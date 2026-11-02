"""Pool causal windows from individually audited nodes using common calendar cuts."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from .io_utils import atomic_write, write_json
from .preprocess import (METRICS, FEATURE_NAMES, FEATURE_UNITS, NS_SECOND,
                         _discover_files, _range_mask, _read_metric,
                         _time_weighted_bins, _weather_asof, _utc)


def weighted_bins(times, values, edges, max_hold_ns, min_coverage):
    """Vectorized equivalent of the reference causal hold for short sensor ages."""
    width = int(edges[1] - edges[0])
    if max_hold_ns > width:
        return _time_weighted_bins(times, values, edges, max_hold_ns, min_coverage)
    count = len(edges) - 1
    ending = np.minimum(np.r_[times[1:], edges[-1]], times + max_hold_ns)
    beginning = np.maximum(times, edges[0])
    ending = np.minimum(ending, edges[-1])
    valid = np.isfinite(values) & (ending > beginning)
    beginning, ending, values = beginning[valid], ending[valid], values[valid]
    first = (beginning - edges[0]) // width
    last = (ending - 1 - edges[0]) // width
    first_seconds = (np.minimum(ending, edges[first + 1]) - beginning) / NS_SECOND
    covered = np.bincount(first, weights=first_seconds, minlength=count)
    total = np.bincount(first, weights=first_seconds * values, minlength=count)
    crosses = first != last
    last_seconds = (ending[crosses] - edges[last[crosses]]) / NS_SECOND
    covered += np.bincount(last[crosses], weights=last_seconds, minlength=count)
    total += np.bincount(last[crosses], weights=last_seconds * values[crosses], minlength=count)
    coverage = covered / (width / NS_SECOND)
    result = np.full(count, np.nan)
    usable = (coverage >= min_coverage) & (covered > 0)
    result[usable] = total[usable] / covered[usable]
    return result, coverage


def discover_nodes(files):
    names = set()
    for path in files:
        source = ds.dataset(str(path), format="parquet")
        if "node" not in source.schema.names:
            raise ValueError(f"{path}: power partition is missing node")
        for batch in source.scanner(columns=["node"], batch_size=262_144).to_batches():
            names.update(value for value in pc.unique(pc.cast(batch["node"], pa.string())).to_pylist()
                         if value is not None)
    return sorted(names, key=lambda node: (0, int(node)) if node.isdigit() else (1, node))


def read_node_groups(files, metric, nodes, month_bounds):
    """Scan each partition once, then stable-sort and deduplicate by node/time."""
    node_chunks, time_chunks, value_chunks = [], [], []
    lookup = pa.array(nodes, type=pa.string())
    for path in files:
        source = ds.dataset(str(path), format="parquet")
        missing = {"node", "timestamp", "value"} - set(source.schema.names)
        if missing:
            raise ValueError(f"{path}: {metric} is missing {sorted(missing)}")
        if not pa.types.is_timestamp(source.schema.field("timestamp").type):
            raise ValueError(f"{path}: timestamp must be an Arrow timestamp")
        for batch in source.scanner(columns=["node", "timestamp", "value"],
                                    batch_size=262_144, batch_readahead=2,
                                    fragment_readahead=1).to_batches():
            valid_time = pc.is_valid(batch["timestamp"])
            ids = pc.fill_null(pc.index_in(pc.cast(pc.filter(batch["node"], valid_time), pa.string()),
                                          value_set=lookup), -1).to_numpy(zero_copy_only=False)
            times = pc.cast(pc.cast(pc.filter(batch["timestamp"], valid_time),
                                   pa.timestamp("ns", tz="UTC")), pa.int64()).to_numpy(zero_copy_only=False)
            values = pc.cast(pc.filter(batch["value"], valid_time), pa.float64()).to_numpy(zero_copy_only=False)
            in_month = np.zeros(len(times), dtype=bool)
            for lower, upper in month_bounds:
                in_month |= (times >= lower) & (times < upper)
            valid = (ids >= 0) & in_month
            if valid.any():
                node_chunks.append(ids[valid].astype(np.int32))
                time_chunks.append(times[valid])
                value_chunks.append(values[valid])
    if not node_chunks:
        return
    ids, times, values = map(np.concatenate, (node_chunks, time_chunks, value_chunks))
    del node_chunks, time_chunks, value_chunks
    before = np.bincount(ids, minlength=len(nodes))
    order = np.lexsort((times, ids))
    ids, times, values = ids[order], times[order], values[order]
    del order
    keep = np.r_[(ids[1:] != ids[:-1]) | (times[1:] != times[:-1]), True]
    ids, times, values = ids[keep], times[keep], values[keep]
    del keep
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(ids)]
    for lower, upper in zip(starts, ends):
        node_index = int(ids[lower])
        t, v = times[lower:upper], values[lower:upper].copy()
        finite = np.isfinite(v)
        valid, rule = _range_mask(metric, v)
        out_of_range = int((finite & ~valid).sum())
        v[~valid] = np.nan
        audit = {"row_count_after_dedup": len(t),
                 "duplicate_timestamp_count": int(before[node_index] - len(t)),
                 "nonfinite_value_count": int((~finite).sum()),
                 "out_of_range_value_count": out_of_range, "validation_rule": rule,
                 "first_timestamp_utc": _utc(t[0]), "last_timestamp_utc": _utc(t[-1])}
        yield node_index, t, v, audit


def node_windows(features, power, edges, history_steps, horizon_steps, boundaries):
    """Generate each node's windows independently; never join across nodes/gaps."""
    origins = np.arange(history_steps, len(power) - horizon_steps + 1)
    missing_history = np.r_[0, np.cumsum(~np.isfinite(features).all(axis=1))]
    missing_target = np.r_[0, np.cumsum(~np.isfinite(power))]
    history_ok = missing_history[origins] == missing_history[origins - history_steps]
    target_ok = missing_target[origins + horizon_steps] == missing_target[origins]
    history = np.lib.stride_tricks.sliding_window_view(features, history_steps, axis=0).transpose(0, 2, 1)
    sums = np.r_[0, np.cumsum(np.where(np.isfinite(power), power, 0.0))]
    result = {}
    for split, (lower, upper) in boundaries.items():
        selected = origins[(origins >= lower) & (origins + horizon_steps <= upper) & history_ok & target_ok]
        result[split] = {
            "X": history[selected - history_steps].astype(np.float32),
            "y": ((sums[selected + horizon_steps] - sums[selected]) / horizon_steps).astype(np.float32)[:, None],
            "origin": edges[selected].astype("datetime64[ns]"),
        }
    return result


def prepare_multinode(dataset_root, output_dir, settings, logger=None):
    logger = logger or logging.getLogger("preprocess")
    root, output_dir = Path(dataset_root).resolve(), Path(output_dir).resolve()
    if "node" in settings:
        raise ValueError("Use either node or nodes, not both")
    requested = settings["nodes"]
    if requested != "all" and (not isinstance(requested, list) or not requested
                                or any(not isinstance(node, str) or not node for node in requested)):
        raise ValueError("nodes must be 'all' or a nonempty list of node ID strings")
    include_gpu = settings.get("include_gpu_utilization", True)
    if not isinstance(include_gpu, bool):
        raise ValueError("include_gpu_utilization must be a JSON boolean")
    time_features = settings.get("time_features", True)
    if not isinstance(time_features, bool):
        raise ValueError("time_features must be a JSON boolean")
    months = sorted(set(settings["months"]))
    month_bounds = []
    for month in months:
        start = datetime.strptime(month, "%y-%m").replace(tzinfo=timezone.utc)
        end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1, tzinfo=timezone.utc)
        month_bounds.append((int(start.timestamp() * NS_SECOND), int(end.timestamp() * NS_SECOND)))
    if not month_bounds:
        raise ValueError("months must be nonempty")
    minutes = int(settings.get("resample_minutes", 10))
    history_minutes = float(settings.get("history_hours", 1)) * 60
    horizon_minutes = int(settings.get("horizon_minutes", 60))
    if minutes <= 0 or history_minutes <= 0 or horizon_minutes <= 0 or history_minutes % minutes or horizon_minutes % minutes:
        raise ValueError("Positive history/horizon must be exact multiples of resample_minutes")
    history_steps, horizon_steps = int(history_minutes / minutes), horizon_minutes // minutes
    train_fraction, val_fraction = float(settings.get("train_fraction", .7)), float(settings.get("val_fraction", .15))
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("Invalid train/validation fractions")
    coverage = float(settings.get("min_bin_coverage", .5))
    hold_seconds = float(settings.get("max_hold_seconds", 300))
    weather_age = float(settings.get("weather_max_age_minutes", 30))
    if not 0 < coverage <= 1 or hold_seconds <= 0 or weather_age <= 0:
        raise ValueError("Invalid coverage or maximum sensor age")
    minimum = settings.get("min_windows_per_split", 1)
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0:
        raise ValueError("min_windows_per_split must be a positive integer")
    metrics = tuple((plugin, metric) for plugin, metric in METRICS if include_gpu or not metric.startswith("Gpu"))
    files = _discover_files(root, months, metrics)
    discovered = discover_nodes(files["total_power"])
    if requested == "all":
        nodes = discovered
    else:
        nodes = sorted(set(requested))
        absent = set(nodes) - set(discovered)
        if absent:
            raise ValueError(f"Requested nodes absent from power partitions: {sorted(absent)}")
    if not nodes:
        raise ValueError("No node IDs found in selected power partitions")
    width = minutes * 60 * NS_SECOND
    edges = np.arange(month_bounds[0][0], month_bounds[-1][1] + width, width, dtype=np.int64)
    bins = len(edges) - 1
    if bins <= history_steps + horizon_steps:
        raise ValueError("Calendar grid is shorter than history plus horizon")
    train_end, val_end = int(bins * train_fraction), int(bins * (train_fraction + val_fraction))
    boundaries = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, bins)}
    local_metrics = [(plugin, metric) for plugin, metric in metrics if plugin != "weather"]
    binned = np.full((len(nodes), bins, len(local_metrics)), np.nan)
    node_audits = {node: {"metrics": {}} for node in nodes}
    logger.info("Auditing %d candidate nodes over common calendar boundaries", len(nodes))
    for column, (_, metric) in enumerate(local_metrics):
        logger.info("Scanning all nodes once for %s", metric)
        present = 0
        for index, t, v, audit in read_node_groups(files[metric], metric, nodes, month_bounds):
            result, time_coverage = weighted_bins(t, v, edges, int(hold_seconds * NS_SECOND), coverage)
            binned[index, :, column] = result
            audit.update(valid_bin_count=int(np.isfinite(result).sum()),
                         mean_bin_time_coverage=float(time_coverage.mean()))
            node_audits[nodes[index]]["metrics"][metric] = audit
            present += 1
        logger.info("Binned %s for %d/%d candidate nodes", metric, present, len(nodes))
    weather, weather_audits = {}, {}
    for metric in ("temp", "humidity"):
        t, v, audit = _read_metric(files[metric], metric, "", True, months, logger)
        weather[metric] = _weather_asof(t, v, edges, int(weather_age * 60 * NS_SECOND))
        weather_audits[metric] = audit
    columns = {metric: index for index, (_, metric) in enumerate(local_metrics)}
    calendar = []
    if time_features:
        seconds = edges[1:] / NS_SECOND
        daily = 2 * np.pi * (seconds % 86400) / 86400
        weekly = 2 * np.pi * ((seconds + 3 * 86400) % (7 * 86400)) / (7 * 86400)
        calendar = [np.sin(daily), np.cos(daily), np.sin(weekly), np.cos(weekly)]
    pooled = {split: {"X": [], "y": [], "origin": [], "node": []} for split in boundaries}
    included = []
    for index, node in enumerate(nodes):
        record = node_audits[node]
        missing = sorted(set(columns) - set(record["metrics"]))
        if missing:
            record.update(status="excluded", reason="missing required metric rows", missing_metrics=missing)
            continue
        value = lambda name: binned[index, :, columns[name]]
        power = value("total_power") / 1000
        with np.errstate(divide="ignore", invalid="ignore"):
            memory = 1 - value("mem_free") / value("mem_total")
        memory[(memory < 0) | (memory > 1)] = np.nan
        features = [power, *(value(name) for name in ("cpu_user", "cpu_system", "cpu_idle", "cpu_wio")), memory]
        if include_gpu:
            features.append(np.column_stack([value(f"Gpu{i}_gpu_utilization") for i in range(4)]).mean(axis=1))
        features.extend([value("ambient"), weather["temp"], weather["humidity"]])
        features.extend(calendar)
        features = np.column_stack(features)
        windows = node_windows(features, power, edges, history_steps, horizon_steps, boundaries)
        counts = {split: len(payload["X"]) for split, payload in windows.items()}
        record["sample_counts"] = counts
        if any(count < minimum for count in counts.values()):
            record.update(status="excluded", reason="insufficient complete windows in a chronological split")
            continue
        node_index = len(included)
        included.append(node)
        record.update(status="included", node_index=node_index)
        for split, payload in windows.items():
            ids = np.full((*payload["X"].shape[:2], 1), node_index, dtype=np.float32)
            pooled[split]["X"].append(np.concatenate((payload["X"], ids), axis=2))
            pooled[split]["y"].append(payload["y"])
            pooled[split]["origin"].append(payload["origin"])
            pooled[split]["node"].append(np.full(len(payload["X"]), node, dtype=f"U{max(map(len, nodes))}"))
    counts = {split: sum(len(chunk) for chunk in payload["X"]) for split, payload in pooled.items()}
    audit_path = output_dir / "results/data_audit.json"
    write_json(audit_path, {"scope": "shared model across usable nodes from one facility",
                           "months": months, "candidate_node_count": len(nodes),
                           "included_node_count": len(included), "excluded_node_count": len(nodes) - len(included),
                           "nodes": node_audits, "weather": weather_audits, "sample_counts": counts,
                           "selection_rule": f"all required sensor rows and at least {minimum} complete windows per split",
                           "grid_start_utc": _utc(edges[0]), "grid_end_exclusive_utc": _utc(edges[-1])})
    if not included:
        raise ValueError(f"No nodes meet coverage requirements; inspect {audit_path}")
    arrays = {}
    for split, payload in pooled.items():
        for key, chunks in payload.items():
            arrays[f"{key}_{split}"] = np.concatenate(chunks)
        payload.clear()
    del binned, pooled
    x_mean = arrays["X_train"].mean(axis=(0, 1), dtype=np.float64)
    x_std = arrays["X_train"].std(axis=(0, 1), dtype=np.float64)
    y_mean = arrays["y_train"].mean(axis=0, dtype=np.float64)
    y_std = arrays["y_train"].std(axis=0, dtype=np.float64)
    # Node identity is categorical, so its index is never standardized.
    x_mean[-1], x_std[-1] = 0, 1
    x_std[x_std < 1e-12] = 1
    y_std[y_std < 1e-12] = 1
    for split in boundaries:
        arrays[f"X_{split}"] = ((arrays[f"X_{split}"] - x_mean) / x_std).astype(np.float32)
        arrays[f"y_{split}"] = ((arrays[f"y_{split}"] - y_mean) / y_std).astype(np.float32)
    arrays["node_names"] = np.asarray(included, dtype=f"U{max(map(len, included))}")
    arrays["power_baseline_coefficients"] = np.asarray([x_std[0] / y_std[0], (x_mean[0] - y_mean[0]) / y_std[0]], dtype=np.float64).reshape(2)
    data_path = output_dir / "prepared/dataset.npz"
    metadata_path = output_dir / "prepared/preprocessing.json"
    atomic_write(data_path, lambda handle: np.savez_compressed(handle, **arrays))
    digest = hashlib.sha256()
    with data_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    pairs = [(name, unit) for name, unit in zip(FEATURE_NAMES, FEATURE_UNITS)
             if include_gpu or name != "gpu_utilization_mean"]
    if time_features:
        pairs.extend((name, "unitless") for name in ("hour_sin", "hour_cos", "week_sin", "week_cos"))
    metadata = {
        "format_version": 2, "settings": settings, "months": months, "nodes": included,
        "node_count": len(included), "feature_names": [name for name, _ in pairs] + ["node_index"],
        "feature_units": [unit for _, unit in pairs] + ["categorical index"],
        "node_index_feature": len(pairs), "history_steps": history_steps,
        "sequence_length": history_steps, "horizon_steps": horizon_steps,
        "input_size": len(pairs) + 1, "target_size": 1,
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
        "normalization_fit": "pooled training windows only; node indices are not standardized",
        "target_name": "next_horizon_mean_node_it_power_kw", "target_unit": "kW",
        "target_definition": "mean power for each node over [origin, origin + horizon); all target bins required",
        "split_boundaries_utc": {"train_end_val_start": _utc(edges[train_end]), "val_end_test_start": _utc(edges[val_end])},
        "split_policy": "common calendar cuts across nodes; full target must remain in split; history may precede cut",
        "origin_unit": "datetime64[ns], UTC", "sample_counts": counts, "dataset_sha256": digest.hexdigest(),
        "source_inventory": [{"path_relative_to_dataset": path.relative_to(root).as_posix(),
                              "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                             for paths in files.values() for path in paths],
    }
    write_json(metadata_path, metadata)
    logger.info("Prepared %d nodes and %d windows: %s", len(included), sum(counts.values()), counts)
    return {"data_path": str(data_path), "preprocessing_path": str(metadata_path),
            "audit_path": str(audit_path), "sample_counts": counts}
