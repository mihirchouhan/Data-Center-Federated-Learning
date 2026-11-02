"""Selective M100 Parquet audit and causal, chronological forecasting windows.

Only one explicitly selected node and the configured metrics are read. GPU
utilization is optional; omitted sensors never become fabricated zero readings.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from .io_utils import atomic_write, write_json


METRICS = (
    ("ipmi", "total_power"), ("ipmi", "ambient"),
    ("ganglia", "cpu_user"), ("ganglia", "cpu_system"),
    ("ganglia", "cpu_idle"), ("ganglia", "cpu_wio"),
    ("ganglia", "mem_free"), ("ganglia", "mem_total"),
    *(("ganglia", f"Gpu{index}_gpu_utilization") for index in range(4)),
    ("weather", "temp"), ("weather", "humidity"),
)
FEATURE_NAMES = [
    "power_kw", "cpu_user", "cpu_system", "cpu_idle", "cpu_wio",
    "memory_used_fraction", "gpu_utilization_mean", "inlet_temperature_mean_c",
    "outdoor_temperature_c", "outdoor_humidity_percent",
]
FEATURE_UNITS = ["kW", "%", "%", "%", "%", "fraction", "%", "degC", "degC", "%"]
RAW_UNITS = {"total_power": "W", "ambient": "degC", "temp": "degC",
             "mem_free": "KB", "mem_total": "KB"}
NS_SECOND = 1_000_000_000


def _utc(timestamp_ns):
    return datetime.fromtimestamp(int(timestamp_ns) / NS_SECOND, timezone.utc).isoformat()


def _discover_files(root, months, metrics=METRICS):
    """Walk directory names once; list Parquet only below requested partitions."""
    result = {metric: [] for _, metric in metrics}
    wanted = {f"year_month={month}" for month in months}
    month_directories = []
    # An extracted tar may add an enclosing directory. rglob here finds only
    # matching month directories; it does not open unselected Parquet contents.
    for month in sorted(wanted):
        month_directories.extend(root.rglob(month))
        if root.name == month:
            month_directories.append(root)
    for directory in sorted(set(month_directories)):
        if not directory.is_dir():
            continue
        for plugin, metric in metrics:
            for alias in (plugin, f"{plugin}_pub"):
                partition = directory / f"plugin={alias}" / f"metric={metric}"
                if partition.is_dir():
                    result[metric].extend(partition.rglob("*.parquet"))
    result = {key: sorted(set(paths)) for key, paths in result.items()}
    missing = [metric for metric, paths in result.items() if not paths]
    if missing:
        raise ValueError(
            "Required metric partitions were not found: " + ", ".join(missing)
            + ". Expected year_month=YY-MM/plugin=ipmi_pub|ganglia_pub|weather_pub/metric=NAME. "
            "Check the extracted dataset root and configured months; absent metrics are not imputed."
        )
    return result


def _range_mask(metric, values):
    finite = np.isfinite(values)
    if metric.startswith("Gpu") or metric in {"cpu_user", "cpu_system", "cpu_idle", "cpu_wio", "humidity"}:
        return finite & (values >= 0) & (values <= 100), "finite, 0 <= value <= 100"
    if metric in {"total_power", "mem_free"}:
        return finite & (values >= 0), "finite, value >= 0"
    if metric == "mem_total":
        return finite & (values > 0), "finite, value > 0"
    return finite, "finite; no speculative temperature limits"


def _read_metric(files, metric, node, weather, months, logger):
    times, values = [], []
    selected_rows = invalid_timestamps = excluded_month_rows = 0
    naive_timestamp_files = []
    bounds = []
    for month in months:
        start = datetime.strptime(month, "%y-%m").replace(tzinfo=timezone.utc)
        end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1, tzinfo=timezone.utc)
        bounds.append((int(start.timestamp() * NS_SECOND), int(end.timestamp() * NS_SECOND)))
    for path in files:
        source = ds.dataset(str(path), format="parquet")
        missing = {"timestamp", "value"} - set(source.schema.names)
        if not weather:
            missing |= {"node"} - set(source.schema.names)
        if missing:
            raise ValueError(f"{path.name}: metric {metric} is missing required columns {sorted(missing)}")
        timestamp_type = source.schema.field("timestamp").type
        if not pa.types.is_timestamp(timestamp_type):
            raise ValueError(f"{path.name}: timestamp must be an Arrow timestamp, found {timestamp_type}; epoch units are not guessed")
        if timestamp_type.tz is None:
            naive_timestamp_files.append(str(path))
        # Cast the predicate so anonymized string, integer and dictionary node
        # columns have the same selection semantics. Arrow scans in bounded batches.
        predicate = None if weather else ds.field("node").cast(pa.string()) == str(node)
        columns = ["timestamp", "value"]
        scanner = source.scanner(columns=columns, filter=predicate, batch_size=65_536,
                                 batch_readahead=2, fragment_readahead=1)
        for batch in scanner.to_batches():
            selected_rows += len(batch)
            valid_time = pc.is_valid(batch["timestamp"])
            invalid_timestamps += len(batch) - int(pc.sum(pc.cast(valid_time, pa.int64())).as_py() or 0)
            stamps = pc.cast(pc.cast(pc.filter(batch["timestamp"], valid_time), pa.timestamp("ns", tz="UTC")), pa.int64())
            numeric = pc.cast(pc.filter(batch["value"], valid_time), pa.float64())
            t = stamps.to_numpy(zero_copy_only=False)
            v = numeric.to_numpy(zero_copy_only=False)
            in_month = np.zeros(len(t), dtype=bool)
            for beginning, ending in bounds:
                in_month |= (t >= beginning) & (t < ending)
            excluded_month_rows += int((~in_month).sum())
            times.append(t[in_month])
            values.append(v[in_month])
    if not times or sum(map(len, times)) == 0:
        hint = (". Verify the dataset node and month coverage, or explicitly set "
                "preprocessing.include_gpu_utilization=false to omit the GPU feature; "
                "missing GPU readings are not imputed" if metric.startswith("Gpu") else "")
        raise ValueError(f"Metric {metric} has no timestamped rows for {'weather' if weather else 'node=' + str(node)} in selected months{hint}")
    t, v = np.concatenate(times), np.concatenate(values)
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    # Files are visited lexically and sorting is stable: last row at a duplicated
    # timestamp wins, including a null value. A null must break stale-value holds.
    keep = np.r_[t[1:] != t[:-1], True]
    duplicate_count = int((~keep).sum())
    t, v = t[keep], v[keep]
    original_finite = np.isfinite(v)
    valid, rule = _range_mask(metric, v)
    invalid_range_count = int((original_finite & ~valid).sum())
    usable_values = v[valid]
    v[~valid] = np.nan
    gaps = np.diff(t) / NS_SECOND
    audit = {
        "metric": metric, "raw_unit": RAW_UNITS.get(metric, "%"),
        "file_count": len(files), "selected_row_count_before_dedup": selected_rows,
        "invalid_timestamp_count": invalid_timestamps, "rows_outside_selected_months": excluded_month_rows,
        "row_count_after_dedup": len(t), "duplicate_timestamp_count": duplicate_count,
        "duplicate_policy": "keep last in lexical file order and original row order",
        "nonfinite_value_count_after_dedup": int((~original_finite).sum()),
        "out_of_range_value_count_after_dedup": invalid_range_count, "validation_rule": rule,
        "first_timestamp_utc": _utc(t[0]), "last_timestamp_utc": _utc(t[-1]),
        "median_gap_seconds": float(np.median(gaps)) if len(gaps) else None,
        "p95_gap_seconds": float(np.percentile(gaps, 95)) if len(gaps) else None,
        "max_gap_seconds": float(gaps.max()) if len(gaps) else None,
        "minimum_usable_value": float(usable_values.min()) if len(usable_values) else None,
        "maximum_usable_value": float(usable_values.max()) if len(usable_values) else None,
        "naive_timestamps_assumed_utc_file_count": len(naive_timestamp_files),
    }
    logger.info("Audited %s: %s selected unique rows, %s duplicates", metric, len(t), duplicate_count)
    return t, v, audit


def _time_weighted_bins(times, values, edges, max_hold_ns, min_coverage):
    """Causal zero-order holds clipped at next observation and maximum age.

    Missing observations break a hold immediately. Values are never carried
    across a source gap for more than max_hold_ns or inferred before first reading.
    """
    total = np.zeros(len(edges) - 1, dtype=np.float64)
    covered = np.zeros(len(total), dtype=np.float64)
    width = int(edges[1] - edges[0])
    next_times = np.r_[times[1:], edges[-1]]
    ends = np.minimum(next_times, times + max_hold_ns)
    for beginning, ending, value in zip(times, ends, values):
        if not np.isfinite(value) or ending <= edges[0] or beginning >= edges[-1]:
            continue
        beginning = max(int(beginning), int(edges[0]))
        ending = min(int(ending), int(edges[-1]))
        if ending <= beginning:
            continue
        first = (beginning - int(edges[0])) // width
        last = (ending - 1 - int(edges[0])) // width
        for index in range(first, last + 1):
            seconds = (min(ending, int(edges[index + 1])) - max(beginning, int(edges[index]))) / NS_SECOND
            covered[index] += seconds
            total[index] += value * seconds
    coverage = covered / (width / NS_SECOND)
    result = np.full(len(total), np.nan)
    valid = (coverage >= min_coverage) & (covered > 0)
    result[valid] = total[valid] / covered[valid]
    return result, coverage


def _weather_asof(times, values, edges, max_age_ns):
    # The right edge itself belongs to the next bin. Only observations available
    # before that edge can describe an input bin completed at a forecast origin.
    right_edges = edges[1:] - 1
    index = np.searchsorted(times, right_edges, side="right") - 1
    safe = np.maximum(index, 0)
    valid = (index >= 0) & (right_edges - times[safe] <= max_age_ns)
    result = np.full(len(index), np.nan)
    result[valid] = values[safe[valid]]
    return result


def prepare_m100(dataset_root: Path, output_dir: Path, settings: dict, logger=None) -> dict:
    """Audit exact source metrics, create train-only normalized NPZ and provenance.

    Target = mean node IT power over [origin, origin + horizon), in kW.
    Inputs = completed bins immediately preceding the origin. Chronological split
    boundaries are fixed on the full observed-power time grid before gap filtering.
    """
    if "nodes" in settings:
        from .multinode import prepare_multinode
        return prepare_multinode(dataset_root, output_dir, settings, logger)
    logger = logger or logging.getLogger("preprocess")
    root, output_dir = Path(dataset_root).resolve(), Path(output_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Extracted dataset directory does not exist: {root}")
    node = str(settings.get("node", "0"))
    include_gpu = settings.get("include_gpu_utilization", True)
    if not isinstance(include_gpu, bool):
        raise ValueError("include_gpu_utilization must be a JSON boolean")
    metrics = tuple((plugin, metric) for plugin, metric in METRICS
                    if include_gpu or not metric.startswith("Gpu"))
    feature_pairs = [(name, unit) for name, unit in zip(FEATURE_NAMES, FEATURE_UNITS)
                     if include_gpu or name != "gpu_utilization_mean"]
    feature_names, feature_units = map(list, zip(*feature_pairs))
    months = sorted(set(settings.get("months", ["22-01", "22-02"])))
    if not months or any(not re.fullmatch(r"\d{2}-(0[1-9]|1[0-2])", month) for month in months):
        raise ValueError("months must be a nonempty list of YY-MM values")
    bin_minutes = int(settings.get("resample_minutes", 10))
    history_minutes = float(settings.get("history_hours", 24)) * 60
    horizon_minutes = int(settings.get("horizon_minutes", 60))
    if bin_minutes <= 0 or history_minutes <= 0 or horizon_minutes <= 0:
        raise ValueError("resampling, history and horizon must be positive")
    if history_minutes % bin_minutes != 0 or horizon_minutes % bin_minutes != 0:
        raise ValueError("history and horizon must be exact multiples of resample_minutes")
    history_steps, horizon_steps = int(history_minutes / bin_minutes), horizon_minutes // bin_minutes
    train_fraction, val_fraction = float(settings.get("train_fraction", .70)), float(settings.get("val_fraction", .15))
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("train_fraction and val_fraction must be positive and sum to less than one")
    coverage_threshold = float(settings.get("min_bin_coverage", .5))
    hold_seconds = float(settings.get("max_hold_seconds", 300))
    weather_age = float(settings.get("weather_max_age_minutes", 30))
    if not 0 < coverage_threshold <= 1 or hold_seconds <= 0 or weather_age <= 0:
        raise ValueError("coverage must be in (0, 1]; telemetry/weather maximum ages must be positive")
    logger.info("Preparing M100 node=%s months=%s history=%g hours GPU utilization=%s",
                node, ",".join(months), history_minutes / 60, include_gpu)
    files = _discover_files(root, months, metrics)
    series, audits = {}, {}
    file_inventory = []
    for plugin, metric in metrics:
        t, v, audit = _read_metric(files[metric], metric, node, plugin == "weather", months, logger)
        series[metric] = (t, v)
        audits[metric] = audit
        for path in files[metric]:
            stat = path.stat()
            file_inventory.append({"path_relative_to_dataset": path.relative_to(root).as_posix(),
                                   "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    width = bin_minutes * 60 * NS_SECOND
    power_times = series["total_power"][0]
    beginning = int(power_times[0]) // width * width
    ending = (int(power_times[-1]) // width + 1) * width
    edges = np.arange(beginning, ending + width, width, dtype=np.int64)
    n_bins = len(edges) - 1
    if n_bins <= history_steps + horizon_steps:
        raise ValueError("Observed data is shorter than the configured history plus forecast horizon")
    binned = {}
    for plugin, metric in metrics:
        t, v = series[metric]
        if plugin == "weather":
            result = _weather_asof(t, v, edges, int(weather_age * 60 * NS_SECOND))
            audits[metric]["aggregation"] = "last observation strictly before bin right edge; no future/backfill interpolation"
        else:
            result, coverage = _time_weighted_bins(t, v, edges, int(hold_seconds * NS_SECOND), coverage_threshold)
            audits[metric].update({"aggregation": "time-weighted causal hold, bounded by next observation and max_hold_seconds",
                                  "mean_bin_time_coverage": float(coverage.mean()),
                                  "bins_below_coverage_threshold": int((coverage < coverage_threshold).sum())})
        binned[metric] = result
        audits[metric]["valid_bin_count"] = int(np.isfinite(result).sum())
    memory = 1 - binned["mem_free"] / binned["mem_total"]
    impossible_memory = np.isfinite(memory) & ((memory < 0) | (memory > 1))
    memory[impossible_memory] = np.nan
    power_kw = binned["total_power"] / 1000
    feature_columns = [
        power_kw, *(binned[name] for name in ("cpu_user", "cpu_system", "cpu_idle", "cpu_wio")),
        memory,
    ]
    if include_gpu:
        gpu = np.stack([binned[f"Gpu{index}_gpu_utilization"] for index in range(4)], axis=1).mean(axis=1)
        feature_columns.append(gpu)
    feature_columns.extend([binned["ambient"], binned["temp"], binned["humidity"]])
    features = np.column_stack(feature_columns)
    train_end = int(n_bins * train_fraction)
    val_end = int(n_bins * (train_fraction + val_fraction))
    bounds = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n_bins)}
    arrays = {}
    dropped = {"history_incomplete": 0, "future_target_incomplete": 0, "crosses_split_boundary": 0}
    by_split = {key: {"X": [], "y": [], "origins": []} for key in bounds}
    for origin in range(history_steps, n_bins - horizon_steps + 1):
        split = next((name for name, (lower, upper) in bounds.items()
                      if origin >= lower and origin + horizon_steps <= upper), None)
        if split is None:
            dropped["crosses_split_boundary"] += 1
            continue
        history = features[origin - history_steps:origin]
        target = power_kw[origin:origin + horizon_steps]
        if not np.isfinite(history).all():
            dropped["history_incomplete"] += 1
            continue
        if not np.isfinite(target).all():
            dropped["future_target_incomplete"] += 1
            continue
        by_split[split]["X"].append(history)
        by_split[split]["y"].append([float(target.mean())])
        by_split[split]["origins"].append(_utc(edges[origin]))
    counts = {split: len(payload["X"]) for split, payload in by_split.items()}
    audit_path = output_dir / "results" / "data_audit.json"
    audit_payload = {
        "node": node, "months": months, "metrics": audits, "grid_bin_count": n_bins,
        "feature_names": feature_names, "include_gpu_utilization": include_gpu,
        "grid_start_utc": _utc(edges[0]), "grid_end_exclusive_utc": _utc(edges[-1]),
        "missing_feature_bin_count": int((~np.isfinite(features).all(axis=1)).sum()),
        "invalid_derived_memory_bin_count": int(impossible_memory.sum()),
        "dropped_windows": dropped, "sample_counts": counts,
        "scope": "one node from one real facility; no inference of independent data centers",
    }
    write_json(audit_path, audit_payload)
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"A chronological split has no complete windows: {counts}. Read {audit_path}; verify node coverage or revise the documented configuration, never fill gaps with zeros.")
    raw_train_x = np.stack(by_split["train"]["X"])
    raw_train_y = np.asarray(by_split["train"]["y"], dtype=np.float64)
    x_mean, x_std = raw_train_x.mean(axis=(0, 1)), raw_train_x.std(axis=(0, 1))
    y_mean, y_std = raw_train_y.mean(axis=0), raw_train_y.std(axis=0)
    constant_features = np.flatnonzero(x_std < 1e-12).tolist()
    x_std[x_std < 1e-12] = 1
    y_std[y_std < 1e-12] = 1
    for split, payload in by_split.items():
        x = np.stack(payload["X"])
        y = np.asarray(payload["y"], dtype=np.float64)
        arrays[f"X_{split}"] = ((x - x_mean) / x_std).astype(np.float32)
        arrays[f"y_{split}"] = ((y - y_mean) / y_std).astype(np.float32)
        arrays[f"origin_{split}"] = np.asarray(payload["origins"], dtype="U32")
    data_path = output_dir / "prepared" / "dataset.npz"
    preprocessing_path = output_dir / "prepared" / "preprocessing.json"
    atomic_write(data_path, lambda handle: np.savez_compressed(handle, **arrays))
    digest = hashlib.sha256()
    with data_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    metadata = {
        "format_version": 1, "node": node, "months": months, "settings": settings,
        "feature_names": feature_names, "feature_units": feature_units,
        "include_gpu_utilization": include_gpu,
        "sequence_length": history_steps, "history_steps": history_steps, "horizon_steps": horizon_steps,
        "input_size": len(feature_names), "target_size": 1,
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
        "constant_feature_indices": constant_features,
        "normalization_fit": "training windows only; repeated overlapping input observations retain their window weights",
        "target_name": "next_horizon_mean_node_it_power_kw", "target_unit": "kW",
        "target_definition": "mean of valid equal-duration power bins in [origin, origin + horizon); all target bins required",
        "input_definition": "completed bins [origin - history, origin); all features required; no future interpolation",
        "weather_definition": "latest observed current weather strictly before each bin end, at most weather_max_age_minutes old",
        "inlet_definition": "IPMI ambient sensor for the selected node, distinct from outdoor weather temp",
        "memory_definition": "1 - bin_mean(mem_free) / bin_mean(mem_total); not a scheduler allocation estimate",
        "split_boundaries_utc": {"train_end_val_start": _utc(edges[train_end]), "val_end_test_start": _utc(edges[val_end])},
        "split_policy": "fixed elapsed-time cuts before invalid windows are removed; complete target must stay in its split; validation/test history may precede cut",
        "sample_counts": counts, "dataset_sha256": digest.hexdigest(),
        "source_inventory": file_inventory,
        "source_inventory_note": "relative paths, sizes and mtimes describe inspected sources; only prepared dataset_sha256 is a full content digest",
    }
    write_json(preprocessing_path, metadata)
    logger.info("Prepared %s chronological windows: %s", sum(counts.values()), counts)
    return {"data_path": str(data_path), "preprocessing_path": str(preprocessing_path),
            "audit_path": str(audit_path), "sample_counts": counts}
