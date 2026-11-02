"""V4 coverage-only data access. No model, fitting, or dense long-history arrays.

Raw rows are scanned once per metric into bounded disk shards. Stable source
order is retained, including duplicate/null records. Only one node is sorted
at a time; the historical binning and weather implementations are reused.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import stat
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from .experiment_data import FEATURES
from .multinode import weighted_bins
from .preprocess import NS_SECOND, _range_mask, _read_metric, _weather_asof

LOCAL_METRICS = ('total_power', 'ambient', 'cpu_user', 'cpu_system', 'cpu_idle',
                 'cpu_wio', 'mem_free', 'mem_total')
RAW_DTYPE = np.dtype([('node', '<i4'), ('time', '<i8'), ('value', '<f8')])
WINDOW_DTYPE = np.dtype([('node_index', '<u4'), ('origin_utc_ns', '<i8'),
                         ('target_scaled', '<f4')])
SPLITS = ('train', 'val', 'test')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f'Unsafe path: {path}')
    return path


def check_inventory(root, inventory):
    """Hash original files, checking recorded metadata without changing it."""
    failures = []
    for name, expected in inventory.items():
        path = inside(root, name)
        if not path.is_file():
            failures.append(f'missing: {name}')
            continue
        current = path.stat()
        if sha256(path) != expected['sha256']:
            failures.append(f'hash: {name}')
        for key, actual in (('bytes', current.st_size), ('mtime_ns', current.st_mtime_ns),
                            ('mode', stat.S_IMODE(current.st_mode))):
            if key in expected and expected[key] != actual:
                failures.append(f'{key}: {name}')
    if failures:
        raise ValueError('Preservation failed: ' + '; '.join(failures))
    return {'checked': len(inventory), 'mismatches': []}


def utc_ns(value):
    return int(np.datetime64(value.replace('+00:00', ''), 'ns').view('i8'))


def mask_weather(x, weather='masked'):
    """Copy-only mask; the fourteen channels and frozen normalization stay fixed."""
    if x.ndim != 3 or x.shape[1] not in (6, 36, 144) or x.shape[2] != 14:
        raise ValueError('Expected [N,6|36|144,14]')
    if weather not in ('masked', 'enabled'):
        raise ValueError('Unknown weather policy')
    result = x.copy()
    if weather == 'masked':
        result[:, :, [7, 8]] = 0
    return result


class ReferenceData:
    """Read-only original arrays. Used solely for exact preparation replay checks."""
    def __init__(self, directory, contract, meta):
        self.directory, self.contract, self.meta = Path(directory), contract, meta
        self.names = np.load(self.directory / 'node_names.npy', mmap_mode='r', allow_pickle=False)
        if self.names.tolist() != meta['nodes'] or len(set(self.names.tolist())) != len(self.names):
            raise ValueError('Frozen node mapping changed')
        self.arrays, self.offsets = {}, {}
        for split in SPLITS:
            arrays = {key: np.load(self.directory / f'{key}_{split}.npy', mmap_mode='r', allow_pickle=False)
                      for key in ('X', 'y', 'node', 'origin')}
            count = len(arrays['y'])
            if (arrays['X'].shape != (count, 6, 14) or arrays['X'].dtype != np.float32
                    or arrays['y'].shape != (count, 1) or arrays['y'].dtype != np.float32
                    or arrays['origin'].shape != (count,) or arrays['origin'].dtype != np.dtype('datetime64[ns]')
                    or arrays['node'].shape != (count,) or arrays['node'].dtype.kind != 'U'):
                raise ValueError(f'Invalid reference schema: {split}')
            starts = np.r_[0, np.flatnonzero(arrays['node'][1:] != arrays['node'][:-1]) + 1]
            if not count or not np.array_equal(arrays['node'][starts], self.names):
                raise ValueError(f'Reference node blocks/order changed: {split}')
            self.offsets[split] = np.r_[starts, count]
            self.arrays[split] = arrays

    def scaled_targets(self, raw_targets):
        # Original preparation rounds targets to float32 BEFORE normalization.
        return ((np.asarray(raw_targets, dtype=np.float32).astype(np.float64)
                 - self.meta['y_mean'][0]) / self.meta['y_std'][0]).astype(np.float32)

    def verify_node(self, node_index, features, edges, selections, targets):
        """Require byte-exact original IDs, ordering, targets AND six-step inputs."""
        checked = {}
        for split in SPLITS:
            lower, upper = self.offsets[split][node_index:node_index + 2]
            old = self.arrays[split]
            selected = selections[split]
            origins = edges[selected].astype('datetime64[ns]')
            if not np.array_equal(old['origin'][lower:upper], origins):
                raise ValueError(f'One-hour reference identity/order mismatch: node={self.names[node_index]} {split}')
            y = self.scaled_targets(targets[selected])
            if y.tobytes() != old['y'][lower:upper, 0].tobytes():
                raise ValueError(f'One-hour reference target mismatch: node={self.names[node_index]} {split}')
            if not np.isfinite(y).all():
                raise ValueError('Nonfinite reference target')
            for start in range(0, len(selected), 256):
                idx = selected[start:start + 256, None] + np.arange(-6, 0)[None, :]
                raw = np.concatenate((features[idx], np.full((*idx.shape, 1), node_index)), axis=2).astype(np.float32)
                scaled = ((raw - np.asarray(self.meta['x_mean'])) / np.asarray(self.meta['x_std'])).astype(np.float32)
                expected = old['X'][lower + start:min(lower + start + 256, upper)]
                if not np.isfinite(scaled).all() or scaled.tobytes() != expected.tobytes():
                    raise ValueError(f'One-hour reference input mismatch: node={self.names[node_index]} {split}')
            checked[split] = len(selected)
        return checked


def spool_metric(files, nodes, edges, scratch, limits):
    """One ordered Arrow scan; store node/time/value records in bounded disk shards."""
    shard_size = limits['nodes_per_shard']
    count = (len(nodes) + shard_size - 1) // shard_size
    paths = [scratch / f'{i:04d}.bin' for i in range(count)]
    totals = np.zeros(count, dtype=np.int64)
    lookup = pa.array(nodes, type=pa.string())
    with ExitStack() as stack:
        handles = [stack.enter_context(path.open('xb')) for path in paths]
        for path in files:
            source = ds.dataset(str(path), format='parquet')
            if (not {'node', 'timestamp', 'value'} <= set(source.schema.names)
                    or not pa.types.is_timestamp(source.schema.field('timestamp').type)):
                raise ValueError(f'Invalid raw schema: {path}')
            scanner = source.scanner(columns=['node', 'timestamp', 'value'], batch_size=limits['arrow_batch_rows'],
                                     batch_readahead=1, fragment_readahead=1, use_threads=False)
            for batch in scanner.to_batches():
                good = pc.is_valid(batch['timestamp'])
                ids = pc.fill_null(pc.index_in(pc.cast(pc.filter(batch['node'], good), pa.string()),
                                              value_set=lookup), -1).to_numpy(zero_copy_only=False)
                times = pc.cast(pc.cast(pc.filter(batch['timestamp'], good), pa.timestamp('ns', tz='UTC')),
                                pa.int64()).to_numpy(zero_copy_only=False)
                values = pc.cast(pc.filter(batch['value'], good), pa.float64()).to_numpy(zero_copy_only=False)
                keep = (ids >= 0) & (times >= edges[0]) & (times < edges[-1])
                rows = np.empty(int(keep.sum()), dtype=RAW_DTYPE)
                rows['node'], rows['time'], rows['value'] = ids[keep], times[keep], values[keep]
                shards = rows['node'] // shard_size
                for shard in np.unique(shards):
                    part = rows[shards == shard]  # Boolean indexing retains original row order.
                    totals[shard] += len(part)
                    if totals[shard] > limits['max_shard_rows']:
                        raise ValueError('Raw shard row limit exceeded; no clients are filtered')
                    handles[shard].write(part.tobytes())
    return paths, totals


def bin_metric(files, metric, nodes, edges, scratch, settings, limits, destination):
    """Sort one node at a time; use the protected vectorized binning unchanged."""
    quality = {}
    with tempfile.TemporaryDirectory(prefix=f'{metric}-', dir=scratch) as temporary:
        paths, counts = spool_metric(files, nodes, edges, Path(temporary), limits)
        for shard, path in enumerate(paths):
            if not counts[shard]:
                continue
            rows = np.memmap(path, mode='r', dtype=RAW_DTYPE)
            try:
                first = shard * limits['nodes_per_shard']
                for node_index in range(first, min(first + limits['nodes_per_shard'], len(nodes))):
                    selected = np.flatnonzero(rows['node'] == node_index)
                    if len(selected) > limits['max_node_rows']:
                        raise ValueError(f'Raw node row limit exceeded: {nodes[node_index]}')
                    if not len(selected):
                        quality[str(nodes[node_index])] = {'raw_rows': 0, 'valid_bins': 0}
                        continue
                    values = rows[selected]
                    order = np.argsort(values['time'], kind='stable')
                    times, raw_values = values['time'][order], values['value'][order]
                    unique = np.r_[times[1:] != times[:-1], True]
                    times, raw_values = times[unique], raw_values[unique]
                    valid, _ = _range_mask(metric, raw_values)
                    invalid_range = np.isfinite(raw_values) & ~valid
                    nonfinite = ~np.isfinite(raw_values)
                    raw_values[~valid] = np.nan
                    binned, coverage = weighted_bins(times, raw_values, edges,
                        int(settings['max_hold_seconds'] * NS_SECOND), settings['min_bin_coverage'])
                    destination[node_index] = binned
                    quality[str(nodes[node_index])] = {
                        'raw_rows': len(selected), 'unique_rows': len(times),
                        'duplicates_keep_last': int(len(selected) - len(times)),
                        'nonfinite_rows_after_dedup': int(nonfinite.sum()),
                        'out_of_range_rows_after_dedup': int(invalid_range.sum()),
                        'valid_bins': int(np.isfinite(binned).sum()),
                        'bins_below_minimum_coverage': int((coverage < settings['min_bin_coverage']).sum())}
            finally:
                del rows
    return quality


def build_binned_grid(raw_root, entries, names, edges, meta, scratch, limits):
    """Disk-backed local metrics; tiny weather arrays stay in memory."""
    files = {}
    for entry in entries:
        metric = next(part[7:] for part in Path(entry['path']).parts if part.startswith('metric='))
        files.setdefault(metric, []).append(inside(raw_root, entry['path']))
    if set(files) != set(LOCAL_METRICS) | {'temp', 'humidity'}:
        raise ValueError('Unexpected required metrics')
    grid = np.lib.format.open_memmap(scratch / 'local_bins.npy', mode='w+', dtype=np.float64,
                                   shape=(len(names), len(edges) - 1, len(LOCAL_METRICS)))
    grid[:] = np.nan
    quality = {}
    for column, metric in enumerate(LOCAL_METRICS):
        print(f'Binning {metric}: one scan, bounded disk shards', flush=True)
        quality[metric] = bin_metric(files[metric], metric, names, edges, scratch,
                                    meta['settings'], limits, grid[:, :, column])
    weather = {}
    import logging
    for metric in ('temp', 'humidity'):
        times, values, audit = _read_metric(files[metric], metric, '', True, meta['months'], logging.getLogger(__name__))
        weather[metric] = _weather_asof(times, values, edges,
            int(meta['settings']['weather_max_age_minutes'] * 60 * NS_SECOND))
        quality[metric] = audit | {'invalid_asof_bins': int((~np.isfinite(weather[metric])).sum())}
    grid.flush()
    return grid, weather, quality


def node_features(local, weather, edges):
    """Reproduce the thirteen non-ID features in their original float64 order."""
    value = {name: local[:, i] for i, name in enumerate(LOCAL_METRICS)}
    power = value['total_power'] / 1000
    with np.errstate(divide='ignore', invalid='ignore'):
        memory = 1 - value['mem_free'] / value['mem_total']
    memory[(memory < 0) | (memory > 1)] = np.nan
    seconds = edges[1:] / NS_SECOND
    daily = 2 * np.pi * (seconds % 86400) / 86400
    weekly = 2 * np.pi * ((seconds + 3 * 86400) % (7 * 86400)) / (7 * 86400)
    return np.column_stack([power, *(value[name] for name in ('cpu_user', 'cpu_system', 'cpu_idle', 'cpu_wio')),
        memory, value['ambient'], weather['temp'], weather['humidity'],
        np.sin(daily), np.cos(daily), np.sin(weekly), np.cos(weekly)])
