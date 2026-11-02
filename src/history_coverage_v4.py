"""Additive M100 history-coverage audit. Never train, infer, score, or freeze X.

CLI modes are mutually exclusive: --preflight-only, --verify-only, or an explicit
coverage run. Completed and interrupted output directories are never reused.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import tempfile
import time

import numpy as np

from .experiment_data import FEATURES
from .history_data_v4 import (ReferenceData, SPLITS, WINDOW_DTYPE, build_binned_grid,
    check_inventory, inside, node_features, sha256, utc_ns)

ROOT = Path(__file__).resolve().parents[1]
HISTORIES = (6, 36, 144)
DAY_NS = 86_400_000_000_000
REASONS = ('insufficient_initial_history', 'target_crosses_split_end',
           'invalid_target_power', 'incomplete_history')


def write_json(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def preflight(root, config, reference_dir, raw_root):
    """Read-only provenance verification; no raw binning or model operations."""
    if (config['version'] != 1 or config['history_steps'] != list(HISTORIES)
            or config['horizon_steps'] != 6 or config['resample_minutes'] != 10
            or config['feature_names'] != FEATURES or config['weather'] != 'masked'
            or config['weather_mask_indices'] != [7, 8]
            or config['scaler_policy'] != 'reuse_original_training_window_scalers'
            or config['node_policy'] != 'unchanged_frozen_mapping_no_filter_or_reindex'):
        raise ValueError('Unsupported audit protocol')
    limits = config['reading_limits']
    if set(limits) != {'nodes_per_shard', 'arrow_batch_rows', 'max_shard_rows', 'max_node_rows'}:
        raise ValueError('Invalid reading limits')
    for key, value in limits.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f'Invalid reading limit: {key}')
    if limits['nodes_per_shard'] > 64 or limits['arrow_batch_rows'] > 262144:
        raise ValueError('Reading limits exceed reviewed bounds')
    if (config['training_eligibility'] != {'minimum_windows': 1024, 'minimum_days': 7}):
        raise ValueError('Training eligibility requirements changed')
    protected = check_inventory(root, config['protected_inventory'])
    sources = check_inventory(root, config['execution_sources'])
    if sha256(reference_dir / 'contract.json') != config['reference_contract_sha256']:
        raise ValueError('Frozen reference contract hash mismatch')
    contract = json.loads((reference_dir / 'contract.json').read_text())
    if sha256(reference_dir / 'preprocessing.json') != contract['metadata_sha256']:
        raise ValueError('Frozen preprocessing hash mismatch')
    meta = json.loads((reference_dir / 'preprocessing.json').read_text())
    if (contract['feature_names'] != FEATURES or meta['feature_names'] != FEATURES
            or contract['history_steps'] != 6 or meta['history_steps'] != 6
            or contract['horizon_steps'] != 6 or meta['horizon_steps'] != 6
            or meta['settings'] != config['reference_settings']
            or meta['split_boundaries_utc'] != config['chronological_cuts']
            or contract['chronological_cuts'] != config['chronological_cuts']
            or meta['sample_counts'] != config['reference_counts']
            or contract['scalers'] != {k: meta[k] for k in ('x_mean', 'x_std', 'y_mean', 'y_std')}
            or contract['weather_mask_indices'] != [7, 8]
            or meta['target_name'] != 'next_horizon_mean_node_it_power_kw'):
        raise ValueError('Frozen reference protocol differs from declaration')
    if (meta['settings']['include_gpu_utilization'] is not False
            or meta['settings']['time_features'] is not True
            or meta['settings']['history_hours'] != 1
            or meta['settings']['resample_minutes'] != 10
            or meta['settings']['horizon_minutes'] != 60):
        raise ValueError('Unexpected original preparation settings')
    for key in ('x_mean', 'x_std', 'y_mean', 'y_std'):
        scaler = np.asarray(meta[key])
        if (scaler.shape != ((14,) if key.startswith('x') else (1,))
                or not np.isfinite(scaler).all() or ('std' in key and (scaler <= 0).any())):
            raise ValueError('Invalid frozen scaler: ' + key)
    if meta['x_mean'][13] != 0 or meta['x_std'][13] != 1:
        raise ValueError('Node identity scaler changed')
    expected_arrays = {f'{key}_{split}.npy' for key in ('X', 'y', 'origin', 'node') for split in SPLITS}
    expected_arrays |= {'node_names.npy', 'power_baseline_coefficients.npy'}
    if set(contract['arrays']) != expected_arrays:
        raise ValueError('Unexpected frozen array inventory')
    for name, expected in contract['arrays'].items():
        path = inside(reference_dir, name)
        if sha256(path) != expected['sha256']:
            raise ValueError('Frozen array hash mismatch: ' + name)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(array.shape) != expected['shape'] or str(array.dtype) != expected['dtype']:
            raise ValueError('Frozen array schema mismatch: ' + name)
    entries = config['raw_files']
    inventory = meta['source_inventory']
    if (len(entries) != len(inventory) or len({e['path'] for e in entries}) != len(entries)
            or [e['path'] for e in entries] != [e['path_relative_to_dataset'] for e in inventory]):
        raise ValueError('Raw input inventory/order changed')
    for entry, recorded in zip(entries, inventory):
        path = inside(raw_root, entry['path'])
        if path.stat().st_size != recorded['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('Raw Parquet hash/size mismatch: ' + entry['path'])
    # Bind the raw hashes to historical extraction receipts, rather than trusting
    # a newly declared hash without its original provenance.
    original_hashes = {}
    for entry in config['extraction_receipts']:
        path = inside(root, entry['path'])
        if sha256(path) != entry['sha256']:
            raise ValueError('Extraction receipt changed: ' + entry['path'])
        for item in json.loads(path.read_text())['files']:
            if item['path'] in original_hashes:
                raise ValueError('Duplicate extraction receipt path')
            original_hashes[item['path']] = item['sha256']
    if any(original_hashes.get(e['path']) != e['sha256'] for e in entries):
        raise ValueError('Raw hashes disagree with original extraction receipts')
    start, end = utc_ns(config['grid_start_utc']), utc_ns(config['grid_end_exclusive_utc'])
    width = 600 * 1_000_000_000
    if end <= start or (end - start) % width:
        raise ValueError('Invalid calendar grid')
    edges = np.arange(start, end + width, width, dtype=np.int64)
    cuts = config['chronological_cuts']
    train_cut, val_cut = utc_ns(cuts['train_end_val_start']), utc_ns(cuts['val_end_test_start'])
    if not start < train_cut < val_cut < end or any((cut - start) % width for cut in (train_cut, val_cut)):
        raise ValueError('Invalid chronological cuts')
    bins = len(edges) - 1
    if (int(bins * meta['settings']['train_fraction']) != (train_cut - start) // width
            or int(bins * (meta['settings']['train_fraction'] + meta['settings']['val_fraction'])) != (val_cut - start) // width):
        raise ValueError('Calendar fraction replay differs from original cuts')
    # Verify the declared grid is the original complete month range.
    from datetime import datetime, timezone
    months = sorted(meta['months'])
    beginning = datetime.strptime(months[0], '%y-%m').replace(tzinfo=timezone.utc)
    last = datetime.strptime(months[-1], '%y-%m').replace(tzinfo=timezone.utc)
    ending = datetime(last.year + (last.month == 12), last.month % 12 + 1, 1, tzinfo=timezone.utc)
    if start != utc_ns(beginning.isoformat()) or end != utc_ns(ending.isoformat()):
        raise ValueError('Calendar month extent changed')
    a, b = (train_cut - start) // width, (val_cut - start) // width
    boundaries = {'train': (0, a), 'val': (a, b), 'test': (b, bins)}
    names = np.load(reference_dir / 'node_names.npy', mmap_mode='r', allow_pickle=False)
    if names.tolist() != meta['nodes'] or len(names) != config['reference_node_count']:
        raise ValueError('Frozen node mapping mismatch')
    coefficients = np.load(reference_dir / 'power_baseline_coefficients.npy', mmap_mode='r', allow_pickle=False)
    expected = np.array([meta['x_std'][0] / meta['y_std'][0],
                         (meta['x_mean'][0] - meta['y_mean'][0]) / meta['y_std'][0]])
    if coefficients.tobytes() != expected.tobytes():
        raise ValueError('Frozen persistence coefficient mismatch')
    return contract, meta, edges, boundaries, {
        'protected': protected, 'execution_sources': sources,
        'frozen_arrays_checked': len(contract['arrays']), 'raw_files_checked': len(entries),
        'mismatches': [], 'training_or_inference_or_metric_evaluation': False}


def audit_node(features, edges, boundaries, histories=HISTORIES):
    """All candidate origins are counted once per split, including edge failures."""
    if features.shape != (len(edges) - 1, 13) or np.any(np.diff(edges) != 600 * 1_000_000_000):
        raise ValueError('Invalid per-node regular grid/features')
    if tuple(histories) not in ((6,), HISTORIES):
        raise ValueError('Unexpected history lengths')
    power = features[:, 0]
    missing = np.r_[0, np.cumsum(~np.isfinite(features).all(axis=1))]
    missing_target = np.r_[0, np.cumsum(~np.isfinite(power))]
    sums = np.r_[0, np.cumsum(np.where(np.isfinite(power), power, 0.0))]
    # Same double cumsum and pre-scaling float32 cast as historical node_windows.
    targets = np.full(len(power), np.nan, dtype=np.float32)
    target_origins = np.arange(len(power) - 6 + 1)
    targets[target_origins] = ((sums[target_origins + 6] - sums[target_origins]) / 6).astype(np.float32)
    selected, diagnostics = {}, {}
    for history in histories:
        selected[history], diagnostics[history] = {}, {}
        for split, (lower, upper) in boundaries.items():
            origins = np.arange(lower, upper)
            have_history = origins >= history
            inside_split = origins + 6 <= upper
            hist_ok = np.zeros(len(origins), dtype=bool)
            hist_ok[have_history] = (missing[origins[have_history]] == missing[origins[have_history] - history])
            target_ok = np.zeros(len(origins), dtype=bool)
            target_ok[inside_split] = (missing_target[origins[inside_split] + 6] == missing_target[origins[inside_split]])
            flags = [~have_history, ~inside_split, inside_split & ~target_ok, have_history & ~hist_ok]
            accepted = have_history & inside_split & hist_ok & target_ok
            selected[history][split] = origins[accepted]
            remaining = np.ones(len(origins), dtype=bool)
            primary = {}
            for reason, flag in zip(REASONS, flags):
                primary[reason] = int((remaining & flag).sum())
                remaining &= ~flag
            if not np.array_equal(remaining, accepted) or sum(primary.values()) + int(accepted.sum()) != len(origins):
                raise ValueError('Exclusion accounting failed')
            channels = {}
            for channel, name in enumerate(FEATURES[:-1]):
                prefix = np.r_[0, np.cumsum(~np.isfinite(features[:, channel]))]
                invalid = prefix[origins[have_history]] != prefix[origins[have_history] - history]
                channels[name] = int(invalid.sum())
            diagnostics[history][split] = {'candidate_origins': len(origins), 'accepted': int(accepted.sum()),
                'primary_exclusions': primary, 'multi_cause_exclusions': {
                    **{reason: int(flag.sum()) for reason, flag in zip(REASONS, flags)},
                    **{'history_channel:' + name: count for name, count in channels.items()}}}
    if len(histories) == 3:
        for split in SPLITS:
            if (not np.isin(selected[144][split], selected[36][split]).all()
                    or not np.isin(selected[36][split], selected[6][split]).all()):
                raise ValueError('History eligibility is not nested')
    return selected, targets, diagnostics


def matched_indices(selected):
    return {split: np.intersect1d(np.intersect1d(selected[6][split], selected[36][split]), selected[144][split])
            for split in SPLITS}


class IdentityDigest:
    def __init__(self):
        self.ids = hashlib.sha256(b'M100 ordered (node UTF-8, origin UTC ns) v1\0')
        self.targets = hashlib.sha256()
        self.count = 0
        self.last_node = -1
        self.last_origin = None

    def add(self, node_index, node, origins, targets):
        if len(origins) != len(targets) or (np.diff(origins) <= 0).any() or not np.isfinite(targets).all():
            raise ValueError('Malformed or unordered cohort identities/targets')
        if node_index < self.last_node or (len(origins) and node_index == self.last_node
                and self.last_origin is not None and origins[0] <= self.last_origin):
            raise ValueError('Noncanonical cohort ordering')
        encoded = str(node).encode('utf8')
        prefix = len(encoded).to_bytes(8, 'little') + encoded
        for origin in origins:
            self.ids.update(prefix)
            self.ids.update(int(origin).to_bytes(8, 'little', signed=True))
        self.targets.update(np.asarray(targets, dtype='<f4').tobytes())
        if len(origins):
            self.last_node, self.last_origin = node_index, int(origins[-1])
        self.count += len(origins)

    def record(self):
        return {'count': self.count, 'ordered_sha256': self.ids.hexdigest(),
                'ordered_scaled_targets_sha256': self.targets.hexdigest(),
                'identity': '(node string, forecast origin UTC datetime64[ns])'}


class Coverage:
    def __init__(self, names, edges):
        self.names, self.edges = names, edges
        self.counts = np.zeros(len(names), dtype=np.int64)
        self.origin_counts = np.zeros(len(edges) - 1, dtype=np.int64)
        self.training_eligible = np.zeros(len(names), dtype=bool)
        self.digest = IdentityDigest()

    def add(self, node_index, indices, targets, handle):
        origins = self.edges[indices]
        self.counts[node_index] = len(indices)
        self.origin_counts[indices] += 1
        days = len(np.unique(origins // DAY_NS))
        self.training_eligible[node_index] = len(indices) >= 1024 and days >= 7
        self.digest.add(node_index, self.names[node_index], origins, targets)
        rows = np.empty(len(indices), dtype=WINDOW_DTYPE)
        rows['node_index'], rows['origin_utc_ns'], rows['target_scaled'] = node_index, origins, targets
        handle.write(rows.tobytes())
        return days

    def summary(self):
        occupied = np.flatnonzero(self.origin_counts)
        return {'windows': int(self.counts.sum()), 'eligible_nodes_at_least_one_window': int((self.counts > 0).sum()),
            'distinct_forecast_origins': len(occupied),
            'zero_coverage_nodes': [str(n) for n, count in zip(self.names, self.counts) if not count],
            'per_node_window_quantiles_min_q25_median_q75_max': np.quantile(self.counts, [0, .25, .5, .75, 1]).tolist(),
            'first_origin_utc': str(self.edges[occupied[0]].astype('datetime64[ns]')) if len(occupied) else None,
            'last_origin_utc': str(self.edges[occupied[-1]].astype('datetime64[ns]')) if len(occupied) else None,
            'training_eligible_nodes_if_training_split': int(self.training_eligible.sum()),
            'window_identity': self.digest.record()}


def _output_path(root, output):
    output = Path(output).absolute()
    allowed = root / 'output/experiments/v4_history/coverage_audit'
    if (not output.resolve().is_relative_to(allowed.resolve()) or output.resolve() == allowed.resolve()
            or output.is_symlink()):
        raise ValueError('Output must be a new directory under output/experiments/v4_history/coverage_audit')
    return output


def _run_coverage(root, config, reference_dir, raw_root, output, config_path, checked):
    contract, meta, edges, boundaries, before = checked
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import pyarrow
    write_json(output / 'manifest.json', {
        'status': 'started', 'config_sha256': sha256(config_path), 'preflight': before,
        'reference_contract_sha256': config['reference_contract_sha256'],
        'reference_dir': str(reference_dir), 'raw_root': str(raw_root),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pyarrow': pyarrow.__version__, 'platform': platform.platform()},
        'weather': 'masked', 'scaler_policy': config['scaler_policy'],
        'scaler_scope': 'Original pooled training-window population, not newly fitted matched-cohort scalers',
        'test_status': 'Previously observed development period; coverage audit does not score predictions',
        'candidate_universe': 'All grid origins in each split for every original frozen node',
        'primary_exclusion_precedence': list(REASONS),
        'multi_cause_counts': 'May overlap; history reasons require available history, target reasons require in-split target',
        'binary_window_dtype': WINDOW_DTYPE.descr, 'training_or_inference_or_metric_evaluation': False})
    shutil.copyfile(config_path, output / 'config.json')
    initial_config_hash = sha256(output / 'config.json')
    for name in config['execution_sources']:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with inside(root, name).open('rb') as incoming, destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if sha256(destination) != config['execution_sources'][name]['sha256']:
            raise ValueError('Execution source changed while snapshotting: ' + name)
    reference = ReferenceData(reference_dir, contract, meta)
    write_json(output / 'node_mapping.json', reference.names.tolist())
    with tempfile.TemporaryDirectory(prefix='.scratch-', dir=output) as temporary:
        scratch = Path(temporary)
        grid, weather, quality = build_binned_grid(raw_root, config['raw_files'], reference.names,
                                                 edges, meta, scratch, config['reading_limits'])
        try:
            totals = {split: 0 for split in SPLITS}
            print('Checking exact original one-hour identities, ordering, targets and inputs', flush=True)
            for index in range(len(reference.names)):
                features = node_features(grid[index], weather, edges)
                selected, targets, _ = audit_node(features, edges, boundaries, histories=(6,))
                counts = reference.verify_node(index, features, edges, selected[6], targets)
                for split in SPLITS:
                    totals[split] += counts[split]
            if totals != config['reference_counts']:
                raise ValueError('One-hour reference aggregate counts differ')
            write_json(output / 'reference_replay.json', {'status': 'exact_match', 'counts': totals,
                'nodes': len(reference.names), 'checked': ['identity', 'node_order', 'origin_order',
                    'split_assignment', 'float32_scaled_targets', 'six_step_scaled_inputs'],
                'longer_history_coverage_accepted': False})
            coverages = {(f'native_steps{h}', split): Coverage(reference.names, edges) for h in HISTORIES for split in SPLITS}
            coverages.update({('matched', split): Coverage(reference.names, edges) for split in SPLITS})
            exclusions = {str(h): {s: {'candidate_origins': 0, 'accepted': 0,
                'primary_exclusions': dict.fromkeys(REASONS, 0), 'multi_cause_exclusions': {}}
                for s in SPLITS} for h in HISTORIES}
            overlaps = {s: {f'{a}_vs_{b}': 0 for a, b in itertools.combinations(HISTORIES, 2)} for s in SPLITS}
            write_json(output / 'source_quality.json', quality)
            with ExitStack() as stack:
                files = {key: stack.enter_context((output / f'{key[0]}_{key[1]}.bin').open('xb')) for key in coverages}
                table = stack.enter_context((output / 'per_node_coverage.csv').open('x', newline=''))
                writer = csv.DictWriter(table, fieldnames=['cohort', 'split', 'node', 'node_index', 'windows',
                    'distinct_days', 'training_eligible', 'lost_to_match', 'matched_retention_percent'])
                writer.writeheader()
                for index, name in enumerate(reference.names):
                    features = node_features(grid[index], weather, edges)
                    selected, targets, diag = audit_node(features, edges, boundaries)
                    matched = matched_indices(selected)
                    scaled = reference.scaled_targets(targets)
                    for split in SPLITS:
                        for h in HISTORIES:
                            aggregate = exclusions[str(h)][split]
                            current = diag[h][split]
                            for key in ('candidate_origins', 'accepted'):
                                aggregate[key] += current[key]
                            for key in ('primary_exclusions', 'multi_cause_exclusions'):
                                for reason, count in current[key].items():
                                    aggregate[key][reason] = aggregate[key].get(reason, 0) + count
                        for a, b in itertools.combinations(HISTORIES, 2):
                            overlaps[split][f'{a}_vs_{b}'] += len(np.intersect1d(selected[a][split], selected[b][split]))
                    selections = {f'native_steps{h}': selected[h] for h in HISTORIES} | {'matched': matched}
                    for cohort, selections_by_split in selections.items():
                        training_indices = selections_by_split['train']
                        eligible = len(training_indices) >= 1024 and len(np.unique(edges[training_indices] // DAY_NS)) >= 7
                        for split, indices in selections_by_split.items():
                            coverage = coverages[cohort, split]
                            days = coverage.add(index, indices, scaled[indices], files[cohort, split])
                            lost = len(indices) - len(matched[split])
                            writer.writerow({'cohort': cohort, 'split': split, 'node': str(name), 'node_index': index,
                                'windows': len(indices), 'distinct_days': days, 'training_eligible': eligible,
                                'lost_to_match': lost,
                                'matched_retention_percent': 100 * len(matched[split]) / len(indices) if len(indices) else ''})
            summaries = {cohort: {split: coverages[cohort, split].summary() for split in SPLITS}
                         for cohort in ('native_steps6', 'native_steps36', 'native_steps144', 'matched')}
            for cohort in summaries:
                eligible = int(coverages[cohort, 'train'].training_eligible.sum())
                for split in SPLITS:
                    summaries[cohort][split].pop('training_eligible_nodes_if_training_split')
                    summaries[cohort][split]['training_eligible_nodes'] = eligible
                    native = summaries[cohort][split]['windows']
                    match = summaries['matched'][split]['windows']
                    original = totals[split]
                    summaries[cohort][split]['lost_from_original_one_hour'] = original - native
                    summaries[cohort][split]['retained_from_original_one_hour_percent'] = 100 * native / original if original else None
                    summaries[cohort][split]['lost_to_matched'] = native - match
                    summaries[cohort][split]['matched_retention_percent'] = 100 * match / native if native else None
            for split in SPLITS:
                if summaries['native_steps6'][split]['windows'] != totals[split]:
                    raise ValueError('Reference counts changed between passes')
                if summaries['native_steps144'][split]['window_identity'] != summaries['matched'][split]['window_identity']:
                    raise ValueError('Nested/matched cohort identity or target mismatch')
            write_json(output / 'summary.json', {'cohorts': summaries, 'pairwise_overlap_windows': overlaps,
                'matched_definition': 'Per-split intersection of node/origin identities at steps 6,36,144',
                'node_filtering': 'None; training eligibility is reported, never applied',
                'model_comparison_ready': all(summaries['matched'][s]['windows'] > 0 for s in SPLITS)
                    and summaries['matched']['train']['training_eligible_nodes'] == len(reference.names),
                'warning': 'No claim that longer history improves forecasting; inspect coverage before any fitting'})
            write_json(output / 'matched_one_hour_reference.json', {
                'history_steps': 6, 'cohort': summaries['matched'],
                'definition': 'One-hour reference rebuilt on the three-way matched identities, with identical targets',
                'original_reference_replay': 'exact_match', 'scalers': contract['scalers'],
                'forecast_metrics_computed': False})
            write_json(output / 'exclusions.json', {'histories': exclusions,
                'reconciliation': 'accepted + mutually exclusive primary exclusions = all candidate origins',
                'source_quality': 'source_quality.json contains marginal sensor diagnostics; these are not causal attribution of every lost window'})
            with (output / 'origin_coverage.csv').open('x', newline='') as handle, (output / 'date_coverage.csv').open('x', newline='') as daily:
                ow, dw = csv.writer(handle), csv.writer(daily)
                ow.writerow(['cohort', 'split', 'origin_utc', 'windows'])
                dw.writerow(['cohort', 'split', 'date_utc', 'windows', 'distinct_origins'])
                for (cohort, split), coverage in coverages.items():
                    indices = np.flatnonzero(coverage.origin_counts)
                    for origin in indices:
                        ow.writerow([cohort, split, str(edges[origin].astype('datetime64[ns]')), int(coverage.origin_counts[origin])])
                    for day in np.unique(edges[indices] // DAY_NS):
                        selected_day = indices[edges[indices] // DAY_NS == day]
                        dw.writerow([cohort, split, str(np.datetime64(int(day), 'D')),
                            int(coverage.origin_counts[selected_day].sum()), len(selected_day)])
        finally:
            del grid
    after = preflight(root, config, reference_dir, raw_root)[-1]
    if sha256(config_path) != initial_config_hash:
        raise ValueError('Audit configuration changed during execution')
    write_json(output / 'preservation.json', {'before': before, 'after': after,
        'excluded_from_protected_inventory': ['.venv', '.git', '__pycache__', '.pytest_cache'],
        'atime_not_compared': True})
    write_json(output / 'resources.json', {'wall_seconds': time.monotonic() - started,
        'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'rss_scope': 'Linux process lifetime high-water mark; no GPU used',
        'dense_36_or_144_step_inputs_created': False})
    artifacts = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'completed.json', {'status': 'coverage_complete', 'artifacts': artifacts,
        'config_sha256': sha256(config_path), 'reference_replay': 'exact_match',
        'training_or_inference_or_metric_evaluation': False})
    return {'status': 'coverage_complete', 'output': str(output), 'reference_counts': totals,
            'matched_windows': {s: summaries['matched'][s]['windows'] for s in SPLITS}}


def verify_output(root, config, reference_dir, raw_root, output, config_path):
    """Hash verification only. Does not rebin raw data or rewrite outputs."""
    before = preflight(root, config, reference_dir, raw_root)[-1]
    receipt = json.loads((output / 'completed.json').read_text())
    if receipt['status'] != 'coverage_complete' or receipt['config_sha256'] != sha256(config_path):
        raise ValueError('Output/configuration receipt mismatch')
    actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p != output / 'completed.json'}
    if actual != set(receipt['artifacts']):
        raise ValueError('Output inventory differs; partial/extra artifacts are preserved')
    for name, expected in receipt['artifacts'].items():
        if sha256(inside(output, name)) != expected:
            raise ValueError('Output artifact changed: ' + name)
    if json.loads((output / 'reference_replay.json').read_text())['status'] != 'exact_match':
        raise ValueError('Reference gate not passed')
    return {'status': 'verified_complete', 'artifacts_checked': len(actual), 'input_integrity': before,
            'raw_rebinning_or_model_inference_or_metric_evaluation': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--reference-data-dir', required=True)
    parser.add_argument('--raw-data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        output = _output_path(ROOT, Path(args.output_dir))
        if not args.verify_only and (output.exists() or output.is_symlink()):
            raise ValueError('Output exists; completed and interrupted audits are never overwritten')
        config_path = Path(args.config).resolve()
        config = json.loads(config_path.read_text())
        reference_dir, raw_root = Path(args.reference_data_dir).resolve(), Path(args.raw_data_dir).resolve()
        if args.verify_only:
            result = verify_output(ROOT, config, reference_dir, raw_root, output, config_path)
        else:
            checked = preflight(ROOT, config, reference_dir, raw_root)
            result = ({'status': 'preflight_passed', **checked[-1]} if args.preflight_only else
                _run_coverage(ROOT, config, reference_dir, raw_root, output, config_path, checked))
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('HISTORY COVERAGE ERROR: ' + str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
