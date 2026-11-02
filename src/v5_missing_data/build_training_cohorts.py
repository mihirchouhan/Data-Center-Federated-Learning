"""Additive six-hour cohort preparation. No model, optimizer, or scoring API.

One explicitly requested raw reconstruction produces compact split-scoped bins,
fill ages, observed power, native/common IDs and observed targets. Verification
uses only these retained bins plus historical hash checks, never a raw decoder.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np

from src.experiment_data import FEATURES
from src.history_coverage_v4 import IdentityDigest, audit_node, preflight, write_json
from src.history_data_v4 import (ReferenceData, build_binned_grid, check_inventory,
    inside, node_features, sha256)
from src.v5_missing_data.forward_fill_audit import causal_forward_fill, eligible_origins

ROOT = Path(__file__).resolve().parents[2]
POLICIES = {'no_fill': 0, 'ff10': 1, 'ff20': 2, 'ff30': 3}
SPLITS = ('train', 'val', 'test')
HISTORY, HORIZON, WIDTH = 36, 6, 600_000_000_000
DAY = 86_400_000_000_000


def read_json(path):
    return json.loads(Path(path).read_text())


def allocate(path, shape, dtype):
    """Create a .npy exclusively, without allocating its full array in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    with path.open('xb') as handle:
        np.lib.format.write_array_header_2_0(handle, {
            'descr': np.lib.format.dtype_to_descr(dtype), 'fortran_order': False, 'shape': tuple(shape)})
        handle.truncate(handle.tell() + int(np.prod(shape)) * dtype.itemsize)
        handle.flush()
        os.fsync(handle.fileno())
    return np.load(path, mmap_mode='r+', allow_pickle=False)


def id_dtype(names):
    width = max(map(len, map(str, names)))
    return np.dtype([('node_index', '<u4'), ('node', f'<U{width}'), ('origin', '<M8[ns]')])


def fill_ages(original, boundaries):
    """Original observation age, reset at cuts; -1 denotes no past observation.

    Weather is never a fill source: 0 if observed, otherwise -1. Positive ages
    exceeding three are retained for provenance, but never authorize a fill.
    """
    if original.ndim != 2 or original.shape[1] != 13 or len(original) > np.iinfo(np.int16).max:
        raise ValueError('Unexpected grid/feature shape for int16 fill ages')
    result = np.full(original.shape, -1, dtype=np.int16)
    previous = 0
    for lower, upper in boundaries.values():
        if lower != previous or not lower < upper <= len(original):
            raise ValueError('Split ranges must partition the original grid')
        positions = np.arange(upper - lower)
        for channel in range(13):
            observed = np.isfinite(original[lower:upper, channel])
            if channel in (7, 8):
                result[lower:upper, channel] = np.where(observed, 0, -1)
                continue
            last = np.maximum.accumulate(np.where(observed, positions, -1))
            result[lower:upper, channel] = np.where(last >= 0, positions - last, -1)
        previous = upper
    if previous != len(original):
        raise ValueError('Split ranges do not cover the grid')
    return result


def fill_from_ages(original, ages, gap):
    if gap not in POLICIES.values() or ages.shape != original.shape:
        raise ValueError('Invalid gap or age shape')
    result = original.copy()
    if gap:
        for channel in range(13):
            if channel in (7, 8):
                continue
            positions = np.flatnonzero((ages[:, channel] > 0) & (ages[:, channel] <= gap))
            sources = positions - ages[positions, channel]
            if (sources < 0).any() or not np.isfinite(original[sources, channel]).all():
                raise ValueError('Fill age points to an invalid original observation')
            result[positions, channel] = original[sources, channel]
    return result


def selections_for_node(original, edges, boundaries):
    """Calculate every policy from original bins; targets never use filled power."""
    selected, diagnostics = {}, {}
    power = original[:, 0].copy()
    power.setflags(write=False)
    ages = fill_ages(original, boundaries)
    _, targets, _ = audit_node(original, edges, boundaries, histories=(6,))
    previous = None
    for policy, gap in POLICIES.items():
        filled = fill_from_ages(original, ages, gap)
        if gap:
            expected = causal_forward_fill(original, gap, boundaries.values())
            if filled.tobytes() != expected.tobytes():
                raise ValueError('Retained age representation differs from audited forward-fill')
        finite = np.isfinite(original)
        if filled[finite].tobytes() != original[finite].tobytes():
            raise ValueError('An original finite observation changed')
        selected[policy], diagnostics[policy] = eligible_origins(
            filled, boundaries, HISTORY, observed_power=power)
        if previous is not None:
            for split in SPLITS:
                if not np.isin(previous[split], selected[policy][split]).all():
                    raise ValueError('Policy nesting failed')
        previous = selected[policy]
    common = {}
    intersections = {'all_policies': tuple(POLICIES),
                     'fill_policies': ('ff10', 'ff20', 'ff30'), 'ff20_ff30': ('ff20', 'ff30')}
    for label, policies in intersections.items():
        common[label] = {}
        for split in SPLITS:
            indices = selected[policies[0]][split]
            for policy in policies[1:]:
                indices = np.intersect1d(indices, selected[policy][split], assume_unique=True)
            if not np.array_equal(indices, selected[policies[0]][split]):
                raise ValueError('Explicit intersection differs from nested-cohort expectation')
            common[label][split] = indices
    return selected, targets, ages, common, diagnostics


def verify_ids(ids, indices, edges, name, node_index, boundaries, split):
    """Check the full identity, not just equal lengths or row positions."""
    expected_origins = edges[indices].astype('datetime64[ns]')
    if (len(ids) != len(indices) or not np.all(ids['node_index'] == node_index)
            or not np.all(ids['node'] == name) or not np.array_equal(ids['origin'], expected_origins)):
        raise ValueError(f'Identity mismatch: {name} {split}')
    times = ids['origin'].view('i8')
    if np.isnat(ids['origin']).any() or (np.diff(times) <= 0).any() or ((times - edges[0]) % WIDTH).any():
        raise ValueError('Duplicate, unordered, NaT or off-grid identities')
    lower, upper = boundaries[split]
    if (indices < lower).any() or (indices < HISTORY).any() or (indices + HORIZON > upper).any():
        raise ValueError('History/target crosses its allowed calendar boundary')


def budget_totals(counts, protocol):
    """Integer accounting only; no row sampling, training loop or model execution."""
    counts = np.asarray(counts, dtype=np.int64)
    if (counts <= 0).any() or not 0 < protocol['clients_per_round'] <= len(counts):
        raise ValueError('Invalid reference client budget/population')
    result = {}
    for seed in protocol['seeds']:
        examples = steps = 0
        for round_id in range(1, protocol['rounds'] + 1):
            positions = np.random.default_rng(np.random.SeedSequence([seed, 1, round_id, 0])).choice(
                len(counts), size=protocol['clients_per_round'], replace=False)
            examples += int(counts[positions].sum())
            steps += int(((counts[positions] + protocol['batch_size'] - 1) // protocol['batch_size']).sum())
        result[str(seed)] = {'examples_processed_per_method_policy': examples,
                             'optimizer_steps_per_method_policy': steps}
    return result


def checked_output(root, output):
    output = Path(output).absolute()
    allowed = root / 'output/experiments/v5_missing_data/model_comparison'
    if (not output.resolve().is_relative_to(allowed.resolve()) or output.resolve() == allowed.resolve()
            or output.is_symlink()):
        raise ValueError('Output must be a new cohort directory below v5 model_comparison')
    return output


def load_inputs(root, plan_path, audit_dir, reference_dir, raw_root):
    plan = read_json(plan_path)
    if (plan['version'] != 1 or plan['stage'] != 'cohort_preparation_only'
            or plan['history_steps'] != HISTORY or plan['horizon_steps'] != HORIZON
            or plan['resample_minutes'] != 10 or plan['policies'] != POLICIES
            or plan['scaler_policy'] != 'reuse_original_training_window_scalers'
            or plan['minimum_windows'] != 1024 or plan['minimum_days'] != 7
            or plan['budget_protocol'] != {'seeds': [42, 43, 44], 'rounds': 200,
                                           'clients_per_round': 98, 'batch_size': 512}):
        raise ValueError('Preparation protocol differs from approval')
    check_inventory(root, plan['execution_sources'])
    check_inventory(root, plan['evidence_files'])
    preparation_path = inside(root, plan['preparation_config']['path'])
    if sha256(preparation_path) != plan['preparation_config']['sha256']:
        raise ValueError('Original preparation declaration changed')
    preparation = read_json(preparation_path)
    checked = preflight(root, preparation, reference_dir, raw_root)
    receipt = read_json(audit_dir / 'completed.json')
    manifest = read_json(audit_dir / 'manifest.json')
    actual = {str(p.relative_to(audit_dir)) for p in audit_dir.rglob('*') if p.is_file() and p != audit_dir / 'completed.json'}
    if (receipt['status'] != 'complete' or actual != set(receipt['artifacts'])
            or receipt['preparation_config_sha256'] != plan['preparation_config']['sha256']):
        raise ValueError('Completed audit inventory/configuration differs')
    for name, expected in receipt['artifacts'].items():
        if sha256(inside(audit_dir, name)) != expected:
            raise ValueError('Completed audit artifact changed: ' + name)
    for name, expected in receipt['sources'].items():
        if sha256(inside(root / 'src/v5_missing_data', name)) != expected:
            raise ValueError('Audited source changed: ' + name)
    baseline_dir = inside(root, plan['baseline_diagnostic_dir'])
    for name, expected in manifest['baseline_artifacts'].items():
        if sha256(inside(baseline_dir, name)) != expected:
            raise ValueError('Baseline diagnostic changed: ' + name)
    summary = read_json(audit_dir / 'summary.json')
    if (summary['original_reference_replay'] != {'status': 'exact_match', 'counts': preparation['reference_counts']}
            or summary['masked_weather_no_fill_windows']['36'] != plan['expected_counts']['no_fill']):
        raise ValueError('Reference replay/expected no-fill cohort differs')
    for policy, gap in POLICIES.items():
        counts = (summary['masked_weather_no_fill_windows']['36'] if not gap else
                  summary['forward_fill_eligible_windows'][str(gap)]['36'])
        if counts != plan['expected_counts'][policy]:
            raise ValueError('Expected native counts disagree with completed audit')
    reference = ReferenceData(reference_dir, checked[0], checked[1])
    if len(reference.names) != plan['expected_nodes']:
        raise ValueError('Original node mapping size differs')
    node_counts = {}
    for policy, gap in POLICIES.items():
        path = (baseline_dir / 'masked_coverage_by_node.csv' if not gap else audit_dir / 'coverage_by_node.csv')
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row['history_steps'] != str(HISTORY) or (gap and row['max_gap_bins'] != str(gap)):
                    continue
                index = int(row['node_index'])
                key = (policy, row['split'], index)
                if (not 0 <= index < len(reference.names) or reference.names[index] != row['node']
                        or key in node_counts or row['split'] not in SPLITS):
                    raise ValueError('Audit per-node mapping/uniqueness mismatch')
                node_counts[key] = int(row['accepted'])
    if len(node_counts) != len(POLICIES) * len(SPLITS) * len(reference.names):
        raise ValueError('Incomplete per-node audit count table')
    for policy in POLICIES:
        for split in SPLITS:
            if sum(node_counts[policy, split, i] for i in range(len(reference.names))) != plan['expected_counts'][policy][split]:
                raise ValueError('Audit CSV and summary disagree')
    return {'plan': plan, 'preparation': preparation, 'checked': checked, 'reference': reference,
            'node_counts': node_counts, 'plan_sha256': sha256(plan_path)}


class PoolWriter:
    def __init__(self, output, label, split, count, names):
        self.label, self.split, self.count, self.cursor = label, split, count, 0
        directory = output / label / split
        self.paths = {'ids': f'{label}/{split}/window_ids.npy',
                      'targets_kw': f'{label}/{split}/targets_kw.npy',
                      'targets_scaled': f'{label}/{split}/targets_scaled.npy'}
        self.ids = allocate(directory / 'window_ids.npy', (count,), id_dtype(names))
        self.raw = allocate(directory / 'targets_kw.npy', (count, 1), '<f4')
        self.scaled = allocate(directory / 'targets_scaled.npy', (count, 1), '<f4')
        self.digest = IdentityDigest()
        self.counts = []

    def add(self, index, name, indices, targets, scaled, edges, boundaries):
        end = self.cursor + len(indices)
        if end > self.count:
            raise ValueError('Generated more windows than audited count')
        rows = self.ids[self.cursor:end]
        rows['node_index'], rows['node'], rows['origin'] = index, name, edges[indices].astype('datetime64[ns]')
        verify_ids(rows, indices, edges, name, index, boundaries, self.split)
        raw, normalized = targets[indices], scaled[indices]
        if not np.isfinite(raw).all() or not np.isfinite(normalized).all():
            raise ValueError('Unobserved/nonfinite target entered a cohort')
        self.raw[self.cursor:end, 0] = raw
        self.scaled[self.cursor:end, 0] = normalized
        self.digest.add(index, name, edges[indices], normalized)
        self.counts.append(len(indices))
        self.cursor = end

    def finish(self):
        if self.cursor != self.count:
            raise ValueError('Generated fewer windows than audited count')
        for array in (self.ids, self.raw, self.scaled):
            array.flush()
        result = {'paths': self.paths, 'count': self.count, 'per_node_counts': self.counts,
                  'nodes_with_windows': sum(c > 0 for c in self.counts),
                  'identity': self.digest.record()}
        for name in ('ids', 'raw', 'scaled'):
            array = getattr(self, name)
            array._mmap.close()
            delattr(self, name)
        return result


def context_specs(boundaries):
    return {split: {'lower': lower, 'upper': upper, 'feature_start': max(0, lower - HISTORY - 3),
                    'features': f'bins/{split}_features.npy', 'ages': f'bins/{split}_fill_ages.npy',
                    'observed_power': f'bins/{split}_observed_power.npy'}
            for split, (lower, upper) in boundaries.items()}


def source_snapshot(root, plan_path, output, plan):
    with plan_path.open('rb') as incoming, (output / 'plan.json').open('xb') as outgoing:
        import shutil
        shutil.copyfileobj(incoming, outgoing)
    for name, record in plan['execution_sources'].items():
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with inside(root, name).open('rb') as incoming, target.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if sha256(target) != record['sha256']:
            raise ValueError('Source changed while snapshotting: ' + name)


def prepare(root, context, plan_path, raw_root, output):
    """Only preparation. Raw decoder is called exactly once across all policies."""
    output = checked_output(root, output)
    if output.exists() or output.is_symlink():
        raise ValueError('Output exists; partial/completed cohorts are never overwritten')
    plan, preparation, reference = context['plan'], context['preparation'], context['reference']
    _, meta, edges, boundaries, before = context['checked']
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_snapshot(root, plan_path, output, plan)
    write_json(output / 'manifest.json', {'stage': 'cohort_preparation_only', 'plan_sha256': context['plan_sha256'],
        'raw_reconstruction_reason': 'Completed audits retain counts, not fill-policy IDs or reusable bin grids',
        'raw_reconstruction_passes': 1, 'original_preflight': before,
        'test_status': 'Previously observed development benchmark; not confirmatory',
        'training_inference_or_prediction_scoring': False})
    write_json(output / 'node_mapping.json', list(map(str, reference.names)))
    write_json(output / 'scalers.json', context['checked'][0]['scalers'])
    totals = dict.fromkeys(SPLITS, 0)
    contexts = context_specs(boundaries)
    specs, clients = {}, []
    with tempfile.TemporaryDirectory(prefix='.cohort-scratch-', dir=output) as temporary:
        grid, weather, quality = build_binned_grid(raw_root, preparation['raw_files'], reference.names,
                                                  edges, meta, Path(temporary), preparation['reading_limits'])
        try:
            print('Verifying complete original one-hour reference before accepting six-hour cohorts', flush=True)
            for index, _ in enumerate(reference.names):
                original = node_features(grid[index], weather, edges)
                selected, targets, _ = audit_node(original, edges, boundaries, histories=(6,))
                for split, count in reference.verify_node(index, original, edges, selected[6], targets).items():
                    totals[split] += count
            if totals != preparation['reference_counts']:
                raise ValueError('Original reference counts differ')
            write_json(output / 'reference_replay.json', {'status': 'exact_match', 'counts': totals,
                'checks': ['node/origin ordering', 'split assignment', 'float32 targets', 'scaled six-step input bytes']})
            writers = {(policy, split): PoolWriter(output, 'native/' + policy, split,
                plan['expected_counts'][policy][split], reference.names) for policy in POLICIES for split in SPLITS}
            writers.update({('common', split): PoolWriter(output, 'common/all_policies', split,
                plan['expected_counts']['no_fill'][split], reference.names) for split in SPLITS})
            recovered_count = plan['expected_counts']['ff20']['test'] - plan['expected_counts']['no_fill']['test']
            recovered = PoolWriter(output, 'recovered/ff20', 'test', recovered_count, reference.names)
            bins = {}
            for split, item in contexts.items():
                bins[split] = (
                    allocate(output / item['features'], (len(reference.names), item['upper'] - item['feature_start'], 13), '<f8'),
                    allocate(output / item['ages'], (len(reference.names), item['upper'] - item['feature_start'], 13), '<i2'),
                    allocate(output / item['observed_power'], (len(reference.names), item['upper'] - item['lower']), '<f8'))
            with (output / 'per_node_coverage.csv').open('x', newline='') as handle:
                table = csv.DictWriter(handle, fieldnames=['node_index', 'node', 'policy', 'split', 'windows', 'days'])
                table.writeheader()
                for index, name in enumerate(reference.names):
                    original = node_features(grid[index], weather, edges)
                    selected, targets, ages, common, _ = selections_for_node(original, edges, boundaries)
                    scaled = reference.scaled_targets(targets)
                    for split, item in contexts.items():
                        bins[split][0][index] = original[item['feature_start']:item['upper']]
                        bins[split][1][index] = ages[item['feature_start']:item['upper']]
                        bins[split][2][index] = original[item['lower']:item['upper'], 0]
                        for policy in POLICIES:
                            indices = selected[policy][split]
                            if len(indices) != context['node_counts'][policy, split, index]:
                                raise ValueError(f'Per-node audited count mismatch: {name} {policy} {split}')
                            writers[policy, split].add(index, name, indices, targets, scaled, edges, boundaries)
                            table.writerow({'node_index': index, 'node': str(name), 'policy': policy, 'split': split,
                                'windows': len(indices), 'days': len(np.unique(edges[indices] // DAY))})
                        writers['common', split].add(index, name, common['all_policies'][split], targets,
                                                       scaled, edges, boundaries)
                    recovered_indices = np.setdiff1d(selected['ff20']['test'], common['all_policies']['test'], assume_unique=True)
                    recovered.add(index, name, recovered_indices, targets, scaled, edges, boundaries)
                    count = len(selected['no_fill']['train'])
                    days = len(np.unique(edges[selected['no_fill']['train']] // DAY))
                    if count < plan['minimum_windows'] or days < plan['minimum_days']:
                        raise ValueError('A client no longer meets the unchanged training requirements')
                    clients.append({'node_index': index, 'node': str(name), 'm_k': count,
                        'no_fill_training_days': days, 'native_training_counts': {
                            policy: len(selected[policy]['train']) for policy in POLICIES}})
                    if index % 100 == 0:
                        print(f'Prepared node {index + 1}/{len(reference.names)}', flush=True)
            for (policy, split), writer in writers.items():
                specs[('common/all_policies' if policy == 'common' else 'native/' + policy) + '/' + split] = writer.finish()
            specs['recovered/ff20/test'] = recovered.finish()
            for arrays in bins.values():
                for array in arrays:
                    array.flush()
                    array._mmap.close()
        finally:
            del grid
    expected_budget = budget_totals([c['m_k'] for c in clients], plan['budget_protocol'])
    if expected_budget != plan['expected_budget_totals']:
        raise ValueError('Revised per-client budget accounting differs from protocol')
    for split in SPLITS:
        if specs['native/no_fill/' + split]['identity'] != specs['common/all_policies/' + split]['identity']:
            raise ValueError('Explicit all-policy intersection identity/targets differ from no-fill')
        for policy in POLICIES:
            if specs['native/' + policy + '/' + split]['nodes_with_windows'] != plan['expected_covered_nodes'][policy][split]:
                raise ValueError('Audited node coverage differs')
    contract = {'version': 1, 'stage': 'cohort_preparation_only', 'history_steps': HISTORY,
        'horizon_steps': HORIZON, 'feature_names': FEATURES, 'node_mapping': 'node_mapping.json',
        'target_name': 'next_horizon_mean_node_it_power_kw', 'target_unit': 'kW',
        'grid_start_ns': int(edges[0]), 'grid_end_ns': int(edges[-1]), 'width_ns': WIDTH,
        'boundaries': boundaries, 'bin_contexts': contexts, 'cohorts': specs,
        'scalers': 'scalers.json', 'scaler_policy': plan['scaler_policy'],
        'fill_age_definition': '-1:no original source since split reset; 0:observed; positive:age in bins; weather never filled',
        'feature_storage': 'Original float64 bins; no normalization refit; no dense history tensors',
        'target_storage': 'Observed six-bin mean float32 kW, then original float64 scaling and float32 cast',
        'policy_training_views': {p: 'native/' + p + '/train' for p in POLICIES},
        'policy_validation_views': {p: 'common/all_policies/val' for p in POLICIES},
        'policy_test_views': {p: 'common/all_policies/test' for p in POLICIES},
        'intersection_aliases': {
            label: {s: 'native/' + p + '/' + s for s in SPLITS}
            for label, p in (('fill_policies', 'ff10'), ('ff20_ff30', 'ff20'))},
        'training_worker_scope': 'Train IDs/targets and train bin files only; evaluation bins are separate coordinator files',
        'training_api_implemented': False}
    write_json(output / 'contract.json', contract)
    write_json(output / 'client_budgets.json', {'clients': clients, 'protocol': plan['budget_protocol'],
        'budget_totals': expected_budget, 'row_sampling_or_optimizer_execution': False})
    write_json(output / 'source_quality.json', quality)
    return contract, {'wall_seconds_excluding_initial_preflight': time.monotonic() - started,
        'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'scope': 'Wall time covers raw reconstruction/materialization, excluding later verification; RSS is Linux process lifetime high-water mark; no GPU/model'}


def retained_node(output, contract, arrays, index):
    """Reassemble contiguous bins from split files, without dropping any missing bin."""
    features, power, age = [], [], []
    for split in SPLITS:
        item = contract['bin_contexts'][split]
        offset = item['lower'] - item['feature_start']
        features.append(arrays[item['features']][index, offset:])
        age.append(arrays[item['ages']][index, offset:])
        power.append(arrays[item['observed_power']][index])
    return np.concatenate(features), np.concatenate(power), np.concatenate(age)


def verify_prepared(output, context):
    """Full semantic check from retained bins only. No raw decoder or model."""
    contract = read_json(output / 'contract.json')
    plan, reference = context['plan'], context['reference']
    _, _, edges, boundaries, _ = context['checked']
    if (contract['history_steps'] != HISTORY or contract['horizon_steps'] != HORIZON
            or contract['target_name'] != 'next_horizon_mean_node_it_power_kw' or contract['target_unit'] != 'kW'
            or contract['feature_names'] != FEATURES or contract['boundaries'] != {s: list(v) for s, v in boundaries.items()}
            or contract['bin_contexts'] != context_specs(boundaries)
            or read_json(output / 'node_mapping.json') != reference.names.tolist()
            or read_json(output / 'scalers.json') != context['checked'][0]['scalers']):
        raise ValueError('Prepared mapping/scalers/features/boundaries changed')
    paths = {p for item in contract['bin_contexts'].values() for p in (item['features'], item['ages'], item['observed_power'])}
    paths |= {p for item in contract['cohorts'].values() for p in item['paths'].values()}
    arrays = {p: np.load(inside(output, p), mmap_mode='r', allow_pickle=False) for p in paths}
    for split, item in contract['bin_contexts'].items():
        length = item['upper'] - item['feature_start']
        for key, shape, dtype in (
                ('features', (len(reference.names), length, 13), np.float64),
                ('ages', (len(reference.names), length, 13), np.int16),
                ('observed_power', (len(reference.names), item['upper'] - item['lower']), np.float64)):
            if arrays[item[key]].shape != shape or arrays[item[key]].dtype != dtype:
                raise ValueError('Split-scoped bin shape/dtype changed: ' + split)
    for spec in contract['cohorts'].values():
        if (len(spec['per_node_counts']) != len(reference.names) or sum(spec['per_node_counts']) != spec['count']
                or spec['nodes_with_windows'] != sum(n > 0 for n in spec['per_node_counts'])):
            raise ValueError('Cohort count metadata changed')
        for key, shape, dtype in (('ids', (spec['count'],), id_dtype(reference.names)),
                                  ('targets_kw', (spec['count'], 1), np.float32),
                                  ('targets_scaled', (spec['count'], 1), np.float32)):
            if arrays[spec['paths'][key]].shape != shape or arrays[spec['paths'][key]].dtype != dtype:
                raise ValueError('Cohort array shape/dtype changed')
    budgets = read_json(output / 'client_budgets.json')
    if budgets['protocol'] != plan['budget_protocol'] or len(budgets['clients']) != len(reference.names):
        raise ValueError('Client budget protocol/population changed')
    cursors = dict.fromkeys(contract['cohorts'], 0)
    digests = {label: IdentityDigest() for label in cursors}
    replay = dict.fromkeys(SPLITS, 0)
    for index, name in enumerate(reference.names):
        original, power, ages = retained_node(output, contract, arrays, index)
        if original[:, 0].tobytes() != power.tobytes():
            raise ValueError('Stored observed power differs from the original unfilled power bins')
        selected6, targets6, _ = audit_node(original, edges, boundaries, histories=(6,))
        for split, count in reference.verify_node(index, original, edges, selected6[6], targets6).items():
            replay[split] += count
        selected, targets, expected_age, common, _ = selections_for_node(original, edges, boundaries)
        if not np.array_equal(ages, expected_age):
            raise ValueError('Retained split-reset fill ages differ')
        for item in contract['bin_contexts'].values():
            lo, hi = item['feature_start'], item['upper']
            if (arrays[item['features']][index].tobytes() != original[lo:hi].tobytes()
                    or arrays[item['ages']][index].tobytes() != expected_age[lo:hi].tobytes()):
                raise ValueError('Overlapping lookback context differs from original causal bins')
        client = budgets['clients'][index]
        actual_days = len(np.unique(edges[selected['no_fill']['train']] // DAY))
        if (client['node_index'] != index or client['node'] != name
                or client['m_k'] != len(selected['no_fill']['train'])
                or client['no_fill_training_days'] != actual_days
                or client['m_k'] < plan['minimum_windows'] or actual_days < plan['minimum_days']
                or client['native_training_counts'] != {p: len(selected[p]['train']) for p in POLICIES}):
            raise ValueError('Client membership/budget/day counts changed')
        scaled = reference.scaled_targets(targets)
        views = {'native/' + policy + '/' + split: selected[policy][split] for policy in POLICIES for split in SPLITS}
        views |= {'common/all_policies/' + split: common['all_policies'][split] for split in SPLITS}
        views['recovered/ff20/test'] = np.setdiff1d(selected['ff20']['test'], common['all_policies']['test'], assume_unique=True)
        for label, indices in views.items():
            split = label.rsplit('/', 1)[1]
            spec = contract['cohorts'][label]
            start, end = cursors[label], cursors[label] + len(indices)
            if len(indices) != spec['per_node_counts'][index]:
                raise ValueError('Stored per-node cohort count changed')
            if label.startswith('native/'):
                policy = label.split('/')[1]
                if len(indices) != context['node_counts'][policy, split, index]:
                    raise ValueError('Retained bins no longer reproduce per-node audit counts')
            ids = arrays[spec['paths']['ids']][start:end]
            if ids.dtype != id_dtype(reference.names):
                raise ValueError('Prepared identity dtype changed')
            verify_ids(ids, indices, edges, name, index, boundaries, split)
            for key, expected in (('targets_kw', targets[indices]), ('targets_scaled', scaled[indices])):
                actual = arrays[spec['paths'][key]][start:end]
                if actual.dtype != np.float32 or actual.shape != (len(indices), 1) or actual[:, 0].tobytes() != expected.tobytes():
                    raise ValueError('Observed target values/scaling changed: ' + label)
            digests[label].add(index, name, edges[indices], scaled[indices])
            cursors[label] = end
    if replay != context['preparation']['reference_counts']:
        raise ValueError('Retained-bin original reference replay differs')
    for label, spec in contract['cohorts'].items():
        if cursors[label] != spec['count'] or digests[label].record() != spec['identity']:
            raise ValueError('Cohort identity/target digest mismatch')
        if len(arrays[spec['paths']['ids']]) != spec['count']:
            raise ValueError('Extra/missing cohort rows')
    for split in ('val', 'test'):
        references = contract['policy_' + ('validation' if split == 'val' else 'test') + '_views']
        if set(references) != set(POLICIES) or set(references.values()) != {'common/all_policies/' + split}:
            raise ValueError('Policies do not share identical evaluation arrays')
    if contract['policy_training_views'] != {p: 'native/' + p + '/train' for p in POLICIES}:
        raise ValueError('Policy training-pool references changed')
    if contract['intersection_aliases'] != {
            label: {s: 'native/' + p + '/' + s for s in SPLITS}
            for label, p in (('fill_policies', 'ff10'), ('ff20_ff30', 'ff20'))}:
        raise ValueError('Secondary intersection references changed')
    counts = [context['node_counts']['no_fill', 'train', i] for i in range(len(reference.names))]
    if (budget_totals(counts, plan['budget_protocol']) != plan['expected_budget_totals']
            or budgets['budget_totals'] != plan['expected_budget_totals']
            or [c['m_k'] for c in budgets['clients']] != counts):
        raise ValueError('Reference client budgets changed')
    return {'status': 'verified_complete', 'original_reference_replay': replay,
            'cohort_arrays_verified': len(cursors), 'raw_decoder_used': False,
            'training_inference_or_prediction_scoring': False}


def write_report(output, contract, resources, semantic):
    lines = ['# V5 six-hour cohort preparation v1', '',
        'Preparation only. No model training, inference, scoring or fit matrix was executed.', '',
        'One bounded raw reconstruction was necessary: completed audits saved counts but no fill-policy IDs or bin grids. '
        'All policies reuse that one reconstructed grid. Split-scoped float64 bins, observed power and int16 fill ages '
        'are now retained; verification uses them without raw decoding.', '',
        '## Verified native cohorts', '', '| Policy | Training | Validation | Test | Train/val/test nodes |',
        '| --- | ---: | ---: | ---: | --- |']
    for policy in POLICIES:
        specs = [contract['cohorts'][f'native/{policy}/{s}'] for s in SPLITS]
        lines.append('| ' + policy + ' | ' + ' | '.join(f"{x['count']:,}" for x in specs)
            + ' | ' + '/'.join(str(x['nodes_with_windows']) for x in specs) + ' |')
    lines += ['', '## Common evaluation and training reference', '']
    for split in SPLITS:
        item = contract['cohorts']['common/all_policies/' + split]
        lines.append(f"- {split}: {item['count']:,} common IDs; {item['nodes_with_windows']} nodes; ordered ID SHA-256 `{item['identity']['ordered_sha256']}`.")
    lines += ['', 'The fill-only three-way intersection has the same ID/target arrays as native ff10; '
        'the ff20/ff30 intersection has the same arrays as native ff20. These aliases were checked by explicit intersections.']
    lines += ['', 'Each policy references the exact same common validation/test arrays. Native training pools remain '
        'different by design; common training IDs define m_k. Secondary fill-only and ff20/ff30 intersections alias '
        'their least-permissive native arrays only after explicit per-node intersections have been verified.', '',
        'Original one-hour identities/order, splits, target bytes and six-step input bytes replayed exactly during '
        'preparation and again from retained bins during verification. All targets remain observed six-bin means; '
        'scalers and node indices are unchanged. Histories may precede a split; targets never cross it. '
        'Fill states reset at cuts, and weather channels 7/8 remain ignored for eligibility.', '',
        '## Revised client budgets', '', 'See `client_budgets.json`: m_k is each original node’s no-fill training count, '
        'not its native filled-pool count. Actual row subsets, model batches and communication rounds are NOT implemented. '
        'The recorded exposure/step totals are integer planning checks only.', '',
        '## Integrity and resources', '', f'- Semantic verification: `{semantic["status"]}`.',
        f'- Raw reconstruction/materialization seconds (excluding initial preflight and later verification): {resources["wall_seconds_excluding_initial_preflight"]:.3f}.',
        f'- Linux process lifetime peak RSS bytes: {resources["peak_process_rss_bytes"]}.',
        '- `preservation.json` records before/after historical hash checks; `completed.json` binds every new artifact.', '',
        '## Scope, unresolved issues and recovery', '',
        'No training/evaluation code was changed. The future training-only adapters, deterministic sampled-pass runner, '
        'checkpoint selection and test-only evaluator still require separate approval and implementation. '
        '24-hour experiments remain outside this preparation. The retained ages support later gap-cause analysis, '
        'but no new 24-hour study or forecasting result is claimed. The test period was previously observed.', '',
        'No resume or overwrite is supported. A completed directory is read-only; verify it with --verify-only. '
        'Preserve an interrupted directory and share its error; do not rerun --prepare into it or edit expected hashes. '
        'Any replacement version needs an unused destination and reviewed input/cache policy.', '',
        '## Generated files', '']
    lines += ['- `' + str(p.relative_to(output)) + '`' for p in sorted(output.rglob('*')) if p.is_file()]
    lines += ['- `REPORT.md` (this file).', '- `completed.json` (written last).', '']
    with (output / 'REPORT.md').open('x') as handle:
        handle.write('\n'.join(lines))


def execute(root, plan_path, audit_dir, reference_dir, raw_root, output):
    output = checked_output(root, output)
    if output.exists() or output.is_symlink():
        raise ValueError('Output exists; never overwrite existing cohorts')
    context = load_inputs(root, plan_path, audit_dir, reference_dir, raw_root)
    contract, resources = prepare(root, context, plan_path, raw_root, output)
    semantic = verify_prepared(output, context)
    after = load_inputs(root, plan_path, audit_dir, reference_dir, raw_root)
    if after['plan_sha256'] != context['plan_sha256']:
        raise ValueError('Preparation plan changed during execution')
    write_json(output / 'preservation.json', {'before': context['checked'][-1], 'after': after['checked'][-1],
        'existing_artifact_changes': [], 'expected_hashes_modified': False})
    write_json(output / 'resources.json', resources)
    write_json(output / 'verification.json', semantic)
    write_report(output, contract, resources, semantic)
    files = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'completed.json', {'status': 'complete', 'plan_sha256': context['plan_sha256'],
        'artifacts': files, 'training_inference_or_prediction_scoring': False})
    for path in output.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    return {**semantic, 'status': 'prepared_verified', 'output': str(output)}


def verify_output(root, plan_path, audit_dir, reference_dir, raw_root, output):
    context = load_inputs(root, plan_path, audit_dir, reference_dir, raw_root)
    receipt = read_json(output / 'completed.json')
    actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p != output / 'completed.json'}
    if receipt['status'] != 'complete' or receipt['plan_sha256'] != context['plan_sha256'] or actual != set(receipt['artifacts']):
        raise ValueError('Prepared output inventory/plan differs or is incomplete')
    for name, expected in receipt['artifacts'].items():
        if sha256(inside(output, name)) != expected:
            raise ValueError('Prepared artifact hash mismatch: ' + name)
    return verify_prepared(output, context)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'audit-dir', 'reference-data-dir', 'raw-data-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'preflight-only', 'verify-only'):
        modes.add_argument('--' + name, action='store_true')
    args = parser.parse_args(argv)
    try:
        paths = [Path(x).resolve() for x in (args.plan, args.audit_dir, args.reference_data_dir, args.raw_data_dir)]
        output = checked_output(ROOT, args.output_dir)
        if args.verify_only:
            result = verify_output(ROOT, *paths, output)
        else:
            if output.exists() or output.is_symlink():
                raise ValueError('Output exists; never overwrite existing cohorts')
            if args.preflight_only:
                context = load_inputs(ROOT, *paths)
                result = {'status': 'preflight_passed', 'historical_integrity': context['checked'][-1], 'raw_decoder_used': False}
            else:
                result = execute(ROOT, *paths, output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('COHORT PREPARATION ERROR: ' + str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
