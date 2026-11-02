"""Synthetic coverage tests only: temporary Parquet/arrays, no model operations."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.experiment_data import FEATURES
from src.history_coverage_v4 import (HISTORIES, Coverage, IdentityDigest, _output_path, _run_coverage,
    audit_node, main, matched_indices, preflight, verify_output)
from src.history_data_v4 import (LOCAL_METRICS, RAW_DTYPE, WINDOW_DTYPE, ReferenceData,
    bin_metric, check_inventory, mask_weather, node_features, sha256, spool_metric)
from src.multinode import node_windows, prepare_multinode, read_node_groups, weighted_bins
from src.preprocess import NS_SECOND, _weather_asof


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def fixture(root):
    """Independent original preparation produces the synthetic reference."""
    raw = root / 'raw'
    settings = {'include_gpu_utilization': False, 'months': ['22-01'], 'resample_minutes': 10,
        'history_hours': 1, 'horizon_minutes': 60, 'train_fraction': .7, 'val_fraction': .15,
        'min_bin_coverage': .5, 'weather_max_age_minutes': 30, 'max_hold_seconds': 300,
        'nodes': ['a', 'b'], 'time_features': True, 'min_windows_per_split': 10}
    base = datetime(2022, 1, 1, tzinfo=timezone.utc)
    for metric in (*LOCAL_METRICS, 'temp', 'humidity'):
        plugin = 'weather' if metric in ('temp', 'humidity') else 'ipmi' if metric in ('total_power', 'ambient') else 'ganglia'
        minutes = np.arange(0, 31 * 1440, 5)
        constants = {'ambient': 23., 'cpu_user': 20., 'cpu_system': 10., 'cpu_idle': 65.,
                     'cpu_wio': 5., 'mem_free': 400., 'mem_total': 1000., 'temp': 12., 'humidity': 60.}
        tables = []
        for index, node in enumerate(('a', 'b') if plugin != 'weather' else ('weather',)):
            use = np.ones(len(minutes), dtype=bool)
            if node == 'b' and metric == 'cpu_user':
                use[(minutes >= 1500) & (minutes < 1600)] = False
            chosen = minutes[use]
            values = (1000 + index * 400 + chosen / 100) if metric == 'total_power' else np.full(len(chosen), constants[metric])
            stamps = [base + timedelta(minutes=int(m)) for m in chosen]
            # Duplicate null must win and break a hold, including across Arrow batches.
            if node == 'a' and metric == 'cpu_system':
                stamps.append(base + timedelta(minutes=1000))
                values = np.r_[values, np.nan]
            columns = {'timestamp': pa.array(stamps, type=pa.timestamp('ms', tz='UTC')),
                       'value': pa.array(values)}
            if plugin != 'weather':
                columns['node'] = pa.array([node] * len(stamps)).dictionary_encode()
            tables.append(pa.table(columns))
        path = raw / 'year_month=22-01' / f'plugin={plugin}_pub' / f'metric={metric}' / 'part.parquet'
        path.parent.mkdir(parents=True)
        pq.write_table(pa.concat_tables(tables), path, row_group_size=257)
    prepared = prepare_multinode(raw, root / 'prepared_reference', settings)
    frozen = root / 'frozen'
    frozen.mkdir()
    meta = json.loads(Path(prepared['preprocessing_path']).read_text())
    shutil.copyfile(prepared['preprocessing_path'], frozen / 'preprocessing.json')
    arrays = {}
    with np.load(prepared['data_path'], allow_pickle=False) as archive:
        for name in archive.files:
            path = frozen / (name + '.npy')
            np.save(path, archive[name], allow_pickle=False)
            arrays[path.name] = {'shape': list(archive[name].shape), 'dtype': str(archive[name].dtype), 'sha256': sha256(path)}
    contract = {'arrays': arrays, 'feature_names': FEATURES, 'history_steps': 6, 'horizon_steps': 6,
        'metadata_sha256': sha256(frozen / 'preprocessing.json'), 'weather_mask_indices': [7, 8],
        'chronological_cuts': meta['split_boundaries_utc'],
        'scalers': {k: meta[k] for k in ('x_mean', 'x_std', 'y_mean', 'y_std')}}
    dump(frozen / 'contract.json', contract)
    entries = [{'path': item['path_relative_to_dataset'], 'sha256': sha256(raw / item['path_relative_to_dataset'])}
               for item in meta['source_inventory']]
    receipt_path = root / 'extraction.json'
    dump(receipt_path, {'files': entries})
    marker = root / 'protected.txt'
    marker.write_text('must remain unchanged\n')
    config = {'version': 1, 'history_steps': [6, 36, 144], 'horizon_steps': 6, 'resample_minutes': 10,
        'feature_names': FEATURES, 'weather': 'masked', 'weather_mask_indices': [7, 8],
        'scaler_policy': 'reuse_original_training_window_scalers',
        'node_policy': 'unchanged_frozen_mapping_no_filter_or_reindex',
        'reference_contract_sha256': sha256(frozen / 'contract.json'),
        'reference_settings': settings, 'chronological_cuts': meta['split_boundaries_utc'],
        'reference_counts': meta['sample_counts'], 'reference_node_count': 2,
        'grid_start_utc': '2022-01-01T00:00:00+00:00', 'grid_end_exclusive_utc': '2022-02-01T00:00:00+00:00',
        'reading_limits': {'nodes_per_shard': 1, 'arrow_batch_rows': 127, 'max_shard_rows': 100000, 'max_node_rows': 100000},
        'training_eligibility': {'minimum_windows': 1024, 'minimum_days': 7},
        'raw_files': entries, 'extraction_receipts': [{'path': 'extraction.json', 'sha256': sha256(receipt_path)}],
        'protected_inventory': {'protected.txt': {'sha256': sha256(marker)}},
        'execution_sources': {'protected.txt': {'sha256': sha256(marker)}}}
    config_path = root / 'config.json'
    dump(config_path, config)
    return config, frozen, raw, config_path


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.edges = np.arange(301, dtype=np.int64) * 600 * NS_SECOND
        self.features = np.ones((300, 13), dtype=np.float64)
        self.features[:, 0] = 1 + np.arange(300) / 100
        self.boundaries = {'train': (0, 160), 'val': (160, 230), 'test': (230, 300)}

    def test_matches_original_window_builder_for_all_lengths(self):
        self.features[150, 2] = np.nan
        self.features[250, 0] = np.nan
        actual, targets, _ = audit_node(self.features, self.edges, self.boundaries)
        for history in HISTORIES:
            expected = node_windows(self.features, self.features[:, 0], self.edges, history, 6, self.boundaries)
            for split in self.boundaries:
                indices = actual[history][split]
                np.testing.assert_array_equal(self.edges[indices], expected[split]['origin'].view('i8'))
                self.assertEqual(targets[indices].tobytes(), expected[split]['y'][:, 0].tobytes())

    def test_missing_bin_is_never_bridged(self):
        self.features[151, 1] = np.nan
        selected, _, _ = audit_node(self.features, self.edges, self.boundaries)
        self.assertNotIn(160, selected[36]['val'])
        self.assertIn(188, selected[36]['val'])
        self.assertNotIn(230, selected[144]['test'])

    def test_nodes_are_audited_separately_without_joining(self):
        a = np.full_like(self.features, np.nan)
        b = a.copy()
        a[-80:] = self.features[-80:]
        b[:80] = self.features[:80]
        for features in (a, b):
            selected, _, _ = audit_node(features, self.edges, self.boundaries)
            self.assertEqual(sum(len(v) for v in selected[144].values()), 0)

    def test_exact_boundaries_allow_past_context_and_not_crossing_targets(self):
        selected, _, _ = audit_node(self.features, self.edges, self.boundaries)
        for history in HISTORIES:
            self.assertIn(154, selected[history]['train'])
            self.assertNotIn(155, selected[history]['train'])
            self.assertIn(160, selected[history]['val'])
            self.assertIn(224, selected[history]['val'])
            self.assertNotIn(225, selected[history]['val'])
            self.assertIn(230, selected[history]['test'])
            self.assertIn(294, selected[history]['test'])

    def test_targets_are_invariant_to_history(self):
        selected, targets, _ = audit_node(self.features, self.edges, self.boundaries)
        expected = self.features[160:166, 0].mean()
        self.assertAlmostEqual(float(targets[160]), expected, places=6)
        for history in HISTORIES:
            self.assertIn(160, selected[history]['val'])
        altered = self.features.copy()
        altered[:140, 1:] = 99
        _, changed, _ = audit_node(altered, self.edges, self.boundaries)
        self.assertEqual(targets.tobytes(), changed.tobytes())

    def test_nested_sets_and_three_way_intersection(self):
        self.features[40, 7] = np.nan
        selected, _, _ = audit_node(self.features, self.edges, self.boundaries)
        matched = matched_indices(selected)
        for split in self.boundaries:
            self.assertTrue(set(selected[144][split]) <= set(selected[36][split]) <= set(selected[6][split]))
            np.testing.assert_array_equal(matched[split], selected[144][split])

    def test_exclusions_reconcile_and_multicauses_are_not_added(self):
        self.features[0, 0] = np.nan
        self.features[152, 7:9] = np.nan
        _, _, diagnostics = audit_node(self.features, self.edges, self.boundaries)
        for history in HISTORIES:
            for split, record in diagnostics[history].items():
                self.assertEqual(record['candidate_origins'], record['accepted'] + sum(record['primary_exclusions'].values()))
                self.assertEqual(record['candidate_origins'], self.boundaries[split][1] - self.boundaries[split][0])
        record = diagnostics[36]['val']
        self.assertGreater(record['multi_cause_exclusions']['history_channel:outdoor_temperature_c'], 0)
        self.assertEqual(record['multi_cause_exclusions']['history_channel:outdoor_temperature_c'],
                         record['multi_cause_exclusions']['history_channel:outdoor_humidity_percent'])

    def test_weather_is_required_before_masking(self):
        self.features[150, 7] = np.nan
        selected, _, _ = audit_node(self.features, self.edges, self.boundaries)
        self.assertNotIn(160, selected[36]['val'])

    def test_irregular_grid_is_rejected(self):
        self.edges[12] += 1
        with self.assertRaisesRegex(ValueError, 'regular grid'):
            audit_node(self.features, self.edges, self.boundaries)

    def test_mask_changes_only_two_channels_without_mutation(self):
        for steps in HISTORIES:
            x = np.arange(2 * steps * 14, dtype=np.float32).reshape(2, steps, 14)
            before = x.tobytes()
            masked = mask_weather(x)
            self.assertEqual(x.tobytes(), before)
            self.assertFalse(np.shares_memory(masked, x))
            np.testing.assert_array_equal(masked[:, :, [7, 8]], 0)
            keep = [i for i in range(14) if i not in (7, 8)]
            np.testing.assert_array_equal(masked[:, :, keep], x[:, :, keep])

    def test_identity_hash_encoding_and_order(self):
        value = IdentityDigest()
        origins = np.array([0, 600 * NS_SECOND])
        targets = np.array([1, 2], dtype=np.float32)
        value.add(0, 'node-z', origins, targets)
        expected = hashlib.sha256(b'M100 ordered (node UTF-8, origin UTC ns) v1\0')
        for origin in origins:
            expected.update((6).to_bytes(8, 'little') + b'node-z' + int(origin).to_bytes(8, 'little', signed=True))
        self.assertEqual(value.record()['ordered_sha256'], expected.hexdigest())
        with self.assertRaises(ValueError): value.add(0, 'node-z', origins, targets)
        with self.assertRaises(ValueError): IdentityDigest().add(0, 'a', origins[::-1], targets)
        with self.assertRaises(ValueError): IdentityDigest().add(0, 'a', origins, np.array([np.nan, 1]))

    def test_weather_right_edge_never_looks_forward(self):
        times = np.array([600, 1200], dtype=np.int64) * NS_SECOND
        edges = np.array([0, 600, 1200, 1800], dtype=np.int64) * NS_SECOND
        result = _weather_asof(times, np.array([5., 9.]), edges, 600 * NS_SECOND)
        self.assertTrue(np.isnan(result[0]))
        np.testing.assert_array_equal(result[1:], [5., 9.])

    def test_future_local_observation_cannot_change_completed_bin(self):
        times = np.array([0, 600], dtype=np.int64) * NS_SECOND
        edges = np.array([0, 600, 1200], dtype=np.int64) * NS_SECOND
        a, _ = weighted_bins(times, np.array([10., 20.]), edges, 300 * NS_SECOND, .5)
        b, _ = weighted_bins(times, np.array([10., 9999.]), edges, 300 * NS_SECOND, .5)
        self.assertEqual(a[0], b[0])

    def test_zero_coverage_and_training_threshold_are_reported_without_filtering(self):
        import io
        coverage = Coverage(['node-z', 'a'], self.edges)
        coverage.add(0, np.array([160]), np.array([1.], dtype=np.float32), io.BytesIO())
        result = coverage.summary()
        self.assertEqual(result['zero_coverage_nodes'], ['a'])
        self.assertEqual(result['eligible_nodes_at_least_one_window'], 1)
        self.assertEqual(result['training_eligible_nodes_if_training_split'], 0)
        self.assertEqual(len(coverage.counts), 2)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.template = Path(cls.temporary.name) / 'template'
        cls.template.mkdir()
        fixture(cls.template)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project'
        shutil.copytree(self.template, self.root)
        self.config_path = self.root / 'config.json'
        self.config = json.loads(self.config_path.read_text())
        self.frozen, self.raw = self.root / 'frozen', self.root / 'raw'
        self.output = self.root / 'output/experiments/v4_history/coverage_audit/v1'

    def checked(self):
        return preflight(self.root, self.config, self.frozen, self.raw)

    def test_preflight_is_read_only_and_does_not_bin(self):
        before = {str(p.relative_to(self.root)): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        with mock.patch('src.history_coverage_v4.build_binned_grid', side_effect=AssertionError('Binning forbidden')):
            checked = self.checked()
        self.assertEqual(checked[-1]['mismatches'], [])
        self.assertFalse(self.output.exists())
        self.assertEqual(before, {str(p.relative_to(self.root)): sha256(p) for p in self.root.rglob('*') if p.is_file()})

    def test_raw_hash_mismatch_is_rejected(self):
        path = self.raw / self.config['raw_files'][0]['path']
        with path.open('ab') as handle: handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Raw Parquet'):
            self.checked()

    def test_contract_and_array_hash_mismatch_are_rejected(self):
        for filename in ('contract.json', 'y_train.npy'):
            path = self.frozen / filename
            old = path.read_bytes()
            path.write_bytes(old + b'changed')
            try:
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    self.checked()
            finally:
                path.write_bytes(old)

    def test_protected_file_changes_are_rejected(self):
        (self.root / 'protected.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Preservation failed'):
            self.checked()

    def test_scaler_policy_and_target_protocol_cannot_change(self):
        for key, value in (('scaler_policy', 'fit_each_history'), ('horizon_steps', 36),
                           ('weather_mask_indices', [6, 7, 8]), ('history_steps', [6, 24, 144])):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                preflight(self.root, config, self.frozen, self.raw)

    def test_bounded_reader_matches_original_dedup_and_binning(self):
        _, meta, edges, _, _ = self.checked()
        names = meta['nodes']
        files = [self.raw / e['path'] for e in self.config['raw_files'] if '/metric=cpu_system/' in e['path']]
        output = np.full((2, len(edges) - 1), np.nan)
        with tempfile.TemporaryDirectory() as temporary:
            quality = bin_metric(files, 'cpu_system', names, edges, Path(temporary), meta['settings'], self.config['reading_limits'], output)
        for index, times, values, _ in read_node_groups(files, 'cpu_system', names, [(edges[0], edges[-1])]):
            expected, _ = weighted_bins(times, values, edges, 300 * NS_SECOND, .5)
            self.assertEqual(output[index].tobytes(), expected.tobytes())
        self.assertEqual(quality['a']['duplicates_keep_last'], 1)
        self.assertEqual(quality['a']['nonfinite_rows_after_dedup'], 1)

    def test_reader_aborts_instead_of_dropping_rows(self):
        _, meta, edges, _, _ = self.checked()
        files = [self.raw / self.config['raw_files'][0]['path']]
        limits = self.config['reading_limits'] | {'max_shard_rows': 1}
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, 'limit exceeded'):
            spool_metric(files, meta['nodes'], edges, Path(temporary), limits)

    def test_synthetic_end_to_end_replay_coverage_and_preservation(self):
        before = {str(p.relative_to(self.root)): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        scalers = copy.deepcopy(self.checked()[0]['scalers'])
        result = _run_coverage(self.root, self.config, self.frozen, self.raw, self.output, self.config_path, self.checked())
        self.assertEqual(result['status'], 'coverage_complete')
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(json.loads((self.output / 'matched_one_hour_reference.json').read_text())['scalers'], scalers)
        for split in ('train', 'val', 'test'):
            self.assertEqual(summary['cohorts']['native_steps6'][split]['windows'], self.config['reference_counts'][split])
            self.assertEqual(summary['cohorts']['matched'][split]['window_identity'], summary['cohorts']['native_steps144'][split]['window_identity'])
            rows = np.fromfile(self.output / f'matched_{split}.bin', dtype=WINDOW_DTYPE)
            self.assertTrue(np.isfinite(rows['target_scaled']).all())
            self.assertEqual(len(rows), result['matched_windows'][split])
            old_nodes = np.load(self.frozen / f'node_{split}.npy', allow_pickle=False)
            old_origins = np.load(self.frozen / f'origin_{split}.npy', allow_pickle=False).view('i8')
            old_targets = np.load(self.frozen / f'y_{split}.npy', allow_pickle=False)[:, 0]
            lookup = {(str(n), int(t)): y for n, t, y in zip(old_nodes, old_origins, old_targets)}
            expected = np.array([lookup[(['a', 'b'][r['node_index']], int(r['origin_utc_ns']))] for r in rows], dtype=np.float32)
            self.assertEqual(rows['target_scaled'].tobytes(), expected.tobytes())
        with mock.patch('src.history_coverage_v4.build_binned_grid', side_effect=AssertionError('Verification must not rebin')):
            self.assertEqual(verify_output(self.root, self.config, self.frozen, self.raw, self.output, self.config_path)['status'], 'verified_complete')
        self.assertEqual(before, {name: sha256(self.root / name) for name in before})
        self.assertFalse(any(self.output.glob('.scratch-*')))
        with (self.output / 'summary.json').open('ab') as handle: handle.write(b' ')
        with self.assertRaisesRegex(ValueError, 'Output artifact changed'):
            verify_output(self.root, self.config, self.frozen, self.raw, self.output, self.config_path)

    def test_output_tampering_and_partial_outputs_fail_closed(self):
        self.output.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            verify_output(self.root, self.config, self.frozen, self.raw, self.output, self.config_path)
        with self.assertRaises(FileExistsError):
            _run_coverage(self.root, self.config, self.frozen, self.raw, self.output, self.config_path, self.checked())

    def test_reference_gate_failure_leaves_no_longer_history_results(self):
        with mock.patch.object(ReferenceData, 'verify_node', side_effect=ValueError('One-hour reference target mismatch')):
            with self.assertRaisesRegex(ValueError, 'reference target mismatch'):
                _run_coverage(self.root, self.config, self.frozen, self.raw, self.output, self.config_path, self.checked())
        self.assertTrue((self.output / 'manifest.json').exists())
        self.assertFalse((self.output / 'summary.json').exists())
        self.assertFalse((self.output / 'completed.json').exists())

    def test_reference_rejects_identity_target_and_input_discrepancies(self):
        contract, meta, edges, boundaries, _ = self.checked()
        ref = ReferenceData(self.frozen, contract, meta)
        a = ref.arrays['train']
        # Small standalone reference data exercises the production comparison method.
        features = np.ones((len(edges) - 1, 13))
        selected, targets, _ = audit_node(features, edges, boundaries, histories=(6,))
        bad_selection = copy.deepcopy(selected[6])
        bad_selection['train'] = bad_selection['train'][1:]
        with self.assertRaisesRegex(ValueError, 'identity/order mismatch'):
            ref.verify_node(0, features, edges, bad_selection, targets)
        lower, upper = ref.offsets['train'][:2]
        origins = a['origin'][lower:upper].view('i8')
        selected[6]['train'] = np.searchsorted(edges, origins)
        with self.assertRaisesRegex(ValueError, 'target mismatch'):
            ref.verify_node(0, features, edges, selected[6], targets)
        expected_y = a['y'][lower:upper, 0]
        with mock.patch.object(ref, 'scaled_targets', return_value=expected_y):
            with self.assertRaisesRegex(ValueError, 'input mismatch'):
                ref.verify_node(0, features, edges, selected[6], targets)

    def test_output_location_and_existing_directory_are_protected(self):
        with self.assertRaises(ValueError): _output_path(self.root, self.frozen)
        self.output.mkdir(parents=True)
        args = ['--config', str(self.config_path), '--reference-data-dir', str(self.frozen),
                '--raw-data-dir', str(self.raw), '--output-dir', str(self.output), '--preflight-only']
        with mock.patch('src.history_coverage_v4.ROOT', self.root), mock.patch('src.history_coverage_v4.preflight', side_effect=AssertionError('Must reject first')):
            self.assertEqual(main(args), 1)


if __name__ == '__main__':
    unittest.main()
