"""Focused cohort-preparation tests: temporary synthetic arrays only, no model."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.history_data_v4 import sha256
from src.v5_missing_data import test_forward_fill_audit as fixture_module
from src.v5_missing_data.forward_fill_audit import causal_forward_fill, eligible_origins
from src.v5_missing_data.build_training_cohorts import (
    POLICIES, SPLITS, PoolWriter, allocate, budget_totals, checked_output, context_specs,
    execute, fill_ages, fill_from_ages, id_dtype, load_inputs, prepare, retained_node,
    selections_for_node, verify_ids, verify_output, verify_prepared,
)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.x = np.ones((300, 13), dtype=np.float64)
        self.x[:, 0] = np.arange(300) / 777 + .7
        self.edges = np.arange(301, dtype=np.int64) * 600_000_000_000
        self.boundaries = {'train': (0, 160), 'val': (160, 230), 'test': (230, 300)}

    def test_stored_ages_reproduce_all_audited_fill_policies(self):
        self.x[10:16, 1] = np.nan
        self.x[160:162, 2] = np.nan
        self.x[:, 7:9] = np.nan
        ages = fill_ages(self.x, self.boundaries)
        self.assertEqual(ages[160, 2], -1)
        self.assertEqual(ages[14, 1], 5)
        for gap in (1, 2, 3):
            actual = fill_from_ages(self.x, ages, gap)
            expected = causal_forward_fill(self.x, gap, self.boundaries.values())
            self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertTrue(np.isnan(actual[:, 7:9]).all())

    def test_observed_target_stays_invalid_when_input_power_is_filled(self):
        self.x[170, 0] = np.nan
        selected, targets, _, _, _ = selections_for_node(self.x, self.edges, self.boundaries)
        for policy in POLICIES:
            self.assertNotIn(166, selected[policy]['val'])
        self.assertIn(176, selected['ff10']['val'])
        self.assertTrue(np.isfinite(targets[176]))

    def test_explicit_intersections_equal_nested_sets_without_changing_targets(self):
        self.x[40:44, 1] = np.nan
        selected, targets, _, common, _ = selections_for_node(self.x, self.edges, self.boundaries)
        for split in SPLITS:
            np.testing.assert_array_equal(common['all_policies'][split], selected['no_fill'][split])
            np.testing.assert_array_equal(common['fill_policies'][split], selected['ff10'][split])
            np.testing.assert_array_equal(common['ff20_ff30'][split], selected['ff20'][split])
        changed = self.x.copy()
        changed[:, 1] = 20
        _, other_targets, _, _, _ = selections_for_node(changed, self.edges, self.boundaries)
        self.assertEqual(targets.tobytes(), other_targets.tobytes())

    def test_target_at_split_end_is_allowed_and_crossing_target_is_rejected(self):
        selected, _, _, _, _ = selections_for_node(self.x, self.edges, self.boundaries)
        for policy in POLICIES:
            self.assertIn(154, selected[policy]['train'])
            self.assertNotIn(155, selected[policy]['train'])
            self.assertIn(160, selected[policy]['val'])
            self.assertIn(294, selected[policy]['test'])

    def test_non_nested_policy_membership_is_rejected(self):
        previous = {s: np.array([36, 37]) if s == 'train' else np.array([], dtype=int) for s in SPLITS}
        following = {s: np.array([36]) if s == 'train' else np.array([], dtype=int) for s in SPLITS}
        with mock.patch('src.v5_missing_data.build_training_cohorts.eligible_origins',
                        side_effect=[(previous, {}), (following, {})]):
            with self.assertRaisesRegex(ValueError, 'nesting failed'):
                selections_for_node(self.x, self.edges, self.boundaries)

    def test_wrong_stage_plan_is_rejected_without_raw_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'version': 1, 'stage': 'training'}))
            with self.assertRaisesRegex(ValueError, 'protocol differs'):
                load_inputs(root, plan, root / 'audit', root / 'frozen', root / 'raw')

    def test_wrong_node_string_index_origin_and_duplicate_ids_fail(self):
        indices = np.array([160, 161])
        ids = np.zeros(2, dtype=id_dtype(['node-z', '47']))
        ids['node_index'], ids['node'], ids['origin'] = 0, 'node-z', self.edges[indices].astype('datetime64[ns]')
        verify_ids(ids, indices, self.edges, 'node-z', 0, self.boundaries, 'val')
        for field, value in (('node', '47'), ('node_index', 1), ('origin', np.datetime64('NaT', 'ns'))):
            bad = ids.copy()
            bad[field][0] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_ids(bad, indices, self.edges, 'node-z', 0, self.boundaries, 'val')
        bad = ids.copy()
        bad['origin'][1] = bad['origin'][0]
        with self.assertRaises(ValueError):
            verify_ids(bad, indices, self.edges, 'node-z', 0, self.boundaries, 'val')

    def test_identity_target_split_is_verified_in_addition_to_length(self):
        indices = np.array([155])
        ids = np.zeros(1, dtype=id_dtype(['a']))
        ids['node'], ids['origin'] = 'a', self.edges[indices].astype('datetime64[ns]')
        with self.assertRaisesRegex(ValueError, 'boundary'):
            verify_ids(ids, indices, self.edges, 'a', 0, self.boundaries, 'train')

    def test_context_padding_keeps_past_fill_sources_but_never_future_training_bins(self):
        contexts = context_specs(self.boundaries)
        self.assertEqual(contexts['train']['feature_start'], 0)
        self.assertEqual(contexts['train']['upper'], 160)
        self.assertEqual(contexts['val']['feature_start'], 160 - 36 - 3)

    def test_budget_accounting_uses_reference_counts_and_partial_batches(self):
        protocol = {'seeds': [42], 'rounds': 3, 'clients_per_round': 2, 'batch_size': 512}
        result = budget_totals([513, 1024], protocol)['42']
        self.assertEqual(result['examples_processed_per_method_policy'], 3 * (513 + 1024))
        self.assertEqual(result['optimizer_steps_per_method_policy'], 3 * (2 + 2))

    def test_exclusive_allocation_never_overwrites_existing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'a.npy'
            a = allocate(path, (2,), '<f4')
            a[:] = [1, 2]
            a.flush()
            before = sha256(path)
            with self.assertRaises(FileExistsError):
                allocate(path, (2,), '<f4')
            self.assertEqual(sha256(path), before)
            a._mmap.close()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the existing tiny independent reference-fixture generator;
        # no tests or production scans run as an import side effect.
        self.fixture = fixture_module.ReferenceAndDriverTests('test_real_reference_interface_and_float32_targets_match_all_splits')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.root, self.frozen, self.edges, self.boundaries = f.root, f.frozen, f.edges, f.boundaries
        self.output = self.root / 'output/experiments/v5_missing_data/model_comparison/v1/cohorts'
        self.plan_path = self.root / 'prep_plan.json'
        node_counts = {}
        expected = {p: dict.fromkeys(SPLITS, 0) for p in POLICIES}
        covered = copy.deepcopy(expected)
        for i, original in enumerate(f.originals):
            for policy, gap in POLICIES.items():
                filled = original if not gap else causal_forward_fill(original, gap, self.boundaries.values())
                indices, _ = eligible_origins(filled, self.boundaries, 36, observed_power=original[:, 0])
                for split in SPLITS:
                    count = len(indices[split])
                    node_counts[policy, split, i] = count
                    expected[policy][split] += count
                    covered[policy][split] += count > 0
        self.plan = {'stage': 'cohort_preparation_only', 'execution_sources': {}, 'expected_counts': expected,
            'expected_covered_nodes': covered, 'minimum_windows': 1, 'minimum_days': 1,
            'scaler_policy': 'reuse_original_training_window_scalers',
            'budget_protocol': {'seeds': [42, 43, 44], 'rounds': 3, 'clients_per_round': 2, 'batch_size': 512}}
        counts = [node_counts['no_fill', 'train', i] for i in range(2)]
        self.plan['expected_budget_totals'] = {str(seed): {
            'examples_processed_per_method_policy': 3 * sum(counts),
            'optimizer_steps_per_method_policy': 3 * sum((n + 511) // 512 for n in counts)} for seed in (42, 43, 44)}
        self.plan_path.write_text(json.dumps(self.plan))
        scalers = {k: f.meta[k] for k in ('x_mean', 'x_std', 'y_mean', 'y_std')}
        checked = ({'scalers': scalers}, f.meta, f.edges, f.boundaries, {'mismatches': []})
        self.context = {'plan': self.plan, 'preparation': {'raw_files': [], 'reading_limits': {}, 'reference_counts': f.expected_counts},
            'reference': f.reference, 'checked': checked, 'node_counts': node_counts, 'plan_sha256': sha256(self.plan_path)}

    def patch_raw(self):
        from contextlib import ExitStack
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.decoder = stack.enter_context(mock.patch('src.v5_missing_data.build_training_cohorts.build_binned_grid',
            return_value=(np.arange(2), {}, {'synthetic': True})))
        stack.enter_context(mock.patch('src.v5_missing_data.build_training_cohorts.node_features',
            side_effect=lambda index, weather, edges: self.fixture.originals[int(index)]))

    def prepare_tiny(self):
        self.patch_raw()
        return prepare(self.root, self.context, self.plan_path, self.root / 'unused_raw', self.output)

    def test_full_synthetic_preparation_replays_reference_and_saves_common_ids(self):
        before = {str(p.relative_to(self.root)): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        contract, _ = self.prepare_tiny()
        self.assertEqual(self.decoder.call_count, 1)
        for split in ('val', 'test'):
            key = 'policy_validation_views' if split == 'val' else 'policy_test_views'
            self.assertEqual(set(contract[key].values()), {'common/all_policies/' + split})
        for split in SPLITS:
            self.assertEqual(contract['cohorts']['native/no_fill/' + split]['identity'],
                             contract['cohorts']['common/all_policies/' + split]['identity'])
        with mock.patch('src.v5_missing_data.build_training_cohorts.build_binned_grid', side_effect=AssertionError('No raw decoding during verification')):
            result = verify_prepared(self.output, self.context)
        self.assertEqual(result['status'], 'verified_complete')
        self.assertEqual(result['original_reference_replay'], self.fixture.expected_counts)
        self.assertEqual(before, {p: sha256(self.root / p) for p in before})
        self.assertFalse(any(self.output.glob('.cohort-scratch-*')))

    def test_per_node_count_mismatch_aborts_instead_of_correcting_expected_counts(self):
        self.context['node_counts']['no_fill', 'train', 0] += 1
        with self.assertRaisesRegex(ValueError, 'audited count mismatch'):
            self.prepare_tiny()
        self.assertFalse((self.output / 'completed.json').exists())

    def test_original_reference_mismatch_stops_before_cohort_acceptance(self):
        self.fixture.originals[0][10, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'reference identity/order mismatch'):
            self.prepare_tiny()
        self.assertFalse((self.output / 'contract.json').exists())

    def test_changed_targets_wrong_node_and_changed_ages_are_rejected(self):
        contract, _ = self.prepare_tiny()
        raw = self.output / contract['cohorts']['native/no_fill/train']['paths']['targets_kw']
        values = np.load(raw, mmap_mode='r+', allow_pickle=False)
        old = values[0, 0].copy()
        values[0, 0] += 1
        values.flush()
        with self.assertRaisesRegex(ValueError, 'target values'):
            verify_prepared(self.output, self.context)
        values[0, 0] = old
        values.flush()
        values._mmap.close()
        age = np.load(self.output / contract['bin_contexts']['val']['ages'], mmap_mode='r+', allow_pickle=False)
        age[0, -1, 1] = 99
        age.flush()
        with self.assertRaisesRegex(ValueError, 'fill ages differ'):
            verify_prepared(self.output, self.context)
        age._mmap.close()

    def test_training_context_cannot_contain_validation_or_test_bins(self):
        contract, _ = self.prepare_tiny()
        spec = contract['bin_contexts']['train']
        train = np.load(self.output / spec['features'], mmap_mode='r', allow_pickle=False)
        self.assertEqual(train.shape, (2, self.boundaries['train'][1], 13))
        self.assertTrue(np.array_equal(train[0], self.fixture.originals[0][:160], equal_nan=True))

    def test_overlap_padding_and_policy_evaluation_references_are_checked(self):
        contract, _ = self.prepare_tiny()
        path = self.output / contract['bin_contexts']['val']['features']
        values = np.load(path, mmap_mode='r+', allow_pickle=False)
        previous = values[0, 0, 1].copy()
        values[0, 0, 1] += 1  # Alter only duplicated lookback, not the reassembled core grid.
        values.flush()
        with self.assertRaisesRegex(ValueError, 'lookback context differs'):
            verify_prepared(self.output, self.context)
        values[0, 0, 1] = previous
        values.flush()
        values._mmap.close()
        contract['policy_test_views']['ff20'] = 'native/ff20/test'
        (self.output / 'contract.json').write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, 'identical evaluation'):
            verify_prepared(self.output, self.context)

    def test_completed_receipt_verification_and_output_preservation(self):
        self.patch_raw()
        with mock.patch('src.v5_missing_data.build_training_cohorts.load_inputs', return_value=self.context):
            result = execute(self.root, self.plan_path, self.root / 'audit', self.frozen,
                             self.root / 'unused_raw', self.output)
            self.assertEqual(result['status'], 'prepared_verified')
            self.assertEqual(verify_output(self.root, self.plan_path, self.root / 'audit', self.frozen,
                                          self.root / 'unused_raw', self.output)['status'], 'verified_complete')
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                execute(self.root, self.plan_path, self.root / 'audit', self.frozen,
                        self.root / 'unused_raw', self.output)
        self.assertTrue((self.output / 'REPORT.md').is_file())

    def test_existing_or_unsafe_output_is_not_overwritten(self):
        with self.assertRaises(ValueError):
            checked_output(self.root, self.frozen)
        self.output.mkdir(parents=True)
        self.patch_raw()
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            prepare(self.root, self.context, self.plan_path, self.root / 'unused_raw', self.output)
        self.decoder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
