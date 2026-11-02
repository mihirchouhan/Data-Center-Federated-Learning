"""Beginner-readable synthetic tests. No raw-data scan or model execution."""
from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.experiment_data import FEATURES as FROZEN_FEATURES
from src.history_data_v4 import ReferenceData, sha256
from src.multinode import node_windows
from src.v5_missing_data.diagnose import masked_coverage
from src.v5_missing_data.forward_fill_audit import (
    GAP_LIMITS, HISTORIES, causal_forward_fill, count_eligible, eligible_origins,
    main, replay_reference_node, run_audit, verify_output,
)


def features(length=300):
    """One node, thirteen feature columns; power varies to test target rounding."""
    x = np.ones((length, 13), dtype=np.float64)
    x[:, 0] = .7 + np.arange(length) / 777
    return x


class ForwardFillTests(unittest.TestCase):
    def fill(self, values, limit):
        x = features(len(values))
        x[:, 1] = values
        return causal_forward_fill(x, limit, [(0, len(x))])[:, 1]

    def test_one_bin_gap_is_filled_with_limit_one(self):
        np.testing.assert_equal(self.fill([10, np.nan, 20], 1), [10, 10, 20])

    def test_two_bin_gap_is_not_both_filled_with_limit_one(self):
        np.testing.assert_equal(self.fill([10, np.nan, np.nan, 20], 1), [10, 10, np.nan, 20])

    def test_two_bin_gap_is_filled_with_limit_two(self):
        np.testing.assert_equal(self.fill([10, np.nan, np.nan, 20], 2), [10, 10, 10, 20])

    def test_long_gap_does_not_restart_age_from_imputed_values(self):
        for limit in GAP_LIMITS:
            result = self.fill([10] + [np.nan] * 8, limit)
            np.testing.assert_equal(result[1:limit + 1], 10)
            self.assertTrue(np.isnan(result[limit + 1:]).all())

    def test_new_observed_value_restarts_age(self):
        np.testing.assert_equal(self.fill([10, np.nan, np.nan, 20, np.nan], 1), [10, 10, np.nan, 20, 20])

    def test_leading_gaps_remain_missing(self):
        np.testing.assert_equal(self.fill([np.nan, np.nan, 20, np.nan], 3), [np.nan, np.nan, 20, 20])

    def test_future_values_never_fill_an_earlier_gap(self):
        before = self.fill([np.nan, 10, np.nan, np.nan, 20], 1)
        after = self.fill([np.nan, 10, np.nan, np.nan, 9999], 1)
        np.testing.assert_equal(before[:4], after[:4])

    def test_different_nodes_do_not_share_fill_state(self):
        a, b = features(3), features(3)
        a[:, 1], b[:, 1] = [10, 20, np.nan], [np.nan, np.nan, 99]
        filled_a = causal_forward_fill(a, 2, [(0, 3)])
        filled_b = causal_forward_fill(b, 2, [(0, 3)])
        self.assertEqual(filled_a[-1, 1], 20)
        self.assertTrue(np.isnan(filled_b[:2, 1]).all())
        with self.assertRaises(ValueError):
            causal_forward_fill(np.stack([a, b]), 2, [(0, 3)])

    def test_channels_do_not_borrow_other_channels_values(self):
        x = features(3)
        x[:, 1], x[:, 2] = [10, np.nan, np.nan], [50, np.nan, 60]
        result = causal_forward_fill(x, 2, [(0, 3)])
        self.assertEqual(result[1, 1], 10)
        self.assertEqual(result[1, 2], 50)

    def test_fill_state_resets_at_both_split_boundaries(self):
        x = features(12)
        x[:, 1] = [10, 10, 10, 10, np.nan, np.nan, 20, np.nan, np.nan, 30, np.nan, 40]
        result = causal_forward_fill(x, 3, [(0, 4), (4, 8), (8, 12)])
        self.assertTrue(np.isnan(result[4:6, 1]).all())
        self.assertEqual(result[7, 1], 20)
        self.assertTrue(np.isnan(result[8, 1]))

    def test_input_arrays_are_never_modified(self):
        x = features(6)
        x[2:4, 1] = np.nan
        before = x.tobytes()
        x.setflags(write=False)
        filled = causal_forward_fill(x, 2, [(0, 6)])
        self.assertEqual(x.tobytes(), before)
        self.assertFalse(np.shares_memory(x, filled))

    def test_masked_weather_is_neither_filled_nor_required(self):
        x = features()
        x[1:, 7:9] = np.nan
        result = causal_forward_fill(x, 3, [(0, 160), (160, 230), (230, 300)])
        np.testing.assert_equal(result[:, 7:9], x[:, 7:9])
        counts = count_eligible(result, {'train': (0, 160), 'val': (160, 230), 'test': (230, 300)},
                                6, observed_power=x[:, 0])
        self.assertEqual(counts['train']['accepted'], 149)
        self.assertEqual(counts['val']['accepted'], 65)

    def test_invalid_limits_and_reset_ranges_are_rejected(self):
        for bad in (0, 4, -1, True, 1.5):
            with self.subTest(limit=bad), self.assertRaises(ValueError):
                causal_forward_fill(features(6), bad, [(0, 6)])
        for ranges in ([], [(1, 6)], [(0, 3), (2, 6)], [(0, 2), (3, 6)], [(0, 7)]):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                causal_forward_fill(features(6), 1, ranges)


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.boundaries = {'train': (0, 160), 'val': (160, 230), 'test': (230, 300)}

    def test_target_power_is_never_imputed_even_when_input_power_is_filled(self):
        original = features()
        original[170, 0] = np.nan
        observed_power = original[:, 0].copy()
        observed_before = observed_power.tobytes()
        filled = causal_forward_fill(original, 3, self.boundaries.values())
        self.assertTrue(np.isfinite(filled[170, 0]))
        for history in HISTORIES:
            selected, counts = eligible_origins(filled, self.boundaries, history, observed_power=observed_power)
            self.assertNotIn(166, selected['val'])  # Future [166,172) includes missing power at 170.
            self.assertIn(176, selected['val'])     # Imputation is permitted in past input context only.
            self.assertGreater(counts['val']['invalid_observed_target'], 0)
        self.assertEqual(observed_power.tobytes(), observed_before)
        with self.assertRaises(TypeError):
            count_eligible(filled, self.boundaries, 6)  # Original target vector is mandatory.

    def test_future_non_power_features_do_not_disqualify_observed_targets(self):
        original = features()
        original[170, 1] = np.nan
        selected, _ = eligible_origins(original, self.boundaries, 6, observed_power=original[:, 0])
        self.assertIn(166, selected['val'])

    def test_targets_end_at_cut_but_never_cross_it_and_past_context_is_allowed(self):
        x = features()
        for history in HISTORIES:
            selected, _ = eligible_origins(x, self.boundaries, history, observed_power=x[:, 0])
            self.assertIn(154, selected['train'])
            self.assertNotIn(155, selected['train'])
            self.assertIn(160, selected['val'])
            self.assertIn(224, selected['val'])
            self.assertNotIn(225, selected['val'])
            self.assertIn(230, selected['test'])
            self.assertIn(294, selected['test'])

    def test_no_fill_agrees_with_existing_v5_eligibility(self):
        x = features()
        x[20:25, 1] = np.nan
        x[100, 0] = np.nan
        x[:, 7:9] = np.nan
        for history in HISTORIES:
            actual = count_eligible(x, self.boundaries, history, observed_power=x[:, 0])
            expected = masked_coverage(x, self.boundaries, history)
            for split in self.boundaries:
                self.assertEqual(actual[split]['accepted'], expected[split]['accepted'])
                self.assertEqual(actual[split]['candidate_origins'], actual[split]['accepted'] +
                                 sum(actual[split][key] for key in ('history_too_short', 'target_crosses_split_end',
                                                                  'invalid_observed_target', 'invalid_history')))

    def test_eligibility_is_nested_for_histories_and_fill_limits(self):
        x = features()
        x[20:25, 1] = np.nan
        previous = {history: eligible_origins(x, self.boundaries, history, observed_power=x[:, 0])[0]
                    for history in HISTORIES}
        for gap in GAP_LIMITS:
            filled = causal_forward_fill(x, gap, self.boundaries.values())
            current = {history: eligible_origins(filled, self.boundaries, history, observed_power=x[:, 0])[0]
                       for history in HISTORIES}
            for split in self.boundaries:
                self.assertTrue(set(current[144][split]) <= set(current[36][split]) <= set(current[6][split]))
                for history in HISTORIES:
                    self.assertTrue(set(previous[history][split]) <= set(current[history][split]))
            previous = current


class ReferenceAndDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frozen = self.root / 'frozen'
        self.frozen.mkdir()
        self.boundaries = {'train': (0, 160), 'val': (160, 230), 'test': (230, 300)}
        self.edges = np.datetime64('2022-01-01', 'ns').view('i8') + np.arange(301, dtype=np.int64) * 600_000_000_000
        self.names = ['node-z', '47']  # Deliberately preserve non-lexical frozen order.
        self.originals = [features(), features()]
        self.originals[0][100, 0] = np.nan
        self.originals[0][250:, 7:9] = np.nan
        self.originals[1][20:25, 1] = np.nan
        payloads = {split: {'X': [], 'y': [], 'origin': [], 'node': []} for split in self.boundaries}
        for index, x in enumerate(self.originals):
            windows = node_windows(x, x[:, 0], self.edges, 6, 6, self.boundaries)
            for split, window in windows.items():
                raw = np.concatenate((window['X'], np.full((*window['X'].shape[:2], 1), index, dtype=np.float32)), axis=2)
                for key, value in {'X': raw, 'y': window['y'], 'origin': window['origin'],
                                   'node': np.full(len(raw), self.names[index])}.items():
                    payloads[split][key].append(value)
        arrays = {f'{key}_{split}': np.concatenate(chunks) for split, values in payloads.items() for key, chunks in values.items()}
        x_mean, x_std = arrays['X_train'].mean((0, 1), dtype=np.float64), arrays['X_train'].std((0, 1), dtype=np.float64)
        y_mean, y_std = arrays['y_train'].mean(0, dtype=np.float64), arrays['y_train'].std(0, dtype=np.float64)
        x_mean[-1], x_std[-1] = 0, 1
        x_std[x_std < 1e-12] = 1
        self.meta = {'nodes': self.names, 'x_mean': x_mean.tolist(), 'x_std': x_std.tolist(),
                     'y_mean': y_mean.tolist(), 'y_std': y_std.tolist()}
        for split in self.boundaries:
            arrays[f'X_{split}'] = ((arrays[f'X_{split}'] - x_mean) / x_std).astype(np.float32)
            arrays[f'y_{split}'] = ((arrays[f'y_{split}'] - y_mean) / y_std).astype(np.float32)
        arrays['node_names'] = np.array(self.names)
        for name, value in arrays.items():
            np.save(self.frozen / (name + '.npy'), value, allow_pickle=False)
        self.contract = {}
        self.reference = ReferenceData(self.frozen, self.contract, self.meta)
        self.expected_counts = {split: len(arrays[f'y_{split}']) for split in self.boundaries}
        self.config = {'reference_counts': self.expected_counts, 'reference_contract_sha256': 'synthetic',
                       'raw_files': [], 'reading_limits': {}, 'feature_names': FROZEN_FEATURES}
        self.config_path = self.root / 'config.json'
        self.config_path.write_text(json.dumps(self.config))
        baseline_counts = {str(h): {s: 0 for s in self.boundaries} for h in HISTORIES}
        for x in self.originals:
            for h in HISTORIES:
                for split, counts in masked_coverage(x, self.boundaries, h).items():
                    baseline_counts[str(h)][split] += counts['accepted']
        self.baseline = self.root / 'diagnostic'
        self.baseline.mkdir()
        (self.baseline / 'summary.json').write_text(json.dumps({'status': 'complete',
            'original_reference_replay': {'status': 'exact_match', 'counts': self.expected_counts},
            'masked_weather_eligible_windows': baseline_counts}))
        (self.baseline / 'masked_coverage_by_node.csv').write_text('synthetic baseline\n')
        self.output = self.root / 'output/experiments/v5_missing_data/forward_fill_audit/v1'
        self.checked = (self.contract, self.meta, self.edges, self.boundaries, {'mismatches': []})

    def patches(self):
        from contextlib import ExitStack
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch('src.v5_missing_data.forward_fill_audit.preflight', return_value=self.checked))
        stack.enter_context(mock.patch('src.v5_missing_data.forward_fill_audit.build_binned_grid',
                                      return_value=(np.arange(2), {}, {'synthetic': True})))
        stack.enter_context(mock.patch('src.v5_missing_data.forward_fill_audit.node_features',
                                      side_effect=lambda index, weather, edges: self.originals[int(index)]))
        return stack

    def run_synthetic(self):
        return run_audit(self.config_path, self.frozen, self.root / 'unused_raw', self.output,
                         self.baseline, root=self.root)

    def test_real_reference_interface_and_float32_targets_match_all_splits(self):
        before = {p.name: sha256(p) for p in self.frozen.iterdir()}
        totals = dict.fromkeys(self.boundaries, 0)
        with mock.patch.object(self.reference, 'verify_node', wraps=self.reference.verify_node) as verify:
            for index, x in enumerate(self.originals):
                for split, count in replay_reference_node(self.reference, index, x, self.edges, self.boundaries).items():
                    totals[split] += count
            self.assertEqual(verify.call_count, 2)  # Once per node, containing all three splits.
            args = verify.call_args.args
            self.assertEqual(tuple(args[3]), tuple(self.boundaries))
            self.assertEqual(args[4].shape, (300,))
            self.assertEqual(args[4].dtype, np.float32)
        self.assertEqual(totals, self.expected_counts)
        self.assertEqual(before, {p.name: sha256(p) for p in self.frozen.iterdir()})

    def test_reference_mismatch_stops_before_fill_or_summary(self):
        self.patches()
        self.originals[0][10, 2] = np.nan
        with mock.patch('src.v5_missing_data.forward_fill_audit.causal_forward_fill', side_effect=AssertionError('Fill must not start')):
            with self.assertRaisesRegex(ValueError, 'reference identity/order mismatch'):
                self.run_synthetic()
        self.assertFalse((self.output / 'summary.json').exists())
        self.assertFalse((self.output / 'completed.json').exists())

    def test_synthetic_full_driver_preserves_reference_targets_and_existing_outputs(self):
        self.patches()
        originals = {str(p.relative_to(self.root)): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        result = self.run_synthetic()
        self.assertEqual(result['original_reference_replay']['counts'], self.expected_counts)
        self.assertEqual(result['status'], 'complete')
        for gap in GAP_LIMITS:
            for split in self.boundaries:
                counts = result['filled_input_bins_by_feature_and_split'][str(gap)][split]
                self.assertEqual(counts['outdoor_temperature_c'], 0)
                self.assertEqual(counts['outdoor_humidity_percent'], 0)
        with (self.output / 'coverage_by_node.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2 * 3 * 3 * 3)
        self.assertEqual(rows[0]['node'], 'node-z')
        self.assertEqual(originals, {name: sha256(self.root / name) for name in originals})
        with mock.patch('src.v5_missing_data.forward_fill_audit.build_binned_grid', side_effect=AssertionError('No scan in verification')):
            status = verify_output(self.config_path, self.frozen, self.root / 'unused_raw', self.output,
                                   self.baseline, root=self.root)
        self.assertEqual(status['status'], 'verified_complete')
        self.assertFalse(any(self.output.glob('.ff-scratch-*')))
        with (self.output / 'summary.json').open('ab') as handle:
            handle.write(b' ')
        with self.assertRaisesRegex(ValueError, 'Changed artifact'):
            verify_output(self.config_path, self.frozen, self.root / 'unused_raw', self.output,
                          self.baseline, root=self.root)

    def test_output_exists_is_rejected_without_scanning(self):
        self.output.mkdir(parents=True)
        marker = self.output / 'keep.txt'
        marker.write_text('preserve me')
        with mock.patch('src.v5_missing_data.forward_fill_audit.preflight', side_effect=AssertionError('No preflight')):
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                self.run_synthetic()
        self.assertEqual(marker.read_text(), 'preserve me')

    def test_preflight_cli_never_builds_bins_or_creates_output(self):
        with mock.patch('src.v5_missing_data.forward_fill_audit.ROOT', self.root), \
             mock.patch('src.v5_missing_data.forward_fill_audit.preflight', return_value=self.checked), \
             mock.patch('src.v5_missing_data.forward_fill_audit.build_binned_grid', side_effect=AssertionError('No raw scan')):
            args = ['--config', str(self.config_path), '--reference-data-dir', str(self.frozen),
                    '--raw-data-dir', str(self.root / 'unused_raw'), '--output-dir', str(self.output),
                    '--baseline-diagnostic-dir', str(self.baseline), '--preflight-only']
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
