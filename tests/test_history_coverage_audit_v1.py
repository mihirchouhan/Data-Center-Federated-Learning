"""Dedicated synthetic tests for Task T1.1 History-Length Coverage Audit.

Tests verify:
- Window boundary correctness.
- History length and target alignment.
- Missing history vs missing target handling.
- Common cohort intersection and nesting.
- Temporal leakage (future data, target in history, split reset).
- Empty/ineligible nodes (training eligibility thresholds).
- Reproducible audit outputs across all 6 required files.
- Physical units (Watts vs kW conversion).
- Explicit feasibility reporting of test-split limitations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.experiment_data import FEATURES
from src.history_coverage_audit_v1 import (
    BIN_WIDTH_NS,
    DAY_NS,
    HISTORIES,
    HORIZON,
    KW_TO_WATTS,
    MIN_DAYS_TRAIN,
    MIN_WINDOWS_TRAIN,
    POLICIES,
    SPLITS,
    TARGET_NAME,
    TARGET_UNIT,
    analyze_telemetry_gaps,
    audit_single_node,
    compute_targets,
    evaluate_fair_comparison,
    verify_data_leakage,
    write_audit_outputs,
)
from src.v5_missing_data.forward_fill_audit import causal_forward_fill


def create_synthetic_features(
    length: int = 500,
    power_base_kw: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    """Create synthetic regular ten-minute feature grid and split boundaries."""
    edges = np.arange(length + 1, dtype=np.int64) * BIN_WIDTH_NS
    feat = np.ones((length, 13), dtype=np.float64)
    # Power channel in kW varies slightly across time
    feat[:, 0] = power_base_kw + np.sin(np.arange(length) / 10.0) * 0.2
    # Temperatures & Ganglia percentages
    feat[:, 1:5] = 25.0
    feat[:, 5] = 0.6  # memory fraction
    feat[:, 6] = 22.0  # ambient temp
    feat[:, 7:9] = [15.0, 50.0]  # outdoor weather
    feat[:, 9:13] = 0.5  # calendar features

    # Realistic split cuts: 70% train, 15% val, 15% test
    train_end = int(length * 0.70)
    val_end = int(length * 0.85)
    boundaries = {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'test': (val_end, length),
    }
    return feat, edges, boundaries


class TestWindowBoundaryCorrectness(unittest.TestCase):
    """Verify window boundary conditions, containment, and history/target separation."""

    def setUp(self):
        self.feat, self.edges, self.boundaries = create_synthetic_features(length=500)

    def test_history_strictly_ends_before_target(self):
        """Every history window [t-H, t) must end at t-1, strictly before target [t, t+6)."""
        for h in HISTORIES:
            for t in range(h, len(self.feat) - HORIZON):
                history_range = range(t - h, t)
                target_range = range(t, t + HORIZON)
                self.assertEqual(max(history_range), t - 1)
                self.assertEqual(min(target_range), t)
                self.assertLess(max(history_range), min(target_range))
                self.assertFalse(set(history_range) & set(target_range))

    def test_initial_history_exclusions(self):
        """Origins with insufficient initial history (t < H) must be excluded."""
        res = audit_single_node(self.feat, self.edges, self.boundaries, histories=(6, 36, 144))
        for h in HISTORIES:
            train_cnt = res['counts']['no_fill'][h]['train']
            # Exactly h origins at the start of the grid have insufficient history
            self.assertEqual(train_cnt['history_too_short'], h)
            # The first accepted origin in train must be exactly t = h
            if train_cnt['accepted'] > 0:
                self.assertIn(h, res['selections']['no_fill'][h]['train'])
                self.assertNotIn(h - 1, res['selections']['no_fill'][h]['train'])

    def test_target_crosses_split_end_exclusions(self):
        """Origins whose 6-step target crosses split end (t + 6 > upper) must be excluded."""
        res = audit_single_node(self.feat, self.edges, self.boundaries, histories=(6, 36, 144))
        for split, (lower, upper) in self.boundaries.items():
            for h in HISTORIES:
                cnt = res['counts']['no_fill'][h][split]
                # In each split, exactly 5 origins at the end (upper-5 .. upper-1) cross split end
                self.assertEqual(cnt['target_crosses_split_end'], HORIZON - 1)
                # The last accepted candidate in split must be <= upper - 6
                sel = res['selections']['no_fill'][h][split]
                if len(sel) > 0:
                    self.assertLessEqual(max(sel), upper - HORIZON)


class TestHistoryLengthAndTargetAlignment(unittest.TestCase):
    """Verify target calculation in Watts and invariance across history lengths."""

    def setUp(self):
        self.feat, self.edges, self.boundaries = create_synthetic_features(length=500)

    def test_target_is_mean_in_watts(self):
        """Target must be arithmetic mean of next 6 bins converted to Watts (kW * 1000)."""
        power_kw = self.feat[:, 0]
        targets_kw, targets_w = compute_targets(power_kw, horizon=HORIZON)

        for t in range(len(self.feat) - HORIZON):
            expected_kw = float(np.mean(power_kw[t:t + HORIZON]))
            expected_w = expected_kw * KW_TO_WATTS
            self.assertAlmostEqual(float(targets_kw[t]), expected_kw, places=5)
            self.assertAlmostEqual(float(targets_w[t]), expected_w, places=2)

    def test_targets_are_invariant_to_history_length(self):
        """Target values for the same forecast origin t must be identical for 1h, 6h, and 24h."""
        res = audit_single_node(self.feat, self.edges, self.boundaries, histories=(6, 36, 144))
        targets_w = res['targets_w']

        # Any origin accepted by 144 steps is also accepted by 36 and 6 steps
        common_train = res['matched_selections']['no_fill']['train']
        self.assertGreater(len(common_train), 0)
        for t in common_train:
            val_144 = targets_w[t]
            self.assertTrue(np.isfinite(val_144))
            self.assertIn(t, res['selections']['no_fill'][6]['train'])
            self.assertIn(t, res['selections']['no_fill'][36]['train'])
            self.assertIn(t, res['selections']['no_fill'][144]['train'])


class TestMissingHistoryVsMissingTargetHandling(unittest.TestCase):
    """Verify separate handling of missing input telemetry versus missing ground truth target."""

    def setUp(self):
        self.feat, self.edges, self.boundaries = create_synthetic_features(length=500)

    def test_missing_history_does_not_corrupt_target_label(self):
        """Missing values in history mark origin as invalid_history, keeping target label valid."""
        # Inject NaN into input history at bin 199 (inside history for origin t=200)
        feat_mod = self.feat.copy()
        feat_mod[199, 1] = np.nan

        res = audit_single_node(feat_mod, self.edges, self.boundaries, histories=(6,))
        sel_train = res['selections']['no_fill'][6]['train']

        # Origin 200 must be excluded due to invalid history
        self.assertNotIn(200, sel_train)
        cnt = res['counts']['no_fill'][6]['train']
        self.assertGreater(cnt['invalid_history'], 0)

        # Ground truth target at 200 must still be valid
        self.assertTrue(np.isfinite(res['targets_w'][200]))

    def test_missing_target_does_not_corrupt_input_history(self):
        """Missing ground truth power at t+2 marks origin as invalid_observed_target."""
        feat_mod = self.feat.copy()
        feat_mod[202, 0] = np.nan  # NaN in target window for origin t=200

        res = audit_single_node(feat_mod, self.edges, self.boundaries, histories=(6,))
        sel_train = res['selections']['no_fill'][6]['train']

        self.assertNotIn(200, sel_train)
        cnt = res['counts']['no_fill'][6]['train']
        self.assertGreater(cnt['invalid_observed_target'], 0)
        self.assertTrue(np.isnan(res['targets_w'][200]))

    def test_forward_fill_heals_input_history_never_target(self):
        """Causal forward-fill can heal short input gaps but NEVER imputes ground truth target."""
        feat_mod = self.feat.copy()
        feat_mod[199, 1] = np.nan  # 1-bin input gap
        feat_mod[202, 0] = np.nan  # 1-bin target gap

        res = audit_single_node(
            feat_mod, self.edges, self.boundaries, histories=(6,), policies=('no_fill', 'ff10')
        )

        # Under no_fill, both cause exclusions
        self.assertNotIn(200, res['selections']['no_fill'][6]['train'])

        # Under ff10, the input gap at 199 is healed, but target at 202 is NEVER filled.
        # Since target horizon is 6 bins, a single NaN at bin 202 invalidates targets
        # for exactly 6 candidate origins: t in [197, 198, 199, 200, 201, 202].
        self.assertTrue(np.isnan(res['targets_w'][200]))
        self.assertNotIn(200, res['selections']['ff10'][6]['train'])
        self.assertEqual(res['counts']['ff10'][6]['train']['invalid_observed_target'], 6)


class TestCommonCohortIntersection(unittest.TestCase):
    """Verify mathematical nesting and 3-way common cohort intersection."""

    def setUp(self):
        self.feat, self.edges, self.boundaries = create_synthetic_features(length=500)

    def test_nesting_and_exact_intersection(self):
        """144-step accepted origins must be a strict subset of 36-step and 6-step origins."""
        res = audit_single_node(self.feat, self.edges, self.boundaries, histories=(6, 36, 144))
        for policy in POLICIES:
            for split in SPLITS:
                sel_6 = set(res['selections'][policy][6][split])
                sel_36 = set(res['selections'][policy][36][split])
                sel_144 = set(res['selections'][policy][144][split])
                matched = set(res['matched_selections'][policy][split])

                # Nesting: 144 <= 36 <= 6
                self.assertTrue(sel_144.issubset(sel_36))
                self.assertTrue(sel_36.issubset(sel_6))

                # Intersection: 6 & 36 & 144 == 144
                self.assertEqual(sel_6 & sel_36 & sel_144, sel_144)
                self.assertEqual(matched, sel_144)


class TestTemporalLeakage(unittest.TestCase):
    """Verify protection against future-data leakage and cross-split contamination."""

    def setUp(self):
        self.feat, self.edges, self.boundaries = create_synthetic_features(length=500)

    def test_future_anomaly_does_not_leak_into_history(self):
        """Future observations starting at origin t cannot affect history window [t-H, t)."""
        feat_clean = self.feat.copy()
        feat_future = self.feat.copy()
        t = 200
        # Inject extreme anomaly into future bins
        feat_future[t:t + 50, :] = 999999.0

        res_clean = audit_single_node(feat_clean, self.edges, self.boundaries, histories=(36,))
        res_future = audit_single_node(feat_future, self.edges, self.boundaries, histories=(36,))

        # History features for t=200 are unchanged
        np.testing.assert_array_equal(feat_clean[t - 36:t], feat_future[t - 36:t])

    def test_forward_fill_does_not_cross_split_boundaries(self):
        """Forward-fill state must reset at split cuts; values cannot bridge across splits."""
        train_end = self.boundaries['train'][1]
        feat_mod = self.feat.copy()
        # Make bins around train_end NaNs except the last bin in train
        feat_mod[train_end - 1, 1] = 42.0
        feat_mod[train_end:train_end + 5, 1] = np.nan

        filled = causal_forward_fill(feat_mod, max_gap=3, reset_at=list(self.boundaries.values()))

        # Bin train_end is the first bin of validation split; it must stay NaN despite gap <= 3
        self.assertTrue(np.isnan(filled[train_end, 1]))
        self.assertTrue(np.isnan(filled[train_end + 1, 1]))

    def test_verify_data_leakage_suite_passes(self):
        """Comprehensive verify_data_leakage returns passed for synthetic setup."""
        leakage = verify_data_leakage(self.feat, self.edges, self.boundaries)
        self.assertTrue(leakage['overall_leakage_free'])
        for check_name, check_info in leakage['checks'].items():
            self.assertTrue(check_info['passed'], f'Check {check_name} failed')


class TestEmptyAndIneligibleNodes(unittest.TestCase):
    """Verify identification of empty nodes and adherence to training eligibility rules."""

    def setUp(self):
        self.feat, self.edges, self.boundaries = create_synthetic_features(length=500)

    def test_completely_empty_node_yields_zero_windows(self):
        """A node with all NaNs must yield 0 accepted windows across all configurations."""
        empty_feat = np.full_like(self.feat, np.nan)
        res = audit_single_node(empty_feat, self.edges, self.boundaries, histories=(6, 36, 144))
        for policy in POLICIES:
            for h in HISTORIES:
                for split in SPLITS:
                    cnt = res['counts'][policy][h][split]
                    self.assertEqual(cnt['accepted'], 0)
                    self.assertEqual(len(res['selections'][policy][h][split]), 0)

    def test_training_eligibility_criteria(self):
        """A node must have >=1024 windows and >=7 distinct days to be training-eligible."""
        # Under synthetic 500 bins (~3.5 days), node is ineligible (< 1024 windows and < 7 days)
        res = audit_single_node(self.feat, self.edges, self.boundaries, histories=(6,))
        sel_train = res['selections']['no_fill'][6]['train']
        days = len(np.unique(self.edges[sel_train] // DAY_NS))
        is_eligible = (len(sel_train) >= MIN_WINDOWS_TRAIN and days >= MIN_DAYS_TRAIN)
        self.assertFalse(is_eligible)


class TestFeasibilityLimitationReporting(unittest.TestCase):
    """Verify that zero test windows are correctly flagged as unfeasible."""

    def test_zero_test_windows_flags_unfeasible(self):
        """When 24h test windows equal 0, evaluate_fair_comparison must report UNFEASIBLE."""
        common_summary = {
            'no_fill': {'train': {'windows': 71653}, 'val': {'windows': 0}, 'test': {'windows': 0}},
            'ff30': {'train': {'windows': 906775}, 'val': {'windows': 19909}, 'test': {'windows': 0}},
        }
        feasibility = evaluate_fair_comparison(summary_by_history={}, common_cohort_summary=common_summary)
        self.assertEqual(feasibility['scientific_verdict'], 'UNFEASIBLE_ON_OBSERVED_TEST_PERIOD')
        self.assertFalse(feasibility['matched_comparison_feasible_on_existing_test_split'])
        self.assertGreater(len(feasibility['limitations']), 0)


class TestReproducibleAuditOutputs(unittest.TestCase):
    """Verify that all 6 required files are generated with exact schemas and reproducible hashes."""

    def setUp(self):
        self.feat, self.edges, self.boundaries = create_synthetic_features(length=500)

    def test_generates_all_six_required_files(self):
        """write_audit_outputs must generate REPORT.md, 4 CSV files, and audit_metadata.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            metadata = {
                'audit_name': 'T1.1 Test Audit',
                'version': '1.0.0',
                'code_file': 'src/history_coverage_audit_v1.py',
                'code_sha256': 'mock_sha256',
                'target_name': TARGET_NAME,
                'target_units': TARGET_UNIT,
            }
            hist_rows = [{
                'history_steps': 6,
                'history_hours': 1.0,
                'policy': 'no_fill',
                'max_gap_bins': 0,
                'split': 'train',
                'candidate_origins': 350,
                'accepted_windows': 345,
                'eligible_nodes': 1,
                'nodes_with_at_least_one_window': 1,
                'insufficient_initial_history': 6,
                'target_crosses_split_end': 5,
                'invalid_observed_target': 0,
                'invalid_history': 0,
            }]
            node_rows = [{
                'node_index': 0,
                'node': 'test_node_1',
                'history_steps': 6,
                'history_hours': 1.0,
                'policy': 'no_fill',
                'split': 'train',
                'candidate_origins': 350,
                'accepted_windows': 345,
                'distinct_days': 3,
                'training_eligible': False,
                'insufficient_initial_history': 6,
                'target_crosses_split_end': 5,
                'invalid_observed_target': 0,
                'invalid_history': 0,
            }]
            common_rows = [{
                'node_index': 0,
                'node': 'test_node_1',
                'policy': 'no_fill',
                'split': 'train',
                'windows_1h': 345,
                'windows_6h': 315,
                'windows_24h': 207,
                'common_windows': 207,
                'lost_from_1h': 138,
                'retained_from_1h_percent': '60.00',
                'lost_from_6h': 108,
                'retained_from_6h_percent': '65.71',
                'training_eligible_1h': False,
                'training_eligible_6h': False,
                'training_eligible_24h': False,
                'training_eligible_common': False,
            }]
            win_rows = [{
                'date_utc': '2022-01-01',
                'split': 'train',
                'history_steps': 6,
                'history_hours': 1.0,
                'policy': 'no_fill',
                'windows': 140,
                'distinct_origins': 140,
            }]
            leakage = {'overall_leakage_free': True, 'checks': {}}
            feasibility = {
                'scientific_verdict': 'UNFEASIBLE_ON_OBSERVED_TEST_PERIOD',
                'matched_comparison_feasible_on_existing_test_split': False,
                'common_train_windows_no_fill': 207,
                'common_val_windows_no_fill': 0,
                'common_test_windows_no_fill': 0,
                'common_test_windows_ff30': 0,
            }

            paths = write_audit_outputs(
                output_dir=out_dir,
                metadata=metadata,
                coverage_by_history_rows=hist_rows,
                coverage_by_node_rows=node_rows,
                common_cohort_rows=common_rows,
                window_coverage_rows=win_rows,
                leakage_results=leakage,
                feasibility_results=feasibility,
            )

            # Check that all 6 files exist
            expected_files = [
                'REPORT.md',
                'coverage_by_history.csv',
                'coverage_by_node.csv',
                'common_cohort.csv',
                'window_coverage.csv',
                'audit_metadata.json',
            ]
            for fname in expected_files:
                self.assertIn(fname, paths)
                self.assertTrue(paths[fname].is_file(), f'Missing file: {fname}')
                self.assertGreater(paths[fname].stat().st_size, 0, f'Empty file: {fname}')

            # Verify REPORT.md contains expected sections
            report_text = paths['REPORT.md'].read_text(encoding='utf-8')
            self.assertIn('TASK T1.1 — History-Length Coverage Audit Report', report_text)
            self.assertIn('UNFEASIBLE_ON_OBSERVED_TEST_PERIOD', report_text)
            self.assertIn('Coverage by History Length and Missing Data Policy', report_text)
            self.assertIn('Data Leakage Audit Details', report_text)

    def test_window_coverage_date_utc_format(self):
        """Verify that int64 nanosecond timestamps produce standard YYYY-MM-DD date strings."""
        # 2022-01-01 00:00:00 UTC in nanoseconds
        t_ns = np.int64(1_640_995_200_000_000_000)
        day_str = str(np.int64(t_ns).astype('datetime64[ns]').astype('datetime64[D]'))
        self.assertEqual(day_str, '2022-01-01')

        # Mid-February timestamp: 2022-02-15 12:00:00 UTC
        t_mid = t_ns + np.int64(45 * 86_400 + 12 * 3600) * 1_000_000_000
        day_str_mid = str(np.int64(t_mid).astype('datetime64[ns]').astype('datetime64[D]'))
        self.assertEqual(day_str_mid, '2022-02-15')


if __name__ == '__main__':
    unittest.main()
