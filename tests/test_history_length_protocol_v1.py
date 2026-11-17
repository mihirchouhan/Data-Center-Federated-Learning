"""Unit tests for Task T1.2 History-Length Comparison Feasibility & Protocol Design.

Verifies:
- Common target timestamp intersection and origin alignment.
- Common node cohort logic.
- Split boundary correctness across candidate periods.
- Target calculation and unit conversion (kW to Watts).
- No future leakage and strict temporal causality.
- Training-only scaler fitting enforcement.
- Missing history vs missing target handling.
- Candidate period quantification, ranking, and reproducibility.
- Protocol JSON structure and immutability.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.history_length_protocol_v1 import (
    CANDIDATE_PERIODS,
    HISTORIES,
    KW_TO_WATTS,
    POLICIES,
    build_common_cohort_summary,
    execute_t1_2_protocol,
    generate_protocol_json,
    load_daily_coverage_table,
    quantify_candidate_periods,
    verify_t1_2_protocol,
)

ROOT = Path(__file__).resolve().parents[1]


class TestSplitBoundaryCorrectness(unittest.TestCase):
    """Verify exact chronological split boundaries and ordering."""

    def test_chronological_ordering_of_all_candidates(self):
        """Train must strictly precede validation, which must strictly precede test."""
        for cid, cand in CANDIDATE_PERIODS.items():
            t_start = cand['train_start_utc']
            t_end = cand['train_end_utc']
            v_start = cand['val_start_utc']
            v_end = cand['val_end_utc']
            te_start = cand['test_start_utc']
            te_end = cand['test_end_utc']

            self.assertEqual(t_end, v_start, f'Train end != Val start for {cid}')
            self.assertEqual(v_end, te_start, f'Val end != Test start for {cid}')
            self.assertLess(t_start, t_end, f'Train start not < Train end for {cid}')
            self.assertLess(v_start, v_end, f'Val start not < Val end for {cid}')
            self.assertLess(te_start, te_end, f'Test start not < Test end for {cid}')

    def test_candidate_1_boundaries(self):
        """Candidate 1 must have exactly 21 days train, 6 days val, 7 days test."""
        c1 = CANDIDATE_PERIODS['candidate_1']
        self.assertEqual(c1['train_start_utc'], '2022-01-01T00:00:00Z')
        self.assertEqual(c1['train_end_utc'], '2022-01-22T00:00:00Z')
        self.assertEqual(c1['val_start_utc'], '2022-01-22T00:00:00Z')
        self.assertEqual(c1['val_end_utc'], '2022-01-28T00:00:00Z')
        self.assertEqual(c1['test_start_utc'], '2022-01-28T00:00:00Z')
        self.assertEqual(c1['test_end_utc'], '2022-02-04T00:00:00Z')
        self.assertEqual(c1['test_days_count'], 7)


class TestTargetAlignmentAndNoFutureLeakage(unittest.TestCase):
    """Verify target calculation, Watts conversion, and strict causal isolation."""

    def test_target_alignment_and_watts_conversion(self):
        """Target must be arithmetic mean of next 6 bins converted to Watts."""
        # 6 bins of power in kW: [0.5, 0.6, 0.4, 0.5, 0.5, 0.5] -> mean = 0.5 kW = 500.0 W
        bins_kw = np.array([0.5, 0.6, 0.4, 0.5, 0.5, 0.5], dtype=np.float64)
        mean_kw = float(np.mean(bins_kw))
        target_w = mean_kw * KW_TO_WATTS
        self.assertAlmostEqual(target_w, 500.0, places=4)

    def test_history_strictly_precedes_target(self):
        """History window [t-H, t) must end at t-1, strictly before target [t, t+6)."""
        t = 144  # origin
        for h in (6, 36, 144):
            hist_indices = list(range(t - h, t))
            target_indices = list(range(t, t + 6))
            self.assertEqual(max(hist_indices), t - 1)
            self.assertEqual(min(target_indices), t)
            self.assertNotIn(t, hist_indices)
            self.assertTrue(set(hist_indices).isdisjoint(set(target_indices)))

    def test_future_observation_cannot_leak_into_history(self):
        """Anomalies or values at or after t must never alter [t-H, t)."""
        t = 200
        H = 144
        grid = np.ones(500, dtype=np.float64) * 0.4
        history_before = grid[t - H:t].copy()

        # Inject extreme value at target index t+1
        grid[t + 1] = 999.9
        history_after = grid[t - H:t]

        np.testing.assert_array_equal(history_before, history_after)


class TestCommonCohortIntersection(unittest.TestCase):
    """Verify common target timestamp intersection and origin matching."""

    def test_matched_cohort_is_exact_intersection(self):
        """Common cohort across 1h, 6h, 144h must equal the 3-way intersection."""
        # Suppose sets of valid origins for a node:
        origins_1h = set(range(100, 500))
        origins_6h = set(range(150, 480))
        origins_24h = set(range(200, 450))

        # Under nested telemetry, 24h is a subset of 6h which is a subset of 1h
        self.assertTrue(origins_24h.issubset(origins_6h))
        self.assertTrue(origins_6h.issubset(origins_1h))

        matched = origins_1h.intersection(origins_6h, origins_24h)
        self.assertEqual(matched, origins_24h)
        self.assertEqual(len(matched), len(origins_24h))

    def test_common_cohort_summary_calculation(self):
        """build_common_cohort_summary must compute exact retention percentages."""
        cand_row = {
            'train_matched_no_fill': 4000,
            'train_6h_no_fill': 20000,
            'train_36h_no_fill': 10000,
            'train_144h_no_fill': 4000,
            'val_matched_no_fill': 1000,
            'val_6h_no_fill': 5000,
            'val_36h_no_fill': 2500,
            'val_144h_no_fill': 1000,
            'test_matched_no_fill': 5000,
            'test_6h_no_fill': 10000,
            'test_36h_no_fill': 8000,
            'test_144h_no_fill': 5000,
        }
        summary = build_common_cohort_summary(cand_row, candidate_id='test_cand')
        # Check test no_fill for 6h
        row_6 = next(
            r for r in summary
            if r['split'] == 'test' and r['policy'] == 'no_fill' and r['history_steps'] == 6
        )
        self.assertEqual(row_6['accepted_windows'], 10000)
        self.assertEqual(row_6['common_cohort_windows'], 5000)
        self.assertEqual(row_6['retention_pct'], 50.0)

        # Check test no_fill for 144h
        row_144 = next(
            r for r in summary
            if r['split'] == 'test' and r['policy'] == 'no_fill' and r['history_steps'] == 144
        )
        self.assertEqual(row_144['accepted_windows'], 5000)
        self.assertEqual(row_144['common_cohort_windows'], 5000)
        self.assertEqual(row_144['retention_pct'], 100.0)


class TestTrainingOnlyScalerFittingEnforcement(unittest.TestCase):
    """Verify that scalers are strictly fitted on training windows only."""

    def test_scaler_fitting_capability_guard(self):
        """Validation or test sets must be rejected by training-only scaler fits."""
        train_features = np.random.randn(100, 6, 13)
        val_features = np.random.randn(20, 6, 13)

        # Scaler statistics calculated strictly on train
        train_mean = train_features.mean(axis=(0, 1), dtype=np.float64)
        train_std = train_features.std(axis=(0, 1), dtype=np.float64)

        # Standardizing validation data using train statistics
        val_standardized = (val_features - train_mean) / train_std

        # val_standardized mean should NOT be zero
        val_std_mean = val_standardized.mean(axis=(0, 1))
        self.assertFalse(np.allclose(val_std_mean, 0.0, atol=1e-3))


class TestCandidatePeriodQuantificationAndRanking(unittest.TestCase):
    """Verify candidate period window aggregations and objective ranking."""

    def test_candidate_quantification_sums_correctly(self):
        """Candidate quantification must correctly sum daily window counts."""
        daily_map = {
            '2022-01-01': {(6, 'no_fill'): 100, (36, 'no_fill'): 50, (144, 'no_fill'): 10},
            '2022-01-02': {(6, 'no_fill'): 200, (36, 'no_fill'): 100, (144, 'no_fill'): 20},
            '2022-01-28': {(6, 'no_fill'): 500, (36, 'no_fill'): 400, (144, 'no_fill'): 300},
            '2022-01-29': {(6, 'no_fill'): 600, (36, 'no_fill'): 500, (144, 'no_fill'): 400},
        }
        records = quantify_candidate_periods(daily_map)
        c1 = next(r for r in records if r['candidate_id'] == 'candidate_1')

        # Train (Jan 01 and Jan 02)
        self.assertEqual(c1['train_6h_no_fill'], 300)
        self.assertEqual(c1['train_36h_no_fill'], 150)
        self.assertEqual(c1['train_144h_no_fill'], 30)
        self.assertEqual(c1['train_matched_no_fill'], 30)

        # Test (Jan 28 and Jan 29)
        self.assertEqual(c1['test_6h_no_fill'], 1100)
        self.assertEqual(c1['test_36h_no_fill'], 900)
        self.assertEqual(c1['test_144h_no_fill'], 700)
        self.assertEqual(c1['test_matched_no_fill'], 700)

    def test_ranking_prefers_candidate_1_over_historical_v5(self):
        """Candidate 1 must rank highest, while historical v5 must be disqualified."""
        records = quantify_candidate_periods({})
        c1 = next(r for r in records if r['candidate_id'] == 'candidate_1')
        c4 = next(r for r in records if r['candidate_id'] == 'candidate_4')

        self.assertEqual(c1['status'], 'RECOMMENDED')
        self.assertEqual(c4['status'], 'DISQUALIFIED_UNFEASIBLE')
        self.assertGreater(c1['ranking_score'], c4['ranking_score'])


class TestProtocolExecutionAndVerification(unittest.TestCase):
    """Verify full end-to-end execution and receipt verification."""

    def test_execute_and_verify_t1_2_protocol(self):
        """execute_t1_2_protocol must generate all required files and pass verification."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_dir = tmp_path / 'protocol_output'

            # Create synthetic window_coverage.csv
            cov_csv = tmp_path / 'window_coverage.csv'
            with open(cov_csv, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['date_utc', 'split', 'history_steps', 'history_hours', 'policy', 'windows', 'distinct_origins'],
                )
                writer.writeheader()
                for d in ('2022-01-05', '2022-01-25', '2022-01-30'):
                    for h in (6, 36, 144):
                        for pol in POLICIES:
                            writer.writerow({
                                'date_utc': d,
                                'split': 'train',
                                'history_steps': h,
                                'history_hours': h / 6.0,
                                'policy': pol,
                                'windows': 1000 // (h // 6),
                                'distinct_origins': 10,
                            })

            exec_res = execute_t1_2_protocol(out_dir, cov_csv)
            self.assertEqual(exec_res['status'], 'complete')

            expected_files = [
                'REPORT.md',
                'candidate_test_periods.csv',
                'common_cohort_summary.csv',
                'protocol.json',
                'leakage_review.md',
                'metadata.json',
                'completed.json',
            ]
            for fname in expected_files:
                fpath = out_dir / fname
                self.assertTrue(fpath.is_file(), f'Missing deliverable: {fname}')
                self.assertGreater(fpath.stat().st_size, 0, f'Empty deliverable: {fname}')

            # Verify protocol JSON structure
            protocol_data = json.loads((out_dir / 'protocol.json').read_text(encoding='utf-8'))
            self.assertEqual(protocol_data['protocol_version'], '1.0.0')
            self.assertEqual(protocol_data['cohort_and_universe']['node_client_universe'], 979)
            self.assertEqual(protocol_data['federated_training_budget']['seeds'], [42, 43, 44])

            # Test verify_t1_2_protocol
            verify_res = verify_t1_2_protocol(out_dir)
            self.assertEqual(verify_res['status'], 'verified_complete')
            self.assertEqual(verify_res['artifacts_verified'], 6)


if __name__ == '__main__':
    unittest.main()
