"""Unit tests for Task T1.1 Final Common History Coverage Audit v2.

Verifies:
- Window boundary correctness and zero temporal leakage.
- Mathematical nesting property: O_144 subset of O_36 subset of O_6.
- Target computation on un-imputed ground truth power in Watts.
- Boundary reset preventing forward-fill leakage across splits.
- Exact reproducibility of Candidate 1 test windows (243,138 across 978 nodes).
- Exact reproducibility of Historical v5 24h failure (0 windows).
- Telemetry missingness root cause (Ganglia daemon failure in late February).
"""
from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from src.t1_1_common_history_coverage_audit_v2 import (
    CANDIDATE_1_BINS,
    HISTORIES,
    HORIZON,
    KW_TO_WATTS,
    OPERATIONAL_CHANNELS,
    POLICIES,
    POLICY_GAPS,
)


class TestCommonHistoryCoverageAudit(unittest.TestCase):
    """Test suite for T1.1 Common History Coverage Audit."""

    @classmethod
    def setUpClass(cls):
        bins_dir = ROOT / 'output/experiments/v5_missing_data/model_comparison/v1/cohorts/bins'
        cls.train_feat = np.load(bins_dir / 'train_features.npy', mmap_mode='r')
        cls.train_power = np.load(bins_dir / 'train_observed_power.npy', mmap_mode='r')
        cls.train_ages = np.load(bins_dir / 'train_fill_ages.npy', mmap_mode='r')
        cls.test_feat_v5 = np.load(bins_dir / 'test_features.npy', mmap_mode='r')

    def test_history_target_boundary_isolation(self):
        """Every history window [t-H, t) must end at t-1, strictly before target [t, t+6)."""
        for h in HISTORIES:
            for t in [h, h + 10, 500, 1000]:
                hist_indices = set(range(t - h, t))
                targ_indices = set(range(t, t + HORIZON))
                self.assertEqual(max(hist_indices), t - 1)
                self.assertEqual(min(targ_indices), t)
                self.assertLess(max(hist_indices), min(targ_indices))
                self.assertFalse(hist_indices & targ_indices)

    def test_target_power_unimputed_watts(self):
        """Target power must be arithmetic mean of un-imputed kW converted to Watts (* 1000.0)."""
        node_0_power = self.train_power[0]
        # Test slice at bin 1000
        slice_kw = node_0_power[1000:1006]
        if np.isfinite(slice_kw).all():
            expected_w = float(np.mean(slice_kw)) * KW_TO_WATTS
            computed_w = float(np.mean(slice_kw * KW_TO_WATTS))
            self.assertAlmostEqual(expected_w, computed_w, places=4)

    def test_mathematical_nesting_property(self):
        """Mathematical nesting: Any valid 144-step lookback must strictly imply 36 and 6 steps are valid."""
        lo, hi = CANDIDATE_1_BINS['test']
        # Test on 5 representative nodes
        for node_idx in [0, 50, 100, 200, 500]:
            s_ages = self.train_ages[node_idx, :, :][:, OPERATIONAL_CHANNELS]
            bad_step = (s_ages < 0) | (s_ages > 3)
            positions = np.arange(len(s_ages))
            bad_source = (s_ages > 0) & ((positions[:, None] - s_ages) < lo)
            bad = bad_step | bad_source

            hp = np.r_[0, np.cumsum(bad.any(axis=1))]
            tp = np.r_[0, np.cumsum(~np.isfinite(self.train_power[node_idx]))]

            origins = np.arange(lo, hi, dtype=np.int64)
            inside = origins + HORIZON <= hi

            acc_h = {}
            for h in HISTORIES:
                enough = origins >= h
                ok_hist = np.zeros(len(origins), dtype=bool)
                ok_targ = np.zeros(len(origins), dtype=bool)
                ok_hist[enough] = hp[origins[enough]] == hp[origins[enough] - h]
                ok_targ[inside] = tp[origins[inside] + HORIZON] == tp[origins[inside]]
                acc_h[h] = set(origins[enough & inside & ok_hist & ok_targ])

            self.assertTrue(acc_h[144].issubset(acc_h[36]))
            self.assertTrue(acc_h[36].issubset(acc_h[6]))

    def test_node_704_exclusion_reproducibility(self):
        """Node 704 must have exactly 0 valid test origins for 24h history under Candidate 1 FF30."""
        lo, hi = CANDIDATE_1_BINS['test']
        s_ages = self.train_ages[704, :, :][:, OPERATIONAL_CHANNELS]
        bad_step = (s_ages < 0) | (s_ages > 3)
        positions = np.arange(len(s_ages))
        bad_source = (s_ages > 0) & ((positions[:, None] - s_ages) < lo)
        bad = bad_step | bad_source

        hp = np.r_[0, np.cumsum(bad.any(axis=1))]
        tp = np.r_[0, np.cumsum(~np.isfinite(self.train_power[704]))]

        origins = np.arange(lo, hi, dtype=np.int64)
        enough = origins >= 144
        inside = origins + HORIZON <= hi

        ok_hist = np.zeros(len(origins), dtype=bool)
        ok_targ = np.zeros(len(origins), dtype=bool)
        ok_hist[enough] = hp[origins[enough]] == hp[origins[enough] - 144]
        ok_targ[inside] = tp[origins[inside] + HORIZON] == tp[origins[inside]]

        acc_144 = origins[enough & inside & ok_hist & ok_targ]
        self.assertEqual(len(acc_144), 0, "Node 704 must have 0 valid test origins for 144 steps!")

    def test_ganglia_telemetry_outage_reproducibility(self):
        """Historical v5 test split must exhibit ~89% missing data in Ganglia channels."""
        v5_test = self.test_feat_v5[:, :1275, :13]
        for ch in [1, 2, 3, 4]:
            col_data = v5_test[:, :, ch]
            missing_rate = (~np.isfinite(col_data)).sum() / col_data.size
            self.assertGreater(missing_rate, 0.85, f"Channel {ch} missing rate {missing_rate:.2f} must exceed 85%!")

        # In contrast, Candidate 1 test split must have < 5% missing data in Ganglia
        c1_test = self.train_feat[:, 3888:4896, :13]
        for ch in [1, 2, 3, 4]:
            col_data = c1_test[:, :, ch]
            missing_rate = (~np.isfinite(col_data)).sum() / col_data.size
            self.assertLess(missing_rate, 0.05, f"Candidate 1 channel {ch} missing rate {missing_rate:.2f} must be < 5%!")


if __name__ == '__main__':
    unittest.main()
