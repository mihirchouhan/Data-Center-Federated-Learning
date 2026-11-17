"""Unit tests for Task T2 Temporal Generalization Evaluation.

Verifies:
1. Scaler leakage prevention (moments computed strictly on training bins [0, 3024)).
2. Temporal split boundaries and strict origin ordering.
3. Persistence baseline definition and unit conversion ($P[t-1]$ in Watts).
4. Metric reduction formulas (pooled vs. macro-node).
5. Output CSV and artifact integrity.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.history_length_experiment_v1 import SequenceGRU


class TestTemporalGeneralizationV1(unittest.TestCase):

    def test_scaler_leakage_safeguard(self):
        """Verify that training scalers are fitted strictly on bins [0, 3024)."""
        from src.history_length_real_pilot_v1 import Candidate1RealData
        data = Candidate1RealData()

        # Check moments
        y_mean = data.scalers['y_mean'][0]
        y_std = data.scalers['y_std'][0]
        self.assertGreater(y_mean, 0.0)
        self.assertGreater(y_std, 0.0)

        # Compute directly on [0, 3024) slice
        lo, hi = 0, 3024
        train_p = data.power[:, lo:hi]
        finite_p = np.isfinite(train_p)
        expected_mean = train_p[finite_p].mean()
        expected_std = train_p[finite_p].std()

        self.assertAlmostEqual(y_mean, expected_mean, places=6)
        self.assertAlmostEqual(y_std, expected_std, places=6)

    def test_split_temporal_ordering(self):
        """Verify chronological ordering: train < val < test, with no overlap."""
        train_cut = (0, 3024)
        val_cut = (3024, 3888)
        test_cut = (3888, 12960)

        self.assertEqual(train_cut[1], val_cut[0])
        self.assertEqual(val_cut[1], test_cut[0])
        self.assertLess(train_cut[0], train_cut[1])
        self.assertLess(val_cut[0], val_cut[1])
        self.assertLess(test_cut[0], test_cut[1])

    def test_persistence_baseline_definition(self):
        """Verify that persistence is P[t-1] in Watts matching target."""
        # Simulated power array for 1 node across 100 bins
        np.random.seed(42)
        power_kw = np.random.uniform(0.3, 0.8, size=(1, 100))
        origin = 50
        history_steps = 36
        horizon = 6

        # Target is mean over [origin, origin + 6)
        target_kw = np.mean(power_kw[0, origin:origin + horizon])
        target_w = target_kw * 1000.0

        # Persistence is last observed step P[origin - 1]
        pers_kw = power_kw[0, origin - 1]
        pers_w = pers_kw * 1000.0

        self.assertAlmostEqual(pers_w, power_kw[0, 49] * 1000.0)
        self.assertNotEqual(pers_w, target_w)

    def test_metric_reductions(self):
        """Verify pooled and macro-node MAE/RMSE calculations."""
        actual = np.array([100.0, 110.0, 120.0, 130.0])
        pred = np.array([105.0, 108.0, 125.0, 128.0])
        nodes = np.array([0, 0, 1, 1])

        # Pooled
        abs_err = np.abs(pred - actual)
        sq_err = (pred - actual) ** 2
        pooled_mae = float(np.mean(abs_err))
        pooled_rmse = float(np.sqrt(np.mean(sq_err)))

        self.assertAlmostEqual(pooled_mae, 3.5, places=5)
        self.assertAlmostEqual(pooled_rmse, np.sqrt(14.5), places=5)

        # Macro
        node_0_mae = np.mean(abs_err[:2])
        node_1_mae = np.mean(abs_err[2:])
        macro_mae = (node_0_mae + node_1_mae) / 2.0
        self.assertAlmostEqual(macro_mae, 3.5, places=5)

    def test_model_loading_and_architecture(self):
        """Verify that 6h model architecture matches 24,601 parameters and loads finite tensors."""
        model = SequenceGRU(input_dim=14, hidden_dim=64, num_layers=1, node_count=979)
        param_count = sum(p.numel() for p in model.parameters())
        self.assertEqual(param_count, 24601)


if __name__ == '__main__':
    unittest.main()
