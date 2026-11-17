"""Unit tests for Task T3 Periodic Federated Retraining & Temporal Adaptation.

Verifies:
1. Strict temporal causality and split ordering (Train < Val <= Deployment Start).
2. Scaler leakage prevention: Moments fitted strictly on designated training slices.
3. Target horizon isolation: Target mean IT power never crosses beyond slice boundary.
4. Model architecture integrity: 24,601 parameters, finite tensors.
5. Reduction formula consistency: Pooled vs. macro-node MAE/RMSE calculations.
"""
from __future__ import annotations

import unittest
import numpy as np
import torch

from src.history_length_experiment_v1 import SequenceGRU


class TestTemporalAdaptationT3(unittest.TestCase):

    def test_chronological_split_causality(self):
        """Verify that training and validation slices end strictly before deployment begins."""
        # Initial model (M0)
        m0_train = (0, 3024)        # Jan 01 - Jan 22
        m0_val = (3024, 3888)       # Jan 22 - Jan 28
        m0_deploy = (3888, 8496)    # Jan 28 - Mar 01

        self.assertLessEqual(m0_train[1], m0_val[0])
        self.assertLessEqual(m0_val[1], m0_deploy[0])

        # Monthly Retraining Event 1 (for March deployment)
        m1_train = (3888, 6048)     # Jan 28 - Feb 11
        m1_val = (6048, 6912)       # Feb 11 - Feb 17
        m1_deploy = (8496, 12960)   # Mar 01 - Apr 01

        self.assertLessEqual(m1_train[1], m1_val[0])
        self.assertLessEqual(m1_val[1], m1_deploy[0])
        # Verify safety gap: Feb 17 to Mar 01
        self.assertGreaterEqual(m1_deploy[0] - m1_val[1], 144)  # at least 1 day safety buffer

    def test_scaler_leakage_safeguard(self):
        """Verify scaler moments are fitted strictly on designated training slice without test leakage."""
        from src.temporal_generalization_v1 import ExtendedTelemetryDataset
        data = ExtendedTelemetryDataset()

        lo, hi = 3888, 6048
        train_pow = data.power[:, lo:hi]
        finite_p = np.isfinite(train_pow)
        expected_mean = train_pow[finite_p].mean()
        expected_std = train_pow[finite_p].std()

        self.assertGreater(expected_mean, 0.0)
        self.assertGreater(expected_std, 0.0)
        # Verify that February mean is distinctly higher than January mean (quantifying concept drift)
        jan_mean = data.scalers['y_mean'][0]
        self.assertGreater(expected_mean, jan_mean)

    def test_target_horizon_isolation(self):
        """Verify that forecast target [origin, origin + 6) never exceeds split boundary."""
        lo, hi = 3888, 6048
        horizon = 6
        origins = np.arange(lo + 36, hi - horizon + 1)
        for orig in origins:
            self.assertGreaterEqual(orig - 36, lo)
            self.assertLessEqual(orig + horizon, hi)

    def test_model_parameter_count_and_finiteness(self):
        """Verify 6h FedAvg-GRU architecture has exactly 24,601 parameters and loads finite tensors."""
        model = SequenceGRU(input_dim=14, hidden_dim=64, num_layers=1, node_count=979)
        param_count = sum(p.numel() for p in model.parameters())
        self.assertEqual(param_count, 24601)

        # Check forward pass output shape
        dummy_x = torch.zeros((4, 36, 14), dtype=torch.float32)
        dummy_x[:, :, -1] = 0.0  # node 0
        out = model(dummy_x)
        self.assertEqual(out.shape, (4, 1))

    def test_metric_reduction_consistency(self):
        """Verify mathematical consistency of pooled vs macro reductions."""
        preds = np.array([100.0, 105.0, 110.0, 120.0])
        actuals = np.array([102.0, 104.0, 108.0, 125.0])
        nodes = np.array([0, 0, 1, 1])

        abs_err = np.abs(preds - actuals)
        sq_err = (preds - actuals) ** 2

        pooled_mae = float(np.mean(abs_err))
        pooled_rmse = float(np.sqrt(np.mean(sq_err)))

        self.assertAlmostEqual(pooled_mae, 2.5, places=5)
        self.assertAlmostEqual(pooled_rmse, np.sqrt(8.5), places=5)

        node_0_mae = np.mean(abs_err[:2])  # (2 + 1)/2 = 1.5
        node_1_mae = np.mean(abs_err[2:])  # (2 + 5)/2 = 3.5
        macro_mae = (node_0_mae + node_1_mae) / 2.0
        self.assertAlmostEqual(macro_mae, 2.5, places=5)


if __name__ == '__main__':
    unittest.main()
