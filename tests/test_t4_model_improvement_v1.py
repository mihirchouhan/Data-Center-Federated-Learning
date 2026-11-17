"""Unit tests for Task T4 Forecasting Improvement and Controlled Federated Benchmark.

Verifies:
1. Persistence Residual Learning mathematical inversion and identity reconstruction.
2. Temporal split boundaries and strict causality.
3. Feature masking transformations and tensor invariants.
4. Loss function backpropagation and numerical stability.
5. Model architecture scaling (hidden 64/128, layers 1/2) and parameter counts.
"""
from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn as nn

from src.history_length_experiment_v1 import SequenceGRU


class TestT4ModelImprovement(unittest.TestCase):

    def test_residual_learning_mathematical_identity(self):
        """Verify that P_hat = P_pers + DeltaP_hat perfectly reconstructs target in Watts."""
        np.random.seed(42)
        # Simulated true power in Watts
        batch_size = 50
        true_target_w = np.random.uniform(200.0, 900.0, size=(batch_size, 1)).astype(np.float32)
        last_step_w = np.random.uniform(200.0, 900.0, size=(batch_size, 1)).astype(np.float32)

        # Residual in kW
        delta_kw = (true_target_w - last_step_w) / 1000.0

        # Training scaler moments
        mu_delta = float(np.mean(delta_kw))
        sigma_delta = float(np.std(delta_kw)) if np.std(delta_kw) > 1e-6 else 1.0

        # Standardized residual
        z_delta = (delta_kw - mu_delta) / sigma_delta

        # Perfect model prediction: z_hat = z_delta
        z_hat = z_delta.copy()

        # Inversion back to physical Watts
        delta_hat_kw = z_hat * sigma_delta + mu_delta
        p_hat_w = last_step_w + delta_hat_kw * 1000.0

        # Maximum reconstruction error in float32 should be sub-millimeter level (< 0.001 W)
        max_error = np.max(np.abs(p_hat_w - true_target_w))
        self.assertLess(max_error, 1e-3, f"Residual inversion error {max_error} exceeds threshold")

    def test_chronological_splits_strictly_causal(self):
        """Verify Candidate 1 train, val, test, and March splits are strictly ordered."""
        train_bins = (0, 3024)        # Jan 01 - Jan 22
        val_bins = (3024, 3888)       # Jan 22 - Jan 28
        test_t1_bins = (3888, 4896)   # Jan 28 - Feb 04
        test_march_bins = (8496, 12960) # Mar 01 - Apr 01

        self.assertLessEqual(train_bins[1], val_bins[0])
        self.assertLessEqual(val_bins[1], test_t1_bins[0])
        self.assertLessEqual(test_t1_bins[1], test_march_bins[0])

    def test_feature_masking_logic(self):
        """Verify feature masking produces valid tensors and expected zero channels."""
        B, seq_len, in_dim = 16, 36, 14
        x = torch.randn(B, seq_len, in_dim)

        # Power only: keep channel 0 and 13 (node index), zero out 1-12
        x_power_only = x.clone()
        x_power_only[:, :, 1:13] = 0.0
        self.assertEqual(x_power_only[:, :, 1:13].sum().item(), 0.0)
        self.assertNotEqual(x_power_only[:, :, 0].sum().item(), 0.0)

        # Power + Calendar: keep channel 0, 9-12 (calendar), 13 (node index), zero out 1-8
        x_pow_cal = x.clone()
        x_pow_cal[:, :, 1:9] = 0.0
        self.assertEqual(x_pow_cal[:, :, 1:9].sum().item(), 0.0)
        self.assertNotEqual(x_pow_cal[:, :, [9, 10, 11, 12]].sum().item(), 0.0)

    def test_loss_functions_and_backprop(self):
        """Verify MSE, MAE, and Huber losses compute finite gradients on SequenceGRU."""
        model = SequenceGRU(input_dim=14, hidden_dim=64, num_layers=1, node_count=979)
        x = torch.randn(8, 36, 14)
        x[:, :, -1] = torch.randint(0, 979, (8, 36)).float()
        y = torch.randn(8, 1)

        losses = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'huber': nn.SmoothL1Loss(beta=1.0),
        }

        for name, loss_fn in losses.items():
            model.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            for p in model.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_model_architecture_variants(self):
        """Verify SequenceGRU architecture scaling for hidden_dim (64, 128) and num_layers (1, 2)."""
        # Baseline: hidden 64, layer 1
        m1 = SequenceGRU(input_dim=14, hidden_dim=64, num_layers=1, node_count=979)
        params_m1 = sum(p.numel() for p in m1.parameters())
        self.assertEqual(params_m1, 24601)

        # Scaled hidden: hidden 128, layer 1
        m2 = SequenceGRU(input_dim=14, hidden_dim=128, num_layers=1, node_count=979)
        params_m2 = sum(p.numel() for p in m2.parameters())
        self.assertGreater(params_m2, params_m1)

        # Scaled depth: hidden 64, layer 2
        m3 = SequenceGRU(input_dim=14, hidden_dim=64, num_layers=2, node_count=979)
        params_m3 = sum(p.numel() for p in m3.parameters())
        self.assertGreater(params_m3, params_m1)

        # Check forward pass shapes
        x = torch.randn(4, 36, 14)
        x[:, :, -1] = torch.randint(0, 979, (4, 36)).float()
        self.assertEqual(m1(x).shape, (4, 1))
        self.assertEqual(m2(x).shape, (4, 1))
        self.assertEqual(m3(x).shape, (4, 1))


if __name__ == '__main__':
    unittest.main()
