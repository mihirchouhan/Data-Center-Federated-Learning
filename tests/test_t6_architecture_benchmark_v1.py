"""
Unit tests for Task T6 Architectural Family Generalization Benchmark infrastructure.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

from src.history_length_experiment_v1 import SequenceGRU
from src.t6_architecture_benchmark_v1 import (
    SequenceLSTM,
    T6_CONDITIONS,
    WEATHER_CHANNELS,
    apply_weather_mask,
)


class TestT6ArchitectureBenchmark(unittest.TestCase):
    def test_weather_channels(self):
        self.assertEqual(WEATHER_CHANNELS, [7, 8])

    def test_conditions_mapping(self):
        self.assertIn('condition_a_fedavg_gru', T6_CONDITIONS)
        self.assertIn('condition_b_fedavg_lstm', T6_CONDITIONS)
        self.assertIn('condition_c_centralized_xgboost', T6_CONDITIONS)

        self.assertEqual(T6_CONDITIONS['condition_a_fedavg_gru']['param_count'], 49561)
        self.assertEqual(T6_CONDITIONS['condition_b_fedavg_lstm']['param_count'], 63449)

    def test_apply_weather_mask(self):
        B, T, F = 4, 36, 14
        x = torch.ones((B, T, F), dtype=torch.float32)
        xa = apply_weather_mask(x, [7, 8])
        self.assertTrue((xa[:, :, [7, 8]] == 0.0).all())
        self.assertTrue((xa[:, :, 0:7] == 1.0).all())
        self.assertTrue((xa[:, :, 9:] == 1.0).all())

    def test_gru_architecture_and_parameter_count(self):
        model = SequenceGRU(
            input_dim=14,
            hidden_dim=64,
            num_layers=2,
            node_count=979,
            node_embedding_dim=8,
        )
        param_count = sum(p.numel() for p in model.parameters())
        self.assertEqual(param_count, 49561)

        bx = torch.randn(8, 36, 14)
        bx[:, :, -1] = 42.0
        out = model(bx)
        self.assertEqual(out.shape, (8, 1))

    def test_lstm_architecture_and_parameter_count(self):
        model = SequenceLSTM(
            input_dim=14,
            hidden_dim=64,
            num_layers=2,
            node_count=979,
            node_embedding_dim=8,
        )
        param_count = sum(p.numel() for p in model.parameters())
        self.assertEqual(param_count, 63449)

        bx = torch.randn(8, 36, 14)
        bx[:, :, -1] = 42.0
        out = model(bx)
        self.assertEqual(out.shape, (8, 1))

    def test_residual_reconstruction_mathematical_identity(self):
        rng = np.random.default_rng(42)
        pers_w = rng.uniform(200.0, 900.0, size=(100, 1)).astype(np.float32)
        target_w = rng.uniform(200.0, 900.0, size=(100, 1)).astype(np.float32)

        res_mean_kw = 0.005
        res_std_kw = 0.045
        kw_to_watts = 1000.0

        # Residual target
        res_kw = (target_w - pers_w) / kw_to_watts
        y_scaled = (res_kw - res_mean_kw) / res_std_kw

        # Inverse reconstruction
        reconstructed_kw = y_scaled * res_std_kw + res_mean_kw
        pred_w = pers_w + reconstructed_kw * kw_to_watts

        max_abs_diff = float(np.max(np.abs(pred_w - target_w)))
        self.assertLess(max_abs_diff, 1e-3)

    def test_config_artifacts(self):
        t6_dir = ROOT / 'output/experiments/t6_architecture_benchmark_v1'
        config_path = t6_dir / 'T6_CONFIGS.json'

        self.assertTrue(config_path.exists())
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.assertEqual(cfg['experiment_id'], 't6_architecture_benchmark_v1')
        self.assertIn('splits', cfg)
        self.assertIn('experimental_conditions', cfg)
        self.assertEqual(len(cfg['experimental_conditions']), 3)


if __name__ == '__main__':
    unittest.main()
