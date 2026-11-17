"""
Unit tests for Task T5 Weather Ablation experiment infrastructure.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

from src.history_length_experiment_v1 import SequenceGRU
from src.t5_weather_ablation_v1 import (
    WEATHER_CONDITIONS,
    WEATHER_CHANNELS,
    apply_weather_mask,
)


class TestT5WeatherAblation(unittest.TestCase):
    def test_weather_channels(self):
        self.assertEqual(WEATHER_CHANNELS['outdoor_temperature_c'], 7)
        self.assertEqual(WEATHER_CHANNELS['outdoor_humidity_percent'], 8)

    def test_weather_conditions_mapping(self):
        self.assertIn('condition_a_no_weather', WEATHER_CONDITIONS)
        self.assertIn('condition_b_temp_only', WEATHER_CONDITIONS)
        self.assertIn('condition_c_humidity_only', WEATHER_CONDITIONS)
        self.assertIn('condition_d_temp_and_humidity', WEATHER_CONDITIONS)

        self.assertEqual(WEATHER_CONDITIONS['condition_a_no_weather']['zero_channels'], [7, 8])
        self.assertEqual(WEATHER_CONDITIONS['condition_b_temp_only']['zero_channels'], [8])
        self.assertEqual(WEATHER_CONDITIONS['condition_c_humidity_only']['zero_channels'], [7])
        self.assertEqual(WEATHER_CONDITIONS['condition_d_temp_and_humidity']['zero_channels'], [])

    def test_apply_weather_mask(self):
        B, T, F = 4, 36, 14
        x = torch.ones((B, T, F), dtype=torch.float32)

        # Condition A: mask 7 and 8
        xa = apply_weather_mask(x, [7, 8])
        self.assertTrue((xa[:, :, [7, 8]] == 0.0).all())
        self.assertTrue((xa[:, :, 0:7] == 1.0).all())
        self.assertTrue((xa[:, :, 9:] == 1.0).all())

        # Condition B: mask 8, keep 7
        xb = apply_weather_mask(x, [8])
        self.assertTrue((xb[:, :, 7] == 1.0).all())
        self.assertTrue((xb[:, :, 8] == 0.0).all())

        # Condition C: mask 7, keep 8
        xc = apply_weather_mask(x, [7])
        self.assertTrue((xc[:, :, 7] == 0.0).all())
        self.assertTrue((xc[:, :, 8] == 1.0).all())

        # Condition D: mask none
        xd = apply_weather_mask(x, [])
        self.assertTrue((xd == 1.0).all())

    def test_model_architecture_and_parameter_count(self):
        model = SequenceGRU(
            input_dim=14,
            hidden_dim=64,
            num_layers=2,
            node_count=979,
            node_embedding_dim=8,
        )
        param_count = sum(p.numel() for p in model.parameters())
        self.assertEqual(param_count, 49561)

        # Forward pass shape check
        bx = torch.randn(8, 36, 14)
        bx[:, :, -1] = 42.0  # valid node index
        out = model(bx)
        self.assertEqual(out.shape, (8, 1))

    def test_config_and_coverage_artifacts(self):
        t5_dir = ROOT / 'output/experiments/t5_weather_ablation_v1'
        config_path = t5_dir / 'T5_CONFIGS.json'
        coverage_path = t5_dir / 'T5_WEATHER_COVERAGE.csv'

        self.assertTrue(config_path.exists())
        self.assertTrue(coverage_path.exists())

        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.assertEqual(cfg['experiment_id'], 't5_weather_ablation_v1')
        self.assertIn('splits', cfg)
        self.assertIn('weather_features', cfg)


if __name__ == '__main__':
    unittest.main()
