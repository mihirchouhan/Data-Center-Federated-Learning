"""
Unit tests for Task T7 Missing Telemetry Robustness experiment infrastructure.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

from src.t7_missing_telemetry_robustness_v1 import (
    FEATURE_GROUPS,
    OPERATIONAL_CHANNELS,
    WEATHER_CHANNELS,
    T7MissingTelemetryManager,
)


class TestT7MissingTelemetryRobustness(unittest.TestCase):
    def test_operational_channels_and_groups(self):
        self.assertEqual(OPERATIONAL_CHANNELS, [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(WEATHER_CHANNELS, [7, 8])

        self.assertIn('power', FEATURE_GROUPS)
        self.assertIn('cpu', FEATURE_GROUPS)
        self.assertIn('memory', FEATURE_GROUPS)
        self.assertIn('inlet_temp', FEATURE_GROUPS)

        self.assertEqual(FEATURE_GROUPS['power'], [0])
        self.assertEqual(FEATURE_GROUPS['cpu'], [1, 2, 3, 4])
        self.assertEqual(FEATURE_GROUPS['memory'], [5])
        self.assertEqual(FEATURE_GROUPS['inlet_temp'], [6])

    def test_random_missingness_and_ff30_logic(self):
        # Create synthetic node timeline of 100 bins
        N, T, C = 2, 100, 7
        features = np.ones((N, T, C), dtype=np.float32) * 50.0
        ages = np.zeros((N, T, C), dtype=np.int16)

        # Inject 10% random missingness
        mgr = T7MissingTelemetryManager(device_str='cpu')
        filled, new_ages, audit = mgr.apply_missingness_and_causal_ff30(
            features=features,
            ages=ages,
            target_channels=[0, 1, 2, 3, 4, 5, 6],
            missing_type='random',
            missing_pct=0.10,
            seed=42,
        )

        self.assertGreater(audit['artificial_missing_values'], 0)
        self.assertGreater(audit['values_filled_by_ff'], 0)
        # Random 10% with FF30 fills almost all isolated missing points
        self.assertGreater(audit['pct_filled_by_ff'], 80.0)

    def test_contiguous_missingness_block_lengths(self):
        N, T, C = 1, 100, 1
        features = np.ones((N, T, C), dtype=np.float32) * 100.0
        ages = np.zeros((N, T, C), dtype=np.int16)

        mgr = T7MissingTelemetryManager(device_str='cpu')

        # L=3: FF30 should fill all 3 bins
        filled_l3, ages_l3, audit_l3 = mgr.apply_missingness_and_causal_ff30(
            features=features,
            ages=ages,
            target_channels=[0],
            missing_type='contiguous',
            missing_pct=0.20,
            block_len=3,
            seed=42,
        )
        self.assertGreater(audit_l3['values_filled_by_ff'], 0)
        self.assertGreaterEqual(audit_l3['pct_filled_by_ff'], 70.0)

        # L=12: FF30 fills first 3, remaining 9 must stay unfilled
        filled_l12, ages_l12, audit_l12 = mgr.apply_missingness_and_causal_ff30(
            features=features,
            ages=ages,
            target_channels=[0],
            missing_type='contiguous',
            missing_pct=0.20,
            block_len=12,
            seed=42,
        )
        self.assertGreater(audit_l12['values_still_unavailable'], 0)
        self.assertLess(audit_l12['pct_filled_by_ff'], 50.0)

    def test_configs_artifact(self):
        cfg_path = ROOT / 'output/experiments/t7_missing_telemetry_robustness_v1/T7_CONFIGS.json'
        self.assertTrue(cfg_path.exists())
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.assertEqual(cfg['experiment_id'], 't7_missing_telemetry_robustness_v1')
        self.assertIn('missingness_levels', cfg)
        self.assertIn('masking_types', cfg)
        self.assertIn('operational_channels', cfg)


if __name__ == '__main__':
    unittest.main()
