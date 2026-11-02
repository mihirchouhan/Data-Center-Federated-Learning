"""Synthetic tests for v5 eligibility; no real data or model operations."""

import unittest
import numpy as np

from src.v5_missing_data.eligibility import (
    FEATURES,
    history_is_eligible,
    target_is_valid,
    missing_counts,
)


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.features = np.ones((40, len(FEATURES)), dtype=np.float64)
        self.power = np.ones(40, dtype=np.float64)

    def test_masked_weather_missing_does_not_reject_history(self):
        self.features[10, 7] = np.nan
        self.features[11, 8] = np.nan
        self.assertTrue(history_is_eligible(self.features, 16, 6))

    def test_missing_cpu_rejects_history(self):
        self.features[12, 1] = np.nan
        self.assertFalse(history_is_eligible(self.features, 16, 6))

    def test_missing_memory_rejects_history(self):
        self.features[12, 5] = np.nan
        self.assertFalse(history_is_eligible(self.features, 16, 6))

    def test_missing_historical_power_rejects_history(self):
        self.features[12, 0] = np.nan
        self.assertFalse(history_is_eligible(self.features, 16, 6))

    def test_unmasked_weather_is_checked(self):
        self.features[12, 7] = np.nan
        self.assertFalse(
            history_is_eligible(self.features, 16, 6, masked_indices=())
        )

    def test_future_target_must_be_observed(self):
        self.assertTrue(target_is_valid(self.power, 16, 6))
        self.power[18] = np.nan
        self.assertFalse(target_is_valid(self.power, 16, 6))

    def test_target_out_of_bounds_is_invalid(self):
        self.assertFalse(target_is_valid(self.power, 38, 6))

    def test_missing_counts_by_feature(self):
        self.features[2, 1] = np.nan
        self.features[3, 7] = np.nan
        counts = missing_counts(self.features, 0, 10)
        self.assertEqual(counts["cpu_user"], 1)
        self.assertEqual(counts["outdoor_temperature_c"], 1)
        self.assertEqual(counts["power_kw"], 0)


if __name__ == "__main__":
    unittest.main()
