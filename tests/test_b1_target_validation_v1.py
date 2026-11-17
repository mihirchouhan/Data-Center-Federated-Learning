"""Synthetic target-array regressions only; no models, file data or scoring."""
import unittest

import numpy as np

from src.b1_target_validation_v1 import target_means
from src.phase2_multioutput.contracts import target_means as historical_target_means


def fixture(dtype=np.float64):
    # Invented bin values, not telemetry or predictions.
    values = np.stack((np.arange(1, 7), np.arange(10, 16)), axis=-1)
    values = values[None].astype(dtype)
    return values, np.ones(values.shape, dtype=bool)


class B1TargetValidationTests(unittest.TestCase):
    def test_fully_unmasked_masked_array_rejected(self):
        f, obs = fixture()
        f = np.ma.array(f, mask=np.zeros(f.shape, dtype=bool))
        self.assertFalse(np.ma.getmaskarray(f).any())
        with self.assertRaisesRegex(ValueError, 'Masked-array targets'):
            target_means(f, obs)

    def test_nomask_masked_array_rejected(self):
        f, obs = fixture()
        with self.assertRaisesRegex(ValueError, 'Masked-array targets'):
            target_means(np.ma.array(f, mask=np.ma.nomask), obs)

    def test_partially_masked_target_rejected(self):
        f, obs = fixture()
        masked = np.ma.array(f, mask=False)
        masked.mask[0, 2, 1] = True
        with self.assertRaisesRegex(ValueError, 'Masked-array targets'):
            target_means(masked, obs)

    def test_all_masked_target_rejected(self):
        f, obs = fixture()
        with self.assertRaisesRegex(ValueError, 'Masked-array targets'):
            target_means(np.ma.array(f, mask=True), obs)

    def test_masked_boolean_target_rejected_before_mask_erasure(self):
        f, obs = fixture(bool)
        with self.assertRaisesRegex(ValueError, 'Masked-array targets'):
            target_means(np.ma.array(f, mask=False), obs)

    def test_boolean_target_array_rejected(self):
        f, obs = fixture(bool)
        with self.assertRaisesRegex(ValueError, 'Boolean target arrays'):
            target_means(f, obs)

    def test_all_false_boolean_target_array_rejected(self):
        f, obs = fixture(bool)
        f[:] = False
        with self.assertRaisesRegex(ValueError, 'Boolean target arrays'):
            target_means(f, obs)

    def test_float32_targets_preserve_values_units_and_order(self):
        f, obs = fixture(np.float32)
        result = target_means(f, obs)
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, [[3500., 12.5]])

    def test_float64_targets_preserve_values_units_and_order(self):
        f, obs = fixture(np.float64)
        np.testing.assert_array_equal(target_means(f, obs), [[3500., 12.5]])

    def test_signed_integer_targets_remain_supported(self):
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            with self.subTest(dtype=dtype):
                f, obs = fixture(dtype)
                np.testing.assert_array_equal(target_means(f, obs), [[3500., 12.5]])

    def test_unsigned_integer_targets_remain_supported(self):
        for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            with self.subTest(dtype=dtype):
                f, obs = fixture(dtype)
                np.testing.assert_array_equal(target_means(f, obs), [[3500., 12.5]])

    def test_valid_numeric_results_match_historical_helper_exactly(self):
        for dtype in (np.float32, np.float64, np.int32, np.uint32):
            with self.subTest(dtype=dtype):
                f, obs = fixture(dtype)
                expected = historical_target_means(f, obs)
                actual = target_means(f, obs)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_nan_and_infinities_rejected_for_each_target(self):
        for dtype in (np.float32, np.float64):
            for value in (np.nan, np.inf, -np.inf):
                for channel in (0, 1):
                    with self.subTest(dtype=dtype, value=value, channel=channel):
                        f, obs = fixture(dtype)
                        f[0, 1, channel] = value
                        with self.assertRaises(ValueError):
                            target_means(f, obs)

    def test_false_observation_in_either_target_rejected(self):
        for channel in (0, 1):
            with self.subTest(channel=channel):
                f, obs = fixture()
                obs[0, 4, channel] = False
                with self.assertRaisesRegex(ValueError, 'future target'):
                    target_means(f, obs)

    def test_masked_observation_flags_rejected_even_when_unmasked(self):
        for masked_position in (None, (0, 1, 1)):
            with self.subTest(masked_position=masked_position):
                f, obs = fixture()
                flags = np.ma.array(obs, mask=False)
                if masked_position is not None:
                    flags.mask[masked_position] = True
                with self.assertRaisesRegex(ValueError, 'Masked-array target observation flags'):
                    target_means(f, flags)

    def test_wrong_observation_dtype_or_shape_rejected(self):
        f, obs = fixture()
        for flags in (obs.astype(np.int8), obs[:, :, 0]):
            with self.subTest(shape=flags.shape, dtype=flags.dtype):
                with self.assertRaises(ValueError):
                    target_means(f, flags)

    def test_incomplete_empty_or_single_output_targets_rejected(self):
        for shape in ((0, 6, 2), (1, 5, 2), (1, 7, 2), (1, 6, 1), (6, 2)):
            with self.subTest(shape=shape):
                f = np.ones(shape)
                with self.assertRaises(ValueError):
                    target_means(f, np.ones(shape, dtype=bool))

    def test_existing_physical_range_checks_preserved(self):
        for channel, value in ((0, -1), (1, -273.16)):
            with self.subTest(channel=channel):
                f, obs = fixture()
                f[0, 0, channel] = value
                with self.assertRaises(ValueError):
                    target_means(f, obs)

    def test_multiple_windows_and_noncontiguous_inputs(self):
        f, obs = fixture()
        f = np.repeat(f, 2, axis=0)[:, ::-1, :]
        obs = np.repeat(obs, 2, axis=0)[:, ::-1, :]
        self.assertFalse(f.flags.c_contiguous)
        np.testing.assert_array_equal(target_means(f, obs), [[3500., 12.5], [3500., 12.5]])

    def test_valid_array_like_inputs_remain_supported(self):
        f, obs = fixture()
        np.testing.assert_array_equal(target_means(f.tolist(), obs.tolist()), [[3500., 12.5]])

    def test_inputs_unchanged_and_output_immutable(self):
        f, obs = fixture()
        before, flags = f.copy(), obs.copy()
        result = target_means(f, obs)
        np.testing.assert_array_equal(f, before)
        np.testing.assert_array_equal(obs, flags)
        self.assertFalse(result.flags.writeable)
        with self.assertRaises(ValueError):
            result[0, 0] = -1


if __name__ == '__main__':
    unittest.main()
