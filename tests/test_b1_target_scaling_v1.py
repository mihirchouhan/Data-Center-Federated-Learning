"""Invented supplied constants only; no statistics are fitted."""
from dataclasses import FrozenInstanceError
import unittest
import numpy as np

from src.b1_target_scaling_v1 import TargetScaling, TARGET_NAMES, TARGET_UNITS


class B1TargetScalingTests(unittest.TestCase):
    def setUp(self):
        self.scaling = TargetScaling((300., 20.), (100., 5.))

    def test_separate_units_and_order(self):
        self.assertEqual(TARGET_UNITS, ('W', 'degC'))
        self.assertEqual(TARGET_NAMES[0], 'next_hour_mean_node_it_power_w')
        np.testing.assert_array_equal(self.scaling.standardize([[350., 15.]]), [[.5, -1.]])

    def test_inverse_without_second_kw_conversion(self):
        np.testing.assert_array_equal(self.scaling.inverse([[.5, -1.]]), [[350., 15.]])

    def test_roundtrip_numeric_dtypes(self):
        for dtype in (np.float32, np.float64, np.int32, np.uint32):
            with self.subTest(dtype=dtype):
                y = np.array([[300, 20], [400, 25]], dtype=dtype)
                np.testing.assert_array_equal(self.scaling.inverse(self.scaling.standardize(y)), y)

    def test_constants_and_outputs_are_immutable(self):
        mean = np.array([300., 20.]); std = np.array([100., 5.])
        scaling = TargetScaling(mean, std)
        mean[0] = -1; std[1] = -1
        self.assertEqual(scaling.mean, (300., 20.))
        self.assertEqual(scaling.std, (100., 5.))
        with self.assertRaises(FrozenInstanceError):
            scaling.mean = (0., 0.)
        self.assertFalse(scaling.standardize([[300., 20.]]).flags.writeable)
        self.assertFalse(scaling.inverse([[0., 0.]]).flags.writeable)

    def test_invalid_supplied_constants_rejected(self):
        cases = [((True, 20), (100, 5)), ((300,), (100, 5)),
                 ((300, np.nan), (100, 5)), ((300, 20), (0, 5)),
                 ((300, 20), (-1, 5)), ((300, 20), (1e-13, 5)),
                 ((300, 20), (100, np.inf)), ((300+1j, 20), (100, 5))]
        for mean, std in cases:
            with self.subTest(mean=mean, std=std):
                with self.assertRaises(ValueError):TargetScaling(mean, std)

    def test_masked_constants_rejected_even_without_active_mask(self):
        with self.assertRaises(ValueError):
            TargetScaling(np.ma.array([300., 20.], mask=False), (100., 5.))

    def test_nonfinite_or_masked_or_boolean_matrices_rejected(self):
        values = [np.ma.array([[300., 20.]], mask=False),
                  np.ma.array([[300., 20.]], mask=True), np.ones((1, 2), bool),
                  [[np.nan, 20]], [[300, np.inf]], [[-np.inf, 20]],
                  np.ones((1, 1)), np.empty((0, 2)), np.ones(2)]
        for value in values:
            for operation in (self.scaling.standardize, self.scaling.inverse):
                with self.subTest(shape=np.shape(value), operation=operation.__name__):
                    with self.assertRaises(ValueError):operation(value)

    def test_overflow_is_rejected(self):
        scaling = TargetScaling((0, 0), (1e308, 1e308))
        with self.assertRaises((ValueError, FloatingPointError)):
            scaling.inverse([[1e308, 1e308]])

    def test_source_arrays_not_mutated(self):
        physical = np.array([[350., 15.]]); before = physical.copy()
        self.scaling.standardize(physical)
        np.testing.assert_array_equal(physical, before)

    def test_metadata_and_digest_are_not_a_fit_claim(self):
        record = self.scaling.record()
        self.assertFalse(record['fit_performed_by_this_object'])
        self.assertEqual(record['target_units'], ['W', 'degC'])
        self.assertEqual(self.scaling.digest, TargetScaling((300, 20), (100, 5)).digest)
        self.assertNotEqual(self.scaling.digest, TargetScaling((301, 20), (100, 5)).digest)
        self.assertEqual(len(self.scaling.digest), 64)


if __name__ == '__main__':unittest.main()
