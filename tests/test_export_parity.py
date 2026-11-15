import unittest

import numpy as np

from continugo_train.export import _assert_parity


class ExportParityTests(unittest.TestCase):
    def test_fp32_intermediate_roundoff_is_accepted(self):
        _assert_parity(np.array([0.018247247]), np.array([0.018224359]), spatial=True)

    def test_near_zero_logit_roundoff_is_accepted(self):
        _assert_parity(
            np.array([[0.016541481, -30.0]]),
            np.array([[0.016557036, -30.0]]),
            logits=True,
        )

    def test_large_logit_errors_are_rejected(self):
        with self.assertRaises(AssertionError):
            _assert_parity(np.array([[0.001, 0.0]]), np.zeros((1, 2)), logits=True)

    def test_final_outputs_keep_strict_tolerance(self):
        with self.assertRaises(AssertionError):
            _assert_parity(np.array([0.018247247]), np.array([0.018224359]))

    def test_large_errors_and_nonfinite_values_are_rejected(self):
        for spatial in (False, True):
            for actual, expected in ((0.001, 0.0), (float("nan"), float("nan"))):
                with self.subTest(spatial=spatial, actual=actual), self.assertRaises(AssertionError):
                    _assert_parity(np.array([actual]), np.array([expected]), spatial=spatial)
