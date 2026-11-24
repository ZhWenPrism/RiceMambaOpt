from decimal import Decimal
import unittest

from ricemambaopt.evaluation.metrics import joint_normalized_rmse


class MetricScaleTypesTest(unittest.TestCase):
    def test_decimal_scales_match_float_scales(self):
        actual = {"a": [0, 0], "b": [1, 1]}
        predicted = {"a": [1, 1], "b": [3, 3]}
        expected = joint_normalized_rmse(actual, predicted, {"a": 0.5, "b": 1.0})
        result = joint_normalized_rmse(actual, predicted,
                                       {"a": Decimal("0.5"), "b": Decimal("1")})
        self.assertEqual(result, expected)
        self.assertEqual(result, 2.0)

    def test_decimal_vectors_and_scales_share_the_same_numeric_contract(self):
        self.assertEqual(joint_normalized_rmse(
            {"a": [Decimal("0"), Decimal("0")]},
            {"a": [Decimal("1"), Decimal("1")]},
            {"a": Decimal("0.5")}), 2.0)

    def test_invalid_decimal_scales_raise_value_error(self):
        for scale in ("0", "-1", "NaN", "Infinity"):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    joint_normalized_rmse({"a": [0]}, {"a": [1]},
                                          {"a": Decimal(scale)})
