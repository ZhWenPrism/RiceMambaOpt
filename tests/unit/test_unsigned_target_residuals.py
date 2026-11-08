import unittest

from ricemambaopt.evaluation.metrics import regression_metrics
from ricemambaopt.optimization import weighted_normalized_squared_error

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class UnsignedTargetTest(unittest.TestCase):
    def test_target_metrics_match_python_numbers(self):
        actual = [10, 20, 30]
        predicted = [8, 20, 31]
        with np.errstate(over="raise"):
            result = regression_metrics(np.array(actual, dtype=np.uint8), np.array(predicted, dtype=np.uint8))
        self.assertEqual(result, regression_metrics(actual, predicted))
        self.assertAlmostEqual(result.bias, -1 / 3)

    def test_objective_uses_the_signed_error_before_normalizing(self):
        with np.errstate(over="raise"):
            value = weighted_normalized_squared_error(
                {"quality": np.uint8(8)}, {"quality": np.uint8(10)}, {"quality": 2},
            )
        self.assertEqual(value, 1.0)
