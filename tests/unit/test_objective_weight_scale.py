import math
import unittest
from ricemambaopt.optimization import weighted_normalized_squared_error


class ObjectiveWeightScaleTest(unittest.TestCase):
    def test_rescaling_weights_keeps_the_same_objective(self):
        predicted, target, scales = {"a": 1., "b": 2.}, {"a": 0., "b": 0.}, {"a": 1., "b": 1.}
        for common in (1., 1e308, 1e-300):
            result = weighted_normalized_squared_error(predicted, target, scales, {"a": common, "b": common})
            self.assertTrue(math.isfinite(result))
            self.assertAlmostEqual(result, 2.5)

    def test_relative_weights_still_control_the_objective(self):
        self.assertAlmostEqual(weighted_normalized_squared_error(
            {"a": 1., "b": 2.}, {"a": 0., "b": 0.}, {"a": 1., "b": 1.},
            {"a": 1e308, "b": 5e307}), 2.)
