import unittest

from ricemambaopt.datasets import ProcessBounds
from ricemambaopt.optimization import ConstraintSet, LinearConstraint


class ConstraintInputTests(unittest.TestCase):
    def test_linear_violation_does_not_treat_nan_as_feasible(self):
        constraint = LinearConstraint("dose", 1.0, 0.1, "<=", 200.0)
        for value in (float("nan"), float("inf"), -float("inf")):
            for time, speed in ((value, 1000.0), (30.0, value)):
                with self.subTest(time=time, speed=speed), self.assertRaises(ValueError):
                    constraint.violation(time, speed)

    def test_projection_requires_finite_settings(self):
        constraints = ConstraintSet(ProcessBounds((10.0, 60.0), (800.0, 1600.0)))
        for value in (float("nan"), float("inf"), -float("inf")):
            for time, speed in ((value, 1000.0), (30.0, value)):
                with self.subTest(time=time, speed=speed), self.assertRaises(ValueError):
                    constraints.project_to_bounds(time, speed)
        self.assertEqual(constraints.project_to_bounds(0.0, 2000.0), (10.0, 1600.0))
