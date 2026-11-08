import unittest

from ricemambaopt.datasets import ProcessBounds
from ricemambaopt.optimization import ConstraintSet, LinearConstraint

try:
    import numpy as np
except ImportError:
    np = None


class ConstraintNumericSlackTest(unittest.TestCase):
    @unittest.skipIf(np is None, "NumPy is not installed")
    def test_unsigned_coefficients_and_bounds_do_not_wrap(self):
        u = np.uint8
        constraint = LinearConstraint("dose", u(2), u(0), "<=", 100)
        bounds = ProcessBounds((u(10), u(200)), (u(1), u(200)))
        with np.errstate(over="raise"):
            result = ConstraintSet(bounds, (constraint,)).evaluate(u(130), u(2))
        self.assertFalse(result.feasible)
        self.assertEqual(dict(result.violations), {"dose": 160.0})

    def test_overflowed_expression_cannot_report_zero_violation(self):
        constraint = LinearConstraint("dose", 1e308, -1e308, "<=", 100.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            constraint.violation(2.0, 2.0)
