import unittest

from ricemambaopt.datasets import ProcessBounds
from ricemambaopt.optimization.constraints import ConstraintSet, LinearConstraint


class ConstraintSetSnapshotTests(unittest.TestCase):
    def test_caller_mutation_does_not_change_feasibility(self):
        constraints = [LinearConstraint("dose", 1., 1., "<=", 5.)]
        configured = ConstraintSet(ProcessBounds((0., 10.), (0., 10.)), constraints)
        before = configured.evaluate(4., 4.)
        constraints.clear()
        self.assertEqual(configured.evaluate(4., 4.), before)
        self.assertFalse(before.feasible)

    def test_ambiguous_names_fail_when_configured(self):
        bound = ProcessBounds((0., 10.), (0., 10.))
        for names in (("dose", "dose"), ("time_lower",)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                ConstraintSet(bound, [LinearConstraint(name, 1., 0., "<=", 5.) for name in names])

    def test_generator_constraints_are_preserved(self):
        bound = ProcessBounds((0., 10.), (0., 10.))
        configured = ConstraintSet(bound, (LinearConstraint("dose", 1., 0., "<=", 5.) for _ in range(1)))
        self.assertEqual(configured.evaluate(6., 1.).violations["dose"], 1.)
        self.assertEqual(configured.evaluate(6., 1.).violations["dose"], 1.)
