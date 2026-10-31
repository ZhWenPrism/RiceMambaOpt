import unittest

from ricemambaopt.datasets import ProcessBounds
from ricemambaopt.optimization import ConstraintSet


class BoundsSnapshotTests(unittest.TestCase):
    def test_constraints_keep_original_equipment_bounds(self):
        times, speeds = [10.0, 60.0], [800.0, 1600.0]
        bounds = ProcessBounds(times, speeds)
        constraints = ConstraintSet(bounds)
        times[0], speeds[1] = 100.0, 0.0
        self.assertTrue(bounds.contains(30.0, 1200.0))
        self.assertTrue(constraints.evaluate(30.0, 1200.0).feasible)
        self.assertEqual(constraints.project_to_bounds(0.0, 2000.0), (10.0, 1600.0))

    def test_iterable_bounds_are_materialized_before_validation(self):
        with self.assertRaises(ValueError):
            ProcessBounds(iter([float("nan"), 60.0]), (800.0, 1600.0))
