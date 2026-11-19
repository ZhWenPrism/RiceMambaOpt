import unittest
from ricemambaopt.evaluation.metrics import per_target_metrics, joint_normalized_rmse


class TargetObservationAlignmentTest(unittest.TestCase):
    def test_targets_cannot_silently_use_different_cohorts(self):
        truth = {"a": [1., 2.], "b": [1.]}
        predicted = {"a": [1., 2.], "b": [1.]}
        for evaluate in (lambda: per_target_metrics(truth, predicted),
                         lambda: joint_normalized_rmse(truth, predicted, {"a": 1., "b": 1.})):
            with self.assertRaisesRegex(ValueError, "same observations"):
                evaluate()
    def test_aligned_targets_keep_their_metrics(self):
        truth = {"a": [1., 2.], "b": [3., 4.]}
        self.assertEqual(joint_normalized_rmse(truth, truth, {"a": 1., "b": 1.}), 0.)
        self.assertEqual(set(per_target_metrics(truth, truth)), {"a", "b"})
