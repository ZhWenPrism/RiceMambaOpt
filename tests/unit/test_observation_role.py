import unittest

from ricemambaopt.datasets import MillingObservation, StudyRole


def observation(role):
    return MillingObservation("sample", "batch", "cultivar", 20, 900, {"yield": 0.8}, role)


class ObservationRoleTest(unittest.TestCase):
    def test_invalid_roles_cannot_enter_a_cohort(self):
        for role in ("test", "hold-out", None):
            with self.subTest(role=role):
                with self.assertRaisesRegex(ValueError, "role must be development, holdout, or prospective"):
                    observation(role)

    def test_serialised_roles_are_restored_as_enum_values(self):
        for role in StudyRole:
            self.assertIs(observation(role.value).role, role)
            self.assertIs(observation(role).role, role)
