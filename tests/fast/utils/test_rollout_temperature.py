import unittest
from argparse import Namespace

from miles.utils.rollout_temperature import validate_rollout_temperature


class TestRolloutTemperature(unittest.TestCase):
    def test_training_rejects_zero(self) -> None:
        args = Namespace(rollout_temperature=0.0, debug_rollout_only=False)

        with self.assertRaisesRegex(ValueError, "greater than 0 for training"):
            validate_rollout_temperature(args)

    def test_training_rejects_negative(self) -> None:
        args = Namespace(rollout_temperature=-1.0, debug_rollout_only=False)

        with self.assertRaisesRegex(ValueError, "non-negative for generation"):
            validate_rollout_temperature(args)

    def test_rollout_only_allows_greedy_zero(self) -> None:
        args = Namespace(rollout_temperature=0.0, debug_rollout_only=True)

        validate_rollout_temperature(args)

    def test_rollout_only_rejects_negative(self) -> None:
        args = Namespace(rollout_temperature=-1.0, debug_rollout_only=True)

        with self.assertRaisesRegex(ValueError, "non-negative for generation"):
            validate_rollout_temperature(args)

    def test_training_accepts_positive(self) -> None:
        args = Namespace(rollout_temperature=1.0, debug_rollout_only=False)

        validate_rollout_temperature(args)


if __name__ == "__main__":
    unittest.main()
