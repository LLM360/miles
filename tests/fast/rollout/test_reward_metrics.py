from types import SimpleNamespace

import pytest

from miles.ray.rollout import _compute_grouped_reward_metrics
from miles.utils.types import Sample


def test_grouped_reward_metrics_separate_raw_and_adjusted_rewards():
    samples = [
        Sample(
            reward=0.9,
            metadata={"raw_reward": 1.0, "raw_reward_adjusted": 0.9},
        ),
        Sample(
            reward=0.4,
            metadata={"raw_reward": 0.5, "raw_reward_adjusted": 0.4},
        ),
    ]

    metrics = _compute_grouped_reward_metrics(
        SimpleNamespace(reward_key=None), samples, "reward", len(samples)
    )

    assert metrics["reward/raw_reward"] == pytest.approx(0.75)
    assert metrics["reward/raw_reward_adjusted"] == pytest.approx(0.65)


def test_grouped_reward_metrics_preserve_legacy_reward_reporting():
    samples = [Sample(reward=0.25), Sample(reward=0.75)]

    metrics = _compute_grouped_reward_metrics(
        SimpleNamespace(reward_key=None), samples, "reward", len(samples)
    )

    assert metrics["reward/raw_reward"] == pytest.approx(0.5)
    assert "reward/raw_reward_adjusted" not in metrics
