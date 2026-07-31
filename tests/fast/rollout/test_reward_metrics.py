from types import SimpleNamespace

import pytest

from miles.ray.rollout import (
    _compute_grouped_reward_metrics,
    _normalize_group_rewards_excluding_truncated,
)
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


def test_group_normalization_excludes_truncated_samples_from_baseline():
    samples = [
        Sample(reward=0.0, status=Sample.Status.COMPLETED),
        Sample(reward=1.0, status=Sample.Status.COMPLETED),
        Sample(reward=100.0, status=Sample.Status.TRUNCATED),
    ]

    rewards = _normalize_group_rewards_excluding_truncated(
        [0.0, 1.0, 100.0],
        samples,
        group_size=3,
        expected_group_count=1,
        std_normalization=False,
    )

    assert rewards == pytest.approx([-0.5, 0.5, 0.0])


def test_group_std_normalization_uses_only_complete_samples():
    samples = [
        Sample(reward=0.0, status=Sample.Status.COMPLETED),
        Sample(reward=2.0, status=Sample.Status.COMPLETED),
        Sample(reward=100.0, status=Sample.Status.TRUNCATED),
    ]

    rewards = _normalize_group_rewards_excluding_truncated(
        [0.0, 2.0, 100.0],
        samples,
        group_size=3,
        expected_group_count=1,
        std_normalization=True,
    )

    assert rewards == pytest.approx(
        [-1 / 2**0.5, 1 / 2**0.5, 0.0],
        abs=1e-6,
    )
