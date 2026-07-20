from types import SimpleNamespace

import pytest

from miles.ray.rollout.metrics import _compute_grouped_reward_metrics
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

    metrics = _compute_grouped_reward_metrics(SimpleNamespace(reward_key=None), samples, "reward", len(samples))

    assert metrics["reward/raw_reward"] == pytest.approx(0.75)
    assert metrics["reward/raw_reward_adjusted"] == pytest.approx(0.65)


def test_grouped_reward_metrics_preserve_legacy_reward_reporting():
    samples = [Sample(reward=0.25), Sample(reward=0.75)]

    metrics = _compute_grouped_reward_metrics(SimpleNamespace(reward_key=None), samples, "reward", len(samples))

    assert metrics["reward/raw_reward"] == pytest.approx(0.5)
    assert "reward/raw_reward_adjusted" not in metrics


def test_adjusted_reward_survives_conversion_and_balanced_or_unbalanced_partitioning():
    from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

    from miles.ray.rollout.train_data_conversion import (
        ROLLOUT_DATA_VALUE_SPEC,
        convert_samples_to_train_data,
        split_train_data_by_dp_raw,
    )

    samples = make_samples_grouped(n_groups=2, group_size=2)
    for i, sample in enumerate(samples):
        sample.metadata.update(raw_reward=float(i), raw_reward_adjusted=float(i) - 0.1)
    for balance in [False, True]:
        args = make_args(rewards_normalization=False, balance_data=balance)
        data = convert_samples_to_train_data(args, samples, {}, None, None)
        assert data["raw_reward"] == [0.0, 1.0, 2.0, 3.0]
        assert data["raw_reward_adjusted"] == [-0.1, 0.9, 1.9, 2.9]
        shards = split_train_data_by_dp_raw(args, data, dp_size=2)
        assert len(shards) == 2
        for shard in shards:
            assert shard["raw_reward_adjusted"] == data["raw_reward_adjusted"]
    assert ROLLOUT_DATA_VALUE_SPEC["raw_reward_adjusted"].codec == "auto"
