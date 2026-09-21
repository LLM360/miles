from types import SimpleNamespace

import pytest

from miles.rollout.value_pretrain_rollout import episode_reward_from_sample


def _sample(**overrides):
    defaults = dict(reward=None, label=None, metadata={})
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_episode_reward_prefers_sample_reward() -> None:
    assert episode_reward_from_sample(_sample(reward=1, label="0")) == 1.0
    assert episode_reward_from_sample(_sample(reward=0.0)) == 0.0
    assert episode_reward_from_sample(_sample(reward=False)) == 0.0


def test_episode_reward_falls_back_to_metadata_then_label() -> None:
    assert episode_reward_from_sample(_sample(metadata={"reward": 0.4})) == 0.4
    assert episode_reward_from_sample(_sample(label="1")) == 1.0
    assert episode_reward_from_sample(_sample(reward={"score": 0.2})) == 0.2


def test_episode_reward_rejects_missing() -> None:
    with pytest.raises(ValueError, match="missing finite reward"):
        episode_reward_from_sample(_sample())
