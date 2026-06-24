from argparse import Namespace

import pytest
import torch

from miles.utils.training_semantics import (
    post_process_rewards_excluding_removed,
    validate_loss_masks_for_removed_samples,
)
from miles.utils.types import Sample


def _args(**overrides):
    return Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        reward_key=None,
        **overrides,
    )


def _sample(reward, *, remove_sample=False):
    return Sample(reward=reward, remove_sample=remove_sample)


def test_removed_samples_are_excluded_from_group_reward_normalization():
    samples = [
        _sample(1.0),
        _sample(None, remove_sample=True),
        _sample(3.0),
        _sample(5.0),
    ]

    raw_rewards, rewards = post_process_rewards_excluding_removed(_args(), samples)

    assert raw_rewards == [1.0, None, 3.0, 5.0]
    assert rewards == [0.0, 0.0, -1.0, 1.0]


def test_non_removed_sample_with_missing_reward_raises():
    with pytest.raises(ValueError, match="non-finite reward"):
        post_process_rewards_excluding_removed(_args(), [_sample(None)])


def test_zero_loss_mask_requires_explicit_removed_sample():
    validate_loss_masks_for_removed_samples(
        [torch.tensor([0, 0]), torch.tensor([1, 0])],
        [2, 2],
        [True, False],
    )

    with pytest.raises(ValueError, match="remove_samples\\[0\\] is false"):
        validate_loss_masks_for_removed_samples(
            [torch.tensor([0, 0])],
            [2],
            [False],
        )
