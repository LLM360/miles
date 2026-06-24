from types import SimpleNamespace

import pytest

from miles.rollout.filter_hub.dynamic_sampling_filters import (
    check_reward_nonzero_std,
    drop_truncated_or_extreme_pass_rate,
    drop_zero_std_groups_and_extreme_pass_rate,
)
from miles.rollout.filter_hub.rollout_filters import mask_truncated_and_llm_judge_failed
from miles.utils.types import Sample


def args(**kwargs):
    return SimpleNamespace(
        **dict(
            reward_key=None,
            dynamic_sampling_min_reward_std=1e-3,
            dynamic_sampling_min_mean_reward=0.1,
            dynamic_sampling_max_mean_reward=0.8,
        )
        | kwargs
    )


@pytest.mark.parametrize(
    "rewards,reason",
    [
        ([], "group_has_no_samples"),
        ([None, 1], "group_has_missing_reward"),
        ([1, 1], "near_zero_std_0.001"),
        ([0.9, 1], "mean_reward_too_high"),
        ([0, 0.1], "mean_reward_too_low"),
        ([0, 0.2], None),
        ([0.6, 1], None),
    ],
)
def test_thresholds_and_boundaries(rewards, reason):
    result = drop_zero_std_groups_and_extreme_pass_rate(args(), [[Sample(reward=r) for r in rewards]])
    assert result.keep == (reason is None)
    assert result.reason == reason


@pytest.mark.parametrize("function", [check_reward_nonzero_std, drop_zero_std_groups_and_extreme_pass_rate])
def test_single_rollout_cannot_estimate_reward_std(function):
    with pytest.raises(ValueError, match="at least 2"):
        function(args(), [Sample(reward=1)])


def test_truncated_group_rejected_before_reward_statistics():
    result = drop_truncated_or_extreme_pass_rate(args(), [[Sample(status=Sample.Status.TRUNCATED, reward=None)]])
    assert result.reason == "group_has_truncated"


def test_mask_preserves_samples_and_existing_removal():
    samples = [
        Sample(metadata=None),
        Sample(status=Sample.Status.TRUNCATED),
        Sample(metadata={"llm_judge_failed": True}),
        Sample(remove_sample=True),
    ]
    mask_truncated_and_llm_judge_failed(args(), [samples])
    assert [s.remove_sample for s in samples] == [False, True, True, True]
