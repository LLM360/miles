import math

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.train_data_conversion import _post_process_rewards
from miles.utils.types import Sample


@pytest.mark.parametrize("std", [False, True])
def test_nested_rollouts_count_once_and_only_truncated_siblings_get_zero(std):
    args = make_args(grpo_std_normalization=std)
    rows = [(10, 0.0, False), (11, 2.0, True), (11, 2.0, False), (11, 2.0, False), (12, 100.0, True)]
    samples = [
        make_sample(
            group_index=0,
            rollout_id=rid,
            reward=reward,
            status=Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED,
        )
        for rid, reward, truncated in rows
    ]
    raw, rewards = _post_process_rewards(args, samples, None)
    scale = math.sqrt(2) + 1e-6 if std else 1.0
    assert raw == [0.0, 2.0, 2.0, 2.0, 100.0]
    assert rewards == pytest.approx([-1 / scale, 0, 1 / scale, 1 / scale, 0])


@pytest.mark.parametrize("complete", [0, 1])
@pytest.mark.parametrize("std", [False, True])
def test_no_baseline_signal_with_fewer_than_two_complete_rollouts(complete, std):
    args = make_args(grpo_std_normalization=std)
    samples = [
        make_sample(
            group_index=0,
            rollout_id=i,
            reward=float(i),
            status=Sample.Status.COMPLETED if i < complete else Sample.Status.TRUNCATED,
        )
        for i in range(3)
    ]
    assert _post_process_rewards(args, samples, None)[1] == [0.0, 0.0, 0.0]


def test_excluded_nonfinite_reward_cannot_poison_complete_rollouts():
    args = make_args()
    samples = [
        make_sample(
            group_index=0,
            rollout_id=i,
            reward=reward,
            status=Sample.Status.TRUNCATED if i == 2 else Sample.Status.COMPLETED,
        )
        for i, reward in enumerate([0.0, 2.0, float("nan")])
    ]
    assert _post_process_rewards(args, samples, None)[1] == [-1.0, 1.0, 0.0]
