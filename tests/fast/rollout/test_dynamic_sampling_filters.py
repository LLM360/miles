from argparse import Namespace

import pytest

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.filter_hub.dynamic_sampling_filters import (
    INFRA_FAILURE_EXIT_STATUSES,
    check_no_infra_failures,
)
from miles.utils.types import Sample


ARGS = Namespace(reward_key=None)


def _sample(
    reward: float | None,
    *,
    status: Sample.Status = Sample.Status.COMPLETED,
    exit_status: str | None = None,
) -> Sample:
    metadata = {} if exit_status is None else {"exit_status": exit_status}
    return Sample(reward=reward, status=status, metadata=metadata)


@pytest.mark.parametrize("exit_status", sorted(INFRA_FAILURE_EXIT_STATUSES))
def test_check_no_infra_failures_rejects_every_infra_exit_status(exit_status: str) -> None:
    # The rewards are mixed, so this group would otherwise pass the std check.
    samples = [_sample(0, exit_status=exit_status), _sample(1)]

    output = check_no_infra_failures(ARGS, samples)

    assert output == DynamicFilterOutput(keep=False, reason=f"group_has_{exit_status}")


def test_check_no_infra_failures_rejects_aborted_before_reward_check() -> None:
    samples = [_sample(None, status=Sample.Status.ABORTED), _sample(1)]

    output = check_no_infra_failures(ARGS, samples)

    assert output == DynamicFilterOutput(keep=False, reason="group_has_aborted")


def test_check_no_infra_failures_keeps_nested_mixed_reward_group() -> None:
    samples = [[_sample(0)], [_sample(1)]]

    output = check_no_infra_failures(ARGS, samples)

    assert bool(output.keep)
    assert output.reason is None


@pytest.mark.parametrize(
    ("reward", "expected_reason"),
    [
        (1, "zero_std_1"),
        (0, "zero_std_0"),
    ],
    ids=["all-correct", "all-incorrect"],
)
def test_check_no_infra_failures_rejects_zero_std_reward_groups(
    reward: int,
    expected_reason: str,
) -> None:
    output = check_no_infra_failures(ARGS, [_sample(reward), _sample(reward)])

    assert not bool(output.keep)
    assert output.reason == expected_reason


def test_check_no_infra_failures_allows_policy_failure_with_mixed_rewards() -> None:
    samples = [_sample(0, exit_status="TestsFailed"), _sample(1)]

    output = check_no_infra_failures(ARGS, samples)

    assert bool(output.keep)
    assert output.reason is None
