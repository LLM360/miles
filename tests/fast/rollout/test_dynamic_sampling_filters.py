from argparse import Namespace

import pytest

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.filter_hub.dynamic_sampling_filters import (
    check_no_infra_failures,
    check_no_invalid_outcomes,
    check_no_invalid_outcomes_then_nonzero_std,
)
from miles.rollout.filter_hub.rollout_filters import mask_nontrainable_outcomes
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


@pytest.mark.parametrize(
    "exit_status",
    [
        "AgentTimeout",
        "AgentTimeoutError",
        "VerifierTimeout",
        "VerifierTimeoutError",
        "_K8sInternalInfraError",
        "Cancelled",
        "RewardFileNotFoundError",
        "BadRequestFromFutureProvider",
    ],
)
def test_invalid_outcome_filter_rejects_partial_fatal_and_unknown_exits(exit_status: str) -> None:
    samples = [_sample(0, exit_status=exit_status), _sample(1)]

    output = check_no_invalid_outcomes(ARGS, samples)

    assert output == DynamicFilterOutput(keep=False, reason=f"group_has_{exit_status}")


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (Sample.Status.ABORTED, "group_has_aborted"),
        (Sample.Status.FAILED, "group_has_failed"),
        (Sample.Status.PENDING, "group_has_pending"),
    ],
)
def test_invalid_outcome_filter_rejects_unusable_sample_statuses(
    status: Sample.Status, reason: str
) -> None:
    output = check_no_invalid_outcomes(ARGS, [_sample(None, status=status), _sample(1)])

    assert output == DynamicFilterOutput(keep=False, reason=reason)


def test_invalid_outcome_filter_keeps_nested_policy_and_token_truncation_group() -> None:
    samples = [
        [_sample(0, exit_status="TestsFailed")],
        [_sample(1, status=Sample.Status.TRUNCATED, exit_status="BadRequestError")],
    ]

    output = check_no_invalid_outcomes(ARGS, samples)

    assert bool(output.keep)
    assert output.reason is None


@pytest.mark.parametrize(
    ("reward", "expected_reason"),
    [(1, "zero_std_1"), (0, "zero_std_0")],
    ids=["all-correct", "all-incorrect"],
)
def test_composed_filter_rejects_zero_std_after_outcome_checks(
    reward: int, expected_reason: str
) -> None:
    output = check_no_invalid_outcomes_then_nonzero_std(
        ARGS, [_sample(reward), _sample(reward)]
    )

    assert not bool(output.keep)
    assert output.reason == expected_reason


def test_composed_filter_reports_invalid_outcome_before_zero_std() -> None:
    output = check_no_invalid_outcomes_then_nonzero_std(
        ARGS,
        [_sample(0, exit_status="AgentTimeout"), _sample(0)],
    )

    assert output == DynamicFilterOutput(
        keep=False,
        reason="group_has_AgentTimeout",
    )


def test_composed_filter_excludes_truncated_sample_from_reward_std() -> None:
    output = check_no_invalid_outcomes_then_nonzero_std(
        ARGS,
        [
            _sample(1),
            _sample(1),
            _sample(
                100,
                status=Sample.Status.TRUNCATED,
                exit_status="LimitsExceeded",
            ),
        ],
    )

    assert not bool(output.keep)
    assert output.reason == "zero_std_1"


def test_composed_filter_rejects_when_truncation_leaves_one_complete_sample() -> None:
    output = check_no_invalid_outcomes_then_nonzero_std(
        ARGS,
        [
            _sample(1),
            _sample(
                0,
                status=Sample.Status.TRUNCATED,
                exit_status="LimitsExceeded",
            ),
        ],
    )

    assert output == DynamicFilterOutput(
        keep=False,
        reason="group_has_insufficient_complete_samples",
    )


def test_compatibility_filter_preserves_composed_behavior() -> None:
    output = check_no_infra_failures(ARGS, [_sample(0), _sample(1)])

    assert bool(output.keep)
    assert output.reason is None


@pytest.mark.parametrize(
    ("exit_status", "status", "expected_mask"),
    [
        ("Submitted", Sample.Status.COMPLETED, False),
        ("BadRequestError", Sample.Status.TRUNCATED, True),
        ("AgentTimeout", Sample.Status.TRUNCATED, True),
        ("Cancelled", Sample.Status.ABORTED, True),
        ("Submitted", Sample.Status.TRUNCATED, True),
    ],
)
def test_nontrainable_sample_filter_is_defensive(
    exit_status: str, status: Sample.Status, expected_mask: bool
) -> None:
    sample = _sample(0, status=status, exit_status=exit_status)

    mask_nontrainable_outcomes(ARGS, [sample])

    assert sample.remove_sample is expected_mask
