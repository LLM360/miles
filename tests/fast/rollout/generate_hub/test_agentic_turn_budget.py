from miles.rollout.generate_hub.agentic_tool_call import _apply_agentic_outcome_status
from miles.utils.types import Sample


def test_limits_exceeded_with_active_tokens_is_truncated() -> None:
    sample = Sample(
        status=Sample.Status.COMPLETED,
        response_length=2,
        loss_mask=[1, 1],
    )

    _apply_agentic_outcome_status([sample], {"exit_status": "LimitsExceeded"})

    assert sample.status is Sample.Status.TRUNCATED


def test_limits_exceeded_without_active_tokens_is_aborted() -> None:
    sample = Sample(status=Sample.Status.COMPLETED, response_length=2, loss_mask=[0, 0])

    _apply_agentic_outcome_status([sample], {"exit_status": "LimitsExceeded"})

    assert sample.status is Sample.Status.ABORTED


def test_submitted_preserves_completed_status() -> None:
    sample = Sample(status=Sample.Status.COMPLETED)

    _apply_agentic_outcome_status([sample], {"exit_status": "Submitted"})

    assert sample.status is Sample.Status.COMPLETED
