import pytest

from miles.rollout._agentic_outcomes import (
    _Outcome,
    classify_exit_status,
    masks_sample,
    rejects_group,
    resolve_sample_status,
)
from miles.utils.types import Sample


PRESERVE_EXIT_STATUSES = ["", "Submitted", "TestsFailed", "FormatError", "RepeatedToolCall"]
TOKEN_TRUNCATION_EXIT_STATUSES = [
    "BadRequestError",
    "ContextWindowExceededError",
    "OutputLengthExceededError",
    "LimitsExceeded",
]
PARTIAL_GROUP_FAILURE_EXIT_STATUSES = [
    "AgentTimeout",
    "AgentTimeoutError",
    "VerifierTimeout",
    "VerifierTimeoutError",
]
FATAL_FAILURE_EXIT_STATUSES = [
    "AgentSetupTimeout",
    "AgentSetupTimeoutError",
    "EnvStartTimeout",
    "EnvironmentStartTimeoutError",
    "TimeoutError",
    "HealthcheckError",
    "_K8sInternalInfraError",
    "SqsConsumerError",
    "AddTestsDirError",
    "NoReadyWorkers",
    "SessionNotFound",
    "SGLangGatewayNotReady",
    "AgentServerError",
    "AgentServerInvalidResponse",
    "Cancelled",
    "RewardFileNotFoundError",
    "ImportError",
    "InvalidInstanceId",
    "TaskNotFound",
    "FinishDeclarationUploadError",
    "Error: RuntimeError",
    "AgentError",
    "AgentFunctionError",
    "Unknown",
    "FutureUnmappedFailure",
]


@pytest.mark.parametrize("exit_status", PRESERVE_EXIT_STATUSES)
@pytest.mark.parametrize("effective_response_length", [0, 7])
def test_preserve_outcomes_keep_engine_status(exit_status: str, effective_response_length: int) -> None:
    outcome = classify_exit_status(exit_status)

    assert outcome is _Outcome.PRESERVE
    assert (
        resolve_sample_status(Sample.Status.COMPLETED, effective_response_length, outcome)
        is Sample.Status.COMPLETED
    )
    assert not rejects_group(outcome)
    assert not masks_sample(outcome)


@pytest.mark.parametrize("exit_status", TOKEN_TRUNCATION_EXIT_STATUSES)
def test_token_truncations_follow_active_token_rule(exit_status: str) -> None:
    outcome = classify_exit_status(exit_status)

    assert outcome is _Outcome.TOKEN_TRUNCATION
    assert resolve_sample_status(Sample.Status.COMPLETED, 7, outcome) is Sample.Status.TRUNCATED
    assert resolve_sample_status(Sample.Status.COMPLETED, 0, outcome) is Sample.Status.ABORTED
    assert not rejects_group(outcome)
    assert masks_sample(outcome)


@pytest.mark.parametrize("exit_status", PARTIAL_GROUP_FAILURE_EXIT_STATUSES)
def test_agent_and_verifier_timeouts_retain_partial_status_but_reject_group(exit_status: str) -> None:
    outcome = classify_exit_status(exit_status)

    assert outcome is _Outcome.PARTIAL_GROUP_FAILURE
    assert resolve_sample_status(Sample.Status.COMPLETED, 7, outcome) is Sample.Status.TRUNCATED
    assert resolve_sample_status(Sample.Status.COMPLETED, 0, outcome) is Sample.Status.ABORTED
    assert rejects_group(outcome)
    assert masks_sample(outcome)


@pytest.mark.parametrize("exit_status", FATAL_FAILURE_EXIT_STATUSES)
@pytest.mark.parametrize("effective_response_length", [0, 7])
def test_other_failures_always_abort_and_reject(
    exit_status: str, effective_response_length: int
) -> None:
    outcome = classify_exit_status(exit_status)

    assert outcome is _Outcome.FATAL_FAILURE
    assert resolve_sample_status(Sample.Status.COMPLETED, effective_response_length, outcome) is Sample.Status.ABORTED
    assert rejects_group(outcome)
    assert masks_sample(outcome)


def test_non_string_nonempty_status_fails_closed() -> None:
    assert classify_exit_status(123) is _Outcome.FATAL_FAILURE
