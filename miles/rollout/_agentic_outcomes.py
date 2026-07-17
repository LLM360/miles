"""Private outcome policy for Harbor-backed agentic rollouts.

Harbor reports a factual ``exit_status``.  Miles owns the separate decisions
about trajectory completeness and whether the result is valid training data.
Keep this module private so those decisions can evolve without creating a
public compatibility surface.
"""

from enum import Enum, auto

from miles.utils.types import Sample


class _Outcome(Enum):
    PRESERVE = auto()
    TOKEN_TRUNCATION = auto()
    PARTIAL_GROUP_FAILURE = auto()
    FATAL_FAILURE = auto()


_PRESERVE_EXIT_STATUSES = frozenset(
    {
        "Submitted",
        "TestsFailed",
        "FormatError",
        "RepeatedToolCall",
    }
)

_TOKEN_TRUNCATION_EXIT_STATUSES = frozenset(
    {
        "BadRequestError",
        "ContextWindowExceededError",
        "OutputLengthExceededError",
        "LimitsExceeded",
    }
)

# These failures can leave a useful partial trajectory for diagnostics, but
# invalidate the whole comparison group because the reward is not comparable.
_PARTIAL_GROUP_FAILURE_EXIT_STATUSES = frozenset(
    {
        "AgentTimeout",
        "AgentTimeoutError",
        "VerifierTimeout",
        "VerifierTimeoutError",
    }
)


def classify_exit_status(exit_status: object) -> _Outcome:
    """Classify a raw Harbor exit status without rewriting it.

    Empty status is the legacy success path.  Unknown non-empty statuses fail
    closed so a newly introduced infrastructure error cannot silently train.
    """
    if exit_status is None or exit_status == "":
        return _Outcome.PRESERVE
    if not isinstance(exit_status, str):
        return _Outcome.FATAL_FAILURE

    status = exit_status.strip()
    if not status:
        return _Outcome.PRESERVE
    if status in _PRESERVE_EXIT_STATUSES:
        return _Outcome.PRESERVE
    if status in _TOKEN_TRUNCATION_EXIT_STATUSES:
        return _Outcome.TOKEN_TRUNCATION
    if status in _PARTIAL_GROUP_FAILURE_EXIT_STATUSES:
        return _Outcome.PARTIAL_GROUP_FAILURE
    return _Outcome.FATAL_FAILURE


def resolve_sample_status(
    current_status: Sample.Status,
    effective_response_length: int,
    outcome: _Outcome,
) -> Sample.Status:
    """Resolve trajectory completeness independently from trainability."""
    if outcome is _Outcome.PRESERVE:
        return current_status
    if outcome in {_Outcome.TOKEN_TRUNCATION, _Outcome.PARTIAL_GROUP_FAILURE}:
        return Sample.Status.TRUNCATED if effective_response_length > 0 else Sample.Status.ABORTED
    return Sample.Status.ABORTED


def rejects_group(outcome: _Outcome) -> bool:
    return outcome in {_Outcome.PARTIAL_GROUP_FAILURE, _Outcome.FATAL_FAILURE}


def masks_sample(outcome: _Outcome) -> bool:
    """Return whether an outcome should receive zero gradient if it survives."""
    return outcome is not _Outcome.PRESERVE
