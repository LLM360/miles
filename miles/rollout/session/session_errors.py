"""Error types for the session module.

Hierarchy
---------
SessionError (base)
├── SessionNotFoundError       → 404  session does not exist
├── MessageValidationError     → 400  messages structure/content invalid
├── TokenizationError          → 500  TITO tokenizer / prefix mismatch
├── UpstreamResponseError      → 502  SGLang response invalid or unexpected
└── SessionStateConflictError  → 409  state changed during unlocked proxy phase
"""


class SessionError(Exception):
    """Base class for all session-related errors."""

    status_code: int = 500


class SessionNotFoundError(SessionError):
    """Raised when the requested session ID does not exist."""

    status_code: int = 404


class MessageValidationError(SessionError):
    """Raised when request messages fail structural validation.

    Examples: user message after assistant, messages not append-only,
    rollback failed (no assistant checkpoint in matched prefix).
    """

    status_code: int = 400


class TokenizationError(SessionError):
    """Raised when TITO tokenization invariants are violated.

    Examples: pretokenized prefix mismatch between stored and new token IDs.
    """

    status_code: int = 500


class UpstreamResponseError(SessionError):
    """Raised when the upstream SGLang response is invalid or unexpected.

    Examples: missing meta_info, assistant content is None,
    output_token_logprobs length mismatch.
    """

    status_code: int = 502


class SessionStateConflictError(SessionError):
    """Raised when session state changed during the unlocked proxy phase.

    The split-lock chat flow releases ``session.lock`` while proxying to
    SGLang.  If another writer commits an assistant turn in that window,
    this writer cannot safely commit its own response: the trajectory's
    accumulated_token_ids would no longer line up with the records list,
    causing the cursor-mismatch assertion in
    ``compute_samples_from_openai_records`` to fire downstream.

    We return 409 so the caller (litellm/harbor) treats this as a
    retryable conflict and does NOT incorporate the dropped turn into its
    local trajectory.  Evidence: run 1711903 (~/run_analysis/1711903/
    1711903_errors_rca.md) — 24 ``state changed during proxy`` warnings
    led to 2 cursor-mismatch failures when the dropped turns were silently
    returned as 200.
    """

    status_code: int = 409
