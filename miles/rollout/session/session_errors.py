"""Error types for the session module.

Hierarchy
---------
SessionError (base)
├── SessionNotFoundError         → 404  session does not exist
├── MessageValidationError       → 400  messages structure/content invalid
├── SessionStateConflictError    → 409  Phase-3 commit guard fired (defense-in-depth)
├── TokenizationError            → 500  TITO tokenizer / prefix mismatch
└── UpstreamResponseError        → 502  SGLang response invalid or unexpected
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


class SessionStateConflictError(SessionError):
    """Raised as a defensive assert when the Phase-3 commit guard fires.

    With the lock-restored chat flow (lock held through Phase 1+2+3), this
    branch is unreachable in practice — no other writer can mutate the
    session while the proxy is in flight. We keep the guard + 409 surface
    as a defense in depth: if a future change reintroduces a split-lock
    window, callers see a clear retryable conflict instead of a silently
    dropped commit that would corrupt the trajectory's accumulated state.
    """

    status_code: int = 409


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
