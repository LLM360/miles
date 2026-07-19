import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pybase64

from miles.rollout.session.errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.types import SessionRecord
from miles.utils.chat_template_utils.message_matcher_hub import (
    SessionMessageMatcher,
    assert_messages_append_only_with_allowed_role,
    strict_message_matches,
)
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

logger = logging.getLogger(__name__)


# TODO: hardcoded to 1 for now; if multi-step rollback is actually needed,
#  raise this limit or make it configurable and remove the restriction.
MAX_ASSISTANT_ROLLBACK_STEPS = 1


def assert_pretokenized_prefix(
    prev: list[int],
    all_token_ids: list[int],
    *,
    max_trim_tokens: int,
    request_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
) -> None:
    """Stored token_ids must be a prefix of the new checkpoint, tolerating up
    to *max_trim_tokens* trailing differences. Pure token-level check, shared
    verbatim by the v1 checkpoint update and the v2 commit."""
    if not prev:
        return
    check_len = len(prev) - max_trim_tokens
    if check_len > 0 and all_token_ids[:check_len] != prev[:check_len]:
        first_mismatch = next(
            (i for i, (a, b) in enumerate(zip(all_token_ids[:check_len], prev[:check_len], strict=True)) if a != b),
            min(len(all_token_ids), check_len),
        )
        raise TokenizationError(
            f"pretokenized prefix mismatch: "
            f"stored {len(prev)} tokens (checking first {check_len}, "
            f"allowing {max_trim_tokens} trailing) are not a prefix of "
            f"prompt_token_ids + completion_token_ids "
            f"({len(all_token_ids)} tokens), "
            f"first mismatch at index {first_mismatch}, "
            f"matched {first_mismatch}/{check_len} prefix tokens\n"
            f"request_messages={request_messages}\n"
            f"assistant_message={assistant_message}"
        )


@dataclass
class LinearTrajectory:
    """State for a linear trajectory.

    Tracks the full message history and accumulated token IDs for one session.

    Session-generated assistant responses create checkpoints; client-injected assistant messages remain prompt history.

    Rollback uses ``generated_checkpoint_message_ends`` instead of inferring checkpoints from message roles.

    The typical message sequence is: [system?, user, assistant, tool, assistant, tool, …],
    but the agent may retry from an earlier point (e.g. re-running a tool call),
    in which case the session is rolled back at most one assistant step.

    Concurrency contract: all mutating methods must be called under ``self.lock``.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    closing: bool = field(default=False, repr=False, compare=False)
    messages: list[dict[str, Any]] = field(default_factory=list)
    records: list[SessionRecord] = field(default_factory=list)
    trajectory_token_ids: list[list[int]] = field(default_factory=list)
    generated_checkpoint_message_ends: list[int] = field(default_factory=list)
    num_assistant: int = 0

    # R3 returns routing for the complete token prefix on every turn.  Only the
    # newest prefix is consumed by the merged training sample, so keep that
    # large blob once per session instead of once per SessionRecord.
    latest_rollout_routed_experts: str | None = field(default=None, repr=False, compare=False)
    latest_rollout_routed_experts_num_tokens: int = field(default=0, repr=False, compare=False)

    @property
    def token_ids(self) -> list[int]:
        """Current token IDs — the latest assistant checkpoint."""
        return self.trajectory_token_ids[-1] if self.trajectory_token_ids else []

    def append_record(self, record: SessionRecord, *, use_addition_r3: bool = False) -> None:
        if use_addition_r3:
            # Incremental payloads are disjoint patches, not redundant prefixes.
            self.records.append(record)
            return
        choice = (record.response.get("choices") or [{}])[0]
        meta_info = choice.get("meta_info")
        routed_experts = meta_info.pop("routed_experts", None) if isinstance(meta_info, dict) else None
        choice_routed_experts = choice.pop("routed_experts", None)
        if routed_experts is None:
            routed_experts = choice_routed_experts

        if routed_experts is None:
            self.latest_rollout_routed_experts = None
            self.latest_rollout_routed_experts_num_tokens = 0
        else:
            if not isinstance(routed_experts, str):
                raise TypeError(f"routed_experts must be a base64 string, got {type(routed_experts).__name__}")
            if not self.token_ids:
                raise ValueError("cannot store routed_experts without a token checkpoint")
            self.latest_rollout_routed_experts = routed_experts
            self.latest_rollout_routed_experts_num_tokens = len(self.token_ids)

        self.records.append(record)

    def get_rollout_routed_experts(self, target_num_tokens: int | None = None) -> str | None:
        """Return the retained R3 blob, optionally truncated to a checkpoint.

        Routed experts are token-major with one row per token except the last.
        Derive the row width from the encoded payload so this state object does
        not need model topology arguments.
        """
        routed_experts = self.latest_rollout_routed_experts
        if routed_experts is None:
            return None

        current_num_tokens = self.latest_rollout_routed_experts_num_tokens
        if target_num_tokens is None:
            target_num_tokens = current_num_tokens
        if target_num_tokens == current_num_tokens:
            return routed_experts
        if target_num_tokens < 1 or target_num_tokens > current_num_tokens:
            raise ValueError(
                f"invalid routed_experts rollback target {target_num_tokens}; "
                f"current token count is {current_num_tokens}"
            )

        decoded = pybase64.b64decode(routed_experts.encode("ascii"))
        current_rows = current_num_tokens - 1
        if current_rows == 0 or len(decoded) % current_rows != 0:
            raise ValueError(
                f"routed_experts byte length {len(decoded)} is not divisible by "
                f"the current row count {current_rows}"
            )

        bytes_per_row = len(decoded) // current_rows
        target_bytes = (target_num_tokens - 1) * bytes_per_row
        return pybase64.b64encode(decoded[:target_bytes]).decode("ascii")

    def _truncate_latest_rollout_routed_experts(self, target_num_tokens: int) -> None:
        """Align the retained R3 blob in-place when a session rolls back."""
        routed_experts = self.get_rollout_routed_experts(target_num_tokens)
        if routed_experts is None:
            return
        self.latest_rollout_routed_experts = routed_experts
        self.latest_rollout_routed_experts_num_tokens = target_num_tokens

    def prepare_pretokenized(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        tito_tokenizer: TITOTokenizer,
        message_matcher: SessionMessageMatcher | None = None,
    ) -> list[int]:
        """Build the full prompt input_ids for *request_messages*.

        Validates that *request_messages* extends the stored history under
        *message_matcher* (defaults to the strict matcher), rolling back at
        most one assistant step on agent retries, then reuses the stored
        token_ids as the pretokenized prefix.  When no stored checkpoint
        is left to build on — the first turn, or a retry of the first turn that
        rolled the session back to empty — renders *request_messages* from
        scratch via the chat template instead.

        Must be called under ``self.lock``.
        """
        matcher = message_matcher if message_matcher is not None else strict_message_matches

        # 1. Detect agent retries and roll back (at most one assistant step). Retrying the
        #    first turn rolls back to the empty checkpoint, clearing token_ids.
        self._try_detect_and_rollback_to_assistant_checkpoint(request_messages, matcher)

        if not self.token_ids:
            return tito_tokenizer.apply_chat_template(
                request_messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
            )

        # 2. Confirm the (possibly rolled-back) stored messages are a prefix of request,
        #    and that each appended message role is in tito_tokenizer.allowed_append_roles.
        try:
            assert_messages_append_only_with_allowed_role(
                self.messages, request_messages, tito_tokenizer.allowed_append_roles, message_matcher=matcher
            )
        except ValueError as e:
            raise MessageValidationError(
                f"{e}; the selected TITO fixed template does not support appending this role"
            ) from e

        effective_messages = self.messages + request_messages[len(self.messages) :]
        return tito_tokenizer.merge_tokens(
            old_messages=self.messages,
            new_messages=effective_messages,
            pretokenized_token_ids=self.token_ids,
            tools=tools,
        )

    def update_pretokenized_state(
        self,
        request_messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        max_trim_tokens: int,
    ) -> None:
        """Store raw token IDs after a successful response.

        Appends ``prompt_token_ids + completion_token_ids`` as a new checkpoint.
        Validates that the previously stored token_ids are a prefix of the new
        checkpoint (tolerating up to ``max_trim_tokens`` trailing differences).
        Must be called under ``self.lock``.
        """
        all_token_ids = prompt_token_ids + completion_token_ids
        assert_pretokenized_prefix(
            self.token_ids,
            all_token_ids,
            max_trim_tokens=max_trim_tokens,
            request_messages=request_messages,
            assistant_message=assistant_message,
        )

        # Commit the same effective history the tokens were built from (stored
        # spellings for the reused prefix, the replay only for the new tail):
        # committing the raw replay would let an accepted-but-reserialized
        # prefix rewrite stored history, so the next canonical replay could
        # no longer match its own session.
        self.messages = self.messages + request_messages[len(self.messages) :] + [assistant_message]
        self.trajectory_token_ids.append(all_token_ids)
        self.generated_checkpoint_message_ends.append(len(request_messages) + 1)
        self.num_assistant = len(self.generated_checkpoint_message_ends)

    def _try_detect_and_rollback_to_assistant_checkpoint(
        self,
        request_messages: list[dict[str, Any]],
        message_matcher: SessionMessageMatcher,
    ) -> None:
        """Detect if *request_messages* diverges from stored history and roll back.

        In agentic workflows the agent may retry from an earlier point — for
        example, re-running a tool call with different arguments.  When that
        happens the new request shares a common prefix with the stored messages
        but diverges before the end.  This method truncates session state back
        to the last generated assistant checkpoint within the matching prefix,
        or to the empty checkpoint when the matching prefix holds no generated
        checkpoint at all.

        Except for a retry of the original prompt, a single-step rollback is allowed (controlled by
        ``MAX_ASSISTANT_ROLLBACK_STEPS``).  Discarding exactly one generated
        checkpoint means the agent is retrying from the preceding checkpoint —
        the request shares the stored prefix up to that generated response and
        then continues with whatever the agent chooses (same or different tool
        result, additional messages, etc.).  Any request that would need to
        discard more than one generated checkpoint (i.e. jump back across
        multiple turns) is rejected with ``MessageValidationError`` and no
        state is modified.

        Example — agent retries after the first tool call::

            stored:  [sys, user, assistant₁, tool₁, assistant₂]
                      ───────────────────── ▲
                      checkpoint 0 (assistant₁)   checkpoint 1 (assistant₂)

            request: [sys, user, assistant₁, tool₁_different, ...]
                                             ↑ diverges here (index 3)

            match_len = 3  (sys, user, assistant₁ all match)
            Last generated checkpoint in matched prefix → assistant₁ (checkpoint 0)
            discard_count = 2 - 1 = 1  (≤ MAX_ASSISTANT_ROLLBACK_STEPS)

            After rollback:
              messages           = [sys, user, assistant₁]
              trajectory_token_ids = [checkpoint_0_ids]
              records              = [record_0]
              num_assistant        = 1

        Example — agent retries the very first turn::

            stored:  [user, assistant₁]
            request: [user]
                           ↑ stored continues past the request (index 1)

            match_len = 1  (user matches), no generated checkpoint in the matched prefix
            Rollback target → the empty checkpoint (index -1)
            discard_count = 1 - 0 = 1  (≤ MAX_ASSISTANT_ROLLBACK_STEPS)

            After rollback the session is empty and the caller re-renders the
            prompt from scratch, so turn 1 regenerates like any later turn.

        No rollback occurs when:
        - The stored history is empty.
        - *request_messages* is a strict extension of stored messages
          (``match_len >= len(stored)``).
        """
        stored = self.messages
        if not stored or not self.trajectory_token_ids:
            return

        match_len = 0
        for i in range(min(len(request_messages), len(stored))):
            if message_matcher(stored[i], request_messages[i]):
                match_len = i + 1
            else:
                break

        if match_len >= len(stored):
            return

        # Only responses generated by this session create checkpoints.
        # Assistant messages won't create new checkpoints.
        checkpoint_index = -1
        for i in reversed(range(len(self.generated_checkpoint_message_ends))):
            if self.generated_checkpoint_message_ends[i] <= match_len:
                checkpoint_index = i
                break

        # No generated checkpoint in the matched prefix means the agent is retrying the
        # first turn, so roll back to the empty checkpoint and retain no messages.
        rollback_msg_end = self.generated_checkpoint_message_ends[checkpoint_index] if checkpoint_index >= 0 else 0
        discard_count = self.num_assistant - (checkpoint_index + 1)
        # A retry containing only the original prompt restarts the rollout.
        # Other retries retain upstream's one-generated-reply rollback limit.
        prompt_reset = checkpoint_index < 0 and match_len == len(request_messages)
        if discard_count > MAX_ASSISTANT_ROLLBACK_STEPS and not prompt_reset:
            raise MessageValidationError(
                f"rollback failed: discard_count={discard_count} exceeds "
                f"max_assistant_rollback_steps={MAX_ASSISTANT_ROLLBACK_STEPS} "
                f"(stored has {len(stored)} messages, "
                f"request has {len(request_messages)} messages)"
            )

        logger.info(
            "Rolling back session: stored %d messages / %d checkpoints -> "
            "checkpoint %d (messages[:%d]), discarding %d generated checkpoint(s)",
            len(stored),
            self.num_assistant,
            checkpoint_index,
            rollback_msg_end,
            discard_count,
        )

        if checkpoint_index < 0:
            self.latest_rollout_routed_experts = None
            self.latest_rollout_routed_experts_num_tokens = 0
        else:
            self._truncate_latest_rollout_routed_experts(len(self.trajectory_token_ids[checkpoint_index]))
        self.messages = stored[:rollback_msg_end]
        self.trajectory_token_ids = self.trajectory_token_ids[: checkpoint_index + 1]
        self.records = self.records[: checkpoint_index + 1]
        self.generated_checkpoint_message_ends = self.generated_checkpoint_message_ends[: checkpoint_index + 1]
        self.num_assistant = len(self.generated_checkpoint_message_ends)


class SessionRegistry:
    """Session ID -> trajectory mapping with shared tokenizer resources.

    Pure CRUD plus read-only computation (compute_session_mismatch).
    Does NOT mutate session state - all mutations are methods on
    LinearTrajectory; called by the route handler under session.lock.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        tito_tokenizer: TITOTokenizer,
        message_matcher: SessionMessageMatcher | None = None,
    ):
        self.sessions: dict[str, LinearTrajectory] = {}
        self._session_last_access: dict[str, float] = {}
        self._deleted_session_ids: set[str] = set()
        self._deleted_order: deque[str] = deque()
        self.tokenizer = tokenizer
        self.tito_tokenizer = tito_tokenizer
        self.comparator = tito_tokenizer.create_comparator()
        self.message_matcher: SessionMessageMatcher = (
            message_matcher if message_matcher is not None else strict_message_matches
        )

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = LinearTrajectory()
        return session_id

    def get_session(self, session_id: str) -> LinearTrajectory:
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        return session

    def get_or_create_session(self, session_id: str) -> LinearTrajectory:
        if session_id in self._deleted_session_ids:
            # Explicitly deleted: do not silently resurrect it. Auto-create is only
            # for ids the server has never seen (e.g. after a router restart).
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        session = self.sessions.get(session_id)
        if session is None:
            self._evict_stale_sessions()
            logger.warning("Auto-creating session %s (not found, likely router restart)", session_id)
            session = LinearTrajectory()
            self.sessions[session_id] = session
            self._session_last_access[session_id] = time.monotonic()
        else:
            self._session_last_access[session_id] = time.monotonic()
        return session

    _SESSION_TTL_SECS: int = 7200  # 2 hours
    _MAX_DELETED_TOMBSTONES: int = 10000  # bound the deleted-id memory

    def _evict_stale_sessions(self) -> None:
        """Remove auto-created sessions older than _SESSION_TTL_SECS."""
        if not self._session_last_access:
            return
        now = time.monotonic()
        stale = [sid for sid, ts in self._session_last_access.items() if now - ts > self._SESSION_TTL_SECS]
        for sid in stale:
            self.sessions.pop(sid, None)
            self._session_last_access.pop(sid, None)
        if stale:
            logger.info("Evicted %d stale auto-created sessions", len(stale))

    def remove_session(self, session_id: str) -> None:
        if self.sessions.pop(session_id, None) is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        self._session_last_access.pop(session_id, None)
        self._tombstone(session_id)

    def is_deleted(self, session_id: str) -> bool:
        """True if this id was explicitly deleted (and must not be auto-recreated)."""
        return session_id in self._deleted_session_ids

    def _tombstone(self, session_id: str) -> None:
        """Remember an explicitly deleted session id so it is not silently
        auto-recreated. Bounded to the most recent _MAX_DELETED_TOMBSTONES ids."""
        if session_id in self._deleted_session_ids:
            return
        if len(self._deleted_order) >= self._MAX_DELETED_TOMBSTONES:
            self._deleted_session_ids.discard(self._deleted_order.popleft())
        self._deleted_order.append(session_id)
        self._deleted_session_ids.add(session_id)

    def compute_session_mismatch(self, session: LinearTrajectory) -> list[dict] | None:
        """Compare accumulated token IDs against canonical chat template output.

        Read-only: does not mutate session state.
        """
        if not session.token_ids:
            return None
        try:
            tools = session.records[-1].request.get("tools") if session.records else None
            expected_ids = self.tito_tokenizer.apply_chat_template(
                session.messages,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )
            mismatches = self.comparator.compare_sequences(expected_ids, session.token_ids)
            return [m.to_dict() for m in mismatches]
        except Exception as e:
            raise TokenizationError(f"failed to compute tito_session_mismatch: {e}") from e
