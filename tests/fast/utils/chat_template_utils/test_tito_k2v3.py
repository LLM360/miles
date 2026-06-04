"""TITO contract tests for the K2V3 family.

Coverage contract — this file protects these invariants:

  (I1) K2V3 canonical chat template renders ``<|im_end|>\\n`` after every
       message (the trailing ``\\n`` comes from jinja block whitespace).
  (I2) Realistic rollout buffers can end at ``<|im_end|>`` WITHOUT the
       trailing ``\\n`` — the model stops at ``<|im_end|>`` on
       autoregressive emission.
  (I3) ``K2V3TITOTokenizer.merge_tokens`` inserts the missing ``\\n``
       when ``prefix[-1] == <|im_end|>``, so the merged buffer matches
       canonical render.
  (I4) Appended env messages (tool / user / system) round-trip through
       ``merge_tokens`` and still match the canonical render — across
       both realistic single-turn buffers and multi-turn parser-driven
       session histories.

The file is split into three banner-marked sections:

  SECTION A — CORE INVARIANT TESTS (I1-I4)
      * ``test_buffer_matches_canonical_under_realistic_rollout``
            — I1 + I2 + I3
      * ``test_append_via_realistic_buffer``
            — I3 + I4 (core; 8 trajectories × 4 env shapes = 32 cases)
      * ``test_chat_template_round_trip_through_real_sglang_parsers``
            — I4 with parser-derived ``parsed_msg`` substituted for raw
            model emit (structural round-trip only)

  SECTION B — INTEGRATION STRESS
      * ``test_end_to_end_realistic_rollout_with_real_parsers``
            — I3 + I4 on parser-tainted multi-turn session.messages;
            failure here that doesn't reproduce in section A is a
            parser-interaction regression specific to accumulated state

  SECTION C — SANITY (orthogonal to I1-I4)
      * ``test_production_prefix_check_raises_on_intentional_violation``
            — runtime defense (``update_pretokenized_state``'s prefix
            check) is alive
      * ``test_k2v3_subclass_is_wired``
            — registry returns the K2V3 subclass, not the base

Why this file exists separately from ``test_tito_tokenizer_model_matrix.py``:
that file builds ``pretokenized`` via ``apply_chat_template(..., add_generation_prompt=False)``,
which already contains the trailing ``\\n``, so the boundary fix path
never fires and the test passes whether the fix exists or not. This file
routes through ``update_pretokenized_state`` instead, producing the
realistic ``prefix[-1] == <|im_end|>`` state that the fix exists for.

Skips at module level if the K2V3 checkpoint is not on this host.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass

import pytest
from transformers import AutoTokenizer

from miles.rollout.session.linear_trajectory import LinearTrajectory
from miles.rollout.session.session_errors import TokenizationError
from miles.utils.chat_template_utils import MismatchType, apply_chat_template, try_get_fixed_chat_template
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType, get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.mock_trajectories import (
    LongChainThinkingTrajectory,
    LongChainTrajectory,
    MultiToolSingleTurnTrajectory,
    MultiTurnThinkingTrajectory,
    MultiTurnTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)

# ---------------------------------------------------------------------------
# Path + fixtures
# ---------------------------------------------------------------------------

K2V3_MODEL_PATH = os.environ.get(
    "TITO_TEST_MODEL_PATH_K2V3",
    "/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-8b-mid3-final",
)
_ALLOWED_APPEND_ROLES = ["tool", "user", "system"]

# K2V3 chat template's generation prompt depends on reasoning_effort
# (high → <think>, medium → <think_fast>, low → <think_faster>). Production
# runs with high effort; pinning here so test is deterministic regardless
# of any future template-default change. Override via env if needed.
_K2V3_REASONING_EFFORT = os.environ.get("TITO_TEST_REASONING_EFFORT_K2V3", "high")
_K2V3_CHAT_TEMPLATE_KWARGS = {"reasoning_effort": _K2V3_REASONING_EFFORT}

# Per-K2V3 SGLang parser names. Defaults match the K2V3 production
# config:
#   SGLANG_TOOL_PARSER=hermes
#   SGLANG_REASONING_PARSER=deepseek-r1
# Both rely on `<think>...</think>` (deepseek-r1) and the hermes
# `<tool_call>\n{json}\n</tool_call>` shape that K2V3's chat template emits.
#
# Older SGLang builds may register `hermes` under a different name (e.g.
# the qwen25 detector handles the same shape). Override via env in those
# environments — e.g. ``TITO_TEST_TOOL_PARSER_K2V3=qwen25``. If the
# configured parser is not registered in this SGLang build, the parser
# round-trip test skips with an explicit reason rather than silently
# turning green.
_K2V3_TOOL_PARSER = os.environ.get("TITO_TEST_TOOL_PARSER_K2V3", "hermes")
_K2V3_REASONING_PARSER = os.environ.get("TITO_TEST_REASONING_PARSER_K2V3", "deepseek-r1")


@pytest.fixture(scope="module")
def tokenizer() -> AutoTokenizer:
    if not os.path.isdir(K2V3_MODEL_PATH):
        pytest.skip(f"K2V3 checkpoint not present on this host: {K2V3_MODEL_PATH}")
    return load_tokenizer(
        K2V3_MODEL_PATH,
        chat_template_path=try_get_fixed_chat_template(K2V3_MODEL_PATH),
        trust_remote_code=True,
    )


@pytest.fixture
def tito_tok(tokenizer):
    return get_tito_tokenizer(
        tokenizer,
        tokenizer_type=TITOTokenizerType.K2V3,
        allowed_append_roles=_ALLOWED_APPEND_ROLES,
        chat_template_kwargs=_K2V3_CHAT_TEMPLATE_KWARGS,
    )


# ---------------------------------------------------------------------------
# Trajectories — realistic conversation shapes from mock_trajectories
# ---------------------------------------------------------------------------


def _with_synthetic_thinking(
    trajectory_cls: type,
    reasoning: str = "Let me work through this step by step.",
) -> type:
    """Synthesize a thinking variant by injecting ``reasoning_content`` on
    each assistant message of the trajectory.

    Used to build coverage shapes that ``mock_trajectories`` doesn't ship
    a native thinking variant for (e.g. multi-tool single-turn with
    thinking — production exercises this combination but no native
    fixture exists).
    """
    new_messages = deepcopy(trajectory_cls.MESSAGES)
    for m in new_messages:
        if m.get("role") == "assistant":
            m["reasoning_content"] = reasoning

    class _Synthesized:
        TOOLS = deepcopy(getattr(trajectory_cls, "TOOLS", None))
        MESSAGES = new_messages

    _Synthesized.__name__ = trajectory_cls.__name__ + "_WithSyntheticThinking"
    return _Synthesized


# Native + synthetic-thinking-injected trajectories. Each entry exercises a
# distinct rollout shape; the thinking variants additionally trigger the
# K2V3 chat template's reasoning-block path (<|im_start|>assistant\n<think>\n
# ... </think>\ncontent<|im_end|>).
CONVERSATIONS: list[tuple[str, type]] = [
    # Single assistant turn — single tool call.
    ("single_tool", SingleToolTrajectory),
    ("single_tool_thinking", SingleToolThinkingTrajectory),
    # Multiple assistant turns — single tool call per turn.
    ("multi_turn", MultiTurnTrajectory),
    ("multi_turn_thinking", MultiTurnThinkingTrajectory),
    # Single assistant turn — multiple parallel tool calls.
    ("multi_tool_single_turn", MultiToolSingleTurnTrajectory),
    # No native thinking variant exists for parallel-tools-single-turn;
    # synthesize by injecting reasoning_content into the assistant turn.
    ("multi_tool_single_turn_thinking", _with_synthetic_thinking(MultiToolSingleTurnTrajectory)),
    # Multiple assistant turns AND tool calls (chain shape).
    ("multi_tool_multi_turn", LongChainTrajectory),
    ("multi_tool_multi_turn_thinking", LongChainThinkingTrajectory),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_text(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
    *,
    add_generation_prompt: bool,
) -> str:
    """``apply_chat_template(...) → str`` with K2V3 chat_template_kwargs auto-applied."""
    return apply_chat_template(
        messages,
        tokenizer=tokenizer,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **_K2V3_CHAT_TEMPLATE_KWARGS,
    )


def _render_ids(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    tools: list[dict] | None,
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """``apply_chat_template(...) → list[int]`` with K2V3 chat_template_kwargs auto-applied."""
    return list(
        apply_chat_template(
            messages,
            tokenizer=tokenizer,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **_K2V3_CHAT_TEMPLATE_KWARGS,
        )
    )


def _first_diff(a, b) -> str:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return f"position {i}: a[{i}]={a[i]} b[{i}]={b[i]}"
    return f"length differs (len(a)={len(a)} len(b)={len(b)})"


def _assistant_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


def _realistic_emit_ids(
    request_messages: list[dict],
    assistant_message: dict,
    tools: list[dict] | None,
    tokenizer: AutoTokenizer,
) -> list[int]:
    """Synthesize completion_token_ids that mirror SGLang's autoregressive emit.

    The model emits starting from inside the assistant generation prompt
    and stops at ``<|im_end|>`` (no trailing ``\\n``). We compute this by
    diffing two chat-template renders:

        full   = render(request + [assistant], add_generation_prompt=False)
        prompt = render(request,               add_generation_prompt=True)
        emit_text = full[len(prompt):]                # what model would emit
        emit_text = emit_text.rstrip("\\n")          # strip jinja's trailing \\n
        assert emit_text.endswith("<|im_end|>")
        emit_ids = tokenizer.encode(emit_text)
    """
    full_text = _render_text(
        request_messages + [assistant_message],
        tokenizer,
        tools,
        add_generation_prompt=False,
    )
    prompt_text = _render_text(
        request_messages,
        tokenizer,
        tools,
        add_generation_prompt=True,
    )
    assert full_text.startswith(prompt_text), (
        "chat template not append-only: prompt-only render is not a prefix "
        "of full render. TITO's premise breaks here."
    )
    emit_text = full_text[len(prompt_text) :]
    # Strip the trailing newline(s) the jinja whitespace adds after
    # `<|im_end|>`. The model autoregressively stops at the stop token
    # without producing them.
    emit_text_stop = emit_text.rstrip("\n")
    assert emit_text_stop.endswith("<|im_end|>"), (
        f"unexpected emit_text shape (does not end with <|im_end|>): " f"{emit_text_stop!r}"
    )
    return list(tokenizer.encode(emit_text_stop, add_special_tokens=False))


def _drive_session_through_trajectory(
    session: LinearTrajectory,
    tito_tok,
    messages: list[dict],
    tools: list[dict] | None,
) -> None:
    """Drive ``session`` turn-by-turn using the trajectory's messages.

    For each assistant message in the trajectory, builds the realistic
    emit_ids and calls ``update_pretokenized_state`` exactly as production
    does. After this call, ``session.token_ids`` reflects what the rollout
    buffer would hold mid-conversation.
    """
    for asst_idx in _assistant_indices(messages):
        request_messages = messages[:asst_idx]
        assistant_message = messages[asst_idx]

        pre = session.prepare_pretokenized(request_messages, tools, tito_tokenizer=tito_tok)
        if pre is None:
            prompt_ids = _render_ids(
                request_messages,
                tito_tok.tokenizer,
                tools,
                add_generation_prompt=True,
            )
        else:
            prompt_ids = list(pre["input_ids"])

        emit_ids = _realistic_emit_ids(request_messages, assistant_message, tools, tito_tok.tokenizer)

        session.update_pretokenized_state(
            request_messages=request_messages,
            assistant_message=assistant_message,
            prompt_token_ids=prompt_ids,
            completion_token_ids=emit_ids,
            max_trim_tokens=tito_tok.max_trim_tokens,
        )


# ###########################################################################
# ###########################################################################
# ##                                                                       ##
# ##  SECTION A — CORE INVARIANT TESTS                                     ##
# ##                                                                       ##
# ##  Each test below leads with the invariant(s) it protects (I1-I4 per   ##
# ##  module docstring). These are the tests a reviewer should read first  ##
# ##  to understand the contract this file enforces.                       ##
# ##                                                                       ##
# ###########################################################################
# ###########################################################################


@pytest.mark.parametrize(
    "name, trajectory_cls",
    CONVERSATIONS,
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_buffer_matches_canonical_under_realistic_rollout(name, trajectory_cls, tito_tok):
    """Invariants I1+I2+I3: rollout buffer ending at ``<|im_end|>`` (no
    trailing ``\\n``) merges back to canonical chat-template render.

    Phase 1 compares the finalized session buffer to canonical. Phase 2
    appends a synthetic tool follow-up so ``merge_tokens`` runs against
    a buffer whose last token is ``<|im_end|>`` even on single-turn
    trajectories (defeats ``trim_trailing_ids`` shielding that would
    otherwise hide a missing boundary fix).

    ``ASSISTANT_TEXT`` mismatches are tolerated (BPE-merge noise,
    non-severe by the comparator); ``SPECIAL_TOKEN_*`` and
    ``NON_ASSISTANT_TEXT`` mismatches fail the test.
    """
    messages = deepcopy(trajectory_cls.MESSAGES)
    tools = deepcopy(getattr(trajectory_cls, "TOOLS", None))

    session = LinearTrajectory()
    _drive_session_through_trajectory(session, tito_tok, messages, tools)

    comparator = tito_tok.create_comparator()

    # Phase 1 — finalized buffer vs canonical (covers structural drift in the
    # whole trajectory, but the comparator's ``trim_trailing_ids`` hides
    # end-of-sequence ``<|im_end|>`` vs ``<|im_end|>\\n`` differences if the
    # trajectory has only ONE assistant turn).
    expected_final = _render_ids(
        session.messages,
        tito_tok.tokenizer,
        tools,
        add_generation_prompt=False,
    )
    actual_final = list(session.token_ids)
    severe_final = [
        m for m in comparator.compare_sequences(expected_final, actual_final) if m.type != MismatchType.ASSISTANT_TEXT
    ]
    if severe_final:
        details = "\n".join(
            f"  {m.type.value} at segment {m.segment_index}: "
            f"expected={m.expected_text!r} actual={m.actual_text!r}" + (f" — {m.detail}" if m.detail else "")
            for m in severe_final[:5]
        )
        pytest.fail(
            f"K2V3 [{name}] phase-1 (finalized buffer) canonical mismatch.\n"
            f"  first_diff: {_first_diff(expected_final, actual_final)}\n{details}"
        )

    # Phase 2 — force the boundary fix path even for single-assistant-turn
    # trajectories: simulate a NEXT-turn env append by calling
    # ``prepare_pretokenized`` with one extra ``tool`` message. This triggers
    # ``tito_tok.merge_tokens(...)`` against a buffer whose last token is
    # ``<|im_end|>`` (the model's autoregressive stop), which is the
    # production state the boundary fix exists for. The follow-up moves the
    # ``<|im_end|>`` from end-of-sequence to mid-sequence, defeating
    # ``trim_trailing_ids`` and surfacing missing-fix bugs that phase 1
    # would hide.
    follow_up = {"role": "tool", "content": "[test] synthetic follow-up env"}
    extended_messages = list(session.messages) + [follow_up]
    pre = session.prepare_pretokenized(extended_messages, tools, tito_tokenizer=tito_tok)
    assert pre is not None, (
        f"K2V3 [{name}] phase-2 setup error: prepare_pretokenized returned "
        f"None even though session has {len(session.messages)} stored messages"
    )
    merged = list(pre["input_ids"])
    expected_next = _render_ids(
        extended_messages,
        tito_tok.tokenizer,
        tools,
        add_generation_prompt=True,
    )
    severe_next = [
        m for m in comparator.compare_sequences(expected_next, merged) if m.type != MismatchType.ASSISTANT_TEXT
    ]
    if severe_next:
        details = "\n".join(
            f"  {m.type.value} at segment {m.segment_index}: "
            f"expected={m.expected_text!r} actual={m.actual_text!r}" + (f" — {m.detail}" if m.detail else "")
            for m in severe_next[:5]
        )
        pytest.fail(
            f"K2V3 [{name}] phase-2 (next-turn merged input_ids) canonical "
            f"mismatch — the per-model boundary fix is likely broken.\n"
            f"  first_diff: {_first_diff(expected_next, merged)}\n{details}"
        )


# ---------------------------------------------------------------------------
# (Section A cont.) Append-case test — mirrors the breadth of
# ``test_tito_tokenizer_model_matrix.py`` but routes through
# ``update_pretokenized_state`` so the buffer used for ``merge_tokens`` has
# the realistic ``<|im_end|>``-end shape (defeats the comparator's
# ``trim_trailing_ids`` shielding that hides missing-fix bugs in the
# model_matrix variant).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EnvAppendShape:
    """Generic env append shape — the messages to be appended after the
    session has been driven through some trajectory."""

    name: str
    appended_messages: list[dict]
    required_contents: tuple[str, ...]


# Generic append shapes. Each gets cross-producted with every trajectory in
# CONVERSATIONS, so we exercise merge_tokens against many distinct buffer
# end-states (single tool, parallel tools, multi-turn with thinking, etc.)
# combined with each env shape (single tool / single user / single system /
# alternating). Strings inside ``required_contents`` are unique markers so
# the in-order check pinpoints exactly which env content the incremental
# tokens dropped if the test fails.
_ENV_APPEND_SHAPES: list[_EnvAppendShape] = [
    _EnvAppendShape(
        name="env_tool",
        appended_messages=[
            {"role": "tool", "tool_call_id": "call_test_xyz", "content": "_marker_tool_xyz_42_"},
        ],
        required_contents=("_marker_tool_xyz_42_",),
    ),
    _EnvAppendShape(
        name="env_user",
        appended_messages=[
            {"role": "user", "content": "_marker_user_abc_99_"},
        ],
        required_contents=("_marker_user_abc_99_",),
    ),
    _EnvAppendShape(
        name="env_system",
        appended_messages=[
            {"role": "system", "content": "_marker_system_def_77_"},
        ],
        required_contents=("_marker_system_def_77_",),
    ),
    _EnvAppendShape(
        name="env_alternating_user_tool",
        appended_messages=[
            {"role": "tool", "tool_call_id": "call_alt_1", "content": "_marker_alt_tool1_aaa_"},
            {"role": "user", "content": "_marker_alt_user1_bbb_"},
            {"role": "tool", "tool_call_id": "call_alt_2", "content": "_marker_alt_tool2_ccc_"},
            {"role": "user", "content": "_marker_alt_user2_ddd_"},
        ],
        required_contents=(
            "_marker_alt_tool1_aaa_",
            "_marker_alt_user1_bbb_",
            "_marker_alt_tool2_ccc_",
            "_marker_alt_user2_ddd_",
        ),
    ),
]


@pytest.mark.parametrize(
    "traj_name, traj_cls",
    CONVERSATIONS,
    ids=lambda x: x if isinstance(x, str) else None,
)
@pytest.mark.parametrize(
    "env_shape",
    _ENV_APPEND_SHAPES,
    ids=lambda s: s.name,
)
def test_append_via_realistic_buffer(traj_name, traj_cls, env_shape, tito_tok):
    """Invariants I3+I4 (core): ``merge_tokens`` against a realistic
    ``<|im_end|>``-terminated buffer matches canonical render, for the
    cross-product of trajectory shape × env append shape.

    8 trajectories × 4 env shapes = 32 ``merge_tokens`` contexts —
    coverage spans buffer end-states (single-tool / parallel-tools /
    thinking) × env shapes (tool / user / system / mixed).

    Checks:
      1. merged input_ids match canonical (modulo ``ASSISTANT_TEXT``).
      2. Each ``required_content`` marker appears IN ORDER in the
         incremental segment (catches dropped/reordered env messages).
    """
    messages = deepcopy(traj_cls.MESSAGES)
    tools = deepcopy(getattr(traj_cls, "TOOLS", None))

    session = LinearTrajectory()
    _drive_session_through_trajectory(session, tito_tok, messages, tools)

    pretokenized_buffer = list(session.token_ids)
    assert pretokenized_buffer and pretokenized_buffer[-1] == tito_tok._im_end_id, (
        f"K2V3 [{traj_name} + {env_shape.name}] setup error: pretokenized "
        f"buffer should end at <|im_end|> after drive, got last token "
        f"{pretokenized_buffer[-1] if pretokenized_buffer else 'EMPTY'}"
    )

    extended = list(session.messages) + list(env_shape.appended_messages)
    pre = session.prepare_pretokenized(extended, tools, tito_tokenizer=tito_tok)
    assert pre is not None, (
        f"K2V3 [{traj_name} + {env_shape.name}] setup error: "
        f"prepare_pretokenized returned None despite stored token_ids of "
        f"length {len(pretokenized_buffer)}"
    )
    merged = list(pre["input_ids"])

    expected = _render_ids(
        extended,
        tito_tok.tokenizer,
        tools,
        add_generation_prompt=True,
    )

    comparator = tito_tok.create_comparator()
    severe = [m for m in comparator.compare_sequences(expected, merged) if m.type != MismatchType.ASSISTANT_TEXT]
    if severe:
        details = "\n".join(
            f"  {m.type.value} at segment {m.segment_index}: "
            f"expected={m.expected_text!r} actual={m.actual_text!r}" + (f" — {m.detail}" if m.detail else "")
            for m in severe[:5]
        )
        pytest.fail(
            f"K2V3 [{traj_name} + {env_shape.name}] merged-vs-canonical "
            f"mismatch under realistic buffer.\n"
            f"  first_diff: {_first_diff(expected, merged)}\n{details}"
        )

    # required-contents-in-order check on the incremental segment.
    incremental_text = tito_tok.tokenizer.decode(merged[len(pretokenized_buffer) :], skip_special_tokens=False)
    cursor = 0
    for content in env_shape.required_contents:
        found = incremental_text.find(content, cursor)
        assert found >= 0, (
            f"K2V3 [{traj_name} + {env_shape.name}] required_content "
            f"{content!r} missing from incremental tokens (or out of order). "
            f"incremental_text={incremental_text!r}"
        )
        cursor = found + len(content)


# ---------------------------------------------------------------------------
# (Section A cont.) Real-SGLang-parser round-trip.
#
# Production server-side parsing flow:
#   raw model text → ReasoningParser → FunctionCallParser
#                  → structured assistant_message in session.messages
#                  → next turn's chat_template re-renders it back to text
#
# If parser output drifts from what chat_template re-emits (whitespace
# stripping, reasoning-block boundaries, tool_call argument formatting),
# the structured message in history fails to round-trip — either causing
# a buffer-vs-canonical mismatch on subsequent turns, or causing
# chat_template to raise (e.g. K2V3's "tool_call.arguments must be dict").
# ---------------------------------------------------------------------------


# (Parser config is declared at the top of the file alongside K2V3_MODEL_PATH.)

_TEST_TOOL_DICT = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two integers and return the product.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
}


def _load_sglang_parsers():
    """Return (FunctionCallParser_cls, ReasoningParser_cls) — either may be
    None if SGLang is missing the corresponding module. Caller decides
    whether to skip."""
    fcp_cls = None
    try:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        fcp_cls = FunctionCallParser
    except ImportError:
        pass
    rp_cls = None
    try:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        rp_cls = ReasoningParser
    except ImportError:
        try:
            from sglang.srt.reasoning_parser import ReasoningParser  # older SGLang layout

            rp_cls = ReasoningParser
        except ImportError:
            pass
    return fcp_cls, rp_cls


def _try_json_decode_tool_args(tool_calls: list[dict]) -> list[dict]:
    """K2V3's chat template requires ``tool_call.arguments`` to be a dict.
    Hermes parser returns it as a JSON string. Decode for template
    compatibility — this mirrors what production agent loops do."""
    import json

    out = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn = {**fn, "arguments": json.loads(args)}
            except Exception:
                pass
        out.append({**tc, "function": fn})
    return out


@pytest.mark.parametrize(
    "traj_name, traj_cls",
    CONVERSATIONS,
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_chat_template_round_trip_through_real_sglang_parsers(traj_name, traj_cls, tito_tok):
    """Invariant I4 with parser substitution: raw assistant emit →
    ReasoningParser + FunctionCallParser → ``parsed_msg`` → re-render via
    chat_template still round-trips structurally to canonical.

    Parametrized over every trajectory in ``CONVERSATIONS``, so each
    parser shape (plain / + tool_calls / + reasoning / + parallel
    tool_calls) gets exercised.

    ``ASSISTANT_TEXT`` mismatches are tolerated — the ``deepseek-r1``
    parser does not ``rstrip`` reasoning content, so re-render inserts
    an extra ``\\n`` before ``</think>``. Production classifies this as
    ``ASSISTANT_TEXT`` and the strict CI check excludes it; this test
    matches that contract.

    Skips if SGLang parsers are unavailable in this environment.
    """
    FCP, RP = _load_sglang_parsers()
    if FCP is None:
        pytest.skip("sglang.srt.function_call.function_call_parser not importable")

    tokenizer = tito_tok.tokenizer
    messages = deepcopy(traj_cls.MESSAGES)
    tools = deepcopy(getattr(traj_cls, "TOOLS", None))

    # Pick the first assistant message — that's our parser-test ``truth_msg``.
    # The messages preceding it (system + user typically) are kept as the
    # request prefix so the chat template renders in correct context.
    first_asst_idx = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    request_messages = messages[:first_asst_idx]
    truth_msg = messages[first_asst_idx]
    has_reasoning = bool(truth_msg.get("reasoning_content"))

    # 1) Render truth_msg via chat_template — that is the raw emit shape.
    full_text = _render_text(
        request_messages + [truth_msg],
        tokenizer,
        tools,
        add_generation_prompt=False,
    )
    prompt_text = _render_text(
        request_messages,
        tokenizer,
        tools,
        add_generation_prompt=True,
    )
    assert full_text.startswith(prompt_text), (
        f"K2V3 [{traj_name}] chat template not append-only: prompt-only " f"render is not a prefix of full render."
    )
    raw_assistant_emit = full_text[len(prompt_text) :].rstrip("\n")
    assert raw_assistant_emit.endswith("<|im_end|>"), (
        f"K2V3 [{traj_name}] unexpected raw_assistant_emit shape: " f"{raw_assistant_emit!r}"
    )

    # 2) Run real ReasoningParser on the raw emit (only if the trajectory's
    #    truth_msg actually has reasoning_content — otherwise there's no
    #    <think>...</think> to extract).
    text_after_reasoning = raw_assistant_emit
    parsed_reasoning = ""
    if _K2V3_REASONING_PARSER and has_reasoning:
        if RP is None:
            pytest.skip("sglang reasoning parser not importable")
        try:
            rp = RP(model_type=_K2V3_REASONING_PARSER)
        except Exception as e:
            pytest.skip(f"reasoning parser {_K2V3_REASONING_PARSER!r} unsupported " f"by this SGLang build: {e}")
        r_out, n_out = rp.parse_non_stream(raw_assistant_emit)
        parsed_reasoning = r_out or ""
        text_after_reasoning = n_out if n_out is not None else ""

    # 3) Run real FunctionCallParser on the post-reasoning text.
    try:
        from sglang.srt.entrypoints.openai.protocol import Tool as SGLangTool
    except ImportError as e:
        pytest.skip(f"sglang.srt.entrypoints.openai.protocol.Tool not importable: {e}")
    sglang_tools = [SGLangTool(**t) for t in (tools or [])]
    try:
        fcp = FCP(tools=sglang_tools, tool_call_parser=_K2V3_TOOL_PARSER)
    except Exception as e:
        pytest.skip(f"tool parser {_K2V3_TOOL_PARSER!r} unsupported by this SGLang " f"build: {e}")
    normal_text, tool_call_items = fcp.parse_non_stream(text_after_reasoning)
    parsed_content = normal_text if normal_text is not None else ""
    parsed_tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": item.name, "arguments": item.parameters},
        }
        for i, item in enumerate(tool_call_items)
    ]
    # Hermes returns arguments as a JSON string; K2V3 chat template requires
    # a dict. Decoding here mirrors what a production agent loop does
    # before storing the assistant message.
    parsed_tool_calls = _try_json_decode_tool_args(parsed_tool_calls)

    parsed_msg: dict = {
        "role": "assistant",
        "content": parsed_content,
        "tool_calls": parsed_tool_calls,
    }
    if has_reasoning:
        parsed_msg["reasoning_content"] = parsed_reasoning

    # 4) Drive session with parser-derived assistant_message.
    # ``raw_assistant_emit`` already ends with ``<|im_end|>`` (the model's
    # autoregressive stop), so the tokenized form is the complete emit.
    # Do NOT append ``tokenizer.eos_token_id`` — for K2V3 that is
    # ``<|endoftext|>``, which the model never emits at turn boundary
    # and would create a spurious extra special-token mismatch.
    emit_ids = list(tokenizer.encode(raw_assistant_emit, add_special_tokens=False))
    prompt_ids = _render_ids(
        request_messages,
        tokenizer,
        tools,
        add_generation_prompt=True,
    )
    session = LinearTrajectory()
    session.update_pretokenized_state(
        request_messages=list(request_messages),
        assistant_message=parsed_msg,
        prompt_token_ids=prompt_ids,
        completion_token_ids=emit_ids,
        max_trim_tokens=tito_tok.max_trim_tokens,
    )

    # 5) Compare ``session.token_ids`` (rollout buffer with raw emit tokens)
    #    against ``apply_chat_template(session.messages)`` canonical (which
    #    re-renders parsed_msg back to text). Severe types only.
    expected = _render_ids(
        session.messages,
        tokenizer,
        tools,
        add_generation_prompt=False,
    )
    actual = list(session.token_ids)
    comparator = tito_tok.create_comparator()
    mismatches = comparator.compare_sequences(expected, actual)
    severe = [m for m in mismatches if m.type != MismatchType.ASSISTANT_TEXT]
    if severe:
        details = "\n".join(
            f"  {m.type.value} at segment {m.segment_index}: "
            f"expected={m.expected_text!r} actual={m.actual_text!r}" + (f" — {m.detail}" if m.detail else "")
            for m in severe[:8]
        )
        pytest.fail(
            f"K2V3 [{traj_name}] chat-template ↔ SGLang parser structural "
            f"round-trip mismatch (tool_parser={_K2V3_TOOL_PARSER!r}, "
            f"reasoning_parser={_K2V3_REASONING_PARSER!r}). "
            f"Severe types only — ASSISTANT_TEXT-only mismatches are "
            f"tolerated (whitespace inside assistant content; production "
            f"already classifies these as non-severe).\n"
            f"{details}\n"
            f"({len(severe)} severe mismatch(es) total; "
            f"showing first {min(8, len(severe))}.)"
        )


# ###########################################################################
# ###########################################################################
# ##                                                                       ##
# ##  SECTION B — INTEGRATION STRESS                                       ##
# ##                                                                       ##
# ##  Chains real parsers across every assistant turn so parser-derived    ##
# ##  ``parsed_msg`` accumulates in ``session.messages``, then runs        ##
# ##  ``prepare_pretokenized → merge_tokens`` against that parser-tainted  ##
# ##  history with a complex env follow-up.                                ##
# ##                                                                       ##
# ##  Section A covers each invariant in isolation. A failure here that    ##
# ##  does NOT reproduce in section A indicates a parser-interaction       ##
# ##  regression specific to accumulated multi-turn state.                 ##
# ##                                                                       ##
# ###########################################################################
# ###########################################################################


@dataclass(frozen=True)
class _BossFlow:
    name: str
    trajectory_cls: type
    final_env: list[dict]


# Build the synthesized thinking variant of the parallel-tools trajectory
# at module load (so it's a stable type referenced in _BOSS_FLOWS).
_MultiToolSingleTurnThinking = _with_synthetic_thinking(MultiToolSingleTurnTrajectory)


_BOSS_FLOWS: list[_BossFlow] = [
    _BossFlow(
        name="multi_turn_thinking + tool_followup",
        trajectory_cls=MultiTurnThinkingTrajectory,
        final_env=[
            {"role": "tool", "tool_call_id": "boss_call_1", "content": "_boss_tool_followup_xyz_42_"},
        ],
    ),
    _BossFlow(
        name="multi_tool_multi_turn_thinking + alternating_user_tool_followup",
        trajectory_cls=LongChainThinkingTrajectory,
        final_env=[
            {"role": "tool", "tool_call_id": "boss_call_2a", "content": "_boss_alt_tool1_aaa_"},
            {"role": "user", "content": "_boss_alt_user1_bbb_"},
            {"role": "tool", "tool_call_id": "boss_call_2b", "content": "_boss_alt_tool2_ccc_"},
            {"role": "user", "content": "_boss_alt_user2_ddd_"},
        ],
    ),
    _BossFlow(
        name="multi_tool_single_turn_thinking + system_inject",
        trajectory_cls=_MultiToolSingleTurnThinking,
        final_env=[
            {"role": "system", "content": "_boss_system_inject_def_77_"},
        ],
    ),
    _BossFlow(
        name="multi_tool_multi_turn_thinking + complex_env_chain",
        trajectory_cls=LongChainThinkingTrajectory,
        final_env=[
            {"role": "tool", "tool_call_id": "boss_call_4a", "content": "_boss_chain_tool1_AAA_"},
            {"role": "user", "content": "_boss_chain_user1_BBB_"},
            {"role": "tool", "tool_call_id": "boss_call_4b", "content": "_boss_chain_tool2_CCC_"},
            {"role": "system", "content": "_boss_chain_system_DDD_"},
            {"role": "tool", "tool_call_id": "boss_call_4c", "content": "_boss_chain_tool3_EEE_"},
        ],
    ),
]


def _run_parsers_on_emit(
    raw_emit: str,
    tools: list[dict] | None,
    *,
    fcp_cls,
    rp_cls,
    has_reasoning: bool,
) -> tuple[str, list[dict], str]:
    """Invoke real SGLang parsers on a raw assistant emit. Returns
    (parsed_content, parsed_tool_calls, parsed_reasoning)."""
    text_after_reasoning = raw_emit
    parsed_reasoning = ""
    if has_reasoning and _K2V3_REASONING_PARSER:
        if rp_cls is None:
            pytest.skip("sglang reasoning parser not importable")
        try:
            rp = rp_cls(model_type=_K2V3_REASONING_PARSER)
        except Exception as e:
            pytest.skip(f"reasoning parser {_K2V3_REASONING_PARSER!r} unsupported " f"by this SGLang build: {e}")
        r_out, n_out = rp.parse_non_stream(raw_emit)
        parsed_reasoning = r_out or ""
        text_after_reasoning = n_out if n_out is not None else ""

    try:
        from sglang.srt.entrypoints.openai.protocol import Tool as SGLangTool
    except ImportError as e:
        pytest.skip(f"sglang.srt.entrypoints.openai.protocol.Tool not importable: {e}")
    sglang_tools = [SGLangTool(**t) for t in (tools or [])]
    try:
        fcp = fcp_cls(tools=sglang_tools, tool_call_parser=_K2V3_TOOL_PARSER)
    except Exception as e:
        pytest.skip(f"tool parser {_K2V3_TOOL_PARSER!r} unsupported by this SGLang " f"build: {e}")
    normal_text, tool_call_items = fcp.parse_non_stream(text_after_reasoning)
    parsed_content = normal_text if normal_text is not None else ""
    parsed_tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": item.name, "arguments": item.parameters},
        }
        for i, item in enumerate(tool_call_items)
    ]
    parsed_tool_calls = _try_json_decode_tool_args(parsed_tool_calls)
    return parsed_content, parsed_tool_calls, parsed_reasoning


def _drive_one_assistant_turn_through_real_parsers(
    session: LinearTrajectory,
    tito_tok,
    *,
    fcp_cls,
    rp_cls,
    request_messages: list[dict],
    truth_assistant_msg: dict,
    tools: list[dict] | None,
) -> dict:
    """Render ``truth_assistant_msg`` to raw_emit, parse it with real
    SGLang parsers, build ``parsed_msg`` from parser output, drive the
    session with ``parsed_msg`` (NOT ``truth_assistant_msg`` — production
    stores parser output in messages history). Returns ``parsed_msg``.
    """
    tokenizer = tito_tok.tokenizer

    full_text = _render_text(
        request_messages + [truth_assistant_msg],
        tokenizer,
        tools,
        add_generation_prompt=False,
    )
    prompt_text = _render_text(
        request_messages,
        tokenizer,
        tools,
        add_generation_prompt=True,
    )
    assert full_text.startswith(prompt_text), (
        "chat template not append-only between " "render(request_messages) and render(request_messages + [truth_msg])"
    )
    raw_emit = full_text[len(prompt_text) :].rstrip("\n")
    assert raw_emit.endswith("<|im_end|>"), f"unexpected raw_emit shape: {raw_emit!r}"

    has_reasoning = bool(truth_assistant_msg.get("reasoning_content"))
    parsed_content, parsed_tool_calls, parsed_reasoning = _run_parsers_on_emit(
        raw_emit,
        tools,
        fcp_cls=fcp_cls,
        rp_cls=rp_cls,
        has_reasoning=has_reasoning,
    )

    parsed_msg: dict = {
        "role": "assistant",
        "content": parsed_content,
        "tool_calls": parsed_tool_calls,
    }
    if has_reasoning:
        parsed_msg["reasoning_content"] = parsed_reasoning

    pre = session.prepare_pretokenized(request_messages, tools, tito_tokenizer=tito_tok)
    if pre is None:
        prompt_ids = _render_ids(
            request_messages,
            tokenizer,
            tools,
            add_generation_prompt=True,
        )
    else:
        prompt_ids = list(pre["input_ids"])

    emit_ids = list(tokenizer.encode(raw_emit, add_special_tokens=False))

    session.update_pretokenized_state(
        request_messages=list(request_messages),
        assistant_message=parsed_msg,
        prompt_token_ids=prompt_ids,
        completion_token_ids=emit_ids,
        max_trim_tokens=tito_tok.max_trim_tokens,
    )
    return parsed_msg


@pytest.mark.parametrize("flow", _BOSS_FLOWS, ids=lambda f: f.name)
def test_end_to_end_realistic_rollout_with_real_parsers(flow: _BossFlow, tito_tok):
    """Invariants I3+I4 under integration stress: drive every assistant
    turn of a multi-turn trajectory through real parsers so
    ``session.messages`` accumulates parser-derived ``parsed_msg`` across
    turns, then append a complex env chain and verify
    ``merge_tokens`` over the parser-tainted history still matches
    canonical.

    Failure here that doesn't reproduce in the simpler per-shape tests
    above indicates a parser-interaction regression specific to
    accumulated session state.

    Skips if SGLang parsers are unavailable.
    """
    FCP, RP = _load_sglang_parsers()
    if FCP is None:
        pytest.skip("sglang.srt.function_call.function_call_parser not importable")

    messages = deepcopy(flow.trajectory_cls.MESSAGES)
    tools = deepcopy(getattr(flow.trajectory_cls, "TOOLS", None))
    asst_indices = _assistant_indices(messages)
    assert asst_indices, f"boss flow {flow.name} has no assistant turns"

    session = LinearTrajectory()

    # Track running messages — these become the request_messages prefix
    # for each subsequent turn, with each prior turn's truth_assistant
    # replaced by its parser-derived parsed_msg.
    running_messages: list[dict] = []

    for k, asst_idx in enumerate(asst_indices):
        if k == 0:
            # Pre-first-assistant: typically [system, user]
            request_messages = list(messages[:asst_idx])
        else:
            # Add env messages from the trajectory between previous
            # assistant and this one (tool results, user follow-ups, etc.)
            prev_asst_idx = asst_indices[k - 1]
            env_between = list(messages[prev_asst_idx + 1 : asst_idx])
            request_messages = list(running_messages) + env_between

        truth_msg = messages[asst_idx]
        parsed_msg = _drive_one_assistant_turn_through_real_parsers(
            session,
            tito_tok,
            fcp_cls=FCP,
            rp_cls=RP,
            request_messages=request_messages,
            truth_assistant_msg=truth_msg,
            tools=tools,
        )
        running_messages = list(request_messages) + [parsed_msg]

    # Final env follow-up — triggers prepare_pretokenized → merge_tokens
    # over a session.messages that has been fully populated by parser-
    # derived parsed_msg's.
    extended = list(session.messages) + list(flow.final_env)
    pre = session.prepare_pretokenized(extended, tools, tito_tokenizer=tito_tok)
    assert pre is not None, (
        f"K2V3 [boss/{flow.name}] setup error: prepare_pretokenized "
        f"returned None even though session has "
        f"{len(session.messages)} stored messages"
    )
    merged = list(pre["input_ids"])

    expected = _render_ids(
        extended,
        tito_tok.tokenizer,
        tools,
        add_generation_prompt=True,
    )

    comparator = tito_tok.create_comparator()
    severe = [m for m in comparator.compare_sequences(expected, merged) if m.type != MismatchType.ASSISTANT_TEXT]
    if severe:
        details = "\n".join(
            f"  {m.type.value} at segment {m.segment_index}: "
            f"expected={m.expected_text!r} actual={m.actual_text!r}" + (f" — {m.detail}" if m.detail else "")
            for m in severe[:8]
        )
        pytest.fail(
            f"K2V3 [boss/{flow.name}] integration mismatch: "
            f"merged input_ids vs canonical render diverge after multi-turn "
            f"parser-driven flow.\n"
            f"  first_diff: {_first_diff(expected, merged)}\n{details}\n"
            f"({len(severe)} severe mismatch(es) total; "
            f"showing first {min(8, len(severe))}.)"
        )

    # Required-content marker check on the incremental segment — ensures
    # the final env chain's content (which includes user/tool/system
    # markers) actually flows into the incremental tokens in order.
    pretokenized_buffer = list(session.token_ids)
    incremental_text = tito_tok.tokenizer.decode(merged[len(pretokenized_buffer) :], skip_special_tokens=False)
    cursor = 0
    for env_msg in flow.final_env:
        marker = env_msg.get("content", "")
        if not marker:
            continue
        found = incremental_text.find(marker, cursor)
        assert found >= 0, (
            f"K2V3 [boss/{flow.name}] env marker {marker!r} missing "
            f"from incremental tokens (or out of order). "
            f"incremental_text={incremental_text!r}"
        )
        cursor = found + len(marker)


# ###########################################################################
# ###########################################################################
# ##                                                                       ##
# ##  SECTION C — SANITY (orthogonal to I1-I4)                             ##
# ##                                                                       ##
# ##  Guards on adjacent runtime defenses and registry wiring — these do   ##
# ##  not test the boundary-fix invariants themselves but catch nearby     ##
# ##  regressions that would silently disable the protection above.        ##
# ##                                                                       ##
# ###########################################################################
# ###########################################################################


def test_production_prefix_check_raises_on_intentional_violation(tito_tok):
    """Validate that production's ``update_pretokenized_state`` prefix check
    fires when fed prompt_token_ids that do not extend the stored prefix.

    If a refactor disables this check, this test fails — protecting the
    runtime defense that catches the same class of bugs in real rollouts.
    """
    session = LinearTrajectory()
    user_q = {"role": "user", "content": "Test."}
    asst1 = {"role": "assistant", "content": "ok"}

    # Seed: drive a single normal turn so the session has stored token_ids.
    prompt_ids = _render_ids(
        [user_q],
        tito_tok.tokenizer,
        tools=None,
        add_generation_prompt=True,
    )
    eos = getattr(tito_tok.tokenizer, "eos_token_id", None)
    completion_ids = list(tito_tok.tokenizer.encode("ok", add_special_tokens=False))
    if eos is not None and (not completion_ids or completion_ids[-1] != int(eos)):
        completion_ids.append(int(eos))
    session.update_pretokenized_state(
        request_messages=[user_q],
        assistant_message=asst1,
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        max_trim_tokens=tito_tok.max_trim_tokens,
    )

    # Now feed bogus prompt_ids — completely different from what's stored.
    bogus_prompt = [99999] * (len(session.token_ids) + 5)
    bogus_completion = [12345]
    asst2 = {"role": "assistant", "content": "next"}
    tool_msg = {"role": "tool", "content": "irrelevant"}

    with pytest.raises(TokenizationError, match=r"pretokenized prefix mismatch"):
        session.update_pretokenized_state(
            request_messages=[user_q, asst1, tool_msg],
            assistant_message=asst2,
            prompt_token_ids=bogus_prompt,
            completion_token_ids=bogus_completion,
            max_trim_tokens=0,
        )


def test_k2v3_subclass_is_wired(tito_tok):
    """Sanity: ``get_tito_tokenizer(..., TITOTokenizerType.K2V3)`` returns
    the K2V3 subclass — not silently falling back to the base
    ``TITOTokenizer``. Catches a future regression where the registry entry
    is removed or pointed elsewhere."""
    from miles.utils.chat_template_utils.tito_tokenizer import K2V3TITOTokenizer

    assert isinstance(tito_tok, K2V3TITOTokenizer), (
        f"expected K2V3TITOTokenizer, got {type(tito_tok).__name__}. "
        f"_TOKENIZER_REGISTRY[TITOTokenizerType.K2V3] may be misregistered."
    )
