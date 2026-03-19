"""Core chat template operations: load from HuggingFace and render from string.

``load_hf_chat_template`` fetches original (unmodified) chat templates via
``hf_hub_download``.  Files are cached locally after the first download —
subsequent calls read from disk without network access.

``apply_chat_template_from_str`` renders a Jinja2 chat template string
without depending on a HuggingFace tokenizer, equivalent to
``tokenizer.apply_chat_template(..., tokenize=False)``.

``apply_chat_template`` is the unified entry point that normalizes tool
arguments, extracts tool dicts, and applies the template with fallback —
works with both a tokenizer object and a raw Jinja2 template string.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from huggingface_hub import hf_hub_download
from jinja2.sandbox import ImmutableSandboxedEnvironment


def load_hf_chat_template(model_id: str) -> str:
    """Load an original chat template from HuggingFace (cached locally).

    Handles two layouts:
    - ``chat_template`` field in ``tokenizer_config.json`` (most models)
    - Separate ``chat_template.jinja`` file (e.g. GLM-5)
    """
    config_path = hf_hub_download(model_id, "tokenizer_config.json")
    with open(config_path) as f:
        config = json.load(f)
    template = config.get("chat_template", "")
    if template:
        if isinstance(template, list):
            for t in template:
                if t.get("name") == "default" or not t.get("name"):
                    return t["template"]
            return template[0]["template"]
        return template

    jinja_path = hf_hub_download(model_id, "chat_template.jinja")
    with open(jinja_path) as f:
        return f.read()


def _tojson(value, ensure_ascii=True, indent=None):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


def _normalize_tool_arguments(messages: list[dict]) -> list[dict]:
    """Deep-copy messages and normalize for template rendering.

    Normalizations:
    - Parse JSON-string tool_call arguments to dicts.  Matches SGLang's
      ``_apply_jinja_template`` / ``_apply_pretokenized_template``
      normalization (serving_chat.py L446-462, L669-682).  Some templates
      (e.g. Qwen3-Coder-Next, GLM-4.7) use ``arguments|items`` which requires
      a mapping.
    - Convert ``content: None`` to ``content: ""`` for assistant messages with
      tool_calls.  The OpenAI API returns ``content: null`` for tool-call-only
      responses; Jinja2 renders Python ``None`` as the literal string "None".
    """
    normalized = copy.deepcopy(messages)
    for msg in normalized:
        if msg.get("role") == "assistant":
            if msg.get("content") is None and msg.get("tool_calls"):
                msg["content"] = ""
            if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                for item in msg["tool_calls"]:
                    if "arguments" in item["function"] and isinstance(item["function"]["arguments"], str):
                        item["function"]["arguments"] = json.loads(item["function"]["arguments"])
    return normalized


def extract_tool_dicts(tools: list[dict] | None) -> list[dict] | None:
    """Extract and canonicalize function definitions from OpenAI tool format.

    Matches SGLang's tool canonicalization before ``apply_chat_template``:
    SGLang validates tools with ``protocol.Tool`` (Pydantic) and then passes
    ``item.function.model_dump()`` into the template.  This stabilizes field
    order and injects defaults like ``strict=False``.

    Falls back to plain extraction when ``sglang`` is not importable.
    """
    if not tools:
        return None

    try:
        from pydantic import TypeAdapter
        from sglang.srt.entrypoints.openai.protocol import Tool

        wrapped = [
            t if isinstance(t, dict) and "function" in t else {"type": "function", "function": t} for t in tools
        ]
        validated = TypeAdapter(list[Tool]).validate_python(wrapped)
        return [tool.function.model_dump() for tool in validated]
    except Exception:
        return [t["function"] if "function" in t else t for t in tools]


def _render_jinja(
    chat_template: str,
    messages: list[dict],
    add_generation_prompt: bool = True,
    tools: list[dict] | None = None,
    **kwargs,
) -> str:
    """Render a Jinja2 chat template string (no normalization)."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    env.filters["tojson"] = _tojson
    template = env.from_string(chat_template)

    render_kwargs = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        render_kwargs["tools"] = tools
    render_kwargs.update(kwargs)
    return template.render(**render_kwargs)


def apply_chat_template_from_str(
    chat_template: str,
    messages: list[dict],
    add_generation_prompt: bool = True,
    tools: list[dict] | None = None,
    **kwargs,
) -> str:
    """Render a Jinja2 chat template string (tokenize=False equivalent)."""
    messages = _normalize_tool_arguments(messages)
    return _render_jinja(chat_template, messages, add_generation_prompt, tools, **kwargs)


_TEMPLATE_RELEVANT_KEYS = ("role", "content", "reasoning_content", "tool_calls")


def _normalize_value(value: Any) -> Any:
    """Normalize falsy sentinels that produce identical Jinja2 output.

    None, "" and [] are all falsy in Jinja2 and render the same way,
    but client libraries may interchange them (e.g. content: null vs ""
    for tool-call-only responses, or tool_calls: null vs []).

    Only collapses falsy values — non-falsy content (including whitespace
    like trailing newlines) is returned as-is.  Message boundary characters
    must be preserved exactly so they tokenize identically across turns.
    """
    if value is None or value == "" or value == []:
        return None
    return value


def message_matches(stored: dict[str, Any], new: dict[str, Any]) -> bool:
    """Compare only the fields that affect chat-template tokenization.

    External client libraries (e.g. litellm) may inject extra keys like
    ``provider_specific_fields`` into messages.  These have no effect on
    the Jinja2 chat template output, so we only compare the keys that
    templates actually read: role, content, reasoning_content, tool_calls.
    """
    for key in _TEMPLATE_RELEVANT_KEYS:
        if _normalize_value(stored.get(key)) != _normalize_value(new.get(key)):
            return False
    return True


def assert_messages_append_only(
    stored_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
) -> None:
    """Assert *new_messages* is an append-only extension of *stored_messages*.

    The stored prefix must match exactly (compared by template-relevant keys),
    and any appended messages must have role ``'tool'`` or ``'system'``.
    """
    if not stored_messages:
        return

    if len(new_messages) < len(stored_messages):
        raise ValueError(
            f"new messages ({len(new_messages)}) are fewer than stored messages ({len(stored_messages)})",
            new_messages,
            stored_messages,
        )

    for i, stored_msg in enumerate(stored_messages):
        if not message_matches(stored_msg, new_messages[i]):
            diffs = {
                key: {"stored": repr(stored_msg.get(key))[:200], "new": repr(new_messages[i].get(key))[:200]}
                for key in _TEMPLATE_RELEVANT_KEYS
                if stored_msg.get(key) != new_messages[i].get(key)
            }
            raise ValueError(
                f"message mismatch at index {i} "
                f"(role: stored={stored_msg.get('role')}, new={new_messages[i].get('role')}). "
                f"Diffs: {diffs}"
            )

    ALLOWED_APPEND_ROLES = {"tool", "system"}
    for j, msg in enumerate(new_messages[len(stored_messages) :]):
        if msg.get("role") not in ALLOWED_APPEND_ROLES:
            raise ValueError(
                f"appended message at index {len(stored_messages) + j} "
                f"has role={msg.get('role')!r}, allowed={ALLOWED_APPEND_ROLES}"
            )


def _hf_apply(tokenizer, messages, tools, *, tokenize, **kwargs):
    """Call ``tokenizer.apply_chat_template`` and normalize the return type.

    HF tokenizers may return ``dict``, ``BatchEncoding``, or ``list`` depending
    on the tokenizer version and ``return_tensors`` setting.  This helper
    always returns ``str`` (tokenize=False) or ``list[int]`` (tokenize=True).
    """
    result = tokenizer.apply_chat_template(messages, tokenize=tokenize, tools=tools, **kwargs)
    if tokenize and not isinstance(result, list):
        result = result["input_ids"]
        if result and hasattr(result[0], "ids"):
            result = result[0].ids
    return result


def apply_chat_template(
    messages: list[dict],
    *,
    tokenizer=None,
    chat_template: str | None = None,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    **kwargs,
) -> str | list[int]:
    """Apply chat template in SGLang style so results match token-for-token.

    Mirrors SGLang's ``serving_chat.py`` preprocessing:
    1. Normalize messages (JSON-string arguments → dict, ``content: null`` → ``""``).
    2. Canonicalize tool definitions (via SGLang's ``protocol.Tool`` Pydantic model).
    3. Render with function-only dicts; fall back to ``{"function": ...}``
       wrapper on failure (templates vary in which format they expect).

    Supports two rendering paths: ``tokenizer`` (HF) or ``chat_template`` (Jinja2).
    """
    if tokenizer is None and chat_template is None:
        raise ValueError("Either tokenizer or chat_template must be provided")

    messages = _normalize_tool_arguments(messages)
    tool_defs = extract_tool_dicts(tools)
    render_kwargs = dict(add_generation_prompt=add_generation_prompt, **kwargs)

    def _render(td):
        # If tokenizer is provided, use HF's apply_chat_template.
        # Otherwise, render the chat template using Jinja2.
        if tokenizer is not None:
            return _hf_apply(tokenizer, messages, td, tokenize=tokenize, **render_kwargs)
        return _render_jinja(chat_template, messages, add_generation_prompt, td, **kwargs)

    # Try function-only tool format first, fall back to wrapped format.
    try:
        return _render(tool_defs)
    except Exception:
        if tool_defs is not None:
            return _render([{"function": t} for t in tool_defs])
        raise
