"""Logic layer of the session server: ``SessionCore``.

HTTP-agnostic: the FastAPI adapter (``sessions.py`` + ``server.py``) turns each request into primitives and calls these methods. Owns one ``SessionRegistry`` (per-session TITO/trajectory state) and one proxy ``backend``.

- ``chat_completions`` omits ``meta_info`` and ``prompt_token_ids`` from v1 replies. The shared v2 renderer strips the R3 replay payloads (``routed_experts`` / ``indexer_topk``) from the client reply copy-on-write; the ``SessionRecord`` keeps the full response for the training path (``GET /sessions/{id}``).
- ``chat_completions`` holds the per-session lock for prep and state update but not across the proxy call; ``closing`` re-checks and the ``num_assistant`` check gate concurrent DELETE/chat.
- ``stream: true`` is served as fake streaming: the backend call stays non-streaming (TITO needs the complete message + meta_info) and the full response is re-rendered as a single SSE chunk plus ``data: [DONE]``. Errors all happen before the SSE body is built, so they keep their real status codes as JSON.
- ``collect_samples`` assembles training Samples from the session's records on the server (compute -> truncate -> merge, in a worker while holding the session lock); deterministic assembly failures return 422 with the assertion text.
"""

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass

import orjson
from starlette.responses import Response

from miles.rollout.generate_utils.sample_utils import drop_samples_after_first_non_completed, merge_samples
from miles.rollout.session.concurrency import run_session_worker
from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.errors import (
    MessageValidationError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.observability import (
    WorkerStats,
    measure_phase,
    measured_session_lock,
    observe_chat,
    publish_response,
    record_request_shape,
    warn_state_change,
)
from miles.rollout.session.samples.codec import encode_samples
from miles.rollout.session.samples.merge import (
    compute_samples_from_openai_records,
    merge_samples_with_addition_r3,
    truncate_samples_by_total_tokens,
)
from miles.rollout.session.types import GetSessionResponse, SessionRecord
from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_enabled

logger = logging.getLogger(__name__)

JSON_MEDIA_TYPE = "application/json"

# Hop-by-hop / length-framing headers dropped from the upstream response so the
# transport layer recomputes them from the body we actually send. "server" and
# "date" are dropped because our own ASGI server always emits them, so echoing
# upstream's copy puts two of each on the wire; aiohttp's parser rejects that
# outright with "Duplicate 'Server' header found" instead of reading the body.
_DROP_RESPONSE_HEADERS = ("content-length", "transfer-encoding", "content-encoding", "server", "date")


@dataclass
class ProxyRequest:
    """Primitive carrier for the proxy backend (replaces fastapi.Request)."""

    method: str
    query: str = ""


def _render_json(payload) -> bytes:
    """Encode like Starlette's JSONResponse (compact, non-ASCII preserved)."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _lcp_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token-ID lists."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _samples_response(payload: bytes) -> Response:
    """The samples-op reply: one safetensors binary payload."""
    return Response(content=payload, status_code=200, media_type="application/octet-stream")


_CLIENT_STRIPPED_META_KEYS = ("routed_experts", "indexer_topk")


def _strip_replay_payloads(response: dict, *, compact: bool = False) -> dict:
    stripped_choices = []
    for choice in response.get("choices", []):
        if compact:
            stripped_choices.append({k: v for k, v in choice.items() if k not in ("meta_info", "prompt_token_ids")})
            continue
        meta = choice.get("meta_info")
        if isinstance(meta, dict) and any(k in meta for k in _CLIENT_STRIPPED_META_KEYS):
            meta = {k: v for k, v in meta.items() if k not in _CLIENT_STRIPPED_META_KEYS}
            choice = {**choice, "meta_info": meta}
        stripped_choices.append(choice)
    return {**response, "choices": stripped_choices}


def _response_to_stream_chunk(response: dict) -> dict:
    """Synthesize the single ``chat.completion.chunk`` for a fake stream.

    Adapted from NVIDIA-NeMo/ProRL-Agent-Server (``gateway/server.py::_response_to_stream_chunk``)
    and THUDM/slime (``agent/adapters/openai.py::_render_stream``).

    One big delta is protocol-legal (streaming deltas concatenate). All
    tool_calls ride in this one chunk with their ``index`` set: some clients
    mis-assemble arguments fragmented across chunks. The chunk carries no
    ``meta_info``; the training path reads ``GET /sessions/{id}`` instead.
    """
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    delta = {"role": message.get("role", "assistant"), "content": message.get("content")}
    if message.get("reasoning_content") is not None:
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [{**tool_call, "index": i} for i, tool_call in enumerate(message["tool_calls"])]
    chunk = {
        "id": response.get("id"),
        "object": "chat.completion.chunk",
        "created": response.get("created"),
        "model": response.get("model"),
        "choices": [{"index": 0, "delta": delta, "finish_reason": choice.get("finish_reason")}],
    }
    if response.get("usage") is not None:
        chunk["usage"] = response["usage"]
    return chunk


def _chat_client_response(result: dict, response: dict, client_stream: bool, *, compact: bool = False) -> Response:
    if client_stream:
        sse = b"data: " + _render_json(_response_to_stream_chunk(response)) + b"\n\ndata: [DONE]\n\n"
        # Fresh headers: upstream's headers describe its JSON body, not this SSE body.
        # X-Accel-Buffering keeps reverse proxies from buffering the stream.
        return Response(
            content=sse,
            status_code=result["status_code"],
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            media_type="text/event-stream",
        )
    headers = {k: v for k, v in result["headers"].items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    return Response(
        content=_render_json(_strip_replay_payloads(response, compact=compact)),
        status_code=result["status_code"],
        headers=headers,
        media_type=JSON_MEDIA_TYPE,
    )


def proxy_result_to_response(result: dict) -> Response:
    """Forward untouched proxy bytes and rebuild transport framing headers."""
    headers = {k: v for k, v in result["headers"].items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    return Response(
        content=result["response_body"],
        status_code=result["status_code"],
        headers=headers,
        media_type=headers.get("content-type"),
    )


def prepare_chat_request(body: bytes, args, tito_tokenizer) -> tuple:
    """Parse and normalize a chat request body — the session-independent half
    of chat dispatch, shared verbatim by the v1 and v2 cores. Returns
    ``(request_body, client_stream, tito_tokenizer)``; the tokenizer may be a
    request-scoped clone.
    """
    try:
        request_body = orjson.loads(body) if body else {}
    except orjson.JSONDecodeError as e:
        raise MessageValidationError(f"invalid JSON body: {e}") from e

    # Fake streaming: the backend must stay non-streaming (TITO needs the
    # complete message + meta_info, and sglang rejects return_meta_info
    # with stream=true), so pop the client's intent here and honor it
    # when rendering the client response.
    client_stream = bool(request_body.pop("stream", False))
    request_body.pop("stream_options", None)

    # TITO token tracking needs Miles-owned input_ids plus SGLang output
    # metadata: logprobs=True populates meta_info.output_token_logprobs and
    # return_meta_info wraps it in choice.meta_info. Hardcoded (not
    # setdefault) so agent-side overrides cannot break token accumulation.
    request_body["logprobs"] = True
    request_body["return_meta_info"] = True
    if getattr(args, "use_rollout_routing_replay", False):
        request_body["return_routed_experts"] = True
    else:
        request_body.pop("return_routed_experts", None)
        request_body.pop("routed_experts_start_len", None)
    if getattr(args, "use_rollout_indexer_replay", False):
        request_body["return_indexer_topk"] = True
    # Must be False so stop-token text is trimmed from assistant content;
    # token IDs still come from logprobs below.
    request_body["no_stop_trim"] = False
    # Serve the adapter being trained instead of the base weights.
    if is_lora_enabled(args):
        request_body["lora_path"] = LORA_ADAPTER_NAME
    # FIXME(session): Only nested `chat_template_kwargs` reach the local renderer;
    # top-level `reasoning` and `reasoning_effort` are not mapped to template kwargs.
    request_ctk = request_body.get("chat_template_kwargs")
    if request_ctk is not None and not isinstance(request_ctk, dict):
        raise MessageValidationError("chat_template_kwargs must be an object")
    if request_ctk:
        try:
            tito_tokenizer = tito_tokenizer.clone_with_chat_template_kwargs(request_ctk)
        except ValueError as e:
            raise MessageValidationError(str(e)) from e
    if tito_tokenizer.chat_template_kwargs:
        request_body["chat_template_kwargs"] = dict(tito_tokenizer.chat_template_kwargs)
    else:
        request_body.pop("chat_template_kwargs", None)
    return request_body, client_stream, tito_tokenizer


def extract_completion(result: dict) -> tuple:
    """Decode and validate the backend chat response — shared verbatim by the
    v1 and v2 cores. Returns ``(response, choice, assistant_message,
    completion_token_ids)``; malformed upstream payloads raise
    ``UpstreamResponseError``.
    """
    response = orjson.loads(result["response_body"])
    choice = response.get("choices", [{}])[0]

    meta_info = choice.get("meta_info")
    if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
        raise UpstreamResponseError("meta_info and output_token_logprobs must be in choice (requires logprobs=True)")
    assistant_message = choice.get("message") or {}
    if assistant_message.get("content") is None:
        raise UpstreamResponseError(
            "assistant message content is None, when tool call parser failed SGLang should still return "
            "an empty content rather than None. Please check your modified SGLang version."
        )

    output_token_logprobs = meta_info["output_token_logprobs"]
    completion_tokens = meta_info["completion_tokens"]

    actual_output_logprobs_len = len(output_token_logprobs)
    if actual_output_logprobs_len != completion_tokens:
        raise UpstreamResponseError(
            "invalid chat completion response: "
            f"len(output_token_logprobs)={actual_output_logprobs_len} "
            f"!= completion_tokens={completion_tokens}. "
            f"Please check whether you use the correct SGLang branch which has fix the tokenizer batch decode issue."
        )

    completion_token_ids = [t[1] for t in output_token_logprobs]
    return response, choice, assistant_message, completion_token_ids


def gate_routed_experts(choice: dict, request_body: dict, *, enabled: bool, use_addition_r3: bool) -> None:
    """Keep routing payloads only when Miles requests them; normalize legacy location."""
    meta = choice["meta_info"]
    if not enabled:
        meta.pop("routed_experts", None)
        choice.pop("routed_experts", None)
        return
    info = meta.get("routed_experts")
    if info is None:
        info = choice.get("routed_experts")
    expected_rows = max(0, len(request_body["input_ids"]) + len(meta["output_token_logprobs"]) - 1)
    empty_addition = use_addition_r3 and request_body.get("routed_experts_start_len") == expected_rows
    if info is None and not empty_addition:
        raise UpstreamResponseError(
            "routed_experts must be in choice or choice.meta_info when use_rollout_routing_replay is enabled"
        )
    if info is not None:
        meta["routed_experts"] = info
    choice.pop("routed_experts", None)


def closed_chat_response(result: dict, client_stream: bool, *, compact: bool = False) -> Response:
    """Forward a closed session's completion without validating or recording it."""
    if result["status_code"] == 200:
        try:
            response = orjson.loads(result["response_body"])
            return _chat_client_response(result, response, client_stream, compact=compact)
        except (orjson.JSONDecodeError, KeyError, TypeError, AttributeError):
            pass
    return proxy_result_to_response(result)


def _is_prefix_rollback_error(result: dict) -> bool:
    body = result.get("response_body") or b""
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return result["status_code"] == 400 and "rollback failed" in body.lower()


def _commit_linear_response(
    session,
    *,
    request_messages,
    assistant_message,
    prompt_token_ids,
    completion_token_ids,
    max_trim_tokens,
    record_fields,
):
    """Publish the checkpoint and its matching record under one worker lifetime."""
    session.update_pretokenized_state(
        request_messages,
        assistant_message,
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        max_trim_tokens=max_trim_tokens,
    )
    session.append_record(SessionRecord(timestamp=time.time(), **record_fields))


class SessionCore:
    """HTTP session operations over one ``SessionRegistry``."""

    def __init__(
        self,
        backend,
        registry: SessionRegistry,
        config: SessionServerConfig,
        session_server_instance_id=None,
        *,
        use_addition_r3=False,
    ):
        self.backend = backend
        self.registry = registry
        self.config = config
        self.instance_id = session_server_instance_id
        self.request_stats = WorkerStats(port=getattr(config, "port", None))
        # Derived from pause_generation_mode at server bootstrap; session code
        # must depend on this capability, never on the weight-update mode.
        self.use_addition_r3 = use_addition_r3

    def _maybe_request_addition_r3(
        self, request_body: dict, checkpoint_token_ids: list[int], prompt_token_ids: list[int]
    ) -> None:
        """Ask SGLang to return only the R3 rows the session has not retained.

        ``checkpoint_token_ids`` is the stored snapshot this request builds on
        (v1: the post-rollback checkpoint; v2: the positioned attach node). The
        checkpoint's N - 1 rows must remain a causal prefix of the new prompt,
        so every persisted patch starts exactly where the previous one ended.
        """
        if not (self.use_addition_r3 and request_body.get("return_routed_experts")):
            return
        previous_rows = max(0, len(checkpoint_token_ids) - 1)
        stable_prefix_tokens = _lcp_len(checkpoint_token_ids, prompt_token_ids)
        assert (
            stable_prefix_tokens >= previous_rows
        ), f"additional R3 requires {previous_rows} stable prefix tokens, got {stable_prefix_tokens}"
        request_body["routed_experts_start_len"] = previous_rows

    async def health(self) -> Response:
        body = {"status": "ok"}
        if self.instance_id is not None:
            body["session_server_instance_id"] = self.instance_id
        return Response(content=_render_json(body), status_code=200, media_type=JSON_MEDIA_TYPE)

    async def create_session(self) -> Response:
        session_id = self.registry.create_session()
        return Response(content=_render_json({"session_id": session_id}), status_code=200, media_type=JSON_MEDIA_TYPE)

    def _session_metadata(self, session_id: str, session) -> dict:
        """The per-session assembly/inspection metadata dict, shared by
        `get_session` (records debug dump) and `collect_samples` (samples op)
        so the two can never drift."""
        metadata: dict = {}
        try:
            mismatch = self.registry.compute_session_mismatch(session)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = session.token_ids
        metadata["max_trim_tokens"] = self.registry.tito_tokenizer.max_trim_tokens
        return metadata

    async def get_session(self, session_id: str) -> Response:
        session = self.registry.sessions.get(session_id)
        if session is None and self.registry.is_deleted(session_id):
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        async with session.lock if session is not None else nullcontext():
            metadata = self._session_metadata(session_id, session) if session is not None else {}
            payload = GetSessionResponse(
                session_id=session_id, records=session.records if session is not None else [], metadata=metadata
            )
            return Response(
                content=_render_json(payload.model_dump(mode="json")), status_code=200, media_type=JSON_MEDIA_TYPE
            )

    async def collect_samples(
        self, session_id: str, *, max_seq_len: int | None, decode_response: bool = True
    ) -> Response:
        """Assemble training Samples from this session's records.

        Validation failures return 422; unexpected errors propagate.
        """
        session = self.registry.get_session(session_id)
        async with session.lock:
            return await run_session_worker(self._assemble_samples, session_id, session, max_seq_len, decode_response)

    def _assemble_samples(self, session_id, session, max_seq_len, decode_response):
        metadata = self._session_metadata(session_id, session)
        tokenizer = self.registry.tokenizer
        if not session.records:
            return _samples_response(encode_samples([], metadata, empty_reason="no_records"))
        try:
            samples = compute_samples_from_openai_records(
                self.config,
                session.records,
                tokenizer,
                accumulated_token_ids=metadata.get("accumulated_token_ids"),
                max_trim_tokens=metadata.get("max_trim_tokens", 0),
                use_addition_r3=self.use_addition_r3,
                decode_response=decode_response,
            )
            samples, num_dropped = drop_samples_after_first_non_completed(samples)
            if num_dropped:
                logger.warning("Session %s dropped %d turns after an incomplete turn", session_id, num_dropped)
                for sample in samples:
                    sample.metadata["dropped_trailing_turns"] = num_dropped
            if max_seq_len is not None:
                samples = truncate_samples_by_total_tokens(samples, max_seq_len, tokenizer)
            if not samples:
                return _samples_response(encode_samples([], metadata, empty_reason="all_truncated"))
            if self.use_addition_r3:
                samples = [merge_samples_with_addition_r3(self.config, samples, session.records, tokenizer)]
            else:
                samples = [merge_samples(samples, tokenizer)]
        except (AssertionError, ValueError) as exc:
            return Response(content=str(exc).encode(), status_code=422, media_type="text/plain")
        if not decode_response:
            metadata.pop("accumulated_token_ids", None)
            metadata.update(
                records_total=len(session.records),
                records_merged=len(session.records) - num_dropped,
                records_dropped_after_first_non_completed=num_dropped,
                accumulated_token_count=len(samples[0].tokens),
            )
        return _samples_response(encode_samples(samples, metadata))

    async def delete_session(self, session_id: str) -> Response:
        session = self.registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        session.closing = True
        # Acquire the lock so an in-flight chat finishes before we drop the session.
        await session.lock.acquire()
        try:
            self.registry.remove_session(session_id)
        finally:
            session.lock.release()
        return Response(status_code=204)

    @observe_chat
    async def chat_completions(
        self, session_id: str, *, method: str, query: str, headers: dict, body: bytes
    ) -> Response:
        """Proxy a chat completion through the backend with TITO token tracking.

        Flow: prepare pretokenized input_ids (lock held briefly) → proxy to
        backend (NO lock) → validate response → update trajectory checkpoint and
        append record (lock held briefly). The lock is NOT held during the long
        inference call so DELETE/other ops are not blocked if the agent disconnects.
        """
        request_timestamp = time.time()
        session = self.registry.get_or_create_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

        # --- Phase 1: prepare request (lock held briefly) ---
        async with measured_session_lock(session.lock):
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            request_body, client_stream, tito_tokenizer = prepare_chat_request(
                body, self.config, self.registry.tito_tokenizer
            )

            request_messages = request_body.get("messages", [])
            record_request_shape(request_messages)
            with measure_phase("tokenize_in_ms"):
                prompt_token_ids = await run_session_worker(
                    session.prepare_pretokenized,
                    request_messages,
                    tools=request_body.get("tools"),
                    tito_tokenizer=tito_tokenizer,
                    message_matcher=self.registry.message_matcher,
                )
            request_body["input_ids"] = prompt_token_ids
            logger.debug("Using TITO input_ids: %d tokens", len(prompt_token_ids))

            # prepare_pretokenized applied any retry rollback, so token_ids is
            # the checkpoint this request builds on.
            self._maybe_request_addition_r3(request_body, session.token_ids, prompt_token_ids)

            proxy_body = orjson.dumps(request_body)
            expected_num_assistant = session.num_assistant
        # --- lock released ---

        # --- Phase 2: proxy to backend (NO lock held) ---
        headers = {**headers, "X-SMG-Routing-Key": session_id}
        with measure_phase("proxy_elapsed_ms"):
            result = await self.backend.do_proxy(
                ProxyRequest(method=method, query=query), "v1/chat/completions", body=proxy_body, headers=headers
            )

        if session.closing:
            return await run_session_worker(closed_chat_response, result, client_stream, compact=True)

        # Stable's backend can recover a prefix-cache rollback failure by
        # rendering the messages again. Keep the retry narrowly scoped.
        retried_without_prefix = _is_prefix_rollback_error(result)
        if retried_without_prefix:
            logger.warning("Retrying session %s without prefix continuation", session_id)
            retry_body = {k: v for k, v in request_body.items() if k != "input_ids"}
            with measure_phase("proxy_elapsed_ms"):
                result = await self.backend.do_proxy(
                    ProxyRequest(method=method, query=query),
                    "v1/chat/completions",
                    body=orjson.dumps(retry_body),
                    headers=headers,
                )

        if session.closing:
            return await run_session_worker(closed_chat_response, result, client_stream, compact=True)

        # Other errors, including a failed retry, pass through unrecorded.
        if result["status_code"] != 200:
            return proxy_result_to_response(result)

        response, choice, assistant_message, completion_token_ids = await run_session_worker(
            extract_completion, result
        )
        if retried_without_prefix:
            # The backend re-rendered the prompt. Use its actual IDs rather
            # than recording the IDs from the rejected request as training data.
            prompt_token_ids = choice.get("prompt_token_ids")
            if not isinstance(prompt_token_ids, list) or not all(type(t) is int for t in prompt_token_ids):
                raise UpstreamResponseError("prefix retry requires backend prompt_token_ids for token tracking")
            request_body = {**request_body, "input_ids": prompt_token_ids}
        gate_routed_experts(
            choice,
            request_body,
            enabled=self.config.use_rollout_routing_replay,
            use_addition_r3=self.use_addition_r3,
        )
        assistant_message = tito_tokenizer.postprocess_completion(
            choice=choice,
            assistant_message=assistant_message,
            completion_token_ids=completion_token_ids,
        )

        # --- Phase 3: update state (lock held briefly) ---
        async with measured_session_lock(session.lock):
            if session.closing:
                logger.warning(f"Session {session_id} closed during proxy, skipping state update")
                return _chat_client_response(result, response, client_stream)

            if session.num_assistant != expected_num_assistant:
                warn_state_change(
                    self.request_stats, session_id, expected_num_assistant, session.num_assistant, headers
                )
                return _chat_client_response(result, response, client_stream)

            stored_request_messages = tito_tokenizer.preserve_server_message_state(
                session.messages,
                request_messages,
            )
            with measure_phase("tokenize_out_ms"):
                await run_session_worker(
                    publish_response,
                    _commit_linear_response,
                    session,
                    request_messages=stored_request_messages,
                    assistant_message=assistant_message,
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=completion_token_ids,
                    max_trim_tokens=self.registry.tito_tokenizer.max_trim_tokens,
                    record_fields={
                        "request_timestamp": request_timestamp,
                        "method": method,
                        "path": "/v1/chat/completions",
                        "status_code": result["status_code"],
                        "request": request_body,
                        "response": response,
                    },
                )
        # --- lock released ---

        return _chat_client_response(result, response, client_stream, compact=True)

    async def proxy(
        self, session_id: str, path: str, *, method: str, query: str, headers: dict, body: bytes
    ) -> Response:
        if method == "GET" and path == "v1/model_info":
            return Response(
                content=_render_json({"id": self.config.sglang_served_model_name or "model", "object": "model"}),
                media_type=JSON_MEDIA_TYPE,
            )
        headers = {**headers, "X-SMG-Routing-Key": session_id}
        result = await self.backend.do_proxy(
            ProxyRequest(method=method, query=query), path, body=body, headers=headers
        )
        return proxy_result_to_response(result)
