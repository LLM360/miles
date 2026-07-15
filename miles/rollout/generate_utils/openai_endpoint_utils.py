"""
Utilities for the OpenAI endpoint
"""

import asyncio
import logging
import random
from argparse import Namespace

import httpx
import orjson

from miles.rollout.session.config import normalize_session_server_urls
from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    SamplesReply,
    decode_samples_and_merge_input_sample,
)
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SESSION_REQUEST_TIMEOUT = 120.0

_HTTP_CONNECT_TIMEOUT = 10.0
_HTTP_READ_TIMEOUT = 120.0
_HTTP_WRITE_TIMEOUT = 30.0
_HTTP_POOL_TIMEOUT = 10.0

_HEALTH_RETRIES = 2
_CREATE_RETRIES = 10
_COLLECT_RETRIES = 3
_DELETE_RETRIES = 3

_BACKOFF_INITIAL_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 10.0
_BACKOFF_JITTER_FRACTION = 0.2

_COLLECT_RECORDS_CONCURRENCY = 256
_COLLECT_RECORDS_SEMAPHORE = asyncio.Semaphore(_COLLECT_RECORDS_CONCURRENCY)


class OpenAIEndpointTracer:
    def __init__(
        self,
        router_url: str,
        session_id: str,
        session_server_instance_id: str | None = None,
        samples_wire_fields: tuple[str, ...] = COMPUTED_FIELDS,
    ):
        self.router_url = router_url.rstrip("/")
        self.session_id = session_id
        self.base_url = f"{self.router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id
        # The samples-wire allowlist must match the server's encode: v1 default,
        # extended under --use-session-server v2 (create() selects from args;
        # direct constructions keep v1).
        self.samples_wire_fields = samples_wire_fields

    @staticmethod
    def _timeout() -> httpx.Timeout:
        return httpx.Timeout(
            timeout=_SESSION_REQUEST_TIMEOUT,
            connect=_HTTP_CONNECT_TIMEOUT,
            read=_HTTP_READ_TIMEOUT,
            write=_HTTP_WRITE_TIMEOUT,
            pool=_HTTP_POOL_TIMEOUT,
        )

    @staticmethod
    def _response_size(response: httpx.Response) -> int:
        try:
            return len(response.content)
        except Exception:
            return -1

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        # attempt is 1-indexed.
        base_delay = _BACKOFF_INITIAL_SECONDS * (2 ** (attempt - 1))
        base_delay = min(base_delay, _BACKOFF_MAX_SECONDS)

        jitter = random.uniform(
            1.0 - _BACKOFF_JITTER_FRACTION,
            1.0 + _BACKOFF_JITTER_FRACTION,
        )
        return min(base_delay * jitter, _BACKOFF_MAX_SECONDS)

    @classmethod
    async def _request(
        cls,
        method: str,
        url: str,
        *,
        phase: str,
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
        expect_json: bool = True,
        return_bytes: bool = False,
    ):
        """Retry transient transport/status failures with stable phase budgets."""
        method = method.upper()
        async with httpx.AsyncClient(timeout=cls._timeout()) as client:
            for retry_index in range(1, max_retries + 1):
                logger.info(
                    "[session-client] request_start phase=%s method=%s url=%s retry=%d/%d",
                    phase,
                    method,
                    url,
                    retry_index,
                    max_retries,
                )
                try:
                    kwargs = {"headers": headers}
                    if method not in {"GET", "DELETE"}:
                        kwargs["json"] = payload or {}
                    response = await client.request(method, url, **kwargs)
                    logger.info(
                        "[session-client] response_received phase=%s status=%d bytes=%d retry=%d/%d",
                        phase,
                        response.status_code,
                        cls._response_size(response),
                        retry_index,
                        max_retries,
                    )
                    response.raise_for_status()
                    if response.status_code == 204 or not response.content:
                        return None
                    if return_bytes:
                        return response.content
                    return await asyncio.to_thread(orjson.loads, response.content) if expect_json else response.text
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 429 and exc.response.status_code < 500:
                        raise
                    failure = exc
                except httpx.TransportError as exc:
                    failure = exc
                logger.info(
                    "[session-client] request_failed phase=%s method=%s url=%s retry=%d/%d error=%r",
                    phase,
                    method,
                    url,
                    retry_index,
                    max_retries,
                    failure,
                )
                if retry_index == max_retries:
                    raise failure
                await asyncio.sleep(cls._backoff_seconds(retry_index))
        raise RuntimeError(f"No request performed for {phase}: max_retries={max_retries}")

    @property
    def session_server_id(self) -> str:
        """``ip:port`` of the instance owning this session, as recorded in sample metadata."""
        return self.router_url.removeprefix("http://").removeprefix("https://")

    @staticmethod
    async def create(args: Namespace):
        session_addrs = getattr(args, "session_server_addrs", None)
        legacy_backends = getattr(args, "session_server_backends", None)
        if session_addrs:
            session_addr = random.choice(session_addrs)
            session_url = normalize_session_server_urls([session_addr])[0]
        elif legacy_backends:
            session_url = random.choice(legacy_backends).rstrip("/")
            session_addr = session_url.removeprefix("http://").removeprefix("https://")
        elif getattr(args, "session_server_ip", None) and getattr(args, "session_server_port", None):
            session_addr = f"{args.session_server_ip}:{args.session_server_port}"
            session_url = normalize_session_server_urls([session_addr])[0]
        else:
            raise RuntimeError(
                "session_server_addrs is not set. Pass --use-session-server to start the session server."
            )
        # Bind the session once. Canonical workers publish instance IDs at startup;
        # legacy callers discover theirs from health, as on stable.
        instance_ids = getattr(args, "session_server_instance_ids", None) or {}
        session_server_instance_id = instance_ids.get(session_addr)
        if not session_addrs or getattr(args, "_session_server_external_pool", False):
            try:
                health = await OpenAIEndpointTracer._request(
                    "GET", f"{session_url}/health", phase="health", max_retries=_HEALTH_RETRIES
                )
                if isinstance(health, dict):
                    session_server_instance_id = health.get("session_server_instance_id")
                    if session_server_instance_id is not None:
                        args.session_server_instance_id = session_server_instance_id
            except Exception as exc:
                logger.warning("Failed to get session server health from %s: %s", session_url, exc)
        response = await OpenAIEndpointTracer._request(
            "POST", f"{session_url}/sessions", phase="create_session", max_retries=_CREATE_RETRIES
        )
        if not isinstance(response, dict) or "session_id" not in response:
            raise RuntimeError(f"invalid create session response from {session_url}: {response!r}")
        session_id = response["session_id"]
        use_v2 = getattr(args, "use_session_server", None) == "v2"
        return OpenAIEndpointTracer(
            router_url=session_url,
            session_id=session_id,
            session_server_instance_id=session_server_instance_id,
            samples_wire_fields=COMPUTED_FIELDS_V2 if use_v2 else COMPUTED_FIELDS,
        )

    async def collect_samples(
        self, input_sample: Sample, *, max_seq_len: int | None, agent_metadata: dict | None = None
    ) -> SamplesReply:
        """Collect the current binary wire with stable v1 failure/retry policy.

        V2 retains its one-request and loud deterministic-error contract.
        """
        use_v2 = self.samples_wire_fields == COMPUTED_FIELDS_V2
        body = {"max_seq_len": max_seq_len}
        if not use_v2:
            body["decode_response"] = False
        if agent_metadata is not None:
            body["metadata"] = agent_metadata
        async with _COLLECT_RECORDS_SEMAPHORE:
            try:
                payload = await self._request(
                    "POST",
                    f"{self.base_url}/samples",
                    phase="collect_samples",
                    payload=body,
                    max_retries=1 if use_v2 else _COLLECT_RETRIES,
                    return_bytes=True,
                )
                return await asyncio.to_thread(
                    decode_samples_and_merge_input_sample, payload, input_sample, fields=self.samples_wire_fields
                )
            except Exception as exc:
                if use_v2:
                    raise
                logger.warning(
                    "[session-client] collect_failed session_id=%s error=%r returning_empty=True", self.session_id, exc
                )
                return SamplesReply(samples=[], session_metadata={}, empty_reason="collection_failed")
            finally:
                await self.delete_session()

    async def delete_session(self) -> None:
        try:
            await self._request(
                "DELETE",
                self.base_url,
                phase="delete_session",
                max_retries=_DELETE_RETRIES,
                expect_json=False,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                logger.warning("Failed to delete session %s: %s", self.session_id, exc)
        except Exception as exc:
            logger.warning("Failed to delete session %s: %s", self.session_id, exc)
