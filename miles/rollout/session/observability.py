"""Per-core request timings and worker heartbeat counters."""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

import psutil

logger = logging.getLogger(__name__)
_PHASES = ("lock_wait_ms", "tokenize_in_ms", "proxy_elapsed_ms", "tokenize_out_ms")


@dataclass
class RequestMetrics:
    req_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    durations: dict = field(default_factory=lambda: dict.fromkeys(_PHASES, 0.0))
    prompt_tokens: int = -1
    completion_tokens: int = -1
    messages_len: int = -1
    committed: bool = False


@dataclass
class WorkerStats:
    port: int | None
    reqs_total: int = 0
    turns_completed: int = 0
    inflight: int = 0


_current_request: ContextVar[RequestMetrics | None] = ContextVar("session_request_metrics", default=None)


def observe_chat(function):
    @wraps(function)
    async def wrapped(self, session_id, *, method, query, headers, body):
        stats = self.request_stats
        metrics = RequestMetrics()
        token = _current_request.set(metrics)
        started = time.monotonic()
        caller_id = next((v for k, v in headers.items() if k.lower() == "x-request-id"), "-")
        stats.reqs_total += 1
        stats.inflight += 1
        logger.debug(
            "[session-server] chat_start worker_port=%s session_id=%s req_id=%s caller_request_id=%s inflight_before=%d",
            stats.port,
            session_id,
            metrics.req_id,
            caller_id,
            stats.inflight - 1,
        )
        try:
            return await function(self, session_id, method=method, query=query, headers=headers, body=body)
        finally:
            stats.inflight -= 1
            stats.turns_completed += int(metrics.committed)
            logger.debug(
                "[session-server] chat_done worker_port=%s session_id=%s req_id=%s caller_request_id=%s "
                "lock_wait_ms=%.1f tokenize_in_ms=%.1f proxy_elapsed_ms=%.1f tokenize_out_ms=%.1f "
                "total_ms=%.1f inflight_now=%d prompt_tokens=%d completion_tokens=%d messages_len=%d",
                stats.port,
                session_id,
                metrics.req_id,
                caller_id,
                *(metrics.durations[name] for name in _PHASES),
                (time.monotonic() - started) * 1000,
                stats.inflight,
                metrics.prompt_tokens,
                metrics.completion_tokens,
                metrics.messages_len,
            )
            _current_request.reset(token)

    return wrapped


@contextmanager
def measure_phase(name):
    started = time.monotonic()
    try:
        yield
    finally:
        metrics = _current_request.get()
        if metrics is not None:
            metrics.durations[name] += (time.monotonic() - started) * 1000


@asynccontextmanager
async def measured_session_lock(lock):
    with measure_phase("lock_wait_ms"):
        await lock.acquire()
    try:
        yield
    finally:
        lock.release()


def record_request_shape(request_messages):
    metrics = _current_request.get()
    if metrics is not None:
        metrics.messages_len = len(request_messages)


def publish_response(function, *args, **kwargs):
    """Mark the actual commit even if its awaiting request gets canceled."""
    result = function(*args, **kwargs)
    metrics = _current_request.get()
    if metrics is not None:
        metrics.committed = True
        metrics.prompt_tokens = len(kwargs["prompt_token_ids"])
        metrics.completion_tokens = len(kwargs["completion_token_ids"])
    return result


async def log_worker_stats(stats, interval_seconds=30.0):
    process = psutil.Process()
    previous = 0
    while True:
        try:
            memory = process.memory_info()
            logger.info(
                "[session-server] stats worker_port=%s reqs_total=%d reqs_since_last=%d "
                "inflight_now=%d turns_completed=%d rss_mb=%.0f vms_mb=%.0f",
                stats.port,
                stats.reqs_total,
                stats.reqs_total - previous,
                stats.inflight,
                stats.turns_completed,
                memory.rss / 1024**2,
                memory.vms / 1024**2,
            )
            previous = stats.reqs_total
        except Exception:
            logger.exception("[session-server] stats logger failed")
        await asyncio.sleep(interval_seconds)


def warn_state_change(stats, session_id, expected, actual, headers):
    metrics = _current_request.get()
    caller_id = next((v for k, v in headers.items() if k.lower() == "x-request-id"), "-")
    logger.warning(
        "[session-server] state_changed_during_proxy worker_port=%s session_id=%s req_id=%s "
        "expected_num_assistant=%d got_num_assistant=%d inflight_chat_count=%d "
        "caller_request_id=%s proxy_elapsed_ms=%.1f",
        stats.port,
        session_id,
        metrics.req_id if metrics else "-",
        expected,
        actual,
        stats.inflight,
        caller_id,
        metrics.durations["proxy_elapsed_ms"] if metrics else 0.0,
    )
