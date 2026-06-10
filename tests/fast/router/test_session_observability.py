"""Session telemetry accounts for every exit without changing request behavior."""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from tests.fast.router.test_session_worker_concurrency import _core
from tests.fast.router.test_stable_session_recovery import _Backend, _chat, _result

from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.errors import SessionNotFoundError
from miles.rollout.session.observability import WorkerStats, log_worker_stats


@pytest.mark.parametrize("version", [1, 2])
def test_success_error_and_missing_session_have_balanced_counters(caplog, version):
    caplog.set_level(logging.INFO)

    async def scenario():
        core, sid = _core(version)
        assert (await _chat(core, sid)).status_code == 200
        core.backend = _Backend([_result(500)])
        assert (await _chat(core, sid)).status_code == 500
        await core.delete_session(sid)
        with pytest.raises(SessionNotFoundError):
            await _chat(core, sid)
        assert core.request_stats.reqs_total == 3
        assert core.request_stats.turns_completed == 1
        assert core.request_stats.inflight == 0

    asyncio.run(scenario())
    logs = [record.getMessage() for record in caplog.records]
    assert sum("chat_start " in line for line in logs) == 3
    done = [line for line in logs if "chat_done " in line]
    assert len(done) == 3
    assert "prompt_tokens=1 completion_tokens=1 messages_len=1" in done[0]
    assert all("inflight_now=0" in line for line in done)
    assert all("lock_wait_ms=" in line and "proxy_elapsed_ms=" in line for line in done)


def test_heartbeat_reports_memory_counts_and_cancels(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(
        "miles.rollout.session.observability.psutil.Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=10 * 1024**2, vms=20 * 1024**2)),
    )

    async def scenario():
        task = asyncio.create_task(log_worker_stats(WorkerStats(port=9000, reqs_total=3, turns_completed=2), 60))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    line = next(record.getMessage() for record in caplog.records if "stats worker_port" in record.getMessage())
    assert "reqs_total=3 reqs_since_last=3 inflight_now=0 turns_completed=2 rss_mb=10 vms_mb=20" in line


@pytest.mark.parametrize(
    "args",
    [
        SimpleNamespace(session_server_backends=["http://worker:1234"]),
        SimpleNamespace(session_server_ip="worker", session_server_port=1234),
    ],
)
def test_legacy_backend_configuration_pins_the_session(monkeypatch, args):
    calls = []

    async def post(url, body, action="post"):
        calls.append((url, action))
        return {"session_server_instance_id": "worker-id"} if action == "get" else {"session_id": "session-id"}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", post)
    tracer = asyncio.run(OpenAIEndpointTracer.create(args))
    assert tracer.base_url == "http://worker:1234/sessions/session-id"
    assert tracer.session_server_instance_id == "worker-id"
    assert calls == [("http://worker:1234/health", "get"), ("http://worker:1234/sessions", "post")]


@pytest.mark.parametrize("health", [None, "not-json", RuntimeError("health unavailable")])
def test_legacy_health_failure_does_not_prevent_creation(monkeypatch, health):
    async def post(url, body, action="post"):
        if action == "get":
            if isinstance(health, Exception):
                raise health
            return health
        return {"session_id": "created"}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", post)
    tracer = asyncio.run(OpenAIEndpointTracer.create(SimpleNamespace(session_server_backends=["http://legacy:1234"])))
    assert tracer.session_id == "created"
    assert tracer.session_server_instance_id is None
