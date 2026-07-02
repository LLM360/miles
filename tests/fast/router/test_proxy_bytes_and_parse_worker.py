import asyncio
import threading

import pytest
from tests.fast.router.test_session_worker_concurrency import _core
from tests.fast.router.test_stable_session_recovery import _chat

from miles.rollout.session.core import proxy_result_to_response


@pytest.mark.parametrize("content", [b'{ "unicode": "\\u00e9", "n": 1.00 }', b"not-json", b""])
def test_proxy_keeps_exact_bytes_and_recomputes_framing(content):
    response = proxy_result_to_response(
        {
            "response_body": content,
            "status_code": 400,
            "headers": {"content-type": "application/json", "content-length": "999", "server": "upstream"},
        }
    )
    assert response.body == content
    assert response.headers["content-length"] == str(len(content))
    assert "server" not in response.headers


@pytest.mark.parametrize("version", [1, 2])
async def test_completion_parse_runs_in_worker_while_health_stays_responsive(monkeypatch, version):
    from miles.rollout.session import core as v1
    from miles.rollout.session.v2 import core as v2

    module = v1 if version == 1 else v2
    original = module.extract_completion
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()

    def parse(result):
        assert threading.get_ident() != loop_thread
        entered.set()
        assert release.wait(5)
        return original(result)

    monkeypatch.setattr(module, "extract_completion", parse)
    core, sid = _core(version)
    task = asyncio.create_task(_chat(core, sid))
    try:
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        assert (await asyncio.wait_for(core.health(), 1)).status_code == 200
        assert not task.done()
    finally:
        release.set()
    assert (await task).status_code == 200
