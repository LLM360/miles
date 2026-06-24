import json

import pytest
from tests.fast.router.test_session_worker_concurrency import _core
from tests.fast.router.test_stable_session_recovery import _chat, _result


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("reply", ["error", "malformed", "valid"])
async def test_closed_during_proxy_skips_retry_validation_and_publication(version, reply):
    core, sid = _core(version)
    session = core.registry.get_session(sid)
    calls = []

    async def proxy(*args, **kwargs):
        calls.append(kwargs)
        session.closing = True
        result = _result(400 if reply == "error" else 200)
        if reply == "malformed":
            result["response_body"] = b"not json"
        elif reply == "valid":
            body = json.loads(result["response_body"])
            body["choices"][0]["meta_info"]["routed_experts"] = [[1]]
            result["response_body"] = json.dumps(body).encode()
        return result

    core.backend.do_proxy = proxy
    result = await _chat(core, sid)
    assert result.status_code == (400 if reply == "error" else 200)
    assert len(calls) == 1
    assert not (session.records if version == 1 else session.active_records())
    assert core.request_stats.turns_completed == 0
    if reply == "valid":
        assert b"routed_experts" not in result.body
