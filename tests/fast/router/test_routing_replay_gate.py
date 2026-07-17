"""R3 ownership is enforced before either session version commits a turn."""

import asyncio
import json

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.router.test_session_worker_concurrency import _core
from tests.fast.router.test_stable_session_recovery import _Backend, _result

from miles.rollout.session.core import gate_routed_experts
from miles.rollout.session.errors import UpstreamResponseError


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("location", ["meta", "choice", "missing"])
def test_routing_ownership_and_atomic_rejection(version, enabled, location):
    async def scenario():
        core, sid = _core(version)
        core.config = make_session_server_config(use_rollout_routing_replay=enabled)
        result = _result()
        body = json.loads(result["response_body"])
        choice = body["choices"][0]
        if location != "missing":
            target = choice["meta_info"] if location == "meta" else choice
            target["routed_experts"] = "AAAAAA=="
        result["response_body"] = json.dumps(body).encode()
        core.backend = _Backend([result])
        request = json.dumps(
            {
                "messages": [{"role": "user", "content": "task"}],
                "return_routed_experts": True,
                "routed_experts_start_len": 999,
            }
        ).encode()
        call = core.chat_completions(sid, method="POST", query="", headers={}, body=request)
        if enabled and location == "missing":
            with pytest.raises(UpstreamResponseError, match="routed_experts must"):
                await call
        else:
            assert (await call).status_code == 200
        sent = core.backend.requests[-1]
        assert sent.get("return_routed_experts", False) is enabled
        if not enabled:
            assert "routed_experts_start_len" not in sent
        records = json.loads((await core.get_session(sid)).body)["records"]
        if enabled and location == "missing":
            assert records == []
        else:
            recorded = records[0]["response"]["choices"][0]
            assert "routed_experts" not in recorded
            assert ("routed_experts" in recorded["meta_info"]) is enabled

    asyncio.run(scenario())


@pytest.mark.parametrize("offset,allowed", [(2, True), (1, False), (3, False)])
def test_only_zero_new_rows_can_omit_incremental_payload(offset, allowed):
    choice = {"meta_info": {"output_token_logprobs": [[-0.5, 10]]}}
    request = {"input_ids": [0, 1], "routed_experts_start_len": offset}
    if allowed:
        gate_routed_experts(choice, request, enabled=True, use_addition_r3=True)
    else:
        with pytest.raises(UpstreamResponseError):
            gate_routed_experts(choice, request, enabled=True, use_addition_r3=True)
