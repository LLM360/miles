"""Stable session recovery with the upstream core and checkpoint representation."""

import asyncio
import json

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.router.test_linear_trajectory import _make_registry

from miles.rollout.session.core import SessionCore
from miles.rollout.session.errors import SessionNotFoundError, UpstreamResponseError


class _Backend:
    def __init__(self, results):
        self.results = iter(results)
        self.requests = []

    async def do_proxy(self, request, path, *, body, headers):
        self.requests.append(json.loads(body))
        return next(self.results)


def _result(status=200, *, prompt_ids=True):
    choice = {
        "message": {"role": "assistant", "content": "answer"},
        "meta_info": {"output_token_logprobs": [[-0.5, 10]], "completion_tokens": 1},
    }
    if prompt_ids:
        choice["prompt_token_ids"] = [0]
    return {
        "status_code": status,
        "response_body": json.dumps({"choices": [choice]} if status == 200 else {"error": "rollback failed"}).encode(),
        "headers": {"content-length": "999", "transfer-encoding": "chunked", "content-encoding": "gzip"},
    }


def _chat(core, session_id="recovered"):
    return core.chat_completions(
        session_id,
        method="POST",
        query="",
        headers={},
        body=json.dumps({"messages": [{"role": "user", "content": "task"}]}).encode(),
    )


def test_unknown_session_recovers_but_explicit_delete_sticks():
    registry = _make_registry()
    backend = _Backend([_result()])
    core = SessionCore(backend, registry, make_session_server_config())
    assert json.loads(asyncio.run(core.get_session("recovered")).body)["records"] == []
    assert asyncio.run(_chat(core)).status_code == 200
    assert len(registry.get_session("recovered").records) == 1
    assert asyncio.run(core.delete_session("recovered")).status_code == 204
    with pytest.raises(SessionNotFoundError):
        asyncio.run(_chat(core))
    with pytest.raises(SessionNotFoundError):
        asyncio.run(core.get_session("recovered"))
    assert len(backend.requests) == 1


def test_prefix_retry_records_actual_prompt_tokens_and_filters_headers():
    registry = _make_registry()
    backend = _Backend([_result(400), _result()])
    core = SessionCore(backend, registry, make_session_server_config())
    response = asyncio.run(_chat(core))
    assert response.status_code == 200
    assert backend.requests[0]["input_ids"] == [0]
    assert "input_ids" not in backend.requests[1]
    session = registry.get_session("recovered")
    assert session.records[0].request["input_ids"] == [0]
    assert session.token_ids == [0, 10]
    assert response.headers["content-length"] == str(len(response.body))
    assert "transfer-encoding" not in response.headers
    assert "content-encoding" not in response.headers


def test_prefix_retry_requires_actual_backend_tokens():
    registry = _make_registry()
    backend = _Backend([_result(400), _result(prompt_ids=False)])
    core = SessionCore(backend, registry, make_session_server_config())
    with pytest.raises(UpstreamResponseError, match="requires backend prompt_token_ids"):
        asyncio.run(_chat(core))
    assert registry.get_session("recovered").records == []


@pytest.mark.parametrize("results,expected_calls", [([_result(400), _result(400)], 2), ([_result(500)], 1)])
def test_backend_retry_is_bounded_and_specific(results, expected_calls):
    registry = _make_registry()
    backend = _Backend(results)
    core = SessionCore(backend, registry, make_session_server_config())
    assert asyncio.run(_chat(core)).status_code in (400, 500)
    assert len(backend.requests) == expected_calls
    assert registry.get_session("recovered").records == []


def test_recovery_expiry_and_bounded_deleted_ids(monkeypatch):
    registry = _make_registry()
    monkeypatch.setattr("miles.rollout.session.linear_trajectory.time.monotonic", lambda: 1.0)
    registry.get_or_create_session("stale")
    monkeypatch.setattr("miles.rollout.session.linear_trajectory.time.monotonic", lambda: 8000.0)
    registry.get_or_create_session("fresh")
    assert "stale" not in registry.sessions
    registry.remove_session("fresh")
    assert "fresh" not in registry._session_last_access
    registry._MAX_DELETED_TOMBSTONES = 2
    for sid in ("second", "third"):
        registry.get_or_create_session(sid)
        registry.remove_session(sid)
    assert not registry.is_deleted("fresh")
    assert registry.is_deleted("second") and registry.is_deleted("third")


def test_role_restriction_survives_request_template_clone():
    tokenizer = _make_registry().tito_tokenizer.with_allowed_append_roles(["tool", "assistant"])
    clone = tokenizer.clone_with_chat_template_kwargs({"enable_thinking": False})
    assert clone.allowed_append_roles == frozenset({"tool", "assistant"})
    with pytest.raises(ValueError, match="unsupported"):
        _make_registry(frozenset({"tool"})).tito_tokenizer.with_allowed_append_roles(["assistant"])


def test_model_info_uses_served_name_without_session_or_backend():
    core = SessionCore(
        _Backend([]), _make_registry(), make_session_server_config(sglang_served_model_name="stable-model")
    )
    response = asyncio.run(core.proxy("unknown", "v1/model_info", method="GET", query="", headers={}, body=b""))
    assert json.loads(response.body) == {"id": "stable-model", "object": "model"}
