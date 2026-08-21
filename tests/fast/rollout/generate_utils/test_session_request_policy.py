import asyncio
from types import SimpleNamespace

import httpx
import pytest

import miles.rollout.generate_utils.openai_endpoint_utils as endpoint
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2
from miles.utils.types import Sample


@pytest.mark.parametrize("first", [429, 500, "transport"])
async def test_transient_request_failure_retries_on_same_url(monkeypatch, first):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.extensions["timeout"] == dict(connect=10.0, read=120.0, write=30.0, pool=10.0)
        if len(calls) == 1:
            if first == "transport":
                raise httpx.ConnectError("temporary", request=request)
            return httpx.Response(first, json={"error": "temporary"})
        return httpx.Response(200, json={"session_id": "created"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        endpoint.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    monkeypatch.setattr(endpoint.OpenAIEndpointTracer, "_backoff_seconds", staticmethod(lambda n: 0))
    reply = await endpoint.OpenAIEndpointTracer._request(
        "POST", "http://worker/sessions", phase="create", max_retries=2
    )
    assert reply == {"session_id": "created"}
    assert len(calls) == 2 and calls[0].url == calls[1].url


@pytest.mark.parametrize("status", [400, 401, 404, 422])
async def test_permanent_client_error_is_not_retried(monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "permanent"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        endpoint.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    with pytest.raises(httpx.HTTPStatusError):
        await endpoint.OpenAIEndpointTracer._request("POST", "http://worker/samples", phase="collect", max_retries=3)
    assert len(calls) == 1


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("error", [httpx.ReadTimeout("slow"), ValueError("invalid wire")])
async def test_collection_policy_is_stable_v1_and_current_v2(monkeypatch, version, error):
    calls = []

    async def request(method, url, **kwargs):
        calls.append((method, kwargs))
        if method == "POST":
            raise error

    monkeypatch.setattr(endpoint.OpenAIEndpointTracer, "_request", staticmethod(request))
    tracer = endpoint.OpenAIEndpointTracer("http://worker", "sid")
    if version == 2:
        tracer.samples_wire_fields = COMPUTED_FIELDS_V2
        with pytest.raises(type(error)):
            await tracer.collect_samples(Sample(), max_seq_len=None)
    else:
        reply = await tracer.collect_samples(Sample(), max_seq_len=None)
        assert reply.samples == [] and reply.empty_reason == "collection_failed"
    assert [method for method, _ in calls] == ["POST", "DELETE"]
    assert calls[0][1]["max_retries"] == (3 if version == 1 else 1)


async def test_collection_cancel_propagates_and_cleans_up(monkeypatch):
    entered = asyncio.Event()
    deleted = []

    async def request(method, url, **kwargs):
        if method == "POST":
            entered.set()
            await asyncio.Event().wait()
        deleted.append(url)

    monkeypatch.setattr(endpoint.OpenAIEndpointTracer, "_request", staticmethod(request))
    tracer = endpoint.OpenAIEndpointTracer("http://worker", "sid")
    task = asyncio.create_task(tracer.collect_samples(Sample(), max_seq_len=None))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert deleted == [tracer.base_url]


async def test_https_external_session_keeps_its_scheme_and_identity(monkeypatch):
    calls = []

    async def request(method, url, **kwargs):
        calls.append(url)
        return {"session_server_instance_id": "external"} if url.endswith("health") else {"session_id": "sid"}

    monkeypatch.setattr(endpoint.OpenAIEndpointTracer, "_request", staticmethod(request))
    args = SimpleNamespace(session_server_addrs=["https://worker:123/"], _session_server_external_pool=True)
    tracer = await endpoint.OpenAIEndpointTracer.create(args)
    assert tracer.base_url == "https://worker:123/sessions/sid"
    assert tracer.session_server_instance_id == "external"
    assert calls == ["https://worker:123/health", "https://worker:123/sessions"]
