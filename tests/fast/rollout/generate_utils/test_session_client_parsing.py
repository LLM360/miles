import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest

from miles.rollout.generate_utils import openai_endpoint_utils as endpoint
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, SamplesReply
from miles.utils.types import Sample


@pytest.mark.parametrize("operation", ["json", "v1", "v2"])
async def test_response_parsing_keeps_event_loop_responsive(monkeypatch, operation):
    entered, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    expected = {"session_id": "sid"} if operation == "json" else SamplesReply([], {}, "no_records")

    def parse(*args, **kwargs):
        assert threading.get_ident() != main_thread
        entered.set()
        assert release.wait(5)
        return expected

    tracer = endpoint.OpenAIEndpointTracer("http://worker", "sid")
    deleted = []
    if operation == "json":
        monkeypatch.setattr(endpoint, "orjson", SimpleNamespace(loads=parse))
        factory = httpx.AsyncClient
        transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b"{}"))
        monkeypatch.setattr(endpoint.httpx, "AsyncClient", lambda **kwargs: factory(transport=transport, **kwargs))
        coro = tracer._request("POST", "http://worker/sessions", phase="create_session")
    else:

        async def request(method, url, **kwargs):
            if method == "DELETE":
                deleted.append(url)
            return b"wire payload"

        monkeypatch.setattr(tracer, "_request", request)
        monkeypatch.setattr(endpoint, "decode_samples_and_merge_input_sample", parse)
        if operation == "v2":
            tracer.samples_wire_fields = COMPUTED_FIELDS_V2
        coro = tracer.collect_samples(Sample(), max_seq_len=None)
    task = asyncio.create_task(coro)
    try:

        async def wait_entered():
            while not entered.is_set():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_entered(), 2)
        assert not task.done()
        await asyncio.sleep(0.01)
        assert not task.done()
    finally:
        release.set()
    assert await asyncio.wait_for(task, 2) is expected
    if operation != "json":
        assert deleted == [tracer.base_url]
