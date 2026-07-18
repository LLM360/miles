import asyncio

import pytest
from tests.fast.rollout.generate_hub.test_agentic_v2 import _generate_input, _patch_agent, _Tracer

from miles.rollout.generate_hub import agentic_tool_call as agent
from miles.rollout.session.samples.codec import SamplesReply
from miles.utils.types import Sample


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("aborted", [False, True])
@pytest.mark.parametrize("collect_timeout", [False, True])
async def test_cancelled_agent_collects_with_abort_deadline(monkeypatch, version, aborted, collect_timeout):
    input = _generate_input(rollout_abort_session_collect_timeout=0.01)
    input.args.use_session_server = version
    input.state.aborted = aborted
    sample = Sample(response="", tokens=[1, 2], response_length=1, status=Sample.Status.COMPLETED)
    tracer = _Tracer(SamplesReply(samples=[sample], session_metadata={}, empty_reason=None))
    collected = asyncio.Event()

    async def collect(*args, **kwargs):
        collected.set()
        if collect_timeout and aborted:
            await asyncio.Event().wait()
        return tracer.reply

    tracer.collect_samples = collect
    _patch_agent(monkeypatch, tracer)

    async def cancel(**kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(agent, "load_function", lambda path: cancel)
    if not aborted:
        with pytest.raises(asyncio.CancelledError):
            await agent.generate(input)
    else:
        output = await asyncio.wait_for(agent.generate(input), 1)
        out = output.samples[0] if version == "v2" else output.samples
        if collect_timeout:
            assert out.status == Sample.Status.ABORTED
        elif version == "v1":
            assert out.tokens == [1, 2]
            assert out.metadata["agent_metrics"]["rollout_abort_cancelled_count"] == 1
            assert out.status == Sample.Status.ABORTED
    assert collected.is_set()
