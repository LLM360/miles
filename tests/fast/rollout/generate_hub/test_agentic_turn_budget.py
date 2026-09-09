import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub import agentic_tool_call
from miles.rollout.generate_hub.agentic_tool_call import _mark_limits_exceeded_truncated
from miles.rollout.session.session_types import MergedSessionSample
from miles.utils.types import Sample


@pytest.mark.parametrize("as_list", [False, True], ids=["merged", "multi-sample"])
def test_limits_exceeded_marks_only_final_sample_truncated(as_list: bool) -> None:
    samples = [
        Sample(status=Sample.Status.COMPLETED),
        Sample(status=Sample.Status.COMPLETED),
    ]
    value = samples if as_list else samples[-1]

    _mark_limits_exceeded_truncated(value, {"exit_status": "LimitsExceeded"})

    assert samples[-1].status == Sample.Status.TRUNCATED
    if as_list:
        assert samples[0].status == Sample.Status.COMPLETED


@pytest.mark.parametrize("agent_metadata", [None, {}, {"exit_status": "Submitted"}])
def test_other_exit_statuses_remain_completed(agent_metadata: dict | None) -> None:
    sample = Sample(status=Sample.Status.COMPLETED)

    _mark_limits_exceeded_truncated(sample, agent_metadata)

    assert sample.status == Sample.Status.COMPLETED


@pytest.mark.parametrize("exit_status", ["LimitsExceeded", "Submitted"])
@pytest.mark.parametrize("empty_session", [False, True])
def test_generate_preserves_server_merged_sample(monkeypatch, exit_status, empty_session):
    merged = MergedSessionSample(
        tokens=[1, 2, 3, 4],
        response="",
        response_length=3,
        loss_mask=[1, 0, 1],
        rollout_log_probs=[-0.1, 0.0, -0.2],
        status="completed",
        weight_versions=["v1", "v2"],
    )
    tracer = SimpleNamespace(
        session_id="test-session",
        session_server_instance_id="test-instance",
        base_url="http://session-server/v1",
        collect_merged_sample=AsyncMock(
            return_value=(None if empty_session else merged, {"session_key": "preserved"})
        ),
    )
    monkeypatch.setattr(agentic_tool_call.OpenAIEndpointTracer, "create", AsyncMock(return_value=tracer))
    agent = AsyncMock(return_value={"exit_status": exit_status, "reward": 1.0})
    monkeypatch.setattr(agentic_tool_call, "load_function", lambda _: agent)
    args = SimpleNamespace(
        session_server_ip="127.0.0.1",
        session_server_port=1234,
        custom_agent_function_path="test-agent",
        generate_multi_samples=False,
    )
    original = Sample(prompt="task", metadata={"task_id": "task-1"})
    result = asyncio.run(
        agentic_tool_call.generate(
            GenerateFnInput(
                state=SimpleNamespace(args=args, tokenizer=None),
                sample=original,
                sampling_params={},
                evaluation=False,
            )
        )
    )

    tracer.collect_merged_sample.assert_awaited_once()
    sample = result.samples
    assert isinstance(sample, Sample)
    assert original.status == Sample.Status.PENDING
    assert original.metadata == {"task_id": "task-1"}
    if empty_session:
        assert sample.status == Sample.Status.ABORTED
        return

    expected_status = Sample.Status.TRUNCATED if exit_status == "LimitsExceeded" else Sample.Status.COMPLETED
    assert sample.status == expected_status
    for field in ("tokens", "response", "response_length", "loss_mask", "rollout_log_probs", "weight_versions"):
        assert getattr(sample, field) == getattr(merged, field)
    assert sample.metadata == {
        "task_id": "task-1",
        "response_decoded": False,
        "exit_status": exit_status,
        "reward": 1.0,
        "session_key": "preserved",
    }
