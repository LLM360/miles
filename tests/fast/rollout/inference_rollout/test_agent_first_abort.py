import asyncio
from types import SimpleNamespace

import pytest

import miles.rollout.inference_rollout.inference_rollout_train as train
from miles.rollout.generate_utils.sample_utils import drop_samples_after_first_non_completed
from miles.utils.types import Sample


@pytest.mark.parametrize("agentic", [False, True])
async def test_abort_signals_agent_before_engines_and_drains_failures(monkeypatch, agentic):
    calls = []
    state = SimpleNamespace(
        aborted=False,
        args=SimpleNamespace(
            use_session_server=agentic,
            custom_agent_function_path="plugin.generate",
            agent_server_url="http://agent",
            partial_rollout=True,
        ),
    )

    async def post(url, body):
        calls.append(url)
        if url.endswith("abort_all"):
            return {"aborted_trials": 1}
        if "engine-b" in url:
            raise RuntimeError("engine unavailable")
        return {}

    async def urls(args):
        return ["http://engine-a", "http://engine-b"]

    async def hook(args):
        calls.append("plugin")

    async def result(group=None):
        if group is None:
            raise ValueError("failed rollout")
        return group

    sample = Sample(response="partial", metadata={"start_rollout_id": 1})
    nested = Sample(response="partial", metadata={})
    tasks = {asyncio.create_task(result([sample, [nested]])), asyncio.create_task(result())}
    cancelled = asyncio.create_task(asyncio.Event().wait())
    cancelled.cancel()
    tasks.add(cancelled)
    monkeypatch.setattr(train, "post", post)
    monkeypatch.setattr(train, "get_worker_urls", urls)
    monkeypatch.setattr(train, "call_agent_abort_hook", hook)
    groups = await train.abort(state, tasks, 7)
    assert calls == (["http://agent/abort_all"] if agentic else ["plugin"]) + [
        "http://engine-a/abort_request",
        "http://engine-b/abort_request",
    ]
    assert groups == [[sample, [nested]]]
    assert sample.metadata["start_rollout_id"] == 1
    assert nested.metadata["start_rollout_id"] == 7
    assert state.aborted and all(task.done() for task in tasks)


async def test_agentic_abort_requires_matching_agent_target():
    state = SimpleNamespace(
        aborted=False, args=SimpleNamespace(use_session_server=True, custom_agent_function_path="a.b")
    )
    with pytest.raises(RuntimeError, match="agent-server-url"):
        await train.abort(state, set(), 1)


@pytest.mark.parametrize("status", [Sample.Status.TRUNCATED, Sample.Status.ABORTED])
def test_only_prefix_through_first_incomplete_turn_is_kept(status):
    samples = [Sample(status=Sample.Status.COMPLETED), Sample(status=status), Sample(status=Sample.Status.COMPLETED)]
    kept, count = drop_samples_after_first_non_completed(samples)
    assert kept == samples[:2] and count == 1


def test_nested_dispatch_metadata_preserves_origin():
    sample = Sample(metadata={"start_rollout_id": 2, "rollout_id": 3})
    train.stamp_rollout_id([[[sample]]], 4)
    assert sample.metadata == {"start_rollout_id": 2, "rollout_id": 4}


@pytest.mark.parametrize("nested", [False, True])
async def test_submission_names_keep_sample_completion_callback(monkeypatch, nested):
    from miles.rollout.inference_rollout import inference_rollout_train as train

    def callback():
        return None

    state = SimpleNamespace(sampling_params={"temperature": 0.7})
    sample = Sample(index=27)
    group = [[sample]] if nested else [sample]
    seen = []

    async def generate(actual_state, actual_group, **kwargs):
        seen.append((actual_state, actual_group, kwargs))
        return actual_group

    monkeypatch.setattr(train, "generate_and_rm_group", generate)
    tasks = train.submit_generate_tasks(state, [group], callback)
    assert [task.get_name() for task in tasks] == ["group-27"]
    assert await tasks[0] is group
    actual_state, actual_group, kwargs = seen[0]
    assert actual_state is state and actual_group is group
    assert kwargs["sample_done_callback"] is callback and kwargs["evaluation"] is False
    assert kwargs["sampling_params"] == state.sampling_params
    assert kwargs["sampling_params"] is not state.sampling_params
