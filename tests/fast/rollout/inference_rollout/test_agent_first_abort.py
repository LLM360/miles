import asyncio
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest

import miles.rollout.inference_rollout.inference_rollout_train as train
from miles.rollout.generate_utils.sample_utils import drop_samples_after_first_non_completed
from miles.utils.types import Sample


def _state(agentic=True, **overrides):
    args = dict(
        use_session_server=agentic,
        custom_agent_function_path="plugin.generate",
        agent_server_url="http://agent",
        partial_rollout=True,
        use_miles_router=True,
        sglang_router_ip="router",
        sglang_router_port=80,
        rollout_abort_timeout_seconds=1.0,
    )
    args.update(overrides)
    return SimpleNamespace(aborted=False, abort_event=asyncio.Event(), args=SimpleNamespace(**args))


def _mock_abort_http(monkeypatch, calls, *, failure=None):
    original = httpx.AsyncClient

    async def respond(request):
        path = request.url.path
        calls.append((str(request.url), json.loads(request.content) if request.content else None))
        if path == "/list_workers":
            return httpx.Response(200, json={"urls": ["http://engine-a", "http://engine-b"]})
        if failure == "engine" and request.url.host == "engine-b":
            return httpx.Response(500, json={})
        if path == "/abort_all":
            return httpx.Response(
                200, json={"status": "containment_failed"} if failure == "harbor" else {"aborted_trials": 1}
            )
        return httpx.Response(200, json={})

    monkeypatch.setattr(
        train.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs)
    )


@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("failure", [None, "engine", "harbor"])
async def test_abort_preserves_sessions_drains_and_checks_confirmation(monkeypatch, agentic, failure):
    calls = []
    state = _state(agentic)
    _mock_abort_http(monkeypatch, calls, failure=failure)

    async def hook(args):
        calls.append(("plugin", None))

    monkeypatch.setattr(train, "call_agent_abort_hook", hook)
    sample = Sample(response="partial", metadata={"start_rollout_id": 1})
    nested = Sample(response="", response_length=1, tokens=[1, 2], metadata={})

    async def result(group=None):
        if group is None:
            raise ValueError("failed rollout")
        return group

    tasks = {asyncio.create_task(result([sample, [nested]])), asyncio.create_task(result())}
    cancelled = asyncio.create_task(asyncio.Event().wait())
    cancelled.cancel()
    tasks.add(cancelled)
    if failure == "engine" or (agentic and failure == "harbor"):
        with pytest.raises(RuntimeError, match="not fully confirmed"):
            await train.abort(state, tasks, 7)
    else:
        assert await train.abort(state, tasks, 7) == [[sample, [nested]]]
    if agentic:
        body = next(body for url, body in calls if url.endswith("abort_all"))
        assert body["close_sessions"] is False and body["rollout_generation"] == 7
        assert 0 < body["timeout_seconds"] < 1
    else:
        assert ("plugin", None) in calls
    assert any("engine-a/abort_request" in url for url, _ in calls)
    assert any("engine-b/abort_request" in url for url, _ in calls)
    assert sample.metadata["start_rollout_id"] == 1
    assert nested.metadata["start_rollout_id"] == 7
    assert state.aborted and state.abort_event.is_set() and all(task.done() for task in tasks)


async def test_agentic_abort_requires_matching_agent_target():
    with pytest.raises(RuntimeError, match="agent-server-url"):
        await train.abort(_state(agent_server_url=None), set(), 1)


async def test_deadline_cancels_local_work_but_keeps_collected_partial(monkeypatch):
    _mock_abort_http(monkeypatch, [])
    sample = Sample(response_length=1, tokens=[1, 2])

    async def work():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return [sample]

    task = asyncio.create_task(work())
    await asyncio.sleep(0)
    assert await asyncio.wait_for(train.abort(_state(), {task}, 9), 2) == [[sample]]
    assert sample.metadata["start_rollout_id"] == 9


@pytest.mark.parametrize("status", [Sample.Status.TRUNCATED, Sample.Status.ABORTED])
def test_only_prefix_through_first_incomplete_turn_is_kept(status):
    samples = [Sample(status=Sample.Status.COMPLETED), Sample(status=status), Sample(status=Sample.Status.COMPLETED)]
    kept, count = drop_samples_after_first_non_completed(samples)
    assert kept == samples[:2] and count == 1


def test_nested_dispatch_metadata_preserves_origin():
    sample = Sample(metadata={"start_rollout_id": 2, "rollout_id": 3})
    train.stamp_rollout_id([[[sample]]], 4)
    first_request = sample.metadata["request_id"]
    assert uuid.UUID(first_request).hex == first_request
    assert sample.metadata == {"start_rollout_id": 2, "rollout_id": 4, "request_id": first_request}
    other = Sample(metadata={})
    train.stamp_rollout_id([[[sample], [other]]], 5)
    assert sample.metadata["request_id"] != first_request
    assert sample.metadata["request_id"] != other.metadata["request_id"]
    assert sample.metadata["start_rollout_id"] == 2
    assert sample.metadata["rollout_id"] == other.metadata["rollout_id"] == 5


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
