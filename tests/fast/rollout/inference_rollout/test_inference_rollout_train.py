import asyncio
from argparse import Namespace

import pytest

from miles.rollout.inference_rollout import inference_rollout_train
from miles.utils.types import Sample


class _FakeState:
    def __init__(self, partial_rollout: bool):
        self.args = Namespace(partial_rollout=partial_rollout)
        self.aborted = False


async def _no_workers(args):
    return []


class TestAbort:
    """A single malformed session must not kill the rollout: abort() must
    collect the surviving groups and skip (not propagate) failed tasks,
    mirroring the per-task error handling of the main generation loop."""

    async def test_failing_task_does_not_propagate(self, monkeypatch):
        monkeypatch.setattr(inference_rollout_train, "get_worker_urls", _no_workers)

        async def good_group():
            return [Sample(response="partial text", status=Sample.Status.ABORTED)]

        async def bad_group():
            raise AssertionError("a.status must be COMPLETED, got Status.ABORTED")

        pendings = {asyncio.create_task(good_group()), asyncio.create_task(bad_group())}
        state = _FakeState(partial_rollout=True)

        aborted_samples = await inference_rollout_train.abort(state, pendings, rollout_id=7)

        assert state.aborted
        assert len(aborted_samples) == 1
        assert aborted_samples[0][0].metadata["start_rollout_id"] == 7

    async def test_failing_task_without_partial_rollout(self, monkeypatch):
        monkeypatch.setattr(inference_rollout_train, "get_worker_urls", _no_workers)

        async def bad_group():
            raise RuntimeError("boom")

        pendings = {asyncio.create_task(bad_group())}
        state = _FakeState(partial_rollout=False)

        aborted_samples = await inference_rollout_train.abort(state, pendings, rollout_id=0)

        assert aborted_samples == []


class TestAbortSessionDrain:
    """abort() must drain session servers BEFORE killing in-flight engine
    requests (so no agent can start a turn on aborted output) and resume
    them only after every pending generate task has returned."""

    def _make_state(self):
        state = _FakeState(partial_rollout=False)
        state.args.use_session_server = True
        state.args.session_server_backends = ["http://ss0", "http://ss1"]
        return state

    async def test_drain_before_engine_abort_and_resume_after_pendings(self, monkeypatch):
        calls = []

        async def fake_get_worker_urls(args):
            return ["http://worker0"]

        async def fake_post(url, payload, **kwargs):
            calls.append(url)
            return {}

        monkeypatch.setattr(inference_rollout_train, "get_worker_urls", fake_get_worker_urls)
        monkeypatch.setattr(inference_rollout_train, "post", fake_post)

        async def pending_group():
            calls.append("pending_task_finished")
            return []

        pendings = {asyncio.create_task(pending_group())}
        await inference_rollout_train.abort(self._make_state(), pendings, rollout_id=0)

        drains = [c for c in calls if c.endswith("/sessions/drain")]
        resumes = [c for c in calls if c.endswith("/sessions/resume")]
        assert sorted(drains) == ["http://ss0/sessions/drain", "http://ss1/sessions/drain"]
        assert sorted(resumes) == ["http://ss0/sessions/resume", "http://ss1/sessions/resume"]

        engine_abort = calls.index("http://worker0/abort_request")
        assert max(calls.index(d) for d in drains) < engine_abort
        assert calls.index("pending_task_finished") < min(calls.index(r) for r in resumes)

    async def test_unreachable_session_server_does_not_crash_abort(self, monkeypatch):
        monkeypatch.setattr(inference_rollout_train, "get_worker_urls", _no_workers)

        async def failing_post(url, payload, **kwargs):
            raise RuntimeError("session server down")

        monkeypatch.setattr(inference_rollout_train, "post", failing_post)

        aborted_samples = await inference_rollout_train.abort(self._make_state(), set(), rollout_id=0)

        assert aborted_samples == []

    async def test_resume_called_even_if_engine_abort_fails(self, monkeypatch):
        # A drained-but-never-resumed session server would deadlock every
        # subsequent rollout, so resume must run even when abort() fails.
        calls = []

        async def failing_get_worker_urls(args):
            raise RuntimeError("router down")

        async def fake_post(url, payload, **kwargs):
            calls.append(url)
            return {}

        monkeypatch.setattr(inference_rollout_train, "get_worker_urls", failing_get_worker_urls)
        monkeypatch.setattr(inference_rollout_train, "post", fake_post)

        with pytest.raises(RuntimeError, match="router down"):
            await inference_rollout_train.abort(self._make_state(), set(), rollout_id=0)

        assert "http://ss0/sessions/resume" in calls
        assert "http://ss1/sessions/resume" in calls

    async def test_no_session_server_no_drain_calls(self, monkeypatch):
        calls = []

        async def fake_get_worker_urls(args):
            return ["http://worker0"]

        async def fake_post(url, payload, **kwargs):
            calls.append(url)
            return {}

        monkeypatch.setattr(inference_rollout_train, "get_worker_urls", fake_get_worker_urls)
        monkeypatch.setattr(inference_rollout_train, "post", fake_post)

        await inference_rollout_train.abort(_FakeState(partial_rollout=False), set(), rollout_id=0)

        assert calls == ["http://worker0/abort_request"]
