import asyncio
from argparse import Namespace

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
