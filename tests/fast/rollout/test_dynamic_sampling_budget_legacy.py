import asyncio
from functools import wraps
from types import SimpleNamespace

import pytest

from miles.rollout import sglang_rollout
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.filter_hub.dynamic_sampling_budget import EvaluatorInvalidReplacementBudgetExceeded
from miles.utils.types import Sample


def _run_async(test_function):
    @wraps(test_function)
    def wrapper(*args, **kwargs):
        return asyncio.run(test_function(*args, **kwargs))

    return wrapper


class _LegacyState:
    def __init__(self):
        self.remaining_batch_size = 0
        self.pendings = set()
        self.reset_calls = 0

    def submit_generate_tasks(self, groups):
        async def finish(group):
            return group

        self.pendings.update(asyncio.create_task(finish(group)) for group in groups)
        self.remaining_batch_size += len(groups)

    def reset(self):
        self.remaining_batch_size = 0
        self.pendings = set()
        self.reset_calls += 1


@_run_async
async def test_legacy_rollout_honors_evaluator_invalid_replacement_budget(monkeypatch):
    args = SimpleNamespace(
        rollout_global_dataset=True,
        dynamic_sampling_filter_path="test:drop_all",
        dynamic_sampling_max_rejected_groups_without_progress=2,
        rollout_batch_size=2,
        n_samples_per_prompt=1,
        over_sampling_batch_size=2,
        reward_key=None,
        partial_rollout=False,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
    )
    state = _LegacyState()

    async def configure_sglang(_args):
        return None

    async def abort(_args, _rollout_id):
        if state.pendings:
            await asyncio.gather(*state.pendings, return_exceptions=True)
            state.pendings.clear()
        return []

    source_calls = 0

    def source(count):
        nonlocal source_calls
        source_calls += 1
        groups = []
        for index in range(count):
            sample = Sample(
                index=index,
                prompt="prompt",
                tokens=[101],
                response="answer",
                response_length=1,
                reward=None,
                status=Sample.Status.ABORTED,
                metadata={"evaluation_failed": True},
            )
            groups.append([sample])
        return groups

    monkeypatch.setattr(sglang_rollout.dumper_utils, "configure_sglang", configure_sglang)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(
        sglang_rollout,
        "load_function",
        lambda _path: lambda _args, _group: DynamicFilterOutput(keep=False, reason="group_has_aborted"),
    )
    monkeypatch.setattr(sglang_rollout, "abort", abort)

    with pytest.raises(EvaluatorInvalidReplacementBudgetExceeded, match="evaluator-invalid groups"):
        await sglang_rollout.generate_rollout_async(args, rollout_id=85, data_source=source)

    assert state.reset_calls == 1
    assert source_calls == 1
