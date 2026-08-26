import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.inference_rollout import adaptive_inference_rollout_train, inference_rollout_train
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.rollout.inference_rollout.inference_rollout_common import (
    GenerationHealthTracker,
    is_usable_generation_group,
)
from miles.utils.types import Sample


ROLLOUT_MODULES = [inference_rollout_train, adaptive_inference_rollout_train]


def _sample(
    status: Sample.Status,
    *,
    index: int = 0,
    with_output: bool = False,
) -> Sample:
    return Sample(
        index=index,
        prompt="prompt",
        tokens=[101] if with_output else [],
        response="answer" if with_output else "",
        response_length=1 if with_output else 0,
        reward=0,
        status=status,
    )


def _args(**overrides):
    values = {
        "rollout_global_dataset": True,
        "dynamic_sampling_filter_path": "test:drop_all",
        "dynamic_sampling_min_reward_std": None,
        "dynamic_sampling_min_mean_reward": None,
        "dynamic_sampling_max_mean_reward": None,
        "rollout_sample_filter_path": None,
        "rollout_all_samples_process_path": None,
        "rollout_batch_size": 2,
        "n_samples_per_prompt": 1,
        "disable_oversampling": False,
        "over_sampling_batch_size": 2,
        "reward_key": None,
        "partial_rollout": False,
        "use_session_server": False,
        "custom_agent_function_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _State:
    def __init__(self, args):
        self.args = args
        self.aborted = False
        self.reset_calls = 0

    def reset(self):
        self.aborted = False
        self.reset_calls += 1


class _UnboundedSource:
    def __init__(self, status: Sample.Status, *, with_output: bool):
        self.status = status
        self.with_output = with_output
        self.calls = 0
        self.next_index = 0

    def __call__(self, count):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("rollout submitted a second wave before rejecting the first")
        groups = []
        for _ in range(count):
            groups.append(
                [
                    _sample(
                        self.status,
                        index=self.next_index,
                        with_output=self.with_output,
                    )
                ]
            )
            self.next_index += 1
        return groups


def _drop_all(_args, _group):
    return DynamicFilterOutput(keep=False, reason="test_drop")


def _install_loop_fakes(monkeypatch, module, *, tasks_raise: bool = False, abort_error: Exception | None = None):
    async def configure_sglang(_args):
        return None

    def load_function(path):
        return _drop_all if path == "test:drop_all" else None

    def submit_generate_tasks(_state, groups):
        async def finish(group, delay):
            # Stagger a wave so FIRST_COMPLETED leaves one request pending.
            # This catches accidental inter-wave top-ups before the original
            # wave reaches its health boundary.
            await asyncio.sleep(delay)
            if tasks_raise:
                raise RuntimeError("generation task failed")
            return group

        return [asyncio.create_task(finish(group, 0 if idx == 0 else 0.02)) for idx, group in enumerate(groups)]

    async def abort(state, pendings, _rollout_id):
        state.aborted = True
        if pendings:
            await asyncio.gather(*pendings, return_exceptions=True)
        if abort_error is not None:
            raise abort_error
        return []

    monkeypatch.setattr(module.dumper_utils, "configure_sglang", configure_sglang)
    monkeypatch.setattr(module, "load_function", load_function)
    monkeypatch.setattr(module, "submit_generate_tasks", submit_generate_tasks)
    monkeypatch.setattr(module, "abort", abort)


@pytest.mark.parametrize("module", ROLLOUT_MODULES)
@pytest.mark.parametrize(
    ("status", "with_output"),
    [
        (Sample.Status.ABORTED, True),
        (Sample.Status.PENDING, True),
        (Sample.Status.FAILED, False),
    ],
)
async def test_unusable_unbounded_wave_fails_before_second_submission(
    monkeypatch,
    module,
    status,
    with_output,
):
    _install_loop_fakes(monkeypatch, module)
    source = _UnboundedSource(status, with_output=with_output)
    state = _State(_args())

    with pytest.raises(RuntimeError, match="generation collapsed"):
        await asyncio.wait_for(
            module.generate_rollout_async(state, rollout_id=7, data_source=source),
            timeout=2,
        )

    assert source.calls == 1
    assert state.reset_calls == 1
    assert not state.aborted


@pytest.mark.parametrize("module", ROLLOUT_MODULES)
async def test_all_task_exceptions_fail_before_second_submission(monkeypatch, module):
    _install_loop_fakes(monkeypatch, module, tasks_raise=True)
    source = _UnboundedSource(Sample.Status.PENDING, with_output=False)
    state = _State(_args())

    with pytest.raises(RuntimeError, match=r"task_exceptions=2"):
        await asyncio.wait_for(
            module.generate_rollout_async(state, rollout_id=8, data_source=source),
            timeout=2,
        )

    assert source.calls == 1
    assert state.reset_calls == 1


@pytest.mark.parametrize("module", ROLLOUT_MODULES)
async def test_later_unusable_wave_fails_after_healthy_filtered_wave(monkeypatch, module):
    _install_loop_fakes(monkeypatch, module)
    state = _State(_args())
    calls = 0
    next_index = 0

    def source(count):
        nonlocal calls, next_index
        calls += 1
        status = Sample.Status.COMPLETED if calls == 1 else Sample.Status.ABORTED
        groups = [
            [_sample(status, index=next_index + offset, with_output=True)]
            for offset in range(count)
        ]
        next_index += count
        return groups

    with pytest.raises(RuntimeError, match="generation collapsed"):
        await module.generate_rollout_async(state, rollout_id=81, data_source=source)

    assert calls == 2
    assert state.reset_calls == 1


@pytest.mark.parametrize("module", ROLLOUT_MODULES)
async def test_healthy_filtered_finite_exhaustion_is_not_a_collapse(monkeypatch, module):
    _install_loop_fakes(monkeypatch, module)
    state = _State(_args())
    calls = 0

    def finite_source(_count):
        nonlocal calls
        calls += 1
        return [[_sample(Sample.Status.COMPLETED, with_output=True)]] if calls == 1 else []

    output, aborted_samples = await module.generate_rollout_async(state, rollout_id=9, data_source=finite_source)

    assert output.stop_training
    assert output.stop_reason == "consumed_prompt_budget_exhausted"
    assert output.samples == []
    assert len(output.all_samples) == 1
    assert aborted_samples == []
    assert calls == 1
    assert state.reset_calls == 1


@pytest.mark.parametrize("module", ROLLOUT_MODULES)
async def test_cleanup_failure_still_resets_rollout_state(monkeypatch, module):
    cleanup_error = RuntimeError("cleanup failed")
    _install_loop_fakes(monkeypatch, module, abort_error=cleanup_error)
    state = _State(_args())

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await module.generate_rollout_async(state, rollout_id=10, data_source=lambda _count: [])

    assert state.reset_calls == 1
    assert not state.aborted


@pytest.mark.parametrize("module", ROLLOUT_MODULES)
async def test_abort_drains_pending_tasks_when_engine_cleanup_fails(monkeypatch, module):
    drained = asyncio.Event()

    async def fail_engine_cleanup(_args):
        raise RuntimeError("engine cleanup failed")

    async def pending_group():
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()

    monkeypatch.setattr(module, "_abort_all_engines", fail_engine_cleanup)
    state = _State(_args())
    task = asyncio.create_task(pending_group())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="engine cleanup failed"):
        await module.abort(state, {task}, rollout_id=11)

    assert drained.is_set()
    assert task.done()
    assert task.cancelled()
    assert state.aborted


async def test_failed_group_task_cancels_and_drains_sibling_generation(monkeypatch):
    sibling_started = asyncio.Event()
    sibling_drained = asyncio.Event()

    async def generate_and_rm(_state, sample, _sampling_params, evaluation=False):
        del evaluation
        if sample.index == 0:
            await sibling_started.wait()
            raise RuntimeError("first sample failed")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_drained.set()

    monkeypatch.setattr(common, "generate_and_rm", generate_and_rm)
    state = SimpleNamespace(
        args=SimpleNamespace(sglang_enable_deterministic_inference=False, group_rm=False),
        aborted=False,
    )

    with pytest.raises(RuntimeError, match="first sample failed"):
        await common.generate_and_rm_group(
            state,
            [_sample(Sample.Status.PENDING, index=0), _sample(Sample.Status.PENDING, index=1)],
            sampling_params={},
        )

    assert sibling_drained.is_set()


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ([_sample(Sample.Status.COMPLETED, with_output=True)], True),
        ([_sample(Sample.Status.TRUNCATED, with_output=True)], True),
        ([_sample(Sample.Status.FAILED, with_output=True)], True),
        ([_sample(Sample.Status.FAILED, with_output=False)], False),
        ([_sample(Sample.Status.PENDING, with_output=True)], False),
        ([_sample(Sample.Status.ABORTED, with_output=True)], False),
        ([], False),
    ],
)
def test_usable_generation_group_contract(group, expected):
    assert is_usable_generation_group(group) is expected


def test_health_window_resets_after_usable_group():
    tracker = GenerationHealthTracker()
    tracker.record_submitted(1)
    tracker.record_returned([_sample(Sample.Status.COMPLETED, with_output=True)])
    tracker.close_window(pending_groups=0)

    assert tracker.submitted_groups == 0
    assert tracker.returned_groups == 0


def test_health_window_rejects_inconsistent_accounting():
    tracker = GenerationHealthTracker()
    tracker.record_submitted(2)
    tracker.record_returned([_sample(Sample.Status.COMPLETED, with_output=True)])

    with pytest.raises(RuntimeError, match="accounting is inconsistent"):
        tracker.close_window(pending_groups=0)
