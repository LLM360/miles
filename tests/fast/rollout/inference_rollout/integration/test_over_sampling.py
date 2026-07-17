import asyncio
import itertools

import pytest
from tests.fast.rollout.inference_rollout.integration.utils import (
    filter_by_reward,
    integration_env_config,
    load_and_call_train,
)

from miles.utils.misc import function_registry
from miles.utils.arguments import _validate_over_sampling_batch_size
from miles.rollout.inference_rollout import inference_rollout_train as rollout_train

_DATA_ROWS = [
    {"input": "What is 1+7?", "label": "8"},
    {"input": "What is 1+8?", "label": "wrong"},
    {"input": "What is 1+9?", "label": "wrong"},
    {"input": "What is 1+6?", "label": "wrong"},
]

_BASE_ARGV = [
    "--over-sampling-batch-size",
    "4",
    "--dynamic-sampling-filter-path",
    "test:filter_by_reward",
]


def _over_sampling_config(rollout_batch_size: int):
    return integration_env_config(["--rollout-batch-size", str(rollout_batch_size)] + _BASE_ARGV, data_rows=_DATA_ROWS)


@pytest.mark.parametrize(
    "rollout_env,expected_rounds",
    [
        pytest.param(_over_sampling_config(1), 1, id="one_round"),
        pytest.param(_over_sampling_config(2), 2, id="two_rounds"),
    ],
    indirect=["rollout_env"],
)
def test_over_sampling_rounds(rollout_env, expected_rounds):
    env = rollout_env

    with function_registry.temporary("test:filter_by_reward", filter_by_reward):
        out = load_and_call_train(env.args, env.data_source)

    assert len(out.samples) == env.args.rollout_batch_size
    assert all(group[0].reward == 1 for group in out.samples)

    requests_count = len(env.mock_server.request_log)
    expected_requests = expected_rounds * env.args.over_sampling_batch_size
    assert requests_count == expected_requests, f"Expected {expected_rounds} round(s) = {expected_requests} requests"

    metrics = out.metrics
    assert metrics["rollout/groups_submitted"] == expected_requests
    assert metrics["rollout/groups_filter_kept"] == expected_rounds
    assert metrics["rollout/groups_filter_rejected"] == expected_requests - expected_rounds
    assert metrics["rollout/groups_failed"] == 0
    assert metrics["rollout/groups_selected"] == env.args.rollout_batch_size
    assert metrics["rollout/groups_unused_completed"] == 0
    assert metrics["rollout/submission_waves"] == expected_rounds
    assert metrics["rollout/refill_waves"] == expected_rounds - 1
    assert metrics["rollout/pending_groups_at_abort"] == 0


def _install_controlled_completion(monkeypatch):
    task_ids = itertools.count()

    async def complete(group):
        first = group[0][0] if isinstance(group[0], list) else group[0]
        if first.label == "error":
            raise RuntimeError("controlled task failure")
        for sample in group:
            sample.reward = 0 if sample.label == "wrong" else 1
            sample.response = "controlled"
        return group

    def submit(_state, groups):
        return [
            asyncio.create_task(complete(group), name=f"controlled-{next(task_ids)}")
            for group in groups
        ]

    async def wait_one(pendings, return_when):
        del return_when
        await asyncio.sleep(0)
        task = min(pendings, key=lambda item: int(item.get_name().rsplit("-", 1)[1]))
        return {task}, pendings - {task}

    async def abort_pending(_state, pendings, _rollout_id):
        for task in pendings:
            task.cancel()
        await asyncio.gather(*pendings, return_exceptions=True)
        return []

    async def configure_sglang(_args):
        return None

    monkeypatch.setattr(rollout_train, "submit_generate_tasks", submit)
    monkeypatch.setattr(rollout_train.asyncio, "wait", wait_one)
    monkeypatch.setattr(rollout_train, "abort", abort_pending)
    monkeypatch.setattr(rollout_train.dumper_utils, "configure_sglang", configure_sglang)


_BOUND_ROWS = [{"input": "rejected", "label": "wrong"}] + [
    {"input": f"accepted-{index}", "label": "valid"} for index in range(191)
]


@pytest.mark.parametrize(
    "rollout_env",
    [
        pytest.param(
            integration_env_config(
                [
                    "--rollout-batch-size",
                    "128",
                    "--n-samples-per-prompt",
                    "16",
                    "--over-sampling-batch-size",
                    "64",
                    "--dynamic-sampling-filter-path",
                    "test:filter_by_reward",
                ],
                data_rows=_BOUND_ROWS,
            ),
            id="production_refill_bound",
        )
    ],
    indirect=["rollout_env"],
)
def test_refill_wave_has_bounded_queued_work(rollout_env, monkeypatch):
    _install_controlled_completion(monkeypatch)

    with function_registry.temporary("test:filter_by_reward", filter_by_reward):
        out = load_and_call_train(rollout_env.args, rollout_env.data_source)

    assert len(out.samples) == 128
    assert out.metrics["rollout/groups_submitted"] == 192
    assert out.metrics["rollout/groups_filter_kept"] == 128
    assert out.metrics["rollout/groups_filter_rejected"] == 1
    assert out.metrics["rollout/groups_selected"] == 128
    assert out.metrics["rollout/pending_groups_at_abort"] == 63
    assert out.metrics["rollout/pending_trajectories_at_abort"] == 1008
    assert out.metrics["rollout/submission_waves"] == 3
    assert out.metrics["rollout/refill_waves"] == 1
    assert out.metrics["rollout/queued_trajectories_peak"] == 3056


@pytest.mark.parametrize(
    "rollout_env",
    [
        pytest.param(
            integration_env_config(
                [
                    "--rollout-batch-size",
                    "1",
                    "--over-sampling-batch-size",
                    "1",
                    "--dynamic-sampling-filter-path",
                    "test:filter_by_reward",
                ],
                data_rows=[
                    {"input": "failed", "label": "error"},
                    {"input": "accepted", "label": "valid"},
                ],
            ),
            id="task_failure_refill",
        )
    ],
    indirect=["rollout_env"],
)
def test_task_failure_starts_a_refill_wave(rollout_env, monkeypatch):
    _install_controlled_completion(monkeypatch)

    with function_registry.temporary("test:filter_by_reward", filter_by_reward):
        out = load_and_call_train(rollout_env.args, rollout_env.data_source)

    assert len(out.samples) == 1
    assert out.metrics["rollout/groups_submitted"] == 2
    assert out.metrics["rollout/groups_filter_kept"] == 1
    assert out.metrics["rollout/groups_failed"] == 1
    assert out.metrics["rollout/submission_waves"] == 2
    assert out.metrics["rollout/refill_waves"] == 1


@pytest.mark.parametrize("value", [0, -1])
def test_over_sampling_batch_size_must_be_positive(value):
    with pytest.raises(AssertionError, match="should be positive"):
        _validate_over_sampling_batch_size(value)


def test_over_sampling_batch_size_can_be_smaller_than_target():
    _validate_over_sampling_batch_size(1)
