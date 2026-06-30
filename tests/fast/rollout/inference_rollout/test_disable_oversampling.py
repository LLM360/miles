"""A fixed submission budget survives filtering and sample-level wakeups."""

import asyncio

import pytest
from tests.fast.rollout.inference_rollout.test_sample_completion_backfill import GROUP_SIZE, Harness, make_args


@pytest.mark.parametrize("granularity", ["group", "sample"])
@pytest.mark.parametrize("all_filtered", [False, True])
async def test_disabled_refill_caps_total_submissions(monkeypatch, granularity, all_filtered):
    harness = Harness(
        monkeypatch,
        make_args(disable_oversampling=True, rollout_batch_size=2, rollout_submission_granularity=granularity),
    )
    task = harness.run()
    await asyncio.sleep(0)
    assert harness.submitted_group_indices == [1, 2]
    for i, group in enumerate(harness.submitted_groups):
        if all_filtered or i == 0:
            group[0].reward = None
        if granularity == "sample":
            harness.finish_samples(i, GROUP_SIZE)
    # A sample wakeup at the submission cap must not spin or refill.
    await asyncio.sleep(0.01)
    assert harness.submitted_group_indices == [1, 2]
    harness.finish_group(0)
    harness.finish_group(1)
    output, _ = await asyncio.wait_for(task, 2)
    assert len(output.samples) == (0 if all_filtered else 1)
    assert harness.submitted_group_indices == [1, 2]
    assert harness.state.reset_count == 1


async def test_empty_source_returns_short_disabled_batch(monkeypatch):
    harness = Harness(monkeypatch, make_args(disable_oversampling=True))
    harness.data_source = lambda n: []
    output, _ = await asyncio.wait_for(harness.run(), 2)
    assert output.samples == []
    assert harness.submitted_group_indices == []


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("tail_groups", [0, 1])
async def test_tail_cut_requires_disabled_refill_and_positive_threshold(monkeypatch, disabled, tail_groups):
    harness = Harness(monkeypatch, make_args(disable_oversampling=disabled, tail_cancel_groups=tail_groups))
    task = harness.run()
    await asyncio.sleep(0)
    harness.finish_group(0)
    await asyncio.sleep(0.01)
    should_cut = disabled and tail_groups == 1
    if not should_cut:
        assert not task.done()
        harness.finish_group(1)
    output, _ = await asyncio.wait_for(task, 2)
    assert len(output.samples) == (1 if should_cut else 2)


@pytest.mark.parametrize("granularity", ["group", "sample"])
@pytest.mark.parametrize("wave_rollouts", [0, 6])
async def test_rolling_start_uses_whole_groups_and_spaces_waves(monkeypatch, granularity, wave_rollouts):
    harness = Harness(
        monkeypatch,
        make_args(
            disable_oversampling=True,
            rollout_batch_size=3,
            rollout_submission_granularity=granularity,
            rolling_start_size=wave_rollouts,
            rolling_start_interval=0.0123,
        ),
    )
    sleeps, requests = [], []
    original_sleep, original_source = asyncio.sleep, harness.data_source

    async def sleep(delay):
        sleeps.append(delay)
        await original_sleep(0)

    def source(n):
        requests.append(n)
        return original_source(n)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    harness.data_source = source
    task = harness.run()
    for _ in range(10):
        await original_sleep(0)
        if len(harness.submitted_groups) == 3:
            break
    assert requests == ([1, 1, 1] if wave_rollouts else [3])
    assert sleeps == ([0.0123, 0.0123] if wave_rollouts else [])
    for i in range(3):
        harness.finish_group(i)
    output, _ = await asyncio.wait_for(task, 2)
    assert len(output.samples) == 3
