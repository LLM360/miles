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
