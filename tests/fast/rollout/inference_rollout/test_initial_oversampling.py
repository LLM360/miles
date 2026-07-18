import asyncio

import pytest
from tests.fast.rollout.inference_rollout.test_sample_completion_backfill import GROUP_SIZE, Harness, make_args


@pytest.mark.parametrize("granularity", ["group", "sample"])
async def test_initial_buffer_returns_to_normal_refill_budget(monkeypatch, granularity):
    harness = Harness(
        monkeypatch, make_args(initial_oversampling_groups=1, rollout_submission_granularity=granularity)
    )
    task = harness.run()
    await asyncio.sleep(0)
    assert harness.submitted_group_indices == [1, 2, 3]
    for i in range(2):
        harness.submitted_groups[i][0].reward = None
        if granularity == "sample":
            harness.finish_samples(i, GROUP_SIZE)
        harness.finish_group(i)
    for _ in range(50):
        await asyncio.sleep(0.001)
        if len(harness.submitted_groups) >= 4:
            break
    assert harness.submitted_group_indices == [1, 2, 3, 4]
    for i in [2, 3]:
        if granularity == "sample":
            harness.finish_samples(i, GROUP_SIZE)
        harness.finish_group(i)
    output, _ = await asyncio.wait_for(task, 2)
    assert len(output.samples) == 2
    assert output.metrics["rollout/groups_submitted"] == 4
    assert output.metrics["rollout/refill_waves"] == 1
