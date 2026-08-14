import pytest

from miles.rollout.generate_hub.agentic_tool_call import _mark_limits_exceeded_truncated
from miles.utils.types import Sample


@pytest.mark.parametrize("as_list", [False, True], ids=["merged", "multi-sample"])
def test_limits_exceeded_marks_only_final_sample_truncated(as_list: bool) -> None:
    samples = [
        Sample(status=Sample.Status.COMPLETED),
        Sample(status=Sample.Status.COMPLETED),
    ]
    value = samples if as_list else samples[-1]

    _mark_limits_exceeded_truncated(value, {"exit_status": "LimitsExceeded"})

    assert samples[-1].status == Sample.Status.TRUNCATED
    if as_list:
        assert samples[0].status == Sample.Status.COMPLETED


@pytest.mark.parametrize("agent_metadata", [None, {}, {"exit_status": "Submitted"}])
def test_other_exit_statuses_remain_completed(agent_metadata: dict | None) -> None:
    sample = Sample(status=Sample.Status.COMPLETED)

    _mark_limits_exceeded_truncated(sample, agent_metadata)

    assert sample.status == Sample.Status.COMPLETED
