import pytest

from miles.rollout.inference_rollout.inference_rollout_common import (
    raise_if_generation_collapsed,
)
from miles.utils.types import Sample


def _sample(status: Sample.Status) -> Sample:
    return Sample(status=status)


@pytest.mark.parametrize(
    "groups",
    [
        [[_sample(Sample.Status.ABORTED)]],
        [
            [_sample(Sample.Status.ABORTED)],
            [_sample(Sample.Status.ABORTED)],
        ],
        [[[_sample(Sample.Status.COMPLETED)], [_sample(Sample.Status.ABORTED)]]],
    ],
)
def test_all_aborted_returned_groups_raise(groups):
    with pytest.raises(RuntimeError, match="generation collapsed"):
        raise_if_generation_collapsed(len(groups), groups)


def test_all_task_exceptions_raise_when_nothing_returned():
    with pytest.raises(RuntimeError, match="missing_groups=3"):
        raise_if_generation_collapsed(3, [])


def test_already_exhausted_source_without_submissions_is_normal():
    raise_if_generation_collapsed(0, [])


def test_healthy_groups_may_all_be_dropped_by_later_filters():
    raise_if_generation_collapsed(1, [[_sample(Sample.Status.COMPLETED)]])


def test_one_healthy_group_prevents_mixed_batch_from_being_classified_as_collapse():
    raise_if_generation_collapsed(
        2,
        [
            [_sample(Sample.Status.ABORTED)],
            [_sample(Sample.Status.COMPLETED)],
        ],
    )


def test_empty_returned_group_does_not_count_as_healthy():
    with pytest.raises(RuntimeError, match="empty_groups=1"):
        raise_if_generation_collapsed(1, [[]])


def test_negative_submission_count_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        raise_if_generation_collapsed(-1, [])
