from miles.rollout.filter_hub.rollout_filters import (
    mask_truncated_and_evaluation_failed,
    mask_truncated_and_llm_judge_failed,
)
from miles.utils.types import Sample


def _sample(*, status=Sample.Status.COMPLETED, metadata=None):
    return Sample(status=status, metadata=metadata or {})


def test_backend_neutral_filter_masks_every_evaluation_failure_kind():
    healthy = _sample()
    generic_failure = _sample(metadata={"evaluation_failed": True})
    legacy_failure = _sample(metadata={"llm_judge_failed": True})
    truncated = _sample(status=Sample.Status.TRUNCATED)

    mask_truncated_and_evaluation_failed(
        None,
        [healthy, generic_failure, legacy_failure, truncated],
    )

    assert healthy.remove_sample is False
    assert generic_failure.remove_sample is True
    assert legacy_failure.remove_sample is True
    assert truncated.remove_sample is True


def test_legacy_filter_name_delegates_to_generic_contract():
    sample = _sample(metadata={"evaluation_failed": True})

    mask_truncated_and_llm_judge_failed(None, [sample])

    assert sample.remove_sample is True
