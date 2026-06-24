from miles.rollout.filter_hub.dynamic_sampling_filters import _flatten_samples
from miles.utils.types import Sample

__all__ = ["mask_truncated", "mask_truncated_and_llm_judge_failed"]


def mask_truncated(args, samples: list[Sample]) -> None:
    """Mask truncated samples so they are excluded from training.

    Usage in config:
        --rollout-sample-filter-path miles.rollout.filter_hub.rollout_filters.mask_truncated
    """
    for sample in _flatten_samples(samples):
        if sample.status == Sample.Status.TRUNCATED:
            sample.remove_sample = True


def mask_truncated_and_llm_judge_failed(args, samples: list[Sample]) -> None:
    """Mask truncated samples and samples where the LLM judge failed so they are excluded from training.

    Usage in config:
        --rollout-sample-filter-path miles.rollout.filter_hub.rollout_filters.mask_truncated_and_llm_judge_failed

    Requires the reward function to return ``llm_judge_failed: True`` in its score dict when judge
    calls fail (stored in ``sample.metadata["llm_judge_failed"]`` by async_rm).
    """
    for sample in _flatten_samples(samples):
        if sample.status == Sample.Status.TRUNCATED or sample.metadata.get("llm_judge_failed"):
            sample.remove_sample = True
