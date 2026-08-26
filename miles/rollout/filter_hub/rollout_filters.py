from miles.rollout.filter_hub.dynamic_sampling_filters import _flatten_samples
from miles.utils.types import Sample

__all__ = [
    "mask_truncated",
    "mask_truncated_and_evaluation_failed",
    "mask_truncated_and_llm_judge_failed",
]


def mask_truncated(args, samples: list[Sample]) -> None:
    """Mask truncated samples so they are excluded from training.

    Usage in config:
        --rollout-sample-filter-path miles.rollout.filter_hub.rollout_filters.mask_truncated
    """
    for sample in _flatten_samples(samples):
        if sample.status == Sample.Status.TRUNCATED:
            sample.remove_sample = True


def mask_truncated_and_evaluation_failed(args, samples: list[Sample]) -> None:
    """Mask truncated samples and samples whose evaluator was unhealthy.

    Usage in config:
        --rollout-sample-filter-path miles.rollout.filter_hub.rollout_filters.mask_truncated_and_evaluation_failed

    ``evaluation_failed`` is backend-neutral: it covers an unavailable CPU
    scorer, an LLM judge failure, a malformed remote verdict, or any other
    failure to obtain a trustworthy score.  The legacy ``llm_judge_failed``
    flag remains accepted while older reward functions migrate.
    """
    for sample in _flatten_samples(samples):
        metadata = sample.metadata or {}
        if (
            sample.status == Sample.Status.TRUNCATED
            or metadata.get("evaluation_failed")
            or metadata.get("llm_judge_failed")
        ):
            sample.remove_sample = True


def mask_truncated_and_llm_judge_failed(args, samples: list[Sample]) -> None:
    """Backward-compatible alias for the backend-neutral evaluation filter."""

    mask_truncated_and_evaluation_failed(args, samples)
