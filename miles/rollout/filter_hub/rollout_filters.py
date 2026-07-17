from miles.rollout._agentic_outcomes import _Outcome, classify_exit_status, masks_sample
from miles.rollout.filter_hub.dynamic_sampling_filters import _flatten_samples
from miles.utils.types import Sample

__all__ = [
    "mask_nontrainable_outcomes",
    "mask_truncated",
    "mask_truncated_and_llm_judge_failed",
    "mask_token_truncated",
]


def mask_nontrainable_outcomes(args, samples: list[Sample]) -> None:
    """Mask every non-trainable agentic outcome as a final safety net.

    Group rejection runs before this filter.  Masking partial/fatal failures
    here prevents accidental gradient if a caller omits the dynamic filter.
    """
    for sample in _flatten_samples(samples):
        outcome = classify_exit_status((sample.metadata or {}).get("exit_status", ""))
        if sample.status == Sample.Status.TRUNCATED or masks_sample(outcome):
            sample.remove_sample = True


def mask_token_truncated(args, samples: list[Sample]) -> None:
    """Mask token/context-truncated samples (zero gradient, reward kept in baseline).

    Keyed on Harbor ``exit_status`` for compatibility with legacy samples whose
    stored status predates the shared resolver; also covers TRUNCATED status.

        --rollout-sample-filter-path miles.rollout.filter_hub.rollout_filters.mask_token_truncated
    """
    for sample in _flatten_samples(samples):
        outcome = classify_exit_status((sample.metadata or {}).get("exit_status", ""))
        if outcome is _Outcome.TOKEN_TRUNCATION or sample.status == Sample.Status.TRUNCATED:
            sample.remove_sample = True


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
