import torch

from miles.rollout._agentic_outcomes import classify_exit_status, rejects_group
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

__all__ = [
    "check_reward_nonzero_std",
    "check_no_aborted",
    "check_no_invalid_outcomes",
    "check_no_invalid_outcomes_then_nonzero_std",
    "check_no_infra_failures",
    "drop_zero_std_groups_and_extreme_pass_rate",
    "drop_truncated_or_extreme_pass_rate",
]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    # Truncated trajectories receive no gradient and are excluded from the
    # GRPO baseline, so they must not create artificial reward variance here.
    baseline_samples = [
        sample
        for sample in _flatten_samples(samples)
        if sample.status != Sample.Status.TRUNCATED
    ]
    if len(baseline_samples) < 2:
        return DynamicFilterOutput(
            keep=False,
            reason="group_has_insufficient_complete_samples",
        )

    rewards = [sample.get_reward_value(args) for sample in baseline_samples]
    # Aborted samples never get a reward computed (see generate_and_rm in
    # sglang_rollout.py). Reject the group when any reward is None, matching
    # the check_no_aborted convention — a group with a missing reward can't
    # yield a reliable std and shouldn't contribute gradient signal.
    if any(r is None for r in rewards):
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-8
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def _flatten_samples(samples):
    """Flatten samples that may contain nested lists (from --generate-multi-samples)."""
    for s in samples:
        if isinstance(s, list):
            yield from s
        else:
            yield s


def check_no_aborted(args, samples: list[Sample], **kwargs):
    """Reject entire group if any sample was aborted (e.g. env timeout, Docker crash)."""
    if any(s.status == Sample.Status.ABORTED for s in _flatten_samples(samples)):
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)


def check_no_invalid_outcomes(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Reject groups containing unusable statuses or non-comparable exits."""
    flat_samples = list(_flatten_samples(samples))
    for sample in flat_samples:
        if sample.status == Sample.Status.ABORTED:
            return DynamicFilterOutput(keep=False, reason="group_has_aborted")
        if sample.status == Sample.Status.FAILED:
            return DynamicFilterOutput(keep=False, reason="group_has_failed")
        if sample.status == Sample.Status.PENDING:
            return DynamicFilterOutput(keep=False, reason="group_has_pending")
        exit_status = (sample.metadata or {}).get("exit_status", "")
        if rejects_group(classify_exit_status(exit_status)):
            return DynamicFilterOutput(keep=False, reason=f"group_has_{exit_status}")
    return DynamicFilterOutput(keep=True)


def check_no_invalid_outcomes_then_nonzero_std(
    args, samples: list[Sample], **kwargs
) -> DynamicFilterOutput:
    """Reject invalid outcomes first, then groups with zero reward variance."""
    flat_samples = list(_flatten_samples(samples))
    outcome = check_no_invalid_outcomes(args, flat_samples, **kwargs)
    if not outcome.keep:
        return outcome
    return check_reward_nonzero_std(args, flat_samples, **kwargs)


def check_no_infra_failures(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Compatibility wrapper for the former infra-plus-zero-variance filter."""
    return check_no_invalid_outcomes_then_nonzero_std(args, samples, **kwargs)


def drop_zero_std_groups_and_extreme_pass_rate(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Filter groups with near-zero reward std or extreme mean rewards.
    For 0/1 rewards, mean reward is equivalent to pass rate --- so this function can be used to filter
    "easy" or "hard" groups

    Usage in config:
        --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.drop_zero_std_groups_and_extreme_pass_rate
        --dynamic-sampling-min-reward-std   (required, default: 1e-3)
        --dynamic-sampling-min-mean-reward  (required, default: 0.1)
        --dynamic-sampling-max-mean-reward  (required, default: 0.8)
    """
    flat_samples = list(_flatten_samples(samples))

    if not flat_samples:
        return DynamicFilterOutput(keep=False, reason="group_has_no_samples")

    rewards = [sample.get_reward_value(args) for sample in flat_samples]
    if any(reward is None for reward in rewards):
        return DynamicFilterOutput(keep=False, reason="group_has_missing_reward")

    if len(rewards) < 2:
        raise ValueError(
            f"expected at least 2 samples per group to check for standard deviation but got {len(rewards)} — set --n-samples-per-prompt >= 2 for GRPO"
        )
    reward_tensor = torch.tensor(rewards, dtype=torch.float64)
    mean_reward = reward_tensor.mean().item()
    std = reward_tensor.std().item()

    # get arguments for min_std, max_mean_reward, and min_mean_reward, see default values in `miles/utils/arguments.py` for reference
    min_std = getattr(args, "dynamic_sampling_min_reward_std", None)
    max_mean_reward = getattr(args, "dynamic_sampling_max_mean_reward", None)
    min_mean_reward = getattr(args, "dynamic_sampling_min_mean_reward", None)

    # check for none values
    if min_std is None:
        raise ValueError(
            "--dynamic-sampling-min-reward-std is required when using drop_zero_std_groups_and_extreme_pass_rate"
        )
    if max_mean_reward is None:
        raise ValueError(
            "--dynamic-sampling-max-mean-reward is required when using drop_zero_std_groups_and_extreme_pass_rate"
        )
    if min_mean_reward is None:
        raise ValueError(
            "--dynamic-sampling-min-mean-reward is required when using drop_zero_std_groups_and_extreme_pass_rate"
        )

    if std < min_std:
        return DynamicFilterOutput(keep=False, reason=f"near_zero_std_{min_std:g}")
    if mean_reward > max_mean_reward:
        return DynamicFilterOutput(keep=False, reason="mean_reward_too_high")
    if mean_reward < min_mean_reward:
        return DynamicFilterOutput(keep=False, reason="mean_reward_too_low")

    return DynamicFilterOutput(keep=True)


def drop_truncated_or_extreme_pass_rate(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Reject groups containing any truncated sample, then apply `drop_zero_std_groups_and_extreme_pass_rate` filter.

    Usage in config:
        --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.drop_truncated_or_extreme_pass_rate
        --dynamic-sampling-min-reward-std   (required, default: 1e-3)
        --dynamic-sampling-min-mean-reward  (required, default: 0.1)
        --dynamic-sampling-max-mean-reward  (required, default: 0.8)
    """
    flat_samples = list(_flatten_samples(samples))

    if any(sample.status == Sample.Status.TRUNCATED for sample in flat_samples):
        return DynamicFilterOutput(keep=False, reason="group_has_truncated")

    return drop_zero_std_groups_and_extreme_pass_rate(args, samples, **kwargs)
