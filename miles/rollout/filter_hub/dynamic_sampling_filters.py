import torch

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_no_aborted", "drop_zero_std_groups_and_extreme_pass_rate", "reject_truncated_or_extreme_pass_rate"]

_DEFAULT_MIN_MEAN_REWARD = 0.1
_DEFAULT_MAX_MEAN_REWARD = 0.8

def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    # Aborted samples never get a reward computed (see generate_and_rm in
    # sglang_rollout.py). Reject the group when any reward is None, matching
    # the check_no_aborted convention — a group with a missing reward can't
    # yield a reliable std and shouldn't contribute gradient signal.
    if any(r is None for r in rewards):
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    assert len(rewards) >= 2, f"expected at least 2 samples per group, got {len(rewards)}"
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


def drop_zero_std_groups_and_extreme_pass_rate(args, samples: list[object], **kwargs) -> DynamicFilterOutput:
    '''
    Do two things:
    1) Filter groups with near-zero reward std.
    2) Drop groups with extreme mean rewards. For 0/1 rewards, mean reward is equivalent to pass rate.
    '''
    flat_samples = list(_flatten_samples(samples))

    if not flat_samples:
        return DynamicFilterOutput(keep=False, reason="group_has_no_samples")

    rewards = [sample.get_reward_value(args) for sample in flat_samples]
    if any(reward is None for reward in rewards):
        return DynamicFilterOutput(keep=False, reason="group_has_missing_reward")

    # this assert should never fail unless user sets `--n-samples-per-prompt` to 1 which is incorrect for grpo
    assert len(rewards) >= 2, f"expected at least 2 samples per group, got {len(rewards)}"
    reward_tensor = torch.tensor(rewards, dtype=torch.float64)
    mean_reward = reward_tensor.mean().item()
    std = reward_tensor.std().item()

    min_std = getattr(args, "dynamic_sampling_min_reward_std", 1e-3)
    if std < min_std:
        return DynamicFilterOutput(keep=False, reason=f"near_zero_std_{min_std:g}")

    max_mean_reward = getattr(args, "dynamic_sampling_max_mean_reward", _DEFAULT_MAX_MEAN_REWARD)
    min_mean_reward = getattr(args, "dynamic_sampling_min_mean_reward", _DEFAULT_MIN_MEAN_REWARD)
    if mean_reward > max_mean_reward:
        return DynamicFilterOutput(keep=False, reason="mean_reward_too_high")
    if mean_reward < min_mean_reward:
        return DynamicFilterOutput(keep=False, reason="mean_reward_too_low")

    return DynamicFilterOutput(keep=True)


def reject_truncated_or_extreme_pass_rate(args, samples: list[object], **kwargs) -> DynamicFilterOutput:
    flat_samples = list(_flatten_samples(samples))

    if any(getattr(getattr(sample, "status", None), "value", getattr(sample, "status", None)) == "truncated" for sample in flat_samples):
        return DynamicFilterOutput(keep=False, reason="group_has_truncated")

    return drop_zero_std_groups_and_extreme_pass_rate(args, samples, **kwargs)
