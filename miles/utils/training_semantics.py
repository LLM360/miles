import math
from typing import Any

import torch

from miles.utils.types import Sample


def _reward_value(args: Any, sample: Sample) -> Any:
    return sample.get_reward_value(args)


def _finite_reward(value: Any, sample_index: int) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"sample {sample_index} has non-finite reward for training: {value!r}")
    return float(value)


def _group_ranges(args: Any, sample_count: int) -> list[range]:
    samples_per_prompt = args.n_samples_per_prompt
    if sample_count == samples_per_prompt * args.rollout_batch_size:
        return [range(start, start + samples_per_prompt) for start in range(0, sample_count, samples_per_prompt)]
    return [range(0, sample_count)]


def post_process_rewards_excluding_removed(args: Any, samples: list[Sample]) -> tuple[list[Any], list[float]]:
    raw_rewards = [_reward_value(args, sample) for sample in samples]
    remove_samples = [sample.remove_sample for sample in samples]

    processed_rewards = [0.0] * len(samples)
    if not (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        for i, raw_reward in enumerate(raw_rewards):
            if not remove_samples[i]:
                processed_rewards[i] = _finite_reward(raw_reward, i)
        return raw_rewards, processed_rewards

    for group in _group_ranges(args, len(samples)):
        kept_indices = [i for i in group if not remove_samples[i]]
        if not kept_indices:
            continue

        kept_rewards = torch.tensor(
            [_finite_reward(raw_rewards[i], i) for i in kept_indices],
            dtype=torch.float,
        )
        kept_rewards = kept_rewards - kept_rewards.mean()

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            if kept_rewards.numel() > 1:
                std = kept_rewards.std()
            else:
                std = torch.tensor(0.0, dtype=kept_rewards.dtype)
            kept_rewards = kept_rewards / (std + 1e-6)

        for i, reward in zip(kept_indices, kept_rewards.tolist(), strict=True):
            processed_rewards[i] = reward

    return raw_rewards, processed_rewards


def validate_loss_masks_for_removed_samples(
    loss_masks: list[Any],
    response_lengths: list[int],
    remove_samples: list[bool] | None,
) -> None:
    if remove_samples is None:
        remove_samples = [False] * len(loss_masks)
    if len(remove_samples) != len(loss_masks):
        raise ValueError(f"remove_samples length {len(remove_samples)} != loss_masks length {len(loss_masks)}")

    for i, (loss_mask, response_length, remove_sample) in enumerate(
        zip(loss_masks, response_lengths, remove_samples, strict=True)
    ):
        mask_sum = loss_mask.sum().item() if torch.is_tensor(loss_mask) else sum(loss_mask)
        if mask_sum <= 0 and not remove_sample:
            raise ValueError(
                f"loss_masks[{i}] has no active tokens, sum={mask_sum}, "
                f"response_len={response_length}, remove_samples[{i}] is false"
            )
