"""Hybrid PPO baselines: combine a leave-one-out group mean with the critic.

`--ppo-baseline` picks the baseline b_t subtracted from the per-token return
(advantage A_t = R_t - b_t). The critic is always trained on the same returns;
only the advantage changes.

    value  b_t = V_t                      (standard PPO, GAE with lambda=1)
    group  b_t = g                        (GRPO-style leave-one-out group mean)
    max    b_t = g + clamp(V_t - g, 0, cap)   optimistic
    min    b_t = g - clamp(g - V_t, 0, cap)   pessimistic
    delta  b_t = g + clamp(V_t - V_0, -cap, cap)

g is the mean reward of the other valid samples of the same prompt, so it does
not depend on the sample's own action. Samples without a valid g (no other
valid group member) fall back to b_t = V_t.
"""

from collections import defaultdict

import torch

PPO_BASELINE_MODES = ("value", "group", "max", "min", "delta")


def compute_group_loo_baselines(
    group_ids: list, rewards: list[float], valid: list[bool]
) -> tuple[list[float], list[int]]:
    """Leave-one-out mean reward over the other valid members of each group.

    Returns (baselines, valid_flags); a sample whose group has no other valid
    member gets baseline 0.0 and flag 0.
    """
    assert len(group_ids) == len(rewards) == len(valid)
    total = defaultdict(float)
    count = defaultdict(int)
    for g, r, ok in zip(group_ids, rewards, valid, strict=True):
        if ok:
            total[g] += float(r)
            count[g] += 1
    baselines, flags = [], []
    for g, r, ok in zip(group_ids, rewards, valid, strict=True):
        s, n = total[g], count[g]
        if ok:
            s, n = s - float(r), n - 1
        if n > 0:
            baselines.append(s / n)
            flags.append(1)
        else:
            baselines.append(0.0)
            flags.append(0)
    return baselines, flags


def hybrid_baseline(
    mode: str, values: torch.Tensor, group_baseline: float, cap: float | None = None
) -> torch.Tensor:
    """Per-token baseline for one full (un-sliced) response."""
    if mode == "value":
        return values
    g = torch.full_like(values, float(group_baseline))
    if mode == "group":
        return g
    hi = float("inf") if cap is None else float(cap)
    if mode == "max":
        return g + (values - g).clamp(min=0.0, max=hi)
    if mode == "min":
        return g - (g - values).clamp(min=0.0, max=hi)
    if mode == "delta":
        shift = values - values[:1] if values.numel() else values
        return g + shift.clamp(min=-hi, max=hi)
    raise ValueError(f"unknown ppo baseline mode {mode!r}; expected one of {PPO_BASELINE_MODES}")


def hybrid_advantages(
    mode: str,
    values: torch.Tensor,
    returns: torch.Tensor,
    loss_mask: torch.Tensor,
    group_baseline: float,
    group_valid: bool,
    cap: float | None = None,
) -> tuple[torch.Tensor, float, float]:
    """A_t = returns_t - b_t for one full response.

    Returns (advantages, mean baseline, fraction of tokens where the critic
    moved the baseline away from g), both over loss-masked tokens.
    """
    if not group_valid:
        b = values
        moved = torch.ones_like(values, dtype=torch.bool)
    else:
        b = hybrid_baseline(mode, values, group_baseline, cap)
        moved = (b - group_baseline).abs() > 1e-6
    m = loss_mask.to(values.dtype)
    n = m.sum().clamp(min=1.0)
    return returns - b, float((b * m).sum() / n), float((moved.to(values.dtype) * m).sum() / n)
