"""ESS-guided learning-rate scaling (VCPO-style; arXiv:2602.17616).

In fully-async / off-policy RL a few stale trajectories can dominate the update
(heavy-tailed importance weights), collapsing the effective sample size (ESS).
When that happens we shrink the effective LR by ``sqrt(ESS / B)``.

Per trajectory ``i`` (SEQUENCE-level, length-normalized = geometric-mean IS):

    m_i      = mean_t (train_logp_t - rollout_logp_t)  over response (mask==1) tokens
    w_i      = exp(m_i)                                  # geom-mean IS weight
    ESS      = (sum_i w_i)^2 / (sum_i w_i^2) ; rho = ESS / B  in (0, 1]
    lr_scale = sqrt(rho), clamped to [args.ess_lr_floor, 1.0]

``ess_lr_compute`` runs once per ROLLOUT over the full batch (from
``compute_advantages_and_returns``, before the rollout's optimizer-step loop) and
stashes the scale in ``_ESS_LR_STATE``; the same scale is reused for every
optimizer step in that rollout. ``megatron_utils/model.py`` broadcasts it across
the PP group and applies it around ``optimizer.step()`` (only for the policy/actor
loss, not the critic). Everything is a no-op (bit-exact legacy) unless
``args.use_ess_lr`` is set.
"""

from argparse import Namespace

import torch

from miles.utils.types import RolloutBatch

from .cp_utils import get_logits_and_tokens_offset_with_cp
from .parallel import ParallelState

# Step-level scale/rho, written by ess_lr_compute() on the last PP stage and
# synced across the PP group at optimizer-step time. Read by model.py.
_ESS_LR_STATE: dict[str, float] = {"scale": 1.0, "rho_ess": 1.0}


def ess_lr_scale_from_sums(
    sum_w: torch.Tensor, sum_w2: torch.Tensor, batch: torch.Tensor, floor: float
) -> tuple[float, float]:
    """Map the (already DP-reduced) ESS sums to ``(lr_scale, rho_ess)``. Pure / testable."""
    rho = (sum_w * sum_w) / (batch * sum_w2 + 1e-8)  # ESS / B in (0, 1]
    scale = float(torch.sqrt(rho.clamp_min(1e-8)).clamp(min=floor, max=1.0).item())
    return scale, float(rho)


def ess_lr_compute(args: Namespace, parallel_state: ParallelState, rollout_data: RolloutBatch) -> None:
    """Compute the ESS-guided LR scale and stash it in ``_ESS_LR_STATE``.

    No-op unless ``args.use_ess_lr``. Only the last pipeline stage holds the
    train/rollout log-probs, so this early-returns elsewhere; the computed scale
    is broadcast across the PP group at optimizer-step time (see model.py).
    """
    if not getattr(args, "use_ess_lr", False):
        return
    # Reset up front so a missing-input early-return (e.g. --use-rollout-logprobs
    # where train log-probs are not recomputed, the critic path, or a non-last PP
    # stage) yields a no-op scale=1 for THIS rollout instead of silently reusing
    # the previous rollout's value.
    _ESS_LR_STATE["scale"] = 1.0
    _ESS_LR_STATE["rho_ess"] = 1.0
    train_log_probs = rollout_data.get("log_probs")
    rollout_log_probs = rollout_data.get("rollout_log_probs")
    loss_masks = rollout_data.get("loss_masks")
    if not train_log_probs or not rollout_log_probs or not loss_masks:
        return  # not last pp stage, or rollout log-probs unavailable
    total_lengths = rollout_data.get("total_lengths")
    response_lengths = rollout_data.get("response_lengths")
    max_seq_lens = rollout_data.get("max_seq_lens", None)
    cp_size = parallel_state.cp.size
    device = train_log_probs[0].device

    local_num: list[torch.Tensor] = []  # CP-local-chunk sum of (train - rollout) log-prob over masked tokens
    full_cnt: list[torch.Tensor] = []  # full-trajectory masked token count
    for i in range(len(train_log_probs)):
        d = train_log_probs[i].float() - rollout_log_probs[i].float()  # CP-local per-token log-IS
        full_mask = loss_masks[i].to(device=device).float()  # FULL response mask
        if cp_size == 1:
            local_mask = full_mask
        else:
            # Mirror the CP mask-chunking used by advantage whitening in loss.py so
            # the local mask aligns with the CP-local log-prob chunk `d`.
            prompt_len = int(total_lengths[i]) - int(response_lengths[i])
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                int(total_lengths[i]), int(response_lengths[i]), args.qkv_format, max_seq_len
            )
            (s0, e0), (s1, e1) = token_offsets[0], token_offsets[1]
            res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
            res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)
            parts = []
            if res_e0 > res_s0:
                parts.append(full_mask[res_s0:res_e0])
            if res_e1 > res_s1:
                parts.append(full_mask[res_s1:res_e1])
            local_mask = torch.cat(parts) if parts else torch.zeros(0, device=device, dtype=full_mask.dtype)
        n = min(d.numel(), local_mask.numel())
        local_num.append((d[:n] * local_mask[:n]).sum())
        full_cnt.append(full_mask.sum())

    if not local_num:
        return
    local_num_t = torch.stack(local_num).float()  # [B_local]
    full_cnt_t = torch.stack(full_cnt).float().clamp_min(1.0)
    if cp_size > 1:
        torch.distributed.all_reduce(local_num_t, group=parallel_state.cp.group)  # -> full-trajectory numerator
    m = local_num_t / full_cnt_t  # per-traj mean log-IS (geom-mean exponent)
    w = torch.exp(m.clamp(min=-30.0, max=30.0))  # geom-mean IS weight (clamp guards exp overflow)
    stat = torch.stack([w.sum(), (w * w).sum(), torch.tensor(float(w.numel()), device=device)])
    if parallel_state.intra_dp.size > 1:
        # Skip when dp_size == 1: the local sums are already global, and
        # intra_dp.group may be None (-> all_reduce would fall back to WORLD and
        # deadlock, since only the last PP stage reaches this line).
        torch.distributed.all_reduce(stat, group=parallel_state.intra_dp.group)  # ESS sums over DP
    scale, rho = ess_lr_scale_from_sums(stat[0], stat[1], stat[2], float(args.ess_lr_floor))
    _ESS_LR_STATE["scale"] = scale
    _ESS_LR_STATE["rho_ess"] = rho
