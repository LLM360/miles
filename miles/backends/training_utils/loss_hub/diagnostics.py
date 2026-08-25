"""Stable loss diagnostics, observed before upstream finite-value handling."""

import logging
import sys
from argparse import Namespace

import torch

from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


def _nan_dbg_flush_logs() -> None:
    """Flush logger/stdout/stderr so crash diagnostics survive abrupt failures."""
    for handler in logging.getLogger().handlers + logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass


def _nan_dbg_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return -1


def _nan_dbg_scalar(value: float | int, device: torch.device) -> torch.Tensor:
    return torch.tensor(value, device=device, dtype=torch.float32)


def _nan_dbg_finite_stats(x: torch.Tensor) -> tuple[float, float, float, int]:
    x = x.detach()
    finite = torch.isfinite(x)
    bad = int((~finite).sum().item())
    if x.numel() == 0 or not bool(finite.any().item()):
        return float("nan"), float("nan"), float("nan"), bad
    xf = x[finite].float()
    return float(xf.min().item()), float(xf.max().item()), float(xf.abs().max().item()), bad


def _nan_dbg_batch_stats(batch: RolloutBatch) -> tuple[int, int, int, int]:
    response_len_max = max((int(x) for x in batch.get("response_lengths", [])), default=0)
    response_len_sum = sum(int(x) for x in batch.get("response_lengths", []))
    loss_tokens = [int(m.sum().item()) for m in batch.get("loss_masks", [])]
    loss_tokens_max = max(loss_tokens, default=0)
    loss_tokens_sum = sum(loss_tokens)
    return response_len_max, response_len_sum, loss_tokens_max, loss_tokens_sum


def _nan_dbg_warn_bad_tensor(name: str, x: torch.Tensor, *, batch: RolloutBatch, extra: str = "") -> None:
    x = x.detach()
    bad = int((~torch.isfinite(x)).sum().item())
    if bad == 0:
        return
    response_len_max, response_len_sum, loss_tokens_max, loss_tokens_sum = _nan_dbg_batch_stats(batch)
    logger.error(
        "NANDBG_BAD_TENSOR "
        f"rank={_nan_dbg_rank()} "
        f"name={name} "
        f"shape={tuple(x.shape)} "
        f"dtype={x.dtype} "
        f"nonfinite={bad} "
        f"response_len_max={response_len_max} "
        f"response_len_sum={response_len_sum} "
        f"loss_tokens_max={loss_tokens_max} "
        f"loss_tokens_sum={loss_tokens_sum} "
        f"{extra}"
    )
    _nan_dbg_flush_logs()


def _nan_dbg_warn_long_batch(args: Namespace, batch: RolloutBatch) -> None:
    response_len_max, response_len_sum, loss_tokens_max, loss_tokens_sum = _nan_dbg_batch_stats(batch)
    if response_len_max <= 65536 and loss_tokens_max <= 65536:
        return
    logger.warning(
        "NANDBG_LONG_BATCH "
        f"rank={_nan_dbg_rank()} "
        f"num_samples={len(batch.get('response_lengths', []))} "
        f"response_len_max={response_len_max} "
        f"response_len_sum={response_len_sum} "
        f"loss_tokens_max={loss_tokens_max} "
        f"loss_tokens_sum={loss_tokens_sum} "
        f"calculate_per_token_loss={args.calculate_per_token_loss} "
        f"loss_agg_mode={getattr(args, 'loss_agg_mode', None)} "
        f"use_dynamic_global_batch_size={args.use_dynamic_global_batch_size}"
    )
    _nan_dbg_flush_logs()


@torch.no_grad()
def policy_nan_metrics(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    ppo_kl: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Observe raw inputs without changing training tensors or their gradients."""
    _nan_dbg_warn_bad_tensor("logits", logits, batch=batch)
    _nan_dbg_warn_bad_tensor("log_probs", log_probs, batch=batch)
    _nan_dbg_warn_bad_tensor("old_log_probs", old_log_probs, batch=batch)
    _nan_dbg_warn_bad_tensor("advantages", advantages, batch=batch)
    _nan_dbg_warn_bad_tensor("ppo_kl", ppo_kl, batch=batch)

    ratio_delta = -ppo_kl
    ratio = ratio_delta.exp()
    ratio_delta_min, ratio_delta_max, _, _ = _nan_dbg_finite_stats(ratio_delta)
    _nan_dbg_warn_bad_tensor(
        "ratio_exp_current_minus_old",
        ratio,
        batch=batch,
        extra=f"ratio_delta_min={ratio_delta_min} ratio_delta_max={ratio_delta_max}",
    )

    log_probs_min, log_probs_max, _, log_probs_bad = _nan_dbg_finite_stats(log_probs)
    old_log_probs_min, old_log_probs_max, _, old_log_probs_bad = _nan_dbg_finite_stats(old_log_probs)
    _, _, advantage_absmax, advantages_bad = _nan_dbg_finite_stats(advantages)
    ppo_kl_min, ppo_kl_max, _, ppo_kl_bad = _nan_dbg_finite_stats(ppo_kl)
    _, ratio_max, _, ratio_bad = _nan_dbg_finite_stats(ratio)
    response_len_max, response_len_sum, loss_tokens_max, loss_tokens_sum = _nan_dbg_batch_stats(batch)
    tis_delta_min = float("nan")
    tis_delta_max = float("nan")
    tis_nonfinite_count = 0
    if (args.get_mismatch_metrics or args.use_tis) and batch.get("rollout_log_probs") is not None:
        rollout_log_probs_cat = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_tis_delta = old_log_probs - rollout_log_probs_cat
        tis_for_dbg = torch.exp(train_rollout_tis_delta)
        tis_delta_min, tis_delta_max, _, _ = _nan_dbg_finite_stats(train_rollout_tis_delta)
        _, _, _, tis_nonfinite_count = _nan_dbg_finite_stats(tis_for_dbg)

    return {
        "nan_dbg/log_probs_min": _nan_dbg_scalar(log_probs_min, log_probs.device),
        "nan_dbg/log_probs_max": _nan_dbg_scalar(log_probs_max, log_probs.device),
        "nan_dbg/old_log_probs_min": _nan_dbg_scalar(old_log_probs_min, log_probs.device),
        "nan_dbg/old_log_probs_max": _nan_dbg_scalar(old_log_probs_max, log_probs.device),
        "nan_dbg/advantage_absmax": _nan_dbg_scalar(advantage_absmax, log_probs.device),
        "nan_dbg/ppo_kl_min": _nan_dbg_scalar(ppo_kl_min, log_probs.device),
        "nan_dbg/ppo_kl_max": _nan_dbg_scalar(ppo_kl_max, log_probs.device),
        "nan_dbg/ratio_max": _nan_dbg_scalar(ratio_max, log_probs.device),
        "nan_dbg/response_len_max": _nan_dbg_scalar(response_len_max, log_probs.device),
        "nan_dbg/response_len_sum": _nan_dbg_scalar(response_len_sum, log_probs.device),
        "nan_dbg/loss_tokens_max": _nan_dbg_scalar(loss_tokens_max, log_probs.device),
        "nan_dbg/loss_tokens_sum": _nan_dbg_scalar(loss_tokens_sum, log_probs.device),
        "nan_dbg/nonfinite_count": _nan_dbg_scalar(
            log_probs_bad + old_log_probs_bad + advantages_bad + ppo_kl_bad + ratio_bad,
            log_probs.device,
        ),
        "nan_dbg/tis_delta_min": _nan_dbg_scalar(tis_delta_min, log_probs.device),
        "nan_dbg/tis_delta_max": _nan_dbg_scalar(tis_delta_max, log_probs.device),
        "nan_dbg/tis_nonfinite_count": _nan_dbg_scalar(tis_nonfinite_count, log_probs.device),
    }
