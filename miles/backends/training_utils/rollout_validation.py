"""Local, read-only rollout diagnostics before actor training.

There are no collectives here: failures identify malformed local data, but do
not provide cross-rank agreement or distributed recovery.
"""

import logging
import math
import socket
import traceback

import numpy as np
import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_logits_and_tokens_offset_with_cp
from miles.backends.training_utils.parallel import get_parallel_state

_REQUIRED = ("rewards", "response_lengths", "total_lengths", "loss_masks")
_VECTORS = (
    "log_probs",
    "rollout_log_probs",
    "ref_log_probs",
    "values",
    "advantages",
    "returns",
    "teacher_log_probs",
    "opd_reverse_kl",
)


def validate_rollout_for_grpo_training_step(
    args,
    rollout_data,
    *,
    rollout_id=None,
    where="train_actor.begin",
    logger=None,
    require_log_probs=False,
):
    logger = logger or logging.getLogger(__name__)
    errors, warnings = [], []
    try:
        ps = get_parallel_state()
    except (AssertionError, RuntimeError) as error:
        ps = None
        warnings.append(f"get_parallel_state() failed: {error}")
    prefix = f"ROLLOUT_VALIDATE {where} rollout_id={rollout_id} {_rank_info(ps)} :: "
    configuration = {
        key: getattr(args, key, None)
        for key in (
            "advantage_estimator",
            "normalize_advantages",
            "use_rollout_logprobs",
            "use_critic",
            "qkv_format",
            "compute_advantages_and_returns",
            "n_samples_per_prompt",
            "context_parallel_size",
        )
    }
    logger.warning("%sstart %s", prefix, configuration)
    if not isinstance(rollout_data, dict):
        errors.append(f"rollout_data must be a dict, got {type(rollout_data).__name__}")
    else:
        _validate_batch(args, rollout_data, ps, errors, warnings, require_log_probs=require_log_probs)
    for warning in warnings[:50]:
        logger.warning("%s%s", prefix, warning)
    if len(warnings) > 50:
        logger.warning("%s%s additional warnings omitted", prefix, len(warnings) - 50)
    summary = _batch_summary(rollout_data)
    if errors:
        logger.error("%ssummary_before_failure %s", prefix, summary)
        for error in errors[:100]:
            logger.error("%sfailure %s", prefix, error)
        if len(errors) > 100:
            logger.error("%s%s additional errors omitted", prefix, len(errors) - 100)
        logger.error("%strace_at_validation_failure\n%s", prefix, "".join(traceback.format_stack(limit=12)))
        raise ValueError(f"{where}: rollout validation failed with {len(errors)} error(s); see logs above")
    logger.warning("%ssuccess %s", prefix, summary)


def _validate_batch(args, data, ps, errors, warnings, *, require_log_probs):
    for key in _REQUIRED:
        if data.get(key) is None:
            errors.append(f"missing required key {key!r}")
        elif not _is_sequence(data[key]):
            errors.append(f"{key!r} must be a sequence, got {type(data[key]).__name__}")
    if errors:
        return
    count = len(data["rewards"])
    keys = (*_REQUIRED[1:], "tokens", "input_ids", "max_seq_lens", *_VECTORS)
    for key in keys:
        if data.get(key) is not None and (not _is_sequence(data[key]) or len(data[key]) != count):
            errors.append(f"{key!r} length mismatch or bad sequence; expected {count} samples")
    logprob_key = "rollout_log_probs" if getattr(args, "use_rollout_logprobs", False) else "log_probs"
    if require_log_probs and data.get(logprob_key) is None:
        errors.append(f"require_log_probs=True but {logprob_key!r} is missing/None")
    if errors:
        return
    if not count:
        warnings.append("empty local rollout batch")
    if data.get("tokens") is None and data.get("input_ids") is None:
        warnings.append("neither 'tokens' nor 'input_ids' present; cannot check total_lengths against tokens")
    for index in range(count):
        _validate_sample(args, data, index, ps, errors, warnings)
    for key in ("n_samples_per_prompt", "grpo_group_size"):
        size = int(getattr(args, key, 0) or 0)
        if size > 0 and count % size:
            warnings.append(f"sample count {count} not divisible by {key}={size}")
    if getattr(args, "compute_advantages_and_returns", False) and getattr(args, "normalize_advantages", False):
        if data.get(logprob_key) is not None:
            warnings.append(f"normalization enters distributed whitening with {logprob_key}; verify other ranks' data")


def _validate_sample(args, data, index, ps, errors, warnings):
    try:
        if not math.isfinite(float(data["rewards"][index])):
            errors.append(f"rewards[{index}] is not finite")
        response = _integer(data["response_lengths"][index])
        total = _integer(data["total_lengths"][index])
        if not 0 <= response <= total or total <= 0:
            errors.append(f"invalid lengths at sample {index}: response={response}, total={total}")
            return
        maximum = _integer(data["max_seq_lens"][index]) if data.get("max_seq_lens") is not None else None
        if maximum is not None and maximum < total:
            errors.append(f"max_seq_lens[{index}]={maximum} < total_lengths[{index}]={total}")
            return
    except (TypeError, ValueError, OverflowError) as error:
        errors.append(f"bad reward or lengths at sample {index}: {error}")
        return
    for key in ("tokens", "input_ids"):
        if data.get(key) is not None:
            _check_vector(data[key][index], total, f"{key}[{index}]", errors)
    mask = _check_vector(data["loss_masks"][index], response, f"loss_masks[{index}]", errors)
    if mask is not None:
        mask_sum = mask.sum().item()
        # Final stable's warning-only correction (ae563ec96) is intentionally brought forward.
        if mask_sum <= 0:
            warnings.append(f"loss_masks[{index}] has no active tokens, sum={mask_sum}, response_len={response}")
        if mask_sum > response:
            warnings.append(f"loss_masks[{index}] sum={mask_sum} exceeds response_len={response} (weighted mask)")
        if not bool(((mask == 0) | (mask == 1)).all()):
            warnings.append(f"loss_masks[{index}] is not binary 0/1")
    try:
        local_length = _local_response_length(args, ps, total, response, maximum)
    except (AssertionError, TypeError, ValueError) as error:
        errors.append(f"invalid CP layout at sample {index}: {error}")
        return
    for key in _VECTORS:
        if data.get(key) is None:
            continue
        # Actor entry data uses zigzag response slices, including allgather-CP
        # outputs redistributed by the forward pass back to this layout.
        _check_vector(data[key][index], local_length, f"{key}[{index}]", errors)


def _is_sequence(value) -> bool:
    return isinstance(value, (list, tuple)) or isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim > 0


def _integer(value) -> int:
    result = int(value)
    if result != value:
        raise ValueError(f"length is not an integer: {value}")
    return result


def _check_vector(value, expected, name, errors):
    try:
        if not _is_sequence(value):
            raise TypeError("expected a vector")
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim != 1:
            errors.append(f"{name} must be 1D, got shape={tuple(tensor.shape)}")
        elif tensor.numel() != expected:
            errors.append(f"{name} length {tensor.numel()} != expected {expected}")
        elif not bool(torch.isfinite(tensor).all()):
            errors.append(f"{name} contains NaN/Inf")
        else:
            return tensor
    except (TypeError, ValueError, RuntimeError) as error:
        errors.append(f"{name} contains non-numeric values or an invalid vector: {error}")
    return None


def _local_response_length(args, ps, total, response, maximum) -> int:
    if ps is None or ps.cp.size == 1 or response == 0:
        return response
    _, _, offsets, _ = get_logits_and_tokens_offset_with_cp(
        total,
        response,
        getattr(args, "qkv_format", "thd"),
        maximum,
        cp_rank=ps.cp.rank,
        cp_size=ps.cp.size,
    )
    return sum(stop - start for start, stop in offsets)


def _rank_info(ps) -> str:
    parts = [f"host={socket.gethostname()}"]
    parts.append(
        f"global_rank={dist.get_rank()}/{dist.get_world_size()}"
        if dist.is_initialized()
        else "global_rank=dist_not_initialized"
    )
    if ps is not None:
        for name in ("effective_dp", "intra_dp", "indep_dp", "cp", "tp", "pp", "ep"):
            group = getattr(ps, name, None)
            if group is not None:
                parts.append(f"{name}_rank={group.rank}/{group.size}")
    return " ".join(parts)


def _batch_summary(data) -> str:
    if not isinstance(data, dict):
        return type(data).__name__
    lines = [f"keys={','.join(sorted(data))}"]
    for key in (*_REQUIRED, "tokens", "input_ids", "max_seq_lens", *_VECTORS):
        values = data.get(key)
        if not _is_sequence(values):
            lines.append(f"{key}={type(values).__name__}")
            continue
        descriptions = [
            dict(
                shape=getattr(v, "shape", (len(v),) if _is_sequence(v) else ()),
                dtype=str(getattr(v, "dtype", type(v).__name__)),
                device=str(getattr(v, "device", "cpu")),
            )
            for v in values[:3]
        ]
        lines.append(f"{key}: len={len(values)} first={descriptions}")
        if key in _REQUIRED:
            try:
                numbers = [float(torch.as_tensor(v).sum()) if key == "loss_masks" else float(v) for v in values]
                lines.append(
                    f"{key}: sum={sum(numbers):.6g} min={min(numbers, default=None)} max={max(numbers, default=None)}"
                )
            except (TypeError, ValueError, RuntimeError):
                lines.append(f"{key}: numeric summary unavailable")
    return " | ".join(lines)
