"""Read-only diagnostics for the existing DP shard and object-store paths."""

import logging
import sys

import numpy as np
import torch

logger = logging.getLogger(__name__)


def log_rollout_dp_shards(shards: list[dict], refs: list) -> None:
    summaries = []
    for rank, (shard, ref) in enumerate(zip(shards, refs, strict=True)):
        tokens = _stats([len(value) for value in shard["tokens"]])
        responses = _stats(shard.get("response_lengths", []))
        mask_lengths = _stats([len(value) for value in shard.get("loss_masks", [])])
        payload_mb = _estimate_payload_bytes(shard) / 1024**2
        summaries.append((tokens["n"], tokens["sum"], payload_mb))
        logger.warning(
            "ROLLOUT_DP_SHARD dp=%s samples=%s token_sum=%s token_min=%s token_max=%s token_avg=%s "
            "response_sum=%s response_min=%s response_max=%s response_avg=%s "
            "loss_mask_sum=%s payload_mb_est=%.2f object_ref=%s partition=%s",
            rank,
            tokens["n"],
            tokens["sum"],
            tokens["min"],
            tokens["max"],
            tokens["avg"],
            responses["sum"],
            responses["min"],
            responses["max"],
            responses["avg"],
            mask_lengths["sum"],
            payload_mb,
            getattr(ref, "inner", ref),
            list(shard["partition"]),
        )
    counts, token_sums, payloads = (
        (list(values) for values in zip(*summaries, strict=True)) if summaries else ([], [], [])
    )
    logger.warning(
        "ROLLOUT_DP_IMBALANCE dp_size=%s total_samples=%s total_tokens=%s "
        "sample_counts=%s sample_ratio=%s token_sums=%s token_ratio=%s payload_mbs=%s payload_ratio=%s",
        len(shards),
        sum(counts),
        sum(token_sums),
        counts,
        _ratio(counts),
        token_sums,
        _ratio(token_sums),
        payloads,
        _ratio(payloads),
    )


def _stats(values) -> dict:
    count = len(values)
    total = sum(values)
    return dict(
        n=count,
        sum=total,
        min=min(values) if count else 0,
        max=max(values) if count else 0,
        avg=round(total / count, 1) if count else 0,
    )


def _ratio(values) -> float:
    if not values:
        return 0.0
    return round(max(values) / min(values), 3) if min(values) else float("inf")


def _estimate_payload_bytes(value, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (str, bytes, bytearray)):
        return len(value)
    if isinstance(value, dict):
        return sum(_estimate_payload_bytes(k, seen) + _estimate_payload_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple, range)):
        return sum(_estimate_payload_bytes(item, seen) for item in value)
    return sys.getsizeof(value)
