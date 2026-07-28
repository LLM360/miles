"""Helpers for routing rollout payloads to their actual training consumers."""

from collections.abc import Sequence

import numpy as np


ROLLOUT_DATA_REF_FORMAT = "destination_sharded_v1"
ROUTED_EXPERTS_SHARD_META_KEY = "_rollout_routed_experts_shard"


def rollout_destination_key(dp_rank: int, pp_rank: int, cp_rank: int) -> str:
    """Return a stable, Ray-serializable key for a training destination."""
    return f"{dp_rank}:{pp_rank}:{cp_rank}"


def validate_routed_experts(
    routed_experts: Sequence[np.ndarray],
    token_lengths: Sequence[int],
) -> None:
    """Validate the source routing tensors before any lossy PP/CP compaction."""
    if len(routed_experts) != len(token_lengths):
        raise ValueError(
            f"routing-replay sample count {len(routed_experts)} does not match "
            f"token-length count {len(token_lengths)}"
        )

    expected_tail = None
    for sample_idx, (sample, token_length) in enumerate(
        zip(routed_experts, token_lengths, strict=True)
    ):
        shape = getattr(sample, "shape", None)
        if not isinstance(sample, np.ndarray) or shape is None or len(shape) != 3:
            raise ValueError(
                f"routing-replay sample {sample_idx} must be a rank-3 numpy array, got shape={shape}"
            )
        expected_rows = token_length - 1
        if shape[0] != expected_rows:
            raise ValueError(
                f"routing-replay sample {sample_idx} has {shape[0]} token rows; "
                f"expected {expected_rows} for {token_length} tokens"
            )
        if expected_tail is None:
            expected_tail = shape[1:]
        elif shape[1:] != expected_tail:
            raise ValueError(
                f"routing-replay sample {sample_idx} layer/topk shape {shape[1:]} "
                f"does not match {expected_tail}"
            )


def shard_routed_experts_for_destination(
    routed_experts: Sequence[np.ndarray],
    *,
    layer_indices: Sequence[int],
    cp_rank: int,
    cp_size: int,
    qkv_format: str,
    max_seq_len: int | None = None,
) -> list[np.ndarray]:
    """Shard routing data for one PP/CP destination."""
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if qkv_format not in {"thd", "bshd"}:
        raise ValueError(f"unsupported qkv_format {qkv_format!r}")
    if qkv_format == "bshd" and max_seq_len is None:
        raise ValueError("max_seq_len is required for qkv_format='bshd'")

    layers = tuple(int(layer_idx) for layer_idx in layer_indices)
    return [
        _shard_one_routed_experts(
            sample,
            layer_indices=layers,
            cp_rank=cp_rank,
            cp_size=cp_size,
            qkv_format=qkv_format,
            max_seq_len=max_seq_len,
        )
        for sample in routed_experts
    ]


def _shard_one_routed_experts(
    routed_experts: np.ndarray,
    *,
    layer_indices: tuple[int, ...],
    cp_rank: int,
    cp_size: int,
    qkv_format: str,
    max_seq_len: int | None,
) -> np.ndarray:
    if not isinstance(routed_experts, np.ndarray) or routed_experts.ndim != 3:
        shape = getattr(routed_experts, "shape", None)
        raise ValueError(f"routed experts must be a rank-3 numpy array, got shape={shape}")

    num_routed_tokens, num_layers, topk = routed_experts.shape
    if any(layer_idx < 0 or layer_idx >= num_layers for layer_idx in layer_indices):
        raise ValueError(
            f"layer indices {layer_indices} are outside routed-expert layer dimension {num_layers}"
        )

    # The replay path appends one row for the final token before CP slicing.
    num_tokens = num_routed_tokens + 1
    target_length = num_tokens if qkv_format == "thd" else int(max_seq_len)
    if target_length < num_tokens:
        raise ValueError(
            f"max_seq_len {target_length} is shorter than sample token length {num_tokens}"
        )

    if cp_size == 1:
        ranges = ((0, target_length),)
    else:
        chunk_size = (target_length + 2 * cp_size - 1) // (2 * cp_size)
        ranges = (
            (chunk_size * cp_rank, chunk_size * (cp_rank + 1)),
            (
                chunk_size * (2 * cp_size - cp_rank - 1),
                chunk_size * (2 * cp_size - cp_rank),
            ),
        )

    output_length = sum(end - start for start, end in ranges)
    output = np.full(
        (output_length, len(layer_indices), topk),
        fill_value=-1,
        dtype=routed_experts.dtype,
    )

    output_start = 0
    for start, end in ranges:
        chunk_length = end - start
        valid_end = min(end, num_routed_tokens)
        if start < valid_end:
            valid_length = valid_end - start
            output[output_start : output_start + valid_length] = routed_experts[
                start:valid_end, layer_indices, :
            ]
        output_start += chunk_length

    return output
