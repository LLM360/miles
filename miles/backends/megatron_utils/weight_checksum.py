"""Per-rank per-step weight checksum dumper for cross-replica consistency verification."""

import hashlib
import logging
from argparse import Namespace
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer.optimizer import MegatronOptimizer

from miles.backends.megatron_utils.ci_utils import _hash_tensor_bytes
from miles.utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.event_logger.models import LocalWeightChecksumEvent

logger = logging.getLogger(__name__)


def compute_and_dump_weight_checksums(
    args: Namespace,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    step: int,
) -> None:
    """Compute and dump weight checksums if enabled."""

    if not args.save_local_weight_checksum:
        return

    if not is_event_logger_initialized():
        logger.warning("EventLogger not initialized, skipping weight checksum dump")
        return

    info = _compute_weight_checksums(model=model, optimizer=optimizer, step=step, rank=torch.distributed.get_rank())
    get_event_logger().log(info, print_log=False)


def _compute_weight_checksums(
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    step: int,
    rank: int,
) -> LocalWeightChecksumEvent:
    param_hashes = _hash_named_tensors(model, accessor="named_parameters")
    buffer_hashes = _hash_named_tensors(model, accessor="named_buffers")

    name_by_fp32_id = _build_name_by_fp32_id(model)
    master_param_hashes, optimizer_state_hashes = _collect_master_and_optimizer_hashes(
        optimizer=optimizer,
        name_by_fp32_id=name_by_fp32_id,
    )

    return LocalWeightChecksumEvent(
        step=step,
        rank=rank,
        param_hashes=param_hashes,
        buffer_hashes=buffer_hashes,
        master_param_hashes=master_param_hashes,
        optimizer_state_hashes=optimizer_state_hashes,
    )


def _hash_named_tensors(model: Sequence[DDP], *, accessor: str) -> dict[str, str]:
    """Hash all named tensors from model chunks using the given accessor method."""
    hashes: dict[str, str] = {}
    for pp_idx, model_chunk in enumerate(model):
        for name, tensor in sorted(getattr(model_chunk, accessor)(), key=lambda x: x[0]):
            if tensor is None:
                continue
            hashes[f"pp{pp_idx}.{name}"] = _hash_tensor_sha256(tensor)
    return hashes


def _build_name_by_fp32_id(model: Sequence[DDP]) -> dict[int, str]:
    """Build id(fp32_master_param) → name mapping from model parameters."""
    name_map: dict[int, str] = {}
    for pp_idx, model_chunk in enumerate(model):
        for name, param in model_chunk.named_parameters():
            if param is None:
                continue
            main_param = getattr(param, "main_param", None)
            if main_param is not None:
                name_map[id(main_param)] = f"pp{pp_idx}.{name}"
    return name_map


def _collect_master_and_optimizer_hashes(
    optimizer: MegatronOptimizer,
    name_by_fp32_id: dict[int, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Collect fp32 master weight hashes and optimizer state hashes via inner PyTorch optimizer."""
    master_param_hashes: dict[str, str] = {}
    optimizer_state_hashes: dict[str, str] = {}

    for fp32_param, state in _iter_fp32_params_and_states(optimizer):
        param_name = name_by_fp32_id.get(id(fp32_param))
        if param_name is None:
            continue

        master_param_hashes[param_name] = _hash_tensor_sha256(fp32_param)

        for state_key in ("exp_avg", "exp_avg_sq"):
            if state_key in state:
                optimizer_state_hashes[f"{param_name}/{state_key}"] = _hash_tensor_sha256(state[state_key])

    return master_param_hashes, optimizer_state_hashes


def _iter_fp32_params_and_states(
    optimizer: MegatronOptimizer,
) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """Yield (fp32_param, optimizer_state_dict) from all sub-optimizers.

    Works uniformly for both Float16OptimizerWithFloat16Params and DistributedOptimizer,
    because both place fp32 master params into inner optimizer's param_groups.
    """
    for sub_opt in _iter_sub_optimizers(optimizer):
        inner = getattr(sub_opt, "optimizer", None)
        if inner is None:
            continue

        for group in inner.param_groups:
            for fp32_param in group["params"]:
                state = inner.state.get(fp32_param, {})
                yield fp32_param, state


def _iter_sub_optimizers(optimizer: MegatronOptimizer) -> Iterator[MegatronOptimizer]:
    """Flatten ChainedOptimizer into individual sub-optimizers."""
    if hasattr(optimizer, "chained_optimizers"):
        for sub in optimizer.chained_optimizers:
            yield from _iter_sub_optimizers(sub)
    else:
        yield optimizer


def _hash_tensor_sha256(tensor: torch.Tensor) -> str:
    raw_bytes = _hash_tensor_bytes(tensor)
    return hashlib.sha256(raw_bytes).hexdigest()
