"""Per-rank per-step weight checksum dumper for cross-replica consistency verification."""

import hashlib
import logging
from argparse import Namespace
from collections.abc import Sequence

import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer.optimizer import MegatronOptimizer

from miles.backends.megatron_utils.ci_utils import _hash_tensor_bytes
from miles.utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.event_logger.models import WeightChecksumDumped
from miles.utils.pydantic_utils import StrictBaseModel

logger = logging.getLogger(__name__)


class WeightChecksumEntry(StrictBaseModel):
    param_hashes: dict[str, str]
    buffer_hashes: dict[str, str]
    master_param_hashes: dict[str, str]
    optimizer_state_hashes: dict[str, str]


def compute_and_dump_weight_checksums(
    args: Namespace,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    step: int,
) -> None:
    """Compute and dump weight checksums if enabled."""

    if not args.save_local_weight_checksum:
        return

    entry = compute_weight_checksums(model=model, optimizer=optimizer)
    rank: int = torch.distributed.get_rank()

    dump_weight_checksums(entry=entry, step=step, rank=rank)


def compute_weight_checksums(
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
) -> WeightChecksumEntry:
    """Compute SHA-256 checksums of all model weights, buffers, master params, and optimizer states."""

    master_param_hashes: dict[str, str] = {}
    optimizer_state_hashes: dict[str, str] = {}

    param_hashes = _hash_named_tensors(model, accessor="named_parameters")
    buffer_hashes = _hash_named_tensors(model, accessor="named_buffers")

    if hasattr(optimizer, "chained_optimizers"):
        for chained_optimizer in optimizer.chained_optimizers:
            _collect_master_and_optimizer_hashes(
                chained_optimizer=chained_optimizer,
                master_param_hashes=master_param_hashes,
                optimizer_state_hashes=optimizer_state_hashes,
            )

    return WeightChecksumEntry(
        param_hashes=param_hashes,
        buffer_hashes=buffer_hashes,
        master_param_hashes=master_param_hashes,
        optimizer_state_hashes=optimizer_state_hashes,
    )


def dump_weight_checksums(
    entry: WeightChecksumEntry,
    step: int,
    rank: int,
) -> None:
    """Log a weight checksum entry via the EventLogger."""
    if not is_event_logger_initialized():
        logger.warning("EventLogger not initialized, skipping weight checksum dump")
        return

    event = WeightChecksumDumped(
        step=step,
        rank=rank,
        param_hashes=entry.param_hashes,
        buffer_hashes=entry.buffer_hashes,
        master_param_hashes=entry.master_param_hashes,
        optimizer_state_hashes=entry.optimizer_state_hashes,
    )
    get_event_logger().log(event)
    logger.info("Weight checksum logged for step=%d rank=%d", step, rank)


def _hash_named_tensors(model: Sequence[DDP], *, accessor: str) -> dict[str, str]:
    """Hash all named tensors from model chunks using the given accessor method."""
    hashes: dict[str, str] = {}
    for pp_idx, model_chunk in enumerate(model):
        for name, tensor in sorted(getattr(model_chunk, accessor)(), key=lambda x: x[0]):
            if tensor is None:
                continue
            hashes[f"pp{pp_idx}.{name}"] = _hash_tensor_sha256(tensor)
    return hashes


def _collect_master_and_optimizer_hashes(
    chained_optimizer: object,
    master_param_hashes: dict[str, str],
    optimizer_state_hashes: dict[str, str],
) -> None:
    """Collect fp32 master weight hashes and optimizer state hashes from a single chained optimizer."""

    from megatron.core.optimizer import Float16OptimizerWithFloat16Params

    if not isinstance(chained_optimizer, Float16OptimizerWithFloat16Params):
        return

    fp16_params: list[torch.nn.Parameter] = []
    for group in chained_optimizer.float16_groups:
        fp16_params.extend(group)

    fp32_params: list[torch.nn.Parameter] = []
    for group in chained_optimizer.fp32_from_float16_groups:
        fp32_params.extend(group)

    if len(fp16_params) != len(fp32_params):
        logger.warning(
            "fp16_params count (%d) != fp32_params count (%d), skipping master param hashing",
            len(fp16_params),
            len(fp32_params),
        )
        return

    for fp16_param, fp32_param in zip(fp16_params, fp32_params, strict=True):
        param_name = _get_param_name(fp16_param)
        master_param_hashes[param_name] = _hash_tensor_sha256(fp32_param)

        state = chained_optimizer.optimizer.state.get(fp32_param, {})
        for state_key in ("exp_avg", "exp_avg_sq"):
            if state_key in state:
                state_name = f"{param_name}/{state_key}"
                optimizer_state_hashes[state_name] = _hash_tensor_sha256(state[state_key])


def _get_param_name(param: torch.nn.Parameter) -> str:
    """Extract a human-readable name for a parameter."""
    if hasattr(param, "main_param"):
        main = param.main_param
        if hasattr(main, "_param_name"):
            return main._param_name
    if hasattr(param, "_param_name"):
        return param._param_name
    logger.warning("Parameter has no _param_name attribute, using id() as fallback (non-deterministic across ranks)")
    return str(id(param))


def _hash_tensor_sha256(tensor: torch.Tensor) -> str:
    raw_bytes = _hash_tensor_bytes(tensor)
    return hashlib.sha256(raw_bytes).hexdigest()
