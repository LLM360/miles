"""Per-rank per-step weight checksum dumper for cross-replica consistency verification."""

import hashlib
import logging
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path

import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer.optimizer import MegatronOptimizer

from miles.backends.megatron_utils.ci_utils import _hash_tensor_bytes
from miles.utils.pydantic_utils import StrictBaseModel

logger = logging.getLogger(__name__)


class WeightChecksumEntry(StrictBaseModel):
    param_hashes: dict[str, str]
    buffer_hashes: dict[str, str]
    master_param_hashes: dict[str, str]
    optimizer_state_hashes: dict[str, str]


def _hash_tensor_sha256(tensor: torch.Tensor) -> str:
    raw_bytes = _hash_tensor_bytes(tensor)
    return hashlib.sha256(raw_bytes).hexdigest()


def compute_weight_checksums(
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
) -> WeightChecksumEntry:
    """Compute SHA-256 checksums of all model weights, buffers, master params, and optimizer states."""

    param_hashes: dict[str, str] = {}
    buffer_hashes: dict[str, str] = {}
    master_param_hashes: dict[str, str] = {}
    optimizer_state_hashes: dict[str, str] = {}

    # Hash model parameters
    for pp_idx, model_chunk in enumerate(model):
        for name, param in sorted(model_chunk.named_parameters(), key=lambda x: x[0]):
            if param is None:
                continue
            full_name = f"pp{pp_idx}.{name}"
            param_hashes[full_name] = _hash_tensor_sha256(param)

    # Hash model buffers
    for pp_idx, model_chunk in enumerate(model):
        for name, buffer in sorted(model_chunk.named_buffers(), key=lambda x: x[0]):
            if buffer is None:
                continue
            full_name = f"pp{pp_idx}.{name}"
            buffer_hashes[full_name] = _hash_tensor_sha256(buffer)

    # Hash fp32 master weights and optimizer states
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


def _collect_master_and_optimizer_hashes(
    chained_optimizer: object,
    master_param_hashes: dict[str, str],
    optimizer_state_hashes: dict[str, str],
) -> None:
    """Collect fp32 master weight hashes and optimizer state hashes from a single chained optimizer."""

    from megatron.core.optimizer import Float16OptimizerWithFloat16Params

    if not isinstance(chained_optimizer, Float16OptimizerWithFloat16Params):
        return

    # Build name mapping: match fp16 params to fp32 master params by position
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

    # Get names from the fp16 params via main_param attribute
    for fp16_param, fp32_param in zip(fp16_params, fp32_params, strict=True):
        param_name = _get_param_name(fp16_param)
        master_param_hashes[param_name] = _hash_tensor_sha256(fp32_param)

        # Hash optimizer states
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
    return str(id(param))


def dump_weight_checksums(
    entry: WeightChecksumEntry,
    output_dir: Path,
    step: int,
    rank: int,
) -> None:
    """Write a weight checksum entry to a JSON file."""
    file_path = output_dir / "weight_checksum" / f"step_{step:07d}" / f"rank_{rank:04d}.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(entry.model_dump_json())
    logger.info("Weight checksum dumped to %s", file_path)


def compute_and_dump_weight_checksums(
    args: Namespace,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    step: int,
) -> None:
    """Compute and dump weight checksums if enabled."""

    if not args.weight_checksum_enable:
        return

    interval = args.weight_checksum_interval
    if step % interval != 0:
        return

    entry = compute_weight_checksums(model=model, optimizer=optimizer)

    output_dir = Path(args.weight_checksum_dir)
    rank: int = torch.distributed.get_rank()

    dump_weight_checksums(
        entry=entry,
        output_dir=output_dir,
        step=step,
        rank=rank,
    )
