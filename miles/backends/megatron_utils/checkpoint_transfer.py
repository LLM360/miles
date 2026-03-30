import logging
from collections.abc import Sequence
from datetime import timedelta

import torch

from miles.backends.megatron_utils.in_memory_checkpoint import InMemoryCheckpointManager, save_to_memory
from miles.utils.process_group_utils import GroupInfo

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 120


def _create_transport(indep_dp: GroupInfo, timeout_seconds: int) -> tuple:
    from torchft.checkpointing.pg_transport import PGTransport

    timeout = timedelta(seconds=timeout_seconds)
    transport = PGTransport(
        pg=indep_dp.group,
        timeout=timeout,
        device=torch.device("cuda"),
    )
    return transport, timeout


def send_ckpt(
    *,
    indep_dp: GroupInfo,
    model: Sequence,
    optimizer: object,
    opt_param_scheduler: object,
    iteration: int,
    dst_rank: int,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Send in-memory checkpoint to a destination cell via torchft PGTransport.

    Args:
        indep_dp: Independent DP group info (provides the torchft PG).
        model: Megatron model chunks.
        optimizer: Megatron optimizer.
        opt_param_scheduler: LR scheduler.
        iteration: Current training iteration / rollout_id.
        dst_rank: Destination cell_id in the indep_dp process group.
        timeout_seconds: Timeout for the NCCL send operation.
    """
    state_dict = save_to_memory(
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
    )

    transport, timeout = _create_transport(indep_dp, timeout_seconds)
    transport.send_checkpoint(
        dst_ranks=[dst_rank],
        step=iteration,
        state_dict=state_dict,
        timeout=timeout,
    )
    transport.disallow_checkpoint()
    logger.info(f"Sent checkpoint to cell {dst_rank}")


def recv_ckpt(
    *,
    indep_dp: GroupInfo,
    src_rank: int,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> InMemoryCheckpointManager:
    """Receive checkpoint from a healthy cell via torchft PGTransport.

    Returns an InMemoryCheckpointManager containing the received state_dict,
    ready to be passed to initialize_model_and_optimizer.

    Args:
        indep_dp: Independent DP group info (provides the torchft PG).
        src_rank: Source cell_id in the indep_dp process group.
        timeout_seconds: Timeout for the NCCL recv operation.

    Returns:
        InMemoryCheckpointManager with state_dict loaded, ready for
        initialize_model_and_optimizer to consume.
    """
    transport, timeout = _create_transport(indep_dp, timeout_seconds)
    state_dict = transport.recv_checkpoint(
        src_rank=src_rank,
        metadata=transport.metadata(),
        step=0,
        timeout=timeout,
    )
    logger.info(f"Received checkpoint from cell {src_rank}")

    manager = InMemoryCheckpointManager()
    manager._state_dict = state_dict
    manager.latest_iteration = 0
    return manager
