import logging
from collections.abc import Sequence

import torch

from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

logger = logging.getLogger(__name__)


class InMemoryCheckpointManager:
    """ref: nvidia_resiliency_ext's LocalCheckpointManager."""

    def __init__(self) -> None:
        self.latest_iteration: int = -1
        self._state_dict: object = None
        self.local_ckpt_dir: str = "<in-memory>"

    def save(self, state_dict: object, iteration: int, is_async: bool = False) -> None:
        """Store state_dict object reference in memory."""
        assert not is_async

        assert self._state_dict is None
        self._state_dict = state_dict
        self.latest_iteration = iteration

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return None

    def find_latest(self) -> int:
        return self.latest_iteration

    def load(self) -> tuple[object, str]:
        assert self.latest_iteration >= 0, "No in-memory checkpoint available"
        assert self._state_dict is not None
        ans = self._state_dict
        self._state_dict = None

        return ans, f"in-memory-ckpt-iter-{self.latest_iteration}"


def _assert_args_for_in_memory_checkpoint(args: object) -> None:
    assert getattr(args, 'non_persistent_ckpt_type', None) == 'local', (
        f"Expected non_persistent_ckpt_type='local', "
        f"got {getattr(args, 'non_persistent_ckpt_type', None)!r}"
    )
    assert getattr(args, 'non_persistent_local_ckpt_algo', None) is not None, (
        "args.non_persistent_local_ckpt_algo must be set"
    )


def save_to_memory(
    iteration: int,
    model: Sequence,
    optimizer: object,
    opt_param_scheduler: object,
) -> object:
    """Save checkpoint to in-memory manager via Megatron's save_checkpoint."""
    args = get_args()
    _assert_args_for_in_memory_checkpoint(args)

    manager = InMemoryCheckpointManager()
    save_checkpoint(
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        num_floating_point_operations_so_far=0,
        checkpointing_context={'local_checkpoint_manager': manager},
        non_persistent_ckpt=True,
    )
    state_dict, _ = manager.load()
    return state_dict
