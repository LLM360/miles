import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from miles.utils.event_logger.logger import get_event_logger
from miles.utils.event_logger.models import WitnessEvent

logger = logging.getLogger(__name__)


class DataWitness(nn.Module):
    def __init__(self, num_ids: int) -> None:
        super().__init__()
        self.witness = nn.Embedding(num_ids, 1)
        nn.init.zeros_(self.witness.weight)

    def forward(self, witness_ids: Tensor) -> Tensor:
        w = self.witness(witness_ids)  # (*, 1)
        return w - w.detach()  # forward: bitwise 0 (for finite w), backward: d/dw = I


class WitnessIdAllocator:
    def __init__(self, *, ring_buffer_size: int, witness: DataWitness) -> None:
        self._ring_buffer_size = ring_buffer_size
        self._witness = witness
        self._counter: int = 0

    @property
    def ring_buffer_size(self) -> int:
        return self._ring_buffer_size

    @property
    def counter(self) -> int:
        return self._counter

    def allocate_for_sequences(self, num_sequences: int) -> list[int]:
        ids = []
        for _ in range(num_sequences):
            ids.append(self._counter % self._ring_buffer_size)
            self._counter += 1
        return ids

    def clear_stale(self, *, optimizer: torch.optim.Optimizer) -> None:
        stale_ids = self._compute_stale_ids()
        if not stale_ids:
            return

        idx = torch.tensor(stale_ids, dtype=torch.long, device=self._witness.witness.weight.device)

        model_weight = self._witness.witness.weight
        model_weight.data[idx] = 0.0

        opt_weight = getattr(model_weight, "main_param", model_weight)
        if opt_weight is not model_weight:
            opt_weight.data[idx] = 0.0

        main_grad = getattr(model_weight, "main_grad", None)
        if main_grad is not None:
            main_grad.data[idx] = 0.0

        if opt_weight in optimizer.state:
            state = optimizer.state[opt_weight]
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][idx] = 0.0

    def _compute_stale_ids(self) -> list[int]:
        if self._counter <= self._ring_buffer_size:
            return []

        head = self._counter % self._ring_buffer_size
        newest_start = (head - 1) % self._ring_buffer_size

        stale = []
        for offset in range(1, self._ring_buffer_size):
            candidate = (newest_start + offset) % self._ring_buffer_size
            if candidate == head:
                break
            stale.append(candidate)
        return stale


class WitnessGradRecorder:
    def record_and_log(
        self,
        *,
        step: int,
        quorum_id: int,
        rank: int,
        witness: DataWitness,
    ) -> None:
        grad = getattr(witness.witness.weight, "main_grad", None)
        if grad is None:
            grad = witness.witness.weight.grad
        if grad is None:
            logger.warning("No gradient found on witness at step %d rank %d", step, rank)
            return

        nonzero_ids = grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

        get_event_logger().log(
            WitnessEvent(
                step=step,
                quorum_id=quorum_id,
                rank=rank,
                nonzero_ids=nonzero_ids,
            ),
            print_log=False,
        )


def install_witness_hook(model: nn.Module, witness: "DataWitness") -> None:
    """Attach a DataWitness submodule and register a pre-decoder hook on a GPTModel.

    The hook reads ``model._pending_witness_ids`` (set before each forward call)
    and adds the witness output to decoder_input. Megatron only sees a generic
    pre-decoder hook — no witness-specific code in Megatron.
    """
    model.head_witness = witness

    def _witness_hook(gpt_model: nn.Module, decoder_input: Tensor) -> Tensor:
        witness_ids = getattr(gpt_model, "_pending_witness_ids", None)
        if witness_ids is None:
            return decoder_input
        witness_out = gpt_model.head_witness(witness_ids)
        return decoder_input + witness_out

    model.register_pre_decoder_hook(_witness_hook)


_witness_id_allocator: Optional[WitnessIdAllocator] = None


def set_witness_id_allocator(allocator: Optional[WitnessIdAllocator]) -> None:
    global _witness_id_allocator
    _witness_id_allocator = allocator


def get_witness_id_allocator() -> WitnessIdAllocator:
    if _witness_id_allocator is None:
        raise RuntimeError("WitnessIdAllocator not initialized. Call set_witness_id_allocator() first.")
    return _witness_id_allocator
