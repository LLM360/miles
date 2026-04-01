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
        self.witness = nn.Embedding(num_embeddings=num_ids, embedding_dim=1)
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
        ids = [
            (self._counter + i) % self._ring_buffer_size
            for i in range(num_sequences)
        ]
        self._counter += num_sequences
        return ids

    def clear_stale(self, *, optimizer: torch.optim.Optimizer) -> None:
        stale_ids = self._compute_stale_ids()
        if not stale_ids:
            return

        idx = torch.tensor(stale_ids, dtype=torch.long, device=self._witness.witness.weight.device)

        model_weight = self._witness.witness.weight
        model_weight.data[idx] = 0.0

        # Megatron stores fp32 copy as main_param; clear it if present
        opt_weight: Tensor = getattr(model_weight, "main_param", model_weight)
        if opt_weight is not model_weight:
            opt_weight.data[idx] = 0.0

        # Megatron accumulates grads in main_grad; clear it if present
        main_grad: Optional[Tensor] = getattr(model_weight, "main_grad", None)
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

        # Active IDs are those allocated in the most recent ring_buffer_size allocations.
        # head points to the next slot to be allocated.
        head = self._counter % self._ring_buffer_size
        # The active IDs occupy a contiguous range (mod ring_buffer_size) ending
        # just before head. Everything else is stale.
        # With ring_buffer_size=N, all N slots are active when exactly N allocations
        # have been made, so there are no stale IDs. When counter > N, the number of
        # stale slots equals (counter - ring_buffer_size) capped at ring_buffer_size,
        # but since we already returned [] for counter <= N, all remaining slots
        # outside the active window [head - ring_buffer_size .. head) are stale.
        # In practice, every slot is always active because allocate_for_sequences
        # fills them round-robin, so stale IDs only exist transiently between
        # allocation and gradient clearing. The truly stale set is always empty
        # unless the caller allocated fewer than ring_buffer_size IDs recently.
        active_count = min(self._counter, self._ring_buffer_size)
        active_ids: set[int] = {
            (head - 1 - i) % self._ring_buffer_size
            for i in range(active_count)
        }

        return [i for i in range(self._ring_buffer_size) if i not in active_ids]


def _get_witness_grad(witness: DataWitness) -> Optional[Tensor]:
    """Return the gradient tensor for the witness embedding, preferring main_grad."""
    # Megatron stores gradients in main_grad when using distributed optimizer
    main_grad: Optional[Tensor] = getattr(witness.witness.weight, "main_grad", None)
    if main_grad is not None:
        return main_grad
    return witness.witness.weight.grad


def record_and_log_witness_grad(
    *,
    step: int,
    quorum_id: int,
    rank: int,
    witness: DataWitness,
) -> None:
    grad = _get_witness_grad(witness)
    if grad is None:
        logger.warning("No gradient found on witness at step %d rank %d", step, rank)
        return

    nonzero_ids: list[int] = grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    get_event_logger().log(
        WitnessEvent(
            step=step,
            quorum_id=quorum_id,
            rank=rank,
            nonzero_ids=nonzero_ids,
        ),
        print_log=False,
    )


def install_witness_hook(model: nn.Module, witness: DataWitness) -> None:
    """Attach a DataWitness submodule and register a pre-decoder hook on a GPTModel.

    The hook reads ``model._pending_witness_ids`` (set before each forward call)
    and adds the witness output to decoder_input. After consumption the attribute
    is cleared to ``None`` so stale IDs cannot leak into subsequent forward passes.
    Megatron only sees a generic pre-decoder hook -- no witness-specific code there.
    """
    model.head_witness = witness
    model._pending_witness_ids = None

    def _witness_hook(gpt_model: nn.Module, decoder_input: Optional[Tensor]) -> Optional[Tensor]:
        witness_ids: Optional[Tensor] = gpt_model._pending_witness_ids
        if witness_ids is None:
            return decoder_input

        gpt_model._pending_witness_ids = None

        witness_out: Tensor = gpt_model.head_witness(witness_ids)

        if decoder_input is not None:
            # pre_process stage: decoder_input comes from embedding
            return decoder_input + witness_out
        else:
            # non-pre_process stage: hidden states live in decoder.input_tensor
            gpt_model.decoder.input_tensor = gpt_model.decoder.input_tensor + witness_out
            return None

    model.register_pre_decoder_hook(_witness_hook)


_witness_id_allocator: Optional[WitnessIdAllocator] = None


def set_witness_id_allocator(allocator: Optional[WitnessIdAllocator]) -> None:
    global _witness_id_allocator
    _witness_id_allocator = allocator


def get_witness_id_allocator() -> WitnessIdAllocator:
    if _witness_id_allocator is None:
        raise RuntimeError("WitnessIdAllocator not initialized. Call set_witness_id_allocator() first.")
    return _witness_id_allocator
