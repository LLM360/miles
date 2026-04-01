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

    def clear_stale(self, *, optimizer: torch.optim.Optimizer, keep_count: int) -> None:
        """Zero out witness rows (and their optimizer state) that are NOT among
        the ``keep_count`` most recently allocated IDs.

        Args:
            optimizer: The optimizer whose state should also be cleared.
            keep_count: Number of most recent allocations to keep.
                Typically equals the number of sequences in the current step.
        """
        stale_ids = self._compute_stale_ids(keep_count=keep_count)
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

    def _compute_stale_ids(self, *, keep_count: int) -> list[int]:
        if self._counter == 0:
            return []

        n = self._ring_buffer_size
        actual_keep = min(keep_count, self._counter, n)
        if actual_keep >= n:
            return []

        head = self._counter % n
        # Active IDs occupy the contiguous range [active_start, head) mod n.
        # Stale IDs are the complement: one or two contiguous ranges.
        active_start = (head - actual_keep) % n

        if active_start < head:
            return list(range(0, active_start)) + list(range(head, n))
        else:
            # Wrapped: active spans [active_start, n) + [0, head)
            return list(range(head, active_start))


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


def set_pending_witness_ids(model: nn.Module, witness_ids: Optional[Tensor]) -> None:
    """Set witness IDs for consumption by the pre-decoder hook.

    This avoids callers directly touching the private ``_pending_witness_ids``
    attribute that ``install_witness_hook`` manages.
    """
    model._pending_witness_ids = witness_ids


_witness_id_allocator: Optional[WitnessIdAllocator] = None


def set_witness_id_allocator(allocator: Optional[WitnessIdAllocator]) -> None:
    global _witness_id_allocator
    _witness_id_allocator = allocator


def get_witness_id_allocator() -> WitnessIdAllocator:
    if _witness_id_allocator is None:
        raise RuntimeError("WitnessIdAllocator not initialized. Call set_witness_id_allocator() first.")
    return _witness_id_allocator
