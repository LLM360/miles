import logging
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from miles.utils.event_logger.logger import get_event_logger
from miles.utils.event_logger.models import WitnessEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API (called by actor.py, model.py, model_provider.py, data.py)
# ---------------------------------------------------------------------------


def init_witness_allocator(*, model_chunks: Sequence[nn.Module], ring_buffer_size: int) -> None:
    """Find all witnesses in model chunks and set up the global allocator."""
    witnesses = _find_all_witnesses_in_model_chunks(model_chunks)
    if witnesses:
        _set_witness_id_allocator(WitnessIdAllocator(
            ring_buffer_size=ring_buffer_size,
            witnesses=witnesses,
        ))


def get_witness_id_allocator() -> "WitnessIdAllocator":
    if _witness_id_allocator is None:
        raise RuntimeError("WitnessIdAllocator not initialized. Call init_witness_allocator() first.")
    return _witness_id_allocator


def install_witness(model: nn.Module, *, num_ids: int) -> None:
    """Attach head and tail DataWitness submodules to a GPTModel.

    Both participate in DDP, optimizer, and checkpointing automatically.
    Callers pass ``witness_ids`` to ``GPTModel.forward()`` to activate them.
    head_witness probes gradients flowing into the decoder;
    tail_witness probes gradients flowing out of the decoder.
    """
    model.head_witness = DataWitness(num_ids=num_ids)
    model.tail_witness = DataWitness(num_ids=num_ids)


def dump_witness_params(
    *,
    model_chunks: Sequence[nn.Module],
    step: int,
) -> None:
    """Find all witness submodules (head + tail) in model chunks and log nonzero param rows."""
    for chunk in model_chunks:
        for attr in _WITNESS_ATTRS:
            witness: Optional[DataWitness] = getattr(chunk.module, attr, None)
            if witness is not None:
                _record_and_log_witness_param(
                    step=step,
                    witness=witness,
                    position=attr,
                )


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class DataWitness(nn.Module):
    def __init__(self, num_ids: int) -> None:
        super().__init__()
        self.witness = nn.Embedding(num_embeddings=num_ids, embedding_dim=1)
        nn.init.zeros_(self.witness.weight)

    def forward(self, input_ids: Tensor, witness_ids: Tensor) -> Tensor:
        assert input_ids.shape == witness_ids.shape
        w = self.witness(witness_ids)  # (*, 1)
        out = w - w.detach()  # forward: bitwise 0 (for finite w), backward: d/dw = I
        return out


class WitnessIdAllocator:
    def __init__(self, *, ring_buffer_size: int, witnesses: list[DataWitness]) -> None:
        self._ring_buffer_size = ring_buffer_size
        self._witnesses = witnesses
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
        """
        stale_ids = self._compute_stale_ids(keep_count=keep_count)
        if not stale_ids:
            return

        for witness in self._witnesses:
            idx = torch.tensor(stale_ids, dtype=torch.long, device=witness.witness.weight.device)
            _zero_witness_rows(witness=witness, idx=idx, optimizer=optimizer)

    def _compute_stale_ids(self, *, keep_count: int) -> list[int]:
        if self._counter == 0:
            return []

        n = self._ring_buffer_size
        actual_keep = min(keep_count, self._counter, n)
        if actual_keep >= n:
            return []

        head = self._counter % n
        active_start = (head - actual_keep) % n

        if active_start < head:
            return list(range(0, active_start)) + list(range(head, n))
        else:
            return list(range(head, active_start))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


_witness_id_allocator: Optional[WitnessIdAllocator] = None


def _set_witness_id_allocator(allocator: Optional[WitnessIdAllocator]) -> None:
    global _witness_id_allocator
    _witness_id_allocator = allocator


_WITNESS_ATTRS = ("head_witness", "tail_witness")


def _find_all_witnesses_in_model_chunks(model_chunks: Sequence[nn.Module]) -> list[DataWitness]:
    witnesses: list[DataWitness] = []
    for chunk in model_chunks:
        for attr in _WITNESS_ATTRS:
            w: Optional[DataWitness] = getattr(chunk.module, attr, None)
            if w is not None:
                witnesses.append(w)
        if witnesses:
            break
    return witnesses


def _zero_witness_rows(*, witness: DataWitness, idx: Tensor, optimizer: torch.optim.Optimizer) -> None:
    model_weight = witness.witness.weight
    model_weight.data[idx] = 0.0

    opt_weight: Tensor = getattr(model_weight, "main_param", model_weight)
    if opt_weight is not model_weight:
        opt_weight.data[idx] = 0.0

    main_grad: Optional[Tensor] = getattr(model_weight, "main_grad", None)
    if main_grad is not None:
        main_grad.data[idx] = 0.0

    if opt_weight in optimizer.state:
        state = optimizer.state[opt_weight]
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key][idx] = 0.0


def _record_and_log_witness_param(
    *,
    step: int,
    witness: DataWitness,
    position: str,
) -> None:
    weight = witness.witness.weight.data
    nonzero_ids: list[int] = weight.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    get_event_logger().log(
        WitnessEvent,
        dict(
            step=step,
            position=position,
            nonzero_ids=nonzero_ids,
        ),
        print_log=False,
    )
