import logging
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from miles.utils.event_logger.logger import get_event_logger
from miles.utils.event_logger.models import WitnessEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_witness_allocator(*, model: Sequence[nn.Module], optimizer: torch.optim.Optimizer) -> None:
    """Find all witnesses in model chunks and set up the global allocator."""
    witnesses = list(_get_all_witnesses_in_model(model))
    assert len(witnesses) > 0
    _set_witness_id_allocator(
        WitnessIdAllocator(
            witnesses=witnesses,
            optimizer=optimizer,
        )
    )


def get_witness_id_allocator() -> "WitnessIdAllocator":
    if _witness_id_allocator is None:
        raise RuntimeError("WitnessIdAllocator not initialized. Call init_witness_allocator() first.")
    return _witness_id_allocator


def install_witness(model: nn.Module, *, buffer_size: int) -> None:
    model.head_witness = _DataWitness(buffer_size=buffer_size)
    model.tail_witness = _DataWitness(buffer_size=buffer_size)


def witness_dump_and_clear_stale(*, model: Sequence[nn.Module]) -> None:
    """Log nonzero witness param rows, then clear stale ring buffer entries."""
    for chunk in model:
        for attr in _WITNESS_ATTRS:
            witness: _DataWitness = getattr(chunk.module, attr)
            _record_and_log_witness_param(witness=witness, position=attr.replace("_witness", ""))

    get_witness_id_allocator().clear_stale()


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class _DataWitness(nn.Module):
    def __init__(self, buffer_size: int) -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self.witness = nn.Embedding(num_embeddings=buffer_size, embedding_dim=1)
        nn.init.zeros_(self.witness.weight)

    def forward(self, input_ids: Tensor, witness_ids: Tensor) -> Tensor:
        assert input_ids.shape == witness_ids.shape
        w = self.witness(witness_ids)  # (*, 1)
        out = w - w.detach()  # forward: bitwise 0 (for finite w), backward: d/dw = I
        return out


class WitnessIdAllocator:
    def __init__(self, *, witnesses: list[_DataWitness], optimizer: torch.optim.Optimizer) -> None:
        buffer_sizes = [x.buffer_size for x in witnesses]
        assert all(buffer_sizes[0] == x for x in buffer_sizes)
        self._buffer_size = buffer_sizes[0]

        self._witnesses = witnesses
        self._optimizer = optimizer

        self._counter: int = 0

    def allocate_for_sequences(self, num_sequences: int) -> list[int]:
        ids = [(self._counter + i) % self._buffer_size for i in range(num_sequences)]
        self._counter += num_sequences
        return ids

    def clear_stale(self) -> None:
        self._clear_stale(keep_count=int(self._buffer_size * 0.7))

    def _clear_stale(self, *, keep_count: int) -> None:
        stale_ids = self._compute_stale_ids(
            keep_count=keep_count, counter=self._counter, buffer_size=self._buffer_size
        )
        if not stale_ids:
            return

        for witness in self._witnesses:
            idx = torch.tensor(stale_ids, dtype=torch.long, device=witness.witness.weight.device)
            _zero_witness_rows(witness=witness, idx=idx, optimizer=self._optimizer)

    @classmethod
    def _compute_stale_ids(cls, *, keep_count: int, counter: int, buffer_size: int) -> list[int]:
        num_stale = buffer_size - min(keep_count, counter, buffer_size)
        if num_stale == 0:
            return []

        head = counter % buffer_size
        return [(head + i) % buffer_size for i in range(num_stale)]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


_witness_id_allocator: WitnessIdAllocator | None = None


def _set_witness_id_allocator(allocator: WitnessIdAllocator | None) -> None:
    global _witness_id_allocator
    _witness_id_allocator = allocator


_WITNESS_ATTRS = ("head_witness", "tail_witness")


def _get_all_witnesses_in_model(model_chunks: Sequence[nn.Module]) -> Iterable[_DataWitness]:
    for chunk in model_chunks:
        for attr in _WITNESS_ATTRS:
            yield getattr(chunk.module, attr)


def _zero_witness_rows(*, witness: _DataWitness, idx: Tensor, optimizer: torch.optim.Optimizer) -> None:
    model_weight = witness.witness.weight
    model_weight.data[idx] = 0.0

    opt_weight: Tensor = getattr(model_weight, "main_param", model_weight)
    if opt_weight is not model_weight:
        opt_weight.data[idx] = 0.0

    if opt_weight in optimizer.state:
        state = optimizer.state[opt_weight]
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key][idx] = 0.0


def _record_and_log_witness_param(
    *,
    witness: _DataWitness,
    position: str,
) -> None:
    weight = witness.witness.weight.data
    nonzero_ids: list[int] = weight.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    get_event_logger().log(
        WitnessEvent,
        dict(
            position=position,
            nonzero_ids=nonzero_ids,
        ),
        print_log=False,
    )
