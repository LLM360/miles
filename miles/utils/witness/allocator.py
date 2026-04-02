import torch
import torch.nn as nn

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class WitnessInfo(FrozenStrictBaseModel):
    witness_ids: list[int]


class WitnessIdAllocator:
    def __init__(self, *, witnesses: list, optimizer: torch.optim.Optimizer) -> None:
        buffer_sizes = [x.buffer_size for x in witnesses]
        assert all(buffer_sizes[0] == x for x in buffer_sizes)
        self._buffer_size = buffer_sizes[0]

        self._witnesses = witnesses
        self._optimizer = optimizer

        self._counter: int = 0

    # TODO: rename to allocate()
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

        from miles.utils.witness.module import _zero_witness_rows

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
