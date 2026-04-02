TODO_split_to_test_allocator_and_test_module

"""Tests for miles.utils.witness: DataWitness, WitnessIdAllocator."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from miles.utils.witness import WitnessIdAllocator, _DataWitness, _record_and_log_witness_param, install_witness

class TestWitnessIdAllocator:
    def _make_allocator(
        self, witness: _DataWitness, optimizer: torch.optim.Optimizer | None = None
    ) -> WitnessIdAllocator:
        if optimizer is None:
            optimizer = torch.optim.Adam(witness.parameters(), lr=0.1)
        return WitnessIdAllocator(witnesses=[witness], optimizer=optimizer)

    def test_monotonic_and_wraps(self) -> None:
        witness = _DataWitness(num_ids=5)
        allocator = self._make_allocator(witness)

        ids1 = allocator.allocate_for_sequences(3)
        assert ids1 == [0, 1, 2]

        ids2 = allocator.allocate_for_sequences(4)
        assert ids2 == [3, 4, 0, 1]

    def test_allocate_for_sequences_returns_correct_count(self) -> None:
        witness = _DataWitness(num_ids=100)
        allocator = self._make_allocator(witness)
        ids = allocator.allocate_for_sequences(7)
        assert len(ids) == 7

    def test_per_sequence_id_same_for_all_tokens(self) -> None:
        witness = _DataWitness(num_ids=100)
        allocator = self._make_allocator(witness)
        seq_ids = allocator.allocate_for_sequences(3)

        token_lengths = [10, 20, 15]
        witness_ids_list = [
            torch.full((length,), fill_value=sid, dtype=torch.long) for length, sid in zip(token_lengths, seq_ids)
        ]

        for wids, sid in zip(witness_ids_list, seq_ids):
            assert torch.all(wids == sid)

    def test_auto_clear_stale_zeros_oldest_ids(self) -> None:
        """allocate_for_sequences auto-clears stale IDs (keep 70% of buffer)."""
        # buffer=10, keep=7 → 3 stale slots
        witness = _DataWitness(num_ids=10)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.1)
        allocator = WitnessIdAllocator(witnesses=[witness], optimizer=optimizer)

        witness.witness.weight.data.fill_(1.0)

        # Allocate 8 IDs → counter=8, head=8
        # Active (last 7): IDs 1..7. Stale: IDs 8,9,0
        allocator.allocate_for_sequences(8)

        for i in [1, 2, 3, 4, 5, 6, 7]:
            assert witness.witness.weight.data[i].item() != 0.0
        for i in [8, 9, 0]:
            assert witness.witness.weight.data[i].item() == 0.0

    def test_auto_clear_stale_zeros_optimizer_state(self) -> None:
        witness = _DataWitness(num_ids=10)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.1)

        # Run a step to populate optimizer state
        ids = torch.arange(10)
        out = witness(ids)
        out.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        param = witness.witness.weight
        assert param in optimizer.state

        allocator = WitnessIdAllocator(witnesses=[witness], optimizer=optimizer)
        # Allocate 8 → stale IDs (8,9,0) optimizer state should be zeroed
        allocator.allocate_for_sequences(8)

        for i in [8, 9, 0]:
            assert optimizer.state[param]["exp_avg"][i].item() == 0.0
            assert optimizer.state[param]["exp_avg_sq"][i].item() == 0.0


class TestComputeStaleIds:
    """Direct tests for WitnessIdAllocator._compute_stale_ids classmethod."""

    _compute = WitnessIdAllocator._compute_stale_ids

    def test_counter_zero_returns_empty(self) -> None:
        assert self._compute(keep_count=5, counter=0, buffer_size=10) == []

    def test_counter_less_than_keep_count_returns_all_unused(self) -> None:
        # counter=3, buffer=10, keep=7 → active=min(7,3,10)=3 → stale=7 slots
        result = self._compute(keep_count=7, counter=3, buffer_size=10)
        assert set(result) == {3, 4, 5, 6, 7, 8, 9}

    def test_keep_count_equals_buffer_size_returns_empty(self) -> None:
        # All slots are active
        assert self._compute(keep_count=10, counter=15, buffer_size=10) == []

    def test_keep_count_exceeds_buffer_size_returns_empty(self) -> None:
        assert self._compute(keep_count=20, counter=15, buffer_size=10) == []

    def test_basic_no_wrap(self) -> None:
        # counter=8, buffer=10, keep=5 → stale=5, head=8 → stale=[8,9,0,1,2]
        result = self._compute(keep_count=5, counter=8, buffer_size=10)
        assert result == [8, 9, 0, 1, 2]

    def test_basic_wrap(self) -> None:
        # counter=3, buffer=10, keep=5 → active=min(5,3,10)=3 → stale=7, head=3 → stale=[3,4,5,6,7,8,9]
        result = self._compute(keep_count=5, counter=3, buffer_size=10)
        assert result == [3, 4, 5, 6, 7, 8, 9]

    def test_head_at_zero(self) -> None:
        # counter=10, buffer=10, keep=3 → stale=7, head=0 → stale=[0,1,2,3,4,5,6]
        result = self._compute(keep_count=3, counter=10, buffer_size=10)
        assert result == [0, 1, 2, 3, 4, 5, 6]

    def test_keep_one(self) -> None:
        # counter=5, buffer=10, keep=1 → stale=9, head=5 → stale=[5,6,7,8,9,0,1,2,3]
        result = self._compute(keep_count=1, counter=5, buffer_size=10)
        assert len(result) == 9
        assert 4 not in result

    def test_keep_zero(self) -> None:
        # All slots stale
        result = self._compute(keep_count=0, counter=5, buffer_size=10)
        assert len(result) == 10
        assert set(result) == set(range(10))

    def test_stale_and_active_are_disjoint_and_cover_buffer(self) -> None:
        for counter in [0, 1, 5, 10, 13, 20, 100]:
            for keep in [0, 1, 3, 7, 10, 15]:
                stale = self._compute(keep_count=keep, counter=counter, buffer_size=10)
                active_count = min(keep, counter, 10)
                assert len(stale) == 10 - active_count, f"counter={counter}, keep={keep}"
                assert len(set(stale)) == len(stale), "no duplicates"
                assert all(0 <= x < 10 for x in stale), "all in range"

    def test_buffer_size_one(self) -> None:
        assert self._compute(keep_count=1, counter=5, buffer_size=1) == []
        assert self._compute(keep_count=0, counter=5, buffer_size=1) == [0]
