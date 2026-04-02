"""Tests for miles.utils.witness: DataWitness, WitnessIdAllocator."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from miles.utils.witness import _DataWitness, WitnessIdAllocator, _record_and_log_witness_param


class TestDataWitnessForward:
    def test_forward_is_bitwise_zero(self) -> None:
        witness = _DataWitness(num_ids=10)
        ids = torch.tensor([0, 1, 2, 3])
        out = witness(ids)
        assert torch.all(out == 0.0)
        assert out.shape == (4, 1)

    def test_forward_zero_after_optimizer_step(self) -> None:
        witness = _DataWitness(num_ids=10)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.1)

        ids = torch.tensor([0, 1, 2])
        out = witness(ids)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # After optimizer update, weights are nonzero, but output is still zero
        assert not torch.all(witness.witness.weight == 0.0)
        out2 = witness(ids)
        assert torch.all(out2 == 0.0)

    def test_backward_records_gradient(self) -> None:
        witness = _DataWitness(num_ids=10)
        ids = torch.tensor([2, 5])

        # Need a downstream loss to propagate grads
        out = witness(ids)
        loss = out.sum()
        loss.backward()

        grad = witness.witness.weight.grad
        assert grad is not None

        # Only IDs 2 and 5 should have nonzero grad
        nonzero_rows = grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()
        assert set(nonzero_rows) == {2, 5}

    def test_no_effect_on_main_model(self) -> None:
        torch.manual_seed(42)
        embed = nn.Embedding(100, 16)
        linear = nn.Linear(16, 1)

        tokens = torch.tensor([1, 2, 3, 4])

        # Compute loss without witness
        torch.manual_seed(0)
        out_no_witness = linear(embed(tokens)).sum()
        out_no_witness.backward()
        grad_embed_no = embed.weight.grad.clone()
        grad_linear_no = linear.weight.grad.clone()

        embed.zero_grad()
        linear.zero_grad()

        # Compute loss with witness
        witness = _DataWitness(num_ids=10)
        ids = torch.tensor([0, 0, 0, 0])
        witness_out = witness(ids)  # (4, 1) of zeros

        torch.manual_seed(0)
        h = embed(tokens)  # (4, 16)
        h = h + witness_out  # broadcast (4,1) → adds 0
        out_with_witness = linear(h).sum()
        out_with_witness.backward()

        assert torch.equal(grad_embed_no, embed.weight.grad)
        assert torch.equal(grad_linear_no, linear.weight.grad)


class TestWitnessIdAllocator:
    def test_monotonic_and_wraps(self) -> None:
        witness = _DataWitness(num_ids=5)
        allocator = WitnessIdAllocator(ring_buffer_size=5, witnesses=[witness])

        ids1 = allocator.allocate_for_sequences(3)
        assert ids1 == [0, 1, 2]

        ids2 = allocator.allocate_for_sequences(4)
        assert ids2 == [3, 4, 0, 1]

    def test_allocate_for_sequences_returns_correct_count(self) -> None:
        witness = _DataWitness(num_ids=100)
        allocator = WitnessIdAllocator(ring_buffer_size=100, witnesses=[witness])
        ids = allocator.allocate_for_sequences(7)
        assert len(ids) == 7

    def test_per_sequence_id_same_for_all_tokens(self) -> None:
        witness = _DataWitness(num_ids=100)
        allocator = WitnessIdAllocator(ring_buffer_size=100, witnesses=[witness])
        seq_ids = allocator.allocate_for_sequences(3)

        token_lengths = [10, 20, 15]
        witness_ids_list = [
            torch.full((length,), fill_value=sid, dtype=torch.long)
            for length, sid in zip(token_lengths, seq_ids)
        ]

        for wids, sid in zip(witness_ids_list, seq_ids):
            assert torch.all(wids == sid)

    def test_clear_stale_zeros_oldest_ids(self) -> None:
        witness = _DataWitness(num_ids=5)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.1)

        # Make all weights nonzero
        witness.witness.weight.data.fill_(1.0)

        allocator = WitnessIdAllocator(ring_buffer_size=5, witnesses=[witness])
        # Allocate 7 IDs → counter=7, head=2
        # IDs allocated: [0,1,2,3,4,0,1], last 2 are [0,1]
        allocator.allocate_for_sequences(7)

        # Keep only the 2 most recently allocated IDs (0, 1)
        allocator.clear_stale(optimizer=optimizer, keep_count=2)

        # IDs 2,3,4 are stale (not in last 2), should be zeroed
        for i in [2, 3, 4]:
            assert witness.witness.weight.data[i].item() == 0.0

        # IDs 0,1 are fresh (most recently allocated), should be nonzero
        for i in [0, 1]:
            assert witness.witness.weight.data[i].item() != 0.0

    def test_clear_stale_zeros_optimizer_state(self) -> None:
        witness = _DataWitness(num_ids=5)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.1)

        # Run a step to populate optimizer state
        ids = torch.tensor([0, 1, 2, 3, 4])
        out = witness(ids)
        out.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        # Verify optimizer state exists
        param = witness.witness.weight
        assert param in optimizer.state
        assert "exp_avg" in optimizer.state[param]

        allocator = WitnessIdAllocator(ring_buffer_size=5, witnesses=[witness])
        allocator.allocate_for_sequences(7)
        allocator.clear_stale(optimizer=optimizer, keep_count=2)

        # Stale IDs (2,3,4) optimizer state should be zeroed
        for i in [2, 3, 4]:
            assert optimizer.state[param]["exp_avg"][i].item() == 0.0
            assert optimizer.state[param]["exp_avg_sq"][i].item() == 0.0

    def test_clear_stale_no_stale_when_all_recent(self) -> None:
        witness = _DataWitness(num_ids=5)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.1)

        witness.witness.weight.data.fill_(1.0)

        allocator = WitnessIdAllocator(ring_buffer_size=5, witnesses=[witness])
        allocator.allocate_for_sequences(7)

        # Keep all 5 (ring_buffer_size) → nothing stale
        allocator.clear_stale(optimizer=optimizer, keep_count=5)

        for i in range(5):
            assert witness.witness.weight.data[i].item() != 0.0


class TestRecordAndLogWitnessParam:
    def test_logs_nonzero_weight_rows(self) -> None:
        witness = _DataWitness(num_ids=10)
        witness.witness.weight.data[3] = 1.0
        witness.witness.weight.data[7] = 2.0

        with patch("miles.utils.witness.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _record_and_log_witness_param(step=0, witness=witness, position="head_witness")

            mock_logger.log.assert_called_once()
            # New API: log(event_cls, partial_dict)
            partial = mock_logger.log.call_args[0][1]
            assert set(partial["nonzero_ids"]) == {3, 7}
            assert partial["position"] == "head_witness"

    def test_record_and_log_event_fields(self) -> None:
        witness = _DataWitness(num_ids=10)
        witness.witness.weight.data[1] = 0.5
        witness.witness.weight.data[4] = -0.3

        with patch("miles.utils.witness.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _record_and_log_witness_param(step=5, witness=witness, position="tail_witness")

            mock_logger.log.assert_called_once()
            from miles.utils.event_logger.models import WitnessEvent

            assert mock_logger.log.call_args[0][0] is WitnessEvent
            partial = mock_logger.log.call_args[0][1]
            assert partial["step"] == 5
            assert partial["position"] == "tail_witness"
            assert set(partial["nonzero_ids"]) == {1, 4}
