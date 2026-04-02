"""Tests for miles.utils.witness: DataWitness, WitnessIdAllocator."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from miles.utils.witness import _DataWitness, WitnessIdAllocator, _record_and_log_witness_param, install_witness


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
    def _make_allocator(self, witness: _DataWitness, optimizer: torch.optim.Optimizer | None = None) -> WitnessIdAllocator:
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
            torch.full((length,), fill_value=sid, dtype=torch.long)
            for length, sid in zip(token_lengths, seq_ids)
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


# ---------------------------------------------------------------------------
# Fake GPTModel for install_witness / forward integration tests
# ---------------------------------------------------------------------------


class _FakeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_tensor: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states


class _FakeGPTModel(nn.Module):
    def __init__(self, *, pre_process: bool = True) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.decoder = _FakeDecoder()
        self.embedding = nn.Embedding(100, 16)

    def forward(self, input_ids: torch.Tensor, witness_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.pre_process:
            decoder_input = self.embedding(input_ids)
        else:
            decoder_input = None

        if hasattr(self, "head_witness") and witness_ids is not None:
            witness_out = self.head_witness(input_ids, witness_ids)
            if decoder_input is not None:
                decoder_input = decoder_input + witness_out
            else:
                self.decoder.input_tensor = self.decoder.input_tensor + witness_out

        if decoder_input is None:
            decoder_input = self.decoder.input_tensor

        return self.decoder(hidden_states=decoder_input)


class TestInstallWitness:
    def test_witness_is_submodule(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        assert "head_witness" in dict(model.named_modules())
        assert "tail_witness" in dict(model.named_modules())

    def test_witness_in_parameters(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        param_names = [name for name, _ in model.named_parameters()]
        assert any("head_witness" in name for name in param_names)
        assert any("tail_witness" in name for name in param_names)

    def test_forward_adds_zero(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        tokens = torch.tensor([[1, 2, 3]])
        out_no = model(tokens)
        out_with = model(tokens, witness_ids=torch.tensor([[0, 1, 2]]))
        assert torch.equal(out_no, out_with)

    def test_forward_produces_gradient(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        tokens = torch.tensor([[1, 2, 3]])
        out = model(tokens, witness_ids=torch.tensor([[5, 5, 5]]))
        out.sum().backward()
        grad = model.head_witness.witness.weight.grad
        assert grad is not None
        assert 5 in grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    def test_no_witness_ids_no_effect(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        out = model(torch.tensor([[1, 2, 3]]))
        assert out is not None

    def test_witness_in_state_dict(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        sd = model.state_dict()
        assert any("head_witness" in k for k in sd)
        assert any("tail_witness" in k for k in sd)

    def test_checkpoint_roundtrip(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        model.head_witness.witness.weight.data.fill_(42.0)
        sd = model.state_dict()

        model2 = _FakeGPTModel()
        install_witness(model2, buffer_size=10)
        model2.load_state_dict(sd)
        assert torch.equal(model2.head_witness.witness.weight.data, model.head_witness.witness.weight.data)

    def test_disabled_no_submodule(self) -> None:
        model = _FakeGPTModel()
        assert not hasattr(model, "head_witness")

    def test_middle_pp_stage_modifies_input_tensor(self) -> None:
        model = _FakeGPTModel(pre_process=False)
        install_witness(model, buffer_size=10)
        hidden = torch.randn(1, 4, 16)
        model.decoder.input_tensor = hidden.clone()
        out = model(torch.tensor([[1, 2, 3, 4]]), witness_ids=torch.tensor([[0, 1, 2, 3]]))
        assert torch.equal(out, hidden)

    def test_middle_pp_stage_produces_gradient(self) -> None:
        model = _FakeGPTModel(pre_process=False)
        install_witness(model, buffer_size=10)
        model.decoder.input_tensor = torch.randn(1, 4, 16, requires_grad=True)
        out = model(torch.tensor([[1, 2, 3, 4]]), witness_ids=torch.tensor([[5, 5, 5, 5]]))
        out.sum().backward()
        assert 5 in model.head_witness.witness.weight.grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    def test_forward_bitwise_zero_bf16(self) -> None:
        witness = _DataWitness(buffer_size=10).to(dtype=torch.bfloat16)
        witness.witness.weight.data.fill_(3.14)
        ids = torch.tensor([0, 1, 2])
        out = witness(ids, ids)
        assert torch.all(out == 0.0)
