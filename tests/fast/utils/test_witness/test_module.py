TODO_split_to_test_allocator_and_test_module

"""Tests for miles.utils.witness: DataWitness, WitnessIdAllocator."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from miles.utils.witness import WitnessIdAllocator, _DataWitness, _record_and_log_witness_param, install_witness


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
            from miles.utils.event_logger.models import WitnessSnapshotParamEvent

            assert mock_logger.log.call_args[0][0] is WitnessSnapshotParamEvent
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
