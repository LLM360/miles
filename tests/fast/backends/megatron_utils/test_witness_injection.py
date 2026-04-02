"""Tests for witness injection into GPTModel via forward parameter."""

import torch
import torch.nn as nn

from miles.utils.witness import DataWitness, install_witness


class _FakeDecoder(nn.Module):
    """Minimal decoder stub that just returns its input."""

    def __init__(self) -> None:
        super().__init__()
        self.input_tensor: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states


class _FakeGPTModel(nn.Module):
    """Minimal GPTModel stub that mirrors Megatron's witness_ids handling."""

    def __init__(self, *, pre_process: bool = True) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.decoder = _FakeDecoder()
        self.embedding = nn.Embedding(100, 16)

    def forward(self, input_ids: torch.Tensor, witness_ids: torch.Tensor | None = None) -> torch.Tensor:
        # Simulate _preprocess
        if self.pre_process:
            decoder_input = self.embedding(input_ids)
        else:
            decoder_input = None

        # Witness injection (mirrors Megatron GPTModel.forward)
        if hasattr(self, 'head_witness') and witness_ids is not None:
            witness_out = self.head_witness(witness_ids)
            if decoder_input is not None:
                decoder_input = decoder_input + witness_out
            else:
                self.decoder.input_tensor = self.decoder.input_tensor + witness_out

        if decoder_input is None:
            decoder_input = self.decoder.input_tensor

        return self.decoder(hidden_states=decoder_input)


class TestInstallWitnessHook:
    def test_witness_is_submodule(self) -> None:
        model = _FakeGPTModel()
        witness = DataWitness(num_ids=10)
        install_witness(model, witness)

        assert hasattr(model, "head_witness")
        assert model.head_witness is witness
        assert "head_witness" in dict(model.named_modules())

    def test_witness_in_parameters(self) -> None:
        model = _FakeGPTModel()
        witness = DataWitness(num_ids=10)
        install_witness(model, witness)

        param_names = [name for name, _ in model.named_parameters()]
        assert any("head_witness" in name for name in param_names)

    def test_hook_adds_zero_to_decoder_input(self) -> None:
        model = _FakeGPTModel()
        witness = DataWitness(num_ids=10)
        install_witness(model, witness)

        tokens = torch.tensor([[1, 2, 3]])
        witness_ids = torch.tensor([[0, 1, 2]])

        # Forward without witness
        out_no_witness = model(tokens)

        # Forward with witness (should add zero)
        out_with_witness = model(tokens, witness_ids=witness_ids)

        assert torch.equal(out_no_witness, out_with_witness)

    def test_hook_produces_gradient_on_witness(self) -> None:
        model = _FakeGPTModel()
        witness = DataWitness(num_ids=10)
        install_witness(model, witness)

        tokens = torch.tensor([[1, 2, 3]])
        witness_ids = torch.tensor([[5, 5, 5]])

        out = model(tokens, witness_ids=witness_ids)
        loss = out.sum()
        loss.backward()

        grad = witness.witness.weight.grad
        assert grad is not None
        nonzero = grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()
        assert 5 in nonzero

    def test_no_witness_ids_no_effect(self) -> None:
        model = _FakeGPTModel()
        witness = DataWitness(num_ids=10)
        install_witness(model, witness)

        tokens = torch.tensor([[1, 2, 3]])
        out = model(tokens)  # No witness_ids
        assert out is not None

    def test_witness_in_state_dict(self) -> None:
        model = _FakeGPTModel()
        witness = DataWitness(num_ids=10)
        install_witness(model, witness)

        sd = model.state_dict()
        assert any("head_witness" in k for k in sd.keys())

    def test_witness_checkpoint_roundtrip(self) -> None:
        model = _FakeGPTModel()
        witness = DataWitness(num_ids=10)
        install_witness(model, witness)

        # Make witness weights nonzero
        model.head_witness.witness.weight.data.fill_(42.0)

        sd = model.state_dict()

        # Create new model and load
        model2 = _FakeGPTModel()
        witness2 = DataWitness(num_ids=10)
        install_witness(model2, witness2)
        model2.load_state_dict(sd)

        assert torch.equal(model2.head_witness.witness.weight.data, model.head_witness.witness.weight.data)

    def test_witness_disabled_no_submodule(self) -> None:
        model = _FakeGPTModel()
        # Don't install witness
        assert not hasattr(model, "head_witness")

    def test_witness_middle_pp_stage_modifies_input_tensor(self) -> None:
        """Non-pre_process stage: witness should modify decoder.input_tensor, not decoder_input."""
        model = _FakeGPTModel(pre_process=False)
        witness = DataWitness(num_ids=10)
        install_witness(model, witness)

        # Simulate hidden states from previous PP stage
        hidden = torch.randn(1, 4, 16)
        model.decoder.input_tensor = hidden.clone()

        tokens = torch.tensor([[1, 2, 3, 4]])
        witness_ids = torch.tensor([[0, 1, 2, 3]])

        out = model(tokens, witness_ids=witness_ids)
        # Output should equal hidden (witness adds zero)
        assert torch.equal(out, hidden)

    def test_witness_middle_pp_stage_produces_gradient(self) -> None:
        model = _FakeGPTModel(pre_process=False)
        witness = DataWitness(num_ids=10)
        install_witness(model, witness)

        hidden = torch.randn(1, 4, 16, requires_grad=True)
        model.decoder.input_tensor = hidden

        tokens = torch.tensor([[1, 2, 3, 4]])
        witness_ids = torch.tensor([[5, 5, 5, 5]])

        out = model(tokens, witness_ids=witness_ids)
        out.sum().backward()

        grad = witness.witness.weight.grad
        assert grad is not None
        nonzero = grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()
        assert 5 in nonzero

    def test_witness_forward_bitwise_zero_bf16(self) -> None:
        witness = DataWitness(num_ids=10).to(dtype=torch.bfloat16)
        # Make weights nonzero
        witness.witness.weight.data.fill_(3.14)
        ids = torch.tensor([0, 1, 2])
        out = witness(ids)
        assert torch.all(out == 0.0)
