"""Value-head output-width wiring for categorical `--value-loss-type` heads.

Verifies that `SharedValueHead` and `LinearForLastLayer` produce the right
last-dim width for each `value_head_output_size(args)` result (1 for the
default `mse` scalar critic; K bins for hl_gauss/twohot/onehot; 2 for
bernoulli), across both `linear` and `mlp` `SharedValueHead` head types.
Requires a real `megatron.core.TransformerConfig`, so this needs the repo's
Megatron-LM/miles stack importable (not a pure-CPU-only torch env).
"""

from argparse import Namespace

import pytest
import torch
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.backends.megatron_utils.model_provider import LinearForLastLayer, SharedValueHead
from miles.utils.value_head_utils import value_head_output_size


@pytest.fixture
def config() -> TransformerConfig:
    return TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1, sequence_parallel=False)


def _mse_args() -> Namespace:
    return Namespace(value_loss_type="mse")


def _categorical_args(value_loss_type: str, num_bins: int = 51) -> Namespace:
    return Namespace(value_loss_type=value_loss_type, value_num_bins=num_bins, value_min=0.0, value_max=1.0)


def test_shared_value_head_linear_defaults_to_scalar_width(config: TransformerConfig) -> None:
    head = SharedValueHead(input_size=8, config=config, output_size=value_head_output_size(_mse_args()))
    out, _ = head(torch.randn(3, 2, 8))
    assert out.shape == (3, 2, 1)


def test_linear_for_last_layer_defaults_to_scalar_width(config: TransformerConfig) -> None:
    layer = LinearForLastLayer(input_size=8, output_size=value_head_output_size(_mse_args()), config=config)
    out, _ = layer(torch.randn(3, 2, 8))
    assert out.shape == (3, 2, 1)


@pytest.mark.parametrize("value_loss_type,expected_width", [("hl_gauss", 51), ("twohot", 51), ("onehot", 51)])
def test_shared_value_head_linear_matches_num_bins(
    config: TransformerConfig, value_loss_type: str, expected_width: int
) -> None:
    args = _categorical_args(value_loss_type, num_bins=expected_width)
    head = SharedValueHead(input_size=8, config=config, output_size=value_head_output_size(args))
    out, _ = head(torch.randn(3, 2, 8))
    assert out.shape == (3, 2, expected_width)


def test_shared_value_head_mlp_matches_num_bins(config: TransformerConfig) -> None:
    args = _categorical_args("hl_gauss", num_bins=51)
    head = SharedValueHead(
        input_size=8,
        config=config,
        head_type="mlp",
        mlp_num_hidden_layers=2,
        output_size=value_head_output_size(args),
    )
    out, _ = head(torch.randn(3, 2, 8))
    assert out.shape == (3, 2, 51)


def test_linear_for_last_layer_matches_num_bins(config: TransformerConfig) -> None:
    args = _categorical_args("hl_gauss", num_bins=51)
    layer = LinearForLastLayer(input_size=8, output_size=value_head_output_size(args), config=config)
    out, _ = layer(torch.randn(3, 2, 8))
    assert out.shape == (3, 2, 51)


def test_shared_value_head_bernoulli_width_is_two(config: TransformerConfig) -> None:
    args = Namespace(value_loss_type="bernoulli")
    head = SharedValueHead(input_size=8, config=config, output_size=value_head_output_size(args))
    out, _ = head(torch.randn(3, 2, 8))
    assert out.shape == (3, 2, 2)


def test_shared_value_head_sequence_parallel_marks_all_linears() -> None:
    sp_config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
    )
    args = _categorical_args("hl_gauss", num_bins=51)
    head = SharedValueHead(
        input_size=8, config=sp_config, head_type="mlp", mlp_num_hidden_layers=2, output_size=value_head_output_size(args)
    )
    linears = [m for m in head.net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3  # 2 hidden + 1 output
    for linear in linears:
        assert linear.weight.sequence_parallel is True
        assert linear.bias.sequence_parallel is True
