"""Check the xLLM checkpoint contract without importing GPU quantizers."""

from argparse import Namespace
from pathlib import Path

import pytest
import torch

from miles.utils.external_utils.model_args_utils import import_module_from_path

_ROOT = Path(__file__).resolve().parents[4]
_CONVERTER = import_module_from_path(
    _ROOT / "miles/backends/megatron_utils/megatron_to_hf/xllm.py", "test_xllm_converter"
).convert_xllm_to_hf
_ARGS = Namespace(hidden_size=8, num_attention_heads=4, num_query_groups=2, kv_channels=2)
_LAYER = "module.module.decoder.layers.3."


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("module.module.embedding.word_embeddings.weight", "model.embed_tokens.weight"),
        ("module.module.output_layer.weight", "lm_head.weight"),
        ("module.module.decoder.final_layernorm.weight", "model.norm.weight"),
        (_LAYER + "self_attention.linear_proj.weight", "model.layers.3.self_attn.o_proj.weight"),
        (_LAYER + "input_layernorm.weight", "model.layers.3.input_layernorm.weight"),
        (_LAYER + "self_attention.linear_qkv.layer_norm_weight", "model.layers.3.input_layernorm.weight"),
        (_LAYER + "pre_mlp_layernorm.weight", "model.layers.3.post_attention_layernorm.weight"),
        (_LAYER + "mlp.linear_fc1.layer_norm_weight", "model.layers.3.post_attention_layernorm.weight"),
        (_LAYER + "mlp.router.weight", "model.layers.3.mlp.gate.weight"),
        (_LAYER + "mlp.router.expert_bias", "model.layers.3.mlp.gate.bias"),
        (_LAYER + "mlp.linear_fc2.weight", "model.layers.3.mlp.down_proj.weight"),
        (_LAYER + "mlp.experts.linear_fc2.weight7", "model.layers.3.mlp.experts.7.down_proj.weight"),
        (_LAYER + "mlp.shared_experts.linear_fc2.weight", "model.layers.3.mlp.shared_experts.down_proj.weight"),
    ],
)
def test_checkpoint_names_preserve_tensor(source, target):
    tensor = torch.arange(16).reshape(2, 8)
    [(name, converted)] = _CONVERTER(_ARGS, source, tensor)
    assert name == target
    assert converted is tensor


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("mlp.linear_fc1.weight", "mlp"),
        ("mlp.experts.linear_fc1.weight7", "mlp.experts.7"),
        ("mlp.shared_experts.linear_fc1.weight", "mlp.shared_experts"),
    ],
)
def test_gate_and_up_weights_keep_their_order(source, target):
    gate = torch.full((3, 8), 17.0)
    up = torch.full((3, 8), 29.0)
    converted = dict(_CONVERTER(_ARGS, _LAYER + source, torch.cat([gate, up])))
    assert set(converted) == {f"model.layers.3.{target}.gate_proj.weight", f"model.layers.3.{target}.up_proj.weight"}
    torch.testing.assert_close(converted[f"model.layers.3.{target}.gate_proj.weight"], gate)
    torch.testing.assert_close(converted[f"model.layers.3.{target}.up_proj.weight"], up)


@pytest.mark.parametrize("kv_channels", [2, None, "missing"])
def test_grouped_query_attention_restores_q_k_v_order(kv_channels):
    # Megatron packs q0,q1,k0,v0,q2,q3,k1,v1 for these two query groups.
    q = torch.arange(64).reshape(8, 8)
    k = torch.arange(32).reshape(4, 8) + 100
    v = torch.arange(32).reshape(4, 8) + 200
    packed = torch.cat([q[:4], k[:2], v[:2], q[4:], k[2:], v[2:]])
    args = Namespace(**vars(_ARGS))
    if kv_channels == "missing":
        del args.kv_channels
    else:
        args.kv_channels = kv_channels
    converted = dict(_CONVERTER(args, _LAYER + "self_attention.linear_qkv.weight", packed))
    assert set(converted) == {f"model.layers.3.self_attn.{name}_proj.weight" for name in ("q", "k", "v")}
    for name, expected in [("q", q), ("k", k), ("v", v)]:
        torch.testing.assert_close(converted[f"model.layers.3.self_attn.{name}_proj.weight"], expected)


def test_unknown_parameter_fails_instead_of_silently_dropping_weights():
    with pytest.raises(ValueError, match="Unknown parameter name"):
        _CONVERTER(_ARGS, _LAYER + "unsupported.weight", torch.zeros(1))
