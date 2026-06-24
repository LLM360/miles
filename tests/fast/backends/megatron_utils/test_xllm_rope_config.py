"""Exercise the real bridge configuration with CPU-only mbridge parent doubles."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("experts", [0, 192])
@pytest.mark.parametrize("rope_dim,percent,partial", [(None, 1.0, False), (64, 0.5, True), (32, 0.25, False)])
def test_checkpoint_rope_dimensions_reach_megatron(monkeypatch, experts, rope_dim, percent, partial):
    class Parent:
        _MLP_MAPPING = {}

        def _build_base_config(self, **kwargs):
            return kwargs

    core = ModuleType("mbridge.core")
    core.register_model = lambda name: lambda cls: cls
    models = ModuleType("mbridge.models")
    models.Qwen2Bridge = models.Qwen2MoEBridge = Parent
    monkeypatch.setitem(sys.modules, "mbridge", ModuleType("mbridge"))
    monkeypatch.setitem(sys.modules, "mbridge.core", core)
    monkeypatch.setitem(sys.modules, "mbridge.models", models)
    path = Path(__file__).parents[4] / "miles_plugins/mbridge/xllm.py"
    spec = importlib.util.spec_from_file_location("xllm_rope_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bridge = module.XllmBridge()
    bridge.hf_config = SimpleNamespace(
        hidden_size=6144,
        num_attention_heads=48,
        num_experts=experts,
        moe_intermediate_size=1792,
        num_experts_per_tok=8,
    )
    if rope_dim is not None:
        bridge.hf_config.head_dim = 128
        bridge.hf_config.rope_head_dim = rope_dim
    config = bridge._build_config()
    assert config["rotary_percent"] == percent
    assert config["xllm_partial_rope_layout"] is partial
    assert ("num_moe_experts" in config) == bool(experts)
