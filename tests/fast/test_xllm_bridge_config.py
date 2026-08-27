import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_bridge_module():
    mbridge_mod = types.ModuleType("mbridge")
    mbridge_core_mod = types.ModuleType("mbridge.core")
    mbridge_models_mod = types.ModuleType("mbridge.models")

    def register_model(_name):
        def decorator(cls):
            return cls

        return decorator

    class Qwen2Bridge:
        _MLP_MAPPING = {}

    class Qwen2MoEBridge:
        _MLP_MAPPING = {}

        def _build_base_config(self, **kwargs):
            return kwargs

        def _get_gptmodel_args(self):
            return {"base": "preserved"}

    mbridge_core_mod.register_model = register_model
    mbridge_models_mod.Qwen2Bridge = Qwen2Bridge
    mbridge_models_mod.Qwen2MoEBridge = Qwen2MoEBridge

    sys.modules["mbridge"] = mbridge_mod
    sys.modules["mbridge.core"] = mbridge_core_mod
    sys.modules["mbridge.models"] = mbridge_models_mod

    module_path = Path(__file__).resolve().parents[2] / "miles_plugins" / "mbridge" / "xllm.py"
    module_name = "test_xllm_bridge_config_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bridge(*, head_dim=128, rope_head_dim=128):
    module = _load_bridge_module()
    bridge = module.XllmBridge.__new__(module.XllmBridge)
    bridge.hf_config = types.SimpleNamespace(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        hidden_size=4096,
        num_attention_heads=32,
        num_experts=0,
    )
    return bridge


def test_full_rope_is_a_model_argument_not_a_transformer_config_argument():
    bridge = _bridge()

    config = bridge._build_config()
    model_args = bridge._get_gptmodel_args()

    assert "rotary_percent" not in config
    assert config["xllm_partial_rope_layout"] is False
    assert model_args == {"base": "preserved", "rotary_percent": 1.0}


def test_missing_rope_head_dimension_means_full_rope():
    bridge = _bridge(rope_head_dim=None)

    assert bridge._build_config()["xllm_partial_rope_layout"] is False
    assert bridge._get_gptmodel_args()["rotary_percent"] == 1.0


def test_half_head_partial_rope_enables_xllm_layout_and_model_fraction():
    bridge = _bridge(rope_head_dim=64)

    config = bridge._build_config()
    model_args = bridge._get_gptmodel_args()

    assert "rotary_percent" not in config
    assert config["xllm_partial_rope_layout"] is True
    assert model_args["rotary_percent"] == 0.5


@pytest.mark.parametrize(
    ("head_dim", "rope_head_dim"),
    [(128, 96), (128, 0), (128, 256), (0, 0)],
)
def test_unsupported_rope_dimensions_fail_closed(head_dim, rope_head_dim):
    bridge = _bridge(head_dim=head_dim, rope_head_dim=rope_head_dim)

    with pytest.raises(ValueError, match="xLLM"):
        bridge._build_config()

    with pytest.raises(ValueError, match="xLLM"):
        bridge._get_gptmodel_args()
