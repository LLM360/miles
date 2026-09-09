from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miles.backends.megatron_utils import model_provider as model_provider_module


def _provider_args(*, mova_num_value_experts: int) -> Namespace:
    return Namespace(
        custom_model_provider_path=None,
        mova_num_value_experts=mova_num_value_experts,
        train_backend="megatron",
        megatron_to_hf_mode="raw",
        use_legacy_models=False,
        yaml_cfg=None,
        spec=None,
        mtp_num_layers=None,
        multi_latent_attention=False,
        heterogeneous_layers_config_path=None,
        attention_output_gate=True,
        rotary_interleaved=True,
        group_query_attention=True,
        num_experts=100 if mova_num_value_experts else 0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tensor_model_parallel_size=8,
        sequence_parallel=True,
        transformer_impl="transformer_engine",
        moe_grouped_gemm=True,
        moe_use_legacy_grouped_gemm=False,
        qk_layernorm=False,
        fp8_param_gather=False,
        padded_vocab_size=256,
        max_position_embeddings=32768,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10_000_000,
        use_rope_scaling=False,
        use_rollout_routing_replay=False,
    )


def test_native_mova_provider_uses_mova_config_and_heterogeneous_spec(monkeypatch):
    args = _provider_args(mova_num_value_experts=64)
    fake_config_class = type("FakeMoVATransformerConfig", (), {})
    fake_config = SimpleNamespace(hidden_size=2560)
    config_builder = MagicMock(return_value=fake_config)
    spec_builder = MagicMock(return_value="mova-block-spec")
    gpt_model = MagicMock(return_value=SimpleNamespace(config=fake_config))

    monkeypatch.setattr(model_provider_module, "core_transformer_config_from_args", config_builder)
    monkeypatch.setattr(
        model_provider_module,
        "_get_mova_model_components",
        lambda: (fake_config_class, spec_builder),
    )
    monkeypatch.setattr(model_provider_module, "GPTModel", gpt_model)

    provider = model_provider_module.get_model_provider_func(args, role="actor")
    model = provider(pre_process=False, post_process=True, vp_stage=2)

    assert model.config is fake_config
    config_builder.assert_called_once_with(args, fake_config_class)
    spec_builder.assert_called_once_with(
        fake_config,
        use_transformer_engine=True,
        moe_grouped_gemm=True,
        moe_use_legacy_grouped_gemm=False,
        vp_stage=2,
    )
    gpt_model.assert_called_once_with(
        config=fake_config,
        transformer_layer_spec="mova-block-spec",
        vocab_size=256,
        max_sequence_length=32768,
        pre_process=False,
        post_process=True,
        fp16_lm_cross_entropy=False,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10_000_000,
        rope_scaling=False,
        vp_stage=2,
    )


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_actor_and_critic_share_native_mova_provider(monkeypatch, role):
    args = _provider_args(mova_num_value_experts=64)
    fake_config_class = type("FakeMoVATransformerConfig", (), {})
    fake_config = SimpleNamespace(hidden_size=2560, sequence_parallel=True)
    spec_builder = MagicMock(return_value="mova-block-spec")
    model = SimpleNamespace(config=fake_config)
    model.output_layer = "original-output-layer"

    monkeypatch.setattr(
        model_provider_module,
        "core_transformer_config_from_args",
        MagicMock(return_value=fake_config),
    )
    monkeypatch.setattr(
        model_provider_module,
        "_get_mova_model_components",
        lambda: (fake_config_class, spec_builder),
    )
    monkeypatch.setattr(model_provider_module, "GPTModel", MagicMock(return_value=model))
    critic_head = MagicMock(return_value="critic-output-layer")
    monkeypatch.setattr(model_provider_module, "LinearForLastLayer", critic_head)

    provider = model_provider_module.get_model_provider_func(args, role=role)
    result = provider()

    assert result is model
    spec_builder.assert_called_once()
    if role == "critic":
        critic_head.assert_called_once_with(
            input_size=2560,
            output_size=1,
            config=fake_config,
        )
        assert model.output_layer == "critic-output-layer"
    else:
        critic_head.assert_not_called()
        assert model.output_layer == "original-output-layer"


def test_non_mova_provider_preserves_standard_transformer_path(monkeypatch):
    args = _provider_args(mova_num_value_experts=0)
    fake_config = SimpleNamespace(hidden_size=2560)
    config_builder = MagicMock(return_value=fake_config)
    te_spec_builder = MagicMock(return_value="standard-te-layer-spec")
    gpt_model = MagicMock(return_value=SimpleNamespace(config=fake_config))
    mova_components = MagicMock(side_effect=AssertionError("legacy path imported MoVA"))

    monkeypatch.setattr(model_provider_module, "core_transformer_config_from_args", config_builder)
    monkeypatch.setattr(model_provider_module, "get_gpt_layer_with_transformer_engine_spec", te_spec_builder)
    monkeypatch.setattr(model_provider_module, "_get_mova_model_components", mova_components)
    monkeypatch.setattr(model_provider_module, "GPTModel", gpt_model)

    provider = model_provider_module.get_model_provider_func(args, role="actor")
    provider(pre_process=True, post_process=False)

    config_builder.assert_called_once_with(args)
    mova_components.assert_not_called()
    te_spec_builder.assert_called_once_with(
        num_experts=0,
        moe_grouped_gemm=True,
        qk_layernorm=False,
        multi_latent_attention=False,
        moe_use_legacy_grouped_gemm=False,
    )
    assert gpt_model.call_args.kwargs["transformer_layer_spec"] == "standard-te-layer-spec"
    assert "vp_stage" not in gpt_model.call_args.kwargs


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("megatron_to_hf_mode", "bridge", "raw"),
        ("custom_model_provider_path", "some.module.provider", "owns its model provider"),
        ("spec", "some.module.spec", "heterogeneous layer spec"),
        ("mtp_num_layers", 1, "MTP"),
    ],
)
def test_mova_provider_fails_before_selecting_incompatible_provider(field, value, message):
    args = _provider_args(mova_num_value_experts=64)
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        model_provider_module.get_model_provider_func(args)


def test_mova_components_report_incompatible_megatron_cleanly(monkeypatch):
    real_import = __import__

    def reject_mova(name, *args, **kwargs):
        if name in {
            "megatron.core.models.gpt.mova_layer_specs",
            "megatron.core.transformer.mova",
        }:
            raise ImportError("no native MoVA")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_mova)

    with pytest.raises(RuntimeError, match="does not provide MoVATransformerConfig"):
        model_provider_module._get_mova_model_components()
