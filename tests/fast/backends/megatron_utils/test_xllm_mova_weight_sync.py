from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from miles.backends.megatron_utils.megatron_to_hf.xllm import convert_xllm_to_hf
from miles.backends.megatron_utils.update_weight.common import (
    all_gather_param,
    collect_named_tensors_for_weight_transfer,
    is_ffn_expert_parameter,
    named_params_and_buffers,
    validate_weight_update_cache_mode,
)


def _args(*, num_value_experts: int = 3, pause_generation_mode: str = "retract") -> Namespace:
    return Namespace(
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        mova_num_value_experts=num_value_experts,
        pause_generation_mode=pause_generation_mode,
    )


def _projection(rows: int, hidden_size: int, offset: int) -> torch.Tensor:
    return torch.arange(
        offset,
        offset + rows * hidden_size,
        dtype=torch.float32,
    ).reshape(rows, hidden_size)


def _pack_by_query_group(*projections: torch.Tensor, args: Namespace) -> torch.Tensor:
    query_heads_per_group = args.num_attention_heads // args.num_query_groups
    head_dim = args.kv_channels
    grouped = []
    for index, projection in enumerate(projections):
        heads_per_group = query_heads_per_group if index < 2 else 1
        grouped.append(
            projection.reshape(
                args.num_query_groups,
                heads_per_group,
                head_dim,
                args.hidden_size,
            )
        )
    return torch.cat(grouped, dim=1).reshape(-1, args.hidden_size)


def _permute_qk_to_hf(weight: torch.Tensor, num_heads: int, args: Namespace) -> torch.Tensor:
    return (
        weight.reshape(num_heads, args.kv_channels // 2, 2, args.hidden_size)
        .transpose(1, 2)
        .reshape_as(weight)
    )


def _restore_native_qk(weight: torch.Tensor, num_heads: int, args: Namespace) -> torch.Tensor:
    """Independent inverse used by the Megatron HF checkpoint reader."""

    return (
        weight.reshape(num_heads, 2, args.kv_channels // 2, args.hidden_size)
        .transpose(1, 2)
        .reshape_as(weight)
    )


def test_dense_mova_qkgv_conversion_preserves_gate_and_value() -> None:
    args = _args()
    query = _projection(8, args.hidden_size, 0)
    gate = _projection(8, args.hidden_size, 1_000)
    key = _projection(4, args.hidden_size, 2_000)
    value = _projection(4, args.hidden_size, 3_000)
    packed = _pack_by_query_group(query, gate, key, value, args=args)

    converted = dict(
        convert_xllm_to_hf(
            args,
            "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
            packed,
        )
    )

    torch.testing.assert_close(
        converted["model.layers.0.self_attn.q_proj.weight"],
        _permute_qk_to_hf(query, args.num_attention_heads, args),
    )
    torch.testing.assert_close(
        _restore_native_qk(
            converted["model.layers.0.self_attn.q_proj.weight"],
            args.num_attention_heads,
            args,
        ),
        query,
    )
    torch.testing.assert_close(converted["model.layers.0.self_attn.attn_gate_proj.weight"], gate)
    torch.testing.assert_close(
        converted["model.layers.0.self_attn.k_proj.weight"],
        _permute_qk_to_hf(key, args.num_query_groups, args),
    )
    torch.testing.assert_close(
        _restore_native_qk(
            converted["model.layers.0.self_attn.k_proj.weight"],
            args.num_query_groups,
            args,
        ),
        key,
    )
    torch.testing.assert_close(converted["model.layers.0.self_attn.v_proj.weight"], value)


def test_sparse_mova_qkg_conversion_emits_no_dense_value() -> None:
    args = _args()
    query = _projection(8, args.hidden_size, 0)
    gate = _projection(8, args.hidden_size, 1_000)
    key = _projection(4, args.hidden_size, 2_000)
    packed = _pack_by_query_group(query, gate, key, args=args)

    converted = dict(
        convert_xllm_to_hf(
            args,
            "module.module.decoder.layers.3.self_attention.linear_qkg.weight",
            packed,
        )
    )

    assert set(converted) == {
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.attn_gate_proj.weight",
    }
    torch.testing.assert_close(
        converted["model.layers.3.self_attn.q_proj.weight"],
        _permute_qk_to_hf(query, args.num_attention_heads, args),
    )
    torch.testing.assert_close(converted["model.layers.3.self_attn.attn_gate_proj.weight"], gate)
    torch.testing.assert_close(
        converted["model.layers.3.self_attn.k_proj.weight"],
        _permute_qk_to_hf(key, args.num_query_groups, args),
    )


def test_grouped_value_experts_convert_input_to_output_sharding_layout() -> None:
    args = _args()
    value_width = args.num_query_groups * args.kv_channels
    # MCore all-gather result: [expert, full hidden input, full value output].
    grouped_weight = torch.arange(
        args.mova_num_value_experts * args.hidden_size * value_width,
        dtype=torch.float32,
    ).reshape(args.mova_num_value_experts, args.hidden_size, value_width)

    converted = convert_xllm_to_hf(
        args,
        "module.module.decoder.layers.3.self_attention.value_projection.experts.weight",
        grouped_weight,
    )

    assert [name for name, _ in converted] == [
        f"model.layers.3.self_attn.v_experts.{expert}.weight"
        for expert in range(args.mova_num_value_experts)
    ]
    for expert, (_, weight) in enumerate(converted):
        assert weight.is_contiguous()
        torch.testing.assert_close(weight, grouped_weight[expert].transpose(0, 1))


def test_grouped_value_experts_require_regular_tp_gather_first() -> None:
    args = _args()
    local_input_shard = torch.empty(args.mova_num_value_experts, args.hidden_size // 2, 4)
    with pytest.raises(ValueError, match="gathered across regular attention TP"):
        convert_xllm_to_hf(
            args,
            "module.module.decoder.layers.3.self_attention.value_projection.experts.weight",
            local_input_shard,
        )


@pytest.mark.parametrize(
    ("megatron_suffix", "hf_suffix"),
    [
        ("router.weight", "v_router.weight"),
        ("router.expert_bias", "v_router.bias"),
    ],
)
def test_value_router_parameter_and_buffer_are_both_synchronized(
    megatron_suffix: str, hf_suffix: str
) -> None:
    args = _args()
    tensor = torch.randn(args.mova_num_value_experts, args.hidden_size)
    if megatron_suffix.endswith("expert_bias"):
        tensor = torch.randn(args.mova_num_value_experts)

    converted = convert_xllm_to_hf(
        args,
        f"module.module.decoder.layers.3.self_attention.value_projection.{megatron_suffix}",
        tensor,
    )
    assert len(converted) == 1
    assert converted[0][0] == f"model.layers.3.self_attn.{hf_suffix}"
    assert converted[0][1] is tensor


def test_legacy_xllm_qkv_conversion_is_unchanged_when_mova_is_disabled() -> None:
    args = _args(num_value_experts=0)
    query = _projection(8, args.hidden_size, 0)
    key = _projection(4, args.hidden_size, 1_000)
    value = _projection(4, args.hidden_size, 2_000)
    # Legacy Q/K/V has Q first, so construct its own three-way group packing.
    query_grouped = query.reshape(2, 2, 2, 8)
    key_grouped = key.reshape(2, 1, 2, 8)
    value_grouped = value.reshape(2, 1, 2, 8)
    packed = torch.cat((query_grouped, key_grouped, value_grouped), dim=1).reshape(-1, 8)

    converted = dict(
        convert_xllm_to_hf(
            args,
            "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
            packed,
        )
    )

    torch.testing.assert_close(converted["model.layers.0.self_attn.q_proj.weight"], query)
    torch.testing.assert_close(converted["model.layers.0.self_attn.k_proj.weight"], key)
    torch.testing.assert_close(converted["model.layers.0.self_attn.v_proj.weight"], value)
    assert "model.layers.0.self_attn.attn_gate_proj.weight" not in converted


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("module.module.decoder.layers.3.mlp.experts.linear_fc1.weight7", True),
        ("module.module.mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight7", True),
        ("module.module.decoder.layers.3.self_attention.value_projection.experts.weight", False),
        (
            "module.module.decoder.layers.3.self_attention.value_projection.experts.experts.7.weight",
            False,
        ),
        ("model.layers.3.self_attn.v_experts.7.weight", False),
    ],
)
def test_only_ffn_experts_use_expert_parallelism(name: str, expected: bool) -> None:
    assert is_ffn_expert_parameter(name) is expected


@pytest.mark.parametrize(
    ("name", "expected_group"),
    [
        (
            "module.module.decoder.layers.3.self_attention.value_projection.experts.weight",
            "regular-tp",
        ),
        ("module.module.decoder.layers.3.mlp.experts.weight", "expert-tp"),
    ],
)
def test_tensor_gather_uses_the_correct_tp_group(name: str, expected_group: str) -> None:
    param = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    param.tensor_model_parallel = True
    param.parallel_mode = None
    param.partition_dim = 0
    param.partition_stride = 1

    def fake_all_gather(partitions, source, *, group):
        assert group == expected_group
        partitions[0].copy_(source)

    with (
        patch(
            "miles.backends.megatron_utils.update_weight.common.mpu.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "miles.backends.megatron_utils.update_weight.common.mpu.get_tensor_model_parallel_group",
            return_value="regular-tp",
        ),
        patch(
            "miles.backends.megatron_utils.update_weight.common.mpu.get_expert_tensor_parallel_world_size",
            return_value=1,
        ),
        patch(
            "miles.backends.megatron_utils.update_weight.common.mpu.get_expert_tensor_parallel_group",
            return_value="expert-tp",
        ),
        patch(
            "miles.backends.megatron_utils.update_weight.common.dist.all_gather",
            side_effect=fake_all_gather,
        ),
    ):
        gathered = all_gather_param(Namespace(swiglu=False), name, param)

    torch.testing.assert_close(gathered, param)


def test_weight_transfer_partition_keeps_mova_values_with_non_experts() -> None:
    tensors = [
        ("module.module.decoder.layers.3.mlp.experts.linear_fc1.weight7", torch.tensor(1)),
        (
            "module.module.decoder.layers.3.self_attention.value_projection.experts.weight",
            torch.tensor(2),
        ),
        ("module.module.decoder.layers.3.self_attention.linear_qkg.weight", torch.tensor(3)),
    ]
    with patch(
        "miles.backends.megatron_utils.update_weight.common.named_params_and_buffers",
        return_value=iter(tensors),
    ):
        regular = list(collect_named_tensors_for_weight_transfer(_args(), [], is_expert=False))
    with patch(
        "miles.backends.megatron_utils.update_weight.common.named_params_and_buffers",
        return_value=iter(tensors),
    ):
        experts = list(collect_named_tensors_for_weight_transfer(_args(), [], is_expert=True))

    assert [name for name, _ in regular] == [tensors[1][0], tensors[2][0]]
    assert [name for name, _ in experts] == [tensors[0][0]]


def test_value_and_ffn_expert_bias_buffers_are_enumerated_for_sync() -> None:
    class ModuleWithRouterBuffers(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("value_projection_router_expert_bias", torch.arange(3.0))
            self.register_buffer("mlp_router_expert_bias", torch.arange(4.0))
            self.register_buffer("unrelated_statistics", torch.arange(5.0))

    names = [
        name
        for name, _ in named_params_and_buffers(
            _args(),
            [ModuleWithRouterBuffers()],
            convert_to_global_name=False,
        )
    ]
    assert names == [
        "vp_stages.0.value_projection_router_expert_bias",
        "vp_stages.0.mlp_router_expert_bias",
    ]


def test_p2p_staging_contract_requires_all_value_expert_shards() -> None:
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p import (
        UpdateWeightP2P,
    )

    updater = object.__new__(UpdateWeightP2P)
    num_value_experts = 3
    packed_name = "model.layers.3.self_attn.v_experts.weight"
    updater._shared_params_dict = {packed_name: torch.empty(num_value_experts, 4, 8)}
    updater._shared_param_mapper = MagicMock()
    updater._shared_param_mapper.map.return_value = SimpleNamespace(
        sglang_name=packed_name,
        num_shards=num_value_experts,
        num_local_experts=None,
    )
    updater._staged_tensors = {}
    updater._tensor_update_pending = {}

    ready_names = []
    ready_tensors = []
    for expert in range(num_value_experts):
        names, tensors = updater._get_transfer_ready_params(
            [(f"model.layers.3.self_attn.v_experts.{expert}.weight", torch.full((4, 8), expert))]
        )
        ready_names.extend(names)
        ready_tensors.extend(tensors)

    assert ready_names == [packed_name]
    assert [name for name, _ in ready_tensors] == [
        f"model.layers.3.self_attn.v_experts.{expert}.weight"
        for expert in range(num_value_experts)
    ]
    assert updater._staged_tensors == {}
    assert updater._tensor_update_pending == {}


@pytest.mark.parametrize(
    ("runtime_dtype", "expected_dtype"),
    [
        ("bfloat16", torch.bfloat16),
        ("auto", torch.float16),
    ],
)
def test_p2p_cpu_replica_uses_runtime_rollout_dtype(runtime_dtype: str, expected_dtype: torch.dtype) -> None:
    from contextlib import nullcontext

    from sglang.srt.configs.model_config import _get_and_verify_dtype

    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import p2p

    updater = object.__new__(p2p.UpdateWeightP2P)
    updater._shared_params_dict = {}
    server_args = SimpleNamespace(dtype=runtime_dtype, rl_quant_profile=None)
    observed = {}

    def fake_model_config(model_path: str, dtype: str = "auto") -> SimpleNamespace:
        # The MoVA HF artifact declares float32. SGLang resolves that to fp16
        # for auto, but must honor an explicit bfloat16 rollout dtype.
        hf_config = {"model_type": "xllm", "torch_dtype": "float32"}
        observed["model_path"] = model_path
        observed["requested_dtype"] = dtype
        return SimpleNamespace(dtype=_get_and_verify_dtype(hf_config, dtype))

    def fake_get_model(*, model_config, load_config, device_config) -> nn.Module:
        observed["validation_dtype"] = model_config.dtype
        return nn.Module()

    with (
        patch.object(p2p, "ModelConfig", side_effect=fake_model_config),
        patch.object(p2p, "LoadConfig", return_value=object()),
        patch.object(p2p, "DeviceConfig", return_value=object()),
        patch.object(p2p, "ParallelismContext", side_effect=lambda _: nullcontext()),
        patch.object(p2p, "get_model", side_effect=fake_get_model),
        patch.object(p2p, "initialize_moe_config"),
        patch.object(p2p, "initialize_fp8_gemm_config"),
        patch.object(p2p, "initialize_fp4_gemm_config"),
        patch.object(p2p.torch.cuda, "empty_cache"),
    ):
        updater.create_cpu_replica(
            parallelism_config=object(),
            model_path="/models/mova-float32-config",
            server_args=server_args,
        )

    assert observed == {
        "model_path": "/models/mova-float32-config",
        "requested_dtype": runtime_dtype,
        "validation_dtype": expected_dtype,
    }


def test_broadcast_path_preserves_all_converted_value_expert_metadata() -> None:
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
        update_weights_from_distributed,
    )

    tensors = [
        (f"model.layers.3.self_attn.v_experts.{expert}.weight", torch.full((4, 8), expert))
        for expert in range(3)
    ]
    engine = MagicMock()
    engine.update_weights_from_distributed.remote.return_value = "engine-ref"
    handle = MagicMock()

    with patch(
        "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast.dist.broadcast",
        return_value=handle,
    ) as broadcast:
        refs = update_weights_from_distributed(
            "mova-test",
            MagicMock(),
            7,
            [engine],
            tensors,
        )

    assert refs == ["engine-ref"]
    kwargs = engine.update_weights_from_distributed.remote.call_args.kwargs
    assert kwargs["names"] == [name for name, _ in tensors]
    assert kwargs["dtypes"] == [tensor.dtype for _, tensor in tensors]
    assert kwargs["shapes"] == [tensor.shape for _, tensor in tensors]
    assert kwargs["weight_version"] == "7"
    assert broadcast.call_count == len(tensors)
    assert handle.wait.call_count == len(tensors)


@pytest.mark.parametrize("mode", ["retract", "abort"])
def test_mova_cache_safe_update_modes(mode: str) -> None:
    validate_weight_update_cache_mode(_args(pause_generation_mode=mode))


def test_mova_rejects_in_place_cache_preservation() -> None:
    with pytest.raises(ValueError, match="routed-value KV cache must be flushed"):
        validate_weight_update_cache_mode(_args(pause_generation_mode="in_place"))


def test_legacy_model_may_retain_existing_in_place_behavior() -> None:
    validate_weight_update_cache_mode(_args(num_value_experts=0, pause_generation_mode="in_place"))
