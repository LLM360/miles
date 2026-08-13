import argparse
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.utils.arguments import (
    _maybe_apply_dumper_overrides,
    get_miles_extra_args_provider,
    validate_mova_args,
)
from miles.utils.misc import function_registry

PATH_ARGS = ["--rollout-function-path", "--custom-generate-function-path"]
REQUIRED_ARGS = ["--rollout-batch-size", "64"]


class TestMoVAArguments:
    def _parse(self, flags):
        with patch.object(sys, "argv", ["test", *flags]):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            return parser.parse_args(flags)

    def test_defaults_match_native_megatron_entry_point(self):
        args = self._parse(REQUIRED_ARGS)

        assert args.mova_num_value_experts == 0
        assert args.mova_router_topk == 1
        assert args.mova_router_score_function == "sigmoid"
        assert args.mova_router_topk_scaling_factor == 1.0
        assert args.mova_router_enable_expert_bias is False
        assert args.mova_router_bias_update_rate == 1.0e-3
        assert args.mova_router_aux_loss_coeff == 0.0
        assert args.mova_router_load_balancing_type == "none"
        assert args.mova_num_dense_layers == 0
        assert args.mova_norm_num_groups == 1
        assert args.mova_attention_gate_function == "softplus"
        assert args.mova_value_backend == "grouped_gemm"
        assert args.mova_use_torch_rms_norm is False
        assert args.xllm_router_compatibility is True
        assert args.xllm_router_gemm_partitions == 1

    def test_generated_k2mova_rl_flags_are_accepted(self):
        args = self._parse(
            REQUIRED_ARGS
            + [
                "--mova-num-value-experts",
                "64",
                "--mova-router-topk",
                "4",
                "--mova-router-score-function",
                "sigmoid",
                "--mova-router-topk-scaling-factor",
                "2.5",
                "--mova-router-enable-expert-bias",
                "--mova-router-bias-update-rate",
                "0.001",
                "--mova-router-aux-loss-coeff",
                "0",
                "--mova-router-load-balancing-type",
                "none",
                "--mova-num-dense-layers",
                "3",
                "--mova-norm-num-groups",
                "2",
                "--mova-attention-gate-function",
                "softplus",
                "--mova-value-backend",
                "grouped_gemm",
                "--no-mova-use-torch-rms-norm",
                "--xllm-router-compatibility",
                "--xllm-router-gemm-partitions",
                "1",
            ]
        )

        assert args.mova_num_value_experts == 64
        assert args.mova_router_topk == 4
        assert args.mova_router_topk_scaling_factor == 2.5
        assert args.mova_router_enable_expert_bias is True
        assert args.mova_num_dense_layers == 3
        assert args.mova_norm_num_groups == 2
        assert args.mova_value_backend == "grouped_gemm"
        assert args.mova_use_torch_rms_norm is False
        assert args.xllm_router_compatibility is True

    def test_registration_does_not_change_shared_transformer_defaults(self):
        with patch.object(sys, "argv", ["test", *REQUIRED_ARGS]):
            parser = argparse.ArgumentParser()
            parser.add_argument("--attention-dropout", type=float, default=0.1)
            parser.add_argument("--hidden-dropout", type=float, default=0.1)
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args(REQUIRED_ARGS)

        assert args.attention_dropout == 0.1
        assert args.hidden_dropout == 0.1

    def test_native_validation_accepts_exact_k2mova_contract(self):
        args = SimpleNamespace(
            mova_num_value_experts=64,
            train_backend="megatron",
            megatron_to_hf_mode="raw",
            custom_model_provider_path=None,
            use_legacy_models=False,
            yaml_cfg=None,
            spec=None,
            mtp_num_layers=None,
            multi_latent_attention=False,
            heterogeneous_layers_config_path=None,
            attention_output_gate=True,
            rotary_interleaved=True,
            group_query_attention=True,
            num_experts=100,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            tensor_model_parallel_size=8,
            sequence_parallel=True,
        )

        validate_mova_args(args)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("megatron_to_hf_mode", "bridge", "raw"),
            ("attention_output_gate", False, "attention-output-gate"),
            ("rotary_interleaved", False, "rotary-interleaved"),
            ("attention_dropout", 0.1, "attention-dropout"),
            ("sequence_parallel", False, "sequence-parallel"),
        ],
    )
    def test_native_validation_rejects_incompatible_rl_flags(self, field, value, message):
        args = SimpleNamespace(
            mova_num_value_experts=64,
            train_backend="megatron",
            megatron_to_hf_mode="raw",
            custom_model_provider_path=None,
            use_legacy_models=False,
            yaml_cfg=None,
            spec=None,
            mtp_num_layers=None,
            multi_latent_attention=False,
            heterogeneous_layers_config_path=None,
            attention_output_gate=True,
            rotary_interleaved=True,
            group_query_attention=True,
            num_experts=100,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            tensor_model_parallel_size=8,
            sequence_parallel=True,
        )
        setattr(args, field, value)

        with pytest.raises(ValueError, match=message):
            validate_mova_args(args)


def make_class_with_add_arguments():
    class MyFn:
        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument("--my-custom-arg", type=int, default=42)

    return MyFn


def make_function_with_add_arguments():
    def my_fn():
        pass

    my_fn.add_arguments = lambda parser: parser.add_argument("--my-custom-arg", type=int, default=42)
    return my_fn


def make_function_without_add_arguments():
    def my_fn():
        pass

    return my_fn


@pytest.mark.parametrize("path_arg", PATH_ARGS)
class TestAddArgumentsSupport:

    @pytest.mark.parametrize("fn_factory", [make_class_with_add_arguments, make_function_with_add_arguments])
    def test_add_arguments_is_called_and_arg_is_parsed(self, path_arg, fn_factory):
        fn = fn_factory()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn", "--my-custom-arg", "100"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()
            assert args.my_custom_arg == 100

    def test_skips_function_without_add_arguments(self, path_arg):
        fn = make_function_without_add_arguments()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)


class TestMaybeApplyDumperOverrides:
    def _make_args(
        self,
        *,
        dumper_enable: bool = False,
        use_fault_tolerance: bool = False,
        router_disable_health_check: bool = False,
        rollout_health_check_interval: float = 30.0,
        start_rollout_id: int | None = None,
        num_rollout: int = 10,
        eval_interval: int | None = 5,
        save_interval: int | None = 5,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            dumper_enable=dumper_enable,
            use_fault_tolerance=use_fault_tolerance,
            router_disable_health_check=router_disable_health_check,
            rollout_health_check_interval=rollout_health_check_interval,
            start_rollout_id=start_rollout_id,
            num_rollout=num_rollout,
            eval_interval=eval_interval,
            save_interval=save_interval,
        )

    def test_noop_when_dumper_disabled(self) -> None:
        args = self._make_args(
            dumper_enable=False,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is True
        assert args.router_disable_health_check is False
        assert args.rollout_health_check_interval == 30.0
        assert args.num_rollout == 10
        assert args.eval_interval == 5
        assert args.save_interval == 5

    def test_disables_all_heartbeats(self) -> None:
        args = self._make_args(
            dumper_enable=True,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is False
        assert args.router_disable_health_check is True
        assert args.rollout_health_check_interval == 1e18

    def test_forces_single_rollout(self) -> None:
        args = self._make_args(dumper_enable=True, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.num_rollout == 1
        assert args.eval_interval is None
        assert args.save_interval is None

    def test_respects_start_rollout_id(self) -> None:
        args = self._make_args(dumper_enable=True, start_rollout_id=5, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.num_rollout == 6
