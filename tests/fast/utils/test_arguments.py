import argparse
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.utils.arguments import (
    _maybe_apply_dumper_overrides,
    apply_value_pretrain_args,
    get_miles_extra_args_provider,
)
from miles.utils.misc import function_registry

PATH_ARGS = ["--rollout-function-path", "--custom-generate-function-path"]
REQUIRED_ARGS = ["--rollout-batch-size", "64"]


@pytest.mark.parametrize("reward_type", ["logr", "k3"])
def test_opd_reward_type_is_parsed(reward_type: str) -> None:
    with patch.object(sys, "argv", ["test", "--opd-reward-type", reward_type] + REQUIRED_ARGS):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args, _ = parser.parse_known_args()

    assert args.opd_reward_type == reward_type


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


def test_value_pretrain_flag_is_parsed() -> None:
    with patch.object(sys, "argv", ["test", "--value-pretrain"] + REQUIRED_ARGS):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args, _ = parser.parse_known_args()

    assert args.value_pretrain is True


def test_apply_value_pretrain_args_forces_mc_critic_path() -> None:
    args = SimpleNamespace(
        value_pretrain=True,
        debug_train_only=False,
        advantage_estimator="grpo",
        compute_advantages_and_returns=False,
        kl_coef=0.1,
        n_samples_per_prompt=8,
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        num_critic_only_steps=0,
        num_rollout=12,
    )
    apply_value_pretrain_args(args)

    assert args.debug_train_only is True
    assert args.advantage_estimator == "ppo"
    assert args.compute_advantages_and_returns is True
    assert args.kl_coef == 0.0
    assert args.n_samples_per_prompt == 1
    assert args.rollout_function_path == "miles.rollout.value_pretrain_rollout.generate_rollout"
    assert args.num_critic_only_steps == 12


def test_apply_value_pretrain_args_keeps_custom_rollout_and_is_noop_when_off() -> None:
    custom = "miles.rollout.custom.generate_rollout"
    args = SimpleNamespace(
        value_pretrain=True,
        debug_train_only=False,
        advantage_estimator="grpo",
        compute_advantages_and_returns=False,
        kl_coef=0.0,
        n_samples_per_prompt=1,
        rollout_function_path=custom,
        num_critic_only_steps=3,
        num_rollout=None,
    )
    apply_value_pretrain_args(args)
    assert args.rollout_function_path == custom
    assert args.num_critic_only_steps == 10**9

    off = SimpleNamespace(value_pretrain=False, advantage_estimator="grpo")
    apply_value_pretrain_args(off)
    assert off.advantage_estimator == "grpo"
