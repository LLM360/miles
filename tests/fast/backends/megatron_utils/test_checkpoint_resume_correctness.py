import importlib
import sys
import types
from argparse import Namespace
from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _import_model_module():
    try:
        return importlib.import_module("miles.backends.megatron_utils.model")
    except (ImportError, ModuleNotFoundError) as error:
        if "sglang" not in str(error):
            raise

    # The checkpoint paths under test do not use the optional dumper. Keep the
    # unit test runnable in lightweight environments that omit SGLang.
    dumper_stub = types.ModuleType("miles.utils.dumper_utils")

    class DumperPhase(Enum):
        FWD_ONLY = "fwd_only"
        FWD_BWD = "fwd_bwd"

    dumper_stub.DumperMegatronUtil = object
    dumper_stub.DumperPhase = DumperPhase
    previous_dumper_module = sys.modules.get(dumper_stub.__name__)
    sys.modules[dumper_stub.__name__] = dumper_stub
    try:
        return importlib.import_module("miles.backends.megatron_utils.model")
    finally:
        if previous_dumper_module is None:
            sys.modules.pop(dumper_stub.__name__, None)
        else:
            sys.modules[dumper_stub.__name__] = previous_dumper_module


model_module = _import_model_module()


def test_save_uses_megatron_common_state_preprocessor(monkeypatch: pytest.MonkeyPatch) -> None:
    save_checkpoint = Mock()
    monkeypatch.setattr(
        model_module,
        "get_args",
        lambda: Namespace(ci_test=False, ci_save_model_hash=False, save="unused"),
    )
    monkeypatch.setattr(model_module, "is_lora_model", lambda model: False)
    monkeypatch.setattr(model_module, "should_disable_forward_pre_hook", lambda args: False)
    monkeypatch.setattr(model_module, "is_megatron_main_rank", lambda: False)
    monkeypatch.setattr(model_module, "save_checkpoint", save_checkpoint)
    monkeypatch.setattr(model_module, "clear_memory", Mock())

    model = [object()]
    optimizer = object()
    scheduler = object()
    model_module.save(12, model, optimizer, scheduler)

    save_checkpoint.assert_called_once_with(
        12,
        model,
        optimizer,
        scheduler,
        num_floating_point_operations_so_far=0,
        checkpointing_context=None,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=model_module.preprocess_common_state_dict,
    )


@pytest.mark.parametrize(
    ("finetune", "no_load_optim", "expected_scheduler_steps"),
    [
        (False, False, 0),
        (True, False, 1),
        (False, True, 1),
    ],
    ids=["normal-resume", "finetune", "optimizer-state-not-loaded"],
)
def test_scheduler_is_only_reconstructed_when_checkpoint_state_was_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
    finetune: bool,
    no_load_optim: bool,
    expected_scheduler_steps: int,
) -> None:
    args = Namespace(
        finetune=finetune,
        no_load_optim=no_load_optim,
        global_batch_size=32,
    )
    model = [SimpleNamespace()]
    optimizer = object()
    scheduler = Mock()
    load_checkpoint = Mock(return_value=(7, 0))

    monkeypatch.setattr(
        model_module,
        "setup_model_and_optimizer",
        lambda args, role: (model, optimizer, scheduler),
    )
    monkeypatch.setattr(model_module, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(model_module, "check_peak_gpu_memory_after_load", Mock())
    monkeypatch.setattr(model_module, "check_model_hashes", Mock())
    monkeypatch.setattr(model_module, "is_megatron_main_rank", lambda: False)
    monkeypatch.setattr(model_module, "clear_memory", Mock())

    result = model_module.initialize_model_and_optimizer(args, role="actor")

    assert result == (model, optimizer, scheduler, 7)
    assert model[0].role == "actor"
    load_checkpoint.assert_called_once_with(
        model,
        optimizer,
        scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    if expected_scheduler_steps:
        scheduler.step.assert_called_once_with(increment=7 * args.global_batch_size)
    else:
        scheduler.step.assert_not_called()
