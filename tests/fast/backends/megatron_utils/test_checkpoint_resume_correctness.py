"""Exercise complete checkpoint orchestration with CPU boundary doubles."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.fast.backends.training_utils.test_train_step_callers import _load_function

from miles.backends.training_utils.train_step_counter import TrainStepCounter, counter_for

MODEL = "miles/backends/megatron_utils/model.py"


def test_save_keeps_preprocessor_context_and_train_step():
    save_checkpoint = Mock()
    preprocessor = object()
    checkpoint_context = object()
    step_context = Mock(side_effect=lambda step: nullcontext())
    save = _load_function(
        MODEL,
        "save",
        {
            "get_args": lambda: SimpleNamespace(ci_test=False),
            "should_disable_forward_pre_hook": lambda args: False,
            "clear_memory": Mock(),
            "is_lora_model": lambda model: False,
            "checkpoint_train_step": step_context,
            "counter_for": counter_for,
            "save_checkpoint": save_checkpoint,
            "preprocess_common_state_dict": preprocessor,
        },
    )
    model = [SimpleNamespace(train_step_counter=TrainStepCounter(17))]
    optimizer, scheduler = object(), object()
    save(12, model, optimizer, scheduler, checkpointing_context=checkpoint_context, non_persistent_ckpt=True)
    step_context.assert_called_once_with(17)
    save_checkpoint.assert_called_once_with(
        12,
        model,
        optimizer,
        scheduler,
        num_floating_point_operations_so_far=0,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=preprocessor,
        checkpointing_context=checkpoint_context,
        non_persistent_ckpt=True,
    )


@pytest.mark.parametrize("finetune", [False, True])
@pytest.mark.parametrize("no_load_optim", [False, True])
@pytest.mark.parametrize("use_checkpoint_opt_param_scheduler", [False, True])
@pytest.mark.parametrize("has_scheduler", [False, True])
def test_scheduler_resume(finetune, no_load_optim, use_checkpoint_opt_param_scheduler, has_scheduler):
    args = SimpleNamespace(
        finetune=finetune,
        no_load_optim=no_load_optim,
        use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler,
        global_batch_size=32,
    )
    model = [SimpleNamespace()]
    optimizer, scheduler, context = object(), Mock() if has_scheduler else None, object()
    load_checkpoint = Mock(return_value=(7, 0))
    initialize = _load_function(
        MODEL,
        "initialize_model_and_optimizer",
        {
            "setup_model_and_optimizer": lambda args, role: (model, optimizer, scheduler),
            "TrainStepCounter": TrainStepCounter,
            "clear_memory": Mock(),
            "is_multi_lora_enabled": lambda args: False,
            "is_lora_enabled": lambda args: False,
            "nullcontext": nullcontext,
            "load_checkpoint": load_checkpoint,
            "check_peak_gpu_memory_after_load": Mock(),
            "check_model_hashes": Mock(),
        },
    )
    assert initialize(args, role="actor", checkpointing_context=context) == (model, optimizer, scheduler, 7)
    assert model[0].role == "actor"
    load_checkpoint.assert_called_once_with(
        model,
        optimizer,
        scheduler,
        checkpointing_context=context,
        skip_load_to_model_and_opt=False,
        restore_train_step=True,
    )
    if has_scheduler:
        if finetune or no_load_optim:
            scheduler.step.assert_called_once_with(increment=7 * 32)
        else:
            scheduler.step.assert_not_called()
