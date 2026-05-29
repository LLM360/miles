"""Run the actual trainer orchestration functions with CPU dependency doubles.

AST extraction avoids importing Megatron/TE. Whole function bodies run; only
the GPU/optimizer/rollout boundaries are replaced. This verifies counter wiring,
not GPU training correctness.
"""

import ast
import logging
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from miles.backends.training_utils import log_utils
from miles.backends.training_utils.train_step_counter import counter_for

ROOT = Path(__file__).resolve().parents[4]


def _load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), path, "exec"), namespace)
    return namespace[name]


class Outcome(Enum):
    NORMAL = 1
    DISCARDED_SHOULD_RETRY = 2


@pytest.mark.parametrize("main_rank", [True, False])
def test_megatron_counts_normal_steps_on_all_ranks_and_reuses_id_for_ci(monkeypatch, main_rank):
    args = SimpleNamespace(
        debug_disable_optimizer=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        reset_optimizer_states=False,
        manual_gc=False,
        enable_mtp_training=False,
        ci_test=True,
        ci_disable_kl_checker=False,
    )
    logs, checks = [], []
    outcomes = iter([Outcome.NORMAL] * 3 + [Outcome.DISCARDED_SHOULD_RETRY] + [Outcome.NORMAL])
    monkeypatch.setattr(log_utils.tracking, "log", lambda args, values, **kwargs: logs.append(values))
    namespace = {
        "get_args": lambda: args,
        "get_parallel_state": lambda: SimpleNamespace(
            indep_dp=SimpleNamespace(size=1), effective_dp=SimpleNamespace(rank=0)
        ),
        "get_model_config": lambda model: SimpleNamespace(),
        "DDP": type("DDP", (), {}),
        "finalize_model_grads_with_empty_cache": None,
        "should_disable_forward_pre_hook": lambda args: False,
        "counter_for": counter_for,
        "TrainStepOutcome": Outcome,
        "train_one_step": lambda *a, **k: ({"loss": 1.0}, 0, next(outcomes)),
        "is_first_replica_megatron_main_rank": lambda: main_rank,
        "log_train_step": log_utils.log_train_step,
        "check_kl": lambda args, log, step, cumulative: checks.append((log["train/step"], cumulative)),
        "check_grad_norm": lambda **kwargs: None,
        "logger": logging.getLogger(__name__),
    }
    train = _load_function("miles/backends/megatron_utils/model.py", "train", namespace)
    model = [SimpleNamespace(train=lambda: None, role="actor")]
    iterator = [SimpleNamespace(reset=lambda: None)]
    for rollout_id, num_steps in enumerate([3, 1, 1]):
        train(rollout_id, model, None, None, iterator, [1] * num_steps, [1] * num_steps, None, 0)
    assert counter_for(model[0]).next_step == 4
    if main_rank:
        assert [row["train/step"] for row in logs] == [0, 1, 2, 3]
        assert checks == [(0, 0), (1, 1), (2, 2), (3, 3)]
    else:
        assert logs == checks == []


def test_fsdp_advances_count_once_per_step_and_not_on_failed_training(monkeypatch):
    logs = []
    monkeypatch.setattr(log_utils.tracking, "log", lambda args, values, **kwargs: logs.append(values))
    monkeypatch.setattr(log_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.nn.utils, "clip_grad_norm_", lambda *a: SimpleNamespace(full_tensor=lambda: torch.tensor(0.0))
    )
    counts = iter([3, 1, 1])
    iterator = SimpleNamespace(reset=lambda: None)
    routing = SimpleNamespace(
        fill=lambda *a: None,
        stage=lambda *a: nullcontext(),
        log_prob_stage=lambda args: None,
        rewind=lambda: None,
        reset=lambda: None,
        REPLAY_BACKWARD="backward",
    )
    namespace = {
        "torch": torch,
        "dist": log_utils.dist,
        "get_data_iterator": lambda *a: ([iterator], [1] * next(counts)),
        "routing_replay": routing,
        "compute_advantages_and_returns": lambda *a: None,
        "log_rollout_data": lambda *a: None,
        "timer": lambda *a: nullcontext(),
        "tqdm": lambda values, **kwargs: values,
        "get_batch": lambda *a, **k: {},
        "aggregate_train_losses": lambda values: {"loss": 1.0},
        "log_train_step": log_utils.log_train_step,
    }
    train = _load_function(
        "miles/backends/fsdp_utils/actor.py", "_train_core", namespace, class_name="FSDPTrainRayActor"
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(
            micro_batch_size=1,
            global_batch_size=1,
            data_pad_size_multiplier=1,
            qkv_format="bshd",
            clip_grad=1,
            ci_test=False,
            save_debug_train_data=None,
            ref_update_interval=None,
        ),
        ref_model=None,
        model=SimpleNamespace(parameters=lambda: []),
        optimizer=Mock(param_groups=[]),
        lr_scheduler=Mock(),
        global_step=17,
        _compute_log_prob=lambda *a, **k: {},
        _train_step=lambda **kwargs: {},
        prof=SimpleNamespace(iterate_train_actor=lambda values: values, step=lambda **kwargs: None),
    )
    train(actor, 10, {})
    train(actor, 11, {})
    assert [row["train/step"] for row in logs] == [17, 18, 19, 20]
    assert actor.global_step == 21
    assert actor.optimizer.step.call_count == 4
    actor._train_step = Mock(side_effect=RuntimeError("training failed"))
    with pytest.raises(RuntimeError, match="training failed"):
        train(actor, 12, {})
    assert actor.global_step == 21
