from types import SimpleNamespace

import pytest

from miles.ray import rollout
from miles.rollout._agentic_outcomes import TOKEN_TRUNCATION_EXIT_STATUSES
from miles.utils import wandb_utils


def test_eval_hook_can_add_metrics_when_rollout_output_has_none(monkeypatch):
    captured = {}

    def custom_hook(_rollout_id, _args, _data, extra_metrics):
        extra_metrics["file_format/eval/heldout/docx/pass_at_1"] = 1.0
        return False

    monkeypatch.setattr(rollout, "load_function", lambda _path: custom_hook)
    monkeypatch.setattr(
        rollout.tracking_utils,
        "log",
        lambda _args, metrics, step_key: captured.update(metrics),
    )
    args = SimpleNamespace(
        custom_eval_rollout_log_function_path="test:hook",
        log_passrate=False,
        wandb_always_use_train_step=False,
    )

    result = rollout._log_eval_rollout_data(
        3, args, {"heldout": {"rewards": [1.0]}}, extra_metrics=None
    )

    key = "file_format/eval/heldout/docx/pass_at_1"
    assert result[key] == 1.0
    assert captured[key] == 1.0
    assert captured["eval/step"] == 3


def test_train_hook_adds_metrics_to_existing_tracking_call(monkeypatch):
    captured = {}

    def custom_hook(_rollout_id, _args, _samples, extra_metrics, _rollout_time):
        extra_metrics["file_format/rollout/docx/pass_at_1"] = 0.5
        return False

    monkeypatch.setattr(rollout, "load_function", lambda _path: custom_hook)
    monkeypatch.setattr(rollout, "compute_metrics_from_samples", lambda _args, _samples: {})
    monkeypatch.setattr(rollout, "compute_perf_metrics_from_samples", lambda _args, _samples, _time: {})
    monkeypatch.setattr(
        rollout.tracking_utils,
        "log",
        lambda _args, metrics, step_key: captured.update(metrics),
    )
    args = SimpleNamespace(
        custom_rollout_log_function_path="test:hook",
        load_debug_rollout_data=None,
        wandb_always_use_train_step=False,
        rollout_batch_size=64,
        n_samples_per_prompt=16,
        global_batch_size=1024,
    )

    rollout._log_rollout_data(3, args, [], {}, 1.0)

    assert captured["file_format/rollout/docx/pass_at_1"] == 0.5
    assert captured["rollout/step"] == 3


def test_eval_logs_dataset_scoped_token_truncation_counts(monkeypatch):
    captured = {}
    monkeypatch.setattr(rollout, "compute_metrics_from_samples", lambda _args, _samples: {})
    monkeypatch.setattr(
        rollout.tracking_utils,
        "log",
        lambda _args, metrics, step_key: captured.update(metrics),
    )
    args = SimpleNamespace(
        custom_eval_rollout_log_function_path=None,
        log_passrate=False,
        wandb_always_use_train_step=False,
    )

    def sample(status):
        return SimpleNamespace(metadata={"exit_status": status})

    data = {
        "first": {
            "rewards": [0.0, 0.0],
            "samples": [sample("BadRequestError"), sample("LimitsExceeded")],
        },
        "second": {
            "rewards": [0.0],
            "samples": [sample("OutputLengthExceededError")],
        },
    }

    rollout._log_eval_rollout_data(3, args, data)

    for status in TOKEN_TRUNCATION_EXIT_STATUSES:
        assert captured[f"eval/first/exit_status/{status}/count"] == int(
            status in {"BadRequestError", "LimitsExceeded"}
        )
        assert captured[f"eval/second/exit_status/{status}/count"] == int(
            status == "OutputLengthExceededError"
        )


def test_eval_rejects_invalid_rewards_with_status_summary():
    args = SimpleNamespace(
        custom_eval_rollout_log_function_path=None,
        log_passrate=False,
        wandb_always_use_train_step=False,
    )
    sample = SimpleNamespace(metadata={"exit_status": "Cancelled"})

    with pytest.raises(ValueError, match="1/1 invalid rewards.*Cancelled"):
        rollout._log_eval_rollout_data(
            3,
            args,
            {"heldout": {"rewards": [None], "samples": [sample]}},
        )


def test_eval_rejects_empty_rewards():
    args = SimpleNamespace(
        custom_eval_rollout_log_function_path=None,
        log_passrate=False,
        wandb_always_use_train_step=False,
    )

    with pytest.raises(ValueError, match="has no rewards"):
        rollout._log_eval_rollout_data(3, args, {"heldout": {"rewards": []}})


def test_file_format_metrics_use_rollout_and_eval_axes(monkeypatch):
    definitions = []
    monkeypatch.setattr(
        wandb_utils.wandb,
        "define_metric",
        lambda name, **kwargs: definitions.append((name, kwargs.get("step_metric"))),
    )

    wandb_utils._init_wandb_common()

    assert ("file_format/rollout/*", "rollout/step") in definitions
    assert ("file_format/eval/*", "eval/step") in definitions
