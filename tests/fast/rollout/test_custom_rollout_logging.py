from types import SimpleNamespace

from miles.ray import rollout
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
