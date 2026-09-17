import json
import shlex
from pathlib import Path

import examples.dapo_math_17k.run_dapo_math_17k as launcher
import pytest
from examples.dapo_math_17k.run_dapo_math_17k import ScriptArgs, _execute
from tests.fast.launch_scripts.py_harness import format_recording, freeze_environment, install_command_recorder
from tests.fast.launch_scripts.sh_harness import assert_matches_snapshot


def test_default_launcher_submission_snapshot(monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    checkpoint = tmp_path / "models" / "Qwen3-8B"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type": "qwen3"}')
    args = ScriptArgs(
        model_dir=str(tmp_path / "models"),
        output_dir=str(tmp_path / "output"),
        run_id="260101-000000-000",
        num_nodes=1,
    )
    launcher._prepare(args)
    launcher._execute(args)
    snapshot = (
        Path(__file__).resolve().parents[3]
        / "snapshots/launch_scripts/py/examples/dapo_math_17k/run_dapo_math_17k.py/main.txt"
    )
    assert_matches_snapshot(snapshot, format_recording(recording, sandbox=tmp_path), "DAPO Math Qwen3-8B default")
    assert "frozen-wandb-api-key" not in "\n".join(recording.commands)


def test_model_repo_selects_local_checkpoint_name() -> None:
    args = ScriptArgs(model_dir="/models", model_repo="Qwen/Qwen3-4B")

    assert args.hf_checkpoint == "/models/Qwen3-4B"
    assert args.megatron_model_type == "qwen3-4B"


def test_default_recipe_selects_qwen3_8b() -> None:
    args = ScriptArgs(model_dir="/models")
    assert args.model_repo == "Qwen/Qwen3-8B"
    assert args.hf_checkpoint == "/models/Qwen3-8B"
    assert args.megatron_model_type == "qwen3-8B"


def test_explicit_checkpoint_takes_precedence_over_model_repo() -> None:
    args = ScriptArgs(
        model_dir="/models",
        model_repo="Qwen/Qwen3-4B",
        hf_checkpoint="/checkpoints/step-100",
    )

    assert args.hf_checkpoint == "/checkpoints/step-100"


def test_prepare_rejects_non_qwen3_dense_checkpoint(monkeypatch, tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    install_command_recorder(monkeypatch)
    with pytest.raises(ValueError, match="dense Qwen3 checkpoint"):
        launcher._prepare(ScriptArgs(hf_checkpoint=str(tmp_path), output_dir=str(tmp_path)))


def test_unsupported_model_definition_is_rejected() -> None:
    with pytest.raises(ValueError, match="supports Qwen3-4B and Qwen3-8B"):
        ScriptArgs(model_repo="Qwen/Qwen3-VL-8B-Instruct")


def test_execute_forwards_learning_rate_schedule(monkeypatch, tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    captured = {}
    monkeypatch.setattr(launcher.U, "get_default_wandb_args", lambda *args, **kwargs: "")
    monkeypatch.setattr(launcher.U, "execute_train", lambda **kwargs: captured.update(kwargs))

    _execute(
        ScriptArgs(
            hf_checkpoint=str(tmp_path),
            lr=5e-6,
            min_lr=2e-6,
            lr_decay_style="linear",
            lr_decay_iters=500,
            lr_warmup_init=1.414e-6,
            lr_warmup_iters=10,
        )
    )

    train_args = captured["train_args"]
    assert "--lr 5e-06 " in train_args
    assert "--min-lr 2e-06 " in train_args
    assert "--lr-decay-style linear " in train_args
    assert "--lr-decay-iters 500 " in train_args
    assert "--lr-warmup-init 1.414e-06 " in train_args
    assert "--lr-warmup-iters 10 " in train_args
    assert train_args.count("--lr ") == 1


def _capture_execute(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    captured = {}
    monkeypatch.setattr(launcher.U, "get_default_wandb_args", lambda *args, **kwargs: "")
    monkeypatch.setattr(launcher.U, "execute_train", lambda **kwargs: captured.update(kwargs))
    return captured


def test_enable_thinking_is_pinned_explicitly() -> None:
    assert launcher._chat_template_args(None) == ""
    assert launcher._chat_template_args(True) == "--apply-chat-template-kwargs '{\"enable_thinking\": true}' "
    assert launcher._chat_template_args(False) == "--apply-chat-template-kwargs '{\"enable_thinking\": false}' "


def test_execute_forwards_seeds_and_thinking_switch(monkeypatch, tmp_path) -> None:
    captured = _capture_execute(monkeypatch, tmp_path)

    _execute(ScriptArgs(hf_checkpoint=str(tmp_path), seed=42, rollout_seed=42, enable_thinking=False))

    train_args = captured["train_args"]
    assert "--seed 42 " in train_args
    assert "--start-rollout-id 0 " in train_args
    assert "--rollout-seed 42 " in train_args
    assert "--apply-chat-template " in train_args
    assert "--apply-chat-template-kwargs '{\"enable_thinking\": false}' " in train_args
    assert "--dump-details" not in train_args
    assert captured["secret_env_vars"] == {}


def test_dump_details_targets_the_output_dir(monkeypatch, tmp_path) -> None:
    captured = _capture_execute(monkeypatch, tmp_path)

    _execute(ScriptArgs(hf_checkpoint=str(tmp_path), output_dir="/out/run", dump_details=True))

    assert "--dump-details /out/run/debug " in captured["train_args"]


def test_explicit_wandb_project_keeps_the_key_out_of_argv(monkeypatch, tmp_path) -> None:
    captured = _capture_execute(monkeypatch, tmp_path)
    monkeypatch.setenv("WANDB_API_KEY", "secret-key")

    _execute(
        ScriptArgs(
            hf_checkpoint=str(tmp_path),
            output_dir="/out/run",
            wandb_project="dapo-exp",
            wandb_run_name="qwen3-4b_warmup_thinking_seed42",
        )
    )

    train_args = captured["train_args"]
    assert "--use-wandb --wandb-mode online " in train_args
    assert "--wandb-project dapo-exp " in train_args
    assert "--wandb-group qwen3-4b_warmup_thinking_seed42 " in train_args
    assert "--disable-wandb-random-suffix " in train_args
    assert "--wandb-dir /out/run/wandb " in train_args
    assert "secret-key" not in train_args
    assert "--wandb-key" not in train_args
    assert captured["secret_env_vars"] == {"WANDB_API_KEY": "secret-key"}


def test_explicit_wandb_project_requires_the_api_key(monkeypatch, tmp_path) -> None:
    _capture_execute(monkeypatch, tmp_path)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        _execute(ScriptArgs(hf_checkpoint=str(tmp_path), wandb_project="dapo-exp"))


def test_default_wandb_path_is_unchanged(monkeypatch, tmp_path) -> None:
    captured = _capture_execute(monkeypatch, tmp_path)
    monkeypatch.setattr(launcher.U, "get_default_wandb_args", lambda *args, **kwargs: "--use-wandb DEFAULT ")

    _execute(ScriptArgs(hf_checkpoint=str(tmp_path)))

    assert "--use-wandb DEFAULT " in captured["train_args"]
    assert captured["secret_env_vars"] == {}


def test_default_wandb_key_is_forwarded_only_through_runtime_env(monkeypatch, tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    captured = {}
    monkeypatch.setattr(launcher.U, "execute_train", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setenv("WANDB_API_KEY", "test-default-secret")
    _execute(ScriptArgs(hf_checkpoint=str(tmp_path)))
    assert "test-default-secret" not in captured["train_args"]
    assert "--wandb-key" not in captured["train_args"]
    assert captured["secret_env_vars"] == {"WANDB_API_KEY": "test-default-secret"}


@pytest.mark.parametrize("size", [16, 32, 64])
def test_small_submission_batches_are_supported(size):
    assert ScriptArgs(rollout_batch_size=64, over_sampling_batch_size=size).over_sampling_batch_size == size


@pytest.mark.parametrize("size", [0, -1])
def test_nonpositive_submission_batches_are_rejected(size):
    with pytest.raises(ValueError, match="over_sampling_batch_size must be positive"):
        ScriptArgs(over_sampling_batch_size=size)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_gpus_per_node": 3}, "multiple of TP"),
        ({"num_gpus_per_node": 16, "context_parallel_size": 8}, "KV heads"),
        ({"tensor_model_parallel_size": 0}, "multiple of TP"),
        ({"num_gpus_per_node": 8, "global_batch_size": 3}, "data parallel"),
        (
            {"num_gpus_per_node": 8, "context_parallel_size": 4, "qwen3_long_context": True},
            "requires max_tokens_per_gpu",
        ),
    ],
)
def test_incompatible_training_layout_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        ScriptArgs(num_nodes=1, **overrides)


def test_resume_skips_conversion_and_selects_only_the_requested_checkpoint(monkeypatch, tmp_path):
    captured = _capture_execute(monkeypatch, tmp_path)
    monkeypatch.setattr(launcher.U, "exec_command_cpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher.U, "convert_checkpoint", lambda **kwargs: pytest.fail("Resume must skip conversion"))
    args = ScriptArgs(hf_checkpoint=str(tmp_path), output_dir=str(tmp_path), load_path="/checkpoints/megatron")
    launcher._prepare(args)
    launcher._execute(args)
    assert captured["train_args"].count("--load ") == 1
    assert "--load /checkpoints/megatron " in captured["train_args"]
    assert "--start-rollout-id" not in captured["train_args"]
    assert captured["megatron_model_type"] == "qwen3-8B"


def test_checkpoint_conversion_preserves_paths_with_spaces(monkeypatch, tmp_path):
    checkpoint = tmp_path / "source weights"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "qwen3"}')
    recording = install_command_recorder(monkeypatch)
    args = ScriptArgs(hf_checkpoint=str(checkpoint), output_dir=str(tmp_path / "output run"))
    launcher._prepare(args)
    command = next(command for command in recording.commands if "convert_hf_to_torch_dist.py" in command)
    tokens = shlex.split(command)
    assert tokens[tokens.index("--hf-checkpoint") + 1] == str(checkpoint)
    assert tokens[tokens.index("--save") + 1] == args.initial_megatron_checkpoint
