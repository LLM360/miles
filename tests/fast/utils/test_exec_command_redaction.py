import subprocess

import pytest

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.exec_command import exec_command_cpu


def test_failed_command_uses_redacted_display_command_in_exception() -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        exec_command_cpu("exit 7 # secret-value", display_cmd="exit 7 # <redacted>")

    assert "secret-value" not in str(error.value)
    assert "<redacted>" in str(error.value)


def test_ray_submission_redacts_secrets_but_preserves_worker_environment(monkeypatch):
    commands = []
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")
    monkeypatch.setattr(command_utils, "exec_command_cpu", lambda cmd, **kwargs: commands.append((cmd, kwargs)))
    command_utils.execute_train(
        train_args="--train-backend fsdp",
        num_gpus_per_node=2,
        megatron_model_type=None,
        secret_env_vars={"WANDB_API_KEY": "test-worker-secret"},
    )
    command, options = commands[-1]
    assert "ray job submit" in command
    assert "test-worker-secret" in command
    assert "test-worker-secret" not in options["display_cmd"]
    assert "WANDB_API_KEY" in options["display_cmd"]
    assert "<redacted>" in options["display_cmd"]
