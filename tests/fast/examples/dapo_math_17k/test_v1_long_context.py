import hashlib
import json
import shlex
from pathlib import Path

import examples.dapo_math_17k.run_dapo_math_17k as launcher
import pytest
from examples.dapo_math_17k.long_context import (
    CONTEXT_LENGTH,
    ROPE_SCALING,
    megatron_yarn_args,
    prepare_checkpoint,
    validate_prompt_budget,
)
from examples.dapo_math_17k.v1_config import EXPERIMENTS
from tests.fast.launch_scripts.sh_harness import assert_matches_snapshot
from transformers import AutoConfig

from miles.utils.external_utils.model_args_utils import load_model_args


def _flags(command):
    result = {}
    key = None
    for value in shlex.split(command):
        if value.startswith("--"):
            assert value not in result, f"Duplicate flag: {value}"
            key = value
            result[key] = []
        else:
            result[key].append(value)
    return result


def test_eight_v1_commands_preserve_plan_and_only_vary_experiment_factors(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type": "qwen3"}')
    commands = []
    monkeypatch.setenv("WANDB_API_KEY", "test-secret-key")
    monkeypatch.setattr(launcher.U, "execute_train", lambda **kwargs: commands.append(kwargs))
    common = None
    assert len(EXPERIMENTS) == 8
    for name, experiment in EXPERIMENTS.items():
        args = launcher.ScriptArgs(
            v1_experiment=name,
            output_dir=str(tmp_path),
            hf_checkpoint=str(source),
            wandb_project="dapo-long-context-test",
        )
        launcher._execute(args)
        assert args.model_repo == f"Qwen/Qwen3-{experiment.model_size.upper()}"
        assert args.lr_warmup_iters == (10 if experiment.warmup else 0)
        assert args.lr_warmup_init == (1.414e-6 if experiment.warmup else 5e-6)
        assert args.enable_thinking == experiment.thinking
        assert args.output_dir == str(tmp_path / name)
        assert args.rollout_max_response_len == 128000
        assert args.max_tokens_per_gpu == 32768
        assert commands[-1]["megatron_model_type"] == f"qwen3-{experiment.model_size.upper()}"
        flags = _flags(commands[-1]["train_args"])
        for flag, expected in {
            "--rollout-max-response-len": "128000",
            "--sglang-context-length": "131072",
            "--max-tokens-per-gpu": "32768",
            "--tensor-model-parallel-size": "2",
            "--context-parallel-size": "4",
            "--pipeline-model-parallel-size": "1",
            "--cp-comm-type": "a2a",
            "--rollout-batch-size": "64",
            "--n-samples-per-prompt": "8",
            "--global-batch-size": "512",
            "--num-rollout": "100",
            "--start-rollout-id": "0",
            "--lr-decay-iters": "100",
            "--save-interval": "50",
            "--seed": "42",
            "--rollout-seed": "42",
            "--rollout-temperature": "1.0",
            "--rollout-top-p": "0.95",
            "--rollout-top-k": "20",
            "--log-probs-chunk-size": "512",
            "--attention-backend": "flash",
            "--train-backend": "megatron",
            "--recompute-granularity": "full",
            "--recompute-method": "uniform",
            "--recompute-num-layers": "1",
            "--seq-length": "131072",
            "--position-embedding-type": "yarn",
            "--rotary-scaling-factor": "4.0",
            "--yarn-original-max-position-embeddings": "32768",
            "--sglang-max-running-requests": "8",
            "--sglang-router-policy": "round_robin",
            "--sglang-page-size": "64",
            "--sglang-chunked-prefill-size": "8192",
            "--sglang-kv-cache-dtype": "fp8_e4m3",
            "--sglang-attention-backend": "fa3",
            "--sglang-cuda-graph-backend-decode": "full",
            "--sglang-cuda-graph-max-bs-decode": "8",
            "--rollout-required-context-len": "131072",
            "--rollout-max-context-len": "131070",
        }.items():
            assert flags[flag] == [expected]
        assert flags["--hf-checkpoint"] == [str(tmp_path / name / "hf_long_context")]
        assert flags["--load"] == [
            str(tmp_path / name / "megatron_init" / f"Qwen3-{experiment.model_size.upper()}_torch_dist")
        ]
        assert "--sequence-parallel" in flags
        assert "--sglang-disable-cuda-graph" not in flags
        assert "--bf16" in flags
        assert "--gradient-checkpointing" not in flags
        assert "--attn-implementation" not in flags
        assert json.loads(flags["--apply-chat-template-kwargs"][0]) == {"enable_thinking": experiment.thinking}
        assert commands[-1]["secret_env_vars"] == {"WANDB_API_KEY": "test-secret-key"}
        assert commands[-1]["extra_env_vars"] == {"SGLANG_MAX_NEW_TOKENS_LIMIT": "128000"}
        assert "test-secret-key" not in commands[-1]["train_args"]
        for varying in (
            "--hf-checkpoint",
            "--load",
            "--save",
            "--wandb-dir",
            "--wandb-group",
            "--lr-warmup-iters",
            "--lr-warmup-init",
            "--apply-chat-template-kwargs",
        ):
            flags.pop(varying)
        if common is None:
            common = flags
        else:
            assert flags == common

    recordings = []
    for name, command in zip(EXPERIMENTS, commands, strict=True):
        recordings.extend(
            [
                f"### {name}",
                "model definition: " + command["megatron_model_type"],
                "runtime env: " + json.dumps(command["extra_env_vars"], sort_keys=True),
                "secret env names: " + ", ".join(sorted(command["secret_env_vars"])),
            ]
        )
        for flag, values in _flags(command["train_args"]).items():
            recordings.append(shlex.join([flag, *values]))
        recordings.append("")
    actual = "\n".join(recordings).replace(str(tmp_path), "<OUTPUT_ROOT>")
    actual = actual.replace(str(launcher.U.repo_base_dir), "<REPO_ROOT>")
    snapshot = (
        Path(__file__).resolve().parents[3]
        / "snapshots/launch_scripts/py"
        / "examples/dapo_math_17k/run_dapo_math_17k.py/v1_experiments.txt"
    )
    assert_matches_snapshot(snapshot, actual, "DAPO Math v1 eight-experiment matrix")


def test_yarn_checkpoint_view_preserves_source_and_weight_files(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "run" / "hf_long_context"
    source.mkdir()
    config = {"model_type": "qwen3", "max_position_embeddings": 40960, "rope_theta": 1000000, "rope_scaling": None}
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors").write_bytes(b"test-weight-bytes")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    prepare_checkpoint(source, destination, context_length=CONTEXT_LENGTH)
    prepare_checkpoint(source, destination, context_length=CONTEXT_LENGTH)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    assert (destination / "model.safetensors").is_symlink()
    assert (destination / "model.safetensors").resolve() == source / "model.safetensors"
    loaded = AutoConfig.from_pretrained(destination, local_files_only=True)
    assert loaded.max_position_embeddings == CONTEXT_LENGTH
    rope = getattr(loaded, "rope_parameters", None) or loaded.rope_scaling
    assert all(rope[key] == value for key, value in ROPE_SCALING.items())
    with pytest.raises(ValueError, match="differ"):
        prepare_checkpoint(source, source, context_length=CONTEXT_LENGTH)
    (destination / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        prepare_checkpoint(source, destination, context_length=CONTEXT_LENGTH)


def test_prompt_budget_checks_chat_template_and_keeps_all_prompts(monkeypatch, tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
            return "prefix" + messages[0]["content"]

        def encode(self, text, **kwargs):
            assert kwargs == dict(add_special_tokens=False)
            return list(range(len(text)))

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: Tokenizer())
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"prompt": [{"role": "user", "content": "question"}]}) + "\n")
    before = data.read_bytes()
    assert (
        validate_prompt_budget(data, tmp_path, response_length=128000, enable_thinking=True, expected_prompts=1) == 14
    )
    with pytest.raises(ValueError, match="exceeds"):
        validate_prompt_budget(data, tmp_path, response_length=131070, enable_thinking=True)
    with pytest.raises(ValueError, match="Expected 640"):
        validate_prompt_budget(data, tmp_path, response_length=128000, enable_thinking=True, expected_prompts=640)
    assert data.read_bytes() == before


def test_v1_rejects_argv_overrides_that_could_change_the_plan():
    with pytest.raises(ValueError, match="extra-args"):
        launcher.ScriptArgs(v1_experiment=next(iter(EXPERIMENTS)), extra_args="--rollout-max-response-len 16000")
    with pytest.raises(ValueError, match="Unknown v1"):
        launcher.ScriptArgs(v1_experiment="qwen3-4b_typo")


@pytest.mark.parametrize("model_type", ["qwen3-4B", "qwen3-8B"])
def test_model_definition_allows_yarn_override(model_type):
    model_flags = _flags(load_model_args(model_type))
    assert "--use-rotary-position-embeddings" not in model_flags
    assert model_flags["--position-embedding-type"] == ["rope"]
    yarn_flags = _flags(megatron_yarn_args())
    assert yarn_flags["--rotary-base"] == ["1000000"]
    assert yarn_flags["--yarn-beta-fast"] == ["32"]
    assert yarn_flags["--yarn-beta-slow"] == ["1"]
    assert yarn_flags["--mscale"] == ["1"]
    assert yarn_flags["--mscale-all-dim"] == ["0"]
    assert "--yarn-correction-range-round-to-int" in yarn_flags


def test_long_context_conversion_and_training_share_yarn_settings(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type": "qwen3"}')
    captured = {}
    monkeypatch.setattr(launcher.U, "exec_command_cpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher.U, "convert_checkpoint", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(launcher, "validate_prompt_budget", lambda *args, **kwargs: 100)
    args = launcher.ScriptArgs(
        v1_experiment="qwen3-8b_warmup_thinking_seed42", hf_checkpoint=str(source), output_dir=str(tmp_path)
    )
    launcher._prepare(args)
    assert captured["megatron_model_type"] == "qwen3-8B"
    assert captured["hf_checkpoint"] == args.prepared_hf_checkpoint
    assert captured["dir_dst"] == args.conversion_dir
    assert captured["extra_args"] == megatron_yarn_args()
    assert megatron_yarn_args() in launcher._backend_args(args)
    config = json.loads((Path(args.prepared_hf_checkpoint) / "config.json").read_text())
    assert config["rope_scaling"] == ROPE_SCALING


def test_preset_honors_eight_node_allocation_and_keeps_one_update_per_rollout(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "8")
    args = launcher.ScriptArgs(v1_experiment="qwen3-8b_warmup_thinking_seed42")
    assert args.num_nodes == 8
    assert args.num_gpus_per_node == 8
    data_parallel_size = (
        args.num_nodes * args.num_gpus_per_node // (args.tensor_model_parallel_size * args.context_parallel_size)
    )
    assert data_parallel_size == 8
    assert args.global_batch_size == args.rollout_batch_size * args.n_samples_per_prompt == 512
    assert args.num_rollout == args.lr_decay_iters == 100
    assert args.save_interval == 50
    explicit = launcher.ScriptArgs(v1_experiment="qwen3-8b_warmup_thinking_seed42", num_nodes=1)
    assert explicit.num_nodes == 1
