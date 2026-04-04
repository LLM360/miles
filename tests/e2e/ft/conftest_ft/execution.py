# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import json
import os
import tempfile
from pathlib import Path

import miles.utils.external_utils.command_utils as U
from tests.e2e.conftest_dumper import MEGATRON_PATCHER_YAMLS
from tests.e2e.ft.conftest_ft.modes import DEBUG_ROLLOUT_DATA_HF_REPO, FTTestMode

_RUN_DIR: Path = Path(tempfile.mkdtemp(prefix="ft_test_dumper_"))
_MEGATRON_SOURCE_PATCHER_CONFIG_PATH: Path = _RUN_DIR / "megatron_source_patcher.yaml"
_MEGATRON_PATH: str = os.environ.get("MILES_SCRIPT_MEGATRON_PATH", "/root/Megatron-LM")


def _get_hf_num_layers(model_path: str) -> int:
    with open(f"{model_path}/config.json") as f:
        return json.load(f)["num_hidden_layers"]


def prepare(mode: FTTestMode) -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download {mode.model_hf_repo} --local-dir /root/models/{mode.model_name}")

    hf_model_path = f"/root/models/{mode.model_name}"
    num_layers = _get_hf_num_layers(hf_model_path)
    convert_gpus = min(mode.train_gpus_per_node, num_layers)

    U.convert_checkpoint(
        model_name=mode.model_name,
        megatron_model_type=mode.megatron_model_type,
        num_gpus_per_node=convert_gpus,
        megatron_path=_MEGATRON_PATH,
    )
    U.hf_download_dataset(DEBUG_ROLLOUT_DATA_HF_REPO)
    if mode.has_rollout:
        U.hf_download_dataset("zhuzilin/gsm8k")

    megatron_yaml: str = MEGATRON_PATCHER_YAMLS["thd"]
    _MEGATRON_SOURCE_PATCHER_CONFIG_PATH.write_text(megatron_yaml)


def get_common_train_args(mode: FTTestMode, *, dump_dir: str, num_steps: int | None = None) -> str:
    ckpt_args = (
        f"--hf-checkpoint /root/models/{mode.model_name} "
        f"--ref-load /root/{mode.model_name}_torch_dist "
    )

    optimizer_args = (
        "--optimizer adam "
        # NOTE: deliberately use huge lr and small adam to make weight change vivid
        "--lr 1e-3 --lr-decay-style constant --adam-beta1 0.1 --adam-beta2 0.9 "
        "--lr-warmup-fraction 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
    )

    rollout_args: str
    if not mode.has_rollout:
        rollout_args = (
            "--load-debug-rollout-data /root/datasets/miles-test-rollout-Qwen3-30B-A3B/{rollout_id}.pt "
            "--debug-train-only "
            f"--rollout-batch-size {mode.global_batch_size} "
        )
    else:
        rollout_args = (
            "--prompt-data /root/datasets/gsm8k/train.parquet "
            "--input-key messages "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type math "
            "--rollout-max-response-len 3 "
            "--rollout-temperature 0.8 "
            "--rollout-batch-size 1 "
            "--n-samples-per-prompt 1 "
            "--sglang-disable-cuda-graph "
            f"--rollout-num-gpus {mode.total_rollout_gpus} "
            f"--rollout-num-gpus-per-engine {mode.rollout_gpus_per_engine} "
        )

    event_logger_args = f"--save-debug-event-data {dump_dir}/events "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {mode.train_num_nodes} "
        f"--actor-num-gpus-per-node {mode.train_gpus_per_node} "
        f"--global-batch-size {mode.global_batch_size} "
        "--decrease-batch-size-if-needed "
        "--delay-split-train-data-by-dp "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
        "--moe-token-dispatcher-type alltoall "
        "--advantage-estimator grpo "
        "--eps-clip 0.2 "
        f"--num-rollout {num_steps if num_steps is not None else mode.num_steps} "
    )

    dumper_args = (
        f"--dumper-dir {dump_dir}/dumps "
        f"--dumper-fwd-bwd enable=1 enable_model_value=1 enable_model_grad=1 "
        f"--dumper-source-patcher-config-train {_MEGATRON_SOURCE_PATCHER_CONFIG_PATH} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{optimizer_args} "
        f"{rollout_args} "
        f"{event_logger_args} "
        f"{mode.parallel_args} "
        f"{misc_args} "
        f"{dumper_args} "
        f"{U.get_default_wandb_args(__file__)} "
    )

    return train_args


def get_ft_args(mode: FTTestMode) -> str:
    return (
        "--use-fault-tolerance "
        "--ft-components train "
        "--control-server-port 0 "
    )


def run_training(
    train_args: str,
    mode: FTTestMode,
    *,
    extra_env_vars: dict[str, str] | None = None,
) -> None:
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=mode.train_gpus_per_node,
        megatron_model_type=mode.megatron_model_type,
        extra_env_vars=extra_env_vars or {},
        megatron_path=_MEGATRON_PATH,
    )
