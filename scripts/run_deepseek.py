"""
This file is in preview, and will be further refined and optimized.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


def _get_layer_count(model_name: str) -> int | None:
    match = re.search(r"(\d+)layer", model_name)
    return int(match.group(1)) if match is not None else None


def _get_rollout_model_name(args) -> str:
    return f"{args.model_name}-NVFP4" if args.rollout_nvfp4 else args.model_name


def _get_rollout_runtime_model_name(args) -> str:
    if args.rollout_nvfp4:
        return f"{args.model_name}-NVFP4-current"
    return _get_rollout_model_name(args)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "deepseek-ai"
    model_name: str = "DeepSeek-V3"
    megatron_model_type: str = "deepseek-v3"
    num_gpus_per_node: int = 4
    enable_eval: bool = True
    extra_args: str = ""
    task: Literal["dapo_aime", "gsm8k"] = "dapo_aime"
    data_dir: str = "/root/datasets"
    model_dir: str = "/shared/zhichen/models"
    model_local_dir: str = "/shared/zhichen/models"
    megatron_path: str = "/root/Megatron-LM"
    rollout_nvfp4: bool = False
    rollout_nvfp4_restart_sync: bool = False
    nvfp4_keep_first_n: int = 0
    nvfp4_keep_last_n: int = 0
    optimizer_cpu_offload: bool = False
    # Megatron parallelism overrides (0 = auto-select based on model/GPU count)
    train_tp: int = 0
    train_pp: int = 0
    train_ep: int = 0

    def __post_init__(self):
        if (layer_count := _get_layer_count(self.model_name)) is not None:
            if self.model_org == "deepseek-ai":
                self.model_org = "chwan"
            self.megatron_model_type = f"deepseek-v3-{layer_count}layer"


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    match args.task:
        case "dapo_aime":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
            U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _prepare_bf16_ckpt(args: ScriptArgs):
    U.fp8_cast_bf16(
        path_src=f"{args.model_dir}/{args.model_name}",
        path_dst=f"{args.model_dir}/{args.model_name}-bf16/",
    )


def _prepare_nvfp4_ckpt(args: ScriptArgs):
    if not args.rollout_nvfp4:
        return

    path_dst = Path(args.model_dir) / f"{args.model_name}-NVFP4"
    if (path_dst / "model.safetensors.index.json").exists():
        print(f"nvfp4 conversion skip {path_dst} since model.safetensors.index.json exists")
        return

    keep_first_arg = f" --keep-first-n {args.nvfp4_keep_first_n}" if args.nvfp4_keep_first_n > 0 else ""
    keep_last_arg = f" --keep-last-n {args.nvfp4_keep_last_n}" if args.nvfp4_keep_last_n > 0 else ""
    U.exec_command(
        f"python tools/convert_hf_to_nvfp4.py "
        f"--model-dir {args.model_dir}/{args.model_name}-bf16 "
        f"--save-dir {path_dst}{keep_first_arg}{keep_last_arg}"
    )


def _get_5layer_parallel_config(args: ScriptArgs) -> tuple[int, int, int]:
    """Return (TP, PP, EP) for 5-layer model. Respects explicit overrides."""
    tp = args.train_tp or 1
    pp = args.train_pp or (1 if args.rollout_nvfp4 else 2)
    ep = args.train_ep or (4 if args.rollout_nvfp4 else 1)
    return tp, pp, ep


def _prepare_megatron_ckpt(args: ScriptArgs):
    # TODO unify 5layer w/ 20layer, also maybe unify the whole script
    extra_args = "--expert-tensor-parallel-size 1 "
    num_gpus_per_node = args.num_gpus_per_node
    multinode = True
    num_nodes = None
    layer_count = _get_layer_count(args.model_name)
    if layer_count == 5:
        tp, pp, ep = _get_5layer_parallel_config(args)
        extra_args += (
            f"--tensor-model-parallel-size {tp} "
            f"--pipeline-model-parallel-size {pp} "
            f"--expert-model-parallel-size {ep} "
        )
        if pp >= 2:
            last_pp_layers = 5 - (5 // pp) * (pp - 1)
            extra_args += f"--decoder-last-pipeline-num-layers {last_pp_layers} "
        num_gpus_per_node = min(4, num_gpus_per_node)
        multinode = False
    elif layer_count == 20:
        extra_args += (
            "--expert-model-parallel-size 4 "
            # PP info will be auto determined by converter script
        )
        num_nodes = 2
    else:
        extra_args += (
            "--pipeline-model-parallel-size 8 "
            "--expert-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 7 "
            "--decoder-last-pipeline-num-layers 6 "
        )

    U.convert_checkpoint(
        model_name=args.model_name,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}-bf16",
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_per_node,
        multinode=multinode,
        num_nodes=num_nodes,
        extra_args=extra_args,
        dir_dst=args.model_dir,
        megatron_path=args.megatron_path,
    )


def _prepare_cp(args: ScriptArgs):
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.model_name}_torch_dist",
        path_dst=f"{args.model_local_dir}/{args.model_name}_torch_dist",
    )
    rollout_src = f"{args.model_dir}/{_get_rollout_model_name(args)}"
    rollout_dst = f"{args.model_local_dir}/{_get_rollout_model_name(args)}"
    U.rsync_simple(
        path_src=rollout_src,
        path_dst=rollout_dst,
    )
    if args.rollout_nvfp4:
        runtime_link = f"{args.model_local_dir}/{_get_rollout_runtime_model_name(args)}"
        U.exec_command(f"mkdir -p {args.model_local_dir} && ln -sfn {rollout_dst} {runtime_link}")


def _execute_train(args: ScriptArgs):
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    rollout_model_name = _get_rollout_runtime_model_name(args)
    ckpt_args = (
        f"--hf-checkpoint {args.model_local_dir}/{rollout_model_name} "
        f"--ref-load {args.model_local_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
        "--save-retain-interval 20 "
    )
    if args.rollout_nvfp4 and args.rollout_nvfp4_restart_sync:
        ckpt_args += (
            "--rollout-nvfp4-restart-sync "
            f"--bridge-hf-checkpoint {args.model_dir}/{args.model_name}-bf16 "
            f"--rollout-refresh-parent-dir {args.output_dir}/{args.run_id}/rollout_sync "
            f"--rollout-refresh-keep-first-n {args.nvfp4_keep_first_n} "
            f"--rollout-refresh-keep-last-n {args.nvfp4_keep_last_n} "
        )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 1 "
        # ------------
        "--num-steps-per-rollout 4 "
        "--balance-data "
    )

    if args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 256 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    # sometimes disable eval to speed up debugging
    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 1 "

    match args.task:
        case "dapo_aime":
            rollout_args += (
                f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 32768} "
            )
            eval_args += (
                f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 8 "
                "--eval-max-response-len 32768 "
            )
        case "gsm8k":
            rollout_args += (
                f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
                "--input-key messages "
                # Deliberately make it very short for this easy task
                "--rollout-max-response-len 256 "
            )
            eval_args += (
                f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
                "--n-samples-per-eval-prompt 1 "
                "--eval-max-response-len 256 "
            )

    layer_count = _get_layer_count(args.model_name)
    if args.num_nodes <= 2 and layer_count == 5 and args.rollout_nvfp4:
        tp, pp, ep = _get_5layer_parallel_config(args)
        perf_args = (
            f"--tensor-model-parallel-size {tp} "
            f"--pipeline-model-parallel-size {pp} "
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {ep} "
            "--expert-tensor-parallel-size 1 "
        )
        if pp >= 2:
            last_pp_layers = 5 - (5 // pp) * (pp - 1)
            perf_args += f"--decoder-last-pipeline-num-layers {last_pp_layers} "
        if tp > 1:
            perf_args += "--sequence-parallel "
    elif args.num_nodes <= 2 and layer_count == 5:
        perf_args = (
            "--tensor-model-parallel-size 1 "
            "--pipeline-model-parallel-size 2 "
            "--decoder-last-pipeline-num-layers 2 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
        )
    elif args.num_nodes <= 2:
        perf_args = (
            "--tensor-model-parallel-size 1 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 4 "
            "--expert-model-parallel-size 4 "
            "--expert-tensor-parallel-size 1 "
        )
    elif args.num_nodes <= 4:
        # TODO remove this temp cfg
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 4 "
            "--expert-model-parallel-size 4 "
            "--expert-tensor-parallel-size 1 "
        )
    else:
        # TODO choose a good config (currently randomly change to suit 64gpu)
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            f"--pipeline-model-parallel-size {1 if _get_layer_count(args.model_name) == 5 else 4} "
            "--context-parallel-size 4 "
            "--expert-model-parallel-size 16 "
            "--expert-tensor-parallel-size 1 "
        )
        if re.search(r"(\d+)layer", args.model_name) is None:
            perf_args += "--decoder-last-pipeline-num-layers 13 "
    perf_args += (
        # ------------
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # ------------
        "--use-dynamic-batch-size "
        # TODO temp use tiny value
        "--max-tokens-per-gpu 2048 "
        # "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        # TODO run-deepseek-r1.sh enables use-kl-loss but w/ coef 0. can we just disable it like this?
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    low_memory_optimizer = args.rollout_nvfp4 and layer_count == 5 and not args.optimizer_cpu_offload

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        # ------------
        # "--optimizer-cpu-offload "
        # "--overlap-cpu-optimizer-d2h-h2d "
        # "--use-precision-aware-optimizer "
    )
    if args.optimizer_cpu_offload:
        optimizer_args += (
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )
    elif args.rollout_nvfp4 and layer_count == 5:
        optimizer_args += (
            "--use-precision-aware-optimizer "
            "--exp-avg-dtype bf16 "
            "--exp-avg-sq-dtype bf16 "
            "--main-grads-dtype bf16 "
        )

    sglang_decode_max_bs = 256
    sglang_world_size = 4 if args.num_nodes <= 4 else 64
    sglang_attn_dp_size = 1 if args.num_nodes <= 4 else 8
    sglang_attn_tp_size = sglang_world_size // sglang_attn_dp_size
    sglang_mem_fraction_static = 0.55 if (args.rollout_nvfp4 and layer_count == 5) else 0.7
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        f"--sglang-mem-fraction-static {sglang_mem_fraction_static} "
        f"--sglang-tp-size {sglang_world_size} "
        f"--sglang-ep-size {sglang_world_size} "
        # dp attention
        "--sglang-enable-dp-attention "
        f"--sglang-dp-size {sglang_attn_dp_size} "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        # make every dp rank has 128 concurrency
        "--sglang-server-concurrency 1024 "
        f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
        f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
        f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
        # For quick experiments
        # """--sglang-json-model-override-args '{"num_hidden_layers": 5}' """
    )
    sglang_extra_env_vars = {}
    if layer_count != 5:
        sglang_args += "--sglang-moe-a2a-backend deepep " "--sglang-deepep-mode low_latency "
        sglang_extra_env_vars["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = f"{sglang_decode_max_bs}"
    if args.rollout_nvfp4:
        sglang_args += (
            "--sglang-quantization modelopt_fp4 "
            "--sglang-attention-backend trtllm_mla "
            "--sglang-moe-runner-backend flashinfer_trtllm "
            "--sglang-kv-cache-dtype fp8_e4m3 "
            "--sglang-model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 64}' "
        )

    misc_args = (
        # default dropout in megatron is 0.1
        "--bf16 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
    )
    if low_memory_optimizer:
        misc_args += "--grad-reduce-in-bf16 "
    else:
        misc_args += "--accumulate-allreduce-grads-in-fp32 "

    update_weight_buffer_size = 512 * 1024**2 if low_memory_optimizer else 4 * 1024**3

    misc_args += (
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        # "--attention-backend flash "
        f"--update-weight-buffer-size {update_weight_buffer_size} "
        # TODO maybe enable it
        # use deepep for megatron
        # "--moe-enable-deepep "
        # "--moe-token-dispatcher-type flex "
        # ------------
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
        "--disable-weights-backuper "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        # TODO may get it from `config`
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**sglang_extra_env_vars},
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    _prepare_download(args)
    _prepare_bf16_ckpt(args)
    _prepare_nvfp4_ckpt(args)
    _prepare_megatron_ckpt(args)
    _prepare_cp(args)
    _execute_train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
