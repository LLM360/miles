"""
Kimi K2.5 VL Training Script

=====================

Args:
  --model-name: Model variant to use.
      Kimi-K2.5       Full model (requires multi-node)
      Kimi-K2.5-5layer  5-layer pruned model (single-node testing)
  --num-nodes: Number of nodes for training.
      1  -> for Kimi-K2.5-5layer minimal test
      4+ -> for full Kimi-K2.5 model
  --mode: "normal" or "debug_minimal" (shorter response length for quick testing)

=====================

Usage for single node minimal test (5-layer):
  python scripts/run_kimi_k25_vl.py --model-name Kimi-K2.5-5layer --num-nodes 1
"""

import re
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_name: str = "Kimi-K2.5"
    megatron_model_type: str = "kimi-k2"
    num_gpus_per_node: int | None = None
    hardware: Literal["H100", "B200", "B300", "GB200", "GB300"] = "H100"
    enable_eval: bool = True
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    rollout_fp8: bool = False
    rollout_int4: bool = False
    train_fp8: bool = False

    def __post_init__(self):
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]
        if self.rollout_int4:
            assert not self.rollout_fp8, "rollout_int4 and rollout_fp8 cannot be enabled at the same time"
        if _is_pruned(self):
            self.mode = "debug_minimal"
        if (m := re.search(r"(\d+)layer", self.model_name)) is not None:
            self.megatron_model_type = f"kimi-k2-{m.group(1)}layer"


def _is_pruned(args: ScriptArgs):
    return re.search(r"(\d+)layer", args.model_name) is not None


def _model_org(model_name: str) -> str:
    if _is_pruned(type("", (), {"model_name": model_name})()):
        return "Pinaster"
    return "moonshotai"


def _patch_kimi_k25_modeling(model_dir: str):
    """Patch upstream bug: MoonViT3dEncoder references self.use_deterministic_attn before it's set."""
    import pathlib

    modeling_file = pathlib.Path(model_dir) / "modeling_kimi_k25.py"
    if not modeling_file.exists():
        return
    text = modeling_file.read_text()
    bad = "use_deterministic_attn=self.use_deterministic_attn)"
    fix = 'use_deterministic_attn=getattr(self, "use_deterministic_attn", False))'
    if bad in text:
        modeling_file.write_text(text.replace(bad, fix))
        print(f"Patched {modeling_file}")


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    org = _model_org(args.model_name)
    U.exec_command(f"huggingface-cli download {org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    _patch_kimi_k25_modeling(f"{args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)

    if args.rollout_fp8:
        U.exec_command(
            f"python tools/convert_hf_to_fp8.py "
            f"--model-dir {args.model_dir}/{args.model_name} "
            f"--save-dir {args.model_dir}/{args.model_name}-FP8"
        )

    if args.rollout_int4:
        U.exec_command(
            f"python tools/convert_hf_to_int4_direct.py "
            f"--model-dir {args.model_dir}/{args.model_name} "
            f"--save-dir {args.model_dir}/{args.model_name}-INT4"
        )


def execute(args: ScriptArgs):
    # Kimi K2.5 VL always uses megatron-bridge — no pre-conversion needed.
    hf_model_dir = f"{args.model_dir}/{args.model_name}"
    if args.rollout_fp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-FP8"
    elif args.rollout_int4:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-INT4"
    else:
        hf_checkpoint = hf_model_dir

    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {hf_checkpoint}/ "
        f"--ref-load {hf_model_dir}/ "
        # Don't pass --load: bridge mode auto-sets load=ref_load when load is None.
        # Passing a non-existent save path would cause an assertion error.
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        f"--rollout-batch-size {8 if _is_pruned(args) else 32} "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 16384} "
        "--rollout-temperature 1 "
        f"--global-batch-size {64 if _is_pruned(args) else 256} "
        "--balance-data "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {2048 if _is_pruned(args) else 16384} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    # megatron-bridge mode, no flash attention (MLA)
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--megatron-to-hf-mode bridge "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    misc_env_vars = {}

    if args.rollout_int4:
        misc_env_vars |= {
            "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
            "OPEN_TRAINING_INT4_GROUP_SIZE": "128",
        }

    if args.train_fp8:
        match args.hardware:
            case "B200" | "B300" | "GB200" | "GB300":
                misc_args += (
                    "--transformer-impl transformer_engine " "--bf16 " "--fp8-format e4m3 " "--fp8-recipe mxfp8 "
                )
            case "H100" | "H200":
                misc_args += (
                    "--transformer-impl transformer_engine " "--bf16 " "--fp8-format e4m3 " "--fp8-recipe blockwise "
                )
                misc_env_vars |= {
                    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
                }

    if args.num_nodes == 1:
        assert _is_pruned(args), (
            "num_nodes=1 only supports 5-layer model."
            "Full model requires num_nodes >= 32."
        )
        sglang_world_size = min(8, args.num_gpus_per_node)
        perf_args += (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {args.num_gpus_per_node} "
            "--expert-tensor-parallel-size 1 "
        )
        sglang_args = (
            f"--rollout-num-gpus-per-engine {sglang_world_size} "
            "--sglang-mem-fraction-static 0.7 "
            "--sglang-enable-dp-attention "
            f"--sglang-dp-size {sglang_world_size} "
            "--sglang-moe-dense-tp-size 1 "
            "--sglang-enable-dp-lm-head "
            f"--sglang-ep-size {sglang_world_size} "
            "--sglang-cuda-graph-max-bs 256 "
        )
    elif args.num_nodes % 32 == 0:
        # Multi-node full model (same parallelism as kimi-k2-Thinking)
        perf_args += (
            "--tensor-model-parallel-size 8 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 8 "
            "--context-parallel-size 4 "
            "--expert-model-parallel-size 32 "
            "--expert-tensor-parallel-size 1 "
            "--decoder-last-pipeline-num-layers 5 "
        )
        sglang_args = (
            "--rollout-num-gpus-per-engine 16 "
            "--sglang-mem-fraction-static 0.7 "
            "--sglang-enable-dp-attention "
            "--sglang-dp-size 8 "
            "--sglang-moe-dense-tp-size 1 "
            "--sglang-enable-dp-lm-head "
            "--sglang-ep-size 16 "
            "--sglang-server-concurrency 1024 "
        )
        optimizer_args += (
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )
        misc_args += "--moe-enable-deepep " "--moe-token-dispatcher-type flex "
    else:
        raise NotImplementedError(
            f"num_nodes={args.num_nodes} does not have a default parallelism config. "
            "Supported: num_nodes=1 (single-node pruned model test), "
            "num_nodes=32/64/... (multiples of 32, full model). "
            "Either customize the parallelism config in run_kimi_k25_vl.py or change num_nodes."
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
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**misc_env_vars},
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
