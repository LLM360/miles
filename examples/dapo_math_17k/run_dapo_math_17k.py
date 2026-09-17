"""Train dense Qwen3 on DAPO Math 17K with Megatron GRPO and Math-Verify rewards.

Install ``requirements.txt`` in the training environment first. The launcher
downloads missing model weights, converts them to Megatron torch_dist, and uses
the bundled dataset. Each invocation
starts one colocated training/rollout job through Miles' command helpers.

Args:
    --model-dir / --model-repo: Local model root and Hugging Face repository.
    --hf-checkpoint: Override the model checkpoint path.
    --dataset-path: Training JSONL; defaults to the bundled 17,398 examples.
    --output-dir / --save-dir: Run artifacts and training checkpoints.
    --load-path: Load a Megatron checkpoint, skipping initial weight conversion.
    --megatron-path: Compatible Megatron checkout added to the runtime PYTHONPATH.
    --enable-thinking / --no-enable-thinking: Pin Qwen3's chat-template switch.
    --seed / --rollout-seed: Training and rollout seeds.
    --lr / --min-lr / --lr-decay-style / --lr-decay-iters: Learning-rate schedule.
    --lr-warmup-init / --lr-warmup-iters: Initial warmup rate and update count.
    --wandb-project / --wandb-run-name: Optional W&B overrides; the project
        override requires WANDB_API_KEY in the environment.
    --dump-details: Save rollout/train data under the run's debug directory.
    --v1-experiment: Pin one of the eight 128K-response presets in README.md.
    --qwen3-long-context: Prepare a shared YaRN checkpoint view with 131072
        positions and validate every formatted prompt against the response budget.
    --context-parallel-size: Number of GPUs sharing each training sequence.
    --tensor-model-parallel-size: Number of GPUs sharing each model layer.
    --log-probs-chunk-size: Bound temporary vocabulary-sized log-prob tensors.
    --sglang-page-size / --sglang-kv-cache-dtype: Rollout KV-cache layout and precision.
    --sglang-max-running-requests: Concurrent requests per single-GPU engine.
    --sglang-cuda-graph-max-bs-decode: Largest captured decode batch.
    --extra-args: Additional train.py flags, unavailable with a fixed preset.

Example:
    python examples/dapo_math_17k/run_dapo_math_17k.py \
        --model-dir /path/to/models --model-repo Qwen/Qwen3-8B \
        --output-dir /path/to/run --num-gpus-per-node 2
"""

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import typer
from examples.dapo_math_17k.long_context import (
    CONTEXT_LENGTH,
    GENERATION_RESERVE,
    megatron_yarn_args,
    prepare_checkpoint,
    validate_prompt_budget,
)
from examples.dapo_math_17k.v1_config import experiment_settings

import miles.utils.external_utils.command_utils as U

_DEFAULT_DATASET_PATH = U.repo_base_dir / "examples/dapo_math_17k/dapo-math-17k.jsonl"
_REWARD_FUNCTION = "examples.dapo_math_17k.reward.reward_func"
_MODEL_TYPES = {"Qwen3-4B": "qwen3-4B", "Qwen3-8B": "qwen3-8B"}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    model_dir: str = "/root/models"
    model_repo: str = "Qwen/Qwen3-8B"
    dataset_path: str = str(_DEFAULT_DATASET_PATH)
    megatron_path: str = "/root/Megatron-LM"
    hf_checkpoint: str | None = None
    save_dir: str | None = None
    load_path: str | None = None
    v1_experiment: str | None = None
    qwen3_long_context: bool = False
    tensor_model_parallel_size: int = 2
    context_parallel_size: int = 1
    log_probs_chunk_size: int = -1

    num_gpus_per_node: int = 2
    num_rollout: int = 3000
    rollout_batch_size: int = 32
    over_sampling_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 0.7
    rollout_top_p: float = 0.8
    rollout_top_k: int = 20
    global_batch_size: int = 256
    max_tokens_per_gpu: int = 6144
    seed: int = 1234
    rollout_seed: int = 42
    enable_thinking: bool | None = None

    lr: float = 1e-6
    min_lr: float = 0.0
    lr_decay_style: str = "constant"
    lr_decay_iters: int | None = None
    lr_warmup_init: float = 0.0
    lr_warmup_iters: int = 0
    clip_grad: float = 1.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28

    sglang_mem_fraction_static: float = 0.6
    sglang_chunked_prefill_size: int = 4096
    disable_cuda_graph: bool = True
    sglang_context_length: int | None = None
    sglang_max_running_requests: int | None = None
    sglang_router_policy: str | None = None
    sglang_page_size: int | None = None
    sglang_kv_cache_dtype: str | None = None
    sglang_attention_backend: str | None = None
    sglang_cuda_graph_backend_decode: str | None = None
    sglang_cuda_graph_max_bs_decode: int | None = None
    save_interval: int = 100
    dump_details: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    extra_args: str = ""

    def __post_init__(self) -> None:
        if self.v1_experiment is not None:
            if self.extra_args:
                raise ValueError("v1-experiment pins the recipe; extra-args could override those settings")
            for key, value in experiment_settings(
                self.v1_experiment, repo_dir=U.repo_base_dir, output_root=self.output_dir
            ).items():
                setattr(self, key, value)
        if self.model_name not in _MODEL_TYPES:
            raise ValueError("This Megatron recipe supports Qwen3-4B and Qwen3-8B; set model-repo accordingly")
        if self.hf_checkpoint is None:
            self.hf_checkpoint = str(Path(self.model_dir) / self.model_name)
        if self.save_dir is None:
            self.save_dir = str(Path(self.output_dir) / "dapo_math_17k")
        if self.qwen3_long_context:
            if self.sglang_context_length is None:
                self.sglang_context_length = 131072
            if self.sglang_context_length != 131072:
                raise ValueError("Qwen3 long-context mode requires sglang_context_length=131072")
            if not 0 < self.rollout_max_response_len < self.sglang_context_length:
                raise ValueError("The response budget must leave room for the prompt")
        if self.over_sampling_batch_size <= 0:
            raise ValueError("over_sampling_batch_size must be positive")
        _validate_parallelism(self)

    @property
    def model_name(self) -> str:
        return self.model_repo.rstrip("/").rsplit("/", maxsplit=1)[-1]

    @property
    def megatron_model_type(self) -> str:
        return _MODEL_TYPES[self.model_name]

    @property
    def conversion_dir(self) -> str:
        return str(Path(self.output_dir) / "megatron_init")

    @property
    def initial_megatron_checkpoint(self) -> str:
        return str(Path(self.conversion_dir) / f"{self.model_name}_torch_dist")

    @property
    def prepared_hf_checkpoint(self) -> str:
        if self.qwen3_long_context:
            return str(Path(self.output_dir) / "hf_long_context")
        return self.hf_checkpoint

    @property
    def wandb_dir(self) -> str:
        return str(Path(self.output_dir) / "wandb")

    @property
    def dump_details_dir(self) -> str:
        return str(Path(self.output_dir) / "debug")


def _read_model_type(hf_checkpoint: str) -> str:
    model_config = Path(hf_checkpoint) / "config.json"
    try:
        config = json.loads(model_config.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read Hugging Face model config: {model_config}") from error
    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        raise ValueError(f"Hugging Face model config has no string model_type: {model_config}")
    return model_type


def _validate_parallelism(args: ScriptArgs) -> None:
    tp, cp = args.tensor_model_parallel_size, args.context_parallel_size
    world_size = args.num_nodes * args.num_gpus_per_node
    if min(tp, cp, args.num_nodes, args.num_gpus_per_node) < 1 or world_size % (tp * cp):
        raise ValueError("Training GPU count must be a positive multiple of TP * CP (pipeline parallelism is 1)")
    if 8 % tp or (8 // tp) % cp:
        raise ValueError("Qwen3's 8 KV heads must divide evenly across TP and a2a context parallelism")
    dp = world_size // (tp * cp)
    if args.global_batch_size <= 0 or args.global_batch_size % dp:
        raise ValueError("global_batch_size must be a positive multiple of data parallel size")
    if args.max_tokens_per_gpu <= 0:
        raise ValueError("max_tokens_per_gpu must be positive")
    if args.qwen3_long_context and args.max_tokens_per_gpu * cp < CONTEXT_LENGTH:
        raise ValueError("128K context requires max_tokens_per_gpu * context_parallel_size >= 131072")


def _backend_args(args: ScriptArgs) -> str:
    return (
        "--train-backend megatron --bf16 "
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        f"{'--sequence-parallel ' if args.tensor_model_parallel_size > 1 else ''}"
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {args.context_parallel_size} "
        f"{'--cp-comm-type a2a ' if args.context_parallel_size > 1 else ''}"
        "--attention-backend flash --attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--update-weight-buffer-size 536870912 "
        f"{megatron_yarn_args() if args.qwen3_long_context else ''}"
    )


def _chat_template_args(enable_thinking: bool | None) -> str:
    if enable_thinking is None:
        return ""
    return f"--apply-chat-template-kwargs {shlex.quote(json.dumps({'enable_thinking': enable_thinking}))} "


def _wandb_args(args: ScriptArgs) -> tuple[str, dict[str, str]]:
    if args.wandb_project is None:
        default_args = U.get_default_wandb_args(__file__, run_id=args.wandb_run_name or args.run_id)
        flags = shlex.split(default_args)
        if "--wandb-key" not in flags:
            return default_args, {}
        key_index = flags.index("--wandb-key")
        wandb_key = flags[key_index + 1]
        del flags[key_index : key_index + 2]
        return shlex.join(flags) + " ", {"WANDB_API_KEY": wandb_key}
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        raise ValueError("--wandb-project requires WANDB_API_KEY in the launcher environment")
    return (
        "--use-wandb --wandb-mode online "
        f"--wandb-project {shlex.quote(args.wandb_project)} "
        f"--wandb-group {shlex.quote(args.wandb_run_name or args.run_id)} "
        "--disable-wandb-random-suffix "
        f"--wandb-dir {shlex.quote(args.wandb_dir)} ",
        {"WANDB_API_KEY": wandb_key},
    )


def _prepare(args: ScriptArgs) -> None:
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"DAPO Math 17K dataset not found: {dataset_path}")
    if args.v1_experiment and args.load_path is None and any(Path(args.save_dir).glob("iter_*")):
        raise ValueError(f"Existing checkpoints in {args.save_dir}; use --load-path or a fresh output root")
    U.exec_command_cpu(f"mkdir -p {shlex.quote(args.model_dir)} {shlex.quote(args.save_dir)}")
    if not (Path(args.hf_checkpoint) / "config.json").exists():
        U.exec_command_cpu(f"hf download {shlex.quote(args.model_repo)} --local-dir {shlex.quote(args.hf_checkpoint)}")
    if _read_model_type(args.hf_checkpoint) != "qwen3":
        raise ValueError("This Megatron recipe requires a dense Qwen3 checkpoint")
    if args.qwen3_long_context:
        longest = validate_prompt_budget(
            dataset_path,
            Path(args.hf_checkpoint),
            response_length=args.rollout_max_response_len,
            enable_thinking=args.enable_thinking,
            expected_prompts=640 if args.v1_experiment else None,
        )
        prepare_checkpoint(
            Path(args.hf_checkpoint), Path(args.prepared_hf_checkpoint), context_length=args.sglang_context_length
        )
        print(f"Qwen3 long context: longest formatted prompt={longest}, response={args.rollout_max_response_len}")
    if args.load_path is None:
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            hf_checkpoint=args.prepared_hf_checkpoint,
            dir_dst=args.conversion_dir,
            megatron_path=args.megatron_path,
            extra_args=megatron_yarn_args() if args.qwen3_long_context else "",
        )


def _rollout_args(args: ScriptArgs) -> str:
    return (
        f"--prompt-data {shlex.quote(args.dataset_path)} "
        "--input-key prompt --label-key label --apply-chat-template "
        f"{_chat_template_args(args.enable_thinking)}"
        f"--rollout-shuffle --rollout-seed {args.rollout_seed} --balance-data "
        f"--custom-rm-path {_REWARD_FUNCTION} "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--over-sampling-batch-size {args.over_sampling_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--rollout-temperature {args.rollout_temperature} "
        f"--rollout-top-p {args.rollout_top_p} --rollout-top-k {args.rollout_top_k} "
        f"--global-batch-size {args.global_batch_size} "
    )


def _optimizer_args(args: ScriptArgs) -> str:
    decay = f"--lr-decay-iters {args.lr_decay_iters} " if args.lr_decay_iters is not None else ""
    return (
        f"--optimizer adam --lr {args.lr} --min-lr {args.min_lr} "
        f"--lr-decay-style {shlex.quote(args.lr_decay_style)} {decay}"
        f"--lr-warmup-init {args.lr_warmup_init} --lr-warmup-iters {args.lr_warmup_iters} "
        f"--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad {args.clip_grad} "
    )


def _runtime_extensions(args: ScriptArgs) -> str:
    values = {
        "--sglang-context-length": args.sglang_context_length,
        "--sglang-max-running-requests": args.sglang_max_running_requests,
        "--sglang-router-policy": args.sglang_router_policy,
        "--sglang-page-size": args.sglang_page_size,
        "--sglang-kv-cache-dtype": args.sglang_kv_cache_dtype,
        "--sglang-attention-backend": args.sglang_attention_backend,
        "--sglang-cuda-graph-backend-decode": args.sglang_cuda_graph_backend_decode,
        "--sglang-cuda-graph-max-bs-decode": args.sglang_cuda_graph_max_bs_decode,
    }
    if args.qwen3_long_context:
        values.update(
            {
                "--rollout-required-context-len": args.sglang_context_length,
                "--rollout-max-context-len": args.sglang_context_length - GENERATION_RESERVE,
            }
        )
    flags = [item for flag, value in values.items() if value is not None for item in (flag, str(value))]
    return shlex.join(flags) + " --skip-eval-before-train "


def _execute(args: ScriptArgs) -> None:
    checkpoint_args = (
        f"--hf-checkpoint {shlex.quote(args.prepared_hf_checkpoint)} "
        f"--load {shlex.quote(args.load_path or args.initial_megatron_checkpoint)} "
    )
    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.0 --kl-loss-type low_var_kl "
        "--kl-coef 0.00 --entropy-coef 0.00 "
        f"--eps-clip {args.eps_clip} --eps-clip-high {args.eps_clip_high} "
    )
    sglang_args = (
        f"--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-chunked-prefill-size {args.sglang_chunked_prefill_size} "
    )
    if args.disable_cuda_graph:
        sglang_args += "--sglang-disable-cuda-graph "
    perf_args = f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
    if args.log_probs_chunk_size > 0:
        perf_args += f"--log-probs-chunk-size {args.log_probs_chunk_size} --recompute-loss-function "
    save_args = f"--save {shlex.quote(args.save_dir)} --save-interval {args.save_interval} "
    misc_args = f"--seed {args.seed} "
    if args.load_path is None:
        misc_args += "--start-rollout-id 0 "
    if args.dump_details:
        misc_args += f"--dump-details {shlex.quote(args.dump_details_dir)} "
    topology_args = (
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} --colocate "
    )
    wandb_args, secret_env_vars = _wandb_args(args)
    U.execute_train(
        train_args=(
            f"{checkpoint_args}{_rollout_args(args)}{grpo_args}{_optimizer_args(args)}"
            f"{sglang_args}{_backend_args(args)}{perf_args}{save_args}{misc_args}{topology_args}"
            f"{_runtime_extensions(args)}{wandb_args}{args.extra_args} "
        ),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        secret_env_vars=secret_env_vars,
        extra_env_vars=(
            {"SGLANG_MAX_NEW_TOKENS_LIMIT": str(args.rollout_max_response_len)} if args.qwen3_long_context else None
        ),
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    _prepare(args)
    _execute(args)


if __name__ == "__main__":
    typer.run(main)
