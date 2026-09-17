"""The eight fixed GRPO experiments in README.md, with 128000-token responses.

Scientific settings follow the plan. Megatron TP=2/CP=4 and chunked log-probs
provide the long-sequence training path. Rollout uses eight requests per engine,
FP8 KV cache and decode CUDA graphs on Hopper GPUs.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Experiment:
    model_size: str
    warmup: bool
    thinking: bool


EXPERIMENTS = {
    f"qwen3-{size}_{'warmup' if warmup else 'nowarmup'}_{'thinking' if thinking else 'nothinking'}_seed42": _Experiment(
        size, warmup, thinking
    )
    for size in ("4b", "8b")
    for warmup in (True, False)
    for thinking in (True, False)
}


def experiment_settings(name: str, *, repo_dir: Path, output_root: str) -> dict:
    """Resolve one plan row; paths are rooted in the caller's configured directories."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown v1 experiment {name!r}; choose from {', '.join(EXPERIMENTS)}")
    experiment = EXPERIMENTS[name]
    return dict(
        model_repo=f"Qwen/Qwen3-{experiment.model_size.upper()}",
        dataset_path=str(repo_dir / "examples/dapo_math_17k/dapo-math-640-seed42.jsonl"),
        output_dir=str(Path(output_root) / name),
        num_gpus_per_node=8,
        num_rollout=100,
        rollout_batch_size=64,
        over_sampling_batch_size=32,
        n_samples_per_prompt=8,
        global_batch_size=512,
        rollout_max_response_len=128000,
        max_tokens_per_gpu=32768,
        rollout_temperature=1.0,
        rollout_top_p=0.95,
        rollout_top_k=20,
        seed=42,
        rollout_seed=42,
        enable_thinking=experiment.thinking,
        lr=5e-6,
        min_lr=2e-6,
        lr_decay_style="linear",
        lr_decay_iters=100,
        lr_warmup_init=1.414e-6 if experiment.warmup else 5e-6,
        lr_warmup_iters=10 if experiment.warmup else 0,
        save_interval=50,
        wandb_run_name=name,
        tensor_model_parallel_size=2,
        context_parallel_size=4,
        log_probs_chunk_size=512,
        qwen3_long_context=True,
        sglang_context_length=131072,
        sglang_max_running_requests=8,
        sglang_router_policy="round_robin",
        sglang_page_size=64,
        sglang_chunked_prefill_size=8192,
        sglang_kv_cache_dtype="fp8_e4m3",
        sglang_attention_backend="fa3",
        disable_cuda_graph=False,
        sglang_cuda_graph_backend_decode="full",
        sglang_cuda_graph_max_bs_decode=8,
    )
