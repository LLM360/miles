import os

import miles.utils.external_utils.command_utils as U

TIGHT_HOST_MEMORY = bool(int(os.environ.get("MILES_TEST_TIGHT_HOST_MEMORY", "1")))

MODEL_TYPE = os.environ.get("MILES_TEST_MOVA_MODEL_TYPE", "k2mova-36b")
HF_DIR = os.environ.get(
    "MILES_TEST_MOVA_HF_DIR",
    "/mnt/weka/shrd/k2m/yash.akhauri/k2mova36b/k2mova-36b-mid3_v3/checkpoint_0005500",
)
TORCH_DIST = os.environ.get(
    "MILES_TEST_MOVA_TORCH_DIST",
    "/mnt/weka/shrd/k2m/yash.akhauri/k2mova36b/megatron-k2mova-36b-mid3-v3-0005500-native-router-meta",
)
PROMPT_DATA = os.environ.get(
    "MILES_TEST_MOVA_PROMPT_DATA",
    "/mnt/weka/shrd/k2pta/datasets/dapo-math-17k/dapo-math-17k.jsonl",
)
RAY_PYTHONPATH = os.environ.get("MILES_TEST_MOVA_RAY_PYTHONPATH", "/root/Megatron-LM")
NUM_GPUS = 8


def execute():
    for path in (HF_DIR, TORCH_DIST, PROMPT_DATA):
        if not os.path.exists(path):
            raise FileNotFoundError(f"MoVA R3 e2e needs pre-staged inputs; missing: {path}")

    ckpt_args = f"--hf-checkpoint {HF_DIR} --ref-load {TORCH_DIST} "

    rollout_args = (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 1 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1 "
        "--global-batch-size 8 "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        f"{'' if TIGHT_HOST_MEMORY else '--use-kl-loss '}"
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
        "--use-rollout-routing-replay "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-dtype bfloat16 "
        """--sglang-json-model-override-args '{"xllm_source_router_gemm_partitions": 2}' """
        f"--sglang-mem-fraction-static {0.7 if TIGHT_HOST_MEMORY else 0.8} "
        "--sglang-max-running-requests 512 "
        # The image's Rust gateway predates R3 and drops `return_routed_experts`
        # from the request; miles' Python router proxies the body verbatim.
        "--use-miles-router "
    )

    ci_args = "--ci-test --ci-disable-kl-checker "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={"PYTHONPATH": RAY_PYTHONPATH},
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
