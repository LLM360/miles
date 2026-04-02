import miles.utils.external_utils.command_utils as U

from tests.e2e.ft.conftest_ft.modes import FTTestMode


def get_common_train_args(mode: FTTestMode, *, dump_dir: str, num_steps: int | None = None) -> str:
    """Base args shared by all FT tests.

    Includes: checkpoint paths, amplified-error optimizer hyperparams,
    debug rollout data, dumper config, event logger, wandb args.
    """
    ckpt_args = (
        f"--hf-checkpoint /root/models/{mode.model_name} "
        f"--ref-load /root/{mode.model_name}_torch_dist "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-3 --lr-decay-style constant "
        "--adam-beta1 0.1 --adam-beta2 0.9 "
        "--lr-warmup-fraction 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
    )

    train_gpus: int = mode.num_gpus_total - mode.rollout_gpus

    debug_rollout_args: str
    if mode.rollout_gpus == 0:
        debug_rollout_args = (
            "--load-debug-rollout-data /root/datasets/ft-test-debug-rollout-data/{rollout_id}.pt "
            "--debug-train-only "
        )
    else:
        debug_rollout_args = (
            "--prompt-data /root/datasets/gsm8k/train.parquet "
            "--input-key messages --label-key label --apply-chat-template "
            "--rollout-shuffle --rm-type math "
            "--rollout-max-response-len 3 --rollout-temperature 0.8 "
            "--rollout-batch-size 1 --n-samples-per-prompt 1 "
            "--sglang-disable-cuda-graph "
        )

    event_logger_args = f"--save-debug-event-data {dump_dir}/events "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {train_gpus} "
        "--global-batch-size 1 "
        "--moe-token-dispatcher-type alltoall "
        "--advantage-estimator grpo --eps-clip 0.2 "
    )

    misc_args += f"--num-rollout {num_steps if num_steps is not None else mode.num_steps} "

    if mode.rollout_gpus > 0:
        misc_args += f"--rollout-num-gpus {mode.rollout_gpus} --colocate "

    return " ".join([
        ckpt_args,
        optimizer_args,
        debug_rollout_args,
        event_logger_args,
        mode.parallel_args,
        misc_args,
        U.get_default_wandb_args(__file__),
    ])


def get_indep_dp_args(mode: FTTestMode) -> str:
    """Args for indep_dp (fault-tolerant) training."""
    return (
        "--use-fault-tolerance --ft-components train "
        "--control-server-port 0 "
    )


def run_training(train_args: str, mode: FTTestMode) -> None:
    """Wrapper around execute_train for FT tests."""
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=mode.num_gpus_total,
        megatron_model_type=mode.megatron_model_type,
    )
