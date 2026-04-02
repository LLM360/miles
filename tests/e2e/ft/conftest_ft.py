# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

import typer

import miles.utils.external_utils.command_utils as U
from miles.utils.event_logger.logger import read_events
from miles.utils.event_logger.models import MetricEvent

MODEL_NAME: str = "Qwen3-30B-A3B-5layer"
MODEL_HF_REPO: str = f"fzyzcjy/{MODEL_NAME}"
MODEL_TYPE: str = "qwen3-30B-A3B"
DEBUG_ROLLOUT_DATA_HF_REPO: str = "fzyzcjy/miles-ft-test-debug-rollout-data"


@dataclass(frozen=True)
class FTTestMode:
    name: str
    model_name: str
    megatron_model_type: str
    num_gpus_total: int
    num_cells: int
    parallel_args: str
    rollout_gpus: int
    num_steps: int = 10


MODES: dict[str, FTTestMode] = {
    "dp2_cp2_tp2_ep2": FTTestMode(
        name="dp2_cp2_tp2_ep2",
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_total=8,
        num_cells=2,
        rollout_gpus=0,
        parallel_args=(
            "--tensor-model-parallel-size 2 --context-parallel-size 2 "
            "--expert-model-parallel-size 2 --sequence-parallel"
        ),
    ),
    "dp2_cp2_pp2": FTTestMode(
        name="dp2_cp2_pp2",
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_total=8,
        num_cells=2,
        rollout_gpus=0,
        parallel_args="--pipeline-model-parallel-size 2 --context-parallel-size 2",
    ),
    "dp4_cp2": FTTestMode(
        name="dp4_cp2",
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_total=8,
        num_cells=4,
        rollout_gpus=0,
        parallel_args="--context-parallel-size 2",
    ),
    "dp2_cp2_real_rollout": FTTestMode(
        name="dp2_cp2_real_rollout",
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_total=8,
        num_cells=2,
        rollout_gpus=4,
        parallel_args="--context-parallel-size 2",
    ),
}


def resolve_mode(mode: str) -> FTTestMode:
    if mode not in MODES:
        raise typer.BadParameter(f"Unknown mode {mode!r}, valid: {list(MODES.keys())}")
    return MODES[mode]


def prepare(mode: FTTestMode) -> None:
    """Download trimmed model, convert checkpoint, download debug rollout data."""
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download {MODEL_HF_REPO} --local-dir /root/models/{mode.model_name}")
    U.convert_checkpoint(
        model_name=mode.model_name,
        megatron_model_type=mode.megatron_model_type,
        num_gpus_per_node=mode.num_gpus_total,
    )
    U.exec_command(
        f"hf download --repo-type dataset {DEBUG_ROLLOUT_DATA_HF_REPO} "
        f"--local-dir /root/datasets/ft-test-debug-rollout-data"
    )


def get_common_train_args(mode: FTTestMode, *, dump_dir: str) -> str:
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
            "--num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 1 "
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


def get_normal_dp_args(mode: FTTestMode) -> str:
    """Args for normal DP baseline training (no fault tolerance)."""
    return ""


def run_training(train_args: str, mode: FTTestMode) -> None:
    """Wrapper around execute_train for FT tests."""
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=mode.num_gpus_total,
        megatron_model_type=mode.megatron_model_type,
    )


def read_metric_events(dump_dir: Path) -> list[MetricEvent]:
    """Read MetricEvents from the event logger output directory."""
    events_dir: Path = dump_dir / "events"
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, MetricEvent)]


def compare_metrics(
    baseline_dir: str,
    target_dir: str,
    *,
    rtol: float = 1e-3,
    key_prefixes: list[str] | None = None,
) -> None:
    """Compare MetricEvents between baseline and target runs.

    Focuses on train/* metrics by default (grad_norm, loss).
    """
    if key_prefixes is None:
        key_prefixes = ["train/"]

    baseline_metrics = read_metric_events(Path(baseline_dir))
    target_metrics = read_metric_events(Path(target_dir))

    assert len(baseline_metrics) > 0, f"No MetricEvents found in baseline dir: {baseline_dir}"
    assert len(target_metrics) > 0, f"No MetricEvents found in target dir: {target_dir}"
    assert len(baseline_metrics) == len(target_metrics), (
        f"MetricEvent count mismatch: baseline={len(baseline_metrics)}, target={len(target_metrics)}"
    )

    for step_idx, (b_event, t_event) in enumerate(zip(baseline_metrics, target_metrics, strict=True)):
        for key in b_event.metrics:
            if not any(key.startswith(prefix) for prefix in key_prefixes):
                continue
            if key not in t_event.metrics:
                continue

            b_val = b_event.metrics[key]
            t_val = t_event.metrics[key]
            if b_val == 0.0 and t_val == 0.0:
                continue

            rel_diff = abs(b_val - t_val) / max(abs(b_val), abs(t_val), 1e-12)
            assert rel_diff <= rtol, (
                f"Step {step_idx}, metric '{key}': baseline={b_val}, target={t_val}, "
                f"rel_diff={rel_diff:.6f} > rtol={rtol}"
            )

    print(f"MetricEvent comparison passed: {len(baseline_metrics)} steps compared")


@dataclass
class _ComparisonAppConfig:
    build_baseline_args: Callable[[FTTestMode, str], str]
    build_target_args: Callable[[FTTestMode, str], str]
    compare_fn: Callable[[str, FTTestMode], None]
    phases: list[str] = field(default_factory=lambda: [])


def create_comparison_app(
    *,
    build_baseline_args: Callable[[FTTestMode, str], str],
    build_target_args: Callable[[FTTestMode, str], str],
    compare_fn: Callable[[str, FTTestMode], None],
    phases: list[str] | None = None,
) -> typer.Typer:
    """Generate a typer app with baseline/target/compare/run commands.

    For simple (no-phase) tests, leave phases empty.
    For multi-phase tests (e.g. with_failure), provide phase names like ["phase_a", "phase_b"].
    """
    app: typer.Typer = typer.Typer()
    effective_phases: list[str] = phases or [""]

    def _get_dump_subdir(side: str, phase: str) -> str:
        if phase:
            return f"{side}/{phase}"
        return side

    @app.command()
    def baseline(
        mode: Annotated[str, typer.Option(help="Test mode variant")] = "dp2_cp2_tp2_ep2",
        dump_dir: Annotated[str, typer.Option(help="Dump base directory")] = "",
        phase: Annotated[str, typer.Option(help="Phase name (multi-phase tests)")] = "",
    ) -> None:
        """Run baseline (normal DP) training."""
        ft_mode = resolve_mode(mode)
        if not dump_dir:
            dump_dir = str(Path(tempfile.mkdtemp(prefix="ft_test_")) / "dumps")
        sub = _get_dump_subdir("baseline", phase)
        full_dump_dir = f"{dump_dir}/{sub}"
        args = build_baseline_args(ft_mode, full_dump_dir)
        prepare(ft_mode)
        run_training(train_args=args, mode=ft_mode)

    @app.command()
    def target(
        mode: Annotated[str, typer.Option(help="Test mode variant")] = "dp2_cp2_tp2_ep2",
        dump_dir: Annotated[str, typer.Option(help="Dump base directory")] = "",
        phase: Annotated[str, typer.Option(help="Phase name (multi-phase tests)")] = "",
    ) -> None:
        """Run target (indep_dp) training."""
        ft_mode = resolve_mode(mode)
        if not dump_dir:
            dump_dir = str(Path(tempfile.mkdtemp(prefix="ft_test_")) / "dumps")
        sub = _get_dump_subdir("target", phase)
        full_dump_dir = f"{dump_dir}/{sub}"
        args = build_target_args(ft_mode, full_dump_dir)
        prepare(ft_mode)
        run_training(train_args=args, mode=ft_mode)

    @app.command()
    def compare(
        mode: Annotated[str, typer.Option(help="Test mode variant")] = "dp2_cp2_tp2_ep2",
        dump_dir: Annotated[str, typer.Option(help="Dump base directory")] = "",
    ) -> None:
        """Compare baseline and target dumps."""
        ft_mode = resolve_mode(mode)
        compare_fn(dump_dir, ft_mode)

    @app.command()
    def run(
        mode: Annotated[str, typer.Option(help="Test mode variant")] = "dp2_cp2_tp2_ep2",
    ) -> None:
        """Full pipeline: prepare + all phases + compare."""
        ft_mode = resolve_mode(mode)
        dump_dir: str = str(Path(tempfile.mkdtemp(prefix="ft_test_")) / "dumps")
        print(f"Dump directory: {dump_dir}")

        prepare(ft_mode)

        for phase in effective_phases:
            sub_baseline = _get_dump_subdir("baseline", phase)
            sub_target = _get_dump_subdir("target", phase)

            baseline_args = build_baseline_args(ft_mode, f"{dump_dir}/{sub_baseline}")
            run_training(train_args=baseline_args, mode=ft_mode)

            target_args = build_target_args(ft_mode, f"{dump_dir}/{sub_target}")
            run_training(train_args=target_args, mode=ft_mode)

        compare_fn(dump_dir, ft_mode)

    return app


def create_non_comparison_app(
    *,
    build_args: Callable[[FTTestMode, str], str],
    verify_fn: Callable[[str, FTTestMode], None] | None = None,
) -> typer.Typer:
    """Generate a typer app with a single 'run' command for non-comparison tests."""
    app: typer.Typer = typer.Typer()

    @app.command()
    def run(
        mode: Annotated[str, typer.Option(help="Test mode variant")] = "dp2_cp2_tp2_ep2",
    ) -> None:
        """Full pipeline: prepare + execute + verify."""
        ft_mode = resolve_mode(mode)
        dump_dir: str = str(Path(tempfile.mkdtemp(prefix="ft_test_")) / "dumps")
        print(f"Dump directory: {dump_dir}")

        prepare(ft_mode)
        args = build_args(ft_mode, dump_dir)
        run_training(train_args=args, mode=ft_mode)

        if verify_fn is not None:
            verify_fn(dump_dir, ft_mode)

    return app
