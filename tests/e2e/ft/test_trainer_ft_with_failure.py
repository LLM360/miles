# NOTE: Please refer to tests/e2e/ft/README.md for documentations and source-of-truth
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import sys
from pathlib import Path

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

from tests.e2e.ft.conftest_ft.app import create_comparison_app
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_indep_dp_args
from miles.utils.test_utils.metric_comparison import compare_dumps, compare_metrics
from tests.e2e.ft.conftest_ft.modes import FTTestMode

NUM_PHASE_A_STEPS: int = 1
NUM_PHASE_B_STEPS: int = 4


def _build_phase_args(mode: FTTestMode, dump_dir: str, *, is_target: bool) -> str:
    is_phase_a: bool = dump_dir.endswith("phase_a")
    num_steps = NUM_PHASE_A_STEPS if is_phase_a else NUM_PHASE_B_STEPS
    base = get_common_train_args(mode, dump_dir=dump_dir, num_steps=num_steps)

    if is_target:
        base += get_indep_dp_args(mode)

    if is_phase_a:
        base += f"--save {dump_dir}/ckpt --save-interval 1 "
    else:
        phase_a_dir = dump_dir.replace("/phase_b", "/phase_a")
        base += f"--load {phase_a_dir}/ckpt "
        if is_target:
            base += "--ci-ft-test-scenario with_failure "

    return base


def _build_baseline_args(mode: FTTestMode, dump_dir: str) -> str:
    return _build_phase_args(mode, dump_dir, is_target=False)


def _build_target_args(mode: FTTestMode, dump_dir: str) -> str:
    return _build_phase_args(mode, dump_dir, is_target=True)


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    compare_metrics(
        baseline_dir=f"{dump_dir}/baseline/phase_b",
        target_dir=f"{dump_dir}/target/phase_b",
        rtol=5e-2,
        key_prefixes=["train/"],
    )
    compare_dumps(
        baseline_dir=f"{dump_dir}/baseline/phase_b",
        target_dir=f"{dump_dir}/target/phase_b",
    )
    print("With-failure comparison test PASSED")


app = create_comparison_app(
    build_baseline_args=_build_baseline_args,
    build_target_args=_build_target_args,
    compare_fn=_compare,
    phases=["phase_a", "phase_b"],
)

if __name__ == "__main__":
    app()
