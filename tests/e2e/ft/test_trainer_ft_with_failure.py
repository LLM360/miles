# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

# Usage:
#   python test_trainer_ft_with_failure.py baseline phase_a --mode dp4_cp2 --dump-dir /tmp/ft
#   python test_trainer_ft_with_failure.py target   phase_a --mode dp4_cp2 --dump-dir /tmp/ft
#   python test_trainer_ft_with_failure.py baseline phase_b --mode dp4_cp2 --dump-dir /tmp/ft
#   python test_trainer_ft_with_failure.py target   phase_b --mode dp4_cp2 --dump-dir /tmp/ft
#   python test_trainer_ft_with_failure.py compare          --mode dp4_cp2 --dump-dir /tmp/ft
#   python test_trainer_ft_with_failure.py run              --mode dp4_cp2

import sys
from pathlib import Path

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

from tests.e2e.ft.conftest_ft.app import create_comparison_app
from tests.e2e.ft.conftest_ft.args import get_common_train_args, get_indep_dp_args
from tests.e2e.ft.conftest_ft.comparison import compare_metrics
from tests.e2e.ft.conftest_ft.modes import FTTestMode

NUM_PHASE_A_STEPS: int = 1
NUM_PHASE_B_STEPS: int = 4


def _build_baseline_args_phase_a(mode: FTTestMode, dump_dir: str) -> str:
    base = get_common_train_args(mode, dump_dir=dump_dir, num_steps=NUM_PHASE_A_STEPS)
    base += f"--save {dump_dir}/ckpt --save-interval 1 "
    return base


def _build_target_args_phase_a(mode: FTTestMode, dump_dir: str) -> str:
    base = get_common_train_args(mode, dump_dir=dump_dir, num_steps=NUM_PHASE_A_STEPS)
    base += get_indep_dp_args(mode)
    base += f"--save {dump_dir}/ckpt --save-interval 1 "
    return base


def _build_baseline_args_phase_b(mode: FTTestMode, dump_dir: str) -> str:
    phase_a_dir = dump_dir.replace("/phase_b", "/phase_a")
    base = get_common_train_args(mode, dump_dir=dump_dir, num_steps=NUM_PHASE_B_STEPS)
    base += f"--load {phase_a_dir}/ckpt "
    return base


def _build_target_args_phase_b(mode: FTTestMode, dump_dir: str) -> str:
    phase_a_dir = dump_dir.replace("/phase_b", "/phase_a")
    base = get_common_train_args(mode, dump_dir=dump_dir, num_steps=NUM_PHASE_B_STEPS)
    base += get_indep_dp_args(mode)
    base += f"--load {phase_a_dir}/ckpt "
    base += "--ci-ft-test-scenario with_failure "
    return base


def _build_baseline_args(mode: FTTestMode, dump_dir: str) -> str:
    if dump_dir.endswith("phase_a"):
        return _build_baseline_args_phase_a(mode, dump_dir)
    return _build_baseline_args_phase_b(mode, dump_dir)


def _build_target_args(mode: FTTestMode, dump_dir: str) -> str:
    if dump_dir.endswith("phase_a"):
        return _build_target_args_phase_a(mode, dump_dir)
    return _build_target_args_phase_b(mode, dump_dir)


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    compare_metrics(
        baseline_dir=f"{dump_dir}/baseline/phase_b",
        target_dir=f"{dump_dir}/target/phase_b",
        rtol=5e-2,
        key_prefixes=["train/"],
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
