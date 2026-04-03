# NOTE: Please refer to tests/e2e/ft/README.md for documentations and source-of-truth
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import json
import sys
from pathlib import Path

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

from miles.utils.test_utils.comparisons import compare_dumps, compare_metrics
from tests.e2e.ft.conftest_ft.app import create_comparison_app
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_indep_dp_args
from tests.e2e.ft.conftest_ft.modes import FTTestMode

NUM_PHASE_A_STEPS: int = 1
NUM_PHASE_B_STEPS: int = 5

# rollout_id in phase_b starts from NUM_PHASE_A_STEPS (ckpt resume offset)
_DETERMINISTIC_ACTIONS: list[dict] = [
    {"after_step": NUM_PHASE_A_STEPS + 1, "action": "stop_cell", "cell_index": -1},
    {"after_step": NUM_PHASE_A_STEPS + 1, "action": "start_cell", "cell_index": -1},
    {"after_step": NUM_PHASE_A_STEPS + 2, "action": "stop_cell", "cell_index": -1},
    {"after_step": NUM_PHASE_A_STEPS + 3, "action": "start_cell", "cell_index": -1},
]


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
            base += (
                f"--ci-ft-test-actions '{json.dumps(_DETERMINISTIC_ACTIONS)}' "
                "--save-local-weight-checksum "
                "--enable-event-analyzer "
            )

    return base


def _build_baseline_args(mode: FTTestMode, dump_dir: str) -> str:
    return _build_phase_args(mode, dump_dir, is_target=False)


def _build_target_args(mode: FTTestMode, dump_dir: str) -> str:
    return _build_phase_args(mode, dump_dir, is_target=True)


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    compare_metrics(
        baseline_dir=f"{dump_dir}/baseline/phase_b",
        target_dir=f"{dump_dir}/target/phase_b",
        rtol=1e-2,
        key_prefixes=["train/"],
    )
    compare_dumps(
        baseline_dir=f"{dump_dir}/baseline/phase_b",
        target_dir=f"{dump_dir}/target/phase_b",
    )
    print("Deterministic healing comparison test PASSED")


app = create_comparison_app(
    build_baseline_args=_build_baseline_args,
    build_target_args=_build_target_args,
    compare_fn=_compare,
    phases=["phase_a", "phase_b"],
)

if __name__ == "__main__":
    app()
