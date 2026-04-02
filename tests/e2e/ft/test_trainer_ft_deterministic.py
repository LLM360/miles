# NOTE: Please refer to tests/e2e/ft/README.md for documentations and source-of-truth
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import sys
from pathlib import Path

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

from miles.utils.test_utils.metric_comparison import compare_metrics
from tests.e2e.ft.conftest_ft.app import create_comparison_app
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_indep_dp_args
from tests.e2e.ft.conftest_ft.modes import FTTestMode


def _build_baseline_args(mode: FTTestMode, dump_dir: str) -> str:
    return get_common_train_args(mode, dump_dir=dump_dir)


def _build_target_args(mode: FTTestMode, dump_dir: str) -> str:
    return (
        get_common_train_args(mode, dump_dir=dump_dir)
        + get_indep_dp_args(mode)
        + "--ci-ft-test-scenario deterministic "
        + "--save-local-weight-checksum "
        + "--enable-event-analyzer "
    )


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    compare_metrics(
        baseline_dir=f"{dump_dir}/baseline",
        target_dir=f"{dump_dir}/target",
        rtol=1e-2,
        key_prefixes=["train/"],
    )
    print("Deterministic healing comparison test PASSED")


app = create_comparison_app(
    build_baseline_args=_build_baseline_args,
    build_target_args=_build_target_args,
    compare_fn=_compare,
)

if __name__ == "__main__":
    app()
