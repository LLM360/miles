# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

# Usage:
#   python test_trainer_ft_deterministic.py run --mode dp2_cp2_tp2_ep2

# This is a non-comparison test. It verifies that after healing (stop+start
# without missing any step), the healed cell has bitwise-equal weights and
# optimizer state compared to the source cell. Verification is done via the
# existing LocalWeightChecksumEvent + event_analyzer cross_replica_weight_checksum rule.

import sys
from pathlib import Path

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

from tests.e2e.ft.conftest_ft.app import create_non_comparison_app
from tests.e2e.ft.conftest_ft.args import get_common_train_args, get_indep_dp_args
from tests.e2e.ft.conftest_ft.comparison import assert_events_dir_exists
from tests.e2e.ft.conftest_ft.modes import FTTestMode


def _build_args(mode: FTTestMode, dump_dir: str) -> str:
    return (
        get_common_train_args(mode, dump_dir=dump_dir)
        + get_indep_dp_args(mode)
        + "--ci-ft-test-scenario deterministic "
        + "--save-local-weight-checksum "
        + "--enable-event-analyzer "
    )


def _verify(dump_dir: str, mode: FTTestMode) -> None:
    """Verification is handled by the event analyzer inside the training job.

    The event_analyzer cross_replica_weight_checksum rule asserts bitwise
    equality of LocalWeightChecksumEvent across all alive cells after healing.
    If the analyzer finds a mismatch, the training job exits non-zero.
    """
    assert_events_dir_exists(dump_dir)
    print("Deterministic healing verification test PASSED (event analyzer checked in-job)")


app = create_non_comparison_app(
    build_args=_build_args,
    verify_fn=_verify,
)

if __name__ == "__main__":
    app()
