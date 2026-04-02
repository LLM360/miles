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

from tests.e2e.ft.conftest_ft import (
    FTTestMode,
    create_non_comparison_app,
    get_common_train_args,
    get_indep_dp_args,
)


def _build_args(mode: FTTestMode, dump_dir: str) -> str:
    base = get_common_train_args(mode, dump_dir=dump_dir)
    base += get_indep_dp_args(mode)
    base += "--ci-ft-test-scenario deterministic "
    base += "--save-local-weight-checksum "
    base += "--enable-event-analyzer "
    return base


def _verify(dump_dir: str, mode: FTTestMode) -> None:
    """Verification is handled by the event analyzer inside the training job.

    The event_analyzer cross_replica_weight_checksum rule asserts bitwise
    equality of LocalWeightChecksumEvent across all alive cells after healing.
    If the analyzer finds a mismatch, the training job exits non-zero.

    Here we just confirm the events directory was written.
    """
    events_dir: Path = Path(dump_dir) / "events"
    assert events_dir.exists(), f"Events directory not found: {events_dir}"

    jsonl_files = list(events_dir.glob("**/*.jsonl"))
    assert len(jsonl_files) > 0, f"No event files found in {events_dir}"
    print("Deterministic healing verification test PASSED (event analyzer checked in-job)")


app = create_non_comparison_app(
    build_args=_build_args,
    verify_fn=_verify,
)

if __name__ == "__main__":
    app()
