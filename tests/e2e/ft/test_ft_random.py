# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

# Usage:
#   python test_ft_random_failure.py run --mode dp4_cp2
#   python test_ft_random_failure.py run --mode dp4_cp2 --seed 42 --num-steps 50

# This is a non-comparison soak test. Train actors randomly crash themselves
# via ray concurrency group background threads. The mini FT controller
# auto-recovers crashed cells. Verifies no hangs, no assertion failures,
# and final weights are loadable.

import sys
import tempfile
from pathlib import Path
from typing import Annotated

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

import typer

from tests.e2e.ft.conftest_ft.comparison import assert_events_dir_exists
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_indep_dp_args, prepare, run_training
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode

app: typer.Typer = typer.Typer()

DEFAULT_NUM_STEPS: int = 30
DEFAULT_CRASH_PROBABILITY: float = 0.1


@app.command()
def run(
    mode: Annotated[str, typer.Option(help="Test mode variant")],
    seed: Annotated[int, typer.Option(help="Random seed for fault injection")] = 42,
    num_steps: Annotated[int, typer.Option(help="Number of train() calls")] = DEFAULT_NUM_STEPS,
    crash_probability: Annotated[float, typer.Option(help="Per-step crash probability per cell")] = DEFAULT_CRASH_PROBABILITY,
) -> None:
    """Run random failure injection soak test.

    The mini FT controller handles automatic recovery. The test verifies
    that training completes without hanging and all assertions pass.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    dump_dir: str = str(Path(tempfile.mkdtemp(prefix="ft_random_failure_")) / "dumps")
    print(f"Dump directory: {dump_dir}")
    print(f"Seed: {seed}, Steps: {num_steps}, Crash probability: {crash_probability}")

    prepare(ft_mode)

    base = (
        get_common_train_args(ft_mode, dump_dir=dump_dir, num_steps=num_steps)
        + get_indep_dp_args(ft_mode)
        + "--mini-ft-controller-enable "
        + f"--ci-ft-test-scenario random_failure "
        + f"--ci-ft-random-seed {seed} "
        + f"--ci-ft-crash-probability {crash_probability} "
        + "--save-local-weight-checksum "
        + "--enable-event-analyzer "
    )

    run_training(train_args=base, mode=ft_mode)

    assert_events_dir_exists(dump_dir)
    print(f"Random failure soak test PASSED (seed={seed}, steps={num_steps})")


if __name__ == "__main__":
    app()
