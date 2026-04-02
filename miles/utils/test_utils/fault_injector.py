# NOTE: Please refer to tests/e2e/ft/README.md for documentations and source-of-truth
"""Actor-side fault injection for FT soak tests.

Runs in a dedicated ray concurrency group thread inside each train actor.
The main training thread is unaffected until the fault fires — the crash
is genuinely unexpected from the training code's perspective.

Failure modes are modeled after torchft's failure.py:
https://github.com/meta-pytorch/torchft/blob/main/examples/monarch/utils/failure.py

Integration:
  1. Add "fault_injector": 1 to the actor's concurrency_groups dict
  2. Call start_fault_injector() from the actor, decorated with
     @ray.method(concurrency_group="fault_injector")
"""

import ctypes
import logging
import os
import random
import signal
import time
from enum import Enum

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    SIGKILL = "sigkill"
    EXIT = "exit"
    SEGFAULT = "segfault"
    DEADLOCK = "deadlock"


def _execute_failure(mode: FailureMode) -> None:
    logger.warning("FaultInjector: executing failure mode %s (pid=%d)", mode.value, os.getpid())

    match mode:
        case FailureMode.SIGKILL:
            os.kill(os.getpid(), signal.SIGKILL)

        case FailureMode.EXIT:
            os._exit(1)

        case FailureMode.SEGFAULT:
            crash_func = ctypes.CFUNCTYPE(None)()
            crash_func()

        case FailureMode.DEADLOCK:
            libc = ctypes.PyDLL(None)
            libc.sleep.argtypes = (ctypes.c_uint,)
            libc.sleep.restype = ctypes.c_uint
            libc.sleep(600)


_FAILURE_MODES_FAST: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]
_FAILURE_MODES_ALL: list[FailureMode] = list(FailureMode)


def run_fault_injector(
    *,
    seed: int,
    crash_probability: float,
    step_interval_seconds: float = 5.0,
    include_deadlock: bool = False,
) -> None:
    """Blocking loop that randomly crashes the current process.

    Designed to run in a ray concurrency group background thread.
    Sleeps between checks. When the RNG fires, picks a random failure
    mode and executes it — the process dies and ray detects the failure.

    Args:
        seed: Random seed for reproducibility.
        crash_probability: Probability of crashing on each check interval.
        step_interval_seconds: Seconds between crash-or-not decisions.
        include_deadlock: Whether to include DEADLOCK in the failure mode pool.
    """
    rng = random.Random(seed)
    modes = _FAILURE_MODES_ALL if include_deadlock else _FAILURE_MODES_FAST
    logger.info(
        "FaultInjector: started (pid=%d, seed=%d, prob=%.3f, interval=%.1fs, modes=%s)",
        os.getpid(), seed, crash_probability, step_interval_seconds,
        [m.value for m in modes],
    )

    while True:
        time.sleep(step_interval_seconds)
        if rng.random() < crash_probability:
            mode = rng.choice(modes)
            _execute_failure(mode)
