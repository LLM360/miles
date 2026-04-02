"""Fault tolerance test scenarios executed inside the training job.

Each scenario function is invoked at step boundaries by the training loop
when ``--ci-ft-test-scenario`` is set. Scenarios perform coordinated fault
injection (stop/start cells) and are deterministic.

The scenario is called as a *step callback* — it receives the current
``FTTestContext`` and decides what to do before/after each ``train()`` call.
"""

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import requests

from miles.utils.test_utils.fault_injector import FailureMode

if TYPE_CHECKING:
    from miles.ray.train.group import RayTrainGroup

logger = logging.getLogger(__name__)

_FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]

SCENARIOS: dict[str, "type[FTTestScenarioBase]"] = {}


def register_scenario(name: str):
    """Decorator to register a scenario class by name."""
    def _decorator(cls: type) -> type:
        SCENARIOS[name] = cls
        return cls
    return _decorator


@dataclass
class FTTestContext:
    """Runtime context shared across scenario callbacks."""
    group: "RayTrainGroup"
    num_cells: int
    control_server_port: int = 0
    current_step: int = 0
    random_seed: int = 42
    crash_probability: float = 0.1


class FTTestScenarioBase:
    """Base class for FT test scenarios.

    Subclasses override ``before_step`` and ``after_step`` to inject faults.
    """

    def __init__(self, ctx: FTTestContext) -> None:
        self.ctx = ctx
        self._target_cell_index: int = ctx.num_cells - 1

    def before_step(self, step: int) -> None:
        """Called before each train() invocation."""

    def after_step(self, step: int) -> None:
        """Called after each train() invocation."""

    def on_complete(self) -> None:
        """Called after all steps are done."""


@register_scenario("with_failure")
class WithFailureScenario(FTTestScenarioBase):
    """Task 2: Coordinated stop/start sequence.

    Timeline (phase_b, resuming from ckpt):
      train() #0: normal (N cells)
      after #0:   stop_cell(target_cell_index)
      train() #1: N-1 cells (retry on DISCARDED_SHOULD_RETRY)
      after #1:   start_cell(target_cell_index)
      train() #2: _refresh_cells() heals the cell, N cells run
      train() #3: N cells stable
    """

    def after_step(self, step: int) -> None:
        if step == 0:
            logger.info(
                "WithFailureScenario: stopping cell %d after step %d",
                self._target_cell_index, step,
            )
            self.ctx.group.stop_cell(self._target_cell_index)

        elif step == 1:
            logger.info(
                "WithFailureScenario: starting cell %d after step %d",
                self._target_cell_index, step,
            )
            self.ctx.group.start_cell(self._target_cell_index)

    def on_complete(self) -> None:
        logger.info("WithFailureScenario: completed successfully")


@register_scenario("deterministic")
class DeterministicScenario(FTTestScenarioBase):
    """Task 3: Deterministic healing + degraded retry + checkpoint resume verification.

    Timeline (phase_b, resuming from ckpt):
      train() #0, #1: all N cells normal (2 good steps)
      after #1:       stop_cell(X) + start_cell(X) (marks pending, healing at next step)
      train() #2:     healing happens at start, then normal execution.
                      after #2: stop_cell(X) again (create degraded state)
      train() #3:     N-1 cells, allreduce fails -> should_commit=false ->
                      DISCARDED_SHOULD_RETRY -> retry succeeds with N-1 cells.
                      after #3: start_cell(X) to restore
      train() #4:     healing + normal execution
    """

    def after_step(self, step: int) -> None:
        if step == 1:
            logger.info(
                "DeterministicScenario: stop+start cell %d after step %d (trigger healing)",
                self._target_cell_index, step,
            )
            self.ctx.group.stop_cell(self._target_cell_index)
            self.ctx.group.start_cell(self._target_cell_index)

        elif step == 2:
            logger.info(
                "DeterministicScenario: stopping cell %d after step %d (create degraded state)",
                self._target_cell_index, step,
            )
            self.ctx.group.stop_cell(self._target_cell_index)

        elif step == 3:
            logger.info(
                "DeterministicScenario: starting cell %d after step %d (restore for healing)",
                self._target_cell_index, step,
            )
            self.ctx.group.start_cell(self._target_cell_index)

    def on_complete(self) -> None:
        logger.info("DeterministicScenario: completed successfully")


@register_scenario("random_failure")
class RandomFailureScenario(FTTestScenarioBase):
    """Task 4: Random unexpected crashes via control server fault injection API.

    At each step boundary, rolls the dice per cell. If triggered, sends
    POST /api/v1/cells/{name}/inject-fault to the control server, which
    calls actor.inject_fault.remote(mode) in a dedicated concurrency group
    thread. The actor process dies immediately (SIGKILL, os._exit, segfault).

    The health checker detects dead actors via heartbeat timeout.
    The mini FT controller auto-recovers (suspend → resume).
    """

    def __init__(self, ctx: FTTestContext) -> None:
        super().__init__(ctx)
        assert ctx.control_server_port > 0, (
            "RandomFailureScenario requires --control-server-port > 0"
        )
        self._rng = random.Random(ctx.random_seed)
        self._base_url = f"http://localhost:{ctx.control_server_port}"
        logger.info(
            "RandomFailureScenario: seed=%d, crash_probability=%.3f, base_url=%s",
            ctx.random_seed, ctx.crash_probability, self._base_url,
        )

    def after_step(self, step: int) -> None:
        if self._rng.random() >= self.ctx.crash_probability:
            return

        alive_cells = [
            i for i in range(self.ctx.num_cells)
            if self.ctx.group._cells[i].is_alive
        ]
        if len(alive_cells) <= 1:
            logger.info(
                "RandomFailureScenario: skipping crash at step %d (only %d alive)",
                step, len(alive_cells),
            )
            return

        target_cell = self._rng.choice(alive_cells)
        cell = self.ctx.group._cells[target_cell]
        sub_index = self._rng.randrange(len(cell._actors))
        mode = self._rng.choice(_FAILURE_MODES)
        cell_name = f"actor-{target_cell}"

        logger.info(
            "RandomFailureScenario: injecting %s into %s actor %d at step %d",
            mode, cell_name, sub_index, step,
        )
        try:
            resp = requests.post(
                f"{self._base_url}/api/v1/cells/{cell_name}/inject-fault",
                json={"mode": mode.value, "sub_index": sub_index},
                timeout=5,
            )
            resp.raise_for_status()
        except Exception:
            logger.warning(
                "RandomFailureScenario: failed to inject fault into %s",
                cell_name, exc_info=True,
            )

    def on_complete(self) -> None:
        logger.info("RandomFailureScenario: completed")


def get_scenario(name: str, ctx: FTTestContext) -> FTTestScenarioBase:
    """Look up and instantiate a scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown FT test scenario: {name!r}. Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[name](ctx)
