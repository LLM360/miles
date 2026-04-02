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
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from miles.ray.train.group import RayTrainGroup

logger = logging.getLogger(__name__)

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
    current_step: int = 0
    random_seed: int = 42
    crash_probability: float = 0.1


@dataclass(frozen=True)
class _FaultEvent:
    step: int
    cell: int
    action: Literal["crash", "restart"]


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
    """Task 3: Stop + start (no missed steps) for bitwise healing verification.

    Timeline:
      train() #0, #1: all N cells normal
      after #1:       stop_cell(X) + start_cell(X)
      train() #2:     _refresh_cells() heals X from cell_0, all N cells run
    """

    def after_step(self, step: int) -> None:
        if step == 1:
            logger.info(
                "DeterministicScenario: stop+start cell %d after step %d",
                self._target_cell_index, step,
            )
            self.ctx.group.stop_cell(self._target_cell_index)
            self.ctx.group.start_cell(self._target_cell_index)

    def on_complete(self) -> None:
        logger.info("DeterministicScenario: completed successfully")


@register_scenario("random_failure")
class RandomFailureScenario(FTTestScenarioBase):
    """Task 4: Random unexpected crashes via background thread in train actors.

    Unlike tasks 1-3, this uses the mini FT controller for automatic recovery
    rather than coordinated stop/start. The scenario randomly selects a cell
    to crash at each step boundary. The actual crash is simulated by calling
    stop_cell() to emulate an unexpected failure detected by the health checker.

    NOTE: True unexpected crashes (os.kill, sys.exit in actor background thread)
    require ray concurrency group integration in the train actor. This scenario
    provides the orchestration layer; the actor-side crash injection is a
    separate concern that plugs into the actor's concurrency group.
    """

    def __init__(self, ctx: FTTestContext) -> None:
        super().__init__(ctx)
        self._crash_probability: float = ctx.crash_probability
        self._rng: random.Random = random.Random(ctx.random_seed)
        self._fault_log: list[_FaultEvent] = []
        logger.info(
            "RandomFailureScenario: seed=%d, crash_probability=%.3f",
            ctx.random_seed, self._crash_probability,
        )

    def after_step(self, step: int) -> None:
        if self._rng.random() < self._crash_probability:
            alive_cells = [
                i for i in range(self.ctx.num_cells)
                if self.ctx.group._cells[i].is_alive
            ]
            if len(alive_cells) <= 1:
                logger.info(
                    "RandomFailureScenario: skipping crash at step %d (only %d alive cells)",
                    step, len(alive_cells),
                )
                return

            target = self._rng.choice(alive_cells)
            logger.info(
                "RandomFailureScenario: crashing cell %d at step %d",
                target, step,
            )
            self.ctx.group.stop_cell(target)
            self._fault_log.append(_FaultEvent(step=step, cell=target, action="crash"))

            self.ctx.group.start_cell(target)
            self._fault_log.append(_FaultEvent(step=step, cell=target, action="restart"))

    def on_complete(self) -> None:
        crash_count = sum(1 for e in self._fault_log if e.action == "crash")
        logger.info(
            "RandomFailureScenario: completed. Total faults injected: %d",
            crash_count,
        )
        for entry in self._fault_log:
            logger.info("  Fault event: %s", entry)


def get_scenario(name: str, ctx: FTTestContext) -> FTTestScenarioBase:
    """Look up and instantiate a scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown FT test scenario: {name!r}. Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[name](ctx)
