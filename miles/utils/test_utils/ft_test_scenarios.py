"""Fault tolerance test scenarios executed inside the training job.

Each scenario function is invoked at step boundaries by the training loop
when ``--ci-ft-test-scenario`` is set. Scenarios perform coordinated fault
injection (stop/start cells) and are deterministic.

The scenario is called as a *step callback* — it receives the current
``FTTestContext`` and decides what to do before/after each ``train()`` call.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    """Task 4: Random unexpected crashes in train actor background threads.

    Unlike tasks 1-3 (coordinated stop/start from the orchestrator), this
    scenario injects *genuine* unexpected crashes inside the train actors:

    1. On ``before_step`` of step 0, fires ``start_fault_injector.remote()``
       on every actor in every cell. Each actor runs a background loop (in a
       dedicated ray concurrency group thread) that randomly crashes the
       process (SIGKILL, os._exit, segfault, or GIL deadlock).

    2. The health checker detects dead actors via heartbeat timeout.

    3. The mini FT controller auto-recovers (suspend → resume).

    The scenario itself does nothing after step 0 — all fault injection
    happens autonomously inside actors.
    """

    def before_step(self, step: int) -> None:
        if step == 0:
            self._arm_all_actors()

    def _arm_all_actors(self) -> None:
        ctx = self.ctx
        logger.info(
            "RandomFailureScenario: arming fault injectors on all actors "
            "(seed=%d, crash_probability=%.3f)",
            ctx.random_seed, ctx.crash_probability,
        )

        actor_index = 0
        for cell_index in range(ctx.num_cells):
            cell = ctx.group._cells[cell_index]
            if not cell.is_alive:
                continue
            for actor in cell._actors:
                actor.start_fault_injector.remote(
                    seed=ctx.random_seed + actor_index,
                    crash_probability=ctx.crash_probability,
                )
                actor_index += 1

        logger.info(
            "RandomFailureScenario: armed %d actors with fault injectors",
            actor_index,
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
