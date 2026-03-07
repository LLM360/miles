from enum import Enum

from pydantic import ConfigDict

from miles.utils.ft.models.base import FtBaseModel


class RecoveryPhase(str, Enum):
    # Entry point: inspect collected alerts for known hardware/network faults
    CHECK_ALERTS = "check_alerts"
    # Restart training without evicting nodes (ephemeral or unknown fault)
    REATTEMPTING = "reattempting"
    # Watch the reattempted run for iteration progress to confirm recovery
    MONITORING = "monitoring"
    # Run diagnostic pipeline (GPU checks, NCCL tests, etc.) to locate bad nodes
    DIAGNOSING = "diagnosing"
    # Evict confirmed bad nodes from the cluster and resubmit training
    EVICT_AND_RESTART = "evict_and_restart"
    # Escalate to humans — automated recovery could not resolve the issue
    NOTIFY = "notify"
    # Terminal state: recovery workflow complete
    DONE = "done"


class ControllerMode(str, Enum):
    MONITORING = "monitoring"
    RECOVERY = "recovery"


class RecoverySnapshot(FtBaseModel):
    model_config = ConfigDict(frozen=True)

    in_progress: bool
    phase: RecoveryPhase | None
    phase_history: list[RecoveryPhase] | None
    diagnosing_nodes: list[str]
    bad_nodes_confirmed: bool


class ControllerStatus(FtBaseModel):
    mode: ControllerMode
    recovery_phase: RecoveryPhase | None
    phase_history: list[RecoveryPhase] | None
    tick_count: int
    active_run_id: str | None
    bad_nodes: list[str]
    recovery_in_progress: bool
    bad_nodes_confirmed: bool
    latest_iteration: int | None


RECOVERY_PHASE_TO_INT: dict[RecoveryPhase, int] = {
    RecoveryPhase.CHECK_ALERTS: 1,
    RecoveryPhase.REATTEMPTING: 2,
    RecoveryPhase.MONITORING: 3,
    RecoveryPhase.DIAGNOSING: 4,
    RecoveryPhase.EVICT_AND_RESTART: 5,
    RecoveryPhase.NOTIFY: 6,
    RecoveryPhase.DONE: 7,
}

BAD_NODES_CONFIRMED_PHASES: frozenset[RecoveryPhase] = frozenset({
    RecoveryPhase.EVICT_AND_RESTART,
    RecoveryPhase.NOTIFY,
    RecoveryPhase.DONE,
})
