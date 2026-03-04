from enum import Enum

from pydantic import BaseModel, ConfigDict


class FtBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricSample(FtBaseModel):
    name: str
    labels: dict[str, str]
    value: float


class CollectorOutput(FtBaseModel):
    metrics: list[MetricSample]


class ActionType(str, Enum):
    NONE = "none"
    MARK_BAD_AND_RESTART = "mark_bad_and_restart"
    ENTER_RECOVERY = "enter_recovery"
    NOTIFY_HUMAN = "notify_human"


class Decision(FtBaseModel):
    action: ActionType
    bad_node_ids: list[str] = []
    reason: str
    trigger: str = ""


class DiagnosticResult(FtBaseModel):
    diagnostic_type: str
    node_id: str
    passed: bool
    details: str


class RecoveryPhase(str, Enum):
    CHECK_ALERTS = "check_alerts"
    REATTEMPTING = "reattempting"
    MONITORING = "monitoring"
    DIAGNOSING = "diagnosing"
    EVICT_AND_RESTART = "evict_and_restart"
    NOTIFY = "notify"
    DONE = "done"


RECOVERY_PHASE_TO_INT: dict[RecoveryPhase, int] = {
    RecoveryPhase.CHECK_ALERTS: 1,
    RecoveryPhase.REATTEMPTING: 2,
    RecoveryPhase.MONITORING: 3,
    RecoveryPhase.DIAGNOSING: 4,
    RecoveryPhase.EVICT_AND_RESTART: 5,
    RecoveryPhase.NOTIFY: 6,
    RecoveryPhase.DONE: 7,
}
