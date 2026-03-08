from enum import Enum

from miles.utils.ft.utils.base_model import FtBaseModel


class ControllerMode(str, Enum):
    MONITORING = "monitoring"
    RECOVERY = "recovery"


class ControllerStatus(FtBaseModel):
    mode: ControllerMode
    recovery_phase: str | None
    phase_history: list[str] | None
    tick_count: int
    active_run_id: str | None
    bad_nodes: list[str]
    recovery_in_progress: bool
    bad_nodes_confirmed: bool
    latest_iteration: int | None
