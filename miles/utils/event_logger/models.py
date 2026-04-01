from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Discriminator

from miles.utils.process_identity import ProcessIdentity
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class EventBase(FrozenStrictBaseModel):
    timestamp: datetime | None = None
    source: ProcessIdentity | None = None


class CellStateChanged(EventBase):
    type: Literal["cell_state_changed"] = "cell_state_changed"
    cell_index: int
    old_state: str
    new_state: str


class QuorumChanged(EventBase):
    type: Literal["quorum_changed"] = "quorum_changed"
    quorum_id: int
    alive_cell_indices: list[int]
    num_cells: int


class HeartbeatTimeout(EventBase):
    type: Literal["heartbeat_timeout"] = "heartbeat_timeout"
    cell_index: int
    last_active_timestamp: float
    staleness_seconds: float


class CellRefreshStarted(EventBase):
    type: Literal["cell_refresh_started"] = "cell_refresh_started"
    pending_indices: list[int]
    alive_indices: list[int]
    will_alive_indices: list[int]


class CellRefreshCompleted(EventBase):
    type: Literal["cell_refresh_completed"] = "cell_refresh_completed"
    alive_indices: list[int]


class CellRefreshFailed(EventBase):
    type: Literal["cell_refresh_failed"] = "cell_refresh_failed"
    error_message: str
    pending_indices: list[int]


class CheckpointTransferStarted(EventBase):
    type: Literal["checkpoint_transfer_started"] = "checkpoint_transfer_started"
    src_cell_index: int
    dst_cell_indices: list[int]


class CheckpointTransferCompleted(EventBase):
    type: Literal["checkpoint_transfer_completed"] = "checkpoint_transfer_completed"
    src_cell_index: int
    dst_cell_indices: list[int]
    duration_seconds: float


class GenericEvent(EventBase):
    type: Literal["generic"] = "generic"
    message: str
    details: dict[str, Any]


class LocalWeightChecksumEvent(EventBase):
    type: Literal["local_weight_checksum"] = "local_weight_checksum"
    step: int
    rank: int
    param_hashes: dict[str, str]
    buffer_hashes: dict[str, str]
    master_param_hashes: dict[str, str]
    optimizer_state_hashes: dict[str, str]


Event = Annotated[
    CellStateChanged
    | QuorumChanged
    | HeartbeatTimeout
    | CellRefreshStarted
    | CellRefreshCompleted
    | CellRefreshFailed
    | CheckpointTransferStarted
    | CheckpointTransferCompleted
    | GenericEvent
    | LocalWeightChecksumEvent,
    Discriminator("type"),
]
