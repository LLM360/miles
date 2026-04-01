from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Discriminator

from miles.utils.process_identity import ProcessIdentity
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class EventBase(FrozenStrictBaseModel):
    timestamp: datetime | None = None
    source: ProcessIdentity | None = None


class GenericEvent(EventBase):
    type: Literal["generic"] = "generic"
    message: str
    details: dict[str, Any]


class CellStateChangedEvent(EventBase):
    type: Literal["cell_state_changed"] = "cell_state_changed"
    cell_index: int
    old_state: str
    new_state: str


class QuorumChangedEvent(EventBase):
    type: Literal["quorum_changed"] = "quorum_changed"
    quorum_id: int
    alive_cell_indices: list[int]
    num_cells: int


class HeartbeatTimeoutEvent(EventBase):
    type: Literal["heartbeat_timeout"] = "heartbeat_timeout"
    cell_index: int
    last_active_timestamp: float
    staleness_seconds: float


class CellRefreshStartedEvent(EventBase):
    type: Literal["cell_refresh_started"] = "cell_refresh_started"
    pending_indices: list[int]
    alive_indices: list[int]
    will_alive_indices: list[int]


class CellRefreshCompletedEvent(EventBase):
    type: Literal["cell_refresh_completed"] = "cell_refresh_completed"
    alive_indices: list[int]


class CellRefreshFailedEvent(EventBase):
    type: Literal["cell_refresh_failed"] = "cell_refresh_failed"
    error_message: str
    pending_indices: list[int]


class CheckpointTransferStartedEvent(EventBase):
    type: Literal["checkpoint_transfer_started"] = "checkpoint_transfer_started"
    src_cell_index: int
    dst_cell_indices: list[int]


class CheckpointTransferCompletedEvent(EventBase):
    type: Literal["checkpoint_transfer_completed"] = "checkpoint_transfer_completed"
    src_cell_index: int
    dst_cell_indices: list[int]
    duration_seconds: float


class OptimizerStateInfo(FrozenStrictBaseModel):
    """Snapshot of one sub-optimizer's state with tensors replaced by hashes."""
    param_names: dict[int, str]
    state_dict: dict[str, Any]


class LocalWeightChecksumState(FrozenStrictBaseModel):
    param_hashes: dict[str, str]
    buffer_hashes: dict[str, str]
    optimizer_hashes: list[OptimizerStateInfo]


class LocalWeightChecksumEvent(EventBase):
    type: Literal["local_weight_checksum"] = "local_weight_checksum"
    step: int
    cell_index: int
    rank_within_cell: int
    state: LocalWeightChecksumState


Event = Annotated[
    GenericEvent
    | CellStateChangedEvent
    | QuorumChangedEvent
    | HeartbeatTimeoutEvent
    | CellRefreshStartedEvent
    | CellRefreshCompletedEvent
    | CellRefreshFailedEvent
    | CheckpointTransferStartedEvent
    | CheckpointTransferCompletedEvent
    | LocalWeightChecksumEvent,
    Discriminator("type"),
]


def _to_snake_case(name: str) -> str:
    import re
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


def _check_event_naming() -> None:
    import typing
    event_types = typing.get_args(typing.get_args(Event)[0])
    for cls in event_types:
        type_value = cls.model_fields["type"].default
        expected_snake = type_value + "_event"
        actual_snake = _to_snake_case(cls.__name__)
        assert actual_snake == expected_snake, (
            f"Event class {cls.__name__} (snake: {actual_snake}) does not match "
            f"type '{type_value}' (expected snake: {expected_snake})"
        )


_check_event_naming()
