import json
from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from miles.utils.event_logger.models import (
    CellStateChanged,
    CheckpointTransferCompleted,
    Event,
    GenericEvent,
    HeartbeatTimeout,
    QuorumChanged,
)

_event_adapter = TypeAdapter(Event)

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestEventModelsDiscriminatedUnion:
    def test_cell_state_changed_roundtrip(self) -> None:
        event = CellStateChanged(
            timestamp=_FIXED_TS,
            cell_index=0,
            old_state="pending",
            new_state="alive",
        )
        raw = event.model_dump_json()
        parsed = _event_adapter.validate_json(raw)
        assert isinstance(parsed, CellStateChanged)
        assert parsed.type == "cell_state_changed"
        assert parsed.cell_index == 0

    def test_quorum_changed_roundtrip(self) -> None:
        event = QuorumChanged(
            timestamp=_FIXED_TS,
            quorum_id=3,
            alive_cell_indices=[0, 2],
            num_cells=4,
        )
        raw = event.model_dump_json()
        parsed = _event_adapter.validate_json(raw)
        assert isinstance(parsed, QuorumChanged)
        assert parsed.quorum_id == 3

    def test_heartbeat_timeout_roundtrip(self) -> None:
        event = HeartbeatTimeout(
            timestamp=_FIXED_TS,
            cell_index=1,
            last_active_timestamp=1000.0,
            staleness_seconds=60.5,
        )
        raw = event.model_dump_json()
        parsed = _event_adapter.validate_json(raw)
        assert isinstance(parsed, HeartbeatTimeout)

    def test_checkpoint_transfer_completed_roundtrip(self) -> None:
        event = CheckpointTransferCompleted(
            timestamp=_FIXED_TS,
            src_cell_index=0,
            dst_cell_indices=[1, 2],
            duration_seconds=3.14,
        )
        raw = event.model_dump_json()
        parsed = _event_adapter.validate_json(raw)
        assert isinstance(parsed, CheckpointTransferCompleted)
        assert parsed.duration_seconds == pytest.approx(3.14)

    def test_discriminator_distinguishes_types(self) -> None:
        e1 = CellStateChanged(
            timestamp=_FIXED_TS,
            cell_index=0,
            old_state="a",
            new_state="b",
        )
        e2 = QuorumChanged(
            timestamp=_FIXED_TS,
            quorum_id=1,
            alive_cell_indices=[0],
            num_cells=1,
        )
        p1 = _event_adapter.validate_json(e1.model_dump_json())
        p2 = _event_adapter.validate_json(e2.model_dump_json())
        assert type(p1) is not type(p2)


class TestEventModelsStrictRejectExtraFields:
    def test_extra_field_rejected(self) -> None:
        data = {
            "type": "cell_state_changed",
            "timestamp": "2026-01-01T00:00:00Z",
            "cell_index": 0,
            "old_state": "a",
            "new_state": "b",
            "bogus_field": 123,
        }
        with pytest.raises(ValidationError, match="bogus_field"):
            CellStateChanged.model_validate(data)


class TestEventBaseTimestampOptional:
    def test_timestamp_defaults_to_none(self) -> None:
        event = CellStateChanged(cell_index=0, old_state="a", new_state="b")
        assert event.timestamp is None

    def test_timestamp_accepted_when_provided(self) -> None:
        event = CellStateChanged(timestamp=_FIXED_TS, cell_index=0, old_state="a", new_state="b")
        assert event.timestamp == _FIXED_TS


class TestGenericEvent:
    def test_generic_event_fields(self) -> None:
        event = GenericEvent(
            timestamp=_FIXED_TS,
            message="something happened",
            details={"key": "value", "count": 42},
        )
        parsed = json.loads(event.model_dump_json())
        assert parsed["type"] == "generic"
        assert parsed["message"] == "something happened"
        assert parsed["details"]["count"] == 42

    def test_generic_event_roundtrip_via_discriminator(self) -> None:
        event = GenericEvent(
            timestamp=_FIXED_TS,
            message="test",
            details={},
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, GenericEvent)
