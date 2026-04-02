from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from miles.utils.event_logger.models import CellStateChangedEvent, Event, GenericEvent, QuorumChangedEvent
from miles.utils.process_identity import MainProcessIdentity

_event_adapter = TypeAdapter(Event)

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_FIXED_SOURCE = MainProcessIdentity()


class TestEventModelsDiscriminatedUnion:
    def test_roundtrip_via_discriminator(self) -> None:
        event = CellStateChangedEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            cell_index=0,
            old_state="pending",
            new_state="alive",
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, CellStateChangedEvent)
        assert parsed.cell_index == 0

    def test_discriminator_distinguishes_types(self) -> None:
        e1 = CellStateChangedEvent(
            timestamp=_FIXED_TS, source=_FIXED_SOURCE, cell_index=0, old_state="a", new_state="b"
        )
        e2 = QuorumChangedEvent(
            timestamp=_FIXED_TS, source=_FIXED_SOURCE, quorum_id=1, alive_cell_indices=[0], num_cells=1
        )
        p1 = _event_adapter.validate_json(e1.model_dump_json())
        p2 = _event_adapter.validate_json(e2.model_dump_json())
        assert type(p1) is not type(p2)


class TestEventModelsStrictRejectExtraFields:
    def test_extra_field_rejected(self) -> None:
        data = {
            "type": "cell_state_changed",
            "timestamp": "2026-01-01T00:00:00Z",
            "source": {"component": "main"},
            "cell_index": 0,
            "old_state": "a",
            "new_state": "b",
            "bogus_field": 123,
        }
        with pytest.raises(ValidationError, match="bogus_field"):
            CellStateChangedEvent.model_validate(data)


class TestGenericEvent:
    def test_roundtrip_via_discriminator(self) -> None:
        event = GenericEvent(timestamp=_FIXED_TS, source=_FIXED_SOURCE, message="test", details={"k": 1})
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, GenericEvent)
        assert parsed.details["k"] == 1
