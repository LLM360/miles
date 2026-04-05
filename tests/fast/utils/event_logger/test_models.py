from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from miles.backends.megatron_utils.types import TrainStepOutcome
from miles.utils.event_logger.models import (
    Event,
    RolloutGenerateCompletedEvent,
    TrainGroupStepEndEvent,
    WitnessAllocateIdEvent,
    WitnessSnapshotParamEvent,
)
from miles.utils.process_identity import MainProcessIdentity, TrainProcessIdentity

_event_adapter = TypeAdapter(Event)

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_FIXED_SOURCE = MainProcessIdentity()
_TRAIN_SOURCE = TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0)


class TestEventModelsDiscriminatedUnion:
    def test_roundtrip_via_discriminator(self) -> None:
        event = RolloutGenerateCompletedEvent(
            timestamp=_FIXED_TS, source=_FIXED_SOURCE, rollout_id=0, sample_indices=[0, 1]
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, RolloutGenerateCompletedEvent)
        assert parsed.sample_indices == [0, 1]

    def test_discriminator_distinguishes_types(self) -> None:
        e1 = RolloutGenerateCompletedEvent(timestamp=_FIXED_TS, source=_FIXED_SOURCE, rollout_id=0, sample_indices=[0])
        e2 = WitnessAllocateIdEvent(
            timestamp=_FIXED_TS, source=_FIXED_SOURCE, rollout_id=0, attempt=0, witness_id_to_sample_index={0: 0}
        )
        p1 = _event_adapter.validate_json(e1.model_dump_json())
        p2 = _event_adapter.validate_json(e2.model_dump_json())
        assert type(p1) is not type(p2)


class TestEventModelsStrictRejectExtraFields:
    def test_extra_field_rejected(self) -> None:
        data = {
            "type": "rollout_generate_completed",
            "timestamp": "2026-01-01T00:00:00Z",
            "source": {"component": "main"},
            "rollout_id": 0,
            "sample_indices": [0],
            "bogus_field": 123,
        }
        with pytest.raises(ValidationError, match="bogus_field"):
            RolloutGenerateCompletedEvent.model_validate(data)


class TestRolloutGenerateCompletedEvent:
    def test_json_roundtrip(self) -> None:
        event = RolloutGenerateCompletedEvent(
            timestamp=_FIXED_TS, source=_FIXED_SOURCE, rollout_id=1, sample_indices=[0, 1, 2]
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, RolloutGenerateCompletedEvent)
        assert parsed.rollout_id == 1
        assert parsed.sample_indices == [0, 1, 2]


class TestWitnessAllocateIdEvent:
    def test_json_roundtrip(self) -> None:
        event = WitnessAllocateIdEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=2,
            attempt=0,
            witness_id_to_sample_index={10: 0, 11: 1},
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessAllocateIdEvent)
        assert parsed.rollout_id == 2
        assert parsed.attempt == 0
        assert parsed.witness_id_to_sample_index == {10: 0, 11: 1}


class TestTrainGroupStepEndEvent:
    def test_json_roundtrip(self) -> None:
        event = TrainGroupStepEndEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=3,
            cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: "error"},
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, TrainGroupStepEndEvent)
        assert parsed.rollout_id == 3
        assert parsed.cell_outcomes[0] == [TrainStepOutcome.NORMAL]
        assert parsed.cell_outcomes[1] == "error"


class TestWitnessSnapshotParamEventWithStaleIds:
    def test_json_roundtrip(self) -> None:
        event = WitnessSnapshotParamEvent(
            timestamp=_FIXED_TS,
            source=_TRAIN_SOURCE,
            rollout_id=5,
            instance_id="actor_cell0_rank0",
            nonzero_witness_ids=[10, 11, 12],
            stale_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessSnapshotParamEvent)
        assert parsed.stale_ids == [0, 1, 2, 3, 4, 5, 6, 7]
        assert parsed.nonzero_witness_ids == [10, 11, 12]


class TestDiscriminatedUnionParsesAllEvents:
    def test_all_event_types_parse(self) -> None:
        events = [
            RolloutGenerateCompletedEvent(timestamp=_FIXED_TS, source=_FIXED_SOURCE, rollout_id=0, sample_indices=[0]),
            WitnessAllocateIdEvent(
                timestamp=_FIXED_TS, source=_FIXED_SOURCE, rollout_id=0, attempt=0, witness_id_to_sample_index={0: 0}
            ),
            TrainGroupStepEndEvent(
                timestamp=_FIXED_TS,
                source=_FIXED_SOURCE,
                rollout_id=0,
                cell_outcomes={0: [TrainStepOutcome.NORMAL]},
            ),
        ]
        for event in events:
            parsed = _event_adapter.validate_json(event.model_dump_json())
            assert type(parsed) is type(event)


class TestCheckEventNaming:
    def test_naming_convention_holds(self) -> None:
        from miles.utils.event_logger.models import _check_event_naming

        _check_event_naming()
