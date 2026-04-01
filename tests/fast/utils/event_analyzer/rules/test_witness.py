"""Tests for event_analyzer rules/witness."""

from pydantic import TypeAdapter

from miles.utils.event_analyzer.rules.witness import WitnessMismatch, check
from miles.utils.event_logger.models import Event, WitnessEvent

_event_adapter = TypeAdapter(Event)


def _make_event(
    step: int,
    quorum_id: int,
    rank: int,
    nonzero_ids: list[int],
) -> WitnessEvent:
    return WitnessEvent(
        step=step,
        quorum_id=quorum_id,
        rank=rank,
        nonzero_ids=nonzero_ids,
    )


class TestWitnessCheck:
    def test_happy_path_all_ranks_agree(self) -> None:
        events = [
            _make_event(step=0, quorum_id=0, rank=0, nonzero_ids=[1, 2, 3]),
            _make_event(step=0, quorum_id=0, rank=1, nonzero_ids=[1, 2, 3]),
            _make_event(step=0, quorum_id=0, rank=2, nonzero_ids=[1, 2, 3]),
        ]
        assert check(events) == []

    def test_cross_rank_mismatch(self) -> None:
        events = [
            _make_event(step=0, quorum_id=0, rank=0, nonzero_ids=[1, 2, 3]),
            _make_event(step=0, quorum_id=0, rank=1, nonzero_ids=[1, 2, 4]),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].step == 0
        assert mismatches[0].quorum_id == 0
        assert "rank 0" in mismatches[0].description
        assert "rank 1" in mismatches[0].description

    def test_groups_by_quorum_id(self) -> None:
        events = [
            # Quorum 0: match
            _make_event(step=0, quorum_id=0, rank=0, nonzero_ids=[1, 2]),
            _make_event(step=0, quorum_id=0, rank=1, nonzero_ids=[1, 2]),
            # Quorum 1: mismatch
            _make_event(step=0, quorum_id=1, rank=0, nonzero_ids=[1, 2]),
            _make_event(step=0, quorum_id=1, rank=1, nonzero_ids=[1, 3]),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].quorum_id == 1

    def test_empty_events(self) -> None:
        assert check([]) == []

    def test_single_rank_no_comparison(self) -> None:
        events = [_make_event(step=0, quorum_id=0, rank=0, nonzero_ids=[1, 2])]
        assert check(events) == []


class TestWitnessEventSerialization:
    def test_roundtrip(self) -> None:
        event = _make_event(step=5, quorum_id=2, rank=1, nonzero_ids=[10, 20])
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessEvent)
        assert parsed.step == 5
        assert parsed.quorum_id == 2
        assert parsed.rank == 1
        assert parsed.nonzero_ids == [10, 20]
