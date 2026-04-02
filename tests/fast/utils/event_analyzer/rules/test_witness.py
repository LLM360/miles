"""Tests for event_analyzer rules/witness."""

from datetime import datetime, timezone

from pydantic import TypeAdapter

from miles.utils.event_analyzer.rules.witness import WitnessDataMismatchIssue, check
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
_MAIN_SOURCE = MainProcessIdentity()


def _make_source(cell_index: int = 0, rank_within_cell: int = 0) -> TrainProcessIdentity:
    return TrainProcessIdentity(component="actor", cell_index=cell_index, rank_within_cell=rank_within_cell)


def _make_snapshot(
    rollout_id: int,
    nonzero_witness_ids: list[int],
    instance_id: str = "pp0.head",
    cell_index: int = 0,
    rank_within_cell: int = 0,
    stale_threshold: int = 0,
) -> WitnessSnapshotParamEvent:
    return WitnessSnapshotParamEvent(
        timestamp=_FIXED_TS,
        source=_make_source(cell_index=cell_index, rank_within_cell=rank_within_cell),
        rollout_id=rollout_id,
        instance_id=instance_id,
        nonzero_witness_ids=nonzero_witness_ids,
        stale_threshold=stale_threshold,
    )


def _make_rollout_completed(rollout_id: int, sample_indices: list[int]) -> RolloutGenerateCompletedEvent:
    return RolloutGenerateCompletedEvent(
        timestamp=_FIXED_TS,
        source=_MAIN_SOURCE,
        rollout_id=rollout_id,
        sample_indices=sample_indices,
    )


def _make_allocate(
    rollout_id: int,
    witness_id_to_sample_index: dict[int, int],
    attempt: int = 0,
) -> WitnessAllocateIdEvent:
    return WitnessAllocateIdEvent(
        timestamp=_FIXED_TS,
        source=_MAIN_SOURCE,
        rollout_id=rollout_id,
        attempt=attempt,
        witness_id_to_sample_index=witness_id_to_sample_index,
    )


def _make_step_end(rollout_id: int, cell_outcomes: dict[int, str]) -> TrainGroupStepEndEvent:
    return TrainGroupStepEndEvent(
        timestamp=_FIXED_TS,
        source=_MAIN_SOURCE,
        rollout_id=rollout_id,
        cell_outcomes=cell_outcomes,
    )


class TestWitnessCheck:
    def test_empty_events(self) -> None:
        assert check([]) == []

    def test_normal_step_with_correct_cumulative_witness_ids_returns_no_issues(self) -> None:
        events: list[Event] = [
            _make_rollout_completed(rollout_id=0, sample_indices=[0, 1]),
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11]),
            _make_step_end(rollout_id=0, cell_outcomes={0: "NORMAL"}),
        ]
        assert check(events) == []

    def test_normal_step_with_missing_witness_id_returns_issue(self) -> None:
        events: list[Event] = [
            _make_rollout_completed(rollout_id=0, sample_indices=[0, 1]),
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: "NORMAL"}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert issues[0].rollout_id == 0
        assert 11 in issues[0].expected_witness_ids
        assert 11 not in issues[0].actual_witness_ids

    def test_normal_step_with_extra_witness_id_returns_issue(self) -> None:
        events: list[Event] = [
            _make_rollout_completed(rollout_id=0, sample_indices=[0, 1]),
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11, 99]),
            _make_step_end(rollout_id=0, cell_outcomes={0: "NORMAL"}),
        ]
        issues = check(events)
        assert len(issues) == 1
        assert 99 in issues[0].actual_witness_ids
        assert 99 not in issues[0].expected_witness_ids

    def test_discarded_step_is_ignored(self) -> None:
        events: list[Event] = [
            _make_rollout_completed(rollout_id=0, sample_indices=[0, 1]),
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: "DISCARDED"}),
        ]
        assert check(events) == []

    def test_stale_threshold_ids_are_ignored(self) -> None:
        """IDs in [0, stale_threshold) are ignored in both expected and actual."""
        events: list[Event] = [
            _make_rollout_completed(rollout_id=0, sample_indices=[0, 1, 2]),
            _make_allocate(rollout_id=0, witness_id_to_sample_index={2: 0, 5: 1, 8: 2}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[5, 8], stale_threshold=3),
            _make_step_end(rollout_id=0, cell_outcomes={0: "NORMAL"}),
        ]
        assert check(events) == []

    def test_multiple_cells_independent_checking(self) -> None:
        """Each cell is checked independently."""
        events: list[Event] = [
            _make_rollout_completed(rollout_id=0, sample_indices=[0, 1]),
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10, 11]),
            _make_step_end(rollout_id=0, cell_outcomes={0: "NORMAL", 1: "NORMAL"}),
        ]
        assert check(events) == []

    def test_retry_uses_latest_attempt_allocation(self) -> None:
        """When retries happen, only the latest attempt's allocation is used."""
        events: list[Event] = [
            _make_rollout_completed(rollout_id=0, sample_indices=[0, 1]),
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0, 11: 1}, attempt=0),
            _make_allocate(rollout_id=0, witness_id_to_sample_index={20: 0, 21: 1}, attempt=1),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[20, 21]),
            _make_step_end(rollout_id=0, cell_outcomes={0: "NORMAL"}),
        ]
        assert check(events) == []

    def test_cumulative_across_rollouts(self) -> None:
        """Expected witness IDs are cumulative from rollout 0 to current."""
        events: list[Event] = [
            _make_rollout_completed(rollout_id=0, sample_indices=[0]),
            _make_allocate(rollout_id=0, witness_id_to_sample_index={10: 0}),
            _make_snapshot(rollout_id=0, nonzero_witness_ids=[10]),
            _make_step_end(rollout_id=0, cell_outcomes={0: "NORMAL"}),

            _make_rollout_completed(rollout_id=1, sample_indices=[1]),
            _make_allocate(rollout_id=1, witness_id_to_sample_index={11: 1}),
            _make_snapshot(rollout_id=1, nonzero_witness_ids=[10, 11]),
            _make_step_end(rollout_id=1, cell_outcomes={0: "NORMAL"}),
        ]
        assert check(events) == []


class TestWitnessEventSerialization:
    def test_roundtrip(self) -> None:
        event = _make_snapshot(
            rollout_id=5,
            nonzero_witness_ids=[10, 20],
            instance_id="pp0.tail",
            cell_index=1,
            stale_threshold=3,
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessSnapshotParamEvent)
        assert parsed.rollout_id == 5
        assert parsed.instance_id == "pp0.tail"
        assert parsed.nonzero_witness_ids == [10, 20]
        assert parsed.stale_threshold == 3
