import logging
from dataclasses import dataclass, field

from miles.backends.megatron_utils.model import TrainStepOutcome
from miles.utils.event_logger.models import (
    Event,
    TrainGroupStepEndEvent,
    WitnessAllocateIdEvent,
    WitnessSnapshotParamEvent,
)
from miles.utils.process_identity import TrainProcessIdentity
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


class WitnessDataMismatchIssue(FrozenStrictBaseModel):
    rollout_id: int
    cell_index: int
    description: str
    expected_witness_ids: list[int]
    actual_witness_ids: list[int]


def check(events: list[Event]) -> list[WitnessDataMismatchIssue]:
    """
    Related events:
    * RolloutGenerateCompletedEvent: when a rollout is executed and some data are obtained
    * WitnessAllocateIdEvent: when allocating `witness_id` to `sample_id`
    * WitnessSnapshotParamEvent: near the end of each train() step in MegatronTrainRayActor
        * If a witness_id appears in the weight, it means the corresponding data is consumed at least once.
    * TrainGroupStepEndEvent: after each train() step in RayTrainGroup

    Check:
    1. For each (rollout_id, cell_index),
       if TrainGroupStepEndEvent claims the cell ends with TrainStepOutcome.NORMAL,
       then its WitnessSnapshotParamEvent should observe *EXACTLY* the training data in rollout_id=0~curr.

    Remarks:
    * To correlate witness_id vs sample_id utilize WitnessAllocateIdEvent.
    * To get *all* samples used in a step, must use RolloutGenerateCompletedEvent as source of truth.
    * Witness' ring buffer will remove old data, thus we need to ignore the appearance/disappearance of
      all values in `WitnessSnapshotParamEvent.stale_ids`
    """
    parsed = _parse_events(events)
    cumulative_expected = _build_cumulative_expected(parsed.allocations_by_rollout)
    return _find_mismatches(
        step_end_events=parsed.step_end_events,
        snapshot_events=parsed.snapshot_events,
        cumulative_expected=cumulative_expected,
    )


@dataclass
class _ParsedEvents:
    allocations_by_rollout: dict[int, dict[int, int]] = field(default_factory=dict)
    step_end_events: list[TrainGroupStepEndEvent] = field(default_factory=list)
    snapshot_events: list[WitnessSnapshotParamEvent] = field(default_factory=list)


def _parse_events(events: list[Event]) -> _ParsedEvents:
    parsed = _ParsedEvents()
    max_attempt_by_rollout: dict[int, int] = {}

    for event in events:
        if isinstance(event, WitnessAllocateIdEvent):
            prev_attempt = max_attempt_by_rollout.get(event.rollout_id, -1)
            if event.attempt > prev_attempt:
                max_attempt_by_rollout[event.rollout_id] = event.attempt
                parsed.allocations_by_rollout[event.rollout_id] = event.witness_id_to_sample_index

        elif isinstance(event, TrainGroupStepEndEvent):
            parsed.step_end_events.append(event)

        elif isinstance(event, WitnessSnapshotParamEvent):
            assert isinstance(event.source, TrainProcessIdentity)
            parsed.snapshot_events.append(event)

    return parsed


def _build_cumulative_expected(allocations_by_rollout: dict[int, dict[int, int]]) -> dict[int, set[int]]:
    """Precompute cumulative expected witness IDs per rollout_id."""
    cumulative: dict[int, set[int]] = {}
    running: set[int] = set()
    for rid in sorted(allocations_by_rollout.keys()):
        running = running | set(allocations_by_rollout[rid].keys())
        cumulative[rid] = set(running)
    return cumulative


def _find_mismatches(
    *,
    step_end_events: list[TrainGroupStepEndEvent],
    snapshot_events: list[WitnessSnapshotParamEvent],
    cumulative_expected: dict[int, set[int]],
) -> list[WitnessDataMismatchIssue]:
    issues: list[WitnessDataMismatchIssue] = []

    for step_end in step_end_events:
        rollout_id = step_end.rollout_id
        expected_witness_ids = cumulative_expected.get(rollout_id, set())

        matching_snapshots = [
            snap for snap in snapshot_events
            if snap.rollout_id == rollout_id
        ]

        for cell_index, outcome in step_end.cell_outcomes.items():
            match outcome:
                case "error":
                    continue
                case steps if any(r != TrainStepOutcome.NORMAL for r in steps):
                    continue

            cell_snapshots = [
                snap for snap in matching_snapshots
                if snap.source.cell_index == cell_index
            ]

            for snap in cell_snapshots:
                stale_set = set(snap.stale_ids)
                filtered_expected = expected_witness_ids - stale_set
                filtered_actual = set(snap.nonzero_witness_ids) - stale_set

                if filtered_expected != filtered_actual:
                    issues.append(WitnessDataMismatchIssue(
                        rollout_id=rollout_id,
                        cell_index=cell_index,
                        description=(
                            f"Witness data mismatch for instance {snap.instance_id}: "
                            f"missing={sorted(filtered_expected - filtered_actual)}, "
                            f"extra={sorted(filtered_actual - filtered_expected)}"
                        ),
                        expected_witness_ids=sorted(filtered_expected),
                        actual_witness_ids=sorted(filtered_actual),
                    ))

    return issues
