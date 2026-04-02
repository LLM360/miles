import logging
from dataclasses import dataclass, field
from typing import Literal

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
    * Witness' ring buffer will remove old data, thus we need to ignore the appearance/disappearance of
      all values in `WitnessSnapshotParamEvent.stale_ids`
    """
    parsed = _parse_events(events)
    expected_witness_ids = _build_expected_witness_ids(parsed.allocations_by_rollout)
    return _find_mismatches(
        step_end_events=parsed.step_end_events,
        witness_events=parsed.witness_events,
        expected_witness_ids=expected_witness_ids,
    )


@dataclass
class _ParsedEvents:
    allocations_by_rollout: dict[int, dict[int, int]] = field(default_factory=dict)
    step_end_events: list[TrainGroupStepEndEvent] = field(default_factory=list)
    witness_events: list[WitnessSnapshotParamEvent] = field(default_factory=list)


def _parse_events(events: list[Event]) -> _ParsedEvents:
    """Events are assumed to arrive in chronological order."""
    parsed = _ParsedEvents()

    for event in events:
        match event:
            case WitnessAllocateIdEvent(rollout_id=rid, witness_id_to_sample_index=mapping):
                parsed.allocations_by_rollout[rid] = mapping

            case TrainGroupStepEndEvent():
                parsed.step_end_events.append(event)

            case WitnessSnapshotParamEvent(source=source):
                assert isinstance(source, TrainProcessIdentity)
                parsed.witness_events.append(event)

    return parsed


def _build_expected_witness_ids(allocations_by_rollout: dict[int, dict[int, int]]) -> dict[int, set[int]]:
    """Precompute cumulative expected witness IDs per rollout_id."""
    ans: dict[int, set[int]] = {}
    running: set[int] = set()
    for rid in sorted(allocations_by_rollout.keys()):
        running = running | set(allocations_by_rollout[rid].keys())
        ans[rid] = set(running)
    return ans


def _find_mismatches(
    *,
    step_end_events: list[TrainGroupStepEndEvent],
    witness_events: list[WitnessSnapshotParamEvent],
    expected_witness_ids: dict[int, set[int]],
) -> list[WitnessDataMismatchIssue]:
    issues: list[WitnessDataMismatchIssue] = []

    for step_end in step_end_events:
        rollout_id = step_end.rollout_id

        matching_snapshots = [
            event for event in witness_events
            if event.rollout_id == rollout_id
        ]

        for cell_index, outcome in step_end.cell_outcomes.items():
            if _is_non_normal_outcome(outcome):
                continue

            cell_snapshots = [
                event for event in matching_snapshots
                if event.source.cell_index == cell_index
            ]

            for event in cell_snapshots:
                issue = _compare_snapshot(
                    event=event, expected=expected_witness_ids.get(rollout_id, set()),
                    rollout_id=rollout_id, cell_index=cell_index,
                )
                if issue is not None:
                    issues.append(issue)

    return issues


def _is_non_normal_outcome(outcome: Literal["error"] | list[TrainStepOutcome]) -> bool:
    return outcome == "error" or any(r != TrainStepOutcome.NORMAL for r in outcome)


def _compare_snapshot(
    *,
    event: WitnessSnapshotParamEvent,
    expected: set[int],
    rollout_id: int,
    cell_index: int,
) -> WitnessDataMismatchIssue | None:
    stale_set = set(event.stale_ids)
    filtered_expected = expected - stale_set
    filtered_actual = set(event.nonzero_witness_ids) - stale_set

    if filtered_expected == filtered_actual:
        return None

    return WitnessDataMismatchIssue(
        rollout_id=rollout_id,
        cell_index=cell_index,
        description=(
            f"Witness data mismatch for instance {event.instance_id}: "
            f"missing={sorted(filtered_expected - filtered_actual)}, "
            f"extra={sorted(filtered_actual - filtered_expected)}"
        ),
        expected_witness_ids=sorted(filtered_expected),
        actual_witness_ids=sorted(filtered_actual),
    )
