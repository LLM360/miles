import logging
from dataclasses import dataclass, field
from typing import Literal, Type

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


class WitnessMissingSnapshotIssue(FrozenStrictBaseModel):
    rollout_id: int
    cell_index: int
    description: str


WitnessIssue = WitnessDataMismatchIssue | WitnessMissingSnapshotIssue


def check(events: list[Event]) -> list[WitnessIssue]:
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

    allocations_by_rollout = {e.rid: e.mapping for e in _filter_by_type(events, WitnessAllocateIdEvent)}
    expected_witness_ids = _compute_expected_witness_ids(allocations_by_rollout)

    return _find_mismatches(
        all_step_events=_filter_by_type(events, TrainGroupStepEndEvent),
        all_witness_events=_filter_by_type(events, WitnessSnapshotParamEvent),
        expected_witness_ids=expected_witness_ids,
    )


def _filter_by_type(arr: list, ty: Type) -> list:
    return [x for x in arr if isinstance(x, ty)]


def _compute_expected_witness_ids(allocations_by_rollout: dict[int, dict[int, int]]) -> dict[int, set[int]]:
    """Precompute cumulative expected witness IDs per rollout_id."""
    ans: dict[int, set[int]] = {}
    running: set[int] = set()
    for rid in sorted(allocations_by_rollout.keys()):
        running = running | set(allocations_by_rollout[rid].keys())
        ans[rid] = set(running)
    return ans


def _find_mismatches(
    *,
    all_step_events: list[TrainGroupStepEndEvent],
    all_witness_events: list[WitnessSnapshotParamEvent],
    expected_witness_ids: dict[int, set[int]],
) -> list[WitnessIssue]:
    issues: list[WitnessIssue] = []

    for step_event in all_step_events:
        rollout_id = step_event.rollout_id

        for cell_index, cell_outcome in step_event.cell_outcomes.items():
            if not all(r == TrainStepOutcome.NORMAL for r in cell_outcome):
                continue

            witness_events_of_cell = [
                e for e in all_witness_events
                if e.rollout_id == rollout_id and e.source.cell_index == cell_index
            ]

            if not witness_events_of_cell:
                issues.append(WitnessMissingSnapshotIssue(
                    rollout_id=rollout_id,
                    cell_index=cell_index,
                    description=f"Cell {cell_index} reported NORMAL for rollout {rollout_id} but no WitnessSnapshotParamEvent was found",
                ))
                continue

            for event in witness_events_of_cell:
                issue = _compare_snapshot(
                    event=event, expected=expected_witness_ids.get(rollout_id, set()),
                    rollout_id=rollout_id, cell_index=cell_index,
                )
                if issue is not None:
                    issues.append(issue)

    return issues


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
