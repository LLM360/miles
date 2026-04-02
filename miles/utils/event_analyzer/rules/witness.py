import logging
from dataclasses import dataclass, field
from typing import Literal

from miles.backends.megatron_utils.model import TrainStepOutcome
from miles.utils.event_logger.models import (
    Event,
    RolloutGenerateCompletedEvent,
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


class WitnessAllocationMismatchIssue(FrozenStrictBaseModel):
    rollout_id: int
    description: str
    rollout_sample_indices: list[int]
    allocated_sample_indices: list[int]


def check(events: list[Event]) -> list[WitnessDataMismatchIssue | WitnessAllocationMismatchIssue]:
    """
    Related events:
    * RolloutGenerateCompletedEvent: when a rollout is executed and some data are obtained
    * WitnessAllocateIdEvent: when allocating `witness_id` to `sample_id`
    * WitnessSnapshotParamEvent: near the end of each train() step in MegatronTrainRayActor
        * If a witness_id appears in the weight, it means the corresponding data is consumed at least once.
    * TrainGroupStepEndEvent: after each train() step in RayTrainGroup

    Check:
    1. For each rollout_id, verify that the samples from RolloutGenerateCompletedEvent (source of truth)
       match the samples allocated witness IDs in WitnessAllocateIdEvent.
    2. For each (rollout_id, cell_index),
       if TrainGroupStepEndEvent claims the cell ends with TrainStepOutcome.NORMAL,
       then its WitnessSnapshotParamEvent should observe *EXACTLY* the training data in rollout_id=0~curr.

    Remarks:
    * RolloutGenerateCompletedEvent is the source of truth for which samples exist per rollout.
    * WitnessAllocateIdEvent provides the witness_id <-> sample_index mapping.
    * Witness' ring buffer will remove old data, thus we need to ignore the appearance/disappearance of
      all values in `WitnessSnapshotParamEvent.stale_ids`
    """
    parsed = _parse_events(events)

    issues: list[WitnessDataMismatchIssue | WitnessAllocationMismatchIssue] = []

    # Step 1: Cross-validate rollout samples vs witness allocation
    issues.extend(_check_allocation_coverage(
        sample_indices_by_rollout=parsed.sample_indices_by_rollout,
        allocations_by_rollout=parsed.allocations_by_rollout,
    ))

    # Step 2: Check witness snapshots against cumulative expected witness IDs
    cumulative_expected = _build_cumulative_expected(parsed.allocations_by_rollout)
    issues.extend(_find_mismatches(
        step_end_events=parsed.step_end_events,
        snapshot_events=parsed.snapshot_events,
        cumulative_expected=cumulative_expected,
    ))

    return issues


@dataclass
class _ParsedEvents:
    sample_indices_by_rollout: dict[int, list[int]] = field(default_factory=dict)
    allocations_by_rollout: dict[int, dict[int, int]] = field(default_factory=dict)
    step_end_events: list[TrainGroupStepEndEvent] = field(default_factory=list)
    snapshot_events: list[WitnessSnapshotParamEvent] = field(default_factory=list)


def _parse_events(events: list[Event]) -> _ParsedEvents:
    """Events are assumed to arrive in chronological order."""
    parsed = _ParsedEvents()

    for event in events:
        match event:
            case RolloutGenerateCompletedEvent(rollout_id=rid, sample_indices=indices):
                parsed.sample_indices_by_rollout[rid] = indices

            case WitnessAllocateIdEvent(rollout_id=rid, witness_id_to_sample_index=mapping):
                parsed.allocations_by_rollout[rid] = mapping

            case TrainGroupStepEndEvent():
                parsed.step_end_events.append(event)

            case WitnessSnapshotParamEvent(source=source):
                assert isinstance(source, TrainProcessIdentity)
                parsed.snapshot_events.append(event)

    return parsed


def _check_allocation_coverage(
    *,
    sample_indices_by_rollout: dict[int, list[int]],
    allocations_by_rollout: dict[int, dict[int, int]],
) -> list[WitnessAllocationMismatchIssue]:
    """Verify that witness allocation covers exactly the samples from each rollout."""
    issues: list[WitnessAllocationMismatchIssue] = []

    for rid, rollout_samples in sample_indices_by_rollout.items():
        if rid not in allocations_by_rollout:
            issues.append(WitnessAllocationMismatchIssue(
                rollout_id=rid,
                description=f"Rollout {rid} produced samples but no WitnessAllocateIdEvent was emitted",
                rollout_sample_indices=sorted(rollout_samples),
                allocated_sample_indices=[],
            ))
            continue

        allocated_samples = sorted(allocations_by_rollout[rid].values())
        if sorted(rollout_samples) != allocated_samples:
            issues.append(WitnessAllocationMismatchIssue(
                rollout_id=rid,
                description=(
                    f"Rollout {rid} sample set mismatch: "
                    f"rollout has {len(rollout_samples)} samples, allocation has {len(allocated_samples)}"
                ),
                rollout_sample_indices=sorted(rollout_samples),
                allocated_sample_indices=allocated_samples,
            ))

    return issues


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
            if _is_non_normal_outcome(outcome):
                continue

            cell_snapshots = [
                snap for snap in matching_snapshots
                if snap.source.cell_index == cell_index
            ]

            for snap in cell_snapshots:
                issue = _compare_snapshot(
                    snap=snap, expected=expected_witness_ids,
                    rollout_id=rollout_id, cell_index=cell_index,
                )
                if issue is not None:
                    issues.append(issue)

    return issues


def _is_non_normal_outcome(outcome: Literal["error"] | list[TrainStepOutcome]) -> bool:
    return outcome == "error" or any(r != TrainStepOutcome.NORMAL for r in outcome)


def _compare_snapshot(
    *,
    snap: WitnessSnapshotParamEvent,
    expected: set[int],
    rollout_id: int,
    cell_index: int,
) -> WitnessDataMismatchIssue | None:
    stale_set = set(snap.stale_ids)
    filtered_expected = expected - stale_set
    filtered_actual = set(snap.nonzero_witness_ids) - stale_set

    if filtered_expected == filtered_actual:
        return None

    return WitnessDataMismatchIssue(
        rollout_id=rollout_id,
        cell_index=cell_index,
        description=(
            f"Witness data mismatch for instance {snap.instance_id}: "
            f"missing={sorted(filtered_expected - filtered_actual)}, "
            f"extra={sorted(filtered_actual - filtered_expected)}"
        ),
        expected_witness_ids=sorted(filtered_expected),
        actual_witness_ids=sorted(filtered_actual),
    )
