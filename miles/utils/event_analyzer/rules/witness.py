import logging

from miles.utils.event_logger.models import (
    Event,
    RolloutGenerateCompletedEvent,
    TrainGroupStepEndEvent,
    WitnessAllocateIdEvent,
    WitnessSnapshotParamEvent,
)
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
      all values in the range of 0..`WitnessSnapshotParamEvent.stale_threshold`
    """

    allocations_by_rollout: dict[int, dict[int, int]] = {}
    max_attempt_by_rollout: dict[int, int] = {}
    step_end_events: list[TrainGroupStepEndEvent] = []
    snapshot_events: list[WitnessSnapshotParamEvent] = []

    for event in events:
        if isinstance(event, WitnessAllocateIdEvent):
            prev_attempt = max_attempt_by_rollout.get(event.rollout_id, -1)
            if event.attempt > prev_attempt:
                max_attempt_by_rollout[event.rollout_id] = event.attempt
                allocations_by_rollout[event.rollout_id] = event.witness_id_to_sample_index

        elif isinstance(event, TrainGroupStepEndEvent):
            step_end_events.append(event)

        elif isinstance(event, WitnessSnapshotParamEvent):
            snapshot_events.append(event)

    # Precompute cumulative expected witness IDs per rollout_id to avoid O(N²) rebuild
    cumulative_expected: dict[int, set[int]] = {}
    running: set[int] = set()
    for rid in sorted(allocations_by_rollout.keys()):
        running = running | set(allocations_by_rollout[rid].keys())
        cumulative_expected[rid] = set(running)

    issues: list[WitnessDataMismatchIssue] = []

    for step_end in step_end_events:
        rollout_id = step_end.rollout_id
        expected_witness_ids = cumulative_expected.get(rollout_id, set())

        matching_snapshots = [
            snap for snap in snapshot_events
            if snap.rollout_id == rollout_id
        ]

        for cell_index, outcome_str in step_end.cell_outcomes.items():
            if outcome_str != "NORMAL":
                continue

            for snap in matching_snapshots:
                stale_range = set(range(snap.stale_threshold))

                filtered_expected = expected_witness_ids - stale_range
                filtered_actual = set(snap.nonzero_witness_ids) - stale_range

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
