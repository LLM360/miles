"""Rule: cross-rank witness gradient consistency."""

from collections import defaultdict

from miles.utils.event_logger.models import Event, WitnessEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class WitnessMismatch(FrozenStrictBaseModel):
    step: int
    quorum_id: int
    description: str


def check(events: list[Event]) -> list[WitnessMismatch]:
    """Verify that all ranks within the same (quorum_id, step) see identical nonzero witness IDs.

    Returns:
        List of mismatches found. Empty list means all ranks agree.
    """
    witness_events = [e for e in events if isinstance(e, WitnessEvent)]
    if not witness_events:
        return []

    grouped: dict[tuple[int, int], list[WitnessEvent]] = defaultdict(list)
    for event in witness_events:
        grouped[(event.quorum_id, event.step)].append(event)

    mismatches: list[WitnessMismatch] = []

    for (quorum_id, step), group in sorted(grouped.items()):
        reference = set(group[0].nonzero_ids)
        reference_rank = group[0].rank

        for event in group[1:]:
            current = set(event.nonzero_ids)
            if current != reference:
                only_in_ref = reference - current
                only_in_cur = current - reference
                mismatches.append(
                    WitnessMismatch(
                        step=step,
                        quorum_id=quorum_id,
                        description=(
                            f"rank {reference_rank} vs rank {event.rank}: "
                            f"only_in_{reference_rank}={sorted(only_in_ref)}, "
                            f"only_in_{event.rank}={sorted(only_in_cur)}"
                        ),
                    )
                )

    return mismatches
