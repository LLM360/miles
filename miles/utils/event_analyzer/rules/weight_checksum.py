"""Rule: cross-replica weight checksum consistency."""

import logging
from collections import defaultdict
from collections.abc import Callable

from miles.utils.event_logger.models import Event, LocalWeightChecksumEvent
from miles.utils.pydantic_utils import StrictBaseModel

logger = logging.getLogger(__name__)


class ChecksumMismatch(StrictBaseModel):
    step: int
    tensor_category: str
    tensor_name: str
    cell_indices: list[int]
    hashes: list[str]


def check_weight_checksums(events: list[Event]) -> list[ChecksumMismatch]:
    """Verify cross-replica weight checksum consistency from events.

    Returns:
        List of mismatches found. Empty list means all replicas match.
    """
    checksum_events = [e for e in events if isinstance(e, LocalWeightChecksumEvent)]
    if not checksum_events:
        return []

    entries_by_step: dict[int, list[tuple[int, LocalWeightChecksumEvent]]] = defaultdict(list)
    for event in checksum_events:
        entries_by_step[event.step].append((event.rank, event))

    all_mismatches: list[ChecksumMismatch] = []

    for step in sorted(entries_by_step.keys()):
        step_entries = entries_by_step[step]

        categories: list[tuple[str, Callable[[LocalWeightChecksumEvent], dict[str, str]]]] = [
            ("param", lambda e: e.param_hashes),
            ("buffer", lambda e: e.buffer_hashes),
            ("master_param", lambda e: e.master_param_hashes),
            ("optimizer_state", lambda e: e.optimizer_state_hashes),
        ]

        for category_name, accessor in categories:
            group = [(rank, accessor(entry)) for rank, entry in step_entries]
            mismatches = _find_mismatches_in_group(
                step=step,
                category=category_name,
                entries=group,
            )
            all_mismatches.extend(mismatches)

    return all_mismatches


def _find_mismatches_in_group(
    step: int,
    category: str,
    entries: list[tuple[int, dict[str, str]]],
) -> list[ChecksumMismatch]:
    """Compare hash dicts across replicas for a single (step, category) group."""
    mismatches: list[ChecksumMismatch] = []

    all_keys: set[str] = set()
    for _, hashes in entries:
        all_keys.update(hashes.keys())

    for key in sorted(all_keys):
        hash_by_rank: dict[str, list[int]] = defaultdict(list)
        for rank, hashes in entries:
            h = hashes.get(key, "<missing>")
            hash_by_rank[h].append(rank)

        if len(hash_by_rank) > 1:
            cell_indices: list[int] = []
            hash_values: list[str] = []
            for h, ranks in sorted(hash_by_rank.items(), key=lambda x: x[1][0]):
                for r in ranks:
                    cell_indices.append(r)
                    hash_values.append(h)

            mismatches.append(
                ChecksumMismatch(
                    step=step,
                    tensor_category=category,
                    tensor_name=key,
                    cell_indices=cell_indices,
                    hashes=hash_values,
                )
            )

    return mismatches
