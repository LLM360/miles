"""Rule: cross-replica weight checksum consistency."""

import logging
from collections import defaultdict
from typing import Any

from miles.utils.event_logger.models import Event, LocalWeightChecksumEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


class ChecksumMismatchIssue(FrozenStrictBaseModel):
    step: int
    category: str
    key: str
    cell_indices: list[int]
    values: list[str]


def check_weight_checksums(events: list[Event]) -> list[ChecksumMismatchIssue]:
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

    all_mismatches: list[ChecksumMismatchIssue] = []

    for step in sorted(entries_by_step.keys()):
        step_entries = entries_by_step[step]

        all_mismatches.extend(_compare_flat_dicts(
            step=step,
            category="param",
            entries=[(rank, e.param_hashes) for rank, e in step_entries],
        ))
        all_mismatches.extend(_compare_flat_dicts(
            step=step,
            category="buffer",
            entries=[(rank, e.buffer_hashes) for rank, e in step_entries],
        ))

        for opt_idx in range(len(step_entries[0][1].optimizer_hashes)):
            flat_dicts = []
            for rank, event in step_entries:
                assert opt_idx < len(event.optimizer_hashes), (
                    f"step {step} rank {rank}: expected optimizer_hashes[{opt_idx}] but only has {len(event.optimizer_hashes)}"
                )
                flat = _flatten_nested(event.optimizer_hashes[opt_idx].state_dict, prefix=f"opt{opt_idx}")
                flat_dicts.append((rank, flat))

            all_mismatches.extend(_compare_flat_dicts(
                step=step,
                category="optimizer",
                entries=flat_dicts,
            ))

    return all_mismatches


def _flatten_nested(obj: Any, *, prefix: str) -> dict[str, str]:
    """Flatten a nested dict/list into a flat dict with dot-separated keys. Only keeps str leaf values (hashes)."""
    result: dict[str, str] = {}

    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda x: str(x[0])):
            result.update(_flatten_nested(v, prefix=f"{prefix}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            result.update(_flatten_nested(v, prefix=f"{prefix}[{i}]"))
    elif isinstance(obj, str):
        result[prefix] = obj

    return result


def _compare_flat_dicts(
    step: int,
    category: str,
    entries: list[tuple[int, dict[str, str]]],
) -> list[ChecksumMismatchIssue]:
    """Compare flat string dicts across replicas."""
    mismatches: list[ChecksumMismatchIssue] = []

    all_keys: set[str] = set()
    for _, d in entries:
        all_keys.update(d.keys())

    for key in sorted(all_keys):
        value_by_rank: dict[str, list[int]] = defaultdict(list)
        for rank, d in entries:
            v = d.get(key, "<missing>")
            value_by_rank[v].append(rank)

        if len(value_by_rank) > 1:
            cell_indices: list[int] = []
            values: list[str] = []
            for v, ranks in sorted(value_by_rank.items(), key=lambda x: x[1][0]):
                for r in ranks:
                    cell_indices.append(r)
                    values.append(v)

            mismatches.append(ChecksumMismatchIssue(
                step=step,
                category=category,
                key=key,
                cell_indices=cell_indices,
                values=values,
            ))

    return mismatches
