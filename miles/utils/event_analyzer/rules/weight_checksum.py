"""Rule: cross-replica weight checksum consistency."""

from collections import defaultdict
from typing import Any, NamedTuple, Iterable

from miles.utils.event_logger.models import Event, LocalWeightChecksumEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ChecksumMismatchIssue(FrozenStrictBaseModel):
    step: int
    category: str
    key: str
    cell_indices: list[int]
    values: list[str]


def check(events: list[Event]) -> list[ChecksumMismatchIssue]:
    """Verify cross-replica weight checksum consistency from events.

    Returns:
        List of mismatches found. Empty list means all replicas match.
    """
    checksum_events = [e for e in events if isinstance(e, LocalWeightChecksumEvent)]
    if not checksum_events:
        return []

    events_by_step: dict[int, list[LocalWeightChecksumEvent]] = defaultdict(list)
    for event in checksum_events:
        events_by_step[event.step].append(event)

    all_mismatches: list[ChecksumMismatchIssue] = []

    for step in sorted(events_by_step.keys()):
        all_mismatches += list(_check_one_step(step, events=events_by_step[step]))

    return all_mismatches


class _RankHashes(NamedTuple):
    rank: int
    hashes: dict[str, str]


def _check_one_step(step: int, events: list[LocalWeightChecksumEvent]) -> Iterable[ChecksumMismatchIssue]:
    for i in range(1, len(events)):
        yield from _compare_flat_dicts(a=_flatten_nested(events[0]), b=_flatten_nested(events[i]))


# TODO temporarily commenterd out
# def _check_one_step(step: int, events: list[LocalWeightChecksumEvent]):
#     yield from _compare_flat_dicts(
#         step=step,
#         category="param",
#         entries=[_RankHashes(rank=e.rank, hashes=e.param_hashes) for e in events],
#     )
#
#     yield from _compare_flat_dicts(
#         step=step,
#         category="buffer",
#         entries=[_RankHashes(rank=e.rank, hashes=e.buffer_hashes) for e in events],
#     )
#
#     for opt_idx in range(len(events[0].optimizer_hashes)):
#         flat_dicts: list[_RankHashes] = []
#         for e in events:
#             assert opt_idx < len(e.optimizer_hashes), (
#                 f"step {step} rank {e.rank}: expected optimizer_hashes[{opt_idx}] but only has {len(e.optimizer_hashes)}"
#             )
#             flat = _flatten_nested(e.optimizer_hashes[opt_idx].state_dict, prefix=f"opt{opt_idx}")
#             flat_dicts.append(_RankHashes(rank=e.rank, hashes=flat))
#
#         yield from _compare_flat_dicts(
#             step=step,
#             category="optimizer",
#             entries=flat_dicts,
#         )


def _flatten_nested(obj: Any, *, prefix: str, _result: dict[str, str] | None = None) -> dict[str, str]:
    """Flatten a nested dict/list into a flat dict with dot-separated keys. Only keeps str leaf values (hashes)."""
    if _result is None:
        _result = {}

    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda x: str(x[0])):
            _flatten_nested(v, prefix=f"{prefix}.{k}", _result=_result)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten_nested(v, prefix=f"{prefix}[{i}]", _result=_result)
    elif isinstance(obj, str):
        _result[prefix] = obj

    return _result


def _compare_flat_dicts(a: dict[str, Any], b: dict[str, Any]):
    TODO


# def _compare_flat_dicts(
#     step: int,
#     category: str,
#     entries: list[_RankHashes],
# ) -> list[ChecksumMismatchIssue]:
#     """Compare flat string dicts across replicas."""
#     mismatches: list[ChecksumMismatchIssue] = []
#
#     all_keys: set[str] = set()
#     for entry in entries:
#         all_keys.update(entry.hashes.keys())
#
#     for key in sorted(all_keys):
#         value_by_rank: dict[str, list[int]] = defaultdict(list)
#         for entry in entries:
#             v = entry.hashes.get(key, "<missing>")
#             value_by_rank[v].append(entry.rank)
#
#         if len(value_by_rank) > 1:
#             cell_indices: list[int] = []
#             values: list[str] = []
#             for v, ranks in sorted(value_by_rank.items(), key=lambda x: x[1][0]):
#                 for r in ranks:
#                     cell_indices.append(r)
#                     values.append(v)
#
#             mismatches.append(ChecksumMismatchIssue(
#                 step=step,
#                 category=category,
#                 key=key,
#                 cell_indices=cell_indices,
#                 values=values,
#             ))
#
#     return mismatches
