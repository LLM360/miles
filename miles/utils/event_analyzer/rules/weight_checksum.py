"""Rule: cross-replica weight checksum consistency."""

from collections.abc import Iterable
from typing import Any

from miles.utils.event_logger.models import Event, LocalWeightChecksumEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel

_PRIMITIVE_TYPES = (str, int, float, bool)


class ChecksumMismatchIssue(FrozenStrictBaseModel):
    key: str
    value_a: str
    value_b: str


def check(events: list[Event]) -> list[ChecksumMismatchIssue]:
    """Verify cross-replica weight checksum consistency from events.

    Returns:
        List of mismatches found. Empty list means all replicas match.
    """
    checksum_events = [e for e in events if isinstance(e, LocalWeightChecksumEvent)]
    if not checksum_events:
        return []

    all_mismatches: list[ChecksumMismatchIssue] = []

    events_by_step: dict[int, list[LocalWeightChecksumEvent]] = {}
    for event in checksum_events:
        events_by_step.setdefault(event.step, []).append(event)

    for step in sorted(events_by_step.keys()):
        all_mismatches += list(_check_one_step(events=events_by_step[step]))

    return all_mismatches


def _check_one_step(events: list[LocalWeightChecksumEvent]) -> Iterable[ChecksumMismatchIssue]:
    event_a = events[0]
    for i in range(1, len(events)):
        event_b = events[i]
        yield from _compare_flat_dicts(
            a=_flatten_event(event_a),
            b=_flatten_event(event_b),
            label_a=_compute_label(event_a),
            label_b=_compute_label(event_b),
        )


def _compute_label(event: LocalWeightChecksumEvent):
    return f"step_{event.step}/cell_{event.cell_index}/rank_{event.rank_within_cell}"


def _flatten_event(event: LocalWeightChecksumEvent) -> dict[str, Any]:
    """Flatten all fields of an event into a flat dict with dot-separated keys."""
    return _flatten_nested(event.state.model_dump(), prefix="")


def _compare_flat_dicts(
    a: dict[str, Any],
    b: dict[str, Any],
    label_a: str,
    label_b: str,
) -> Iterable[ChecksumMismatchIssue]:
    """Compare two flat dicts and yield mismatches."""
    all_keys = sorted(set(a.keys()) | set(b.keys()))

    for key in all_keys:
        val_a = a.get(key, "<missing>")
        val_b = b.get(key, "<missing>")
        if val_a != val_b:
            yield ChecksumMismatchIssue(
                key=key,
                value_a=f"{label_a}: {val_a}",
                value_b=f"{label_b}: {val_b}",
            )


def _flatten_nested(obj: Any, *, prefix: str, _result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten a nested dict/list into a flat dict with dot-separated keys. Keeps all primitive leaf values."""
    if _result is None:
        _result = {}

    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda x: str(x[0])):
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten_nested(v, prefix=child_prefix, _result=_result)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten_nested(v, prefix=f"{prefix}[{i}]", _result=_result)
    elif isinstance(obj, _PRIMITIVE_TYPES):
        _result[prefix] = obj

    return _result
