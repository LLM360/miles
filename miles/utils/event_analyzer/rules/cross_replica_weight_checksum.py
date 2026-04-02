from collections.abc import Iterable
from typing import Any

from miles.utils.event_logger.models import Event, LocalWeightChecksumEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ChecksumMismatchIssue(FrozenStrictBaseModel):
    key: str
    label_a: str
    label_b: str
    value_a: str
    value_b: str


def check(events: list[Event]) -> list[ChecksumMismatchIssue]:
    """Weight checksum across replicas should be exactly the same."""
    checksum_events = [e for e in events if isinstance(e, LocalWeightChecksumEvent)]
    if not checksum_events:
        return []

    all_mismatches: list[ChecksumMismatchIssue] = []

    events_by_rollout: dict[int, list[LocalWeightChecksumEvent]] = {}
    for event in checksum_events:
        events_by_rollout.setdefault(event.rollout_id, []).append(event)

    for rollout_id in sorted(events_by_rollout.keys()):
        all_mismatches += list(_check_one_step(events=events_by_rollout[rollout_id]))

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


def _compute_label(event: LocalWeightChecksumEvent) -> str:
    return f"rollout_{event.rollout_id}/{event.source.to_name()}"


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
        value_a = a.get(key, "<missing>")
        value_b = b.get(key, "<missing>")
        if value_a != value_b:
            yield ChecksumMismatchIssue(
                key=key,
                label_a=label_a,
                label_b=label_b,
                value_a=str(value_a),
                value_b=str(value_b),
            )


def _flatten_nested(obj: Any, *, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict/list into a flat dict with dot-separated keys. Keeps all primitive leaf values."""
    result: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda x: str(x[0])):
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            result.update(_flatten_nested(v, prefix=child_prefix))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            result.update(_flatten_nested(v, prefix=f"{prefix}[{i}]"))
    else:
        result[prefix] = obj

    return result
