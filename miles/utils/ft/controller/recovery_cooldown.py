from __future__ import annotations

from datetime import datetime, timedelta, timezone

from miles.utils.ft.models.fault import TriggerType


class RecoveryCooldown:
    """Tracks recovery frequency per trigger and throttles when a limit is exceeded.

    Within a sliding window of ``window_minutes``, if the same trigger fires
    ``max_count`` or more times, ``is_throttled`` returns True.
    """

    def __init__(self, window_minutes: float, max_count: int) -> None:
        self._window_minutes = window_minutes
        self._max_count = max_count
        self._history: list[tuple[TriggerType, datetime]] = []

    def record(self, trigger: TriggerType) -> None:
        self._history.append((trigger, datetime.now(timezone.utc)))

    def is_throttled(self, trigger: TriggerType) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self._window_minutes)
        recent_count = sum(
            1 for t, ts in self._history
            if t == trigger and ts >= cutoff
        )
        return recent_count >= self._max_count

