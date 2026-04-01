import time
from dataclasses import dataclass


@dataclass(frozen=True)
class HeartbeatStatus:
    last_active_timestamp: float
    bump_count: int


class SimpleHeartbeat:
    def __init__(self) -> None:
        self._last_active_timestamp: float = 0.0
        self._bump_count: int = 0

    def bump(self) -> None:
        self._last_active_timestamp = time.time()
        self._bump_count += 1

    def status(self) -> HeartbeatStatus:
        return HeartbeatStatus(
            last_active_timestamp=self._last_active_timestamp,
            bump_count=self._bump_count,
        )
