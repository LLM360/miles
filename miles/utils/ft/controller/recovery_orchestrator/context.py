from __future__ import annotations

import math
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from miles.utils.ft.models import RecoveryPhase

_PENDING_TIMEOUT_SECONDS: int = 300
_MAX_RETRIES: int = 3

_StepHandler = Callable[[], Coroutine[Any, Any, None]]


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


@dataclass
class RecoveryContext:
    trigger: str
    phase: RecoveryPhase = RecoveryPhase.CHECK_ALERTS
    reattempt_start_time: datetime | None = None
    reattempt_base_iteration: int | None = None
    recovery_start_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    global_timeout_seconds: int = 1800
    monitoring_success_iterations: int = 10
    monitoring_timeout_seconds: int = 600
    phase_before_notify: RecoveryPhase | None = None
