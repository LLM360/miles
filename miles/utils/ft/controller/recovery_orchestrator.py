from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from miles.utils.ft.models import RecoveryPhase

logger = logging.getLogger(__name__)


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
