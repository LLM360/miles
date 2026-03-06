from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from miles.utils.ft.controller.recovery_cooldown import RecoveryCooldown
from miles.utils.ft.models.fault import TriggerType


class TestRecoveryCooldown:
    def test_not_throttled_below_max_count(self) -> None:
        cooldown = RecoveryCooldown(window_minutes=30.0, max_count=3)
        cooldown.record(TriggerType.CRASH)
        cooldown.record(TriggerType.CRASH)
        assert not cooldown.is_throttled(TriggerType.CRASH)

    def test_throttled_at_max_count(self) -> None:
        cooldown = RecoveryCooldown(window_minutes=30.0, max_count=3)
        cooldown.record(TriggerType.CRASH)
        cooldown.record(TriggerType.CRASH)
        cooldown.record(TriggerType.CRASH)
        assert cooldown.is_throttled(TriggerType.CRASH)

    def test_different_triggers_tracked_separately(self) -> None:
        cooldown = RecoveryCooldown(window_minutes=30.0, max_count=2)
        cooldown.record(TriggerType.CRASH)
        cooldown.record(TriggerType.CRASH)
        assert cooldown.is_throttled(TriggerType.CRASH)
        assert not cooldown.is_throttled(TriggerType.HANG)

    def test_old_entries_outside_window_ignored(self) -> None:
        cooldown = RecoveryCooldown(window_minutes=10.0, max_count=2)

        old_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        with patch(
            "miles.utils.ft.controller.recovery_cooldown.datetime",
        ) as mock_dt:
            mock_dt.now.return_value = old_time
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            cooldown.record(TriggerType.CRASH)

        cooldown.record(TriggerType.CRASH)
        assert not cooldown.is_throttled(TriggerType.CRASH)

    def test_empty_history_not_throttled(self) -> None:
        cooldown = RecoveryCooldown(window_minutes=30.0, max_count=1)
        assert not cooldown.is_throttled(TriggerType.CRASH)
