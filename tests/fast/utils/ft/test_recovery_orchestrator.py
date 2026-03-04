from __future__ import annotations

from datetime import datetime, timezone

from miles.utils.ft.controller.recovery_orchestrator import RecoveryContext
from miles.utils.ft.models import RECOVERY_PHASE_TO_INT, RecoveryPhase


class TestRecoveryPhase:
    def test_enum_values(self) -> None:
        assert RecoveryPhase.CHECK_ALERTS == "check_alerts"
        assert RecoveryPhase.REATTEMPTING == "reattempting"
        assert RecoveryPhase.MONITORING == "monitoring"
        assert RecoveryPhase.DIAGNOSING == "diagnosing"
        assert RecoveryPhase.EVICT_AND_RESTART == "evict_and_restart"
        assert RecoveryPhase.NOTIFY == "notify"
        assert RecoveryPhase.DONE == "done"
        assert len(RecoveryPhase) == 7

    def test_phase_to_int_mapping(self) -> None:
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.CHECK_ALERTS] == 1
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.REATTEMPTING] == 2
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.MONITORING] == 3
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.DIAGNOSING] == 4
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.EVICT_AND_RESTART] == 5
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.NOTIFY] == 6
        assert RECOVERY_PHASE_TO_INT[RecoveryPhase.DONE] == 7
        assert len(RECOVERY_PHASE_TO_INT) == len(RecoveryPhase)


class TestRecoveryContext:
    def test_defaults(self) -> None:
        ctx = RecoveryContext(trigger="crash")
        assert ctx.trigger == "crash"
        assert ctx.phase == RecoveryPhase.CHECK_ALERTS
        assert ctx.reattempt_start_time is None
        assert ctx.reattempt_base_iteration is None
        assert ctx.global_timeout_seconds == 1800
        assert ctx.monitoring_success_iterations == 10
        assert ctx.monitoring_timeout_seconds == 600
        assert ctx.recovery_start_time.tzinfo == timezone.utc

    def test_recovery_start_time_is_utc(self) -> None:
        before = datetime.now(timezone.utc)
        ctx = RecoveryContext(trigger="hang")
        after = datetime.now(timezone.utc)
        assert before <= ctx.recovery_start_time <= after
