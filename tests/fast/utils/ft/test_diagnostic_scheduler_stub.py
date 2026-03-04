from __future__ import annotations

import asyncio

from miles.utils.ft.controller.diagnostic_scheduler_stub import StubDiagnosticScheduler
from miles.utils.ft.models import ActionType


class TestStubDiagnosticScheduler:
    def test_returns_notify_human(self) -> None:
        scheduler = StubDiagnosticScheduler()
        decision = asyncio.run(
            scheduler.run_diagnostic_pipeline(trigger_reason="crash"),
        )
        assert decision.action == ActionType.NOTIFY_HUMAN
        assert "stub" in decision.reason.lower()

    def test_accepts_suspect_node_ids(self) -> None:
        scheduler = StubDiagnosticScheduler()
        decision = asyncio.run(
            scheduler.run_diagnostic_pipeline(
                trigger_reason="hang",
                suspect_node_ids=["node-0", "node-1"],
            ),
        )
        assert decision.action == ActionType.NOTIFY_HUMAN
