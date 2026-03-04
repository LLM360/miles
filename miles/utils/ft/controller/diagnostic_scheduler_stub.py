from __future__ import annotations

import logging

from miles.utils.ft.models import ActionType, Decision

logger = logging.getLogger(__name__)


class StubDiagnosticScheduler:
    async def run_diagnostic_pipeline(
        self,
        trigger_reason: str,
        suspect_node_ids: list[str] | None = None,
    ) -> Decision:
        logger.info(
            "stub_diagnostic_pipeline trigger_reason=%s suspect_node_ids=%s",
            trigger_reason, suspect_node_ids,
        )
        return Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="all diagnostics passed (stub scheduler — no real diagnostics available)",
        )
