from __future__ import annotations

import pytest

from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus
from tests.fast.utils.ft.conftest import StubDiagnostic, make_test_controller


class TestRegisterNodeAgentScrapeTarget:
    def test_register_node_agent_adds_scrape_target(self) -> None:
        harness = make_test_controller()
        ctrl = harness.controller

        stub = StubDiagnostic(passed=True)

        class _FakeAgent:
            async def run_diagnostic(
                self, diagnostic_type: str, timeout_seconds: int = 120, **kwargs: object,
            ):
                return await stub.run(node_id="node-0", timeout_seconds=timeout_seconds)

        ctrl.register_node_agent(
            node_id="node-0",
            agent=_FakeAgent(),
            exporter_address="http://node-0:9100",
        )

        assert "node-0" in ctrl._agents
        assert isinstance(ctrl._scrape_target_manager, MiniPrometheus)
        assert "node-0" in ctrl._scrape_target_manager._scrape_targets

    def test_register_node_agent_without_exporter_address_skips_scrape(self) -> None:
        harness = make_test_controller()
        ctrl = harness.controller

        class _FakeAgent:
            async def run_diagnostic(
                self, diagnostic_type: str, timeout_seconds: int = 120, **kwargs: object,
            ):
                pass

        ctrl.register_node_agent(node_id="node-1", agent=_FakeAgent())

        assert "node-1" in ctrl._agents
        assert isinstance(ctrl._scrape_target_manager, MiniPrometheus)
        assert "node-1" not in ctrl._scrape_target_manager._scrape_targets

    def test_register_node_agent_no_scrape_manager_is_safe(self) -> None:
        harness = make_test_controller()
        ctrl = harness.controller
        ctrl._scrape_target_manager = None

        class _FakeAgent:
            async def run_diagnostic(
                self, diagnostic_type: str, timeout_seconds: int = 120, **kwargs: object,
            ):
                pass

        ctrl.register_node_agent(
            node_id="node-2",
            agent=_FakeAgent(),
            exporter_address="http://node-2:9100",
        )

        assert "node-2" in ctrl._agents
