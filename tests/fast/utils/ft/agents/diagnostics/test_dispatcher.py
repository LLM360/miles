from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.agents.diagnostics.dispatcher import NodeDiagnosticDispatcher
from miles.utils.ft.agents.types import DiagnosticResult, UnknownDiagnosticError


class _StubExecutor:
    """Minimal NodeExecutorProtocol implementation for testing."""

    def __init__(self, diagnostic_type: str, *, passed: bool = True) -> None:
        self.diagnostic_type = diagnostic_type
        self._passed = passed

    async def run(self, node_id: str, timeout_seconds: int = 120) -> DiagnosticResult:
        factory = DiagnosticResult.pass_result if self._passed else DiagnosticResult.fail_result
        return factory(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            details=f"stub {self.diagnostic_type}",
        )


class _SlowExecutor:
    """Executor that sleeps longer than the timeout to trigger TimeoutError."""

    diagnostic_type = "slow"

    async def run(self, node_id: str, timeout_seconds: int = 120) -> DiagnosticResult:
        await asyncio.sleep(timeout_seconds + 60)
        return DiagnosticResult.pass_result(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            details="should not reach here",
        )


class _CrashingExecutor:
    """Executor that always raises."""

    diagnostic_type = "crash"

    async def run(self, node_id: str, timeout_seconds: int = 120) -> DiagnosticResult:
        raise RuntimeError("boom")


class TestAvailableTypes:
    def test_returns_sorted_registered_types(self) -> None:
        runner = NodeDiagnosticDispatcher(
            node_id="node-1",
            diagnostics=[_StubExecutor("zeta"), _StubExecutor("alpha"), _StubExecutor("mid")],
        )
        assert runner.available_types == ["alpha", "mid", "zeta"]

    def test_empty_when_no_diagnostics(self) -> None:
        runner = NodeDiagnosticDispatcher(node_id="node-1")
        assert runner.available_types == []


class TestRunSelected:
    def test_runs_all_selected_types_in_order(self) -> None:
        runner = NodeDiagnosticDispatcher(
            node_id="node-1",
            diagnostics=[_StubExecutor("a", passed=True), _StubExecutor("b", passed=False)],
        )
        results = asyncio.run(runner.run_selected(["a", "b"], timeout_seconds=10))

        assert len(results) == 2
        assert results[0].diagnostic_type == "a"
        assert results[0].passed is True
        assert results[1].diagnostic_type == "b"
        assert results[1].passed is False

    def test_unknown_type_raises(self) -> None:
        runner = NodeDiagnosticDispatcher(
            node_id="node-1",
            diagnostics=[_StubExecutor("known")],
        )
        with pytest.raises(UnknownDiagnosticError):
            asyncio.run(runner.run_selected(["unknown"], timeout_seconds=10))

    def test_timeout_returns_fail_result(self) -> None:
        runner = NodeDiagnosticDispatcher(
            node_id="node-1",
            diagnostics=[_SlowExecutor()],
        )
        results = asyncio.run(runner.run_selected(["slow"], timeout_seconds=1))

        assert len(results) == 1
        assert results[0].passed is False
        assert "timed out" in results[0].details

    def test_exception_returns_fail_result(self) -> None:
        runner = NodeDiagnosticDispatcher(
            node_id="node-1",
            diagnostics=[_CrashingExecutor()],
        )
        results = asyncio.run(runner.run_selected(["crash"], timeout_seconds=10))

        assert len(results) == 1
        assert results[0].passed is False
        assert "exception" in results[0].details

    def test_empty_selection_returns_empty_list(self) -> None:
        runner = NodeDiagnosticDispatcher(
            node_id="node-1",
            diagnostics=[_StubExecutor("a")],
        )
        results = asyncio.run(runner.run_selected([], timeout_seconds=10))
        assert results == []
