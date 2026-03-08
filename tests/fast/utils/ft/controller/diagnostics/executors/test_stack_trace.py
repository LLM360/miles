"""Tests for StackTraceClusterExecutor — directly tests execute() without going through the orchestrator."""

from __future__ import annotations


import pytest
from tests.fast.utils.ft.utils import (
    SAMPLE_PYSPY_JSON_DIFFERENT_STUCK,
    SAMPLE_PYSPY_JSON_STUCK,
    make_fake_agents,
    make_rank_pids_provider,
    make_trace_result,
    mock_stack_trace_diagnostic,
)

from miles.utils.ft.controller.diagnostics.executors import StackTraceClusterExecutor


class TestStackTraceClusterExecutorBasic:
    @pytest.mark.anyio
    async def test_all_traces_same_returns_no_bad_nodes(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": True},
            }
        )
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100},
                "node-1": {1: 200},
            }
        )

        with mock_stack_trace_diagnostic(
            [
                make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
                make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
            ]
        ) as mock_diag_cls:
            executor = StackTraceClusterExecutor(rank_pids_provider=pids_provider)
            bad_nodes = await executor.execute(agents=agents, timeout_seconds=120)

            assert mock_diag_cls.call_count == 2
            assert bad_nodes == []

    @pytest.mark.anyio
    async def test_outlier_detected_returns_as_bad_node(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": False},
                "node-1": {"gpu": False},
                "node-2": {"gpu": False},
            }
        )
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100},
                "node-1": {1: 200},
                "node-2": {2: 300},
            }
        )

        with mock_stack_trace_diagnostic(
            [
                make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
                make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
                make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_JSON_DIFFERENT_STUCK),
            ]
        ):
            executor = StackTraceClusterExecutor(rank_pids_provider=pids_provider)
            bad_nodes = await executor.execute(agents=agents, timeout_seconds=120)

            assert bad_nodes == ["node-2"]


class TestStackTraceClusterExecutorFailures:
    @pytest.mark.anyio
    async def test_collection_failure_makes_node_bad(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": False},
            }
        )
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100},
                "node-1": {1: 200},
            }
        )

        with mock_stack_trace_diagnostic(
            [
                make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
                make_trace_result("node-1", passed=False, details="failed to collect"),
            ]
        ):
            executor = StackTraceClusterExecutor(rank_pids_provider=pids_provider)
            bad_nodes = await executor.execute(agents=agents, timeout_seconds=120)

            assert "node-1" in bad_nodes

    @pytest.mark.anyio
    async def test_exception_during_collection_makes_node_bad(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": False},
                "node-2": {"gpu": True},
            }
        )
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100},
                "node-1": {1: 200},
                "node-2": {2: 300},
            }
        )

        with mock_stack_trace_diagnostic(
            [
                make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
                RuntimeError("py-spy crashed"),
                make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
            ]
        ):
            executor = StackTraceClusterExecutor(rank_pids_provider=pids_provider)
            bad_nodes = await executor.execute(agents=agents, timeout_seconds=120)

            assert "node-1" in bad_nodes
            assert "node-0" not in bad_nodes

    @pytest.mark.anyio
    async def test_rank_pids_provider_exception_isolates_node(self) -> None:
        agents = make_fake_agents(
            {
                "node-0": {"gpu": False},
                "node-1": {"gpu": True},
                "node-2": {"gpu": True},
            }
        )

        def raising_provider(node_id: str) -> dict[int, int]:
            if node_id == "node-0":
                raise RuntimeError("cannot query pids for node-0")
            return {0: 100}

        with mock_stack_trace_diagnostic(
            [
                make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
                make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
            ]
        ):
            executor = StackTraceClusterExecutor(rank_pids_provider=raising_provider)
            bad_nodes = await executor.execute(agents=agents, timeout_seconds=120)

        assert "node-0" in bad_nodes


class TestStackTraceClusterExecutorIntegrationWithPipeline:
    """Test StackTraceClusterExecutor used as pre_executor in the orchestrator pipeline."""

    @pytest.mark.anyio
    async def test_outlier_evicted_via_pre_executor(self) -> None:
        """When used as pre_executor, outlier is evicted before hardware diagnostics run."""
        from miles.utils.ft.controller.diagnostics.executors import GpuClusterExecutor
        from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator

        agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": True},
                "node-2": {"gpu": True},
            }
        )
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100},
                "node-1": {1: 200},
                "node-2": {2: 300},
            }
        )

        with mock_stack_trace_diagnostic(
            [
                make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
                make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
                make_trace_result("node-2", passed=True, details=SAMPLE_PYSPY_JSON_DIFFERENT_STUCK),
            ]
        ):
            orchestrator = DiagnosticOrchestrator(
                agents=agents,
                pipeline=[GpuClusterExecutor()],
            )
            decision = await orchestrator.run_diagnostic_pipeline(
                pre_executors=[StackTraceClusterExecutor(rank_pids_provider=pids_provider)],
            )

            assert decision.bad_node_ids == ["node-2"]

    @pytest.mark.anyio
    async def test_no_outlier_falls_through_to_gpu_executor(self) -> None:
        """When StackTraceClusterExecutor finds no outlier, subsequent executors run normally."""
        from miles.utils.ft.controller.diagnostics.executors import GpuClusterExecutor
        from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator

        agents = make_fake_agents(
            {
                "node-0": {"gpu": True},
                "node-1": {"gpu": False},
            }
        )
        pids_provider = make_rank_pids_provider(
            {
                "node-0": {0: 100},
                "node-1": {1: 200},
            }
        )

        with mock_stack_trace_diagnostic(
            [
                make_trace_result("node-0", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
                make_trace_result("node-1", passed=True, details=SAMPLE_PYSPY_JSON_STUCK),
            ]
        ):
            orchestrator = DiagnosticOrchestrator(
                agents=agents,
                pipeline=[GpuClusterExecutor()],
            )
            decision = await orchestrator.run_diagnostic_pipeline(
                pre_executors=[StackTraceClusterExecutor(rank_pids_provider=pids_provider)],
            )

            assert decision.bad_node_ids == ["node-1"]
