"""ABC compliance tests for NodeAgentProtocol and NodeExecutorProtocol."""

from __future__ import annotations

import pytest

from miles.utils.ft.adapters.types import NodeAgentProtocol, NodeExecutorProtocol
from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.agents.diagnostics.dispatcher import NodeDiagnosticDispatcher
from miles.utils.ft.agents.diagnostics.executors.gpu import GpuNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.nccl import NcclNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.stack_trace import StackTraceNodeExecutor


class TestNodeAgentProtocolCompliance:
    def test_ft_node_agent_satisfies_protocol(self) -> None:
        agent = FtNodeAgent(node_id="test-node")
        assert isinstance(agent, NodeAgentProtocol)
        agent._exporter.shutdown()

    def test_diagnostic_runner_satisfies_protocol(self) -> None:
        runner = NodeDiagnosticDispatcher(node_id="test-node")
        assert isinstance(runner, NodeAgentProtocol)

    def test_incomplete_subclass_raises_type_error(self) -> None:
        class _Incomplete(NodeAgentProtocol):
            pass

        with pytest.raises(TypeError):
            _Incomplete()


class TestNodeExecutorProtocolCompliance:
    def test_gpu_diagnostic_satisfies_protocol(self) -> None:
        assert isinstance(GpuNodeExecutor(), NodeExecutorProtocol)

    def test_stack_trace_diagnostic_satisfies_protocol(self) -> None:
        assert isinstance(StackTraceNodeExecutor(), NodeExecutorProtocol)

    def test_nccl_diagnostic_satisfies_protocol(self) -> None:
        assert isinstance(
            NcclNodeExecutor(diagnostic_type="nccl_simple", expected_bandwidth_gbps=350.0),
            NodeExecutorProtocol,
        )

    def test_incomplete_subclass_raises_type_error(self) -> None:
        class _Incomplete(NodeExecutorProtocol):
            diagnostic_type = "test"

        with pytest.raises(TypeError):
            _Incomplete()
