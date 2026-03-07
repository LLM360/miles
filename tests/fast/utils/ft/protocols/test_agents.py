"""Protocol compliance tests for NodeAgentProtocol and DiagnosticProtocol."""

from __future__ import annotations

from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.controller.diagnostics.gpu_diagnostic import GpuDiagnostic
from miles.utils.ft.controller.diagnostics.nccl.intra_machine import IntraMachineCommDiagnostic
from miles.utils.ft.controller.diagnostics.stack_trace import StackTraceDiagnostic
from miles.utils.ft.protocols.agents import DiagnosticProtocol, NodeAgentProtocol


class TestNodeAgentProtocolCompliance:
    def test_ft_node_agent_satisfies_protocol(self) -> None:
        agent = FtNodeAgent(node_id="test-node")
        assert isinstance(agent, NodeAgentProtocol)
        agent._exporter.shutdown()

    def test_conforming_class_passes_isinstance(self) -> None:
        class _Conforming:
            async def run_diagnostic(
                self, diagnostic_type: str, timeout_seconds: int = 120, **kwargs: object,
            ) -> object:
                return None

        assert isinstance(_Conforming(), NodeAgentProtocol)

    def test_missing_method_fails_isinstance(self) -> None:
        class _Empty:
            pass

        assert not isinstance(_Empty(), NodeAgentProtocol)


class TestDiagnosticProtocolCompliance:
    def test_gpu_diagnostic_satisfies_protocol(self) -> None:
        assert isinstance(GpuDiagnostic(), DiagnosticProtocol)

    def test_stack_trace_diagnostic_satisfies_protocol(self) -> None:
        assert isinstance(StackTraceDiagnostic(), DiagnosticProtocol)

    def test_intra_machine_comm_diagnostic_satisfies_protocol(self) -> None:
        assert isinstance(IntraMachineCommDiagnostic(), DiagnosticProtocol)

    def test_conforming_class_passes_isinstance(self) -> None:
        class _Conforming:
            diagnostic_type = "test"

            async def run(self, node_id: str, timeout_seconds: int = 120) -> object:
                return None

        assert isinstance(_Conforming(), DiagnosticProtocol)

    def test_missing_run_method_fails_isinstance(self) -> None:
        class _MissingRun:
            diagnostic_type = "test"

        assert not isinstance(_MissingRun(), DiagnosticProtocol)
