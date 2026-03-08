from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from miles.utils.ft.agents.types import DiagnosticPipelineResult
    from miles.utils.ft.adapters.types import ClusterExecutorProtocol

REGISTER_TIMEOUT_SECONDS: float = 10


@runtime_checkable
class DiagnosticOrchestratorProtocol(Protocol):
    async def run_diagnostic_pipeline(
        self,
        pre_executors: list[ClusterExecutorProtocol] | None = None,
    ) -> DiagnosticPipelineResult: ...


@runtime_checkable
class ControllerClientProtocol(Protocol):
    """Agent-side interface for communicating with the FtController.

    Implementations hide the transport (Ray, in-process, stub) so that
    agent code never imports ray or calls .remote().
    """

    def register_training_rank(
        self,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
        pid: int,
        timeout_seconds: float = REGISTER_TIMEOUT_SECONDS,
    ) -> None: ...

    def log_step(
        self,
        run_id: str,
        step: int,
        metrics: dict[str, float],
    ) -> None: ...


def ft_controller_actor_name(ft_id: str) -> str:
    if not ft_id:
        return "ft_controller"
    return f"ft_controller_{ft_id}"


def ft_node_agent_actor_name(ft_id: str, node_id: str) -> str:
    prefix = f"ft_node_agent_{ft_id}" if ft_id else "ft_node_agent"
    return f"{prefix}_{node_id}"
