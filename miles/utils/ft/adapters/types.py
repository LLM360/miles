"""Cross-layer boundary contracts: Protocols and constants shared across layers."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from miles.utils.ft.agents.types import DiagnosticResult


# ---------------------------------------------------------------------------
# protocols/agents.py — controller calls agents
# ---------------------------------------------------------------------------

DIAGNOSTIC_TIMEOUT_SECONDS: int = 120


@runtime_checkable
class NodeAgentProtocol(Protocol):
    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
        **kwargs: object,
    ) -> DiagnosticResult: ...


@runtime_checkable
class NodeExecutorProtocol(Protocol):
    diagnostic_type: str

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
    ) -> DiagnosticResult: ...


@runtime_checkable
class ClusterExecutorProtocol(Protocol):
    """Strategy for executing one diagnostic step within the pipeline.

    Returns bad_node_ids (empty if all healthy).
    """

    async def execute(
        self,
        agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> list[str]: ...


# ---------------------------------------------------------------------------
# protocols/controller.py — agents call controller
# ---------------------------------------------------------------------------

REGISTER_TIMEOUT_SECONDS: float = 10


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


# ---------------------------------------------------------------------------
# protocols/platform.py — controller calls platform
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    PENDING = "pending"


STOP_TRAINING_TIMEOUT_SECONDS: int = 300


@runtime_checkable
class NodeManagerProtocol(Protocol):
    async def mark_node_bad(self, node_id: str, reason: str) -> None: ...

    async def unmark_node_bad(self, node_id: str) -> None: ...

    async def get_bad_nodes(self) -> list[str]: ...


@runtime_checkable
class TrainingJobProtocol(Protocol):
    async def stop_training(self, timeout_seconds: int = STOP_TRAINING_TIMEOUT_SECONDS) -> None: ...

    async def submit_training(
        self,
        excluded_node_ids: list[str] | None = None,
    ) -> str: ...

    async def get_training_status(self) -> JobStatus: ...


@runtime_checkable
class NotifierProtocol(Protocol):
    async def send(self, title: str, content: str, severity: str) -> None: ...

    async def aclose(self) -> None: ...
