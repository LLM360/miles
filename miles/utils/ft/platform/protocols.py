from enum import Enum
from typing import Protocol


class JobStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    PENDING = "pending"


class NodeManagerProtocol(Protocol):
    async def mark_node_bad(self, node_id: str, reason: str) -> None: ...

    async def unmark_node_bad(self, node_id: str) -> None: ...

    async def get_bad_nodes(self) -> list[str]: ...


class TrainingJobProtocol(Protocol):
    async def stop_training(self, timeout_seconds: int = 300) -> None: ...

    async def submit_training(self) -> str: ...

    async def get_training_status(self) -> JobStatus: ...


class NotificationProtocol(Protocol):
    async def send(self, title: str, content: str, severity: str) -> None: ...
