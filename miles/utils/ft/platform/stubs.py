from uuid import uuid4

import structlog

from miles.utils.ft.platform.protocols import JobStatus

log = structlog.get_logger(__name__)


class StubNodeManager:
    """Logs operations but does not call real K8s API."""

    async def mark_node_bad(self, node_id: str, reason: str) -> None:
        log.info("stub_mark_node_bad", node_id=node_id, reason=reason)

    async def unmark_node_bad(self, node_id: str) -> None:
        log.info("stub_unmark_node_bad", node_id=node_id)

    async def get_bad_nodes(self) -> list[str]:
        return []


class StubTrainingJob:
    """Logs operations but does not call real Ray Job API."""

    async def stop_training(self, timeout_seconds: int = 300) -> None:
        log.info("stub_stop_training", timeout_seconds=timeout_seconds)

    async def submit_training(self) -> str:
        run_id = uuid4().hex[:8]
        log.info("stub_submit_training", run_id=run_id)
        return run_id

    async def get_training_status(self) -> JobStatus:
        return JobStatus.RUNNING
