from __future__ import annotations

import logging
from collections.abc import Callable

from miles.utils.ft.protocols.platform import JobStatus, NodeManagerProtocol, TrainingJobProtocol
from miles.utils.ft.utils.retry import RetryResult, retry_async

logger = logging.getLogger(__name__)


async def stop_and_submit(
    training_job: TrainingJobProtocol,
    excluded_node_ids: list[str] | None = None,
    on_new_run: Callable[[str], None] | None = None,
) -> bool:
    """Stop training, submit new job, notify caller of new run_id. Returns True on success."""
    stop_result = await retry_async(
        training_job.stop_training,
        description="stop_training",
        max_retries=2,
    )

    if not stop_result.ok:
        try:
            status = await training_job.get_training_status()
        except Exception:
            logger.error("get_status_after_stop_failure_also_failed", exc_info=True)
            return False

        if status not in (JobStatus.STOPPED, JobStatus.FAILED):
            logger.error(
                "stop_training_failed_job_still_active status=%s, skipping submit",
                status.value,
            )
            return False

    try:
        run_id = await training_job.submit_training(excluded_node_ids=excluded_node_ids)
    except Exception:
        logger.error("submit_training_failed", exc_info=True)
        return False

    if on_new_run is not None:
        on_new_run(run_id)
    return True


async def get_already_bad_nodes(node_manager: NodeManagerProtocol) -> set[str]:
    return set(await node_manager.get_bad_nodes())


async def retry_mark_node_bad(
    node_manager: NodeManagerProtocol,
    node_id: str,
    reason: str,
) -> RetryResult[None]:
    return await retry_async(
        lambda: node_manager.mark_node_bad(node_id, reason=reason),
        description=f"mark_node_bad({node_id})",
    )
