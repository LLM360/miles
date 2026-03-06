"""Shared recovery primitives used by both FtController and RecoveryOrchestrator."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from miles.utils.ft.models.fault import TriggerType
from miles.utils.ft.protocols.platform import JobStatus, NodeManagerProtocol, NotificationProtocol, TrainingJobProtocol
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
    try:
        return set(await node_manager.get_bad_nodes())
    except Exception:
        logger.warning("get_bad_nodes_failed, proceeding without filter", exc_info=True)
        return set()


async def retry_mark_node_bad(
    node_manager: NodeManagerProtocol,
    node_id: str,
    reason: str,
) -> RetryResult[None]:
    return await retry_async(
        lambda: node_manager.mark_node_bad(node_id, reason=reason),
        description=f"mark_node_bad({node_id})",
    )


async def safe_notify(
    notifier: NotificationProtocol | None,
    title: str,
    content: str,
    severity: str = "critical",
) -> None:
    if notifier is None:
        return
    try:
        await notifier.send(title=title, content=content, severity=severity)
    except Exception:
        logger.exception("notifier_send_failed title=%s", title)


async def evict_and_notify(
    node_manager: NodeManagerProtocol,
    training_job: TrainingJobProtocol,
    bad_node_ids: list[str],
    reason: str,
    notifier: NotificationProtocol | None = None,
    excluded_node_ids: list[str] | None = None,
    on_new_run: Callable[[str], None] | None = None,
    fail_fast: bool = True,
) -> bool:
    """Mark nodes bad, restart training, and send a success notification.

    Returns True when the full eviction + restart sequence succeeds.

    When ``fail_fast=True`` (default), marks nodes in parallel and returns
    False immediately if any mark fails.  When ``fail_fast=False``, marks
    nodes sequentially and continues even if some fail.
    """
    already_bad = await get_already_bad_nodes(node_manager)
    nodes_to_mark = [nid for nid in bad_node_ids if nid not in already_bad]
    if len(nodes_to_mark) < len(bad_node_ids):
        logger.info(
            "evict_skipped_already_bad skipped=%s",
            sorted(set(bad_node_ids) - set(nodes_to_mark)),
        )

    if fail_fast:
        results = await asyncio.gather(*(
            retry_mark_node_bad(node_manager, node_id=nid, reason=reason)
            for nid in nodes_to_mark
        ))
        if not all(r.ok for r in results):
            return False
    else:
        failed_nodes: list[str] = []
        for nid in nodes_to_mark:
            result = await retry_mark_node_bad(node_manager, node_id=nid, reason=reason)
            if not result.ok:
                failed_nodes.append(nid)
        if failed_nodes:
            logger.error("mark_bad_partial_failure nodes=%s", failed_nodes)

    success = await stop_and_submit(
        training_job, excluded_node_ids=excluded_node_ids, on_new_run=on_new_run,
    )
    if not success:
        return False

    if bad_node_ids:
        await safe_notify(
            notifier,
            title="Node Eviction Succeeded",
            content=f"Evicted nodes {bad_node_ids} reason={reason}",
            severity="warning",
        )
    return True


class SlidingWindowThrottle:
    """Tracks event frequency per trigger and throttles when a limit is exceeded.

    Within a sliding window of ``window_minutes``, if the same trigger fires
    ``max_count`` or more times, ``is_throttled`` returns True.
    """

    def __init__(self, window_minutes: float, max_count: int) -> None:
        self._window_minutes = window_minutes
        self._max_count = max_count
        self._history: list[tuple[TriggerType, datetime]] = []

    def record(self, trigger: TriggerType) -> None:
        self._history.append((trigger, datetime.now(timezone.utc)))

    def is_throttled(self, trigger: TriggerType) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self._window_minutes)
        recent_count = sum(
            1 for t, ts in self._history
            if t == trigger and ts >= cutoff
        )
        return recent_count >= self._max_count
