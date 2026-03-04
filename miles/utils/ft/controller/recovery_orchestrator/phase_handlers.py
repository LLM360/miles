from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Any

from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery_orchestrator.alert_checker import AlertChecker
from miles.utils.ft.controller.recovery_orchestrator.context import (
    RecoveryContext,
    _MAX_RETRIES,
    _PENDING_TIMEOUT_SECONDS,
    _is_finite,
)
from miles.utils.ft.models import ActionType, RecoveryPhase
from miles.utils.ft.platform.protocols import (
    DiagnosticSchedulerProtocol,
    JobStatus,
    NodeManagerProtocol,
    NotificationProtocol,
    TrainingJobProtocol,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# CHECK_ALERTS
# -------------------------------------------------------------------


async def step_check_alerts(
    ctx: RecoveryContext,
    alert_checker: AlertChecker,
) -> RecoveryPhase:
    bad_node_ids, reasons = alert_checker.check_alerts()

    if bad_node_ids:
        ctx.bad_node_ids = bad_node_ids
        logger.info("check_alerts_found bad_nodes=%s reasons=%s", ctx.bad_node_ids, reasons)
        return RecoveryPhase.EVICT_AND_RESTART

    logger.info("check_alerts_clean trigger=%s", ctx.trigger)
    return RecoveryPhase.REATTEMPTING


# -------------------------------------------------------------------
# REATTEMPTING
# -------------------------------------------------------------------


async def step_reattempting(
    ctx: RecoveryContext,
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
) -> RecoveryPhase | None:
    if not ctx.reattempt_submitted:
        return await _reattempt_submit(ctx, training_job, mini_wandb)
    return await _reattempt_poll(ctx, training_job, mini_wandb)


async def _reattempt_submit(
    ctx: RecoveryContext,
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
) -> RecoveryPhase | None:
    success = await _stop_clear_submit(training_job, mini_wandb)
    if not success:
        return RecoveryPhase.NOTIFY

    ctx.reattempt_submitted = True
    ctx.reattempt_submit_time = datetime.now(timezone.utc)
    logger.info("reattempt_submitted trigger=%s", ctx.trigger)
    return None


async def _reattempt_poll(
    ctx: RecoveryContext,
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
) -> RecoveryPhase | None:
    status = await training_job.get_training_status()

    if status == JobStatus.RUNNING:
        iteration = mini_wandb.latest(metric_name="iteration", rank=0)
        ctx.reattempt_start_time = datetime.now(timezone.utc)
        ctx.reattempt_base_iteration = (
            int(iteration) if iteration is not None and _is_finite(iteration) else 0
        )
        logger.info("reattempt_running base_iteration=%s", ctx.reattempt_base_iteration)
        return RecoveryPhase.MONITORING

    if status == JobStatus.FAILED:
        logger.warning("reattempt_immediately_failed trigger=%s", ctx.trigger)
        return RecoveryPhase.DIAGNOSING

    if ctx.reattempt_submit_time is not None:
        elapsed = (datetime.now(timezone.utc) - ctx.reattempt_submit_time).total_seconds()
        if elapsed > _PENDING_TIMEOUT_SECONDS:
            logger.warning("reattempt_pending_timeout elapsed=%.0f", elapsed)
            return RecoveryPhase.NOTIFY

    return None


# -------------------------------------------------------------------
# MONITORING
# -------------------------------------------------------------------


async def step_monitoring(
    ctx: RecoveryContext,
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
) -> RecoveryPhase | None:
    status = await training_job.get_training_status()
    progress = _iteration_progress(ctx, mini_wandb)

    if status == JobStatus.FAILED:
        logger.warning(
            "monitoring_training_failed progress_iterations=%d trigger=%s",
            progress, ctx.trigger,
        )
        return RecoveryPhase.DIAGNOSING

    if status == JobStatus.RUNNING and progress >= ctx.monitoring_success_iterations:
        logger.info(
            "monitoring_success progress_iterations=%d threshold=%d",
            progress, ctx.monitoring_success_iterations,
        )
        return RecoveryPhase.DONE

    if ctx.reattempt_start_time is not None:
        elapsed = (datetime.now(timezone.utc) - ctx.reattempt_start_time).total_seconds()
        if elapsed > ctx.monitoring_timeout_seconds:
            logger.warning("monitoring_timeout elapsed=%.0f trigger=%s", elapsed, ctx.trigger)
            return RecoveryPhase.DIAGNOSING

    return None


def _iteration_progress(ctx: RecoveryContext, mini_wandb: MiniWandb) -> int:
    current_iteration = mini_wandb.latest(metric_name="iteration", rank=0)
    if current_iteration is None or not _is_finite(current_iteration):
        return 0
    base = ctx.reattempt_base_iteration or 0
    return int(current_iteration) - base


# -------------------------------------------------------------------
# DIAGNOSING
# -------------------------------------------------------------------


async def step_diagnosing(
    ctx: RecoveryContext,
    diagnostic_scheduler: DiagnosticSchedulerProtocol,
) -> RecoveryPhase:
    decision = await diagnostic_scheduler.run_diagnostic_pipeline(
        trigger_reason=ctx.trigger,
    )

    if decision.action == ActionType.MARK_BAD_AND_RESTART:
        ctx.bad_node_ids = list(decision.bad_node_ids)
        logger.info("diagnosing_found_bad_nodes bad_nodes=%s", ctx.bad_node_ids)
        return RecoveryPhase.EVICT_AND_RESTART

    logger.info("diagnosing_all_passed trigger=%s", ctx.trigger)
    return RecoveryPhase.NOTIFY


# -------------------------------------------------------------------
# EVICT_AND_RESTART
# -------------------------------------------------------------------


async def step_evict_and_restart(
    ctx: RecoveryContext,
    node_manager: NodeManagerProtocol,
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
) -> RecoveryPhase:
    for node_id in ctx.bad_node_ids:
        success = await _retry_async(
            lambda nid=node_id: node_manager.mark_node_bad(
                nid, reason=f"recovery eviction: {ctx.trigger}",
            ),
            description=f"mark_node_bad({node_id})",
        )
        if not success:
            return RecoveryPhase.NOTIFY

    success = await _stop_clear_submit(training_job, mini_wandb)
    if not success:
        return RecoveryPhase.NOTIFY

    logger.info("evict_and_restart_done bad_nodes=%s trigger=%s", ctx.bad_node_ids, ctx.trigger)
    return RecoveryPhase.DONE


# -------------------------------------------------------------------
# NOTIFY
# -------------------------------------------------------------------


async def step_notify(
    ctx: RecoveryContext,
    notifier: NotificationProtocol | None,
) -> RecoveryPhase:
    prev = ctx.phase_before_notify
    message = (
        f"Recovery requires human intervention. "
        f"trigger={ctx.trigger} "
        f"phase_before_notify={prev.value if prev else 'unknown'}"
    )
    logger.warning("recovery_notify reason=%s", message)

    if notifier is not None:
        try:
            await notifier.send(title="Recovery Alert", content=message, severity="critical")
        except Exception:
            logger.exception("recovery_notifier_send_failed")

    return RecoveryPhase.DONE


# -------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------


async def _stop_clear_submit(
    training_job: TrainingJobProtocol,
    mini_wandb: MiniWandb,
) -> bool:
    """Stop training, clear metrics, submit new job. Returns True on success."""
    try:
        await training_job.stop_training()
    except Exception:
        logger.warning("stop_training_failed", exc_info=True)

    mini_wandb.clear()

    return await _retry_async(
        training_job.submit_training,
        description="submit_training",
    )


async def _retry_async(
    func: Callable[[], Coroutine[Any, Any, Any]],
    description: str,
    max_retries: int = _MAX_RETRIES,
) -> bool:
    for attempt in range(max_retries):
        try:
            await func()
            return True
        except Exception:
            logger.warning(
                "retry_failed description=%s attempt=%d/%d",
                description, attempt + 1, max_retries,
                exc_info=True,
            )
    logger.error("retry_exhausted description=%s", description)
    return False
