"""Tests for recovery_orchestrator/helpers.py (retry_async, stop_and_submit, SlidingWindowThrottle, evict_and_notify)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle, evict_and_notify, stop_and_submit
from miles.utils.ft.models.fault import TriggerType
from miles.utils.ft.utils.retry import RetryResult, retry_async
from miles.utils.ft.protocols.platform import JobStatus
from tests.fast.utils.ft.conftest import (
    FakeNodeManager,
    FakeNotifier,
    FakeTrainingJob,
    make_failing_node_manager,
    make_failing_training_job,
)


class TestRetryAsyncEdgePaths:
    def test_succeeds_on_second_attempt(self) -> None:
        call_count = 0

        async def flaky_fn() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")

        result = asyncio.run(retry_async(func=flaky_fn, description="test_retry"))

        assert isinstance(result, RetryResult)
        assert result.ok is True
        assert call_count == 2

    def test_all_retries_fail_returns_failure(self) -> None:
        async def always_fail() -> None:
            raise RuntimeError("permanent error")

        result = asyncio.run(retry_async(
            func=always_fail, description="test_fail", max_retries=2,
        ))

        assert isinstance(result, RetryResult)
        assert result.ok is False
        assert result.error is not None
        assert "permanent error" in result.error

    def test_preserves_return_value(self) -> None:
        async def returns_value() -> str:
            return "run-42"

        result = asyncio.run(retry_async(func=returns_value, description="test_return"))

        assert result.ok is True
        assert result.value == "run-42"

    def test_exponential_backoff_between_retries(self) -> None:
        sleep_durations: list[float] = []

        async def record_sleep(seconds: float) -> None:
            sleep_durations.append(seconds)

        async def always_fail() -> None:
            raise RuntimeError("fail")

        asyncio.run(retry_async(
            func=always_fail,
            description="test_backoff",
            max_retries=4,
            sleep_fn=record_sleep,
        ))

        assert sleep_durations == [1.0, 2.0, 4.0]

    def test_no_sleep_on_immediate_success(self) -> None:
        sleep_durations: list[float] = []

        async def record_sleep(seconds: float) -> None:
            sleep_durations.append(seconds)

        async def succeed() -> str:
            return "ok"

        asyncio.run(retry_async(
            func=succeed,
            description="test_no_sleep",
            sleep_fn=record_sleep,
        ))

        assert sleep_durations == []

    def test_backoff_caps_at_max(self) -> None:
        sleep_durations: list[float] = []

        async def record_sleep(seconds: float) -> None:
            sleep_durations.append(seconds)

        async def always_fail() -> None:
            raise RuntimeError("fail")

        asyncio.run(retry_async(
            func=always_fail,
            description="test_cap",
            max_retries=8,
            sleep_fn=record_sleep,
        ))

        assert all(d <= 30.0 for d in sleep_durations)
        assert sleep_durations[-1] == 30.0


class TestStopAndSubmit:
    @pytest.mark.anyio
    async def test_happy_path_returns_true(self) -> None:
        training_job = FakeTrainingJob()

        result = await stop_and_submit(training_job)

        assert result is True
        assert training_job._stopped
        assert training_job._submitted

    @pytest.mark.anyio
    async def test_stop_failure_but_job_stopped_still_submits(self) -> None:
        training_job = make_failing_training_job(
            fail_stop=True, status_sequence=[JobStatus.STOPPED],
        )

        result = await stop_and_submit(training_job)

        assert result is True
        assert training_job._submitted

    @pytest.mark.anyio
    async def test_stop_failure_job_still_running_skips_submit(self) -> None:
        training_job = make_failing_training_job(
            fail_stop=True, status_sequence=[JobStatus.RUNNING],
        )

        result = await stop_and_submit(training_job)

        assert result is False
        assert not training_job._submitted

    @pytest.mark.anyio
    async def test_submit_failure_returns_false(self) -> None:
        training_job = make_failing_training_job(fail_submit=True)

        result = await stop_and_submit(training_job)

        assert result is False
        assert training_job._stopped

    @pytest.mark.anyio
    async def test_excluded_node_ids_passed_to_submit(self) -> None:
        training_job = FakeTrainingJob()

        result = await stop_and_submit(
            training_job, excluded_node_ids=["node-x", "node-y"],
        )

        assert result is True
        assert training_job._last_excluded_node_ids == ["node-x", "node-y"]

    @pytest.mark.anyio
    async def test_on_new_run_called_after_successful_submit(self) -> None:
        training_job = FakeTrainingJob()
        calls: list[str] = []

        result = await stop_and_submit(
            training_job, on_new_run=lambda run_id: calls.append(run_id),
        )

        assert result is True
        assert len(calls) == 1
        assert calls[0].startswith("fake-")

    @pytest.mark.anyio
    async def test_on_new_run_not_called_on_submit_failure(self) -> None:
        training_job = make_failing_training_job(fail_submit=True)
        calls: list[str] = []

        result = await stop_and_submit(
            training_job, on_new_run=lambda run_id: calls.append(run_id),
        )

        assert result is False
        assert len(calls) == 0

    @pytest.mark.anyio
    async def test_stop_failure_job_failed_still_submits(self) -> None:
        training_job = make_failing_training_job(
            fail_stop=True, status_sequence=[JobStatus.FAILED],
        )

        result = await stop_and_submit(training_job)

        assert result is True
        assert training_job._submitted

    @pytest.mark.anyio
    async def test_submit_called_exactly_once(self) -> None:
        """stop_and_submit calls submit_training exactly once (no outer retry)."""
        training_job = FakeTrainingJob()

        result = await stop_and_submit(training_job)

        assert result is True
        assert training_job._submit_call_count == 1

    @pytest.mark.anyio
    async def test_submit_exception_returns_false_without_retry(self) -> None:
        """When submit_training raises, stop_and_submit returns False without retrying."""
        training_job = make_failing_training_job(fail_submit=True)

        result = await stop_and_submit(training_job)

        assert result is False


class TestSlidingWindowThrottle:
    def test_not_throttled_below_max_count(self) -> None:
        throttle = SlidingWindowThrottle(window_minutes=30.0, max_count=3)
        throttle.record(TriggerType.CRASH)
        throttle.record(TriggerType.CRASH)
        assert not throttle.is_throttled(TriggerType.CRASH)

    def test_throttled_at_max_count(self) -> None:
        throttle = SlidingWindowThrottle(window_minutes=30.0, max_count=3)
        throttle.record(TriggerType.CRASH)
        throttle.record(TriggerType.CRASH)
        throttle.record(TriggerType.CRASH)
        assert throttle.is_throttled(TriggerType.CRASH)

    def test_different_triggers_tracked_separately(self) -> None:
        throttle = SlidingWindowThrottle(window_minutes=30.0, max_count=2)
        throttle.record(TriggerType.CRASH)
        throttle.record(TriggerType.CRASH)
        assert throttle.is_throttled(TriggerType.CRASH)
        assert not throttle.is_throttled(TriggerType.HANG)

    def test_old_entries_outside_window_ignored(self) -> None:
        throttle = SlidingWindowThrottle(window_minutes=10.0, max_count=2)

        old_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        with patch(
            "miles.utils.ft.controller.recovery_orchestrator.helpers.datetime",
        ) as mock_dt:
            mock_dt.now.return_value = old_time
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            throttle.record(TriggerType.CRASH)

        throttle.record(TriggerType.CRASH)
        assert not throttle.is_throttled(TriggerType.CRASH)

    def test_empty_history_not_throttled(self) -> None:
        throttle = SlidingWindowThrottle(window_minutes=30.0, max_count=1)
        assert not throttle.is_throttled(TriggerType.CRASH)


class TestEvictAndNotify:
    @pytest.mark.anyio
    async def test_success_sends_warning_notification(self) -> None:
        node_manager = FakeNodeManager()
        training_job = FakeTrainingJob()
        notifier = FakeNotifier()

        result = await evict_and_notify(
            node_manager=node_manager,
            training_job=training_job,
            bad_node_ids=["node-0"],
            reason="test eviction",
            notifier=notifier,
        )

        assert result is True
        assert node_manager.is_node_bad("node-0")
        assert len(notifier.calls) == 1
        title, content, severity = notifier.calls[0]
        assert title == "Node Eviction Succeeded"
        assert severity == "warning"
        assert "node-0" in content

    @pytest.mark.anyio
    async def test_fail_fast_returns_false_on_mark_failure(self) -> None:
        node_manager = make_failing_node_manager()
        training_job = FakeTrainingJob()
        notifier = FakeNotifier()

        result = await evict_and_notify(
            node_manager=node_manager,
            training_job=training_job,
            bad_node_ids=["node-0"],
            reason="test eviction",
            notifier=notifier,
            fail_fast=True,
        )

        assert result is False
        assert not training_job._submitted
        assert len(notifier.calls) == 0

    @pytest.mark.anyio
    async def test_no_fail_fast_continues_on_mark_failure(self) -> None:
        """With fail_fast=False, partial mark failures still proceed to restart."""
        node_manager = FakeNodeManager()
        training_job = FakeTrainingJob()
        notifier = FakeNotifier()

        async def fail_for_bad(node_id: str, reason: str = "") -> None:
            if node_id == "node-bad":
                raise RuntimeError("K8s API unreachable")
            node_manager._bad_nodes.add(node_id)

        node_manager.mark_node_bad = fail_for_bad  # type: ignore[assignment]

        result = await evict_and_notify(
            node_manager=node_manager,
            training_job=training_job,
            bad_node_ids=["node-ok", "node-bad"],
            reason="test eviction",
            notifier=notifier,
            fail_fast=False,
        )

        assert result is True
        assert training_job._submitted
        assert len(notifier.calls) == 1
        assert notifier.calls[0][0] == "Node Eviction Succeeded"

    @pytest.mark.anyio
    async def test_restart_failure_returns_false_no_success_notification(self) -> None:
        node_manager = FakeNodeManager()
        training_job = make_failing_training_job(fail_submit=True)
        notifier = FakeNotifier()

        result = await evict_and_notify(
            node_manager=node_manager,
            training_job=training_job,
            bad_node_ids=["node-0"],
            reason="test eviction",
            notifier=notifier,
        )

        assert result is False
        assert node_manager.is_node_bad("node-0")
        assert len(notifier.calls) == 0

    @pytest.mark.anyio
    async def test_skips_already_bad_nodes(self) -> None:
        node_manager = FakeNodeManager()
        await node_manager.mark_node_bad("node-a", reason="previous")
        training_job = FakeTrainingJob()
        notifier = FakeNotifier()

        result = await evict_and_notify(
            node_manager=node_manager,
            training_job=training_job,
            bad_node_ids=["node-a", "node-b"],
            reason="test eviction",
            notifier=notifier,
        )

        assert result is True
        assert node_manager.is_node_bad("node-a")
        assert node_manager.is_node_bad("node-b")
        assert len(notifier.calls) == 1
