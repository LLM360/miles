from __future__ import annotations

import asyncio

from miles.utils.ft.controller.recovery_orchestrator.helpers import RetryResult, retry_async


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
