"""Tests for miles.utils.ft.retry (retry_sync, retry_async_or_raise)."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from miles.utils.ft.retry import RetryResult, retry_async_or_raise, retry_sync


class TestRetrySyncHappyPath:
    def test_immediate_success(self) -> None:
        result = retry_sync(func=lambda: "ok", description="test")
        assert result == RetryResult(ok=True, value="ok")

    def test_succeeds_on_second_attempt(self) -> None:
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            return "recovered"

        with patch("miles.utils.ft.retry.time.sleep"):
            result = retry_sync(func=flaky, description="flaky")

        assert result.ok is True
        assert result.value == "recovered"
        assert call_count == 2


class TestRetrySyncFailure:
    def test_all_retries_fail(self) -> None:
        with patch("miles.utils.ft.retry.time.sleep"):
            result = retry_sync(
                func=lambda: (_ for _ in ()).throw(RuntimeError("permanent")),
                description="always_fail",
                max_retries=3,
            )

        assert result.ok is False
        assert result.error is not None
        assert "permanent" in result.error

    def test_returns_none_value_on_failure(self) -> None:
        with patch("miles.utils.ft.retry.time.sleep"):
            result = retry_sync(
                func=lambda: (_ for _ in ()).throw(ValueError("err")),
                description="fail",
                max_retries=2,
            )

        assert result.value is None


class TestRetrySyncBackoff:
    def test_fixed_delay_when_base_equals_max(self) -> None:
        sleep_calls: list[float] = []

        def record_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with patch("miles.utils.ft.retry.time.sleep", side_effect=record_sleep):
            retry_sync(
                func=lambda: (_ for _ in ()).throw(RuntimeError("fail")),
                description="fixed",
                max_retries=4,
                backoff_base=0.5,
                max_backoff=0.5,
            )

        assert sleep_calls == [0.5, 0.5, 0.5]

    def test_exponential_backoff(self) -> None:
        sleep_calls: list[float] = []

        def record_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with patch("miles.utils.ft.retry.time.sleep", side_effect=record_sleep):
            retry_sync(
                func=lambda: (_ for _ in ()).throw(RuntimeError("fail")),
                description="exponential",
                max_retries=4,
                backoff_base=1.0,
                max_backoff=30.0,
            )

        assert sleep_calls == [1.0, 2.0, 4.0]


class TestRetryAsyncOrRaiseHappyPath:
    @pytest.mark.anyio
    async def test_immediate_success_returns_value(self) -> None:
        async def fn() -> str:
            return "ok"

        result = await retry_async_or_raise(func=fn, description="test")
        assert result == "ok"

    @pytest.mark.anyio
    async def test_succeeds_on_third_attempt(self) -> None:
        call_count = 0

        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "recovered"

        result = await retry_async_or_raise(
            func=flaky, description="flaky", max_retries=3,
        )
        assert result == "recovered"
        assert call_count == 3


class TestRetryAsyncOrRaiseFailure:
    @pytest.mark.anyio
    async def test_raises_last_exception_on_exhaustion(self) -> None:
        async def always_fail() -> str:
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            await retry_async_or_raise(
                func=always_fail, description="fail", max_retries=2,
            )

    @pytest.mark.anyio
    async def test_raises_the_last_specific_exception(self) -> None:
        call_count = 0

        async def varying_error() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first")
            raise TypeError("second")

        with pytest.raises(TypeError, match="second"):
            await retry_async_or_raise(
                func=varying_error, description="varying", max_retries=2,
            )


class TestRetryAsyncOrRaisePerCallTimeout:
    @pytest.mark.anyio
    async def test_per_call_timeout_triggers(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(10)
            return "late"

        with pytest.raises(asyncio.TimeoutError):
            await retry_async_or_raise(
                func=slow,
                description="slow",
                max_retries=1,
                per_call_timeout=0.01,
            )
