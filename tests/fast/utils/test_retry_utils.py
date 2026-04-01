import pytest

from miles.utils.retry_utils import retry_until_true

pytestmark = pytest.mark.asyncio


class TestRetryUntilTrue:
    async def test_succeeds_immediately(self):
        call_count = 0

        async def fn() -> bool:
            nonlocal call_count
            call_count += 1
            return True

        await retry_until_true(fn)
        assert call_count == 1

    async def test_retries_then_succeeds(self):
        call_count = 0

        async def fn() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 4

        await retry_until_true(fn)
        assert call_count == 4

    async def test_single_retry(self):
        call_count = 0

        async def fn() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 2

        await retry_until_true(fn)
        assert call_count == 2

    async def test_exception_propagates(self):
        """fn raising an exception is not caught — it propagates to the caller."""

        async def fn() -> bool:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await retry_until_true(fn)

    async def test_logs_on_retry(self, caplog):
        call_count = 0

        async def fn() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 3

        with caplog.at_level("WARNING"):
            await retry_until_true(fn, description="my_op")

        assert sum("my_op" in r.message and "retrying" in r.message for r in caplog.records) == 2

    async def test_no_log_on_first_success(self, caplog):
        async def fn() -> bool:
            return True

        with caplog.at_level("WARNING"):
            await retry_until_true(fn)

        assert not any("retrying" in r.message for r in caplog.records)
