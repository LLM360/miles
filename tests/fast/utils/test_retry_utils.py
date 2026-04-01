import pytest

from miles.utils.retry_utils import retry

pytestmark = pytest.mark.asyncio


class TestRetry:
    async def test_succeeds_immediately(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1

        await retry(fn)
        assert call_count == 1

    async def test_retries_then_succeeds(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ValueError("not yet")

        await retry(fn)
        assert call_count == 4

    async def test_single_retry(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail once")

        await retry(fn)
        assert call_count == 2

    async def test_logs_on_retry(self, caplog):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("boom")

        with caplog.at_level("WARNING"):
            await retry(fn)

        retry_messages = [r for r in caplog.records if "retrying" in r.message]
        assert len(retry_messages) == 2

    async def test_no_log_on_first_success(self, caplog):
        async def fn():
            pass

        with caplog.at_level("WARNING"):
            await retry(fn)

        assert not any("retrying" in r.message for r in caplog.records)

    async def test_logs_include_exc_info(self, caplog):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("detail")

        with caplog.at_level("WARNING"):
            await retry(fn)

        retry_records = [r for r in caplog.records if "retrying" in r.message]
        assert len(retry_records) == 1
        assert retry_records[0].exc_info is not None
