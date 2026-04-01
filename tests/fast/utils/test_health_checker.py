import asyncio

import pytest

from miles.utils.health_checker import SimpleHealthChecker


def _make_checker(
    *,
    check_fn=None,
    on_failure=None,
    interval: float = 0.01,
    first_wait: float = 0.0,
    name: str = "test",
) -> SimpleHealthChecker:
    if check_fn is None:

        async def check_fn() -> None:
            pass

    if on_failure is None:
        on_failure = lambda: None

    return SimpleHealthChecker(
        name=name,
        check_fn=check_fn,
        on_failure=on_failure,
        interval=interval,
        first_wait=first_wait,
    )


class TestStartStop:
    async def test_start_creates_task(self):
        checker = _make_checker()
        assert checker._task is None

        await checker.start()
        assert checker._task is not None

        checker.stop()
        assert checker._task is None

    async def test_start_is_idempotent(self):
        checker = _make_checker()
        await checker.start()
        task = checker._task

        await checker.start()
        assert checker._task is task

        checker.stop()

    async def test_stop_without_start_is_noop(self):
        checker = _make_checker()
        checker.stop()


class TestCheckFnCalled:
    async def test_check_fn_called_periodically(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker = _make_checker(check_fn=check_fn, interval=0.01)
        await checker.start()

        await asyncio.sleep(0.05)
        checker.stop()

        assert call_count >= 2

    async def test_first_wait_delays_first_check(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker = _make_checker(check_fn=check_fn, interval=0.01, first_wait=0.1)
        await checker.start()

        await asyncio.sleep(0.05)
        assert call_count == 0

        await asyncio.sleep(0.1)
        checker.stop()

        assert call_count >= 1


class TestOnFailure:
    async def test_on_failure_called_when_check_raises(self):
        failure_count = 0

        async def check_fn() -> None:
            raise RuntimeError("boom")

        def on_failure() -> None:
            nonlocal failure_count
            failure_count += 1

        checker = _make_checker(check_fn=check_fn, on_failure=on_failure, interval=0.01)
        await checker.start()

        await asyncio.sleep(0.05)
        checker.stop()

        assert failure_count >= 1

    async def test_loop_continues_after_failure(self):
        """on_failure is called but the loop keeps running."""
        failure_count = 0

        async def check_fn() -> None:
            raise RuntimeError("boom")

        def on_failure() -> None:
            nonlocal failure_count
            failure_count += 1

        checker = _make_checker(check_fn=check_fn, on_failure=on_failure, interval=0.01)
        await checker.start()

        await asyncio.sleep(0.05)
        checker.stop()

        assert failure_count >= 2

    async def test_intermittent_failure_does_not_stop_loop(self):
        """Failure on some ticks, success on others — loop keeps going."""
        call_count = 0
        failure_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("intermittent")

        def on_failure() -> None:
            nonlocal failure_count
            failure_count += 1

        checker = _make_checker(check_fn=check_fn, on_failure=on_failure, interval=0.01)
        await checker.start()

        await asyncio.sleep(0.08)
        checker.stop()

        assert call_count >= 4
        assert failure_count >= 1


class TestPauseResume:
    async def test_paused_checker_does_not_call_check_fn(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker = _make_checker(check_fn=check_fn, interval=0.01)
        checker.pause()

        await checker.start()
        await asyncio.sleep(0.05)
        checker.stop()

        assert call_count == 0

    async def test_resume_after_pause_resumes_checking(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker = _make_checker(check_fn=check_fn, interval=0.01)
        checker.pause()

        await checker.start()
        await asyncio.sleep(0.03)
        assert call_count == 0

        checker.resume()
        await asyncio.sleep(0.05)
        checker.stop()

        assert call_count >= 2

    async def test_pause_resume_flags(self):
        checker = _make_checker()
        assert not checker._paused

        checker.pause()
        assert checker._paused

        checker.resume()
        assert not checker._paused
