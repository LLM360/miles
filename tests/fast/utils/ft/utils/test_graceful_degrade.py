"""Tests for miles.utils.ft.utils.graceful_degrade."""
from __future__ import annotations

import logging

import pytest

from miles.utils.ft.utils.graceful_degrade import graceful_degrade


class TestSyncGracefulDegrade:
    def test_success_returns_value(self) -> None:
        @graceful_degrade()
        def add(a: int, b: int) -> int:
            return a + b

        assert add(1, 2) == 3

    def test_exception_returns_default_none(self, caplog: pytest.LogCaptureFixture) -> None:
        @graceful_degrade()
        def boom() -> str:
            raise RuntimeError("kaboom")

        with caplog.at_level(logging.WARNING):
            result = boom()

        assert result is None
        assert "boom failed" in caplog.text
        assert "kaboom" in caplog.text

    def test_exception_returns_custom_default(self) -> None:
        @graceful_degrade(default=[])
        def boom() -> list[int]:
            raise ValueError("oops")

        assert boom() == []

    def test_custom_msg(self, caplog: pytest.LogCaptureFixture) -> None:
        @graceful_degrade(msg="custom failure message")
        def boom() -> None:
            raise RuntimeError("inner")

        with caplog.at_level(logging.WARNING):
            boom()

        assert "custom failure message" in caplog.text

    def test_custom_log_level(self, caplog: pytest.LogCaptureFixture) -> None:
        @graceful_degrade(log_level=logging.ERROR)
        def boom() -> None:
            raise RuntimeError("err")

        with caplog.at_level(logging.ERROR):
            boom()

        assert caplog.records[0].levelno == logging.ERROR

    def test_preserves_function_metadata(self) -> None:
        @graceful_degrade()
        def my_func() -> None:
            """My docstring."""

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_qualname_in_default_msg(self, caplog: pytest.LogCaptureFixture) -> None:
        class Foo:
            @graceful_degrade()
            def bar(self) -> None:
                raise RuntimeError("x")

        with caplog.at_level(logging.WARNING):
            Foo().bar()

        assert "Foo.bar failed" in caplog.text

    def test_works_with_staticmethod(self) -> None:
        class Foo:
            @staticmethod
            @graceful_degrade(default=-1)
            def compute(x: int) -> int:
                raise RuntimeError("nope")

        assert Foo.compute(42) == -1

    def test_log_includes_call_args(self, caplog: pytest.LogCaptureFixture) -> None:
        @graceful_degrade()
        def process(name: str, count: int) -> None:
            raise RuntimeError("fail")

        with caplog.at_level(logging.WARNING):
            process("widget", count=7)

        assert "process failed" in caplog.text
        assert "name='widget'" in caplog.text
        assert "count=7" in caplog.text

    def test_log_excludes_self_and_cls(self, caplog: pytest.LogCaptureFixture) -> None:
        class Svc:
            @graceful_degrade()
            def run(self, phase: str) -> None:
                raise RuntimeError("x")

        with caplog.at_level(logging.WARNING):
            Svc().run("init")

        assert "phase='init'" in caplog.text
        assert "self=" not in caplog.text

    def test_long_arg_repr_is_truncated(self, caplog: pytest.LogCaptureFixture) -> None:
        @graceful_degrade()
        def ingest(data: str) -> None:
            raise RuntimeError("big")

        with caplog.at_level(logging.WARNING):
            ingest("x" * 500)

        log_line = caplog.text
        assert "data=" in log_line
        assert "..." in log_line
        assert len(log_line) < 1000


class TestAsyncGracefulDegrade:
    @pytest.mark.anyio
    async def test_success_returns_value(self) -> None:
        @graceful_degrade()
        async def fetch() -> str:
            return "data"

        assert await fetch() == "data"

    @pytest.mark.anyio
    async def test_exception_returns_default_none(self, caplog: pytest.LogCaptureFixture) -> None:
        @graceful_degrade()
        async def fetch() -> str:
            raise ConnectionError("timeout")

        with caplog.at_level(logging.WARNING):
            result = await fetch()

        assert result is None
        assert "fetch failed" in caplog.text
        assert "timeout" in caplog.text

    @pytest.mark.anyio
    async def test_exception_returns_custom_default(self) -> None:
        @graceful_degrade(default=[])
        async def fetch() -> list[str]:
            raise IOError("bad")

        assert await fetch() == []

    @pytest.mark.anyio
    async def test_preserves_coroutine_function_flag(self) -> None:
        import inspect

        @graceful_degrade()
        async def coro() -> None:
            pass

        assert inspect.iscoroutinefunction(coro)

    @pytest.mark.anyio
    async def test_async_log_includes_call_args(self, caplog: pytest.LogCaptureFixture) -> None:
        @graceful_degrade()
        async def fetch(url: str, retries: int = 3) -> str:
            raise ConnectionError("down")

        with caplog.at_level(logging.WARNING):
            await fetch("https://example.com")

        assert "url='https://example.com'" in caplog.text
        assert "retries=3" in caplog.text
