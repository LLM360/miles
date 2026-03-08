"""Tests for miles.utils.ft.utils.subprocess."""

from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.utils.subprocess import run_subprocess_with_timeout


class TestRunSubprocessWithTimeout:
    def test_successful_command_returns_stdout_stderr_returncode(self) -> None:
        stdout, stderr, rc = asyncio.run(
            run_subprocess_with_timeout(
                cmd=["echo", "hello"],
                timeout_seconds=5,
            )
        )

        assert b"hello" in stdout
        assert rc == 0

    def test_nonzero_exit_code_returned(self) -> None:
        stdout, stderr, rc = asyncio.run(
            run_subprocess_with_timeout(
                cmd=["bash", "-c", "exit 42"],
                timeout_seconds=5,
            )
        )

        assert rc == 42

    def test_stderr_captured(self) -> None:
        stdout, stderr, rc = asyncio.run(
            run_subprocess_with_timeout(
                cmd=["bash", "-c", "echo err_msg >&2"],
                timeout_seconds=5,
            )
        )

        assert b"err_msg" in stderr

    def test_timeout_raises_timeout_error(self) -> None:
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(
                run_subprocess_with_timeout(
                    cmd=["sleep", "60"],
                    timeout_seconds=1,
                )
            )

    def test_command_not_found_raises_os_error(self) -> None:
        with pytest.raises(FileNotFoundError):
            asyncio.run(
                run_subprocess_with_timeout(
                    cmd=["nonexistent_binary_xyz_123"],
                    timeout_seconds=5,
                )
            )

    def test_env_parameter_passed_to_subprocess(self) -> None:
        stdout, stderr, rc = asyncio.run(
            run_subprocess_with_timeout(
                cmd=["bash", "-c", "echo $MY_TEST_VAR"],
                timeout_seconds=5,
                env={"MY_TEST_VAR": "test_value", "PATH": "/usr/bin:/bin"},
            )
        )

        assert b"test_value" in stdout
        assert rc == 0
