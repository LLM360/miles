from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from miles.utils.ft.agents.collectors.kernel_log_reader import (
    DmesgSubprocessReader,
    KmsgFileReader,
)


class TestKmsgFileReader:
    def test_read_after_close_returns_empty(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_bytes(b"init line\n")

        reader = KmsgFileReader(kmsg_path=kmsg_file)
        reader.close()

        assert reader.read_new_lines() == []

    def test_close_idempotent(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_bytes(b"")

        reader = KmsgFileReader(kmsg_path=kmsg_file)
        reader.close()
        reader.close()
        assert reader._file_handle is None

    def test_close_handles_os_error(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_bytes(b"")

        reader = KmsgFileReader(kmsg_path=kmsg_file)
        os.close(reader._file_handle)
        reader.close()
        assert reader._file_handle is None

    def test_blocking_io_error_breaks_loop(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_bytes(b"")

        reader = KmsgFileReader(kmsg_path=kmsg_file)

        with patch("os.read", side_effect=[b"line1\nline2\n", BlockingIOError]):
            lines = reader.read_new_lines()

        assert lines == ["line1", "line2"]
        reader.close()

    def test_empty_read_returns_empty(self, tmp_path: Path) -> None:
        kmsg_file = tmp_path / "kmsg"
        kmsg_file.write_bytes(b"")

        reader = KmsgFileReader(kmsg_path=kmsg_file)

        with patch("os.read", return_value=b""):
            lines = reader.read_new_lines()

        assert lines == []
        reader.close()


class TestDmesgSubprocessReader:
    def test_returns_lines_on_success(self) -> None:
        reader = DmesgSubprocessReader()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "kernel: message 1\nkernel: message 2\n"

        with patch("subprocess.run", return_value=mock_result):
            lines = reader.read_new_lines()

        assert lines == ["kernel: message 1", "kernel: message 2"]

    def test_returns_empty_on_nonzero_returncode(self) -> None:
        reader = DmesgSubprocessReader()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            lines = reader.read_new_lines()

        assert lines == []

    def test_returns_empty_on_timeout(self) -> None:
        reader = DmesgSubprocessReader()

        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="dmesg", timeout=5)):
            lines = reader.read_new_lines()

        assert lines == []

    def test_returns_empty_on_empty_stdout(self) -> None:
        reader = DmesgSubprocessReader()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            lines = reader.read_new_lines()

        assert lines == []

    def test_close_is_noop(self) -> None:
        reader = DmesgSubprocessReader()
        reader.close()
