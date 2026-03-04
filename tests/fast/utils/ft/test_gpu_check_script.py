"""Unit tests for _gpu_check_script — pynvml + matmul checks."""
from __future__ import annotations

import json
from dataclasses import asdict
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.controller.diagnostics._gpu_check_script import (
    GpuCheckResult,
    _check_matmul,
    _check_nvml,
    _check_single_gpu,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pynvml_mock(
    *,
    device_count: int = 1,
    ecc_uncorrectable: int = 0,
    retired_pages: list[object] | None = None,
    power_state: int = 0,
    remap_info: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> MagicMock:
    mock = MagicMock()
    mock.NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    mock.NVML_VOLATILE_ECC_COUNTER_TYPE = 0
    mock.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR = 0

    mock.nvmlInit.return_value = None
    mock.nvmlShutdown.return_value = None
    mock.nvmlDeviceGetCount.return_value = device_count
    mock.nvmlDeviceGetHandleByIndex.side_effect = lambda i: f"handle-{i}"
    mock.nvmlDeviceGetTotalEccErrors.return_value = ecc_uncorrectable
    mock.nvmlDeviceGetRetiredPages.return_value = retired_pages or []
    mock.nvmlDeviceGetPowerState.return_value = power_state
    mock.nvmlDeviceGetRemappedRows.return_value = remap_info
    return mock


# ---------------------------------------------------------------------------
# _check_nvml tests
# ---------------------------------------------------------------------------

class TestCheckNvml:
    def test_healthy_gpu(self) -> None:
        mock_pynvml = _make_pynvml_mock()
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(gpu_index=0, handle="handle-0")

        assert result["ecc_errors_uncorrectable"] == 0
        assert result["retired_pages_count"] == 0
        assert result["power_state_abnormal"] is False
        assert result["row_remap_failure"] is False

    def test_ecc_uncorrectable_errors(self) -> None:
        mock_pynvml = _make_pynvml_mock(ecc_uncorrectable=5)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(gpu_index=0, handle="handle-0")

        assert result["ecc_errors_uncorrectable"] == 5

    def test_retired_pages(self) -> None:
        mock_pynvml = _make_pynvml_mock(retired_pages=["page1", "page2", "page3"])
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(gpu_index=0, handle="handle-0")

        assert result["retired_pages_count"] == 3

    def test_abnormal_power_state_8(self) -> None:
        mock_pynvml = _make_pynvml_mock(power_state=8)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(gpu_index=0, handle="handle-0")

        assert result["power_state_abnormal"] is True

    def test_abnormal_power_state_15(self) -> None:
        mock_pynvml = _make_pynvml_mock(power_state=15)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(gpu_index=0, handle="handle-0")

        assert result["power_state_abnormal"] is True

    def test_normal_power_state(self) -> None:
        mock_pynvml = _make_pynvml_mock(power_state=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(gpu_index=0, handle="handle-0")

        assert result["power_state_abnormal"] is False

    def test_row_remap_failure(self) -> None:
        mock_pynvml = _make_pynvml_mock(remap_info=(0, 0, 0, 1))
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvml(gpu_index=0, handle="handle-0")

        assert result["row_remap_failure"] is True


# ---------------------------------------------------------------------------
# _check_single_gpu tests
# ---------------------------------------------------------------------------

class TestCheckSingleGpu:
    def test_all_pass(self) -> None:
        mock_pynvml = _make_pynvml_mock()
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
        ):
            result = _check_single_gpu(gpu_index=0, handle="handle-0")

        assert result.passed is True
        assert result.matmul_passed is True
        assert result.details == "all checks passed"

    def test_ecc_failure(self) -> None:
        mock_pynvml = _make_pynvml_mock(ecc_uncorrectable=3)
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
        ):
            result = _check_single_gpu(gpu_index=0, handle="handle-0")

        assert result.passed is False
        assert "ECC" in result.details

    def test_matmul_mismatch(self) -> None:
        mock_pynvml = _make_pynvml_mock()
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=False,
            ),
        ):
            result = _check_single_gpu(gpu_index=0, handle="handle-0")

        assert result.passed is False
        assert result.matmul_passed is False
        assert "matmul" in result.details

    def test_multiple_failures(self) -> None:
        mock_pynvml = _make_pynvml_mock(
            ecc_uncorrectable=1,
            remap_info=(0, 0, 0, 1),
        )
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=False,
            ),
        ):
            result = _check_single_gpu(gpu_index=0, handle="handle-0")

        assert result.passed is False
        assert "ECC" in result.details
        assert "row remap" in result.details
        assert "matmul" in result.details

    def test_retired_pages_failure(self) -> None:
        mock_pynvml = _make_pynvml_mock(retired_pages=["p1"])
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
        ):
            result = _check_single_gpu(gpu_index=0, handle="handle-0")

        assert result.passed is False
        assert "retired" in result.details

    def test_power_state_abnormal_failure(self) -> None:
        mock_pynvml = _make_pynvml_mock(power_state=8)
        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
        ):
            result = _check_single_gpu(gpu_index=0, handle="handle-0")

        assert result.passed is False
        assert "power state" in result.details


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_healthy_output(self) -> None:
        mock_pynvml = _make_pynvml_mock(device_count=2)
        stdout_capture = StringIO()

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
            patch("sys.stdout", stdout_capture),
        ):
            main()

        output = json.loads(stdout_capture.getvalue())
        assert len(output) == 2
        assert output[0]["gpu_index"] == 0
        assert output[0]["passed"] is True
        assert output[1]["gpu_index"] == 1
        assert output[1]["passed"] is True

    def test_main_mixed_results(self) -> None:
        mock_pynvml = _make_pynvml_mock(device_count=2, ecc_uncorrectable=5)
        stdout_capture = StringIO()

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(
                "miles.utils.ft.controller.diagnostics._gpu_check_script._check_matmul",
                return_value=True,
            ),
            patch("sys.stdout", stdout_capture),
        ):
            main()

        output = json.loads(stdout_capture.getvalue())
        assert len(output) == 2
        assert output[0]["passed"] is False
        assert output[0]["ecc_errors_uncorrectable"] == 5

    def test_main_nvml_shutdown_on_error(self) -> None:
        mock_pynvml = _make_pynvml_mock(device_count=1)
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError("GPU error")

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            pytest.raises(RuntimeError, match="GPU error"),
        ):
            main()

        mock_pynvml.nvmlShutdown.assert_called_once()


# ---------------------------------------------------------------------------
# GpuCheckResult dataclass tests
# ---------------------------------------------------------------------------

class TestGpuCheckResult:
    def test_serialization(self) -> None:
        result = GpuCheckResult(
            gpu_index=0,
            passed=True,
            ecc_errors_uncorrectable=0,
            retired_pages_count=0,
            power_state_abnormal=False,
            row_remap_failure=False,
            matmul_passed=True,
            details="all checks passed",
        )
        d = asdict(result)
        assert d["gpu_index"] == 0
        assert d["passed"] is True

        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored == d
