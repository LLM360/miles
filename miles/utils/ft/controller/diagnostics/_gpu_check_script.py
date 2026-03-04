"""Standalone GPU health-check script, executed as a subprocess.

Runs pynvml extended checks and matmul correctness verification on all
visible GPUs, then prints a JSON array of per-GPU results to stdout.

Usage::

    python -m miles.utils.ft.controller.diagnostics._gpu_check_script

The caller (GpuDiagnostic) launches this via asyncio.create_subprocess_exec
so that pynvml init/shutdown and torch computation happen in an isolated
process and never block the NodeAgent event loop (see 3-discussions.md #48).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass


@dataclass
class GpuCheckResult:
    gpu_index: int
    passed: bool
    ecc_errors_uncorrectable: int
    retired_pages_count: int
    power_state_abnormal: bool
    row_remap_failure: bool
    matmul_passed: bool
    details: str


_ABNORMAL_POWER_STATES = frozenset({8, 15})


def _check_nvml(gpu_index: int, handle: object) -> dict[str, object]:
    """Run pynvml extended checks on a single GPU handle.

    Returns a dict with keys consumed by _build_result.
    """
    import pynvml

    ecc_uncorrectable = pynvml.nvmlDeviceGetTotalEccErrors(
        handle,
        pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
        pynvml.NVML_VOLATILE_ECC_COUNTER_TYPE,
    )

    retired_double_bit = pynvml.nvmlDeviceGetRetiredPages(
        handle, pynvml.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR,
    )
    retired_count = len(retired_double_bit)

    power_state: int = pynvml.nvmlDeviceGetPowerState(handle)
    power_state_abnormal = power_state in _ABNORMAL_POWER_STATES

    remap_info = pynvml.nvmlDeviceGetRemappedRows(handle)
    row_remap_failure = bool(remap_info[3])

    return {
        "ecc_errors_uncorrectable": ecc_uncorrectable,
        "retired_pages_count": retired_count,
        "power_state_abnormal": power_state_abnormal,
        "row_remap_failure": row_remap_failure,
    }


def _check_matmul(gpu_index: int) -> bool:
    """Run deterministic matmul and compare against CPU reference.

    Returns True if the result matches within tolerance.
    """
    import torch

    generator = torch.Generator(device="cpu").manual_seed(42)

    size = 1024
    a_fp32 = torch.randn(size, size, generator=generator)
    b_fp32 = torch.randn(size, size, generator=generator)

    expected = (a_fp32.half() @ b_fp32.half()).float()

    device = torch.device(f"cuda:{gpu_index}")
    a_gpu = a_fp32.half().to(device)
    b_gpu = b_fp32.half().to(device)
    actual = (a_gpu @ b_gpu).float().cpu()

    return bool(torch.allclose(actual, expected, atol=1e-2, rtol=1e-3))


def _check_single_gpu(gpu_index: int, handle: object) -> GpuCheckResult:
    """Run all checks on one GPU and produce a GpuCheckResult."""
    failures: list[str] = []

    nvml_info = _check_nvml(gpu_index, handle)

    if nvml_info["ecc_errors_uncorrectable"] > 0:
        failures.append(
            f"uncorrectable ECC errors: {nvml_info['ecc_errors_uncorrectable']}"
        )
    if nvml_info["retired_pages_count"] > 0:
        failures.append(
            f"retired pages: {nvml_info['retired_pages_count']}"
        )
    if nvml_info["power_state_abnormal"]:
        failures.append("abnormal power state")
    if nvml_info["row_remap_failure"]:
        failures.append("row remap failure")

    matmul_passed = _check_matmul(gpu_index)
    if not matmul_passed:
        failures.append("matmul mismatch")

    passed = len(failures) == 0
    details = "; ".join(failures) if failures else "all checks passed"

    return GpuCheckResult(
        gpu_index=gpu_index,
        passed=passed,
        ecc_errors_uncorrectable=nvml_info["ecc_errors_uncorrectable"],
        retired_pages_count=nvml_info["retired_pages_count"],
        power_state_abnormal=nvml_info["power_state_abnormal"],
        row_remap_failure=nvml_info["row_remap_failure"],
        matmul_passed=matmul_passed,
        details=details,
    )


def main() -> None:
    import pynvml

    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        results: list[GpuCheckResult] = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            result = _check_single_gpu(i, handle)
            results.append(result)
    finally:
        pynvml.nvmlShutdown()

    json.dump([asdict(r) for r in results], sys.stdout)


if __name__ == "__main__":
    main()
