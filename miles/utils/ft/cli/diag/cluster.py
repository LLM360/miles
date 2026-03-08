from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated

import typer

from miles.utils.ft.cli.diag.output import exit_with_results, print_results
from miles.utils.ft.models.diagnostics import DiagnosticResult

logger = logging.getLogger(__name__)

ClusterCheckFn = Callable[[int], Awaitable[list[DiagnosticResult]]]


def _build_cluster_registry() -> dict[str, ClusterCheckFn]:
    from miles.utils.ft.platform.ray_wrappers.standalone_diagnostic import (
        run_gpu_diagnostics,
        run_inter_machine_diagnostics,
        run_intra_machine_diagnostics,
    )

    async def _gpu_check(timeout: int) -> list[DiagnosticResult]:
        results, outlier_ids = await run_gpu_diagnostics(timeout_seconds=timeout)
        if outlier_ids:
            results.append(
                DiagnosticResult.fail_result(
                    diagnostic_type="gpu_hash_comparison",
                    node_id="cluster",
                    details=f"outlier nodes: {', '.join(outlier_ids)}",
                )
            )
        return results

    async def _intra_machine_check(timeout: int) -> list[DiagnosticResult]:
        return await run_intra_machine_diagnostics(timeout_seconds=timeout)

    async def _inter_machine_check(timeout: int) -> list[DiagnosticResult]:
        bad_nodes = await run_inter_machine_diagnostics(timeout_seconds=timeout)
        if bad_nodes:
            return [
                DiagnosticResult.fail_result(
                    diagnostic_type="inter_machine",
                    node_id="cluster",
                    details=f"bad nodes: {', '.join(bad_nodes)}",
                )
            ]
        return [
            DiagnosticResult.pass_result(
                diagnostic_type="inter_machine",
                node_id="cluster",
                details="all inter-machine NCCL checks passed",
            )
        ]

    return {
        "gpu": _gpu_check,
        "intra_machine": _intra_machine_check,
        "inter_machine": _inter_machine_check,
    }


async def _run_cluster_checks(
    registry: dict[str, ClusterCheckFn],
    checks: list[str],
    timeout: int,
) -> list[DiagnosticResult]:
    all_results: list[DiagnosticResult] = []
    for name in checks:
        try:
            all_results.extend(await registry[name](timeout))
        except Exception:
            logger.error("cluster check %s failed with exception", name, exc_info=True)
            all_results.append(
                DiagnosticResult.fail_result(
                    diagnostic_type=name,
                    node_id="cluster",
                    details="exception during check (see logs)",
                )
            )
    return all_results


def cluster(
    checks: Annotated[list[str] | None, typer.Argument(help="Checks to run (default: all)")] = None,
    ray_address: Annotated[str, typer.Option(help="Ray cluster address")] = "auto",
    timeout: Annotated[int, typer.Option(help="Per-check timeout in seconds")] = 180,
    json_output: Annotated[bool, typer.Option("--json", help="Output results as JSON")] = False,
) -> None:
    """Run diagnostic checks across a Ray cluster."""
    import ray

    registry = _build_cluster_registry()
    selected = checks or list(registry.keys())
    unknown = set(selected) - set(registry.keys())
    if unknown:
        typer.echo(f"Unknown checks: {', '.join(sorted(unknown))}", err=True)
        raise typer.Exit(code=1)

    ray.init(address=ray_address)

    try:
        results = asyncio.run(_run_cluster_checks(registry, selected, timeout))
    finally:
        ray.shutdown()

    print_results(results, json_output=json_output, node_id="cluster")
    exit_with_results(results)
