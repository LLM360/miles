"""Standalone NCCL diagnostic smoke test.

Runs intra-machine and inter-machine NCCL bandwidth tests on every GPU
node in the Ray cluster.  Does NOT require training, FT controller, or
any other E2E infrastructure — only a live Ray cluster with GPU nodes
that have nccl-tests binaries installed.

Expected result: all pass (healthy cluster).  Also serves as a cluster
network health canary.
"""

from __future__ import annotations

import pytest

from miles.utils.ft.platform.ray_wrappers.standalone_diagnostic import (
    run_inter_machine_diagnostics,
    run_intra_machine_diagnostics,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(600),
]


async def test_intra_machine_all_pass(ray_cluster: None) -> None:
    """Every GPU node should pass intra-machine all_reduce_perf."""
    results = await run_intra_machine_diagnostics(timeout_seconds=120)
    failed = [r for r in results if not r.passed]
    assert not failed, f"Intra-machine NCCL diagnostic failed on: {[r.node_id for r in failed]}"


async def test_inter_machine_all_pass(ray_cluster: None) -> None:
    """All node pairs should pass inter-machine all_gather_perf."""
    bad_nodes = await run_inter_machine_diagnostics(timeout_seconds=180)
    assert bad_nodes == [], f"Inter-machine NCCL diagnostic identified bad nodes: {bad_nodes}"
