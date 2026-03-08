from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from miles.utils.ft.agents.diagnostics.executors.collector_based import CollectorBasedNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.gpu import GpuNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.intra_machine import IntraMachineNodeExecutor
from miles.utils.ft.platform.node_agent_factory import build_local_diagnostics


class TestBuildLocalDiagnostics:
    def test_returns_five_executors_with_expected_types(self) -> None:
        diagnostics = build_local_diagnostics(num_gpus=4)
        types = [d.diagnostic_type for d in diagnostics]
        assert types == ["gpu", "intra_machine", "disk", "network", "xid"]

    def test_gpu_and_intra_machine_are_dedicated_executors(self) -> None:
        diagnostics = build_local_diagnostics(num_gpus=4)
        assert isinstance(diagnostics[0], GpuNodeExecutor)
        assert isinstance(diagnostics[1], IntraMachineNodeExecutor)

    def test_collector_based_executors_for_disk_network_xid(self) -> None:
        diagnostics = build_local_diagnostics(num_gpus=4)
        for diag in diagnostics[2:]:
            assert isinstance(diag, CollectorBasedNodeExecutor)

    def test_custom_disk_mounts_passed_through(self) -> None:
        mounts = [Path("/data"), Path("/scratch")]
        diagnostics = build_local_diagnostics(num_gpus=4, disk_mounts=mounts)
        disk_executor = diagnostics[2]
        assert isinstance(disk_executor, CollectorBasedNodeExecutor)
        assert disk_executor._collector._disk_mounts == mounts

    def test_xid_since_passed_through(self) -> None:
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        diagnostics = build_local_diagnostics(num_gpus=4, xid_since=since)
        xid_executor = diagnostics[4]
        assert isinstance(xid_executor, CollectorBasedNodeExecutor)
        assert xid_executor._collector._since == since

    def test_defaults_produce_valid_executors(self) -> None:
        diagnostics = build_local_diagnostics()
        assert len(diagnostics) == 5
        for d in diagnostics:
            assert hasattr(d, "diagnostic_type")
            assert hasattr(d, "run")
