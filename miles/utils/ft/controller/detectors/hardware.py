import polars as pl

from miles.utils.ft.metric_names import (
    DCGM_FI_DEV_GPU_TEMP,
    GPU_AVAILABLE,
    NODE_FILESYSTEM_AVAIL_BYTES,
    NODE_NETWORK_UP,
    XID_CODE_RECENT,
)
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.models import Decision, NodeFault

_CRITICAL_XID_CODES: frozenset[int] = frozenset({48, 62, 64, 79})
_DISK_AVAILABLE_THRESHOLD_BYTES: float = 1e9  # 1 GB


class HighConfidenceHardwareDetector(BaseFaultDetector):
    def __init__(
        self,
        critical_xid_codes: frozenset[int] = _CRITICAL_XID_CODES,
        disk_available_threshold_bytes: float = _DISK_AVAILABLE_THRESHOLD_BYTES,
    ) -> None:
        self._critical_xid_codes = critical_xid_codes
        self._disk_available_threshold_bytes = disk_available_threshold_bytes

    def evaluate(self, ctx: DetectorContext) -> Decision:
        faults: list[NodeFault] = [
            *self._check_gpu_lost(ctx.metric_store),
            *self._check_critical_xid(ctx.metric_store),
            *self._check_disk_fault(ctx.metric_store),
            *self._check_majority_nic_down(ctx.metric_store),
        ]

        return Decision.from_node_faults(
            faults,
            fallback_reason="no high-confidence hardware faults",
        )

    def _check_gpu_lost(self, metric_store: MetricStoreProtocol) -> list[NodeFault]:
        df = metric_store.query_latest(GPU_AVAILABLE)
        if df.is_empty():
            return []

        bad = df.filter(pl.col("value") == 0.0)
        if bad.is_empty():
            return []

        return [
            NodeFault(node_id=node_id, reason=f"GPU unavailable on {node_id}")
            for node_id in bad["node_id"].unique().to_list()
        ]

    def _check_critical_xid(self, metric_store: MetricStoreProtocol) -> list[NodeFault]:
        df = metric_store.query_latest(XID_CODE_RECENT)
        if df.is_empty():
            return []

        faults: list[NodeFault] = []
        for row in df.iter_rows(named=True):
            xid_code = int(row.get("xid", -1))
            if xid_code in self._critical_xid_codes:
                node_id = row["node_id"]
                faults.append(NodeFault(node_id=node_id, reason=f"critical XID {xid_code} on {node_id}"))
        return faults

    def _check_disk_fault(self, metric_store: MetricStoreProtocol) -> list[NodeFault]:
        df = metric_store.query_latest(NODE_FILESYSTEM_AVAIL_BYTES)
        if df.is_empty():
            return []

        bad = df.filter(pl.col("value") < self._disk_available_threshold_bytes)
        if bad.is_empty():
            return []

        return [
            NodeFault(
                node_id=row["node_id"],
                reason=f"disk space low on {row['node_id']} ({row['value']:.0f} bytes)",
            )
            for row in bad.iter_rows(named=True)
        ]

    def _check_majority_nic_down(self, metric_store: MetricStoreProtocol) -> list[NodeFault]:
        df = metric_store.query_latest(NODE_NETWORK_UP)
        if df.is_empty():
            return []

        node_stats: dict[str, tuple[int, int]] = {}
        for row in df.iter_rows(named=True):
            node_id = row["node_id"]
            down_count, total_count = node_stats.get(node_id, (0, 0))
            total_count += 1
            if row["value"] == 0.0:
                down_count += 1
            node_stats[node_id] = (down_count, total_count)

        return [
            NodeFault(node_id=node_id, reason=f"majority NIC down on {node_id} ({down_count}/{total_count})")
            for node_id, (down_count, total_count) in node_stats.items()
            if total_count > 0 and down_count > total_count / 2
        ]
