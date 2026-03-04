from __future__ import annotations

import polars as pl

from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.metric_names import (
    GPU_AVAILABLE,
    NODE_FILESYSTEM_AVAIL_BYTES,
    NODE_NETWORK_UP,
    XID_CODE_RECENT,
)

_CRITICAL_XID_CODES: frozenset[int] = frozenset({48, 62, 64, 79})
_DISK_AVAILABLE_THRESHOLD_BYTES: float = 1e9


class AlertChecker:
    def __init__(self, metric_store: MetricStoreProtocol) -> None:
        self._metric_store = metric_store

    def check_alerts(self) -> tuple[list[str], list[str]]:
        """Return (sorted bad_node_ids, reasons)."""
        bad_nodes: set[str] = set()
        reasons: list[str] = []

        self._query_gpu_lost(bad_nodes, reasons)
        self._query_critical_xid(bad_nodes, reasons)
        self._query_disk_fault(bad_nodes, reasons)
        self._query_majority_nic_down(bad_nodes, reasons)

        return sorted(bad_nodes), reasons

    def _query_gpu_lost(
        self, bad_nodes: set[str], reasons: list[str],
    ) -> None:
        df = self._metric_store.query_latest(GPU_AVAILABLE)
        if df.is_empty():
            return
        df_bad = df.filter(pl.col("value") == 0.0)
        for node_id in df_bad["node_id"].unique().to_list():
            bad_nodes.add(node_id)
            reasons.append(f"GPU unavailable on {node_id}")

    def _query_critical_xid(
        self, bad_nodes: set[str], reasons: list[str],
    ) -> None:
        df = self._metric_store.query_latest(XID_CODE_RECENT)
        if df.is_empty():
            return
        for row in df.iter_rows(named=True):
            xid_code = int(row.get("xid", -1))
            if xid_code in _CRITICAL_XID_CODES:
                node_id = row["node_id"]
                bad_nodes.add(node_id)
                reasons.append(f"critical XID {xid_code} on {node_id}")

    def _query_disk_fault(
        self, bad_nodes: set[str], reasons: list[str],
    ) -> None:
        df = self._metric_store.query_latest(NODE_FILESYSTEM_AVAIL_BYTES)
        if df.is_empty():
            return
        df_bad = df.filter(pl.col("value") < _DISK_AVAILABLE_THRESHOLD_BYTES)
        for row in df_bad.iter_rows(named=True):
            node_id = row["node_id"]
            bad_nodes.add(node_id)
            reasons.append(f"disk space low on {node_id}")

    def _query_majority_nic_down(
        self, bad_nodes: set[str], reasons: list[str],
    ) -> None:
        df = self._metric_store.query_latest(NODE_NETWORK_UP)
        if df.is_empty():
            return

        node_stats: dict[str, tuple[int, int]] = {}
        for row in df.iter_rows(named=True):
            node_id = row["node_id"]
            down_count, total_count = node_stats.get(node_id, (0, 0))
            total_count += 1
            if row["value"] == 0.0:
                down_count += 1
            node_stats[node_id] = (down_count, total_count)

        for node_id, (down_count, total_count) in node_stats.items():
            if total_count > 0 and down_count > total_count / 2:
                bad_nodes.add(node_id)
                reasons.append(f"majority NIC down on {node_id} ({down_count}/{total_count})")
