from __future__ import annotations

import asyncio
import fnmatch
import logging
from pathlib import Path

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models import CollectorOutput, MetricSample

logger = logging.getLogger(__name__)

_DEFAULT_INCLUDE_PATTERNS = ["ib*", "eth*", "en*"]
_DEFAULT_EXCLUDE_PATTERNS = ["lo", "docker*", "veth*"]

_STAT_FILES = ["rx_errors", "tx_errors", "rx_dropped", "tx_dropped"]


class NetworkCollector(BaseCollector):
    collect_interval: float = 30.0

    def __init__(
        self,
        sysfs_net_path: Path = Path("/sys/class/net"),
        interface_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self._sysfs_net_path = sysfs_net_path
        self._include_patterns = interface_patterns or _DEFAULT_INCLUDE_PATTERNS
        self._exclude_patterns = exclude_patterns or _DEFAULT_EXCLUDE_PATTERNS

    async def collect(self) -> CollectorOutput:
        metrics = await asyncio.to_thread(self._collect_sync)
        return CollectorOutput(metrics=metrics)

    def _collect_sync(self) -> list[MetricSample]:
        if not self._sysfs_net_path.exists():
            logger.warning("sysfs net path %s does not exist", self._sysfs_net_path)
            return []

        samples: list[MetricSample] = []

        for iface_dir in sorted(self._sysfs_net_path.iterdir()):
            iface_name = iface_dir.name
            if not self._should_collect(iface_name):
                continue

            iface_label = {"interface": iface_name}
            self._collect_operstate(iface_dir, iface_label, samples)
            self._collect_statistics(iface_dir, iface_label, samples)

        return samples

    def _should_collect(self, iface_name: str) -> bool:
        for pattern in self._exclude_patterns:
            if fnmatch.fnmatch(iface_name, pattern):
                return False

        for pattern in self._include_patterns:
            if fnmatch.fnmatch(iface_name, pattern):
                return True

        return False

    @staticmethod
    def _collect_operstate(
        iface_dir: Path,
        iface_label: dict[str, str],
        samples: list[MetricSample],
    ) -> None:
        operstate_file = iface_dir / "operstate"
        try:
            state = operstate_file.read_text().strip().lower()
            value = 1.0 if state == "up" else 0.0
            samples.append(MetricSample(name="nic_up", labels=iface_label, value=value))
        except Exception:
            logger.warning("Failed to read operstate for %s", iface_label["interface"])

    @staticmethod
    def _collect_statistics(
        iface_dir: Path,
        iface_label: dict[str, str],
        samples: list[MetricSample],
    ) -> None:
        stats_dir = iface_dir / "statistics"
        if not stats_dir.exists():
            return

        for stat_name in _STAT_FILES:
            stat_file = stats_dir / stat_name
            try:
                value = int(stat_file.read_text().strip())
                metric_name = f"nic_{stat_name}"
                samples.append(MetricSample(name=metric_name, labels=iface_label, value=float(value)))
            except Exception:
                logger.warning(
                    "Failed to read %s for %s",
                    stat_name,
                    iface_label["interface"],
                )
