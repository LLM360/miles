from __future__ import annotations

import logging
import os
from pathlib import Path

import miles.utils.ft.models.metric_names as mn
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models.metrics import GaugeSample
from miles.utils.ft.utils.graceful_degrade import graceful_degrade

logger = logging.getLogger(__name__)


class DiskCollector(BaseCollector):
    collect_interval: float = 60.0

    def __init__(self, *, disk_mounts: list[Path] | None = None) -> None:
        self._disk_mounts = disk_mounts or []
        if not self._disk_mounts:
            logger.warning(
                "DiskCollector initialized with no disk_mounts — " "filesystem metrics will not be collected"
            )

    def _collect_sync(self) -> list[GaugeSample]:
        samples: list[GaugeSample] = []
        samples.extend(self._collect_disk_avail())
        samples.extend(self._collect_disk_io_time())
        return samples

    def _collect_disk_avail(self) -> list[GaugeSample]:
        samples: list[GaugeSample] = []
        for mount in self._disk_mounts:
            try:
                stat = os.statvfs(mount)
                available_bytes = stat.f_bavail * stat.f_frsize
                samples.append(
                    GaugeSample(
                        name=mn.NODE_FILESYSTEM_AVAIL_BYTES,
                        labels={"mountpoint": str(mount)},
                        value=float(available_bytes),
                    )
                )
            except Exception:
                logger.warning("Failed to statvfs %s", mount, exc_info=True)
        return samples

    def _collect_disk_io_time(self) -> list[GaugeSample]:
        sys_block = Path("/sys/block")
        if not sys_block.exists():
            return []

        samples: list[GaugeSample] = []
        for device_dir in sorted(sys_block.iterdir()):
            samples.extend(self._collect_single_device_io(device_dir))
        return samples

    @staticmethod
    @graceful_degrade(default=[])
    def _collect_single_device_io(device_dir: Path) -> list[GaugeSample]:
        stat_file = device_dir / "stat"
        text = stat_file.read_text().strip()
        fields = text.split()
        if len(fields) >= 10:
            io_time_ms = int(fields[9])
            return [
                GaugeSample(
                    name=mn.NODE_DISK_IO_TIME_SECONDS_TOTAL,
                    labels={"device": device_dir.name},
                    value=io_time_ms / 1000.0,
                )
            ]
        return []
