from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from miles.utils.ft.agents.types import MetricSample
from miles.utils.ft.controller.metrics.mini_prometheus.in_memory_store import InMemoryMetricStore
from miles.utils.ft.controller.metrics.mini_prometheus.query import SeriesKey
from miles.utils.ft.controller.metrics.mini_prometheus.scrape_loop import ScrapeLoop
from miles.utils.ft.controller.types import MetricStoreProtocol, ScrapeTargetManagerProtocol


@dataclass
class MiniPrometheusConfig:
    scrape_interval: timedelta = field(default_factory=lambda: timedelta(seconds=10))
    retention: timedelta = field(default_factory=lambda: timedelta(minutes=60))


class MiniPrometheus(InMemoryMetricStore, MetricStoreProtocol, ScrapeTargetManagerProtocol):
    def __init__(self, config: MiniPrometheusConfig | None = None) -> None:
        super().__init__()
        self._config = config or MiniPrometheusConfig()
        self._last_eviction_time: datetime | None = None

        self._scrape_loop = ScrapeLoop(
            store=self,
            scrape_interval_seconds=self._config.scrape_interval.total_seconds(),
        )

    # -------------------------------------------------------------------
    # Scrape target management (delegated to ScrapeLoop)
    # -------------------------------------------------------------------

    def add_scrape_target(self, target_id: str, address: str) -> None:
        self._scrape_loop.add_target(target_id=target_id, address=address)

    def remove_scrape_target(self, target_id: str) -> None:
        self._scrape_loop.remove_target(target_id)

    # -------------------------------------------------------------------
    # Scrape lifecycle (delegated to ScrapeLoop)
    # -------------------------------------------------------------------

    async def scrape_once(self) -> None:
        await self._scrape_loop.scrape_once()

    async def start(self) -> None:
        await self._scrape_loop.start()

    async def stop(self) -> None:
        await self._scrape_loop.stop()

    # -------------------------------------------------------------------
    # Data ingestion (extends base with eviction)
    # -------------------------------------------------------------------

    def ingest_samples(
        self,
        target_id: str,
        samples: list[MetricSample],
        timestamp: datetime | None = None,
    ) -> None:
        super().ingest_samples(target_id, samples, timestamp)
        self._maybe_evict()

    @property
    def _scrape_targets(self) -> dict[str, str]:
        return self._scrape_loop.targets

    # -------------------------------------------------------------------
    # Internal: eviction
    # -------------------------------------------------------------------

    def _maybe_evict(self) -> None:
        now = datetime.now(timezone.utc)
        evict_interval = self._config.retention / 10
        if self._last_eviction_time is not None and now - self._last_eviction_time < evict_interval:
            return
        self._last_eviction_time = now
        self._evict_expired()

    def _evict_expired(self) -> None:
        cutoff = datetime.now(timezone.utc) - self._config.retention
        empty_keys: list[SeriesKey] = []

        for key, samples in self._series.items():
            while samples and samples[0].timestamp < cutoff:
                samples.popleft()
            if not samples:
                empty_keys.append(key)

        for key in empty_keys:
            metric_name, _ = key
            del self._series[key]
            self._label_maps.pop(key, None)
            index_set = self._name_index.get(metric_name)
            if index_set is not None:
                index_set.discard(key)
                if not index_set:
                    del self._name_index[metric_name]
