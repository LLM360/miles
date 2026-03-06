from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import polars as pl

from miles.utils.ft.controller.metrics.aggregation_mixin import RangeAggregationMixin
from miles.utils.ft.controller.metrics.mini_prometheus.query import SeriesKey, TimeSeriesSample
from miles.utils.ft.protocols.metrics import MetricStoreProtocol, ScrapeTargetManagerProtocol
from miles.utils.ft.controller.metrics.mini_prometheus.query import query_latest as _query_latest
from miles.utils.ft.controller.metrics.mini_prometheus.query import query_range as _query_range
from miles.utils.ft.controller.metrics.mini_prometheus.query import range_aggregate as _range_aggregate
from miles.utils.ft.controller.metrics.mini_prometheus.scrape_loop import ScrapeLoop
from miles.utils.ft.models.metrics import MetricSample


@dataclass
class MiniPrometheusConfig:
    scrape_interval: timedelta = field(default_factory=lambda: timedelta(seconds=10))
    retention: timedelta = field(default_factory=lambda: timedelta(minutes=60))


class MiniPrometheus(MetricStoreProtocol, ScrapeTargetManagerProtocol, RangeAggregationMixin):
    def __init__(self, config: MiniPrometheusConfig | None = None) -> None:
        self._config = config or MiniPrometheusConfig()
        self._series: dict[SeriesKey, deque[TimeSeriesSample]] = {}
        self._label_maps: dict[SeriesKey, dict[str, str]] = {}
        self._name_index: dict[str, set[SeriesKey]] = {}
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
    # Data ingestion
    # -------------------------------------------------------------------

    def ingest_samples(
        self,
        target_id: str,
        samples: list[MetricSample],
        timestamp: datetime | None = None,
    ) -> None:
        ts = timestamp or datetime.now(timezone.utc)
        for sample in samples:
            labels = dict(sample.labels)
            labels.setdefault("node_id", target_id)
            key: SeriesKey = (sample.name, frozenset(labels.items()))

            if key not in self._series:
                self._series[key] = deque()
                self._label_maps[key] = labels
                self._name_index.setdefault(sample.name, set()).add(key)

            self._series[key].append(TimeSeriesSample(timestamp=ts, value=sample.value))

        self._maybe_evict()

    # -------------------------------------------------------------------
    # Query API (MetricStoreProtocol)
    # -------------------------------------------------------------------

    def query_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return _query_latest(self._series, self._label_maps, self._name_index, metric_name, label_filters)

    def query_range(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        return _query_range(self._series, self._label_maps, self._name_index, metric_name, window, label_filters)

    def _dispatch_range_function(
        self,
        func_name: str,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None,
    ) -> pl.DataFrame:
        return _range_aggregate(
            self._series,
            self._label_maps,
            self._name_index,
            func_name,
            metric_name,
            window,
            label_filters,
        )

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
