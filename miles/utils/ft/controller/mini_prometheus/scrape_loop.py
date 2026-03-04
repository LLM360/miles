from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from miles.utils.ft.controller.mini_prometheus.scraper import parse_prometheus_text
from miles.utils.ft.models import MetricSample

logger = logging.getLogger(__name__)


class _IngestTarget(Protocol):
    def ingest_samples(
        self,
        target_id: str,
        samples: list[MetricSample],
    ) -> None: ...


class ScrapeLoop:
    def __init__(self, store: _IngestTarget, scrape_interval_seconds: float) -> None:
        self._store = store
        self._scrape_interval_seconds = scrape_interval_seconds
        self._targets: dict[str, str] = {}
        self._running = False

    def add_target(self, target_id: str, address: str) -> None:
        self._targets[target_id] = address

    def remove_target(self, target_id: str) -> None:
        self._targets.pop(target_id, None)

    @property
    def targets(self) -> dict[str, str]:
        return self._targets

    async def scrape_once(self) -> None:
        import httpx

        targets = list(self._targets.items())
        if not targets:
            return

        async with httpx.AsyncClient(timeout=10.0) as client:

            async def _scrape_target(target_id: str, address: str) -> None:
                try:
                    response = await client.get(f"{address}/metrics")
                    response.raise_for_status()
                    samples = parse_prometheus_text(response.text)
                    self._store.ingest_samples(target_id=target_id, samples=samples)
                except Exception:
                    logger.warning(
                        "Failed to scrape target %s at %s",
                        target_id,
                        address,
                        exc_info=True,
                    )

            await asyncio.gather(*(
                _scrape_target(target_id, address)
                for target_id, address in targets
            ))

    async def start(self) -> None:
        self._running = True
        while self._running:
            await self.scrape_once()
            await asyncio.sleep(self._scrape_interval_seconds)

    async def stop(self) -> None:
        self._running = False
