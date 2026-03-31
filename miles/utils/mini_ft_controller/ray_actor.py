from __future__ import annotations

import logging
from typing import Any

import httpx
import ray

from miles.utils.mini_ft_controller.controller import CellSnapshot, MiniFTController

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0.001)
class MiniFTControllerActor:
    def __init__(
        self,
        *,
        control_server_url: str,
        poll_interval: float,
        resume_delay: float,
        max_consecutive_failures: int,
    ) -> None:
        self._url = control_server_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._url, timeout=30.0)
        self._controller = MiniFTController(
            get_cells=self._get_cells,
            suspend_cell=self._suspend_cell,
            resume_cell=self._resume_cell,
            poll_interval=poll_interval,
            resume_delay=resume_delay,
            max_consecutive_failures=max_consecutive_failures,
        )

    async def run(self) -> None:
        try:
            await self._controller.run()
        finally:
            await self._client.aclose()

    def request_stop(self) -> None:
        self._controller.request_stop()

    async def _get_cells(self) -> list[CellSnapshot]:
        resp = await self._client.get("/api/v1/cells")
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

        snapshots: list[CellSnapshot] = []
        for item in data["items"]:
            name: str = item["metadata"]["name"]
            healthy_status: str = "Unknown"
            healthy_reason: str | None = None

            for condition in item.get("status", {}).get("conditions", []):
                if condition["type"] == "Healthy":
                    healthy_status = condition["status"]
                    healthy_reason = condition.get("reason")
                    break

            snapshots.append(CellSnapshot(
                name=name,
                healthy_status=healthy_status,
                healthy_reason=healthy_reason,
            ))

        return snapshots

    async def _suspend_cell(self, name: str) -> None:
        await self._patch_cell_suspend(name=name, suspend=True)

    async def _resume_cell(self, name: str) -> None:
        await self._patch_cell_suspend(name=name, suspend=False)

    async def _patch_cell_suspend(self, *, name: str, suspend: bool) -> None:
        resp = await self._client.patch(
            f"/api/v1/cells/{name}",
            json={"spec": {"suspend": suspend}},
        )
        resp.raise_for_status()


def maybe_start_mini_ft_controller(args: Any) -> None:
    if not args.mini_ft_controller_enabled:
        return

    controller = MiniFTControllerActor.remote(
        control_server_url=f"http://127.0.0.1:{args.control_server_port}",
        poll_interval=args.mini_ft_controller_poll_interval,
        resume_delay=args.mini_ft_controller_resume_delay,
        max_consecutive_failures=args.mini_ft_controller_max_consecutive_failures,
    )
    controller.run.remote()  # fire-and-forget
    logger.info("Started MiniFTControllerActor")
