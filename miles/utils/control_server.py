from __future__ import annotations

import abc
import asyncio
import logging
import threading
from typing import Literal

import ray
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from miles.ray.train.group import RayTrainGroup

logger = logging.getLogger(__name__)


def start_control_server(actor_model: RayTrainGroup, rollout_manager: object, port: int) -> None:
    registry = _CellRegistry()

    for i in range(len(actor_model._cells)):
        registry.register(_ActorCellHandle(group=actor_model, cell_index=i))

    num_rollout_cells = ray.get(rollout_manager.get_cell_count.remote())
    for i in range(num_rollout_cells):
        registry.register(
            _RolloutCellHandle(
                rollout_manager=rollout_manager,
                cell_id=f"rollout-{i}",
            )
        )

    _start_control_server_raw(registry=registry, port=port)


def _start_control_server_raw(registry: _CellRegistry, port: int) -> None:
    app = _create_control_app(registry)

    def _run() -> None:
        uvicorn.run(app, host="0.0.0.0", port=port)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("Control server started on port %d", port)


class _CellInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_id: str
    cell_type: Literal["actor", "rollout"]
    status: Literal["running", "stopped", "pending", "errored"]
    node_ids: list[str]


class _StopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int = 30


class _OkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


def _create_control_app(registry: _CellRegistry) -> FastAPI:
    app = FastAPI()

    # -------------------------- APIs ------------------------------

    @app.get("/health")
    async def health() -> _OkResponse:
        return _OkResponse()

    @app.get("/cells")
    async def get_cells() -> list[_CellInfo]:
        handles = registry.get_all()

        async def _fetch(handle: _CellHandle) -> _CellInfo:
            status, node_ids = await asyncio.gather(handle.get_status(), handle.get_node_ids())
            return _CellInfo(
                cell_id=handle.cell_id,
                cell_type=handle.cell_type,
                status=status,
                node_ids=node_ids,
            )

        return list(await asyncio.gather(*(_fetch(h) for h in handles)))

    @app.post("/cells/{cell_id}/stop")
    async def stop_cell(cell_id: str, body: _StopRequest | None = None) -> _OkResponse:
        if body is None:
            body = _StopRequest()
        handle = _get_handle(cell_id)
        return await _call_handle(cell_id, "stop", handle.stop(timeout_seconds=body.timeout_seconds))

    @app.post("/cells/{cell_id}/start")
    async def start_cell(cell_id: str) -> _OkResponse:
        handle = _get_handle(cell_id)
        return await _call_handle(cell_id, "start", handle.start())

    # -------------------------- utils ------------------------------

    def _get_handle(cell_id: str) -> _CellHandle:
        try:
            return registry.get(cell_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Cell '{cell_id}' not found") from None

    async def _call_handle(cell_id: str, action: str, coro) -> _OkResponse:
        try:
            await coro
        except Exception:
            logger.error("Failed to %s cell %s", action, cell_id, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to {action} cell '{cell_id}'") from None
        return _OkResponse()

    return app


class _CellRegistry:
    def __init__(self) -> None:
        self._handles: dict[str, _CellHandle] = {}

    def register(self, handle: _CellHandle) -> None:
        if handle.cell_id in self._handles:
            raise ValueError(f"Cell '{handle.cell_id}' is already registered")
        self._handles[handle.cell_id] = handle

    def get_all(self) -> list[_CellHandle]:
        return list(self._handles.values())

    def get(self, cell_id: str) -> _CellHandle:
        try:
            return self._handles[cell_id]
        except KeyError as e:
            raise KeyError(f"Cell '{cell_id}' not found") from e


class _CellHandle(abc.ABC):
    @property
    @abc.abstractmethod
    def cell_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def cell_type(self) -> str: ...

    @abc.abstractmethod
    async def stop(self, timeout_seconds: int) -> None: ...

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def get_status(self) -> str: ...

    @abc.abstractmethod
    async def get_node_ids(self) -> list[str]: ...


class _ActorCellHandle(_CellHandle):
    def __init__(self, *, group: RayTrainGroup, cell_index: int) -> None:
        self._group = group
        self._cell_index = cell_index

    @property
    def cell_id(self) -> str:
        return f"actor-{self._cell_index}"

    @property
    def cell_type(self) -> str:
        return "actor"

    async def stop(self, timeout_seconds: int) -> None:
        self._group.stop(self._cell_index)

    async def start(self) -> None:
        self._group.start(self._cell_index)

    async def get_status(self) -> str:
        cell = self._group._cells[self._cell_index]
        if cell.is_errored:
            return "errored"
        elif cell.is_running:
            return "running"
        elif cell.is_pending:
            return "pending"
        else:
            return "stopped"

    async def get_node_ids(self) -> list[str]:
        return []


class _RolloutCellHandle(_CellHandle):
    def __init__(self, rollout_manager: object, cell_id: str) -> None:
        self._rollout_manager = rollout_manager
        self._cell_id = cell_id

    @property
    def cell_id(self) -> str:
        return self._cell_id

    @property
    def cell_type(self) -> str:
        return "rollout"

    async def stop(self, timeout_seconds: int) -> None:
        await self._rollout_manager.stop_cell.remote(self._cell_id, timeout_seconds)

    async def start(self) -> None:
        await self._rollout_manager.start_cell.remote(self._cell_id)

    async def get_status(self) -> str:
        return await self._rollout_manager.get_cell_status.remote(self._cell_id)

    async def get_node_ids(self) -> list[str]:
        return await self._rollout_manager.get_cell_node_ids.remote(self._cell_id)
