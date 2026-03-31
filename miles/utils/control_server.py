import abc
import asyncio
import logging
import threading
from typing import Literal

import ray
import uvicorn
from fastapi import FastAPI, HTTPException

from miles.ray.train.group import RayTrainGroup
from miles.utils.pydantic_utils import StrictBaseModel

logger = logging.getLogger(__name__)


# ────────────────────── Pydantic models ──────────────────────


class CellInfo(StrictBaseModel):
    cell_id: str
    cell_type: Literal["actor", "rollout"]
    state: Literal["running", "stopped", "pending"]
    node_ids: list[str]


_DEFAULT_STOP_TIMEOUT_SECONDS = 30


class StopRequest(StrictBaseModel):
    timeout_seconds: int = _DEFAULT_STOP_TIMEOUT_SECONDS


class OkResponse(StrictBaseModel):
    status: str = "ok"


# ────────────────────── Cell handle ABC ──────────────────────


class CellHandle(abc.ABC):
    @property
    @abc.abstractmethod
    def cell_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def cell_type(self) -> Literal["actor", "rollout"]: ...

    @abc.abstractmethod
    def get_info(self) -> CellInfo: ...

    @abc.abstractmethod
    def stop(self, timeout_seconds: int) -> None: ...

    @abc.abstractmethod
    def start(self) -> None: ...


# ────────────────────── Actor cell handle ──────────────────────


class ActorCellHandle(CellHandle):
    def __init__(self, *, group: RayTrainGroup, cell_index: int) -> None:
        self._group = group
        self._cell_index = cell_index

    @property
    def cell_id(self) -> str:
        return f"actor-{self._cell_index}"

    @property
    def cell_type(self) -> Literal["actor", "rollout"]:
        return "actor"

    def get_info(self) -> CellInfo:
        cell = self._group._cells[self._cell_index]

        if cell.is_running:
            state: Literal["running", "stopped", "pending"] = "running"
        elif cell.is_pending:
            state = "pending"
        else:
            state = "stopped"

        return CellInfo(
            cell_id=self.cell_id,
            cell_type=self.cell_type,
            state=state,
            node_ids=[],
        )

    def stop(self, timeout_seconds: int) -> None:
        logger.info(f"Stopping actor cell {self._cell_index} (timeout_seconds={timeout_seconds})")
        self._group.stop(self._cell_index)

    def start(self) -> None:
        self._group.start(self._cell_index)


# ────────────────────── Rollout cell handle ──────────────────────


class RolloutCellHandle(CellHandle):
    """Ported from rl-resilience/miles_part_dev:_RolloutSubsystemHandle with minimal changes:
    async→sync (ray.get), cell_id str→cell_index int, adapted to CellHandle ABC."""

    def __init__(self, *, rollout_manager: object, cell_index: int) -> None:
        self._rollout_manager = rollout_manager
        self._cell_index = cell_index

    @property
    def cell_id(self) -> str:
        return f"rollout-{self._cell_index}"

    @property
    def cell_type(self) -> Literal["actor", "rollout"]:
        return "rollout"

    def get_info(self) -> CellInfo:
        status = ray.get(self._rollout_manager.get_cell_status.remote(self._cell_index))
        node_ids = ray.get(self._rollout_manager.get_cell_node_ids.remote(self._cell_index))
        return CellInfo(
            cell_id=self.cell_id,
            cell_type=self.cell_type,
            state=status,
            node_ids=node_ids,
        )

    def stop(self, timeout_seconds: int) -> None:
        ray.get(self._rollout_manager.stop_cell.remote(self._cell_index, timeout_seconds))

    def start(self) -> None:
        ray.get(self._rollout_manager.start_cell.remote(self._cell_index))


# ────────────────────── Control server ──────────────────────


class ControlServer:
    def __init__(self, *, port: int) -> None:
        self._port = port
        self._handles: dict[str, CellHandle] = {}
        self._stop_lock = threading.Lock()
        self._app = FastAPI()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

        self._setup_routes()

    def register(self, handle: CellHandle) -> None:
        self._handles[handle.cell_id] = handle

    def start(self) -> None:
        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self._port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)

        def _run() -> None:
            asyncio.run(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        logger.info(f"Control server started on port {self._port}")

    def _get_handle(self, cell_id: str) -> CellHandle:
        handle = self._handles.get(cell_id)
        if handle is None:
            raise HTTPException(status_code=404, detail=f"Cell {cell_id!r} not found")
        return handle

    def _setup_routes(self) -> None:
        app = self._app

        @app.get("/health")
        def health() -> OkResponse:
            return OkResponse()

        @app.get("/cells")
        def list_cells() -> list[CellInfo]:
            return [handle.get_info() for handle in self._handles.values()]

        @app.post("/cells/{cell_id}/stop")
        def stop_cell(cell_id: str, request: StopRequest | None = None) -> OkResponse:
            handle = self._get_handle(cell_id)
            timeout = request.timeout_seconds if request is not None else _DEFAULT_STOP_TIMEOUT_SECONDS

            with self._stop_lock:
                info = handle.get_info()
                if handle.cell_type == "actor" and info.state == "running":
                    running_actor_count = sum(
                        1
                        for h in self._handles.values()
                        if h.cell_type == "actor" and h.get_info().state == "running"
                    )
                    if running_actor_count <= 1:
                        raise HTTPException(
                            status_code=409,
                            detail="Cannot stop the last running actor cell",
                        )
                handle.stop(timeout_seconds=timeout)

            return OkResponse()

        @app.post("/cells/{cell_id}/start")
        def start_cell(cell_id: str) -> OkResponse:
            handle = self._get_handle(cell_id)
            handle.start()
            return OkResponse()
