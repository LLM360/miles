from __future__ import annotations

import asyncio
import logging
import threading

import ray
import uvicorn
from fastapi import FastAPI
from starlette.responses import JSONResponse

from miles.ray.train.group import RayTrainGroup
from miles.utils.control_server.handles import _ActorCellHandle, _CellHandle, _RolloutCellHandle
from miles.utils.control_server.models import (
    Cell,
    CellList,
    CellPatch,
    K8sStatus,
    _OkResponse,
)
from miles.utils.control_server.registry import _CellRegistry

logger = logging.getLogger(__name__)


# -------------------------- entrypoint ------------------------------


def start_control_server(
    *,
    actor_model: RayTrainGroup,
    rollout_manager: object,
    port: int,
    ft_components: frozenset[str],
) -> None:
    registry = _CellRegistry()

    if "train" in ft_components:
        for i in range(len(actor_model._cells)):
            registry.register(_ActorCellHandle(group=actor_model, cell_index=i))

    if "rollout" in ft_components:
        # TODO the code will NOT work before implementing rollout ft
        num_rollout_cells = ray.get(rollout_manager.get_cell_count.remote())
        for i in range(num_rollout_cells):
            registry.register(
                _RolloutCellHandle(
                    rollout_manager=rollout_manager,
                    cell_index=i,
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


# -------------------------- main app ------------------------------


def _create_control_app(registry: _CellRegistry) -> FastAPI:
    app = FastAPI()

    # -------------------------- APIs ------------------------------

    @app.get("/api/v1/health")
    async def health() -> _OkResponse:
        return _OkResponse()

    @app.get("/api/v1/cells")
    async def get_cells() -> CellList:
        handles = registry.get_all()
        cells = list(await asyncio.gather(*(h.get_cell() for h in handles)))
        return CellList(items=cells)

    @app.get("/api/v1/cells/{name}")
    async def get_cell(name: str) -> Cell | JSONResponse:
        handle = _get_handle_or_404(name)
        if isinstance(handle, JSONResponse):
            return handle
        return await handle.get_cell()

    @app.patch("/api/v1/cells/{name}")
    async def patch_cell(name: str, body: CellPatch) -> Cell | JSONResponse:
        handle = _get_handle_or_404(name)
        if isinstance(handle, JSONResponse):
            return handle

        if body.spec is not None and body.spec.suspend is not None:
            try:
                if body.spec.suspend:
                    await handle.suspend()
                else:
                    await handle.resume()
            except Exception:
                logger.error("Failed to patch cell %s", name, exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content=K8sStatus(
                        message=f"Failed to patch cell '{name}'",
                        reason="InternalError",
                        code=500,
                    ).model_dump(),
                )

        return await handle.get_cell()

    # -------------------------- utils ------------------------------

    def _get_handle_or_404(name: str) -> _CellHandle | JSONResponse:
        try:
            return registry.get(name)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content=K8sStatus(
                    message=f"Cell '{name}' not found",
                    reason="NotFound",
                    code=404,
                ).model_dump(),
            )

    return app
