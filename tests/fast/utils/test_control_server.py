from __future__ import annotations

import asyncio

import httpx
import pytest

from miles.utils.control_server.handles import _ActorCellHandle, _RolloutCellHandle
from miles.utils.control_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus
from miles.utils.control_server.registry import _CellRegistry
from miles.utils.control_server.server import _create_control_app


class _MockHandle:
    def __init__(
        self,
        cell_id: str,
        cell_type: str,
        cell_index: int = 0,
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        is_suspended: bool = False,
        suspend_error: Exception | None = None,
        resume_error: Exception | None = None,
    ) -> None:
        self.cell_id = cell_id
        self.cell_type = cell_type
        self._cell_index = cell_index
        self._phase = phase
        self._conditions = conditions or [
            {"type": "Allocated", "status": "True"},
            {"type": "Ready", "status": "True"},
        ]
        self._is_suspended = is_suspended
        self._suspend_error = suspend_error
        self._resume_error = resume_error
        self.suspend_calls: int = 0
        self.resume_calls: int = 0

    @property
    def cell_index(self) -> int:
        return self._cell_index

    async def get_cell(self) -> Cell:
        return Cell(
            metadata=CellMetadata(
                name=self.cell_id,
                labels={
                    "miles.io/cell-type": self.cell_type,
                    "miles.io/cell-index": str(self._cell_index),
                },
            ),
            spec=CellSpec(suspend=self._is_suspended),
            status=CellStatus(
                phase=self._phase,
                conditions=[CellCondition(**c) for c in self._conditions],
            ),
        )

    async def suspend(self) -> None:
        if self._suspend_error:
            raise self._suspend_error
        self.suspend_calls += 1
        self._is_suspended = True
        self._phase = "Suspended"
        self._conditions = [
            {"type": "Allocated", "status": "False"},
            {"type": "Ready", "status": "False"},
        ]

    async def resume(self) -> None:
        if self._resume_error:
            raise self._resume_error
        self.resume_calls += 1
        self._is_suspended = False
        self._phase = "Running"
        self._conditions = [
            {"type": "Allocated", "status": "True"},
            {"type": "Ready", "status": "True"},
        ]


class _MockRemoteCall:
    def __init__(self, return_value: object) -> None:
        self._return_value = return_value
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args: object, **kwargs: object) -> asyncio.Future[object]:
        self.calls.append((args, kwargs))
        future: asyncio.Future[object] = asyncio.get_event_loop().create_future()
        future.set_result(self._return_value)
        return future


class _MockRolloutManager:
    def __init__(
        self,
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        is_suspended: bool = False,
    ) -> None:
        self.stop_cell = _MockRemoteCall(None)
        self.start_cell = _MockRemoteCall(None)
        self.get_cell_phase = _MockRemoteCall(phase)
        self.get_cell_conditions = _MockRemoteCall(
            conditions
            or [
                {"type": "Allocated", "status": "True"},
                {"type": "Ready", "status": "True"},
            ]
        )
        self.get_cell_is_suspended = _MockRemoteCall(is_suspended)


@pytest.fixture
def registry() -> _CellRegistry:
    return _CellRegistry()


@pytest.fixture
def async_client(registry: _CellRegistry) -> httpx.AsyncClient:
    app = _create_control_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


class TestCellRegistry:
    def test_register_and_get_by_id(self, registry: _CellRegistry) -> None:
        handle = _MockHandle(cell_id="cell-0", cell_type="rollout")
        registry.register(handle)
        assert registry.get("cell-0") is handle

    def test_get_unknown_id_raises_key_error(self, registry: _CellRegistry) -> None:
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_get_all_returns_all_registered(self, registry: _CellRegistry) -> None:
        h1 = _MockHandle(cell_id="cell-0", cell_type="rollout")
        h2 = _MockHandle(cell_id="cell-1", cell_type="rollout")
        registry.register(h1)
        registry.register(h2)

        all_handles = registry.get_all()
        assert len(all_handles) == 2
        assert h1 in all_handles
        assert h2 in all_handles

    def test_register_duplicate_id_raises(self, registry: _CellRegistry) -> None:
        h1 = _MockHandle(cell_id="cell-0", cell_type="rollout")
        h2 = _MockHandle(cell_id="cell-0", cell_type="rollout")
        registry.register(h1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(h2)


class TestGetHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestGetCells:
    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty_cell_list(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.get("/api/v1/cells")
        assert resp.status_code == 200
        data = resp.json()
        assert data["apiVersion"] == "miles.io/v1"
        assert data["kind"] == "CellList"
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_returns_all_cells_with_k8s_structure(
        self, registry: _CellRegistry, async_client: httpx.AsyncClient
    ) -> None:
        registry.register(
            _MockHandle(
                cell_id="actor-0",
                cell_type="actor",
                cell_index=0,
                phase="Running",
            )
        )
        registry.register(
            _MockHandle(
                cell_id="rollout-0",
                cell_type="rollout",
                cell_index=0,
                phase="Suspended",
                is_suspended=True,
                conditions=[
                    {"type": "Allocated", "status": "False"},
                    {"type": "Ready", "status": "False"},
                ],
            )
        )

        resp = await async_client.get("/api/v1/cells")
        assert resp.status_code == 200
        data = resp.json()
        assert data["kind"] == "CellList"
        assert len(data["items"]) == 2

        by_name = {item["metadata"]["name"]: item for item in data["items"]}

        actor = by_name["actor-0"]
        assert actor["apiVersion"] == "miles.io/v1"
        assert actor["kind"] == "Cell"
        assert actor["metadata"]["labels"]["miles.io/cell-type"] == "actor"
        assert actor["spec"]["suspend"] is False
        assert actor["status"]["phase"] == "Running"

        rollout = by_name["rollout-0"]
        assert rollout["spec"]["suspend"] is True
        assert rollout["status"]["phase"] == "Suspended"


class TestGetCell:
    @pytest.mark.asyncio
    async def test_returns_single_cell(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        registry.register(
            _MockHandle(cell_id="actor-0", cell_type="actor", phase="Running")
        )

        resp = await async_client.get("/api/v1/cells/actor-0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["kind"] == "Cell"
        assert data["metadata"]["name"] == "actor-0"
        assert data["status"]["phase"] == "Running"

    @pytest.mark.asyncio
    async def test_not_found_returns_k8s_status(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.get("/api/v1/cells/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert data["kind"] == "Status"
        assert data["reason"] == "NotFound"
        assert "nonexistent" in data["message"]


class TestPatchCell:
    @pytest.mark.asyncio
    async def test_suspend_cell_via_patch(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = _MockHandle(cell_id="actor-0", cell_type="actor", phase="Running")
        registry.register(handle)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 200
        assert handle.suspend_calls == 1

        data = resp.json()
        assert data["status"]["phase"] == "Suspended"
        assert data["spec"]["suspend"] is True

    @pytest.mark.asyncio
    async def test_resume_cell_via_patch(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = _MockHandle(
            cell_id="actor-0",
            cell_type="actor",
            phase="Suspended",
            is_suspended=True,
        )
        registry.register(handle)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": False}})
        assert resp.status_code == 200
        assert handle.resume_calls == 1

        data = resp.json()
        assert data["status"]["phase"] == "Running"

    @pytest.mark.asyncio
    async def test_patch_with_no_spec_is_noop(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = _MockHandle(cell_id="actor-0", cell_type="actor", phase="Running")
        registry.register(handle)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={})
        assert resp.status_code == 200
        assert handle.suspend_calls == 0
        assert handle.resume_calls == 0

    @pytest.mark.asyncio
    async def test_patch_not_found_returns_k8s_status(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.patch("/api/v1/cells/nonexistent", json={"spec": {"suspend": True}})
        assert resp.status_code == 404
        data = resp.json()
        assert data["kind"] == "Status"
        assert data["reason"] == "NotFound"

    @pytest.mark.asyncio
    async def test_patch_suspend_idempotent(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = _MockHandle(
            cell_id="actor-0",
            cell_type="actor",
            phase="Suspended",
            is_suspended=True,
        )
        registry.register(handle)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 200
        assert handle.suspend_calls == 1

    @pytest.mark.asyncio
    async def test_patch_error_returns_500_k8s_status(
        self, registry: _CellRegistry, async_client: httpx.AsyncClient
    ) -> None:
        handle = _MockHandle(
            cell_id="actor-0",
            cell_type="actor",
            suspend_error=RuntimeError("engine crashed"),
        )
        registry.register(handle)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 500
        data = resp.json()
        assert data["kind"] == "Status"
        assert data["reason"] == "InternalError"


class TestActorCellHandle:
    def test_cell_id_and_type(self) -> None:
        group = _make_mock_group([_MockRayTrainCell()])
        handle = _ActorCellHandle(group=group, cell_index=0)
        assert handle.cell_id == "actor-0"
        assert handle.cell_type == "actor"

    @pytest.mark.asyncio
    async def test_get_cell_returns_full_cell_structure(self) -> None:
        group = _make_mock_group([
            _MockRayTrainCell(
                phase="Running",
                conditions=[
                    {"type": "Allocated", "status": "True"},
                    {"type": "Ready", "status": "True"},
                ],
                is_stopped=False,
            )
        ])
        handle = _ActorCellHandle(group=group, cell_index=0)
        cell = await handle.get_cell()

        assert cell.apiVersion == "miles.io/v1"
        assert cell.kind == "Cell"
        assert cell.metadata.name == "actor-0"
        assert cell.metadata.labels["miles.io/cell-type"] == "actor"
        assert cell.metadata.labels["miles.io/cell-index"] == "0"
        assert cell.spec.suspend is False
        assert cell.status.phase == "Running"
        assert len(cell.status.conditions) == 2
        assert cell.status.conditions[0].type == "Allocated"
        assert cell.status.conditions[0].status == "True"
        assert cell.status.conditions[1].type == "Ready"
        assert cell.status.conditions[1].status == "True"

    @pytest.mark.asyncio
    async def test_get_cell_suspended(self) -> None:
        group = _make_mock_group([
            _MockRayTrainCell(
                phase="Suspended",
                conditions=[
                    {"type": "Allocated", "status": "False"},
                    {"type": "Ready", "status": "False"},
                ],
                is_stopped=True,
            )
        ])
        handle = _ActorCellHandle(group=group, cell_index=0)
        cell = await handle.get_cell()

        assert cell.spec.suspend is True
        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_suspend_delegates_to_group(self) -> None:
        from unittest.mock import MagicMock

        group = _make_mock_group([_MockRayTrainCell()])
        group.stop_cell = MagicMock()
        handle = _ActorCellHandle(group=group, cell_index=2)
        await handle.suspend()
        group.stop_cell.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_resume_delegates_to_group(self) -> None:
        from unittest.mock import MagicMock

        group = _make_mock_group([_MockRayTrainCell()])
        group.start_cell = MagicMock()
        handle = _ActorCellHandle(group=group, cell_index=1)
        await handle.resume()
        group.start_cell.assert_called_once_with(1)


class TestRolloutCellHandle:
    @pytest.mark.asyncio
    async def test_get_cell_delegates_to_manager(self) -> None:
        manager = _MockRolloutManager(
            phase="Running",
            conditions=[
                {"type": "Allocated", "status": "True"},
                {"type": "Ready", "status": "True"},
            ],
            is_suspended=False,
        )

        handle = _RolloutCellHandle(rollout_manager=manager, cell_index=0)
        cell = await handle.get_cell()

        assert cell.metadata.name == "rollout-0"
        assert cell.metadata.labels["miles.io/cell-type"] == "rollout"
        assert cell.status.phase == "Running"
        assert cell.spec.suspend is False

    @pytest.mark.asyncio
    async def test_suspend_delegates_to_manager(self) -> None:
        manager = _MockRolloutManager()

        handle = _RolloutCellHandle(rollout_manager=manager, cell_index=0)
        await handle.suspend()

        assert manager.stop_cell.calls == [((0,), {})]

    @pytest.mark.asyncio
    async def test_resume_delegates_to_manager(self) -> None:
        manager = _MockRolloutManager()

        handle = _RolloutCellHandle(rollout_manager=manager, cell_index=0)
        await handle.resume()

        assert manager.start_cell.calls == [((0,), {})]

    def test_cell_type_is_rollout(self) -> None:
        handle = _RolloutCellHandle(rollout_manager=object(), cell_index=0)
        assert handle.cell_type == "rollout"
        assert handle.cell_id == "rollout-0"


class _MockRayTrainCell:
    def __init__(
        self,
        *,
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        is_stopped: bool = False,
    ) -> None:
        self._phase = phase
        self._conditions = conditions or [
            {"type": "Allocated", "status": "True"},
            {"type": "Ready", "status": "True"},
        ]
        self._is_stopped = is_stopped

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def conditions(self) -> list[dict[str, str | None]]:
        return self._conditions

    @property
    def is_stopped(self) -> bool:
        return self._is_stopped


def _make_mock_group(cells: list[_MockRayTrainCell]) -> object:
    from miles.ray.train.group import RayTrainGroup

    group = object.__new__(RayTrainGroup)
    group._cells = cells
    group._indep_dp_quorum_id = 0
    group._alive_cell_ids = frozenset()
    return group
