from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miles.utils.control_server.handles import _ActorCellHandle, _RolloutCellHandle

from .conftest import MockRayTrainCell, MockRolloutManager, make_mock_group


class TestActorCellHandle:
    def test_cell_id_and_type(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        handle = _ActorCellHandle(group=group, cell_index=0)
        assert handle.cell_id == "actor-0"
        assert handle.cell_type == "actor"

    @pytest.mark.asyncio
    async def test_get_cell_returns_full_cell_structure(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        handle = _ActorCellHandle(group=group, cell_index=0)
        cell = await handle.get_cell()

        assert cell.model_dump() == {
            "apiVersion": "miles.io/v1",
            "kind": "Cell",
            "metadata": {
                "name": "actor-0",
                "labels": {
                    "miles.io/cell-type": "actor",
                    "miles.io/cell-index": "0",
                },
            },
            "spec": {"suspend": False},
            "status": {
                "phase": "Running",
                "conditions": [
                    {
                        "type": "Allocated",
                        "status": "True",
                        "reason": None,
                        "message": None,
                        "lastTransitionTime": None,
                    },
                    {"type": "Healthy", "status": "True", "reason": None, "message": None, "lastTransitionTime": None},
                ],
            },
        }

    @pytest.mark.asyncio
    async def test_get_cell_suspended(self) -> None:
        group = make_mock_group(
            [
                MockRayTrainCell(
                    phase="Suspended",
                    conditions=[
                        {"type": "Allocated", "status": "False"},
                        {"type": "Healthy", "status": "False"},
                    ],
                    is_stopped=True,
                )
            ]
        )
        handle = _ActorCellHandle(group=group, cell_index=0)
        cell = await handle.get_cell()

        assert cell.spec.suspend is True
        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_suspend_delegates_to_group(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        group.stop_cell = MagicMock()
        handle = _ActorCellHandle(group=group, cell_index=2)
        await handle.suspend()
        group.stop_cell.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_resume_delegates_to_group(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        group.start_cell = MagicMock()
        handle = _ActorCellHandle(group=group, cell_index=1)
        await handle.resume()
        group.start_cell.assert_called_once_with(1)


class TestRolloutCellHandle:
    @pytest.mark.asyncio
    async def test_get_cell_delegates_to_manager(self) -> None:
        manager = MockRolloutManager()
        handle = _RolloutCellHandle(rollout_manager=manager, cell_index=0)
        cell = await handle.get_cell()

        assert cell.metadata.name == "rollout-0"
        assert cell.metadata.labels["miles.io/cell-type"] == "rollout"
        assert cell.status.phase == "Running"
        assert cell.spec.suspend is False

    @pytest.mark.asyncio
    async def test_suspend_delegates_to_manager(self) -> None:
        manager = MockRolloutManager()
        handle = _RolloutCellHandle(rollout_manager=manager, cell_index=0)
        await handle.suspend()
        assert manager.stop_cell.calls == [((0,), {})]

    @pytest.mark.asyncio
    async def test_resume_delegates_to_manager(self) -> None:
        manager = MockRolloutManager()
        handle = _RolloutCellHandle(rollout_manager=manager, cell_index=0)
        await handle.resume()
        assert manager.start_cell.calls == [((0,), {})]

    def test_cell_type_is_rollout(self) -> None:
        handle = _RolloutCellHandle(rollout_manager=object(), cell_index=0)
        assert handle.cell_type == "rollout"
        assert handle.cell_id == "rollout-0"
