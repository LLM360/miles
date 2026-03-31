from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from miles.utils.mini_ft_controller.controller import CellSnapshot
from miles.utils.mini_ft_controller.ray_actor import MiniFTControllerActor


def _build_cell_json(
    *,
    name: str = "actor-0",
    cell_type: str = "actor",
    cell_index: int = 0,
    phase: str = "Running",
    healthy_status: str = "True",
    healthy_reason: str | None = None,
    is_suspended: bool = False,
) -> dict[str, Any]:
    return {
        "apiVersion": "miles.io/v1",
        "kind": "Cell",
        "metadata": {
            "name": name,
            "labels": {"miles.io/cell-type": cell_type, "miles.io/cell-index": str(cell_index)},
        },
        "spec": {"suspend": is_suspended},
        "status": {
            "phase": phase,
            "conditions": [
                {"type": "Allocated", "status": "True", "reason": None, "message": None, "lastTransitionTime": None},
                {
                    "type": "Healthy",
                    "status": healthy_status,
                    "reason": healthy_reason,
                    "message": None,
                    "lastTransitionTime": None,
                },
            ],
        },
    }


def _build_cell_list_json(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "apiVersion": "miles.io/v1",
        "kind": "CellList",
        "items": cells,
    }


def _create_actor_instance() -> MiniFTControllerActor:
    """Create a MiniFTControllerActor instance directly (without Ray)."""
    actor = MiniFTControllerActor.__new__(MiniFTControllerActor)
    actor.__init__(
        control_server_url="http://127.0.0.1:8080",
        poll_interval=10.0,
        resume_delay=5.0,
        max_consecutive_failures=5,
    )
    return actor


def _mock_response(*, status_code: int = 200, json_data: Any = None) -> httpx.Response:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=response,
        )
    return response


@pytest.mark.asyncio
async def test_get_cells_parses_response() -> None:
    """Mock httpx returning CellList JSON → verify CellSnapshot list correct."""
    actor = _create_actor_instance()

    cells_json = _build_cell_list_json([
        _build_cell_json(name="actor-0", healthy_status="True"),
        _build_cell_json(name="actor-1", healthy_status="False", healthy_reason="Fatal"),
    ])

    actor._client = AsyncMock()
    actor._client.get = AsyncMock(return_value=_mock_response(json_data=cells_json))

    result = await actor._get_cells()

    assert len(result) == 2
    assert result[0] == CellSnapshot(name="actor-0", healthy_status="True", healthy_reason=None)
    assert result[1] == CellSnapshot(name="actor-1", healthy_status="False", healthy_reason="Fatal")


@pytest.mark.asyncio
async def test_get_cells_extracts_healthy_condition() -> None:
    """Verify Healthy condition status and reason correctly extracted."""
    actor = _create_actor_instance()

    cells_json = _build_cell_list_json([
        _build_cell_json(name="cell-0", healthy_status="False", healthy_reason="Fatal"),
    ])

    actor._client = AsyncMock()
    actor._client.get = AsyncMock(return_value=_mock_response(json_data=cells_json))

    result = await actor._get_cells()

    assert result[0].healthy_status == "False"
    assert result[0].healthy_reason == "Fatal"


@pytest.mark.asyncio
async def test_suspend_cell_sends_correct_patch() -> None:
    """Verify PATCH body for suspend."""
    actor = _create_actor_instance()
    actor._client = AsyncMock()
    actor._client.patch = AsyncMock(return_value=_mock_response())

    await actor._suspend_cell("actor-0")

    actor._client.patch.assert_called_once_with(
        "/api/v1/cells/actor-0",
        json={"spec": {"suspend": True}},
    )


@pytest.mark.asyncio
async def test_resume_cell_sends_correct_patch() -> None:
    """Verify PATCH body for resume."""
    actor = _create_actor_instance()
    actor._client = AsyncMock()
    actor._client.patch = AsyncMock(return_value=_mock_response())

    await actor._resume_cell("actor-0")

    actor._client.patch.assert_called_once_with(
        "/api/v1/cells/actor-0",
        json={"spec": {"suspend": False}},
    )


@pytest.mark.asyncio
async def test_get_cells_http_error_raises() -> None:
    """Verify HTTP 4xx/5xx propagated."""
    actor = _create_actor_instance()
    actor._client = AsyncMock()
    actor._client.get = AsyncMock(return_value=_mock_response(status_code=500))

    with pytest.raises(httpx.HTTPStatusError):
        await actor._get_cells()


def test_argument_validation_requires_control_server_port() -> None:
    """mini_ft_controller_enabled=True + control_server_port=0 → error."""
    from miles.utils.arguments import miles_validate_args

    args = argparse.Namespace(
        mini_ft_controller_enabled=True,
        control_server_port=0,
        use_fault_tolerance=False,
        ft_components=None,
        eval_datasets=None,
        eval_data=None,
    )

    with pytest.raises(ValueError, match="--mini-ft-controller-enabled requires --control-server-port"):
        miles_validate_args(args)
