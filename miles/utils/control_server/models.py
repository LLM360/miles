from __future__ import annotations

from typing import Literal

from miles.utils.pydantic_utils import StrictBaseModel


class _OkResponse(StrictBaseModel):
    status: str = "ok"


class CellCondition(StrictBaseModel):
    type: Literal["Allocated", "Ready"]
    status: Literal["True", "False", "Unknown"]
    reason: str | None = None
    message: str | None = None
    lastTransitionTime: str | None = None


class CellStatus(StrictBaseModel):
    phase: Literal["Pending", "Running", "Suspended"]
    conditions: list[CellCondition]


class CellSpec(StrictBaseModel):
    suspend: bool = False


class CellMetadata(StrictBaseModel):
    name: str
    labels: dict[str, str]


class Cell(StrictBaseModel):
    apiVersion: Literal["miles.io/v1"] = "miles.io/v1"
    kind: Literal["Cell"] = "Cell"
    metadata: CellMetadata
    spec: CellSpec
    status: CellStatus


class CellList(StrictBaseModel):
    apiVersion: Literal["miles.io/v1"] = "miles.io/v1"
    kind: Literal["CellList"] = "CellList"
    items: list[Cell]


class CellPatchSpec(StrictBaseModel):
    suspend: bool | None = None


class CellPatch(StrictBaseModel):
    spec: CellPatchSpec | None = None


class K8sStatus(StrictBaseModel):
    apiVersion: Literal["v1"] = "v1"
    kind: Literal["Status"] = "Status"
    status: Literal["Failure"] = "Failure"
    message: str
    reason: str
    code: int
