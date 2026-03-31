from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _OkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


class CellCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    status: Literal["True", "False", "Unknown"]
    reason: str | None = None
    message: str | None = None
    lastTransitionTime: str | None = None


class CellStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: Literal["Pending", "Running", "Suspended"]
    conditions: list[CellCondition]


class CellSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suspend: bool = False


class CellMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    labels: dict[str, str]


class Cell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apiVersion: str = "miles.io/v1"
    kind: str = "Cell"
    metadata: CellMetadata
    spec: CellSpec
    status: CellStatus


class CellList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apiVersion: str = "miles.io/v1"
    kind: str = "CellList"
    items: list[Cell]


class CellPatchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suspend: bool | None = None


class CellPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: CellPatchSpec | None = None


class K8sStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apiVersion: str = "v1"
    kind: str = "Status"
    status: str = "Failure"
    message: str
    reason: str
    code: int
