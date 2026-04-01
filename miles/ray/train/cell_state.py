import ray

from pydantic import BaseModel, ConfigDict

from miles.utils.indep_dp import IndepDPInfo


class _StateBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class _StatePending(_StateBase):
    pass


class _StateAllocatedBase(_StateBase):
    actor_handles: list[ray.actor.ActorHandle]


class _StateAllocatedUninitialized(_StateAllocatedBase):
    pass


class _StateAllocatedAlive(_StateAllocatedBase):
    indep_dp_info: IndepDPInfo


class _StateAllocatedErrored(_StateAllocatedBase):
    indep_dp_info: IndepDPInfo


class _StateStopped(_StateBase):
    pass


_CellState = _StatePending | _StateAllocatedBase | _StateStopped

