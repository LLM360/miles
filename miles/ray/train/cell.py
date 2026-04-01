import asyncio
import logging
import time
from collections.abc import Callable

import ray
from pydantic import BaseModel, ConfigDict

from miles.utils.control_server.models import CellCondition, CellStatus, TriState
from miles.utils.health_checker import BaseHealthChecker, SimpleHealthChecker, SimpleHealthCheckerConfig
from miles.utils.indep_dp import IndepDPInfo

logger = logging.getLogger(__name__)


ActorFactory = Callable[[], list[ray.actor.ActorHandle]]


class RayTrainCell:
    def __init__(
        self,
        *,
        args,
        role: str,
        with_ref: bool,
        cell_index: int,
        actor_factory: ActorFactory,
        rollout_manager: object | None,
        health_checker: BaseHealthChecker,
    ) -> None:
        self.args = args
        self.cell_index = cell_index
        self.role = role
        self.with_ref = with_ref
        self.rollout_manager = rollout_manager
        self._actor_factory = actor_factory

        # NOTE: do *NOT* directly modify `self._state`, but instead use `self._change_state`
        self._state: _CellState = _StatePending()
        self.health_checker = health_checker
        self.allocate_for_pending()

    # ------------------------ state transition ------------------------

    def stop(self) -> None:
        if self.is_stopped:
            logger.info(f"stop: cell {self.cell_index} already stopped, skipping")
            return

        def _core():
            if self.is_allocated:
                handles = self._get_actor_handles()
                for actor in handles:
                    ray.kill(actor)

            return _StateStopped()

        self._change_state("stop", (_StatePending, _StateAllocatedBase), _core)

    def mark_as_pending(self) -> None:
        if self.is_pending or self.is_allocated:
            logger.info(f"mark_as_pending: cell {self.cell_index} already {type(self._state).__name__}, skipping")
            return

        self._change_state("mark_as_pending", _StateStopped, _StatePending)

    def allocate_for_pending(self) -> None:
        def _core():
            actor_handles = self._actor_factory()
            return _StateAllocatedUninitialized(actor_handles=actor_handles)

        self._change_state("allocate_for_pending", _StatePending, _core)

    def _mark_as_alive(self, indep_dp_info: IndepDPInfo) -> None:
        self._change_state(
            "_mark_as_alive",
            _StateAllocatedUninitialized,
            lambda: _StateAllocatedAlive(
                actor_handles=self._state.actor_handles,
                indep_dp_info=indep_dp_info,
            ),
        )

    def _update_indep_dp_info(self, indep_dp_info: IndepDPInfo) -> None:
        self._change_state(
            "_update_indep_dp_info",
            _StateAllocatedAlive,
            lambda: _StateAllocatedAlive(
                actor_handles=self._state.actor_handles,
                indep_dp_info=indep_dp_info,
            ),
        )

    def _mark_as_errored(self) -> None:
        self._change_state(
            "_mark_as_errored",
            (_StateAllocatedAlive, _StateAllocatedErrored),
            lambda: _StateAllocatedErrored(
                actor_handles=self._state.actor_handles,
                indep_dp_info=self._state.indep_dp_info,
            ),
        )

    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type["_CellState"] | tuple[type["_CellState"], ...],
        fn: Callable[[], "_CellState"],
    ):
        logger.info(f"{debug_name} start {self.cell_index=} old={self._state}")
        assert isinstance(self._state, old_state_cls), f"{self.cell_index=} {self._state=}"
        self._state = fn()
        logger.info(f"{debug_name} end {self.cell_index=} new={self._state}")

    # ------------------------ cooperatively prepare ------------------------

    async def prepare_indep_dp_mode_alive(
        self,
        indep_dp_info: IndepDPInfo,
        send_ckpt_dst_ranks: list[int],
    ):
        await self.execute("reconfigure_indep_dp", indep_dp_info=indep_dp_info)
        self._update_indep_dp_info(indep_dp_info)

        for dst_rank in send_ckpt_dst_ranks:
            await self.execute("send_ckpt", dst_rank=dst_rank)

    async def prepare_indep_dp_mode_healing(
        self,
        indep_dp_info: IndepDPInfo,
        recv_ckpt_src_rank: int | None,
    ):
        await self.init(
            indep_dp_info=indep_dp_info,
            recv_ckpt_src_rank=recv_ckpt_src_rank,
        )

        await self.set_rollout_manager()

    # ------------------------ forward calls to actors ------------------------

    async def execute(self, fn_name: str, *args, mark_errored_on_failure: bool = True, **kwargs) -> list:
        return await self._execute_raw(
            fn_name,
            compute_args=lambda _: args,
            compute_kwargs=lambda _: kwargs,
            mark_errored_on_failure=mark_errored_on_failure,
        )

    async def _execute_raw(
        self,
        fn_name: str,
        compute_args,
        compute_kwargs,
        mark_errored_on_failure: bool = True,
    ) -> list:
        handles = self._get_actor_handles()
        try:
            return await asyncio.gather(
                *[
                    getattr(actor, fn_name).remote(*compute_args(i), **compute_kwargs(i))
                    for i, actor in enumerate(handles)
                ]
            )
        except Exception:
            logger.error(f"Cell {self.cell_index} failed in {fn_name}", exc_info=True)
            if mark_errored_on_failure:
                self._mark_as_errored()
            raise

    # ------------------------ TODO: move these methods up and down ------------------------

    async def connect(self, critic_cell: "RayTrainCell") -> list:
        critic_handles = critic_cell._get_actor_handles()
        return await self._execute_raw(
            "connect_actor_critic",
            compute_args=lambda i: (critic_handles[i],),
            compute_kwargs=lambda _: {},
        )

    async def init(
        self,
        *,
        indep_dp_info: IndepDPInfo,
        recv_ckpt_src_rank: int | None = None,
    ):
        self._mark_as_alive(indep_dp_info=indep_dp_info)
        await self.execute(
            "init",
            args=self.args,
            role=self.role,
            with_ref=self.with_ref,
            indep_dp_info=indep_dp_info,
            recv_ckpt_src_rank=recv_ckpt_src_rank,
        )
        await self.health_checker.start()

    async def set_rollout_manager(self):
        if (m := self.rollout_manager) is not None:
            return await self.execute("set_rollout_manager", m)
        return []

    # ------------------------ state helpers ------------------------

    @property
    def is_pending(self) -> bool:
        return isinstance(self._state, _StatePending)

    @property
    def is_allocated(self) -> bool:
        return isinstance(self._state, _StateAllocatedBase)

    @property
    def is_alive(self) -> bool:
        return isinstance(self._state, _StateAllocatedAlive)

    @property
    def is_errored(self) -> bool:
        return isinstance(self._state, _StateAllocatedErrored)

    @property
    def is_stopped(self) -> bool:
        return isinstance(self._state, _StateStopped)

    def cell_status(self) -> CellStatus:
        return _compute_cell_status(self._state, self.health_checker.status)

    @property
    def indep_dp_info(self) -> IndepDPInfo:
        assert isinstance(self._state, (_StateAllocatedAlive, _StateAllocatedErrored))
        return self._state.indep_dp_info

    def _get_actor_handles(self) -> list[ray.actor.ActorHandle]:
        assert isinstance(
            self._state, _StateAllocatedBase
        ), f"Cell {self.cell_index} is not allocated (state={type(self._state).__name__})"
        return self._state.actor_handles


# ------------------------ states ------------------------


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


# ------------------------ Actor factory ------------------------


# ------------------------ misc ------------------------


def create_trainer_cell_health_checker(
    *,
    cell: RayTrainCell,
    config: SimpleHealthCheckerConfig,
    max_heartbeat_age: float,
) -> SimpleHealthChecker:
    async def _check() -> None:
        if not cell.is_alive:
            return

        now = time.time()
        results = await cell.execute("get_heartbeat_status", mark_errored_on_failure=False)

        for status in results:
            delta = now - status.last_active_timestamp
            if delta > max_heartbeat_age:
                raise RuntimeError(
                    f"Heartbeat stale: last_active={status.last_active_timestamp:.1f}, "
                    f"now={now:.1f}, delta={delta:.1f}s, bump_count={status.bump_count}"
                )

    return SimpleHealthChecker(
        name=f"trainer-cell-{cell.cell_index}",
        check_fn=_check,
        config=config,
    )


def _compute_cell_status(state: _CellState, health_checker_status: TriState) -> CellStatus:
    match state:
        case _StateAllocatedAlive():
            if health_checker_status == TriState.FALSE:
                healthy = CellCondition.healthy(TriState.FALSE, reason="HealthCheckFailed")
            else:
                healthy = CellCondition.healthy(TriState.TRUE)
            return CellStatus(phase="Running", conditions=[CellCondition.allocated(TriState.TRUE), healthy])

        case _StateAllocatedUninitialized():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.healthy(TriState.TRUE),
                ],
            )

        case _StateAllocatedErrored():
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.healthy(TriState.FALSE, reason="ExecutionErrored"),
                ],
            )

        case _StatePending():
            return CellStatus(phase="Pending", conditions=[CellCondition.allocated(TriState.FALSE)])

        case _StateStopped():
            return CellStatus(phase="Suspended", conditions=[CellCondition.allocated(TriState.FALSE)])

        case _:
            raise NotImplementedError(f"Unknown state: {state}")
