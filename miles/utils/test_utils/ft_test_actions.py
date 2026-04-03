import logging
import os
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from miles.utils.pydantic_utils import FrozenStrictBaseModel

if TYPE_CHECKING:
    from miles.ray.train.group import RayTrainGroup

logger = logging.getLogger(__name__)


class FTTestAction(FrozenStrictBaseModel):
    at_rollout: int
    action: Literal["stop_cell_at_end", "start_cell_at_end", "crash_before_allreduce"]
    cell_index: int = -1  # -1 = last cell
    rank: int = 0  # for actor-level actions: which rank within the cell
    attempt: int = 0  # for actor-level actions: which attempt (0 = first try)


_ACTION_LIST_ADAPTER: TypeAdapter[list[FTTestAction]] = TypeAdapter(list[FTTestAction])


def _parse_ft_test_actions(raw: str | None) -> list[FTTestAction]:
    if not raw:
        return []
    return _ACTION_LIST_ADAPTER.validate_json(raw)


_GROUP_ACTIONS = {"stop_cell_at_end", "start_cell_at_end"}
_ACTOR_ACTIONS = {"crash_before_allreduce"}


class FTTestActionGroupExecutor:
    """Runs in RayTrainGroup.train() after each step. Handles stop_cell_at_end / start_cell_at_end."""

    def __init__(self, *, actions: list[FTTestAction], group: "RayTrainGroup") -> None:
        self._actions = actions
        self._group = group

    @staticmethod
    def from_args(args: object, *, group: "RayTrainGroup") -> "FTTestActionGroupExecutor":
        raw: str | None = getattr(args, "ci_ft_test_actions", None)
        all_actions = _parse_ft_test_actions(raw)
        actions = [a for a in all_actions if a.action in _GROUP_ACTIONS]
        if actions:
            logger.info("FT test group actions activated: %d actions", len(actions))
        return FTTestActionGroupExecutor(actions=actions, group=group)

    def run_after_step(self, rollout_id: int) -> None:
        for action in self._actions:
            if action.at_rollout == rollout_id:
                cell_index = action.cell_index if action.cell_index >= 0 else self._group.num_cells - 1
                logger.info("FT test action: %s cell %d after rollout %d", action.action, cell_index, rollout_id)

                if action.action == "stop_cell_at_end":
                    self._group.stop_cell(cell_index)
                elif action.action == "start_cell_at_end":
                    self._group.start_cell(cell_index)


class FTTestActionActorExecutor:
    """Runs in train_one_step() before allreduce. Handles crash_before_allreduce."""

    def __init__(self, *, actions: list[FTTestAction]) -> None:
        self._actions = actions

    @staticmethod
    def from_args(args: object) -> "FTTestActionActorExecutor":
        raw: str | None = getattr(args, "ci_ft_test_actions", None)
        all_actions = _parse_ft_test_actions(raw)
        actions = [a for a in all_actions if a.action in _ACTOR_ACTIONS]
        if actions:
            logger.info("FT test actor actions activated: %d actions", len(actions))
        return FTTestActionActorExecutor(actions=actions)

    def maybe_crash(self, *, rollout_id: int, attempt: int, cell_index: int, rank: int, num_cells: int) -> None:
        for action in self._actions:
            resolved_cell = action.cell_index if action.cell_index >= 0 else num_cells - 1
            if (
                action.at_rollout == rollout_id
                and action.attempt == attempt
                and resolved_cell == cell_index
                and action.rank == rank
            ):
                logger.warning(
                    "FT test action: crash_before_allreduce at rollout %d attempt %d cell %d rank %d — calling os._exit(1)",
                    rollout_id,
                    attempt,
                    cell_index,
                    rank,
                )
                os._exit(1)
