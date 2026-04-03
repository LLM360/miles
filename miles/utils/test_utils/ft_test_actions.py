import logging
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from miles.utils.pydantic_utils import FrozenStrictBaseModel

if TYPE_CHECKING:
    from miles.ray.train.group import RayTrainGroup

logger = logging.getLogger(__name__)


class FTTestAction(FrozenStrictBaseModel):
    after_step: int
    action: Literal["stop_cell", "start_cell"]
    cell_index: int  # -1 = last cell


class FTTestActionExecutor:
    def __init__(self, *, actions: list[FTTestAction], group: "RayTrainGroup") -> None:
        self._actions = actions
        self._group = group

    @staticmethod
    def from_args(args: object, *, group: "RayTrainGroup") -> "FTTestActionExecutor":
        raw: str | None = getattr(args, "ci_ft_test_actions", None)
        actions = _parse_ft_test_actions(raw)
        if actions:
            logger.info("FT test actions activated: %d actions", len(actions))
        return FTTestActionExecutor(actions=actions, group=group)

    def run_after_step(self, rollout_id: int) -> None:
        for action in self._actions:
            if action.after_step == rollout_id:
                cell_index = action.cell_index if action.cell_index >= 0 else self._group.num_cells - 1
                logger.info("FT test action: %s cell %d after rollout %d", action.action, cell_index, rollout_id)

                if action.action == "stop_cell":
                    self._group.stop_cell(cell_index)
                elif action.action == "start_cell":
                    self._group.start_cell(cell_index)


_ACTION_LIST_ADAPTER: TypeAdapter[list[FTTestAction]] = TypeAdapter(list[FTTestAction])


def _parse_ft_test_actions(raw: str | None) -> list[FTTestAction]:
    if not raw:
        return []
    return _ACTION_LIST_ADAPTER.validate_json(raw)
