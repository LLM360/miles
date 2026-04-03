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
    def __init__(self, *, actions: list[FTTestAction], group: "RayTrainGroup", num_cells: int) -> None:
        self._actions = actions
        self._group = group
        self._num_cells = num_cells
        self._step: int = 0

    @staticmethod
    def from_args(args: object, *, group: "RayTrainGroup", num_cells: int) -> "FTTestActionExecutor":
        raw: str | None = getattr(args, "ci_ft_test_actions", None)
        actions = _parse_ft_test_actions(raw)
        if actions:
            logger.info("FT test actions activated: %d actions", len(actions))
        return FTTestActionExecutor(actions=actions, group=group, num_cells=num_cells)

    def run_after_step(self) -> None:
        for action in self._actions:
            if action.after_step == self._step:
                cell_index = action.cell_index if action.cell_index >= 0 else self._num_cells - 1
                logger.info("FT test action: %s cell %d after step %d", action.action, cell_index, self._step)
                getattr(self._group, action.action)(cell_index)
        self._step += 1


_ACTION_LIST_ADAPTER: TypeAdapter[list[FTTestAction]] = TypeAdapter(list[FTTestAction])


def _parse_ft_test_actions(raw: str | None) -> list[FTTestAction]:
    if not raw:
        return []
    return _ACTION_LIST_ADAPTER.validate_json(raw)
