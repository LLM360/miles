# TODO: move from module.py
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class WitnessIdAllocator:
    def allocate(self, num_ids: int) -> "WitnessInfo":
        do_allocate_things
        return TODO


class WitnessInfo(FrozenStrictBaseModel):
    witness_ids: list[int]
