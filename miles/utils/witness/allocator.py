from miles.utils.pydantic_utils import FrozenStrictBaseModel


class WitnessInfo(FrozenStrictBaseModel):
    witness_ids: list[int]
    stale_ids: list[int]


class WitnessIdAllocator:
    def __init__(self, *, buffer_size: int) -> None:
        self._buffer_size = buffer_size
        self._counter: int = 0

    def allocate(self, num_ids: int) -> WitnessInfo:
        assert num_ids <= self._buffer_size, (
            f"num_ids ({num_ids}) exceeds buffer_size ({self._buffer_size}). " f"Increase --witness-buffer-size."
        )
        ids = [(self._counter + i) % self._buffer_size for i in range(num_ids)]
        stale_ids = _compute_stale_ids(
            keep_count=int(self._buffer_size * 0.7),
            counter=self._counter + num_ids,
            buffer_size=self._buffer_size,
        )
        self._counter += num_ids
        return WitnessInfo(witness_ids=ids, stale_ids=stale_ids)


def _compute_stale_ids(*, keep_count: int, counter: int, buffer_size: int) -> list[int]:
    if counter == 0:
        return []
    num_stale = buffer_size - min(keep_count, counter, buffer_size)
    if num_stale == 0:
        return []

    head = counter % buffer_size
    return [(head + i) % buffer_size for i in range(num_stale)]
