from miles.ray.train.cell import RayTrainCell, _StateAllocatedAlive, _StateAllocatedErrored, _StatePending, _StateStopped, _StateAllocatedUninitialized


def _make_cell_with_state(state) -> RayTrainCell:
    """Create a RayTrainCell with a pre-set state, bypassing __init__."""
    cell = object.__new__(RayTrainCell)
    cell.cell_index = 0
    cell._state = state
    return cell


class TestStopIdempotent:
    def test_stop_already_stopped_is_noop(self):
        """Calling stop() on an already-stopped cell does not crash and state remains Stopped."""
        cell = _make_cell_with_state(_StateStopped())

        cell.stop()

        assert cell.is_stopped

    def test_stop_pending_transitions_to_stopped(self):
        """Calling stop() on a Pending cell transitions to Stopped (normal path)."""
        cell = _make_cell_with_state(_StatePending())

        cell.stop()

        assert cell.is_stopped


class TestMarkAsPendingIdempotent:
    def test_mark_as_pending_already_pending_is_noop(self):
        """Calling mark_as_pending() on an already-pending cell does not crash."""
        cell = _make_cell_with_state(_StatePending())

        cell.mark_as_pending()

        assert cell.is_pending

    def test_mark_as_pending_already_allocated_is_noop(self):
        """Calling mark_as_pending() on an alive cell does not crash and state stays."""
        cell = _make_cell_with_state(_StateAllocatedAlive(actor_handles=[]))

        cell.mark_as_pending()

        assert cell.is_running


class TestPhaseTransitions:
    def test_uninitialized_state(self):
        """Uninitialized state is allocated and running but not errored."""
        cell = _make_cell_with_state(_StateAllocatedUninitialized(actor_handles=[]))

        assert cell.is_allocated
        assert cell.is_running
        assert not cell.is_errored

    def test_mark_as_alive(self):
        """mark_as_alive transitions from Uninitialized to Alive."""
        cell = _make_cell_with_state(_StateAllocatedUninitialized(actor_handles=[]))

        cell._mark_as_alive()

        assert cell.is_running
        assert not cell.is_errored

    def test_mark_as_errored(self):
        """mark_as_errored transitions from Alive to Errored."""
        cell = _make_cell_with_state(_StateAllocatedAlive(actor_handles=[]))

        cell.mark_as_errored()

        assert cell.is_errored
        assert not cell.is_running
        assert cell.is_allocated
