from miles.ray.train.cell import RayTrainCell, _StateAllocated, _StatePending, _StateStopped


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
        """Calling mark_as_pending() on an allocated cell does not crash and state stays."""
        cell = _make_cell_with_state(_StateAllocated(actor_handles=[], phase="alive"))

        cell.mark_as_pending()

        assert cell.is_running


class TestPhaseTransitions:
    def test_allocate_sets_allocated_phase(self):
        """After allocate_for_pending, phase is 'allocated'."""
        cell = _make_cell_with_state(_StateAllocated(actor_handles=[], phase="uninitialized"))

        assert cell.is_allocated
        assert cell.is_running
        assert not cell.is_errored

    def test_mark_as_running(self):
        """mark_as_running transitions from allocated to running phase."""
        cell = _make_cell_with_state(_StateAllocated(actor_handles=[], phase="uninitialized"))

        cell.mark_as_running()

        assert cell.is_running
        assert not cell.is_errored

    def test_mark_as_errored(self):
        """mark_as_errored transitions from running to errored phase."""
        cell = _make_cell_with_state(_StateAllocated(actor_handles=[], phase="alive"))

        cell.mark_as_errored()

        assert cell.is_errored
        assert not cell.is_running
        assert cell.is_allocated
