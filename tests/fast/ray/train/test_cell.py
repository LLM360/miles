import asyncio

import pytest
import ray

from tests.fast.ray.train.conftest import make_alive_cell, make_cell, make_indep_dp_info


class TestInitialState:
    def test_starts_as_uninitialized_after_init(self):
        """After __init__, cell is allocated (uninitialized) — actors created but not init'd."""
        cell = make_cell()

        assert cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_pending
        assert not cell.is_stopped

    def test_actor_handles_are_real_ray_actors(self):
        cell = make_cell(actor_count=3)

        handles = cell._get_actor_handles()
        assert len(handles) == 3
        assert all(isinstance(h, ray.actor.ActorHandle) for h in handles)


class TestStopTransitions:
    def test_stop_from_uninitialized_kills_actors(self):
        cell = make_cell(actor_count=2)

        cell.stop()

        assert cell.is_stopped
        assert not cell.is_allocated

    def test_stop_from_alive_kills_actors(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])

        cell.stop()

        assert cell.is_stopped

    def test_stop_from_pending_transitions_to_stopped(self):
        cell = make_cell()
        cell.stop()
        cell.mark_as_pending()

        cell.stop()

        assert cell.is_stopped

    def test_stop_already_stopped_is_idempotent(self):
        cell = make_cell()
        cell.stop()

        cell.stop()

        assert cell.is_stopped


class TestMarkAsPending:
    def test_from_stopped(self):
        cell = make_cell()
        cell.stop()

        cell.mark_as_pending()

        assert cell.is_pending

    def test_idempotent_when_pending(self):
        cell = make_cell()
        cell.stop()
        cell.mark_as_pending()

        cell.mark_as_pending()

        assert cell.is_pending

    def test_idempotent_when_allocated(self):
        cell = make_cell()

        cell.mark_as_pending()

        assert cell.is_allocated


class TestAllocateForPending:
    def test_reallocate_after_stop_start(self):
        """After stop → pending → allocate, cell has fresh actors."""
        cell = make_cell(actor_count=2)
        old_handles = cell._get_actor_handles()

        cell.stop()
        cell.mark_as_pending()
        cell.allocate_for_pending()

        assert cell.is_allocated
        new_handles = cell._get_actor_handles()
        assert len(new_handles) == 2
        assert new_handles != old_handles


class TestMarkAsAlive:
    def test_transitions_uninitialized_to_alive(self):
        cell = make_cell()
        info = make_indep_dp_info(alive_cell_indices=[0, 1, 2])

        cell._mark_as_alive(indep_dp_info=info)

        assert cell.is_alive
        assert cell.indep_dp_info == info

    def test_preserves_actor_handles(self):
        cell = make_cell(actor_count=3)
        handles_before = cell._get_actor_handles()

        cell._mark_as_alive(indep_dp_info=make_indep_dp_info())

        assert cell._get_actor_handles() == handles_before

    def test_rejects_from_alive(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])

        with pytest.raises(AssertionError):
            cell._mark_as_alive(indep_dp_info=make_indep_dp_info())


class TestUpdateIndepDPInfo:
    def test_updates_stored_info(self):
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        cell._update_indep_dp_info(new_info)

        assert cell.indep_dp_info == new_info

    def test_preserves_actor_handles(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])
        handles = cell._get_actor_handles()

        cell._update_indep_dp_info(make_indep_dp_info(quorum_id=5))

        assert cell._get_actor_handles() == handles

    def test_rejects_from_uninitialized(self):
        cell = make_cell()

        with pytest.raises(AssertionError):
            cell._update_indep_dp_info(make_indep_dp_info())


class TestMarkAsErrored:
    def test_transitions_alive_to_errored(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])
        info = cell.indep_dp_info

        cell._mark_as_errored()

        assert cell.is_errored
        assert not cell.is_alive
        assert cell.is_allocated
        assert cell.indep_dp_info == info

    def test_errored_is_idempotent(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()

        cell._mark_as_errored()

        assert cell.is_errored


class TestAsyncInit:
    def test_dispatches_init_and_marks_alive(self):
        cell = make_cell(actor_count=2)
        info = make_indep_dp_info()

        refs = cell.async_init(indep_dp_info=info)

        assert len(refs) == 2
        assert cell.is_alive
        assert cell.indep_dp_info == info
        ray.get(refs)

        for handle in cell._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert len(calls) == 1
            assert calls[0][0] == "init"


class TestPrepareIndepDPModeAlive:
    def test_reconfigure_and_update_info(self):
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        asyncio.get_event_loop().run_until_complete(
            cell.prepare_indep_dp_mode_alive(indep_dp_info=new_info, send_ckpt_dst_ranks=[])
        )

        assert cell.indep_dp_info == new_info
        assert cell.is_alive

        for handle in cell._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "reconfigure_indep_dp" for c in calls)

    def test_sends_ckpt_to_dst_ranks(self):
        cell = make_alive_cell(0, alive_cell_indices=[0, 1, 2])

        new_info = make_indep_dp_info(alive_cell_indices=[0, 1, 2], quorum_id=2)
        asyncio.get_event_loop().run_until_complete(
            cell.prepare_indep_dp_mode_alive(indep_dp_info=new_info, send_ckpt_dst_ranks=[1, 2])
        )

        handle = cell._get_actor_handles()[0]
        calls = ray.get(handle.get_calls.remote())
        send_calls = [c for c in calls if c[0] == "send_ckpt"]
        assert len(send_calls) == 2


class TestPrepareIndepDPModeHealing:
    def test_healing_inits_and_marks_alive(self):
        cell = make_cell(actor_count=1)
        info = make_indep_dp_info()

        asyncio.get_event_loop().run_until_complete(
            cell.prepare_indep_dp_mode_healing(indep_dp_info=info, recv_ckpt_src_rank=None)
        )

        assert cell.is_alive
        assert cell.indep_dp_info == info

        handle = cell._get_actor_handles()[0]
        calls = ray.get(handle.get_calls.remote())
        assert any(c[0] == "init" for c in calls)


class TestStatePredicates:
    def test_pending(self):
        cell = make_cell()
        cell.stop()
        cell.mark_as_pending()

        assert cell.is_pending
        assert not cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_errored
        assert not cell.is_stopped

    def test_uninitialized(self):
        cell = make_cell()

        assert not cell.is_pending
        assert cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_errored
        assert not cell.is_stopped

    def test_alive(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])

        assert not cell.is_pending
        assert cell.is_allocated
        assert cell.is_alive
        assert not cell.is_errored
        assert not cell.is_stopped

    def test_errored(self):
        cell = make_alive_cell(0, alive_cell_indices=[0])
        cell._mark_as_errored()

        assert not cell.is_pending
        assert cell.is_allocated
        assert not cell.is_alive
        assert cell.is_errored
        assert not cell.is_stopped

    def test_stopped(self):
        cell = make_cell()
        cell.stop()

        assert not cell.is_pending
        assert not cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_errored
        assert cell.is_stopped


class TestFullLifecycle:
    def test_full_stop_start_cycle(self):
        """Full lifecycle: init → alive → stop → pending → allocate → alive."""
        # Step 1: Create (Pending → Uninitialized)
        cell = make_cell(actor_count=2)
        assert cell.is_allocated and not cell.is_alive

        # Step 2: Alive
        info_v1 = make_indep_dp_info(alive_cell_indices=[0, 1, 2], quorum_id=1)
        cell._mark_as_alive(indep_dp_info=info_v1)
        assert cell.is_alive

        # Step 3: Stop
        cell.stop()
        assert cell.is_stopped

        # Step 4: Pending
        cell.mark_as_pending()
        assert cell.is_pending

        # Step 5: Allocate (new actors)
        cell.allocate_for_pending()
        assert cell.is_allocated and not cell.is_alive

        # Step 6: Alive again with new config
        info_v2 = make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        cell._mark_as_alive(indep_dp_info=info_v2)
        assert cell.is_alive
        assert cell.indep_dp_info.quorum_id == 2
        assert cell.indep_dp_info.alive_size == 2
