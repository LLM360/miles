from types import SimpleNamespace

from tests.fast.ray.specs.test_inference import _make_session_server_args
from tests.fast.utils.workers.test_ray_worker_manager import _launch, _make_spec

from miles.ray.rollout.router_manager import wait_session_server_ready
from miles.ray.specs.inference import spec_session_server


async def test_spread_option_reaches_ray_actor_creation(fake_ray_cluster):
    spec = _make_spec("session-server", num_cells=2)
    scheduling = spec.scheduling.model_copy(update={"ray_scheduling_strategy": "SPREAD", "num_cpus_per_worker": 0.1})
    spec = spec.model_copy(update={"scheduling": scheduling})
    await _launch([spec])
    assert len(fake_ray_cluster.handles) == 2
    assert all(handle.options["scheduling_strategy"] == "SPREAD" for handle in fake_ray_cluster.handles)
    assert all(handle.options["num_cpus"] == 0.1 for handle in fake_ray_cluster.handles)


async def test_external_workers_need_no_local_checkpoint_or_provider():
    args = SimpleNamespace(use_session_server="v1", session_server_addrs=["worker-a:123", "https://worker-b:456/"])
    await wait_session_server_ready(args)
    assert args.session_server_addrs == ["http://worker-a:123", "https://worker-b:456"]
    assert args.session_server_backends == args.session_server_addrs
    assert args._session_server_external_pool is True


def test_external_workers_suppress_managed_pool_but_published_addresses_do_not():
    args = _make_session_server_args()
    args.session_server_addrs = ["worker:123"]
    assert spec_session_server(args).scheduling.num_cells == 0
    args._session_servers_managed = True
    assert spec_session_server(args).scheduling.num_cells == 2
