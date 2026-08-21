import asyncio
import logging

from miles.ray.specs.inference import compute_router_pool_id, compute_session_server_instance_id
from miles.rollout.session.config import has_external_session_servers, normalize_session_server_urls
from miles.utils.http_utils import wait_tcp_ready_async
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)

# Readiness budget for the spawned router/session-server children. The spawn
# context re-imports the heavy transformers/megatron chain (~13s typical in
# CI), and transient CI stalls have pushed startup past a 30s budget.
_SERVER_READY_TIMEOUT_SECS = 120


async def wait_router_ready(model_idx: int) -> HostAndPort:
    """Wait until the model's router, launched by the RayWorkerManager, is reachable and return its address."""
    provider: BaseWorkerProvider = RayWorkerProvider.create()  # TODO inject instance
    worker_name = compute_worker_name(pool_id=compute_router_pool_id(model_idx))
    router_addr = (await provider.get_addrs(worker_name=worker_name))["primary"]
    await wait_tcp_ready_async(router_addr.host, router_addr.port, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(f"Router ready at {router_addr}")
    return router_addr


async def wait_session_server_ready(args):
    """Start the standalone session servers when ``--use-session-server`` is set.

    One independent single-process server per resolved port; the rollout side
    picks one per session and its URL carries the affinity from then on.
    Always runs standalone regardless of whether ``--use-miles-router`` is
    active.
    """
    if not getattr(args, "use_session_server", False):
        return

    if has_external_session_servers(args):
        args.session_server_addrs = normalize_session_server_urls(args.session_server_addrs)
        args.session_server_backends = list(args.session_server_addrs)
        args._session_server_external_pool = True
        logger.info("Using external session workers: %s", args.session_server_addrs)
        return

    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        raise ValueError("--use-session-server requires --hf-checkpoint to be set.")

    if args.session_server_workers < 1:
        raise ValueError("--session-server-workers must be at least 1.")

    provider: BaseWorkerProvider = RayWorkerProvider.create()  # TODO inject instance
    addrs = await asyncio.gather(*[_wait_session_worker(provider, i) for i in range(args.session_server_workers)])
    # The canonical driver-side value; rollout code picks from this list. Instances may sit on
    # different hosts, so each one is addressed in full rather than by a port under a shared ip.
    args._session_servers_managed = True
    args.session_server_addrs = [f"{x.host}:{x.port}" for x in addrs]
    # Stable clients use full URLs; both lists describe the same owned workers.
    args.session_server_backends = [f"http://{addr}" for addr in args.session_server_addrs]

    # Children are started concurrently by the worker manager; publish only ready addresses.
    instance_ids: dict[str, str] = {}
    for instance_index, addr in enumerate(args.session_server_addrs):
        instance_ids[addr] = compute_session_server_instance_id(args, instance_index)
    # The per-address map OpenAIEndpointTracer.create reads instance ids from,
    # replacing the per-session /health probe.
    args.session_server_instance_ids = instance_ids
    logger.info(f"Session servers ready at {args.session_server_addrs} ({len(addrs)} instances)")


async def _restart_session_worker(cell_index: int, *, restart: bool) -> None:
    manager = RayWorkerManager.get_handle()
    cell_ids = [compute_cell_id(pool_id="session-server", cell_index=cell_index)]
    await manager.stop_cells.remote(cell_ids)
    if restart:
        await manager.start_cells.remote(cell_ids)


async def _wait_session_worker(provider: BaseWorkerProvider, cell_index: int) -> HostAndPort:
    worker_name = compute_worker_name(pool_id="session-server", cell_index=cell_index)
    for retry in range(10):
        try:
            addr = (await provider.get_addrs(worker_name=worker_name))["primary"]
            await wait_tcp_ready_async(addr.host, addr.port, timeout=_SERVER_READY_TIMEOUT_SECS)
            return addr
        except (RuntimeError, OSError):
            logger.warning("Session worker %s failed startup (%d/10)", worker_name, retry + 1, exc_info=True)
            await _restart_session_worker(cell_index, restart=retry < 9)
            if retry == 9:
                raise
