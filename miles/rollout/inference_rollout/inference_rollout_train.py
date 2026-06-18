import asyncio
import logging
from argparse import Namespace
from collections.abc import Callable

import sglang_router
from packaging.version import parse
from tqdm import tqdm

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.inference_rollout.abort_utils import collect_finished_rollouts, engines_idle
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.utils import dumper_utils
from miles.utils.http_utils import get, post
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def _signal_harbor(harbor_url: str, rollout_id: int) -> None:
    """Tell the harbor server to abort all in-flight trials of this rollout step.
    Raises on failure -- an abort we could not signal must not be silently dropped."""
    await post(f"{harbor_url}/rollouts/{rollout_id}/abort", {})


async def _abort_all_engines(args: Namespace) -> None:
    urls = await get_worker_urls(args)
    logger.info(f"[abort] abort_all -> {urls}")
    results = await asyncio.gather(
        *[post(f"{url}/abort_request", {"abort_all": True}) for url in urls],
        return_exceptions=True,
    )
    for url, r in zip(urls, results, strict=True):
        if isinstance(r, Exception):
            logger.warning(f"[abort] abort_all failed for {url}: {r!r}")


async def abort(state: GenerateState, pendings: set, rollout_id: int) -> list[list[Sample]]:
    """End-of-step rollout abort.

    Agentic rollouts signal the harbor server once: each trial closes its
    sessions, which cancels the in-flight SGLang request and tears down the
    environment. Vanilla rollouts broadcast abort_all to the engines. Either way
    the trainer then awaits the pending tasks and asserts the engines are idle
    before the weight update, so it never starts on a live decode. Partial-sample
    collection is preserved.
    """
    args = state.args
    assert not state.aborted
    state.aborted = True

    is_agentic = bool(getattr(args, "use_session_server", False) and getattr(args, "custom_agent_function_path", None))

    if is_agentic:
        harbor_url = getattr(args, "agent_server_url", None)
        if not harbor_url:
            raise RuntimeError("agentic rollout abort requires --agent-server-url")
        await _signal_harbor(harbor_url, rollout_id)
    else:
        await _abort_all_engines(args)

    aborted_samples = await collect_finished_rollouts(
        pendings,
        partial_rollout=args.partial_rollout,
        rollout_id=rollout_id,
    )

    if args.partial_rollout:
        logger.info(f"[abort] collected {sum(len(x) for x in aborted_samples)} partial samples")

    await assert_engines_idle(args)
    return aborted_samples


async def get_worker_urls(args: Namespace):
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        return response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        return [worker["url"] for worker in response["workers"]]


async def assert_engines_idle(args: Namespace) -> None:
    """Hard-fail the step unless every engine is idle before the weight update.
    A residual in-flight decode would poison the weight update."""
    urls = await get_worker_urls(args)

    async def _fetch():
        # Each worker's /get_load returns a LIST of per-shard GetLoadReqOutput dicts.
        results = await asyncio.gather(*[get(f"{u}/get_load") for u in urls], return_exceptions=True)
        loads: list[dict] = []
        for r in results:
            if isinstance(r, Exception):
                # Unreachable worker -> treat as busy (force retry/fail).
                loads.append({"num_reqs": 1, "num_waiting_reqs": 0})
            elif isinstance(r, list):
                loads.extend(r)
            elif isinstance(r, dict):
                loads.append(r)
        return loads

    ok = await engines_idle(
        _fetch,
        retries=getattr(args, "rollout_idle_check_retries", 15),
        interval=getattr(args, "rollout_idle_check_interval", 0.5),
    )
    if not ok:
        raise RuntimeError(
            "[abort] engines NOT idle after abort -- refusing the weight update "
            "(a residual in-flight decode would poison training)."
        )
    logger.info("[abort] engines idle -- weight update may proceed")


def stamp_rollout_id(samples: list[list[Sample]], rollout_id: int) -> None:
    """Stamp the dispatching rollout id into every sample's metadata.

    sample.metadata flows through the agent function into the harbor /run
    payload, so trial records carry the step that executed them (a
    partial-rollout sample re-dispatched later is re-stamped; its origin
    stays in metadata["start_rollout_id"]).
    """
    for group in samples:
        for sample in group:
            sample.metadata["rollout_id"] = rollout_id


def submit_generate_tasks(state: GenerateState, samples: list[list[Sample]]):
    return [
        asyncio.create_task(
            # submit a group of samples as a single task.
            generate_and_rm_group(
                state,
                group,
                sampling_params=state.sampling_params.copy(),
                evaluation=False,
            )
        )
        for group in samples
    ]


async def generate_rollout_async(
    state: GenerateState, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    args = state.args
    assert args.rollout_global_dataset

    await dumper_utils.configure_sglang(args)

    # instantiate data filters
    dynamic_filter = load_function(args.dynamic_sampling_filter_path)

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    pendings = set()
    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while len(data) + len(pendings) < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            stamp_rollout_id(samples, rollout_id)
            pendings.update(submit_generate_tasks(state, samples))

        # wait for the generation to finish
        logger.debug(f"[rollout] Waiting on {len(pendings)} pending tasks, data={len(data)}/{target_data_size}")
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        logger.debug(f"[rollout] asyncio.wait returned: {len(done)} done, {len(pendings)} pending")
        for task in done:
            try:
                group: list[Sample] = task.result()
            except Exception as e:
                logger.error(f"[rollout] Task raised exception: {e!r}", exc_info=True)
                continue

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(state, pendings, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()

    if f := load_function(args.rollout_sample_filter_path):
        f(args, data)
    # There can be circumstances where users want to process all samples including filtered ones.
    if f := load_function(args.rollout_all_samples_process_path):
        f(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples
