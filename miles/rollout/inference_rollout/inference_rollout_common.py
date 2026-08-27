import asyncio
import logging
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from miles.rollout.base_types import (
    GenerateFnInput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.generate_hub.single_turn import generate
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _flatten_samples(value) -> list[Sample]:
    if isinstance(value, Sample):
        return [value]
    if isinstance(value, (list, tuple)):
        return [sample for item in value for sample in _flatten_samples(item)]
    return []


def is_responsive_generation_group(group: list[Sample]) -> bool:
    """Return whether every sample proves that generation made progress.

    Generation health is intentionally separate from training eligibility.
    ``ABORTED`` samples can contain a complete token chain that a downstream
    filter must reject for training (for example, because an attempt reached
    its token cap), while still proving that the generation backend responded.
    ``PENDING`` samples and terminal samples without generated tokens do not
    prove liveness.
    """

    samples = _flatten_samples(group)
    if not samples:
        return False

    terminal_statuses = {
        Sample.Status.COMPLETED,
        Sample.Status.TRUNCATED,
        Sample.Status.ABORTED,
        Sample.Status.FAILED,
    }
    return all(
        sample.status in terminal_statuses
        and sample.response_length > 0
        and len(sample.tokens) >= sample.response_length
        for sample in samples
    )


@dataclass
class GenerationHealthTracker:
    """Fail a rollout as soon as a complete wave has no responsive group.

    The tracker resets after each healthy wave. This makes the health decision
    online: an unbounded prompt source cannot hide a dead generation backend by
    continuously supplying replacement prompts. Whether a responsive group is
    suitable for training remains the dynamic filter's responsibility.
    """

    submitted_groups: int = 0
    returned_groups: int = 0
    responsive_groups: int = 0
    unresponsive_groups: int = 0
    task_exceptions: int = 0

    def record_submitted(self, count: int) -> None:
        if count < 0:
            raise ValueError("submitted group count must be non-negative")
        self.submitted_groups += count

    def record_returned(self, group: list[Sample]) -> None:
        self.returned_groups += 1
        if is_responsive_generation_group(group):
            self.responsive_groups += 1
        else:
            self.unresponsive_groups += 1

    def record_task_exception(self) -> None:
        self.task_exceptions += 1

    def close_window(self, *, pending_groups: int, allow_incomplete: bool = False) -> None:
        """Validate and reset a drained health window.

        ``allow_incomplete`` is used only after the caller intentionally aborted
        surplus requests because the target batch was already filled.
        """

        if pending_groups < 0:
            raise ValueError("pending group count must be non-negative")
        if self.submitted_groups == 0 or pending_groups:
            return

        accounted_groups = self.returned_groups + self.task_exceptions
        missing_groups = max(0, self.submitted_groups - accounted_groups)
        if not allow_incomplete and accounted_groups != self.submitted_groups:
            raise RuntimeError(
                "rollout generation health accounting is inconsistent: "
                f"submitted_groups={self.submitted_groups}, returned_groups={self.returned_groups}, "
                f"task_exceptions={self.task_exceptions}, missing_groups={missing_groups}"
            )

        if self.responsive_groups == 0:
            raise RuntimeError(
                "rollout generation collapsed before producing a responsive group: "
                f"submitted_groups={self.submitted_groups}, returned_groups={self.returned_groups}, "
                f"responsive_groups={self.responsive_groups}, "
                f"unresponsive_groups={self.unresponsive_groups}, "
                f"task_exceptions={self.task_exceptions}, missing_groups={missing_groups}"
            )

        self.reset_window()

    def reset_window(self) -> None:
        self.submitted_groups = 0
        self.returned_groups = 0
        self.responsive_groups = 0
        self.unresponsive_groups = 0
        self.task_exceptions = 0


def raise_if_generation_collapsed(
    submitted_groups: int,
    returned_groups: list[list[Sample]],
) -> None:
    """Compatibility wrapper for callers that validate one completed wave."""

    tracker = GenerationHealthTracker()
    tracker.record_submitted(submitted_groups)
    for group in returned_groups:
        tracker.record_returned(group)
    for _ in range(max(0, submitted_groups - len(returned_groups))):
        tracker.record_task_exception()
    tracker.close_window(pending_groups=0)


class GenerateState:
    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.generate_fn_semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = compute_sampling_params(
            args,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
        )

        self._generate_functions: dict[str, Any] = {}
        default_path = args.custom_generate_function_path
        self.generate_function = load_generate_function(default_path) or generate
        if default_path:
            self._generate_functions[default_path] = self.generate_function

        self.reset()

    def resolve_generate_function(self, path: str | None):
        """Resolve one dataset's generator without mutating shared state."""

        if path is None:
            return self.generate_function
        if path not in self._generate_functions:
            generate_function = load_generate_function(path)
            if generate_function is None:
                raise ValueError(f"could not load custom generate function {path!r}")
            self._generate_functions[path] = generate_function
        return self._generate_functions[path]

    def reset(self) -> None:
        self.aborted = False


async def generate_and_rm(
    state: GenerateState,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    generate_function=None,
) -> Sample | list[Sample]:
    args = state.args

    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    # generate
    log_prefix = f"[sample={getattr(sample, 'index', '?')}]"
    logger.debug(f"{log_prefix} Waiting for semaphore...")
    async with state.generate_fn_semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        logger.debug(f"{log_prefix} Acquired semaphore, calling generate_function")
        selected_generate_function = generate_function or state.generate_function
        output = await selected_generate_function(
            GenerateFnInput(
                state=state,
                sample=sample,
                sampling_params=deepcopy(sampling_params),
                evaluation=evaluation,
            )
        )
        sample = output.samples
        logger.debug(f"{log_prefix} generate_function returned")

    # TODO change to `if not args.group_rm: do reward model` for more clarity after the refactor below
    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # TODO: unify the two branches into one if we decide to use list as output type
    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        await batched_async_rm(args, samples_need_reward, inplace_set_reward_field=True)
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    logger.debug(f"{log_prefix} generate_and_rm complete")
    return sample


async def generate_and_rm_group(
    state: GenerateState, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    args = state.args

    if state.aborted:
        return group

    log_prefix = f"[group indices={[getattr(s, 'index', '?') for s in group]}]"
    logger.debug(f"{log_prefix} Starting group with {len(group)} samples")
    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            current_sampling_params["sampling_seed"] = args.rollout_seed + idx
        tasks.append(
            asyncio.create_task(generate_and_rm(state, sample, current_sampling_params, evaluation=evaluation))
        )

    try:
        group = await asyncio.gather(*tasks)
    except BaseException:
        # ``asyncio.gather`` propagates the first failure without waiting for
        # sibling tasks. Cancel and drain them so a failed group cannot leave
        # generation work running after the rollout has been reset.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    logger.debug(f"{log_prefix} [group] All {len(group)} samples completed")
    if state.aborted:
        return group

    if args.group_rm:
        await batched_async_rm(args, group, inplace_set_reward_field=True)

    return group


def compute_sampling_params(
    args,
    *,
    # after unifying configuration, this can be further refactored
    temperature,
    top_p,
    top_k,
    max_new_tokens,
):
    return dict(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )


class InferenceRolloutFn:
    def __init__(self, input: RolloutFnConstructorInput):
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        self.eval_prompt_dataset_cache = {}

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        return await self._call_train(input)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        from miles.rollout.inference_rollout.inference_rollout_train import generate_rollout_async

        output, aborted_samples = await generate_rollout_async(
            self.state, input.rollout_id, self.data_source.get_samples
        )
        self.data_source.add_samples(aborted_samples)
        return output

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        from miles.rollout.inference_rollout.inference_rollout_eval import eval_rollout_single_dataset

        assert not self.state.args.group_rm, "Group RM is not supported for eval rollout"

        coros = []
        for dataset_cfg in getattr(self.state.args, "eval_datasets", []) or []:
            coros.append(eval_rollout_single_dataset(self.state, dataset_cfg, self.eval_prompt_dataset_cache))
        results_list = await asyncio.gather(*coros)
        results = {k: v for r in results_list for k, v in r.items()}
        return RolloutFnEvalOutput(data=results)
