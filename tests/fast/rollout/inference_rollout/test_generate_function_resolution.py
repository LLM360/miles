import asyncio
from argparse import Namespace

from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.utils.types import Sample


def _state(default_generate):
    state = object.__new__(common.GenerateState)
    state.generate_function = default_generate
    state._generate_functions = {"global.generate": default_generate}
    return state


def test_dataset_generate_function_resolution_is_cached_without_state_leak(monkeypatch):
    async def global_generate(_input):
        raise AssertionError("global generator should not be called")

    async def dataset_a_generate(_input):
        raise AssertionError("not executed in this resolver test")

    async def dataset_b_generate(_input):
        raise AssertionError("not executed in this resolver test")

    loaded_paths = []
    functions = {
        "dataset_a.generate": dataset_a_generate,
        "dataset_b.generate": dataset_b_generate,
    }

    def load(path):
        loaded_paths.append(path)
        return functions[path]

    monkeypatch.setattr(common, "load_generate_function", load)
    state = _state(global_generate)

    assert state.resolve_generate_function(None) is global_generate
    assert state.resolve_generate_function("dataset_a.generate") is dataset_a_generate
    assert state.resolve_generate_function("dataset_b.generate") is dataset_b_generate
    assert state.resolve_generate_function("dataset_a.generate") is dataset_a_generate
    assert loaded_paths == ["dataset_a.generate", "dataset_b.generate"]
    assert state.generate_function is global_generate


def test_generate_and_rm_uses_explicit_eval_generator_and_forwards_eval_flag():
    calls = []

    async def global_generate(_input):
        raise AssertionError("global generator leaked into dataset override")

    async def eval_generate(input):
        calls.append(input.evaluation)
        sample = input.sample
        sample.tokens = [1]
        sample.response = "answer"
        sample.response_length = 1
        sample.status = Sample.Status.COMPLETED
        sample.reward = {"success": 1.0}
        return GenerateFnOutput(samples=sample)

    state = _state(global_generate)
    state.args = Namespace(partial_rollout=False, group_rm=False)
    state.generate_fn_semaphore = asyncio.Semaphore(1)
    state.aborted = False
    sample = Sample(index=4)

    output = asyncio.run(
        common.generate_and_rm(
            state,
            sample,
            sampling_params={"max_new_tokens": 8},
            evaluation=True,
            generate_function=eval_generate,
        )
    )

    assert output is sample
    assert output.reward == {"success": 1.0}
    assert calls == [True]
