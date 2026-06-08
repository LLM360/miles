import argparse

import pytest

from miles.rollout.generate_hub.agentic_tool_call import generate
from miles.utils.misc import function_registry
from miles.utils.test_utils import mock_tools
from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.types import Sample
from tests.fast.fixtures.generation_fixtures import generation_env, make_sample, run_generate

_ = generation_env

PROMPT = [{"role": "user", "content": "What is 1+1?"}]
RESPONSE = "The answer is 2."


@pytest.fixture
def variant():
    return "agentic_tool_call_single_sample"


def _parse_agentic_args(*argv: str):
    parser = argparse.ArgumentParser()
    generate.add_arguments(parser)
    return parser.parse_args(list(argv))


def test_custom_agent_config_json_object_is_parsed():
    args = _parse_agentic_args(
        "--custom-agent-config-json",
        '{"step_limit": 3, "nested": {"enabled": true}}',
    )

    assert args.custom_agent_config == {"step_limit": 3, "nested": {"enabled": True}}


@pytest.mark.parametrize("value", ["{", "[]", "1", '"scalar"', "true"])
def test_custom_agent_config_json_rejects_invalid_or_non_object(value, capsys):
    with pytest.raises(SystemExit) as exc_info:
        _parse_agentic_args("--custom-agent-config-json", value)

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "--custom-agent-config-json" in stderr
    assert "valid JSON" in stderr or "JSON object" in stderr


@pytest.mark.parametrize(
    "generation_env",
    [
        {
            "args_kwargs": {
                "extra_argv": [
                    "--custom-agent-config-json",
                    '{"step_limit": 3, "nested": {"mode": "strict"}}',
                ],
            }
        }
    ],
    indirect=True,
)
def test_custom_agent_config_json_is_passed_to_custom_agent_function(variant, generation_env):
    seen = {}

    async def agent(base_url, prompt, request_kwargs=None, metadata=None, agent_config=None):
        seen["agent_config"] = agent_config
        seen["metadata"] = metadata
        return await mock_tools.run_agentic_tool_call(
            base_url=base_url,
            prompt=prompt,
            request_kwargs=request_kwargs,
            metadata=metadata,
            agent_config=agent_config,
        )

    sample = make_sample(prompt=PROMPT)
    sample.metadata = {"task_id": "task-1", "agent_config": {"step_limit": 9}}
    generation_env.args.custom_agent_function_path = "test:agent-with-config"
    generation_env.mock_server.process_fn = lambda _: ProcessResult(text=RESPONSE, finish_reason="stop")

    with function_registry.temporary("test:agent-with-config", agent):
        result = run_generate(generation_env, sample, variant=variant)

    assert result.sample.status == Sample.Status.COMPLETED
    assert seen["agent_config"] == {"step_limit": 3, "nested": {"mode": "strict"}}
    assert seen["metadata"]["task_id"] == "task-1"
    assert seen["metadata"]["agent_config"] == {"step_limit": 9}


def test_absent_custom_agent_config_json_does_not_add_agent_config_kwarg(variant, generation_env):
    seen = {}

    async def agent(base_url, prompt, request_kwargs=None, metadata=None):
        seen["called"] = True
        return await mock_tools.run_agentic_tool_call(
            base_url=base_url,
            prompt=prompt,
            request_kwargs=request_kwargs,
            metadata=metadata,
        )

    generation_env.args.custom_agent_function_path = "test:agent-without-config-kwarg"
    generation_env.mock_server.process_fn = lambda _: ProcessResult(text=RESPONSE, finish_reason="stop")

    with function_registry.temporary("test:agent-without-config-kwarg", agent):
        result = run_generate(generation_env, make_sample(prompt=PROMPT), variant=variant)

    assert seen["called"] is True
    assert result.sample.status == Sample.Status.COMPLETED
