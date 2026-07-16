import argparse

import pytest
from tests.fast.fixtures.generation_fixtures import generation_env, listify, make_sample, run_generate

from miles.rollout.generate_hub.agentic_tool_call import generate
from miles.utils.misc import function_registry
from miles.utils.test_utils import mock_tools
from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.types import Sample

_ = generation_env

PROMPT = [{"role": "user", "content": "What is 1+1?"}]
RESPONSE = "The answer is 2."
AGENTIC_VARIANTS = ["agentic_tool_call_single_sample"]
HARBOR_EXIT_STATUSES_TO_TRUNCATE = [
    "BadRequestError",
    "VerifierTimeout",
    "OutputLengthExceededError",
    "AgentTimeout",
    "Cancelled",
]


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


@pytest.mark.parametrize("variant", AGENTIC_VARIANTS)
@pytest.mark.parametrize("exit_status", HARBOR_EXIT_STATUSES_TO_TRUNCATE)
def test_harbor_exit_status_with_records_is_truncated(variant, generation_env, exit_status):
    generation_env.mock_server.process_fn = lambda _: ProcessResult(text=RESPONSE, finish_reason="stop")
    mock_tools.AGENTIC_RETURN_METADATA = {"exit_status": exit_status, "reward": 0.0}

    result = run_generate(generation_env, make_sample(prompt=PROMPT), variant=variant)

    samples = listify(result.sample)
    assert samples[-1].status == Sample.Status.TRUNCATED
    assert all(s.metadata["exit_status"] == exit_status for s in samples)


@pytest.mark.parametrize("variant", AGENTIC_VARIANTS)
def test_k8s_internal_infra_error_is_not_forced_truncated(variant, generation_env):
    generation_env.mock_server.process_fn = lambda _: ProcessResult(text=RESPONSE, finish_reason="stop")
    mock_tools.AGENTIC_RETURN_METADATA = {"exit_status": "_K8sInternalInfraError", "reward": 0.0}

    result = run_generate(generation_env, make_sample(prompt=PROMPT), variant=variant)

    samples = listify(result.sample)
    assert samples[-1].status == Sample.Status.COMPLETED
    assert all(s.metadata["exit_status"] == "_K8sInternalInfraError" for s in samples)


def test_harbor_exit_status_truncates_server_merged_sample(variant, generation_env):
    generation_env.mock_server.process_fn = mock_tools.TwoTurnStub.process_fn
    mock_tools.AGENTIC_RETURN_METADATA = {"exit_status": "VerifierTimeout", "reward": 0.0}

    result = run_generate(generation_env, make_sample(prompt=mock_tools.TwoTurnStub.PROMPT), variant=variant)

    samples = listify(result.sample)
    assert [s.status for s in samples] == [Sample.Status.TRUNCATED]


@pytest.mark.parametrize("variant", AGENTIC_VARIANTS)
@pytest.mark.parametrize("exit_status", ["BadRequestError", "Cancelled"])
def test_harbor_exit_status_without_records_stays_aborted(variant, generation_env, exit_status):
    async def agent_returns_failure_without_model_calls(**kwargs):
        return {
            "exit_status": exit_status,
            "reward": 0.0,
            "agent_metrics": {"agent_timeout_count": 1},
        }

    generation_env.args.custom_agent_function_path = "test:failure-no-records"

    with function_registry.temporary("test:failure-no-records", agent_returns_failure_without_model_calls):
        result = run_generate(generation_env, make_sample(prompt=PROMPT), variant=variant)

    samples = listify(result.sample)
    assert len(samples) == 1
    assert samples[0].status == Sample.Status.ABORTED
    assert samples[0].metadata["exit_status"] == exit_status
    assert samples[0].metadata["agent_metrics"] == {
        "agent_timeout_count": 1,
        "empty_records_count": 1,
    }


@pytest.mark.parametrize("variant", AGENTIC_VARIANTS)
@pytest.mark.parametrize("exit_status", ["BadRequestError", "Cancelled"])
def test_harbor_exit_status_with_zero_active_tokens_is_aborted(variant, generation_env, exit_status):
    generation_env.mock_server.process_fn = lambda _: ProcessResult(text="", finish_reason="stop")
    mock_tools.AGENTIC_RETURN_METADATA = {"exit_status": exit_status, "reward": 0.0}

    result = run_generate(generation_env, make_sample(prompt=PROMPT), variant=variant)

    samples = listify(result.sample)
    assert samples[-1].status == Sample.Status.ABORTED
    assert samples[-1].effective_response_length == 0
    assert all(s.metadata["exit_status"] == exit_status for s in samples)
