"""Custom generate function that wraps the agentic tool-call flow with
logprob equivalence verification.

After the agent finishes its multi-turn conversation through the session
server, this function replays each turn by sending the same messages
directly to SGLang ``/v1/chat/completions`` with ``temperature=0``
(greedy, no ``input_ids`` — fresh tokenization from scratch) and
verifies that:

1. **prompt_token_ids** match exactly (core TITO invariant)
2. **output_token_logprobs** match within tolerance (same decode path,
   same prompt → same logprobs)

Both the session path and the replay path use the decode phase, so
logprobs are computed via the same GPU kernels.  Any mismatch means
TITO produced different prompt tokens than a fresh tokenization.

When ``use_rollout_routing_replay`` is enabled (MoE models), routed
expert arrays are also compared.
"""

import argparse
import logging
import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import pybase64
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    compute_samples_from_openai_records,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

LOGPROB_TOL = 1e-5


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Run the agent, collect session records, then re-prefill and compare."""

    # === Step 1: normal agentic flow ===
    tracer = await OpenAIEndpointTracer.create(input.args)

    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert (
        custom_agent_function is not None
    ), f"Custom agent function {input.args.custom_agent_function_path} not found"

    agent_metadata = await custom_agent_function(
        base_url=tracer.base_url,
        prompt=input.sample.prompt,
        request_kwargs=build_chat_request_kwargs(input.sampling_params),
        metadata=input.sample.metadata,
    )

    records, session_metadata = await tracer.collect_records()

    if not records:
        logger.warning("No model calls recorded for sample")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    # === Step 2: re-prefill verification for each turn ===
    assert len(records) >= 2, f"Expected at least 2 turns for TITO verification, got {len(records)}"

    sglang_url = f"http://{input.args.sglang_router_ip}:{input.args.sglang_router_port}"
    tokenizer = input.state.tokenizer
    use_r3 = getattr(input.args, "use_rollout_routing_replay", False)

    for i, record in enumerate(records):
        await _verify_logprob_equivalence(
            sglang_url,
            record,
            tokenizer,
            use_r3,
            turn_idx=i,
        )

    # === Step 3: session-level verification ===
    mismatch = session_metadata.get("tito_session_mismatch")
    assert mismatch == [], f"tito_session_mismatch is not empty: {mismatch}"

    accumulated = session_metadata.get("accumulated_token_ids")
    assert accumulated and len(accumulated) > 0, "accumulated_token_ids is empty"

    logger.info(
        "Logprob equivalence verified: %d turns, %d accumulated tokens",
        len(records),
        len(accumulated),
    )

    # === Step 4: build samples as usual (same as agentic_tool_call.generate) ===
    samples = compute_samples_from_openai_records(
        input.args,
        input.sample,
        records,
        tokenizer,
        accumulated_token_ids=accumulated,
        max_trim_tokens=session_metadata.get("max_trim_tokens", 0),
    )

    for s in samples:
        s.metadata.update(agent_metadata or {})

    if not input.args.generate_multi_samples:
        samples = merge_samples(samples, tokenizer)
        samples.metadata.update(session_metadata)
    else:
        samples[-1].metadata.update(session_metadata)
    return GenerateFnOutput(samples=samples)


async def _verify_logprob_equivalence(
    sglang_url: str,
    record,
    tokenizer,
    use_r3: bool,
    turn_idx: int,
) -> None:
    """Replay a turn via /v1/chat/completions (fresh tokenization) and compare.

    Sends the same messages directly to SGLang with temperature=0 and no
    ``input_ids``, so SGLang tokenizes from scratch.  With greedy decoding
    and identical prompt tokens, the output tokens and logprobs must match.
    """
    choice = record.response["choices"][0]
    session_prompt_ids = choice["prompt_token_ids"]
    session_output_logprobs = choice["meta_info"]["output_token_logprobs"]
    session_output_ids = [t[1] for t in session_output_logprobs]

    if not session_output_ids:
        logger.warning("Turn %d: no output tokens, skipping verification", turn_idx)
        return

    # Step A: replay the same messages directly to SGLang (no input_ids = fresh tokenization)
    req = record.request
    payload = {
        "messages": req["messages"],
        "temperature": 0,
        "max_tokens": req.get("max_tokens", 1024),
        "logprobs": True,
        "return_prompt_token_ids": True,
        "return_meta_info": True,
        "no_stop_trim": False,
    }
    if req.get("tools"):
        payload["tools"] = req["tools"]
    if req.get("tool_choice"):
        payload["tool_choice"] = req["tool_choice"]
    if use_r3:
        payload["return_routed_experts"] = True

    replay_resp = await post(f"{sglang_url}/v1/chat/completions", payload)
    replay_choice = replay_resp["choices"][0]

    # Step B: verify prompt_token_ids match (core TITO invariant)
    replay_prompt_ids = replay_choice["prompt_token_ids"]
    assert session_prompt_ids == replay_prompt_ids, (
        f"Turn {turn_idx}: prompt_token_ids mismatch — "
        f"session has {len(session_prompt_ids)} tokens, "
        f"fresh replay has {len(replay_prompt_ids)} tokens"
    )

    # Step C: compare output_token_logprobs
    replay_output_logprobs = replay_choice["meta_info"]["output_token_logprobs"]
    replay_output_ids = [t[1] for t in replay_output_logprobs]

    # With identical prompt and greedy decoding, output tokens must match
    assert session_output_ids == replay_output_ids, (
        f"Turn {turn_idx}: output token IDs mismatch — "
        f"session={session_output_ids[:10]}..., replay={replay_output_ids[:10]}..."
    )

    mismatches = []
    for j, (replay_entry, session_entry) in enumerate(
        zip(replay_output_logprobs, session_output_logprobs, strict=True)
    ):
        replay_logprob = replay_entry[0]
        session_logprob = session_entry[0]

        if replay_logprob is None or session_logprob is None:
            continue
        if not math.isclose(replay_logprob, session_logprob, abs_tol=LOGPROB_TOL):
            mismatches.append(
                f"  token {j}: logprob {replay_logprob:.8f} vs {session_logprob:.8f} "
                f"(diff={abs(replay_logprob - session_logprob):.2e})"
            )

    assert not mismatches, f"Turn {turn_idx}: logprob mismatches:\n" + "\n".join(mismatches)

    # Step D: verify routed_experts (R3) if available
    if use_r3:
        session_re = choice["meta_info"].get("routed_experts")
        replay_re = replay_choice["meta_info"].get("routed_experts")
        if session_re is not None or replay_re is not None:
            assert (
                session_re is not None and replay_re is not None
            ), f"Turn {turn_idx}: routed_experts present on one side but not the other"
            s_arr = np.frombuffer(pybase64.b64decode(session_re.encode("ascii")), dtype=np.int32)
            f_arr = np.frombuffer(pybase64.b64decode(replay_re.encode("ascii")), dtype=np.int32)
            np.testing.assert_array_equal(
                s_arr,
                f_arr,
                err_msg=f"Turn {turn_idx}: routed_experts mismatch",
            )

    logger.info(
        "Turn %d: logprob equivalence verified (%d prompt + %d output tokens)",
        turn_idx,
        len(session_prompt_ids),
        len(session_output_ids),
    )


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--custom-agent-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true", default=False)


generate.add_arguments = _add_arguments


def build_chat_request_kwargs(sampling_params: dict[str, Any]) -> dict[str, Any]:
    """Same as agentic_tool_call.build_chat_request_kwargs."""
    request_kwargs = dict(sampling_params)
    key_map = {
        "max_new_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "sampling_seed": "seed",
    }
    for src, dst in key_map.items():
        if src in request_kwargs:
            if dst not in request_kwargs:
                request_kwargs[dst] = request_kwargs[src]
            request_kwargs.pop(src, None)

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in request_kwargs.items() if key in allowed_keys and value is not None}
