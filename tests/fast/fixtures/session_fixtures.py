from __future__ import annotations

import base64
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np

from miles.rollout.session.config import SessionServerConfig
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer


def make_session_server_config(**overrides: Any) -> SessionServerConfig:
    defaults: dict[str, Any] = dict(
        host="127.0.0.1",
        port=0,
        instance_id=None,
        backend_url="http://127.0.0.1:0",
        timeout=30,
        hf_checkpoint=None,
        chat_template_path=None,
        tito_model="default",
        apply_chat_template_kwargs=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
        num_layers=None,
        moe_router_topk=None,
        save_debug_trajectory_data=None,
        lora_rank=0,
        lora_adapter_path=None,
        use_session_server=None,
        session_message_matcher="strict",
        pause_generation_mode=None,
        session_sample_picker_path=None,
        session_sample_postprocessor_path=None,
    )
    defaults.update(overrides)
    return SessionServerConfig(**defaults)


@contextmanager
def mock_requested_routing():
    """Supply one-layer/top-one routing for precisely the requested token rows."""
    original = MockSGLangServer._compute_chat_completions_response

    def response(server, payload):
        result = original(server, payload)
        if payload.get("return_routed_experts"):
            meta = result["choices"][0]["meta_info"]
            rows = len(payload["input_ids"]) + len(meta["output_token_logprobs"]) - 1
            rows -= payload.get("routed_experts_start_len", 0)
            meta["routed_experts"] = base64.b64encode(np.zeros((rows, 1, 1), dtype=np.int32).tobytes()).decode()
        return result

    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=response):
        yield
