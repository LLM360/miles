from copy import deepcopy

import pytest

from miles.rollout.session.core import _compact_agent_choice, closed_chat_response


@pytest.mark.parametrize("events", [None, [], "invalid", [{"type": "fallback"}]])
def test_only_nonempty_parser_event_lists_survive_compaction(events):
    choice = {
        "message": {"content": "answer"},
        "prompt_token_ids": [1],
        "routed_experts": "large",
        "meta_info": {
            "tool_parser_fallback_events": events,
            "output_token_logprobs": [[-1, 2]],
            "routed_experts": "large",
            "other": 1,
        },
    }
    original = deepcopy(choice)
    compact = _compact_agent_choice(choice)
    expected = {"message": {"content": "answer"}}
    if isinstance(events, list) and events:
        expected["meta_info"] = {"tool_parser_fallback_events": events}
    assert compact == expected
    assert choice == original


def test_late_closed_response_keeps_diagnostics_without_routing():
    import json

    response = {
        "choices": [
            {
                "message": {"content": "answer"},
                "prompt_token_ids": [1],
                "routed_experts": "large",
                "meta_info": {"reasoning_parser_fallback_events": [{"type": "fallback"}], "routed_experts": "large"},
            }
        ]
    }
    result = {"status_code": 200, "response_body": json.dumps(response).encode(), "headers": {}}
    out = json.loads(closed_chat_response(result, False, compact=True).body)
    assert out["choices"] == [
        {"message": {"content": "answer"}, "meta_info": {"reasoning_parser_fallback_events": [{"type": "fallback"}]}}
    ]
