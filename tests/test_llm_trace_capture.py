from types import SimpleNamespace

from agent_io_tracing.adapters.llm_trace import (
    capture_messages,
    capture_request_params,
    capture_response,
)


def test_capture_messages_preserves_tool_call_structure() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"sample":"A"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]

    assert capture_messages(messages) == messages


def test_capture_request_params_excludes_credentials_and_internal_metadata() -> None:
    params = capture_request_params({
        "temperature": 0,
        "max_tokens": 20,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "api_key": "secret",
        "metadata": {"private": "value"},
    })

    assert params["temperature"] == 0
    assert params["max_tokens"] == 20
    assert "tools" in params
    assert "api_key" not in params
    assert "metadata" not in params


def test_capture_response_preserves_reasoning_and_tools() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    reasoning="inspect evidence",
                    tool_calls=[{"id": "call-1"}],
                ),
                finish_reason="tool_calls",
            )
        ]
    )

    assert capture_response(response) == {
        "reasoning": "inspect evidence",
        "tool_calls": [{"id": "call-1"}],
        "finish_reason": "tool_calls",
    }
