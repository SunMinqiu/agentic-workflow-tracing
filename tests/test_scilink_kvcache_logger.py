import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from agent_io_tracing.adapters.llm_trace import cache_key
from agent_io_tracing.adapters.scilink.logger import LiteLLMToolLogger


def _jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_scilink_logger_writes_joinable_prompt_usage_and_timing(tmp_path):
    logger = LiteLLMToolLogger(tmp_path)
    started = datetime(2026, 7, 29, 12, 0, 0)
    ended = started + timedelta(seconds=2.5)
    messages = [
        {"role": "system", "content": "Analyze carefully."},
        {"role": "user", "content": "Inspect the spectrum."},
    ]
    response = SimpleNamespace(
        id="request-1",
        model="gpt-4o-mini-2024-07-18",
        usage=SimpleNamespace(
            prompt_tokens=1200,
            completion_tokens=80,
            total_tokens=1280,
            prompt_tokens_details=SimpleNamespace(cached_tokens=768),
        ),
    )

    logger.on_llm_success(
        {
            "model": response.model,
            "messages": messages,
            "metadata": {"agent_role": "AnalysisOrchestratorAgent"},
        },
        response,
        started,
        ended,
    )

    events = _jsonl(tmp_path / "pi_events.jsonl")
    prompts = _jsonl(tmp_path / "messages.jsonl")
    assert [event["type"] for event in events] == ["message_start", "message_end"]
    assert events[0]["run_id"] == events[1]["run_id"] == prompts[0]["run_id"]
    assert events[1]["message"]["usage"]["cacheRead"] == 768
    assert events[1]["provider_request_id"] == "request-1"
    assert events[1]["agent_role"] == "AnalysisOrchestratorAgent"
    assert events[1]["message"]["timestamp"] - events[0]["message"]["timestamp"] == 2500
    assert prompts[0]["messages"] == messages


def test_full_prompt_cache_key_is_stable_for_mapping_key_order():
    first = [{"role": "user", "content": {"b": 2, "a": 1}}]
    second = [{"content": {"a": 1, "b": 2}, "role": "user"}]
    assert cache_key("openai", "model", first) == cache_key("openai", "model", second)
    assert cache_key("openai", "model", first) != cache_key(
        "openai", "other-model", first
    )


def test_scilink_emits_litellm_first_token_timestamp(tmp_path):
    logger = LiteLLMToolLogger(tmp_path)
    started = datetime(2026, 7, 29, 12, 0, 0)
    first = started + timedelta(milliseconds=400)
    ended = started + timedelta(seconds=2)
    response = SimpleNamespace(
        id="request-2",
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens=1100,
            completion_tokens=20,
            total_tokens=1120,
        ),
    )

    logger.on_llm_success(
        {
            "model": response.model,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "completion_start_time": first,
        },
        response,
        started,
        ended,
    )

    events = _jsonl(tmp_path / "pi_events.jsonl")
    assert [event["type"] for event in events] == [
        "message_start",
        "message_first_token",
        "message_last_token",
        "message_end",
    ]
    assert events[1]["message"]["timestamp"] - events[0]["message"]["timestamp"] == 400
    assert events[-1]["stream_timing_available"] is True
