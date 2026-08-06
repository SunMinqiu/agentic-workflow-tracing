import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from agent_io_tracing.adapters.llm_trace import (
    cache_key,
    runtime_vendor,
    serving_config,
    trace_cache_config,
)
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
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="analysis complete",
                    reasoning="checked the spectrum",
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
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
            "temperature": 0.2,
            "max_tokens": 80,
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
    assert prompts[0]["request"]["messages"] == messages
    assert prompts[0]["request_params"] == {
        "temperature": 0.2,
        "max_tokens": 80,
    }
    assert prompts[0]["response_text"] == "analysis complete"
    assert prompts[0]["response"]["reasoning"] == "checked the spectrum"


def test_full_prompt_cache_key_is_stable_for_mapping_key_order():
    first = [{"role": "user", "content": {"b": 2, "a": 1}}]
    second = [{"content": {"a": 1, "b": 2}, "role": "user"}]
    assert cache_key("openai", "model", first) == cache_key("openai", "model", second)
    assert cache_key("openai", "model", first) != cache_key(
        "openai", "other-model", first
    )


def test_serving_arm_manifest_is_stamped_without_losing_request_controls(
    monkeypatch,
):
    monkeypatch.setenv(
        "KVCACHE_ARM_JSON",
        json.dumps({
            "arm": "A1_prefix",
            "backend": "vllm",
            "resolved": {
                "block_size": 16,
                "prefix_match_unit": 16,
                "kv_cache_memory_bytes": 4_000_000_000,
            },
        }),
    )

    config = trace_cache_config({"prompt_cache_key": "group-a"})

    assert config["prompt_cache_key"] == "group-a"
    assert config["serving"]["arm"] == "A1_prefix"
    assert config["serving"]["resolved"]["block_size"] == 16
    assert serving_config()["backend"] == "vllm"
    assert runtime_vendor("openai", "Qwen3.6-27B") == "vLLM"


def test_invalid_serving_manifest_is_visible_in_trace(monkeypatch):
    monkeypatch.setenv("KVCACHE_ARM_JSON", "{bad json")

    config = trace_cache_config()

    assert "manifest_error" in config["serving"]


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
