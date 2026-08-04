import json
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agent_io_tracing.adapters.genomas.logger import (
    GenoMASToolLogger,
    _install_openai_stream_timing,
)


def _read_events(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _client():
    return SimpleNamespace(
        config=SimpleNamespace(provider="openai", model_name="test-model")
    )


def _result(trace_timing=None):
    result = {
        "content": "answer",
        "usage": {"input_tokens": 1200, "output_tokens": 30, "cost": 0},
        "raw_response": {
            "id": "req_test",
            "usage": {"prompt_tokens_details": {"cached_tokens": 896}},
        },
    }
    if trace_timing is not None:
        result["_trace_timing"] = trace_timing
    return result


def test_genomas_logger_emits_real_stream_timing_events(tmp_path, monkeypatch):
    monkeypatch.delenv("GENOMAS_LLM_CACHE_PATH", raising=False)
    monkeypatch.delenv("GENOMAS_LLM_REPLAY", raising=False)
    logger = GenoMASToolLogger(tmp_path)
    started = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=2)
    start_ms = started.timestamp() * 1000

    logger.on_llm_call(
        _client(),
        [{"role": "user", "content": "hello"}],
        _result({
            "response_headers_ms": start_ms + 100,
            "first_token_ms": start_ms + 350,
            "last_token_ms": start_ms + 1900,
        }),
        started,
        ended,
        cache_key="hash",
        run_id="run-1",
        started_monotonic_ns=10,
        ended_monotonic_ns=20,
    )

    events = _read_events(tmp_path / "pi_events.jsonl")
    assert [event["type"] for event in events] == [
        "message_request_start",
        "message_start",
        "message_response_headers",
        "message_first_token",
        "message_last_token",
        "message_end",
    ]
    assert events[-1]["stream_timing_available"] is True
    assert events[-1]["provider_request_id"] == "req_test"
    assert events[-1]["message"]["usage"]["cacheRead"] == 896


def test_genomas_logger_does_not_fabricate_first_token(tmp_path, monkeypatch):
    monkeypatch.delenv("GENOMAS_LLM_CACHE_PATH", raising=False)
    monkeypatch.delenv("GENOMAS_LLM_REPLAY", raising=False)
    logger = GenoMASToolLogger(tmp_path)
    started = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)

    logger.on_llm_call(
        _client(),
        [{"role": "user", "content": "hello"}],
        _result(),
        started,
        started + timedelta(seconds=1),
        run_id="run-2",
    )

    events = _read_events(tmp_path / "pi_events.jsonl")
    assert "message_first_token" not in [event["type"] for event in events]
    assert events[-1]["stream_timing_available"] is False


def test_genomas_openai_compatible_stream_produces_real_timing(monkeypatch):
    monkeypatch.setenv("GENOMAS_CAPTURE_STREAM_TIMING", "1")

    class Stream:
        def __init__(self):
            self._chunks = iter([
                SimpleNamespace(
                    id="req-1",
                    model="qwen3.6-35b",
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content="hello", reasoning_content=None)
                    )],
                    usage=None,
                ),
                SimpleNamespace(
                    id="req-1",
                    model="qwen3.6-35b",
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=1200,
                        completion_tokens=1,
                        model_dump=lambda: {
                            "prompt_tokens": 1200,
                            "completion_tokens": 1,
                            "prompt_tokens_details": {"cached_tokens": 960},
                        },
                    ),
                ),
            ])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration

    class Completions:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True
            assert kwargs["stream_options"] == {"include_usage": True}
            return Stream()

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=Completions())

    class OpenAIClient:
        config = SimpleNamespace(extra_message_params={})
        model_name = "qwen3.6-35b"

        def __init__(self):
            self.client = Client()

        async def generate_completion(self, messages):
            raise AssertionError("stream timing hook was not installed")

        def _format_response(self, content, input_tokens, output_tokens, raw_response):
            return {
                "content": content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "raw_response": raw_response,
            }

        def handle_exception(self, exc):
            raise exc

    module = SimpleNamespace(
        OpenAIClient=OpenAIClient,
        check_recent_openai_model=lambda _: False,
    )
    assert _install_openai_stream_timing(module) is True
    result = asyncio.run(OpenAIClient().generate_completion(
        [{"role": "user", "content": "hello"}]
    ))

    assert result["content"] == "hello"
    assert result["usage"] == {"input_tokens": 1200, "output_tokens": 1}
    assert result["raw_response"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 960
    assert result["_trace_timing"]["first_token_ms"] is not None
