import json
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agent_io_tracing.adapters.genomas.logger import (
    GenoMASToolLogger,
    _build_exact_reference_block,
    _build_exact_prompt_prefix,
    _cohort_from_messages,
    _insert_reference_before_task,
    _install_openai_stream_timing,
    _make_async_wrapper,
    _ordered_reference_text,
    _prepend_prompt_prefix,
    _strip_cohort_reference,
    _strip_prompt_prefix,
    _to_pi_usage,
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


def _result(trace_timing=None, server_metrics=None):
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
    if server_metrics is not None:
        result["_trace_server_metrics"] = server_metrics
    return result


class _CharacterTokenizerEndpoint:
    """ASCII test tokenizer with one token per character."""

    def post_json(self, path, payload):
        if path == "/tokenize":
            return {"tokens": [ord(char) for char in payload["prompt"]]}
        if path == "/detokenize":
            return {"prompt": "".join(chr(token) for token in payload["tokens"])}
        raise AssertionError(path)


def test_genomas_builds_exact_server_tokenized_prompt_prefix():
    prefix, metadata = _build_exact_prompt_prefix(
        32768,
        _CharacterTokenizerEndpoint(),
        "test-model",
    )

    assert len(prefix) == 32768
    assert metadata["target_tokens"] == 32768
    assert metadata["actual_tokens"] == 32768
    assert metadata["characters"] == 32768
    assert prefix.startswith("[GENOMAS BENCHMARK REFERENCE PREFIX]")
    assert prefix.endswith("[END GENOMAS BENCHMARK REFERENCE PREFIX]\n\n")


def test_genomas_prompt_prefix_is_copied_before_first_message():
    original = [
        {"role": "system", "content": "original system"},
        {"role": "user", "content": "task"},
    ]

    padded = _prepend_prompt_prefix(original, "fixed prefix\n")

    assert padded[0]["content"] == "fixed prefix\noriginal system"
    assert padded[1] is original[1]
    assert original[0]["content"] == "original system"
    assert _strip_prompt_prefix(padded, "fixed prefix\n") == original


def test_genomas_cohort_reference_is_role_ordered_and_inserted_before_task():
    paper = """Paper title and abstract.

INTRODUCTION
intro evidence

RESULTS
result evidence

DISCUSSION
discussion evidence

METHODS
method evidence
"""
    geo = _ordered_reference_text(paper, "GEOAgent")
    expert = _ordered_reference_text(paper, "DomainExpertAgent")

    assert geo.index("METHODS") < geo.index("RESULTS")
    assert expert.index("RESULTS") < expert.index("METHODS")

    source = [{
        "role": "user",
        "content": 'cohort: "GSE131528"\n\n**Task**:\nDo the original work.',
    }]
    block, metadata = _build_exact_reference_block(
        source_text=geo * 100,
        title="Study paper",
        cohort="GSE131528",
        role="GEOAgent",
        target_tokens=2048,
        endpoint=_CharacterTokenizerEndpoint(),
        model="test-model",
    )
    injected = _insert_reference_before_task(source, block)

    assert metadata["actual_tokens"] == 2048
    assert _cohort_from_messages(source) == "GSE131528"
    assert injected is not source
    assert source[0]["content"].count("GENOMAS COHORT REFERENCE") == 0
    content = injected[0]["content"]
    assert content.index("[GENOMAS COHORT REFERENCE]") < content.index("**Task**")
    assert content.rstrip().endswith("Do the original work.")
    assert _strip_cohort_reference(injected) == source


def test_genomas_reference_skips_reviewer_and_preserves_upstream_messages(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("GENOMAS_PROMPT_PREFIX_TOKENS", raising=False)
    monkeypatch.delenv("GENOMAS_REFERENCE_TOKENS", raising=False)
    monkeypatch.delenv("GENOMAS_LLM_CACHE_PATH", raising=False)
    monkeypatch.delenv("GENOMAS_LLM_REPLAY", raising=False)
    logger = GenoMASToolLogger(tmp_path)
    logger._reference_roles = {"GEOAgent", "DomainExpertAgent"}
    metadata = {
        "cohort": "GSE123086",
        "role": "GEOAgent",
        "actual_tokens": 8000,
        "sha256": "abc",
    }
    logger._cohort_references[("GSE123086", "GEOAgent")] = (
        "\n\n[GENOMAS COHORT REFERENCE]\npaper\n"
        "[END GENOMAS COHORT REFERENCE]\n"
        "Follow the original GenoMAS task below.\n\n",
        metadata,
    )
    source = [{
        "role": "user",
        "content": 'cohort: "GSE123086"\n\n**Task**:\nRun preprocessing.',
    }]

    geo_messages, geo_metadata = logger.add_cohort_reference(source, "GEOAgent")
    reviewer_messages, reviewer_metadata = logger.add_cohort_reference(
        source, "CodeReviewerAgent"
    )

    assert geo_messages is not source
    assert geo_metadata == metadata
    assert reviewer_messages is source
    assert reviewer_metadata is None
    assert "GENOMAS COHORT REFERENCE" not in source[0]["content"]


def test_genomas_wrapper_sends_and_records_padded_prompt(tmp_path, monkeypatch):
    monkeypatch.delenv("GENOMAS_PROMPT_PREFIX_TOKENS", raising=False)
    monkeypatch.delenv("GENOMAS_NOCACHE", raising=False)
    monkeypatch.delenv("GENOMAS_LLM_CACHE_PATH", raising=False)
    monkeypatch.delenv("GENOMAS_LLM_REPLAY", raising=False)
    logger = GenoMASToolLogger(tmp_path)
    logger._prompt_prefix = "fixed prefix\n"
    seen = []

    async def original(client, messages):
        seen.append(messages)
        return _result()

    wrapped = _make_async_wrapper(original, logger)
    source = [{"role": "system", "content": "original system"}]
    asyncio.run(wrapped(_client(), source))

    assert seen[0][0]["content"] == "fixed prefix\noriginal system"
    assert source[0]["content"] == "original system"
    prompts = _read_events(tmp_path / "messages.jsonl")
    assert prompts[0]["messages"][0]["content"] == (
        "fixed prefix\noriginal system"
    )


def test_genomas_nocache_tag_precedes_entire_prompt_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("GENOMAS_PROMPT_PREFIX_TOKENS", raising=False)
    monkeypatch.setenv("GENOMAS_NOCACHE", "1")
    monkeypatch.delenv("GENOMAS_LLM_CACHE_PATH", raising=False)
    monkeypatch.delenv("GENOMAS_LLM_REPLAY", raising=False)
    logger = GenoMASToolLogger(tmp_path)
    logger._prompt_prefix = "fixed prefix\n"
    seen = []

    async def original(client, messages):
        seen.append(messages)
        return _result()

    wrapped = _make_async_wrapper(original, logger)
    asyncio.run(wrapped(
        _client(),
        [{"role": "system", "content": "original system"}],
    ))

    sent = seen[0][0]["content"]
    assert sent.startswith("[nocache:")
    assert "\nfixed prefix\noriginal system" in sent
    recorded = _read_events(tmp_path / "messages.jsonl")[0]
    assert recorded["messages"][0]["content"] == (
        "fixed prefix\noriginal system"
    )


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
        _result(
            {
                "response_headers_ms": start_ms + 100,
                "first_token_ms": start_ms + 350,
                "last_token_ms": start_ms + 1900,
            },
            {"time_to_first_token_ms": 320.0, "queue_time_ms": 12.0},
        ),
        started,
        ended,
        cache_key="hash",
        run_id="run-1",
        started_monotonic_ns=10,
        ended_monotonic_ns=20,
        request_params={"temperature": 0, "max_tokens": 30},
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
    assert events[-1]["server_metrics"]["queue_time_ms"] == 12.0
    prompts = _read_events(tmp_path / "messages.jsonl")
    assert prompts[0]["request_params"] == {
        "temperature": 0,
        "max_tokens": 30,
    }
    assert prompts[0]["request"]["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    assert prompts[0]["response_text"] == "answer"


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


def test_genomas_explicit_zero_cached_tokens_is_measured():
    usage = _to_pi_usage({
        "usage": {"input_tokens": 100, "output_tokens": 10},
        "raw_response": {
            "usage": {"prompt_tokens_details": {"cached_tokens": 0}}
        },
        "_trace_cache": {"attributable": True, "cacheRead": 64},
    })
    assert usage["cacheRead"] == 0
    assert usage["cacheReadAvailable"] is True
    assert usage["cacheReadSource"] == "response"


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
                    model_extra={
                        "metrics": SimpleNamespace(
                            model_dump=lambda: {
                            "time_to_first_token_ms": 11.0,
                            "mean_itl_ms": 4.0,
                            },
                        ),
                    },
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
    assert result["_trace_server_metrics"]["mean_itl_ms"] == 4.0
