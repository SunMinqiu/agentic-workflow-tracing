import json

from agent_io_tracing.replay.bundle import build_bundle
from agent_io_tracing.replay.runner import _peak_concurrency


class FakeEndpoint:
    def tokenize_chat(self, model, messages):
        assert model == "served-model"
        return [len(messages), 10, 20]


def test_build_bundle_joins_usage_and_preserves_arrival_offsets(tmp_path) -> None:
    messages = [
        {
            "run_id": "a",
            "timestamp": 1000,
            "model": "source-model",
            "messages": [{"role": "user", "content": "first"}],
            "request_params": {"temperature": 0},
            "response_text": "one",
        },
        {
            "run_id": "b",
            "timestamp": 1250,
            "model": "source-model",
            "messages": [{"role": "user", "content": "second"}],
        },
    ]
    events = [
        {
            "type": "message_start",
            "run_id": run_id,
            "message": {"timestamp": start},
        }
        for run_id, start in (("a", 1000), ("b", 1250))
    ] + [
        {
            "type": "message_end",
            "run_id": run_id,
            "message": {
                "timestamp": end,
                "usage": {"input": 3, "output": output, "cacheRead": 0},
            },
        }
        for run_id, end, output in (("a", 1500, 4), ("b", 1400, 2))
    ]
    (tmp_path / "messages.jsonl").write_text(
        "\n".join(json.dumps(row) for row in messages),
        encoding="utf-8",
    )
    (tmp_path / "pi_events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in events),
        encoding="utf-8",
    )

    bundle = build_bundle(tmp_path, FakeEndpoint(), "served-model")

    assert bundle["requests"][0]["arrival_offset_ms"] == 0
    assert bundle["requests"][1]["arrival_offset_ms"] == 250
    assert bundle["requests"][0]["original_duration_ms"] == 500
    assert bundle["requests"][0]["max_tokens"] == 4
    assert bundle["requests"][0]["prompt_token_ids"] == [1, 10, 20]
    assert bundle["capture_completeness"]["missing_fields"] == [
        "request_params",
        "response_text",
    ]


def test_peak_concurrency_uses_original_intervals() -> None:
    requests = [
        {"arrival_offset_ms": 0, "original_duration_ms": 100},
        {"arrival_offset_ms": 50, "original_duration_ms": 100},
        {"arrival_offset_ms": 200, "original_duration_ms": 10},
    ]

    assert _peak_concurrency(requests) == 2
