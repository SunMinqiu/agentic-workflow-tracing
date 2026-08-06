import json
from pathlib import Path

from agent_io_tracing.analysis.kvcache.segments import (
    SEGMENTS_CSV,
    SEGMENTS_MD,
    analyze_cell_segments,
    _content_breakdown,
    attach_realized,
    classify_content,
    build_radix_trie,
    collect_segments,
    write_markdown,
    write_tables,
)


class _FakeEnc:
    name = "fake"

    def decode(self, tokens):
        return "".join(chr(96 + t) for t in tokens)


def _segments(sequences, roles=None):
    roles = roles or ["r"] * len(sequences)
    root = build_radix_trie(sequences)
    return collect_segments(root, _FakeEnc(), roles)


def test_segments_partition_the_cache_exactly():
    """Segment tokens sum to distinct tokens; segments never overlap."""
    sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 9, 9],
        [1, 2, 3, 4, 5, 6],
        [7, 7, 7],
    ]
    segments = _segments(sequences)
    distinct = {
        tuple(seq[:i + 1])
        for seq in sequences
        for i in range(len(seq))
    }
    assert sum(s["tokens"] for s in segments) == len(distinct)


def test_reuse_plus_cache_size_equals_tokens_sent():
    sequences = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 9], [5, 6]]
    segments = _segments(sequences)
    cache_size = sum(s["tokens"] for s in segments)
    reuse = sum(s["reuse_tokens"] for s in segments)
    assert cache_size + reuse == sum(len(s) for s in sequences)


def test_first_call_is_the_creating_call():
    sequences = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 5]]
    segments = {s["segment"]: s for s in _segments(sequences)}
    shared = [s for s in segments.values() if s["tokens"] == 2 and s["first_call"] == 0]
    assert shared, "the [1,2] prefix must be attributed to call 0"
    assert shared[0]["n_calls"] == 3
    # the [3] split created at call 1 is traversed by calls 1 and 2
    split = [s for s in segments.values() if s["first_call"] == 1 and s["n_calls"] == 2]
    assert split and split[0]["tokens"] == 1


def test_identical_prompts_create_no_new_segment_tokens():
    once = _segments([[1, 2, 3]])
    twice = _segments([[1, 2, 3], [1, 2, 3]])
    assert sum(s["tokens"] for s in once) == sum(s["tokens"] for s in twice)
    assert twice[0]["n_calls"] == 2


def test_divergent_prompts_share_nothing():
    segments = _segments([[1, 2, 3], [4, 5, 6]])
    assert all(s["n_calls"] == 1 for s in segments)
    assert all(s["reuse_tokens"] == 0 for s in segments)


def test_content_tag_takes_the_most_volatile_thing_present():
    """A block of fixed boilerplate is unrepeatable once an output lands in it."""
    assert classify_content("just some standing instructions")[0] == "system prompt"
    assert classify_content("STEP 3 [Chosen action unit]: x")[0] == "history dialog"
    assert classify_content("STEP 3\n[Code]:\nimport pandas")[0] == "code"
    assert classify_content("STEP 3\n[Code]: x\n[Output]:\n3.14159")[0] == "raw data"


def test_raw_data_share_counts_measurements_not_step_numbers():
    _, plain = classify_content("STEP 3 of 12, attempt 2, year 2026")
    assert plain == 0.0
    _, dense = classify_content("[1.500529793, 7.306074269] GSM3721554 '1552263_at'")
    assert dense > 0.5


def test_content_breakdown_shares_and_leverage():
    segments = [
        {"content_tag": "system prompt", "tokens": 10, "realized_tokens": 90,
         "raw_data_share": 0.0},
        {"content_tag": "raw data", "tokens": 90, "realized_tokens": 10,
         "raw_data_share": 0.9},
    ]
    cb = _content_breakdown(segments)
    boiler = cb["by_tag"]["system prompt"]
    output = cb["by_tag"]["raw data"]
    assert boiler["cache_share"] == 0.1 and boiler["realized_share"] == 0.9
    assert boiler["leverage"] == 9.0, "cheap content returning most hits"
    assert output["leverage"] == 0.11


def test_unmatched_share_flags_a_corpus_the_markers_do_not_fit():
    segments = [
        {"content_tag": "system prompt", "tokens": 100, "realized_tokens": 0,
         "raw_data_share": 0.0},
    ]
    assert _content_breakdown(segments)["unmatched_share"] == 1.0


def _calls(cache_reads, lengths):
    return [
        {"input": n, "cacheRead": k, "vendor": "vLLM", "model": "m"}
        for k, n in zip(cache_reads, lengths)
    ]


def test_realized_attribution_reconstructs_reported_hits():
    """Segments partition each call's prefix, so hits must add back up."""
    sequences = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 9, 9], [1, 2, 3, 4, 5, 7]]
    segments = _segments(sequences)
    calls = _calls([0, 4, 2], [6, 6, 6])
    out = attach_realized(segments, calls, sequences)
    assert out["attribution_residual"] == 0
    assert out["realized_reuse_tokens"] == 6


def test_realized_prefix_ending_mid_segment_is_counted_partially():
    sequences = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]
    segments = _segments(sequences)
    assert len(segments) == 1 and segments[0]["tokens"] == 8
    # second call reports a 3-token hit inside the single 8-token segment
    out = attach_realized(segments, _calls([0, 3], [8, 8]), sequences)
    assert segments[0]["realized_tokens"] == 3
    assert segments[0]["gap_tokens"] == 5
    assert out["capture_rate"] == 0.375


def test_creating_call_is_never_credited_as_a_hit():
    sequences = [[1, 2, 3, 4]]
    segments = _segments(sequences)
    attach_realized(segments, _calls([4], [4]), sequences)
    assert segments[0]["realized_tokens"] == 0
    assert segments[0]["realized_hits"] == 0


def test_zero_cache_reads_give_zero_capture():
    sequences = [[1, 2, 3], [1, 2, 3]]
    segments = _segments(sequences)
    out = attach_realized(segments, _calls([0, 0], [3, 3]), sequences)
    assert out["realized_reuse_tokens"] == 0
    assert out["capture_rate"] == 0.0
    assert out["gap_tokens"] == 3


def _real_cell(tmp_path: Path) -> Path:
    cell = tmp_path / "cell"
    cell.mkdir()
    messages, events = [], []
    for i in range(6):
        run_id = f"r{i}"
        role = "GEO" if i % 2 == 0 else "Expert"
        messages.append({
            "run_id": run_id,
            "agent_role": role,
            "model": "gpt-4o-mini",
            "provider": "openai",
            "timestamp": 1000.0 + i,
            "messages": [
                {"role": "system", "content": f"you are {role}"},
                {"role": "user", "content": "SHARED GUIDELINES " * 30 + f"step {i}"},
            ],
        })
        events.append({
            "run_id": run_id,
            "type": "message_end",
            "message": {"timestamp": 1000.0 + i, "usage": {"input": 500, "output": 10, "cacheRead": 100}},
        })
        events.append({
            "run_id": run_id,
            "type": "message_start",
            "message": {"timestamp": 999.0 + i},
        })
    (cell / "messages.jsonl").write_text(
        "\n".join(json.dumps(m) for m in messages), encoding="utf-8"
    )
    (cell / "pi_events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8"
    )
    return cell


def test_analyze_cell_segments_end_to_end(tmp_path):
    cell = _real_cell(tmp_path)
    summary = analyze_cell_segments(cell)

    assert summary["n_calls"] == 6
    assert summary["n_segments"] > 0
    assert (
        summary["cache_size_tokens"]
        + sum(s["reuse_tokens"] for s in summary["segments"])
        == summary["prompt_tokens_total"]
    )
    assert summary["resend_ratio"] > 1.0

    write_tables(summary, cell)
    write_markdown(summary, cell)
    assert (cell / SEGMENTS_CSV).read_text(encoding="utf-8").count("\n") > 1
    assert "KV cache contents" in (cell / SEGMENTS_MD).read_text(encoding="utf-8")


def test_empty_cell_is_handled(tmp_path):
    cell = tmp_path / "empty"
    cell.mkdir()
    summary = analyze_cell_segments(cell)
    assert summary["n_calls"] == 0
    assert summary["segments"] == []
