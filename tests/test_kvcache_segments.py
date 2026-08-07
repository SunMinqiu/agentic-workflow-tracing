import json
from pathlib import Path

from agent_io_tracing.analysis.kvcache.segments import (
    SEGMENTS_CSV,
    analyze_cell_segments,
    _content_breakdown,
    attach_realized,
    build_radix_trie,
    collect_segments,
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


def test_sample_elides_the_middle_of_long_text():
    from agent_io_tracing.analysis.kvcache.segments import _sample
    text = "A" * 2000 + "B" * 2000
    kept = _sample(text)
    assert kept.startswith("AAAA") and kept.endswith("BBBB")
    assert len(kept) < len(text)
    assert _sample("short") == "short"


def _seg(tag_tokens, realized=None):
    return {
        "hit_by_tag": dict(tag_tokens),
        "hit_tag": max(tag_tokens, key=lambda t: tag_tokens[t]) if tag_tokens else None,
        "realized_tokens": realized if realized is not None else sum(tag_tokens.values()),
    }


TAGS = ["instructions", "code", "raw data", "history dialog", "unlabeled"]


def test_token_shares_add_up_to_one_whole():
    """Tokens are split across the sections a hit covers, never double-counted."""
    segments = [_seg({"instructions": 90}), _seg({"raw data": 10})]
    cb = _content_breakdown(segments, TAGS)
    assert cb["by_tag"]["instructions"]["realized_share"] == 0.9
    assert cb["by_tag"]["raw data"]["realized_share"] == 0.1
    assert sum(cb["by_tag"][t]["realized_tokens"] for t in TAGS) == 100


def test_one_hit_spanning_two_types_pays_into_both():
    segments = [_seg({"code": 60, "raw data": 40})]
    cb = _content_breakdown(segments, TAGS)
    assert cb["by_tag"]["code"]["realized_tokens"] == 60
    assert cb["by_tag"]["raw data"]["realized_tokens"] == 40
    # one segment, counted under both types: segment shares exceed 100%
    assert cb["n_served_segments"] == 1
    assert cb["by_tag"]["code"]["segment_share"] == 1.0
    assert cb["by_tag"]["raw data"]["segment_share"] == 1.0


def test_every_type_with_tokens_gets_its_own_example():
    """A type in the table and nothing under it in the details is a bug."""
    from agent_io_tracing.analysis.kvcache.segments import examples_by_tag
    segments = [{
        "hit_by_tag": {"code": 60, "raw data": 40},
        "hit_samples": {
            "code": {"section": "Code", "chars": 9, "sample": "import os"},
            "raw data": {"section": "Output", "chars": 4, "sample": "3.14"},
        },
        "tokens": 100, "first_call": 0, "realized_hits": 1,
        "realized_tokens": 100, "n_calls": 2,
    }]
    ex = examples_by_tag(segments, TAGS)
    assert ex["code"][0]["hit_sample"] == "import os"
    assert ex["raw data"][0]["hit_sample"] == "3.14", "the label decides the text"
    assert ex["code"][0]["tag_tokens"] == 60
    assert ex["instructions"] == []


def test_never_served_segments_stay_out_of_the_segment_count():
    cb = _content_breakdown([_seg({}, realized=0), _seg({"code": 5})], TAGS)
    assert cb["n_served_segments"] == 1


def test_unlabeled_share_reports_what_the_grammar_missed():
    cb = _content_breakdown([_seg({"unlabeled": 40, "code": 60})], TAGS)
    assert cb["unlabeled_share"] == 0.4



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
    assert out["gap_tokens"] == 5


def test_hit_tokens_is_the_longest_prefix_any_call_was_served():
    sequences = [[1, 2, 3, 4, 5, 6, 7, 8]] * 3
    segments = _segments(sequences)
    attach_realized(segments, _calls([0, 3, 6], [8, 8, 8]), sequences)
    assert segments[0]["hit_tokens"] == 6, "the deepest hit, not the sum"
    assert segments[0]["realized_tokens"] == 9, "but the total still sums"


def test_creating_call_is_never_credited_as_a_hit():
    sequences = [[1, 2, 3, 4]]
    segments = _segments(sequences)
    attach_realized(segments, _calls([4], [4]), sequences)
    assert segments[0]["realized_tokens"] == 0
    assert segments[0]["realized_hits"] == 0


def test_zero_cache_reads_leave_the_whole_reuse_as_gap():
    sequences = [[1, 2, 3], [1, 2, 3]]
    segments = _segments(sequences)
    out = attach_realized(segments, _calls([0, 0], [3, 3]), sequences)
    assert out["realized_reuse_tokens"] == 0
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

    # the whole point of splitting by section: the parts add back up
    for s in summary["segments"]:
        assert sum(s["hit_by_tag"].values()) == s["realized_tokens"]
    cb = summary["content_breakdown"]
    assert (
        sum(cb["by_tag"][t]["realized_tokens"] for t in cb["tags"])
        == cb["realized_tokens_total"]
    )

    write_tables(summary, cell)
    assert (cell / SEGMENTS_CSV).read_text(encoding="utf-8").count("\n") > 1


def test_empty_cell_is_handled(tmp_path):
    cell = tmp_path / "empty"
    cell.mkdir()
    summary = analyze_cell_segments(cell)
    assert summary["n_calls"] == 0
    assert summary["segments"] == []
