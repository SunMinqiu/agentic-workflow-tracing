import json
from pathlib import Path

from agent_io_tracing.analysis.kvcache.demand import (
    CTX_GROWTH_PNG,
    plot_context_growth,
)
from agent_io_tracing.analysis.kvcache.logical import (
    CACHE_WARMING_PNG,
    PREFIX_LINEAGE_PNG,
    PREFIX_DUMP,
    _logical_reuse_fraction,
    _cache_block_references,
    _lineage_parent_index,
    _lineage_marker_size,
    _per_call_reuse,
    _reuse_distance_summary,
    analyze_cell_logical,
    plot_cache_warming,
)
from agent_io_tracing.analysis.kvcache.latency import (
    LATENCY_TIMELINE_PNG,
    OUTPUT_LATENCY_PNG,
    plot_latency_breakdown,
    plot_output_vs_latency,
    plot_ttft_vs_fresh_input,
    plot_ttft_vs_prefix_age,
)
from agent_io_tracing.analysis.kvcache.report import build_report


def _logical_summary(n: int = 10) -> dict:
    return {
        "cell": "cell-a",
        "per_call": [
            {
                "input": 100,
                "cacheRead": i * 5,
                "our_tokens": 100,
                "logical": i * 8,
                "logical_aligned": i * 8,
            }
            for i in range(n)
        ],
    }


def test_per_call_reuse_returns_each_call_ratio() -> None:
    x, realized, logical = _per_call_reuse(_logical_summary(3))

    assert x == [0, 1, 2]
    assert realized == [0.0, 0.05, 0.1]
    assert logical == [0.0, 0.08, 0.16]


def test_per_call_reuse_handles_zero_input() -> None:
    summary = _logical_summary(1)
    summary["per_call"][0]["input"] = 0
    x, realized, logical = _per_call_reuse(summary)

    assert x == [0]
    assert realized == [0.0]
    assert logical == [0.0]


def test_reuse_distance_counts_distinct_blocks_since_last_touch() -> None:
    summary = _reuse_distance_summary(["a", "b", "c", "a", "b", "a"])

    assert summary["cumulative_unique_blocks"] == 3
    assert summary["block_references"] == 6
    assert summary["cold_references"] == 3
    assert summary["reference_order"] == "request_start_then_prefix"
    assert summary["reuse_distance_blocks"] == {"1": 1, "2": 2}
    assert summary["peak_resident_blocks"] == {
        "policy": "unbounded",
        "blocks": 3,
    }


def test_cache_block_identity_includes_the_parent_prefix() -> None:
    first = _cache_block_references([1, 2, 3, 4], block_size=2)
    second = _cache_block_references([9, 9, 3, 4], block_size=2)

    assert len(first) == 2
    assert first[1] != second[1]


def test_lineage_uses_logical_prefix_candidates_without_cache_read() -> None:
    call = {"cacheRead": 0, "logical_source_candidates": [1, 3]}
    assert _lineage_parent_index(call) == 3

    call["logical_source_candidates"] = []
    assert _lineage_parent_index(call) is None


def test_lineage_size_uses_logical_reuse_percentage() -> None:
    assert _logical_reuse_fraction({"input": 200, "logical_aligned": 50}) == 0.25
    assert _logical_reuse_fraction({"input": 0, "logical_aligned": 50}) == 0.0
    assert _logical_reuse_fraction({"input": 100, "logical_aligned": 150}) == 1.0
    assert _lineage_marker_size(0.0) < _lineage_marker_size(0.5)
    assert _lineage_marker_size(0.5) < _lineage_marker_size(1.0)


def test_kvcache_plots_write_pngs(tmp_path: Path) -> None:
    calls = [
        {"role": "agent-a", "input": 100, "cacheRead": 0},
        {"role": "agent-b", "input": 150, "cacheRead": 20},
        {"role": "agent-a", "input": 200, "cacheRead": 50},
        {"role": "agent-b", "input": 300, "cacheRead": 100},
    ]
    paths = [
        tmp_path / "context.png",
        tmp_path / "warming.png",
    ]

    plot_context_growth(calls, paths[0])
    plot_cache_warming(_logical_summary(), paths[1])

    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_stream_timing_plots_write_pngs(tmp_path: Path) -> None:
    summary = {
        "per_call": [
            {
                "role": "agent-a",
                "input": 1200,
                "output": 20,
                "cacheRead": cache_read,
                "fresh_input": 1200 - cache_read,
                "start_ms": 1000.0 + i * 3000,
                "first_token_ms": 1400.0 + i * 3000,
                "last_token_ms": 2400.0 + i * 3000,
                "end_ms": 2500.0 + i * 3000,
                "newest_possible_source_age_s": float(i + 1),
                "capture_rate": cache_read / 1024,
            }
            for i, cache_read in enumerate([0, 256, 768, 1024])
        ]
    }
    outputs = [tmp_path / f"stream-{i}.png" for i in range(3)]

    plot_ttft_vs_fresh_input(summary, outputs[0])
    plot_latency_breakdown(summary, outputs[1])
    plot_ttft_vs_prefix_age(summary, outputs[2])

    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_output_latency_plot_writes_png(tmp_path: Path) -> None:
    summary = {
        "per_call": [
            {
                "role": "agent-a",
                "output": output,
                "duration_ms": output * 14.3,
            }
            for output in [100, 500, 1000]
        ]
    }
    output = tmp_path / OUTPUT_LATENCY_PNG

    plot_output_vs_latency(summary, output)

    assert output.is_file() and output.stat().st_size > 0


def test_report_embeds_kvcache_figures(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    rows = [
        {
            "cell": "cell-a",
            "has_logical": True,
            "has_prefix_dump": True,
            "has_stream_timing": False,
            "n_calls": 4,
            "total_input": 1000,
            "out_in": 0.125,
            "runtime": {
                "vendors": ["FreeInference"],
                "providers": ["openai"],
                "models": ["qwen3.6-35b"],
                "cache_configs": [{}],
            },
            "latency": {
                "overall": {
                    "n": 1,
                    "median_duration_s": 2.0,
                    "p90_duration_s": 2.0,
                    "max_duration_s": 2.0,
                },
                "spearman": {},
                "slowest": [
                    {
                        "global_call": 3,
                        "role": "GEOAgent",
                        "duration_s": 2.0,
                        "input": 100,
                        "cacheRead": 50,
                        "fresh_input": 50,
                        "output": 10,
                    }
                ],
            },
        },
        {
            "cell": "cell-b",
            "has_logical": True,
            "has_prefix_dump": True,
            "has_stream_timing": False,
            "n_calls": 2,
            "total_input": 500,
            "out_in": 0.1,
            "runtime": {
                "vendors": ["OpenAI"],
                "providers": ["openai"],
                "models": ["gpt-4o-mini"],
                "cache_configs": [{}],
            },
        },
    ]
    for cell in ["cell-a", "cell-b"]:
        visualization_dir = run_dir / cell / "visualizations"
        visualization_dir.mkdir(parents=True)
        for filename in [
            CTX_GROWTH_PNG,
            CACHE_WARMING_PNG,
            PREFIX_LINEAGE_PNG,
            LATENCY_TIMELINE_PNG,
            OUTPUT_LATENCY_PNG,
        ]:
            (visualization_dir / filename).write_bytes(b"png")
        (run_dir / cell / PREFIX_DUMP).write_text("prefixes", encoding="utf-8")

    report = build_report(run_dir, rows, tmp_path).read_text(encoding="utf-8")
    html_report = (run_dir / "kvcache_report.html").read_text(encoding="utf-8")

    assert report.count("data:image/png;base64,cG5n") == 10
    assert html_report.count('<img alt="') == 10
    assert html_report.count("data:image/png;base64,cG5n") == 10
    assert PREFIX_DUMP in report
    assert PREFIX_DUMP in html_report
    assert "<details>" in html_report
    assert "<pre>prefixes</pre>" in html_report
    assert "data:text/plain" not in html_report
    assert "## Summary — tokens" in report
    assert "## Summary — time" in report
    assert "FreeInference" in report
    assert "qwen3.6-35b" in report
    assert "provider-managed/default" in report
    assert "\n---\n\n### cell-b" in report
    assert "ctx_grow/call" not in report
    assert "x-agent%" not in report
    assert "verbatim%" not in report
    assert "local, no API" not in report
    assert "## Reading" not in report
    assert "**Runtime**" not in report
    assert "**Context growth**" not in report
    assert "**Cache warming**" not in report

    cell_report = build_report(
        run_dir / "cell-a",
        rows[:1],
        tmp_path,
        cells_root=run_dir,
    )
    assert cell_report == run_dir / "cell-a" / "kvcache_report.md"
    assert cell_report.with_suffix(".html").is_file()


def test_report_survives_failed_cell_without_kvcache_figures(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-a"
    failed_cell = run_dir / "failed-cell"
    failed_cell.mkdir(parents=True)
    rows = [
        {
            "cell": "failed-cell",
            "has_logical": False,
            "has_prefix_dump": False,
            "has_stream_timing": False,
        }
    ]

    output = build_report(run_dir, rows, tmp_path)
    report = output.read_text(encoding="utf-8")

    assert output.is_file()
    assert output.with_suffix(".html").is_file()
    assert "no successful token-bearing LLM calls" in report
    assert "kvcache_context_growth.png` was not generated" in report


def test_prefix_dump_has_agent_index_and_role_local_call_numbers(
    tmp_path: Path,
) -> None:
    messages = [
        {
            "run_id": f"run-{i}",
            "genomas_role": role,
            "timestamp": i,
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"prompt {i}"}],
        }
        for i, role in enumerate(["agent-a", "agent-b", "agent-a"])
    ]
    events = [
        {
            "type": "message_end",
            "run_id": f"run-{i}",
            "message": {"usage": {"input": 10, "cacheRead": 0}},
        }
        for i in range(3)
    ]
    (tmp_path / "messages.jsonl").write_text(
        "\n".join(json.dumps(row) for row in messages),
        encoding="utf-8",
    )
    (tmp_path / "pi_events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in events),
        encoding="utf-8",
    )

    analyze_cell_logical(tmp_path, dump_prefixes=True)
    prefix_dump = (tmp_path / "kvcache_prefixes.txt").read_text(encoding="utf-8")

    assert "agent-a: 2 calls | global_calls=[0, 2]" in prefix_dump
    assert "agent-b: 1 calls | global_calls=[1]" in prefix_dump
    assert "global_call=2  role=agent-a  role_call=2/2" in prefix_dump
    summary = analyze_cell_logical(tmp_path)
    assert summary is not None
    assert "logical_source_candidates" in summary["per_call"][2]
    assert "duration_ms" in summary["per_call"][2]


def test_logical_analysis_reads_cache_geometry_from_serving_manifest(
    tmp_path: Path,
) -> None:
    messages = [
        {
            "run_id": "run-0",
            "timestamp": 0,
            "model": "gpt-4o-mini",
            "cache_config": {
                "serving": {
                    "backend": "vllm",
                    "resolved": {
                        "block_size": 4,
                        "prefix_match_unit": 2,
                    },
                },
            },
            "messages": [{"role": "user", "content": "repeat this prompt " * 10}],
        },
        {
            "run_id": "run-1",
            "timestamp": 1,
            "model": "gpt-4o-mini",
            "cache_config": {
                "serving": {
                    "backend": "vllm",
                    "resolved": {
                        "block_size": 4,
                        "prefix_match_unit": 2,
                    },
                },
            },
            "messages": [{"role": "user", "content": "repeat this prompt " * 10}],
        },
    ]
    events = [
        {
            "type": "message_end",
            "run_id": f"run-{i}",
            "message": {"usage": {"input": 40, "cacheRead": 0}},
        }
        for i in range(2)
    ]
    (tmp_path / "messages.jsonl").write_text(
        "\n".join(json.dumps(row) for row in messages),
        encoding="utf-8",
    )
    (tmp_path / "pi_events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in events),
        encoding="utf-8",
    )

    summary = analyze_cell_logical(tmp_path)

    assert summary is not None
    assert summary["cache_geometry"] == {
        "block_size": 4,
        "prefix_match_unit": 2,
    }
    assert summary["logical_aligned_frac"] > 0
    assert summary["block_demand"]["cumulative_unique_blocks"] > 0
    assert summary["block_demand"]["reuse_distance_blocks"]
