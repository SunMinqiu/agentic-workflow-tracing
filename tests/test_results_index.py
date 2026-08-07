"""The index and the per-task reports must agree, because they are one table.

The index exists to compare tasks side by side. If its numbers were computed
separately from the numbers on each task's own page, the two would drift and
the comparison would be worthless — so both render through kvcache/summary.py
and these tests pin that down.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent_io_tracing.analysis.kvcache.summary import (
    TIME_HEADERS, TOKEN_HEADERS, time_table, token_table,
)
from agent_io_tracing.analysis.results_index import (
    IO_REPORT, KV_REPORT, LEAD_HEADERS, _lead_cells,
    discover_tasks, group_rows, task_row, write_results_index,
)

ROW = {
    "cell": "A_c2_w1",
    "run": "GenoMAS_A_c2_w1_20260805_152115",
    "workflow": "GenoMAS",
    "stamp": "20260805_152115",
    "rel": "GenoMAS_A_c2_w1_20260805_152115/A_c2_w1",
    "n_calls": 12,
    "total_input": 74798,
    "median_input": 5406,
    "total_output": 29438,
    "realized_frac": 0.29,
    "logical_frac": 0.36,
    "gap_frac": 0.06,
    "latency": {"overall": {"total_duration_s": 431.0}, "wall_clock_span_s": 433.0,
                "stream_timing": {"median_ttft_s": 0.7, "median_tpot_s": 0.0143}},
    "runtime": {"vendors": ["vLLM"], "models": ["Qwen3.6-27B"]},
    "segments": {"cache_size_tokens": 45376, "resend_ratio": 1.552},
}


def _data_row(lines: list[str]) -> str:
    return [l for l in lines if l.startswith("| A_c2_w1")][0]


def test_index_and_report_show_the_same_numbers():
    """Same row, two renderings: only the leading columns may differ."""
    report = _data_row(token_table([ROW]))
    index = _data_row(token_table([ROW], LEAD_HEADERS, _lead_cells))
    assert report.split(" | ")[1:] == index.split(" | ")[2:]


def test_both_tables_have_a_column_for_every_header():
    for lines, headers in ((token_table([ROW]), TOKEN_HEADERS),
                           (time_table([ROW]), TIME_HEADERS)):
        assert lines[0].count("|") == len(headers) + 2
        assert _data_row(lines).count("|") == len(headers) + 2


def test_every_task_links_to_both_of_its_reports():
    task, links = _lead_cells(ROW)
    assert "A_c2_w1" in task and "08-05" in task
    assert f"{ROW['rel']}/{IO_REPORT}" in links
    assert f"{ROW['rel']}/{KV_REPORT}" in links


def test_newest_run_comes_first_inside_a_workflow():
    older = dict(ROW, stamp="20260729_144227", cell="A_c4_w2")
    grouped = [r for r in group_rows([older, ROW]) if r.get("stamp")]
    assert [r["stamp"] for r in grouped] == ["20260805_152115", "20260729_144227"]


def test_workflows_are_separated_and_the_empty_ones_still_show():
    grouped = group_rows([ROW])
    groups = [r["_group"] for r in grouped if r.get("_group")]
    assert groups[0] == "GenoMAS"
    assert "SciLink" in groups, "a workflow with no runs must still be visible"
    empty = [r for r in grouped if r.get("_empty")]
    assert empty and all(r.get("_empty") != "GenoMAS" for r in empty)


def test_an_unknown_run_name_does_not_lose_the_task(tmp_path):
    cell = tmp_path / "some_old_run" / "cell"
    cell.mkdir(parents=True)
    (cell / "messages.jsonl").write_text("", encoding="utf-8")
    rows = discover_tasks(tmp_path)
    assert len(rows) == 1 and rows[0]["workflow"] == "other"


def test_a_task_with_no_analysis_artifacts_still_gets_a_row(tmp_path):
    """A failed run has to be visible in the index, not silently absent."""
    cell = tmp_path / "GenoMAS_x_20260806_101500" / "x"
    cell.mkdir(parents=True)
    (cell / "messages.jsonl").write_text("", encoding="utf-8")
    row = task_row(cell, tmp_path)
    assert row["n_calls"] == "?" and row["segments"] == {}
    rendered = token_table([row], LEAD_HEADERS, _lead_cells)[-1]
    assert "n/a" in rendered


def test_write_results_index_writes_one_page(tmp_path):
    cell = tmp_path / "GenoMAS_x_20260806_101500" / "x"
    cell.mkdir(parents=True)
    (cell / "messages.jsonl").write_text("", encoding="utf-8")
    (cell / "kvcache_demand.json").write_text(json.dumps({"n_token_calls": 3}))
    out = write_results_index(tmp_path)
    page = out.read_text(encoding="utf-8")
    assert out.name == "index.html"
    assert "Summary — tokens" in page and "Summary — time" in page
    assert "visualizations/index.html" in page and "kvcache_report.html" in page


def test_no_run_level_report_is_written(tmp_path):
    """A run holding three tasks is three experiments, not one page."""
    for run in Path("results").iterdir():
        if run.is_dir():
            assert not (run / "kvcache_report.html").exists(), run
