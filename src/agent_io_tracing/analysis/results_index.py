#!/usr/bin/env python3
"""One page listing every task under results/, with a link to each report.

The two tables are the same ones that open every per-task report — see
kvcache/summary.py — so a number here and the number on the task's own page
cannot disagree. Each row adds the two reports that task produced: I/O pattern
(generated on the cluster) and KV cache (regenerated locally on pull).

Rows are grouped by workflow, newest run first inside each group. A task whose
analysis artifacts are missing still gets a row, with n/a in every column: a
run that failed has to be visible, not absent.

    PYTHONPATH=src python3 -m agent_io_tracing.analysis.results_index
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any

import markdown

from agent_io_tracing.analysis.kvcache.summary import (
    PAGE_CSS, token_table, time_table,
)
from agent_io_tracing.replay.sweep import pretty_arm

INDEX_HTML = "index.html"
IO_REPORT = "visualizations/index.html"
KV_REPORT = "kvcache_report.html"
SWEEP_REPORT = "kvcache_report.html"

# Run directories are named <Workflow>_<task>_<timestamp> by run_dir_name.
RUN_NAME = re.compile(r"^(?P<workflow>[A-Za-z0-9]+)_.*?(?P<stamp>\d{8}_\d{6})$")
WORKFLOWS = ["GenoMAS", "SciLink", "SRAgent", "ChemGraph", "Montage",
             "1000Genome", "Pi"]


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def task_row(cell: Path, results_dir: Path) -> dict[str, Any]:
    """The shape report.analyze_run returns, read back instead of recomputed."""
    demand = _load(cell / "kvcache_demand.json")
    logical = _load(cell / "kvcache_logical.json")
    seg = _load(cell / "kvcache_segments.json")
    run = cell.parent
    match = RUN_NAME.match(run.name)
    return {
        "cell": cell.name,
        "run": run.name,
        "workflow": match.group("workflow") if match else "other",
        "stamp": match.group("stamp") if match else "",
        "rel": cell.relative_to(results_dir).as_posix(),
        "n_calls": demand.get("n_token_calls") or logical.get("n_calls") or "?",
        "total_input": demand.get("total_input_tokens", 0),
        "median_input": (demand.get("input_tokens") or {}).get("median", 0),
        "total_output": demand.get("total_output_tokens", 0),
        "realized_frac": demand.get("cacheread_fraction"),
        "cached_tokens": demand.get("total_cacheread_tokens"),
        "cache_read_available": demand.get("cache_read_available", False),
        "logical_frac": logical.get("logical_frac"),
        "logical_aligned_frac": logical.get("logical_aligned_frac"),
        # Backends that leave cacheRead null record their realized reuse here
        # instead; the summary tables fall back to it.
        "server_prefix_cache": logical.get("server_prefix_cache"),
        "realized_provenance": logical.get("realized_provenance"),
        "gap_frac": logical.get("gap_frac"),
        "latency": logical.get("latency") or {},
        "runtime": logical.get("runtime") or {},
        "serving_config": logical.get("serving_config") or {},
        "segments": {
            "realized_vs_logical": seg.get("realized_vs_logical") or {},
        } if seg else {},
    }


def discover_tasks(results_dir: Path) -> list[dict[str, Any]]:
    cells = sorted(
        c for c in results_dir.glob("*/*")
        if c.is_dir() and (c / "messages.jsonl").is_file()
    )
    return [task_row(c, results_dir) for c in cells]


def _source_case(summary: dict[str, Any]) -> tuple[str, str]:
    """The bundle's source cell, formatted exactly like a task index row."""
    bundle_path = Path(str(summary.get("bundle") or ""))
    bundle = _load(bundle_path)
    source = Path(str(bundle.get("source_cell") or ""))
    parts = source.parts
    try:
        results_index = len(parts) - 1 - parts[::-1].index("results")
        run = parts[results_index + 1]
        cell = parts[results_index + 2]
    except (ValueError, IndexError):
        return bundle_path.stem or "unknown source", ""
    match = RUN_NAME.match(run)
    stamp = match.group("stamp") if match else ""
    return f"{cell} · {_pretty_stamp(stamp)}", stamp


def discover_sweep_arms(results_dir: Path) -> list[dict[str, Any]]:
    """Find replay sweep arms and read their saved summaries."""
    replay_dir = results_dir.parent / "replay"
    if not replay_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for report in sorted(replay_dir.rglob(SWEEP_REPORT)):
        sweep_dir = report.parent
        for arm_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
            reps = [
                _load(path)
                for path in sorted(arm_dir.glob("rep*/summary.json"))
            ]
            reps = [row for row in reps if row]
            if not reps:
                continue

            def median(key: str) -> float | None:
                values = [row.get(key) for row in reps]
                numbers = [value for value in values if isinstance(value, (int, float))]
                return statistics.median(numbers) if numbers else None

            hit_rates = [
                (row.get("prefix_cache") or {}).get("hit_rate") for row in reps
            ]
            hit_rates = [value for value in hit_rates if isinstance(value, (int, float))]
            first = reps[0]
            source_case, source_stamp = _source_case(first)
            rows.append({
                "case": source_case,
                "case_stamp": source_stamp,
                "sweep": sweep_dir.relative_to(replay_dir).as_posix(),
                "arm": arm_dir.name,
                "report": os.path.relpath(report, results_dir),
                "reps": len(reps),
                "bundle": Path(str(first.get("bundle") or "")).name,
                "mode": first.get("mode"),
                "cache_state": ",".join(sorted({str(row.get("cache_state")) for row in reps})),
                "requests": first.get("requests"),
                "prompt_tokens": first.get("prompt_tokens"),
                "output_tokens": median("output_tokens"),
                "hit_rate": statistics.median(hit_rates) if hit_rates else None,
                "median_ttft_ms": median("median_ttft_ms"),
                "median_tpot_ms": median("median_tpot_ms"),
                "median_total_ms": median("median_total_ms"),
                "wall_s": median("wall_s"),
                "recorded_at": first.get("started_at_epoch_s") or 0,
            })
    rows = sorted(rows, key=lambda row: row["recorded_at"])
    return sorted(rows, key=lambda row: row["case_stamp"], reverse=True)


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Workflow headings, newest run first, cells alphabetical inside a run."""
    extra = sorted({r["workflow"] for r in rows} - set(WORKFLOWS))
    out: list[dict[str, Any]] = []
    for workflow in WORKFLOWS + extra:
        group = sorted(
            (r for r in rows if r["workflow"] == workflow),
            key=lambda r: (r["stamp"], r["cell"]),
            reverse=True,
        )
        out.append({"_group": workflow})
        out.extend(group or [{"_empty": workflow}])
    return out


def _pretty_stamp(stamp: str) -> str:
    try:
        return _dt.datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%m-%d %H:%M")
    except ValueError:
        return stamp or "?"


def _lead_cells(r: dict[str, Any]) -> list[str]:
    if r.get("_empty"):
        return ["_no runs yet_", ""]
    return [
        f"{r['cell']} · {_pretty_stamp(r['stamp'])}",
        f"[IO]({r['rel']}/{IO_REPORT}) · [KV]({r['rel']}/{KV_REPORT})",
    ]


LEAD_HEADERS = ["task", "reports"]


def _sweep_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| arm | report | calls | total input | total output | realized reuse | median TTFT | median TPOT | median request latency | wall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    current_case = None
    for row in rows:
        if row["case"] != current_case:
            current_case = row["case"]
            lines.append(f"| **{current_case}** |" + " |" * 9)
        hit = f"{row['hit_rate']:.2%}" if isinstance(row.get("hit_rate"), (int, float)) else "n/a"
        ttft = f"{row['median_ttft_ms'] / 1000:,.2f}s" if isinstance(row.get("median_ttft_ms"), (int, float)) else "n/a"
        tpot = f"{row['median_tpot_ms']:,.2f}ms/token" if isinstance(row.get("median_tpot_ms"), (int, float)) else "n/a"
        total = f"{row['median_total_ms'] / 1000:,.2f}s" if isinstance(row.get("median_total_ms"), (int, float)) else "n/a"
        wall = f"{row['wall_s']:,.2f}s" if isinstance(row.get("wall_s"), (int, float)) else "n/a"
        prompt = f"{int(row['prompt_tokens']):,}" if isinstance(row.get("prompt_tokens"), (int, float)) else "n/a"
        output = f"{int(row['output_tokens']):,}" if isinstance(row.get("output_tokens"), (int, float)) else "n/a"
        arm = pretty_arm({"label": row["arm"], "recorded_at": row["recorded_at"]})
        test_name = (
            row["sweep"].rsplit("/", 1)[-1]
            .replace("-", " ")
            .replace("_", " ")
        )
        lines.append(
            f"| {arm} | [{test_name}]({row['report']}) | {row['requests']} | {prompt} | "
            f"{output} | {hit} | {ttft} | {tpot} | {total} | {wall} |"
        )
    return lines


def build_index(
    rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]] | None = None,
) -> str:
    grouped = group_rows(rows)
    n_runs = len({r["run"] for r in rows})
    L = ["# Experiments", ""]
    L.append(
        f"_{len(rows)} task(s) across {n_runs} run(s) · "
        f"generated {_dt.date.today().isoformat()}._"
    )
    L.append("")
    L.append("## Fixed-input tests")
    L.append("")
    if sweep_rows:
        L.extend(_sweep_table(sweep_rows))
    else:
        L.append("_No fixed-input tests recorded._")
    L.append("")
    for title, table in (
        ("## Summary — tokens", token_table),
        ("## Summary — time", time_table),
    ):
        L.append(title)
        L.append("")
        L.extend(table(grouped, LEAD_HEADERS, _lead_cells))
        L.append("")
    return "\n".join(L)


def write_results_index(results_dir: Path) -> Path:
    rows = discover_tasks(results_dir)
    text = build_index(rows, discover_sweep_arms(results_dir))
    body = markdown.markdown(text, extensions=["tables"])
    out = results_dir / INDEX_HTML
    out.write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Experiments</title>\n<style>\n" + PAGE_CSS + "</style>\n"
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", type=Path)
    args = ap.parse_args()
    out = write_results_index(args.results.resolve())
    print(f"index → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
