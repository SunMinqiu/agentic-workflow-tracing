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
import re
from pathlib import Path
from typing import Any

import markdown

from agent_io_tracing.analysis.kvcache.summary import (
    PAGE_CSS, token_table, time_table,
)

INDEX_HTML = "index.html"
IO_REPORT = "visualizations/index.html"
KV_REPORT = "kvcache_report.html"

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
        "logical_frac": logical.get("logical_frac"),
        "gap_frac": logical.get("gap_frac"),
        "latency": logical.get("latency") or {},
        "runtime": logical.get("runtime") or {},
        "segments": {
            "cache_size_tokens": seg.get("cache_size_tokens"),
            "resend_ratio": seg.get("resend_ratio"),
            "realized_vs_logical": seg.get("realized_vs_logical") or {},
        } if seg else {},
    }


def discover_tasks(results_dir: Path) -> list[dict[str, Any]]:
    cells = sorted(
        c for c in results_dir.glob("*/*")
        if c.is_dir() and (c / "messages.jsonl").is_file()
    )
    return [task_row(c, results_dir) for c in cells]


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


def build_index(rows: list[dict[str, Any]]) -> str:
    grouped = group_rows(rows)
    n_runs = len({r["run"] for r in rows})
    L = ["# Experiments", ""]
    L.append(
        f"_{len(rows)} task(s) across {n_runs} run(s) · "
        f"generated {_dt.date.today().isoformat()}._"
    )
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
    text = build_index(rows)
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
