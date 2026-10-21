#!/usr/bin/env python3
"""Full experiment metrics archive with links to each task report.

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
from agent_io_tracing.viz.format_utils import jload
from agent_io_tracing.replay.sweep import pretty_arm
from agent_io_tracing.analysis.kvcache.catalog import Catalog, COLORS, LEGEND

INDEX_HTML = "all-experiments.html"
IO_REPORT = "visualizations/index.html"
KV_REPORT = "kvcache_report.html"

# Run directories are named <Workflow>_<task>_<timestamp> by run_dir_name.
RUN_NAME = re.compile(r"^(?P<workflow>[A-Za-z0-9]+)_.*?(?P<stamp>\d{8}_\d{6})$")
WORKFLOWS = ["GenoMAS", "SciLink", "SRAgent", "ChemGraph", "Montage",
             "1000Genome", "Pi"]


def task_row(cell: Path, results_dir: Path) -> dict[str, Any]:
    """The shape report.analyze_run returns, read back instead of recomputed."""
    demand = jload(cell / "kvcache_demand.json")
    logical = jload(cell / "kvcache_logical.json")
    seg = jload(cell / "kvcache_segments.json")
    run = cell.parent
    match = RUN_NAME.match(run.name)
    return {
        "cell": cell.name,
        "run": run.name,
        "workflow": match.group("workflow") if match else "other",
        "stamp": match.group("stamp") if match else "",
        "rel": cell.relative_to(results_dir).as_posix(),
        "available_reports": [name for name in (IO_REPORT, KV_REPORT) if (cell / name).is_file()],
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
        "prometheus_window": logical.get("prometheus_window") or {},
        "server_config_changed_during_cell": logical.get(
            "server_config_changed_during_cell", False
        ),
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


def discover_sweep_arms(results_dir: Path, catalog: Catalog | None = None) -> list[dict[str, Any]]:
    """Discover replay data and link every configuration to its source task."""
    catalog = catalog or Catalog(results_dir.parent)
    groups = {}
    for row in catalog.records:
        groups.setdefault((row['rep'].parent, row['source']), []).append(row)
    rows = []
    for (arm, source), records in groups.items():
        valid = [r for r in records if not r['review'].get('exclude_from_comparison')]
        first = records[0]
        def median(key):
            values = [r.get(key) for r in valid if isinstance(r.get(key), (int, float))]
            return statistics.median(values) if values else None
        rates = [r['measurement']['rate'] for r in valid if r['measurement']['rate'] is not None]
        sources = {r['measurement']['source'] for r in valid}
        stamp_match = RUN_NAME.match(source.parent.name) if source else None
        stamp = stamp_match.group('stamp') if stamp_match else ''
        rows.append({
            'case': f'{source.name} · {_pretty_stamp(stamp)}' if source else first['source_kind'],
            'case_stamp': stamp, 'sweep': os.path.relpath(arm.parent, catalog.root/'replay'),
            'arm': arm.name, 'configuration_anchor': first['arm_id'],
            'report': os.path.relpath(catalog.page(first), results_dir),
            'block_returns_highlight': any(catalog.blue(r) for r in records),
            'reps': len(valid), 'recorded_reps': len(records), 'requests': first.get('requests'),
            'prompt_tokens': first.get('prompt_tokens'), 'output_tokens': median('output_tokens'),
            'hit_rate': statistics.median(rates) if rates and len(sources)==1 else None,
            'window_shared': False, 'median_ttft_ms': median('median_ttft_ms'),
            'p95_ttft_ms': median('p95_ttft_ms'), 'median_tpot_ms': median('median_tpot_ms'),
            'median_total_ms': median('median_total_ms'), 'wall_s': median('wall_s'),
            'recorded_at': first.get('started_at_epoch_s') or 0,
        })
    return sorted(rows, key=lambda row: (row['case_stamp'], row['sweep'], row['recorded_at']), reverse=True)


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
        " · ".join(f"[{label}]({r['rel']}/{name})" for name, label in [(IO_REPORT, "IO"), (KV_REPORT, "KV")] if name in r.get("available_reports", [])) or "Reports not generated",
    ]


LEAD_HEADERS = ["task", "reports"]


def _sweep_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| arm | comparison | calls | total input | total output | realized reuse | median TTFT | p95 TTFT | median TPOT | median request latency | wall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    current_case = None
    current_sweep = None
    for row in rows:
        if row["case"] != current_case:
            current_case = row["case"]
            current_sweep = None
            lines.append(f"| **{current_case}** |" + " |" * 10)
        test_name = (
            row["sweep"].rsplit("/", 1)[-1]
            .replace("-", " ")
            .replace("_", " ")
        )
        if row["sweep"] != current_sweep:
            current_sweep = row["sweep"]
            lines.append(
                f"| **{test_name}** | [Task KVCache experiments]({row['report']}#replay-experiments) |"
                + " |" * 9
            )
        hit = f"{row['hit_rate']:.2%}" if isinstance(row.get("hit_rate"), (int, float)) else "n/a"
        if row.get("window_shared") and hit != "n/a":
            hit += " †"
        ttft = f"{row['median_ttft_ms'] / 1000:,.2f}s" if isinstance(row.get("median_ttft_ms"), (int, float)) else "n/a"
        p95_ttft = f"{row['p95_ttft_ms'] / 1000:,.2f}s" if isinstance(row.get("p95_ttft_ms"), (int, float)) else "n/a"
        tpot = f"{row['median_tpot_ms']:,.2f}ms/token" if isinstance(row.get("median_tpot_ms"), (int, float)) else "n/a"
        total = f"{row['median_total_ms'] / 1000:,.2f}s" if isinstance(row.get("median_total_ms"), (int, float)) else "n/a"
        wall = f"{row['wall_s']:,.2f}s" if isinstance(row.get("wall_s"), (int, float)) else "n/a"
        prompt = f"{int(row['prompt_tokens']):,}" if isinstance(row.get("prompt_tokens"), (int, float)) else "n/a"
        output = f"{int(row['output_tokens']):,}" if isinstance(row.get("output_tokens"), (int, float)) else "n/a"
        arm = pretty_arm({"label": row["arm"], "recorded_at": row["recorded_at"]})
        detail = f"[Configuration and {row['reps']} valid run(s)]({row['report']}#{row['configuration_anchor']})"
        if row.get("block_returns_highlight"):
            arm = '<span class="kv-swap">' + arm + '</span>'
        lines.append(
            f"| {arm} | {detail} | {row['requests']} | {prompt} | "
            f"{output} | {hit} | {ttft} | {p95_ttft} | "
            f"{tpot} | {total} | {wall} |"
        )
    if any(row.get("window_shared") for row in rows):
        lines.append("")
        lines.append(
            "† arms replayed concurrently share one Prometheus window, so "
            "their realized reuse is the per-request cached_tokens sum; the "
            "window delta is not attributable to a single arm."
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
    L.append(LEGEND)
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


def write_results_index(results_dir: Path, catalog: Catalog | None = None) -> Path:
    rows = discover_tasks(results_dir)
    text = build_index(rows, discover_sweep_arms(results_dir, catalog))
    body = markdown.markdown(text, extensions=["tables"])
    out = results_dir / INDEX_HTML
    out.write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Experiments</title>\n<style>\n" + PAGE_CSS + COLORS + "</style>\n"
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n",
        encoding="utf-8",
    )
    catalog = catalog or Catalog(results_dir.parent)
    from agent_io_tracing.analysis.kvcache.replay_report import render_replays, CSS
    if catalog.task_records(None):
        content = out.read_text().replace('</style>', CSS + '</style>')
        out.write_text(content.replace('</body>', render_replays(catalog, None, results_dir) + '</body>'))
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
