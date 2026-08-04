#!/usr/bin/env python3
"""One-command KV-cache report for a run: run demand + logical, emit markdown.

    PYTHONPATH=src python3 -m agent_io_tracing.analysis.kvcache_report --runs <run_id>

For every cell under the run it (re)computes the demand analysis
(kvcache_demand.json + figures) and the logical-vs-realized analysis
(kvcache_logical.json + figure), then writes one

    results/<run_id>/kvcache_report.md

with a cross-cell summary table and per-cell sections containing tables and
embedded figures. Run it after each pull and the report regenerates itself.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import html
from collections import defaultdict
from pathlib import Path
from typing import Any

import markdown

from agent_io_tracing.analysis.kvcache.demand import (
    discover_cells, analyze_cell as demand_analyze,
    plot_context_growth,
    CELL_JSON as DEMAND_JSON, CTX_GROWTH_PNG,
)
from agent_io_tracing.analysis.kvcache.logical import (
    analyze_cell_logical, plot_cache_warming, plot_prefix_lineage,
    CELL_JSON as LOGICAL_JSON, CACHE_WARMING_PNG,
    PREFIX_LINEAGE_PNG, PREFIX_DUMP,
)
from agent_io_tracing.analysis.kvcache.latency import (
    analyze_latency, plot_inference_latency_timeline,
    plot_fresh_input_vs_latency, has_stream_timing,
    plot_ttft_vs_fresh_input, plot_latency_breakdown,
    plot_ttft_vs_prefix_age,
    LATENCY_TIMELINE_PNG, FRESH_INPUT_LATENCY_PNG,
    TTFT_FRESH_INPUT_PNG, LATENCY_BREAKDOWN_PNG,
    TTFT_PREFIX_AGE_PNG,
)
import json


def _fmt(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _pct(x: Any) -> str:
    return f"{x:.0%}" if isinstance(x, (int, float)) else "n/a"


def _dur(seconds: Any) -> str:
    """Seconds as m:ss above a minute, plain seconds below it."""
    if not isinstance(seconds, (int, float)):
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"


def analyze_run(cell: Path, dump_prefixes: bool) -> dict[str, Any]:
    """Run both analyses for one cell, write their artifacts, return merged row."""
    row: dict[str, Any] = {"cell": cell.name}

    d = demand_analyze(cell)
    if d is not None:
        s = d["summary"]
        (cell / DEMAND_JSON).write_text(json.dumps(s, indent=1), encoding="utf-8")
        if s["tokens_available"]:
            viz = cell / "visualizations"
            viz.mkdir(parents=True, exist_ok=True)
        row.update({
            "n_calls": s["n_token_calls"],
            "total_input": s["total_input_tokens"],
            "median_input": s["input_tokens"].get("median", 0),
            "max_context": s.get("max_context", 0),
            "realized_frac": s["cacheread_fraction"],
            "out_in": round(s["total_output_tokens"] / max(s["total_input_tokens"], 1), 3),
        })

    lg = analyze_cell_logical(cell, dump_prefixes=dump_prefixes)
    if lg is not None:
        latency = analyze_latency(lg)
        lg["latency"] = latency
        (cell / LOGICAL_JSON).write_text(json.dumps(lg, indent=1), encoding="utf-8")
        viz = cell / "visualizations"
        viz.mkdir(parents=True, exist_ok=True)
        plot_cache_warming(lg, viz / CACHE_WARMING_PNG)
        plot_prefix_lineage(lg, viz / PREFIX_LINEAGE_PNG)
        plot_inference_latency_timeline(lg, viz / LATENCY_TIMELINE_PNG)
        plot_fresh_input_vs_latency(lg, viz / FRESH_INPUT_LATENCY_PNG)
        if has_stream_timing(lg):
            plot_ttft_vs_fresh_input(lg, viz / TTFT_FRESH_INPUT_PNG)
            plot_latency_breakdown(lg, viz / LATENCY_BREAKDOWN_PNG)
            plot_ttft_vs_prefix_age(lg, viz / TTFT_PREFIX_AGE_PNG)
        row.update({
            "logical_frac": lg["logical_frac"],
            "logical_128_frac": lg["logical_128_frac"],
            "gap_frac": lg["gap_frac"],
            "candidate_count_table": lg["candidate_count_table"],
            "temporal_metrics": lg["temporal_metrics"],
            "latency": latency,
            "has_stream_timing": has_stream_timing(lg),
            "runtime": lg.get("runtime") or {},
        })
    if d is not None and d["summary"]["tokens_available"]:
        viz = cell / "visualizations"
        viz.mkdir(parents=True, exist_ok=True)
        plot_context_growth(
            d["calls"],
            viz / CTX_GROWTH_PNG,
            logical_summary=lg,
        )
    row["has_prefix_dump"] = (cell / PREFIX_DUMP).is_file()
    row["has_logical"] = lg is not None
    return row


def build_report(
    run_dir: Path,
    rows: list[dict[str, Any]],
    results_dir: Path,
    *,
    cells_root: Path | None = None,
) -> Path:
    rows = sorted(rows, key=lambda r: r.get("total_input", 0), reverse=True)
    artifact_root = cells_root or run_dir

    def embedded_image(alt: str, cell: str, filename: str) -> str:
        path = artifact_root / cell / "visualizations" / filename
        if not path.is_file():
            return f"_Unavailable: `{filename}` was not generated for this workload._"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"![{alt}](data:image/png;base64,{encoded})"

    def embedded_prefix_details(cell: str) -> str:
        path = artifact_root / cell / PREFIX_DUMP
        content = html.escape(path.read_text(encoding="utf-8"))
        return (
            f"<details><summary>View {PREFIX_DUMP} per-call details</summary>"
            f"<pre>{content}</pre></details>"
        )

    L = ["# KV-Cache Report — " + run_dir.name, ""]
    L.append(f"_Generated {_dt.date.today().isoformat()} · {len(rows)} cell(s)._")
    L.append("")

    L.append("## Summary")
    L.append("")
    L.append("| cell | vendor | model | calls | total_input | output/input | median | max_ctx | Σinfer | mean infer | span | realized% | logical% | gap% |")
    L.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        runtime = r.get("runtime") or {}
        overall = ((r.get("latency") or {}).get("overall") or {})
        L.append("| {cell} | {vendor} | {model} | {calls} | {ti} | {oi} | {med} | {mx} | {tot} | {avg} | {span} | {rz} | {lg} | {gp} |".format(
            cell=r["cell"], calls=r.get("n_calls", "?"),
            vendor=", ".join(runtime.get("vendors") or ["unknown"]),
            model=", ".join(runtime.get("models") or ["unknown"]),
            ti=_fmt(r.get("total_input", 0)), med=_fmt(r.get("median_input", 0)),
            oi=_pct(r.get("out_in")),
            mx=_fmt(r.get("max_context", 0)),
            tot=_dur(overall.get("total_duration_s")),
            avg=_dur(overall.get("mean_duration_s")),
            span=_dur((r.get("latency") or {}).get("wall_clock_span_s")),
            rz=_pct(r.get("realized_frac")), lg=_pct(r.get("logical_frac")),
            gp=_pct(r.get("gap_frac")),
        ))
    L.append("")
    L.append("**output/input** = Σoutput tokens / Σinput tokens · **Σinfer** = summed LLM call duration, which exceeds **span** when calls overlap · **mean infer** = Σinfer / calls · **span** = first call start to last call end · **realized%** = vendor-reported cacheRead / input · **logical%** = aligned reusable prefix / input · **gap%** = logical reuse − realized reuse.")
    L.append("")

    L.append("## Per cell")
    L.append("")
    for cell_index, r in enumerate(rows):
        if cell_index:
            L.append("---")
            L.append("")
        L.append(f"### {r['cell']}")
        L.append("")
        runtime = r.get("runtime") or {}
        cache_configs = runtime.get("cache_configs") or []
        configured_cache = [config for config in cache_configs if config]
        cache_text = (
            ", ".join(
                json.dumps(config, ensure_ascii=False, sort_keys=True)
                for config in configured_cache
            )
            if configured_cache else "provider-managed/default; no request-level cache controls recorded"
        )
        L.append(f"- **KV-cache request settings**: {cache_text}")
        if r.get("has_stream_timing"):
            stream = (r.get("latency") or {}).get("stream_timing") or {}
            L.append(
                f"- **Stream timing**: {stream.get('n', 0)} calls · "
                f"median TTFT {stream.get('median_ttft_s')}s · "
                f"median TPOT {stream.get('median_tpot_s')}s/token"
            )
        else:
            L.append(
                "- **Stream timing**: unavailable in this trace. TTFT and TPOT "
                "figures are omitted because first-token timestamps cannot be "
                "reconstructed after the run."
            )
        L.append("")
        if r.get("has_logical"):
            L.append("**Compatible source candidates**")
            L.append("")
            L.append("| compatible candidates | calls | share of calls |")
            L.append("| ---: | ---: | ---: |")
            for candidate_row in r.get("candidate_count_table") or []:
                L.append(
                    f"| {candidate_row['compatible_candidates']} | "
                    f"{candidate_row['calls']} | "
                    f"{_pct(candidate_row['call_fraction'])} |"
                )
            L.append("")
            temporal = r.get("temporal_metrics") or {}
            L.append(
                "- **Time since latest compatible prefix** uses current request start "
                "minus the latest completed compatible request end. Calls without a "
                "completed compatible source are excluded from the age table."
            )
            L.append("")
            L.append("| time since latest compatible prefix | calls | logical reusable tokens | realized cacheRead | token capture |")
            L.append("| --- | ---: | ---: | ---: | ---: |")
            for age_bin in temporal.get("age_bins", []):
                L.append(
                    f"| {age_bin['age']} | {age_bin['n']} | "
                    f"{_fmt(age_bin['logical_reusable_tokens'])} | "
                    f"{_fmt(age_bin['realized_cache_read_tokens'])} | "
                    f"{_pct(age_bin['token_capture_rate'])} |"
                )
            L.append("")
            latency = r.get("latency") or {}
            overall = latency.get("overall") or {}
            if overall.get("n"):
                correlations = latency.get("spearman") or {}
                L.append(
                    f"- **Inference latency**: median {overall.get('median_duration_s')}s · "
                    f"p90 {overall.get('p90_duration_s')}s · "
                    f"max {overall.get('max_duration_s')}s · "
                    f"wall span {latency.get('wall_clock_span_s')}s"
                )
                L.append(
                    "- **Latency Spearman ρ**: "
                    f"input {correlations.get('duration_vs_input')} · "
                    f"fresh input {correlations.get('duration_vs_fresh_input')} · "
                    f"output {correlations.get('duration_vs_output')}"
                )
                L.append("")
                L.append("**Top 5 slowest calls**")
                L.append("")
                L.append("| global call | role | duration | input | cacheRead | fresh | output |")
                L.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
                for slow in latency.get("slowest", []):
                    L.append(
                        f"| {slow['global_call']} | {slow['role']} | "
                        f"{slow['duration_s']}s | {_fmt(slow['input'])} | "
                        f"{_fmt(slow['cacheRead'])} | {_fmt(slow['fresh_input'])} | "
                        f"{_fmt(slow['output'])} |"
                    )
                L.append("")
            if r.get("has_prefix_dump"):
                L.append(embedded_prefix_details(r["cell"]))
                L.append("")
        else:
            L.append(
                "realized **n/a** · logical **n/a** because this workload has "
                "no successful token-bearing LLM calls"
            )
        L.append("")
        L.append(embedded_image("context growth", r["cell"], CTX_GROWTH_PNG))
        L.append("")
        if r.get("has_logical"):
            L.append(embedded_image("per-call prefix reuse", r["cell"], CACHE_WARMING_PNG))
            L.append("")
            L.append(
                "Realized KV-cache lineage: an edge exists only when the child call "
                "reports `cacheRead > 0`. The parent is the latest prior call among "
                "those sharing the longest exact prompt prefix. Cache hits without an "
                "identifiable source inside this trace are omitted from the lineage "
                "rather than assigned a synthetic source. Node size represents "
                "`cacheRead / input`, and calls unrelated to a realized hit are omitted. "
                "The size legend is expressed as percentages. A double ring marks an "
                "exact full-prompt repeat."
            )
            L.append("")
            L.append(embedded_image("prefix cache lineage", r["cell"], PREFIX_LINEAGE_PNG))
            L.append("")
            L.append(embedded_image("inference latency timeline", r["cell"], LATENCY_TIMELINE_PNG))
            L.append("")
            L.append(
                "The fresh-input plot uses end-to-end latency, including output decoding."
            )
            L.append("")
            L.append(embedded_image("fresh input vs latency", r["cell"], FRESH_INPUT_LATENCY_PNG))
            L.append("")
            if r.get("has_stream_timing"):
                for alt, filename in [
                    ("TTFT and TPOT vs fresh input", TTFT_FRESH_INPUT_PNG),
                    ("latency breakdown", LATENCY_BREAKDOWN_PNG),
                    ("TTFT vs prefix age", TTFT_PREFIX_AGE_PNG),
                ]:
                    L.append(embedded_image(alt, r["cell"], filename))
                    L.append("")

    markdown_text = "\n".join(L)
    out = run_dir / "kvcache_report.md"
    out.write_text(markdown_text, encoding="utf-8")
    body = markdown.markdown(markdown_text, extensions=["tables"])
    html_out = run_dir / "kvcache_report.html"
    html_out.write_text(
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>"""
        + html.escape(f"KV-Cache Report — {run_dir.name}")
        + """</title>
<style>
body { color: #202124; font: 16px/1.55 system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; padding: 0 1.25rem; }
table { border-collapse: collapse; display: block; overflow-x: auto; }
th, td { border: 1px solid #c7c7c7; padding: 0.4rem 0.65rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
img { display: block; height: auto; margin: 1.25rem auto 2rem; max-width: 100%; }
code { background: #f1f3f4; padding: 0.1rem 0.25rem; }
</style>
</head>
<body>
"""
        + body
        + """
</body>
</html>
""",
        encoding="utf-8",
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--dump-prefixes", action="store_true",
                    help="Also write kvcache_prefixes.txt per cell (reused-prefix text).")
    args = ap.parse_args()

    results_dir = args.results.resolve()
    cells = discover_cells(results_dir, args.runs)
    if not cells:
        print(f"no cells found under {results_dir} for runs={args.runs}")
        return 1

    by_run: dict[Path, list[dict]] = defaultdict(list)
    for cell in cells:
        row = analyze_run(cell, dump_prefixes=args.dump_prefixes)
        by_run[cell.parent].append(row)
        print(f"  {cell.name}: realized={_pct(row.get('realized_frac'))} "
              f"logical={_pct(row.get('logical_frac'))} gap={_pct(row.get('gap_frac'))}")

    for run_dir, rows in by_run.items():
        out = build_report(run_dir, rows, results_dir)
        print(f"report → {out}")
        print(f"browser report → {out.with_suffix('.html')}")
        for row in rows:
            cell_dir = run_dir / row["cell"]
            cell_out = build_report(
                cell_dir,
                [row],
                results_dir,
                cells_root=run_dir,
            )
            print(f"cell report → {cell_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
