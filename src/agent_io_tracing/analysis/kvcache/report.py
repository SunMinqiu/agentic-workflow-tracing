#!/usr/bin/env python3
"""One-command KV-cache report for a run: run demand + logical, emit markdown.

    PYTHONPATH=src python3 -m agent_io_tracing.analysis.kvcache.report --runs <run_id>

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
from agent_io_tracing.analysis.kvcache.segments import (
    CONTENT_TAGS,
    analyze_cell_segments,
    write_markdown as write_segments_markdown,
    write_tables as write_segments_tables,
)
from agent_io_tracing.analysis.kvcache.latency import (
    analyze_latency, plot_inference_latency_timeline,
    plot_output_vs_latency, has_stream_timing,
    plot_ttft_vs_fresh_input, plot_latency_breakdown,
    plot_ttft_vs_prefix_age,
    LATENCY_TIMELINE_PNG, OUTPUT_LATENCY_PNG,
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
            "total_output": s["total_output_tokens"],
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
        plot_output_vs_latency(lg, viz / OUTPUT_LATENCY_PNG)
        if has_stream_timing(lg):
            plot_ttft_vs_fresh_input(lg, viz / TTFT_FRESH_INPUT_PNG)
            plot_latency_breakdown(lg, viz / LATENCY_BREAKDOWN_PNG)
            plot_ttft_vs_prefix_age(lg, viz / TTFT_PREFIX_AGE_PNG)
        row.update({
            "logical_frac": lg["logical_frac"],
            "logical_aligned_frac": lg["logical_aligned_frac"],
            "logical_128_frac": lg["logical_128_frac"],
            "gap_frac": lg["gap_frac"],
            "cache_geometry": lg["cache_geometry"],
            "block_demand": lg["block_demand"],
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
    seg = analyze_cell_segments(cell)
    if seg["n_calls"]:
        write_segments_tables(seg, cell)
        write_segments_markdown(seg, cell)
        row["segments"] = {
            "n_segments": seg["n_segments"],
            "cache_size_tokens": seg["cache_size_tokens"],
            "prompt_tokens_total": seg["prompt_tokens_total"],
            "resend_ratio": seg["resend_ratio"],
            "content_breakdown": seg["content_breakdown"],
            "realized_vs_logical": seg["realized_vs_logical"],
            "top_hits": sorted(
                seg["segments"], key=lambda s: -s["realized_tokens"]
            )[:8],
        }
    row["has_segments"] = bool(seg["n_calls"])
    row["has_prefix_dump"] = (cell / PREFIX_DUMP).is_file()
    row["has_logical"] = lg is not None
    return row


def _segment_section(r: dict[str, Any]) -> list[str]:
    """What the never-evicting cache would hold, and what the real one served."""
    seg = r["segments"]
    L: list[str] = []
    L.append("### Cache contents")
    L.append("")
    cb = seg["content_breakdown"]
    L.append("| content | segments | cache space | real hits | hits ÷ space |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    for tag in CONTENT_TAGS:
        e = cb["by_tag"][tag]
        if not e["n_segments"]:
            continue
        L.append(
            f"| {tag} | {e['n_segments']} | {_fmt(e['cache_size_tokens'])} "
            f"({e['cache_share']:.1%}) | {_fmt(e['realized_tokens'])} "
            f"({e['realized_share']:.1%}) | {e['leverage']} |"
        )
    L.append("")
    if cb["unmatched_share"] > 0.95:
        L.append(
            "**Caution:** no content markers matched; the table above is meaningless "
            "on this corpus."
        )
        L.append("")

    hits = [s for s in seg["top_hits"] if s["realized_tokens"] > 0]
    if hits:
        L.append("**What the real cache served:**")
        L.append("")
        L.append("| seg | tokens | content | starts at | hits | tokens served | excerpt |")
        L.append("| --- | ---: | --- | ---: | ---: | ---: | --- |")
        for s in hits:
            L.append(
                f"| {s['segment']} | {_fmt(s['tokens'])} | {s['content_tag']} | "
                f"{_fmt(s['start_offset'])} | {s['realized_hits']} | "
                f"{_fmt(s['realized_tokens'])} | "
                f"`{s['excerpt'][:150].replace('|', chr(92) + '|')}` |"
            )
        L.append("")

    return L


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

    def shared_cells(r: dict[str, Any]) -> str:
        runtime = r.get("runtime") or {}
        return (
            f"| {r['cell']} | {', '.join(runtime.get('vendors') or ['unknown'])} | "
            f"{', '.join(runtime.get('models') or ['unknown'])} | "
            f"{r.get('n_calls', '?')} |"
        )

    L.append("## Summary — tokens")
    L.append("")
    L.append(
        "| cell | vendor | model | calls | total_input | median_input | total_output "
        "| cache_size | resend× | realized% | logical% | gap% | captured% |"
    )
    L.append("| --- | --- | --- |" + " ---: |" * 10)
    for r in rows:
        seg_row = r.get("segments") or {}
        L.append(
            shared_cells(r)
            + f" {_fmt(r.get('total_input', 0))} |"
            f" {_fmt(r.get('median_input', 0))} |"
            f" {_fmt(r.get('total_output', 0))} |"
            f" {_fmt(seg_row['cache_size_tokens']) if seg_row else 'n/a'} |"
            f" {f'{seg_row['resend_ratio']}x' if seg_row.get('resend_ratio') else 'n/a'} |"
            f" {_pct(r.get('realized_frac'))} | {_pct(r.get('logical_frac'))} |"
            f" {_pct(r.get('gap_frac'))} |"
            f" {_pct((seg_row.get('realized_vs_logical') or {}).get('capture_rate'))} |"
        )
    L.append("")

    L.append("## Summary — time")
    L.append("")
    L.append("| cell | vendor | model | calls | Σinfer | span | median TTFT | median TPOT |")
    L.append("| --- | --- | --- |" + " ---: |" * 5)
    for r in rows:
        latency = r.get("latency") or {}
        overall = latency.get("overall") or {}
        stream = latency.get("stream_timing") or {}
        tpot = stream.get("median_tpot_s")
        L.append(
            shared_cells(r)
            + f" {_dur(overall.get('total_duration_s'))} |"
            f" {_dur(latency.get('wall_clock_span_s'))} |"
            f" {_dur(stream.get('median_ttft_s'))} |"
            + (f" {tpot * 1000:.1f}ms/token |" if isinstance(tpot, (int, float)) else " n/a |")
        )
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
        L.append("")
        if r.get("has_logical"):
            L.append("**Possible sources per call**")
            L.append("")
            L.append("| possible sources | calls | share of calls |")
            L.append("| ---: | ---: | ---: |")
            for candidate_row in r.get("candidate_count_table") or []:
                L.append(
                    f"| {candidate_row['possible_sources']} | "
                    f"{candidate_row['calls']} | "
                    f"{_pct(candidate_row['call_fraction'])} |"
                )
            L.append("")
            temporal = r.get("temporal_metrics") or {}
            L.append(
                "- **possible sources** = earlier calls whose prompt shares this "
                "call's matched prefix, i.e. how many of them could have put it in "
                "the cache."
            )
            L.append("")
            L.append("| age of newest possible source | calls | logical reusable tokens | realized cacheRead | token capture |")
            L.append("| --- | ---: | ---: | ---: | ---: |")
            for age_bin in temporal.get("age_bins", []):
                L.append(
                    f"| {age_bin['age']} | {age_bin['n']} | "
                    f"{_fmt(age_bin['logical_reusable_tokens'])} | "
                    f"{_fmt(age_bin['realized_cache_read_tokens'])} | "
                    f"{_pct(age_bin['token_capture_rate'])} |"
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
        if r.get("has_segments"):
            L.extend(_segment_section(r))
        L.append(embedded_image("context growth", r["cell"], CTX_GROWTH_PNG))
        L.append("")
        if r.get("has_logical"):
            L.append(embedded_image("per-call prefix reuse", r["cell"], CACHE_WARMING_PNG))
            L.append("")
            L.append(
                "Logical KV-cache lineage: every call with a reusable exact prompt "
                "prefix is connected to the latest prior call sharing its longest "
                "prefix. These edges describe logical reuse opportunities, not the "
                "physical source of vendor-reported `cacheRead`. Node size represents "
                "aligned logical reusable prefix tokens divided by input tokens. "
                "Calls without logical prefix reuse are omitted. "
                "The size legend is expressed as percentages. A double ring marks an "
                "exact full-prompt repeat."
            )
            L.append("")
            L.append(embedded_image("prefix cache lineage", r["cell"], PREFIX_LINEAGE_PNG))
            L.append("")
            L.append(embedded_image("inference latency timeline", r["cell"], LATENCY_TIMELINE_PNG))
            L.append("")
            L.append(embedded_image("output length vs latency", r["cell"], OUTPUT_LATENCY_PNG))
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
