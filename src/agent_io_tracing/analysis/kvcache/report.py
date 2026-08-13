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
    realized_provenance,
    read_server_prefix_cache,
    CELL_JSON as LOGICAL_JSON, CACHE_WARMING_PNG,
PREFIX_LINEAGE_PNG, PREFIX_DUMP,
)
from agent_io_tracing.analysis.kvcache.summary import (
    PAGE_CSS, token_table, time_table,
)
from agent_io_tracing.analysis.results_index import write_results_index
from agent_io_tracing.analysis.kvcache.segments import (
    analyze_cell_segments,
    write_tables as write_segments_tables,
)
from agent_io_tracing.analysis.kvcache.latency import (
    analyze_latency, plot_inference_latency_timeline,
    plot_output_vs_latency, has_stream_timing,
    plot_ttft_vs_fresh_input, plot_latency_breakdown,
    LATENCY_TIMELINE_PNG, OUTPUT_LATENCY_PNG,
    TTFT_FRESH_INPUT_PNG, LATENCY_BREAKDOWN_PNG,
)
import json


LINEAGE_NOTES = "kvcache_lineage_notes.md"


def _fmt(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _pct(x: Any) -> str:
    return f"{x:.2%}" if isinstance(x, (int, float)) else "n/a"


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
            "cached_tokens": s["total_cacheread_tokens"],
            "cache_read_available": s.get("cache_read_available", False),
            "out_in": round(s["total_output_tokens"] / max(s["total_input_tokens"], 1), 3),
        })

    lg = analyze_cell_logical(cell, dump_prefixes=dump_prefixes)
    if lg is not None:
        latency = analyze_latency(lg)
        lg["latency"] = latency
        # Prefer complete per-request response data. Fall back to the cell-level
        # Prometheus delta only when cached_tokens was absent from one or more
        # responses. An explicit cached_tokens=0 remains a measured zero.
        server_cache = read_server_prefix_cache(cell, row.get("total_input"))
        if server_cache is not None:
            lg["server_prefix_cache"] = server_cache
            row["server_prefix_cache"] = server_cache
        # Record whether realized was measured at all, so the tables can print
        # "n/a" instead of a 0 that reads as a finding.
        lg["realized_provenance"] = realized_provenance(
            cell, lg.get("per_call") or [], lg.get("server_prefix_cache"),
        )
        row["realized_provenance"] = lg["realized_provenance"]
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
            "tokenizer": lg.get("tokenizer"),
            "token_source": lg.get("token_source") or {},
            "serving_config": lg.get("serving_config") or {},
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
        row["segments"] = {
            "n_segments": seg["n_segments"],
            "prompt_tokens_total": seg["prompt_tokens_total"],
            "content_breakdown": seg["content_breakdown"],
            "content_examples": seg["content_examples"],
            "realized_vs_logical": seg["realized_vs_logical"],
        }
    row["has_segments"] = bool(seg["n_calls"])
    row["has_prefix_dump"] = (cell / PREFIX_DUMP).is_file()
    row["has_logical"] = lg is not None
    return row


def _content_examples(examples: dict[str, list], tags: list[str]) -> list[str]:
    """One collapsible block per content type, with real served text inside."""
    L: list[str] = []
    for tag in tags:
        rows = examples.get(tag) or []
        if not rows:
            L.append(
                f"<p><i>{html.escape(tag)}: nothing of this type was ever served "
                "from cache.</i></p>"
            )
            continue
        L.append(
            f"<details><summary>{html.escape(tag)} — top {len(rows)} by tokens served"
            "</summary>"
        )
        for rank, s in enumerate(rows, 1):
            section = html.escape(s.get("section") or "?")
            L.append(
                f"<p><b>{rank}. {section} — served {_fmt(s['tag_tokens'])} "
                f"{tag} tokens</b> · to {s['realized_hits']} call(s) · first sent "
                f"by call {s['first_call']} · in {s['n_calls']} calls' prompts</p>"
            )
            L.append(f"<pre>{html.escape(s['hit_sample'])}</pre>")
        L.append("</details>")
    L.append("")
    return L


def _segment_section(r: dict[str, Any]) -> list[str]:
    """What the real cache served, split by what the text is."""
    seg = r["segments"]
    L: list[str] = []
    L.append("### Cache contents")
    L.append("")
    cb = seg["content_breakdown"]
    tags = cb["tags"]
    served = cb["n_served_segments"]
    L.append("| content | tokens served | share of tokens | hit segments | share of segments |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    # every type is listed, a zero included: "never served" is an answer
    ranked = sorted(tags, key=lambda t: -cb["by_tag"][t]["realized_tokens"])
    for tag in ranked:
        e = cb["by_tag"][tag]
        L.append(
            f"| {tag} | {_fmt(e['realized_tokens'])} | {e['realized_share']:.2%} | "
            f"{e['n_segments']} | {e['segment_share']:.2%} |"
        )
    L.append(
        f"| **all** | **{_fmt(cb['realized_tokens_total'])}** | **100.00%** | "
        f"**{served}** | — |"
    )
    L.append("")
    L.append(
        f"Tokens split each hit across the sections it covers, so they add to 100%. "
        f"Segments count a hit under every type it touched, so they do not."
    )
    L.append("")
    L.extend(_content_examples(seg.get("content_examples") or {}, tags))
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

    L.append("## Summary — tokens")
    L.append("")
    L.extend(token_table(rows))
    L.append("")

    L.append("## Summary — time")
    L.append("")
    L.extend(time_table(rows))
    L.append("")

    L.append("## Per task")
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
        # Only worth a line when the request actually carried cache controls;
        # "none recorded" is the default and says nothing.
        if configured_cache:
            L.append(f"- **KV-cache request settings**: {cache_text}")
        serving = r.get("serving_config") or {}
        if serving:
            L.append(
                "- **Server config**: "
                + ", ".join(
                    f"`{k}`={serving[k]}"
                    for k in ("cache_dtype", "block_size", "prefix_match_unit",
                              "enable_prefix_caching",
                              "num_gpu_blocks", "gpu_memory_utilization")
                    if k in serving
                )
            )
        L.append("")
        # Which tokenizer produced logical.  Without this a reader cannot tell
        # a cell corrected against the engine's own tokenizer from one still
        # counted with tiktoken, and the two differ by orders of magnitude on
        # tool- or image-bearing workloads.
        source = r.get("token_source") or {}
        n_server = source.get("server_tokenized_calls") or 0
        if n_server:
            n_exact = source.get("n_exact")
            n_tokenized = source.get("n_tokenized")
            residual = source.get("residual_pct")
            detail = (
                f" ({n_exact}/{n_tokenized} exact against the server's recorded "
                f"`input`"
                if n_exact is not None and n_tokenized else ""
            )
            shortfall = (
                f", {residual}% short where the trace lost `tool_calls`)"
                if residual else ")" if detail else ""
            )
            L.append(f"- **Token counts**: from the engine's own `/tokenize`{detail}{shortfall}.")
        elif r.get("tokenizer"):
            L.append(
                f"- **Token counts**: re-tokenized locally with "
                f"`{r['tokenizer']}`; absolute totals differ from the provider's."
            )
        L.append("")
        server = r.get("server_prefix_cache")
        if server:
            # The summary table already prints the rate with its † marker.
            # Only the failure case needs saying.
            if not server.get("sole_client"):
                L.append(
                    f"- **Realized reuse**: {server['hit_rate']:.2%} read from "
                    f"the engine's counters, but **not attributable** — "
                    f"{_fmt(server['queries'])} queried tokens against this "
                    f"cell's {_fmt(r.get('total_input', 0))} input tokens means "
                    f"another client shared the server."
                )
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
            L.append("| age of newest possible source | calls | logical reusable tokens | realized cacheRead |")
            L.append("| --- | ---: | ---: | ---: |")
            for age_bin in temporal.get("age_bins", []):
                L.append(
                    f"| {age_bin['age']} | {age_bin['n']} | "
                    f"{_fmt(age_bin['logical_reusable_tokens'])} | "
                    f"{_fmt(age_bin['realized_cache_read_tokens'])} | "
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
            lineage_notes = artifact_root / r["cell"] / LINEAGE_NOTES
            if lineage_notes.is_file():
                L.append(lineage_notes.read_text(encoding="utf-8").strip())
                L.append("")
            L.append(embedded_image("inference latency timeline", r["cell"], LATENCY_TIMELINE_PNG))
            L.append("")
            L.append(embedded_image("output length vs latency", r["cell"], OUTPUT_LATENCY_PNG))
            L.append("")
            if r.get("has_stream_timing"):
                for alt, filename in [
                    ("TTFT and TPOT vs fresh input", TTFT_FRESH_INPUT_PNG),
                    ("latency breakdown", LATENCY_BREAKDOWN_PNG),
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
""" + PAGE_CSS + """</style>
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

    # one report per task: a run holding three tasks is three experiments, and
    # the results index is what puts them side by side
    for run_dir, rows in by_run.items():
        for row in rows:
            out = build_report(
                run_dir / row["cell"], [row], results_dir, cells_root=run_dir
            )
            print(f"report → {out.with_suffix('.html')}")
    write_results_index(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
