#!/usr/bin/env python3
"""The two summary tables, rendered from one place.

Every per-task report opens with them and the results index is nothing but
them for all tasks at once. Writing the columns twice would let the two drift,
so both callers come here; the index only adds a leading column carrying the
task name and its two report links.

A row is the shape ``report.analyze_run`` returns. ``results_index`` rebuilds
that shape from the per-cell JSON artifacts instead of re-running the analysis.
"""
from __future__ import annotations

from typing import Any, Callable

PAGE_CSS = """\
body { color: #202124; font: 16px/1.55 system-ui, sans-serif; margin: 2rem auto; max-width: 1400px; padding: 0 1.25rem; }
table { border-collapse: collapse; display: block; overflow-x: auto; }
th, td { border: 1px solid #c7c7c7; padding: 0.4rem 0.65rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
img { display: block; height: auto; margin: 1.25rem auto 2rem; max-width: 100%; }
code { background: #f1f3f4; padding: 0.1rem 0.25rem; }
"""


def fmt(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def pct(x: Any) -> str:
    return f"{x:.2%}" if isinstance(x, (int, float)) else "n/a"


def dur(seconds: Any) -> str:
    """Seconds with two decimal places, using minutes above one minute."""
    if not isinstance(seconds, (int, float)):
        return "n/a"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds) // 60
    remainder = seconds - minutes * 60
    return f"{minutes}m{remainder:05.2f}s"


TOKEN_HEADERS = [
    "vendor", "model", "calls", "total input", "median input", "total output",
    "cache size", "realized reuse", "logical reuse", "reuse gap",
]
TIME_HEADERS = [
    "vendor", "model", "calls", "total inference", "wall", "median TTFT",
    "median TPOT",
]


def _identity(r: dict[str, Any]) -> list[str]:
    runtime = r.get("runtime") or {}
    return [
        ", ".join(runtime.get("vendors") or ["unknown"]),
        ", ".join(runtime.get("models") or ["unknown"]),
        str(r.get("n_calls", "?")),
    ]


def realized_cell(r: dict[str, Any]) -> str:
    """Realized reuse, from the vendor when it reports it, else the server.

    vLLM leaves ``usage.prompt_tokens_details`` null, so ``cacheRead`` is 0 on
    every call even while its prefix cache runs at 46%.  Printing that 0 in the
    headline table while the body of the same report states the real figure is
    worse than printing nothing.  The dagger marks the number as cell-level,
    read from the engine's counters rather than summed from per-call usage.
    """
    realized = r.get("realized_frac")
    if realized:
        return pct(realized)
    server = r.get("server_prefix_cache") or {}
    rate = server.get("hit_rate")
    if rate is not None:
        return f"{pct(rate)}†" if server.get("sole_client") else f"{pct(rate)}†?"
    # Nothing reported anything.  "0%" would read as "the cache did nothing",
    # which is a claim we cannot make -- one run displayed 0% while the server
    # was at 29%, because its counters were never read.
    if r.get("realized_provenance") == "unmeasured":
        return "unmeasured"
    return pct(realized)


def gap_cell(r: dict[str, Any]) -> str:
    """How much of the ideal reuse the backend left on the table.

    ``gap_frac`` was computed against a realized of 0, so on vLLM it just
    restated logical.  When the server reports its own hit rate, subtract that
    instead -- the difference between 47% ideal and 46% achieved is the whole
    point of the column.
    """
    if not r.get("realized_frac"):
        rate = (r.get("server_prefix_cache") or {}).get("hit_rate")
        aligned = r.get("logical_aligned_frac")
        if rate is not None and aligned is not None:
            return f"{pct(round(aligned - rate, 4))}†"
        # gap = logical - realized is undefined when realized is unknown;
        # printing logical again under a different heading invents a finding.
        if r.get("realized_provenance") == "unmeasured":
            return "unmeasured"
    return pct(r.get("gap_frac"))


def token_cells(r: dict[str, Any]) -> list[str]:
    seg = r.get("segments") or {}
    return _identity(r) + [
        fmt(r.get("total_input", 0)),
        fmt(r.get("median_input", 0)),
        fmt(r.get("total_output", 0)),
        fmt(seg["cache_size_tokens"]) if seg else "n/a",
        realized_cell(r),
        pct(r.get("logical_frac")),
        gap_cell(r),
    ]


def time_cells(r: dict[str, Any]) -> list[str]:
    latency = r.get("latency") or {}
    stream = latency.get("stream_timing") or {}
    tpot = stream.get("median_tpot_s")
    return _identity(r) + [
        dur((latency.get("overall") or {}).get("total_duration_s")),
        dur(latency.get("wall_clock_span_s")),
        dur(stream.get("median_ttft_s")),
        f"{tpot * 1000:.2f}ms/token" if isinstance(tpot, (int, float)) else "n/a",
    ]


def _table(
    rows: list[dict[str, Any]],
    headers: list[str],
    cells: Callable[[dict[str, Any]], list[str]],
    lead_headers: list[str],
    lead_cells: Callable[[dict[str, Any]], list[str]],
) -> list[str]:
    """One markdown table.

    ``lead_headers`` are the caller's own leading columns: a report names the
    cell, the index names the task and links to its two reports. A row carrying
    ``_group`` renders as a heading instead of data, which is how the index
    separates one workflow from the next inside a single table.
    """
    width = len(lead_headers) + len(headers)
    # the leading columns plus vendor and model are text; every number is right
    text_columns = len(lead_headers) + 2
    L = [
        "| " + " | ".join(lead_headers + headers) + " |",
        "|" + " --- |" * text_columns + " ---: |" * (width - text_columns),
    ]
    for r in rows:
        if r.get("_group"):
            L.append(f"| **{r['_group']}** |" + " |" * (width - 1))
        elif r.get("_empty"):
            # a workflow with no runs yet: say so once, do not fake zeroes
            L.append("| " + " | ".join(lead_cells(r)) + " |"
                     + " |" * len(headers))
        else:
            L.append("| " + " | ".join(lead_cells(r) + cells(r)) + " |")
    return L


def _cell_name(r: dict[str, Any]) -> list[str]:
    return [str(r["cell"])]


def token_table(
    rows: list[dict[str, Any]],
    lead_headers: list[str] | None = None,
    lead_cells: Callable[[dict[str, Any]], list[str]] = _cell_name,
) -> list[str]:
    L = _table(rows, TOKEN_HEADERS, token_cells,
               lead_headers or ["cell"], lead_cells)
    # Explain the dagger where it is used, not in a distant legend.
    if any(r.get("server_prefix_cache") for r in rows):
        L.append("")
        L.append(
            "† cell-level, from the engine's own counters. `?` = another "
            "client shared the server; not attributable."
        )
    return L


def time_table(
    rows: list[dict[str, Any]],
    lead_headers: list[str] | None = None,
    lead_cells: Callable[[dict[str, Any]], list[str]] = _cell_name,
) -> list[str]:
    return _table(rows, TIME_HEADERS, time_cells,
                  lead_headers or ["cell"], lead_cells)
