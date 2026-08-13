#!/usr/bin/env python3
"""Logical KV-cache demand analysis from existing agent LLM traces (Phase 1).

Scope (what today's traces can answer, GenoMAS-grade cells):

  Q1  How long is each LLM call's prompt/context?      -> input-token stats,
                                                           per role / per phase.
  Q2  How does context grow along the call sequence?   -> input tokens over
                                                           chronological call order,
                                                           per-role growth slope.
  Q3* How much prefix is reused between calls?          -> *amount only*, taken from
                                                           the backend-reported
                                                           ``cacheRead`` tokens. The
                                                           token-level *structure* of
                                                           the shared prefix needs the
                                                           raw ``messages`` array, which
                                                           the current trace does not
                                                           persist (only a sha256
                                                           ``cache_key`` of the whole
                                                           payload). Not computed here.
  Q5  How much already-existing context was re-submitted?
                                                        -> served-cache view:
                                                           sum(cacheRead); logical
                                                           verbatim view: identical
                                                           ``cache_key`` resubmissions.
  Q6  Which workloads most need prefix caching?         -> cross-cell ranking by
                                                           backend-observed reusable
                                                           tokens (a lower bound on the
                                                           reuse opportunity).

Data source: each cell's ``pi_events.jsonl`` (emitted by the GenoMAS/SciLink
loggers). Only ``message_end`` records are read -- each one is self-contained:
``{run_id, message.usage{input,output,cacheRead,totalTokens}, genomas_role,
phase, cache_key, cache_hit}``. ``run_id`` pairs a start/end; we key a call by it.

Caveats baked into the output, not hidden:
  - ``cacheRead`` is what *one specific backend* (FreeInference/qwen here) happened
    to cache, not a backend-independent logical reuse upper bound. It is a realized
    lower bound on reuse opportunity, reported as such.
  - Only GenoMAS cells currently carry real tokens + role. Cells without role are
    still summarized (role="(unknown)"); cells with all-zero usage are flagged
    ``tokens_available=false`` and skipped from token stats.
  - Some runs emit a ``message_end`` twice (same cache_key/input/output, <100ms,
    different run_id). These are collapsed and counted as ``duplicate_emissions``.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

CTX_GROWTH_PNG = "kvcache_context_growth.png"
CELL_JSON = "kvcache_demand.json"

# Colorblind-safe qualitative palette (Okabe-Ito), assigned to roles in first-seen order.
_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
]
_DUP_WINDOW_MS = 100.0


# --------------------------------------------------------------------------- load

def load_calls(cell: Path) -> list[dict[str, Any]]:
    """Return the chronological list of LLM calls for one cell.

    Each call: {t, run_id, role, phase, input, output, cacheRead, total,
    cache_key, cache_hit}. Near-duplicate message_end emissions are collapsed.
    """
    path = cell / "pi_events.jsonl"
    if not path.is_file():
        return []
    raw: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") != "message_end":
            continue
        if o.get("error"):
            continue
        msg = o.get("message") or {}
        usage = msg.get("usage") or {}
        raw.append(
            {
                "t": float(msg.get("timestamp") or 0.0),
                "run_id": o.get("run_id"),
                "role": o.get("agent_role") or o.get("genomas_role") or "(unknown)",
                "phase": o.get("phase") or "(unknown)",
                "input": int(usage.get("input", 0) or 0),
                "output": int(usage.get("output", 0) or 0),
                "cacheRead": int(usage.get("cacheRead", 0) or 0),
                "cacheReadAvailable": usage.get("cacheReadAvailable"),
                "cacheReadSource": usage.get("cacheReadSource"),
                "total": int(usage.get("totalTokens", 0) or 0),
                "cache_key": o.get("cache_key"),
                "cache_hit": bool(o.get("cache_hit")),
            }
        )
    raw.sort(key=lambda c: c["t"])

    # Collapse duplicate emissions: same (cache_key, input, output) within a short
    # window. Two identical LLM responses ~18ms apart are a logging artifact, not
    # two real roundtrips.
    calls: list[dict[str, Any]] = []
    dups = 0
    for c in raw:
        if calls:
            p = calls[-1]
            same = (
                c["cache_key"] == p["cache_key"]
                and c["input"] == p["input"]
                and c["output"] == p["output"]
                and abs(c["t"] - p["t"]) <= _DUP_WINDOW_MS
            )
            if same:
                dups += 1
                continue
        calls.append(c)
    if calls:
        calls[0]["_duplicate_emissions"] = dups
    return calls


# ------------------------------------------------------------------------ metrics

def _slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys vs xs; 0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return num / den


def _least_squares_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return slope and intercept for a least-squares line."""
    if not xs or len(xs) != len(ys):
        return 0.0, 0.0
    slope = _slope(xs, ys)
    intercept = sum(ys) / len(ys) - slope * sum(xs) / len(xs)
    return slope, intercept


def _stats(vals: list[int]) -> dict[str, float]:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "sum": sum(s),
        "min": s[0],
        "median": statistics.median(s),
        "mean": round(statistics.mean(s), 1),
        "p90": s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))],
        "max": s[-1],
    }


def analyze_cell(cell: Path) -> dict[str, Any] | None:
    calls = load_calls(cell)
    if not calls:
        return None
    dup = calls[0].get("_duplicate_emissions", 0)

    token_calls = [c for c in calls if c["input"] > 0 or c["total"] > 0]
    tokens_available = bool(token_calls)

    inputs = [c["input"] for c in token_calls]
    outputs = [c["output"] for c in token_calls]
    cachereads = [c["cacheRead"] for c in token_calls]

    total_input = sum(inputs)
    total_output = sum(outputs)
    total_cacheread = sum(cachereads)
    explicit_cache_flags = [
        c["cacheReadAvailable"] for c in token_calls
        if c.get("cacheReadAvailable") is not None
    ]
    if explicit_cache_flags:
        measured_calls = sum(bool(c.get("cacheReadAvailable")) for c in token_calls)
        cache_read_available = measured_calls == len(token_calls)
    else:
        # Legacy traces did not record availability. Preserve their established
        # interpretation when they contain at least one nonzero observation.
        measured_calls = len(token_calls) if total_cacheread > 0 else 0
        cache_read_available = total_cacheread > 0

    # Q1: per-call length, overall + per role + per phase.
    by_role: dict[str, list[int]] = {}
    by_phase: dict[str, list[int]] = {}
    for c in token_calls:
        by_role.setdefault(c["role"], []).append(c["input"])
        by_phase.setdefault(c["phase"], []).append(c["input"])

    # Q2: context growth. Global series over chronological order + per-role slope.
    growth_by_role: dict[str, dict[str, Any]] = {}
    for role, seq in _role_sequences(token_calls).items():
        idx = list(range(len(seq)))
        ys = [c["input"] for c in seq]
        growth_by_role[role] = {
            "n_calls": len(seq),
            "first_input": ys[0] if ys else 0,
            "max_input": max(ys) if ys else 0,
            "slope_tokens_per_call": round(_slope([float(i) for i in idx], [float(y) for y in ys]), 1),
        }

    # Q5: re-submitted existing context.
    #  (a) served-cache view: backend recognized this many prefix tokens as reusable.
    #  (b) logical verbatim view: identical cache_key sent more than once.
    key_counts: dict[str, dict[str, Any]] = {}
    for c in token_calls:
        k = c["cache_key"]
        if k is None:
            continue
        e = key_counts.setdefault(k, {"count": 0, "input": c["input"]})
        e["count"] += 1
    repeat_keys = {k: v for k, v in key_counts.items() if v["count"] > 1}
    verbatim_resubmissions = sum(v["count"] - 1 for v in repeat_keys.values())
    verbatim_resubmitted_tokens = sum(v["input"] * (v["count"] - 1) for v in repeat_keys.values())
    cache_hits = sum(1 for c in token_calls if c["cache_hit"])

    cacheread_frac = (total_cacheread / total_input) if total_input else 0.0
    verbatim_frac = (verbatim_resubmitted_tokens / total_input) if total_input else 0.0
    # Reusable tokens = backend-realized reuse, plus verbatim resubmissions the
    # backend did NOT already count as cacheRead (avoid double counting per call).
    reusable_tokens = sum(max(c["cacheRead"], 0) for c in token_calls)

    summary = {
        "cell": cell.name,
        "path": str(cell),
        "tokens_available": tokens_available,
        "n_calls": len(calls),
        "n_token_calls": len(token_calls),
        "duplicate_emissions": dup,
        "roles": sorted(by_role),
        "phases": sorted(by_phase),
        # Q1
        "input_tokens": _stats(inputs),
        "output_tokens": _stats(outputs),
        "input_by_role": {r: _stats(v) for r, v in by_role.items()},
        "input_by_phase": {r: _stats(v) for r, v in by_phase.items()},
        # Q2
        "context_growth_by_role": growth_by_role,
        "max_context": max(inputs) if inputs else 0,
        # Q5
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cacheread_tokens": total_cacheread,
        "cacheread_fraction": round(cacheread_frac, 4) if cache_read_available else None,
        "cache_read_available": cache_read_available,
        "cache_read_measured_calls": measured_calls,
        "cache_read_total_calls": len(token_calls),
        "verbatim_resubmissions": verbatim_resubmissions,
        "verbatim_resubmitted_tokens": verbatim_resubmitted_tokens,
        "verbatim_fraction": round(verbatim_frac, 4),
        "cache_hits": cache_hits,
        # Q6 inputs
        "reusable_tokens": reusable_tokens,
        "reuse_potential": round(max(cacheread_frac, verbatim_frac), 4),
    }
    return {"summary": summary, "calls": token_calls}


def _role_sequences(calls: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    seqs: dict[str, list[dict[str, Any]]] = {}
    for c in calls:
        seqs.setdefault(c["role"], []).append(c)
    return seqs


# -------------------------------------------------------------------------- plots

def _role_colors(roles: list[str]) -> dict[str, str]:
    return {r: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(roles)}


def plot_context_growth(
    calls: list[dict[str, Any]],
    out_png: Path,
    logical_summary: dict[str, Any] | None = None,
) -> None:
    """Two panels sharing chronological call index:
    (A) input tokens per call, colored by role  -> Q2 context growth.
    (B) realized, unrealized-reusable, and intrinsically-new prompt tokens.
    """
    roles = list(_role_sequences(calls))
    colors = _role_colors(roles)
    x = list(range(len(calls)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Panel A: line of input tokens, points colored by role, per-role fit line.
    ax1.plot(x, [c["input"] for c in calls], color="#999999", lw=0.8, zorder=1)
    for role in roles:
        xs = [i for i, c in enumerate(calls) if c["role"] == role]
        ys = [calls[i]["input"] for i in xs]
        ax1.scatter(xs, ys, s=22, color=colors[role], label=role, zorder=2)
        if len(xs) >= 2:
            fx = [float(i) for i in xs]
            slope, intercept = _least_squares_fit(fx, [float(v) for v in ys])
            x0, x1 = min(xs), max(xs)
            ax1.plot([x0, x1], [slope * x0 + intercept, slope * x1 + intercept],
                     color=colors[role], lw=1.6, ls="--", zorder=3)
            ax1.annotate(f"{slope:+.0f}/call", (x1, slope * x1 + intercept),
                         color=colors[role], fontsize=7, va="bottom", ha="right")
    ax1.set_ylabel("input (prompt) tokens")
    ax1.set_title("Context growth")
    ax1.legend(fontsize=8, ncol=min(len(roles), 4), loc="upper left")
    ax1.grid(True, alpha=0.25)

    logical_by_run = {
        call["run_id"]: call
        for call in (logical_summary or {}).get("per_call", [])
    }
    realized = [min(max(c["cacheRead"], 0), c["input"]) for c in calls]
    logical_reuse: list[float] = []
    for call, reused in zip(calls, realized):
        logical_call = logical_by_run.get(call.get("run_id"))
        estimate = (
            float(logical_call.get("logical_aligned", 0))
            if logical_call is not None else float(reused)
        )
        logical_reuse.append(max(float(reused), min(estimate, float(call["input"]))))
    intrinsically_new = [
        max(call["input"] - reusable, 0.0)
        for call, reusable in zip(calls, logical_reuse)
    ]
    if logical_summary is not None:
        ax2.bar(
            x, logical_reuse,
            color="#56B4E9", label="logical reuse",
        )
    ax2.bar(
        x, realized,
        color="#E69F00", label="realized reuse (cacheRead)", zorder=3,
    )
    ax2.bar(
        x, intrinsically_new, bottom=logical_reuse,
        color="#B0B0B0", label="intrinsically new",
    )
    ax2.set_ylabel("tokens")
    ax2.set_xlabel("LLM call index (chronological)")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title("Input reuse composition")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def discover_cells(results_dir: Path, run_roots: list[str] | None) -> list[Path]:
    cells: list[Path] = []
    bases = [results_dir / root for root in run_roots] if run_roots else [results_dir]
    for base in bases:
        if base.is_file() and base.name == "pi_events.jsonl":
            cells.append(base.parent)
        elif base.is_dir() and (base / "pi_events.jsonl").is_file():
            cells.append(base)
        elif base.exists():
            for p in sorted(base.rglob("pi_events.jsonl")):
                cells.append(p.parent)
    # de-dup preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for c in cells:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq
