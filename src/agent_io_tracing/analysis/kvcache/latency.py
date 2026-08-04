#!/usr/bin/env python3
"""Inference-latency analysis for KV-cache traces."""
from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATENCY_TIMELINE_PNG = "kvcache_inference_latency_timeline.png"
FRESH_INPUT_LATENCY_PNG = "kvcache_fresh_input_vs_latency.png"
TTFT_FRESH_INPUT_PNG = "kvcache_ttft_vs_fresh_input.png"
LATENCY_BREAKDOWN_PNG = "kvcache_latency_breakdown.png"
TTFT_PREFIX_AGE_PNG = "kvcache_ttft_vs_prefix_age.png"

_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_mean = statistics.mean(left_ranks)
    right_mean = statistics.mean(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left_ranks)
        * sum((y - right_mean) ** 2 for y in right_ranks)
    )
    return numerator / denominator if denominator else None


def _stats(calls: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [call["duration_ms"] / 1000.0 for call in calls]
    if not durations:
        return {"n": 0}
    output_rates = [
        call["output"] / duration
        for call, duration in zip(calls, durations)
        if duration > 0
    ]
    return {
        "n": len(calls),
        "total_duration_s": round(sum(durations), 3),
        "mean_duration_s": round(statistics.mean(durations), 3),
        "median_duration_s": round(statistics.median(durations), 3),
        "p90_duration_s": round(_percentile(durations, 0.90), 3),
        "max_duration_s": round(max(durations), 3),
        "median_input": round(statistics.median(call["input"] for call in calls), 1),
        "median_fresh_input": round(
            statistics.median(call["fresh_input"] for call in calls), 1
        ),
        "median_cache_read": round(
            statistics.median(call["cacheRead"] for call in calls), 1
        ),
        "median_output": round(statistics.median(call["output"] for call in calls), 1),
        "median_e2e_output_tps": (
            round(statistics.median(output_rates), 3) if output_rates else 0.0
        ),
    }


def analyze_latency(summary: dict[str, Any]) -> dict[str, Any]:
    calls = [call for call in summary["per_call"] if call["duration_ms"] >= 0]
    by_role: dict[str, list[dict[str, Any]]] = {}
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        by_role.setdefault(call["role"], []).append(call)
        by_phase.setdefault(call["phase"], []).append(call)
    durations = [call["duration_ms"] / 1000.0 for call in calls]
    slowest = sorted(
        enumerate(calls), key=lambda item: item[1]["duration_ms"], reverse=True
    )[:5]
    stream_calls = [
        call for call in calls if call.get("first_token_ms") is not None
    ]
    ttft = [
        (call["first_token_ms"] - call["start_ms"]) / 1000.0
        for call in stream_calls
    ]
    tpot = [
        ((call.get("last_token_ms") or call["end_ms"]) - call["first_token_ms"])
        / 1000.0 / (call["output"] - 1)
        for call in stream_calls
        if call["output"] > 1
    ]
    return {
        "overall": _stats(calls),
        "wall_clock_span_s": round(
            (max((call["end_ms"] for call in calls), default=0)
             - min((call["start_ms"] for call in calls), default=0)) / 1000.0,
            3,
        ),
        "by_role": {role: _stats(group) for role, group in by_role.items()},
        "by_phase": {phase: _stats(group) for phase, group in by_phase.items()},
        "spearman": {
            "duration_vs_input": _spearman(
                durations, [float(call["input"]) for call in calls]
            ),
            "duration_vs_fresh_input": _spearman(
                durations, [float(call["fresh_input"]) for call in calls]
            ),
            "duration_vs_output": _spearman(
                durations, [float(call["output"]) for call in calls]
            ),
        },
        "stream_timing": {
            "n": len(stream_calls),
            "median_ttft_s": round(statistics.median(ttft), 4) if ttft else None,
            "p90_ttft_s": round(_percentile(ttft, 0.90), 4) if ttft else None,
            "median_tpot_s": round(statistics.median(tpot), 4) if tpot else None,
            "p90_tpot_s": round(_percentile(tpot, 0.90), 4) if tpot else None,
        },
        "slowest": [
            {
                "global_call": index,
                "role": call["role"],
                "phase": call["phase"],
                "duration_s": round(call["duration_ms"] / 1000.0, 3),
                "input": call["input"],
                "cacheRead": call["cacheRead"],
                "fresh_input": call["fresh_input"],
                "output": call["output"],
            }
            for index, call in slowest
        ],
    }


def _role_colors(calls: list[dict[str, Any]]) -> dict[str, str]:
    roles = list(dict.fromkeys(call["role"] for call in calls))
    return {role: _PALETTE[i % len(_PALETTE)] for i, role in enumerate(roles)}


def plot_inference_latency_timeline(summary: dict[str, Any], out_png: Path) -> None:
    calls = summary["per_call"]
    colors = _role_colors(calls)
    x = [call["elapsed_s"] for call in calls]
    durations = [call["duration_ms"] / 1000.0 for call in calls]
    positive_gaps = [b - a for a, b in zip(x, x[1:]) if b > a]
    width = max(statistics.median(positive_gaps) * 0.55, 0.12) if positive_gaps else 0.5

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for role, color in colors.items():
        indices = [i for i, call in enumerate(calls) if call["role"] == role]
        ax.bar(
            [x[i] for i in indices],
            [durations[i] for i in indices],
            width=width,
            color=color,
            alpha=0.78,
            label=role,
        )
        ax.scatter(
            [x[i] for i in indices],
            [durations[i] for i in indices],
            s=[20 + min(calls[i]["output"] / 4, 90) for i in indices],
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.set_xlabel("elapsed wall time (s)")
    ax.set_ylabel("end-to-end inference duration (s)")
    ax.set_title("Inference latency over time")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fresh_input_vs_latency(summary: dict[str, Any], out_png: Path) -> None:
    calls = summary["per_call"]
    colors = _role_colors(calls)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for role, color in colors.items():
        selected = [call for call in calls if call["role"] == role]
        ax.scatter(
            [call["fresh_input"] for call in selected],
            [call["duration_ms"] / 1000.0 for call in selected],
            s=42,
            color=color,
            marker="o",
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
            label=role,
        )
    ax.set_xlabel("fresh input tokens (input − cacheRead)")
    ax.set_ylabel("end-to-end inference duration (s)")
    ax.set_title("Fresh input vs. end-to-end latency")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def has_stream_timing(summary: dict[str, Any]) -> bool:
    return any(call.get("first_token_ms") is not None for call in summary["per_call"])


def plot_ttft_vs_fresh_input(summary: dict[str, Any], out_png: Path) -> None:
    calls = [call for call in summary["per_call"] if call.get("first_token_ms") is not None]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2))
    colors = _role_colors(calls)
    for role, color in colors.items():
        group = [call for call in calls if call["role"] == role]
        marker_sizes = [25 + min(call["input"] / 250, 100) for call in group]
        left.scatter(
            [call["fresh_input"] for call in group],
            [(call["first_token_ms"] - call["start_ms"]) / 1000.0 for call in group],
            s=marker_sizes,
            color=color,
            label=role,
            alpha=0.8,
        )
        decode_group = [call for call in group if call["output"] > 1]
        right.scatter(
            [call["fresh_input"] for call in decode_group],
            [
                ((call.get("last_token_ms") or call["end_ms"]) - call["first_token_ms"])
                / 1000.0
                / (call["output"] - 1)
                for call in decode_group
            ],
            s=[25 + min(call["input"] / 250, 100) for call in decode_group],
            color=color,
            alpha=0.8,
        )
    left.set_ylabel("TTFT (s)")
    left.set_title("TTFT")
    right.set_ylabel("TPOT (s/token)")
    right.set_title("decode TPOT")
    for ax in (left, right):
        ax.set_xlabel("fresh input tokens")
        ax.grid(True, alpha=0.25)
    left.legend(fontsize=8)
    fig.suptitle("TTFT and TPOT vs. fresh input")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latency_breakdown(summary: dict[str, Any], out_png: Path) -> None:
    calls = [call for call in summary["per_call"] if call.get("first_token_ms") is not None]
    x = list(range(len(calls)))
    first = [(call["first_token_ms"] - call["start_ms"]) / 1000.0 for call in calls]
    decode = [
        ((call.get("last_token_ms") or call["end_ms"]) - call["first_token_ms"]) / 1000.0
        for call in calls
    ]
    finalize = [
        (call["end_ms"] - (call.get("last_token_ms") or call["end_ms"])) / 1000.0
        for call in calls
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, first, color="#E69F00", label="request → first token")
    ax.bar(x, decode, bottom=first, color="#0072B2", label="decode")
    ax.bar(
        x, finalize,
        bottom=[a + b for a, b in zip(first, decode)],
        color="#999999", label="client finalization",
    )
    ax.set_xlabel("stream-timed inference index")
    ax.set_ylabel("seconds")
    ax.set_title("Inference latency breakdown")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ttft_vs_prefix_age(summary: dict[str, Any], out_png: Path) -> None:
    calls = [
        call for call in summary["per_call"]
        if call.get("first_token_ms") is not None
        and call.get("time_since_latest_compatible_prefix_s") is not None
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    scatter = ax.scatter(
        [max(call["time_since_latest_compatible_prefix_s"], 0.01) for call in calls],
        [(call["first_token_ms"] - call["start_ms"]) / 1000.0 for call in calls],
        c=[call.get("capture_rate") or 0 for call in calls],
        cmap="viridis", vmin=0, vmax=1, s=55,
    )
    ax.set_xscale("log")
    ax.set_xlabel("time since latest compatible prefix (s)")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("TTFT vs. prefix interval")
    fig.colorbar(scatter, ax=ax, label="capture rate")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
