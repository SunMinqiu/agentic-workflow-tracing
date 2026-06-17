#!/usr/bin/env python3
from __future__ import annotations
"""
Visualize parsed strace data from Claude code runs.

Generates interactive HTML (Plotly) and static PNG (Matplotlib) visualizations
to analyze I/O behavior patterns from Linux strace output.

This is adapted from visualize_traces.py for the strace parser output format.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from extract_phases import PhaseAnalysis, load_phases


# =============================================================================
# Color Schemes
# =============================================================================

# =============================================================================
# Color schemes — three independent palettes
# =============================================================================
#
# Three lanes / scopes in the visualizations, three palettes that DO NOT share
# hex codes so the same color cannot be misread across lanes:
#
#   TOOL_COLORS         — pi-coding-agent tools (warm, saturated)
#   SUBAGENT_COLORS     — LangGraph subagents (mid-tone purples/pinks)
#   SYSCALL_CATEGORY_COLORS — FS / syscall categories (cool, desaturated)
#
# The legacy `category_colors` dict (previously identical to TOOL_COLORS hex
# codes, causing the matplotlib timeline to silently mis-label scatter dots)
# is replaced by SYSCALL_CATEGORY_COLORS below.

# Tool type colors (warm/saturated, consistent across all visualizations).
# Known pi tools get fixed colors so their identity is stable across traces.
TOOL_COLORS = {
    "Bash":  "#e74c3c",     # Red
    "Read":  "#e67e22",     # Dark orange
    "Write": "#f39c12",     # Orange
    "Edit":  "#f1c40f",     # Yellow
    "Glob":  "#d35400",     # Burnt orange
    "Grep":  "#c0392b",     # Dark red
}

# Fallback palette for unknown tools (SRAgent tools etc.).  Selected by
# deterministic hash so the same tool name always picks the same color across
# traces — important for cross-trace comparison.
TOOL_FALLBACK_PALETTE = [
    "#FF6B9D", "#FFA07A", "#FFD166", "#F4A261", "#E76F51",
    "#FF8C42", "#FF6F61", "#D4A373", "#C9A227", "#B8860B",
]
SCRIPT_EXEC_COLOR = "#9B59B6"


def color_for_tool(name: str) -> str:
    """Deterministic color for any tool name, warm palette."""
    if name in {"ScriptExec", "SubprocessExec"}:
        return SCRIPT_EXEC_COLOR
    if name in TOOL_COLORS:
        return TOOL_COLORS[name]
    # hash() is salted per-Python-process in 3.3+; use a stable hash.
    import hashlib
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    return TOOL_FALLBACK_PALETTE[h % len(TOOL_FALLBACK_PALETTE)]


# Subagent colors (mid-tone purples/pinks/magentas — distinct from tool warm
# palette and from syscall cool palette).
SUBAGENT_KNOWN_COLORS: dict[str, str] = {}  # filled in by users if needed
SUBAGENT_FALLBACK_PALETTE = [
    "#9b59b6", "#8e44ad", "#c39bd3", "#d2b4de", "#bb8fce",
    "#a569bd", "#7d3c98", "#884ea0", "#af7ac5", "#76448a",
]


def color_for_subagent(name: str) -> str:
    """Deterministic mid-tone color for any subagent name."""
    if name in SUBAGENT_KNOWN_COLORS:
        return SUBAGENT_KNOWN_COLORS[name]
    import hashlib
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    return SUBAGENT_FALLBACK_PALETTE[h % len(SUBAGENT_FALLBACK_PALETTE)]


# Syscall category colors — cool/desaturated so they recede visually and
# don't compete with the warm tool bars in the same chart.  Hex codes
# deliberately chosen to NOT collide with TOOL_COLORS or SUBAGENT_*.
SYSCALL_CATEGORY_COLORS = {
    "metadata": "#5DADE2",   # Light blue
    "data":     "#48C9B0",   # Mint
    "control":  "#7FB3D5",   # Sky blue
    "modify":   "#A2D9CE",   # Pale teal
    "process":  "#3498DB",   # Blue
    "blocking": "#85C1E9",   # Pale blue (currently never appears — see notes)
    "network":  "#2980B9",   # Deep blue
    "other":    "#AAB7B8",   # Cool gray
}

# Distinct hues so adjacent segments are easy to tell apart. LLM=green,
# File-IO=blue, CPU=yellow, Process-mgmt=dark slate, Orchestration=orange.
# (Previously File-IO was teal #48C9B0, nearly identical to LLM green.)
RESOURCE_COLORS = {
    "LLM": "#2ECC71",
    "Tool File-IO": "#3498DB",
    "Tool CPU compute": "#F1C40F",
    "Tool Wait": "#AEB6BF",
    "File-IO": "#3498DB",
    "CPU compute": "#F1C40F",
    "Wait": "#AEB6BF",
    "Process-mgmt": "#34495E",
    "Orch File-IO": "#5DADE2",
    "Orch CPU compute": "#E67E22",
    "Orch Wait": "#CACFD2",
    "Unaccounted": "#7F8C8D",
    "Parallel": "#9b59b6",
    "Idle": "#E5E7E9",
    "Orchestration": "#E67E22",
}

# LLM segment color (used in agent_timeline semantic lane).  Saturated
# accent that is distinct from tool warm + subagent purple + syscall cool.
LLM_COLOR = "#2ECC71"  # Green — readable on white, no overlap with above.
LLM_SUBAGENT_COLOR = "#A8E6A3"

# Backwards-compat alias for old call sites that reference `category_colors`
# inline.  Existing code uses a local dict identical to this; consolidating
# here removes the duplicate-definition bug that allowed silent drift.
category_colors = SYSCALL_CATEGORY_COLORS

# Operation type colors (top 10) — used by other viz that aren't categorized.
OPERATION_COLORS = px.colors.qualitative.Set3

# Syscall category classification
SYSCALL_CATEGORIES = {
    "metadata": {
        "stat", "fstat", "lstat", "statx", "fstatat64", "newfstatat",
        "access", "faccessat",
        "getdents64", "getdents",
        "readlink", "readlinkat",
    },
    "data": {
        "read", "write", "pread64", "pwrite64",
        "readv", "writev", "preadv", "pwritev", "preadv2", "pwritev2",
    },
    "control": {
        "open", "openat", "close",
        "lseek", "fcntl", "ioctl",
        "chdir", "fchdir", "getcwd",
        "mmap", "munmap",
        "dup", "dup2", "dup3",
    },
    "modify": {
        "mkdir", "mkdirat", "rmdir",
        "unlink", "unlinkat",
        "rename", "renameat", "renameat2",
        "chmod", "fchmod", "chown", "fchown",
        "truncate", "ftruncate", "fsync", "fdatasync", "sync_file_range",
    },
    "process": {
        "clone", "clone3", "fork", "vfork",
        "execve",
        "wait4", "waitpid", "waitid",
    },
    "blocking": {
        "select", "pselect6", "poll", "ppoll",
        "epoll_wait", "epoll_pwait",
        "futex",
        "nanosleep", "clock_nanosleep",
    },
    "network": {
        "recvfrom", "sendto",
        "accept", "connect",
        "socket", "bind", "listen",
        "recv", "send",
        "recvmsg", "sendmsg",
    },
}

SYSCALL_CATEGORY_TO_RESOURCE = {
    "metadata": "File-IO",
    "data": "File-IO",
    "control": "File-IO",
    "modify": "File-IO",
    "blocking": "Wait",
    "process": "Process-mgmt",
}

# Reverse lookup syscall -> category, built once for vectorized classification
# (pandas .map) instead of a per-row classify_syscall() call over millions of rows.
_SYSCALL_TO_CATEGORY = {
    sc: cat for cat, scs in SYSCALL_CATEGORIES.items() for sc in scs
}


def classify_syscall(syscall: str) -> str:
    """Classify a syscall into a category."""
    for category, syscalls in SYSCALL_CATEGORIES.items():
        if syscall in syscalls:
            return category
    return "other"


def resource_for_syscall(syscall: str) -> str | None:
    """Map a syscall name to the resource taxonomy used by time accounting.

    Network is out of scope (None). Blocking maps to "Wait" so it is both shown
    as its own segment AND subtracted from the CPU-compute residual (keeping CPU
    honest). The sum pie bounds Wait per tool so it can't overcount.
    """
    category = classify_syscall(syscall)
    if category == "network":
        return None
    return SYSCALL_CATEGORY_TO_RESOURCE.get(category)


def _is_code_exec_tool(tool_name: str) -> bool:
    return tool_name in {
        "Bash",
        "CodeExec",
        "ScriptExec",
        "SubprocessExec",
        "PythonExec",
        "ShellExec",
    }


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class StraceData:
    """Container for loaded and processed strace data."""
    tool_calls_df: pd.DataFrame
    fs_entries_df: pd.DataFrame
    summary: dict
    start_time: datetime
    end_time: datetime
    duration_seconds: float


# Process-lifetime cache: parsed.json is large (millions of fs_entries) and a
# full visualization run loads it from several entry points (strace charts +
# each agent chart via _load_agent_timeline_data). Parsing + pd.to_datetime on
# millions of rows several times dominates runtime. Cache the immutable result
# by resolved path so it is parsed exactly once per run. Every mutating consumer
# already does .copy() before touching the DataFrames, so sharing is safe.
_PARSED_JSON_CACHE: dict[str, StraceData] = {}


def load_parsed_json(filepath: Path) -> StraceData:
    """Load parsed.json and convert to pandas DataFrames (cached per path)."""
    cache_key = str(Path(filepath).resolve())
    cached = _PARSED_JSON_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with open(filepath) as f:
        data = json.load(f)

    # Convert tool calls to DataFrame
    tool_calls = data["tool_calls"]
    tc_df = pd.DataFrame(tool_calls)
    tc_df["start_time"] = pd.to_datetime(tc_df["start_time"], format="ISO8601")
    tc_df["end_time"] = pd.to_datetime(tc_df["end_time"], format="ISO8601")
    tc_df["duration_ms"] = (tc_df["end_time"] - tc_df["start_time"]).dt.total_seconds() * 1000

    # Convert fs entries to DataFrame
    fs_entries = data["fs_entries"]
    fs_df = pd.DataFrame(fs_entries)
    fs_df["timestamp"] = pd.to_datetime(fs_df["timestamp"], format="ISO8601")

    # Map syscall -> operation for consistency with original visualize_traces.py conventions
    if "syscall" in fs_df.columns:
        fs_df["operation"] = fs_df["syscall"]

    # Calculate time bounds. Vectorized min/max over the datetime columns avoids
    # materializing a multi-million-element Python list (the old `list(...) +
    # list(...)`), which was a large, pointless allocation on big traces.
    min_candidates = [tc_df["start_time"].min(), tc_df["end_time"].min()]
    max_candidates = [tc_df["start_time"].max(), tc_df["end_time"].max()]
    if len(fs_df) and "timestamp" in fs_df.columns:
        min_candidates.append(fs_df["timestamp"].min())
        max_candidates.append(fs_df["timestamp"].max())
    start_time = min(t for t in min_candidates if pd.notna(t))
    end_time = max(t for t in max_candidates if pd.notna(t))
    duration = (end_time - start_time).total_seconds()

    # Add relative time columns (seconds from start)
    tc_df["start_rel"] = (tc_df["start_time"] - start_time).dt.total_seconds()
    tc_df["end_rel"] = (tc_df["end_time"] - start_time).dt.total_seconds()
    fs_df["time_rel"] = (fs_df["timestamp"] - start_time).dt.total_seconds()

    # Sort tool calls by start time so the timeline y-axis is chronological.
    # The raw log is in completion order which misrepresents concurrent calls.
    tc_df = tc_df.sort_values("start_time").reset_index(drop=True)

    result = StraceData(
        tool_calls_df=tc_df,
        fs_entries_df=fs_df,
        summary=data["summary"],
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration,
    )
    _PARSED_JSON_CACHE[cache_key] = result
    return result


# =============================================================================
# Visualization: Timeline
# =============================================================================

PROCESS_SPAWN_SYSCALLS = {"clone", "clone3", "fork", "vfork"}


def _extract_child_pid(entry: pd.Series) -> int | None:
    """Extract child PID from a process-spawn syscall entry."""
    return_value = entry.get("return_value")
    if isinstance(return_value, (int, np.integer)) and return_value > 0:
        return int(return_value)
    if isinstance(return_value, float) and np.isfinite(return_value) and return_value.is_integer() and return_value > 0:
        return int(return_value)

    args = entry.get("args")
    if isinstance(args, str):
        match = re.search(r"child_pid=(\d+)", args)
        if match:
            return int(match.group(1))
    return None


def _build_process_info(data: StraceData) -> pd.DataFrame:
    """Reconstruct process lifetimes and parent relationships from fs entries."""
    from collections import deque as _deque

    fs_df = data.fs_entries_df.copy()
    if len(fs_df) == 0:
        return pd.DataFrame(columns=[
            "pid", "ppid", "first_seen", "last_seen", "lifespan_s",
            "row_index", "is_main", "label", "depth",
        ])

    if "operation" not in fs_df.columns and "syscall" in fs_df.columns:
        fs_df["operation"] = fs_df["syscall"]

    lifespan_df = (
        fs_df.groupby("pid", observed=True)["time_rel"]
        .agg(first_seen="min", last_seen="max")
        .reset_index()
    )

    parent_by_pid: dict[int, int] = {}
    spawn_entries = fs_df[fs_df["operation"].isin(PROCESS_SPAWN_SYSCALLS)].sort_values("time_rel")
    for _, entry in spawn_entries.iterrows():
        parent_pid = entry.get("pid")
        if not isinstance(parent_pid, (int, np.integer)):
            continue
        child_pid = _extract_child_pid(entry)
        if child_pid is None or child_pid == int(parent_pid):
            continue
        # Keep the earliest observed parent assignment for stability.
        parent_by_pid.setdefault(child_pid, int(parent_pid))

    process_df = lifespan_df.copy()
    process_df["ppid"] = process_df["pid"].map(parent_by_pid)
    process_df["lifespan_s"] = process_df["last_seen"] - process_df["first_seen"]
    process_df = process_df.sort_values(["first_seen", "pid"]).reset_index(drop=True)
    process_df["row_index"] = np.arange(len(process_df))

    main_pid = None
    summary_pids = data.summary.get("pids", [])
    if summary_pids:
        main_pid = summary_pids[0]
    elif len(process_df) > 0:
        main_pid = int(process_df.iloc[0]["pid"])

    process_df["is_main"] = process_df["pid"] == main_pid if main_pid is not None else False
    process_df["label"] = process_df.apply(
        lambda row: f"PID {int(row['pid'])}" + (" (main)" if row["is_main"] else ""),
        axis=1,
    )

    # Compute tree depth via BFS from the main PID.
    children_of: dict[int, list[int]] = {}
    for _, row in process_df.iterrows():
        ppid = row["ppid"]
        if pd.notna(ppid):
            children_of.setdefault(int(ppid), []).append(int(row["pid"]))

    depth_map: dict[int, int] = {}
    if main_pid is not None:
        mp = int(main_pid)
        depth_map[mp] = 0
        queue = _deque([mp])
        while queue:
            cur = queue.popleft()
            for ch in children_of.get(cur, []):
                if ch not in depth_map:
                    depth_map[ch] = depth_map[cur] + 1
                    queue.append(ch)

    fallback_depth = max(depth_map.values(), default=0) + 1
    process_df["depth"] = process_df["pid"].apply(
        lambda p: depth_map.get(int(p), fallback_depth)
    )

    # Add per-PID syscall event count for importance ranking.
    event_counts = fs_df.groupby("pid", observed=True).size()
    process_df["event_count"] = process_df["pid"].map(event_counts).fillna(0).astype(int)

    # When the tree is flat (most non-main processes share one depth level),
    # subdivide by activity so the collapse buttons offer a useful middle
    # ground between "main only" and "all 150+ processes".
    non_main = process_df[~process_df["is_main"]]
    if len(non_main) > 20:
        dominant_depth = non_main["depth"].mode().iloc[0] if len(non_main) > 0 else 1
        same_depth_count = (non_main["depth"] == dominant_depth).sum()
        if same_depth_count / len(non_main) > 0.8:
            top_quartile = non_main["event_count"].quantile(0.75)
            process_df["depth"] = process_df.apply(
                lambda row: (
                    row["depth"] if row["is_main"]
                    else int(dominant_depth) if row["event_count"] >= top_quartile
                    else int(dominant_depth) + 1
                ),
                axis=1,
            )

    return process_df


def create_process_timeline_plotly(data: StraceData, output_path: Path) -> None:
    """Create interactive per-process timeline with collapsible depth levels.

    Processes are grouped by tree depth (distance from the main PID) and
    rendered as separate Plotly traces.  Buttons let the viewer expand or
    collapse subprocess levels so the chart height stays manageable even for
    traces with hundreds of PIDs.
    """
    tc_df = data.tool_calls_df
    process_df = _build_process_info(data)

    fig = go.Figure()

    # -- Tool-call window shading (layout shapes, always visible). ----------
    seen_tools: set[str] = set()
    for _, row in tc_df.iterrows():
        tool_name = row["tool_name"]
        color = color_for_tool(tool_name)
        fig.add_vrect(
            x0=row["start_rel"],
            x1=row["end_rel"],
            fillcolor=color,
            opacity=0.12,
            line_width=0,
            layer="below",
        )
        if tool_name not in seen_tools:
            seen_tools.add(tool_name)
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color, symbol="square"),
                name=f"{tool_name} window",
                showlegend=True,
            ))

    n_always_visible = len(fig.data)

    if len(process_df) == 0:
        fig.update_layout(
            title="Per-Process Timeline (Lifespan, Parent Links, Tool Call Windows)",
            xaxis_title="Time (seconds from start)",
            yaxis_title="Process",
        )
        fig.write_html(output_path)
        return

    # -- One Bar trace per depth level. -------------------------------------
    max_depth = int(process_df["depth"].max())
    depth_palette = ["#2c3e50", "#34495e", "#5d6d7e", "#7f8c8d", "#95a5a6", "#bdc3c7"]
    depth_names = {
        0: "Main process",
        1: "Active subprocesses",
        2: "Other subprocesses",
        3: "Deep subprocesses",
    }

    bar_trace_depths: list[int] = []
    for d in range(max_depth + 1):
        group = process_df[process_df["depth"] == d]
        if len(group) == 0:
            continue
        bar_color = depth_palette[min(d, len(depth_palette) - 1)]
        colors = np.where(group["is_main"], "#2c3e50", bar_color)
        dname = depth_names.get(d, f"Subprocess depth {d}")

        customdata = np.stack([
            group["pid"].astype(int).astype(str),
            group["ppid"].fillna(-1).astype(int).astype(str),
            group["first_seen"].round(6).astype(str),
            group["last_seen"].round(6).astype(str),
            group["lifespan_s"].round(6).astype(str),
            group["depth"].astype(str),
        ], axis=1)

        fig.add_trace(go.Bar(
            x=group["lifespan_s"],
            y=group["label"],
            base=group["first_seen"],
            orientation="h",
            marker_color=colors,
            name=dname,
            customdata=customdata,
            hovertemplate=(
                "<b>PID %{customdata[0]}</b> (depth %{customdata[5]})<br>"
                "Parent PID: %{customdata[1]}<br>"
                "First seen: %{customdata[2]}s<br>"
                "Last seen: %{customdata[3]}s<br>"
                "Lifespan: %{customdata[4]}s<br>"
                "<extra></extra>"
            ),
        ))
        bar_trace_depths.append(d)

    # -- Precompute parent-link annotation data. ----------------------------
    pid_to_label = {int(r["pid"]): r["label"] for _, r in process_df.iterrows()}
    arrow_data: list[dict] = []
    for _, row in process_df.iterrows():
        ppid = row["ppid"]
        if pd.isna(ppid):
            continue
        ppid_int = int(ppid)
        if ppid_int not in pid_to_label:
            continue
        arrow_data.append({
            "x": float(row["first_seen"]),
            "y": row["label"],
            "ay": pid_to_label[ppid_int],
            "child_depth": int(row["depth"]),
        })

    def _make_annotations(max_d: int) -> list[dict]:
        return [
            dict(
                x=a["x"], y=a["y"],
                xref="x", yref="y",
                ax=a["x"], ay=a["ay"],
                axref="x", ayref="y",
                text="", showarrow=True,
                arrowhead=2, arrowsize=0.8, arrowwidth=1,
                arrowcolor="#8e44ad", opacity=0.6,
            )
            for a in arrow_data
            if a["child_depth"] <= max_d
        ]

    # -- Build depth-filter buttons. ----------------------------------------
    thresholds: list[int] = []
    for d in [0, 1, 2]:
        if d <= max_depth:
            thresholds.append(d)
    if max_depth not in thresholds:
        thresholds.append(max_depth)

    buttons = []
    for threshold in thresholds:
        vis: list[bool] = [True] * n_always_visible
        for d in bar_trace_depths:
            vis.append(d <= threshold)

        cats = process_df[process_df["depth"] <= threshold]["label"].tolist()
        h = max(450, 260 + len(cats) * 22)

        if threshold == max_depth and max_depth > 2:
            btn_label = f"All ({len(process_df)} procs)"
        elif threshold == 0:
            n = len(cats)
            btn_label = f"Main only ({n})"
        else:
            n = len(cats)
            btn_label = f"Depth \u2264 {threshold} ({n} procs)"

        buttons.append(dict(
            label=btn_label,
            method="update",
            args=[
                {"visible": vis},
                {
                    "yaxis.categoryarray": cats[::-1],
                    "yaxis.categoryorder": "array",
                    "height": h,
                    "annotations": _make_annotations(threshold),
                },
            ],
        ))

    # -- Choose a sensible default view. ------------------------------------
    if len(process_df) > 30 and any(t == 1 for t in thresholds):
        default_threshold = 1
    else:
        default_threshold = max_depth
    default_idx = next(
        (i for i, t in enumerate(thresholds) if t == default_threshold),
        len(thresholds) - 1,
    )

    default_cats = process_df[process_df["depth"] <= default_threshold]["label"].tolist()

    # Set initial visibility for non-default depth traces.
    for i, d in enumerate(bar_trace_depths):
        fig.data[n_always_visible + i].visible = (d <= default_threshold)

    fig.update_layout(
        title="Per-Process Timeline (Lifespan, Parent Links, Tool Call Windows)",
        xaxis_title="Time (seconds from start)",
        yaxis_title="Process",
        barmode="overlay",
        height=max(450, 260 + len(default_cats) * 22),
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(
            categoryorder="array",
            categoryarray=default_cats[::-1],
        ),
        annotations=_make_annotations(default_threshold),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.0,
            y=1.15,
            xanchor="left",
            yanchor="top",
            pad=dict(r=10, t=10),
            showactive=True,
            buttons=buttons,
            active=default_idx,
        )],
    )
    fig.update_xaxes(range=[-0.1, data.duration_seconds + 0.1])

    fig.write_html(output_path)


def create_process_timeline_matplotlib(data: StraceData, output_path: Path) -> None:
    """Create static per-process timeline with parent links and tool windows."""
    tc_df = data.tool_calls_df
    process_df = _build_process_info(data)

    fig, ax = plt.subplots(figsize=(15, max(6, 2 + len(process_df) * 0.35)))

    for _, row in tc_df.iterrows():
        color = color_for_tool(row["tool_name"])
        ax.axvspan(row["start_rel"], row["end_rel"], color=color, alpha=0.12, zorder=0)

    if len(process_df) > 0:
        bar_colors = ["#2c3e50" if is_main else "#7f8c8d" for is_main in process_df["is_main"]]
        ax.barh(
            process_df["row_index"],
            process_df["lifespan_s"],
            left=process_df["first_seen"],
            color=bar_colors,
            alpha=0.9,
            height=0.65,
            zorder=2,
            label="Process lifespan",
        )

        pid_to_row = {int(row["pid"]): int(row["row_index"]) for _, row in process_df.iterrows()}
        for _, row in process_df.iterrows():
            parent_pid = row["ppid"]
            if pd.isna(parent_pid):
                continue
            parent_pid = int(parent_pid)
            if parent_pid not in pid_to_row:
                continue
            child_time = float(row["first_seen"])
            ax.annotate(
                "",
                xy=(child_time, float(row["row_index"])),
                xytext=(child_time, float(pid_to_row[parent_pid])),
                arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1, alpha=0.7),
                zorder=3,
            )

        ax.set_yticks(process_df["row_index"])
        ax.set_yticklabels(process_df["label"], fontsize=8)

    legend_items = [
        mpatches.Patch(color="#2c3e50", label="Main process"),
        mpatches.Patch(color="#7f8c8d", label="Child/other process"),
    ]
    # Auto-extend legend to every tool that actually appeared in tc_df, using
    # color_for_tool so SRAgent / unknown tools also get a deterministic color
    # and a legend entry (instead of falling to gray "Unknown").
    for tool_name in tc_df["tool_name"].unique() if len(tc_df) else []:
        legend_items.append(
            mpatches.Patch(color=color_for_tool(tool_name), alpha=0.5,
                           label=f"{tool_name} window")
        )
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)

    ax.set_xlabel("Time (seconds from start)")
    ax.set_ylabel("Process")
    ax.set_title("Per-Process Timeline (Lifespan, Parent Links, Tool Call Windows)")
    ax.set_xlim(-0.1, data.duration_seconds + 0.1)
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_timeline_plotly(data: StraceData, output_path: Path) -> None:
    """Create interactive timeline visualization with Plotly."""
    tc_df = data.tool_calls_df
    fs_df = data.fs_entries_df
    
    fig = go.Figure()
    
    # Add tool calls as horizontal bars
    for idx, row in tc_df.iterrows():
        color = color_for_tool(row["tool_name"])
        
        # Extract command preview for Bash tools
        label = row["tool_name"]
        if row["tool_name"] == "Bash" and isinstance(row["input_params"], dict):
            cmd = row["input_params"].get("command", "")
            label = f"Bash: {cmd[:40]}..." if len(cmd) > 40 else f"Bash: {cmd}"
        elif isinstance(row["input_params"], dict) and "file_path" in row["input_params"]:
            fp = row["input_params"]["file_path"]
            label = f"{row['tool_name']}: {Path(fp).name}"
        
        fig.add_trace(go.Bar(
            x=[row["end_rel"] - row["start_rel"]],
            y=[idx],
            base=[row["start_rel"]],
            orientation='h',
            name=row["tool_name"],
            marker_color=color,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Start: {row['start_rel']:.3f}s<br>"
                f"Duration: {row['duration_ms']:.1f}ms<br>"
                f"Tool ID: {row['tool_id'][:20]}...<br>"
                "<extra></extra>"
            ),
            showlegend=bool(idx == tc_df[tc_df["tool_name"] == row["tool_name"]].index[0]),
            legendgroup=row["tool_name"],
        ))
    
    # Add FS operations as a category-colored point cloud. Instead of randomly
    # sampling 5000 of millions of points (lossy, non-deterministic), rasterize
    # event TIMES to a sub-pixel grid and emit one marker per occupied
    # (category, time-bucket): full coverage, deterministic, bounded count.
    # Sub-pixel density and per-point hover are dropped (invisible / meaningless
    # at this density).
    span = data.duration_seconds
    if len(fs_df) > 0 and span > 0:
        bucket = span / _RASTER_PX
        cat_series = fs_df["operation"].map(_SYSCALL_TO_CATEGORY).fillna("other")
        bidx = (fs_df["time_rel"].to_numpy() / bucket).astype("int64")
        xs: list[float] = []
        colors: list[str] = []
        for cat in cat_series.unique():
            occ = np.unique(bidx[(cat_series == cat).to_numpy()])
            xs.extend((occ * bucket).tolist())
            colors.extend([SYSCALL_CATEGORY_COLORS.get(cat, "#95a5a6")] * len(occ))
        if xs:
            fig.add_trace(go.Scatter(
                x=xs,
                y=np.random.uniform(-0.8, len(tc_df) - 0.2, len(xs)),
                mode='markers',
                marker=dict(size=3, color=colors, opacity=0.5),
                hovertemplate="time: %{x:.3f}s<extra></extra>",
                name="FS Operations",
                showlegend=True,
            ))
    
    fig.update_layout(
        title="Tool Calls and FS Operations Timeline (strace)",
        xaxis_title="Time (seconds from start)",
        yaxis_title="Tool Call Index",
        barmode='overlay',
        height=400 + len(tc_df) * 30,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="closest",
    )
    
    fig.write_html(output_path)


def create_timeline_matplotlib(data: StraceData, output_path: Path) -> None:
    """Create static timeline visualization with Matplotlib."""
    tc_df = data.tool_calls_df
    fs_df = data.fs_entries_df
    
    fig, ax = plt.subplots(figsize=(14, max(6, 2 + len(tc_df) * 0.5)))
    
    # Plot tool calls as horizontal bars
    for idx, row in tc_df.iterrows():
        color = color_for_tool(row["tool_name"])
        ax.barh(idx, row["end_rel"] - row["start_rel"], left=row["start_rel"],
                color=color, alpha=0.8, height=0.6)
        
        # Add label
        label = row["tool_name"]
        if row["tool_name"] == "Bash" and isinstance(row["input_params"], dict):
            cmd = row["input_params"].get("command", "")
            cmd_short = cmd.split()[0] if cmd else ""
            label = f"{row['tool_name']} ({cmd_short})"
        ax.text(row["start_rel"], idx, f" {label}", va='center', fontsize=8)
    
    # Plot FS operations as a category-colored point cloud, rasterized to a
    # sub-pixel time grid (one marker per occupied (category, bucket)) instead
    # of a random sample — full coverage, deterministic, bounded. Colors come
    # from SYSCALL_CATEGORY_COLORS (cool palette), distinct from the tool bars.
    cat_series = fs_df["operation"].map(_SYSCALL_TO_CATEGORY).fillna("other") \
        if len(fs_df) else pd.Series(dtype=object)
    cats_in_data: list[str] = sorted(set(cat_series.unique())) if len(cat_series) else []
    span = data.duration_seconds
    if len(fs_df) > 0 and span > 0:
        bucket = span / _RASTER_PX
        bidx = (fs_df["time_rel"].to_numpy() / bucket).astype("int64")
        xs: list[float] = []
        colors_list: list[str] = []
        for cat in cats_in_data:
            occ = np.unique(bidx[(cat_series == cat).to_numpy()])
            xs.extend((occ * bucket).tolist())
            colors_list.extend([SYSCALL_CATEGORY_COLORS.get(cat, "#AAB7B8")] * len(occ))
        if xs:
            ax.scatter(
                xs,
                np.random.uniform(-0.5, len(tc_df) - 0.5, len(xs)),
                c=colors_list, s=2, alpha=0.3,
            )

    # Legend: TWO blocks — tool bars (from tc_df, using actual per-tool colors
    # including auto-assigned ones) AND syscall categories that actually
    # appeared in fs_sample.  Fixes the prior bug where the legend listed
    # TOOL_COLORS labels but the scatter dots were colored by syscall category.
    tool_names_in_data = list(tc_df["tool_name"].unique()) if len(tc_df) else []
    tool_patches = [
        mpatches.Patch(color=color_for_tool(t), label=f"tool: {t}")
        for t in tool_names_in_data
    ]
    cat_patches = [
        mpatches.Patch(color=SYSCALL_CATEGORY_COLORS.get(c, "#AAB7B8"), label=f"syscall: {c}")
        for c in cats_in_data
    ]
    ax.legend(handles=tool_patches + cat_patches, loc='upper right', fontsize=7,
              ncol=2 if (len(tool_patches) + len(cat_patches)) > 8 else 1)
    
    ax.set_xlabel("Time (seconds from start)")
    ax.set_ylabel("Tool Call Index")
    ax.set_title("Tool Calls and FS Operations Timeline (strace)")
    ax.set_xlim(-0.1, data.duration_seconds + 0.1)
    ax.set_ylim(-1, len(tc_df))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Visualization: I/O Rate Timeline
# =============================================================================

def _get_tool_label(row: pd.Series, max_len: int = 40) -> str:
    """Extract a short label for a tool call, showing up to max_len chars of the command."""
    tool_name = row["tool_name"]
    params = row["input_params"]

    if tool_name == "Bash" and isinstance(params, dict):
        cmd = params.get("command", "")
        # Just show up to max_len characters of the command
        cmd_trunc = cmd[:max_len] + ("..." if len(cmd) > max_len else "")
        return f"{tool_name}: {cmd_trunc}"
    elif isinstance(params, dict):
        if "file_path" in params:
            fp = params["file_path"]
            name = Path(fp).name
            if len(name) > max_len:
                name = name[:max_len] + "..."
            return f"{tool_name}: {name}"
        elif "pattern" in params:
            pat = params["pattern"]
            if len(pat) > max_len:
                pat = pat[:max_len] + "..."
            return f"{tool_name}: {pat}"
        elif "query" in params:
            q = params["query"]
            if len(q) > max_len:
                q = q[:max_len] + "..."
            return f"{tool_name}: {q}"
    return tool_name


def _syscall_size_hover_fragment(entry: pd.Series) -> str:
    """Return a hover line for syscall size/bytes when present, else empty."""
    # parse_ebpf.py writes `bytes_transferred`; keep `size` as a compatibility
    # fallback for other parsed.json producers.
    size_val = entry.get("bytes_transferred")
    if pd.isna(size_val):
        size_val = entry.get("size")
    if pd.isna(size_val):
        return ""

    if isinstance(size_val, (int, np.integer)):
        if int(size_val) <= 0:
            return ""
        size_text = f"{int(size_val)} bytes"
    elif isinstance(size_val, float):
        if not np.isfinite(size_val) or size_val <= 0:
            return ""
        if np.isfinite(size_val) and size_val.is_integer():
            size_text = f"{int(size_val)} bytes"
        else:
            size_text = f"{size_val:g} bytes"
    else:
        size_text = str(size_val)

    return f"Size: {size_text}<br>"


def _syscall_access_mode_hover_fragment(entry: pd.Series) -> str:
    """Return a hover line for openat access mode when present."""
    op = entry.get("operation")
    if pd.isna(op):
        op = entry.get("syscall")
    if op != "openat":
        return ""

    access_mode = entry.get("access_mode")
    if pd.isna(access_mode) or not access_mode:
        open_flags = entry.get("open_flags")
        if pd.isna(open_flags) or not open_flags:
            return ""
        # parse_ebpf.py encodes open flags like "O_RDONLY|O_CLOEXEC|..."
        access_mode = str(open_flags).split("|", 1)[0]

    return f"Access mode: {access_mode}<br>"


def _compute_label_positions(tc_df: pd.DataFrame, y_max: float, min_gap: float = 0.5) -> list[float]:
    """Compute y positions for labels to avoid overlap.
    
    Uses alternating levels and checks for horizontal proximity.
    """
    positions = []
    levels = [0.95, 0.85, 0.75, 0.65]  # Fractions of y_max
    
    for i, (_, row) in enumerate(tc_df.iterrows()):
        start = row["start_rel"]
        
        # Find the best level that doesn't conflict with recent labels
        best_level = 0
        for level_idx, level in enumerate(levels):
            # Check if any recent label at this level would overlap
            conflict = False
            for j in range(max(0, i - len(levels)), i):
                if positions[j] == level * y_max:
                    prev_end = tc_df.iloc[j]["end_rel"]
                    if start - prev_end < min_gap:
                        conflict = True
                        break
            if not conflict:
                best_level = level_idx
                break
        
        positions.append(levels[best_level] * y_max)
    
    return positions


def create_io_rate_plotly(data: StraceData, output_path: Path) -> None:
    """Create I/O rate over time chart with Plotly."""
    fs_df = data.fs_entries_df.copy()
    tc_df = data.tool_calls_df
    
    # Ensure errno column exists
    if "errno" not in fs_df.columns:
        fs_df["errno"] = None
    
    # Bin operations by time (100ms bins)
    bin_size = 0.1  # seconds
    bins = np.arange(0, data.duration_seconds + bin_size, bin_size)
    fs_df["time_bin"] = pd.cut(fs_df["time_rel"], bins=bins)
    
    # Count per bin
    bin_counts = fs_df.groupby("time_bin", observed=True).size()
    bin_centers = [(b.left + b.right) / 2 for b in bin_counts.index]
    
    # Count errors per bin
    error_counts = fs_df[fs_df["errno"].notna()].groupby("time_bin", observed=True).size()
    
    # Calculate max y for label positioning
    y_max = bin_counts.max() if len(bin_counts) > 0 else 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add I/O rate line first (so vrects appear behind)
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=bin_counts.values,
        mode='lines',
        name='Syscalls per 100ms',
        line=dict(color='#2c3e50', width=2),
        fill='tozeroy',
        fillcolor='rgba(44, 62, 80, 0.3)',
    ), secondary_y=False)
    
    # Add error rate as markers on secondary y-axis
    if len(error_counts) > 0:
        error_centers = [(b.left + b.right) / 2 for b in error_counts.index]
        fig.add_trace(go.Scatter(
            x=error_centers,
            y=error_counts.values,
            mode='markers',
            name='Errors',
            marker=dict(color='#e74c3c', size=10, symbol='x'),
            hovertemplate="Time: %{x:.2f}s<br>Errors: %{y}<extra></extra>",
        ), secondary_y=True)
    
    # Compute label positions to avoid overlap
    label_y_positions = _compute_label_positions(tc_df, y_max, min_gap=0.3)
    
    # Add tool call regions as vertical bands (without annotations)
    for i, (_, row) in enumerate(tc_df.iterrows()):
        color = color_for_tool(row["tool_name"])
        fig.add_vrect(
            x0=row["start_rel"],
            x1=row["end_rel"],
            fillcolor=color,
            opacity=0.2,
            line_width=0,
        )
        
        # Add label as a separate annotation with computed y position
        label = _get_tool_label(row, max_len=35)
        label_x = (row["start_rel"] + row["end_rel"]) / 2
        label_y = label_y_positions[i]
        
        fig.add_annotation(
            x=label_x,
            y=label_y,
            text=label,
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowwidth=1,
            arrowcolor=color,
            ax=0,
            ay=-20,
            font=dict(size=9, color=color),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color,
            borderwidth=1,
            borderpad=2,
            textangle=-30,  # Diagonal labels to prevent overlap
        )
    
    fig.update_layout(
        title="I/O Rate Over Time (with Error Markers)",
        xaxis_title="Time (seconds from start)",
        height=500,  # Increased height for labels
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        # Add margin at top for labels
        margin=dict(t=80),
    )
    
    fig.update_yaxes(title_text="Syscalls per 100ms", secondary_y=False)
    fig.update_yaxes(title_text="Errors per 100ms", secondary_y=True, color='#e74c3c')
    
    fig.write_html(output_path)


def create_io_rate_matplotlib(data: StraceData, output_path: Path) -> None:
    """Create I/O rate over time chart with Matplotlib."""
    fs_df = data.fs_entries_df.copy()
    tc_df = data.tool_calls_df
    
    # Ensure errno column exists
    if "errno" not in fs_df.columns:
        fs_df["errno"] = None
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax2 = ax.twinx()  # Secondary y-axis for errors
    
    # Bin operations
    bin_size = 0.1
    bins = np.arange(0, data.duration_seconds + bin_size, bin_size)
    counts, edges = np.histogram(fs_df["time_rel"], bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Bin errors
    error_times = fs_df[fs_df["errno"].notna()]["time_rel"]
    error_counts, _ = np.histogram(error_times, bins=bins)
    
    # Calculate max y for label positioning
    y_max = counts.max() if len(counts) > 0 else 100
    
    # Plot rate
    ax.fill_between(centers, counts, alpha=0.3, color='#2c3e50')
    ax.plot(centers, counts, color='#2c3e50', linewidth=1.5, label='Syscalls')
    
    # Plot error markers
    error_mask = error_counts > 0
    if error_mask.any():
        ax2.scatter(centers[error_mask], error_counts[error_mask], 
                   color='#e74c3c', s=50, marker='x', label='Errors', zorder=5)
    
    # Compute label positions to avoid overlap
    label_y_positions = _compute_label_positions(tc_df, y_max, min_gap=0.3)
    
    # Add tool call regions and labels
    for i, (_, row) in enumerate(tc_df.iterrows()):
        color = color_for_tool(row["tool_name"])
        ax.axvspan(row["start_rel"], row["end_rel"], alpha=0.2, color=color)
        
        # Add label
        label = _get_tool_label(row, max_len=25)
        label_x = (row["start_rel"] + row["end_rel"]) / 2
        label_y = label_y_positions[i]
        
        # Draw label with arrow pointing down, diagonal text to prevent overlap
        ax.annotate(
            label,
            xy=(label_x, y_max * 0.1),  # Arrow points to bottom of chart
            xytext=(label_x, label_y),
            fontsize=7,
            color=color,
            ha='left',
            va='bottom',
            rotation=-30,  # Diagonal labels to prevent overlap
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
        )
    
    ax.set_xlabel("Time (seconds from start)")
    ax.set_ylabel("Syscalls per 100ms", color='#2c3e50')
    ax2.set_ylabel("Errors per 100ms", color='#e74c3c')
    ax.set_title("I/O Rate Over Time (with Error Markers)")
    ax.set_xlim(0, data.duration_seconds)
    ax.set_ylim(0, y_max * 1.15)  # Add headroom for labels
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Visualization: Tool Call Syscalls (per-tool breakdown)
# =============================================================================

# Category colors for syscall types
CATEGORY_COLORS = {
    "metadata": "#9b59b6",  # Purple
    "data": "#3498db",      # Blue
    "control": "#f39c12",   # Orange
    "modify": "#e74c3c",    # Red
    "process": "#2ecc71",   # Green
    "blocking": "#e67e22",  # Dark Orange
    "network": "#1abc9c",   # Teal
    "other": "#95a5a6",     # Gray
}


def _limit_syscalls_per_type(
    df: pd.DataFrame,
    max_entries_per_type: int,
    group_col: str = "operation",
    sort_col: str = "duration_ms",
) -> pd.DataFrame:
    """Limit syscall rows independently per syscall type.

    Keeps at most ``max_entries_per_type`` entries per ``group_col`` value.
    When limiting is needed for a type, the longest-duration rows are kept.
    """
    if len(df) == 0:
        return df
    if max_entries_per_type <= 0:
        return df.iloc[0:0].copy()

    limited_dfs: list[pd.DataFrame] = []
    for _, type_df in df.groupby(group_col, sort=False):
        if len(type_df) <= max_entries_per_type:
            limited_dfs.append(type_df)
        else:
            limited_dfs.append(type_df.nlargest(max_entries_per_type, sort_col))

    if not limited_dfs:
        return df.iloc[0:0].copy()
    return pd.concat(limited_dfs, ignore_index=True)


MAX_TOOL_CALL_SUBPLOTS = 100


def _select_top_tool_calls(tc_df: pd.DataFrame, max_tools: int) -> pd.DataFrame:
    """Select the top tool calls by duration when there are too many for subplots."""
    if len(tc_df) <= max_tools:
        return tc_df
    return tc_df.nlargest(max_tools, "duration_ms").sort_values("start_time").reset_index(drop=True)


def create_tool_syscalls_plotly(
    data: StraceData,
    output_path: Path,
    max_syscalls_per_type: int = 5000,
    max_tool_calls: int = MAX_TOOL_CALL_SUBPLOTS,
) -> None:
    """Create interactive visualization showing syscalls for each tool call.
    
    Each tool call gets its own subplot showing syscalls as horizontal bars
    representing their duration. Y-axis = syscall type, X-axis = time.
    Long syscalls are immediately visible as long bars.
    """
    tc_df = data.tool_calls_df
    fs_df = data.fs_entries_df

    if len(tc_df) == 0:
        return

    # Sort fs entries by timestamp ONCE so each tool's time window can be sliced
    # with searchsorted (O(log n)) instead of a full-DataFrame boolean mask per
    # tool (the old code scanned all millions of rows ~3× per tool). Selected
    # rows are identical; only the cost changes.
    fs_sorted = fs_df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    _ts_sorted = fs_sorted["timestamp"].to_numpy()

    # Pre-aggregate matched syscall time per tool once (was a full scan per tool).
    if len(fs_sorted) and "matched_tool_call" in fs_sorted.columns:
        _matched_ms_by_tool = (
            fs_sorted.groupby("matched_tool_call")["duration"].sum() * 1000.0
        )
    else:
        _matched_ms_by_tool = pd.Series(dtype=float)

    def _window_slice(tool_start, tool_end) -> pd.DataFrame:
        lo = np.searchsorted(_ts_sorted, np.datetime64(tool_start), side="left")
        hi = np.searchsorted(_ts_sorted, np.datetime64(tool_end), side="right")
        return fs_sorted.iloc[lo:hi]

    total_tool_calls = len(tc_df)
    tc_df = _select_top_tool_calls(tc_df, max_tool_calls)
    capped = total_tool_calls > len(tc_df)
    
    # Calculate grid dimensions (favor more columns as tool count grows)
    n_tools = len(tc_df)
    if n_tools <= 6:
        n_cols = min(3, n_tools)
    elif n_tools <= 20:
        n_cols = 4
    else:
        n_cols = 5
    n_rows = (n_tools + n_cols - 1) // n_cols
    
    # Create subplot titles with syscall time summary
    subplot_titles = []
    for _, row in tc_df.iterrows():
        tool_id = row["tool_id"]
        tool_start = row["start_time"]
        tool_end = row["end_time"]
        
        # Count syscalls in time window
        tool_syscalls = _window_slice(tool_start, tool_end)
        n_syscalls = len(tool_syscalls)

        total_syscall_ms = float(_matched_ms_by_tool.get(tool_id, 0.0))
        
        label = _get_tool_label(row, max_len=40)
        if label.startswith("Bash: "):
            label = label[len("Bash: "):]
        duration_ms = row["duration_ms"]
        
        # Show sampling indicator only when any syscall type exceeds the per-type limit
        per_type_counts = tool_syscalls["operation"].value_counts()
        has_per_type_sampling = (per_type_counts > max_syscalls_per_type).any()
        metadata = f"{duration_ms:.0f}ms wall | {total_syscall_ms:.1f}ms syscall | {n_syscalls} calls"
        if has_per_type_sampling:
            metadata += " | sampled"
        subplot_titles.append(f"{label}<br><span style='font-size:10px;color:#666'>{metadata}</span>")
    
    v_spacing = min(0.06, 0.8 / max(n_rows - 1, 1))
    h_spacing = min(0.06, 0.8 / max(n_cols - 1, 1))
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=v_spacing,
        horizontal_spacing=h_spacing,
    )
    
    # Process each tool call
    for idx, (_, tc_row) in enumerate(tc_df.iterrows()):
        row_num = idx // n_cols + 1
        col_num = idx % n_cols + 1
        
        tool_id = tc_row["tool_id"]
        tool_start = tc_row["start_time"]
        tool_end = tc_row["end_time"]
        
        # Filter fs entries to this tool's time window (searchsorted slice)
        tool_fs = _window_slice(tool_start, tool_end).copy()
        
        if len(tool_fs) == 0:
            continue
        
        # Calculate relative time (ms from tool start)
        tool_fs["time_rel_ms"] = (tool_fs["timestamp"] - tool_start).dt.total_seconds() * 1000
        tool_fs["duration_ms"] = tool_fs["duration"] * 1000  # Convert to ms
        
        # Classify syscalls and determine match status (vectorized map)
        tool_fs["category"] = tool_fs["operation"].map(_SYSCALL_TO_CATEGORY).fillna("other")
        tool_fs["is_matched"] = tool_fs["matched_tool_call"] == tool_id
        
        # Store FULL stats before sampling (for summary display)
        full_syscall_counts = tool_fs.groupby("operation").size()
        full_syscall_durations = tool_fs.groupby("operation")["duration_ms"].sum()
        total_syscalls_before_sample = len(tool_fs)
        
        # Limit independently per syscall type (not as one total budget).
        has_per_type_sampling = (full_syscall_counts > max_syscalls_per_type).any()
        if has_per_type_sampling:
            tool_fs = _limit_syscalls_per_type(
                tool_fs,
                max_entries_per_type=max_syscalls_per_type,
                group_col="operation",
                sort_col="duration_ms",
            )
        
        # Track how many were sampled per type
        sampled_syscall_counts = tool_fs.groupby("operation").size()
        
        # Get unique syscall types for y-axis ordering (sorted by FULL total duration)
        syscall_types = list(full_syscall_durations.sort_values(ascending=False).index)
        # Only include types that appear in sampled data
        syscall_types = [s for s in syscall_types if s in tool_fs["operation"].values]
        syscall_to_y = {s: i for i, s in enumerate(syscall_types)}
        tool_fs["y_pos"] = tool_fs["operation"].map(syscall_to_y)
        
        # Add small jitter to y position for overlapping syscalls
        tool_fs["y_pos_jitter"] = tool_fs["y_pos"] + np.random.uniform(-0.35, 0.35, len(tool_fs))
        
        # Plot syscalls using batched line segments (much faster than individual shapes)
        # Each category gets one trace with line segments separated by None
        # 
        # Minimum visual width: Most syscalls are <0.1ms (93%+ typically), which would be
        # invisible on a multi-second plot. Use 0.5% of tool duration as minimum width
        # so every syscall is at least visible as a small mark.
        tool_duration_ms = tc_row["duration_ms"]
        min_visual_width = max(tool_duration_ms * 0.005, 0.1)  # 0.5% of tool duration, min 0.1ms
        
        for is_matched in [True, False]:
            subset = tool_fs[tool_fs["is_matched"] == is_matched]
            if len(subset) == 0:
                continue
            
            line_width = 4 if is_matched else 2
            opacity = 0.9 if is_matched else 0.3
            match_label = "matched" if is_matched else "unmatched"
            
            for category in subset["category"].unique():
                cat_df = subset[subset["category"] == category]
                color = CATEGORY_COLORS.get(category, "#95a5a6")
                
                # Build batched line coordinates with None separators
                x_coords = []
                y_coords = []
                hover_texts = []
                
                for _, entry in cat_df.iterrows():
                    x_start = entry["time_rel_ms"]
                    x_end = x_start + max(entry["duration_ms"], min_visual_width)
                    y_val = entry["y_pos_jitter"]
                    size_hover = _syscall_size_hover_fragment(entry)
                    access_mode_hover = _syscall_access_mode_hover_fragment(entry)
                    
                    # Line segment: start -> end -> None (separator)
                    x_coords.extend([x_start, x_end, None])
                    y_coords.extend([y_val, y_val, None])
                    
                    # Hover text for the midpoint
                    path_str = str(entry["path"])[:50] if entry["path"] else "N/A"
                    hover_texts.extend([
                        f"<b>{entry['operation']}</b><br>"
                        f"PID: {entry['pid']}<br>"
                        f"Duration: {entry['duration_ms']:.3f}ms<br>"
                        f"{access_mode_hover}"
                        f"{size_hover}"
                        f"Start: {x_start:.2f}ms<br>"
                        f"Path: {path_str}",
                        None,  # End point doesn't need hover
                        None,  # Separator
                    ])
                
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(color=color, width=line_width),
                        opacity=opacity,
                        name=f"{category} ({match_label})",
                        hoverinfo='text',
                        hovertext=hover_texts,
                        showlegend=(idx == 0),
                        legendgroup=f"{match_label}_{category}",
                    ),
                    row=row_num, col=col_num
                )
        
        # Update y-axis to show syscall names with counts (sampled/total)
        y_labels = []
        for syscall in syscall_types:
            sampled = sampled_syscall_counts.get(syscall, 0)
            total = full_syscall_counts.get(syscall, 0)
            if sampled < total:
                y_labels.append(f"{syscall} ({sampled}/{total})")
            else:
                y_labels.append(f"{syscall} ({total})")
        
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(syscall_types))),
            ticktext=y_labels,
            tickfont=dict(size=8),
            row=row_num, col=col_num
        )
        
        # Set explicit x-axis range to tool call duration (syscall durations can extend beyond)
        fig.update_xaxes(
            title_text="Time (ms)" if row_num == n_rows else "",
            tickfont=dict(size=8),
            range=[0, tool_duration_ms],
            row=row_num, col=col_num
        )
    
    # Determine main process PID (first/lowest from summary, or most common in data)
    main_pid = None
    pids = data.summary.get("pids", [])
    if pids:
        main_pid = pids[0]
    elif len(fs_df) > 0 and "pid" in fs_df.columns:
        main_pid = fs_df["pid"].mode().iloc[0]
    
    title = "Syscalls Per Tool Call (bar length = duration, thick = matched)"
    if capped:
        title += f" — top {len(tc_df)} of {total_tool_calls} by duration"
    if main_pid is not None:
        title += f" — main PID: {main_pid}"
    
    fig.update_layout(
        title=title,
        height=max(500, 320 * n_rows),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
        ),
        hovermode="closest",
        margin=dict(t=130, b=50, l=40, r=30),
    )
    
    fig.write_html(output_path)


def create_tool_syscall_durations_plotly(
    data: StraceData,
    output_path: Path,
    max_syscalls_per_type: int = 5000,
    max_tool_calls: int = MAX_TOOL_CALL_SUBPLOTS,
) -> None:
    """Create per-tool violin plots of syscall duration distributions."""
    tc_df = data.tool_calls_df
    fs_df = data.fs_entries_df

    if len(tc_df) == 0:
        return

    # Sort once + searchsorted per tool instead of a full-DataFrame mask per tool.
    fs_sorted = fs_df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    _ts_sorted = fs_sorted["timestamp"].to_numpy()

    def _window_slice(tool_start, tool_end) -> pd.DataFrame:
        lo = np.searchsorted(_ts_sorted, np.datetime64(tool_start), side="left")
        hi = np.searchsorted(_ts_sorted, np.datetime64(tool_end), side="right")
        return fs_sorted.iloc[lo:hi]

    total_tool_calls = len(tc_df)
    tc_df = _select_top_tool_calls(tc_df, max_tool_calls)
    capped = total_tool_calls > len(tc_df)

    # Calculate grid dimensions (match tool_syscalls layout)
    n_tools = len(tc_df)
    if n_tools <= 6:
        n_cols = min(3, n_tools)
    elif n_tools <= 20:
        n_cols = 4
    else:
        n_cols = 5
    n_rows = (n_tools + n_cols - 1) // n_cols

    subplot_titles = []
    for _, row in tc_df.iterrows():
        tool_start = row["start_time"]
        tool_end = row["end_time"]

        tool_syscalls = _window_slice(tool_start, tool_end)
        n_syscalls = len(tool_syscalls)

        per_type_counts = tool_syscalls["operation"].value_counts()
        has_per_type_sampling = (per_type_counts > max_syscalls_per_type).any()

        label = _get_tool_label(row, max_len=35)
        if label.startswith("Bash: "):
            label = label[len("Bash: "):]
        duration_ms = row["duration_ms"]

        metadata = f"{duration_ms:.0f}ms wall | {n_syscalls} calls"
        if has_per_type_sampling:
            metadata += " | sampled"
        subplot_titles.append(
            f"{label}<br><span style='font-size:10px;color:#666'>{metadata}</span>"
        )

    v_spacing = min(0.06, 0.8 / max(n_rows - 1, 1))
    h_spacing = min(0.06, 0.8 / max(n_cols - 1, 1))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=v_spacing,
        horizontal_spacing=h_spacing,
    )

    legend_categories_shown: set[str] = set()

    for idx, (_, tc_row) in enumerate(tc_df.iterrows()):
        row_num = idx // n_cols + 1
        col_num = idx % n_cols + 1

        tool_start = tc_row["start_time"]
        tool_end = tc_row["end_time"]

        tool_fs = _window_slice(tool_start, tool_end).copy()
        if len(tool_fs) == 0:
            continue

        tool_fs["duration_ms"] = tool_fs["duration"] * 1000
        full_syscall_counts = tool_fs.groupby("operation").size()
        full_syscall_durations = tool_fs.groupby("operation")["duration_ms"].sum()

        if (full_syscall_counts > max_syscalls_per_type).any():
            tool_fs = _limit_syscalls_per_type(
                tool_fs,
                max_entries_per_type=max_syscalls_per_type,
                group_col="operation",
                sort_col="duration_ms",
            )

        if len(tool_fs) == 0:
            continue

        # Log axes require strictly positive values.
        tool_fs["duration_plot_ms"] = np.maximum(tool_fs["duration_ms"], 1e-4)
        syscall_types = list(full_syscall_durations.sort_values(ascending=False).index)
        syscall_types = [s for s in syscall_types if s in tool_fs["operation"].values]

        for syscall in syscall_types:
            syscall_df = tool_fs[tool_fs["operation"] == syscall]
            if len(syscall_df) == 0:
                continue

            category = classify_syscall(syscall)
            color = CATEGORY_COLORS.get(category, "#95a5a6")
            show_legend = idx == 0 and category not in legend_categories_shown
            if show_legend:
                legend_categories_shown.add(category)

            fig.add_trace(
                go.Violin(
                    x=[syscall] * len(syscall_df),
                    y=syscall_df["duration_plot_ms"],
                    customdata=np.stack(
                        [
                            syscall_df["duration_ms"],
                            syscall_df["pid"].astype(str),
                            syscall_df["path"].fillna("N/A").astype(str),
                            syscall_df.apply(_syscall_size_hover_fragment, axis=1),
                            syscall_df.apply(_syscall_access_mode_hover_fragment, axis=1),
                        ],
                        axis=1,
                    ),
                    name=category,
                    legendgroup=category,
                    showlegend=show_legend,
                    line_color=color,
                    fillcolor=color,
                    opacity=0.65,
                    box_visible=True,
                    meanline_visible=True,
                    points="outliers",
                    pointpos=0,
                    spanmode="hard",
                    hovertemplate=(
                        "<b>" + syscall + "</b><br>"
                        "Duration: %{customdata[0]:.4f}ms<br>"
                        "%{customdata[4]}"
                        "%{customdata[3]}"
                        "PID: %{customdata[1]}<br>"
                        "Path: %{customdata[2]}<extra></extra>"
                    ),
                ),
                row=row_num,
                col=col_num,
            )

        fig.update_xaxes(
            title_text="Syscall type" if row_num == n_rows else "",
            tickangle=45,
            tickfont=dict(size=8),
            type="category",
            categoryorder="array",
            categoryarray=syscall_types,
            row=row_num,
            col=col_num,
        )
        fig.update_yaxes(
            title_text="Duration (ms)" if col_num == 1 else "",
            type="log",
            tickfont=dict(size=8),
            row=row_num,
            col=col_num,
        )

    title = "Syscall Duration Distributions Per Tool Call (violin)"
    if capped:
        title += f" — top {len(tc_df)} of {total_tool_calls} by duration"
    fig.update_layout(
        title=title,
        height=max(500, 360 * n_rows),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
        ),
        hovermode="closest",
        margin=dict(t=90, b=80, l=40, r=30),
        violinmode="group",
    )

    fig.write_html(output_path)


# Phase colors
PHASE_COLORS = {
    "tool_execution": "#3498db",      # Blue
    "model_completion": "#e74c3c",    # Red
}


# =============================================================================
# Visualization: Phase Breakdown
# =============================================================================

def _exclusive_role_wall_tiles(
    role_intervals: dict[str, list[tuple[float, float]]],
    e2e_s: float,
) -> dict[str, float]:
    """Tile wall clock into role-exclusive, Parallel, and Idle slices."""
    points = {0.0, max(0.0, e2e_s)}
    merged_by_role = {
        role: _merge_intervals(intervals)
        for role, intervals in role_intervals.items()
        if intervals
    }
    for intervals in merged_by_role.values():
        for s, e in intervals:
            points.add(max(0.0, min(e2e_s, s)))
            points.add(max(0.0, min(e2e_s, e)))

    out = {role: 0.0 for role in merged_by_role}
    out["Parallel"] = 0.0
    out["Idle"] = 0.0
    ordered = sorted(points)
    for s, e in zip(ordered, ordered[1:]):
        if e <= s:
            continue
        mid = (s + e) / 2.0
        active = [
            role for role, intervals in merged_by_role.items()
            if any(a <= mid < b for a, b in intervals)
        ]
        duration = e - s
        if len(active) == 0:
            out["Idle"] += duration
        elif len(active) == 1:
            out[active[0]] += duration
        else:
            out["Parallel"] += duration
    return {label: value for label, value in out.items() if value > 0 or label == "Idle"}

def _phase_breakdown_stats(trace_dir: Path) -> dict:
    """
    Compute the stats annotated alongside the pie chart:
      - e2e_s:           true end-to-end (all sources unioned)
      - llm_sum_s:       SUM of LLM self-time (per-event interval with direct
                         children subtracted, so a nested LLM under a tool is
                         not also counted toward the enclosing tool).
      - tool_sum_s:      SUM of tool self-time (same construction).  This is
                         what prevents the old bug where Run_analysis (parent)
                         and RunPreprocessing/ScriptExec (children) all got
                         summed, inflating tool_sum by 2-3x.
      - phase_span_s:    llm_sum_s + tool_sum_s.  With self-time, this is the
                         total non-overlapping "primary activity" across all
                         agents — for a fully serial trace it equals e2e.
      - llm_union_s:     time union of LLM intervals (wall-clock — overlaps
                         counted once)
      - tool_union_s:    time union of real-tool intervals
      - subagent_union_s:time union of subagent intervals
      - measured_union_s:union(LLM ∪ tool ∪ subagent)
      - unaccounted_s:   e2e - measured_union_s
      - concurrency_factor: phase_span_s / e2e.  This is a work concurrency
                         factor, not a parallel time ratio: 1.0 means fully
                         serial busy time, >1.0 indicates overlapping work.
      - speedup:         alias of concurrency_factor (Amdahl-style speedup —
                         the number to report: 1.0x serial, ~Nx for N busy
                         workers).
      - wall_both_s:     wall-clock time where LLM AND tool are both active
                         (cross-worker parallelism + nested LLM-in-tool)
      - wall_llm_only_s: llm_union - wall_both
      - wall_tool_only_s:tool_union - wall_both
      - wall_idle_s:     e2e - union(LLM ∪ tool); the four wall_* slices tile
                         e2e exactly
      - n_llm / n_tool:  interval counts
    """
    default_sum_labels = [
        "LLM", "Tool File-IO", "Tool CPU compute", "Tool Wait",
        "Process-mgmt", "Orch File-IO", "Orch CPU compute", "Orch Wait",
        "Unaccounted",
    ]
    default_stats = {
        "e2e_s": 0.0,
        "llm_sum_s": 0.0, "tool_sum_s": 0.0, "phase_span_s": 0.0,
        "llm_union_s": 0.0, "tool_union_s": 0.0,
        "subagent_union_s": 0.0, "measured_union_s": 0.0,
        "unaccounted_s": 0.0, "concurrency_factor": 1.0,
        "speedup": 1.0, "wall_both_s": 0.0, "wall_llm_only_s": 0.0,
        "wall_tool_only_s": 0.0, "wall_idle_s": 0.0,
        "n_llm": 0, "n_tool": 0,
        "sum_labels": default_sum_labels,
        "sum_values": [0.0] * len(default_sum_labels),
        "wall_labels": ["Idle"],
        "wall_values": [0.0],
        "role_intervals": {},
    }
    bundle = _load_agent_timeline_data(trace_dir)
    if bundle is None:
        return default_stats
    e2e = bundle["total_span_s"]
    llm_iv = [(s["start_rel"], s["end_rel"]) for s in bundle["llm_segments"]]
    tool_iv = [(s, e) for s, e, _, _ in bundle["tool_intervals"]]
    sub_iv = [(s, e) for s, e, _, _ in bundle["subagent_intervals"]]
    sum_by_label = {label: 0.0 for label in default_sum_labels}
    role_intervals: dict[str, list[tuple[float, float]]] = {
        "LLM": llm_iv[:],
        "Tool": [],
        "Orchestration": [],
    }

    # Hierarchy-correct LLM/tool sums: use compute_parallelism's self-intervals
    # so nested calls (Run_analysis ⊃ RunPreprocessing ⊃ ScriptExec) are not
    # counted at every level. Falls back to naive sum if the inputs needed by
    # load_events are missing (e.g. legacy traces without pi_events.jsonl).
    try:
        from compute_parallelism import (
            build_children,
            compute_self_intervals,
            load_events,
        )
        events = load_events(trace_dir)
        children = build_children(events)
        self_iv_by_id = compute_self_intervals(events, children)

        llm_sum_ms = 0.0
        tool_sum_ms = 0.0
        t0_ms = bundle["t0"].timestamp() * 1000.0
        tool_summaries = {}
        strace = bundle.get("strace_data")
        if strace is not None:
            tool_summaries = (strace.summary or {}).get("tool_summaries") or {}
            # Some parsed.json files keep tool_summaries at the top level;
            # load_parsed_json stores only summary, so read directly when needed.
            parsed_path = trace_dir / "parsed.json"
            if not tool_summaries and parsed_path.exists():
                try:
                    parsed_payload = json.loads(parsed_path.read_text(encoding="utf-8"))
                    tool_summaries = parsed_payload.get("tool_summaries") or {}
                except (OSError, json.JSONDecodeError):
                    tool_summaries = {}
        for rid, ivs in self_iv_by_id.items():
            ev = events.get(rid)
            if ev is None:
                continue
            self_ms = sum(max(0.0, e - s) for s, e in ivs)
            if ev.kind == "llm":
                llm_sum_ms += self_ms
                sum_by_label["LLM"] += self_ms / 1000.0
            elif ev.kind == "tool":
                tool_sum_ms += self_ms
                rel_ivs = [
                    ((s - t0_ms) / 1000.0, (e - t0_ms) / 1000.0)
                    for s, e in ivs if e > s
                ]
                has_children = bool(children.get(rid))
                is_orch = has_children and not _is_code_exec_tool(ev.name)
                if is_orch:
                    role_intervals["Orchestration"].extend(rel_ivs)
                else:
                    role_intervals["Tool"].extend(rel_ivs)

                summary = tool_summaries.get(rid) or {}
                rb = summary.get("resource_breakdown") or {}
                if not rb:
                    if is_orch:
                        sum_by_label["Orch CPU compute"] += self_ms / 1000.0
                    elif _is_code_exec_tool(ev.name):
                        sum_by_label["Tool CPU compute"] += self_ms / 1000.0
                    else:
                        sum_by_label["Unaccounted"] += self_ms / 1000.0
                    continue
                # Partition this tool's SELF-TIME (bounded, deduped) into
                # File-IO / Wait / Process-mgmt / other / CPU. Raw syscall latency
                # sums can exceed self-time under multithreading (many threads
                # blocked on futex at once); if so, scale the measured buckets down
                # to fit self-time so Wait can't overcount (the old 94% bug). CPU is
                # whatever self-time is left after the measured syscall buckets.
                self_s = self_ms / 1000.0
                file_s = float(rb.get("file_io_ms", 0.0) or 0.0) / 1000.0
                wait_s = float(rb.get("wait_ms", 0.0) or 0.0) / 1000.0
                proc_s = float(rb.get("process_mgmt_ms", 0.0) or 0.0) / 1000.0
                other_s = (
                    float(rb.get("other_syscall_ms", 0.0) or 0.0)
                    + float(rb.get("network_ms", 0.0) or 0.0)
                ) / 1000.0
                busy_s = file_s + wait_s + proc_s + other_s
                if busy_s > self_s and busy_s > 0:
                    f = self_s / busy_s
                    file_s *= f; wait_s *= f; proc_s *= f; other_s *= f
                    cpu_s = 0.0
                else:
                    cpu_s = self_s - busy_s
                if is_orch:
                    sum_by_label["Orch File-IO"] += file_s
                    sum_by_label["Orch Wait"] += wait_s
                    sum_by_label["Process-mgmt"] += proc_s
                    sum_by_label["Orch CPU compute"] += cpu_s
                    sum_by_label["Unaccounted"] += other_s
                else:
                    sum_by_label["Tool File-IO"] += file_s
                    sum_by_label["Tool Wait"] += wait_s
                    sum_by_label["Process-mgmt"] += proc_s
                    sum_by_label["Tool CPU compute"] += cpu_s
                    sum_by_label["Unaccounted"] += other_s
        llm_sum = llm_sum_ms / 1000.0
        tool_sum = tool_sum_ms / 1000.0
    except (FileNotFoundError, ImportError):
        llm_sum = sum(max(0.0, e - s) for s, e in llm_iv)
        tool_sum = sum(max(0.0, e - s) for s, e in tool_iv)
        sum_by_label["LLM"] = llm_sum
        sum_by_label["Unaccounted"] = tool_sum
        role_intervals["Tool"] = tool_iv[:]

    llm_u = _time_union_seconds(llm_iv)
    tool_u = _time_union_seconds(tool_iv)
    sub_u = _time_union_seconds(sub_iv)
    meas_u = _time_union_seconds(llm_iv + tool_iv + sub_iv)
    unaccounted_s = max(0.0, e2e - meas_u)
    sum_by_label["Unaccounted"] += unaccounted_s
    phase_span = sum(sum_by_label.values())

    # Wall-clock decomposition into n+1 tiles: one exclusive slice per role,
    # one aggregate Parallel slice for >=2 active roles/lanes, plus Idle.
    wall_by_label = _exclusive_role_wall_tiles(role_intervals, e2e)
    wall_both = wall_by_label.get("Parallel", 0.0)
    wall_llm_only = wall_by_label.get("LLM", 0.0)
    wall_tool_only = wall_by_label.get("Tool", 0.0)
    wall_idle = wall_by_label.get("Idle", 0.0)

    speedup = (phase_span / e2e) if e2e > 0 else 1.0
    sum_labels = [label for label in default_sum_labels if sum_by_label.get(label, 0.0) > 0 or label == "Unaccounted"]
    wall_labels = [label for label in ["LLM", "Tool", "Orchestration", "Parallel", "Idle"] if label in wall_by_label]
    return {
        "e2e_s": e2e,
        "llm_sum_s": llm_sum,
        "tool_sum_s": tool_sum,
        "phase_span_s": phase_span,
        "llm_union_s": llm_u,
        "tool_union_s": tool_u,
        "subagent_union_s": sub_u,
        "measured_union_s": meas_u,
        "unaccounted_s": unaccounted_s,
        "concurrency_factor": speedup,
        "speedup": speedup,
        "wall_both_s": wall_both,
        "wall_llm_only_s": wall_llm_only,
        "wall_tool_only_s": wall_tool_only,
        "wall_idle_s": wall_idle,
        "n_llm": len(llm_iv),
        "n_tool": len(tool_iv),
        "sum_labels": sum_labels,
        "sum_values": [sum_by_label[label] for label in sum_labels],
        "wall_labels": wall_labels,
        "wall_values": [wall_by_label[label] for label in wall_labels],
        "role_intervals": role_intervals,
    }


def _fmt_stats_line(stats: dict) -> str:
    """One-line monospace footer under the two pies."""
    return (
        f"n_llm={stats['n_llm']}  n_tool={stats['n_tool']}  "
        f"LLM_union={stats['llm_union_s']:.1f}s  "
        f"tool_union={stats['tool_union_s']:.1f}s  "
        f"subagent_union={stats['subagent_union_s']:.1f}s  "
        f"unaccounted={stats['unaccounted_s']:.1f}s"
    )


def _colors_for_labels(labels: list[str]) -> list[str]:
    fallback = "#95A5A6"
    label_colors = {
        "LLM": RESOURCE_COLORS["LLM"],
        "Tool": "#3498db",
        "Orchestration": RESOURCE_COLORS["Orchestration"],
        "Parallel": RESOURCE_COLORS["Parallel"],
        "Idle": RESOURCE_COLORS["Idle"],
    }
    return [RESOURCE_COLORS.get(label, label_colors.get(label, fallback)) for label in labels]


def create_phase_breakdown_plotly(trace_dir: Path, output_path: Path) -> None:
    """
    Time accounting — two donuts + speedup.

    Left  (sum view):  phase span = LLM self-time sum + tool self-time sum.
                       Answers "what KIND of work fills the busy time".
    Right (wall view): e2e tiled into role-exclusive slices / parallel / idle.
                       Answers "how much of the wall clock is parallel".
    Below: speedup = phase span / e2e (1.0x serial, ~Nx for N busy workers).
    """
    stats = _phase_breakdown_stats(trace_dir)
    if stats["e2e_s"] <= 0:
        print(f"  phase_breakdown: no data found in {trace_dir}", file=sys.stderr)
        return

    fig = go.Figure()
    sum_labels = stats["sum_labels"]
    sum_values = stats["sum_values"]
    wall_labels = stats["wall_labels"]
    wall_values = stats["wall_values"]

    # Pie 1 — sum view. textposition='inside' + textinfo='percent' keeps labels
    # ON the wedges (auto-hidden when a slice is too small), so tiny slices no
    # longer collide as external labels. Full label/value is in the hover.
    fig.add_trace(
        go.Pie(
            labels=sum_labels,
            values=sum_values,
            marker_colors=_colors_for_labels(sum_labels),
            hole=0.45,
            textinfo='percent',
            textposition='inside',
            insidetextorientation='horizontal',
            hovertemplate="<b>%{label}</b><br>%{value:.2f}s (%{percent})<extra></extra>",
            domain=dict(x=[0.02, 0.46], y=[0.16, 0.92]),
            sort=False,
            legendgroup="sum",
        )
    )
    # Pie 2 — wall view.  Idle is always present (0% slice still shows in
    # legend + hover), per design.
    fig.add_trace(
        go.Pie(
            labels=wall_labels,
            values=wall_values,
            marker_colors=_colors_for_labels(wall_labels),
            hole=0.45,
            textinfo='percent',
            textposition='inside',
            insidetextorientation='horizontal',
            hovertemplate="<b>%{label}</b><br>%{value:.2f}s (%{percent})<extra></extra>",
            domain=dict(x=[0.54, 0.98], y=[0.16, 0.92]),
            sort=False,
            legendgroup="wall",
        )
    )

    # Donut centers: the two headline numbers.
    fig.add_annotation(
        xref="paper", yref="paper", x=0.24, y=0.54,
        showarrow=False, align="center",
        font=dict(size=16, color="#2c3e50"),
        text=f"<b>{stats['phase_span_s']:.1f}s</b><br>phase span",
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.76, y=0.54,
        showarrow=False, align="center",
        font=dict(size=16, color="#2c3e50"),
        text=f"<b>{stats['e2e_s']:.1f}s</b><br>e2e (wall)",
    )

    # Captions above each donut.
    fig.add_annotation(
        xref="paper", yref="paper", x=0.24, y=0.99,
        showarrow=False, font=dict(size=13, color="#34495e"),
        text="<b>Sum view</b> — total busy time by kind",
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.76, y=0.99,
        showarrow=False, font=dict(size=13, color="#34495e"),
        text="<b>Wall view</b> — wall clock tiled by activity",
    )

    # Headline metric: speedup, centered under both pies.
    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=0.075,
        showarrow=False, align="center",
        font=dict(size=18, color="#2c3e50"),
        text=(f"<b>speedup = phase span / e2e = "
              f"{stats['speedup']:.2f}×</b>  "
              f"<span style='font-size:12px;color:#7f8c8d'>"
              f"(1.0× = fully serial; ≈N× = N workers continuously busy)"
              f"</span>"),
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=0.0,
        showarrow=False, align="center",
        font=dict(family="monospace", size=11, color="#566573"),
        text=_fmt_stats_line(stats),
    )

    fig.update_layout(
        title=(
            f"<b>Time accounting</b> — speedup {stats['speedup']:.2f}× "
            f"(phase span {stats['phase_span_s']:.1f}s / e2e {stats['e2e_s']:.1f}s)"
        ),
        height=620,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=70, b=40),
        paper_bgcolor="white",
    )

    fig.write_html(output_path)


def create_phase_breakdown_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """
    Time accounting — two donuts + speedup (PNG twin of the plotly version).
    """
    stats = _phase_breakdown_stats(trace_dir)
    if stats["e2e_s"] <= 0:
        print(f"  phase_breakdown: no data found in {trace_dir}", file=sys.stderr)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    # Pie 1 — sum view.  Labels go into a legend (with seconds + %) instead of
    # inline pie labels, which otherwise collide badly when several slices are
    # tiny (0.x%). autopct only annotates slices large enough to read.
    sum_labels = stats["sum_labels"]
    sum_sizes = stats["sum_values"]
    sum_colors = _colors_for_labels(sum_labels)
    phase_span = stats["phase_span_s"]
    sum_legend = [
        f"{name} — {v:.1f}s ({(v / phase_span * 100.0) if phase_span > 0 else 0.0:.1f}%)"
        for name, v in zip(sum_labels, sum_sizes)
    ]
    ax1.pie(
        sum_sizes, colors=sum_colors, startangle=90,
        autopct=lambda p: f"{p:.0f}%" if p >= 4.0 else "",
        wedgeprops=dict(width=0.6),
    )
    ax1.legend(
        sum_legend, loc="center right", bbox_to_anchor=(-0.05, 0.5),
        fontsize=8, frameon=False,
    )
    ax1.set_title("Sum view — total busy time by kind", fontsize=11)
    ax1.text(
        0, 0, f"{stats['phase_span_s']:.1f}s\nphase span",
        ha='center', va='center', fontsize=12, fontweight='bold',
    )

    # Pie 2 — wall view.  All four slices always present; labels go into a
    # legend (with absolute seconds) so a 0% Idle slice stays visible.
    wall_labels = stats["wall_labels"]
    wall_vals = stats["wall_values"]
    wall_colors = _colors_for_labels(wall_labels)
    e2e = stats["e2e_s"]
    legend_labels = [
        f"{name} — {v:.1f}s ({(v / e2e * 100.0) if e2e > 0 else 0.0:.1f}%)"
        for name, v in zip(wall_labels, wall_vals)
    ]
    ax2.pie(
        wall_vals, colors=wall_colors, startangle=90,
        autopct=lambda p: f"{p:.1f}%" if p >= 1.0 else "",
        wedgeprops=dict(width=0.6),
    )
    ax2.legend(
        legend_labels, loc="center left", bbox_to_anchor=(0.92, 0.5),
        fontsize=9, frameon=False,
    )
    ax2.set_title("Wall view — wall clock tiled by activity", fontsize=11)
    ax2.text(
        0, 0, f"{stats['e2e_s']:.1f}s\ne2e (wall)",
        ha='center', va='center', fontsize=12, fontweight='bold',
    )

    # Headline metric + footer.
    fig.text(
        0.5, 0.115,
        f"speedup = phase span / e2e = {stats['speedup']:.2f}×",
        ha='center', fontsize=15, fontweight='bold', color="#2c3e50",
    )
    fig.text(
        0.5, 0.075,
        "(1.0× = fully serial; ≈N× = N workers continuously busy)",
        ha='center', fontsize=9, color="#7f8c8d",
    )
    fig.text(
        0.5, 0.03, _fmt_stats_line(stats),
        ha='center', family="monospace", fontsize=8.5, color="#566573",
    )

    plt.suptitle(
        f"Time accounting — speedup {stats['speedup']:.2f}× "
        f"(phase span {stats['phase_span_s']:.1f}s / e2e {stats['e2e_s']:.1f}s)",
        fontsize=13,
    )
    plt.tight_layout(rect=[0.14, 0.15, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Visualization: Agent Concurrency (n lanes, n = number of agents)
# =============================================================================

def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of intervals as a sorted list of disjoint (start, end) pairs."""
    merged: list[tuple[float, float]] = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


# Horizontal resolution (in "pixels") the dense-timeline rasterizer quantizes
# to. The agent-concurrency chart is ~1400px wide; anything finer than one pixel
# is not visible. Using 2000 keeps quantization strictly sub-pixel at any trace
# length, so the rendered image is identical while the bar count is bounded.
_RASTER_PX = 2000


def _rasterize_intervals(
    intervals: list[tuple[float, float]],
    span_s: float,
    px: int = _RASTER_PX,
) -> list[tuple[float, float]]:
    """Quantize intervals to a sub-pixel time grid, then merge adjacent cells.

    A busy trace produces hundreds of thousands of one-syscall intervals whose
    sub-pixel gaps (futex between two reads, etc.) block plain union-merge, so
    the bar count stays huge and the chart is both unrenderable and a solid
    smear. Snapping each interval to a grid of ``px`` buckets and merging
    consecutive occupied buckets bounds the output to O(px) bars per resource
    while staying visually identical: every collapsed feature is narrower than
    one on-screen pixel. Falls back to an exact union when the span is unknown.
    """
    if span_s <= 0 or not intervals:
        return _merge_intervals(intervals)
    bucket = span_s / px
    occupied: set[int] = set()
    for s, e in intervals:
        if e <= s:
            continue
        i0 = int(s // bucket)
        i1 = int(e // bucket)
        for i in range(i0, i1 + 1):
            occupied.add(i)
    runs: list[list[int]] = []
    for i in sorted(occupied):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(r[0] * bucket, (r[1] + 1) * bucket) for r in runs]


def _subtract_many(
    base: list[tuple[float, float]],
    blockers: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Subtract blocker intervals from a list of base intervals."""
    remaining = base[:]
    for bs, be in _merge_intervals(blockers):
        next_remaining: list[tuple[float, float]] = []
        for s, e in remaining:
            if be <= s or bs >= e:
                next_remaining.append((s, e))
                continue
            if s < bs:
                next_remaining.append((s, max(s, bs)))
            if be < e:
                next_remaining.append((min(e, be), e))
        remaining = next_remaining
    return [(s, e) for s, e in remaining if e > s]


def _agent_concurrency_data(trace_dir: Path) -> dict | None:
    """
    Per-agent lane data for the agent concurrency chart.

    Lane = one agent (genomas_role; traces without roles collapse to a single
    "agent" lane).  Bars are resource segments: LLM windows, syscall-latency
    resources, and residual CPU/orchestration self-time.  Blank space is
    explicit lane idle.

    Returns dict with:
      lanes: [{role, label, segments, busy, busy_s, pct}]  (sorted by first start)
      wall_s, t0_s, max_concurrent, parallel_s, parallel_pct
    """
    from compute_parallelism import build_children, compute_self_intervals, load_events

    try:
        events = load_events(trace_dir)
    except FileNotFoundError as exc:
        print(f"  agent_concurrency: {exc}", file=sys.stderr)
        return None
    if not events:
        return None

    t0 = min(ev.start_ms for ev in events.values())
    wall_ms = max(ev.end_ms for ev in events.values()) - t0

    children = build_children(events)
    self_iv_by_id = compute_self_intervals(events, children)
    event_role = {rid: (ev.role or "agent") for rid, ev in events.items()}
    event_name = {rid: ev.name for rid, ev in events.items()}
    segments_by_role: dict[str, list[dict]] = {}
    syscall_blockers_by_tool: dict[str, list[tuple[float, float]]] = {}

    def add_segment(role: str, resource: str, s: float, e: float, label: str = "") -> None:
        if e <= s:
            return
        segments_by_role.setdefault(role, []).append({
            "resource": resource,
            "start": max(0.0, s),
            "end": max(0.0, e),
            "label": label,
        })

    for rid, ev in events.items():
        if ev.kind == "llm":
            add_segment(
                event_role[rid],
                "LLM",
                (ev.start_ms - t0) / 1000.0,
                (ev.end_ms - t0) / 1000.0,
                ev.name,
            )

    parsed_json = trace_dir / "parsed.json"
    if parsed_json.exists():
        # Reuse the cached, vectorized-parse DataFrame instead of re-reading the
        # (potentially multi-million-row) parsed.json and calling pd.to_datetime
        # per entry — both were major costs on big traces.
        fs_df = load_parsed_json(parsed_json).fs_entries_df
        t0_dt = datetime.fromtimestamp(t0 / 1000.0)
        if len(fs_df) and "timestamp" in fs_df.columns:
            df = fs_df[["matched_tool_call", "syscall", "timestamp", "duration"]].copy()
            df = df[df["matched_tool_call"].isin(event_role.keys())]
            df["resource"] = df["syscall"].astype(str).map(resource_for_syscall)
            df = df[df["resource"].notna()]
        else:
            df = None

        if df is not None and len(df):
            # Align every timestamp's DATE onto t0_dt's date (keep HMS) —
            # vectorized equivalent of the old per-row replace(year/month/day).
            t0_ts = pd.Timestamp(t0_dt)
            t0_midnight = pd.Timestamp(t0_dt.date())
            ts = df["timestamp"]
            aligned = t0_midnight + (ts - ts.dt.normalize())

            # TZ-mismatch shift detection (same threshold / 15-min quantum).
            shift_s = 0
            gap_s = (aligned.min() - t0_ts).total_seconds()
            if abs(gap_s) >= 1800:
                shift_s = round(gap_s / 900) * 900

            end_rel = (aligned - t0_ts).dt.total_seconds() - shift_s
            duration_s = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
            start_rel = end_rel - duration_s
            keep = (end_rel >= -1.0) & (start_rel <= (wall_ms / 1000.0) + 1.0)

            for tid, resource, syscall, s, e in zip(
                df["matched_tool_call"][keep],
                df["resource"][keep],
                df["syscall"][keep].astype(str),
                start_rel[keep],
                end_rel[keep],
            ):
                label = "Process-mgmt" if resource == "Process-mgmt" else resource
                add_segment(event_role[tid], label, float(s), float(e), syscall)
                syscall_blockers_by_tool.setdefault(tid, []).append((float(s), float(e)))

    for rid, ev in events.items():
        if ev.kind != "tool":
            continue
        rel_self = [
            ((s - t0) / 1000.0, (e - t0) / 1000.0)
            for s, e in self_iv_by_id.get(rid, [])
            if e > s
        ]
        if not rel_self:
            continue
        gaps = _subtract_many(rel_self, syscall_blockers_by_tool.get(rid, []))
        if _is_code_exec_tool(event_name.get(rid, "")):
            resource = "CPU compute"
        elif children.get(rid):
            resource = "Orchestration"
        else:
            continue
        for s, e in gaps:
            add_segment(event_role[rid], resource, s, e, event_name.get(rid, "tool"))

    # Collapse the one-bar-per-syscall segments into a bounded number of bars per
    # (role, resource) via sub-pixel rasterization. A busy trace has hundreds of
    # thousands of syscalls; one plotly bar each is unrenderable (minutes to
    # build, multi-hundred-MB HTML) and a solid smear on screen. Plain union
    # leaves the count huge because sub-pixel gaps (futex between reads) block
    # merging. Rasterizing to a sub-pixel grid bounds the bar count while keeping
    # the rendered image identical; only per-syscall hover detail is lost, which
    # is meaningless at this density. Idle gaps wider than a pixel are preserved.
    # busy / concurrency are computed from the EXACT (un-rasterized) segments so
    # the reported numbers are unaffected; rasterization only collapses the
    # DISPLAY bars.
    display_segments_by_role: dict[str, list[dict]] = {}
    for role in list(segments_by_role.keys()):
        by_resource: dict[str, list[tuple[float, float]]] = {}
        for seg in segments_by_role[role]:
            by_resource.setdefault(seg["resource"], []).append(
                (seg["start"], seg["end"])
            )
        merged_segments: list[dict] = []
        for resource, intervals in by_resource.items():
            for s, e in _rasterize_intervals(intervals, wall_ms / 1000.0):
                merged_segments.append({
                    "resource": resource,
                    "start": s,
                    "end": e,
                    "label": resource,
                })
        display_segments_by_role[role] = merged_segments

    lanes = []
    for role, segments in segments_by_role.items():
        busy = _merge_intervals([(seg["start"], seg["end"]) for seg in segments])
        busy_s = sum(e - s for s, e in busy)
        display_segments = display_segments_by_role.get(role, [])
        lanes.append({
            "role": role,
            "segments": sorted(display_segments, key=lambda seg: (seg["start"], seg["end"])),
            "busy": busy,
            "busy_s": busy_s,
        })
    lanes.sort(key=lambda lane: lane["busy"][0][0] if lane["busy"] else 0.0)

    wall_s = wall_ms / 1000.0
    for lane in lanes:
        lane["label"] = lane["role"]

    # Sweep over per-agent unions: max simultaneous agents + time at >=2.
    points: list[tuple[float, int]] = []
    for lane in lanes:
        for s, e in lane["busy"]:
            points.append((s, 1))
            points.append((e, -1))
    points.sort()
    active = 0
    last: float | None = None
    parallel_s = 0.0
    max_concurrent = 0
    for t, delta in points:
        if last is not None and t > last and active >= 2:
            parallel_s += t - last
        active += delta
        max_concurrent = max(max_concurrent, active)
        last = t

    return {
        "lanes": lanes,
        "wall_s": wall_s,
        "max_concurrent": max_concurrent,
        "parallel_s": parallel_s,
        "parallel_pct": (parallel_s / wall_s * 100.0) if wall_s > 0 else 0.0,
    }


# Deterministic lane colors (tab10-like, reused in order of first activity).
AGENT_LANE_COLORS = [
    "#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def create_agent_concurrency_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """Agent activity timeline — one lane per agent, colored by resource."""
    data = _agent_concurrency_data(trace_dir)
    if data is None or not data["lanes"]:
        print(f"  agent_concurrency: no data found in {trace_dir}", file=sys.stderr)
        return

    lanes = data["lanes"]
    n = len(lanes)
    fig, ax = plt.subplots(figsize=(14, max(2.6, 0.85 * n + 1.6)))

    for i, lane in enumerate(lanes):
        y = n - 1 - i  # first-active agent on top
        for seg in lane["segments"]:
            color = RESOURCE_COLORS.get(seg["resource"], "#95A5A6")
            ax.broken_barh(
                [(seg["start"], seg["end"] - seg["start"])],
                (y - 0.36, 0.72),
                facecolors=color,
                edgecolors="white",
                linewidth=0.3,
            )

    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([lane["label"] for lane in lanes], fontsize=10)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, data["wall_s"] * 1.005)
    ax.set_xlabel("time (s)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(
        f"max {data['max_concurrent']} agents concurrent · "
        f"agents in parallel (≥2 active): {data['parallel_s']:.0f}s "
        f"({data['parallel_pct']:.0f}% of wall)",
        fontsize=10, color="#555555",
    )

    legend_handles = [
        mpatches.Patch(facecolor=RESOURCE_COLORS.get(label, "#95A5A6"), label=label)
        for label in ["LLM", "File-IO", "CPU compute", "Wait", "Process-mgmt", "Orchestration"]
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, frameon=False, fontsize=9)

    fig.suptitle("Agent activity timeline", fontsize=13)
    plt.tight_layout(rect=[0, 0.10, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_agent_concurrency_plotly(trace_dir: Path, output_path: Path) -> None:
    """Interactive twin of create_agent_concurrency_matplotlib."""
    data = _agent_concurrency_data(trace_dir)
    if data is None or not data["lanes"]:
        print(f"  agent_concurrency: no data found in {trace_dir}", file=sys.stderr)
        return

    lanes = data["lanes"]
    n = len(lanes)
    fig = go.Figure()
    shown_resources: set[str] = set()
    for _i, lane in enumerate(lanes):
        y = lane["label"]
        for seg in lane["segments"]:
            resource = seg["resource"]
            showlegend = resource not in shown_resources
            shown_resources.add(resource)
            fig.add_trace(go.Bar(
                y=[y],
                x=[seg["end"] - seg["start"]],
                base=[seg["start"]],
                orientation="h",
                marker=dict(
                    color=RESOURCE_COLORS.get(resource, "#95A5A6"),
                    line=dict(width=0.3, color="white"),
                ),
                name=resource,
                legendgroup=resource,
                showlegend=showlegend,
                customdata=[[seg["end"], seg.get("label", "")]],
                hovertemplate=(
                    f"<b>{lane['role']}</b><br>"
                    f"{resource}<br>"
                    "start: %{base:.3f}s<br>"
                    "end: %{customdata[0]:.3f}s<br>"
                    "detail: %{customdata[1]}<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=(
            f"<b>Agent activity timeline</b> — "
            f"max {data['max_concurrent']} agents concurrent · "
            f"in parallel (≥2 active): {data['parallel_s']:.0f}s "
            f"({data['parallel_pct']:.0f}% of wall)"
        ),
        barmode="overlay",
        height=max(300, 90 * n + 160),
        width=1400,
        xaxis=dict(title="time (s)", range=[0, data["wall_s"] * 1.005]),
        yaxis=dict(categoryorder="array",
                   categoryarray=[lane["label"] for lane in reversed(lanes)]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=70, b=50),
    )
    fig.write_html(output_path)


# =============================================================================
# Visualization: Agent Timeline (three-lane Gantt)
# =============================================================================
#
# Three vertically stacked lanes, shared x-axis:
#   1. Semantic lane — LLM segments (top row) + each subagent's full lifespan
#      below it (indented per parent_subagent_id so nested subagents render
#      under their parent).
#   2. Tool lane    — one row per unique tool name (real tools only; subagents
#      have been moved to subagent_calls.log by langchain_tool_logger.py).
#   3. System lane  — FS syscalls split into the 8 syscall categories
#      (metadata / data / control / modify / process / blocking / network /
#      other). Bars (not dots), width = duration. The tracer captures common
#      blocking/wait syscalls (futex / poll / epoll_wait / nanosleep / wait4)
#      so long model waits and subprocess waits do not render as blank gaps.
#
# Health check: after computing all bar intervals, the time-union of
# LLM ∪ tool ∪ subagent intervals should approximately cover the trace's
# overall execution span. Any "residue" is agent dispatch / framework
# overhead OR a logger-coverage gap. If residue exceeds REASONABLE_RESIDUE_PCT
# of total span we emit a warning to stderr (does not block rendering).


REASONABLE_RESIDUE_PCT = 5.0   # warn if > 5% of total span is unattributed


def _pair_llm_segments(events_path: Path) -> list[dict]:
    """
    Walk pi_events.jsonl and pair message_start with message_end into LLM
    segments. Returns list of {start_ms, end_ms, run_id, parent_subagent_run_id}.

    Pairing strategy:
      - If both start and end events carry `run_id`, pair by run_id (correct
        even when concurrent LLM calls in different subagents overlap).
      - Otherwise fall back to LIFO stack order (one start, one end, single
        thread). This matches the existing pi_events behavior where
        message_start/end events from the pi extension don't have run_id.
    """
    if not events_path.exists():
        return []

    segments: list[dict] = []
    open_by_id: dict[str, dict] = {}
    open_stack: list[dict] = []

    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            et = ev.get("type")
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
            ts = msg.get("timestamp")
            if not isinstance(ts, (int, float)):
                continue
            role = msg.get("role")
            if role != "assistant":
                continue
            run_id = ev.get("run_id")
            parent = ev.get("parent_run_id") or ev.get("parent_subagent_run_id")
            if not isinstance(parent, str):
                parent = None

            if et == "message_start":
                entry = {"start_ms": ts, "run_id": run_id, "parent_run_id": parent}
                if isinstance(run_id, str):
                    open_by_id[run_id] = entry
                else:
                    open_stack.append(entry)
            elif et == "message_end":
                entry = None
                if isinstance(run_id, str) and run_id in open_by_id:
                    entry = open_by_id.pop(run_id)
                elif open_stack:
                    entry = open_stack.pop()
                if entry is None:
                    # Stray end with no matching start — skip but log.
                    continue
                entry["end_ms"] = ts
                # Prefer end's parent_run_id (more accurately attributed).
                if isinstance(parent, str):
                    entry["parent_run_id"] = parent
                segments.append(entry)

    # Unmatched starts (LLM in progress when trace ended): close them with the
    # latest known timestamp so they still render as bars.
    if open_by_id or open_stack:
        last_ts = max(
            (s.get("start_ms", 0) for s in list(open_by_id.values()) + open_stack),
            default=0,
        )
        for entry in list(open_by_id.values()) + open_stack:
            entry["end_ms"] = max(entry["start_ms"], last_ts)
            entry["unmatched"] = True
            segments.append(entry)

    return segments


def _load_parent_run_ids(events_path: Path) -> dict[str, str | None]:
    """Map tool_call_id -> parent_run_id from pi_events.jsonl."""
    parent_by_id: dict[str, str | None] = {}
    if not events_path.exists():
        return parent_by_id
    with events_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            parent = ev.get("parent_run_id") or ev.get("parent_subagent_run_id")
            if not isinstance(parent, str):
                parent = None
            et = ev.get("type")
            if et == "message_update":
                assistant_event = ev.get("assistantMessageEvent")
                if not isinstance(assistant_event, dict):
                    continue
                tool_call = assistant_event.get("toolCall")
                if not isinstance(tool_call, dict):
                    continue
                tool_id = tool_call.get("id")
                if isinstance(tool_id, str):
                    parent_by_id[tool_id] = parent
            elif et == "tool_execution_end":
                tool_id = ev.get("toolCallId")
                if isinstance(tool_id, str):
                    parent_by_id.setdefault(tool_id, parent)
    return parent_by_id


def _load_tool_results(events_path: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    if not events_path.exists():
        return results
    with events_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "tool_execution_end":
                continue
            tool_id = ev.get("toolCallId")
            if not isinstance(tool_id, str):
                continue
            text = ""
            result = ev.get("result")
            if isinstance(result, dict):
                content = result.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            text += item["text"]
            try:
                parsed = json.loads(text) if text else {}
            except json.JSONDecodeError:
                parsed = {"raw": text}
            if isinstance(parsed, dict):
                results[tool_id] = parsed
    return results


def _time_union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Total covered seconds after merging overlapping intervals."""
    if not intervals:
        return 0.0
    sorted_iv = sorted(intervals)
    total = 0.0
    cur_s, cur_e = sorted_iv[0]
    for s, e in sorted_iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return max(0.0, total)


def _datetime_from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0)


# Bundle cache: phase_breakdown, agent_timeline and agent_concurrency all call
# _load_agent_timeline_data(trace_dir) with the same argument and consume the
# result read-only. Without caching, each recomputes the full alignment AND
# reloads/​copies the millions of fs_entries. Cache the bundle per trace_dir so
# the work (including the big DataFrame date-shift copy) happens once per run.
_AGENT_TIMELINE_BUNDLE_CACHE: dict[str, dict | None] = {}


def _load_agent_timeline_data(trace_dir: Path) -> dict | None:
    cache_key = str(Path(trace_dir).resolve())
    if cache_key in _AGENT_TIMELINE_BUNDLE_CACHE:
        return _AGENT_TIMELINE_BUNDLE_CACHE[cache_key]
    bundle = _load_agent_timeline_data_uncached(trace_dir)
    _AGENT_TIMELINE_BUNDLE_CACHE[cache_key] = bundle
    return bundle


def _load_agent_timeline_data_uncached(trace_dir: Path) -> dict | None:
    """
    Load and align LLM segments, tool calls, subagent calls, and FS entries
    onto a single t=0 origin. Returns None if nothing is plottable.
    """
    from parse_strace import parse_tool_calls_log as parse_log

    events_path = trace_dir / "pi_events.jsonl"
    tool_log = trace_dir / "tool_calls.log"
    sub_log = trace_dir / "subagent_calls.log"
    parsed_json = trace_dir / "parsed.json"

    llm_raw = _pair_llm_segments(events_path)
    tool_parent_by_id = _load_parent_run_ids(events_path)
    tool_results_by_id = _load_tool_results(events_path)
    tool_calls = parse_log(tool_log) if tool_log.exists() else []
    subagent_calls = parse_log(sub_log) if sub_log.exists() else []
    strace_data = load_parsed_json(parsed_json) if parsed_json.exists() else None

    # Date alignment.  parse_tool_calls_log uses datetime.now() to assign a
    # date to HMS-only tool/subagent times.  If the viz is run on a different
    # day than the trace was made (e.g., today is May 13 but the trace is
    # from May 12), tool dates land on TODAY while LLM unix ms parses to the
    # real trace date, and the chart's t-axis stretches by N×24h.
    # Fix: rewrite tool/subagent dates to match the LLM events' true date.
    if llm_raw and (tool_calls or subagent_calls):
        import dataclasses as _dc
        true_date = _datetime_from_ms(min(s["start_ms"] for s in llm_raw)).date()

        def _set_date(dt: datetime, d) -> datetime:
            return dt.replace(year=d.year, month=d.month, day=d.day)

        tool_calls = [
            _dc.replace(
                tc,
                start_time=_set_date(tc.start_time, true_date),
                end_time=_set_date(tc.end_time, true_date),
            )
            for tc in tool_calls
        ]
        subagent_calls = [
            _dc.replace(
                sc,
                start_time=_set_date(sc.start_time, true_date),
                end_time=_set_date(sc.end_time, true_date),
            )
            for sc in subagent_calls
        ]
        # strace_data carries its own start_time + ISO datetime columns
        # (parsed from parsed.json), which were set when parse_ebpf.py ran.
        # If that was a different day than `true_date`, shift the dates
        # uniformly by an integer-day delta.
        if strace_data is not None and strace_data.start_time.date() != true_date:
            day_delta = timedelta(days=(true_date - strace_data.start_time.date()).days)
            new_tc_df = strace_data.tool_calls_df.copy()
            new_tc_df["start_time"] = new_tc_df["start_time"] + day_delta
            new_tc_df["end_time"] = new_tc_df["end_time"] + day_delta
            new_fs_df = strace_data.fs_entries_df.copy()
            new_fs_df["timestamp"] = new_fs_df["timestamp"] + day_delta
            strace_data = StraceData(
                tool_calls_df=new_tc_df,
                fs_entries_df=new_fs_df,
                summary=strace_data.summary,
                start_time=strace_data.start_time + day_delta,
                end_time=strace_data.end_time + day_delta,
                duration_seconds=strace_data.duration_seconds,
            )

    # TZ mismatch correction.  When the trace was made on a machine whose
    # local TZ differs from this machine's, the LLM unix-ms timestamps (which
    # get converted to LOCAL naive datetimes here via fromtimestamp) end up
    # off by a constant N-hour offset from the tool HMS times (which got
    # parsed onto this machine's today).  Detect by comparing the first
    # event of each kind: in a healthy trace they're within seconds; in a
    # TZ-skewed trace they differ by ≥1 hour.  Shift LLM event ms values to
    # bring them into the tool naive-time frame.
    if llm_raw:
        first_llm_dt = _datetime_from_ms(min(s["start_ms"] for s in llm_raw))
        anchor_candidates = []
        if tool_calls:
            anchor_candidates.append(min(tc.start_time for tc in tool_calls))
        if subagent_calls:
            anchor_candidates.append(min(sc.start_time for sc in subagent_calls))
        # Also use fs_entries (strace_data) as an anchor.  Without this,
        # a trace whose tool_calls.log is empty (e.g. GenoMAS with
        # logger Step A applied) loses TZ correction and the LLM lane
        # ends up 1 hour ahead of the syscall lane.
        if strace_data is not None:
            anchor_candidates.append(strace_data.start_time)
        if anchor_candidates:
            first_other_dt = min(anchor_candidates)
            gap_s = (first_other_dt - first_llm_dt).total_seconds()
            # Threshold 30 min — see reclassify_subagents.detect_tz_offset for
            # rationale (1-hr TZ offsets register as ~3596–3604s gaps).
            if abs(gap_s) >= 1800:
                quanta = round(gap_s / 900) * 900    # nearest 15 min
                print(
                    f"  agent_timeline: detected {gap_s:+.0f}s gap between "
                    f"first LLM event and first tool — likely TZ mismatch "
                    f"between tracing machine and this machine. Shifting "
                    f"LLM timestamps by {quanta:+d}s ({quanta // 3600:+d}h).",
                    file=sys.stderr,
                )
                for seg in llm_raw:
                    seg["start_ms"] += quanta * 1000
                    if "end_ms" in seg:
                        seg["end_ms"] += quanta * 1000

    # Anchor t=0 to earliest of: first LLM start, first tool start, first
    # subagent start, first ebpf event. This way the chart starts at the
    # genuine beginning of activity.
    candidates: list[datetime] = []
    if llm_raw:
        candidates.append(_datetime_from_ms(min(s["start_ms"] for s in llm_raw)))
    if tool_calls:
        candidates.append(min(tc.start_time for tc in tool_calls))
    if subagent_calls:
        candidates.append(min(sc.start_time for sc in subagent_calls))
    if strace_data is not None:
        candidates.append(strace_data.start_time)
    if not candidates:
        return None
    t0 = min(candidates)

    def _to_rel(dt: datetime) -> float:
        return (dt - t0).total_seconds()

    llm_segments = []
    for seg in llm_raw:
        s_rel = _to_rel(_datetime_from_ms(seg["start_ms"]))
        e_rel = _to_rel(_datetime_from_ms(seg["end_ms"]))
        llm_segments.append({
            "start_rel": s_rel,
            "end_rel": e_rel,
            "duration_s": max(0.0, e_rel - s_rel),
            "run_id": seg.get("run_id"),
            "parent_run_id": seg.get("parent_run_id"),
            "parent_subagent_run_id": seg.get("parent_run_id"),
            "unmatched": seg.get("unmatched", False),
        })

    tool_intervals = [(_to_rel(tc.start_time), _to_rel(tc.end_time), tc.tool_name, tc.tool_id)
                      for tc in tool_calls]
    tool_args_by_id = {tc.tool_id: tc.input_params for tc in tool_calls}

    agent_name_by_id: dict[object, str] = {}
    for result in tool_results_by_id.values():
        if "agent_id" in result and isinstance(result.get("agent_name"), str):
            agent_name_by_id[result["agent_id"]] = result["agent_name"]

    tool_display_by_id: dict[str, str] = {}
    for _s, _e, name, tid in tool_intervals:
        display = name
        args = tool_args_by_id.get(tid) or {}
        if name == "Run_analysis":
            agent_name = agent_name_by_id.get(args.get("agent_id"))
            if agent_name:
                display = f"{name} -> {agent_name}"
        tool_display_by_id[tid] = display
    subagent_intervals = [(_to_rel(sc.start_time), _to_rel(sc.end_time), sc.tool_name, sc.tool_id)
                          for sc in subagent_calls]

    # Backward-compatible fallback for old traces: when no explicit parent
    # metadata exists, attribute LLMs fully contained by a tool to the nearest
    # enclosing tool so Run_analysis internals become visible.
    if llm_segments and tool_intervals and not any(
        isinstance(seg.get("parent_run_id"), str) for seg in llm_segments
    ):
        for seg in llm_segments:
            best_parent: tuple[float, str] | None = None
            for s_rel, e_rel, _name, tid in tool_intervals:
                if s_rel <= seg["start_rel"] and e_rel >= seg["end_rel"]:
                    span = e_rel - s_rel
                    if best_parent is None or span < best_parent[0]:
                        best_parent = (span, tid)
            if best_parent is not None:
                seg["parent_run_id"] = best_parent[1]
                seg["parent_subagent_run_id"] = best_parent[1]

    # Determine total span (use whichever source extends furthest).
    spans = []
    if llm_segments:
        spans.append(max(s["end_rel"] for s in llm_segments))
    if tool_intervals:
        spans.append(max(e for _, e, _, _ in tool_intervals))
    if subagent_intervals:
        spans.append(max(e for _, e, _, _ in subagent_intervals))
    if strace_data is not None:
        spans.append((strace_data.start_time + timedelta(seconds=strace_data.duration_seconds)
                      - t0).total_seconds())
    total_span = max(spans) if spans else 0.0

    # Subagent parent-child structure: compute from TIME CONTAINMENT between
    # subagent intervals, NOT from pi_events.jsonl's parent_subagent_run_id
    # field. The pi_events field is set at trace time by the run_id tree walk,
    # which breaks for SRAgent (it invokes sub-agents without forwarding
    # RunnableConfig).  After reclassify_subagents.py has moved entries into
    # subagent_calls.log, the run_id-tree-based parent info is incomplete or
    # absent. Time containment is robust against this: if A's interval contains
    # B's interval, A is B's parent (subject to picking the *nearest* enclosing
    # subagent, not just any ancestor).
    parent_of_subagent: dict[str, str | None] = {}
    for s_a, e_a, _name_a, id_a in subagent_intervals:
        best_parent: tuple[float, str] | None = None  # (parent_span, parent_id)
        for s_b, e_b, _name_b, id_b in subagent_intervals:
            if id_b == id_a:
                continue
            # b strictly contains a?
            if s_b <= s_a and e_b >= e_a and (e_b - s_b) > (e_a - s_a):
                span = e_b - s_b
                # We want the smallest enclosing subagent (nearest ancestor).
                if best_parent is None or span < best_parent[0]:
                    best_parent = (span, id_b)
        parent_of_subagent[id_a] = best_parent[1] if best_parent else None

    return {
        "t0": t0,
        "total_span_s": total_span,
        "llm_segments": llm_segments,
        "tool_intervals": tool_intervals,
        "tool_parent_by_id": tool_parent_by_id,
        "tool_display_by_id": tool_display_by_id,
        "subagent_intervals": subagent_intervals,
        "parent_of_subagent": parent_of_subagent,
        "strace_data": strace_data,
    }


def _compute_residue_warning(bundle: dict) -> tuple[float, float]:
    """
    Returns (residue_seconds, residue_pct).  Residue = total_span - union of
    LLM ∪ tool ∪ subagent intervals.  Health-check helper for the timeline.
    """
    intervals: list[tuple[float, float]] = []
    intervals += [(s["start_rel"], s["end_rel"]) for s in bundle["llm_segments"]]
    intervals += [(s, e) for s, e, _, _ in bundle["tool_intervals"]]
    intervals += [(s, e) for s, e, _, _ in bundle["subagent_intervals"]]
    covered = _time_union_seconds(intervals)
    total = bundle["total_span_s"]
    residue = max(0.0, total - covered)
    pct = (residue / total * 100.0) if total > 0 else 0.0
    return residue, pct


def _aggregate_subagent_rows(
    subagent_intervals: list[tuple[float, float, str, str]],
    parent_of_subagent: dict[str, str | None],   # kept for signature compat, unused
) -> list[dict]:
    """
    Aggregate subagent invocations by NAME — one row per unique name, with
    the label decorated `name (Nx, avg T)` and all invocations drawn as
    multiple bars on the same row.

    Row order: by first-occurrence start time.

    Note: parent-child indent was REMOVED in favor of a flat layout because
    the run_id parent tree is unreliable (SRAgent's broken-config callback
    chain) and the time-containment fallback over-attributes concurrent
    siblings as nested children.  Showing call hierarchy correctly requires
    the actual code; the chart's purpose is timing/parallelism, not topology.

    Returns list of dicts with keys: name, intervals=[(s,e,id), ...],
                                     count, avg_s, label, depth=0.
    """
    by_name: dict[str, dict] = {}
    for s, e, name, sid in subagent_intervals:
        entry = by_name.setdefault(name, {
            "name": name, "intervals": [], "first_s": s,
        })
        entry["intervals"].append((s, e, sid))
        entry["first_s"] = min(entry["first_s"], s)

    rows = sorted(by_name.values(), key=lambda r: r["first_s"])
    for r in rows:
        durs = [e - s for s, e, _ in r["intervals"]]
        r["count"] = len(r["intervals"])
        r["avg_s"] = sum(durs) / len(durs) if durs else 0.0
        r["label"] = (
            f"{r['name']} ({r['count']}x, avg {_fmt_duration(r['avg_s'])})"
        )
        r["depth"] = 0
    return rows


def _aggregate_named_rows(
    intervals: list[tuple[float, float, str, str]],
) -> list[dict]:
    """
    Same as _aggregate_subagent_rows but for tools (no parent-depth). Returns
    list of dicts with: name, intervals=[(s,e,id)...], count, avg_s, label.
    """
    by_name: dict[str, dict] = {}
    for s, e, name, ident in intervals:
        entry = by_name.setdefault(name, {
            "name": name, "intervals": [], "first_s": s,
        })
        entry["intervals"].append((s, e, ident))
        entry["first_s"] = min(entry["first_s"], s)
    rows = sorted(by_name.values(), key=lambda r: r["first_s"])
    for r in rows:
        durs = [e - s for s, e, _ in r["intervals"]]
        r["count"] = len(r["intervals"])
        r["avg_s"] = sum(durs) / len(durs) if durs else 0.0
        r["label"] = f"{r['name']} ({r['count']}x, avg {_fmt_duration(r['avg_s'])})"
    return rows


def _aggregate_tool_tree_rows(
    tool_intervals: list[tuple[float, float, str, str]],
    llm_segments: list[dict],
    tool_parent_by_id: dict[str, str | None],
    tool_display_by_id: dict[str, str] | None = None,
) -> list[dict]:
    """Rows for Phase 3: top-level tool followed by indented child-type rows."""
    tool_display_by_id = tool_display_by_id or {}
    llm_by_parent: dict[str, list[dict]] = {}
    for seg in llm_segments:
        parent = seg.get("parent_run_id")
        if isinstance(parent, str):
            llm_by_parent.setdefault(parent, []).append(seg)

    tool_by_parent: dict[str, list[tuple[float, float, str, str]]] = {}
    top_tools: list[tuple[float, float, str, str]] = []
    for item in tool_intervals:
        _s, _e, _name, tid = item
        parent = tool_parent_by_id.get(tid)
        if parent:
            tool_by_parent.setdefault(parent, []).append(item)
        else:
            top_tools.append(item)

    rows: list[dict] = []
    for s_rel, e_rel, name, tid in sorted(top_tools, key=lambda item: item[0]):
        duration_s = max(0.0, e_rel - s_rel)
        display_name = tool_display_by_id.get(tid, name)
        rows.append({
            "label": f"{display_name} ({_fmt_duration(duration_s)})",
            "name": name,
            "kind": "tool",
            "color": color_for_tool(name),
            "intervals": [(s_rel, e_rel, tid, name)],
        })

        child_llms = sorted(llm_by_parent.get(tid, []), key=lambda seg: seg["start_rel"])
        if child_llms:
            total_s = sum(seg["duration_s"] for seg in child_llms)
            rows.append({
                "label": f"  -> {display_name} LLM ({len(child_llms)}x, total {_fmt_duration(total_s)})",
                "name": f"{display_name} LLM",
                "kind": "llm_child",
                "color": LLM_SUBAGENT_COLOR,
                "intervals": [
                    (seg["start_rel"], seg["end_rel"], seg.get("run_id"), "LLM")
                    for seg in child_llms
                ],
            })

        by_child_name: dict[str, list[tuple[float, float, str, str]]] = {}
        for child in sorted(tool_by_parent.get(tid, []), key=lambda item: item[0]):
            by_child_name.setdefault(child[2], []).append(child)
        for child_name, child_items in by_child_name.items():
            total_s = sum(max(0.0, e - s) for s, e, _name, _cid in child_items)
            rows.append({
                "label": f"  -> {child_name} ({len(child_items)}x, total {_fmt_duration(total_s)})",
                "name": child_name,
                "kind": "tool_child",
                "color": color_for_tool(child_name),
                "intervals": [
                    (s, e, cid, child_name) for s, e, _name, cid in child_items
                ],
            })

    # Backward-compatible fallback for traces without parent metadata.
    if not rows:
        for r in _aggregate_named_rows(tool_intervals):
            rows.append({
                "label": r["label"],
                "name": r["name"],
                "kind": "tool",
                "color": color_for_tool(r["name"]),
                "intervals": [
                    (s, e, tid, r["name"]) for s, e, tid in r["intervals"]
                ],
            })
    return rows


def _aggregate_llm_rows(
    llm_segments: list[dict],
    tool_display_by_id: dict[str, str],
) -> list[dict]:
    by_label: dict[str, dict] = {}
    for seg in llm_segments:
        parent = seg.get("parent_run_id")
        if isinstance(parent, str):
            parent_label = tool_display_by_id.get(parent, parent[:8])
            label_base = f"LLM in {parent_label}"
            color = LLM_SUBAGENT_COLOR
        else:
            label_base = "LLM top-level"
            color = LLM_COLOR
        entry = by_label.setdefault(label_base, {
            "name": label_base,
            "segments": [],
            "first_s": seg["start_rel"],
            "color": color,
        })
        entry["segments"].append(seg)
        entry["first_s"] = min(entry["first_s"], seg["start_rel"])

    rows = sorted(by_label.values(), key=lambda r: r["first_s"])
    for row in rows:
        total_s = sum(seg["duration_s"] for seg in row["segments"])
        count = len(row["segments"])
        avg_s = total_s / count if count else 0.0
        row["label"] = f"{row['name']} ({count}x, avg {_fmt_duration(avg_s)})"
    return rows


def _fmt_duration(s: float) -> str:
    """Format a duration in seconds compactly: us / ms / s as appropriate."""
    if s < 1e-3:
        return f"{s*1e6:.0f}µs"   # µs
    if s < 1.0:
        return f"{s*1000:.0f}ms"
    return f"{s:.2f}s"


# System lane:  syscalls with duration < SYS_BAR_MIN_S are too narrow to be
# visible as bars on a multi-second timeline → render those as scatter dots
# (faithful position, no misleading width).  Syscalls >= the threshold get
# real width-encoded bars.
SYS_BAR_MIN_S = 0.1   # 100 ms — user-tunable threshold


def create_agent_timeline_plotly(trace_dir: Path, output_path: Path) -> None:
    """Three-lane Gantt: semantic (LLM + subagents) / tool / system (FS)."""
    bundle = _load_agent_timeline_data(trace_dir)
    if bundle is None:
        print(f"  agent_timeline: no data found in {trace_dir}", file=sys.stderr)
        return

    residue_s, residue_pct = _compute_residue_warning(bundle)
    if residue_pct > REASONABLE_RESIDUE_PCT:
        print(
            f"  agent_timeline: WARNING — {residue_s:.2f}s ({residue_pct:.1f}%) "
            "of trace span is not covered by any LLM / tool / subagent bar. "
            "Either logger missed events or this is agent dispatch overhead.",
            file=sys.stderr,
        )

    # --- Build row labels for each lane ------------------------------------
    # Semantic lane: one LLM row per parent context, not one anonymous bucket.
    sub_rows = _aggregate_subagent_rows(
        bundle["subagent_intervals"], bundle["parent_of_subagent"])

    llm_rows = _aggregate_llm_rows(
        bundle["llm_segments"], bundle["tool_display_by_id"])

    semantic_labels: list[str] = [r["label"] for r in llm_rows]
    sub_name_to_label: dict[str, str] = {}
    for r in sub_rows:
        semantic_labels.append(r["label"])
        sub_name_to_label[r["name"]] = r["label"]

    # Tool lane: each top-level tool, followed by indented child rows.
    tool_rows = _aggregate_tool_tree_rows(
        bundle["tool_intervals"],
        bundle["llm_segments"],
        bundle["tool_parent_by_id"],
        bundle["tool_display_by_id"],
    )
    tool_labels = [r["label"] for r in tool_rows]

    # System lane: fixed 8-row order so output is stable across traces.
    syscall_rows = ["metadata", "data", "control", "modify", "process",
                    "network", "blocking", "other"]

    # --- Make subplot scaffolding -----------------------------------------
    sem_h = max(1, len(semantic_labels))
    tool_h = max(1, len(tool_labels))
    sys_h = len(syscall_rows)
    total_h = sem_h + tool_h + sys_h
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[sem_h / total_h, tool_h / total_h, sys_h / total_h],
        vertical_spacing=0.04,
        subplot_titles=(
            "Semantic — LLM + subagents",
            "Tool — real tools (from tool_calls.log)",
            f"System — FS syscalls by category (bars = duration ≥{int(SYS_BAR_MIN_S*1000)}ms, dots = shorter)",
        ),
    )

    # --- Lane 1: LLM rows grouped by parent tool / selected agent ----------
    for r in llm_rows:
        for seg in r["segments"]:
            fig.add_trace(go.Bar(
                x=[seg["duration_s"]],
                y=[r["label"]],
                base=[seg["start_rel"]],
                orientation="h",
                marker_color=r["color"],
                opacity=0.5 if seg.get("unmatched") else 0.85,
                name=r["name"],
                legendgroup=f"llm:{r['name']}",
                showlegend=False,
                customdata=[[seg["start_rel"], seg["duration_s"], str(seg.get("run_id", ""))[:8]]],
                hovertemplate=(
                    f"<b>{r['name']}</b><br>"
                    "start: %{customdata[0]:.3f}s<br>"
                    "duration: %{customdata[1]:.4f}s<br>"
                    "id: %{customdata[2]}...<br>"
                    "<extra></extra>"
                ),
            ), row=1, col=1)

    for r in sub_rows:
        for s_rel, e_rel, sid in r["intervals"]:
            fig.add_trace(go.Bar(
                x=[e_rel - s_rel],
                y=[r["label"]],
                base=[s_rel],
                orientation="h",
                marker_color=color_for_subagent(r["name"]),
                opacity=0.85,
                name=r["name"],
                legendgroup=f"sub:{r['name']}",
                showlegend=False,
                customdata=[[s_rel, e_rel - s_rel, sid[:8]]],
                hovertemplate=(
                    f"<b>{r['name']}</b> (subagent)<br>"
                    "start: %{customdata[0]:.3f}s<br>"
                    "duration: %{customdata[1]:.4f}s<br>"
                    "id: %{customdata[2]}…<br>"
                    "<extra></extra>"
                ),
            ), row=1, col=1)

    # --- Lane 2: top-level tool bars plus indented child rows --------------
    for r in tool_rows:
        for s_rel, e_rel, tid, item_name in r["intervals"]:
            fig.add_trace(go.Bar(
                x=[e_rel - s_rel],
                y=[r["label"]],
                base=[s_rel],
                orientation="h",
                marker_color=r["color"],
                opacity=0.85,
                name=item_name,
                legendgroup=f"tool:{item_name}",
                showlegend=False,
                customdata=[[s_rel, e_rel - s_rel, str(tid or "")[:8]]],
                hovertemplate=(
                    f"<b>{item_name}</b> ({r['kind']})<br>"
                    "start: %{customdata[0]:.3f}s<br>"
                    "duration: %{customdata[1]:.4f}s<br>"
                    "id: %{customdata[2]}…<br>"
                    "<extra></extra>"
                ),
            ), row=2, col=1)

    # --- Lane 3: FS syscall coverage per category (sub-pixel rasterized) ---
    # Each category's syscall intervals [time_rel, time_rel+duration] are
    # rasterized to a bounded set of coverage bars (full data, visually faithful)
    # rather than randomly subsampling to a few thousand individual marks. This
    # keeps the lane fast and honest even at millions of syscalls; sub-pixel
    # detail (incl. the old <0.1s dots) is collapsed because it cannot be drawn.
    strace = bundle["strace_data"]
    if strace is not None and len(strace.fs_entries_df) > 0:
        fs_df = strace.fs_entries_df
        span_s = bundle["total_span_s"]
        # Vectorized classification over the whole column (no per-row .apply).
        cat_series = fs_df["operation"].map(_SYSCALL_TO_CATEGORY).fillna("other")
        dur = pd.to_numeric(fs_df.get("duration", 0.0), errors="coerce").fillna(0.0)
        start = fs_df["time_rel"].astype(float)
        end = start + dur
        for cat in syscall_rows:
            mask = (cat_series == cat).to_numpy()
            if not mask.any():
                continue
            color = SYSCALL_CATEGORY_COLORS.get(cat, SYSCALL_CATEGORY_COLORS["other"])
            intervals = list(zip(start[mask].tolist(), end[mask].tolist()))
            runs = _rasterize_intervals(intervals, span_s)
            if not runs:
                continue
            fig.add_trace(go.Bar(
                x=[e - s for s, e in runs],
                y=[cat] * len(runs),
                base=[s for s, e in runs],
                orientation="h",
                marker_color=color,
                marker_line_width=0,
                opacity=0.7,
                name=f"sys: {cat}",
                legendgroup=f"sys:{cat}",
                showlegend=True,
                hovertemplate=(
                    "<b>" + cat + "</b><br>"
                    "start: %{base:.3f}s<br>"
                    "<extra></extra>"
                ),
            ), row=3, col=1)

    # --- Layout polish ----------------------------------------------------
    title_extra = ""
    if residue_pct > REASONABLE_RESIDUE_PCT:
        title_extra = (f" — ⚠ {residue_s:.1f}s ({residue_pct:.1f}%) unattributed")
    fig.update_layout(
        title=f"Agent Timeline (total {bundle['total_span_s']:.1f}s){title_extra}",
        barmode="overlay",
        height=200 + 40 * total_h,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=1.0),
        margin=dict(l=140, r=200, t=80, b=40),
    )
    fig.update_xaxes(title_text="time (s)", row=3, col=1, range=[0, bundle["total_span_s"]])
    fig.update_yaxes(categoryorder="array", categoryarray=semantic_labels[::-1], row=1, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=tool_labels[::-1], row=2, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=syscall_rows[::-1], row=3, col=1)

    fig.write_html(output_path)


def create_agent_timeline_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """Static three-lane Gantt PNG version."""
    bundle = _load_agent_timeline_data(trace_dir)
    if bundle is None:
        print(f"  agent_timeline: no data found in {trace_dir}", file=sys.stderr)
        return

    residue_s, residue_pct = _compute_residue_warning(bundle)
    if residue_pct > REASONABLE_RESIDUE_PCT:
        print(
            f"  agent_timeline: WARNING — {residue_s:.2f}s ({residue_pct:.1f}%) "
            "of trace span is not covered by any LLM / tool / subagent bar.",
            file=sys.stderr,
        )

    sub_rows = _aggregate_subagent_rows(
        bundle["subagent_intervals"], bundle["parent_of_subagent"])
    tool_rows_agg = _aggregate_tool_tree_rows(
        bundle["tool_intervals"],
        bundle["llm_segments"],
        bundle["tool_parent_by_id"],
        bundle["tool_display_by_id"],
    )

    llm_rows = _aggregate_llm_rows(
        bundle["llm_segments"], bundle["tool_display_by_id"])

    semantic_labels = [r["label"] for r in llm_rows]
    sub_name_to_row: dict[str, int] = {}
    for r in sub_rows:
        semantic_labels.append(r["label"])
        sub_name_to_row[r["name"]] = len(semantic_labels) - 1

    tool_labels = [r["label"] for r in tool_rows_agg]
    tool_row_by_label: dict[str, int] = {
        r["label"]: i for i, r in enumerate(tool_rows_agg)
    }

    syscall_rows = ["metadata", "data", "control", "modify", "process",
                    "network", "blocking", "other"]

    sem_h = max(1, len(semantic_labels))
    tool_h = max(1, len(tool_labels))
    sys_h = len(syscall_rows)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(16, max(8, 0.45 * (sem_h + tool_h + sys_h) + 3)),
        sharex=True,
        gridspec_kw={"height_ratios": [sem_h, tool_h, sys_h]},
    )
    ax_sem, ax_tool, ax_sys = axes

    # Lane 1: LLM rows grouped by parent tool / selected agent.
    for row_idx, r in enumerate(llm_rows):
        for seg in r["segments"]:
            ax_sem.barh(
                y=row_idx,
                width=seg["end_rel"] - seg["start_rel"],
                left=seg["start_rel"],
                color=r["color"],
                alpha=0.5 if seg.get("unmatched") else 0.85,
                height=0.7,
            )
    # Lane 1: one row per unique subagent name; bars for each invocation.
    for r in sub_rows:
        row_idx = sub_name_to_row[r["name"]]
        for s_rel, e_rel, _sid in r["intervals"]:
            ax_sem.barh(y=row_idx, width=e_rel - s_rel, left=s_rel,
                        color=color_for_subagent(r["name"]), alpha=0.85, height=0.7)
    ax_sem.set_yticks(range(len(semantic_labels)))
    ax_sem.set_yticklabels(semantic_labels, fontsize=8)
    ax_sem.set_ylim(-0.6, len(semantic_labels) - 0.4)
    ax_sem.invert_yaxis()
    ax_sem.set_title("Semantic — LLM + subagents", fontsize=10, loc="left")
    ax_sem.grid(axis="x", alpha=0.2)

    # Lane 2: one row per unique tool name; bars for each invocation.
    for r in tool_rows_agg:
        row_idx = tool_row_by_label[r["label"]]
        for s_rel, e_rel, _tid, _item_name in r["intervals"]:
            ax_tool.barh(y=row_idx, width=e_rel - s_rel, left=s_rel,
                         color=r["color"], alpha=0.85, height=0.7)
    ax_tool.set_yticks(range(len(tool_labels)))
    ax_tool.set_yticklabels(tool_labels, fontsize=8)
    ax_tool.set_ylim(-0.6, max(0, len(tool_labels) - 1) + 0.6)
    ax_tool.invert_yaxis()
    ax_tool.set_title("Tool — real tools", fontsize=10, loc="left")
    ax_tool.grid(axis="x", alpha=0.2)

    # Lane 3: FS syscall coverage per category (sub-pixel rasterized) — full
    # data, visually faithful, bounded bar count (matches the plotly twin).
    strace = bundle["strace_data"]
    if strace is not None and len(strace.fs_entries_df) > 0:
        fs_df = strace.fs_entries_df
        span_s = bundle["total_span_s"]
        cat_series = fs_df["operation"].map(_SYSCALL_TO_CATEGORY).fillna("other")
        dur = pd.to_numeric(fs_df.get("duration", 0.0), errors="coerce").fillna(0.0)
        start = fs_df["time_rel"].astype(float)
        end = start + dur
        for cat_idx, cat in enumerate(syscall_rows):
            mask = (cat_series == cat).to_numpy()
            if not mask.any():
                continue
            color = SYSCALL_CATEGORY_COLORS.get(cat, SYSCALL_CATEGORY_COLORS["other"])
            runs = _rasterize_intervals(
                list(zip(start[mask].tolist(), end[mask].tolist())), span_s)
            if not runs:
                continue
            ax_sys.barh(
                y=[cat_idx] * len(runs),
                width=[e - s for s, e in runs],
                left=[s for s, e in runs],
                color=color, alpha=0.7, height=0.7, edgecolor="none",
            )
    ax_sys.set_yticks(range(len(syscall_rows)))
    ax_sys.set_yticklabels(syscall_rows, fontsize=8)
    ax_sys.set_ylim(-0.6, len(syscall_rows) - 0.4)
    ax_sys.invert_yaxis()
    ax_sys.set_title(
        "System — FS syscalls (per-category coverage)",
        fontsize=10, loc="left",
    )
    ax_sys.set_xlabel("time (s)")
    ax_sys.grid(axis="x", alpha=0.2)
    ax_sys.set_xlim(0, bundle["total_span_s"] if bundle["total_span_s"] > 0 else 1.0)

    # Two-block legend on the figure (right margin): tool palette + subagent
    # palette + syscall palette. Labels are bare names (the row tick labels
    # already include the (Nx, avg T) decoration).
    seen_tool_legend: set[str] = set()
    tool_patches = []
    for r in tool_rows_agg:
        if r["name"] in seen_tool_legend:
            continue
        seen_tool_legend.add(r["name"])
        tool_patches.append(
            mpatches.Patch(color=r["color"], label=f"tool: {r['name']}")
        )
    sub_patches = [mpatches.Patch(color=color_for_subagent(r["name"]),
                                  label=f"sub: {r['name']}")
                   for r in sub_rows]
    cat_patches = [mpatches.Patch(color=SYSCALL_CATEGORY_COLORS[c], label=f"sys: {c}")
                   for c in syscall_rows]
    llm_patch = [
        mpatches.Patch(color=LLM_COLOR, label="LLM top-level"),
        mpatches.Patch(color=LLM_SUBAGENT_COLOR, label="LLM subagent"),
    ]
    fig.legend(handles=llm_patch + sub_patches + tool_patches + cat_patches,
               loc="center right", fontsize=7, frameon=False,
               bbox_to_anchor=(1.0, 0.5))

    title = f"Agent Timeline (total {bundle['total_span_s']:.1f}s)"
    if residue_pct > REASONABLE_RESIDUE_PCT:
        title += f"  —  ⚠ {residue_s:.1f}s ({residue_pct:.1f}%) unattributed"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.85, 0.97])
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Index HTML Dashboard
# =============================================================================

def create_index_html(output_dir: Path, visualizations: list[str]) -> None:
    """Create index.html dashboard linking all visualizations."""
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strace Visualization Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #eee;
        }
        .header {
            background: rgba(0,0,0,0.3);
            padding: 2rem;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .header h1 {
            font-size: 2rem;
            font-weight: 300;
            letter-spacing: 2px;
            color: #fff;
        }
        .header p {
            margin-top: 0.5rem;
            color: #aaa;
            font-size: 0.9rem;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        }
        .card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .card-body {
            padding: 1.25rem;
        }
        .card-body h3 {
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 0.75rem;
            color: #fff;
        }
        .card-body .links {
            display: flex;
            gap: 0.75rem;
        }
        .card-body a {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: rgba(52, 152, 219, 0.2);
            color: #3498db;
            text-decoration: none;
            border-radius: 6px;
            font-size: 0.85rem;
            transition: background 0.2s;
        }
        .card-body a:hover {
            background: rgba(52, 152, 219, 0.4);
        }
        .card-body a.png {
            background: rgba(46, 204, 113, 0.2);
            color: #2ecc71;
        }
        .card-body a.png:hover {
            background: rgba(46, 204, 113, 0.4);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Strace Visualization Dashboard</h1>
        <p>Interactive analysis of filesystem operations from strace output</p>
    </div>
    <div class="container">
        <div class="grid">
"""
    
    viz_titles = {
        "timeline": "Timeline View",
        "process_timeline": "Per-Process Timeline",
        "io_rate": "I/O Rate Over Time",
        "tool_syscalls": "Syscalls Per Tool Call",
        "tool_syscall_durations": "Syscall Duration Distributions",
        "phase_breakdown": "Time Accounting",
        "agent_timeline": "Agent Timeline (3-lane Gantt)",
        "agent_concurrency": "Agent Activity Timeline",
    }

    viz_descriptions = {
        "timeline": "Tool calls and filesystem operations over time",
        "process_timeline": "Process lifespans with parent-child links and tool call windows",
        "io_rate": "Syscalls per 100ms with tool call overlays",
        "tool_syscalls": "Per-tool breakdown showing syscall durations as horizontal bars",
        "tool_syscall_durations": "Per-tool violin plots showing duration distribution of each syscall type",
        "phase_breakdown": (
            "Two donuts over the same trace. Sum view (left): total busy time "
            "split into LLM, tool resources, process management, orchestration, "
            "and unaccounted residual. Wall view (right): e2e wall clock tiled "
            "into role-exclusive slices, aggregate parallel time, and idle. "
            "Headline: speedup = phase span / e2e."
        ),
        "agent_timeline": "Semantic / tool / system three-lane Gantt: LLM + subagents on top, real tools in the middle, FS syscalls (by category) on the bottom",
        "agent_concurrency": (
            "One lane per agent with resource-colored segments: LLM, File-IO, "
            "CPU compute, Wait, Process-mgmt, and Orchestration. Blank lane space "
            "is idle; vertical overlap between lanes = agents in parallel."
        ),
    }
    
    # Union: viz names generated THIS run + any viz files already on disk from
    # previous runs.  This way `--only X` doesn't wipe other viz from the index.
    on_disk = set()
    for known in VISUALIZATIONS:
        if (output_dir / f"{known}.html").exists() or (output_dir / f"{known}.png").exists():
            on_disk.add(known)
    all_viz = list(dict.fromkeys(list(visualizations) + sorted(on_disk)))

    for viz_name in all_viz:
        title = viz_titles.get(viz_name, viz_name.replace("_", " ").title())
        desc = viz_descriptions.get(viz_name, "")

        html_file = f"{viz_name}.html"
        png_file = f"{viz_name}.png"

        has_html = (output_dir / html_file).exists()
        has_png = (output_dir / png_file).exists()
        
        html_content += f"""
            <div class="card">
                {"<img src='" + png_file + "' alt='" + title + "'>" if has_png else ""}
                <div class="card-body">
                    <h3>{title}</h3>
                    <p style="color: #888; font-size: 0.85rem; margin-bottom: 0.75rem;">{desc}</p>
                    <div class="links">
                        {"<a href='" + html_file + "'>Interactive HTML</a>" if has_html else ""}
                        {"<a href='" + png_file + "' class='png'>Static PNG</a>" if has_png else ""}
                    </div>
                </div>
            </div>
"""

    trace_dir = output_dir.parent
    dag_html = trace_dir / "call_dag.html"
    dag_dot = trace_dir / "call_dag.dot"
    parallelism_json = trace_dir / "parallelism_summary.json"
    if dag_html.exists() or dag_dot.exists() or parallelism_json.exists():
        html_content += """
            <div class="card">
                <div class="card-body">
                    <h3>Call DAG + Parallelism</h3>
                    <p style="color: #888; font-size: 0.85rem; margin-bottom: 0.75rem;">Parent-child execution graph and numeric parallelism metrics</p>
                    <div class="links">
"""
        if dag_html.exists():
            html_content += "                        <a href='../call_dag.html'>Call DAG HTML</a>\n"
        if dag_dot.exists():
            html_content += "                        <a href='../call_dag.dot'>Graphviz DOT</a>\n"
        if parallelism_json.exists():
            html_content += "                        <a href='../parallelism_summary.json'>Metrics JSON</a>\n"
        html_content += """
                    </div>
                </div>
            </div>
"""
    
    # Storage / lineage figures (produced by lineage_analyzer.py into
    # ../lineage/). PNG-only; show a preview card + link when present.
    lineage_dir = trace_dir / "lineage"
    lineage_figs = [
        ("fig1_size_distribution.png", "File Size Distribution",
         "Per-syscall I/O request size and per-file artifact size by category."),
        ("fig2_reader_fanout.png", "Reader Fan-out",
         "Distinct CodeExec calls that read each artifact (caching candidates)."),
        ("fig3_staleness_cdf.png", "Write→First-Read Staleness",
         "Gap from first write to first read, per artifact."),
        ("fig4_lifecycle.png", "Artifact Lifecycle",
         "Per-artifact reclaimable window over time."),
        ("fig5_artifact_lifecycle.png", "Top Artifact Lifecycles",
         "Write/read events for the most-touched artifacts."),
    ]
    for fname, title, desc in lineage_figs:
        if not (lineage_dir / fname).exists():
            continue
        rel = f"../lineage/{fname}"
        html_content += f"""
            <div class="card">
                <img src='{rel}' alt='{title}'>
                <div class="card-body">
                    <h3>{title}</h3>
                    <p style="color: #888; font-size: 0.85rem; margin-bottom: 0.75rem;">{desc}</p>
                    <div class="links">
                        <a href='{rel}' class='png'>Static PNG</a>
                    </div>
                </div>
            </div>
"""

    html_content += """
        </div>
    </div>
</body>
</html>
"""

    (output_dir / "index.html").write_text(html_content)


# =============================================================================
# Main Visualization Runner
# =============================================================================

# Visualizations that use StraceData (from parsed.json)
STRACE_VISUALIZATIONS = {
    "timeline": (create_timeline_plotly, create_timeline_matplotlib),
    # process_timeline and io_rate intentionally disabled (not wanted). The
    # create_* functions are kept in case they're re-enabled later.
    "tool_syscalls": (create_tool_syscalls_plotly, None),
    "tool_syscall_durations": (create_tool_syscall_durations_plotly, None),
}

# Visualizations that use PhaseAnalysis (from tool_calls.log) — empty now,
# kept for backwards compat and as an extension point.  Time Accounting
# (phase_breakdown) was moved to AGENT_VISUALIZATIONS so it can also load
# LLM + subagent data.
PHASE_VISUALIZATIONS: dict = {}

# Visualizations that operate directly on the trace directory (load whatever
# combination of pi_events.jsonl, tool_calls.log, subagent_calls.log, parsed.json
# they need on their own). These take (trace_dir, output_path) signatures.
AGENT_VISUALIZATIONS = {
    "agent_timeline": (create_agent_timeline_plotly, create_agent_timeline_matplotlib),
    "phase_breakdown": (create_phase_breakdown_plotly, create_phase_breakdown_matplotlib),
    "agent_concurrency": (create_agent_concurrency_plotly, create_agent_concurrency_matplotlib),
}

# Combined for CLI help and validation
VISUALIZATIONS = {**STRACE_VISUALIZATIONS, **PHASE_VISUALIZATIONS, **AGENT_VISUALIZATIONS}


def generate_visualizations(
    trace_dir: Path,
    only: list[str] | None = None,
    html_only: bool = False,
    png_only: bool = False,
    max_syscalls_per_type: int = 5000,
    max_tool_calls: int = MAX_TOOL_CALL_SUBPLOTS,
) -> None:
    """Generate all or selected visualizations."""
    
    # Create output directory
    output_dir = trace_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    generated = []
    
    # Determine which visualizations to generate
    viz_names = only if only else list(VISUALIZATIONS.keys())
    
    # Separate strace, phase, and agent visualizations
    strace_viz_to_gen = [v for v in viz_names if v in STRACE_VISUALIZATIONS]
    phase_viz_to_gen = [v for v in viz_names if v in PHASE_VISUALIZATIONS]
    agent_viz_to_gen = [v for v in viz_names if v in AGENT_VISUALIZATIONS]

    # Check for unknown visualizations
    unknown = [v for v in viz_names if v not in VISUALIZATIONS]
    for v in unknown:
        print(f"Warning: Unknown visualization '{v}', skipping", file=sys.stderr)
    
    # Load strace data if needed
    strace_data = None
    if strace_viz_to_gen:
        parsed_json = trace_dir / "parsed.json"
        if parsed_json.exists():
            print(f"Loading strace data from {parsed_json}...", file=sys.stderr)
            strace_data = load_parsed_json(parsed_json)
            print(f"  Loaded {len(strace_data.tool_calls_df)} tool calls, "
                  f"{len(strace_data.fs_entries_df)} fs entries", file=sys.stderr)
            print(f"  Duration: {strace_data.duration_seconds:.2f} seconds", file=sys.stderr)
        else:
            print(f"Warning: {parsed_json} not found, skipping strace visualizations", 
                  file=sys.stderr)
            strace_viz_to_gen = []
    
    # Load phase data if needed
    phase_data = None
    if phase_viz_to_gen:
        tool_log = trace_dir / "tool_calls.log"
        if tool_log.exists():
            print(f"Loading phase data from {tool_log}...", file=sys.stderr)
            phase_data = load_phases(trace_dir)
            print(f"  Found {len(phase_data.phases)} phases, {len(phase_data.batches)} batches", 
                  file=sys.stderr)
            print(f"  Tool execution: {phase_data.tool_execution_pct:.1f}%, "
                  f"Model completion: {phase_data.model_completion_pct:.1f}%", file=sys.stderr)
        else:
            print(f"Warning: {tool_log} not found, skipping phase visualizations", 
                  file=sys.stderr)
            phase_viz_to_gen = []
    
    # Generate strace visualizations
    for viz_name in strace_viz_to_gen:
        plotly_fn, matplotlib_fn = STRACE_VISUALIZATIONS[viz_name]
        
        if not png_only and plotly_fn:
            html_path = output_dir / f"{viz_name}.html"
            print(f"Generating {html_path.name}...", file=sys.stderr)
            try:
                if viz_name in {"tool_syscalls", "tool_syscall_durations"}:
                    plotly_fn(
                        strace_data,
                        html_path,
                        max_syscalls_per_type=max_syscalls_per_type,
                        max_tool_calls=max_tool_calls,
                    )
                else:
                    plotly_fn(strace_data, html_path)
                generated.append(viz_name)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
        
        if not html_only and matplotlib_fn:
            png_path = output_dir / f"{viz_name}.png"
            print(f"Generating {png_path.name}...", file=sys.stderr)
            try:
                if viz_name in {"tool_syscalls", "tool_syscall_durations"}:
                    matplotlib_fn(
                        strace_data,
                        png_path,
                        max_syscalls_per_type=max_syscalls_per_type,
                    )
                else:
                    matplotlib_fn(strace_data, png_path)
                if viz_name not in generated:
                    generated.append(viz_name)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
    
    # Generate phase visualizations
    for viz_name in phase_viz_to_gen:
        plotly_fn, matplotlib_fn = PHASE_VISUALIZATIONS[viz_name]

        if not png_only and plotly_fn:
            html_path = output_dir / f"{viz_name}.html"
            print(f"Generating {html_path.name}...", file=sys.stderr)
            try:
                plotly_fn(phase_data, html_path)
                generated.append(viz_name)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)

        if not html_only and matplotlib_fn:
            png_path = output_dir / f"{viz_name}.png"
            print(f"Generating {png_path.name}...", file=sys.stderr)
            try:
                matplotlib_fn(phase_data, png_path)
                if viz_name not in generated:
                    generated.append(viz_name)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)

    # Generate agent visualizations (these load their own data from trace_dir).
    for viz_name in agent_viz_to_gen:
        plotly_fn, matplotlib_fn = AGENT_VISUALIZATIONS[viz_name]

        if not png_only and plotly_fn:
            html_path = output_dir / f"{viz_name}.html"
            print(f"Generating {html_path.name}...", file=sys.stderr)
            try:
                plotly_fn(trace_dir, html_path)
                generated.append(viz_name)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)

        if not html_only and matplotlib_fn:
            png_path = output_dir / f"{viz_name}.png"
            print(f"Generating {png_path.name}...", file=sys.stderr)
            try:
                matplotlib_fn(trace_dir, png_path)
                if viz_name not in generated:
                    generated.append(viz_name)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)

    # Create index dashboard
    print("Creating index.html dashboard...", file=sys.stderr)
    create_index_html(output_dir, generated)
    
    print(f"\nVisualizations saved to {output_dir}/", file=sys.stderr)
    print(f"Open {output_dir}/index.html in a browser to view.", file=sys.stderr)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from parsed strace data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s traces/20260115_190208/terraform-main/
  %(prog)s traces/20260115_190208/terraform-main/ --only timeline,operations
  %(prog)s traces/20260115_190208/terraform-main/ --html-only
        """
    )
    
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory containing parsed.json (from parse_strace.py)"
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Comma-separated list of visualizations to generate: " + 
             ", ".join(VISUALIZATIONS.keys())
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Generate only interactive HTML visualizations"
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Generate only static PNG visualizations"
    )
    parser.add_argument(
        "--max-syscalls-per-type",
        type=int,
        default=5000,
        help="Maximum syscalls to show per syscall type per tool call (default: 5000)"
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=MAX_TOOL_CALL_SUBPLOTS,
        help=f"Maximum tool calls to show in per-tool subplot charts (default: {MAX_TOOL_CALL_SUBPLOTS})"
    )
    
    args = parser.parse_args()

    # Parse --only argument
    only = args.only.split(",") if args.only else None

    # Cheap pre-check: if the trace produced 0 tool calls (agent likely
    # crashed at startup before any tool fired), don't try to draw anything.
    # Print a clear message and exit 0 so the orchestrator continues.
    parsed_json = args.trace_dir / "parsed.json"
    if parsed_json.exists():
        try:
            with open(parsed_json) as f:
                _peek = json.load(f)
            if not _peek.get("tool_calls"):
                print(
                    f"[visualize_strace] {args.trace_dir}: tool_calls is empty "
                    f"— agent did not invoke any tool (likely crashed in init). "
                    f"Skipping visualization. See sragent.err for the cause.",
                    file=sys.stderr,
                )
                sys.exit(0)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[visualize_strace] cannot peek at parsed.json: {e}", file=sys.stderr)

    generate_visualizations(
        trace_dir=args.trace_dir,
        only=only,
        html_only=args.html_only,
        png_only=args.png_only,
        max_syscalls_per_type=args.max_syscalls_per_type,
        max_tool_calls=args.max_tool_calls,
    )


if __name__ == "__main__":
    main()
