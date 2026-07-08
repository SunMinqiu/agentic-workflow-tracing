#!/usr/bin/env python3
from __future__ import annotations
"""
Visualize parsed strace data from Claude code runs.

Generates interactive HTML (Plotly) and static PNG (Matplotlib) visualizations
to analyze I/O behavior patterns from Linux strace output.

This is adapted from visualize_traces.py for the strace parser output format.
"""

import argparse
import csv
import html
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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agent_io_tracing.parsing.phases import PhaseAnalysis, load_phases


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
    "Tool-other": "#E67E22",
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
    "modify": "File-IO",
    "blocking": "Wait",
    "process": "Process-mgmt",
}

STORAGE_CONTROL_SYSCALLS = {"open", "openat", "close"}
NON_STORAGE_CONTROL_SYSCALLS = {
    "lseek", "fcntl", "ioctl", "chdir", "fchdir", "getcwd",
    "mmap", "munmap", "dup", "dup2", "dup3",
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
    if category == "control":
        return "File-IO" if syscall in STORAGE_CONTROL_SYSCALLS else "Other"
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
        cat_series = _effective_category_series(fs_df)
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
    cat_series = _effective_category_series(fs_df) \
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
        "ΣLLM": RESOURCE_COLORS["LLM"],
        "Tool": "#3498db",
        "ΣTool": "#3498db",
        "Orchestration": RESOURCE_COLORS["Orchestration"],
        "Parallel": RESOURCE_COLORS["Parallel"],
        "Idle": RESOURCE_COLORS["Idle"],
        "Residual": "#B0BEC5",
    }
    return [RESOURCE_COLORS.get(label, label_colors.get(label, fallback)) for label in labels]


def _phase_times(trace_dir: Path) -> dict | None:
    """Canonical ΣLLM / ΣTool / residual / total-work self-times (seconds),
    read from parallelism_summary.json so the Time Accounting donut shows the
    SAME numbers as the cross-cell summary. Falls back to recomputing via
    compute_parallelism. These are summed WORK times (robust to worker count);
    'total' is NOT wall clock.
    """
    def _from(d: dict):
        keys = ("llm_self_time_s", "tool_self_time_s",
                "residual_self_time_s", "total_work_s")
        if all(d.get(k) is not None for k in keys):
            return {"llm": float(d["llm_self_time_s"]),
                    "tool": float(d["tool_self_time_s"]),
                    "residual": float(d["residual_self_time_s"]),
                    "total": float(d["total_work_s"])}
        return None

    try:
        got = _from(json.loads((trace_dir / "parallelism_summary.json").read_text("utf-8")))
        if got:
            return got
    except Exception:
        pass
    try:
        from agent_io_tracing.analysis.parallelism import load_events, compute_summary
        got = _from(compute_summary(load_events(trace_dir)))
        if got:
            return got
    except Exception:
        pass
    return None


def create_phase_breakdown_plotly(trace_dir: Path, output_path: Path) -> None:
    """Time accounting — total work split into ΣLLM / ΣTool / residual.

    total = ΣLLM + ΣTool + residual (de-nested self-time + per-worker gap). It
    is total WORK (robust to worker count), NOT wall clock. No speedup framing.
    """
    t = _phase_times(trace_dir)
    if t is None or t["total"] <= 0:
        print(f"  phase_breakdown: no self-time data in {trace_dir}", file=sys.stderr)
        return

    labels = ["ΣLLM", "ΣTool", "Residual"]
    values = [t["llm"], t["tool"], t["residual"]]
    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=labels, values=values,
            marker_colors=_colors_for_labels(labels),
            hole=0.5, textinfo='percent', textposition='inside',
            insidetextorientation='horizontal',
            hovertemplate="<b>%{label}</b><br>%{value:.2f}s (%{percent})<extra></extra>",
            domain=dict(x=[0.05, 0.7], y=[0.08, 0.92]),
            sort=False,
        )
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.375, y=0.5,
        showarrow=False, align="center",
        font=dict(size=16, color="#2c3e50"),
        text=f"<b>{t['total']:.1f}s</b><br>total",
    )
    fig.update_layout(
        title="<b>Time accounting</b> — ΣLLM / ΣTool / residual (total = Σ work, de-nested)",
        height=560, width=1000, showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=0.72),
        margin=dict(l=20, r=20, t=70, b=40),
        paper_bgcolor="white",
    )
    fig.write_html(output_path)


def create_phase_breakdown_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """Time accounting — total work split into ΣLLM / ΣTool / residual (PNG).

    total = ΣLLM + ΣTool + residual (de-nested self-time + per-worker gap), the
    total WORK (robust to worker count), NOT wall clock. No speedup framing.
    """
    t = _phase_times(trace_dir)
    if t is None or t["total"] <= 0:
        print(f"  phase_breakdown: no self-time data in {trace_dir}", file=sys.stderr)
        return

    labels = ["ΣLLM", "ΣTool", "Residual"]
    vals = [t["llm"], t["tool"], t["residual"]]
    colors = _colors_for_labels(labels)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    ax.pie(
        vals, colors=colors, startangle=90,
        autopct=lambda p: f"{p:.1f}%" if p >= 1.0 else "",
        wedgeprops=dict(width=0.6),
    )
    ax.legend(
        [f"{n} — {v:.1f}s" for n, v in zip(labels, vals)],
        loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=False,
    )
    ax.text(
        0, 0, f"{t['total']:.1f}s\ntotal",
        ha='center', va='center', fontsize=13, fontweight='bold',
    )
    ax.set_title("Time accounting — ΣLLM / ΣTool / residual", fontsize=13)
    fig.text(
        0.5, 0.03,
        "total = ΣLLM + ΣTool + residual (de-nested self-time, summed work; not wall)",
        ha='center', fontsize=9, color="#7f8c8d",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close()


# I/O-abstraction layer colors (H1: interface choice). stdio/posix = "wrong for
# HPC" warm tones; structured/mpiio = "HPC-appropriate" cool tones.
_IO_LAYER_COLORS = {
    "stdio": "#d62728",          # buffered text I/O (the thing we expect too much of)
    "posix_raw": "#ff7f0e",      # raw POSIX / shell file tools
    "structured": "#2ca02c",     # HDF5/AnnData/Parquet/Zarr/sqlite
    "mpiio": "#1f77b4",          # mpi4py MPI.File
    "vector_index": "#9467bd",   # FAISS/Chroma/Qdrant
}
_IO_LAYER_ORDER = ["stdio", "posix_raw", "structured", "mpiio", "vector_index"]


def create_interface_mix_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """I/O-abstraction mix (H1) — how many code-exec calls used each I/O layer.

    Source: phase1_metrics.json['interface_mix'], itself an aggregate of
    generated_code.jsonl produced by io_api_classifier. If no code was captured
    (interface_mix.total_execs == 0) we draw an explicit placeholder rather than
    a blank, so a missing-capture run is obvious at a glance.
    """
    try:
        p1 = json.loads((trace_dir / "phase1_metrics.json").read_text())
    except Exception:
        p1 = {}
    mix = (p1.get("interface_mix") or {})
    total = mix.get("total_execs") or 0

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if not total:
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "No generated code captured\n(interface_mix.total_execs = 0)\n\n"
            "io_api_classifier did not run for this cell — re-run with "
            "io_api_classifier.py on sys.path.",
            ha="center", va="center", fontsize=12, color="#7f8c8d",
        )
        ax.set_title("I/O-abstraction mix (H1) — no data", fontsize=13)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    counts = mix.get("layer_exec_counts") or {}
    io_execs = mix.get("execs_with_file_io") or 0
    if not io_execs or not counts:
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            f"Generated code captured: {total} execution(s)\n"
            "File-I/O API usage in generated code: 0\n\n"
            "This means the captured SciLink dynamic-analysis code ran in memory. "
            "Any disk writes in this run were performed by SciLink/framework "
            "controllers rather than by the generated snippet itself.",
            ha="center", va="center", fontsize=12, color="#7f8c8d", wrap=True,
        )
        ax.set_title("Generated-code I/O API mix (H1) — no snippet file I/O", fontsize=13)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    layers = [l for l in _IO_LAYER_ORDER if l in counts] + \
             [l for l in counts if l not in _IO_LAYER_ORDER]
    vals = [counts[l] for l in layers]
    colors = [_IO_LAYER_COLORS.get(l, "#7f8c8d") for l in layers]

    y = np.arange(len(layers))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(layers, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("# code-exec calls using this I/O layer", fontsize=10)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v}", va="center", fontsize=9)

    pct_stdio = mix.get("pct_stdio_only")
    pct_struct = mix.get("pct_structured_any")
    sub = (f"{total} execs · {io_execs} with file I/O · "
           f"stdio-only {pct_stdio if pct_stdio is not None else 'n/a'}% · "
           f"structured-any {pct_struct if pct_struct is not None else 'n/a'}%")
    ax.set_title("Generated-code I/O API mix (H1) — interface choice\n" + sub, fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _fmt_bytes_short(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "0B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _no_data_placeholder(ax, title: str, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#7f8c8d")
    ax.set_title(title, fontsize=13)


def create_io_autocorrelation_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """I/O read/write autocorrelation at 1/5/25-minute windows.

    Source: phase1_metrics.json['io_autocorrelation']. The lag-1/2/3 values
    are precomputed by phase1_metrics, so this function only visualizes them.
    """
    try:
        p1 = json.loads((trace_dir / "phase1_metrics.json").read_text())
    except Exception:
        p1 = {}
    ac = p1.get("io_autocorrelation") or {}

    windows = ("1min", "5min", "25min")
    lags = (1, 2, 3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
    fig.suptitle("I/O read/write autocorrelation by time-window size", fontsize=14)

    any_data = False
    bar_w = 0.35
    for ax, wname in zip(axes, windows):
        w = ac.get(wname) or {}
        read_ac = w.get("read_autocorr") or {}
        write_ac = w.get("write_autocorr") or {}
        read_vals = [read_ac.get(f"lag{lag}") for lag in lags]
        write_vals = [write_ac.get(f"lag{lag}") for lag in lags]
        xcorr = w.get("read_write_xcorr_lag0")
        xcorr_txt = f"{xcorr:.2f}" if isinstance(xcorr, (int, float)) else "n/a"
        title = f"{wname} · rw xcorr lag0={xcorr_txt}"

        if not any(isinstance(v, (int, float)) for v in read_vals + write_vals):
            _no_data_placeholder(
                ax,
                title,
                "No autocorrelation data\n(bins too few or constant series)",
            )
            ax.set_ylim(-1, 1)
            continue

        any_data = True
        x = np.arange(len(lags))
        rv = [v if isinstance(v, (int, float)) else np.nan for v in read_vals]
        wv = [v if isinstance(v, (int, float)) else np.nan for v in write_vals]
        ax.bar(x - bar_w / 2, rv, width=bar_w, color="#1f77b4", label="read")
        ax.bar(x + bar_w / 2, wv, width=bar_w, color="#ff7f0e", label="write")
        ax.axhline(0, color="#34495e", linewidth=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(lag) for lag in lags])
        ax.set_xlabel("lag")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-1, 1)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("autocorrelation coefficient")
    if any_data:
        handles = [
            mpatches.Patch(color="#1f77b4", label="read"),
            mpatches.Patch(color="#ff7f0e", label="write"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_intensity_phases_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """I/O intensity phases over 60-second windows.

    The byte series is recomputed from parsed.json via phase1_metrics._binned_series
    with READ ∪ WRITE syscalls, matching the metric computation path.
    """
    try:
        parsed = json.loads((trace_dir / "parsed.json").read_text())
    except Exception:
        parsed = {}
    try:
        from agent_io_tracing.analysis.phase1_metrics import (
            READ_SYSCALLS_STRICT,
            WRITE_SYSCALLS_STRICT,
            _binned_series,
            percentile,
        )
        series = _binned_series(parsed, 60.0, READ_SYSCALLS_STRICT | WRITE_SYSCALLS_STRICT)
    except Exception:
        series = []

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    nonzero = [x for x in series if x > 0]
    if len(nonzero) < 4:
        _no_data_placeholder(
            ax,
            "I/O intensity phases — no data",
            "Too few active 60-second bins to segment\n(need at least 4 non-empty bins)",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    hi = percentile(nonzero, 75)
    lo = percentile(nonzero, 25)
    if hi is None or lo is None:
        _no_data_placeholder(
            ax,
            "I/O intensity phases — no data",
            "Could not compute intensity thresholds",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    high_mask = [x >= hi for x in series]
    low_mask = [0 < x <= lo for x in series]

    def _segments(mask: list[bool]) -> list[int]:
        segs, run = [], 0
        for matched in mask:
            if matched:
                run += 1
            elif run:
                segs.append(run)
                run = 0
        if run:
            segs.append(run)
        return segs

    hi_segs = _segments(high_mask)
    lo_segs = _segments(low_mask)
    hi_mean = (sum(hi_segs) / len(hi_segs)) if hi_segs else 0.0
    lo_mean = (sum(lo_segs) / len(lo_segs)) if lo_segs else 0.0

    colors = [
        "#d62728" if high else "#1f77b4" if low else "#9aa0a6"
        for high, low in zip(high_mask, low_mask)
    ]
    x = np.arange(len(series))
    ax.bar(x, series, color=colors, edgecolor="black", linewidth=0.2)
    ax.axhline(
        hi, color="#d62728", linestyle="--", linewidth=1.2,
        label=f"75th pct ({_fmt_bytes_short(hi)})",
    )
    ax.axhline(
        lo, color="#1f77b4", linestyle="--", linewidth=1.2,
        label=f"25th pct ({_fmt_bytes_short(lo)})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("60-second time-window index")
    ax.set_ylabel("bytes per window (log scale)")
    ax.set_title(
        f"{len(hi_segs)} high phases (mean len {hi_mean:.1f} bins) · "
        f"{len(lo_segs)} low phases (mean len {lo_mean:.1f} bins)",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    legend_handles = [
        mpatches.Patch(color="#d62728", label="high intensity"),
        mpatches.Patch(color="#1f77b4", label="low intensity"),
        mpatches.Patch(color="#9aa0a6", label="middle"),
    ]
    ax.legend(handles=legend_handles + ax.get_legend_handles_labels()[0], frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_reread_attribution_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """Reread attribution (Overview.md §5 three-bucket scheme) — how much of
    the observed file rereading is agent-induced vs. residual (a different
    tool_call_id, not proven to be a different pipeline stage).

    Source: phase1_metrics.json['reread_attribution'], joined on the specific
    tool_call_id (not the coarser phase:role label — see
    compute_reread_attribution's docstring in phase1_metrics.py for why that
    distinction matters).
    """
    try:
        p1 = json.loads((trace_dir / "phase1_metrics.json").read_text())
    except Exception:
        p1 = {}
    rr = p1.get("reread_attribution") or {}
    ai = rr.get("agent_induced") or {}
    same_step = ai.get("same_step_reopen") or {}
    after_bt = ai.get("reread_after_backtrack") or {}
    residual = rr.get("different_tool_call_id") or {}

    labels = ["same-step\nreopen\n(agent-induced)", "reread after\nbacktrack\n(agent-induced)",
             "different\ntool_call_id\n(residual)"]
    buckets = [same_step, after_bt, residual]
    counts = [b.get("count") or 0 for b in buckets]
    byte_vals = [b.get("bytes") or 0 for b in buckets]
    colors = ["#e67e22", "#c0392b", "#2980b9"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if not any(counts):
        _no_data_placeholder(
            ax, "Reread attribution — no data",
            "No rereads of any file were observed\n(every file was touched at most once)",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    x = np.arange(len(labels))
    ax.bar(x, counts, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("# reread touches", fontsize=10)
    for xi, c, b in zip(x, counts, byte_vals):
        ax.text(xi, c, f" {c}\n({_fmt_bytes_short(b)})", ha="center", va="bottom", fontsize=9)

    agent_induced_total = (same_step.get("count") or 0) + (after_bt.get("count") or 0)
    residual_total = residual.get("count") or 0
    ax.set_title(
        "Reread attribution (§5 three-bucket scheme)\n"
        f"{agent_induced_total} agent-induced vs. {residual_total} residual (different tool_call_id)",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_directory_scan_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """Directory scan (getdents64) count and rescan pattern — how many
    directories the agent's code re-listed via os.listdir()-style calls
    instead of caching the result once.

    Source: phase1_metrics.json['directory_scan'].
    """
    try:
        p1 = json.loads((trace_dir / "phase1_metrics.json").read_text())
    except Exception:
        p1 = {}
    ds = p1.get("directory_scan") or {}
    top = ds.get("top_rescanned") or []

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    if not ds.get("total_scans"):
        _no_data_placeholder(
            ax, "Directory scans — no data",
            "No getdents64 (directory listing) syscalls observed",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    if not top:
        total_scans = ds.get("total_scans", 0)
        unique_dirs = ds.get("unique_directories_scanned", 0)
        avg = (total_scans / unique_dirs) if unique_dirs else None
        _no_data_placeholder(
            ax, "Directory scans — no rescans",
            f"{total_scans} scans over {unique_dirs} dirs"
            + (f" (avg {avg:.1f}x),\n" if avg is not None else ",\n")
            + "0 directories rescanned >=2x",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    paths = [p for p, _ in top]
    counts = [c for _, c in top]
    short_paths = [p if len(p) <= 46 else "…" + p[-45:] for p in paths]

    y = np.arange(len(short_paths))
    ax.barh(y, counts, color="#8e44ad", edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(short_paths, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("# times scanned (getdents64 calls)", fontsize=10)
    for yi, c in zip(y, counts):
        ax.text(c, yi, f" {c}", va="center", fontsize=9)

    total_scans = ds.get("total_scans", 0)
    unique_dirs = ds.get("unique_directories_scanned", 0)
    rescanned_dirs = ds.get("rescanned_directories", 0)
    avg = (total_scans / unique_dirs) if unique_dirs else None
    ax.set_title(
        "Directory rescans (top offenders)\n"
        f"{total_scans} scans over {unique_dirs} dirs"
        + (f" (avg {avg:.1f}x)" if avg is not None else "")
        + f" · {rescanned_dirs}/{unique_dirs} rescanned >=2x",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_inter_arrival_cdf_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """Axis 2: CDF of the time gap between successive accesses to the same file.
    A summary table only shows p50/p95/p99; this shows the full shape and the
    long tail. The 1-second line makes the "amnesiac reread" story visceral:
    the fraction of re-accesses that recur near-instantly.

    Source: recomputed from parsed.json via
    phase1_metrics.inter_arrival_deltas (shared with the metric).
    """
    try:
        parsed = json.loads((trace_dir / "parsed.json").read_text())
    except Exception:
        parsed = {}
    try:
        from agent_io_tracing.analysis.phase1_metrics import inter_arrival_deltas
        deltas, n_files = inter_arrival_deltas(parsed)
    except Exception:
        deltas, n_files = [], 0

    pos = [d for d in deltas if d > 0]
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    if len(pos) < 2:
        _no_data_placeholder(
            ax, "Inter-arrival time — no data",
            "Fewer than 2 repeat accesses to any file were observed",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    def _fmt_secs(s: float) -> str:
        if s < 1e-3:
            return f"{s * 1e6:.0f}µs"
        if s < 1.0:
            return f"{s * 1e3:.0f}ms"
        return f"{s:.1f}s"

    arr = np.sort(np.asarray(pos))
    ys = np.arange(1, len(arr) + 1) / len(arr)
    frac_lt_1s = float(np.mean(arr < 1.0))
    p50, p95, p99 = (float(np.percentile(arr, q)) for q in (50, 95, 99))

    ax.plot(arr, ys, color="#1f77b4", lw=2)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.axvline(1.0, color="#d62728", ls="--", lw=1)
    ax.annotate(
        f"{frac_lt_1s * 100:.1f}% recur < 1s",
        xy=(1.0, frac_lt_1s), xytext=(6, -14), textcoords="offset points",
        fontsize=9, fontweight="bold", color="#d62728",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", lw=0.6, alpha=0.9),
    )
    ax.set_xlabel("inter-arrival time between successive accesses to the same file", fontsize=10)
    ax.set_ylabel("cumulative fraction of re-accesses", fontsize=10)
    ax.set_title(
        "Repeat-access inter-arrival CDF\n"
        f"{len(pos)} intervals across {n_files} re-accessed files · "
        f"p50={_fmt_secs(p50)}  p95={_fmt_secs(p95)}  p99={_fmt_secs(p99)}",
        fontsize=12,
    )
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
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


def _coalesce_intervals(
    intervals: list[tuple[float, float]], eps: float = 0.0
) -> list[tuple[float, float]]:
    """Union of intervals, also bridging gaps smaller than ``eps``.

    Like _merge_intervals but additionally joins intervals separated by a gap
    < eps. Used by the System lane to draw syscalls at REAL width while bounding
    the bar count: thousands of back-to-back microsecond reads (sub-pixel gaps)
    coalesce into a few real-width runs, instead of one bar each (unrenderable)
    OR pixel-inflated blocks (the bug where 1.7s of reads looked like 100s).
    """
    merged: list[tuple[float, float]] = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if merged and s <= merged[-1][1] + eps:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


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


def _intersect_intervals(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Intersection of two interval lists (each merged first)."""
    a = _merge_intervals(a)
    b = _merge_intervals(b)
    res: list[tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            res.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return res


def _dominant_bucket_segments(
    state_intervals: dict[str, list[tuple[float, float]]],
    span_s: float,
    px: int = _RASTER_PX,
) -> list[dict]:
    """Color each sub-pixel time bucket by the state with the MOST time in it.

    Unlike presence-rasterization (any event paints the bucket), this assigns
    each bucket to whichever state actually consumed the most wall-time in that
    bucket, then merges consecutive same-state buckets. So a long tool region
    with a few tiny reads stays "Tool-other", not "File-IO"; a region that is
    mostly file-IO stays "File-IO". Buckets with no activity are left blank
    (idle). Output is bounded to O(px) bars per state.
    """
    if span_s <= 0:
        return []
    bucket = span_s / px
    acc: dict[int, dict[str, float]] = {}
    for state, ivs in state_intervals.items():
        for s, e in ivs:
            if e <= s:
                continue
            i0 = int(s // bucket)
            i1 = int(min(e, span_s) // bucket)
            for i in range(i0, i1 + 1):
                bs = i * bucket
                ov = min(e, bs + bucket) - max(s, bs)
                if ov > 0:
                    acc.setdefault(i, {}).setdefault(state, 0.0)
                    acc[i][state] += ov
    chosen: dict[int, str] = {}
    for i, states in acc.items():
        chosen[i] = max(states.items(), key=lambda kv: kv[1])[0]
    runs: list[list] = []
    for i in sorted(chosen):
        st = chosen[i]
        if runs and runs[-1][2] == st and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i, st])
    return [
        {"resource": st, "start": a * bucket, "end": (b + 1) * bucket, "label": st}
        for a, b, st in runs
    ]


def _dominant_lane_segments(llm_iv, fileio_iv, tool_iv, span_s, px=_RASTER_PX):
    """Per-bucket dominant state for one agent lane: LLM / File-IO / Tool-other.

    Tool-other = tool time minus file-IO time, derived PER BUCKET (no O(n^2)
    interval subtraction — that hung on 455k-syscall traces). Each bucket is
    colored by whichever of LLM / File-IO / (tool−fileio) consumed the most time.
    """
    if span_s <= 0:
        return []
    bucket = span_s / px
    L: dict[int, float] = {}
    F: dict[int, float] = {}
    T: dict[int, float] = {}

    def _acc(ivs, d):
        for s, e in ivs:
            if e <= s:
                continue
            i0 = int(s // bucket)
            i1 = int(min(e, span_s) // bucket)
            for i in range(i0, i1 + 1):
                bs = i * bucket
                ov = min(e, bs + bucket) - max(s, bs)
                if ov > 0:
                    d[i] = d.get(i, 0.0) + ov

    _acc(llm_iv, L)
    _acc(fileio_iv, F)
    _acc(tool_iv, T)
    chosen: dict[int, str] = {}
    for i in set(L) | set(F) | set(T):
        opts = {"LLM": L.get(i, 0.0), "File-IO": F.get(i, 0.0),
                "Tool-other": max(0.0, T.get(i, 0.0) - F.get(i, 0.0))}
        best = max(opts.items(), key=lambda kv: kv[1])
        if best[1] > 0:
            chosen[i] = best[0]
    runs: list[list] = []
    for i in sorted(chosen):
        st = chosen[i]
        if runs and runs[-1][2] == st and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i, st])
    return [
        {"resource": st, "start": a * bucket, "end": (b + 1) * bucket, "label": st}
        for a, b, st in runs
    ]


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
    from agent_io_tracing.analysis.parallelism import (
        build_children,
        compute_self_intervals,
        load_events,
    )

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
    wall_s = wall_ms / 1000.0
    FILE_IO_CATS = {"metadata", "data", "control", "modify"}

    # Per-role interval sets: LLM windows, tool self-time, file-IO within tools.
    # Everything in a tool that is NOT file-IO (compute, framework, waiting,
    # process-mgmt) is lumped into "Tool-other" — we do not claim to separate
    # compute from wait (not possible from syscall latency alone).
    llm_iv_by_role: dict[str, list[tuple[float, float]]] = {}
    tool_iv_by_role: dict[str, list[tuple[float, float]]] = {}
    fileio_iv_by_role: dict[str, list[tuple[float, float]]] = {}

    for rid, ev in events.items():
        role = event_role[rid]
        if ev.kind == "llm":
            llm_iv_by_role.setdefault(role, []).append(
                ((ev.start_ms - t0) / 1000.0, (ev.end_ms - t0) / 1000.0))
        elif ev.kind == "tool":
            for s, e in self_iv_by_id.get(rid, []):
                if e > s:
                    tool_iv_by_role.setdefault(role, []).append(
                        ((s - t0) / 1000.0, (e - t0) / 1000.0))

    parsed_json = trace_dir / "parsed.json"
    if parsed_json.exists():
        fs_df = load_parsed_json(parsed_json).fs_entries_df
        t0_dt = datetime.fromtimestamp(t0 / 1000.0)
        if len(fs_df) and "timestamp" in fs_df.columns:
            cols = ["matched_tool_call", "syscall", "timestamp", "duration"]
            if "path" in fs_df.columns:
                cols.append("path")
            df = fs_df[cols].copy()
            df = df[df["matched_tool_call"].isin(event_role.keys())]
            df["cat"] = _effective_category_series(df.rename(columns={"syscall": "operation"}))
            df = df[df["cat"].isin(FILE_IO_CATS)]
        else:
            df = None
        if df is not None and len(df):
            t0_ts = pd.Timestamp(t0_dt)
            t0_midnight = pd.Timestamp(t0_dt.date())
            ts = df["timestamp"]
            aligned = t0_midnight + (ts - ts.dt.normalize())
            shift_s = 0
            gap_s = (aligned.min() - t0_ts).total_seconds()
            if abs(gap_s) >= 1800:
                shift_s = round(gap_s / 900) * 900
            end_rel = (aligned - t0_ts).dt.total_seconds() - shift_s
            duration_s = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
            start_rel = end_rel - duration_s
            keep = (end_rel >= -1.0) & (start_rel <= wall_s + 1.0)
            for tid, s, e in zip(
                df["matched_tool_call"][keep], start_rel[keep], end_rel[keep]
            ):
                fileio_iv_by_role.setdefault(event_role[tid], []).append(
                    (max(0.0, float(s)), max(0.0, float(e))))

    roles = set(llm_iv_by_role) | set(tool_iv_by_role) | set(fileio_iv_by_role)
    lanes = []
    for role in roles:
        llm_iv = _merge_intervals(llm_iv_by_role.get(role, []))
        tool_iv = _merge_intervals(tool_iv_by_role.get(role, []))
        fileio_iv = _intersect_intervals(
            tool_iv, _merge_intervals(fileio_iv_by_role.get(role, [])))
        busy = _merge_intervals(llm_iv + tool_iv)
        busy_s = sum(e - s for s, e in busy)
        # Per-bucket dominant (Tool-other derived as tool−fileio inside the
        # bucketizer — no O(n^2) interval subtraction).
        display = _dominant_lane_segments(llm_iv, fileio_iv, tool_iv, wall_s)
        lanes.append({
            "role": role, "label": role,
            "segments": display, "busy": busy, "busy_s": busy_s,
        })
    lanes.sort(key=lambda lane: lane["busy"][0][0] if lane["busy"] else 0.0)

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
        for label in ["LLM", "File-IO", "Tool-other"]
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
    from agent_io_tracing.parsing.tool_log import parse_tool_calls_log as parse_log

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

# Categories that count as file-system I/O (vs process/network/blocking/other).
FILE_IO_CATEGORIES = ("metadata", "data", "control", "modify")


def _effective_category_series(fs_df):
    """syscall -> category, but read/write on NON-files are reclassified to 'other'.

    A read()/write() on a pipe / stdin / stdout / socket (no resolved file path)
    is NOT file I/O — it is IPC / waiting (e.g. a blocking read on a subprocess's
    stdout pipe that blocks for the whole subprocess runtime). Counting that as
    'data'/File-IO is what made File-IO look like ~135s when real file I/O was
    ~0.6s. Such path-less data syscalls become 'other'.
    """
    op = fs_df["operation"]
    cat = op.map(_SYSCALL_TO_CATEGORY).fillna("other")
    # wait4/waitpid/waitid = waiting for a child process, NOT process creation.
    cat = cat.where(~op.isin({"wait4", "waitpid", "waitid"}), "wait")
    # Only open/close control calls are storage time. mmap/ioctl/fcntl/chdir
    # are process/control overhead and must not inflate File-IO.
    cat = cat.where(~((cat == "control") & ~op.isin(STORAGE_CONTROL_SYSCALLS)), "other")
    # read/write on a NON-file (pipe / stdin / stdout / socket; no resolved path)
    # is IPC / blocking-on-a-subprocess, not file I/O. Both go to the "wait"
    # sentinel, which is NOT one of the System-lane rows, so they are excluded.
    if "path" in fs_df.columns:
        path = fs_df["path"]
        pathless = path.isna() | (path.astype(str).str.len() == 0) | (path.astype(str) == "None")
    else:
        pathless = pd.Series(True, index=fs_df.index)
    return cat.where(~((cat == "data") & pathless), "wait")


def _syscall_category_seconds(
    fs_df, categories: list[str]
) -> tuple[dict[str, float], float]:
    """Per-category UNION-length seconds + File-system-I/O total (union).

    Each category's syscall intervals [time_rel, time_rel+duration] are merged
    (overlaps once) and summed → honest wall-seconds that category was active.
    The File-IO total is the union of ALL file-IO categories' intervals together
    (NOT the sum of the per-row numbers — they overlap), so it never exceeds e2e.
    """
    if fs_df is None or len(fs_df) == 0 or "time_rel" not in fs_df.columns:
        return {c: 0.0 for c in categories}, 0.0
    cat_series = _effective_category_series(fs_df)
    dur = pd.to_numeric(fs_df.get("duration", 0.0), errors="coerce").fillna(0.0)
    start = fs_df["time_rel"].astype(float)
    end = start + dur
    per: dict[str, float] = {}
    fileio_iv: list[tuple[float, float]] = []
    for cat in categories:
        m = (cat_series == cat).to_numpy()
        if not m.any():
            per[cat] = 0.0
            continue
        ivs = list(zip(start[m].tolist(), end[m].tolist()))
        per[cat] = sum(e - s for s, e in _merge_intervals(ivs))
        if cat in FILE_IO_CATEGORIES:
            fileio_iv.extend(ivs)
    fileio_total = sum(e - s for s, e in _merge_intervals(fileio_iv))
    return per, fileio_total


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

    # System lane: fixed row order (blocking/wait dropped — not charted).
    syscall_rows = ["metadata", "data", "control", "modify", "process",
                    "network", "other"]

    # Per-category union seconds + File-system-I/O total (union of the 4 file-IO
    # rows; NOT the sum). Used to annotate the row labels and the lane title.
    _strace = bundle.get("strace_data")
    _fs_df = _strace.fs_entries_df if _strace is not None else None
    cat_secs, fileio_total_s = _syscall_category_seconds(_fs_df, syscall_rows)
    _e2e = bundle["total_span_s"]
    fileio_pct = (fileio_total_s / _e2e * 100.0) if _e2e > 0 else 0.0
    sys_row_label = {c: f"{c} — {cat_secs.get(c, 0.0):.1f}s" for c in syscall_rows}

    # --- Make subplot scaffolding -----------------------------------------
    sem_h = max(1, len(semantic_labels))
    tool_h = max(1, len(tool_labels))
    sys_h = len(syscall_rows)
    total_h = sem_h + tool_h + sys_h
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=False,
        column_widths=[0.84, 0.16],
        row_heights=[sem_h / total_h, tool_h / total_h, sys_h / total_h],
        vertical_spacing=0.04,
        horizontal_spacing=0.035,
        specs=[
            [{"type": "xy"}, {"type": "domain"}],
            [{"type": "xy"}, {"type": "domain"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Semantic — LLM + subagents",
            "",
            "Tool — real tools (from tool_calls.log)",
            "",
            "System — FS syscalls · per-row = union seconds (not summable)",
            "Total seconds",
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
        cat_series = _effective_category_series(fs_df)
        dur = pd.to_numeric(fs_df.get("duration", 0.0), errors="coerce").fillna(0.0)
        start = fs_df["time_rel"].astype(float)
        end = start + dur
        _minw = span_s / _RASTER_PX
        for cat in syscall_rows:
            mask = (cat_series == cat).to_numpy()
            if not mask.any():
                continue
            color = SYSCALL_CATEGORY_COLORS.get(cat, SYSCALL_CATEGORY_COLORS["other"])
            cstart = start[mask].to_numpy()
            cend = end[mask].to_numpy()
            label = sys_row_label[cat]
            # Bars: union runs at REAL width, sub-pixel runs dropped (drawing a
            # μs read ≥1px would inflate the total into a solid block — the bug).
            runs = [(s, e) for s, e in
                    _merge_intervals(list(zip(cstart.tolist(), cend.tolist())))
                    if e - s >= _minw]
            if runs:
                fig.add_trace(go.Bar(
                    x=[e - s for s, e in runs], y=[label] * len(runs),
                    base=[s for s, e in runs], orientation="h",
                    marker_color=color, marker_line_width=0, opacity=0.7,
                    name=f"sys: {cat}", legendgroup=f"sys:{cat}", showlegend=True,
                    hovertemplate=f"<b>{cat}</b> — {cat_secs.get(cat, 0.0):.2f}s total<extra></extra>",
                ), row=3, col=1)
            # Dots: one marker per syscall to show WHERE activity is, WITHOUT
            # faking width. Sampled when very many (purely positional density).
            xs = cstart
            if len(xs) > 6000:
                xs = xs[np.linspace(0, len(xs) - 1, 6000).astype(int)]
            fig.add_trace(go.Scatter(
                x=xs, y=[label] * len(xs), mode="markers",
                marker=dict(color=color, size=3, opacity=0.45),
                name=f"sys: {cat}", legendgroup=f"sys:{cat}", showlegend=not runs,
                hovertemplate=f"<b>{cat}</b> — {cat_secs.get(cat, 0.0):.2f}s total<extra></extra>",
            ), row=3, col=1)

    cat_total = sum(cat_secs.get(c, 0.0) for c in syscall_rows)
    if cat_total > 0:
        left = 0.0
        for cat in syscall_rows:
            val = cat_secs.get(cat, 0.0)
            if val <= 0:
                continue
            fig.add_trace(go.Bar(
                x=[val],
                y=["total"],
                base=[left],
                orientation="h",
                marker_color=SYSCALL_CATEGORY_COLORS.get(cat, SYSCALL_CATEGORY_COLORS["other"]),
                marker_line_width=0,
                name=f"total: {cat}",
                legendgroup=f"sys:{cat}",
                showlegend=False,
                hovertemplate=f"<b>{cat}</b><br>%{{x:.2f}}s union time<extra></extra>",
            ), row=3, col=2)
            left += val

    # --- Layout polish ----------------------------------------------------
    title_extra = ""
    if residue_pct > REASONABLE_RESIDUE_PCT:
        title_extra = (f" — ⚠ {residue_s:.1f}s ({residue_pct:.1f}%) unattributed")
    fig.update_layout(
        title=(f"Agent Timeline — total {bundle['total_span_s']:.1f}s · "
               f"File-system I/O {fileio_total_s:.1f}s ({fileio_pct:.0f}% of e2e)"
               f"{title_extra}"),
        barmode="overlay",
        height=200 + 40 * total_h,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=1.0),
        margin=dict(l=140, r=180, t=80, b=40),
    )
    for row in (1, 2, 3):
        fig.update_xaxes(range=[0, bundle["total_span_s"]], row=row, col=1)
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    fig.update_xaxes(title_text="seconds", row=3, col=2, range=[0, max(cat_total, 1e-9)])
    fig.update_yaxes(categoryorder="array", categoryarray=semantic_labels[::-1], row=1, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=tool_labels[::-1], row=2, col=1)
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[sys_row_label[c] for c in syscall_rows][::-1],
        row=3, col=1,
    )
    fig.update_yaxes(showticklabels=False, row=3, col=2)

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
                    "network", "other"]

    _strace = bundle.get("strace_data")
    _fs_df = _strace.fs_entries_df if _strace is not None else None
    cat_secs, fileio_total_s = _syscall_category_seconds(_fs_df, syscall_rows)
    _e2e = bundle["total_span_s"]
    fileio_pct = (fileio_total_s / _e2e * 100.0) if _e2e > 0 else 0.0

    sem_h = max(1, len(semantic_labels))
    tool_h = max(1, len(tool_labels))
    sys_h = len(syscall_rows)

    fig = plt.figure(figsize=(16, max(8, 0.45 * (sem_h + tool_h + sys_h) + 3)))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[sem_h, tool_h, sys_h],
        width_ratios=[18, 2.4],
        hspace=0.18,
        wspace=0.08,
    )
    ax_sem = fig.add_subplot(gs[0, 0])
    ax_tool = fig.add_subplot(gs[1, 0], sharex=ax_sem)
    ax_sys = fig.add_subplot(gs[2, 0], sharex=ax_sem)
    ax_total = fig.add_subplot(gs[2, 1])

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
        cat_series = _effective_category_series(fs_df)
        dur = pd.to_numeric(fs_df.get("duration", 0.0), errors="coerce").fillna(0.0)
        start = fs_df["time_rel"].astype(float)
        end = start + dur
        _minw = span_s / _RASTER_PX
        for cat_idx, cat in enumerate(syscall_rows):
            mask = (cat_series == cat).to_numpy()
            if not mask.any():
                continue
            color = SYSCALL_CATEGORY_COLORS.get(cat, SYSCALL_CATEGORY_COLORS["other"])
            cstart = start[mask].to_numpy()
            cend = end[mask].to_numpy()
            # Bars: union runs at REAL width, sub-pixel dropped — see plotly twin.
            runs = [(s, e) for s, e in
                    _merge_intervals(list(zip(cstart.tolist(), cend.tolist())))
                    if e - s >= _minw]
            if runs:
                ax_sys.barh(
                    y=[cat_idx] * len(runs),
                    width=[e - s for s, e in runs],
                    left=[s for s, e in runs],
                    color=color, alpha=0.7, height=0.7, edgecolor="none",
                )
            # Dots: per-syscall position (no width faking), sampled if many.
            xs = cstart
            if len(xs) > 6000:
                xs = xs[np.linspace(0, len(xs) - 1, 6000).astype(int)]
            ax_sys.scatter(xs, [cat_idx] * len(xs), c=color, s=3, alpha=0.4,
                           linewidths=0)
    ax_sys.set_yticks(range(len(syscall_rows)))
    ax_sys.set_yticklabels(
        [f"{c} — {cat_secs.get(c, 0.0):.1f}s" for c in syscall_rows], fontsize=8)
    ax_sys.set_ylim(-0.6, len(syscall_rows) - 0.4)
    ax_sys.invert_yaxis()
    ax_sys.set_title(
        "System — FS syscalls · per-row = union seconds (not summable)",
        fontsize=10, loc="left",
    )
    ax_sys.set_xlabel("time (s)")
    ax_sys.grid(axis="x", alpha=0.2)
    ax_sys.set_xlim(0, bundle["total_span_s"] if bundle["total_span_s"] > 0 else 1.0)

    cat_total = sum(cat_secs.get(c, 0.0) for c in syscall_rows)
    left = 0.0
    for cat in syscall_rows:
        val = cat_secs.get(cat, 0.0)
        if val <= 0:
            continue
        ax_total.barh(
            y=0, width=val, left=left,
            color=SYSCALL_CATEGORY_COLORS.get(cat, SYSCALL_CATEGORY_COLORS["other"]),
            height=0.55, edgecolor="none",
        )
        left += val
    ax_total.set_xlim(0, cat_total if cat_total > 0 else 1.0)
    ax_total.set_yticks([])
    ax_total.set_xlabel("seconds", fontsize=8)
    ax_total.set_title("Total seconds", fontsize=9)
    ax_total.grid(axis="x", alpha=0.2)
    for spine in ("left", "right", "top"):
        ax_total.spines[spine].set_visible(False)

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

    title = (f"Agent Timeline — total {bundle['total_span_s']:.1f}s · "
             f"File-system I/O {fileio_total_s:.1f}s ({fileio_pct:.0f}% of e2e)")
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
    trace_dir = output_dir.parent
    lineage_dir = trace_dir / "lineage"

    def infer_title_parts(path: Path) -> tuple[str, str]:
        parts = list(path.parts)
        try:
            rel_parts = parts[parts.index("results") + 1:]
        except ValueError:
            rel_parts = [path.name]
        workload = rel_parts[-1] if rel_parts else path.name
        run_id = rel_parts[-2] if len(rel_parts) >= 2 else workload
        for part in reversed(rel_parts[:-1]):
            if re.search(r"\d{8}(?:_\d{6})?", part):
                run_id = part
                break
        return run_id, workload

    run_id, workload = infer_title_parts(trace_dir)
    page_title = f"{run_id} · {workload} · I/O Pattern"

    def jload(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def load_artifact_rows() -> list[dict]:
        path = lineage_dir / "artifacts.csv"
        if not path.is_file():
            return []
        try:
            with path.open(newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except Exception:
            return []

    def fmt_num(value, digits: int = 2) -> str:
        if value is None or value == "":
            return "n/a"
        if isinstance(value, (int, float)):
            if value == 0:
                return "0"
            if digits == 0:
                return f"{value:,.0f}"
            if abs(value) >= 1000:
                return f"{value:,.0f}"
            return f"{value:.{digits}f}".rstrip("0").rstrip(".")
        return str(value)

    def fmt_bytes(value) -> str:
        if value is None or value == "":
            return "n/a"
        try:
            x = float(value)
        except (TypeError, ValueError):
            return str(value)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(x) < 1024 or unit == "TB":
                return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
            x /= 1024.0
        return f"{x:.1f} TB"

    def esc(text) -> str:
        return html.escape(str(text), quote=True)

    def metric(label: str, value: str) -> str:
        return f"<div class='metric'><span>{esc(label)}</span><strong>{esc(value)}</strong></div>"

    p1 = jload(trace_dir / "phase1_metrics.json")
    par = jload(trace_dir / "parallelism_summary.json")
    io_summary = jload(lineage_dir / "io_summary.json") or jload(trace_dir / "io_summary.json")
    artifact_rows = load_artifact_rows()
    ratios = p1.get("metadata_data_ratio") or {}
    fsz = p1.get("file_size_cdf") or {}
    files_per_tool = ((p1.get("namespace") or {}).get("files_per_tool_call") or {})
    fs_non_llm = p1.get("fs_io_non_llm") or {}
    amp = p1.get("analytical_optimum_amplification") or {}
    wl = (io_summary.get("workload") or {})

    # Setup / Global only — descriptive aggregates that belong to no single
    # axis. Axis-specific numbers (file size, amplification, request size, …)
    # live in their axis sections below, not up here.
    fs_io_s = fs_non_llm.get("fs_io_ms_sum") / 1000.0 if fs_non_llm.get("fs_io_ms_sum") is not None else None
    fs_io_pct_wall = fs_non_llm.get("fs_io_pct_of_wall")
    wall_s = par.get("wall_clock_s")
    llm_s = (
        fs_non_llm.get("llm_ms") / 1000.0
        if fs_non_llm.get("llm_ms") is not None
        else par.get("total_self_time_s")
    )
    llm_pct_wall = None
    if isinstance(wall_s, (int, float)) and wall_s > 0 and isinstance(llm_s, (int, float)):
        llm_pct_wall = 100.0 * llm_s / wall_s
    headline = [
        ("read/write bytes", f"{fmt_bytes(wl.get('read_bytes'))} / {fmt_bytes(wl.get('write_bytes'))}"),
        ("distinct files (generated)", f"{fmt_num(wl.get('distinct_files'))} ({fmt_num(amp.get('actual_generated_files'))})"),
        ("distinct files / tool_call (mean, p95)", f"{fmt_num(files_per_tool.get('mean'))} / {fmt_num(files_per_tool.get('p95'))}"),
        ("metadata/data ratio", fmt_num(ratios.get("storage_metadata_to_data_ops"))),
        ("FS-I/O time", f"{fmt_num(fs_io_s)} s" + (f" ({fmt_num(fs_io_pct_wall)}% of wall)" if fs_io_pct_wall is not None else "")),
        ("LLM time / wall", f"{fmt_num(llm_s)} s / {fmt_num(wall_s)} s" + (f" ({fmt_num(llm_pct_wall)}%)" if llm_pct_wall is not None else "")),
    ]

    latency = p1.get("latency_by_phase") or {}
    latency_rows = []
    for phase, vals in sorted(latency.items()):
        latency_rows.append(
            "<tr>"
            f"<td>{esc(phase)}</td>"
            f"<td>{fmt_num(vals.get('p50_ms'))}</td>"
            f"<td>{fmt_num(vals.get('p95_ms'))}</td>"
            f"<td>{fmt_num(vals.get('p99_ms'))}</td>"
            f"<td>{fmt_num(vals.get('count'), 0)}</td>"
            "</tr>"
        )
    latency_table = ""
    if latency_rows:
        latency_table = (
            "<p class='muted'>phase = tag logged by the workflow's own adapter per "
            "tool_call (e.g. code_exec, action_unit_backtrack), not inferred from syscalls</p>"
            "<table class='compact'><thead><tr><th>phase</th><th>p50 ms</th>"
            "<th>p95 ms</th><th>p99 ms</th><th>n</th></tr></thead><tbody>"
            + "".join(latency_rows) + "</tbody></table>"
        )

    # --- Attribution summary tables (Overview.md §5/§6 rollups) -----------

    def kv_table(headers: list[str], rows: list[list]) -> str:
        if not rows:
            return "<p class='muted'>No data.</p>"
        head = "".join(f"<th>{esc(h)}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>"
            for row in rows
        )
        return f"<table class='compact'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    iface = p1.get("interface_mix") or {}
    iface_bytes = p1.get("interface_byte_mix") or {}
    iface_total = iface.get("total_execs") or 0
    iface_io_execs = iface.get("execs_with_file_io") or 0
    observed_read_ops = amp.get("actual_read_ops")
    observed_write_ops = amp.get("actual_write_ops")
    observed_io_note = (
        f"Observed by eBPF: {fmt_num(observed_read_ops, 0)} read-family syscalls and "
        f"{fmt_num(observed_write_ops, 0)} write-family syscalls across the process. "
        "These are different layers: generated-code API choice vs kernel-level I/O."
    )
    iface_note = (
        f"{fmt_num(iface_total, 0)} generated-code execution(s); "
        f"{fmt_num(iface_io_execs, 0)} used file-I/O APIs in the generated snippet. "
        "(static, from generated source — not measured syscall bytes). "
        f"{observed_io_note}"
        if iface_total else (
            "No generated code captured for this cell. "
            "(static generated-source mix unavailable — not measured syscall bytes). "
            f"{observed_io_note}"
        )
    )
    iface_byte_table = kv_table(["interface byte mix", "value"], [
        ["STDIO bytes (fread/fwrite)", fmt_bytes(iface_bytes.get("stdio_bytes"))],
        ["POSIX observed bytes (read/write syscalls)", fmt_bytes(iface_bytes.get("posix_observed_bytes"))],
        ["direct POSIX bytes est.", fmt_bytes(iface_bytes.get("posix_direct_bytes_est"))],
        ["STDIO / direct POSIX est.", f"{fmt_num(iface_bytes.get('stdio_pct_deoverlapped'))}% / {fmt_num(iface_bytes.get('posix_direct_pct_deoverlapped'))}%"],
    ])
    iface_byte_note = iface_bytes.get("note") or (
        "Requires new traces with libc fread/fwrite uprobes; older traces show zeros."
    )
    generated_code_records = []
    generated_code_path = trace_dir / "generated_code.jsonl"
    if generated_code_path.is_file():
        for line in generated_code_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            generated_code_records.append(rec)

    saved_generated_scripts = sorted((trace_dir / "scilink_session").glob("results/**/*.py"))

    def rel_from_output(path: Path) -> str:
        try:
            return "../" + path.relative_to(trace_dir).as_posix()
        except ValueError:
            return str(path)

    def generated_code_evidence_table() -> str:
        rows = []
        for rec in generated_code_records[:8]:
            rid = str(rec.get("run_id") or "")[:8] or "?"
            imports = ", ".join(rec.get("imports") or []) or "(none)"
            layers = ", ".join(rec.get("io_layers") or []) or "(no file-I/O API)"
            rows.append(
                "<tr>"
                f"<td>{esc(rid)}</td>"
                f"<td>{esc(fmt_num(rec.get('code_len'), 0))}</td>"
                f"<td>{esc(imports)}</td>"
                f"<td>{esc(layers)}</td>"
                "</tr>"
            )
        if not rows:
            rows.append("<tr><td colspan='4' class='muted'>No generated_code.jsonl records.</td></tr>")
        source_links = []
        if generated_code_path.is_file():
            source_links.append(
                f"<a href='../generated_code.jsonl'>generated_code.jsonl</a>"
            )
        for script in saved_generated_scripts[:4]:
            source_links.append(
                f"<a href='{esc(rel_from_output(script))}'>{esc(script.name)}</a>"
            )
        links_html = (
            "<p class='muted'>Evidence: " + " · ".join(source_links) + "</p>"
            if source_links else ""
        )
        return (
            "<table class='compact'><thead><tr>"
            "<th>run id</th><th>code len</th><th>imports</th><th>classified I/O API layer</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            + links_html
        )

    state = p1.get("state_file_rewrite_frequency") or {}
    state_rows = [
        [Path(r.get("path") or "").name or "?", fmt_num(r.get("n_writes"), 0),
         fmt_num(r.get("n_reads"), 0), fmt_bytes(r.get("total_write_bytes"))]
        for r in sorted(state.get("per_file") or [], key=lambda r: -(r.get("n_writes") or 0))
    ]
    state_table = kv_table(["state file", "n_writes", "n_reads", "write bytes"], state_rows)
    state_note = "matches: " + " · ".join(state.get("path_hints") or [])

    ds = p1.get("directory_scan") or {}
    total_scans = ds.get("total_scans") or 0
    unique_dirs = ds.get("unique_directories_scanned") or 0
    rescanned_dirs = ds.get("rescanned_directories") or 0
    avg_scans_per_dir = (total_scans / unique_dirs) if unique_dirs else None
    ds_note = (
        f"{fmt_num(total_scans, 0)} scans over {fmt_num(unique_dirs, 0)} dirs"
        + (f" (avg {fmt_num(avg_scans_per_dir)}×)" if avg_scans_per_dir is not None else "")
        + f" · {fmt_num(rescanned_dirs, 0)}/{fmt_num(unique_dirs, 0)} rescanned ≥2×"
    )

    fos = p1.get("failed_open_stat") or {}
    fos_rows = [[syscall, fmt_num(count, 0)] for syscall, count in (fos.get("by_syscall") or {}).items()]
    fos_table = kv_table(["failed syscall", "# failures"], fos_rows)

    elr = p1.get("error_log_reads") or {}
    elr_note = (
        f"{fmt_num(elr.get('log_files'), 0)} log file(s), "
        f"{fmt_num(elr.get('log_files_ever_read'), 0)} ever read, "
        f"{fmt_num(elr.get('total_reads'), 0)} total reads "
        f"({fmt_bytes(elr.get('total_read_bytes'))})"
    )

    bop = p1.get("bytes_ops_by_phase") or {}
    bop_rows = [
        [phase, fmt_num(vals.get("read_ops"), 0), fmt_bytes(vals.get("read_bytes")),
         fmt_num(vals.get("write_ops"), 0), fmt_bytes(vals.get("write_bytes"))]
        for phase, vals in sorted(bop.items())
    ]
    bop_table = kv_table(["phase", "read ops", "read bytes", "write ops", "write bytes"], bop_rows)

    # --- Newly implemented axis metrics (phase1_metrics.json) --------------

    rh = p1.get("access_type_rhwhrw") or {}
    rhp = rh.get("by_class_pct") or {}
    rh_table = kv_table(["access type", "share"], [
        ["read-heavy", fmt_num(rhp.get("read_heavy")) + "%"],
        ["write-heavy", fmt_num(rhp.get("write_heavy")) + "%"],
        ["read-write", fmt_num(rhp.get("read_write")) + "%"],
    ])
    rh_note = (
        f"{rh.get('definition') or ''} · n={fmt_num(rh.get('n_files_classified'), 0)} "
        "files with I/O bytes (RH+WH+RW denominator)"
    ).strip(" ·")

    eo = p1.get("exploration_overhead") or {}
    eo_ratio = eo.get("exploration_overhead_ratio")
    eo_table = kv_table(["exploration overhead", "value"], [
        ["overhead ratio", (fmt_num(eo_ratio * 100) + "%") if isinstance(eo_ratio, (int, float)) else "n/a"],
        ["backtrack-phase bytes", fmt_bytes(eo.get("backtrack_phase_bytes"))],
        ["dead-write bytes", fmt_bytes(eo.get("dead_write_bytes"))],
        ["total data bytes", fmt_bytes(eo.get("total_data_bytes"))],
    ])

    req = p1.get("request_size_cdf") or {}
    req_table = kv_table(["request size", "value"], [
        ["p50 / p95 / p99", " / ".join(fmt_bytes(req.get(k)) for k in ("p50_bytes", "p95_bytes", "p99_bytes"))],
        ["% <4KB", fmt_num(req.get("pct_lt_4kb")) + "%"],
        ["% <10MB", fmt_num(req.get("pct_lt_10mb")) + "%"],
    ])
    req_note = "one request = one read/write-family syscall (metadata calls excluded)"

    rca = amp.get("read_call_amplification")
    wca = amp.get("write_call_amplification")
    amp_table = kv_table(["I/O batching efficiency", "value"], [
        ["read call amplification", (fmt_num(rca) + "×") if rca is not None else "n/a"],
        ["write call amplification", (fmt_num(wca) + "×") if wca is not None else "n/a"],
        ["actual / optimum read ops", f"{fmt_num(amp.get('actual_read_ops'), 0)} / {fmt_num(amp.get('optimum_read_ops'), 0)}"],
    ])
    amp_note = "amplification = actual ops / ceil(bytes / 4MB baseline)"

    fsz_table = kv_table(["file size", "value"], [
        ["p50 / p95 / p99", " / ".join(fmt_bytes(fsz.get(k)) for k in ("p50_bytes", "p95_bytes", "p99_bytes"))],
        ["% <1MB", fmt_num(fsz.get("pct_lt_1mb")) + "%"],
        ["% <1GB", fmt_num(fsz.get("pct_lt_1gb")) + "%"],
    ])

    seq_note = (
        "Planned categories: consecutive / backward / gap-random. Not measurable here: "
        "plain read/write syscalls carry no offset argument, and offset-bearing variants "
        "(pread64/pwrite64/preadv/pwritev) had zero occurrences in every trace so far."
    )

    fca = amp.get("file_count_amplification")
    moa = amp.get("metadata_op_amplification")
    logphys_table = kv_table(["logical→physical", "value"], [
        ["file count amplification", (fmt_num(fca) + "×") if fca is not None else "n/a"],
        ["metadata op amplification", (fmt_num(moa) + "×") if moa is not None else "n/a"],
    ])

    pdeg = (par.get("parallel_degree") or {}).get("semantic_events") or {}
    pardeg_table = kv_table(["workflow concurrency", "value"], [
        ["avg active when busy", fmt_num(pdeg.get("avg_active_when_busy"))],
        ["max active", fmt_num(pdeg.get("max_active"), 0)],
        ["parallel time ratio", fmt_num(pdeg.get("parallel_time_ratio"))],
    ])
    io_pdeg = (par.get("parallel_degree") or {}).get("io_busy_workers") or {}
    io_pardeg_table = kv_table(["I/O concurrency", "value"], [
        ["avg I/O-busy workers", fmt_num(io_pdeg.get("avg_active_when_busy"))],
        ["max I/O-busy workers", fmt_num(io_pdeg.get("max_active"), 0)],
        ["I/O parallel time ratio", fmt_num(io_pdeg.get("parallel_time_ratio"))],
        ["I/O-busy workers / events", f"{fmt_num(io_pdeg.get('workers'), 0)} / {fmt_num(io_pdeg.get('events'), 0)}"],
        ["I/O bytes counted", fmt_bytes(io_pdeg.get("bytes"))],
    ])

    def links_for(base: str, rel_prefix: str = "") -> str:
        pieces = []
        html_file = output_dir / f"{base}.html"
        png_file = output_dir / f"{base}.png"
        if html_file.exists():
            pieces.append(f"<a href='{rel_prefix}{base}.html'>HTML</a>")
        if png_file.exists():
            pieces.append(f"<a href='{rel_prefix}{base}.png'>PNG</a>")
        return "".join(pieces)

    def figure_card(title: str, img_rel: str | None, links: str, caption: str = "") -> str:
        if not links and (not img_rel or not (output_dir / img_rel).exists()):
            return ""
        img = f"<img src='{esc(img_rel)}' alt='{esc(title)}'>" if img_rel else ""
        caption_html = f"<p class='muted'>{esc(caption)}</p>" if caption else ""
        return f"<article class='card'>{img}<div><h3>{esc(title)}</h3>{caption_html}<nav>{links}</nav></div></article>"

    def viz_card(viz: str, title: str) -> str:
        links = links_for(viz)
        img_rel = f"{viz}.png" if (output_dir / f"{viz}.png").exists() else None
        return figure_card(title, img_rel, links)

    def lineage_card(fname: str, title: str, caption: str = "") -> str:
        path = lineage_dir / fname
        if not path.exists():
            return ""
        rel = f"../lineage/{fname}"
        return figure_card(title, rel, f"<a href='{rel}'>PNG</a>", caption)

    def external_card(rel: str, title: str) -> str:
        if not (output_dir / rel).exists():
            return ""
        return figure_card(title, None, f"<a href='{rel}'>Open</a>")

    def as_float(value) -> float | None:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def median(values: list[float]) -> float | None:
        vals = [float(v) for v in values if v is not None]
        return float(np.median(vals)) if vals else None

    def artifact_summary_from_rows() -> dict:
        unknown = [r for r in artifact_rows if (r.get("size_source") or "") == "unknown"]
        stale_vals = [as_float(r.get("staleness_s")) for r in artifact_rows]
        stale_vals = [v for v in stale_vals if v is not None]
        generated = [r for r in artifact_rows if as_float(r.get("total_write_bytes")) and as_float(r.get("total_write_bytes")) > 0]
        write_once_leaf = [r for r in generated if (r.get("lifecycle_class") or "") == "ephemeral_leaf"]
        reclaimable = [
            r for r in generated
            if (as_float(r.get("dead_seconds")) or 0.0) > 0
        ]
        return {
            "artifact_sizes": {
                "has_unknown_size": bool(unknown),
                "unknown_size_files": len(unknown),
            },
            "staleness": {
                "n": len(stale_vals),
                "median_s": median(stale_vals),
            },
            "lifecycle": {
                "generated_files": len(generated),
                "write_once_leaf_files": len(write_once_leaf),
                "write_once_leaf_pct": (
                    100.0 * len(write_once_leaf) / len(generated) if generated else None
                ),
                "reclaimable_files": len(reclaimable),
                "reclaimable_pct": (
                    100.0 * len(reclaimable) / len(generated) if generated else None
                ),
                "generated_median_dead_s": median([
                    as_float(r.get("dead_seconds")) or 0.0 for r in generated
                ]),
                "reclaimable_median_dead_s": median([
                    as_float(r.get("dead_seconds")) or 0.0 for r in reclaimable
                ]),
            },
        }

    row_summary = artifact_summary_from_rows() if artifact_rows else {}
    artifact_size_summary = io_summary.get("artifact_sizes") or row_summary.get("artifact_sizes") or {}
    staleness_summary = io_summary.get("staleness") or row_summary.get("staleness") or {}
    lifecycle_summary = {
        **(row_summary.get("lifecycle") or {}),
        **(io_summary.get("lifecycle") or {}),
    }
    if (
        lifecycle_summary.get("reclaimable_pct") is None
        and lifecycle_summary.get("generated_files")
        and lifecycle_summary.get("reclaimable_files") is not None
    ):
        lifecycle_summary["reclaimable_pct"] = (
            100.0
            * float(lifecycle_summary["reclaimable_files"])
            / float(lifecycle_summary["generated_files"])
        )

    reader_fanout_caption = (
        "(?) = size unknown — read-only input not stat-able after the run "
        "(not recorded in artifact_sizes.json)"
        if artifact_size_summary.get("has_unknown_size")
        else ""
    )
    staleness_title = "Write→First-Read Staleness"
    if staleness_summary.get("median_s") is not None:
        staleness_title += f" (median={fmt_num(staleness_summary.get('median_s'))}s)"

    lifecycle_title = "Artifact Lifecycle"
    lifecycle_bits = []
    generated_files = lifecycle_summary.get("generated_files")
    write_once_leaf_files = lifecycle_summary.get("write_once_leaf_files")
    reclaimable_files = lifecycle_summary.get("reclaimable_files")
    if lifecycle_summary.get("write_once_leaf_pct") is not None:
        prefix = ""
        if generated_files is not None and write_once_leaf_files is not None:
            prefix = f"{fmt_num(write_once_leaf_files, 0)}/{fmt_num(generated_files, 0)} generated "
        lifecycle_bits.append(
            f"{prefix}write-once leaf ({fmt_num(lifecycle_summary.get('write_once_leaf_pct'), 0)}%)"
        )
    if reclaimable_files is not None:
        reclaim = ""
        if generated_files is not None:
            pct_val = lifecycle_summary.get("reclaimable_pct")
            pct_text = f" ({fmt_num(pct_val, 0)}%)" if pct_val is not None else ""
            reclaim = f"{fmt_num(reclaimable_files, 0)}/{fmt_num(generated_files, 0)} reclaimable{pct_text}"
        else:
            reclaim = f"{fmt_num(reclaimable_files, 0)} reclaimable"
        lifecycle_bits.append(reclaim)
    if lifecycle_summary.get("generated_median_dead_s") is not None:
        lifecycle_bits.append(
            f"median dead {fmt_num(lifecycle_summary.get('generated_median_dead_s'))}s"
        )
    if (
        lifecycle_summary.get("reclaimable_median_dead_s") is not None
        and lifecycle_summary.get("reclaimable_median_dead_s") != lifecycle_summary.get("generated_median_dead_s")
    ):
        lifecycle_bits.append(
            f"reclaimable median dead {fmt_num(lifecycle_summary.get('reclaimable_median_dead_s'))}s"
        )
    if lifecycle_bits:
        lifecycle_title += " (" + " · ".join(lifecycle_bits) + ")"

    # --- Figures grouped by axis ------------------------------------------
    setup_figs = "".join([
        lineage_card("fig0_io_volume_summary.png", "I/O Volume Summary"),
        viz_card("agent_timeline", "Agent Timeline"),
        viz_card("phase_breakdown", "Time Accounting"),
        external_card("../call_dag.html", "Call DAG with I/O"),
    ])
    ax1_figs = ""
    ax2_figs = "".join([
        viz_card("directory_scan", "Directory Rescans"),
        viz_card("inter_arrival_cdf", "Inter-arrival CDF"),
        viz_card("reread_attribution", "Reread Attribution"),
        lineage_card(
            "fig2_reader_fanout.png", "Reader Fan-out",
            reader_fanout_caption,
        ),
        lineage_card("fig3_staleness_cdf.png", staleness_title),
        lineage_card("fig6_reuse_pattern.png", "Reuse Pattern"),
    ])
    _iface_card = viz_card("interface_mix", "Generated-Code I/O API Mix")
    ax3_figs = _iface_card + (f"<p class='muted'>{esc(iface_note)}</p>" if _iface_card else "")
    ax4_figs = lineage_card("fig4_lifecycle.png", lifecycle_title)
    ax5_figs = lineage_card("fig1_size_distribution.png", "File Size Distribution")
    ax6_figs = "".join([
        viz_card("agent_concurrency", "Agent Activity Timeline"),
        lineage_card("fig7_role_io_attribution.png", "Who Does the I/O"),
        viz_card("io_rate", "I/O Rate Over Time"),
        viz_card("io_autocorrelation", "I/O Autocorrelation"),
        viz_card("intensity_phases", "Intensity Phases"),
    ])

    # --- Per-axis table grids ---------------------------------------------
    def panel(title: str, table: str, note: str = "") -> str:
        note_html = f'<p class="muted">{esc(note)}</p>' if note else ""
        return f"<div><h3>{esc(title)}</h3>{note_html}{table}</div>"

    def grid(*cells: str) -> str:
        inner = "".join(c for c in cells if c)
        return f'<div class="attr-grid">{inner}</div>' if inner else ""

    ax1_tables = grid(
    )
    ax2_tables = grid(
        panel("Directory rescans", "", ds_note),
        panel("Access type RH/WH/RW", rh_table, rh_note),
    )
    ax3_tables = grid(
        panel("Generated-code evidence", generated_code_evidence_table(), iface_note),
        panel("Measured STDIO/POSIX byte mix", iface_byte_table, iface_byte_note),
        panel("State file rewrite frequency", state_table, state_note),
        panel("Logical→physical amplification", logphys_table),
    )
    ax4_tables = grid(
        panel("Failed open/stat", fos_table),
        panel("Error-log reads", "", elr_note),
        panel("Bytes/ops by phase", bop_table),
    )
    ax5_tables = grid(
        panel("Request size", req_table, req_note),
        panel("I/O Batching Efficiency", amp_table, amp_note),
        panel("File size", fsz_table),
        panel("Access pattern", "", seq_note),
    )
    ax6_tables = grid(
        panel(
            "Workflow concurrency (opportunity)",
            pardeg_table,
            "time-weighted active semantic events; busy time excludes idle gaps, "
            "parallel time ratio = time with >=2 active events / busy time",
        ),
        panel(
            "I/O concurrency",
            io_pardeg_table,
            "time-weighted overlap of worker I/O-busy intervals from read/write-family syscalls "
            "and libc fread/fwrite probes when available",
        ),
    )
    detail_links = []
    for viz, title in [
        ("timeline", "Timeline View"),
        ("tool_syscalls", "Syscalls Per Tool Call"),
        ("tool_syscall_durations", "Syscall Duration Distributions"),
    ]:
        links = links_for(viz)
        if links:
            detail_links.append(f"<div class='linkrow'><span>{esc(title)}</span><nav>{links}</nav></div>")

    data_links = [
        ("phase1_metrics.json", "../phase1_metrics.json"),
        ("io_summary.json", "../lineage/io_summary.json"),
        ("artifacts.csv", "../lineage/artifacts.csv"),
        ("tool_call_attribution.csv", "../lineage/tool_call_attribution.csv"),
        ("generated_code.jsonl", "../generated_code.jsonl"),
        ("manifest.json", "../manifest.json"),
        ("call_dag.html", "../call_dag.html"),
        ("parallelism_summary.json", "../parallelism_summary.json"),
    ]
    data_html = []
    for label, rel in data_links:
        if (output_dir / rel).exists():
            data_html.append(f"<a href='{rel}'>{esc(label)}</a>")

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(page_title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f8fb;
      color: #1f2933;
    }}
    header {{
      padding: 28px 32px 18px;
      background: #ffffff;
      border-bottom: 1px solid #d9e2ec;
    }}
    h1 {{ margin: 0; font-size: 26px; }}
    h2 {{ margin: 28px 0 14px; font-size: 18px; }}
    .wrap {{ max-width: 1440px; margin: 0 auto; padding: 0 24px 36px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .metric {{
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 12px;
    }}
    .metric span {{ display: block; color: #627d98; font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 17px; }}
    .zone {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      overflow: hidden;
    }}
    .card img {{
      display: block;
      width: 100%;
      height: 180px;
      object-fit: cover;
      border-bottom: 1px solid #d9e2ec;
      background: #eef2f7;
    }}
    .card div {{ padding: 12px; }}
    .card h3 {{ margin: 0 0 10px; font-size: 15px; }}
    nav {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    a {{
      display: inline-block;
      padding: 7px 10px;
      border-radius: 6px;
      background: #e6f0ff;
      color: #0b5cad;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
    }}
    .linkrow {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 8px;
    }}
    .data {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    table.compact {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      overflow: hidden;
      margin-top: 12px;
    }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf2f7; text-align: left; font-size: 13px; }}
    th {{ color: #52606d; background: #f0f4f8; }}
    .attr-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-top: 14px;
    }}
    .attr-grid h3 {{ margin: 0 0 6px; font-size: 14px; }}
    .attr-summary {{ margin-top: 6px; }}
    .muted {{ color: #627d98; font-size: 12px; margin: 4px 0; }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>{esc(page_title)}</h1>
      <div class="metrics">{''.join(metric(k, v) for k, v in headline)}</div>
      {latency_table}
    </div>
  </header>
  <main class="wrap">
    <h2>Global</h2>
    <section class="zone">{setup_figs or '<p class="muted">No setup figures found.</p>'}</section>
    <h2>Axis 1 · Runtime Decision Timing</h2>
    <section class="zone">{ax1_figs}</section>
    <section class="attr-summary">{ax1_tables}</section>
    <h2>Axis 2 · Context-Limited State Persistence</h2>
    <section class="zone">{ax2_figs}</section>
    <section class="attr-summary">{ax2_tables}</section>
    <h2>Axis 3 · Generated-Code I/O Interface Layer</h2>
    <section class="zone">{ax3_figs}</section>
    <section class="attr-summary">{ax3_tables}</section>
    <h2>Axis 4 · Exploratory Branching and Backtracking</h2>
    <section class="zone">{ax4_figs}</section>
    <section class="attr-summary">{ax4_tables}</section>
    <h2>Axis 5 · Reasoning-Step/Data-Granularity Alignment</h2>
    <section class="zone">{ax5_figs}</section>
    <section class="attr-summary">{ax5_tables}</section>
    <h2>Axis 6 · Uncoordinated Agent Concurrency</h2>
    <section class="zone">{ax6_figs}</section>
    <section class="attr-summary">{ax6_tables}</section>
    <h2>Detail</h2>
    <section>{''.join(detail_links) or '<p>No detail views found.</p>'}</section>
    <h2>Data & Artifacts</h2>
    <section class="data">{''.join(data_html) or '<p>No data artifacts found.</p>'}</section>
  </main>
</body>
</html>
"""

    (output_dir / "index.html").write_text(html_content, encoding="utf-8")


# =============================================================================
# Main Visualization Runner
# =============================================================================

# Visualizations that use StraceData (from parsed.json)
STRACE_VISUALIZATIONS = {
    "timeline": (create_timeline_plotly, create_timeline_matplotlib),
    # io_rate (syscalls/100ms with tool-call overlays) re-enabled: it is the
    # per-cell "storage rate over time" view and is wanted for the high-
    # concurrency runs later. process_timeline stays disabled (redundant with
    # agent_timeline).
    "io_rate": (create_io_rate_plotly, create_io_rate_matplotlib),
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
    "interface_mix": (None, create_interface_mix_matplotlib),
    "inter_arrival_cdf": (None, create_inter_arrival_cdf_matplotlib),
    "reread_attribution": (None, create_reread_attribution_matplotlib),
    "directory_scan": (None, create_directory_scan_matplotlib),
    "io_autocorrelation": (None, create_io_autocorrelation_matplotlib),
    "intensity_phases": (None, create_intensity_phases_matplotlib),
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
        help="Directory containing parsed.json (from parse_ebpf.py)"
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
