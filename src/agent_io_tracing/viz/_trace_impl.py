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


def create_io_rate_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """Bytes/s read/write rate with inference intensity overlay."""
    parsed_json = trace_dir / "parsed.json"
    if not parsed_json.exists():
        fig, ax = plt.subplots(figsize=(10, 4))
        _no_data_placeholder(ax, "I/O rate — no data", "Missing parsed.json")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return
    data = load_parsed_json(parsed_json)
    fs_df = data.fs_entries_df.copy()
    if fs_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        _no_data_placeholder(ax, "I/O rate — no data", "No filesystem events")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    syscalls = fs_df["syscall"].astype(str) if "syscall" in fs_df.columns else pd.Series([], dtype=str)
    sizes = (
        fs_df.get("bytes_transferred", 0).fillna(0)
        if "bytes_transferred" in fs_df.columns else pd.Series(0, index=fs_df.index)
    )
    read_mask = syscalls.isin(["read", "pread64", "readv", "preadv", "preadv2"]) & (sizes > 0)
    write_mask = syscalls.isin(["write", "pwrite64", "writev", "pwritev", "pwritev2"]) & (sizes > 0)

    llm_intervals: list[tuple[float, float]] = []
    llm_abs_intervals: list[tuple[float, float]] = []
    # (start_ms, end_ms, output_tokens) for calls that reported usage. Every
    # adapter's launcher records usage on message_end, so this works on every
    # trace without streaming deltas (which litellm-based agents never emit).
    llm_abs_tokens: list[tuple[float, float, float]] = []
    try:
        from agent_io_tracing.analysis.parallelism import _load_llm_events

        llms, _, _ = _load_llm_events(trace_dir / "pi_events.jsonl")
        llm_abs_intervals = [
            (ev.start_ms, ev.end_ms)
            for ev in llms
            if ev.end_ms > ev.start_ms
        ]
        for ev in llms:
            if ev.end_ms <= ev.start_ms:
                continue
            out = (ev.usage or {}).get("output")
            if isinstance(out, (int, float)) and out > 0:
                llm_abs_tokens.append((ev.start_ms, ev.end_ms, float(out)))
    except Exception:
        llm_abs_intervals = []
        llm_abs_tokens = []

    if "ts_ms" in fs_df.columns and fs_df["ts_ms"].notna().any():
        fs_abs_ms = fs_df["ts_ms"].astype(float)
        abs_candidates = [float(fs_abs_ms.min()), float(fs_abs_ms.max())]
        abs_candidates.extend(t for iv in llm_abs_intervals for t in iv)
        t0_ms = min(abs_candidates)
        t1_ms = max(abs_candidates)
        fs_time_rel = (fs_abs_ms - t0_ms) / 1000.0
        llm_intervals = [((s - t0_ms) / 1000.0, (e - t0_ms) / 1000.0) for s, e in llm_abs_intervals]
        duration_seconds = max((t1_ms - t0_ms) / 1000.0, 0.0)
    else:
        t0_ms = data.start_time.timestamp() * 1000.0
        fs_abs_start = float(fs_df["time_rel"].min()) * 1000.0 + t0_ms
        fs_abs_end = float(fs_df["time_rel"].max()) * 1000.0 + t0_ms
        if llm_abs_intervals:
            ev_start = min(s for s, _ in llm_abs_intervals)
            ev_end = max(e for _, e in llm_abs_intervals)
            if min(fs_abs_end, ev_end) - max(fs_abs_start, ev_start) < -1000.0:
                raise RuntimeError(
                    "Cannot overlay LLM intensity on I/O rate: parsed.json lacks absolute ts_ms "
                    "and filesystem timestamps are skewed from LLM epoch timestamps. Re-parse "
                    "ebpf_events.log with the current parser."
                )
        fs_time_rel = fs_df["time_rel"]
        llm_intervals = [((s - t0_ms) / 1000.0, (e - t0_ms) / 1000.0) for s, e in llm_abs_intervals]
        duration_seconds = data.duration_seconds

    bin_size = 1.0 if duration_seconds <= 1800 else 2.0 if duration_seconds <= 7200 else 5.0
    bins = np.arange(0, max(duration_seconds, bin_size) + bin_size, bin_size)
    centers = (bins[:-1] + bins[1:]) / 2
    read_bytes, _ = np.histogram(fs_time_rel[read_mask], bins=bins, weights=sizes[read_mask])
    write_bytes, _ = np.histogram(fs_time_rel[write_mask], bins=bins, weights=sizes[write_mask])
    read_rate = read_bytes / bin_size
    write_rate = write_bytes / bin_size
    # Output-token rate: spread each call's output tokens uniformly over its
    # own [start, end] and accumulate the overlap with each bin. Decoding is
    # roughly steady within a call, so this is a good decode-intensity signal.
    token_rate = np.zeros(len(centers))
    token_rate_available = False
    for s_ms, e_ms, tokens in llm_abs_tokens:
        s = (s_ms - t0_ms) / 1000.0
        e = (e_ms - t0_ms) / 1000.0
        span = e - s
        if span <= 0:
            continue
        tokens_per_s = tokens / span
        lo = max(int(s / bin_size), 0)
        hi = min(int(e / bin_size) + 1, len(centers))
        for i in range(lo, hi):
            overlap = min(e, bins[i + 1]) - max(s, bins[i])
            if overlap > 0:
                token_rate[i] += tokens_per_s * overlap / bin_size
                token_rate_available = True

    # Scale the I/O-rate axis to KB/s at minimum, stepping up to MB/s when the
    # peak rate warrants it; raw bytes/s forces unreadable 1e7 tick labels.
    peak_rate = float(max(read_rate.max(initial=0.0), write_rate.max(initial=0.0)))
    if peak_rate >= 1024 ** 2:
        rate_div, rate_unit = 1024 ** 2, "MB/s"
    else:
        rate_div, rate_unit = 1024, "KB/s"

    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax2 = ax.twinx()
    ax.plot(centers, read_rate / rate_div, color="#1f77b4", lw=1.6, label=f"read {rate_unit}")
    ax.plot(centers, write_rate / rate_div, color="#ff7f0e", lw=1.6, label=f"write {rate_unit}")
    if token_rate_available:
        ax2.plot(centers, token_rate, color="#2ca02c", lw=1.3,
                 drawstyle="steps-post", label="output tokens/s")
        ax2.set_ylabel("output tokens/s")
    ax.set_xlabel("Time (seconds from run start)")
    ax.set_ylabel(rate_unit)
    ax.set_xlim(0, duration_seconds)
    ax.grid(alpha=0.25)
    ax.set_title("I/O Rate Over Time")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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


def create_measured_interface_layers_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """Measured I/O interface layers from eBPF syscalls and uprobes."""
    try:
        p1 = json.loads((trace_dir / "phase1_metrics.json").read_text())
    except Exception:
        p1 = {}
    measured = p1.get("measured_interface_layers") or {}
    layers = measured.get("layers") or {}

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    order = ["STDIO", "POSIX", "MMAP", "HDF5", "MPI-IO"]
    names = list(order)
    vals = [int((layers.get(name) or {}).get("ops") or 0) for name in names]
    if not any(vals):
        _no_data_placeholder(
            ax,
            "Measured I/O interface mix (uprobe/syscall) — no data",
            "No measured interface-layer events.\n"
            "Older traces lack HDF5/MPI uprobes; workloads without HDF5/MPI will show zero.",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    colors = {
        "STDIO": _IO_LAYER_COLORS["stdio"],
        "POSIX": _IO_LAYER_COLORS["posix_raw"],
        "MMAP": _IO_LAYER_COLORS.get("vector_index", "#8c564b"),
        "HDF5": _IO_LAYER_COLORS["structured"],
        "MPI-IO": _IO_LAYER_COLORS["mpiio"],
    }
    y = np.arange(len(names))
    ax.barh(y, vals, color=[colors.get(n, "#7f8c8d") for n in names],
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("# measured calls", fontsize=10)
    for yi, name, val in zip(y, names, vals):
        layer = layers.get(name) or {}
        suffix = ""
        if val == 0:
            suffix = ""
        elif layer.get("bytes_resolved") and layer.get("bytes") is not None:
            suffix = f" · {_fmt_bytes_short(layer.get('bytes'))}"
        elif name == "MMAP":
            mapped = layer.get("mapped_bytes_upper_bound")
            suffix = f" · ≤{_fmt_bytes_short(mapped)} mapped" if mapped else " · bytes unresolved"
        elif name in {"HDF5", "MPI-IO"}:
            suffix = " · bytes unresolved"
        ax.text(val, yi, f" {val}{suffix}", va="center", fontsize=9)
    ax.set_title("Measured I/O interface mix (uprobe/syscall)", fontsize=13)
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
    """Directory scan count histogram from phase1_metrics.json."""
    try:
        p1 = json.loads((trace_dir / "phase1_metrics.json").read_text())
    except Exception:
        p1 = {}
    ds = p1.get("directory_scan") or {}
    hist = ds.get("scans_per_dir_hist") or {}

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

    labels = [*(str(i) for i in range(1, 10)), ">=10"]
    counts = [int(hist.get(label, 0) or 0) for label in labels]
    x = np.arange(len(labels))
    ax.bar(x, counts, color="#8e44ad", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("# scans of one directory (getdents calls)", fontsize=10)
    ax.set_ylabel("# directories", fontsize=10)
    for xi, c in zip(x, counts):
        if c:
            ax.text(xi, c, f" {c}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Directory Re-scans (getdents64)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_inter_arrival_cdf_matplotlib(trace_dir: Path, output_path: Path) -> None:
    """Axis 1: histogram of gaps between successive accesses to the same file."""
    try:
        p1 = json.loads((trace_dir / "phase1_metrics.json").read_text())
    except Exception:
        p1 = {}
    ia = p1.get("inter_arrival") or {}
    hist = ia.get("hist") or {}
    labels = ia.get("hist_bins") or ["<1s", "1-30s", "30s-5min", "5-30min", ">30min"]
    counts = [int(hist.get(label, 0) or 0) for label in labels]
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    if not any(counts):
        _no_data_placeholder(
            ax, "Inter-arrival time — no data",
            "Fewer than 2 repeat accesses to any file were observed",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    x = np.arange(len(labels))
    ax.bar(x, counts, color="#1f77b4", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("gap between successive accesses to the same file", fontsize=10)
    ax.set_ylabel("# inter-arrival intervals", fontsize=10)
    ax.set_title("Inter-arrival Time", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_effective_bandwidth_matplotlib(trace_dir: Path, output_path: Path) -> None:
    try:
        p1 = json.loads((trace_dir / "phase1_metrics.json").read_text())
    except Exception:
        p1 = {}
    ebw = p1.get("effective_bandwidth") or {}
    by_phase = ebw.get("by_phase") or {}
    rows = []
    for phase, vals in by_phase.items():
        read = vals.get("read") or {}
        write = vals.get("write") or {}
        total_bytes = (read.get("bytes") or 0) + (write.get("bytes") or 0)
        has_reliable_bw = isinstance(read.get("effective_Bps"), (int, float)) or isinstance(write.get("effective_Bps"), (int, float))
        if total_bytes > 0 and has_reliable_bw:
            rows.append((phase, read, write, total_bytes))
    rows.sort(key=lambda r: r[3], reverse=True)
    rows = rows[:12]

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(rows) + 1.8)))
    if not rows:
        _no_data_placeholder(ax, "Effective bandwidth — no data", "No reliable workload read/write bandwidth samples")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    y = np.arange(len(rows))

    def mbps(stat: dict) -> float:
        v = stat.get("effective_Bps")
        return float(v) / (1024 * 1024) if isinstance(v, (int, float)) else 0.0

    read_vals = [mbps(r[1]) for r in rows]
    write_vals = [mbps(r[2]) for r in rows]
    height = 0.36
    ax.barh(y - height / 2, read_vals, height, color="#1f77b4", label="read")
    ax.barh(y + height / 2, write_vals, height, color="#ff7f0e", label="write")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("effective bandwidth (MiB/s)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Effective Bandwidth by Phase")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_access_pattern_matplotlib(trace_dir: Path, output_path: Path) -> None:
    try:
        p1 = json.loads((trace_dir / "phase1_metrics.json").read_text())
    except Exception:
        p1 = {}
    seq = p1.get("sequentiality") or {}
    four = seq.get("four_cell") or {}
    vals = [
        int(four.get("seq_read") or 0),
        int(four.get("rand_read") or 0),
        int(four.get("seq_write") or 0),
        int(four.get("rand_write") or 0),
    ]

    fig, ax0 = plt.subplots(1, 1, figsize=(8.5, 4.6))
    total_plotted = sum(vals)
    if not total_plotted:
        note = seq.get("note") or "No offset-bearing workload data operations"
        _no_data_placeholder(ax0, "Access pattern — no classified transitions", note)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    labels = ["Seq R", "Rand R", "Seq W", "Rand W"]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]
    xs = np.arange(len(labels))
    ax0.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.3)
    ax0.set_xticks(xs)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("# classified transitions / append ops")
    ax0.grid(axis="y", alpha=0.25)
    ax0.set_title("Access Pattern")
    # A stream is classified Seq/Rand only when it has a *transition* (>=2 ops on
    # the same fd/open-generation) or is an append. Single-write outputs (open ->
    # write once -> close, no O_APPEND) produce no transition, so writes can be
    # present in the trace yet show 0 here.
    wr_by_kind = (seq.get("ops_with_offset_by_kind") or {})
    if vals[2] == 0 and vals[3] == 0 and int(wr_by_kind.get("write") or 0) > 0:
        fig.text(
            0.5, -0.02,
            f"note: {int(wr_by_kind.get('write'))} write ops occurred but none formed a "
            "seq/rand transition (single-write streams, no O_APPEND) -> Seq/Rand W = 0",
            ha="center", va="top", fontsize=8, color="#a55",
        )
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


# Bundle cache: phase_breakdown and agent_timeline both call
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

    def fmt_seconds(value) -> str:
        if value is None or value == "":
            return "not resolved"
        return fmt_num(value) + "s"

    def esc(text) -> str:
        return html.escape(str(text), quote=True)

    def metric(label: str, value: str) -> str:
        return f"<div class='metric'><span>{esc(label)}</span><strong>{esc(value)}</strong></div>"

    p1 = jload(trace_dir / "phase1_metrics.json")
    par = jload(trace_dir / "parallelism_summary.json")
    io_summary = jload(lineage_dir / "io_summary.json") or jload(trace_dir / "io_summary.json")
    artifact_rows = load_artifact_rows()
    ratios = p1.get("metadata_data_ratio") or {}
    req = p1.get("request_size_cdf") or {}
    files_per_tool = ((p1.get("namespace") or {}).get("files_per_tool_call") or {})
    fs_non_llm = p1.get("fs_io_non_llm") or {}
    amp = p1.get("analytical_optimum_amplification") or {}
    seq = p1.get("sequentiality") or {}
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

    measured_iface = p1.get("measured_interface_layers") or {}

    state = p1.get("state_file_rewrite_frequency") or {}
    state_rows = [
        [Path(r.get("path") or "").name or "?", fmt_num(r.get("n_writes"), 0),
         fmt_num(r.get("n_reads"), 0), fmt_bytes(r.get("total_write_bytes"))]
        for r in sorted(state.get("per_file") or [], key=lambda r: -(r.get("n_writes") or 0))
    ]
    state_table = kv_table(["state file", "n_writes", "n_reads", "write bytes"], state_rows)
    state_note = "matches: " + " · ".join(state.get("path_hints") or [])

    fos = p1.get("failed_open_stat") or {}
    fos_rows = [[syscall, fmt_num(count, 0)] for syscall, count in (fos.get("by_syscall") or {}).items()]
    if not fos_rows and "total_failed" in fos:
        fos_rows = [["all tracked open/stat syscalls", "0 failures"]]
    fos_table = kv_table(["failed syscall", "# failures"], fos_rows)
    _fr = fos.get("failed_rate")
    fos_note = (
        f"{fmt_num(fos.get('total_failed'), 0)} agent-level failed probe(s)"
        + (f", {fmt_num(_fr * 100)}% of agent open/stat attempts" if _fr is not None else "")
        + f" · excluded {fmt_num(fos.get('import_probe_failed_excluded'), 0)} CPython import probe(s)"
        f" of {fmt_num(fos.get('total_failed_raw'), 0)} raw failures"
    )

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

    merge = seq.get("mergeability") or {}
    amp_table = kv_table(["I/O batching efficiency", "value"], [
        ["mergeable ops saved", f"{fmt_num(merge.get('saved_ops'), 0)} / {fmt_num(merge.get('actual_ops_with_offset'), 0)}"],
        ["saved ops share", f"{fmt_num(merge.get('saved_ops_pct_of_actual_ops'))}%"],
        ["bytes in consecutive runs", f"{fmt_bytes(merge.get('bytes_in_consecutive_runs'))} / {fmt_bytes(merge.get('bytes_total_with_offset'))}"],
        ["consecutive-run byte share", f"{fmt_num(merge.get('bytes_in_consecutive_runs_pct'))}%"],
    ])
    if not merge.get("actual_ops_with_offset"):
        amp_note = seq.get("note") or (
            "Offset coverage unavailable; old traces need VFS offset capture before mergeability can be computed."
        )
    else:
        amp_note = merge.get("note") or ""

    pc_counts = {"1-1": 0, "1-n": 0, "n-1": 0, "n-n": 0}
    pc_total = 0
    for _key, _n in ((io_summary.get("fanout") or {}).get("reader_writer_joint") or {}).items():
        try:
            _w, _r = (int(x) for x in str(_key).split(",", 1))
            _n = int(_n)
        except (TypeError, ValueError):
            continue
        pc_total += _n
        if _w <= 1 and _r <= 1:
            pc_counts["1-1"] += _n
        elif _w <= 1:
            pc_counts["1-n"] += _n
        elif _r <= 1:
            pc_counts["n-1"] += _n
        else:
            pc_counts["n-n"] += _n
    pc_table = kv_table(
        ["producer-consumer", "files", "share"],
        [
            [label, fmt_num(pc_counts[label], 0), f"{fmt_num(100.0 * pc_counts[label] / pc_total)}%"]
            for label in ["1-1", "1-n", "n-1", "n-n"]
        ],
    ) if pc_total else ""

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
        caption_html = (
            "<p class='muted'>"
            + "<br>".join(esc(line) for line in caption.split("\n") if line.strip())
            + "</p>"
        ) if caption else ""
        return f"<article class='card'>{img}<div><h3>{esc(title)}</h3>{caption_html}<nav>{links}</nav></div></article>"

    def viz_card(viz: str, title: str, caption: str = "") -> str:
        links = links_for(viz)
        img_rel = f"{viz}.png" if (output_dir / f"{viz}.png").exists() else None
        return figure_card(title, img_rel, links, caption)

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
    write_read_gap = io_summary.get("write_read_gap_s") or {}
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

    # The reader/writer fan-out figure is a histogram of fan-out k; it draws no
    # per-file size markers, so there is no "(?)" glyph to annotate anymore.
    # (The old caption explained a size marker from a prior per-bar version.)
    reader_fanout_caption = ""
    staleness_title = "Write→Read Gap"

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
    def caption_join(parts: list[str]) -> str:
        return "\n".join(part for part in parts if part)

    lifecycle_caption = caption_join(lifecycle_bits)

    def write_read_gap_caption() -> str:
        return caption_join([
            f"n={fmt_num(write_read_gap.get('n_pairs'), 0)} write→read pairs",
            f"p50={fmt_seconds(write_read_gap.get('p50_s'))}",
            f"p95={fmt_seconds(write_read_gap.get('p95_s'))}",
            f"<1s={fmt_num(write_read_gap.get('pct_lt_1s'))}%",
        ])

    def inter_arrival_caption() -> str:
        ia = p1.get("inter_arrival") or {}
        return caption_join([
            f"n={fmt_num(ia.get('n_intervals'), 0)} re-access intervals",
            f"p50={fmt_seconds(ia.get('p50_s'))}",
            f"p95={fmt_seconds(ia.get('p95_s'))}",
            f"<1s={fmt_num(ia.get('pct_lt_1s'))}%",
        ])

    def directory_caption() -> str:
        ds = p1.get("directory_scan") or {}
        return caption_join([
            f"{fmt_num(ds.get('total_scans'), 0)} scans over {fmt_num(ds.get('unique_directories_scanned'), 0)} workload directories",
            f"{fmt_num(ds.get('rescanned_directories'), 0)} rescanned",
            f"p95 scans/dir={fmt_num(ds.get('p95_scans_per_dir'))}",
        ])

    def file_size_caption() -> str:
        return caption_join([
            f"Requests: <4KB={fmt_num(req.get('pct_lt_4kb'))}%",
            f"<64KB={fmt_num(req.get('pct_lt_64kb'))}%",
            f"bytes-weighted mean request={fmt_bytes(req.get('bytes_weighted_mean_request_bytes'))}",
        ])

    def access_pattern_caption() -> str:
        pct = seq.get("four_cell_pct") or {}
        cov = seq.get("pct_ops_with_offset")
        by_kind = seq.get("ops_with_offset_by_kind") or {}
        tid_streams = seq.get("by_stream_tid_fd_open_generation") or {}
        read_transitions = ((tid_streams.get("read") or {}).get("transitions") or 0)
        write_transitions = ((tid_streams.get("write") or {}).get("transitions") or 0)
        return caption_join([
            "Seq = consecutive/gap or append; Rand = backward/random.",
            f"read={fmt_num(by_kind.get('read'), 0)} ops / {fmt_num(read_transitions, 0)} transitions",
            f"write={fmt_num(by_kind.get('write'), 0)} ops / {fmt_num(write_transitions, 0)} transitions",
            f"Seq R={fmt_num(pct.get('seq_read'))}%",
            f"Rand R={fmt_num(pct.get('rand_read'))}%",
            f"Seq W={fmt_num(pct.get('seq_write'))}%",
            f"Rand W={fmt_num(pct.get('rand_write'))}%",
            f"offset coverage={fmt_num(cov)}%" if cov is not None else "",
        ])

    def io_rate_caption() -> str:
        return (
            "output tokens/s = each LLM call's API-reported output-token count "
            "spread evenly over the call's wall-clock duration."
        )

    def effective_bw_caption() -> str:
        eff = p1.get("effective_bandwidth") or {}
        glob = eff.get("global") or {}
        gread = glob.get("read") or {}
        gwrite = glob.get("write") or {}
        return caption_join([
            f"Global read={fmt_bytes(gread.get('effective_Bps'))}/s",
            f"global write={fmt_bytes(gwrite.get('effective_Bps'))}/s",
            "effective bandwidth = bytes / sum(syscall duration; guarded by min ops and min I/O time).",
        ])

    def measured_interface_caption() -> str:
        stdio = ((measured_iface.get("layers") or {}).get("STDIO") or {})
        pt_ops = stdio.get("process_tree_ops")
        pt_bytes = stdio.get("process_tree_bytes")
        return caption_join([
            (
                f"pathless STDIO process-tree context: {fmt_num(pt_ops, 0)} ops / {fmt_bytes(pt_bytes)}"
                if pt_ops else ""
            ),
        ])

    def io_volume_caption() -> str:
        cov = io_summary.get("coverage_pct") or {}
        return caption_join([
            f"R:W bytes={fmt_num(wl.get('rw_byte_ratio'))}:1" if wl.get("rw_byte_ratio") else "",
            f"coverage read/write={fmt_num(cov.get('read'))}%/{fmt_num(cov.get('write'))}%" if cov else "",
        ])

    # --- Figures grouped by axis ------------------------------------------
    setup_figs = "".join([
        lineage_card("fig0_io_volume_summary.png", "I/O Volume Summary", io_volume_caption()),
        viz_card("agent_timeline", "Agent Timeline"),
        viz_card("phase_breakdown", "Time Accounting"),
        external_card("../call_dag.html", "Call DAG with I/O"),
        # Per-run I/O characterization (paper Fig 2 / Fig 3 analogs), computed
        # from this run's own workload files by
        # agent_io_tracing.analysis.per_run_io_char.
        viz_card("file_access_volume", "File Access Frequency x Volume",
                 "Per-run workload files grouped by access frequency and data volume."),
        viz_card("rw_asymmetry", "Read/Write Asymmetry",
                 "Per-run workload-file classes and per-file read/write byte skew."),
    ])
    ax1_figs = "".join([
        viz_card("directory_scan", "Directory Re-scans (getdents64)", directory_caption()),
        viz_card("inter_arrival_cdf", "Inter-arrival Histogram", inter_arrival_caption()),
        viz_card("reread_attribution", "Reread Attribution"),
        lineage_card(
            "fig2_fanout.png", "Reader & Writer Fan-out",
            reader_fanout_caption,
        ),
        lineage_card("fig3_staleness_cdf.png", staleness_title, write_read_gap_caption()),
    ])
    ax2_figs = viz_card("measured_interface_layers", "Measured I/O Interface Mix",
                        measured_interface_caption())
    ax3_figs = lineage_card("fig4_lifecycle.png", lifecycle_title, lifecycle_caption)
    ax4_figs = "".join([
        lineage_card("fig1_size_distribution.png", "File and Request Size", file_size_caption()),
        viz_card("access_pattern", "Access Pattern", access_pattern_caption()),
    ])
    ax5_figs = "".join([
        viz_card("io_rate", "I/O Rate Over Time", io_rate_caption()),
        viz_card("effective_bandwidth", "Effective BW by Phase", effective_bw_caption()),
        viz_card("io_autocorrelation", "I/O Autocorrelation"),
    ])

    # --- Per-axis table grids ---------------------------------------------
    def panel(title: str, table: str, note: str = "") -> str:
        lines = [seg.strip() for seg in note.replace("\n", " · ").split(" · ") if seg.strip()]
        note_html = (
            '<p class="muted">' + "<br>".join(esc(seg) for seg in lines) + "</p>"
        ) if lines else ""
        return f"<div><h3>{esc(title)}</h3>{note_html}{table}</div>"

    def grid(*cells: str) -> str:
        inner = "".join(c for c in cells if c)
        return f'<div class="attr-grid">{inner}</div>' if inner else ""

    ax1_tables = grid(
        panel("Access type RH/WH/RW", rh_table, rh_note),
        panel("State file rewrite frequency", state_table, state_note),
        panel("Producer-consumer classes", pc_table) if pc_table else "",
    )
    ax2_tables = ""
    ax3_tables = grid(
        panel("Failed open/stat", fos_table, fos_note),
        panel("Error-log reads", "", elr_note),
        panel("Bytes/ops by phase", bop_table),
    )
    ax4_tables = grid(
        panel("I/O Batching Efficiency", amp_table, amp_note),
    )
    ax5_tables = grid(
        panel(
            "Workflow concurrency (opportunity)",
            pardeg_table,
        ),
        panel(
            "I/O concurrency",
            io_pardeg_table,
        ),
    )

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
    <h2>Axis 1 · Context-Limited State Persistence</h2>
    <section class="zone">{ax1_figs}</section>
    <section class="attr-summary">{ax1_tables}</section>
    <h2>Axis 2 · Measured I/O Interface Layer</h2>
    <section class="zone">{ax2_figs}</section>
    <section class="attr-summary">{ax2_tables}</section>
    <h2>Axis 3 · Exploratory Branching and Backtracking</h2>
    <section class="zone">{ax3_figs}</section>
    <section class="attr-summary">{ax3_tables}</section>
    <h2>Axis 4 · Reasoning-Step/Data-Granularity Alignment</h2>
    <section class="zone">{ax4_figs}</section>
    <section class="attr-summary">{ax4_tables}</section>
    <h2>Axis 5 · Uncoordinated Agent Concurrency</h2>
    <section class="zone">{ax5_figs}</section>
    <section class="attr-summary">{ax5_tables}</section>
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

# Visualizations that use StraceData (from parsed.json). The old detail-only
# strace views were retired from the cleaned dashboard.
STRACE_VISUALIZATIONS = {}

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
    "measured_interface_layers": (None, create_measured_interface_layers_matplotlib),
    "inter_arrival_cdf": (None, create_inter_arrival_cdf_matplotlib),
    "reread_attribution": (None, create_reread_attribution_matplotlib),
    "directory_scan": (None, create_directory_scan_matplotlib),
    "io_rate": (None, create_io_rate_matplotlib),
    "effective_bandwidth": (None, create_effective_bandwidth_matplotlib),
    "access_pattern": (None, create_access_pattern_matplotlib),
    "io_autocorrelation": (None, create_io_autocorrelation_matplotlib),
}

# Combined for CLI help and validation
VISUALIZATIONS = {**STRACE_VISUALIZATIONS, **PHASE_VISUALIZATIONS, **AGENT_VISUALIZATIONS}

RETIRED_VISUALIZATION_STEMS = {
    "timeline",
    "tool_syscalls",
    "tool_syscall_durations",
    "agent_concurrency",
    "intensity_phases",
    "interface_mix",
}

EXTRA_VISUALIZATION_STEMS = {
    "index",
    "file_access_volume",
    "rw_asymmetry",
}


def cleanup_retired_visualizations(output_dir: Path) -> None:
    """Remove stale PNG/HTML files for visualizations no longer generated."""
    if not output_dir.exists():
        return
    allowed = set(VISUALIZATIONS) | EXTRA_VISUALIZATION_STEMS
    for path in output_dir.iterdir():
        if path.suffix not in {".png", ".html"}:
            continue
        remove = path.stem in RETIRED_VISUALIZATION_STEMS or path.stem not in allowed
        if not remove and path.stem in VISUALIZATIONS:
            plotly_fn, matplotlib_fn = VISUALIZATIONS[path.stem]
            remove = (path.suffix == ".html" and plotly_fn is None) or (
                path.suffix == ".png" and matplotlib_fn is None
            )
        if not remove:
            continue
        try:
            path.unlink()
        except OSError:
            pass


def generate_visualizations(
    trace_dir: Path,
    only: list[str] | None = None,
    html_only: bool = False,
    png_only: bool = False,
) -> None:
    """Generate all or selected visualizations."""
    
    # Create output directory
    output_dir = trace_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    cleanup_retired_visualizations(output_dir)
    
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
                plotly_fn(strace_data, html_path)
                generated.append(viz_name)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
        
        if not html_only and matplotlib_fn:
            png_path = output_dir / f"{viz_name}.png"
            print(f"Generating {png_path.name}...", file=sys.stderr)
            try:
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
  %(prog)s traces/20260115_190208/terraform-main/ --only agent_timeline,io_rate
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
    )


if __name__ == "__main__":
    main()
