"""Storage / lineage characterization from a single GenoMAS trace cell.

Computes 4 metrics from parsed.json + pi_events.jsonl:
  1. File size distribution (read vs write)
  2. Reader fan-out per artifact
  3. Write→first-read staleness
  4. Lineage depth (number of distinct CodeExec calls touching each artifact)

Outputs to <trace_dir>/lineage/:
  - artifacts.csv              one row per data artifact
  - tool_call_attribution.csv  one row per tool_call run_id
  - fig1_size_distribution.png log-x histogram of read/write sizes
  - fig2_reader_fanout.png     P(fan-out = k) per CodeExec / per role
  - fig3_staleness_cdf.png     CDF of write→first-read latency
  - fig4_lineage_depth.png     P(depth = k)
  - fig5_artifact_lifecycle.png top-N artifacts' read/write timeline (optional)

Usage:
    python3 lineage_analyzer.py <trace_dir>

trace_dir must contain:
    parsed.json
    pi_events.jsonl
"""
from __future__ import annotations
import json
import sys
import csv
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration -------------------------------------------------

# Paths we treat as real workload artifacts. Anything outside is noise
# (Python imports, /proc, /dev, our own logger files, ...).
#
# GenoMAS's LLM-generated code writes via RELATIVE paths ("./output/..."),
# so we accept both absolute and relative forms.  BCC records whatever
# string was passed to openat/write; it does not resolve to absolute.
DATA_PATH_PREFIXES = (
    # Inputs
    "/mnt/lustrefs/genomas_data/",            # GEO + TCGA matrix files
    "/users/Minqiu/GenoMAS/metadata/",        # task_info.json + gene synonyms
    # Outputs (absolute)
    "/users/Minqiu/GenoMAS/output/",
    # Outputs (relative, because GenoMAS runs with cwd=~/GenoMAS)
    "./output/",
    "output/",
    # Scratch
    "/tmp/genomas_work",
)

# Logger / instrumentation paths we DROP even if they sit under data prefixes.
# These are our own measurement artifacts, not GenoMAS's workload.
EXCLUDE_PATH_SUBSTRINGS = (
    "pi-ebpf-tracing-handoff/traces/",   # our trace output (pi_events.jsonl etc.)
    "/output/log_",                       # GenoMAS top-level log file (high frequency)
    "output/log_",                        # relative form of the same
    "/.venv/",                            # Python library imports
)

READ_SYSCALLS = ("read", "pread64", "readv", "preadv")
WRITE_SYSCALLS = ("write", "pwrite64", "writev", "pwritev")

# --- Artifact categories -------------------------------------------
# These are the storage-placement decision classes.  Every figure is
# colored by category so we can read off rules like "raw inputs are the
# high-fan-out files" or "intermediates are write-once read-once".
RAW_INPUT_PREFIXES = (
    "/mnt/lustrefs/genomas_data/",
)
METADATA_PREFIXES = (
    "/users/Minqiu/GenoMAS/metadata/",
)
# Coordination / state files: shared task trackers that many roles
# read+write. Matched by basename anywhere under the output tree.
COORDINATION_BASENAMES = (
    "completed_tasks.json",
)

# Fixed display order + colors so every figure is consistent.
CATEGORY_ORDER = (
    "raw_input",
    "metadata",
    "intermediate",
    "terminal_output",
    "coordination",
    "code",
    "scratch",
)
CATEGORY_COLORS = {
    "raw_input":       "#1f77b4",  # blue
    "metadata":        "#17becf",  # cyan
    "intermediate":    "#ff7f0e",  # orange
    "terminal_output": "#2ca02c",  # green
    "coordination":    "#d62728",  # red
    "code":            "#9467bd",  # purple
    "scratch":         "#8c564b",  # brown
}


def is_workload_artifact(path: str) -> bool:
    if not path:
        return False
    if not any(path.startswith(p) for p in DATA_PATH_PREFIXES):
        return False
    return not any(s in path for s in EXCLUDE_PATH_SUBSTRINGS)


def classify_artifact(path: str, rec: dict) -> str:
    """Assign a storage-placement category to one artifact.

    `rec` is its per-artifact summary (used to split generated files into
    intermediate=written-then-read vs terminal_output=written-never-read).
    """
    base = path.rsplit("/", 1)[-1]
    if any(path.startswith(p) for p in RAW_INPUT_PREFIXES):
        return "raw_input"
    if any(path.startswith(p) for p in METADATA_PREFIXES):
        return "metadata"
    if base.endswith(".tmp"):
        return "scratch"
    if base in COORDINATION_BASENAMES:
        return "coordination"
    if base.endswith(".py") or "/code/" in path:
        return "code"
    # Generated data output: intermediate if a later step read it back,
    # otherwise a terminal (leaf) output.
    if rec["reads"]:
        return "intermediate"
    return "terminal_output"


def artifact_size_bytes(rec: dict) -> int:
    """Best per-FILE size proxy (not per-syscall).

    For generated files, bytes written ≈ on-disk size (writes build the
    file once, even when split across many syscalls). For read-only inputs
    we fall back to bytes read. This is an approximation — re-reads of an
    input can inflate it — but it is per-file, which is what storage sizing
    needs, unlike the per-syscall histogram.
    """
    total_write = sum(s for _, s in rec["writes"])
    if total_write > 0:
        return total_write
    return sum(s for _, s in rec["reads"])


def human_bytes(x: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.0f} {unit}"
        x /= 1024
    return f"{x:.0f} TB"


def parse_iso_ts(s: str) -> float:
    """ISO 8601 → epoch seconds (float)."""
    return datetime.fromisoformat(s).timestamp()


# --- Step 1: load + attribute --------------------------------------

def load_codeexec_index(trace_dir: Path) -> dict:
    """Build {tool_call_run_id: {role, code_len, t_start_ms, t_end_ms}}.

    Source: pi_events.jsonl tool_execution_start / tool_execution_end.
    """
    pending = {}
    index = {}
    with (trace_dir / "pi_events.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            t = e.get("type")
            rid = e.get("run_id")
            if t == "tool_execution_start":
                pending[rid] = {
                    "role": e.get("genomas_role", "?"),
                    "code_len": e.get("code_len", 0),
                    "t_start_ms": e.get("timestamp", 0.0),
                }
            elif t == "tool_execution_end":
                if rid in pending:
                    entry = pending.pop(rid)
                    entry["t_end_ms"] = e.get("timestamp", entry["t_start_ms"])
                    entry["stdout_len"] = e.get("stdout_len", 0)
                    entry["error"] = e.get("error")
                    index[rid] = entry
    return index


def load_data_io_events(trace_dir: Path) -> list[dict]:
    """Filter fs_entries to read/write on workload artifacts with size>0."""
    parsed = json.load((trace_dir / "parsed.json").open())
    out = []
    for e in parsed.get("fs_entries", []):
        syscall = e.get("syscall")
        if syscall not in READ_SYSCALLS + WRITE_SYSCALLS:
            continue
        size = e.get("bytes_transferred", 0) or 0
        if size <= 0:
            continue
        path = e.get("path") or ""
        if not is_workload_artifact(path):
            continue
        out.append({
            "path": path,
            "syscall": syscall,
            "kind": "R" if syscall in READ_SYSCALLS else "W",
            "size": int(size),
            "pid": e.get("pid"),
            "ts": parse_iso_ts(e["timestamp"]),
            "tool_call_id": e.get("matched_tool_call"),
        })
    out.sort(key=lambda x: x["ts"])
    return out


# --- Step 2: per-artifact analysis ---------------------------------

def per_artifact_summary(io_events: list[dict],
                         codeexec_index: dict) -> dict:
    """Return {path: {summary fields}}."""
    by_path = defaultdict(lambda: {
        "reads": [], "writes": [],
        "reader_tool_ids": set(), "writer_tool_ids": set(),
        "reader_roles": Counter(), "writer_roles": Counter(),
        "first_write_ts": None, "first_read_after_write_ts": None,
        "all_tool_ids": set(),
    })
    for e in io_events:
        rec = by_path[e["path"]]
        tid = e["tool_call_id"]
        role = codeexec_index.get(tid, {}).get("role") if tid else None
        if e["kind"] == "R":
            rec["reads"].append((e["ts"], e["size"]))
            if tid:
                rec["reader_tool_ids"].add(tid)
                rec["all_tool_ids"].add(tid)
            if role:
                rec["reader_roles"][role] += 1
            # staleness: only count first read AFTER first write
            if rec["first_write_ts"] is not None and rec["first_read_after_write_ts"] is None:
                if e["ts"] > rec["first_write_ts"]:
                    rec["first_read_after_write_ts"] = e["ts"]
        else:  # write
            rec["writes"].append((e["ts"], e["size"]))
            if tid:
                rec["writer_tool_ids"].add(tid)
                rec["all_tool_ids"].add(tid)
            if role:
                rec["writer_roles"][role] += 1
            if rec["first_write_ts"] is None:
                rec["first_write_ts"] = e["ts"]
    return dict(by_path)


def annotate_categories(per_artifact: dict):
    """Add category + size_bytes to every artifact record (in place)."""
    for path, rec in per_artifact.items():
        rec["category"] = classify_artifact(path, rec)
        rec["size_bytes"] = artifact_size_bytes(rec)


def compute_generations(per_artifact: dict):
    """Add `generation` to every artifact: how many write→read hops it sits
    downstream of a raw input (true lineage depth, not touch count).

    Edge A→B exists when some CodeExec read A and wrote B. generation(B) =
    1 + max(generation of B's parents); a file with no parent (raw input or
    first-generation output) is generation 0.
    """
    # tool_call -> artifacts it read / wrote
    reads_by_tool = defaultdict(set)
    writes_by_tool = defaultdict(set)
    for path, rec in per_artifact.items():
        for t in rec["reader_tool_ids"]:
            reads_by_tool[t].add(path)
        for t in rec["writer_tool_ids"]:
            writes_by_tool[t].add(path)

    parents = defaultdict(set)  # artifact -> set of upstream artifacts
    for t in set(reads_by_tool) | set(writes_by_tool):
        for child in writes_by_tool.get(t, ()):
            for parent in reads_by_tool.get(t, ()):
                if parent != child:
                    parents[child].add(parent)

    memo = {}

    def depth(node, stack):
        if node in memo:
            return memo[node]
        if node in stack:          # cycle guard (e.g. read+write same file)
            return 0
        ps = parents.get(node)
        if not ps:
            memo[node] = 0
            return 0
        stack.add(node)
        d = 1 + max(depth(p, stack) for p in ps)
        stack.discard(node)
        memo[node] = d
        return d

    for path, rec in per_artifact.items():
        rec["generation"] = depth(path, set())


def annotate_lifecycle(per_artifact: dict, run_start: float, run_end: float,
                       dead_frac: float = 0.05):
    """Add lifecycle / cleanup fields to every artifact (in place).

    Cleanup semantics: an artifact is reclaimable once nobody will read it
    again. So the reclaim threshold is its LAST READ (for consumed files) or
    its creation time (for write-once-never-read leaves). Everything between
    that threshold and run end is wasted residency.

    Fields added:
      t_create            first write ts (first read ts for read-only inputs)
      t_last_read         last read ts (None if never read)
      t_reclaimable       point after which the file is dead weight
      dead_seconds        run_end - t_reclaimable
      waste_byte_seconds  size_bytes * dead_seconds (storage-time wasted)
      lifecycle_class     input | ephemeral_leaf | transient | live_to_end
    """
    run_span = max(run_end - run_start, 1e-9)
    for path, rec in per_artifact.items():
        read_ts = [t for t, _ in rec["reads"]]
        write_ts = [t for t, _ in rec["writes"]]
        n_reads, n_writes = len(read_ts), len(write_ts)

        rec["t_create"] = min(write_ts) if write_ts else (min(read_ts) if read_ts else run_start)
        rec["t_last_read"] = max(read_ts) if read_ts else None

        if n_writes == 0:
            # External read-only input: not ours to delete.
            rec["lifecycle_class"] = "input"
            rec["t_reclaimable"] = None
            rec["dead_seconds"] = 0.0
            rec["waste_byte_seconds"] = 0.0
            continue

        if n_reads == 0:
            # Written, never read back -> dead the moment it lands.
            rec["t_reclaimable"] = rec["t_create"]
            rec["lifecycle_class"] = "ephemeral_leaf"
        else:
            rec["t_reclaimable"] = rec["t_last_read"]

        dead = max(run_end - rec["t_reclaimable"], 0.0)
        rec["dead_seconds"] = dead
        rec["waste_byte_seconds"] = rec["size_bytes"] * dead
        if n_reads > 0:
            rec["lifecycle_class"] = (
                "transient" if dead > dead_frac * run_span else "live_to_end"
            )


# --- Step 3: outputs -----------------------------------------------

def write_csvs(per_artifact: dict, codeexec_index: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "artifacts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "path", "category", "size_bytes",
            "n_reads", "n_writes", "total_read_bytes", "total_write_bytes",
            "fanout_codeexec", "fanout_roles",
            "reader_roles", "writer_roles",
            "first_write_ts", "first_read_after_write_ts", "staleness_s",
            "lineage_depth_generations", "n_touching_calls",
            "lifecycle_class", "t_create", "t_last_read",
            "dead_seconds", "waste_byte_seconds",
        ])
        for path, r in per_artifact.items():
            staleness = (
                r["first_read_after_write_ts"] - r["first_write_ts"]
                if r["first_write_ts"] is not None and r["first_read_after_write_ts"] is not None
                else ""
            )
            w.writerow([
                path,
                r.get("category", ""),
                r.get("size_bytes", 0),
                len(r["reads"]),
                len(r["writes"]),
                sum(s for _, s in r["reads"]),
                sum(s for _, s in r["writes"]),
                len(r["reader_tool_ids"]),
                len(set(r["reader_roles"]) | set(r["writer_roles"])),
                ";".join(f"{k}:{v}" for k, v in r["reader_roles"].items()),
                ";".join(f"{k}:{v}" for k, v in r["writer_roles"].items()),
                r["first_write_ts"] or "",
                r["first_read_after_write_ts"] or "",
                staleness,
                r.get("generation", ""),
                len(r["all_tool_ids"]),
                r.get("lifecycle_class", ""),
                r.get("t_create", ""),
                r.get("t_last_read") if r.get("t_last_read") is not None else "",
                round(r.get("dead_seconds", 0.0), 3),
                int(r.get("waste_byte_seconds", 0.0)),
            ])

    with (out_dir / "tool_call_attribution.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool_call_id", "role", "code_len", "stdout_len",
                    "t_start_ms", "t_end_ms", "duration_s", "error"])
        for tid, info in codeexec_index.items():
            duration = (info.get("t_end_ms", 0) - info.get("t_start_ms", 0)) / 1000.0
            w.writerow([tid, info["role"], info["code_len"], info.get("stdout_len", 0),
                        info["t_start_ms"], info.get("t_end_ms", 0),
                        round(duration, 4), info.get("error") or ""])


_SIZE_TICKS = [1, 1024, 1024**2, 10 * 1024**2, 100 * 1024**2, 1024**3]


def _apply_size_xaxis(ax):
    """Log x-axis with human-readable byte ticks (1 B / 1 KB / 1 MB ...)."""
    ax.set_xscale("log")
    ticks = [t for t in _SIZE_TICKS]
    ax.set_xticks(ticks)
    ax.set_xticklabels([human_bytes(t) for t in ticks])


def fig_size_distribution(io_events: list[dict], per_artifact: dict, out_path: Path):
    """Two panels: per-syscall I/O request sizes (top) and per-FILE artifact
    sizes by category (bottom). The first informs block size / read-ahead;
    the second informs capacity and storage tier — which is goal #1."""
    bins = np.logspace(0, np.log10(1024**3), 40)  # 1 B → 1 GB
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    # --- top: per-syscall ---
    reads = [e["size"] for e in io_events if e["kind"] == "R"]
    writes = [e["size"] for e in io_events if e["kind"] == "W"]
    ax1.hist(reads, bins=bins, alpha=0.6, label=f"read (n={len(reads)})", color="#2ca02c")
    ax1.hist(writes, bins=bins, alpha=0.6, label=f"write (n={len(writes)})", color="#d62728")
    _apply_size_xaxis(ax1)
    ax1.set_xlabel("I/O request size (per syscall)")
    ax1.set_ylabel("# syscalls")
    ax1.set_title("Per-syscall I/O request size  (informs block size / read-ahead)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # --- bottom: per-file, stacked by category ---
    by_cat = defaultdict(list)
    for rec in per_artifact.values():
        if rec["size_bytes"] > 0:
            by_cat[rec["category"]].append(rec["size_bytes"])
    cats = [c for c in CATEGORY_ORDER if c in by_cat]
    ax2.hist(
        [by_cat[c] for c in cats],
        bins=bins, stacked=True,
        color=[CATEGORY_COLORS[c] for c in cats],
        label=[f"{c} (n={len(by_cat[c])})" for c in cats],
    )
    _apply_size_xaxis(ax2)
    ax2.set_xlabel("approx file size (bytes written, or bytes read for inputs)")
    ax2.set_ylabel("# artifacts")
    ax2.set_title("Per-file artifact size by category  (informs capacity / tier)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _short(path: str, n: int = 48) -> str:
    return path if len(path) <= n else "…" + path[-(n - 1):]


def _category_legend(ax, cats):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in cats]
    ax.legend(handles=handles, fontsize=8, loc="lower right")


def fig_reader_fanout(per_artifact: dict, out_path: Path):
    """One bar per artifact (named), length = #distinct CodeExec calls that
    read it, colored by category. Answers 'which file has high fan-out, and
    is it raw input or intermediate' — goal #2."""
    items = [(p, r) for p, r in per_artifact.items()
             if r["reads"] and len(r["reader_tool_ids"]) > 0]
    if not items:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No artifacts had any attributed read events.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    items.sort(key=lambda kv: len(kv[1]["reader_tool_ids"]))
    labels = [_short(p) for p, _ in items]
    values = [len(r["reader_tool_ids"]) for _, r in items]
    colors = [CATEGORY_COLORS[r["category"]] for _, r in items]
    cats_present = [c for c in CATEGORY_ORDER
                    if any(r["category"] == c for _, r in items)]

    fig, ax = plt.subplots(figsize=(10, max(4, len(items) * 0.32)))
    y = np.arange(len(items))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7, family="monospace")
    ax.set_xlabel("reader fan-out = # distinct CodeExec calls that read this file")
    ax.set_title("Reader fan-out per artifact (colored by category)")
    ax.grid(axis="x", alpha=0.3)
    _category_legend(ax, cats_present)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_staleness_cdf(per_artifact: dict, out_path: Path):
    """One dot per artifact (named): write→first-read gap, colored by
    category. With only a handful of artifacts a CDF is misleading, so we
    show each file directly — goal #3."""
    items = [
        (p, r, r["first_read_after_write_ts"] - r["first_write_ts"])
        for p, r in per_artifact.items()
        if r["first_write_ts"] is not None and r["first_read_after_write_ts"] is not None
    ]
    fig, ax = plt.subplots(figsize=(10, max(3, len(items) * 0.4)))
    if not items:
        ax.text(0.5, 0.5,
                "No artifact had both a write and a subsequent read in this trace.\n"
                "Likely cause: BCC missed GenoMAS output files written from\n"
                "asyncio executor threads; only logger writes were captured.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#888")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    items.sort(key=lambda t: t[2])
    y = np.arange(len(items))
    # clamp to a small floor so sub-ms gaps still render on a log axis
    xs = [max(s, 1e-3) for _, _, s in items]
    colors = [CATEGORY_COLORS[r["category"]] for _, r, _ in items]
    ax.scatter(xs, y, c=colors, s=80, edgecolor="black", linewidth=0.4, zorder=3)
    ax.hlines(y, 1e-3, xs, color="#cccccc", linewidth=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(p) for p, _, _ in items], fontsize=7, family="monospace")
    ax.set_xscale("log")
    ax.set_xlabel("write → first-read staleness (seconds, log scale)")
    ax.set_title(f"Write→first-read staleness per artifact (n={len(items)})")
    ax.grid(axis="x", alpha=0.3)
    cats_present = [c for c in CATEGORY_ORDER if any(r["category"] == c for _, r, _ in items)]
    _category_legend(ax, cats_present)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_lifecycle_spans(per_artifact: dict, run_start: float, run_end: float,
                        out_path: Path, top_n: int = 25):
    """Per-artifact live/reclaimable window — answers WHEN each file can be
    cleaned up (goal #4, lifecycle framing).

    One row per artifact (top-N by file size, since cleanup payoff scales
    with bytes), x = seconds from run start:
      - solid bar  t_create → t_reclaimable : still needed, colored by category
      - gray hatch t_reclaimable → run_end  : dead weight, reclaimable
      - read-only inputs: solid access span only (we don't delete inputs)
      - write-once leaves (never read): red ▽ at write + full gray bar
    Each row is labeled with the file's size so the selection is explicit."""
    items = [(p, r) for p, r in per_artifact.items() if r["size_bytes"] > 0]
    if not items:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No artifacts in trace.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    items.sort(key=lambda kv: kv[1]["size_bytes"], reverse=True)
    items = items[:top_n]
    items.reverse()  # biggest at top after invert

    t0 = run_start
    end = run_end - t0
    fig, ax = plt.subplots(figsize=(12, max(4, len(items) * 0.34)))
    for y, (path, r) in enumerate(items):
        cat = r["category"]
        color = CATEGORY_COLORS[cat]
        create = r["t_create"] - t0
        if r["lifecycle_class"] == "input":
            # read-only: just the access span, no reclaim tail
            last = (r["t_last_read"] or r["t_create"]) - t0
            ax.barh(y, max(last - create, end * 0.002), left=create, height=0.6,
                    color=color, edgecolor="black", linewidth=0.3, zorder=3)
        elif r["lifecycle_class"] == "ephemeral_leaf":
            # dead on arrival: full gray reclaimable bar + write marker
            ax.barh(y, end - create, left=create, height=0.6,
                    color="#dddddd", hatch="///", edgecolor="#999999",
                    linewidth=0.3, zorder=2)
            ax.scatter(create, y, marker="v", s=55, color="#d62728",
                       edgecolor="black", linewidth=0.4, zorder=4)
        else:
            # transient / live_to_end: needed window + reclaimable tail
            reclaim = r["t_reclaimable"] - t0
            ax.barh(y, max(reclaim - create, end * 0.002), left=create, height=0.6,
                    color=color, edgecolor="black", linewidth=0.3, zorder=3)
            if end - reclaim > 0:
                ax.barh(y, end - reclaim, left=reclaim, height=0.6,
                        color="#dddddd", hatch="///", edgecolor="#999999",
                        linewidth=0.3, zorder=2)

    labels = [f"{human_bytes(r['size_bytes']):>8}  {_short(p, 42)}" for p, r in items]
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=7, family="monospace")
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlim(0, end * 1.02)
    ax.set_xlabel("time from run start (s)   —   solid = needed, gray ▨ = reclaimable, ▽ = write-once leaf")
    ax.set_title(f"Artifact lifecycle / reclaimable window (top {len(items)} by size)")
    ax.grid(axis="x", alpha=0.3)

    cats_present = [c for c in CATEGORY_ORDER if any(r["category"] == c for _, r in items)]
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in cats_present]
    handles.append(Patch(facecolor="#dddddd", hatch="///", edgecolor="#999999",
                         label="reclaimable (dead weight)"))
    ax.legend(handles=handles, fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_artifact_lifecycle(per_artifact: dict, codeexec_index: dict, out_path: Path):
    """Gantt-like: top-N artifacts by (n_reads + n_writes), show every R/W event."""
    ranked = sorted(per_artifact.items(),
                    key=lambda kv: -(len(kv[1]["reads"]) + len(kv[1]["writes"])))[:12]
    if not ranked:
        return

    all_ts = [t for _, r in ranked for t, _ in (r["reads"] + r["writes"])]
    if not all_ts:
        return
    t0 = min(all_ts)
    fig, ax = plt.subplots(figsize=(12, max(4, len(ranked) * 0.45)))

    role_palette = plt.get_cmap("tab10").colors
    role_color = {}
    next_idx = [0]
    def color_for(role):
        if role not in role_color:
            role_color[role] = role_palette[next_idx[0] % len(role_palette)]
            next_idx[0] += 1
        return role_color[role]

    for y, (path, r) in enumerate(ranked):
        for ts, size in r["writes"]:
            ax.scatter(ts - t0, y, marker="^", s=70, color="#d62728",
                       edgecolor="black", linewidth=0.5, zorder=3)
        for ts, size in r["reads"]:
            ax.scatter(ts - t0, y, marker="o", s=40, color="#2ca02c",
                       edgecolor="black", linewidth=0.3, zorder=2)

    short_paths = [p[-60:] for p, _ in ranked]
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(short_paths, fontsize=8, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("time from first artifact event (s)")
    ax.set_title("Top-12 artifact lifecycles (▲ = write, ● = read)")
    ax.grid(axis="x", alpha=0.3)

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#d62728",
               markersize=10, markeredgecolor="black", label="write"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=9, markeredgecolor="black", label="read"),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(per_artifact: dict, io_events: list[dict]):
    print("=" * 64)
    print(f"  Artifacts identified : {len(per_artifact)}")
    print(f"  I/O events on data   : {len(io_events)}")
    if io_events:
        n_attr = sum(1 for e in io_events if e['tool_call_id'])
        print(f"  Attributed to CodeExec: {n_attr} ({100*n_attr/len(io_events):.0f}%)")
    reads = [e for e in io_events if e["kind"] == "R"]
    writes = [e for e in io_events if e["kind"] == "W"]
    if reads:
        print(f"  Read size:  min={min(e['size'] for e in reads)}  "
              f"median={int(np.median([e['size'] for e in reads]))}  "
              f"max={max(e['size'] for e in reads)}")
    if writes:
        print(f"  Write size: min={min(e['size'] for e in writes)}  "
              f"median={int(np.median([e['size'] for e in writes]))}  "
              f"max={max(e['size'] for e in writes)}")
    print()
    print("  Top 5 most-touched artifacts:")
    ranked = sorted(per_artifact.items(),
                    key=lambda kv: -(len(kv[1]["reads"]) + len(kv[1]["writes"])))[:5]
    for path, r in ranked:
        print(f"    R={len(r['reads']):>3} W={len(r['writes']):>3} "
              f"depth={len(r['all_tool_ids']):>2}  ...{path[-60:]}")
    print("=" * 64)


# --- main ----------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("usage: lineage_analyzer.py <trace_dir>", file=sys.stderr)
        sys.exit(2)
    trace_dir = Path(sys.argv[1]).resolve()
    if not (trace_dir / "parsed.json").is_file():
        print(f"ERROR: {trace_dir}/parsed.json not found", file=sys.stderr)
        sys.exit(2)
    if not (trace_dir / "pi_events.jsonl").is_file():
        print(f"ERROR: {trace_dir}/pi_events.jsonl not found", file=sys.stderr)
        sys.exit(2)

    out_dir = trace_dir / "lineage"
    out_dir.mkdir(exist_ok=True)

    codeexec_index = load_codeexec_index(trace_dir)
    io_events = load_data_io_events(trace_dir)
    per_artifact = per_artifact_summary(io_events, codeexec_index)
    annotate_categories(per_artifact)
    compute_generations(per_artifact)
    run_start = min((e["ts"] for e in io_events), default=0.0)
    run_end = max((e["ts"] for e in io_events), default=0.0)
    annotate_lifecycle(per_artifact, run_start, run_end)

    write_csvs(per_artifact, codeexec_index, out_dir)
    fig_size_distribution(io_events, per_artifact, out_dir / "fig1_size_distribution.png")
    fig_reader_fanout    (per_artifact,    out_dir / "fig2_reader_fanout.png")
    fig_staleness_cdf    (per_artifact,    out_dir / "fig3_staleness_cdf.png")
    fig_lifecycle_spans  (per_artifact, run_start, run_end, out_dir / "fig4_lifecycle.png")
    fig_artifact_lifecycle(per_artifact, codeexec_index, out_dir / "fig5_artifact_lifecycle.png")

    # Old generational-depth figure retired; remove stale file if present so
    # the index never shows the deprecated chart.
    old_fig4 = out_dir / "fig4_lineage_depth.png"
    if old_fig4.exists():
        old_fig4.unlink()

    print_summary(per_artifact, io_events)
    print(f"\nOutputs in: {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
