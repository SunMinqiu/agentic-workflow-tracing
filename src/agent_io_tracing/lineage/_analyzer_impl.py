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
  - fig4_lifecycle.png         per-artifact reclaimable window over time
  - fig6_reuse_pattern.png     dead-write % and read-reuse factor
  - fig7_role_io_attribution.png read/write bytes per agent role

Usage:
    python3 lineage_analyzer.py <trace_dir>

trace_dir must contain:
    parsed.json
    pi_events.jsonl
"""
from __future__ import annotations
import json
import os
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
    "metadata/",                              # task_info.json + gene synonyms
    "./metadata/",
    # Outputs
    # Outputs (relative, because GenoMAS runs with cwd=~/GenoMAS)
    "./output/",
    "output/",
    # Scratch
    "/tmp/genomas_work",
)


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return tuple(part for part in (p.strip() for p in raw.split(":")) if part)


DATA_PATH_PREFIXES = _env_tuple("LINEAGE_DATA_PATH_PREFIXES", DATA_PATH_PREFIXES)

# Logger / instrumentation paths we DROP even if they sit under data prefixes.
# These are our own measurement artifacts, not GenoMAS's workload.
EXCLUDE_PATH_SUBSTRINGS = (
    "pi-ebpf-tracing-handoff/results/",  # our trace output (pi_events.jsonl etc.)
    "/.venv/",                            # Python library imports
)
EXCLUDE_PATH_SUBSTRINGS = _env_tuple(
    "LINEAGE_EXCLUDE_PATH_SUBSTRINGS",
    EXCLUDE_PATH_SUBSTRINGS,
)

READ_SYSCALLS = ("read", "pread64", "readv", "preadv")
WRITE_SYSCALLS = ("write", "pwrite64", "writev", "pwritev")
META_SYSCALLS = {
    "stat", "fstat", "lstat", "statx", "fstatat64", "newfstatat",
    "access", "faccessat", "getdents64", "getdents",
    "readlink", "readlinkat", "open", "openat", "close",
    "mkdir", "mkdirat", "rmdir", "unlink", "unlinkat",
    "rename", "renameat", "renameat2", "chmod", "fchmod",
    "chown", "fchown", "truncate", "ftruncate", "fsync",
    "fdatasync", "sync_file_range",
}

# --- Artifact categories -------------------------------------------
# These are the storage-placement decision classes.  Every figure is
# colored by category so we can read off rules like "raw inputs are the
# high-fan-out files" or "intermediates are write-once read-once".
RAW_INPUT_PREFIXES = (
    "/mnt/lustrefs/genomas_data/",
)
RAW_INPUT_PREFIXES = _env_tuple("LINEAGE_RAW_INPUT_PREFIXES", RAW_INPUT_PREFIXES)
METADATA_PREFIXES = (
    "metadata/",
    "./metadata/",
)
METADATA_PREFIXES = _env_tuple("LINEAGE_METADATA_PREFIXES", METADATA_PREFIXES)
# Coordination / state files: shared task trackers that many roles
# read+write. Matched by basename anywhere under the output tree.
COORDINATION_BASENAMES = (
    "completed_tasks.json",
)

# Fixed display order + colors so every figure is consistent.
CATEGORY_ORDER = (
    "dataset",
    "tool_inputs",
    "tool_outputs",
    "agent_memory",
    "summaries",
    "semantic_index",
    "logs",
    "tmp",
    "environment",
)
CATEGORY_COLORS = {
    "dataset":        "#1f77b4",
    "tool_inputs":    "#17becf",
    "tool_outputs":   "#2ca02c",
    "agent_memory":   "#9467bd",
    "summaries":      "#8e44ad",
    "semantic_index": "#2c3e50",
    "logs":           "#d62728",
    "tmp":            "#ff7f0e",
    "environment":    "#8c564b",
}


def _load_manifest_paths(trace_dir: Path) -> tuple[str, ...]:
    manifest = trace_dir / "manifest.json"
    if not manifest.is_file():
        return ()
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return ()
    paths = []
    for key in ("genomas_repo", "data_dir", "work_dir", "output_dir"):
        val = data.get(key)
        if isinstance(val, str) and val:
            paths.append(val.rstrip("/") + "/")
            if key == "genomas_repo":
                paths.extend([
                    val.rstrip("/") + "/metadata/",
                    val.rstrip("/") + "/output/",
                ])
    return tuple(dict.fromkeys(paths))


def configure_paths_from_manifest(trace_dir: Path) -> None:
    """Broaden path filters using manifest.json when available.

    The historic defaults had /users/Minqiu hardcoded. Keep env overrides, but
    otherwise derive absolute GenoMAS/data/work paths from the run manifest so
    another user or CloudLab image does not silently drop every artifact.
    """
    global DATA_PATH_PREFIXES, RAW_INPUT_PREFIXES, METADATA_PREFIXES
    if os.environ.get("LINEAGE_DATA_PATH_PREFIXES"):
        return
    manifest_paths = _load_manifest_paths(trace_dir)
    if not manifest_paths:
        return
    DATA_PATH_PREFIXES = tuple(dict.fromkeys(DATA_PATH_PREFIXES + manifest_paths))
    data_roots = [p for p in manifest_paths if p.endswith("genomas_data/")]
    if data_roots and not os.environ.get("LINEAGE_RAW_INPUT_PREFIXES"):
        RAW_INPUT_PREFIXES = tuple(dict.fromkeys(RAW_INPUT_PREFIXES + tuple(data_roots)))
    meta_roots = [p for p in manifest_paths if p.endswith("/metadata/")]
    if meta_roots and not os.environ.get("LINEAGE_METADATA_PREFIXES"):
        METADATA_PREFIXES = tuple(dict.fromkeys(METADATA_PREFIXES + tuple(meta_roots)))


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
        return "dataset"
    # GEO/TCGA cohort input files, wherever they are physically staged (the
    # symlink views may live under /tmp). Identity beats location, so this MUST
    # come before the "/tmp/ -> tmp" rule below, otherwise the real input data
    # gets mislabeled as a temp file.
    if ("/GEO/" in path and "GSE" in base) or "/TCGA/" in path:
        return "dataset"
    if any(path.startswith(p) for p in METADATA_PREFIXES):
        return "tool_inputs"
    lower = path.lower()
    if "/memory" in lower or "memory" in base.lower() or "snippet" in base.lower():
        return "agent_memory"
    if "summary" in base.lower() or "/summar" in lower:
        return "summaries"
    if any(k in lower for k in ("faiss", "chroma", "qdrant", "index", "sqlite", ".db")):
        return "semantic_index"
    if base.startswith("log_") or base.endswith(".log") or "/logs/" in lower:
        return "logs"
    if base.endswith(".tmp") or "/tmp/" in lower or path.startswith("/tmp/"):
        return "tmp"
    if base in COORDINATION_BASENAMES:
        return "agent_memory"
    if base.endswith(".py") or "/code/" in path:
        return "tool_outputs"
    # Generated data output: intermediate if a later step read it back,
    # otherwise a terminal (leaf) output.
    if rec["reads"]:
        return "tool_inputs"
    return "tool_outputs"


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


def human_bytes1(x: float) -> str:
    """Like human_bytes but keeps one decimal (for headline totals)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


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
                    entry["phase"] = e.get("phase")
                    entry["io_layers"] = e.get("io_layers") or []
                    index[rid] = entry
    return index


def load_parsed_entries(trace_dir: Path) -> list[dict]:
    parsed = json.load((trace_dir / "parsed.json").open())
    return parsed.get("fs_entries", [])


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


CREATE_SYSCALLS = {"openat", "mkdirat"}
RENAME_SYSCALLS = {"rename", "renameat", "renameat2"}
UNLINK_SYSCALLS = {"unlink", "unlinkat", "rmdir"}


def _entry_ts(entry: dict) -> float | None:
    ts = entry.get("timestamp")
    if not isinstance(ts, str):
        return None
    try:
        return parse_iso_ts(ts)
    except Exception:
        return None


def _is_create_entry(entry: dict) -> bool:
    s = entry.get("syscall")
    if s == "mkdirat":
        return (entry.get("return_value") or 0) >= 0
    if s != "openat" or (entry.get("return_value") or -1) < 0:
        return False
    flags = str(entry.get("open_flags") or "")
    return "O_CREAT" in flags


def build_namespace_summary(parsed_entries: list[dict],
                            per_artifact: dict) -> dict:
    files_by_tool = defaultdict(set)
    create_by_tool = Counter()
    rename_by_tool = Counter()
    unlink_by_tool = Counter()
    dir_fanout = Counter()
    tmp_create_ts: dict[str, float] = {}
    tmp_lifetimes: list[float] = []

    for e in parsed_entries:
        path = e.get("path") or ""
        if path and is_workload_artifact(path):
            tid = e.get("matched_tool_call")
            if tid:
                files_by_tool[tid].add(path)
            parent = str(Path(path).parent)
            dir_fanout[parent] += 1

        syscall = e.get("syscall")
        tid = e.get("matched_tool_call")
        if _is_create_entry(e):
            if tid:
                create_by_tool[tid] += 1
            if path and classify_artifact(path, {"reads": [], "writes": [(0, 0)]}) == "tmp":
                ts = _entry_ts(e)
                if ts is not None:
                    tmp_create_ts[path] = ts
        elif syscall in RENAME_SYSCALLS:
            if tid:
                rename_by_tool[tid] += 1
        elif syscall in UNLINK_SYSCALLS:
            if tid:
                unlink_by_tool[tid] += 1
            if path in tmp_create_ts:
                ts = _entry_ts(e)
                if ts is not None and ts >= tmp_create_ts[path]:
                    tmp_lifetimes.append(ts - tmp_create_ts[path])

    files_per_tool = [len(v) for v in files_by_tool.values()]
    fanouts = list(dir_fanout.values())
    total_create = sum(create_by_tool.values())
    total_rename = sum(rename_by_tool.values())
    total_unlink = sum(unlink_by_tool.values())
    churn_den = max(total_create, 1)
    return {
        "files_per_tool_call": {
            "by_tool_call": {k: len(v) for k, v in files_by_tool.items()},
            "mean": float(np.mean(files_per_tool)) if files_per_tool else 0.0,
            "p50": float(np.percentile(files_per_tool, 50)) if files_per_tool else 0.0,
            "p95": float(np.percentile(files_per_tool, 95)) if files_per_tool else 0.0,
            "max": max(files_per_tool) if files_per_tool else 0,
        },
        "directory_fanout": {
            "top": dir_fanout.most_common(20),
            "mean": float(np.mean(fanouts)) if fanouts else 0.0,
            "p95": float(np.percentile(fanouts, 95)) if fanouts else 0.0,
            "max": max(fanouts) if fanouts else 0,
        },
        "churn": {
            "create_count": int(total_create),
            "rename_count": int(total_rename),
            "unlink_count": int(total_unlink),
            "rename_per_create": total_rename / churn_den,
            "unlink_per_create": total_unlink / churn_den,
            "churn_ops_per_create": (total_rename + total_unlink) / churn_den,
            "by_tool_call": {
                tid: {
                    "creates": create_by_tool.get(tid, 0),
                    "renames": rename_by_tool.get(tid, 0),
                    "unlinks": unlink_by_tool.get(tid, 0),
                }
                for tid in sorted(set(create_by_tool) | set(rename_by_tool) | set(unlink_by_tool))
            },
        },
        "tmp_lifetime_s": {
            "count": len(tmp_lifetimes),
            "p50": float(np.percentile(tmp_lifetimes, 50)) if tmp_lifetimes else None,
            "p95": float(np.percentile(tmp_lifetimes, 95)) if tmp_lifetimes else None,
            "max": max(tmp_lifetimes) if tmp_lifetimes else None,
        },
    }


def load_all_captured_io_totals(trace_dir: Path) -> dict:
    """Read/write byte totals over the WHOLE agent process tree (no workload
    filter), straight from parsed.json.

    Used only for the coverage line on fig0 — i.e. "how much of the captured
    read/write is real workload data vs Python imports / .venv / logger".
    NB: this is the agent's process tree, not literally every process on the
    box; the tracer only follows root-pid + descendants.
    """
    parsed = json.load((trace_dir / "parsed.json").open())
    r_bytes = w_bytes = 0
    n_r = n_w = 0
    for e in parsed.get("fs_entries", []):
        s = e.get("syscall")
        b = e.get("bytes_transferred", 0) or 0
        if b <= 0:
            continue
        if s in READ_SYSCALLS:
            r_bytes += b
            n_r += 1
        elif s in WRITE_SYSCALLS:
            w_bytes += b
            n_w += 1
    return {"read_bytes": r_bytes, "write_bytes": w_bytes,
            "n_reads": n_r, "n_writes": n_w}


# --- Reuse: how much IO touches new data vs re-touches old data ----
#
# Two flavors, both characterize an agentic pipeline's data behavior:
#   - dead writes        : produced but NEVER read back  (pure waste)
#   - produced reuse     : an output read by 1 vs many later CodeExec calls
#   - input reuse        : a read-only input read by 1 vs many calls
# read_amplification = total_read_bytes / true_file_size: how many times the
# file's content was re-read (>1 ⇒ a cache would have saved IO).

REUSE_CLASSES = (
    "dead_write",          # written, never read -> waste
    "produced_read_once",  # written, read by exactly one CodeExec call
    "produced_read_many",  # written, read by >=2 CodeExec calls (shared)
    "input_read_once",     # read-only input, one reader call
    "input_read_many",     # read-only input, >=2 reader calls (re-loaded)
)
REUSE_LABELS = {
    "dead_write":         "written, never read (waste)",
    "produced_read_once": "written, read once",
    "produced_read_many": "written, read by many",
    "input_read_once":    "input, read once",
    "input_read_many":    "input, read by many",
}
REUSE_COLORS = {
    "dead_write":         "#d62728",  # red  = waste
    "produced_read_once": "#ff7f0e",  # orange
    "produced_read_many": "#2ca02c",  # green = good reuse
    "input_read_once":    "#9edae5",  # light cyan
    "input_read_many":    "#1f77b4",  # blue
}


# Real dataset roots, used to recover the true size of GEO/TCGA input files
# whose traced path is a symlink view (e.g. /tmp/genomas_fanout_views/...) that
# no longer exists at analysis time. Override with LINEAGE_DATASET_ROOTS.
DATASET_ROOTS = _env_tuple("LINEAGE_DATASET_ROOTS", ("/mnt/lustrefs/genomas_data",))


def _dataset_real_size(path: str):
    """True on-disk size of a GEO/TCGA file by re-rooting its /GEO|/TCGA suffix
    under a real dataset root (the staged symlink view may be gone). None if not
    found."""
    for marker in ("/GEO/", "/TCGA/"):
        if marker in path:
            suffix = marker + path.split(marker, 1)[1]
            for root in DATASET_ROOTS:
                try:
                    return os.path.getsize(root.rstrip("/") + suffix)
                except OSError:
                    continue
    return None


def load_true_sizes(trace_dir: Path, per_artifact: dict):
    """Set rec['true_size'] and rec['size_source'] on every artifact.

    Priority: artifact_sizes.json sidecar (path->bytes, produced by statting
    the files on the node) > local os.stat > re-rooted dataset stat >
    generated-file write bytes > None. Generated files (n_writes>0) always have
    a reliable size = bytes written, so they never need a stat; only read-only
    inputs do.
    """
    sidecar = {}
    sc_path = trace_dir / "artifact_sizes.json"
    if sc_path.is_file():
        try:
            sidecar = json.load(sc_path.open())
        except Exception:
            sidecar = {}

    for path, rec in per_artifact.items():
        n_writes = len(rec["writes"])
        if path in sidecar and sidecar[path] is not None:
            rec["true_size"] = int(sidecar[path])
            rec["size_source"] = "stat"
        elif os.path.exists(path):
            rec["true_size"] = os.path.getsize(path)
            rec["size_source"] = "stat"
        elif n_writes == 0 and _dataset_real_size(path) is not None:
            # read-only GEO/TCGA input whose symlink-view path is gone: stat the
            # real file under the dataset root instead.
            rec["true_size"] = _dataset_real_size(path)
            rec["size_source"] = "stat-dataroot"
        elif n_writes > 0:
            # generated file: bytes written == on-disk size
            rec["true_size"] = sum(s for _, s in rec["writes"])
            rec["size_source"] = "write_bytes"
        else:
            # read-only input we couldn't stat: size unknown
            rec["true_size"] = None
            rec["size_source"] = "unknown"

    # Persist freshly-statted input sizes to a portable sidecar so later
    # re-runs anywhere (e.g. pulled back to a laptop with no Lustre mount)
    # reuse the true size instead of falling back to inflated read bytes.
    statted = {p: r["true_size"] for p, r in per_artifact.items()
               if r.get("size_source") in ("stat", "stat-dataroot")
               and r.get("true_size") is not None}
    if statted:
        merged = dict(sidecar)
        merged.update(statted)
        try:
            sc_path.write_text(json.dumps(merged, indent=0, sort_keys=True))
        except OSError:
            pass


def classify_reuse(rec: dict) -> str:
    n_reads = len(rec["reads"])
    n_writes = len(rec["writes"])
    n_reader_calls = len(rec["reader_tool_ids"])
    if n_writes == 0:
        return "input_read_many" if n_reader_calls >= 2 else "input_read_once"
    if n_reads == 0:
        return "dead_write"
    return "produced_read_many" if n_reader_calls >= 2 else "produced_read_once"


def annotate_reuse(per_artifact: dict):
    """Add reuse_class + read_amplification to every artifact (in place)."""
    for rec in per_artifact.values():
        rec["reuse_class"] = classify_reuse(rec)
        total_read = sum(s for _, s in rec["reads"])
        ts = rec.get("true_size")
        rec["read_amplification"] = (
            (total_read / ts) if (ts and ts > 0 and total_read > 0) else None
        )


def build_reuse_summary(per_artifact: dict) -> dict:
    """Aggregate reuse: per-class file count + bytes, dead-write waste, and a
    working-set reuse factor (total read bytes / unique read bytes)."""
    by_class_files = Counter()
    by_class_rbytes = Counter()
    by_class_wbytes = Counter()
    total_write = unique_read_bytes = total_read_bytes = 0
    dead_write_bytes = 0
    for rec in per_artifact.values():
        c = rec["reuse_class"]
        rb = sum(s for _, s in rec["reads"])
        wb = sum(s for _, s in rec["writes"])
        by_class_files[c] += 1
        by_class_rbytes[c] += rb
        by_class_wbytes[c] += wb
        total_write += wb
        total_read_bytes += rb
        if c == "dead_write":
            dead_write_bytes += wb
        if rb > 0 and rec.get("true_size"):
            unique_read_bytes += rec["true_size"]

    return {
        "by_class": {
            c: {
                "files": by_class_files.get(c, 0),
                "read_bytes": by_class_rbytes.get(c, 0),
                "write_bytes": by_class_wbytes.get(c, 0),
            }
            for c in REUSE_CLASSES
        },
        "dead_write_bytes": dead_write_bytes,
        "dead_write_pct_of_write": (
            100 * dead_write_bytes / total_write if total_write else None
        ),
        "unique_read_bytes": unique_read_bytes,
        "total_read_bytes": total_read_bytes,
        "read_reuse_factor": (
            total_read_bytes / unique_read_bytes if unique_read_bytes else None
        ),
        "n_inputs_size_unknown": sum(
            1 for r in per_artifact.values()
            if r.get("size_source") == "unknown" and r["reads"]
        ),
    }


def _median_or_none(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.median(vals)) if vals else None


def build_artifact_size_summary(per_artifact: dict) -> dict:
    unknown = [
        path for path, rec in per_artifact.items()
        if rec.get("size_source") == "unknown"
    ]
    unknown_read_only = [
        path for path, rec in per_artifact.items()
        if rec.get("size_source") == "unknown" and rec.get("reads") and not rec.get("writes")
    ]
    return {
        "has_unknown_size": bool(unknown),
        "unknown_size_files": len(unknown),
        "unknown_read_only_files": len(unknown_read_only),
    }


def build_staleness_summary(per_artifact: dict) -> dict:
    vals = []
    for rec in per_artifact.values():
        first_write = rec.get("first_write_ts")
        first_read_after = rec.get("first_read_after_write_ts")
        if first_write is None or first_read_after is None:
            continue
        vals.append(max(float(first_read_after) - float(first_write), 0.0))
    return {
        "n": len(vals),
        "median_s": _median_or_none(vals),
    }


def build_lifecycle_summary(per_artifact: dict) -> dict:
    by_class = Counter(rec.get("lifecycle_class", "unknown") for rec in per_artifact.values())
    generated = [rec for rec in per_artifact.values() if rec.get("writes")]
    write_once_leaf = [rec for rec in generated if rec.get("lifecycle_class") == "ephemeral_leaf"]
    reclaimable = [
        rec for rec in generated
        if (rec.get("dead_seconds") or 0) > 0
    ]
    generated_dead = [float(rec.get("dead_seconds") or 0.0) for rec in generated]
    reclaimable_dead = [float(rec.get("dead_seconds") or 0.0) for rec in reclaimable]
    return {
        "by_class": dict(by_class),
        "generated_files": len(generated),
        "write_once_leaf_files": len(write_once_leaf),
        "write_once_leaf_pct": (
            100.0 * len(write_once_leaf) / len(generated) if generated else None
        ),
        "reclaimable_files": len(reclaimable),
        "reclaimable_pct": (
            100.0 * len(reclaimable) / len(generated) if generated else None
        ),
        "generated_median_dead_s": _median_or_none(generated_dead),
        "reclaimable_median_dead_s": _median_or_none(reclaimable_dead),
    }


# --- Who does the IO: per-agent-role attribution -------------------
#
# The agentic-specific cut: which agent role actually read/wrote the bytes.
# Every io_event carries a tool_call_id; codeexec_index maps it to the leaf
# CodeExec agent's role. Orchestration / logger IO is already filtered out
# upstream, so this is "which agent's code touched the data".

def build_role_io_attribution(io_events: list[dict], codeexec_index: dict,
                              per_artifact: dict) -> dict:
    roles = defaultdict(lambda: {
        "read_bytes": 0, "write_bytes": 0, "n_reads": 0, "n_writes": 0,
        "read_by_category": defaultdict(int),
        "write_by_category": defaultdict(int),
    })
    tot_r = tot_w = 0
    for e in io_events:
        tid = e["tool_call_id"]
        role = (codeexec_index.get(tid, {}).get("role") if tid else None) or "(unattributed)"
        cat = per_artifact.get(e["path"], {}).get("category", "scratch")
        d = roles[role]
        if e["kind"] == "R":
            d["read_bytes"] += e["size"]
            d["n_reads"] += 1
            d["read_by_category"][cat] += e["size"]
            tot_r += e["size"]
        else:
            d["write_bytes"] += e["size"]
            d["n_writes"] += 1
            d["write_by_category"][cat] += e["size"]
            tot_w += e["size"]

    by_role = {}
    for role, d in roles.items():
        by_role[role] = {
            "read_bytes": d["read_bytes"],
            "write_bytes": d["write_bytes"],
            "n_reads": d["n_reads"],
            "n_writes": d["n_writes"],
            "read_pct": (100 * d["read_bytes"] / tot_r) if tot_r else None,
            "write_pct": (100 * d["write_bytes"] / tot_w) if tot_w else None,
            "read_by_category": dict(d["read_by_category"]),
            "write_by_category": dict(d["write_by_category"]),
        }
    top_reader = max(by_role, key=lambda r: by_role[r]["read_bytes"]) if by_role else None
    top_writer = max(by_role, key=lambda r: by_role[r]["write_bytes"]) if by_role else None
    return {
        "by_role": by_role,
        "totals": {"read_bytes": tot_r, "write_bytes": tot_w},
        "top_reader": top_reader,
        "top_writer": top_writer,
    }


def build_io_volume_summary(io_events: list[dict], per_artifact: dict,
                            all_captured: dict) -> dict:
    """Headline IO volume: total read/write bytes, R:W ratio, per-category
    bytes, and coverage vs all-captured. This is the machine-readable twin of
    fig0 and answers the first characterization question: 'how much, and how
    is read vs write balanced'."""
    wl_reads = [e for e in io_events if e["kind"] == "R"]
    wl_writes = [e for e in io_events if e["kind"] == "W"]
    wl_r_bytes = sum(e["size"] for e in wl_reads)
    wl_w_bytes = sum(e["size"] for e in wl_writes)

    by_cat = defaultdict(lambda: {"read_bytes": 0, "write_bytes": 0})
    for rec in per_artifact.values():
        c = rec["category"]
        by_cat[c]["read_bytes"] += sum(s for _, s in rec["reads"])
        by_cat[c]["write_bytes"] += sum(s for _, s in rec["writes"])

    ac_r = all_captured["read_bytes"]
    ac_w = all_captured["write_bytes"]
    return {
        "workload": {
            "read_bytes": wl_r_bytes,
            "write_bytes": wl_w_bytes,
            "n_reads": len(wl_reads),
            "n_writes": len(wl_writes),
            "rw_byte_ratio": (wl_r_bytes / wl_w_bytes) if wl_w_bytes else None,
            "mean_read_bytes": (wl_r_bytes / len(wl_reads)) if wl_reads else 0,
            "mean_write_bytes": (wl_w_bytes / len(wl_writes)) if wl_writes else 0,
            "distinct_files": len(per_artifact),
            "bytes_by_category": {c: dict(v) for c, v in by_cat.items()},
        },
        "all_captured_process_tree": all_captured,
        "coverage_pct": {
            "read": (100 * wl_r_bytes / ac_r) if ac_r else None,
            "write": (100 * wl_w_bytes / ac_w) if ac_w else None,
        },
        "reuse": build_reuse_summary(per_artifact),
    }


def write_io_summary_json(summary: dict, out_dir: Path):
    with (out_dir / "io_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


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

def build_tool_call_io_stats(io_events: list[dict],
                             parsed_entries: list[dict]) -> dict[str, dict[str, int]]:
    stats = defaultdict(lambda: {"read_bytes": 0, "write_bytes": 0, "meta_ops": 0})
    for e in io_events:
        tid = e.get("tool_call_id")
        if not tid:
            continue
        if e.get("kind") == "R":
            stats[tid]["read_bytes"] += int(e.get("size") or 0)
        elif e.get("kind") == "W":
            stats[tid]["write_bytes"] += int(e.get("size") or 0)

    for e in parsed_entries:
        tid = e.get("matched_tool_call")
        if not tid:
            continue
        syscall = e.get("syscall")
        if syscall not in META_SYSCALLS:
            continue
        path = e.get("path") or ""
        if path and not is_workload_artifact(path):
            continue
        stats[tid]["meta_ops"] += 1
    return {tid: dict(vals) for tid, vals in stats.items()}


def write_csvs(per_artifact: dict, codeexec_index: dict, out_dir: Path,
               io_events: list[dict] | None = None,
               parsed_entries: list[dict] | None = None):
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
            "reuse_class", "true_size", "size_source", "read_amplification",
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
                r.get("reuse_class", ""),
                r.get("true_size") if r.get("true_size") is not None else "",
                r.get("size_source", ""),
                round(r["read_amplification"], 3) if r.get("read_amplification") is not None else "",
            ])

    tool_io_stats = build_tool_call_io_stats(io_events or [], parsed_entries or [])
    with (out_dir / "tool_call_attribution.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool_call_id", "role", "code_len", "stdout_len",
                    "t_start_ms", "t_end_ms", "duration_s", "error",
                    "read_bytes", "write_bytes", "meta_ops"])
        for tid, info in codeexec_index.items():
            duration = (info.get("t_end_ms", 0) - info.get("t_start_ms", 0)) / 1000.0
            io = tool_io_stats.get(tid, {})
            w.writerow([tid, info["role"], info["code_len"], info.get("stdout_len", 0),
                        info["t_start_ms"], info.get("t_end_ms", 0),
                        round(duration, 4), info.get("error") or "",
                        io.get("read_bytes", 0), io.get("write_bytes", 0),
                        io.get("meta_ops", 0)])


def fig_io_volume_summary(summary: dict, out_path: Path):
    """fig0 — the headline. Big total READ / WRITE numbers + R:W ratio on top,
    a per-category stacked byte bar (read & write rows) below, and an explicit
    coverage / scope caption so the totals can't be misread."""
    wl = summary["workload"]
    cov = summary["coverage_pct"]
    rw = wl["rw_byte_ratio"]
    rw_str = f"{rw:.1f} : 1" if rw else "n/a"

    # one-line shape read-off: contrast read vs write granularity
    mr, mw = wl["mean_read_bytes"], wl["mean_write_bytes"]
    if mr > 0 and mw > 0:
        if mr >= 4 * mw:
            shape = "reads few-and-large, writes many-and-small"
        elif mw >= 4 * mr:
            shape = "writes few-and-large, reads many-and-small"
        else:
            shape = "read and write request sizes comparable"
    else:
        shape = ""

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.5)

    # --- top: headline text ---
    axt = fig.add_subplot(gs[0])
    axt.set_axis_off()
    axt.text(0.0, 1.0, "IO Volume — workload data artifacts",
             transform=axt.transAxes, fontsize=15, fontweight="bold",
             va="top")
    big = (
        f"READ   {human_bytes1(wl['read_bytes']):>10}   ({wl['n_reads']:,} syscalls)\n"
        f"WRITE  {human_bytes1(wl['write_bytes']):>10}   ({wl['n_writes']:,} syscalls)"
    )
    axt.text(0.0, 0.74, big, transform=axt.transAxes, fontsize=16,
             family="monospace", va="top")
    sub = f"R:W bytes = {rw_str}"
    if shape:
        sub += f"   ·   {shape}"
    sub += f"\ndistinct files: {wl['distinct_files']}"
    axt.text(0.0, 0.30, sub, transform=axt.transAxes, fontsize=11,
             family="monospace", va="top", color="#333333")

    cov_r = f"{cov['read']:.1f}%" if cov["read"] is not None else "n/a"
    cov_w = f"{cov['write']:.1f}%" if cov["write"] is not None else "n/a"
    caption = (
        f"Scope: workload data only (read/write families, size>0; excludes "
        f"Python imports / .venv / logger / trace output).\n"
        f"Covers {cov_r} of read / {cov_w} of write bytes of the whole agent "
        f"process tree (remainder = interpreter imports)."
    )
    axt.text(0.0, -0.02, caption, transform=axt.transAxes, fontsize=8.5,
             va="top", color="#888888")

    # --- bottom: per-category stacked bytes, READ row + WRITE row ---
    axb = fig.add_subplot(gs[1])
    cats = [c for c in CATEGORY_ORDER if c in wl["bytes_by_category"]]
    rows = [("WRITE", "write_bytes"), ("READ", "read_bytes")]  # READ on top
    for i, (_, key) in enumerate(rows):
        left = 0.0
        for c in cats:
            v = wl["bytes_by_category"][c][key]
            if v <= 0:
                continue
            axb.barh(i, v, left=left, color=CATEGORY_COLORS[c],
                     edgecolor="black", linewidth=0.3)
            left += v
    axb.set_yticks(range(len(rows)))
    axb.set_yticklabels([r[0] for r in rows], fontweight="bold")
    axb.set_xlabel("bytes (by storage-placement category)")
    axb.set_title("Where the bytes live")
    axb.grid(axis="x", alpha=0.3)
    cats_present = [c for c in cats
                    if any(wl["bytes_by_category"][c][k] > 0
                           for k in ("read_bytes", "write_bytes"))]
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in cats_present]
    axb.legend(handles=handles, fontsize=8, loc="lower right", ncol=2)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_reuse_pattern(summary: dict, out_path: Path):
    """fig6 — reuse pattern. Two stacked bars (by file count, by bytes) split
    into the reuse classes, with the headline waste number in the title:
    'X% of written bytes are never read back'."""
    reuse = summary["reuse"]
    bc = reuse["by_class"]
    cats = [c for c in REUSE_CLASSES
            if bc[c]["files"] > 0 or bc[c]["read_bytes"] > 0 or bc[c]["write_bytes"] > 0]

    # row 1 = file count, row 2 = bytes (read+write touched)
    counts = {c: bc[c]["files"] for c in cats}
    bytes_touched = {c: bc[c]["read_bytes"] + bc[c]["write_bytes"] for c in cats}
    tot_files = sum(counts.values()) or 1
    tot_bytes = sum(bytes_touched.values()) or 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.2))

    def stacked(ax, values, total, unit):
        left = 0.0
        for c in cats:
            v = values[c]
            if v <= 0:
                continue
            ax.barh(0, v, left=left, color=REUSE_COLORS[c],
                    edgecolor="black", linewidth=0.3)
            if v / total > 0.06:
                ax.text(left + v / 2, 0, f"{100*v/total:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
            left += v
        ax.set_xlim(0, total)
        ax.set_yticks([])
        ax.set_xlabel(unit)

    stacked(ax1, counts, tot_files, f"# files (n={tot_files})")
    stacked(ax2, bytes_touched, tot_bytes, "bytes touched (read+write)")

    dead_pct = reuse["dead_write_pct_of_write"]
    title = "Reuse pattern"
    if dead_pct is not None:
        title += f"  —  {dead_pct:.0f}% of written bytes are NEVER read back (waste)"
    rf = reuse["read_reuse_factor"]
    sub = ""
    if rf is not None:
        sub = (f"\nread reuse factor = {rf:.1f}× "
               f"(total read bytes / unique bytes; >1 ⇒ re-reads a cache could save)")
    if reuse["n_inputs_size_unknown"]:
        sub += (f"\n⚠ {reuse['n_inputs_size_unknown']} input file(s) had no true size "
                f"(not statted) — their reuse is shown by read COUNT only, not amplification")
    ax1.set_title(title + sub, fontsize=10)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=REUSE_COLORS[c], label=REUSE_LABELS[c]) for c in cats]
    ax2.legend(handles=handles, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, -0.35), ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_role_io_attribution(role_attr: dict, out_path: Path):
    """fig7 — who does the IO. Two panels (READ | WRITE), one horizontal bar
    per agent role, each segmented by file category. Title names the dominant
    reader and writer so the producer/consumer split is read off instantly."""
    by = role_attr["by_role"]
    roles = sorted(by, key=lambda r: -(by[r]["read_bytes"] + by[r]["write_bytes"]))
    if not roles:
        return
    y = np.arange(len(roles))
    fig, (axr, axw) = plt.subplots(1, 2, figsize=(13, max(2.5, len(roles) * 0.7)),
                                   sharey=True)

    def draw(ax, key):
        for i, role in enumerate(roles):
            left = 0.0
            for c in CATEGORY_ORDER:
                v = by[role][key].get(c, 0)
                if v <= 0:
                    continue
                ax.barh(i, v, left=left, color=CATEGORY_COLORS[c],
                        edgecolor="black", linewidth=0.3)
                left += v
        ax.set_yticks(y)
        ax.set_yticklabels(roles, fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    draw(axr, "read_by_category")
    axr.set_xlabel("read bytes")
    axr.set_title("READ by agent")
    draw(axw, "write_by_category")
    axw.set_xlabel("write bytes")
    axw.set_title("WRITE by agent")
    axr.invert_yaxis()

    tr, tw = role_attr["top_reader"], role_attr["top_writer"]
    trp = by[tr]["read_pct"] if tr else None
    twp = by[tw]["write_pct"] if tw else None
    head = "Who does the IO"
    if tr and trp is not None:
        head += f"   —   reads: {tr} {trp:.0f}%"
    if tw and twp is not None:
        head += f"   ·   writes: {tw} {twp:.0f}%"
    fig.suptitle(head, fontsize=13, fontweight="bold")

    cats = [c for c in CATEGORY_ORDER
            if any(by[r]["read_by_category"].get(c, 0)
                   + by[r]["write_by_category"].get(c, 0) > 0 for r in roles)]
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in cats]
    axw.legend(handles=handles, fontsize=8, loc="lower right", title="file category")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    n_r, _, _ = ax1.hist(reads, bins=bins, alpha=0.6, label=f"read (n={len(reads)})", color="#2ca02c")
    n_w, _, _ = ax1.hist(writes, bins=bins, alpha=0.6, label=f"write (n={len(writes)})", color="#d62728")

    _ymax = max([*n_r, *n_w, 1])

    def _annotate_peak(counts, color, label):
        """Label the tallest bar with its size bucket (x) and syscall count (y).
        Tall bars (near the axis top) get a side label so it never collides
        with the title."""
        if len(counts) == 0 or max(counts) == 0:
            return
        i = int(np.argmax(counts))
        cnt = int(counts[i])
        lo, hi = bins[i], bins[i + 1]
        xc = (lo * hi) ** 0.5  # geometric center (log x-axis)
        txt = f"{label} peak: {human_bytes(lo)}–{human_bytes(hi)}, n={cnt}"
        if cnt > 0.75 * _ymax:               # near the top → label to the side
            off, ha, va = (10, -4), "left", "top"
        else:                                # room above → label above the bar
            off, ha, va = (0, 10), "center", "bottom"
        ax1.annotate(
            txt, xy=(xc, cnt), xytext=off, textcoords="offset points",
            ha=ha, va=va, fontsize=8, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.6, alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    _annotate_peak(n_r, "#2ca02c", "read")
    _annotate_peak(n_w, "#d62728", "write")
    ax1.set_ylim(top=_ymax * 1.15)
    _apply_size_xaxis(ax1)
    ax1.set_xlabel("I/O request size (per syscall)")
    ax1.set_ylabel("# syscalls")
    ax1.set_title(
        f"Per-syscall I/O request size  (informs block size / read-ahead)\n"
        f"total: READ {human_bytes1(sum(reads))} / WRITE {human_bytes1(sum(writes))}"
        f"  ·  workload data only",
        fontsize=10,
    )
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # --- bottom: per-file, stacked by category ---
    # Use the true on-disk size when known (set by load_true_sizes): for
    # read-only inputs size_bytes is total READ bytes, which re-reads inflate
    # far past the real file size. true_size = stat size for inputs, write
    # bytes for generated files; fall back to size_bytes only when unknown.
    by_cat = defaultdict(list)
    n_skipped = 0
    for rec in per_artifact.values():
        size = rec.get("true_size")
        if size is None:
            # Read-only input we couldn't stat. Its size_bytes is inflated
            # read bytes, so plotting it would mislead — skip instead.
            n_skipped += 1
            continue
        if size > 0:
            by_cat[rec["category"]].append(size)
    cats = [c for c in CATEGORY_ORDER if c in by_cat]
    if cats:
        ax2.hist(
            [by_cat[c] for c in cats],
            bins=bins, stacked=True,
            color=[CATEGORY_COLORS[c] for c in cats],
            label=[f"{c} (n={len(by_cat[c])})" for c in cats],
        )
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No workload artifacts with size > 0",
                 transform=ax2.transAxes, ha="center", va="center",
                 color="#7f8c8d")
    _apply_size_xaxis(ax2)
    ax2.set_xlabel("per-file TRUE on-disk size (stat for inputs, write bytes for generated) — NOT total read/write")
    ax2.set_ylabel("# artifacts")
    _skip_note = (f"  ·  {n_skipped} input(s) skipped (size un-statted)"
                  if n_skipped else "")
    ax2.set_title("Per-file artifact size by category  (informs capacity / tier)\n"
                  "true file size — for I/O volume (with re-reads) see fig0 / io_summary.json"
                  + _skip_note,
                  fontsize=10)
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
    labels = [
        f"{_short(p)} ({human_bytes(r['true_size']) if r.get('true_size') is not None else '?'})"
        for p, r in items
    ]
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
                "This is expected for runs that only read inputs and write final\n"
                "leaf outputs such as reports, images, checkpoints, or logs.\n\n"
                "To populate this metric, the workload must write an intermediate\n"
                "artifact and later read that same artifact within the same trace.",
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


def print_summary(per_artifact: dict, io_events: list[dict], io_summary: dict | None = None):
    print("=" * 64)
    if io_summary is not None:
        wl = io_summary["workload"]
        cov = io_summary["coverage_pct"]
        rw = wl["rw_byte_ratio"]
        rw_str = f"{rw:.1f}:1" if rw else "n/a"
        print(f"  IO VOLUME (workload data):")
        print(f"    READ  {human_bytes1(wl['read_bytes']):>10}  ({wl['n_reads']:,} syscalls)")
        print(f"    WRITE {human_bytes1(wl['write_bytes']):>10}  ({wl['n_writes']:,} syscalls)")
        print(f"    R:W bytes = {rw_str}")
        if cov["read"] is not None:
            print(f"    coverage: {cov['read']:.1f}% read / {cov['write']:.1f}% write of process-tree bytes")
        ru = io_summary.get("reuse", {})
        if ru.get("dead_write_pct_of_write") is not None:
            print(f"  REUSE:")
            print(f"    dead writes (written, never read): {human_bytes1(ru['dead_write_bytes'])} "
                  f"= {ru['dead_write_pct_of_write']:.0f}% of written bytes")
            if ru.get("read_reuse_factor") is not None:
                print(f"    read reuse factor: {ru['read_reuse_factor']:.1f}x (total read / unique bytes)")
            if ru.get("n_inputs_size_unknown"):
                print(f"    note: {ru['n_inputs_size_unknown']} input(s) un-statted -> count-based reuse only")
        br = io_summary.get("by_role", {})
        if br.get("by_role"):
            tr, tw = br.get("top_reader"), br.get("top_writer")
            print(f"  WHO:")
            if tr:
                print(f"    top reader: {tr} ({br['by_role'][tr]['read_pct']:.0f}% of read bytes)")
            if tw:
                print(f"    top writer: {tw} ({br['by_role'][tw]['write_pct']:.0f}% of write bytes)")
        print("-" * 64)
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

    configure_paths_from_manifest(trace_dir)
    codeexec_index = load_codeexec_index(trace_dir)
    parsed_entries = load_parsed_entries(trace_dir)
    io_events = load_data_io_events(trace_dir)
    per_artifact = per_artifact_summary(io_events, codeexec_index)
    annotate_categories(per_artifact)
    compute_generations(per_artifact)
    run_start = min((e["ts"] for e in io_events), default=0.0)
    run_end = max((e["ts"] for e in io_events), default=0.0)
    annotate_lifecycle(per_artifact, run_start, run_end)
    load_true_sizes(trace_dir, per_artifact)
    annotate_reuse(per_artifact)

    all_captured = load_all_captured_io_totals(trace_dir)
    io_summary = build_io_volume_summary(io_events, per_artifact, all_captured)
    io_summary["by_role"] = build_role_io_attribution(io_events, codeexec_index, per_artifact)
    io_summary["namespace"] = build_namespace_summary(parsed_entries, per_artifact)
    io_summary["artifact_sizes"] = build_artifact_size_summary(per_artifact)
    io_summary["staleness"] = build_staleness_summary(per_artifact)
    io_summary["lifecycle"] = build_lifecycle_summary(per_artifact)

    write_csvs(per_artifact, codeexec_index, out_dir, io_events, parsed_entries)
    write_io_summary_json(io_summary, out_dir)
    fig_io_volume_summary(io_summary, out_dir / "fig0_io_volume_summary.png")
    fig_reuse_pattern(io_summary, out_dir / "fig6_reuse_pattern.png")
    fig_role_io_attribution(io_summary["by_role"], out_dir / "fig7_role_io_attribution.png")
    fig_size_distribution(io_events, per_artifact, out_dir / "fig1_size_distribution.png")
    fig_reader_fanout    (per_artifact,    out_dir / "fig2_reader_fanout.png")
    fig_staleness_cdf    (per_artifact,    out_dir / "fig3_staleness_cdf.png")
    fig_lifecycle_spans  (per_artifact, run_start, run_end, out_dir / "fig4_lifecycle.png")

    # Retired figures: remove stale files if present so the index never shows a
    # deprecated chart. (fig4_lineage_depth = old generational-depth figure;
    # fig5_artifact_lifecycle = top-N lifecycle, dropped in favour of fig4.)
    for stale in ("fig4_lineage_depth.png", "fig5_artifact_lifecycle.png"):
        p = out_dir / stale
        if p.exists():
            p.unlink()

    print_summary(per_artifact, io_events, io_summary)
    print(f"\nOutputs in: {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
