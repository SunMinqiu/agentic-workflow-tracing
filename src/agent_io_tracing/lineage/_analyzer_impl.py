"""Storage / lineage characterization from a single GenoMAS trace cell.

Computes 4 metrics from parsed.json + pi_events.jsonl:
  1. File size distribution (read vs write)
  2. Reader/writer fan-out distribution
  3. Adjacent write→read gap distribution
  4. Artifact lifecycle distribution

Outputs to <trace_dir>/lineage/:
  - artifacts.csv              one row per data artifact
  - tool_call_attribution.csv  one row per tool_call run_id
  - fig1_size_distribution.png Darshan-bin histogram of request/file sizes
  - fig2_fanout.png            reader/writer fan-out histograms
  - fig3_staleness_cdf.png     adjacent write→read gap histogram
  - fig4_lifecycle.png         dead-fraction histogram

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

from agent_io_tracing.analysis.size_bins import (
    DARSHAN_SIZE_LABELS,
    darshan_hist,
)
from agent_io_tracing.analysis.labels import role_for_entry

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


def _lineage_config_path(trace_dir: Path) -> Path:
    return trace_dir / "lineage" / "config.json"


def _load_saved_lineage_config(trace_dir: Path) -> bool:
    global DATA_PATH_PREFIXES, EXCLUDE_PATH_SUBSTRINGS, RAW_INPUT_PREFIXES, METADATA_PREFIXES
    path = _lineage_config_path(trace_dir)
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    prefixes = data.get("data_path_prefixes")
    excludes = data.get("exclude_path_substrings")
    raw_inputs = data.get("raw_input_prefixes")
    metadata = data.get("metadata_prefixes")
    if isinstance(prefixes, list) and prefixes:
        DATA_PATH_PREFIXES = tuple(str(p) for p in prefixes if str(p))
    if isinstance(excludes, list):
        EXCLUDE_PATH_SUBSTRINGS = tuple(str(p) for p in excludes if str(p))
    if isinstance(raw_inputs, list):
        RAW_INPUT_PREFIXES = tuple(str(p) for p in raw_inputs if str(p))
    if isinstance(metadata, list):
        METADATA_PREFIXES = tuple(str(p) for p in metadata if str(p))
    return True


def write_lineage_config(trace_dir: Path) -> None:
    path = _lineage_config_path(trace_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "data_path_prefixes": list(DATA_PATH_PREFIXES),
        "exclude_path_substrings": list(EXCLUDE_PATH_SUBSTRINGS),
        "raw_input_prefixes": list(RAW_INPUT_PREFIXES),
        "metadata_prefixes": list(METADATA_PREFIXES),
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def configure_paths_from_manifest(trace_dir: Path) -> None:
    """Broaden path filters using manifest.json when available.

    The historic defaults had /users/Minqiu hardcoded. Keep env overrides, but
    otherwise derive absolute GenoMAS/data/work paths from the run manifest so
    another user or CloudLab image does not silently drop every artifact.
    """
    global DATA_PATH_PREFIXES, RAW_INPUT_PREFIXES, METADATA_PREFIXES
    if os.environ.get("LINEAGE_DATA_PATH_PREFIXES"):
        return
    if _load_saved_lineage_config(trace_dir):
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


WRITE_READ_GAP_BINS = [
    (0.0, 1.0, "<1s"),
    (1.0, 30.0, "1-30s"),
    (30.0, 5 * 60.0, "30s-5min"),
    (5 * 60.0, 30 * 60.0, "5-30min"),
    (30 * 60.0, float("inf"), ">30min"),
]


def _write_read_gap_hist(vals: list[float]) -> dict[str, int]:
    hist = {label: 0 for _, _, label in WRITE_READ_GAP_BINS}
    for v in vals:
        for lo, hi, label in WRITE_READ_GAP_BINS:
            if lo <= v < hi:
                hist[label] += 1
                break
    return hist


def build_write_read_gap_summary(io_events: list[dict]) -> dict:
    vals: list[float] = []
    last_write_by_path: dict[str, float] = {}
    for e in sorted(io_events, key=lambda item: item["ts"]):
        path = e["path"]
        if e["kind"] == "W":
            last_write_by_path[path] = e["ts"]
        elif path in last_write_by_path and e["ts"] >= last_write_by_path[path]:
            vals.append(e["ts"] - last_write_by_path[path])
    return {
        "n_pairs": len(vals),
        "hist": _write_read_gap_hist(vals),
        "hist_note": (
            "Time bins are chosen for page-cache/writeback semantics, not from "
            "the cited inter-arrival literature."
        ),
        "p50_s": float(np.percentile(vals, 50)) if vals else None,
        "p95_s": float(np.percentile(vals, 95)) if vals else None,
        "max_s": max(vals) if vals else None,
        "pct_lt_1s": (100.0 * sum(1 for v in vals if v < 1.0) / len(vals)) if vals else None,
    }


def build_fanout_summary(per_artifact: dict) -> dict:
    def one(side: str) -> dict:
        key = "reader_tool_ids" if side == "reader" else "writer_tool_ids"
        vals = [len(rec[key]) for rec in per_artifact.values() if rec.get(key)]
        hist = Counter(vals)
        return {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "p50": float(np.percentile(vals, 50)) if vals else 0.0,
            "p95": float(np.percentile(vals, 95)) if vals else 0.0,
            "max": max(vals) if vals else 0,
            "pct_ge_2": (100.0 * sum(1 for v in vals if v >= 2) / len(vals)) if vals else None,
            "hist": {str(k): int(v) for k, v in sorted(hist.items())},
        }

    joint = Counter(
        (len(rec["writer_tool_ids"]), len(rec["reader_tool_ids"]))
        for rec in per_artifact.values()
        if rec.get("writer_tool_ids") or rec.get("reader_tool_ids")
    )
    return {
        "reader": one("reader"),
        "writer": one("writer"),
        "reader_writer_joint": {
            f"{w},{r}": int(n) for (w, r), n in sorted(joint.items())
        },
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
    role_tool_calls = {
        tid: {"role": info.get("role")}
        for tid, info in codeexec_index.items()
        if isinstance(tid, str) and isinstance(info, dict)
    }
    for e in io_events:
        tid = e["tool_call_id"]
        role = role_for_entry(e, role_tool_calls, {}) if tid else "(unattributed)"
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
        else:  # write
            rec["writes"].append((e["ts"], e["size"]))
            if tid:
                rec["writer_tool_ids"].add(tid)
                rec["all_tool_ids"].add(tid)
            if role:
                rec["writer_roles"][role] += 1
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
            "lineage_depth_generations", "n_touching_calls",
            "lifecycle_class", "t_create", "t_last_read",
            "dead_seconds", "waste_byte_seconds",
            "reuse_class", "true_size", "size_source", "read_amplification",
        ])
        for path, r in per_artifact.items():
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
    """fig0 — workload read/write bytes split by storage-placement category."""
    wl = summary["workload"]

    # Bytes are unreadable on a raw x-axis; scale to KB at minimum, stepping up
    # to MB/GB when the larger of the two rows warrants it.
    peak = max(wl.get("read_bytes") or 0, wl.get("write_bytes") or 0, 1)
    if peak >= 1024 ** 3:
        div, unit = 1024 ** 3, "GB"
    elif peak >= 1024 ** 2:
        div, unit = 1024 ** 2, "MB"
    else:
        div, unit = 1024, "KB"

    fig, axb = plt.subplots(figsize=(10, 3.2))
    cats = [c for c in CATEGORY_ORDER if c in wl["bytes_by_category"]]
    rows = [("WRITE", "write_bytes", "n_writes"), ("READ", "read_bytes", "n_reads")]  # READ on top
    for i, (_, key, _n) in enumerate(rows):
        left = 0.0
        for c in cats:
            v = wl["bytes_by_category"][c][key]
            if v <= 0:
                continue
            axb.barh(i, v / div, left=left, color=CATEGORY_COLORS[c],
                     edgecolor="black", linewidth=0.3)
            left += v / div
        total = (wl.get(key) or 0) / div
        n_ops = wl.get(_n) or 0
        axb.text(left, i, f"  {total:.1f} {unit} ({n_ops} syscalls)",
                 va="center", ha="left", fontsize=9, fontweight="bold")
    axb.set_yticks(range(len(rows)))
    axb.set_yticklabels([r[0] for r in rows], fontweight="bold")
    axb.set_xlabel(f"{unit} (by storage-placement category)")
    axb.set_title("I/O Volume by Category")
    axb.grid(axis="x", alpha=0.3)
    axb.margins(x=0.18)
    cats_present = [c for c in cats
                    if any(wl["bytes_by_category"][c][k] > 0
                           for k in ("read_bytes", "write_bytes"))]
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in cats_present]
    axb.legend(handles=handles, fontsize=8, loc="lower right", ncol=2)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def fig_size_distribution(io_events: list[dict], per_artifact: dict, out_path: Path):
    """Darshan-bin request-size and per-file size distributions."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    reads = [e["size"] for e in io_events if e["kind"] == "R"]
    writes = [e["size"] for e in io_events if e["kind"] == "W"]
    rh = darshan_hist(reads)
    wh = darshan_hist(writes)
    x = np.arange(len(DARSHAN_SIZE_LABELS))
    width = 0.42
    ax1.bar(x - width / 2, [rh[l] for l in DARSHAN_SIZE_LABELS], width,
            color="#2ca02c", label=f"read (n={len(reads)})")
    ax1.bar(x + width / 2, [wh[l] for l in DARSHAN_SIZE_LABELS], width,
            color="#d62728", label=f"write (n={len(writes)})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(DARSHAN_SIZE_LABELS, rotation=30, ha="right")
    ax1.set_ylabel("# syscalls")
    ax1.set_title("Per-syscall I/O Request Size", fontsize=10)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax1.grid(axis="y", alpha=0.3)

    file_sizes = []
    n_skipped = 0
    for rec in per_artifact.values():
        size = rec.get("true_size")
        if size is None:
            n_skipped += 1
            continue
        if size > 0:
            file_sizes.append(size)
    fh = darshan_hist(file_sizes)
    vals = [fh[l] for l in DARSHAN_SIZE_LABELS]
    if any(vals):
        ax2.bar(x, vals, color="#4c78a8", edgecolor="black", linewidth=0.3)
    else:
        ax2.text(0.5, 0.5, "No workload artifacts with size > 0",
                 transform=ax2.transAxes, ha="center", va="center",
                 color="#7f8c8d")
    ax2.set_xticks(x)
    ax2.set_xticklabels(DARSHAN_SIZE_LABELS, rotation=30, ha="right")
    ax2.set_xlabel("per-file true on-disk size")
    ax2.set_ylabel("# artifacts")
    _skip_note = (f"  ·  {n_skipped} input(s) skipped (size un-statted)"
                  if n_skipped else "")
    ax2.set_title("Per-file Artifact Size", fontsize=10)
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
    """Reader and writer fan-out histograms, one file per sample."""
    reader = [len(r["reader_tool_ids"]) for r in per_artifact.values() if r["reader_tool_ids"]]
    writer = [len(r["writer_tool_ids"]) for r in per_artifact.values() if r["writer_tool_ids"]]
    if not reader and not writer:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No artifacts had attributed read or write events.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    def capped_hist(vals: list[int]) -> list[int]:
        c = Counter(10 if v >= 10 else v for v in vals)
        return [c.get(k, 0) for k in range(1, 11)]

    labels = [str(i) for i in range(1, 10)] + [">=10"]
    x = np.arange(len(labels))
    fig, (axr, axw) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, vals, color, title in (
        (axr, reader, "#4c78a8", "Reader fan-out"),
        (axw, writer, "#f58518", "Writer fan-out"),
    ):
        counts = capped_hist(vals)
        ax.bar(x, counts, color=color, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("fan-out k (# distinct tool calls)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(title)
    axr.set_ylabel("# files")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def fig_staleness_cdf(write_read_gap: dict, out_path: Path):
    """Adjacent write→read gap histogram over read events."""
    hist = write_read_gap.get("hist") or {}
    labels = [label for _, _, label in WRITE_READ_GAP_BINS]
    counts = [int(hist.get(label, 0) or 0) for label in labels]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not any(counts):
        ax.text(0.5, 0.5,
                "No read event had a preceding write to the same path.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#888")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    x = np.arange(len(labels))
    ax.bar(x, counts, color="#7f3c8d", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("# read events")
    ax.set_xlabel("gap from immediately preceding write to this read")
    for xi, c in zip(x, counts):
        if c:
            ax.text(xi, c, f" {c}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Write→Read Gap")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_lifecycle_spans(per_artifact: dict, run_start: float, run_end: float,
                        out_path: Path, top_n: int = 25):
    """Histogram of generated-file dead fraction."""
    run_span = max(run_end - run_start, 1e-9)
    generated = [r for r in per_artifact.values() if r.get("writes")]
    if not generated:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No generated artifacts in trace.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    fractions = [
        max(0.0, min(1.0, float(r.get("dead_seconds") or 0.0) / max(run_end - float(r.get("t_create") or run_start), 1e-9)))
        for r in generated
    ]
    bins = np.linspace(0, 1, 11)
    counts, edges = np.histogram(fractions, bins=bins)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(np.arange(10), counts, color="#4c78a8", edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(10)], rotation=25, ha="right")
    ax.set_xlabel("dead time share")
    ax.set_ylabel("# generated files")
    ax.set_title("Artifact Lifecycle")
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.5, -0.02,
        r"dead time share = dead_seconds / (run_end $-$ t_create);   "
        r"dead_seconds = run_end $-$ last_read  (= run_end $-$ t_create if never read)",
        ha="center", va="top", fontsize=8, color="#555",
    )
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
                rp = br["by_role"][tr].get("read_pct")
                rp_text = f"{rp:.0f}%" if rp is not None else "n/a"
                print(f"    top reader: {tr} ({rp_text} of read bytes)")
            if tw:
                wp = br["by_role"][tw].get("write_pct")
                wp_text = f"{wp:.0f}%" if wp is not None else "n/a"
                print(f"    top writer: {tw} ({wp_text} of write bytes)")
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
    if not per_artifact:
        print(
            "ERROR: lineage found 0 workload artifacts; refusing to write empty artifacts.csv. "
            f"data_path_prefixes={list(DATA_PATH_PREFIXES)} "
            f"exclude_path_substrings={list(EXCLUDE_PATH_SUBSTRINGS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    write_lineage_config(trace_dir)
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
    io_summary["fanout"] = build_fanout_summary(per_artifact)
    io_summary["write_read_gap_s"] = build_write_read_gap_summary(io_events)
    io_summary["lifecycle"] = build_lifecycle_summary(per_artifact)

    write_csvs(per_artifact, codeexec_index, out_dir, io_events, parsed_entries)
    write_io_summary_json(io_summary, out_dir)
    fig_io_volume_summary(io_summary, out_dir / "fig0_io_volume_summary.png")
    fig_size_distribution(io_events, per_artifact, out_dir / "fig1_size_distribution.png")
    fig_reader_fanout    (per_artifact,    out_dir / "fig2_fanout.png")
    fig_staleness_cdf    (io_summary["write_read_gap_s"], out_dir / "fig3_staleness_cdf.png")
    fig_lifecycle_spans  (per_artifact, run_start, run_end, out_dir / "fig4_lifecycle.png")

    current_figures = {
        "fig0_io_volume_summary.png",
        "fig1_size_distribution.png",
        "fig2_fanout.png",
        "fig3_staleness_cdf.png",
        "fig4_lifecycle.png",
    }
    for p in out_dir.glob("fig*.png"):
        if p.name not in current_figures:
            p.unlink()

    print_summary(per_artifact, io_events, io_summary)
    print(f"\nOutputs in: {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
