#!/usr/bin/env python3
"""Emit the Phase-1 closed-loop metrics for one trace directory.

Inputs, when present:
  - parsed.json
  - pi_summary.json
  - pi_events.jsonl
  - generated_code.jsonl
  - lineage/io_summary.json
  - lineage/artifacts.csv
  - manifest.json

Outputs:
  - phase1_metrics.json
  - phase1_comparison.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from agent_io_tracing.analysis.io_api_classifier import aggregate as aggregate_io_api
    from agent_io_tracing.parsing.ebpf import (
        classify_syscall,
        resource_bucket_for_syscall,
    )
    from agent_io_tracing.lineage.analyzer import is_workload_artifact
    from agent_io_tracing.analysis.labels import (
        phase_label_for_tool_call,
        role_for_entry,
        role_for_tool_call,
    )
    from agent_io_tracing.analysis.size_bins import darshan_hist
except Exception:  # pragma: no cover - script still reports partial metrics
    classify_syscall = None  # type: ignore
    resource_bucket_for_syscall = None  # type: ignore
    aggregate_io_api = None  # type: ignore
    is_workload_artifact = None  # type: ignore
    phase_label_for_tool_call = None  # type: ignore
    role_for_entry = None  # type: ignore
    role_for_tool_call = None  # type: ignore
    darshan_hist = None  # type: ignore


DATA_SYSCALLS = {"read", "write", "pread64", "pwrite64", "readv", "writev",
                 "preadv", "pwritev", "preadv2", "pwritev2"}
OPEN_CLOSE = {"open", "openat", "close"}
DURABILITY = {"fsync", "fdatasync", "sync_file_range"}
NAMESPACE = {"mkdir", "mkdirat", "rmdir", "unlink", "unlinkat",
             "rename", "renameat", "renameat2"}
METADATA_EXTRAS = OPEN_CLOSE | DURABILITY | NAMESPACE
DIRECTORY_SCAN_SYSCALLS = {"getdents64", "getdents"}
OPEN_STAT_SYSCALLS = {"open", "openat", "openat2", "stat", "fstat", "lstat",
                      "newfstatat", "statx", "access", "faccessat"}
# Filename fragments that identify a checkpoint/coordination-shaped artifact
# (small JSON files rewritten to record run/task state), as opposed to a
# dataset/result file. Matches GenoMAS's `cohort_info.json` / `completed_tasks.json`
# and generalizes past this one workflow via substring match, not an exact list.
STATE_FILE_PATH_HINTS = ("cohort_info", "completed_tasks", "_state.json",
                         "manifest.json", ".lock")

PAGE_CACHE_TIME_BINS = [
    (0.0, 1.0, "<1s"),
    (1.0, 30.0, "1-30s"),
    (30.0, 5 * 60.0, "30s-5min"),
    (5 * 60.0, 30 * 60.0, "5-30min"),
    (30 * 60.0, float("inf"), ">30min"),
]


def page_cache_time_hist(values: list[float]) -> dict[str, int]:
    hist = {label: 0 for _, _, label in PAGE_CACHE_TIME_BINS}
    for value in values:
        for lo, hi, label in PAGE_CACHE_TIME_BINS:
            if lo <= value < hi:
                hist[label] += 1
                break
    return hist

# Markers that a failed open/stat is the CPython import machinery probing
# sys.path (every `import x` stats/opens the candidate name in each sys.path
# entry, and all-but-one return ENOENT). These failures are interpreter noise,
# not the agent probing candidate data paths, so they must be excluded before
# the count is read as an Axis-4 exploration/backtracking signal.
_IMPORT_ROOT_MARKERS = (
    "site-packages", "dist-packages", "/lib/python", "/lib64/python",
    "lib-dynload", "__pycache__", ".egg", ".dist-info", "/python3.",
)
# Markers of the dynamic linker / import-hook machinery (not the agent probing
# candidate data paths): glibc-hwcaps searches every CPU-feature-optimized .so
# variant (all-but-one ENOENT); editable installs register a finder/path-hook
# that "opens" synthetic names; and code exec reports pseudo-paths like
# <string> / <unknown> / <frozen ...> that are not filesystem candidates at all.
_LOADER_NOISE_MARKERS = (
    "glibc-hwcaps/", "__editable__", ".__path_hook__", ".finder.",
)
_MODULE_SUFFIX_RE = re.compile(r"\.(py[cod]?|so|pyd|abi3\.so)$")
_EXT_MODULE_RE = re.compile(r"\.(cpython-\d+[^/]*|abi3)\.so$")
# A shared object (foo.so, libbar.so.6). A *failed* probe for one is always the
# dynamic linker searching its HWCAP/tls/arch subdir cascade (…/tls/haswell/
# avx512_1/x86_64/lib.so — all-but-one ENOENT), never the agent probing a
# candidate data path, so it is filtered regardless of location.
_SHARED_OBJ_RE = re.compile(r"\.so(\.\d+)*$")


def _is_python_import_probe(path: str | None) -> bool:
    """True if a failed open/stat looks like CPython's import search / dynamic
    linker rather than an agent-issued candidate-path probe."""
    if not path:
        return False
    # Pseudo-paths from exec of code strings (<string>, <unknown>, <frozen ...>)
    # are not filesystem candidates.
    if path.startswith("<") and path.endswith(">"):
        return True
    if any(m in path for m in _LOADER_NOISE_MARKERS):
        return True
    base = path.rsplit("/", 1)[-1]
    if _SHARED_OBJ_RE.search(base):
        return True
    # Compiled-extension names (foo.cpython-311-x86_64-linux-gnu.so, foo.abi3.so)
    # are unambiguous import machinery regardless of location.
    if _EXT_MODULE_RE.search(base):
        return True
    if _MODULE_SUFFIX_RE.search(base) and any(m in path for m in _IMPORT_ROOT_MARKERS):
        return True
    # Namespace/package dir probes under an import root.
    if base == "__init__.py" or any(m in path for m in _IMPORT_ROOT_MARKERS):
        return True
    return False


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], pct: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    rank = (pct / 100.0) * (len(vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def pct(values: list[float], pred) -> float | None:
    if not values:
        return None
    return 100.0 * sum(1 for v in values if pred(v)) / len(values)


def _entry_end_ms(entry: dict[str, Any]) -> float | None:
    ts_ms = entry.get("ts_ms")
    if isinstance(ts_ms, (int, float)):
        return float(ts_ms)
    ts = entry.get("timestamp")
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts).timestamp() * 1000.0
    except ValueError:
        return None


def _entry_end_s(entry: dict[str, Any]) -> float | None:
    ms = _entry_end_ms(entry)
    return ms / 1000.0 if ms is not None else None


def read_artifacts(trace_dir: Path) -> list[dict[str, Any]]:
    path = trace_dir / "lineage" / "artifacts.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_event_phase_index(trace_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    path = trace_dir / "pi_events.jsonl"
    if not path.is_file():
        return index
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = e.get("run_id")
        if not isinstance(rid, str):
            continue
        rec = index.setdefault(rid, {})
        for key in ("phase", "genomas_role", "tool_name", "io_layers",
                    "cache_hit", "cache_key"):
            if key in e and e.get(key) is not None:
                rec[key] = e.get(key)
    return index


def storage_metadata_syscall(syscall: str) -> bool:
    if syscall in METADATA_EXTRAS:
        return True
    if classify_syscall is None:
        return False
    return classify_syscall(syscall) == "metadata"


def is_storage_file_io(entry: dict[str, Any]) -> bool:
    bucket = entry.get("resource_bucket")
    if bucket:
        return bucket == "file_io"
    if resource_bucket_for_syscall is None:
        return False
    return resource_bucket_for_syscall(str(entry.get("syscall"))) == "file_io"


def phase_for_entry(entry: dict[str, Any], phases: dict[str, dict[str, Any]],
                    tool_calls: dict[str, dict[str, Any]]) -> str:
    tid = entry.get("matched_tool_call")
    if isinstance(tid, str):
        if phase_label_for_tool_call:
            return phase_label_for_tool_call(tid, tool_calls.get(tid), phases)
        return str((phases.get(tid) or {}).get("phase") or "uncategorized_tool")
    return "orchestration"


def build_tool_call_map(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        tc.get("tool_id"): tc
        for tc in parsed.get("tool_calls", [])
        if isinstance(tc.get("tool_id"), str)
    }


def compute_latency_by_phase(parsed: dict[str, Any],
                             phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tool_calls = build_tool_call_map(parsed)
    by_phase: dict[str, list[float]] = defaultdict(list)
    for e in parsed.get("fs_entries", []):
        if e.get("resource_bucket") == "interface_probe":
            continue
        dur_ms = float(e.get("duration", 0.0) or 0.0) * 1000.0
        by_phase[phase_for_entry(e, phases, tool_calls)].append(dur_ms)
    return {
        ph: {
            "count": len(vals),
            "p50_ms": percentile(vals, 50),
            "p95_ms": percentile(vals, 95),
            "p99_ms": percentile(vals, 99),
        }
        for ph, vals in sorted(by_phase.items())
    }


READ_SYSCALLS_STRICT = {"read", "pread64", "readv", "preadv", "preadv2"}
WRITE_SYSCALLS_STRICT = {"write", "pwrite64", "writev", "pwritev", "pwritev2"}
STDIO_READ_FUNCS = {"fread"}
STDIO_WRITE_FUNCS = {"fwrite"}
HDF5_READ_FUNCS = {"H5Dread"}
HDF5_WRITE_FUNCS = {"H5Dwrite"}
MPIIO_READ_FUNCS = {"MPI_File_read", "MPI_File_read_at", "MPI_File_read_all"}
MPIIO_WRITE_FUNCS = {"MPI_File_write", "MPI_File_write_at", "MPI_File_write_all"}


def compute_reread_attribution(parsed: dict[str, Any],
                               phases: dict[str, dict[str, Any]],
                               artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify rereads of the same file as agent-induced vs. residual,
    joined on the *specific* tool_call_id (``matched_tool_call``) rather
    than the coarser ``phase:role`` label used by compute_latency_by_phase /
    compute_bytes_ops_by_phase — checked directly on the CloudLab run: one
    coarse phase label ("code_exec:GEOAgent") covers 5 *different*
    tool_call_ids, so joining on the label would misclassify legitimate
    reuse across action units as agent-induced.

    Touch key is (path, tool_call_id, fd), not (path, tool_call_id): a
    single logical `open()`-then-buffered-`read()` in Python emits many
    `read` syscalls on the same fd, which must collapse into one touch, not
    be counted as N rereads. A *second* fd on the same path within the same
    tool_call_id (a genuine re-open-and-reread within one code snippet) is
    kept as a separate touch.

    Rule per subsequent touch of a path (first touch is never a "reread"):
      - same tool_call_id as the immediately preceding touch of this path,
        different fd -> reopened-within-the-same-step -> agent-induced
      - different tool_call_id whose phase is `action_unit_backtrack`
        -> agent-induced (a retry re-reading a file it already had)
      - different tool_call_id, any other phase -> residual. NOTE: this
        only checks tool_call_id, not actual pipeline-stage identity, so it
        does not prove the two touches were in different stages — a stage
        that spans multiple non-backtrack tool_call_ids would also land
        here. Named `different_tool_call_id` for that reason (not
        "cross-stage").
    """
    # Workload data only: otherwise re-imports of .venv libraries across code-exec
    # steps count as "agent-induced reread waste", which they are not.
    _wl = make_workload_filter(artifacts)
    entries = [
        e for e in parsed.get("fs_entries", [])
        if str(e.get("syscall")) in READ_SYSCALLS_STRICT
        and isinstance(e.get("path"), str)
        and isinstance(e.get("matched_tool_call"), str)
        and _wl(e["path"])
    ]

    touches: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for e in entries:
        key = (e["path"], e["matched_tool_call"], e.get("file_descriptor"))
        t = touches.setdefault(key, {
            "path": e["path"], "tool_call_id": e["matched_tool_call"],
            "first_ts": e.get("timestamp"), "bytes": 0,
        })
        if e.get("timestamp") and (not t["first_ts"] or e["timestamp"] < t["first_ts"]):
            t["first_ts"] = e["timestamp"]
        sz = e.get("actual_size") or e.get("requested_size") or 0
        t["bytes"] += sz if isinstance(sz, (int, float)) and sz > 0 else 0

    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in touches.values():
        by_path[t["path"]].append(t)
    for touch_list in by_path.values():
        touch_list.sort(key=lambda t: t["first_ts"] or "")

    same_step_reopen = []
    reread_after_backtrack = []
    different_tool_call_id = []

    for path, touch_list in by_path.items():
        prev = None
        for t in touch_list:
            if prev is None:
                pass  # first touch: not a reread
            elif t["tool_call_id"] == prev["tool_call_id"]:
                same_step_reopen.append(t)
            else:
                phase = (phases.get(t["tool_call_id"]) or {}).get("phase")
                if phase == "action_unit_backtrack":
                    reread_after_backtrack.append(t)
                else:
                    different_tool_call_id.append(t)
            prev = t

    def _summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(items),
            "bytes": sum(i["bytes"] for i in items),
            "distinct_files": len({i["path"] for i in items}),
        }

    return {
        "agent_induced": {
            "same_step_reopen": _summarize(same_step_reopen),
            "reread_after_backtrack": _summarize(reread_after_backtrack),
        },
        "different_tool_call_id": _summarize(different_tool_call_id),
        "note": "joined on matched_tool_call, not the coarser phase:role label",
    }


def compute_bytes_ops_by_phase(parsed: dict[str, Any],
                               phases: dict[str, dict[str, Any]],
                               artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Same phase join as compute_latency_by_phase, but rolling up bytes/op
    counts instead of latency. This is what turns a validated phase tag (e.g.
    ``action_unit_backtrack``) into an actual retry-induced-bytes number.

    Scoped to workload data only (same workload-artifact set as the per-syscall
    histogram and the batching-efficiency table); otherwise interpreter imports
    (.venv), logger output and our own trace files leak in and pollute the
    per-phase attribution (~40% of ops in some traces)."""
    tool_calls = build_tool_call_map(parsed)
    _wl = make_workload_filter(artifacts)
    by_phase: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"ops": 0, "read_ops": 0, "write_ops": 0,
                 "read_bytes": 0, "write_bytes": 0}
    )
    for e in parsed.get("fs_entries", []):
        syscall = str(e.get("syscall"))
        if syscall not in DATA_SYSCALLS:
            continue
        if not _wl(e.get("path") or ""):
            continue
        ph = phase_for_entry(e, phases, tool_calls)
        d = by_phase[ph]
        d["ops"] += 1
        size = e.get("actual_size") or e.get("requested_size") or e.get("bytes_transferred") or 0
        size = size if isinstance(size, (int, float)) and size > 0 else 0
        if syscall in {"write", "pwrite64", "writev", "pwritev", "pwritev2"}:
            d["write_ops"] += 1
            d["write_bytes"] += size
        else:
            d["read_ops"] += 1
            d["read_bytes"] += size
    return dict(sorted(by_phase.items()))


def compute_interface_byte_mix(parsed: dict[str, Any],
                               artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Axis 2 Phase A: measured interface bytes from libc probes + syscalls.

    fread/fwrite uprobes are usable for workload comparison only when the tracer
    captured FILE*->_fileno and the parser resolved that fd to a workload path.
    Pathless STDIO probes are retained separately as process-tree observations.
    """
    wl = make_workload_filter(artifacts)
    raw = {
        "stdio_read_bytes": 0,
        "stdio_write_bytes": 0,
        "stdio_process_tree_read_bytes": 0,
        "stdio_process_tree_write_bytes": 0,
        "posix_read_bytes_observed": 0,
        "posix_write_bytes_observed": 0,
        "stdio_ops": 0,
        "stdio_process_tree_ops": 0,
        "posix_ops": 0,
    }
    for e in parsed.get("fs_entries", []):
        syscall = str(e.get("syscall") or "")
        path = e.get("path") or ""
        is_stdio = syscall in (STDIO_READ_FUNCS | STDIO_WRITE_FUNCS)
        if path and not wl(path):
            continue
        if not path:
            if not is_stdio:
                continue
            scope = "process_tree"
        else:
            scope = "workload"
        size = e.get("actual_size") or e.get("bytes_transferred") or 0
        size = int(size) if isinstance(size, (int, float)) and size > 0 else 0
        if size <= 0:
            continue
        if syscall in STDIO_READ_FUNCS:
            if scope == "workload":
                raw["stdio_read_bytes"] += size
                raw["stdio_ops"] += 1
            else:
                raw["stdio_process_tree_read_bytes"] += size
                raw["stdio_process_tree_ops"] += 1
        elif syscall in STDIO_WRITE_FUNCS:
            if scope == "workload":
                raw["stdio_write_bytes"] += size
                raw["stdio_ops"] += 1
            else:
                raw["stdio_process_tree_write_bytes"] += size
                raw["stdio_process_tree_ops"] += 1
        elif syscall in READ_SYSCALLS_STRICT:
            raw["posix_read_bytes_observed"] += size
            raw["posix_ops"] += 1
        elif syscall in WRITE_SYSCALLS_STRICT:
            raw["posix_write_bytes_observed"] += size
            raw["posix_ops"] += 1

    stdio_bytes = raw["stdio_read_bytes"] + raw["stdio_write_bytes"]
    stdio_process_tree_bytes = (
        raw["stdio_process_tree_read_bytes"] + raw["stdio_process_tree_write_bytes"]
    )
    posix_observed = raw["posix_read_bytes_observed"] + raw["posix_write_bytes_observed"]
    return {
        **raw,
        "stdio_bytes": stdio_bytes,
        "stdio_process_tree_bytes": stdio_process_tree_bytes,
        "posix_observed_bytes": posix_observed,
        "note": (
            "POSIX bytes are workload-scoped. STDIO bytes are workload-scoped only "
            "when FILE*->_fileno was captured and resolved to a workload path; "
            "pathless STDIO is reported separately at process-tree scope and must "
            "not be added to or divided against workload POSIX bytes."
        ),
    }


def compute_measured_interface_layers(parsed: dict[str, Any],
                                      artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Axis 2 Phase B: measured interface-layer calls from uprobes + syscalls.

    The MMAP layer counts file-backed mmap() calls; its bytes are the mapped
    region length (an upper bound on the volume actually read/written through
    the mapping, since page access happens without further syscalls).
    """
    wl = make_workload_filter(artifacts)
    layers: dict[str, dict[str, Any]] = {
        "STDIO": {
            "ops": 0, "read_ops": 0, "write_ops": 0,
            "bytes_resolved": True, "bytes": 0,
            "process_tree_ops": 0, "process_tree_bytes": 0,
            "scope": "workload_when_path_resolved",
            "by_function": Counter(),
        },
        "POSIX": {
            "ops": 0, "read_ops": 0, "write_ops": 0,
            "bytes_resolved": True, "bytes": 0, "by_function": Counter(),
        },
        "MMAP": {
            "ops": 0, "read_ops": 0, "write_ops": 0,
            "bytes_resolved": False, "bytes": None,
            "mapped_bytes_upper_bound": 0, "by_function": Counter(),
        },
        "HDF5": {
            "ops": 0, "read_ops": 0, "write_ops": 0,
            "bytes_resolved": False, "bytes": None, "by_function": Counter(),
        },
        "MPI-IO": {
            "ops": 0, "read_ops": 0, "write_ops": 0,
            "bytes_resolved": False, "bytes": None, "raw_count_total": 0,
            "by_function": Counter(),
        },
    }

    for e in parsed.get("fs_entries", []):
        syscall = str(e.get("syscall") or "")
        path = e.get("path") or ""
        if path and not wl(path):
            continue
        if not path and syscall not in (STDIO_READ_FUNCS | STDIO_WRITE_FUNCS):
            continue
        is_stdio = syscall in (STDIO_READ_FUNCS | STDIO_WRITE_FUNCS)
        stdio_workload_scoped = bool(path)
        size = e.get("actual_size") or e.get("bytes_transferred") or 0
        size = int(size) if isinstance(size, (int, float)) and size > 0 else 0
        if is_stdio:
            d = layers["STDIO"]
            d["by_function"][syscall] += 1
            if stdio_workload_scoped:
                d["ops"] += 1
                d["bytes"] += size
                if syscall in STDIO_READ_FUNCS:
                    d["read_ops"] += 1
                else:
                    d["write_ops"] += 1
            else:
                d["process_tree_ops"] += 1
                d["process_tree_bytes"] += size
        elif syscall in READ_SYSCALLS_STRICT | WRITE_SYSCALLS_STRICT:
            d = layers["POSIX"]
            d["ops"] += 1
            d["by_function"][syscall] += 1
            d["bytes"] += size
            if syscall in READ_SYSCALLS_STRICT:
                d["read_ops"] += 1
            else:
                d["write_ops"] += 1
        elif syscall == "mmap":
            # Only file-backed, successful mappings (resolved path via fd,
            # return != MAP_FAILED). Anonymous maps have no resolved path.
            path = e.get("path")
            ret = e.get("return_value")
            if not isinstance(path, str) or not path:
                continue
            if isinstance(ret, int) and ret == -1:
                continue
            prot = int(e.get("flags") or 0)
            if prot & 0x4:  # PROT_EXEC: shared-library / code loads, not data I/O
                continue
            length = e.get("requested_size") or 0
            length = int(length) if isinstance(length, (int, float)) and length > 0 else 0
            d = layers["MMAP"]
            d["ops"] += 1
            d["by_function"]["mmap"] += 1
            d["mapped_bytes_upper_bound"] += length
            if prot & 0x1:  # PROT_READ
                d["read_ops"] += 1
            if prot & 0x2:  # PROT_WRITE
                d["write_ops"] += 1

    by_tool: dict[str, dict[str, int]] = defaultdict(
        lambda: {"hdf5_write_ops": 0, "physical_write_ops": 0, "physical_pwrite_ops": 0}
    )
    for e in parsed.get("fs_entries", []):
        path = e.get("path") or ""
        if not wl(path):
            continue
        tid = e.get("matched_tool_call")
        if not isinstance(tid, str):
            continue
        syscall = str(e.get("syscall") or "")
        if syscall in WRITE_SYSCALLS_STRICT:
            by_tool[tid]["physical_write_ops"] += 1
        if syscall in {"pwrite64", "pwritev", "pwritev2"}:
            by_tool[tid]["physical_pwrite_ops"] += 1

    for e in parsed.get("lib_io_entries", []):
        library = str(e.get("library") or "")
        function = str(e.get("function") or "")
        if library == "hdf5":
            d = layers["HDF5"]
            d["ops"] += 1
            d["by_function"][function] += 1
            if function in HDF5_READ_FUNCS:
                d["read_ops"] += 1
            elif function in HDF5_WRITE_FUNCS:
                d["write_ops"] += 1
                tid = e.get("matched_tool_call")
                if isinstance(tid, str):
                    by_tool[tid]["hdf5_write_ops"] += 1
        elif library == "mpiio":
            d = layers["MPI-IO"]
            d["ops"] += 1
            d["by_function"][function] += 1
            if function in MPIIO_READ_FUNCS:
                d["read_ops"] += 1
            elif function in MPIIO_WRITE_FUNCS:
                d["write_ops"] += 1
            count = e.get("count")
            if isinstance(count, (int, float)) and count > 0:
                d["raw_count_total"] += int(count)

    tool_ratios = []
    for tid, vals in by_tool.items():
        h5 = vals["hdf5_write_ops"]
        phys = vals["physical_write_ops"]
        pwrite = vals["physical_pwrite_ops"]
        if h5 <= 0:
            continue
        tool_ratios.append({
            "tool_call_id": tid,
            **vals,
            "hdf5_write_to_physical_write_ops": (h5 / phys) if phys else None,
            "physical_write_to_hdf5_write_ops": (phys / h5) if h5 else None,
            "hdf5_write_to_pwrite_ops": (h5 / pwrite) if pwrite else None,
            "pwrite_to_hdf5_write_ops": (pwrite / h5) if h5 else None,
        })

    total_h5_write = layers["HDF5"]["write_ops"]
    total_phys_write = layers["POSIX"]["write_ops"]
    total_pwrite = sum(
        1 for e in parsed.get("fs_entries", [])
        if str(e.get("syscall")) in {"pwrite64", "pwritev", "pwritev2"}
        and wl(e.get("path") or "")
    )
    out_layers = {}
    for name, vals in layers.items():
        clean = dict(vals)
        clean["by_function"] = dict(vals["by_function"])
        out_layers[name] = clean
    return {
        "layers": out_layers,
        "logical_physical_write_ratio": {
            "hdf5_write_ops": total_h5_write,
            "physical_write_ops": total_phys_write,
            "physical_pwrite_ops": total_pwrite,
            "hdf5_write_to_physical_write_ops": (
                total_h5_write / total_phys_write if total_phys_write else None
            ),
            "physical_write_to_hdf5_write_ops": (
                total_phys_write / total_h5_write if total_h5_write else None
            ),
            "hdf5_write_to_pwrite_ops": (
                total_h5_write / total_pwrite if total_pwrite else None
            ),
            "pwrite_to_hdf5_write_ops": (
                total_pwrite / total_h5_write if total_h5_write else None
            ),
            "by_tool_call": tool_ratios[:50],
        },
        "note": (
            "POSIX and mmap layers are scoped to workload artifacts. STDIO is "
            "workload-scoped only when FILE* fd resolution produced a workload path; "
            "otherwise pathless STDIO probes are kept as process-tree-only context. "
            "MMAP bytes are mapped-region length, an upper bound on actual page access. "
            "HDF5/MPI-IO uprobes are pathless and therefore reported as observed calls, not workload-filtered bytes."
        ),
    }


def compute_directory_scan_count(parsed: dict[str, Any],
                                 artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    # Workload directories only: otherwise Python's sys.path / site-packages
    # walking dominates "top rescanned directories".
    _wl = make_workload_dir_filter(artifacts)
    entries = [e for e in parsed.get("fs_entries", [])
              if str(e.get("syscall")) in DIRECTORY_SCAN_SYSCALLS
              and _wl(e.get("path") or "")]
    by_path = Counter(e.get("path") for e in entries if e.get("path"))
    hist = Counter(">=10" if c >= 10 else str(c) for c in by_path.values())
    return {
        "total_scans": len(entries),
        "unique_directories_scanned": len(by_path),
        "rescanned_directories": sum(1 for c in by_path.values() if c > 1),
        "scans_per_dir_hist": {k: hist.get(k, 0) for k in [*(str(i) for i in range(1, 10)), ">=10"]},
        "p95_scans_per_dir": percentile([float(c) for c in by_path.values()], 95),
    }


def compute_failed_open_stat_count(parsed: dict[str, Any]) -> dict[str, Any]:
    # Denominator: every open/stat/access attempt (success or failure), used to
    # normalize the failed-probe count so traces of different length are
    # comparable. CPython import-machinery probes are excluded from both the
    # numerator and denominator so the rate reflects agent-level path probing,
    # not the interpreter statting sys.path on every `import`.
    total_attempts = 0
    total_attempts_agent = 0
    failed_all = 0
    import_probe_failed = 0
    agent_entries: list[dict[str, Any]] = []
    for e in parsed.get("fs_entries", []):
        if str(e.get("syscall")) not in OPEN_STAT_SYSCALLS:
            continue
        is_import = _is_python_import_probe(e.get("path"))
        total_attempts += 1
        if not is_import:
            total_attempts_agent += 1
        rv = e.get("return_value")
        if not (isinstance(rv, (int, float)) and rv < 0):
            continue
        failed_all += 1
        if is_import:
            import_probe_failed += 1
        else:
            agent_entries.append(e)

    by_syscall = Counter(str(e.get("syscall")) for e in agent_entries)
    by_path = Counter(e.get("path") for e in agent_entries if e.get("path"))
    return {
        # Agent-level (import probes excluded) — the Axis-4 exploration signal.
        "total_failed": len(agent_entries),
        "failed_rate": (len(agent_entries) / total_attempts_agent
                        if total_attempts_agent else None),
        "by_syscall": dict(by_syscall),
        "distinct_paths_involved": len(by_path),
        "top_failing_paths": by_path.most_common(5),
        # Transparency: how much interpreter noise was filtered out.
        "total_failed_raw": failed_all,
        "import_probe_failed_excluded": import_probe_failed,
        "total_open_stat_attempts": total_attempts,
    }


def compute_error_log_reads(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Reads of files already classified as `logs` by the lineage pass
    (lineage.classify.classify_artifact) — i.e. the agent reading back its own
    or the workflow's log output, a canonical exploration/debug signal."""
    log_rows = [a for a in artifacts if a.get("category") == "logs"]
    total_reads = sum(int(float(a.get("n_reads") or 0)) for a in log_rows)
    total_read_bytes = sum(int(float(a.get("total_read_bytes") or 0)) for a in log_rows)
    return {
        "log_files": len(log_rows),
        "log_files_ever_read": sum(1 for a in log_rows if int(float(a.get("n_reads") or 0)) > 0),
        "total_reads": total_reads,
        "total_read_bytes": total_read_bytes,
    }


def compute_state_file_rewrite_frequency(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Rewrite frequency for small coordination/progress-state files (path
    matches STATE_FILE_PATH_HINTS) — not process/job checkpointing. High
    n_writes relative to the number of logical tasks tracked is the
    read-modify-write-per-task pattern flagged as a script-caused
    misconfiguration candidate (see tools/preprocess.py's
    validate_and_save_cohort_info)."""
    rows = [
        a for a in artifacts
        if any(h in (a.get("path") or "") for h in STATE_FILE_PATH_HINTS)
    ]
    per_file = [
        {
            "path": a.get("path"),
            "n_writes": int(float(a.get("n_writes") or 0)),
            "n_reads": int(float(a.get("n_reads") or 0)),
            "total_write_bytes": int(float(a.get("total_write_bytes") or 0)),
        }
        for a in rows
    ]
    return {
        "state_shaped_files": len(per_file),
        "total_writes": sum(r["n_writes"] for r in per_file),
        "per_file": per_file,
        "path_hints": list(STATE_FILE_PATH_HINTS),
    }


def compute_op_ratios(parsed: dict[str, Any]) -> dict[str, Any]:
    entries = parsed.get("fs_entries", [])
    cats = Counter()
    storage_metadata = 0
    data_ops = 0
    for e in entries:
        syscall = str(e.get("syscall"))
        cat = classify_syscall(syscall) if classify_syscall else "other"
        cats[cat] += 1
        if syscall in DATA_SYSCALLS:
            data_ops += 1
        if storage_metadata_syscall(syscall):
            storage_metadata += 1
    strict_metadata = cats.get("metadata", 0)
    return {
        "strict_metadata_ops": strict_metadata,
        "storage_metadata_ops": storage_metadata,
        "data_ops": data_ops,
        "strict_metadata_to_data_ops": strict_metadata / data_ops if data_ops else None,
        "storage_metadata_to_data_ops": storage_metadata / data_ops if data_ops else None,
        "by_category": dict(cats),
    }


def compute_request_size_cdf(parsed: dict[str, Any]) -> dict[str, Any]:
    sizes = []
    bytes_total = 0.0
    bytes_weighted_num = 0.0
    for e in parsed.get("fs_entries", []):
        if str(e.get("syscall")) not in DATA_SYSCALLS:
            continue
        sz = e.get("requested_size") or e.get("actual_size") or e.get("bytes_transferred")
        if isinstance(sz, (int, float)) and sz > 0:
            sizes.append(float(sz))
            bytes_total += float(sz)
            bytes_weighted_num += float(sz) * float(sz)
    return {
        "count": len(sizes),
        "p50_bytes": percentile(sizes, 50),
        "p95_bytes": percentile(sizes, 95),
        "p99_bytes": percentile(sizes, 99),
        "pct_lt_4kb": pct(sizes, lambda x: x < 4096),
        "pct_lt_64kb": pct(sizes, lambda x: x < 64 * 1024),
        "pct_lt_10mb": pct(sizes, lambda x: x < 10 * 1024 * 1024),
        "bytes_weighted_mean_request_bytes": (bytes_weighted_num / bytes_total) if bytes_total else None,
    }


def compute_file_size_cdf(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = []
    for row in artifacts:
        raw = row.get("true_size") or row.get("size_bytes") or ""
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0:
            sizes.append(val)
    return {
        "count": len(sizes),
        "p50_bytes": percentile(sizes, 50),
        "p95_bytes": percentile(sizes, 95),
        "p99_bytes": percentile(sizes, 99),
        "pct_lt_1mb": pct(sizes, lambda x: x < 1024 * 1024),
        "pct_lt_1gb": pct(sizes, lambda x: x < 1024 ** 3),
    }


def compute_fs_io_non_llm(parsed: dict[str, Any], pi_summary: dict[str, Any],
                          manifest: dict[str, Any]) -> dict[str, Any]:
    metrics = pi_summary.get("metrics") or {}
    wall_ms = float(metrics.get("overall_execution_time_ms", 0.0) or 0.0)
    llm_ms = float(
        ((metrics.get("llm_completion_metrics") or {}).get("total_llm_time_ms", 0.0))
        or 0.0
    )
    fs_io_ms_sum = 0.0
    for e in parsed.get("fs_entries", []):
        if is_storage_file_io(e):
            fs_io_ms_sum += float(e.get("duration", 0.0) or 0.0) * 1000.0
    non_llm_ms = max(0.0, wall_ms - llm_ms)
    return {
        "wall_ms": wall_ms,
        "llm_ms": llm_ms,
        "non_llm_ms": non_llm_ms,
        "fs_io_ms_sum": fs_io_ms_sum,
        "fs_io_pct_of_non_llm": (100.0 * fs_io_ms_sum / non_llm_ms) if non_llm_ms else None,
        "fs_io_pct_of_wall": (100.0 * fs_io_ms_sum / wall_ms) if wall_ms else None,
        "replay_mode": manifest.get("replay_mode"),
        "definition": "sum(storage syscall latency) / (wall - LLM time)",
    }


def make_workload_filter(artifacts: list[dict[str, Any]]):
    """Predicate: is this path workload data (vs interpreter/.venv/trace noise)?

    Source of truth is the lineage step's already-workload-scoped artifacts.csv,
    which was produced with each run's correct LINEAGE_DATA_PATH_PREFIXES. Using
    that path set makes the analysis env-independent and guarantees it matches
    the per-syscall histogram exactly. Falls back to is_workload_artifact (env
    prefixes) only when artifacts.csv is missing/empty."""
    workload_paths = {row.get("path") for row in artifacts if row.get("path")}
    if workload_paths:
        return lambda p: p in workload_paths
    if is_workload_artifact is not None:
        return is_workload_artifact
    return lambda _p: True


def make_workload_dir_filter(artifacts: list[dict[str, Any]]):
    """Predicate for DIRECTORY paths (getdents), which are never in artifacts.csv
    (that lists files). A directory is workload iff some workload file lives under
    it. Env-independent, same source of truth as make_workload_filter."""
    workload_paths = {row.get("path") for row in artifacts if row.get("path")}
    if not workload_paths:
        if is_workload_artifact is not None:
            return is_workload_artifact
        return lambda _p: True

    def _is_workload_dir(d: str) -> bool:
        if not d:
            return False
        prefix = d.rstrip("/") + "/"
        return any(f == d or f.startswith(prefix) for f in workload_paths)

    return _is_workload_dir


def compute_analytical_optimum(parsed: dict[str, Any],
                               artifacts: list[dict[str, Any]],
                               optimal_request_bytes: int) -> dict[str, Any]:
    # Historical field name retained for compatibility. This now returns only
    # honest workload-scoped counts; synthetic 4 MiB optimum/amplification
    # fields live in neither JSON nor figures.
    _ = optimal_request_bytes
    _wl = make_workload_filter(artifacts)

    generated = []
    for row in artifacts:
        path = row.get("path", "")
        if not _wl(path):
            continue
        try:
            writes = int(float(row.get("n_writes") or 0))
            write_bytes = int(float(row.get("total_write_bytes") or 0))
        except ValueError:
            continue
        if writes > 0 or write_bytes > 0:
            generated.append((path, writes, write_bytes))
    actual_generated_files = len({p for p, _, _ in generated if p})

    actual_write_ops = 0
    actual_write_bytes = 0
    actual_read_ops = 0
    actual_read_bytes = 0
    actual_metadata_ops = 0
    for e in parsed.get("fs_entries", []):
        syscall = str(e.get("syscall"))
        path = e.get("path") or ""
        if not _wl(path):
            continue
        if storage_metadata_syscall(syscall):
            actual_metadata_ops += 1
            continue
        is_read = syscall in READ_SYSCALLS_STRICT
        is_write = syscall in WRITE_SYSCALLS_STRICT
        if not (is_read or is_write):
            continue
        sz = e.get("bytes_transferred") or 0
        if not (isinstance(sz, (int, float)) and sz > 0):
            continue
        if is_read:
            actual_read_ops += 1
            actual_read_bytes += sz
        else:
            actual_write_ops += 1
            actual_write_bytes += sz
    return {
        "actual_generated_files": actual_generated_files,
        "actual_write_bytes": actual_write_bytes,
        "actual_write_ops": actual_write_ops,
        "actual_storage_metadata_ops": actual_metadata_ops,
        "actual_read_bytes": actual_read_bytes,
        "actual_read_ops": actual_read_ops,
        "assumption": (
            "workload data only (same filter as the per-syscall request-size "
            "histogram; excludes Python imports / .venv / logger / trace output)"
        ),
    }


def compute_sequentiality(parsed: dict[str, Any],
                          artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    wl = make_workload_filter(artifacts)
    data_syscalls = READ_SYSCALLS_STRICT | WRITE_SYSCALLS_STRICT
    streams_tid: dict[tuple[str, int, int, int], list[tuple[float, int, int, bool]]] = defaultdict(list)
    streams_pid: dict[tuple[str, int, int, int], list[tuple[float, int, int, bool]]] = defaultdict(list)
    eligible_ops = 0
    ops_with_offset = 0
    ops_with_offset_by_kind = Counter()
    excluded = Counter()

    for e in parsed.get("fs_entries", []):
        syscall = str(e.get("syscall"))
        if syscall not in data_syscalls:
            continue
        if not is_storage_file_io(e):
            excluded["non_storage"] += 1
            continue
        path = e.get("path") or ""
        if not wl(path):
            excluded["non_workload"] += 1
            continue
        size = _io_bytes(e)
        if size <= 0:
            excluded["zero_bytes"] += 1
            continue
        eligible_ops += 1
        off = e.get("offset")
        if not isinstance(off, int):
            excluded["missing_offset"] += 1
            continue
        ops_with_offset += 1
        kind = "write" if syscall in WRITE_SYSCALLS_STRICT else "read"
        ops_with_offset_by_kind[kind] += 1
        fd = e.get("file_descriptor")
        gen = e.get("open_generation")
        pid = e.get("pid")
        tid = e.get("tid")
        if not all(isinstance(x, int) for x in (fd, gen, pid, tid)):
            excluded["missing_stream_identity"] += 1
            continue
        ts = _entry_end_s(e) or 0.0
        item = (ts, off, size, bool(e.get("append")))
        streams_tid[(kind, tid, fd, gen)].append(item)
        streams_pid[(kind, pid, fd, gen)].append(item)

    def classify_streams(
        streams: dict[tuple[str, int, int, int], list[tuple[float, int, int, bool]]]
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "read": Counter(),
            "write": Counter(),
            "stride_hist": {},
            "n_streams": 0,
            "n_streams_with_transitions": 0,
        }
        strides: list[int] = []
        for (kind, _owner, _fd, _gen), events in streams.items():
            events.sort(key=lambda x: x[0])
            if events:
                out["n_streams"] += 1
            prev_last = None
            stream_transitions = 0
            for _ts, off, size, is_append in events:
                if kind == "write" and is_append:
                    out[kind]["append_ops"] += 1
                    prev_last = None
                    continue
                last_byte = off + size - 1
                if prev_last is not None:
                    stream_transitions += 1
                    if off == prev_last + 1:
                        out[kind]["consecutive"] += 1
                    elif off > prev_last + 1:
                        out[kind]["sequential_gap"] += 1
                        strides.append(off - prev_last - 1)
                    else:
                        out[kind]["backward_or_random"] += 1
                prev_last = last_byte
            if stream_transitions:
                out["n_streams_with_transitions"] += 1
        for kind in ("read", "write"):
            c = out[kind]
            transitions = c.get("consecutive", 0) + c.get("sequential_gap", 0) + c.get("backward_or_random", 0)
            seq_ops = c.get("consecutive", 0) + c.get("sequential_gap", 0)
            if kind == "write":
                seq_ops += c.get("append_ops", 0)
            rand_ops = c.get("backward_or_random", 0)
            c["transitions"] = transitions
            c["seq_ops"] = seq_ops
            c["rand_ops"] = rand_ops
            c["pct_consecutive"] = 100.0 * c.get("consecutive", 0) / transitions if transitions else None
            c["pct_sequential_including_gap"] = (
                100.0 * (c.get("consecutive", 0) + c.get("sequential_gap", 0)) / transitions
                if transitions else None
            )
            c["pct_backward_or_random"] = 100.0 * c.get("backward_or_random", 0) / transitions if transitions else None
            out[kind] = dict(c)
        out["stride_hist"] = darshan_hist(strides) if darshan_hist else {}
        return out

    def mergeability(
        streams: dict[tuple[str, int, int, int], list[tuple[float, int, int, bool]]],
        max_chunk_bytes: int = 4 * 1024 * 1024,
    ) -> dict[str, Any]:
        actual_ops = 0
        merged_ops = 0
        saved_ops = 0
        total_bytes = 0
        bytes_in_consecutive_runs = 0

        def finish_run(n_ops: int, n_bytes: int) -> None:
            nonlocal merged_ops, saved_ops, bytes_in_consecutive_runs
            if n_ops <= 0:
                return
            chunks = max(1, math.ceil(n_bytes / max_chunk_bytes))
            merged_ops += chunks
            saved_ops += max(0, n_ops - chunks)
            if n_ops >= 2:
                bytes_in_consecutive_runs += n_bytes

        for (_kind, _owner, _fd, _gen), events in streams.items():
            events.sort(key=lambda x: x[0])
            run_ops = 0
            run_bytes = 0
            prev_last = None
            for _ts, off, size, _is_append in events:
                actual_ops += 1
                total_bytes += size
                last_byte = off + size - 1
                if prev_last is not None and off == prev_last + 1:
                    run_ops += 1
                    run_bytes += size
                else:
                    finish_run(run_ops, run_bytes)
                    run_ops = 1
                    run_bytes = size
                prev_last = last_byte
            finish_run(run_ops, run_bytes)

        return {
            "actual_ops_with_offset": actual_ops,
            "merged_ops_if_consecutive_runs_capped_at_4mb": merged_ops if actual_ops else None,
            "saved_ops": saved_ops if actual_ops else None,
            "saved_ops_pct_of_actual_ops": 100.0 * saved_ops / actual_ops if actual_ops else None,
            "bytes_total_with_offset": total_bytes,
            "bytes_in_consecutive_runs": bytes_in_consecutive_runs,
            "bytes_in_consecutive_runs_pct": (
                100.0 * bytes_in_consecutive_runs / total_bytes if total_bytes else None
            ),
            "max_merge_chunk_bytes": max_chunk_bytes,
            "note": (
                "Only adjacent operations with explicit offsets in the same "
                "(tid, fd, open_generation) stream are considered mergeable; "
                "runs are split at 4 MiB boundaries."
            ),
        }

    pct_with_offset = 100.0 * ops_with_offset / eligible_ops if eligible_ops else None
    tid_classified = classify_streams(streams_tid)
    pid_classified = classify_streams(streams_pid)
    four_cell = {
        "seq_read": (tid_classified.get("read") or {}).get("seq_ops", 0),
        "rand_read": (tid_classified.get("read") or {}).get("rand_ops", 0),
        "seq_write": (tid_classified.get("write") or {}).get("seq_ops", 0),
        "rand_write": (tid_classified.get("write") or {}).get("rand_ops", 0),
    }
    four_total = sum(four_cell.values())
    four_cell_pct = {
        key: (100.0 * val / four_total if four_total else None)
        for key, val in four_cell.items()
    }
    note = (
        "sequentiality uses workload storage data syscalls with kernel/file offsets; "
        "read and write streams are classified separately by fd open generation. "
        "mmap reads are invisible, buffered I/O is measured at the POSIX/VFS layer, "
        "and logical offsets do not imply physical layout."
    )
    if not ops_with_offset:
        note += " No offset-bearing data operations were available in this parsed trace."
    return {
        "eligible_data_ops": eligible_ops,
        "ops_with_offset": ops_with_offset,
        "ops_with_offset_by_kind": {
            "read": int(ops_with_offset_by_kind.get("read", 0)),
            "write": int(ops_with_offset_by_kind.get("write", 0)),
        },
        "pct_ops_with_offset": pct_with_offset,
        "excluded_ops": dict(excluded),
        "four_cell": four_cell,
        "four_cell_pct": four_cell_pct,
        "by_stream_tid_fd_open_generation": tid_classified,
        "by_stream_pid_fd_open_generation": pid_classified,
        "mergeability": mergeability(streams_tid),
        "note": note,
    }


def compute_access_type_rhwhrw(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Axis 1: read-heavy / write-heavy / read-write file classification by
    read/write byte volume (Patel FAST'20 Fig. 3a). Distinct from the lineage
    reuse_class (which splits by reader-call count): this splits by which
    direction the *bytes* flow. A file's read share = rb/(rb+wb); RH if >=2/3,
    WH if <=1/3, RW otherwise."""
    counts = Counter()
    bytes_by_class = defaultdict(int)
    per_class_files = defaultdict(int)
    classified = 0
    for a in artifacts:
        try:
            rb = int(float(a.get("total_read_bytes") or 0))
            wb = int(float(a.get("total_write_bytes") or 0))
        except (TypeError, ValueError):
            continue
        if rb <= 0 and wb <= 0:
            continue
        classified += 1
        share = rb / (rb + wb) if (rb + wb) else 0.0
        cls = "read_heavy" if share >= 2 / 3 else "write_heavy" if share <= 1 / 3 else "read_write"
        counts[cls] += 1
        per_class_files[cls] += 1
        bytes_by_class[cls] += rb + wb
    return {
        "n_files_classified": classified,
        "by_class_files": {
            "read_heavy": counts.get("read_heavy", 0),
            "write_heavy": counts.get("write_heavy", 0),
            "read_write": counts.get("read_write", 0),
        },
        "by_class_pct": {
            c: round(100.0 * n / classified, 1) if classified else 0.0
            for c, n in {
                "read_heavy": counts.get("read_heavy", 0),
                "write_heavy": counts.get("write_heavy", 0),
                "read_write": counts.get("read_write", 0),
            }.items()
        } if classified else {},
        "by_class_bytes": {
            "read_heavy": bytes_by_class.get("read_heavy", 0),
            "write_heavy": bytes_by_class.get("write_heavy", 0),
            "read_write": bytes_by_class.get("read_write", 0),
        },
        "definition": "read_share=rb/(rb+wb); RH>=2/3, WH<=1/3, RW otherwise",
    }


def inter_arrival_deltas(parsed: dict[str, Any],
                         artifacts: list[dict[str, Any]]) -> tuple[list[float], int]:
    """Raw per-file inter-access time gaps (seconds) and the number of files
    that were re-accessed. An access is any data syscall carrying a path;
    consecutive identical timestamps on one file (buffered reads on one fd)
    collapse to a single point so we measure logical re-access gaps, not
    per-syscall noise. Shared by compute_inter_arrival (percentiles) and the
    inter-arrival CDF figure (full distribution).

    Workload data only: otherwise repeated .venv library reads dominate the
    re-access gaps and describe interpreter behaviour, not the workload."""
    _wl = make_workload_filter(artifacts)
    ts_by_path: dict[str, list[float]] = defaultdict(list)
    for e in parsed.get("fs_entries", []):
        if str(e.get("syscall")) not in DATA_SYSCALLS:
            continue
        path = e.get("path")
        ts = e.get("timestamp")
        if not isinstance(path, str) or not isinstance(ts, str):
            continue
        if not _wl(path):
            continue
        t = _entry_end_s(e)
        if t is None:
            continue
        ts_by_path[path].append(t)
    deltas: list[float] = []
    files_reaccessed = 0
    for stamps in ts_by_path.values():
        uniq = sorted(set(stamps))
        if len(uniq) < 2:
            continue
        files_reaccessed += 1
        deltas.extend(uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1))
    return deltas, files_reaccessed


def compute_inter_arrival(parsed: dict[str, Any],
                          artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Axis 1: distribution of the time gap (seconds) between successive
    accesses to the same file (Patel FAST'20 Fig. 5a/6b analogue)."""
    deltas, files_reaccessed = inter_arrival_deltas(parsed, artifacts)
    return {
        "files_with_repeat_access": files_reaccessed,
        "n_intervals": len(deltas),
        "hist": page_cache_time_hist(deltas),
        "hist_bins": [label for _, _, label in PAGE_CACHE_TIME_BINS],
        "hist_note": (
            "Time bins are chosen for page-cache/writeback semantics, not from "
            "the cited inter-arrival literature."
        ),
        "p50_s": percentile(deltas, 50),
        "p95_s": percentile(deltas, 95),
        "p99_s": percentile(deltas, 99),
        "mean_s": (sum(deltas) / len(deltas)) if deltas else None,
        "pct_lt_1s": pct(deltas, lambda x: x < 1.0),
    }


def compute_exploration_overhead(bytes_ops_by_phase: dict[str, Any],
                                 artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Axis 3: assemble the single exploratory-I/O overhead ratio the pieces of
    which already exist. Exploration bytes = I/O in backtrack phases (retry that
    re-does a step) + dead-write bytes (files written but never read = abandoned
    artifacts). Denominator = all data bytes seen across phases.

    Overlap caveat: the two terms can double-count a write that is BOTH inside a
    backtrack phase AND never read (a dead write produced during a retry). This
    is not subtracted here, so exploration_bytes is a slight upper bound in that
    corner case. In practice traces without backtrack tagging have
    backtrack_bytes == 0, so no overlap arises."""
    total_bytes = 0
    backtrack_bytes = 0
    for ph, d in bytes_ops_by_phase.items():
        b = int(d.get("read_bytes", 0)) + int(d.get("write_bytes", 0))
        total_bytes += b
        if "action_unit_backtrack" in str(ph):
            backtrack_bytes += b
    dead_write_bytes = 0
    for a in artifacts:
        if a.get("reuse_class") == "dead_write":
            try:
                dead_write_bytes += int(float(a.get("total_write_bytes") or 0))
            except (TypeError, ValueError):
                pass
    exploration_bytes = backtrack_bytes + dead_write_bytes
    return {
        "backtrack_phase_bytes": backtrack_bytes,
        "dead_write_bytes": dead_write_bytes,
        "exploration_bytes": exploration_bytes,
        "total_data_bytes": total_bytes,
        "exploration_overhead_ratio": (
            exploration_bytes / total_bytes if total_bytes else None
        ),
        "note": (
            "exploration = backtrack-phase I/O + never-read (dead) writes; "
            "GenoMAS-style backtrack tagging required for the first term"
        ),
    }


def _binned_series(parsed: dict[str, Any], window_s: float,
                   timeline: dict[str, Any] | None,
                   syscalls: set[str],
                   artifacts: list[dict[str, Any]] | None = None) -> list[float]:
    """Bytes per fixed time window over the run, for the given syscall set."""
    wl = make_workload_filter(artifacts or []) if artifacts is not None else None
    points: list[tuple[float, float]] = []
    for e in parsed.get("fs_entries", []):
        if str(e.get("syscall")) not in syscalls:
            continue
        if wl is not None and not wl(e.get("path") or ""):
            continue
        t = _entry_end_s(e)
        if t is None:
            continue
        sz = e.get("actual_size") or e.get("requested_size") or e.get("bytes_transferred") or 0
        points.append((t, float(sz) if isinstance(sz, (int, float)) and sz > 0 else 0.0))
    if not points and not timeline:
        return []
    if timeline and timeline.get("wall_s"):
        t0 = float(timeline.get("wall_start_ms") or 0.0) / 1000.0
        t1 = t0 + float(timeline.get("wall_s") or 0.0)
    else:
        t0 = min(t for t, _ in points)
        t1 = max(t for t, _ in points)
    nbins = max(1, int((t1 - t0) / window_s) + 1)
    series = [0.0] * nbins
    for t, sz in points:
        idx = int((t - t0) / window_s)
        if 0 <= idx < nbins:
            series[idx] += sz
    return series


def _autocorr(series: list[float], lag: int) -> float | None:
    n = len(series)
    if n <= lag or n < 2:
        return None
    mean = sum(series) / n
    var = sum((x - mean) ** 2 for x in series)
    if var == 0:
        return None
    cov = sum((series[i] - mean) * (series[i + lag] - mean) for i in range(n - lag))
    return cov / var


def compute_io_autocorrelation(parsed: dict[str, Any],
                               artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Axis 5: read/write autocorrelation at 1/5/25-minute windows and a few
    lags, plus lag-0 read-write cross-correlation (Patel SC'19 Fig. 9)."""
    windows = {"1min": 60.0, "5min": 300.0, "25min": 1500.0}
    lags = [1, 2, 3]
    out: dict[str, Any] = {}
    for name, w in windows.items():
        r = _binned_series(parsed, w, None, READ_SYSCALLS_STRICT, artifacts)
        wr = _binned_series(parsed, w, None, WRITE_SYSCALLS_STRICT, artifacts)
        # Cross-correlation needs aligned bins; rebin both on the same grid.
        cross = None
        m = min(len(r), len(wr))
        if m >= 2:
            a, b = r[:m], wr[:m]
            ma, mb = sum(a) / m, sum(b) / m
            va = sum((x - ma) ** 2 for x in a)
            vb = sum((x - mb) ** 2 for x in b)
            if va > 0 and vb > 0:
                cov = sum((a[i] - ma) * (b[i] - mb) for i in range(m))
                cross = cov / (va ** 0.5 * vb ** 0.5)
        out[name] = {
            "n_bins_read": len(r),
            "n_bins_write": len(wr),
            "read_autocorr": {f"lag{l}": _autocorr(r, l) for l in lags},
            "write_autocorr": {f"lag{l}": _autocorr(wr, l) for l in lags},
            "read_write_xcorr_lag0": cross,
        }
    return out


def compute_intensity_phases(parsed: dict[str, Any],
                             timeline: dict[str, Any] | None = None,
                             window_s: float = 60.0) -> dict[str, Any]:
    """Axis 5: high/low-intensity I/O phase segmentation. Bin total I/O bytes
    into fixed windows, threshold at the 75th/25th percentile of non-empty bins,
    and segment consecutive high / low bins (Patel SC'19 Fig. 7/8)."""
    series = _binned_series(parsed, window_s, timeline,
                            READ_SYSCALLS_STRICT | WRITE_SYSCALLS_STRICT)
    nonzero = [x for x in series if x > 0]
    if len(nonzero) < 4:
        return {"n_bins": len(series), "note": "too few active bins to segment"}
    hi = percentile(nonzero, 75)
    lo = percentile(nonzero, 25)

    def _segments(mask: list[bool]) -> list[int]:
        segs, run = [], 0
        for m in mask:
            if m:
                run += 1
            elif run:
                segs.append(run)
                run = 0
        if run:
            segs.append(run)
        return segs

    hi_segs = _segments([x >= hi for x in series])
    lo_segs = _segments([0 < x <= lo for x in series])
    return {
        "window_s": window_s,
        "n_bins": len(series),
        "high_threshold_bytes": hi,
        "low_threshold_bytes": lo,
        "high_phases": {
            "count": len(hi_segs),
            "mean_len_bins": (sum(hi_segs) / len(hi_segs)) if hi_segs else None,
            "max_len_bins": max(hi_segs) if hi_segs else 0,
        },
        "low_phases": {
            "count": len(lo_segs),
            "mean_len_bins": (sum(lo_segs) / len(lo_segs)) if lo_segs else None,
            "max_len_bins": max(lo_segs) if lo_segs else 0,
        },
    }


def _entry_interval_ms(entry: dict[str, Any]) -> tuple[float, float] | None:
    end_ms = _entry_end_ms(entry)
    if end_ms is None:
        return None
    dur_ms = max(0.0, float(entry.get("duration") or 0.0) * 1000.0)
    return end_ms - dur_ms, end_ms


def _union_length_ms(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    cur_s = cur_e = None
    for s, e in sorted((s, e) for s, e in intervals if e > s):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None and cur_e is not None:
        total += cur_e - cur_s
    return total


def _merged_intervals_ms(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for s, e in sorted((s, e) for s, e in intervals if e > s):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _point_in_intervals_ms(t: float, intervals: list[tuple[float, float]]) -> bool:
    return any(s <= t <= e for s, e in intervals)


def _interval_overlap_ms(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _io_bytes(entry: dict[str, Any]) -> int:
    value = entry.get("bytes_transferred") or entry.get("actual_size") or entry.get("requested_size") or 0
    return int(value) if isinstance(value, (int, float)) and value > 0 else 0


def _fs_entry_bounds_ms(parsed: dict[str, Any]) -> tuple[float, float] | None:
    bounds: list[float] = []
    for e in parsed.get("fs_entries", []):
        iv = _entry_interval_ms(e)
        if iv is not None:
            bounds.extend(iv)
    if not bounds:
        return None
    return min(bounds), max(bounds)


def _llm_intervals_ms(trace_dir: Path) -> list[tuple[float, float]]:
    events_path = trace_dir / "pi_events.jsonl"
    if not events_path.is_file():
        return []
    try:
        from agent_io_tracing.analysis.parallelism import _load_llm_events

        llms, _, _ = _load_llm_events(events_path)
    except Exception:
        return []
    return [(ev.start_ms, ev.end_ms) for ev in llms if ev.end_ms > ev.start_ms]


def _run_timeline_ms(
    trace_dir: Path,
    parsed: dict[str, Any],
    phases: dict[str, dict[str, Any]],
    tool_calls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """One wall-clock timeline shared by inference and bandwidth metrics."""
    events: dict[str, Any] = {}
    try:
        from agent_io_tracing.analysis.parallelism import load_events

        events = load_events(trace_dir)
    except Exception:
        events = {}

    llm_intervals: list[tuple[float, float]] = []
    tool_intervals: list[tuple[float, float]] = []
    phase_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    role_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for ev in events.values():
        s = float(getattr(ev, "start_ms", 0.0))
        e = float(getattr(ev, "end_ms", 0.0))
        if e <= s:
            continue
        if getattr(ev, "kind", None) == "llm":
            llm_intervals.append((s, e))
            continue
        if getattr(ev, "kind", None) != "tool":
            continue
        rid = getattr(ev, "run_id", None)
        if not isinstance(rid, str):
            continue
        tc = tool_calls.get(rid) or {
            "tool_id": rid,
            "tool_name": getattr(ev, "name", None),
            "name": getattr(ev, "name", None),
            "role": getattr(ev, "role", None),
            "input_params": getattr(ev, "args", None) or {},
        }
        iv = (s, e)
        tool_intervals.append(iv)
        if phase_label_for_tool_call:
            phase_intervals[phase_label_for_tool_call(rid, tc, phases)].append(iv)
        if role_for_tool_call:
            role_intervals[role_for_tool_call(rid, tc, phases)].append(iv)

    event_bounds = [
        t
        for iv in [*llm_intervals, *tool_intervals]
        for t in iv
    ]
    fs_bounds = _fs_entry_bounds_ms(parsed)
    timestamp_has_ts_ms = any(
        isinstance(e.get("ts_ms"), (int, float))
        for e in parsed.get("fs_entries", [])
    )
    if fs_bounds and event_bounds:
        fs_start, fs_end = fs_bounds
        ev_start, ev_end = min(event_bounds), max(event_bounds)
        overlap_ms = min(fs_end, ev_end) - max(fs_start, ev_start)
        if overlap_ms < -1000.0:
            skew_ms = ev_start - fs_start
            source = "ts_ms" if timestamp_has_ts_ms else "naive ISO timestamp"
            raise RuntimeError(
                "Filesystem event timestamps and LLM/tool event timestamps are "
                f"on different timebases (source={source}; fs=[{fs_start:.0f}, {fs_end:.0f}] ms, "
                f"events=[{ev_start:.0f}, {ev_end:.0f}] ms, approx skew={skew_ms/1000.0:.1f}s). "
                "Re-parse ebpf_events.log with a parser that writes absolute ts_ms; "
                "do not use inference-overlap metrics from this parsed.json."
            )
    if fs_bounds:
        event_bounds.extend(fs_bounds)

    wall_start = min(event_bounds) if event_bounds else 0.0
    summary = load_json(trace_dir / "parallelism_summary.json", {})
    wall_clock_s = summary.get("wall_clock_s")
    if isinstance(wall_clock_s, (int, float)) and wall_clock_s > 0 and event_bounds:
        wall_end = wall_start + float(wall_clock_s) * 1000.0
    elif event_bounds:
        wall_end = max(event_bounds)
        wall_clock_s = max(0.0, (wall_end - wall_start) / 1000.0)
    else:
        wall_end = wall_start
        wall_clock_s = 0.0

    tool_busy_s = _union_length_ms(tool_intervals) / 1000.0
    orchestration_s = max(0.0, float(wall_clock_s or 0.0) - tool_busy_s)
    if orchestration_s > 0:
        phase_intervals.setdefault("orchestration", [])
        role_intervals.setdefault("(unattributed)", [])

    return {
        "wall_start_ms": wall_start,
        "wall_end_ms": wall_end,
        "wall_s": float(wall_clock_s or 0.0),
        "llm_intervals": _merged_intervals_ms(llm_intervals),
        "tool_intervals": _merged_intervals_ms(tool_intervals),
        "phase_intervals": {k: _merged_intervals_ms(v) for k, v in phase_intervals.items()},
        "role_intervals": {k: _merged_intervals_ms(v) for k, v in role_intervals.items()},
        "phase_wall_s": {
            k: _union_length_ms(v) / 1000.0 for k, v in phase_intervals.items()
        } | {"orchestration": orchestration_s},
        "role_wall_s": {
            k: _union_length_ms(v) / 1000.0 for k, v in role_intervals.items()
        } | {"(unattributed)": orchestration_s},
    }


def compute_io_vs_inference(
    trace_dir: Path,
    parsed: dict[str, Any],
    artifacts: list[dict[str, Any]],
    timeline: dict[str, Any],
) -> dict[str, Any]:
    llm_intervals = timeline.get("llm_intervals") or []
    wl = make_workload_filter(artifacts)
    io_events: list[tuple[float, int]] = []
    for e in parsed.get("fs_entries", []):
        if str(e.get("syscall")) not in (READ_SYSCALLS_STRICT | WRITE_SYSCALLS_STRICT):
            continue
        path = e.get("path") or ""
        if not wl(path):
            continue
        size = _io_bytes(e)
        if size <= 0:
            continue
        iv = _entry_interval_ms(e)
        if iv is None:
            continue
        _, end = iv
        io_events.append((end, size))
    wall_start = float(timeline.get("wall_start_ms") or 0.0)
    wall_end = float(timeline.get("wall_end_ms") or wall_start)
    wall_ms = max(float(timeline.get("wall_s") or 0.0) * 1000.0, wall_end - wall_start)
    if wall_ms <= 0:
        return {
            "pct_time_in_inference": None,
            "pct_bytes_during_inference": None,
            "bandwidth_ratio": None,
            "inference_gap_s": {"count": 0, "p50": None, "p95": None, "max": None},
        }
    inference_ms = _union_length_ms(llm_intervals)
    bytes_busy = sum(size for t, size in io_events if _point_in_intervals_ms(t, llm_intervals))
    total_bytes = sum(size for _, size in io_events)
    idle_ms = max(wall_ms - inference_ms, 0.0)
    busy_bw = bytes_busy / (inference_ms / 1000.0) if inference_ms > 0 else None
    idle_bytes = total_bytes - bytes_busy
    idle_bw = idle_bytes / (idle_ms / 1000.0) if idle_ms > 0 else None
    gaps: list[float] = []
    cursor = wall_start
    for s, e in llm_intervals:
        if s > cursor:
            gaps.append((s - cursor) / 1000.0)
        cursor = max(cursor, e)
    if wall_end > cursor:
        gaps.append((wall_end - cursor) / 1000.0)
    return {
        "pct_time_in_inference": 100.0 * inference_ms / wall_ms if wall_ms else None,
        "pct_bytes_during_inference": 100.0 * bytes_busy / total_bytes if total_bytes else None,
        "bandwidth_ratio": (
            busy_bw / idle_bw
            if isinstance(busy_bw, (int, float)) and isinstance(idle_bw, (int, float)) and idle_bw > 0
            else None
        ),
        "bytes_during_inference": bytes_busy,
        "bytes_outside_inference": idle_bytes,
        "inference_gap_s": {
            "count": len(gaps),
            "p50": percentile(gaps, 50),
            "p95": percentile(gaps, 95),
            "max": max(gaps) if gaps else None,
        },
        "caveat": (
            "workload data only; bytes are attributed by syscall end timestamp, "
            "so a long operation crossing an LLM boundary is counted wholly on "
            "the side where it returns."
        ),
    }


MIN_EFFECTIVE_BW_OPS = 5
MIN_EFFECTIVE_BW_IO_TIME_S = 1e-3


def _bandwidth_stats(items: list[tuple[float, float, int]],
                     wall_s: float | None = None) -> dict[str, Any]:
    if not items:
        return {"ops": 0, "bytes": 0, "io_time_s": 0.0, "busy_time_s": 0.0,
                "wall_s": float(wall_s or 0.0), "effective_Bps": None, "aggregate_Bps": None,
                "duty_cycle": None, "effective_reliable": False,
                "effective_unreliable_reason": "no operations"}
    intervals = [(s, e) for s, e, _ in items]
    bytes_total = sum(size for _, _, size in items)
    io_time_s = sum(max(e - s, 0.0) for s, e, _ in items) / 1000.0
    busy_s = _union_length_ms(intervals) / 1000.0
    if wall_s is None:
        wall_s = (max(e for _, e, _ in items) - min(s for s, _, _ in items)) / 1000.0
    effective_reliable = len(items) >= MIN_EFFECTIVE_BW_OPS and io_time_s >= MIN_EFFECTIVE_BW_IO_TIME_S
    unreliable_reason = None
    if not effective_reliable:
        if len(items) < MIN_EFFECTIVE_BW_OPS:
            unreliable_reason = f"ops<{MIN_EFFECTIVE_BW_OPS}"
        if io_time_s < MIN_EFFECTIVE_BW_IO_TIME_S:
            suffix = f"io_time_s<{MIN_EFFECTIVE_BW_IO_TIME_S:g}"
            unreliable_reason = f"{unreliable_reason}; {suffix}" if unreliable_reason else suffix
    return {
        "ops": len(items),
        "bytes": bytes_total,
        "io_time_s": io_time_s,
        "busy_time_s": busy_s,
        "wall_s": wall_s,
        "effective_Bps": bytes_total / io_time_s if effective_reliable else None,
        "aggregate_Bps": bytes_total / busy_s if busy_s > 0 else None,
        "duty_cycle": busy_s / wall_s if wall_s > 0 else None,
        "effective_reliable": effective_reliable,
        "effective_unreliable_reason": unreliable_reason,
    }


def compute_effective_bandwidth(
    trace_dir: Path,
    parsed: dict[str, Any],
    phases: dict[str, dict[str, Any]],
    artifacts: list[dict[str, Any]],
    timeline: dict[str, Any],
) -> dict[str, Any]:
    tool_calls = build_tool_call_map(parsed)
    wl = make_workload_filter(artifacts)
    llm_intervals = timeline.get("llm_intervals") or []
    by_phase: dict[str, dict[str, list[tuple[float, float, int]]]] = defaultdict(lambda: {"read": [], "write": []})
    by_role: dict[str, dict[str, list[tuple[float, float, int]]]] = defaultdict(lambda: {"read": [], "write": []})
    by_inference: dict[str, dict[str, list[tuple[float, float, int]]]] = defaultdict(lambda: {"read": [], "write": []})
    global_items: dict[str, list[tuple[float, float, int]]] = {"read": [], "write": []}
    for e in parsed.get("fs_entries", []):
        syscall = str(e.get("syscall"))
        if syscall not in (READ_SYSCALLS_STRICT | WRITE_SYSCALLS_STRICT):
            continue
        path = e.get("path") or ""
        if not wl(path):
            continue
        size = _io_bytes(e)
        iv = _entry_interval_ms(e)
        if size <= 0 or iv is None:
            continue
        kind = "write" if syscall in WRITE_SYSCALLS_STRICT else "read"
        item = (iv[0], iv[1], size)
        phase = phase_for_entry(e, phases, tool_calls)
        role = (
            role_for_entry(e, tool_calls, phases)
            if role_for_entry else "(unattributed)"
        )
        inf_state = "inference_busy" if _point_in_intervals_ms(iv[1], llm_intervals) else "inference_idle"
        by_phase[phase][kind].append(item)
        by_role[role][kind].append(item)
        by_inference[inf_state][kind].append(item)
        global_items[kind].append(item)

    phase_wall_s = timeline.get("phase_wall_s") or {}
    role_wall_s = timeline.get("role_wall_s") or {}
    run_wall_s = float(timeline.get("wall_s") or 0.0)
    inference_wall_s = _union_length_ms(llm_intervals) / 1000.0
    inference_walls = {
        "inference_busy": inference_wall_s,
        "inference_idle": max(0.0, run_wall_s - inference_wall_s),
    }

    def finish(
        groups: dict[str, dict[str, list[tuple[float, float, int]]]],
        walls: dict[str, float],
    ) -> dict[str, Any]:
        return {
            name: {
                "read": _bandwidth_stats(vals.get("read", []), walls.get(name)),
                "write": _bandwidth_stats(vals.get("write", []), walls.get(name)),
            }
            for name, vals in sorted(groups.items())
        }

    return {
        "global": {
            "read": _bandwidth_stats(global_items["read"], run_wall_s),
            "write": _bandwidth_stats(global_items["write"], run_wall_s),
        },
        "by_phase": finish(by_phase, phase_wall_s),
        "by_role": finish(by_role, role_wall_s),
        "by_inference_state": finish(by_inference, inference_walls),
        "caveat": (
            "syscall duration is application-observed return latency with page cache; "
            "write may return before durable media, and cache-hit reads need not touch disk. "
            "Bytes are attributed by syscall end timestamp. Duty cycle is wall-clock "
            "storage-busy time / group wall-time union, not worker-time share."
        ),
    }


def verdicts(metrics: dict[str, Any]) -> list[dict[str, str]]:
    iface = metrics.get("interface_mix") or {}
    req = metrics.get("request_size_cdf") or {}
    ratios = metrics.get("metadata_data_ratio") or {}
    fsz = metrics.get("file_size_cdf") or {}
    seq = metrics.get("sequentiality") or {}

    layer_counts = iface.get("layer_exec_counts") or {}
    io_execs = iface.get("execs_with_file_io") or 0
    stdio_posix_pct = (
        100.0 * (layer_counts.get("stdio", 0) + layer_counts.get("posix_raw", 0)) / io_execs
        if io_execs else None
    )
    structured_pct = iface.get("pct_structured_any")
    metadata_ratio = ratios.get("storage_metadata_to_data_ops")
    small_10mb = req.get("pct_lt_10mb")
    sub4k = req.get("pct_lt_4kb")
    file_lt_1gb = fsz.get("pct_lt_1gb")

    out = []
    out.append({
        "metric": "Interface mix",
        "target": "Luu HPDC'15 POSIX-dominant jobs",
        "verdict": (
            "different: text/POSIX-heavy and near-zero structured I/O"
            if stdio_posix_pct is not None and stdio_posix_pct >= 75 and (structured_pct or 0) < 10
            else "same: POSIX-like dominance" if stdio_posix_pct and stdio_posix_pct >= 75
            else "unknown/weak"
        ),
    })
    out.append({
        "metric": "Request-size CDF",
        "target": "Paul MASCOTS'21 most requests <10MB",
        "verdict": (
            "different: most requests are sub-4KB"
            if sub4k is not None and sub4k >= 50
            else "same: small-request regime (<10MB)"
            if small_10mb is not None and small_10mb >= 50
            else "unknown/weak"
        ),
    })
    out.append({
        "metric": "Metadata/data ratio",
        "target": "Luu metadata-heavy job fraction",
        "verdict": (
            "different: metadata storm relative to data ops"
            if metadata_ratio is not None and metadata_ratio > 1.0
            else "same: data-op dominated"
            if metadata_ratio is not None
            else "unknown/weak"
        ),
    })
    out.append({
        "metric": "File size / bytes per task",
        "target": "Luu Fig.2 Edison Lustre file-size distribution",
        "verdict": (
            "different: many small files"
            if fsz.get("pct_lt_1mb") is not None and fsz["pct_lt_1mb"] >= 50
            else "same: many files below 1GB"
            if file_lt_1gb is not None and file_lt_1gb >= 50
            else "unknown/weak"
        ),
    })
    out.append({
        "metric": "Sequential vs random/consecutive",
        "target": "Darshan sequential/consecutive ratios",
        "verdict": (
            "same: mostly consecutive"
            if seq.get("pct_consecutive") is not None and seq["pct_consecutive"] >= 80
            else "different: gap/backward seeks visible"
            if seq.get("pct_consecutive") is not None
            else "unknown: insufficient offset-bearing syscalls"
        ),
    })
    out.append({
        "metric": "Text vs binary fraction",
        "target": "Luu text-only app fraction",
        "verdict": (
            "different: STDIO/text dominates generated code"
            if iface.get("pct_stdio_only") is not None and iface["pct_stdio_only"] >= 50
            else "same/unknown"
        ),
    })
    return out


def write_markdown(trace_dir: Path, metrics: dict[str, Any]) -> None:
    lines = ["# Phase-1 Metrics", ""]
    six = metrics["six_numbers"]
    for key, value in six.items():
        lines.append(f"- **{key}**: `{json.dumps(value, ensure_ascii=False)}`")
    lines.extend(["", "## Same/Different", ""])
    for row in metrics["comparison_verdicts"]:
        lines.append(f"- **{row['metric']}** vs {row['target']}: {row['verdict']}")

    lines.extend(["", "## Attribution (Overview.md §5)", ""])

    iface = metrics.get("interface_mix") or {}
    lines.append(f"- **I/O interface mix**: `{json.dumps(iface, ensure_ascii=False)}`")
    lines.append(
        f"- **Measured interface layers (uprobe/syscall)**: "
        f"`{json.dumps(metrics.get('measured_interface_layers'), ensure_ascii=False)}`"
    )
    reread = metrics.get("reread_attribution") or {}
    ai = reread.get("agent_induced") or {}
    lines.append(
        f"- **Reread attribution**: agent-induced same-step reopen = "
        f"`{json.dumps(ai.get('same_step_reopen'), ensure_ascii=False)}`; "
        f"agent-induced reread-after-backtrack = "
        f"`{json.dumps(ai.get('reread_after_backtrack'), ensure_ascii=False)}`; "
        f"different tool_call_id (residual) = "
        f"`{json.dumps(reread.get('different_tool_call_id'), ensure_ascii=False)}`"
    )

    lines.extend(["", "## Agent-Induced I/O (Overview.md §6.2 rollups)", ""])
    lines.append(f"- **Bytes/ops by phase**: `{json.dumps(metrics.get('bytes_ops_by_phase'), ensure_ascii=False)}`")
    lines.append(f"- **Directory scans**: `{json.dumps(metrics.get('directory_scan'), ensure_ascii=False)}`")
    lines.append(f"- **Failed open/stat**: `{json.dumps(metrics.get('failed_open_stat'), ensure_ascii=False)}`")
    lines.append(f"- **Error-log reads**: `{json.dumps(metrics.get('error_log_reads'), ensure_ascii=False)}`")

    lines.extend(["", "## Task-Misconfigured I/O (Overview.md §6.3 rollups)", ""])
    lines.append(
        f"- **State file rewrite frequency**: "
        f"`{json.dumps(metrics.get('state_file_rewrite_frequency'), ensure_ascii=False)}`"
    )

    lines.append("")
    (trace_dir / "phase1_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def build_metrics(trace_dir: Path, optimal_request_bytes: int) -> dict[str, Any]:
    parsed = load_json(trace_dir / "parsed.json", {})
    pi_summary = load_json(trace_dir / "pi_summary.json", {})
    manifest = load_json(trace_dir / "manifest.json", {})
    lineage = load_json(trace_dir / "lineage" / "io_summary.json", {})
    artifacts = read_artifacts(trace_dir)
    phases = read_event_phase_index(trace_dir)
    tool_calls = build_tool_call_map(parsed)
    timeline = _run_timeline_ms(trace_dir, parsed, phases, tool_calls)

    generated_code = trace_dir / "generated_code.jsonl"
    interface_mix = (
        aggregate_io_api(generated_code) if aggregate_io_api and generated_code.is_file()
        else {}
    )

    metrics = {
        "trace_dir": str(trace_dir),
        "manifest": manifest,
        "metadata_data_ratio": compute_op_ratios(parsed),
        "request_size_cdf": compute_request_size_cdf(parsed),
        "file_size_cdf": compute_file_size_cdf(artifacts),
        "latency_by_phase": compute_latency_by_phase(parsed, phases),
        "fs_io_non_llm": compute_fs_io_non_llm(parsed, pi_summary, manifest),
        "analytical_optimum_amplification": compute_analytical_optimum(
            parsed, artifacts, optimal_request_bytes
        ),
        "interface_mix": interface_mix,
        "interface_byte_mix": compute_interface_byte_mix(parsed, artifacts),
        "measured_interface_layers": compute_measured_interface_layers(parsed, artifacts),
        "namespace": lineage.get("namespace", {}),
        "sequentiality": compute_sequentiality(parsed, artifacts),
        "bytes_ops_by_phase": compute_bytes_ops_by_phase(parsed, phases, artifacts),
        "reread_attribution": compute_reread_attribution(parsed, phases, artifacts),
        "directory_scan": compute_directory_scan_count(parsed, artifacts),
        "failed_open_stat": compute_failed_open_stat_count(parsed),
        "error_log_reads": compute_error_log_reads(artifacts),
        "state_file_rewrite_frequency": compute_state_file_rewrite_frequency(artifacts),
        # Newly implemented axis metrics (data-derived, no oracle required)
        "access_type_rhwhrw": compute_access_type_rhwhrw(artifacts),          # axis 2
        "inter_arrival": compute_inter_arrival(parsed, artifacts),             # axis 2
        "io_autocorrelation": compute_io_autocorrelation(parsed, artifacts),  # axis 6
        "intensity_phases": compute_intensity_phases(parsed, timeline),       # axis 6
        "io_vs_inference": compute_io_vs_inference(trace_dir, parsed, artifacts, timeline),
        "effective_bandwidth": compute_effective_bandwidth(trace_dir, parsed, phases, artifacts, timeline),
    }
    # axis 4: assembled from bytes_ops_by_phase (already computed above) + reuse
    metrics["exploration_overhead"] = compute_exploration_overhead(
        metrics["bytes_ops_by_phase"], artifacts
    )
    metrics["six_numbers"] = {
        "1_metadata_ops_per_data_op": metrics["metadata_data_ratio"].get(
            "storage_metadata_to_data_ops"
        ),
        "2_file_size_cdf": metrics["file_size_cdf"],
        "3_files_created_per_tool_call": (
            (metrics["namespace"].get("files_per_tool_call") or {})
        ),
        "4_latency_p50_p95_p99_by_phase": metrics["latency_by_phase"],
        "5_fs_io_pct_of_non_llm_runtime": (
            metrics["fs_io_non_llm"].get("fs_io_pct_of_non_llm")
        ),
        "6_analytical_optimum_amplification": metrics["analytical_optimum_amplification"],
    }
    metrics["comparison_verdicts"] = verdicts(metrics)
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace_dir", type=Path)
    p.add_argument("--optimal-request-bytes", type=int, default=4 * 1024 * 1024)
    args = p.parse_args()

    trace_dir = args.trace_dir.resolve()
    metrics = build_metrics(trace_dir, args.optimal_request_bytes)
    (trace_dir / "phase1_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(trace_dir, metrics)
    print(f"Wrote {trace_dir / 'phase1_metrics.json'}")
    print(f"Wrote {trace_dir / 'phase1_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
