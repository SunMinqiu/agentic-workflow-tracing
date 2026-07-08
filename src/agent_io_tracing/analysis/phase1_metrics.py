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
except Exception:  # pragma: no cover - script still reports partial metrics
    classify_syscall = None  # type: ignore
    resource_bucket_for_syscall = None  # type: ignore
    aggregate_io_api = None  # type: ignore


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
        if tid in phases and phases[tid].get("phase"):
            role = phases[tid].get("genomas_role")
            return f"{phases[tid]['phase']}:{role}" if role else str(phases[tid]["phase"])
        tc = tool_calls.get(tid)
        if tc:
            inp = tc.get("input_params") or {}
            phase = inp.get("phase") or tc.get("tool_name") or "tool"
            role = inp.get("role")
            return f"{phase}:{role}" if role else str(phase)
        return "uncategorized_tool"
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


def compute_reread_attribution(parsed: dict[str, Any],
                               phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
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
    entries = [
        e for e in parsed.get("fs_entries", [])
        if str(e.get("syscall")) in READ_SYSCALLS_STRICT
        and isinstance(e.get("path"), str)
        and isinstance(e.get("matched_tool_call"), str)
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
                               phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Same phase join as compute_latency_by_phase, but rolling up bytes/op
    counts instead of latency. This is what turns a validated phase tag (e.g.
    ``action_unit_backtrack``) into an actual retry-induced-bytes number."""
    tool_calls = build_tool_call_map(parsed)
    by_phase: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"ops": 0, "read_ops": 0, "write_ops": 0,
                 "read_bytes": 0, "write_bytes": 0}
    )
    for e in parsed.get("fs_entries", []):
        syscall = str(e.get("syscall"))
        if syscall not in DATA_SYSCALLS:
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


def compute_interface_byte_mix(parsed: dict[str, Any]) -> dict[str, Any]:
    """Axis 3 Phase A: measured interface bytes from libc probes + syscalls.

    fread/fwrite uprobes sit above read/write syscalls, so raw STDIO and POSIX
    observations are overlapping layers, not additive. The de-overlapped view
    subtracts observed STDIO bytes from kernel read/write bytes to estimate the
    direct-POSIX lower bound.
    """
    raw = {
        "stdio_read_bytes": 0,
        "stdio_write_bytes": 0,
        "posix_read_bytes_observed": 0,
        "posix_write_bytes_observed": 0,
        "stdio_ops": 0,
        "posix_ops": 0,
    }
    for e in parsed.get("fs_entries", []):
        syscall = str(e.get("syscall") or "")
        size = e.get("actual_size") or e.get("bytes_transferred") or 0
        size = int(size) if isinstance(size, (int, float)) and size > 0 else 0
        if size <= 0:
            continue
        if syscall in STDIO_READ_FUNCS:
            raw["stdio_read_bytes"] += size
            raw["stdio_ops"] += 1
        elif syscall in STDIO_WRITE_FUNCS:
            raw["stdio_write_bytes"] += size
            raw["stdio_ops"] += 1
        elif syscall in READ_SYSCALLS_STRICT:
            raw["posix_read_bytes_observed"] += size
            raw["posix_ops"] += 1
        elif syscall in WRITE_SYSCALLS_STRICT:
            raw["posix_write_bytes_observed"] += size
            raw["posix_ops"] += 1

    stdio_bytes = raw["stdio_read_bytes"] + raw["stdio_write_bytes"]
    posix_observed = raw["posix_read_bytes_observed"] + raw["posix_write_bytes_observed"]
    posix_direct = max(0, posix_observed - stdio_bytes)
    denom = stdio_bytes + posix_direct
    return {
        **raw,
        "stdio_bytes": stdio_bytes,
        "posix_observed_bytes": posix_observed,
        "posix_direct_bytes_est": posix_direct,
        "stdio_pct_deoverlapped": (100.0 * stdio_bytes / denom) if denom else None,
        "posix_direct_pct_deoverlapped": (100.0 * posix_direct / denom) if denom else None,
        "note": (
            "fread/fwrite probes and read/write syscalls are different layers; "
            "deoverlapped POSIX is max(kernel read/write bytes - STDIO bytes, 0)."
        ),
    }


def compute_directory_scan_count(parsed: dict[str, Any]) -> dict[str, Any]:
    entries = [e for e in parsed.get("fs_entries", [])
              if str(e.get("syscall")) in DIRECTORY_SCAN_SYSCALLS]
    by_path = Counter(e.get("path") for e in entries if e.get("path"))
    return {
        "total_scans": len(entries),
        "unique_directories_scanned": len(by_path),
        "rescanned_directories": sum(1 for c in by_path.values() if c > 1),
        "top_rescanned": by_path.most_common(5),
    }


def compute_failed_open_stat_count(parsed: dict[str, Any]) -> dict[str, Any]:
    entries = [
        e for e in parsed.get("fs_entries", [])
        if str(e.get("syscall")) in OPEN_STAT_SYSCALLS
        and isinstance(e.get("return_value"), (int, float))
        and e.get("return_value") < 0
    ]
    by_syscall = Counter(str(e.get("syscall")) for e in entries)
    by_path = Counter(e.get("path") for e in entries if e.get("path"))
    return {
        "total_failed": len(entries),
        "by_syscall": dict(by_syscall),
        "distinct_paths_involved": len(by_path),
        "top_failing_paths": by_path.most_common(5),
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
    for e in parsed.get("fs_entries", []):
        if str(e.get("syscall")) not in DATA_SYSCALLS:
            continue
        sz = e.get("requested_size") or e.get("actual_size") or e.get("bytes_transferred")
        if isinstance(sz, (int, float)) and sz > 0:
            sizes.append(float(sz))
    return {
        "count": len(sizes),
        "p50_bytes": percentile(sizes, 50),
        "p95_bytes": percentile(sizes, 95),
        "p99_bytes": percentile(sizes, 99),
        "pct_lt_4kb": pct(sizes, lambda x: x < 4096),
        "pct_lt_10mb": pct(sizes, lambda x: x < 10 * 1024 * 1024),
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


def compute_analytical_optimum(parsed: dict[str, Any],
                               artifacts: list[dict[str, Any]],
                               optimal_request_bytes: int) -> dict[str, Any]:
    generated = []
    for row in artifacts:
        try:
            writes = int(float(row.get("n_writes") or 0))
            write_bytes = int(float(row.get("total_write_bytes") or 0))
        except ValueError:
            continue
        if writes > 0 or write_bytes > 0:
            generated.append((row.get("path", ""), writes, write_bytes))
    actual_generated_files = len({p for p, _, _ in generated if p})
    actual_write_bytes = sum(wb for _, _, wb in generated)
    actual_write_ops = sum(1 for e in parsed.get("fs_entries", [])
                           if str(e.get("syscall")) in {"write", "pwrite64", "writev", "pwritev"})
    actual_metadata_ops = sum(
        1 for e in parsed.get("fs_entries", [])
        if storage_metadata_syscall(str(e.get("syscall")))
    )
    # Read side (axis 5 small-I/O aggregation potential): mirror the write-side
    # amplification. actual_read_ops / optimum_read_ops answers "how many read
    # calls could be saved if fragmented reads were merged to optimal-size
    # requests," the read analogue of write_call_amplification.
    actual_read_ops = 0
    actual_read_bytes = 0
    for e in parsed.get("fs_entries", []):
        if str(e.get("syscall")) in READ_SYSCALLS_STRICT:
            actual_read_ops += 1
            sz = e.get("actual_size") or e.get("requested_size") or e.get("bytes_transferred") or 0
            actual_read_bytes += sz if isinstance(sz, (int, float)) and sz > 0 else 0
    optimum_read_ops = (
        max(1, math.ceil(actual_read_bytes / optimal_request_bytes))
        if actual_read_bytes else 0
    )
    optimum_files = 1 if actual_generated_files else 0
    optimum_write_ops = (
        max(1, math.ceil(actual_write_bytes / optimal_request_bytes))
        if actual_write_bytes else 0
    )
    # Batched-file lower bound: create/open + close + at least one metadata
    # lookup/attribute update. This is intentionally conservative.
    optimum_metadata_ops = 3 if actual_generated_files else 0
    return {
        "actual_generated_files": actual_generated_files,
        "actual_write_bytes": actual_write_bytes,
        "actual_write_ops": actual_write_ops,
        "actual_storage_metadata_ops": actual_metadata_ops,
        "optimum_files": optimum_files,
        "optimum_write_ops": optimum_write_ops,
        "optimum_storage_metadata_ops": optimum_metadata_ops,
        "file_count_amplification": (
            actual_generated_files / optimum_files if optimum_files else None
        ),
        "write_call_amplification": (
            actual_write_ops / optimum_write_ops if optimum_write_ops else None
        ),
        "actual_read_bytes": actual_read_bytes,
        "actual_read_ops": actual_read_ops,
        "optimum_read_ops": optimum_read_ops,
        "read_call_amplification": (
            actual_read_ops / optimum_read_ops if optimum_read_ops else None
        ),
        "metadata_op_amplification": (
            actual_metadata_ops / optimum_metadata_ops if optimum_metadata_ops else None
        ),
        "assumption": (
            f"analytical optimum is one batched output file with "
            f"{optimal_request_bytes}B read/write requests; no reference run"
        ),
    }


def compute_sequentiality(parsed: dict[str, Any]) -> dict[str, Any]:
    by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for e in parsed.get("fs_entries", []):
        path = e.get("path")
        off = e.get("offset")
        size = e.get("actual_size") or e.get("requested_size")
        if isinstance(path, str) and isinstance(off, int) and isinstance(size, int) and size > 0:
            by_file[path].append((off, size))
    total = consecutive = backward = random = 0
    for events in by_file.values():
        prev_end = None
        for off, size in events:
            if prev_end is not None:
                total += 1
                if off == prev_end:
                    consecutive += 1
                elif off < prev_end:
                    backward += 1
                else:
                    random += 1
            prev_end = off + size
    return {
        "transitions": total,
        "pct_consecutive": 100.0 * consecutive / total if total else None,
        "pct_backward": 100.0 * backward / total if total else None,
        "pct_gap_random": 100.0 * random / total if total else None,
        "note": "requires offset-capable syscalls; read/write without offsets are excluded",
    }


def compute_access_type_rhwhrw(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Axis 2: read-heavy / write-heavy / read-write file classification by
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


def inter_arrival_deltas(parsed: dict[str, Any]) -> tuple[list[float], int]:
    """Raw per-file inter-access time gaps (seconds) and the number of files
    that were re-accessed. An access is any data syscall carrying a path;
    consecutive identical timestamps on one file (buffered reads on one fd)
    collapse to a single point so we measure logical re-access gaps, not
    per-syscall noise. Shared by compute_inter_arrival (percentiles) and the
    inter-arrival CDF figure (full distribution)."""
    ts_by_path: dict[str, list[float]] = defaultdict(list)
    for e in parsed.get("fs_entries", []):
        if str(e.get("syscall")) not in DATA_SYSCALLS:
            continue
        path = e.get("path")
        ts = e.get("timestamp")
        if not isinstance(path, str) or not isinstance(ts, str):
            continue
        try:
            ts_by_path[path].append(datetime.fromisoformat(ts).timestamp())
        except ValueError:
            continue
    deltas: list[float] = []
    files_reaccessed = 0
    for stamps in ts_by_path.values():
        uniq = sorted(set(stamps))
        if len(uniq) < 2:
            continue
        files_reaccessed += 1
        deltas.extend(uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1))
    return deltas, files_reaccessed


def compute_inter_arrival(parsed: dict[str, Any]) -> dict[str, Any]:
    """Axis 2: distribution of the time gap (seconds) between successive
    accesses to the same file (Patel FAST'20 Fig. 5a/6b analogue)."""
    deltas, files_reaccessed = inter_arrival_deltas(parsed)
    return {
        "files_with_repeat_access": files_reaccessed,
        "n_intervals": len(deltas),
        "p50_s": percentile(deltas, 50),
        "p95_s": percentile(deltas, 95),
        "p99_s": percentile(deltas, 99),
        "mean_s": (sum(deltas) / len(deltas)) if deltas else None,
        "pct_lt_1s": pct(deltas, lambda x: x < 1.0),
    }


def compute_exploration_overhead(bytes_ops_by_phase: dict[str, Any],
                                 artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Axis 4: assemble the single exploratory-I/O overhead ratio the pieces of
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
                   syscalls: set[str]) -> list[float]:
    """Bytes per fixed time window over the run, for the given syscall set."""
    points: list[tuple[float, float]] = []
    for e in parsed.get("fs_entries", []):
        if str(e.get("syscall")) not in syscalls:
            continue
        ts = e.get("timestamp")
        if not isinstance(ts, str):
            continue
        try:
            t = datetime.fromisoformat(ts).timestamp()
        except ValueError:
            continue
        sz = e.get("actual_size") or e.get("requested_size") or e.get("bytes_transferred") or 0
        points.append((t, float(sz) if isinstance(sz, (int, float)) and sz > 0 else 0.0))
    if not points:
        return []
    t0 = min(t for t, _ in points)
    t1 = max(t for t, _ in points)
    nbins = max(1, int((t1 - t0) / window_s) + 1)
    series = [0.0] * nbins
    for t, sz in points:
        series[min(nbins - 1, int((t - t0) / window_s))] += sz
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


def compute_io_autocorrelation(parsed: dict[str, Any]) -> dict[str, Any]:
    """Axis 6: read/write autocorrelation at 1/5/25-minute windows and a few
    lags, plus lag-0 read-write cross-correlation (Patel SC'19 Fig. 9)."""
    windows = {"1min": 60.0, "5min": 300.0, "25min": 1500.0}
    lags = [1, 2, 3]
    out: dict[str, Any] = {}
    for name, w in windows.items():
        r = _binned_series(parsed, w, READ_SYSCALLS_STRICT)
        wr = _binned_series(parsed, w, WRITE_SYSCALLS_STRICT)
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
                             window_s: float = 60.0) -> dict[str, Any]:
    """Axis 6: high/low-intensity I/O phase segmentation. Bin total I/O bytes
    into fixed windows, threshold at the 75th/25th percentile of non-empty bins,
    and segment consecutive high / low bins (Patel SC'19 Fig. 7/8)."""
    series = _binned_series(parsed, window_s,
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
        "interface_byte_mix": compute_interface_byte_mix(parsed),
        "namespace": lineage.get("namespace", {}),
        "sequentiality": compute_sequentiality(parsed),
        "bytes_ops_by_phase": compute_bytes_ops_by_phase(parsed, phases),
        "reread_attribution": compute_reread_attribution(parsed, phases),
        "directory_scan": compute_directory_scan_count(parsed),
        "failed_open_stat": compute_failed_open_stat_count(parsed),
        "error_log_reads": compute_error_log_reads(artifacts),
        "state_file_rewrite_frequency": compute_state_file_rewrite_frequency(artifacts),
        # Newly implemented axis metrics (data-derived, no oracle required)
        "access_type_rhwhrw": compute_access_type_rhwhrw(artifacts),          # axis 2
        "inter_arrival": compute_inter_arrival(parsed),                        # axis 2
        "io_autocorrelation": compute_io_autocorrelation(parsed),             # axis 6
        "intensity_phases": compute_intensity_phases(parsed),                 # axis 6
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
