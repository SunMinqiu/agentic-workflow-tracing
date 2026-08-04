#!/usr/bin/env python3
"""Unify agent tool calls and classic tasks as I/O execution units."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from agent_io_tracing.analysis.workload_paths import workload_path_index


DATA_READ = {"read", "pread64", "readv", "preadv", "preadv2"}
DATA_WRITE = {"write", "pwrite64", "writev", "pwritev", "pwritev2"}
METADATA = {
    "open", "openat", "openat2", "close", "stat", "fstat", "lstat",
    "newfstatat", "statx", "access", "faccessat", "mkdir", "mkdirat",
    "rmdir", "unlink", "unlinkat", "rename", "renameat", "renameat2",
    "getdents", "getdents64", "fsync", "fdatasync", "sync_file_range",
}


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    rank = q / 100.0 * (len(vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _entry_ts_ms(entry: dict[str, Any]) -> float | None:
    value = entry.get("ts_ms")
    if isinstance(value, (int, float)):
        return float(value)
    timestamp = entry.get("timestamp")
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp).timestamp() * 1000.0
        except ValueError:
            pass
    return None


def load_execution_units(trace_dir: Path) -> list[dict[str, Any]]:
    path = trace_dir / "execution_units.jsonl"
    if not path.is_file():
        return []
    units: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        unit_id = row.get("execution_unit_id")
        if isinstance(unit_id, str) and unit_id:
            units.append(row)
    return units


def _execution_unit_clock_offset(parsed: dict[str, Any]) -> float:
    """Return the offset from epoch-local time to the parsed trace clock.

    ``execution_units.jsonl`` stores Unix epoch milliseconds, while
    ``parsed.json`` stores timezone-naive ISO timestamps produced on the
    machine that parsed the eBPF trace.  If a result directory is later read
    in a different timezone, converting the execution-unit epochs with the
    reader's ``datetime.fromtimestamp`` moves only the synthetic tool bars.

    Entries that carry both ``ts_ms`` and ``timestamp`` give us an exact
    bridge between those clocks.  Real timezone offsets are quantized to 15
    minutes; ignore smaller differences so timestamp formatting noise does
    not shift task boundaries.
    """
    offsets: list[float] = []
    for entry in parsed.get("fs_entries", []):
        ts_ms = entry.get("ts_ms")
        timestamp = entry.get("timestamp")
        if not isinstance(ts_ms, (int, float)) or not isinstance(timestamp, str):
            continue
        try:
            parsed_time = datetime.fromisoformat(timestamp).replace(tzinfo=None)
        except ValueError:
            continue
        epoch_local_time = datetime.fromtimestamp(float(ts_ms) / 1000.0)
        offsets.append((epoch_local_time - parsed_time).total_seconds())
        if len(offsets) >= 101:
            break

    if not offsets:
        return 0.0
    offset = median(offsets)
    if abs(offset) < 1800.0:
        return 0.0
    return float(round(offset / 900.0) * 900.0)


def annotate_parsed_execution_units(
    parsed: dict[str, Any], units: list[dict[str, Any]]
) -> int:
    """Annotate unmatched entries in memory by task PID and wall interval."""
    if not units:
        return 0
    by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        pid = unit.get("pid")
        if isinstance(pid, int):
            by_pid[pid].append(unit)
    annotated = 0
    for entry in parsed.get("fs_entries", []):
        pid = entry.get("pid")
        ts_ms = _entry_ts_ms(entry)
        if not isinstance(pid, int) or ts_ms is None:
            continue
        matches = [
            unit for unit in by_pid.get(pid, [])
            if isinstance(unit.get("start_ts_ms"), (int, float))
            and isinstance(unit.get("end_ts_ms"), (int, float))
            and float(unit["start_ts_ms"]) <= ts_ms <= float(unit["end_ts_ms"])
        ]
        if not matches:
            continue
        unit = min(
            matches,
            key=lambda item: float(item["end_ts_ms"]) - float(item["start_ts_ms"]),
        )
        unit_id = str(unit["execution_unit_id"])
        entry["execution_unit_id"] = unit_id
        entry["execution_stage"] = unit.get("stage") or "classic_task"
        if not entry.get("matched_tool_call"):
            entry["matched_tool_call"] = unit_id
        annotated += 1

    clock_offset_s = _execution_unit_clock_offset(parsed)

    def execution_time(ms: float) -> str:
        value = datetime.fromtimestamp(ms / 1000.0)
        return (value - timedelta(seconds=clock_offset_s)).isoformat()

    known = {
        call.get("tool_id") for call in parsed.get("tool_calls", [])
        if isinstance(call, dict)
    }
    for unit in units:
        unit_id = str(unit["execution_unit_id"])
        if unit_id in known:
            continue
        start_ms = float(unit.get("start_ts_ms") or 0.0)
        end_ms = float(unit.get("end_ts_ms") or start_ms)
        stage = str(unit.get("stage") or "classic_task")
        parsed.setdefault("tool_calls", []).append(
            {
                "tool_id": unit_id,
                "tool_name": stage,
                "start_time": execution_time(start_ms),
                "end_time": execution_time(end_ms),
                "input_params": {
                    "phase": stage,
                    "execution_unit_kind": "classic_task",
                    "task_id": unit.get("task_id") or unit_id,
                },
            }
        )
    return annotated


def _workload_artifacts(trace_dir: Path) -> list[dict[str, Any]]:
    path = trace_dir / "lineage" / "artifacts.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _merged_seconds(intervals: list[tuple[float, float]]) -> float:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged) / 1000.0


def build_execution_unit_outputs(trace_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = json.loads((trace_dir / "parsed.json").read_text(encoding="utf-8"))
    units = load_execution_units(trace_dir)
    annotate_parsed_execution_units(parsed, units)
    path_index = workload_path_index(_workload_artifacts(trace_dir))
    tool_calls = {
        row.get("tool_id"): row for row in parsed.get("tool_calls", [])
        if isinstance(row, dict) and isinstance(row.get("tool_id"), str)
    }
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "read_ops": 0, "write_ops": 0, "read_bytes": 0,
            "write_bytes": 0, "metadata_ops": 0, "intervals": [],
        }
    )
    for entry in parsed.get("fs_entries", []):
        unit_id = entry.get("execution_unit_id") or entry.get("matched_tool_call")
        if not isinstance(unit_id, str) or not unit_id:
            continue
        path = entry.get("path") or ""
        if path_index.files and not path_index.contains(path):
            continue
        syscall = str(entry.get("syscall") or "")
        size = entry.get("bytes_transferred") or entry.get("actual_size") or 0
        size = int(size) if isinstance(size, (int, float)) and size > 0 else 0
        row = stats[unit_id]
        if syscall in DATA_READ and size > 0:
            row["read_ops"] += 1
            row["read_bytes"] += size
        elif syscall in DATA_WRITE and size > 0:
            row["write_ops"] += 1
            row["write_bytes"] += size
        elif syscall in METADATA:
            row["metadata_ops"] += 1
        if syscall in DATA_READ | DATA_WRITE and size > 0:
            end_ms = _entry_ts_ms(entry)
            duration_ms = float(entry.get("duration") or 0.0) * 1000.0
            if end_ms is not None and duration_ms > 0:
                row["intervals"].append((end_ms - duration_ms, end_ms))

    unit_meta = {str(unit["execution_unit_id"]): unit for unit in units}
    rows: list[dict[str, Any]] = []
    phase = {}
    phase_path = trace_dir / "phase1_metrics.json"
    if phase_path.is_file():
        try:
            phase = json.loads(phase_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            phase = {}
    phase_denominators = (
        (phase.get("byte_normalized_summary") or {}).get("denominators") or {}
    )
    read_total = int(phase_denominators.get("read_bytes") or 0) or sum(
        int(row["read_bytes"]) for row in stats.values()
    )
    write_total = int(phase_denominators.get("write_bytes") or 0) or sum(
        int(row["write_bytes"]) for row in stats.values()
    )
    total_io = read_total + write_total
    gib = float(1024 ** 3)
    for unit_id, values in sorted(stats.items()):
        meta = unit_meta.get(unit_id, {})
        call = tool_calls.get(unit_id, {})
        params = call.get("input_params") if isinstance(call.get("input_params"), dict) else {}
        read_bytes = int(values["read_bytes"])
        write_bytes = int(values["write_bytes"])
        row = {
            "execution_unit_id": unit_id,
            "kind": params.get("execution_unit_kind") or ("classic_task" if meta else "agent_tool_call"),
            "stage": meta.get("stage") or params.get("phase") or call.get("tool_name") or "unattributed",
            "pid": meta.get("pid") or "",
            "start_ts_ms": meta.get("start_ts_ms") or "",
            "end_ts_ms": meta.get("end_ts_ms") or "",
            "read_ops": int(values["read_ops"]),
            "write_ops": int(values["write_ops"]),
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "metadata_ops": int(values["metadata_ops"]),
            "io_busy_s": _merged_seconds(values["intervals"]),
            "read_ops_per_gib_read": values["read_ops"] * gib / read_total if read_total else None,
            "write_ops_per_gib_write": values["write_ops"] * gib / write_total if write_total else None,
            "metadata_ops_per_gib_total_io": values["metadata_ops"] * gib / total_io if total_io else None,
            "read_byte_share_pct": 100.0 * read_bytes / read_total if read_total else None,
            "write_byte_share_pct": 100.0 * write_bytes / write_total if write_total else None,
        }
        rows.append(row)

    byte_totals = [float(row["read_bytes"] + row["write_bytes"]) for row in rows]
    busy = [float(row["io_busy_s"]) for row in rows]

    def skew(values: list[float]) -> dict[str, Any]:
        positive = [value for value in values if value > 0]
        p50 = median(positive) if positive else None
        p95 = _percentile(positive, 95)
        return {
            "n": len(positive),
            "max": max(positive) if positive else None,
            "p50": p50,
            "p95": p95,
            "max_over_median": max(positive) / p50 if p50 else None,
            "p95_over_p50": p95 / p50 if p50 and p95 is not None else None,
        }

    return rows, {
        "schema_version": 1,
        "execution_units_with_io": len(rows),
        "denominators": {
            "read_bytes": read_total,
            "write_bytes": write_total,
            "total_io_bytes": total_io,
        },
        "skew": {"io_bytes": skew(byte_totals), "io_busy_s": skew(busy)},
    }


def write_outputs(trace_dir: Path) -> tuple[Path, Path]:
    rows, summary = build_execution_unit_outputs(trace_dir)
    output_dir = trace_dir / "lineage"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "execution_unit_io.csv"
    fields = list(rows[0]) if rows else [
        "execution_unit_id", "kind", "stage", "pid", "start_ts_ms", "end_ts_ms",
        "read_ops", "write_ops", "read_bytes", "write_bytes", "metadata_ops",
        "io_busy_s", "read_ops_per_gib_read", "write_ops_per_gib_write",
        "metadata_ops_per_gib_total_io", "read_byte_share_pct", "write_byte_share_pct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "execution_unit_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    csv_path, json_path = write_outputs(args.trace_dir.resolve())
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
