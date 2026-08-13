#!/usr/bin/env python3
"""Emit objective per-run trace quality checks used before comparison."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from agent_io_tracing.analysis.execution_units import (
    annotate_parsed_execution_units,
    load_execution_units,
)


DATA_IO = {
    "read", "write", "pread64", "pwrite64", "readv", "writev",
    "preadv", "pwritev", "preadv2", "pwritev2",
}


# A run whose agent could not reach its model still produces a clean-looking
# I/O trace, so the I/O checks alone once passed a SciLink run in which 39% of
# LLM calls were rejected by the server.  Above this share the cell is not a
# valid observation of the workload and must not silently enter comparison.
MAX_LLM_FAILURE_FRACTION = 0.10


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def llm_call_health(trace_dir: Path) -> dict[str, Any]:
    """Success/failure of every LLM call the adapter recorded.

    Returns applicable=False for workloads that make no LLM calls at all
    (Montage, 1000Genome) and for traces predating failure recording, so the
    check stays silent instead of failing them.
    """
    path = trace_dir / "pi_events.jsonl"
    health: dict[str, Any] = {
        "applicable": False, "n_calls": 0, "n_failed": 0,
        "failure_fraction": None, "by_status_code": {}, "by_type": {},
        "examples": [],
    }
    if not path.is_file():
        return health

    failures: list[dict[str, Any]] = []
    n_calls = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "message_end":
                    continue
                n_calls += 1
                # A successful call never carries an "error" key at all, so
                # key presence -- not its value -- is the failure signal.  This
                # keeps traces written before the reason was captured (their
                # error reads the string "None") correctly counted as failures.
                if "error" not in event:
                    continue
                failures.append(event)
    except OSError:
        return health

    if not n_calls:
        return health

    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for event in failures:
        status = event.get("error_status_code")
        by_status[str(status) if status is not None else "unknown"] = (
            by_status.get(str(status) if status is not None else "unknown", 0) + 1
        )
        kind = event.get("error_type") or "unknown"
        by_type[kind] = by_type.get(kind, 0) + 1

    seen: list[str] = []
    for event in failures:
        message = str(event.get("error") or "unrecorded")[:200]
        if message not in seen:
            seen.append(message)
        if len(seen) >= 3:
            break

    health.update({
        "applicable": True,
        "n_calls": n_calls,
        "n_failed": len(failures),
        "failure_fraction": len(failures) / n_calls,
        "by_status_code": by_status,
        "by_type": by_type,
        "examples": seen,
    })
    return health


def build_report(trace_dir: Path) -> dict[str, Any]:
    phase = _json(trace_dir / "phase1_metrics.json")
    io_summary = _json(trace_dir / "lineage" / "io_summary.json")
    parsed = _json(trace_dir / "parsed.json")
    units = load_execution_units(trace_dir)
    annotate_parsed_execution_units(parsed, units)
    artifacts_path = trace_dir / "lineage" / "artifacts.csv"
    paths: set[str] = set()
    if artifacts_path.is_file():
        with artifacts_path.open(newline="", encoding="utf-8") as handle:
            paths = {row["path"] for row in csv.DictReader(handle) if row.get("path")}

    total_bytes = attributed_bytes = offset_ops = data_ops = 0
    for entry in parsed.get("fs_entries", []):
        if str(entry.get("syscall")) not in DATA_IO:
            continue
        if paths and (entry.get("path") or "") not in paths:
            continue
        size = entry.get("bytes_transferred") or entry.get("actual_size") or 0
        if not isinstance(size, (int, float)) or size <= 0:
            continue
        total_bytes += int(size)
        data_ops += 1
        if entry.get("matched_tool_call") or entry.get("execution_unit_id"):
            attributed_bytes += int(size)
        if isinstance(entry.get("offset"), int):
            offset_ops += 1

    bcc_err = (trace_dir / "bcc.err").read_text(
        encoding="utf-8", errors="replace"
    ) if (trace_dir / "bcc.err").is_file() else ""
    matches = re.findall(r"lost_events=(\d+)", bcc_err)
    lost_events = int(matches[-1]) if matches else None
    denominators = (phase.get("byte_normalized_summary") or {}).get("denominators") or {}
    coverage = io_summary.get("coverage_pct") or {}
    llm = llm_call_health(trace_dir)
    checks = {
        "nonzero_read_denominator": int(denominators.get("read_bytes") or 0) > 0,
        "nonzero_write_denominator": int(denominators.get("write_bytes") or 0) > 0,
        "tracer_reported_zero_lost_events": lost_events == 0 if lost_events is not None else None,
        # None keeps workloads without LLM calls exempt: all() below treats
        # only an explicit False as a failure.
        "llm_calls_mostly_succeeded": (
            llm["failure_fraction"] <= MAX_LLM_FAILURE_FRACTION
            if llm["applicable"] else None
        ),
    }
    return {
        "schema_version": 2,
        "checks": checks,
        "llm_call_health": llm,
        "workload_path_coverage_pct": coverage,
        "byte_denominators": denominators,
        "offset_coverage_pct": 100.0 * offset_ops / data_ops if data_ops else None,
        "execution_unit_attribution": {
            "bytes": attributed_bytes,
            "total_bytes": total_bytes,
            "byte_coverage_pct": 100.0 * attributed_bytes / total_bytes if total_bytes else None,
            "classic_execution_units_present": bool(units),
        },
        "trace_drops": {
            "lost_events": lost_events,
            "note": "N/A for traces collected before lost-event accounting was added",
        },
        "ready_for_manual_review": all(value is not False for value in checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    trace_dir = args.trace_dir.resolve()
    output = trace_dir / "trace_quality.json"
    output.write_text(json.dumps(build_report(trace_dir), indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
