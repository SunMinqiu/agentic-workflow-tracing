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


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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
    checks = {
        "nonzero_read_denominator": int(denominators.get("read_bytes") or 0) > 0,
        "nonzero_write_denominator": int(denominators.get("write_bytes") or 0) > 0,
        "tracer_reported_zero_lost_events": lost_events == 0 if lost_events is not None else None,
    }
    return {
        "schema_version": 1,
        "checks": checks,
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
