#!/usr/bin/env python3
"""Print one cell's quality verdict in plain language; exit 1 if unusable.

pull_agentic_run.sh calls this before its file-existence loop so a degraded
cell says *why* it is degraded.  Missing figures are usually a symptom -- an
agent that could not reach its model makes no tool calls, and the visualizer
then declines to draw the comparison figures -- and reporting the symptom
alone sends you looking in the wrong place.

    PYTHONPATH=src python -m agent_io_tracing.analysis.trace_quality_report <cell>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_io_tracing.analysis.trace_quality import (
    MAX_LLM_FAILURE_FRACTION, _json, llm_call_health,
)


def describe(cell: Path) -> tuple[list[str], bool]:
    """Human-readable findings and whether the cell is fit for comparison."""
    lines: list[str] = []
    ok = True

    health = llm_call_health(cell)
    if health["applicable"] and health["n_failed"]:
        fraction = health["failure_fraction"]
        severity = "ERROR" if fraction > MAX_LLM_FAILURE_FRACTION else "NOTE"
        if fraction > MAX_LLM_FAILURE_FRACTION:
            ok = False
        lines.append(
            f"{severity}: {cell.name}: {health['n_failed']} of "
            f"{health['n_calls']} LLM calls failed ({fraction:.0%})"
        )
        by_status = health["by_status_code"]
        if by_status and set(by_status) != {"unknown"}:
            summary = ", ".join(f"{n}x HTTP {code}" for code, n in sorted(by_status.items()))
            lines.append(f"    status codes: {summary}")
        for example in health["examples"]:
            if example in ("None", "unrecorded", ""):
                continue
            lines.append(f"    reason: {example}")
        if set(by_status) == {"unknown"}:
            lines.append(
                "    reason unrecorded by this trace's adapter; see scilink.log "
                "or genomas.log in the cell for the server's message"
            )

    # The visualizer states its own reason for skipping; surface it rather
    # than letting the caller infer one from the missing files.
    visualize_log = cell / "visualize.log"
    if visualize_log.is_file():
        text = visualize_log.read_text(encoding="utf-8", errors="replace")
        if "not eligible for comparison figures" in text:
            lines.append(
                f"NOTE: {cell.name}: comparison figures were skipped because the "
                "cell recorded no tool calls"
            )

    quality = _json(cell / "trace_quality.json")
    if quality.get("ready_for_manual_review") is False:
        failed_checks = [
            name for name, value in (quality.get("checks") or {}).items()
            if value is False
        ]
        lines.append(
            f"ERROR: {cell.name}: trace_quality checks failed: "
            + ", ".join(failed_checks or ["unspecified"])
        )
        ok = False

    return lines, ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", type=Path)
    args = parser.parse_args()

    lines, ok = describe(args.cell)
    for line in lines:
        print(line, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
