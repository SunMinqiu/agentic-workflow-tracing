#!/usr/bin/env python3
"""
Post-hoc subagent reclassification.

Two classification signals are available:

1. **Name regex (PRIMARY, on by default)** — any tool whose normalized name
   matches the regex (default `^Invoke_.*_(agent|workflow)$`, matching
   SRAgent's `Invoke_xxx_agent` and `Invoke_xxx_workflow` naming convention)
   is classified as a subagent.  This is the most reliable signal because
   frameworks like SRAgent use consistent naming for agent-dispatch tools.

2. **Time-interval (SECONDARY, opt-in via --time-interval)** — any tool whose
   [start, end] window contains an LLM `message_start` event is classified
   as a subagent.  Catches subagents whose names don't match the regex, but
   ALSO over-classifies in two real situations seen in SRAgent traces:
     a. concurrent parallel tools whose windows overlap, sweeping up LLM
        starts that actually belong to a different parallel path
     b. real tools (e.g., `Query_vector_db`) that internally call an
        embedding / reranker API — counts as LLM activity but isn't a
        subagent in the orchestration sense
   So time-interval is opt-in, not default.

Why this exists
---------------
langchain_tool_logger.py classifies subagents at run time via a `run_id`
parent-tree walk. This works only when the outer Runnable forwards its
RunnableConfig into the inner agent invocation. SRAgent's
`sub_agent.invoke(input)` calls do not forward config, so the parent tree
breaks and Invoke_*_agent tools end up in tool_calls.log. This post-hoc
script fixes that.

Idempotency
-----------
A tool entry already moved into subagent_calls.log is no longer in
tool_calls.log, so the next run can't move it again. Safe to invoke
repeatedly.

Usage
-----
    python reclassify_subagents.py <trace_dir>
        [--name-regex REGEX]      # default: ^Invoke_.*_(agent|workflow)$
        [--no-name-regex]         # disable name-regex check
        [--time-interval]         # also enable the secondary check
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# LLM event timestamps are now recorded as float ms with sub-ms precision
# (matching the µs precision of tool_calls.log times), so containment can be
# tested exactly.  A microsecond-scale epsilon is kept for floating-point
# rounding safety only — it's small enough not to swallow real adjacent
# events (no tool boundary is within 10µs of another tool boundary in
# practice for any LangChain agent loop).
_PRECISION_TOLERANCE = timedelta(microseconds=10)

# Same line format as parse_ebpf / summarize use, so the rewrite
# is byte-compatible with every existing reader.
TOOL_LOG_LINE = re.compile(
    r"\[(?P<start>\d{2}:\d{2}:\d{2}\.\d+)\s*->\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d+)\]\s*"
    r"\((?P<dur>[\d.]+)ms\)\s*"
    r"(?P<name>\w+)\s*"
    r"\(id=(?P<id>[^)]+)\)\s*"
    r"(?:container=\S+\s*)?"
    r"input=(?P<input>.+)$"
)


def _hms_to_dt(hms: str, date_anchor: datetime) -> datetime:
    """Parse 'HH:MM:SS.uuuuuu' into a datetime on date_anchor's calendar day."""
    parts = hms.split(":")
    h, m = int(parts[0]), int(parts[1])
    sec_parts = parts[2].split(".")
    s = int(sec_parts[0])
    us_str = sec_parts[1] if len(sec_parts) > 1 else "0"
    us = int(us_str[:6].ljust(6, "0"))
    return date_anchor.replace(hour=h, minute=m, second=s, microsecond=us)


def _collect_llm_start_dts(events_path: Path) -> list[datetime]:
    """Return ascending list of `message_start` timestamps as local datetimes."""
    if not events_path.exists():
        return []
    out: list[datetime] = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "message_start":
                continue
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
            ts = msg.get("timestamp")
            if isinstance(ts, (int, float)):
                out.append(datetime.fromtimestamp(ts / 1000.0))
    out.sort()
    return out


def _any_llm_in_window(llm_dts: list[datetime], s: datetime, e: datetime) -> bool:
    """
    Binary-search the sorted LLM dt list for any timestamp in [s, e],
    expanded by _PRECISION_TOLERANCE on each side to absorb the ms-vs-µs
    precision mismatch between LLM event timestamps and tool log times.
    """
    s_tol = s - _PRECISION_TOLERANCE
    e_tol = e + _PRECISION_TOLERANCE
    idx = bisect.bisect_left(llm_dts, s_tol)
    return idx < len(llm_dts) and llm_dts[idx] <= e_tol


def detect_tz_offset(llm_dts: list[datetime], tool_first_dt: datetime) -> timedelta:
    """
    Detect a timezone-mismatch offset between LLM timestamps (parsed via
    fromtimestamp on the parsing machine, which uses the PARSING machine's
    local TZ) and tool HMS times (parsed onto the parsing machine's today,
    so they're effectively in the parsing machine's TZ — but their HMS
    digits were recorded in the TRACING machine's local TZ).

    The first LLM event and the first tool/subagent event should be within
    seconds of each other on the actual wall clock.  If their naive parsed
    representations differ by ≥1 hour, that's a TZ mismatch.  We round to the
    nearest 15-minute quantum to be robust to clock skew.

    Returns the timedelta to ADD to LLM datetimes to bring them into the
    tool naive-time frame (so containment tests work).  Zero if no offset.
    """
    if not llm_dts:
        return timedelta(0)
    gap = (tool_first_dt - llm_dts[0]).total_seconds()
    # Threshold is 30 min: legitimate "agent thinks N times before first tool"
    # gaps are seconds to a few minutes; any TZ offset is ≥30 min (because
    # standard timezones are at 15/30/60-minute boundaries — a 1-hour TZ off
    # is the most common and shows up as a gap just under 3600s once you
    # subtract the few-second orchestrator-LLM lag).
    if abs(gap) < 1800:
        return timedelta(0)
    quanta = round(gap / 900) * 900    # nearest 15 min
    return timedelta(seconds=quanta)


DEFAULT_NAME_REGEX = r"^Invoke_.*_(agent|workflow)$"


def reclassify(
    trace_dir: Path,
    *,
    name_regex: str | None = DEFAULT_NAME_REGEX,
    use_time_interval: bool = False,
) -> dict:
    """
    Reclassify tool_calls.log entries as subagents.  See module docstring for
    the two classification signals.

    Args:
        trace_dir: trace directory containing tool_calls.log, pi_events.jsonl,
                   subagent_calls.log.
        name_regex: a regex string; tools whose name matches are classified as
                    subagents.  None or empty string disables this check.
                    Default: r"^Invoke_.*_(agent|workflow)$".
        use_time_interval: if True, ALSO classify any tool whose [start, end]
                    window contains an LLM message_start.  See module
                    docstring for caveats (over-classifies real tools that
                    internally call embedding APIs, and parallel tools).
                    Default False.
    """
    tool_log = trace_dir / "tool_calls.log"
    subagent_log = trace_dir / "subagent_calls.log"
    events_log = trace_dir / "pi_events.jsonl"

    if not tool_log.exists():
        return {"error": f"no tool_calls.log in {trace_dir}"}

    name_pattern: re.Pattern | None = re.compile(name_regex) if name_regex else None
    if not name_pattern and not use_time_interval:
        return {
            "error": "all classifiers disabled — pass --name-regex or "
                     "--time-interval to enable at least one"
        }

    llm_dts: list[datetime] = []
    if use_time_interval:
        llm_dts = _collect_llm_start_dts(events_log)
        if not llm_dts:
            print(
                "reclassify_subagents: --time-interval requested but no LLM "
                "message_start events found; time-interval check will be a "
                "no-op.",
                file=sys.stderr,
            )

    # TZ-mismatch detection only matters for the time-interval check.
    if use_time_interval and llm_dts:
        first_tool_dt: datetime | None = None
        with tool_log.open("r", encoding="utf-8") as f:
            for line in f:
                m = TOOL_LOG_LINE.match(line.strip())
                if m:
                    first_tool_dt = _hms_to_dt(
                        m.group("start"),
                        llm_dts[0].replace(hour=0, minute=0, second=0, microsecond=0),
                    )
                    break
        tz_offset = detect_tz_offset(llm_dts, first_tool_dt) if first_tool_dt else timedelta(0)
        if tz_offset != timedelta(0):
            print(
                f"reclassify_subagents: detected TZ mismatch — shifting LLM "
                f"timestamps by {tz_offset.total_seconds():+.0f}s to align "
                f"with tool log times (tracing machine TZ != this machine TZ).",
                file=sys.stderr,
            )
            llm_dts = [dt + tz_offset for dt in llm_dts]
        # date_anchor for HMS parsing
        date_anchor = llm_dts[0].replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # No time-interval check; date_anchor unused.  Anchor on today so
        # HMS parsing still produces valid (if unused) datetimes.
        date_anchor = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    moved_lines: list[str] = []
    kept_lines: list[str] = []
    moved_names: list[str] = []
    moved_reasons: dict[str, set[str]] = {}   # name → set of reasons

    # Read BOTH logs into a single pool and re-partition them.  This makes
    # reclassify idempotent across runs with different criteria — running
    # with --time-interval and then without will correctly move things back
    # to tool_calls.log instead of leaving stale subagent entries.
    raw_lines: list[str] = []
    with tool_log.open("r", encoding="utf-8") as f:
        raw_lines.extend(f.readlines())
    if subagent_log.exists():
        with subagent_log.open("r", encoding="utf-8") as f:
            raw_lines.extend(f.readlines())

    for raw in raw_lines:
        if not raw.strip():
            continue
        m = TOOL_LOG_LINE.match(raw.strip())
        if not m:
            kept_lines.append(raw if raw.endswith("\n") else raw + "\n")
            continue

        name = m.group("name")
        reasons: list[str] = []

        # Primary: name regex
        if name_pattern and name_pattern.match(name):
            reasons.append("name")

        # Secondary: time-interval (opt-in)
        if use_time_interval and llm_dts:
            s_dt = _hms_to_dt(m.group("start"), date_anchor)
            e_dt = _hms_to_dt(m.group("end"), date_anchor)
            if _any_llm_in_window(llm_dts, s_dt, e_dt):
                reasons.append("time")

        is_subagent = bool(reasons)
        target = moved_lines if is_subagent else kept_lines
        target.append(raw if raw.endswith("\n") else raw + "\n")
        if is_subagent:
            moved_names.append(name)
            moved_reasons.setdefault(name, set()).update(reasons)

    # Always rewrite both logs from the re-partitioned pool.  Re-partition is
    # idempotent regardless of previous reclassify runs.
    tool_log.write_text("".join(kept_lines), encoding="utf-8")
    subagent_log.write_text("".join(moved_lines), encoding="utf-8")

    return {
        "moved": len(moved_lines),
        "kept": len(kept_lines),
        "moved_names_unique": sorted(set(moved_names)),
        "moved_by_reason": {
            name: sorted(reasons) for name, reasons in moved_reasons.items()
        },
        "name_regex": name_regex if name_pattern else None,
        "time_interval": use_time_interval,
        "subagent_log": str(subagent_log),
        "tool_log": str(tool_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc subagent reclassification.  Default classifier is name "
            "regex; pass --time-interval to additionally use time-window LLM "
            "containment (with caveats — see module docstring)."
        )
    )
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory containing tool_calls.log + pi_events.jsonl",
    )
    parser.add_argument(
        "--name-regex",
        type=str,
        default=DEFAULT_NAME_REGEX,
        help=f"Regex matched against tool names. Default: {DEFAULT_NAME_REGEX!r}",
    )
    parser.add_argument(
        "--no-name-regex",
        action="store_true",
        help="Disable the name-regex check.",
    )
    parser.add_argument(
        "--time-interval",
        action="store_true",
        help=(
            "Also enable the secondary time-interval check (over-classifies "
            "tools that internally call embedding/reranker APIs and parallel "
            "tools that overlap with LLM events on other paths)."
        ),
    )
    args = parser.parse_args()
    regex = None if args.no_name_regex else args.name_regex
    result = reclassify(
        args.trace_dir.resolve(),
        name_regex=regex,
        use_time_interval=args.time_interval,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
