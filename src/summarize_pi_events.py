#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TOOL_CALL_PATTERN = re.compile(
    r"\[(\d{2}:\d{2}:\d{2}\.\d+)\s*->\s*(\d{2}:\d{2}:\d{2}\.\d+)\]\s*"
    r"\(([\d.]+)ms\)\s*"
    r"(\w+)\s*"
    r"\(id=([^)]+)\)\s*"
    r"(?:container=\S+\s*)?"
    r"input=(.+)$"
)


@dataclass
class ToolCallLogEntry:
    tool_id: str
    tool_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    input_params: dict[str, Any]


@dataclass
class LlmSegment:
    start_ms: float
    end_ms: float
    run_id: str | None = None
    parent_subagent_id: str | None = None

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


def parse_time(time_str: str) -> datetime:
    parts = time_str.split(".")
    time_part = parts[0]
    micros = parts[1] if len(parts) > 1 else "0"
    micros = micros[:6].ljust(6, "0")
    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    hour, minute, second = map(int, time_part.split(":"))
    return base.replace(hour=hour, minute=minute, second=second, microsecond=int(micros))


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def stable_serialize(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(value)


def parse_tool_calls_log(path: Path) -> list[ToolCallLogEntry]:
    entries: list[ToolCallLogEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            match = TOOL_CALL_PATTERN.match(line)
            if not match:
                print(f"Warning: could not parse tool_calls.log line {line_num}", file=sys.stderr)
                continue
            start_s, end_s, duration_s, tool_name, tool_id, input_s = match.groups()
            try:
                input_params = ast.literal_eval(input_s)
                if not isinstance(input_params, dict):
                    input_params = {"raw": input_s}
            except (ValueError, SyntaxError):
                input_params = {"raw": input_s}

            entries.append(
                ToolCallLogEntry(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    start_time=parse_time(start_s),
                    end_time=parse_time(end_s),
                    duration_ms=float(duration_s),
                    input_params=input_params,
                )
            )
    return entries


def _extract_text_from_tool_result(result_obj: dict[str, Any]) -> str:
    content = result_obj.get("content", [])
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


def _empty_usage() -> dict[str, int]:
    return {"input": 0, "output": 0, "cacheRead": 0, "total": 0, "calls": 0}


def _add_usage(bucket: dict[str, int], usage: dict[str, Any]) -> None:
    bucket["input"] += int(usage.get("input", 0) or 0)
    bucket["output"] += int(usage.get("output", 0) or 0)
    bucket["cacheRead"] += int(usage.get("cacheRead", 0) or 0)
    bucket["total"] += int(usage.get("totalTokens", 0) or 0)
    bucket["calls"] += 1


def analyze_events(path: Path) -> dict[str, Any]:
    first_assistant_ts: int | None = None
    last_message_ts: int | None = None

    usage_input = 0
    usage_output = 0
    usage_cache_read = 0
    usage_total = 0

    content_tokens_by_tool_id: dict[str, int] = {}
    output_tokens_by_tool_id: dict[str, int] = {}
    # tool_id -> parent_subagent_run_id (or None for top-level). Populated
    # from tool_execution_end events when the logger annotates them.
    parent_subagent_by_tool_id: dict[str, str | None] = {}
    # Per-subagent LLM usage buckets keyed by parent_subagent_run_id.
    # None bucket holds top-level (non-subagent) LLM usage.
    llm_usage_by_parent: dict[str | None, dict[str, int]] = {}
    llm_segments: list[LlmSegment] = []
    llm_starts_by_id: dict[str, dict[str, Any]] = {}
    llm_start_stack: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSON at line {line_num}", file=sys.stderr)
                continue

            msg = event.get("message")
            if isinstance(msg, dict):
                ts = msg.get("timestamp")
                if isinstance(ts, (int, float)):
                    if last_message_ts is None or ts > last_message_ts:
                        last_message_ts = ts

            event_type = event.get("type")

            if event_type == "message_start":
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    ts = msg.get("timestamp")
                    if isinstance(ts, (int, float)) and first_assistant_ts is None:
                        first_assistant_ts = ts
                    if isinstance(ts, (int, float)):
                        run_id = event.get("run_id")
                        parent_subagent = (
                            event.get("parent_run_id")
                            or event.get("parent_subagent_run_id")
                        )
                        start = {
                            "start_ms": float(ts),
                            "run_id": run_id if isinstance(run_id, str) else None,
                            "parent_subagent_id": (
                                parent_subagent if isinstance(parent_subagent, str) else None
                            ),
                        }
                        if isinstance(run_id, str):
                            llm_starts_by_id[run_id] = start
                        else:
                            llm_start_stack.append(start)

            elif event_type == "message_end":
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    ts = msg.get("timestamp")
                    run_id = event.get("run_id")
                    start = None
                    if isinstance(run_id, str) and run_id in llm_starts_by_id:
                        start = llm_starts_by_id.pop(run_id)
                    elif llm_start_stack:
                        start = llm_start_stack.pop()
                    if start is not None and isinstance(ts, (int, float)):
                        parent_subagent = (
                            event.get("parent_run_id")
                            or event.get("parent_subagent_run_id")
                            or start.get("parent_subagent_id")
                        )
                        llm_segments.append(
                            LlmSegment(
                                start_ms=float(start["start_ms"]),
                                end_ms=float(ts),
                                run_id=run_id if isinstance(run_id, str) else start.get("run_id"),
                                parent_subagent_id=(
                                    parent_subagent if isinstance(parent_subagent, str) else None
                                ),
                            )
                        )
                    usage = msg.get("usage", {})
                    if isinstance(usage, dict):
                        usage_input += int(usage.get("input", 0) or 0)
                        usage_output += int(usage.get("output", 0) or 0)
                        usage_cache_read += int(usage.get("cacheRead", 0) or 0)
                        usage_total += int(usage.get("totalTokens", 0) or 0)

                        parent_subagent = (
                            event.get("parent_run_id")
                            or event.get("parent_subagent_run_id")
                        )
                        key = parent_subagent if isinstance(parent_subagent, str) else None
                        bucket = llm_usage_by_parent.setdefault(key, _empty_usage())
                        _add_usage(bucket, usage)

            elif event_type == "message_update":
                assistant_event = event.get("assistantMessageEvent", {})
                if not isinstance(assistant_event, dict):
                    continue
                if assistant_event.get("type") == "toolcall_end":
                    tool_call = assistant_event.get("toolCall", {})
                    if not isinstance(tool_call, dict):
                        continue
                    tool_id = tool_call.get("id")
                    if not isinstance(tool_id, str):
                        continue
                    arguments = tool_call.get("arguments", {})
                    token_est = estimate_tokens_from_text(stable_serialize(arguments))
                    content_tokens_by_tool_id[tool_id] = token_est

            elif event_type == "tool_execution_end":
                tool_id = event.get("toolCallId")
                result = event.get("result", {})
                if isinstance(tool_id, str) and isinstance(result, dict):
                    text = _extract_text_from_tool_result(result)
                    output_tokens_by_tool_id[tool_id] = estimate_tokens_from_text(text)
                if isinstance(tool_id, str):
                    parent_subagent = (
                        event.get("parent_run_id")
                        or event.get("parent_subagent_run_id")
                    )
                    parent_subagent_by_tool_id[tool_id] = (
                        parent_subagent if isinstance(parent_subagent, str) else None
                    )

    return {
        "first_assistant_ts": first_assistant_ts,
        "last_message_ts": last_message_ts,
        "usage_input": usage_input,
        "usage_output": usage_output,
        "usage_cache_read": usage_cache_read,
        "usage_total": usage_total,
        "content_tokens_by_tool_id": content_tokens_by_tool_id,
        "output_tokens_by_tool_id": output_tokens_by_tool_id,
        "parent_subagent_by_tool_id": parent_subagent_by_tool_id,
        "llm_usage_by_parent": llm_usage_by_parent,
        "llm_segments": llm_segments,
    }


def _datetime_to_ms(dt: datetime) -> float:
    return dt.timestamp() * 1000.0


def _replace_date(dt: datetime, anchor_ms: float) -> datetime:
    anchor = datetime.fromtimestamp(anchor_ms / 1000.0)
    return dt.replace(year=anchor.year, month=anchor.month, day=anchor.day)


def _interval_union_ms(intervals: list[tuple[float, float]]) -> float:
    merged = sorted((s, e) for s, e in intervals if e > s)
    if not merged:
        return 0.0
    total = 0.0
    cur_s, cur_e = merged[0]
    for s, e in merged[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def _tool_intervals_ms(
    tool_calls: list[ToolCallLogEntry],
    llm_segments: list[LlmSegment],
) -> list[tuple[float, float]]:
    if not tool_calls:
        return []
    calls = tool_calls
    if llm_segments:
        anchor_ms = min(seg.start_ms for seg in llm_segments)
        calls = [
            ToolCallLogEntry(
                tool_id=tc.tool_id,
                tool_name=tc.tool_name,
                start_time=_replace_date(tc.start_time, anchor_ms),
                end_time=_replace_date(tc.end_time, anchor_ms),
                duration_ms=tc.duration_ms,
                input_params=tc.input_params,
            )
            for tc in tool_calls
        ]
        first_llm_dt = datetime.fromtimestamp(anchor_ms / 1000.0)
        first_tool_dt = min(tc.start_time for tc in calls)
        gap_s = (first_tool_dt - first_llm_dt).total_seconds()
        if abs(gap_s) >= 1800:
            # Same timezone skew heuristic used by the visualizers.
            from datetime import timedelta

            shift_s = round(gap_s / 900) * 900
            calls = [
                ToolCallLogEntry(
                    tool_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    start_time=tc.start_time - timedelta(seconds=shift_s),
                    end_time=tc.end_time - timedelta(seconds=shift_s),
                    duration_ms=tc.duration_ms,
                    input_params=tc.input_params,
                )
                for tc in calls
            ]
    return [(_datetime_to_ms(tc.start_time), _datetime_to_ms(tc.end_time)) for tc in calls]


def build_summary(trace_dir: Path) -> dict[str, Any]:
    events_path = trace_dir / "pi_events.jsonl"
    tool_log_path = trace_dir / "tool_calls.log"
    subagent_log_path = trace_dir / "subagent_calls.log"

    if not events_path.exists():
        raise FileNotFoundError(f"Missing file: {events_path}")
    if not tool_log_path.exists():
        raise FileNotFoundError(f"Missing file: {tool_log_path}")

    event_stats = analyze_events(events_path)
    tool_calls = parse_tool_calls_log(tool_log_path)
    # subagent_calls.log is only written by langchain_tool_logger.py (SRAgent
    # path); pi traces won't have one. Treat missing/empty as "no subagents".
    subagent_calls = (
        parse_tool_calls_log(subagent_log_path) if subagent_log_path.exists() else []
    )

    total_tool_time_ms = sum(tc.duration_ms for tc in tool_calls)
    total_tool_calls = len(tool_calls)
    avg_tool_time_ms = (total_tool_time_ms / total_tool_calls) if total_tool_calls else 0.0

    first_ts = event_stats["first_assistant_ts"]
    last_ts = event_stats["last_message_ts"]
    overall_execution_time_ms = 0.0
    if (
        isinstance(first_ts, (int, float))
        and isinstance(last_ts, (int, float))
        and last_ts >= first_ts
    ):
        overall_execution_time_ms = float(last_ts - first_ts)
    elif tool_calls or subagent_calls:
        all_calls = tool_calls + subagent_calls
        start = min(tc.start_time for tc in all_calls)
        end = max(tc.end_time for tc in all_calls)
        overall_execution_time_ms = max(0.0, (end - start).total_seconds() * 1000.0)

    llm_segments: list[LlmSegment] = event_stats["llm_segments"]
    llm_total_time_ms = sum(seg.duration_ms for seg in llm_segments)
    llm_intervals = [(seg.start_ms, seg.end_ms) for seg in llm_segments]
    tool_intervals = _tool_intervals_ms(tool_calls + subagent_calls, llm_segments)
    measured_union_ms = _interval_union_ms(llm_intervals + tool_intervals)
    unaccounted_time_ms = max(0.0, overall_execution_time_ms - measured_union_ms)
    tool_time_pct = (total_tool_time_ms / overall_execution_time_ms * 100.0) if overall_execution_time_ms else 0.0

    per_tool_call: list[dict[str, Any]] = []
    sum_content_tokens_est = 0
    sum_output_tokens_est = 0
    content_map = event_stats["content_tokens_by_tool_id"]
    output_map = event_stats["output_tokens_by_tool_id"]
    parent_map: dict[str, str | None] = event_stats["parent_subagent_by_tool_id"]

    for tc in tool_calls:
        fallback_content_est = estimate_tokens_from_text(stable_serialize(tc.input_params))
        content_tokens_est = int(content_map.get(tc.tool_id, fallback_content_est))
        output_tokens_est = int(output_map.get(tc.tool_id, 0))
        sum_content_tokens_est += content_tokens_est
        sum_output_tokens_est += output_tokens_est
        per_tool_call.append(
            {
                "tool_id": tc.tool_id,
                "tool_name": tc.tool_name,
                "duration_ms": round(tc.duration_ms, 3),
                "content_tokens_est": content_tokens_est,
                "output_tokens_est": output_tokens_est,
                "parent_subagent_id": parent_map.get(tc.tool_id),
            }
        )

    llm_time_pct = (llm_total_time_ms / overall_execution_time_ms * 100.0) if overall_execution_time_ms else 0.0

    subagent_metrics = _build_subagent_metrics(
        subagent_calls=subagent_calls,
        tool_calls=tool_calls,
        parent_subagent_by_tool_id=parent_map,
        llm_usage_by_parent=event_stats["llm_usage_by_parent"],
    )

    summary = {
        "trace_dir": str(trace_dir),
        "metrics": {
            "overall_execution_time_ms": round(overall_execution_time_ms, 3),
            "tool_call_metrics": {
                "total_tool_time_ms": round(total_tool_time_ms, 3),
                "tool_time_pct": round(tool_time_pct, 3),
                "total_tool_calls": total_tool_calls,
                "avg_tool_time_ms": round(avg_tool_time_ms, 3),
                "sum_tool_content_tokens_est": sum_content_tokens_est,
                "sum_tool_output_tokens_est": sum_output_tokens_est,
                "per_tool_call": per_tool_call,
            },
            "llm_completion_metrics": {
                "total_llm_time_ms": round(llm_total_time_ms, 3),
                "llm_time_pct": round(llm_time_pct, 3),
                "total_llm_calls": len(llm_segments),
                "time_source": "message_start/message_end",
                "total_reasoning_generation_time_ms": None,
                "total_output_generation_time_ms": None,
                "total_input_tokens": event_stats["usage_input"],
                "total_reasoning_tokens": None,
                "total_output_tokens": event_stats["usage_output"],
                "total_cache_read_tokens": event_stats["usage_cache_read"],
                "total_tokens": event_stats["usage_total"],
                "reasoning_time_pct_of_overall": None,
                "output_time_pct_of_overall": None,
                "notes": [
                    "LLM time is measured from pi_events message_start/message_end, not residual overall minus tool time.",
                    "Reasoning/output split metrics are unavailable in current pi_events format.",
                    "Output token counts include both reasoning and visible output for this provider.",
                ],
            },
            "unaccounted_metrics": {
                "unaccounted_time_ms": round(unaccounted_time_ms, 3),
                "unaccounted_pct_of_overall": (
                    round(unaccounted_time_ms / overall_execution_time_ms * 100.0, 3)
                    if overall_execution_time_ms else 0.0
                ),
                "notes": [
                    "Unaccounted is wall-clock residual after measured LLM/tool/subagent intervals are unioned.",
                ],
            },
            "subagent_metrics": subagent_metrics,
        },
    }
    return summary


def _build_subagent_metrics(
    *,
    subagent_calls: list[ToolCallLogEntry],
    tool_calls: list[ToolCallLogEntry],
    parent_subagent_by_tool_id: dict[str, str | None],
    llm_usage_by_parent: dict[str | None, dict[str, int]],
) -> dict[str, Any]:
    """
    Emit summary blocks for subagent activity.

    `per_subagent` attributes inner tool calls and inner LLM token usage to
    the *nearest* enclosing subagent (so nested subagents don't double-count).
    `top_level` is the bucket for tools / LLM calls that ran outside any
    subagent — i.e. the orchestrator's own direct work.

    Sum invariant (non-nested case): for every dimension,
        sum(per_subagent[i].inner_*) + top_level.inner_* == grand total
    """
    total_subagent_calls = len(subagent_calls)
    total_subagent_time_ms = sum(sc.duration_ms for sc in subagent_calls)
    avg_subagent_time_ms = (
        total_subagent_time_ms / total_subagent_calls if total_subagent_calls else 0.0
    )

    # Bucket inner real-tool calls by their parent_subagent_id (None = top).
    tools_by_parent: dict[str | None, list[ToolCallLogEntry]] = {}
    for tc in tool_calls:
        key = parent_subagent_by_tool_id.get(tc.tool_id)
        tools_by_parent.setdefault(key, []).append(tc)

    # The subagent_calls.log entries don't carry their own parent_subagent on
    # disk (line format is unchanged for backwards-compat).  We still want
    # to report nested subagent parentage when known: look it up from the
    # tool_id -> parent_subagent map captured from pi_events.jsonl.
    per_subagent: list[dict[str, Any]] = []
    for sc in subagent_calls:
        own_id = sc.tool_id
        inner_tools = tools_by_parent.get(own_id, [])
        inner_tool_time_ms = sum(t.duration_ms for t in inner_tools)
        inner_llm = llm_usage_by_parent.get(own_id) or _empty_usage()
        per_subagent.append(
            {
                "subagent_id": own_id,
                "subagent_name": sc.tool_name,
                "duration_ms": round(sc.duration_ms, 3),
                "parent_subagent_id": parent_subagent_by_tool_id.get(own_id),
                "inner_tool_calls": len(inner_tools),
                "inner_tool_time_ms": round(inner_tool_time_ms, 3),
                "inner_llm_calls": int(inner_llm.get("calls", 0)),
                "inner_llm_tokens": {
                    "input": int(inner_llm.get("input", 0)),
                    "output": int(inner_llm.get("output", 0)),
                    "cacheRead": int(inner_llm.get("cacheRead", 0)),
                    "total": int(inner_llm.get("total", 0)),
                },
            }
        )

    top_tools = tools_by_parent.get(None, [])
    top_llm = llm_usage_by_parent.get(None) or _empty_usage()
    top_level = {
        "inner_tool_calls": len(top_tools),
        "inner_tool_time_ms": round(sum(t.duration_ms for t in top_tools), 3),
        "inner_llm_calls": int(top_llm.get("calls", 0)),
        "inner_llm_tokens": {
            "input": int(top_llm.get("input", 0)),
            "output": int(top_llm.get("output", 0)),
            "cacheRead": int(top_llm.get("cacheRead", 0)),
            "total": int(top_llm.get("total", 0)),
        },
    }

    return {
        "total_subagent_calls": total_subagent_calls,
        "total_subagent_time_ms": round(total_subagent_time_ms, 3),
        "avg_subagent_time_ms": round(avg_subagent_time_ms, 3),
        "per_subagent": per_subagent,
        "top_level": top_level,
    }


def print_summary(summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    tool = metrics["tool_call_metrics"]
    llm = metrics["llm_completion_metrics"]

    print("=== PI Events Summary ===")
    print(f"Trace dir: {summary['trace_dir']}")
    print(f"Overall execution time: {metrics['overall_execution_time_ms']:.3f} ms")
    print("")
    print("Tool call metrics (real tools only):")
    print(f"  total time: {tool['total_tool_time_ms']:.3f} ms")
    print(f"  pct of overall time: {tool['tool_time_pct']:.3f}%")
    print(f"  total tool calls: {tool['total_tool_calls']}")
    print(f"  avg tool call time: {tool['avg_tool_time_ms']:.3f} ms")
    print(f"  sum content tokens (est): {tool['sum_tool_content_tokens_est']}")
    print(f"  sum output tokens (est): {tool['sum_tool_output_tokens_est']}")
    print("")
    print("LLM completion metrics:")
    print(f"  total llm generation time: {llm['total_llm_time_ms']:.3f} ms")
    print(f"  pct of overall time: {llm['llm_time_pct']:.3f}%")
    print(f"  total input tokens: {llm['total_input_tokens']}")
    print(f"  total output tokens: {llm['total_output_tokens']}")
    print(f"  total cache read tokens: {llm['total_cache_read_tokens']}")
    print(f"  total tokens: {llm['total_tokens']}")
    unaccounted = metrics.get("unaccounted_metrics") or {}
    if unaccounted:
        print("")
        print("Unaccounted metrics:")
        print(f"  unaccounted time: {unaccounted['unaccounted_time_ms']:.3f} ms")
        print(f"  pct of overall time: {unaccounted['unaccounted_pct_of_overall']:.3f}%")

    sub = metrics.get("subagent_metrics") or {}
    if sub.get("total_subagent_calls", 0) > 0:
        print("")
        print("Subagent metrics:")
        print(f"  total subagent calls: {sub['total_subagent_calls']}")
        print(f"  total subagent time: {sub['total_subagent_time_ms']:.3f} ms")
        print(f"  avg subagent time: {sub['avg_subagent_time_ms']:.3f} ms")
        top = sub.get("top_level") or {}
        print(
            "  top-level (orchestrator): "
            f"{top.get('inner_tool_calls', 0)} tools "
            f"({top.get('inner_tool_time_ms', 0.0):.1f} ms), "
            f"{top.get('inner_llm_calls', 0)} llm calls "
            f"({(top.get('inner_llm_tokens') or {}).get('total', 0)} tokens)"
        )
        for entry in sub.get("per_subagent") or []:
            llm_tok = entry.get("inner_llm_tokens") or {}
            print(
                f"  - {entry['subagent_name']} (id={entry['subagent_id'][:8]}…)"
                f" dur={entry['duration_ms']:.1f}ms"
                f" inner_tools={entry['inner_tool_calls']} ({entry['inner_tool_time_ms']:.1f}ms)"
                f" inner_llm={entry['inner_llm_calls']} ({llm_tok.get('total', 0)} tokens)"
            )

    if llm.get("notes"):
        print("")
        for note in llm["notes"]:
            print(f"Note: {note}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize pi_events.jsonl + tool_calls.log metrics for one trace directory."
    )
    parser.add_argument("trace_dir", type=Path, help="Trace directory containing pi_events.jsonl and tool_calls.log")
    args = parser.parse_args()

    trace_dir = args.trace_dir.resolve()
    summary = build_summary(trace_dir)

    output_path = trace_dir / "pi_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_summary(summary)
    print(f"\nWrote JSON summary to: {output_path}")


if __name__ == "__main__":
    main()
