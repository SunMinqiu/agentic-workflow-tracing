#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from summarize_pi_events import parse_tool_calls_log


@dataclass
class Event:
    run_id: str
    kind: str
    name: str
    start_ms: float
    end_ms: float
    parent_run_id: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    args: dict[str, Any] | None = None
    result_text: str | None = None
    # Worker identity (GenoMAS `genomas_role`, or tool input `role`).  Used to
    # scope containment-based parent inference: in a parallel trace, worker A's
    # long tool can contain worker B's LLM in time without any real nesting.
    role: str | None = None

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


def _dt_to_ms(dt: datetime) -> float:
    return dt.timestamp() * 1000.0


def _datetime_from_ms(ms: float) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0)


def _event_parent(ev: dict[str, Any]) -> str | None:
    parent = ev.get("parent_run_id")
    if isinstance(parent, str) and parent:
        return parent
    # Backward compatibility with the LangChain/SRAgent logger.
    parent = ev.get("parent_subagent_run_id")
    return parent if isinstance(parent, str) and parent else None


def _extract_result_text(result_obj: Any) -> str:
    if not isinstance(result_obj, dict):
        return ""
    content = result_obj.get("content", [])
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            chunks.append(item["text"])
    return "".join(chunks)


def _load_llm_events(
    events_path: Path,
) -> tuple[list[Event], dict[str, str | None], dict[str, str]]:
    starts_by_id: dict[str, dict[str, Any]] = {}
    starts_stack: list[dict[str, Any]] = []
    tool_parent_by_id: dict[str, str | None] = {}
    tool_result_by_id: dict[str, str] = {}
    llms: list[Event] = []

    with events_path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping invalid JSON in pi_events.jsonl:{line_num}",
                    file=sys.stderr,
                )
                continue

            et = ev.get("type")
            if et == "message_update":
                assistant_event = ev.get("assistantMessageEvent")
                if not isinstance(assistant_event, dict):
                    continue
                tool_call = assistant_event.get("toolCall")
                if not isinstance(tool_call, dict):
                    continue
                tool_id = tool_call.get("id")
                if isinstance(tool_id, str):
                    tool_parent_by_id[tool_id] = _event_parent(ev)
                continue

            if et == "tool_execution_end":
                tool_id = ev.get("toolCallId")
                if isinstance(tool_id, str):
                    tool_parent_by_id.setdefault(tool_id, _event_parent(ev))
                    tool_result_by_id[tool_id] = _extract_result_text(ev.get("result"))
                continue

            msg = ev.get("message")
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            ts = msg.get("timestamp")
            if not isinstance(ts, (int, float)):
                continue

            run_id = ev.get("run_id")
            parent = _event_parent(ev)
            role = ev.get("genomas_role")
            if not isinstance(role, str) or not role:
                role = None
            if et == "message_start":
                start = {
                    "run_id": run_id, "start_ms": float(ts),
                    "parent": parent, "role": role,
                }
                if isinstance(run_id, str):
                    starts_by_id[run_id] = start
                else:
                    starts_stack.append(start)
            elif et == "message_end":
                start = None
                if isinstance(run_id, str) and run_id in starts_by_id:
                    start = starts_by_id.pop(run_id)
                elif starts_stack:
                    start = starts_stack.pop()
                if start is None:
                    continue
                rid = start.get("run_id")
                if not isinstance(rid, str) or not rid:
                    rid = f"llm_{len(llms) + 1}"
                usage = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}
                llms.append(
                    Event(
                        run_id=rid,
                        kind="llm",
                        name="LLM",
                        start_ms=float(start["start_ms"]),
                        end_ms=float(ts),
                        parent_run_id=parent or start.get("parent"),
                        usage=usage,
                        error=ev.get("error") if isinstance(ev.get("error"), str) else None,
                        role=role or start.get("role"),
                    )
                )

    return llms, tool_parent_by_id, tool_result_by_id


def _load_tool_events(
    tool_log: Path,
    tool_parent_by_id: dict[str, str | None],
    tool_result_by_id: dict[str, str],
    llm_events: list[Event],
) -> list[Event]:
    calls = parse_tool_calls_log(tool_log)
    if not calls:
        return []

    # Tool logs only carry HH:MM:SS.ffffff. Put them on the LLM event date
    # when available so wall-clock arithmetic is stable across days.
    if llm_events:
        true_date = _datetime_from_ms(min(ev.start_ms for ev in llm_events)).date()
        calls = [
            type(tc)(
                tool_id=tc.tool_id,
                tool_name=tc.tool_name,
                start_time=tc.start_time.replace(
                    year=true_date.year, month=true_date.month, day=true_date.day
                ),
                end_time=tc.end_time.replace(
                    year=true_date.year, month=true_date.month, day=true_date.day
                ),
                duration_ms=tc.duration_ms,
                input_params=tc.input_params,
            )
            for tc in calls
        ]

        # Correct common timezone skew between LLM unix timestamps and HMS logs.
        first_llm = _datetime_from_ms(min(ev.start_ms for ev in llm_events))
        first_tool = min(tc.start_time for tc in calls)
        gap_s = (first_tool - first_llm).total_seconds()
        if abs(gap_s) >= 1800:
            quanta = round(gap_s / 900) * 900
            calls = [
                type(tc)(
                    tool_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    start_time=tc.start_time - timedelta(seconds=quanta),
                    end_time=tc.end_time - timedelta(seconds=quanta),
                    duration_ms=tc.duration_ms,
                    input_params=tc.input_params,
                )
                for tc in calls
            ]

    def _role_of(tc: Any) -> str | None:
        args = tc.input_params
        role = args.get("role") if isinstance(args, dict) else None
        return role if isinstance(role, str) and role else None

    return [
        Event(
            run_id=tc.tool_id,
            kind="tool",
            name=tc.tool_name,
            start_ms=_dt_to_ms(tc.start_time),
            end_ms=_dt_to_ms(tc.end_time),
            parent_run_id=tool_parent_by_id.get(tc.tool_id),
            args=tc.input_params,
            result_text=tool_result_by_id.get(tc.tool_id),
            role=_role_of(tc),
        )
        for tc in calls
    ]


def _load_tool_events_from_pi_events(events_path: Path) -> list[Event]:
    """
    Fallback tool loader for GenoMAS-style traces.

    GenoMAS dual-logs tool executions: tool_calls.log AND pi_events.jsonl
    `tool_execution_start` / `tool_execution_end` records (top-level
    `timestamp` in unix ms, `tool_name`, `genomas_role`).  When tool_calls.log
    is missing (e.g. a partial copy of a trace dir) or empty, rebuild tool
    intervals from the events file so DAG/parallelism stats stay correct.
    """
    starts: dict[str, dict[str, Any]] = {}
    tools: list[Event] = []
    with events_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            et = ev.get("type")
            if et not in ("tool_execution_start", "tool_execution_end"):
                continue
            rid = ev.get("run_id")
            ts = ev.get("timestamp")
            if not isinstance(rid, str) or not isinstance(ts, (int, float)):
                continue
            role = ev.get("genomas_role")
            role = role if isinstance(role, str) and role else None
            if et == "tool_execution_start":
                name = ev.get("tool_name")
                starts[rid] = {
                    "start_ms": float(ts),
                    "name": name if isinstance(name, str) and name else "tool",
                    "role": role,
                    "parent": _event_parent(ev),
                }
            elif rid in starts:
                st = starts.pop(rid)
                tools.append(
                    Event(
                        run_id=rid,
                        kind="tool",
                        name=st["name"],
                        start_ms=st["start_ms"],
                        end_ms=float(ts),
                        parent_run_id=st["parent"],
                        role=role or st["role"],
                    )
                )
    return tools


def load_events(trace_dir: Path) -> dict[str, Event]:
    events_path = trace_dir / "pi_events.jsonl"
    tool_log = trace_dir / "tool_calls.log"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing {events_path}")

    llms, tool_parent_by_id, tool_result_by_id = _load_llm_events(events_path)
    tools: list[Event] = []
    if tool_log.exists():
        tools = _load_tool_events(tool_log, tool_parent_by_id, tool_result_by_id, llms)
    if not tools:
        tools = _load_tool_events_from_pi_events(events_path)
    events = {ev.run_id: ev for ev in [*llms, *tools]}
    if events and not any(ev.parent_run_id for ev in events.values()):
        _infer_parents_by_containment(events)
    return events


def _infer_parents_by_containment(events: dict[str, Event]) -> None:
    """
    Backward-compatible fallback for Phase 1/2 traces.

    Old pi_events.jsonl files have no parent_run_id, but long orchestrator
    tools such as Run_analysis fully contain the sub-agent LLM calls.  Infer
    the nearest enclosing tool as parent only when a trace has no explicit
    parent metadata at all.

    Containment is scoped by worker role when both sides carry one: in a
    parallel trace (GenoMAS mw>=2), worker A's long tool can contain worker
    B's LLM purely by timing — that overlap is real parallelism, not nesting,
    and must not eat into A's self-time.
    """
    tools = [ev for ev in events.values() if ev.kind == "tool"]
    for ev in sorted(events.values(), key=lambda item: item.start_ms):
        if ev.kind == "tool":
            continue
        best: tuple[float, str] | None = None
        for tool in tools:
            if tool.run_id == ev.run_id:
                continue
            if ev.role and tool.role and ev.role != tool.role:
                continue
            if tool.start_ms <= ev.start_ms and tool.end_ms >= ev.end_ms:
                span = tool.end_ms - tool.start_ms
                if best is None or span < best[0]:
                    best = (span, tool.run_id)
        if best is not None:
            ev.parent_run_id = best[1]


def union_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
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


def intersection_active(
    a_intervals: list[tuple[float, float]],
    b_intervals: list[tuple[float, float]],
) -> float:
    points: list[tuple[float, int, int]] = []
    for s, e in a_intervals:
        if e > s:
            points.append((s, 1, 0))
            points.append((e, -1, 0))
    for s, e in b_intervals:
        if e > s:
            points.append((s, 0, 1))
            points.append((e, 0, -1))
    if not points:
        return 0.0
    points.sort(key=lambda p: (p[0], p[1] + p[2]))
    active_a = active_b = 0
    last = points[0][0]
    total = 0.0
    for t, da, db in points:
        if t > last and active_a > 0 and active_b > 0:
            total += t - last
        active_a += da
        active_b += db
        last = t
    return total


def k_active(intervals: list[tuple[float, float]], k: int) -> float:
    points: list[tuple[float, int]] = []
    for s, e in intervals:
        if e > s:
            points.append((s, 1))
            points.append((e, -1))
    if not points:
        return 0.0
    points.sort(key=lambda p: (p[0], p[1]))
    active = 0
    last = points[0][0]
    total = 0.0
    for t, delta in points:
        if t > last and active >= k:
            total += t - last
        active += delta
        last = t
    return total


def active_degree(
    intervals: list[tuple[float, float]],
    wall_ms: float,
) -> dict[str, Any]:
    """
    Time-weighted active-count stats for intervals.

    This is the real "parallel degree" family: average/max number of active
    units at the same time.  `avg_active_over_wall` includes idle gaps;
    `avg_active_when_busy` uses only union(intervals) as the denominator.
    """
    points_by_t: dict[float, int] = defaultdict(int)
    for s, e in intervals:
        if e > s:
            points_by_t[s] += 1
            points_by_t[e] -= 1
    if not points_by_t:
        return {
            "avg_active_over_wall": 0.0,
            "avg_active_when_busy": 0.0,
            "max_active": 0,
            "busy_time_s": 0.0,
            "time_at_degree_ge_2_s": 0.0,
            "parallel_time_ratio": 0.0,
            "time_by_degree_s": {},
        }

    active = 0
    last: float | None = None
    active_area = 0.0
    busy_ms = 0.0
    ge2_ms = 0.0
    max_active = 0
    time_by_degree: dict[int, float] = defaultdict(float)

    for t in sorted(points_by_t):
        if last is not None and t > last:
            dur = t - last
            if active > 0:
                active_area += active * dur
                busy_ms += dur
                time_by_degree[active] += dur
                if active >= 2:
                    ge2_ms += dur
        active += points_by_t[t]
        max_active = max(max_active, active)
        last = t

    return {
        "avg_active_over_wall": round(active_area / wall_ms, 6) if wall_ms else 0.0,
        "avg_active_when_busy": round(active_area / busy_ms, 6) if busy_ms else 0.0,
        "max_active": max_active,
        "busy_time_s": round(busy_ms / 1000.0, 6),
        "time_at_degree_ge_2_s": round(ge2_ms / 1000.0, 6),
        "parallel_time_ratio": round(ge2_ms / busy_ms, 6) if busy_ms else 0.0,
        "time_by_degree_s": {
            str(k): round(v / 1000.0, 6)
            for k, v in sorted(time_by_degree.items())
        },
    }


def _load_observed_pid_intervals(trace_dir: Path) -> list[tuple[float, float]]:
    """
    Coarse OS-process parallelism from parsed.json.

    We only have syscall observations, not scheduler state, so a PID interval
    is first_observed_syscall..last_observed_syscall.  This estimates observed
    process overlap and may overstate true CPU activity when a child is alive
    but idle.
    """
    parsed_json = trace_dir / "parsed.json"
    if not parsed_json.exists():
        return []
    try:
        data = json.loads(parsed_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    fs_entries = data.get("fs_entries")
    if not isinstance(fs_entries, list):
        return []
    by_pid: dict[int, list[datetime]] = defaultdict(list)
    for entry in fs_entries:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("pid")
        ts = entry.get("timestamp")
        if not isinstance(pid, int) or not isinstance(ts, str):
            continue
        try:
            by_pid[pid].append(datetime.fromisoformat(ts))
        except ValueError:
            continue
    intervals: list[tuple[float, float]] = []
    for timestamps in by_pid.values():
        if not timestamps:
            continue
        start = min(timestamps)
        end = max(timestamps)
        s_ms = _dt_to_ms(start)
        e_ms = _dt_to_ms(end)
        if e_ms > s_ms:
            intervals.append((s_ms, e_ms))
    return intervals


def _subtract_intervals(
    base: tuple[float, float], blockers: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    pieces = [base]
    for bs, be in sorted(blockers):
        next_pieces: list[tuple[float, float]] = []
        for s, e in pieces:
            if be <= s or bs >= e:
                next_pieces.append((s, e))
                continue
            if bs > s:
                next_pieces.append((s, min(bs, e)))
            if be < e:
                next_pieces.append((max(be, s), e))
        pieces = next_pieces
    return [(s, e) for s, e in pieces if e > s]


def build_children(events: dict[str, Event]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for rid, ev in events.items():
        if ev.parent_run_id and ev.parent_run_id in events:
            children[ev.parent_run_id].append(rid)
    for kids in children.values():
        kids.sort(key=lambda rid: events[rid].start_ms)
    return children


def build_sequence_edges(
    events: dict[str, Event],
    children: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """
    Serial edges between consecutive events with the same parent context.

    Parent-child edges answer "contained by"; sequence edges answer "then".
    They are separate because siblings under Run_analysis are usually serial
    LLM/code-exec steps even though their structural parent is the same.
    """
    buckets: dict[str | None, list[str]] = defaultdict(list)
    for rid, ev in events.items():
        parent = ev.parent_run_id if ev.parent_run_id in events else None
        buckets[parent].append(rid)

    edges: list[tuple[str, str]] = []
    child_pairs = {(parent, child) for parent, kids in children.items() for child in kids}
    for siblings in buckets.values():
        siblings.sort(key=lambda rid: (events[rid].start_ms, events[rid].end_ms))
        for prev, cur in zip(siblings, siblings[1:]):
            if (prev, cur) not in child_pairs and prev != cur:
                edges.append((prev, cur))
    return edges


def compute_self_intervals(
    events: dict[str, Event],
    children: dict[str, list[str]],
) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for rid, ev in events.items():
        blockers = [
            (events[ch].start_ms, events[ch].end_ms)
            for ch in children.get(rid, [])
            if ch in events
        ]
        out[rid] = _subtract_intervals((ev.start_ms, ev.end_ms), blockers)
    return out


def _max_leaf_width(events: dict[str, Event], children: dict[str, list[str]]) -> int:
    if not events:
        return 0
    points = sorted({t for ev in events.values() for t in (ev.start_ms, ev.end_ms)})
    best = 0
    for i in range(len(points) - 1):
        mid = (points[i] + points[i + 1]) / 2.0
        active = [
            rid for rid, ev in events.items()
            if ev.start_ms <= mid < ev.end_ms
        ]
        active_set = set(active)
        leaves = [
            rid for rid in active
            if not any(ch in active_set for ch in children.get(rid, []))
        ]
        best = max(best, len(leaves))
    return best


def _depth(rid: str, children: dict[str, list[str]]) -> int:
    kids = children.get(rid, [])
    if not kids:
        return 1
    return 1 + max(_depth(ch, children) for ch in kids)


def _pairwise(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
) -> dict[str, float]:
    overlap = intersection_active(a, b)
    union = union_length([*a, *b])
    return {
        "overlap_s": round(overlap / 1000.0, 6),
        "union_s": round(union / 1000.0, 6),
        "ratio": round(overlap / union, 6) if union else 0.0,
    }


def compute_summary(events: dict[str, Event]) -> dict[str, Any]:
    if not events:
        return {}
    children = build_children(events)
    self_intervals_by_id = compute_self_intervals(events, children)
    starts = [ev.start_ms for ev in events.values()]
    ends = [ev.end_ms for ev in events.values()]
    wall_ms = max(ends) - min(starts)

    llm_self = [
        iv for rid, intervals in self_intervals_by_id.items()
        if events[rid].kind == "llm"
        for iv in intervals
    ]
    tool_self = [
        iv for rid, intervals in self_intervals_by_id.items()
        if events[rid].kind == "tool"
        for iv in intervals
    ]
    all_self = [iv for intervals in self_intervals_by_id.values() for iv in intervals]
    total_self_ms = sum(e - s for s, e in all_self)

    roots = [
        rid for rid, ev in events.items()
        if not ev.parent_run_id or ev.parent_run_id not in events
    ]
    depth = max((_depth(rid, children) for rid in roots), default=0)

    top_level_tools = [
        ev for ev in events.values()
        if ev.kind == "tool" and (not ev.parent_run_id or ev.parent_run_id not in events)
    ]
    internal_events = [
        ev for ev in events.values()
        if ev.parent_run_id and ev.parent_run_id in events
    ]

    return {
        "wall_clock_s": round(wall_ms / 1000.0, 6),
        "total_self_time_s": round(total_self_ms / 1000.0, 6),
        "workload_concurrency_factor": (
            round(total_self_ms / wall_ms, 6) if wall_ms else 0.0
        ),
        "parallel_time_ratio": {
            "llm_x_tool": _pairwise(llm_self, tool_self),
            "tool_x_tool": {
                "overlap_s": round(k_active(tool_self, 2) / 1000.0, 6),
                "union_s": round(union_length(tool_self) / 1000.0, 6),
                "ratio": (
                    round(k_active(tool_self, 2) / union_length(tool_self), 6)
                    if union_length(tool_self) else 0.0
                ),
            },
            "llm_x_llm": {
                "overlap_s": round(k_active(llm_self, 2) / 1000.0, 6),
                "union_s": round(union_length(llm_self) / 1000.0, 6),
                "ratio": (
                    round(k_active(llm_self, 2) / union_length(llm_self), 6)
                    if union_length(llm_self) else 0.0
                ),
            },
        },
        "parallel_degree": {
            "semantic_events": active_degree(all_self, wall_ms),
            "llm_events": active_degree(llm_self, wall_ms),
            "tool_events": active_degree(tool_self, wall_ms),
        },
        "structural": {
            "depth": depth,
            "width_max": _max_leaf_width(events, children),
            "top_level_tools": len(top_level_tools),
            "subagent_internal_events": len(internal_events),
        },
    }


def compute_trace_summary(trace_dir: Path) -> dict[str, Any]:
    events = load_events(trace_dir)
    summary = compute_summary(events)
    if not summary:
        return summary

    starts = [ev.start_ms for ev in events.values()]
    ends = [ev.end_ms for ev in events.values()]
    wall_ms = max(ends) - min(starts)
    pid_intervals = _load_observed_pid_intervals(trace_dir)
    summary["parallel_degree"]["observed_processes"] = {
        **active_degree(pid_intervals, wall_ms),
        "unit": "pid",
        "notes": [
            "Estimated from parsed.json fs syscall observations.",
            "Interval per PID is first_observed_syscall..last_observed_syscall; this is not scheduler CPU residency.",
            "Thread-level parallelism is unavailable unless the tracer records TIDs separately.",
        ],
    }
    return summary


def _fmt_duration(ms: float) -> str:
    seconds = ms / 1000.0
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _safe_json_loads(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _agent_name_for_tool(ev: Event, events: dict[str, Event]) -> str | None:
    if ev.name != "Run_analysis":
        return None
    agent_id = (ev.args or {}).get("agent_id")
    if agent_id is None:
        return None
    for candidate in events.values():
        if candidate.name != "Select_agent":
            continue
        result = _safe_json_loads(candidate.result_text)
        if not isinstance(result, dict):
            continue
        if result.get("agent_id") == agent_id and isinstance(result.get("agent_name"), str):
            return result["agent_name"]
    return f"agent_id={agent_id}"


def _event_title(ev: Event, events: dict[str, Event], llm_index: dict[str, int]) -> str:
    if ev.kind == "llm":
        return f"LLM #{llm_index.get(ev.run_id, 0)}"
    agent_name = _agent_name_for_tool(ev, events)
    if agent_name:
        return f"Tool: {ev.name} -> {agent_name}"
    return f"Tool: {ev.name}"


def _event_detail(ev: Event) -> str:
    parts = [_fmt_duration(ev.duration_ms)]
    if ev.kind == "llm" and ev.usage:
        total = ev.usage.get("totalTokens") or ev.usage.get("total")
        if total:
            parts.append(f"{total} tokens")
    if ev.args:
        interesting = []
        for key in ("agent_id", "analysis_goal", "objective", "script_len", "working_dir"):
            if key in ev.args and ev.args[key] not in (None, ""):
                value = str(ev.args[key])
                if len(value) > 70:
                    value = value[:67] + "..."
                interesting.append(f"{key}={value}")
        if interesting:
            parts.append("; ".join(interesting))
    if ev.error:
        parts.append("ERROR")
    return " | ".join(parts)


def render_call_tree(events: dict[str, Event]) -> str:
    if not events:
        return "Session (no events)\n"
    children = build_children(events)
    self_intervals = compute_self_intervals(events, children)
    roots = [
        rid for rid, ev in events.items()
        if not ev.parent_run_id or ev.parent_run_id not in events
    ]
    roots.sort(key=lambda rid: events[rid].start_ms)
    start_ms = min(ev.start_ms for ev in events.values())
    end_ms = max(ev.end_ms for ev in events.values())
    total_self_ms = sum(
        e - s for intervals in self_intervals.values() for s, e in intervals
    )
    unaccounted_ms = max((end_ms - start_ms) - union_length(
        [(ev.start_ms, ev.end_ms) for ev in events.values()]
    ), 0.0)

    llm_index: dict[str, int] = {}
    script_index = 0
    for ev in sorted(events.values(), key=lambda e: e.start_ms):
        if ev.kind == "llm":
            llm_index[ev.run_id] = len(llm_index) + 1

    def label_for(rid: str) -> str:
        nonlocal script_index
        ev = events[rid]
        own_self_ms = sum(e - s for s, e in self_intervals.get(rid, []))
        if ev.kind == "llm":
            return f"{_event_title(ev, events, llm_index)} ({_event_detail(ev)})"
        if ev.name == "ScriptExec":
            script_index += 1
            return (
                f"Tool: ScriptExec #{script_index} "
                f"({_fmt_duration(own_self_ms)} self, {_fmt_duration(ev.duration_ms)} total)"
            )
        return (
            f"{_event_title(ev, events, llm_index)} "
            f"({_fmt_duration(own_self_ms)} self, {_fmt_duration(ev.duration_ms)} total)"
        )

    lines = [
        (
            f"Session ({_fmt_duration(end_ms - start_ms)} total, "
            f"{_fmt_duration(total_self_ms)} self, "
            f"{_fmt_duration(unaccounted_ms)} unaccounted)"
        )
    ]

    def walk(rid: str, prefix: str, is_last: bool) -> None:
        branch = "`- " if is_last else "+- "
        lines.append(f"{prefix}{branch}{label_for(rid)}")
        next_prefix = prefix + ("   " if is_last else "|  ")
        kids = children.get(rid, [])
        for idx, ch in enumerate(kids):
            walk(ch, next_prefix, idx == len(kids) - 1)

    for idx, rid in enumerate(roots):
        walk(rid, "", idx == len(roots) - 1)
    return "\n".join(lines) + "\n"


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_call_dag_dot(events: dict[str, Event]) -> str:
    children = build_children(events)
    sequence_edges = build_sequence_edges(events, children)
    llm_index = {
        ev.run_id: idx
        for idx, ev in enumerate(
            [e for e in sorted(events.values(), key=lambda item: item.start_ms) if e.kind == "llm"],
            1,
        )
    }
    lines = [
        "digraph call_dag {",
        "  rankdir=LR;",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\"];",
        "  edge [fontname=\"Helvetica\"];",
    ]
    for rid, ev in sorted(events.items(), key=lambda item: item[1].start_ms):
        color = "#d8f5d2" if ev.kind == "llm" else "#ffd6a5"
        if ev.name == "ScriptExec":
            color = "#d9c2ff"
        label = f"{_event_title(ev, events, llm_index)}\\n{_event_detail(ev)}\\n{rid[:8]}"
        lines.append(f'  "{_dot_escape(rid)}" [label="{_dot_escape(label)}", fillcolor="{color}"];')
    for parent, kids in children.items():
        for child in kids:
            lines.append(
                f'  "{_dot_escape(parent)}" -> "{_dot_escape(child)}" '
                '[label="contains", color="#7f8c8d"];'
            )
    for prev, cur in sequence_edges:
        lines.append(
            f'  "{_dot_escape(prev)}" -> "{_dot_escape(cur)}" '
            '[label="next", style="dashed", color="#34495e", constraint="false"];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_call_dag_html(events: dict[str, Event]) -> str:
    children = build_children(events)
    sequence_edges = build_sequence_edges(events, children)
    roots = [
        rid for rid, ev in events.items()
        if not ev.parent_run_id or ev.parent_run_id not in events
    ]
    roots.sort(key=lambda rid: events[rid].start_ms)
    llm_index = {
        ev.run_id: idx
        for idx, ev in enumerate(
            [e for e in sorted(events.values(), key=lambda item: item.start_ms) if e.kind == "llm"],
            1,
        )
    }

    order: list[str] = []

    def walk(rid: str) -> None:
        order.append(rid)
        for child in children.get(rid, []):
            walk(child)

    for root in roots:
        walk(root)

    depth_by_id: dict[str, int] = {}
    for root in roots:
        stack = [(root, 0)]
        while stack:
            rid, depth = stack.pop()
            depth_by_id[rid] = depth
            for child in reversed(children.get(rid, [])):
                stack.append((child, depth + 1))

    row_h = 92
    col_w = 330
    margin_x = 30
    margin_y = 30
    node_w = 270
    node_h = 58
    height = max(160, margin_y * 2 + len(order) * row_h)
    width = max(900, margin_x * 2 + (max(depth_by_id.values(), default=0) + 1) * col_w)
    pos: dict[str, tuple[int, int]] = {}
    for idx, rid in enumerate(order):
        pos[rid] = (margin_x + depth_by_id.get(rid, 0) * col_w, margin_y + idx * row_h)

    edges_svg: list[str] = []
    for parent, kids in children.items():
        if parent not in pos:
            continue
        px, py = pos[parent]
        for child in kids:
            if child not in pos:
                continue
            cx, cy = pos[child]
            x1, y1 = px + node_w, py + node_h / 2
            x2, y2 = cx, cy + node_h / 2
            mid = (x1 + x2) / 2
            edges_svg.append(
                f'<path d="M{x1},{y1} C{mid},{y1} {mid},{y2} {x2},{y2}" '
                'fill="none" stroke="#7f8c8d" stroke-width="1.5" marker-end="url(#arrow)" />'
            )

    sequence_svg: list[str] = []
    for prev, cur in sequence_edges:
        if prev not in pos or cur not in pos:
            continue
        px, py = pos[prev]
        cx, cy = pos[cur]
        x1, y1 = px + node_w / 2, py + node_h
        x2, y2 = cx + node_w / 2, cy
        mid_y = (y1 + y2) / 2
        sequence_svg.append(
            f'<path d="M{x1},{y1} C{x1},{mid_y} {x2},{mid_y} {x2},{y2}" '
            'fill="none" stroke="#34495e" stroke-width="1.2" stroke-dasharray="5 4" '
            'marker-end="url(#arrowSeq)" opacity="0.75" />'
        )

    nodes_svg: list[str] = []
    for rid in order:
        ev = events[rid]
        x, y = pos[rid]
        fill = "#d8f5d2" if ev.kind == "llm" else "#ffd6a5"
        stroke = "#2ecc71" if ev.kind == "llm" else "#f39c12"
        if ev.name == "ScriptExec":
            fill = "#eadcff"
            stroke = "#9b59b6"
        title = html.escape(_event_title(ev, events, llm_index))
        detail = html.escape(_event_detail(ev))
        rid_short = html.escape(rid[:8])
        nodes_svg.append(
            f'<g class="node">'
            f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="8" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" />'
            f'<text x="{x + 12}" y="{y + 20}" class="title">{title}</text>'
            f'<text x="{x + 12}" y="{y + 39}" class="detail">{detail}</text>'
            f'<text x="{x + node_w - 64}" y="{y + 52}" class="id">{rid_short}</text>'
            f'</g>'
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Call DAG</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; }}
    header {{ padding: 16px 24px; border-bottom: 1px solid #ddd; }}
    h1 {{ margin: 0; font-size: 20px; }}
    p {{ margin: 6px 0 0; color: #52606d; }}
    svg {{ display: block; background: #fff; }}
    text {{ dominant-baseline: middle; }}
    .title {{ font-weight: 700; font-size: 13px; }}
    .detail {{ font-size: 11px; fill: #334e68; }}
    .id {{ font-size: 10px; fill: #627d98; }}
    .legend {{ margin-top: 8px; font-size: 13px; color: #52606d; }}
    .legend span {{ display: inline-block; margin-right: 18px; }}
  </style>
</head>
<body>
  <header>
    <h1>Call DAG</h1>
    <p>Nodes are LLM calls and tool/script executions.</p>
    <div class="legend">
      <span>solid edge: parent contains child</span>
      <span>dashed edge: serial next step under the same parent</span>
    </div>
  </header>
  <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <defs>
      <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
        <path d="M0,0 L0,6 L7,3 z" fill="#7f8c8d" />
      </marker>
      <marker id="arrowSeq" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
        <path d="M0,0 L0,6 L7,3 z" fill="#34495e" />
      </marker>
    </defs>
    {''.join(edges_svg)}
    {''.join(sequence_svg)}
    {''.join(nodes_svg)}
  </svg>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute DAG and parallelism metrics for a PI trace directory."
    )
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()

    trace_dir = args.trace_dir.resolve()
    events = load_events(trace_dir)
    summary = compute_trace_summary(trace_dir)

    summary_path = trace_dir / "parallelism_summary.json"
    tree_path = trace_dir / "call_tree.txt"
    dag_dot_path = trace_dir / "call_dag.dot"
    dag_html_path = trace_dir / "call_dag.html"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tree_path.write_text(render_call_tree(events), encoding="utf-8")
    dag_dot_path.write_text(render_call_dag_dot(events), encoding="utf-8")
    dag_html_path.write_text(render_call_dag_html(events), encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {tree_path}")
    print(f"Wrote {dag_dot_path}")
    print(f"Wrote {dag_html_path}")


if __name__ == "__main__":
    main()
