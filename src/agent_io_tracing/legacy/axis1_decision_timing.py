"""Retired Axis 1 decision-timing analysis.

This metric is kept here only for audit/backward reproduction. It was removed
from the main index because it estimates a path "decision" time from the first
observable path string in tool input, generated code, or assistant text. For
planning-heavy agents and orchestrators, paths can be assembled internally or
represented as action units without ever appearing in observable text, which
systematically makes decisions look later than they were and compresses the
decision-to-access lead time.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent_io_tracing.analysis.phase1_metrics import (
    DATA_SYSCALLS,
    make_workload_filter,
    pct,
    percentile,
)


PATHLIKE_EXTENSIONS = {
    ".csv", ".tsv", ".json", ".jsonl", ".txt", ".log", ".parquet", ".h5",
    ".hdf5", ".hdf", ".nc", ".npy", ".npz", ".pkl", ".pickle", ".fa",
    ".fasta", ".fastq", ".fq", ".bam", ".sam", ".vcf", ".bed", ".gtf",
    ".gff", ".py", ".sh", ".r", ".R", ".out", ".err", ".xml", ".yaml",
    ".yml", ".ini", ".toml", ".db", ".sqlite",
}


def _is_pathlike_string(value: str) -> bool:
    s = value.strip()
    if not s or len(s) > 4096:
        return False
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        return False
    if "/" in s or s.startswith((".", "~")):
        return True
    suffix = Path(s).suffix
    return suffix in PATHLIKE_EXTENSIONS


def _extract_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(_extract_strings(value))
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            out.extend(_extract_strings(value))
    return out


def _extract_code_strings(code: str) -> list[str]:
    strings: list[str] = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                strings.append(node.value)
            elif isinstance(node, ast.Str):
                strings.append(node.s)
    except SyntaxError:
        for match in re.finditer(r"""(['"])(.*?)\1""", code, flags=re.DOTALL):
            strings.append(match.group(2))
    return strings


def _path_join_keys(path_text: str) -> set[str]:
    s = path_text.strip().strip("'\"")
    if not s:
        return set()
    s = s.replace("\\", "/")
    s = re.sub(r"^file://", "", s)
    s = s.split("#", 1)[0].split("?", 1)[0]
    s = re.sub(r"/+", "/", s)
    s = s.rstrip("/")
    if not s:
        return set()
    norm = str(Path(s).as_posix())
    while norm.startswith("./"):
        norm = norm[2:]
    keys = {norm, norm.lstrip("/")}
    parts = [p for p in norm.strip("/").split("/") if p]
    for i in range(len(parts)):
        keys.add("/".join(parts[i:]))
    if parts:
        keys.add(parts[-1])
    return {k for k in keys if k and k != "."}


def _parse_generated_timestamp(value: Any, tz_offset_seconds: float) -> datetime | None:
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000.0 if float(value) > 10_000_000_000 else float(value)
        return datetime.fromtimestamp(seconds) - timedelta(seconds=tz_offset_seconds)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                seconds = float(value) / 1000.0 if float(value) > 10_000_000_000 else float(value)
                return datetime.fromtimestamp(seconds) - timedelta(seconds=tz_offset_seconds)
            except ValueError:
                return None
    return None


def _decision_events_from_tool_calls(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for tc in parsed.get("tool_calls", []):
        try:
            ts = datetime.fromisoformat(str(tc.get("start_time")))
        except ValueError:
            continue
        for value in _extract_strings(tc.get("input_params") or {}):
            if _is_pathlike_string(value):
                decisions.append({"path": value, "timestamp": ts, "source": "toolcall"})
    return decisions


def _decision_events_from_generated_code(
    trace_dir: Path,
    tz_offset_seconds: float,
) -> list[dict[str, Any]]:
    path = trace_dir / "generated_code.jsonl"
    decisions: list[dict[str, Any]] = []
    if not path.is_file():
        return decisions
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = _parse_generated_timestamp(rec.get("timestamp"), tz_offset_seconds)
        code = rec.get("code")
        if ts is None or not isinstance(code, str):
            continue
        for value in _extract_code_strings(code):
            if _is_pathlike_string(value):
                decisions.append({"path": value, "timestamp": ts, "source": "code"})
    return decisions


def _extract_pathlike_text_strings(text: str) -> list[str]:
    strings: list[str] = []
    for match in re.finditer(r"[^\s,;:()\[\]{}<>]+", text):
        value = match.group(0).strip("'\"`“”‘’.,;:")
        if _is_pathlike_string(value):
            strings.append(value)
    return strings


def _decision_events_from_assistant_text(
    trace_dir: Path,
    tz_offset_seconds: float,
) -> list[dict[str, Any]]:
    path = trace_dir / "pi_events.jsonl"
    decisions: list[dict[str, Any]] = []
    if not path.is_file():
        return decisions
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "message_end":
            continue
        msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
        ts = _parse_generated_timestamp(
            msg.get("timestamp") or rec.get("timestamp"),
            tz_offset_seconds,
        )
        if ts is None:
            continue
        for field, source in (("assistant_text", "output"), ("assistant_thinking", "thinking")):
            text = rec.get(field)
            if not isinstance(text, str) or not text:
                continue
            for value in _extract_pathlike_text_strings(text):
                decisions.append({"path": value, "timestamp": ts, "source": source})
    return decisions


def _message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _decision_events_from_prompt(
    trace_dir: Path,
    tz_offset_seconds: float,
) -> list[dict[str, Any]]:
    path = trace_dir / "pi_events.jsonl"
    decisions: list[dict[str, Any]] = []
    if not path.is_file():
        return decisions
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "message_start":
            continue
        msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
        if msg.get("role") != "user":
            continue
        ts = _parse_generated_timestamp(
            msg.get("timestamp") or rec.get("timestamp"),
            tz_offset_seconds,
        )
        if ts is None:
            continue
        for value in _extract_pathlike_text_strings(_message_text(msg)):
            decisions.append({"path": value, "timestamp": ts, "source": "prompt"})
    return decisions


def _first_decisions_by_path(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for d in decisions:
        path = str(d.get("path") or "")
        keys = _path_join_keys(path)
        if not keys:
            continue
        canonical = max(keys, key=lambda k: (k.count("/"), len(k)))
        existing = by_path.get(canonical)
        if existing is None or d["timestamp"] < existing["timestamp"]:
            by_path[canonical] = {**d, "keys": keys, "canonical": canonical}
    return list(by_path.values())


def compute_decision_access_lead_time(
    parsed: dict[str, Any],
    trace_dir: Path,
    artifacts: list[dict[str, Any]],
    prefetch_threshold_s: float = 1.0,
) -> dict[str, Any]:
    _wl = make_workload_filter(artifacts)
    tz_offset_seconds = float((parsed.get("summary") or {}).get("tz_offset_seconds") or 0.0)
    raw_decisions = _decision_events_from_tool_calls(parsed)
    raw_decisions.extend(_decision_events_from_generated_code(trace_dir, tz_offset_seconds))
    raw_decisions.extend(_decision_events_from_assistant_text(trace_dir, tz_offset_seconds))
    raw_decisions.extend(_decision_events_from_prompt(trace_dir, tz_offset_seconds))
    decisions = _first_decisions_by_path(raw_decisions)

    decision_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in decisions:
        for key in d["keys"]:
            decision_by_key[key].append(d)

    first_access_by_path: dict[str, dict[str, Any]] = {}
    access_syscalls = DATA_SYSCALLS | {"open", "openat"}
    for e in parsed.get("fs_entries", []):
        syscall = str(e.get("syscall") or "")
        if syscall not in access_syscalls:
            continue
        path = e.get("path")
        ts_s = e.get("timestamp")
        if not isinstance(path, str) or not isinstance(ts_s, str):
            continue
        if not _wl(path):
            continue
        try:
            ts = datetime.fromisoformat(ts_s)
        except ValueError:
            continue
        key = path.rstrip("/")
        existing = first_access_by_path.get(key)
        if existing is None or ts < existing["timestamp"]:
            first_access_by_path[key] = {"path": path, "timestamp": ts, "syscall": syscall}

    leads: list[float] = []
    by_source = Counter()
    resolved = 0
    unresolvable = 0
    ambiguous = 0
    decided_at_access = 0

    for access in first_access_by_path.values():
        keys = sorted(_path_join_keys(access["path"]), key=lambda k: (-k.count("/"), -len(k)))
        chosen: dict[str, Any] | None = None
        saw_ambiguous = False
        for key in keys:
            candidates = decision_by_key.get(key) or []
            unique = {c["canonical"] for c in candidates}
            if len(unique) > 1:
                saw_ambiguous = True
                continue
            if candidates:
                chosen = min(candidates, key=lambda c: c["timestamp"])
                break
        if chosen is None:
            if saw_ambiguous:
                ambiguous += 1
            else:
                unresolvable += 1
            continue
        lead = (access["timestamp"] - chosen["timestamp"]).total_seconds()
        if lead < 0:
            lead = 0.0
            decided_at_access += 1
        leads.append(lead)
        resolved += 1
        by_source[str(chosen.get("source") or "unknown")] += 1

    total_access_paths = len(first_access_by_path)
    return {
        "n_access_paths": total_access_paths,
        "resolved": resolved,
        "unresolvable": unresolvable,
        "ambiguous": ambiguous,
        "unresolvable_fraction": (
            unresolvable / total_access_paths if total_access_paths else None
        ),
        "lead_time_s": {
            "count": len(leads),
            "p50": percentile(leads, 50),
            "p95": percentile(leads, 95),
            "p99": percentile(leads, 99),
            "mean": (sum(leads) / len(leads)) if leads else None,
        },
        "lead_samples_s": sorted(leads),
        "pct_unprefetchable_lt_1s": pct(leads, lambda x: x < prefetch_threshold_s),
        "decided_at_access_count": decided_at_access,
        "by_source": dict(by_source),
    }


def create_decision_access_lead_time_matplotlib(trace_dir: Path, output_path: Path) -> None:
    try:
        p1 = json.loads((trace_dir / "phase1_metrics.json").read_text())
    except Exception:
        p1 = {}
    metric = p1.get("decision_access_lead_time") or {}
    samples = [float(x) for x in (metric.get("lead_samples_s") or []) if isinstance(x, (int, float))]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if not samples:
        resolved = int(metric.get("resolved") or 0)
        unresolvable = int(metric.get("unresolvable") or 0)
        ambiguous = int(metric.get("ambiguous") or 0)
        counts = [resolved, unresolvable, ambiguous]
        labels = ["resolved", "runtime-built/\nunobserved", "ambiguous"]
        colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
        ax.bar(labels, counts, color=colors)
        for idx, val in enumerate(counts):
            ax.text(idx, val, f" {val}", ha="center", va="bottom", fontsize=10)
        total = int(metric.get("n_access_paths") or sum(counts))
        ax.set_ylabel("first-access paths", fontsize=10)
        ax.set_title(
            "Decision→access path coverage\n"
            f"0 resolved lead-time samples out of {total} access path(s)",
            fontsize=13,
        )
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.5,
            -0.22,
            "No CDF is drawn because no path appeared in observable tool input/generated code "
            "before first access.",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            color="#7f8c8d",
            wrap=True,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    xs = np.array(sorted(samples))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color="#34495e", linewidth=2)
    ax.axvline(1.0, color="#d62728", linestyle="--", linewidth=1, label="<1s threshold")
    ax.set_xlabel("seconds from observable path decision to first access", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    lt1 = metric.get("pct_unprefetchable_lt_1s")
    unresolved = metric.get("unresolvable_fraction")
    subtitle = (
        f"n={len(xs)} · <1s={lt1:.1f}%"
        if isinstance(lt1, (int, float)) else f"n={len(xs)}"
    )
    if isinstance(unresolved, (int, float)):
        subtitle += f" · unresolvable={100.0 * unresolved:.1f}%"
    ax.set_title("Decision→access lead-time CDF\n" + subtitle, fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
