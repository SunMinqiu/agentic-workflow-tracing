#!/usr/bin/env python3
"""
Parse ebpf_events.log and correlate filesystem activity to Claude tool calls.

This keeps output compatible with the packaged `visualize_strace.py` by writing
the same parsed.json schema:
  - tool_calls
  - fs_entries
  - summary
  - tool_summaries
"""

from __future__ import annotations

import argparse
import ast
import json
import posixpath
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


TOOL_CALL_PATTERN = re.compile(
    r"\[(\d{2}:\d{2}:\d{2}\.\d+)\s*->\s*(\d{2}:\d{2}:\d{2}\.\d+)\]\s*"
    r"\([\d.]+ms\)\s*"
    r"(\w+)\s*"
    r"\(id=([^)]+)\)\s*"
    r"(?:container=\S+\s*)?"
    r"input=(.+)$"
)


SYSCALL_CATEGORIES = {
    "metadata": {
        "stat", "fstat", "lstat", "statx", "fstatat64", "newfstatat",
        "access", "faccessat",
        "getdents64", "getdents",
        "readlink", "readlinkat",
    },
    "data": {
        "read", "write", "pread64", "pwrite64",
        "readv", "writev", "preadv", "pwritev", "preadv2", "pwritev2",
    },
    "control": {
        "open", "openat", "close",
        "lseek", "fcntl", "ioctl",
        "chdir", "fchdir", "getcwd",
        "mmap", "munmap",
        "dup", "dup2", "dup3",
    },
    "modify": {
        "mkdir", "mkdirat", "rmdir",
        "unlink", "unlinkat",
        "rename", "renameat", "renameat2",
        "chmod", "fchmod", "chown", "fchown",
        "truncate", "ftruncate", "fsync", "fdatasync", "sync_file_range",
    },
    "process": {
        "clone", "clone3", "fork", "vfork",
        "execve",
        "wait4", "waitpid", "waitid",
    },
    "blocking": {
        "select", "pselect6", "poll", "ppoll",
        "epoll_wait", "epoll_pwait",
        "futex",
        "nanosleep", "clock_nanosleep",
    },
    "network": {
        "recvfrom", "sendto",
        "accept", "accept4", "connect",
        "socket", "bind", "listen",
        "recv", "send",
        "recvmsg", "sendmsg", "recvmmsg", "sendmmsg",
    },
}

SYSCALL_CATEGORY_TO_RESOURCE = {
    "metadata": "file_io",
    "data": "file_io",
    "modify": "file_io",
    "blocking": "wait",
    "process": "process_mgmt",
}

RESOURCE_KEYS = ("file_io", "wait", "process_mgmt", "network", "interface_probe", "other")

# Phase-1 storage-time definition (§5 #5): metadata + data + durability +
# open/close. Do not count generic process control (mmap/ioctl/fcntl) as
# storage time. Namespace mutations live under "modify" and are metadata work
# on Lustre, so they remain in file_io.
STORAGE_CONTROL_SYSCALLS = {"open", "openat", "close"}
NON_STORAGE_CONTROL_SYSCALLS = {
    "lseek", "fcntl", "ioctl", "chdir", "fchdir", "getcwd",
    "mmap", "munmap", "dup", "dup2", "dup3",
}

CODE_EXEC_TOOL_NAMES = {
    "Bash",
    "CodeExec",
    "ScriptExec",
    "SubprocessExec",
    "PythonExec",
    "ShellExec",
}

PURE_IO_TOOL_NAMES = {"Read", "Write", "Edit", "Glob", "Grep"}


def classify_syscall(syscall: str) -> str:
    for category, syscalls in SYSCALL_CATEGORIES.items():
        if syscall in syscalls:
            return category
    return "other"


def resource_bucket_for_syscall(syscall: str) -> str:
    category = classify_syscall(syscall)
    if category == "network":
        return "network"
    if category == "control":
        return "file_io" if syscall in STORAGE_CONTROL_SYSCALLS else "other"
    return SYSCALL_CATEGORY_TO_RESOURCE.get(category, "other")


def is_code_exec_tool(tool_name: str) -> bool:
    return tool_name in CODE_EXEC_TOOL_NAMES


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


@dataclass
class ToolCall:
    start_time: datetime
    end_time: datetime
    tool_name: str
    tool_id: str
    input_params: dict

    def contains(self, ts: datetime) -> bool:
        return self.start_time <= ts <= self.end_time

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "input_params": self.input_params,
        }


@dataclass
class FsEntry:
    pid: int
    timestamp: datetime
    syscall: str
    args: str
    return_value: int | None
    duration: float
    path: str | None
    bytes_transferred: int = 0
    file_descriptor: int | None = None
    matched_tool_call: str | None = None
    errno: str | None = None
    errno_desc: str | None = None
    open_flags: str | None = None
    resource_bucket: str | None = None
    requested_size: int | None = None
    actual_size: int | None = None
    offset: int | None = None
    flags: int | None = None
    dirfd: int | None = None

    def to_dict(self) -> dict:
        out = {
            "pid": self.pid,
            "timestamp": self.timestamp.isoformat(),
            "syscall": self.syscall,
            "args": self.args,
            "return_value": self.return_value,
            "duration": self.duration,
            "path": self.path,
            "bytes_transferred": self.bytes_transferred,
            "matched_tool_call": self.matched_tool_call,
            "resource_bucket": self.resource_bucket,
        }
        if self.errno is not None:
            out["errno"] = self.errno
        if self.errno_desc is not None:
            out["errno_desc"] = self.errno_desc
        if self.file_descriptor is not None:
            out["file_descriptor"] = self.file_descriptor
        if self.open_flags is not None:
            out["open_flags"] = self.open_flags
        if self.requested_size is not None:
            out["requested_size"] = self.requested_size
        if self.actual_size is not None:
            out["actual_size"] = self.actual_size
        if self.offset is not None:
            out["offset"] = self.offset
        if self.flags is not None:
            out["flags"] = self.flags
        if self.dirfd is not None:
            out["dirfd"] = self.dirfd
        return out


@dataclass
class ToolSummary:
    tool_id: str
    tool_name: str
    wall_clock_ms: float
    total_syscall_ms: float
    syscall_count: int
    by_syscall: dict[str, dict]
    time_gap_ms: float
    resource_breakdown: dict[str, float] = field(default_factory=dict)
    cpu_compute_ms: float = 0.0
    unaccounted_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "wall_clock_ms": round(self.wall_clock_ms, 3),
            "total_syscall_ms": round(self.total_syscall_ms, 3),
            "syscall_count": self.syscall_count,
            "by_syscall": self.by_syscall,
            "time_gap_ms": round(self.time_gap_ms, 3),
            "resource_breakdown": {
                key: (round(float(value), 3) if isinstance(value, (int, float)) else value)
                for key, value in self.resource_breakdown.items()
            },
            "cpu_compute_ms": round(self.cpu_compute_ms, 3),
            "unaccounted_ms": round(self.unaccounted_ms, 3),
        }


@dataclass
class ParsedTrace:
    tool_calls: list[ToolCall] = field(default_factory=list)
    fs_entries: list[FsEntry] = field(default_factory=list)
    tool_summaries: dict[str, ToolSummary] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "fs_entries": [e.to_dict() for e in self.fs_entries],
            "summary": self.summary,
        }
        if self.tool_summaries:
            out["tool_summaries"] = {
                tid: summary.to_dict() for tid, summary in self.tool_summaries.items()
            }
        return out


@dataclass
class FDInfo:
    path: str
    open_timestamp: datetime
    tool_id: str | None


class FDTable:
    """Track fd -> path mappings per PID, with CWD tracking for path resolution."""

    def __init__(self) -> None:
        self._tables: dict[int, dict[int, FDInfo]] = {}
        self._cwd: dict[int, str] = {}

    def resolve_path(self, pid: int, path: str, dirfd: int) -> str:
        """Resolve a possibly-relative path using dirfd or CWD."""
        if not path or path.startswith("/"):
            return path
        if dirfd != AT_FDCWD:
            info = self.lookup(pid, dirfd)
            if info and info.path:
                return posixpath.normpath(info.path.rstrip("/") + "/" + path)
            # dirfd not in our table — leave path relative rather than
            # incorrectly resolving against CWD.
            return path
        cwd = self._cwd.get(pid)
        if cwd:
            return posixpath.normpath(cwd.rstrip("/") + "/" + path)
        return path

    def handle_chdir(self, pid: int, path: str) -> None:
        if path.startswith("/"):
            self._cwd[pid] = posixpath.normpath(path)
        elif pid in self._cwd:
            self._cwd[pid] = posixpath.normpath(
                self._cwd[pid].rstrip("/") + "/" + path
            )

    def handle_open(
        self,
        pid: int,
        fd: int,
        path: str,
        timestamp: datetime,
        tool_id: str | None,
    ) -> None:
        if fd < 0:
            return
        self._tables.setdefault(pid, {})[fd] = FDInfo(
            path=path,
            open_timestamp=timestamp,
            tool_id=tool_id,
        )

    def handle_close(self, pid: int, fd: int) -> None:
        if pid in self._tables:
            self._tables[pid].pop(fd, None)

    def copy_table_for_child(self, parent_pid: int, child_pid: int) -> None:
        if parent_pid in self._tables:
            self._tables[child_pid] = dict(self._tables[parent_pid])
        if parent_pid in self._cwd:
            self._cwd[child_pid] = self._cwd[parent_pid]

    def lookup(self, pid: int, fd: int) -> FDInfo | None:
        return self._tables.get(pid, {}).get(fd)


class ProcessTree:
    """Track parent-child PID relationships and root tool ownership."""

    def __init__(self) -> None:
        self._parents: dict[int, int] = {}
        self._root_tool: dict[int, str] = {}

    def handle_fork(self, parent_pid: int, child_pid: int, tool_id: str | None) -> None:
        if child_pid <= 0:
            return
        self._parents[child_pid] = parent_pid
        if parent_pid in self._root_tool:
            self._root_tool[child_pid] = self._root_tool[parent_pid]
        elif tool_id:
            self._root_tool[child_pid] = tool_id

    def get_root_tool(self, pid: int) -> str | None:
        return self._root_tool.get(pid)


def parse_time(time_str: str) -> datetime:
    parts = time_str.split(".")
    hms = parts[0]
    us = parts[1] if len(parts) > 1 else "0"
    us = us[:6].ljust(6, "0")
    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    h, m, s = map(int, hms.split(":"))
    return base.replace(hour=h, minute=m, second=s, microsecond=int(us))


def parse_tool_calls(tool_log: Path) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    with tool_log.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            match = TOOL_CALL_PATTERN.match(line)
            if not match:
                print(f"Warning: could not parse tool log line {idx}", file=sys.stderr)
                continue
            start_s, end_s, tool_name, tool_id, input_s = match.groups()
            try:
                input_params = ast.literal_eval(input_s)
            except (ValueError, SyntaxError):
                input_params = {"raw": input_s}
            tool_calls.append(
                ToolCall(
                    start_time=parse_time(start_s),
                    end_time=parse_time(end_s),
                    tool_name=tool_name,
                    tool_id=tool_id,
                    input_params=input_params,
                )
            )
    # Task wrapper is a parent tool call; remove it for attribution/plots.
    tool_calls = [tc for tc in tool_calls if tc.tool_name != "Task"]

    # Widen each window by 200 ms on the leading edge so we capture the
    # syscalls (e.g. openat, stat) that the runtime issues *just before* the
    # tool handler proper begins.
    #lead = timedelta(milliseconds=200)
    #for tc in tool_calls:
    #    tc.start_time -= lead

    return tool_calls


def parse_events(events_log: Path) -> tuple[list[dict], dict | None]:
    """Parse JSONL events, returning (events, meta_dict).

    The meta dict (first line with ``type == "meta"``) is returned separately
    so callers can use ``wall_start_ns`` for timezone-offset calibration.
    """
    events: list[dict] = []
    meta: dict | None = None
    with events_log.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: invalid JSON at {events_log}:{idx}", file=sys.stderr)
                continue
            if obj.get("type") == "meta":
                if meta is None:
                    meta = obj
                continue
            if "ts_ns" not in obj:
                continue
            events.append(obj)
    events.sort(key=lambda e: int(e["ts_ns"]))
    return events, meta


def _compute_tz_offset(
    wall_start_ns: int,
    tool_calls: list[ToolCall],
) -> float:
    """Compute timezone offset (seconds) between ebpf timestamps and tool-call timestamps.

    ``wall_start_ns`` is a Unix-epoch nanosecond timestamp recorded on the
    *tracing* machine.  ``datetime.fromtimestamp`` converts it to the *parser*
    machine's local timezone.  ``parse_time`` produces datetimes in the
    *tracing* machine's local timezone (because tool_calls.log stores local
    HH:MM:SS).

    If the parser runs in a different timezone the two will disagree by the
    timezone difference.  We detect and correct that here.

    Returns the offset **in seconds** to subtract from ``datetime.fromtimestamp``
    results so they align with the tool-call timebase.
    """
    if not tool_calls:
        return 0.0

    local_wall_start = datetime.fromtimestamp(wall_start_ns / 1_000_000_000)
    first_tool_start = tool_calls[0].start_time

    # wall_start should be slightly *before* the first tool call (typically
    # 1–30 s).  A large absolute difference indicates a timezone mismatch.
    raw_diff = (local_wall_start - first_tool_start).total_seconds()

    # Round to nearest 30-minute boundary (covers all real-world UTC offsets
    # including half-hour zones like UTC+5:30, UTC+9:30, etc.).
    tz_offset_seconds = round(raw_diff / 1800) * 1800

    if tz_offset_seconds != 0:
        print(
            f"Detected timezone offset: {tz_offset_seconds / 3600:+.1f}h "
            f"(ebpf wall_start mapped to {local_wall_start.strftime('%H:%M:%S')}, "
            f"first tool call at {first_tool_start.strftime('%H:%M:%S')}); correcting.",
            file=sys.stderr,
        )

    return float(tz_offset_seconds)


# Module-level offset applied by ns_to_datetime; set once in process_trace_dir.
_tz_offset_seconds: float = 0.0


def ns_to_datetime(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000) - timedelta(seconds=_tz_offset_seconds)


def in_any_tool_window(ts: datetime, tool_calls: list[ToolCall]) -> bool:
    return any(tc.contains(ts) for tc in tool_calls)


def get_active_tool_calls(ts: datetime, tool_calls: list[ToolCall]) -> list[ToolCall]:
    return [tc for tc in tool_calls if tc.contains(ts)]


class ActiveToolIndex:
    """Sweep-line index of currently-active tool calls.

    Replaces the O(events × tool_calls) per-event scan (get_active_tool_calls /
    _single_active_tool_id) with an amortized O(events + tool_calls) sweep.

    REQUIREMENT: ``advance_to(ts)`` must be called with non-decreasing ``ts``
    (events are sorted by ts_ns in parse_events, so both build_state_tables and
    the main loop iterate them in order). ``active()`` then returns exactly the
    tools tc where ``tc.start_time <= ts <= tc.end_time`` — identical semantics
    to ``get_active_tool_calls``, just without rescanning every tool each call.
    """

    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self._by_start = sorted(tool_calls, key=lambda tc: tc.start_time)
        self._by_end = sorted(tool_calls, key=lambda tc: tc.end_time)
        self._si = 0
        self._ei = 0
        self._active: dict[str, ToolCall] = {}

    def advance_to(self, ts: datetime) -> None:
        by_start = self._by_start
        n = len(by_start)
        while self._si < n and by_start[self._si].start_time <= ts:
            tc = by_start[self._si]
            self._active[tc.tool_id] = tc
            self._si += 1
        by_end = self._by_end
        m = len(by_end)
        # end < ts → no longer active (inclusive end: still active at ts == end).
        while self._ei < m and by_end[self._ei].end_time < ts:
            tc = by_end[self._ei]
            self._active.pop(tc.tool_id, None)
            self._ei += 1

    def active(self) -> list[ToolCall]:
        return list(self._active.values())

    def single_active_id(self) -> str | None:
        if len(self._active) == 1:
            return next(iter(self._active.values())).tool_id
        return None


def get_tool_window(tool_calls: list[ToolCall]) -> tuple[datetime | None, datetime | None]:
    if not tool_calls:
        return None, None
    return (min(tc.start_time for tc in tool_calls), max(tc.end_time for tc in tool_calls))


FD_BASED_SYSCALLS = {
    "read",
    "write",
    "pread64",
    "pwrite64",
    "fstat",
    "getdents64",
    "ftruncate",
    "close",
    "fchdir",
}

PATH_BASED_SYSCALLS = {
    "openat",
    "access",
    "faccessat",
    "newfstatat",
    "unlinkat",
    "mkdirat",
    "renameat2",
    "truncate",
    "chdir",
}

AT_FDCWD = (-100) & 0xFFFFFFFFFFFFFFFF

DIRFD_BASED_SYSCALLS = {
    "openat",
    "newfstatat",
    "faccessat",
    "unlinkat",
    "mkdirat",
    "renameat2",
}

O_ACCMODE = 0x3
O_FLAG_BITS = {
    0o00000100: "O_CREAT",
    0o00000200: "O_EXCL",
    0o00000400: "O_NOCTTY",
    0o00001000: "O_TRUNC",
    0o00002000: "O_APPEND",
    0o00004000: "O_NONBLOCK",
    0o00010000: "O_DSYNC",
    0o00040000: "O_DIRECT",
    0o00100000: "O_LARGEFILE",
    0o00200000: "O_DIRECTORY",
    0o00400000: "O_NOFOLLOW",
    0o01000000: "O_NOATIME",
    0o02000000: "O_CLOEXEC",
    0o10000000: "O_PATH",
}


def decode_open_flags(flags: int) -> str:
    access = flags & O_ACCMODE
    parts = [{0: "O_RDONLY", 1: "O_WRONLY", 2: "O_RDWR"}.get(access, f"0x{access:x}")]
    remaining = flags & ~O_ACCMODE
    for bit, name in O_FLAG_BITS.items():
        if remaining & bit:
            parts.append(name)
            remaining &= ~bit
    if remaining:
        parts.append(f"0x{remaining:x}")
    return "|".join(parts)


ENOENT_NOISE_PATTERNS = [
    "/usr/lib/",
    "/usr/local/lib/",
    "/lib/",
    "/etc/ld.so",
    "/etc/ssl/",
    "/etc/localtime",
    ".so.",
    ".pyc",
    "__pycache__",
    "/proc/",
    "/sys/",
    "pyvenv.cfg",
    "._pth",
    "pybuilddir.txt",
    "/root/.claude/",
    "/root/.config/",
]


def _path_match(entry_path: str, tool_path: str) -> bool:
    normalized_entry_path = entry_path.rstrip("/")
    normalized_tool_path = tool_path.lstrip("./").rstrip("/")
    return (
        normalized_entry_path == normalized_tool_path
        or normalized_entry_path.endswith("/" + normalized_tool_path)
    )


def _single_active_tool_id(ts: datetime, tool_calls: list[ToolCall]) -> str | None:
    active = get_active_tool_calls(ts, tool_calls)
    if len(active) == 1:
        return active[0].tool_id
    return None


def build_state_tables(events: list[dict], tool_calls: list[ToolCall]) -> tuple[FDTable, ProcessTree]:
    """
    Pass over all events to:
    - build process tree from fork events
    - build fd->path state from open/close
    - enrich fd-based syscall events with resolved path and opening tool id
    """
    fd_table = FDTable()
    proc_tree = ProcessTree()
    index = ActiveToolIndex(tool_calls)

    for event in events:
        etype = event.get("type")
        ts = ns_to_datetime(int(event["ts_ns"]))
        index.advance_to(ts)
        tool_id = index.single_active_id()

        if etype == "fork":
            parent = int(event.get("pid", -1))
            child = int(event.get("child_pid", -1))
            if parent > 0 and child > 0:
                proc_tree.handle_fork(parent, child, tool_id)
                fd_table.copy_table_for_child(parent, child)
            continue

        if etype != "syscall":
            continue

        syscall = str(event.get("syscall", "unknown"))
        pid = int(event.get("pid", 0))
        fd = int(event.get("arg0", -1)) if syscall in FD_BASED_SYSCALLS else None

        # Enrich fd-based events before any table mutation (especially close).
        if fd is not None and fd >= 0:
            info = fd_table.lookup(pid, fd)
            if info:
                if not event.get("path"):
                    event["_resolved_path"] = info.path
                event["_fd_tool_id"] = info.tool_id

        # Resolve relative paths for *at() syscalls using the dirfd (arg0).
        if syscall in DIRFD_BASED_SYSCALLS:
            raw_path = event.get("path")
            if isinstance(raw_path, str) and raw_path and not raw_path.startswith("/"):
                dirfd = int(event.get("arg0", AT_FDCWD))
                resolved = fd_table.resolve_path(pid, raw_path, dirfd)
                if resolved != raw_path:
                    event["path"] = resolved

        if syscall == "openat":
            ret = int(event.get("ret", -1))
            path = event.get("path")
            if ret >= 0 and isinstance(path, str) and path:
                fd_table.handle_open(pid, ret, path, ts, tool_id)
        elif syscall == "close" and fd is not None and fd >= 0:
            fd_table.handle_close(pid, fd)
        elif syscall == "chdir":
            ret = int(event.get("ret", 0))
            chdir_path = event.get("path")
            if ret == 0 and isinstance(chdir_path, str) and chdir_path:
                fd_table.handle_chdir(pid, chdir_path)
        elif syscall == "fchdir":
            ret = int(event.get("ret", 0))
            if ret == 0 and fd is not None and fd >= 0:
                info = fd_table.lookup(pid, fd)
                if info and info.path:
                    fd_table.handle_chdir(pid, info.path)

    return fd_table, proc_tree


def match_event_to_tool(
    entry: FsEntry,
    active_tools: list[ToolCall],
    fd_tool_id: str | None,
    proc_tree: ProcessTree,
) -> str | None:
    # Signal 1: file descriptor ownership from openat time.
    if fd_tool_id:
        return fd_tool_id

    # Signal 2: timestamp window (active_tools precomputed by the sweep index).
    if not active_tools:
        # Signal 4: process ancestry.
        root_tool = proc_tree.get_root_tool(entry.pid)
        return root_tool

    if len(active_tools) == 1:
        return active_tools[0].tool_id

    # Signal 3: path matching for file tools.
    if entry.path:
        for tc in active_tools:
            if tc.tool_name in {"Read", "Write", "Edit"}:
                tool_path = tc.input_params.get("file_path", "")
                if isinstance(tool_path, str) and tool_path and _path_match(entry.path, tool_path):
                    return tc.tool_id

    # Signal 4 (constrained to active tools).
    root_tool = proc_tree.get_root_tool(entry.pid)
    if root_tool and any(tc.tool_id == root_tool for tc in active_tools):
        return root_tool

    # Signal 5: exec/fork-related calls likely belong to Bash.
    if entry.syscall in {"execve", "clone"}:
        bash_tools = [tc for tc in active_tools if tc.tool_name == "Bash"]
        if len(bash_tools) == 1:
            return bash_tools[0].tool_id

    # Signal 6: innermost-active wins.  LangGraph's hierarchical agents
    # produce nested tool windows: an `Invoke_<x>_agent` wrapper fully
    # contains its leaf tool calls (Esearch/Elink/Efetch/...).  Without
    # this rule the leaf systematically loses every overlapping syscall
    # to "uncategorized" because both parent and leaf are non-Read/Write
    # tools.  Picking the latest-started tool gives the leaf priority,
    # which matches the user's intuition ("attribute to whichever tool
    # is currently 'doing' the work").  Ties are rare; pick any.
    return max(active_tools, key=lambda tc: tc.start_time).tool_id


def is_enoent_noise(entry: FsEntry) -> bool:
    if entry.return_value != -2:
        return False
    if entry.syscall not in PATH_BASED_SYSCALLS:
        return False
    if not entry.path:
        return False
    return any(pattern in entry.path for pattern in ENOENT_NOISE_PATTERNS)


def make_fs_entry(event: dict) -> FsEntry:
    syscall = str(event.get("syscall") or event.get("function") or "unknown")
    ret = event.get("ret")
    if isinstance(ret, bool):
        ret = int(ret)
    elif ret is not None:
        ret = int(ret)

    arg0 = int(event.get("arg0", 0))
    arg1 = int(event.get("arg1", 0))
    arg2 = int(event.get("arg2", 0))
    arg3 = int(event.get("arg3", 0))
    arg4 = int(event.get("arg4", 0))
    latency_ns = int(event.get("latency_ns", 0))

    bytes_xfer = 0
    # Network syscalls also return bytes-transferred on success; include them
    # so HTTP-heavy agents have non-zero throughput in tool_summaries.
    _BYTE_RETURNING_SYSCALLS = {
        "read", "write", "pread64", "pwrite64",
        "readv", "writev", "preadv", "pwritev", "preadv2", "pwritev2",
        "fread", "fwrite",
        "sendto", "recvfrom", "sendmsg", "recvmsg",
        "sendmmsg", "recvmmsg",
    }
    if syscall in _BYTE_RETURNING_SYSCALLS and isinstance(ret, int) and ret > 0:
        bytes_xfer = ret

    fd = None
    if syscall in {
        "read", "write", "readv", "writev",
        "pread64", "pwrite64", "preadv", "pwritev", "preadv2", "pwritev2",
        "close", "fstat", "fsync", "fdatasync",
        "ftruncate", "sync_file_range",
    }:
        fd = arg0

    path = event.get("path")
    if not path:
        path = event.get("_resolved_path")
    if not isinstance(path, str) or not path:
        path = None

    errno = None
    errno_desc = None
    if ret == -2:
        errno = "ENOENT"
        errno_desc = "No such file or directory"

    open_flags = decode_open_flags(arg2) if syscall == "openat" else None

    requested_size = None
    offset = None
    flags = None
    dirfd = None

    if syscall in {"read", "write"}:
        requested_size = arg2
    elif syscall in {"fread", "fwrite"}:
        requested_size = arg1 * arg2 if arg1 > 0 and arg2 > 0 else None
    elif syscall in {"pread64", "pwrite64"}:
        requested_size = arg2
        offset = arg3 if "arg3" in event else None
    elif syscall in {"readv", "writev", "preadv", "pwritev", "preadv2", "pwritev2"}:
        # arg2 is iovcnt, not bytes. The total requested byte count lives in
        # user iovec memory and is intentionally not read by the BPF program.
        requested_size = None
        if syscall in {"preadv", "pwritev", "preadv2", "pwritev2"}:
            offset = arg3 if "arg3" in event else None
            if syscall in {"preadv2", "pwritev2"}:
                flags = arg4 if "arg4" in event else None
    elif syscall == "openat":
        dirfd = arg0
        flags = arg2
    elif syscall == "sync_file_range":
        offset = arg1
        requested_size = arg2
        flags = arg3 if "arg3" in event else None
    elif syscall in DIRFD_BASED_SYSCALLS:
        dirfd = arg0

    actual_size = bytes_xfer if bytes_xfer > 0 else None

    return FsEntry(
        pid=int(event.get("pid", 0)),
        timestamp=ns_to_datetime(int(event["ts_ns"])),
        syscall=syscall,
        args=f"arg0={arg0},arg1={arg1},arg2={arg2},arg3={arg3},arg4={arg4}",
        return_value=ret,
        duration=latency_ns / 1_000_000_000,
        path=path,
        bytes_transferred=bytes_xfer,
        file_descriptor=fd,
        errno=errno,
        errno_desc=errno_desc,
        open_flags=open_flags,
        resource_bucket="interface_probe" if event.get("type") == "libc_io" else resource_bucket_for_syscall(syscall),
        requested_size=requested_size,
        actual_size=actual_size,
        offset=offset,
        flags=flags,
        dirfd=dirfd,
    )


def make_lifecycle_entry(event: dict) -> FsEntry:
    """Create an FsEntry from a fork, exec, or exit event."""
    etype = event["type"]
    pid = int(event.get("pid", 0))
    comm = event.get("comm", "")

    if etype == "fork":
        child_pid = int(event.get("child_pid", 0))
        args = f"child_pid={child_pid}"
    else:
        args = ""

    return FsEntry(
        pid=pid,
        timestamp=ns_to_datetime(int(event["ts_ns"])),
        syscall=etype,
        args=args,
        return_value=None,
        duration=0.0,
        path=comm if comm else None,
    )


def compute_tool_summaries(entries: list[FsEntry], tool_calls: list[ToolCall]) -> dict[str, ToolSummary]:
    summaries: dict[str, ToolSummary] = {}
    # Group entries by their matched tool once (O(entries)) instead of
    # rescanning the full entries list per tool call (O(tool_calls × entries)).
    entries_by_tool: dict[str, list[FsEntry]] = {}
    for e in entries:
        if e.matched_tool_call:
            entries_by_tool.setdefault(e.matched_tool_call, []).append(e)
    for tc in tool_calls:
        wall_ms = (tc.end_time - tc.start_time).total_seconds() * 1000
        selected = entries_by_tool.get(tc.tool_id, [])
        by_syscall: dict[str, dict] = {}
        durations_by_syscall: dict[str, list[float]] = {}
        resource_breakdown = {key: 0.0 for key in RESOURCE_KEYS}
        total_ms = 0.0
        for e in selected:
            by_syscall.setdefault(e.syscall, {"count": 0, "total_ms": 0.0, "total_bytes": 0})
            durations_by_syscall.setdefault(e.syscall, [])
            by_syscall[e.syscall]["count"] += 1
            by_syscall[e.syscall]["total_ms"] += e.duration * 1000
            by_syscall[e.syscall]["total_bytes"] += e.bytes_transferred
            durations_by_syscall[e.syscall].append(e.duration * 1000)
            duration_ms = e.duration * 1000
            bucket = e.resource_bucket or resource_bucket_for_syscall(e.syscall)
            if bucket not in resource_breakdown:
                bucket = "other"
            resource_breakdown[bucket] += duration_ms
            if bucket != "interface_probe":
                total_ms += duration_ms
        for syscall, agg in by_syscall.items():
            agg["total_ms"] = round(agg["total_ms"], 3)
            ds = sorted(durations_by_syscall.get(syscall, []))
            if ds:
                agg["p50_ms"] = round(_percentile(ds, 50), 6)
                agg["p95_ms"] = round(_percentile(ds, 95), 6)
                agg["p99_ms"] = round(_percentile(ds, 99), 6)
            measured_ms = (
                resource_breakdown["file_io"]
                + resource_breakdown["wait"]
                + resource_breakdown["process_mgmt"]
                + resource_breakdown["other"]
                + resource_breakdown["network"]
            )
        residual_ms = max(0.0, wall_ms - measured_ms)
        cpu_compute_ms = residual_ms if is_code_exec_tool(tc.tool_name) else 0.0
        unaccounted_ms = 0.0 if is_code_exec_tool(tc.tool_name) else residual_ms
        out_breakdown = {
            "file_io_ms": resource_breakdown["file_io"],
            "wait_ms": resource_breakdown["wait"],
                "process_mgmt_ms": resource_breakdown["process_mgmt"],
                "network_ms": resource_breakdown["network"],
                "interface_probe_ms": resource_breakdown["interface_probe"],
                "other_syscall_ms": resource_breakdown["other"],
            "cpu_compute_ms": cpu_compute_ms,
            "unaccounted_ms": unaccounted_ms,
            "cpu_source": "residual_estimate" if is_code_exec_tool(tc.tool_name) else "not_applicable",
        }
        summaries[tc.tool_id] = ToolSummary(
            tool_id=tc.tool_id,
            tool_name=tc.tool_name,
            wall_clock_ms=wall_ms,
            total_syscall_ms=total_ms,
            syscall_count=len(selected),
            by_syscall=by_syscall,
            time_gap_ms=wall_ms - total_ms,
            resource_breakdown=out_breakdown,
            cpu_compute_ms=cpu_compute_ms,
            unaccounted_ms=unaccounted_ms,
        )
    return summaries


def compute_resource_summary(entries: list[FsEntry]) -> dict[str, dict[str, float]]:
    """Aggregate syscall latency by resource bucket and broad role."""
    roles = {
        "tool": {key: 0.0 for key in RESOURCE_KEYS},
        "orchestration": {key: 0.0 for key in RESOURCE_KEYS},
    }
    for entry in entries:
        role = (
            "tool"
            if entry.matched_tool_call and entry.matched_tool_call != "uncategorized"
            else "orchestration"
        )
        bucket = entry.resource_bucket or resource_bucket_for_syscall(entry.syscall)
        if bucket not in RESOURCE_KEYS:
            bucket = "other"
        roles[role][bucket] += entry.duration * 1000.0
    return {
        role: {f"{key}_ms": round(value, 3) for key, value in buckets.items()}
        for role, buckets in roles.items()
    }


def process_trace_dir(
    trace_dir: Path,
    workspace_filter: str | None,
) -> ParsedTrace:
    tool_log = trace_dir / "tool_calls.log"
    events_log = trace_dir / "ebpf_events.log"
    if not tool_log.exists():
        raise FileNotFoundError(f"tool_calls.log not found in {trace_dir}")
    if not events_log.exists():
        raise FileNotFoundError(f"ebpf_events.log not found in {trace_dir}")

    tool_calls = parse_tool_calls(tool_log)
    events, meta = parse_events(events_log)

    # Calibrate timezone offset so ebpf epoch timestamps align with the
    # local-time-of-day strings in tool_calls.log.
    global _tz_offset_seconds
    if meta and "wall_start_ns" in meta and tool_calls:
        _tz_offset_seconds = _compute_tz_offset(int(meta["wall_start_ns"]), tool_calls)
    else:
        _tz_offset_seconds = 0.0

    _fd_table, proc_tree = build_state_tables(events, tool_calls)

    window_start, window_end = get_tool_window(tool_calls)

    fs_entries: list[FsEntry] = []
    total_events = 0
    lifecycle_types = {"fork", "exec", "exit"}
    index = ActiveToolIndex(tool_calls)
    for event in events:
        etype = event.get("type")
        if etype in {"syscall", "libc_io"}:
            total_events += 1
            entry = make_fs_entry(event)
        elif etype in lifecycle_types:
            total_events += 1
            entry = make_lifecycle_entry(event)
        else:
            continue

        if window_start and entry.timestamp < window_start:
            continue
        if window_end and entry.timestamp > window_end:
            continue

        # Keep pathless events; only filter when a path is known.
        if workspace_filter and entry.path and workspace_filter not in entry.path:
            continue

        if is_enoent_noise(entry):
            continue

        # Sweep index gives the active tool set in O(1) amortized instead of an
        # O(tool_calls) rescan per event.
        index.advance_to(entry.timestamp)
        active_tools = index.active()

        entry.matched_tool_call = match_event_to_tool(
            entry,
            active_tools,
            fd_tool_id=event.get("_fd_tool_id"),
            proc_tree=proc_tree,
        )

        if entry.matched_tool_call is None and active_tools:
            entry.matched_tool_call = "uncategorized"
        fs_entries.append(entry)

    matched = sum(1 for e in fs_entries if e.matched_tool_call and e.matched_tool_call != "uncategorized")
    uncategorized = sum(1 for e in fs_entries if e.matched_tool_call == "uncategorized")
    resource_summary = compute_resource_summary(fs_entries)

    result = ParsedTrace()
    result.tool_calls = tool_calls
    result.fs_entries = fs_entries
    result.tool_summaries = compute_tool_summaries(fs_entries, tool_calls)
    result.summary = {
        "total_entries": total_events,
        "filtered_entries": len(fs_entries),
        "matched_to_tools": matched,
        "uncategorized": uncategorized,
        "pids": sorted({e.pid for e in fs_entries}),
        "attribution": {
            "process_tree_nodes": len(proc_tree._parents),
            "method": "multi_signal_fd_path_process_tree",
        },
        "resource_summary": resource_summary,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse eBPF JSONL events and correlate to Claude tool calls."
    )
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory containing ebpf_events.log and tool_calls.log",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path (default: <trace_dir>/parsed.json)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON",
    )
    parser.add_argument(
        "--workspace-path",
        type=str,
        default=None,
        help="Optional workspace path filter (default: no filtering)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trace_dir = args.trace_dir
    workspace_filter = args.workspace_path

    result = process_trace_dir(trace_dir, workspace_filter)

    output_path = args.output if args.output else (trace_dir / "parsed.json")
    payload = result.to_dict()
    output_path.write_text(
        json.dumps(payload, indent=None if args.compact else 2),
        encoding="utf-8",
    )
    print(f"Output written to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
