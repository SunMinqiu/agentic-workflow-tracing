#!/usr/bin/env python3
from __future__ import annotations
"""
Parse and correlate strace.log entries with Claude tool calls.

This script filters noise from unrelated syscalls and matches filesystem
operations to their originating tool calls for analysis.

Equivalent to parse_traces.py but for Linux strace output instead of macOS fs_usage.
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ToolCall:
    """Represents a single tool call from tool_calls.log."""
    start_time: datetime
    end_time: datetime
    tool_name: str       # "Bash", "Read", "Write", etc.
    tool_id: str
    input_params: dict   # {'command': '...', ...}
    
    def contains_timestamp(self, ts: datetime) -> bool:
        """Check if a timestamp falls within this tool call's window."""
        return self.start_time <= ts <= self.end_time
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "input_params": self.input_params,
        }


@dataclass
class StraceEntry:
    """Represents a single strace log entry."""
    pid: int
    timestamp: datetime
    syscall: str         # "openat", "read", "stat", etc.
    args: str            # raw argument string
    return_value: int | None
    errno: str | None    # "ENOENT", "EACCES", etc.
    errno_desc: str | None  # "No such file or directory", etc.
    duration: float      # in seconds
    path: str | None     # extracted file path if applicable
    bytes_transferred: int = 0  # for read/write operations
    file_descriptor: int | None = None  # for fd-based operations
    matched_tool_call: str | None = None  # tool_id or "uncategorized"
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {
            "pid": self.pid,
            "timestamp": self.timestamp.isoformat(),
            "syscall": self.syscall,
            "args": self.args,
            "return_value": self.return_value,
            "duration": self.duration,
            "path": self.path,
            "bytes_transferred": self.bytes_transferred,
            "matched_tool_call": self.matched_tool_call,
        }
        if self.errno:
            result["errno"] = self.errno
        if self.errno_desc:
            result["errno_desc"] = self.errno_desc
        if self.file_descriptor is not None:
            result["file_descriptor"] = self.file_descriptor
        return result


@dataclass
class ToolSummary:
    """Aggregated statistics for a single tool call."""
    tool_id: str
    tool_name: str
    wall_clock_ms: float
    total_syscall_ms: float
    syscall_count: int
    by_syscall: dict[str, dict]  # syscall -> {count, total_ms, total_bytes}
    time_gap_ms: float  # wall_clock - total_syscall (unexplained time)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "wall_clock_ms": round(self.wall_clock_ms, 3),
            "total_syscall_ms": round(self.total_syscall_ms, 3),
            "syscall_count": self.syscall_count,
            "by_syscall": self.by_syscall,
            "time_gap_ms": round(self.time_gap_ms, 3),
        }


@dataclass
class ParsedTrace:
    """Container for parsed and filtered trace data."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    strace_entries: list[StraceEntry] = field(default_factory=list)
    pids: set[int] = field(default_factory=set)
    tool_summaries: dict[str, ToolSummary] = field(default_factory=dict)
    
    # Statistics
    total_entries: int = 0
    filtered_entries: int = 0
    matched_to_tools: int = 0
    uncategorized: int = 0
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "fs_entries": [e.to_dict() for e in self.strace_entries],
            "summary": {
                "total_entries": self.total_entries,
                "filtered_entries": self.filtered_entries,
                "matched_to_tools": self.matched_to_tools,
                "uncategorized": self.uncategorized,
                "pids": sorted(self.pids),
            }
        }
        if self.tool_summaries:
            result["tool_summaries"] = {
                tid: s.to_dict() for tid, s in self.tool_summaries.items()
            }
        return result


# =============================================================================
# File Descriptor Tracking
# =============================================================================

@dataclass
class FDInfo:
    """Information about an open file descriptor."""
    path: str
    open_timestamp: datetime
    tool_id: str | None  # tool_id that was active when fd was opened


class FDTable:
    """
    Track file descriptor to path mappings per process.
    
    This allows us to resolve read/write/fstat/close operations back to
    their file paths, even when the open() happened in a different tool call.
    """
    
    def __init__(self):
        # pid -> {fd -> FDInfo}
        self._tables: dict[int, dict[int, FDInfo]] = {}
    
    def handle_open(self, pid: int, fd: int, path: str, timestamp: datetime, tool_id: str | None) -> None:
        """Record an fd -> path mapping from openat/open syscall."""
        if fd < 0:
            return  # Failed open, no fd assigned
        if pid not in self._tables:
            self._tables[pid] = {}
        self._tables[pid][fd] = FDInfo(path=path, open_timestamp=timestamp, tool_id=tool_id)
    
    def handle_close(self, pid: int, fd: int) -> None:
        """Remove fd mapping on close."""
        if pid in self._tables:
            self._tables[pid].pop(fd, None)
    
    def handle_dup(self, pid: int, old_fd: int, new_fd: int) -> None:
        """Copy fd mapping for dup/dup2/dup3."""
        if pid in self._tables and old_fd in self._tables[pid]:
            self._tables[pid][new_fd] = self._tables[pid][old_fd]
    
    def lookup(self, pid: int, fd: int) -> FDInfo | None:
        """Look up the path and tool_id for a given fd."""
        if pid in self._tables:
            return self._tables[pid].get(fd)
        return None
    
    def get_path(self, pid: int, fd: int) -> str | None:
        """Get just the path for a given fd."""
        info = self.lookup(pid, fd)
        return info.path if info else None
    
    def get_tool_id(self, pid: int, fd: int) -> str | None:
        """Get the tool_id that opened this fd."""
        info = self.lookup(pid, fd)
        return info.tool_id if info else None
    
    def copy_table_for_child(self, parent_pid: int, child_pid: int) -> None:
        """Copy fd table from parent to child process (for clone/fork)."""
        if parent_pid in self._tables:
            # Deep copy the fd mappings
            self._tables[child_pid] = dict(self._tables[parent_pid])


# =============================================================================
# Process Tree Tracking
# =============================================================================

class ProcessTree:
    """
    Track parent-child process relationships.
    
    This allows us to attribute syscalls from child processes back to
    the tool call that spawned them (typically Bash tools).
    """
    
    def __init__(self):
        # child_pid -> parent_pid
        self._parents: dict[int, int] = {}
        # pid -> tool_id that is the "root" of this process tree
        self._root_tool: dict[int, str] = {}
    
    def handle_clone(self, parent_pid: int, child_pid: int, timestamp: datetime, tool_id: str | None) -> None:
        """Record parent-child relationship from clone/fork."""
        if child_pid <= 0:
            return  # Failed clone
        self._parents[child_pid] = parent_pid
        
        # Inherit root tool from parent, or set if this is the root
        # NOTE: We don't propagate root_tool to the parent, only to children.
        # This prevents infrastructure processes (spawned before tool calls)
        # from being attributed to tools just because they spawned children later.
        if parent_pid in self._root_tool:
            self._root_tool[child_pid] = self._root_tool[parent_pid]
        elif tool_id:
            self._root_tool[child_pid] = tool_id
    
    def get_parent(self, pid: int) -> int | None:
        """Get the parent PID of a process."""
        return self._parents.get(pid)
    
    def get_root_tool(self, pid: int) -> str | None:
        """Get the tool_id that is the root of this process's tree."""
        return self._root_tool.get(pid)
    
    def get_ancestors(self, pid: int) -> list[int]:
        """Get list of ancestor PIDs (parent, grandparent, etc.)."""
        ancestors = []
        current = pid
        while current in self._parents:
            parent = self._parents[current]
            ancestors.append(parent)
            current = parent
        return ancestors
    
    def set_root_tool(self, pid: int, tool_id: str) -> None:
        """Explicitly set the root tool for a PID."""
        self._root_tool[pid] = tool_id


# =============================================================================
# Parsers
# =============================================================================

# Pattern for tool_calls.log entries:
# [12:51:33.388651 -> 12:51:33.466118] (77.5ms) Bash (id=toolu_01PFV...) input={...}
TOOL_CALL_PATTERN = re.compile(
    r'\[(\d{2}:\d{2}:\d{2}\.\d+)\s*->\s*(\d{2}:\d{2}:\d{2}\.\d+)\]\s*'
    r'\([\d.]+ms\)\s*'
    r'(\w+)\s*'  # tool name
    r'\(id=([^)]+)\)\s*'  # tool id
    r'(?:container=\S+\s*)?'  # optional PTC container id
    r'input=(.+)$'  # input params (Python dict literal)
)


def parse_time(time_str: str) -> datetime:
    """Parse a time string like '12:51:33.388651' into a datetime object.
    
    Uses today's date as the base since logs only contain time.
    """
    parts = time_str.split('.')
    time_part = parts[0]
    microseconds = parts[1] if len(parts) > 1 else '0'
    # Pad or truncate to 6 digits
    microseconds = microseconds[:6].ljust(6, '0')
    
    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    h, m, s = map(int, time_part.split(':'))
    return base.replace(hour=h, minute=m, second=s, microsecond=int(microseconds))


def parse_tool_calls_log(filepath: Path) -> list[ToolCall]:
    """Parse tool_calls.log file into ToolCall objects."""
    tool_calls = []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            match = TOOL_CALL_PATTERN.match(line)
            if not match:
                print(f"Warning: Could not parse line {line_num}: {line[:80]}...", 
                      file=sys.stderr)
                continue
            
            start_str, end_str, tool_name, tool_id, input_str = match.groups()
            
            # Parse the input dict (it's a Python literal)
            try:
                input_params = ast.literal_eval(input_str)
            except (ValueError, SyntaxError):
                input_params = {"raw": input_str}
            
            tool_calls.append(ToolCall(
                start_time=parse_time(start_str),
                end_time=parse_time(end_str),
                tool_name=tool_name,
                tool_id=tool_id,
                input_params=input_params,
            ))
    
    return tool_calls


# =============================================================================
# Strace Line Parsing
# =============================================================================

# Pattern for complete strace line:
# PID     TIMESTAMP syscall(args) = result <duration>
# Examples:
#   9     23:19:29.419475 openat(AT_FDCWD, "/path", O_RDONLY) = 3 <0.000358>
#   9     23:19:29.422384 openat(...) = -1 ENOENT (No such file or directory) <0.000039>
STRACE_COMPLETE_PATTERN = re.compile(
    r'^(\d+)\s+'                           # PID
    r'(\d{2}:\d{2}:\d{2}\.\d+)\s+'         # timestamp
    r'(\w+)\((.*)?\)\s*'                   # syscall(args)
    r'=\s*(-?\d+|0x[0-9a-fA-F]+|\?)'       # return value
    r'(?:\s+([A-Z]+)\s*(?:\([^)]+\))?)?'   # optional errno
    r'\s*<([\d.]+)>$'                      # duration
)

# Pattern for unfinished syscall:
# PID     TIMESTAMP syscall(args <unfinished ...>
STRACE_UNFINISHED_PATTERN = re.compile(
    r'^(\d+)\s+'                           # PID
    r'(\d{2}:\d{2}:\d{2}\.\d+)\s+'         # timestamp
    r'(\w+)\((.*)'                         # syscall(args...
    r'\s*<unfinished\s*\.\.\.>$'           # <unfinished ...>
)

# Pattern for resumed syscall:
# PID     TIMESTAMP <... syscall resumed>rest) = result <duration>
STRACE_RESUMED_PATTERN = re.compile(
    r'^(\d+)\s+'                           # PID
    r'(\d{2}:\d{2}:\d{2}\.\d+)\s+'         # timestamp
    r'<\.\.\.\s*(\w+)\s+resumed>'          # <... syscall resumed>
    r'(.*?)\)\s*'                          # rest of args)
    r'=\s*(-?\d+|0x[0-9a-fA-F]+|\?)'       # return value
    r'(?:\s+([A-Z]+)\s*(?:\([^)]+\))?)?'   # optional errno
    r'\s*<([\d.]+)>$'                      # duration
)

# Pattern to extract errno and description
ERRNO_PATTERN = re.compile(r'([A-Z]+)\s*\(([^)]+)\)')

# Pattern for duration at end
DURATION_PATTERN = re.compile(r'<([\d.]+)>$')

# Pattern to extract path from common syscalls
# Handles: "/path", '/path', or just /path in some contexts
PATH_PATTERN = re.compile(r'"([^"]+)"')

# Pattern for bytes in read/write: read(3, "...", 832) = 832
BYTES_PATTERN = re.compile(r',\s*(\d+)\)\s*=')

# Syscalls that have paths as arguments
# Value indicates which quoted string is the path (0 = first quoted string, etc.)
PATH_SYSCALLS = {
    'openat': 0,      # openat(AT_FDCWD, "/path", ...) - AT_FDCWD not quoted, path is first quoted
    'open': 0,        # open("/path", ...)
    'stat': 0,        # stat("/path", ...)
    'lstat': 0,       # lstat("/path", ...)
    'statx': 0,       # statx(AT_FDCWD, "/path", ...) - path is first quoted string
    'access': 0,      # access("/path", ...)
    'faccessat': 0,   # faccessat(AT_FDCWD, "/path", ...) - path is first quoted string
    'readlink': 0,    # readlink("/path", ...)
    'readlinkat': 0,  # readlinkat(AT_FDCWD, "/path", ...) - path is first quoted string
    'unlink': 0,      # unlink("/path")
    'unlinkat': 0,    # unlinkat(AT_FDCWD, "/path", ...) - path is first quoted string
    'mkdir': 0,       # mkdir("/path", ...)
    'mkdirat': 0,     # mkdirat(AT_FDCWD, "/path", ...) - path is first quoted string
    'rmdir': 0,       # rmdir("/path")
    'rename': 0,      # rename("/old", "/new") - first path
    'renameat': 0,    # renameat(AT_FDCWD, "/old", ...) - first quoted is old path
    'chmod': 0,       # chmod("/path", ...)
    'chown': 0,       # chown("/path", ...)
    'truncate': 0,    # truncate("/path", ...)
    'chdir': 0,       # chdir("/path")
    'getcwd': None,   # getcwd returns path in buffer, extract from result
    'execve': 0,      # execve("/path", ...)
    'fstatat64': 0,   # fstatat64(AT_FDCWD, "/path", ...) - path is first quoted string
    'newfstatat': 0,  # newfstatat(AT_FDCWD, "/path", ...) - path is first quoted string
}

# Syscalls that transfer data (have byte counts)
DATA_TRANSFER_SYSCALLS = {'read', 'write', 'pread64', 'pwrite64', 'readv', 'writev', 'preadv', 'pwritev', 'sendfile'}

# Syscalls we care about for filesystem tracing
FS_SYSCALLS = {
    'openat', 'open', 'close',
    'read', 'write', 'pread64', 'pwrite64',
    'readv', 'writev', 'preadv', 'pwritev',
    'stat', 'fstat', 'lstat', 'statx', 'fstatat64', 'newfstatat',
    'lseek',
    'getdents64', 'getdents',
    'access', 'faccessat',
    'readlink', 'readlinkat',
    'unlink', 'unlinkat',
    'mkdir', 'mkdirat', 'rmdir',
    'rename', 'renameat', 'renameat2',
    'chmod', 'fchmod', 'chown', 'fchown',
    'truncate', 'ftruncate',
    'chdir', 'fchdir', 'getcwd',
    'mmap', 'munmap',
    'fcntl', 'dup', 'dup2', 'dup3',
    'execve',
    # Process creation (for process tree tracking)
    'clone', 'clone3', 'fork', 'vfork',
}

# Additional syscalls for performance analysis (blocking, waiting, process)
PERF_SYSCALLS = FS_SYSCALLS | {
    # Blocking I/O
    'select', 'pselect6', 'poll', 'ppoll', 'epoll_wait', 'epoll_pwait',
    # Sleeping
    'nanosleep', 'clock_nanosleep',
    # Lock contention
    'futex',
    # Process waiting
    'wait4', 'waitpid', 'waitid',
    # Network (often blocking)
    'recvfrom', 'sendto', 'accept', 'connect',
}

# Syscalls exempt from path filtering (process-related, no file path)
PATHLESS_SYSCALLS = {
    'clone', 'clone3', 'fork', 'vfork', 'execve',
    'wait4', 'waitpid', 'waitid', 'futex',
    'nanosleep', 'clock_nanosleep',
    'select', 'pselect6', 'poll', 'ppoll', 'epoll_wait', 'epoll_pwait',
    'recvfrom', 'sendto', 'accept', 'connect',
}

# Optional path filter - only keep entries with paths containing this string
WORKSPACE_PATH: str | None = "/workspace"


def identify_main_pid(entries: list["StraceEntry"]) -> int | None:
    """
    Identify the main process PID (the Python interpreter running the agent).
    
    Heuristic: Find the PID that executed python3 via execve, or fall back to
    the PID with the lowest number (usually the main process started by strace).
    """
    # Look for the process that ran python3
    for entry in entries:
        if entry.syscall == 'execve' and entry.path and 'python' in entry.path:
            return entry.pid
    
    # Fallback: the lowest PID (usually the main process started by strace)
    if entries:
        return min(e.pid for e in entries)
    return None


def extract_path_from_args(syscall: str, args: str) -> str | None:
    """Extract file path from syscall arguments."""
    if syscall not in PATH_SYSCALLS:
        return None
    
    path_index = PATH_SYSCALLS[syscall]
    if path_index is None:
        return None
    
    # Find all quoted strings in args
    paths = PATH_PATTERN.findall(args)
    
    if paths and path_index < len(paths):
        return paths[path_index]
    
    return None


def extract_bytes_from_args(syscall: str, args: str, return_value: int | None) -> int:
    """Extract bytes transferred from read/write syscalls."""
    if syscall not in DATA_TRANSFER_SYSCALLS:
        return 0
    
    # For successful read/write, return value is bytes transferred
    if return_value is not None and return_value > 0:
        return return_value
    
    return 0


def extract_fd_from_args(args: str) -> int | None:
    """Extract file descriptor from first argument."""
    # First arg is usually FD for read, write, fstat, etc.
    match = re.match(r'(\d+)', args.strip())
    if match:
        return int(match.group(1))
    return None


def parse_return_value(val_str: str) -> int | None:
    """Parse return value string to int."""
    if val_str == '?':
        return None
    if val_str.startswith('0x'):
        return int(val_str, 16)
    return int(val_str)


@dataclass
class PendingCall:
    """Stores state for an unfinished syscall."""
    pid: int
    timestamp: datetime
    syscall: str
    args: str


def parse_strace_log(filepath: Path) -> list[StraceEntry]:
    """Parse strace.log file into StraceEntry objects."""
    entries = []
    # Track unfinished syscalls by (pid, syscall) - note: strace can have nested unfinished calls
    pending: dict[tuple[int, str], PendingCall] = {}
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip()
            if not line:
                continue
            
            entry = None
            
            # Try complete pattern first (most common)
            match = STRACE_COMPLETE_PATTERN.match(line)
            if match:
                pid = int(match.group(1))
                timestamp = parse_time(match.group(2))
                syscall = match.group(3)
                args = match.group(4) or ""
                return_val = parse_return_value(match.group(5))
                errno = match.group(6)
                duration = float(match.group(7))
                
                # Extract errno description from original line if errno present
                errno_desc = None
                if errno:
                    errno_match = ERRNO_PATTERN.search(line)
                    if errno_match:
                        errno_desc = errno_match.group(2)
                
                entry = StraceEntry(
                    pid=pid,
                    timestamp=timestamp,
                    syscall=syscall,
                    args=args,
                    return_value=return_val,
                    errno=errno,
                    errno_desc=errno_desc,
                    duration=duration,
                    path=extract_path_from_args(syscall, args),
                    bytes_transferred=extract_bytes_from_args(syscall, args, return_val),
                    file_descriptor=extract_fd_from_args(args) if syscall in ('read', 'write', 'fstat', 'close', 'lseek') else None,
                )
            
            # Try unfinished pattern
            if not entry:
                match = STRACE_UNFINISHED_PATTERN.match(line)
                if match:
                    pid = int(match.group(1))
                    timestamp = parse_time(match.group(2))
                    syscall = match.group(3)
                    args = match.group(4) or ""
                    
                    # Store for later when we see the resumed line
                    pending[(pid, syscall)] = PendingCall(
                        pid=pid,
                        timestamp=timestamp,
                        syscall=syscall,
                        args=args,
                    )
                    continue
            
            # Try resumed pattern
            if not entry:
                match = STRACE_RESUMED_PATTERN.match(line)
                if match:
                    pid = int(match.group(1))
                    resume_timestamp = parse_time(match.group(2))
                    syscall = match.group(3)
                    rest_args = match.group(4) or ""
                    return_val = parse_return_value(match.group(5))
                    errno = match.group(6)
                    duration = float(match.group(7))
                    
                    # Find the pending call
                    key = (pid, syscall)
                    if key in pending:
                        pend = pending.pop(key)
                        # Combine args
                        full_args = pend.args + rest_args
                        
                        # Extract errno description
                        errno_desc = None
                        if errno:
                            errno_match = ERRNO_PATTERN.search(line)
                            if errno_match:
                                errno_desc = errno_match.group(2)
                        
                        entry = StraceEntry(
                            pid=pid,
                            timestamp=pend.timestamp,  # Use original start time
                            syscall=syscall,
                            args=full_args,
                            return_value=return_val,
                            errno=errno,
                            errno_desc=errno_desc,
                            duration=duration,
                            path=extract_path_from_args(syscall, full_args),
                            bytes_transferred=extract_bytes_from_args(syscall, full_args, return_val),
                            file_descriptor=extract_fd_from_args(full_args) if syscall in ('read', 'write', 'fstat', 'close', 'lseek') else None,
                        )
                    else:
                        # No matching pending call, skip
                        continue
            
            if entry:
                entries.append(entry)
    
    return entries


# =============================================================================
# Filtering Logic
# =============================================================================

def extract_task_window(
    tool_calls: list[ToolCall]
) -> tuple[list[ToolCall], datetime | None, datetime | None]:
    """
    Extract the Task wrapper from tool calls and return remaining tools plus task window.
    
    Returns (filtered_tool_calls, task_start_time, task_end_time).
    """
    task_call = None
    remaining = []
    
    for tc in tool_calls:
        if tc.tool_name == "Task":
            task_call = tc
        else:
            remaining.append(tc)
    
    if task_call:
        return remaining, task_call.start_time, task_call.end_time
    return remaining, None, None


def get_tool_window(tool_calls: list[ToolCall]) -> tuple[datetime | None, datetime | None]:
    """Get the overall time window from first to last tool call."""
    if not tool_calls:
        return None, None
    
    start = min(tc.start_time for tc in tool_calls)
    end = max(tc.end_time for tc in tool_calls)
    return start, end


def is_in_tool_window(ts: datetime, tool_calls: list[ToolCall]) -> bool:
    """Check if a timestamp falls within any tool call's time window."""
    return any(tc.contains_timestamp(ts) for tc in tool_calls)


def get_active_tool_calls(ts: datetime, tool_calls: list[ToolCall]) -> list[ToolCall]:
    """Get all tool calls that are active at a given timestamp."""
    return [tc for tc in tool_calls if tc.contains_timestamp(ts)]


def filter_strace_entries(
    entries: list[StraceEntry],
    tool_calls: list[ToolCall],
    task_start: datetime | None = None,
    task_end: datetime | None = None,
    workspace_filter: str | None = WORKSPACE_PATH,
    perf_mode: bool = False,
    exclude_main_pid: int | None = None,
    proc_tree: "ProcessTree | None" = None,
) -> list[StraceEntry]:
    """
    Filter strace entries according to filtering rules:
    
    1. Only keep filesystem-related syscalls (or perf syscalls in perf_mode)
    2. Filter to task/tool window
    3. Optionally filter by workspace path (pathless syscalls exempt in perf_mode)
    4. Skip ENOENT noise from library/config probing
    5. Optionally exclude main process (--children-only mode)
    """
    if not entries:
        return []
    
    # Determine time window
    if task_start is not None:
        window_start = task_start
        window_end = task_end
    else:
        window_start, window_end = get_tool_window(tool_calls)
        if window_start is None:
            print("Warning: No tool calls found, cannot determine time window", file=sys.stderr)
            return []
    
    # Choose syscall set based on mode
    syscall_set = PERF_SYSCALLS if perf_mode else FS_SYSCALLS
    
    filtered = []
    
    for entry in entries:
        # Skip syscalls not in our set
        if entry.syscall not in syscall_set:
            continue
        
        # Skip entries outside the time window
        if entry.timestamp < window_start:
            continue
        if window_end is not None and entry.timestamp > window_end:
            continue
        
        # Exclude main process if requested (--children-only mode)
        # Only keep syscalls from PIDs that were spawned by a tool call
        if exclude_main_pid is not None and proc_tree is not None:
            root_tool = proc_tree.get_root_tool(entry.pid)
            if root_tool is None:
                # This PID wasn't spawned by a tool call - skip it
                continue
        
        # Workspace path filter (but exempt pathless syscalls in perf mode)
        if workspace_filter is not None:
            if perf_mode and entry.syscall in PATHLESS_SYSCALLS:
                pass  # Keep pathless syscalls in perf mode
            elif not entry.path or workspace_filter not in entry.path:
                continue
        
        # Skip ENOENT noise from library/config probing
        if entry.errno == "ENOENT":
            # Skip common noise patterns
            if entry.path:
                noise_patterns = [
                    '/usr/lib/', '/usr/local/lib/', '/lib/',
                    '/etc/ld.so', '/etc/ssl/', '/etc/localtime',
                    '.so.', '.pyc', '__pycache__',
                    '/proc/', '/sys/',
                    'pyvenv.cfg', '._pth', 'pybuilddir.txt',
                    '/root/.claude/', '/root/.config/',
                ]
                if any(pat in entry.path for pat in noise_patterns):
                    continue
        
        filtered.append(entry)
    
    return filtered


# =============================================================================
# Tool Matching
# =============================================================================

def match_entry_to_tool(
    entry: StraceEntry,
    tool_calls: list[ToolCall],
    fd_table: FDTable | None = None,
    proc_tree: ProcessTree | None = None,
) -> str | None:
    """
    Match a strace entry to a tool call using multiple signals.
    
    Matching priority:
    1. If fd was opened by a specific tool, attribute to that tool
    2. If timestamp falls within exactly one tool window, use that
    3. If multiple tools overlap, try path matching for Read/Write/Edit
    4. If process tree shows ancestry to a Bash tool, use that
    5. Fall back to "uncategorized" if in any tool window
    
    Returns tool_id if matched, "uncategorized" if in window but ambiguous,
    or None if not in any window.
    """
    # Signal 1: Check if the fd was opened by a specific tool
    if fd_table and entry.file_descriptor is not None:
        fd_tool_id = fd_table.get_tool_id(entry.pid, entry.file_descriptor)
        if fd_tool_id:
            # Verify this tool is still plausibly active (within reasonable time)
            for tc in tool_calls:
                if tc.tool_id == fd_tool_id:
                    # FD was opened by this tool - strong signal
                    return fd_tool_id
    
    # Signal 2: Timestamp-based matching
    active_tools = get_active_tool_calls(entry.timestamp, tool_calls)
    
    if not active_tools:
        # No tool active at this timestamp - check process tree
        if proc_tree:
            root_tool = proc_tree.get_root_tool(entry.pid)
            if root_tool:
                return root_tool
        return None
    
    if len(active_tools) == 1:
        return active_tools[0].tool_id
    
    # Multiple active tools - try to disambiguate
    
    # Signal 3: Path matching for Read/Write/Edit tools
    if entry.path:
        for tc in active_tools:
            if tc.tool_name in ("Read", "Write", "Edit"):
                tool_path = tc.input_params.get("file_path", "")
                if tool_path:
                    # Normalize paths for comparison
                    normalized_entry_path = entry.path.rstrip('/')
                    normalized_tool_path = tool_path.lstrip('./').rstrip('/')
                    # Check for exact match or if entry path ends with tool path
                    if normalized_entry_path == normalized_tool_path or \
                       normalized_entry_path.endswith('/' + normalized_tool_path):
                        return tc.tool_id
    
    # Signal 4: Check process tree for Bash tool ancestry
    if proc_tree:
        root_tool = proc_tree.get_root_tool(entry.pid)
        if root_tool:
            # Verify root_tool is one of the active tools
            for tc in active_tools:
                if tc.tool_id == root_tool:
                    return root_tool
    
    # Signal 5: If it's a Bash-like operation (execve, etc.), prefer Bash tools
    if entry.syscall in ('execve', 'clone', 'clone3', 'fork', 'vfork'):
        bash_tools = [tc for tc in active_tools if tc.tool_name == "Bash"]
        if len(bash_tools) == 1:
            return bash_tools[0].tool_id
    
    return "uncategorized"


def match_all_entries(
    entries: list[StraceEntry],
    tool_calls: list[ToolCall],
    fd_table: FDTable | None = None,
    proc_tree: ProcessTree | None = None,
) -> list[StraceEntry]:
    """Match all strace entries to tool calls where possible."""
    for entry in entries:
        entry.matched_tool_call = match_entry_to_tool(entry, tool_calls, fd_table, proc_tree)
    return entries


def compute_tool_summaries(
    entries: list[StraceEntry],
    tool_calls: list[ToolCall],
) -> dict[str, ToolSummary]:
    """Compute per-tool aggregation of syscall statistics."""
    summaries = {}
    
    for tc in tool_calls:
        wall_ms = (tc.end_time - tc.start_time).total_seconds() * 1000
        
        # Filter entries for this tool
        tool_entries = [e for e in entries if e.matched_tool_call == tc.tool_id]
        
        # Aggregate by syscall type
        by_syscall: dict[str, dict] = {}
        total_ms = 0.0
        for e in tool_entries:
            if e.syscall not in by_syscall:
                by_syscall[e.syscall] = {"count": 0, "total_ms": 0.0, "total_bytes": 0}
            by_syscall[e.syscall]["count"] += 1
            by_syscall[e.syscall]["total_ms"] += e.duration * 1000
            by_syscall[e.syscall]["total_bytes"] += e.bytes_transferred
            total_ms += e.duration * 1000
        
        # Round the totals in by_syscall
        for syscall_stats in by_syscall.values():
            syscall_stats["total_ms"] = round(syscall_stats["total_ms"], 3)
        
        summaries[tc.tool_id] = ToolSummary(
            tool_id=tc.tool_id,
            tool_name=tc.tool_name,
            wall_clock_ms=wall_ms,
            total_syscall_ms=total_ms,
            syscall_count=len(tool_entries),
            by_syscall=by_syscall,
            time_gap_ms=wall_ms - total_ms,
        )
    
    return summaries


# =============================================================================
# Main Processing
# =============================================================================

def process_trace_directory(
    trace_dir: Path,
    workspace_filter: str | None = WORKSPACE_PATH,
    perf_mode: bool = False,
    children_only: bool = False,
) -> ParsedTrace:
    """Process a trace directory containing strace.log and tool_calls.log."""
    strace_log = trace_dir / "strace.log"
    tool_log = trace_dir / "tool_calls.log"
    
    if not strace_log.exists():
        raise FileNotFoundError(f"strace.log not found in {trace_dir}")
    if not tool_log.exists():
        raise FileNotFoundError(f"tool_calls.log not found in {trace_dir}")
    
    return process_logs(strace_log, tool_log, workspace_filter, perf_mode, children_only)


def build_state_tables(
    entries: list[StraceEntry],
    tool_calls: list[ToolCall],
) -> tuple[FDTable, ProcessTree]:
    """
    Pass 1: Build FD table and process tree from strace entries.
    
    This processes ALL entries (not just filtered ones) to build complete
    state tables that can resolve fd -> path mappings and process ancestry.
    
    IMPORTANT: Also enriches entries with fd -> path mappings inline,
    because we need to resolve paths BEFORE the fd is closed (which removes
    it from the table).
    """
    fd_table = FDTable()
    proc_tree = ProcessTree()
    
    # Syscalls that operate on file descriptors and should be enriched
    fd_enrichable_syscalls = {
        'read', 'write', 'pread64', 'pwrite64', 
        'readv', 'writev', 'preadv', 'pwritev',
        'fstat', 'lseek', 'ftruncate', 'fsync',
        'getdents64', 'getdents', 'fchmod', 'fchown',
    }
    
    for entry in entries:
        # Determine which tool (if any) is active at this timestamp
        active_tools = get_active_tool_calls(entry.timestamp, tool_calls)
        tool_id = active_tools[0].tool_id if len(active_tools) == 1 else None
        
        # Handle open syscalls - record fd -> path mapping
        if entry.syscall in ('openat', 'open'):
            if entry.return_value is not None and entry.return_value >= 0 and entry.path:
                fd_table.handle_open(
                    pid=entry.pid,
                    fd=entry.return_value,
                    path=entry.path,
                    timestamp=entry.timestamp,
                    tool_id=tool_id,
                )
        
        # ENRICH: Before processing close, check if this is an fd-based syscall
        # that needs path enrichment. We do this here (inline) because after close
        # the fd mapping will be removed from the table.
        elif entry.syscall in fd_enrichable_syscalls:
            if entry.file_descriptor is not None and not entry.path:
                path = fd_table.get_path(entry.pid, entry.file_descriptor)
                if path:
                    entry.path = path
        
        # Handle close syscalls - remove fd mapping (AFTER potential enrichment above)
        elif entry.syscall == 'close':
            if entry.file_descriptor is not None:
                fd_table.handle_close(entry.pid, entry.file_descriptor)
        
        # Handle dup syscalls - copy fd mapping
        elif entry.syscall in ('dup', 'dup2', 'dup3'):
            if entry.file_descriptor is not None and entry.return_value is not None and entry.return_value >= 0:
                fd_table.handle_dup(entry.pid, entry.file_descriptor, entry.return_value)
        
        # Handle clone/fork - record parent-child relationship
        elif entry.syscall in ('clone', 'clone3', 'fork', 'vfork'):
            if entry.return_value is not None and entry.return_value > 0:
                # return_value is the child PID
                proc_tree.handle_clone(
                    parent_pid=entry.pid,
                    child_pid=entry.return_value,
                    timestamp=entry.timestamp,
                    tool_id=tool_id,
                )
                # Child inherits parent's fd table
                fd_table.copy_table_for_child(entry.pid, entry.return_value)
    
    return fd_table, proc_tree


def process_logs(
    strace_log: Path,
    tool_log: Path,
    workspace_filter: str | None = WORKSPACE_PATH,
    perf_mode: bool = False,
    children_only: bool = False,
) -> ParsedTrace:
    """Process strace.log and tool_calls.log files."""
    result = ParsedTrace()
    
    if perf_mode:
        print("Performance analysis mode enabled", file=sys.stderr)
    if children_only:
        print("Children-only mode enabled (excluding main process)", file=sys.stderr)
    
    # Parse tool calls
    print(f"Parsing tool calls from {tool_log}...", file=sys.stderr)
    all_tool_calls = parse_tool_calls_log(tool_log)
    print(f"  Found {len(all_tool_calls)} tool calls", file=sys.stderr)
    
    # Extract Task wrapper (remove it from tool_calls, but don't use its window for filtering)
    result.tool_calls, _, _ = extract_task_window(all_tool_calls)
    print(f"  Actual tool calls (excluding Task): {len(result.tool_calls)}", file=sys.stderr)
    
    # Parse strace entries
    print(f"Parsing strace from {strace_log}...", file=sys.stderr)
    all_entries = parse_strace_log(strace_log)
    result.total_entries = len(all_entries)
    result.pids = {e.pid for e in all_entries}
    print(f"  Found {result.total_entries} total entries from {len(result.pids)} PIDs", file=sys.stderr)
    
    # Pass 1: Build FD table and process tree from ALL entries
    # Note: This also enriches entries with fd -> path mappings inline
    print("Building FD table and process tree (with inline path enrichment)...", file=sys.stderr)
    fd_table, proc_tree = build_state_tables(all_entries, result.tool_calls)
    print(f"  FD tables for {len(fd_table._tables)} PIDs", file=sys.stderr)
    print(f"  Process tree: {len(proc_tree._parents)} parent-child relationships", file=sys.stderr)
    
    # Count enriched entries for statistics
    enriched_count = sum(1 for e in all_entries if e.path and e.syscall in 
                         {'read', 'write', 'pread64', 'pwrite64', 'readv', 'writev',
                          'fstat', 'lseek', 'getdents64', 'getdents'})
    print(f"  Enriched {enriched_count} fd-based entries with paths", file=sys.stderr)
    
    # Identify main process PID if children-only mode is enabled
    main_pid = None
    if children_only:
        main_pid = identify_main_pid(all_entries)
        print(f"  Main process PID: {main_pid} (will be excluded)", file=sys.stderr)
    
    # Filter entries (use tool window, not Task window)
    print(f"Filtering entries (workspace_filter={workspace_filter}, perf_mode={perf_mode}, children_only={children_only})...", file=sys.stderr)
    filtered = filter_strace_entries(
        all_entries, result.tool_calls, None, None, workspace_filter, perf_mode,
        exclude_main_pid=main_pid,
        proc_tree=proc_tree if children_only else None
    )
    result.strace_entries = filtered
    result.filtered_entries = len(filtered)
    print(f"  Retained {result.filtered_entries} entries after filtering", file=sys.stderr)
    
    # Match entries to tool calls (now with FD table and process tree context)
    print("Matching entries to tool calls...", file=sys.stderr)
    match_all_entries(result.strace_entries, result.tool_calls, fd_table, proc_tree)
    
    # Calculate statistics
    result.matched_to_tools = sum(
        1 for e in result.strace_entries 
        if e.matched_tool_call and e.matched_tool_call != "uncategorized"
    )
    result.uncategorized = sum(
        1 for e in result.strace_entries 
        if e.matched_tool_call == "uncategorized"
    )
    print(f"  Matched: {result.matched_to_tools}, Uncategorized: {result.uncategorized}", 
          file=sys.stderr)
    
    # Compute per-tool summaries in perf mode
    if perf_mode:
        print("Computing per-tool summaries...", file=sys.stderr)
        result.tool_summaries = compute_tool_summaries(result.strace_entries, result.tool_calls)
        print(f"  Generated summaries for {len(result.tool_summaries)} tools", file=sys.stderr)
    
    return result


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parse and correlate strace logs with Claude tool calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s traces/20260115_181911/academic-writing-main/
  %(prog)s traces/20260115_181911/academic-writing-main/ --compact
  %(prog)s traces/20260115_181911/academic-writing-main/ --no-workspace-filter
  %(prog)s traces/20260115_181911/academic-writing-main/ --perf
  %(prog)s traces/20260115_181911/academic-writing-main/ --children-only
        """
    )
    
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory containing strace.log and tool_calls.log"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (default: <trace_dir>/parsed.json)"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact (non-indented) JSON"
    )
    parser.add_argument(
        "--no-workspace-filter",
        action="store_true",
        help="Don't filter to /workspace paths only"
    )
    parser.add_argument(
        "--workspace-path",
        type=str,
        default="/workspace",
        help="Path filter for workspace (default: /workspace)"
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Performance analysis mode: expanded syscalls, relaxed filtering, per-tool aggregation"
    )
    parser.add_argument(
        "--children-only",
        action="store_true",
        help="Exclude syscalls from the main process, keeping only child processes spawned by tool calls"
    )
    
    args = parser.parse_args()
    
    # Determine workspace filter
    workspace_filter = None if args.no_workspace_filter else args.workspace_path
    
    # Process the trace directory
    result = process_trace_directory(args.trace_dir, workspace_filter, args.perf, args.children_only)
    
    # Determine output path
    output_path = args.output if args.output else args.trace_dir / "parsed.json"
    
    # Output
    output_dict = result.to_dict()
    indent = None if args.compact else 2
    json_str = json.dumps(output_dict, indent=indent)
    
    output_path.write_text(json_str)
    print(f"Output written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

