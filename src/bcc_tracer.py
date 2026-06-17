#!/usr/bin/env python3
"""
Capture process lifecycle and filesystem syscall activity for a target PID tree
using BCC/eBPF, and write JSONL events.

Output file: ebpf_events.log (JSON lines)
"""

import argparse
import ctypes
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict

from bcc import BPF


BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/types.h>
#include <uapi/linux/unistd.h>

enum event_type {
    EVENT_FORK = 1,
    EVENT_EXEC = 2,
    EVENT_EXIT = 3,
    EVENT_SYSCALL = 4,
};

struct syscall_state_t {
    u64 ts_ns;
    u64 syscall_id;
    u64 arg0;
    u64 arg1;
    u64 arg2;
    char filepath[256];
};

struct event_t {
    u64 mono_ts_ns;
    u32 event_type;
    u32 pid;
    u32 tid;
    u32 aux_pid;
    s64 ret;
    u64 syscall_id;
    u64 latency_ns;
    u64 arg0;
    u64 arg1;
    u64 arg2;
    char filepath[256];
    char comm[TASK_COMM_LEN];
};

BPF_HASH(tracked_pids, u32, u8, 65536);
BPF_HASH(inflight, u64, struct syscall_state_t);
BPF_PERF_OUTPUT(events);

static __always_inline int is_tracked(u32 pid) {
    u8 *found = tracked_pids.lookup(&pid);
    return found != 0;
}

static __always_inline int is_traced_syscall(u64 id) {
    switch (id) {
        case __NR_openat:
        case __NR_close:
        case __NR_read:
        case __NR_write:
        case __NR_pread64:
        case __NR_pwrite64:
        case __NR_readv:
        case __NR_writev:
        case __NR_preadv:
        case __NR_pwritev:
        case __NR_preadv2:
        case __NR_pwritev2:
        case __NR_newfstatat:
        case __NR_fstat:
        case __NR_access:
        case __NR_faccessat:
        case __NR_getdents64:
        case __NR_unlinkat:
        case __NR_mkdirat:
        case __NR_renameat2:
        case __NR_truncate:
        case __NR_ftruncate:
        case __NR_fsync:
        case __NR_fdatasync:
        case __NR_sync_file_range:
        case __NR_chdir:
        case __NR_fchdir:
        case __NR_getcwd:
        case __NR_execve:
        case __NR_clone:
        case __NR_poll:
        case __NR_select:
        case __NR_pselect6:
        case __NR_ppoll:
        case __NR_epoll_wait:
        case __NR_epoll_pwait:
        case __NR_futex:
        case __NR_nanosleep:
        case __NR_clock_nanosleep:
        case __NR_wait4:
        case __NR_waitid:
__NET_SYSCALL_CASES__
            return 1;
        default:
            return 0;
    }
}

TRACEPOINT_PROBE(sched, sched_process_fork) {
    /*
     * args->parent_pid is the kernel pid (= userspace TID) of the forking
     * thread, NOT the tgid (= userspace PID).  If a non-main thread forks,
     * args->parent_pid won't match any entry in tracked_pids (which is
     * keyed by tgid).  Use bpf_get_current_pid_tgid() to get the real tgid.
     */
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 parent_tgid = pid_tgid >> 32;
    u32 child = args->child_pid;
    if (!is_tracked(parent_tgid)) {
        return 0;
    }

    u8 one = 1;
    tracked_pids.update(&child, &one);

    struct event_t evt = {};
    evt.mono_ts_ns = bpf_ktime_get_ns();
    evt.event_type = EVENT_FORK;
    evt.pid = parent_tgid;
    evt.tid = (u32)pid_tgid;
    evt.aux_pid = child;
    bpf_get_current_comm(&evt.comm, sizeof(evt.comm));
    events.perf_submit(args, &evt, sizeof(evt));
    return 0;
}

TRACEPOINT_PROBE(sched, sched_process_exec) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    u32 tid = (u32)pid_tgid;
    if (!is_tracked(pid)) {
        return 0;
    }

    struct event_t evt = {};
    evt.mono_ts_ns = bpf_ktime_get_ns();
    evt.event_type = EVENT_EXEC;
    evt.pid = pid;
    evt.tid = tid;
    bpf_get_current_comm(&evt.comm, sizeof(evt.comm));
    events.perf_submit(args, &evt, sizeof(evt));
    return 0;
}

TRACEPOINT_PROBE(sched, sched_process_exit) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    u32 tid = (u32)pid_tgid;
    if (!is_tracked(pid)) {
        return 0;
    }

    struct event_t evt = {};
    evt.mono_ts_ns = bpf_ktime_get_ns();
    evt.event_type = EVENT_EXIT;
    evt.pid = pid;
    evt.tid = tid;
    bpf_get_current_comm(&evt.comm, sizeof(evt.comm));
    events.perf_submit(args, &evt, sizeof(evt));

    /*
     * Remove exited processes from tracked_pids so the BPF hash map
     * doesn't fill up with dead entries.  Only remove when the main
     * thread exits (pid == tid), since threads share the tgid.
     */
    if (pid == tid) {
        tracked_pids.delete(&pid);
    }
    return 0;
}

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    if (!is_tracked(pid)) {
        return 0;
    }

    u64 id = args->id;
    if (!is_traced_syscall(id)) {
        return 0;
    }

    struct syscall_state_t state = {};
    state.ts_ns = bpf_ktime_get_ns();
    state.syscall_id = id;
    state.arg0 = args->args[0];
    state.arg1 = args->args[1];
    state.arg2 = args->args[2];
    switch (id) {
        case __NR_openat:
        case __NR_faccessat:
        case __NR_newfstatat:
        case __NR_unlinkat:
        case __NR_mkdirat:
        case __NR_renameat2:
            bpf_probe_read_user_str(
                &state.filepath,
                sizeof(state.filepath),
                (void *)args->args[1]
            );
            break;
        case __NR_execve:
        case __NR_access:
        case __NR_truncate:
        case __NR_chdir:
            bpf_probe_read_user_str(
                &state.filepath,
                sizeof(state.filepath),
                (void *)args->args[0]
            );
            break;
        default:
            break;
    }
    inflight.update(&pid_tgid, &state);
    return 0;
}

TRACEPOINT_PROBE(raw_syscalls, sys_exit) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    u32 tid = (u32)pid_tgid;
    if (!is_tracked(pid)) {
        return 0;
    }

    u64 id = args->id;
    if (!is_traced_syscall(id)) {
        return 0;
    }

    struct syscall_state_t *state = inflight.lookup(&pid_tgid);
    if (!state) {
        return 0;
    }

    if (state->syscall_id != id) {
        inflight.delete(&pid_tgid);
        return 0;
    }

    struct event_t evt = {};
    evt.mono_ts_ns = bpf_ktime_get_ns();
    evt.event_type = EVENT_SYSCALL;
    evt.pid = pid;
    evt.tid = tid;
    evt.ret = args->ret;
    evt.syscall_id = id;
    evt.latency_ns = evt.mono_ts_ns - state->ts_ns;
    evt.arg0 = state->arg0;
    evt.arg1 = state->arg1;
    evt.arg2 = state->arg2;
    __builtin_memcpy(&evt.filepath, &state->filepath, sizeof(evt.filepath));
    bpf_get_current_comm(&evt.comm, sizeof(evt.comm));
    events.perf_submit(args, &evt, sizeof(evt));

    inflight.delete(&pid_tgid);
    return 0;
}
"""


class EventStruct(ctypes.Structure):
    _fields_ = [
        ("mono_ts_ns", ctypes.c_ulonglong),
        ("event_type", ctypes.c_uint),
        ("pid", ctypes.c_uint),
        ("tid", ctypes.c_uint),
        ("aux_pid", ctypes.c_uint),
        ("ret", ctypes.c_longlong),
        ("syscall_id", ctypes.c_ulonglong),
        ("latency_ns", ctypes.c_ulonglong),
        ("arg0", ctypes.c_ulonglong),
        ("arg1", ctypes.c_ulonglong),
        ("arg2", ctypes.c_ulonglong),
        ("filepath", ctypes.c_char * 256),
        ("comm", ctypes.c_char * 16),
    ]


SYSCALL_ID_TO_NAME = {
    0: "read",
    1: "write",
    3: "close",
    35: "nanosleep",
    7: "poll",
    23: "select",
    56: "clone",
    59: "execve",
    61: "wait4",
    72: "fcntl",
    78: "getdents",
    79: "getcwd",
    80: "chdir",
    81: "fchdir",
    202: "futex",
    230: "clock_nanosleep",
    232: "epoll_wait",
    247: "waitid",
    217: "getdents64",
    257: "openat",
    258: "mkdirat",
    259: "mknodat",
    260: "fchownat",
    261: "futimesat",
    262: "newfstatat",
    263: "unlinkat",
    264: "renameat",
    267: "readlinkat",
    268: "fchmodat",
    269: "faccessat",
    271: "ppoll",
    270: "pselect6",
    281: "epoll_pwait",
    272: "unshare",
    273: "set_robust_list",
    292: "dup3",
    316: "renameat2",
    322: "execveat",
    326: "copy_file_range",
    76: "truncate",
    77: "ftruncate",
    17: "pread64",
    18: "pwrite64",
    19: "readv",
    20: "writev",
    21: "access",
    5: "fstat",
    74: "fsync",
    75: "fdatasync",
    277: "sync_file_range",
    295: "preadv",
    296: "pwritev",
    327: "preadv2",
    328: "pwritev2",
    # Network syscalls (x86_64). Captured when --include-net is enabled
    # so HTTP-heavy agents (e.g. SRAgent) don't show up as a giant time gap.
    41: "socket",
    42: "connect",
    43: "accept",
    44: "sendto",
    45: "recvfrom",
    46: "sendmsg",
    47: "recvmsg",
    48: "shutdown",
    49: "bind",
    50: "listen",
    288: "accept4",
    299: "recvmmsg",
    307: "sendmmsg",
}


# C cases injected into is_traced_syscall when --include-net is on.
# Matches the HTTP client path; keeps the FS whitelist untouched so the
# original measurement set is preserved verbatim.
NET_SYSCALL_CASES_C = """\
        case __NR_socket:
        case __NR_connect:
        case __NR_sendto:
        case __NR_recvfrom:
        case __NR_sendmsg:
        case __NR_recvmsg:
        case __NR_sendmmsg:
        case __NR_recvmmsg:
"""


def build_bpf_program(include_net: bool) -> str:
    placeholder = "__NET_SYSCALL_CASES__"
    replacement = NET_SYSCALL_CASES_C if include_net else ""
    return BPF_PROGRAM.replace(placeholder, replacement)


def syscall_name(syscall_id: int) -> str:
    return SYSCALL_ID_TO_NAME.get(syscall_id, f"sys_{syscall_id}")


def seed_existing_children(tracked_map: object, root_pid: int) -> None:
    """Scan /proc to find already-running descendants of *root_pid* and add
    them to the BPF ``tracked_pids`` map.  This closes the window between
    the agent starting and the BPF probes becoming active."""
    # Python 3.6 (CentOS Stream 8 system python) doesn't accept PEP 585
    # `set[int]` syntax; drop the annotation.  This file must remain 3.6-clean
    # because BCC bindings (python3-bcc) are tied to the system interpreter.
    visited = set()
    queue = [root_pid]
    while queue:
        pid = queue.pop()
        if pid in visited:
            continue
        visited.add(pid)
        children_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            text = children_path.read_text().strip()
        except OSError:
            continue
        if not text:
            continue
        for tok in text.split():
            try:
                child_pid = int(tok)
            except ValueError:
                continue
            if child_pid not in visited:
                tracked_map[ctypes.c_uint(child_pid)] = ctypes.c_ubyte(1)
                queue.append(child_pid)
    seeded = len(visited) - 1  # exclude root_pid itself
    if seeded:
        print(f"Seeded {seeded} existing child PIDs into tracked_pids", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace a PID tree using BCC and emit JSONL events."
    )
    parser.add_argument(
        "--root-pid",
        type=int,
        required=True,
        help="Root PID whose descendants should be tracked.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--ready-fd",
        type=int,
        default=None,
        help="File descriptor to write 'ready\\n' to once probes are active.  "
             "Useful for synchronising with a launch script.",
    )
    parser.set_defaults(include_net=True)
    parser.add_argument(
        "--include-net",
        dest="include_net",
        action="store_true",
        help="Trace socket syscalls (connect/sendto/recvfrom/...) in addition "
             "to FS syscalls.  Default: enabled.",
    )
    parser.add_argument(
        "--no-include-net",
        dest="include_net",
        action="store_false",
        help="Disable network syscall tracing; preserve original FS-only "
             "measurement set.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    root_pid = int(args.root_pid)
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    running = True

    def _stop(_sig: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    wall_start_ns = int(time.time() * 1_000_000_000)
    mono_start_ns = int(time.monotonic() * 1_000_000_000)

    bpf_text = build_bpf_program(include_net=args.include_net)
    print(
        f"BPF program built (include_net={args.include_net})",
        file=sys.stderr,
    )
    bpf = BPF(text=bpf_text)
    tracked_map = bpf.get_table("tracked_pids")
    tracked_map[ctypes.c_uint(root_pid)] = ctypes.c_ubyte(1)

    # Pick up children that were forked before probes became active.
    seed_existing_children(tracked_map, root_pid)

    # Signal readiness so the launch script can start the agent *after*
    # probes are live (when using --ready-fd).
    if args.ready_fd is not None:
        try:
            os.write(args.ready_fd, b"ready\n")
            os.close(args.ready_fd)
        except OSError:
            pass

    with out_path.open("w", encoding="utf-8") as f:
        meta = {
            "type": "meta",
            "wall_start_ns": wall_start_ns,
            "mono_start_ns": mono_start_ns,
            "root_pid": root_pid,
        }
        f.write(json.dumps(meta) + "\n")
        f.flush()

        def _to_wall_ns(evt_mono_ns: int) -> int:
            return wall_start_ns + (int(evt_mono_ns) - mono_start_ns)

        def handle_event(_cpu: int, data: ctypes.c_void_p, _size: int) -> None:
            evt = ctypes.cast(data, ctypes.POINTER(EventStruct)).contents
            evt_type = int(evt.event_type)
            payload = {  # type: Dict[str, object]
                "ts_ns": _to_wall_ns(int(evt.mono_ts_ns)),
                "pid": int(evt.pid),
                "tid": int(evt.tid),
                "comm": evt.comm.decode("utf-8", errors="replace").rstrip("\x00"),
            }

            if evt_type == 1:
                payload["type"] = "fork"
                payload["child_pid"] = int(evt.aux_pid)
            elif evt_type == 2:
                payload["type"] = "exec"
            elif evt_type == 3:
                payload["type"] = "exit"
            elif evt_type == 4:
                sid = int(evt.syscall_id)
                path = evt.filepath.decode("utf-8", errors="replace").rstrip("\x00")
                payload["type"] = "syscall"
                payload["syscall_id"] = sid
                payload["syscall"] = syscall_name(sid)
                payload["ret"] = int(evt.ret)
                payload["latency_ns"] = int(evt.latency_ns)
                payload["arg0"] = int(evt.arg0)
                payload["arg1"] = int(evt.arg1)
                payload["arg2"] = int(evt.arg2)
                payload["path"] = path if path else None
            else:
                return

            f.write(json.dumps(payload) + "\n")

        bpf["events"].open_perf_buffer(handle_event, page_cnt=256)

        print(
            f"Tracing PID tree rooted at {root_pid}; writing to {out_path}",
            file=sys.stderr,
        )
        while running:
            bpf.perf_buffer_poll(timeout=200)
            f.flush()

        # Drain any final events before exiting.
        for _ in range(3):
            bpf.perf_buffer_poll(timeout=50)
        f.flush()

    print("Tracer stopped.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
