from __future__ import annotations

import json
import sys
from pathlib import Path

from agent_io_tracing.analysis.postprocess import run_stage, write_timings
from agent_io_tracing.parsing._ebpf_impl import iter_events
from agent_io_tracing.tracing.bcc_tracer import build_arg_parser, build_bpf_program


def test_wait_syscalls_are_opt_in_and_anonymous_mmap_is_filtered() -> None:
    default_program = build_bpf_program(include_net=True)
    assert "case __NR_futex:" not in default_program
    assert "case __NR_epoll_wait:" not in default_program
    assert "args->args[3] & 0x20" in default_program
    assert "(s32)args->args[4] < 0" in default_program

    debug_program = build_bpf_program(include_net=False, include_waits=True)
    assert "case __NR_futex:" in debug_program
    assert "case __NR_epoll_wait:" in debug_program
    assert "case __NR_recvfrom:" not in debug_program
    assert build_arg_parser().parse_args(
        ["--root-pid", "1", "--output", "trace.jsonl"]
    ).include_waits is False


def test_event_stream_restores_small_cross_cpu_reordering(tmp_path: Path) -> None:
    path = tmp_path / "ebpf_events.log"
    rows = [
        {"type": "meta", "wall_start_ns": 0},
        {"type": "syscall", "ts_ns": 300_000_000, "syscall": "write"},
        {"type": "syscall", "ts_ns": 100_000_000, "syscall": "read"},
        {"type": "syscall", "ts_ns": 350_000_000, "syscall": "close"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert [row["ts_ns"] for row in iter_events(path)] == [
        100_000_000, 300_000_000, 350_000_000,
    ]


def test_parallel_stage_and_timing_manifest(tmp_path: Path) -> None:
    env = {}
    jobs = [
        ("one", [sys.executable, "-c", "print('one')"], tmp_path, "one.log", env),
        ("two", [sys.executable, "-c", "print('two')"], tmp_path, "two.log", env),
    ]
    results = run_stage(jobs, max_workers=2)
    assert [result.returncode for result in results] == [0, 0]
    output = write_timings(tmp_path, results, max_workers=2)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["max_workers"] == 2
    assert [step["name"] for step in payload["steps"]] == ["one", "two"]
    assert (tmp_path / "one.log").read_text().strip() == "one"
