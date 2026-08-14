"""Run one trace's analysis DAG with bounded parallelism and timing records."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class StepResult:
    name: str
    duration_s: float
    returncode: int
    log: str
    skipped: bool = False


def _command(python: str, module: str, *args: str) -> list[str]:
    return [python, "-m", module, *args]


def run_step(
    name: str,
    command: list[str],
    trace_dir: Path,
    log_name: str,
    env: dict[str, str],
) -> StepResult:
    started = time.monotonic()
    log_path = trace_dir / log_name
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    return StepResult(
        name=name,
        duration_s=round(time.monotonic() - started, 3),
        returncode=completed.returncode,
        log=log_name,
    )


def skipped_step(name: str, log_name: str) -> StepResult:
    return StepResult(name, 0.0, 0, log_name, skipped=True)


def run_stage(jobs: list[tuple], max_workers: int) -> list[StepResult]:
    if len(jobs) < 2 or max_workers == 1:
        return [run_step(*job) for job in jobs]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
        futures = [pool.submit(run_step, *job) for job in jobs]
        return [future.result() for future in futures]


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def process_trace(trace_dir: Path, python: str, max_workers: int = 2) -> list[StepResult]:
    trace_dir = trace_dir.resolve()
    env = dict(os.environ)
    path = str(trace_dir)
    results: list[StepResult] = []
    has_semantic_logs = (
        (trace_dir / "pi_events.jsonl").is_file()
        and (trace_dir / "tool_calls.log").is_file()
    )

    results.append(run_step(
        "parse", _command(python, "agent_io_tracing.parsing.ebpf", path),
        trace_dir, "parse.log", env,
    ))
    if results[-1].returncode or not (trace_dir / "parsed.json").is_file():
        return results

    first_stage = [(
        "lineage", _command(python, "agent_io_tracing.lineage.analyzer", path),
        trace_dir, "lineage.log", env,
    )]
    if has_semantic_logs:
        first_stage.append((
            "summary", _command(python, "agent_io_tracing.analysis.summary", path),
            trace_dir, "summarize.log", env,
        ))
    else:
        results.append(skipped_step("summary", "summarize.log"))
    results.extend(run_stage(first_stage, max_workers))

    second_stage = [(
        "per_run_io_char",
        _command(
            python, "agent_io_tracing.analysis.per_run_io_char",
            "--results", path, "--runs", ".",
        ),
        trace_dir, "per_run_io_char.log", env,
    )]
    if has_semantic_logs:
        second_stage.append((
            "parallelism", _command(python, "agent_io_tracing.analysis.parallelism", path),
            trace_dir, "parallelism.log", env,
        ))
    else:
        results.append(skipped_step("parallelism", "parallelism.log"))
    results.extend(run_stage(second_stage, max_workers))

    results.append(run_step(
        "phase1_metrics",
        _command(python, "agent_io_tracing.analysis.phase1_metrics", path),
        trace_dir, "phase1_metrics.log", env,
    ))

    third_stage = [
        (
            "execution_units",
            _command(python, "agent_io_tracing.analysis.execution_units", path),
            trace_dir, "execution_units.log", env,
        ),
        (
            "trace_quality",
            _command(python, "agent_io_tracing.analysis.trace_quality", path),
            trace_dir, "trace_quality.log", env,
        ),
    ]
    results.extend(run_stage(third_stage, max_workers))
    results.append(run_step(
        "visualize", _command(python, "agent_io_tracing.viz.trace", path),
        trace_dir, "visualize.log", env,
    ))
    return results


def write_timings(
    trace_dir: Path,
    results: list[StepResult],
    max_workers: int,
    total_duration_s: float | None = None,
) -> Path:
    trace_stats_path = trace_dir / "trace_stats.json"
    try:
        trace_stats = json.loads(trace_stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        trace_stats = {}
    payload = {
        "schema_version": 1,
        "max_workers": max_workers,
        "total_duration_s": total_duration_s,
        "summed_step_duration_s": round(sum(step.duration_s for step in results), 3),
        "input_bytes": {
            "ebpf_events": _file_size(trace_dir / "ebpf_events.log"),
            "parsed": _file_size(trace_dir / "parsed.json"),
        },
        "trace_stats": trace_stats,
        "steps": [asdict(step) for step in results],
    }
    output = trace_dir / "postprocess_timings.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-workers", type=int, choices=(1, 2), default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.monotonic()
    results = process_trace(args.trace_dir, args.python, args.max_workers)
    wall_s = round(time.monotonic() - started, 3)
    output = write_timings(
        args.trace_dir.resolve(), results, args.max_workers, wall_s,
    )
    for step in results:
        state = "skipped" if step.skipped else "ok" if step.returncode == 0 else "failed"
        print(f"  {step.name}: {state} in {step.duration_s:.1f}s")
    print(f"  timings: {output}")
    return 1 if any(step.returncode for step in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
