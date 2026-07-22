#!/usr/bin/env python3
"""Run the 1000genome task DAG directly, without Pegasus or HTCondor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path


MAIN_VCF = "ALL.chr{chromosome}.250000.vcf"
ANNOTATION_VCF = (
    "ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated_v5."
    "20130502.sites.annotation.vcf"
)


@dataclass(frozen=True)
class Task:
    name: str
    command: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskResult:
    task: Task
    returncode: int
    outputs: dict[str, Path]
    error: str | None = None


def _csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=os.environ.get("CLASSIC_WORKFLOW_REPO"),
        help="1000genome-workflow checkout (default: CLASSIC_WORKFLOW_REPO).",
    )
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--chromosomes", type=_csv, required=True)
    parser.add_argument("--populations", type=_csv, default=["ALL"])
    parser.add_argument("--individual-jobs", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--rows-per-chromosome", type=int, default=250_000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--main-vcf-template", default=MAIN_VCF)
    parser.add_argument("--annotation-vcf-template", default=ANNOTATION_VCF)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Record that all code, dependencies, and inputs were pre-staged.",
    )
    return parser


def _script(repo: Path, name: str) -> str:
    path = repo / "bin" / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"workflow task script not found: {path}")
    return str(path)


def _logical_name(value: str, label: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{label} must be a plain filename, got: {value!r}")
    return value


def build_tasks(args: argparse.Namespace, repo: Path) -> tuple[list[Task], set[str]]:
    if args.individual_jobs < 1:
        raise ValueError("--individual-jobs must be positive")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    if args.rows_per_chromosome < 2:
        raise ValueError("--rows-per-chromosome must be at least 2")
    if len(set(args.chromosomes)) != len(args.chromosomes):
        raise ValueError("--chromosomes contains duplicates")
    if len(set(args.populations)) != len(args.populations):
        raise ValueError("--populations contains duplicates")
    for population in args.populations:
        _logical_name(population, "population")

    individual_jobs = min(args.individual_jobs, args.rows_per_chromosome)
    if args.rows_per_chromosome % individual_jobs:
        raise ValueError(
            "--individual-jobs must divide --rows-per-chromosome "
            f"({individual_jobs} does not divide {args.rows_per_chromosome})"
        )

    scripts = {
        name: _script(repo, name)
        for name in (
            "individuals",
            "individuals_merge",
            "sifting",
            "mutation_overlap",
            "frequency",
        )
    }
    python = str(args.python)
    tasks: list[Task] = []
    initial_inputs = {"columns.txt", *args.populations}
    step = args.rows_per_chromosome // individual_jobs

    for chromosome in args.chromosomes:
        if not chromosome.isdigit() or not 1 <= int(chromosome) <= 22:
            raise ValueError(f"invalid chromosome: {chromosome!r}")
        main_vcf = _logical_name(
            args.main_vcf_template.format(chromosome=chromosome), "main VCF"
        )
        annotation_vcf = _logical_name(
            args.annotation_vcf_template.format(chromosome=chromosome), "annotation VCF"
        )
        initial_inputs.update((main_vcf, annotation_vcf))

        individual_task_names: list[str] = []
        individual_outputs: list[str] = []
        counter = 1
        while counter < args.rows_per_chromosome:
            stop = counter + step
            task_name = f"chr{chromosome}_individuals_{counter}_{stop}"
            output = f"chr{chromosome}n-{counter}-{stop}.tar.gz"
            tasks.append(
                Task(
                    name=task_name,
                    command=(
                        python,
                        scripts["individuals"],
                        main_vcf,
                        chromosome,
                        str(counter),
                        str(stop),
                        str(args.rows_per_chromosome),
                    ),
                    inputs=(main_vcf, "columns.txt"),
                    outputs=(output,),
                )
            )
            individual_task_names.append(task_name)
            individual_outputs.append(output)
            counter += step

        sift_name = f"chr{chromosome}_sifting"
        sifted_output = f"sifted.SIFT.chr{chromosome}.txt"
        tasks.append(
            Task(
                name=sift_name,
                command=(python, scripts["sifting"], annotation_vcf, chromosome),
                inputs=(annotation_vcf,),
                outputs=(sifted_output,),
            )
        )

        merge_name = f"chr{chromosome}_individuals_merge"
        merged_output = f"chr{chromosome}n.tar.gz"
        tasks.append(
            Task(
                name=merge_name,
                command=(
                    python,
                    scripts["individuals_merge"],
                    chromosome,
                    *individual_outputs,
                ),
                inputs=tuple(individual_outputs),
                outputs=(merged_output,),
                dependencies=tuple(individual_task_names),
            )
        )

        for population in args.populations:
            common_inputs = (merged_output, sifted_output, population, "columns.txt")
            dependencies = (merge_name, sift_name)
            tasks.extend(
                (
                    Task(
                        name=f"chr{chromosome}_mutation_overlap_{population}",
                        command=(
                            python,
                            scripts["mutation_overlap"],
                            "-c",
                            chromosome,
                            "-pop",
                            population,
                        ),
                        inputs=common_inputs,
                        outputs=(f"chr{chromosome}-{population}.tar.gz",),
                        dependencies=dependencies,
                    ),
                    Task(
                        name=f"chr{chromosome}_frequency_{population}",
                        command=(
                            python,
                            scripts["frequency"],
                            "-c",
                            chromosome,
                            "-pop",
                            population,
                        ),
                        inputs=common_inputs,
                        outputs=(f"chr{chromosome}-{population}-freq.tar.gz",),
                        dependencies=dependencies,
                    ),
                )
            )
    return tasks, initial_inputs


def _safe_task_dir(tasks_dir: Path, name: str) -> Path:
    safe_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not name or any(character not in safe_characters for character in name):
        raise ValueError(f"unsafe task name: {name!r}")
    task_dir = tasks_dir / name
    task_dir.mkdir(parents=True, exist_ok=False)
    return task_dir


def _run_task(task: Task, tasks_dir: Path, inputs: dict[str, Path]) -> TaskResult:
    try:
        task_dir = _safe_task_dir(tasks_dir, task.name)
        for logical_name in task.inputs:
            source = inputs[logical_name].resolve(strict=True)
            destination = task_dir / logical_name
            destination.symlink_to(source)
        with (task_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
            task_dir / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                task.command,
                cwd=task_dir,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if completed.returncode:
            return TaskResult(task, completed.returncode, {}, "task process failed")
        output_paths: dict[str, Path] = {}
        missing: list[str] = []
        for logical_name in task.outputs:
            output = task_dir / logical_name
            if output.exists():
                output_paths[logical_name] = output.resolve()
            else:
                missing.append(logical_name)
        if missing:
            return TaskResult(
                task,
                1,
                output_paths,
                "declared output(s) missing: " + ", ".join(missing),
            )
        return TaskResult(task, 0, output_paths)
    except Exception as exc:
        return TaskResult(task, 1, {}, str(exc))


def _register_artifact(artifacts_dir: Path, logical_name: str, source: Path) -> None:
    destination = artifacts_dir / logical_name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"duplicate artifact: {logical_name}")
    destination.symlink_to(source.resolve(strict=True))


def run_dag(
    tasks: list[Task],
    initial_inputs: set[str],
    work_dir: Path,
    max_workers: int,
    *,
    offline: bool = False,
) -> int:
    tasks_dir = work_dir / "tasks"
    artifacts_dir = work_dir / "artifacts"
    tasks_dir.mkdir(exist_ok=False)
    artifacts_dir.mkdir(exist_ok=False)

    artifacts: dict[str, Path] = {}
    for logical_name in sorted(initial_inputs):
        source = work_dir / logical_name
        if not source.exists():
            raise FileNotFoundError(f"staged workflow input not found: {source}")
        artifacts[logical_name] = source.resolve()
        _register_artifact(artifacts_dir, logical_name, source)

    pending = {task.name: task for task in tasks}
    completed: set[str] = set()
    running: dict[Future[TaskResult], Task] = {}
    failures: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while pending or running:
            if not failures:
                ready = [
                    task
                    for task in pending.values()
                    if set(task.dependencies).issubset(completed)
                    and all(item in artifacts for item in task.inputs)
                ]
                for task in sorted(ready, key=lambda item: item.name):
                    if len(running) >= max_workers:
                        break
                    task_inputs = {name: artifacts[name] for name in task.inputs}
                    future = pool.submit(_run_task, task, tasks_dir, task_inputs)
                    running[future] = task
                    del pending[task.name]
                    print(f"[1000genome-driver] started {task.name}", flush=True)

            if not running:
                if pending:
                    blocked = ", ".join(sorted(pending))
                    if failures:
                        print(
                            f"[1000genome-driver] not run after failure: {blocked}",
                            file=sys.stderr,
                        )
                    else:
                        failures.append(
                            {"task": "scheduler", "error": f"blocked tasks: {blocked}"}
                        )
                break

            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                result = future.result()
                if result.returncode:
                    message = result.error or f"exit code {result.returncode}"
                    failures.append(
                        {
                            "task": task.name,
                            "returncode": result.returncode,
                            "error": message,
                        }
                    )
                    print(
                        f"[1000genome-driver] FAILED {task.name}: {message}",
                        file=sys.stderr,
                    )
                    continue
                try:
                    for logical_name, source in result.outputs.items():
                        _register_artifact(artifacts_dir, logical_name, source)
                        artifacts[logical_name] = source
                except OSError as exc:
                    failures.append({"task": task.name, "error": str(exc)})
                    print(f"[1000genome-driver] FAILED {task.name}: {exc}", file=sys.stderr)
                    continue
                completed.add(task.name)
                print(f"[1000genome-driver] completed {task.name}", flush=True)

    summary = {
        "offline": offline,
        "tasks_total": len(tasks),
        "tasks_completed": len(completed),
        "tasks_not_run": sorted(pending),
        "failures": failures,
        "artifacts": sorted(artifacts),
    }
    (work_dir / "classic_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repo is None:
        print(
            "[1000genome-driver] --repo or CLASSIC_WORKFLOW_REPO is required",
            file=sys.stderr,
        )
        return 2
    repo = args.repo.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    if not repo.is_dir():
        print(f"[1000genome-driver] repository does not exist: {repo}", file=sys.stderr)
        return 2
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        tasks, initial_inputs = build_tasks(args, repo)
        return run_dag(
            tasks,
            initial_inputs,
            work_dir,
            args.max_workers,
            offline=args.offline,
        )
    except (OSError, ValueError) as exc:
        print(f"[1000genome-driver] setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
