#!/usr/bin/env python3
"""Stage inputs and launch a classic workflow behind a trace-ready stop."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


def _input_spec(value: str) -> tuple[Path, Path]:
    source_text, separator, destination_text = value.partition(":")
    source = Path(source_text).expanduser().resolve()
    destination = Path(destination_text) if separator else Path(source.name)
    if not source.exists():
        raise ValueError(f"input does not exist: {source}")
    if destination.is_absolute() or ".." in destination.parts or destination == Path("."):
        raise ValueError(f"input destination must stay below work_dir: {destination}")
    return source, destination


def _stage_input(work_dir: Path, value: str, *, copy: bool) -> None:
    source, relative_destination = _input_spec(value)
    destination = work_dir / relative_destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace staged input: {destination}")
    if copy:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def _environment(values: list[str], repo: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if repo is not None:
        env["CLASSIC_WORKFLOW_REPO"] = str(repo)
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError(f"--env must be KEY=VAL, got: {value!r}")
        env[key] = item
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--cmd", required=True, help="Command parsed with shlex.split().")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--input", action="append", default=[], metavar="SRC[:DEST]")
    parser.add_argument("--copy-input", action="append", default=[], metavar="SRC[:DEST]")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VAL")
    parser.add_argument(
        "--no-self-stop",
        action="store_true",
        help="Do not SIGSTOP before exec (intended for tests and direct use).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    ours, extra = _split_argv(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(ours)

    work_dir = args.work_dir.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    repo = args.repo.expanduser().resolve() if args.repo else None
    if repo is not None and not repo.is_dir():
        print(f"[classic-launcher] repository does not exist: {repo}", file=sys.stderr)
        return 2

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        for value in args.input:
            _stage_input(work_dir, value, copy=False)
        for value in args.copy_input:
            _stage_input(work_dir, value, copy=True)
        (log_dir / "pi_events.jsonl").write_text("", encoding="utf-8")
        (log_dir / "tool_calls.log").write_text("", encoding="utf-8")
        env = _environment(args.env, repo)
        command = shlex.split(args.cmd)
        if repo is not None:
            command = [part.replace("{repo}", str(repo)) for part in command]
        command.extend(extra)
        if not command:
            raise ValueError("--cmd produced an empty command")
    except (OSError, ValueError) as exc:
        print(f"[classic-launcher] setup failed: {exc}", file=sys.stderr)
        return 2

    if not args.no_self_stop:
        print("[classic-launcher] staged inputs; stopping for tracer attach", file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGSTOP)

    print(f"[classic-launcher] exec: {shlex.join(command)}", file=sys.stderr, flush=True)
    try:
        with (log_dir / "classic.stdout").open("w", encoding="utf-8") as stdout, (
            log_dir / "classic.stderr"
        ).open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command,
                cwd=work_dir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            previous_handlers: dict[signal.Signals, object] = {}

            def forward_signal(signum: int, _frame: object) -> None:
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

            for forwarded_signal in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[forwarded_signal] = signal.signal(
                    forwarded_signal, forward_signal
                )
            try:
                returncode = process.wait()
            finally:
                for forwarded_signal, previous in previous_handlers.items():
                    signal.signal(forwarded_signal, previous)
    except OSError as exc:
        print(f"[classic-launcher] launch failed: {exc}", file=sys.stderr)
        return 127
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
