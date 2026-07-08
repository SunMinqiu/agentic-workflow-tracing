#!/usr/bin/env python3
"""W0 ground-truth I/O microbench for Phase-1 counter calibration.

It emits a deliberately simple known pattern:
  1. small-file create/write/read/unlink loop
  2. one batched file write/read

If mdtest/ior are installed, it can also run them, but the Python pattern is
the calibration anchor because its expected syscall/file counts are explicit
and portable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def run(cmd: list[str], cwd: Path, timeout: int) -> dict:
    try:
        p = subprocess.run(
            cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout,
        )
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except FileNotFoundError:
        return {"cmd": cmd, "skipped": "not found"}
    except subprocess.TimeoutExpired as e:
        return {"cmd": cmd, "skipped": "timeout", "stdout": e.stdout, "stderr": e.stderr}


def python_pattern(root: Path, n_files: int, file_size: int,
                   batch_size: int) -> dict:
    small_dir = root / "small_files"
    small_dir.mkdir(parents=True, exist_ok=True)
    payload = b"x" * file_size
    t0 = time.time()
    for i in range(n_files):
        p = small_dir / f"f_{i:06d}.dat"
        with p.open("wb") as f:
            f.write(payload)
    create_write_done = time.time()
    total_read = 0
    for i in range(n_files):
        p = small_dir / f"f_{i:06d}.dat"
        with p.open("rb") as f:
            total_read += len(f.read())
    read_done = time.time()
    for i in range(n_files):
        (small_dir / f"f_{i:06d}.dat").unlink()
    unlink_done = time.time()

    batch_path = root / "batched.dat"
    batch_payload = b"y" * batch_size
    with batch_path.open("wb") as f:
        f.write(batch_payload)
    with batch_path.open("rb") as f:
        batch_read = len(f.read())
    batch_path.unlink()
    t1 = time.time()

    return {
        "n_small_files": n_files,
        "small_file_size": file_size,
        "small_total_write_bytes": n_files * file_size,
        "small_total_read_bytes": total_read,
        "small_expected_min": {
            "creates": n_files,
            "write_calls": n_files,
            "read_calls": n_files,
            "unlinks": n_files,
            "files_touched": n_files,
        },
        "batch_size": batch_size,
        "batch_read_bytes": batch_read,
        "batch_expected_min": {
            "creates": 1,
            "write_calls": 1,
            "read_calls": 1,
            "unlinks": 1,
            "files_touched": 1,
        },
        "timing_s": {
            "small_create_write": create_write_done - t0,
            "small_read": read_done - create_write_done,
            "small_unlink": unlink_done - read_done,
            "batch_total": t1 - unlink_done,
            "total": t1 - t0,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True,
                   help="Directory on the target filesystem, e.g. Lustre.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-files", type=int, default=1024)
    p.add_argument("--file-size", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=4 * 1024 * 1024)
    p.add_argument("--run-ior-mdtest", action="store_true")
    args = p.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    bench_dir = root / f"w0_{int(time.time())}_{os.getpid()}"
    bench_dir.mkdir()
    result = {
        "root": str(root),
        "bench_dir": str(bench_dir),
        "python_pattern": python_pattern(
            bench_dir, args.n_files, args.file_size, args.batch_size
        ),
        "external": {},
    }

    if args.run_ior_mdtest:
        if shutil.which("mdtest"):
            result["external"]["mdtest"] = run(
                ["mdtest", "-d", str(bench_dir / "mdtest"), "-n", str(args.n_files),
                 "-F", "-C", "-T", "-r"],
                bench_dir,
                timeout=600,
            )
        else:
            result["external"]["mdtest"] = {"skipped": "mdtest not found"}
        if shutil.which("ior"):
            result["external"]["ior"] = run(
                ["ior", "-o", str(bench_dir / "ior_file"), "-w", "-r",
                 "-b", str(args.batch_size), "-t", str(args.file_size)],
                bench_dir,
                timeout=600,
            )
        else:
            result["external"]["ior"] = {"skipped": "ior not found"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Bench dir left in place for tracing/stat inspection: {bench_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
