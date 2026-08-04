#!/usr/bin/env python3
"""Run a fixed-input Montage mosaic and record stage-level execution units."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Stage:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


STAGES = (
    Stage("header", (), ("region.hdr",)),
    Stage("raw_table", ("raw",), ("rimages.tbl",), ("header",)),
    Stage(
        "projection",
        ("raw", "rimages.tbl", "region.hdr"),
        ("projected",),
        ("raw_table",),
    ),
    Stage("projected_table", ("projected",), ("pimages.tbl",), ("projection",)),
    Stage("overlaps", ("pimages.tbl",), ("diffs.tbl",), ("projected_table",)),
    Stage(
        "difference_fitting",
        ("projected", "diffs.tbl", "region.hdr"),
        ("fits.tbl",),
        ("overlaps",),
    ),
    Stage(
        "background_model",
        ("pimages.tbl", "fits.tbl"),
        ("corrections.tbl",),
        ("difference_fitting",),
    ),
    Stage(
        "background_correction",
        ("projected", "pimages.tbl", "corrections.tbl"),
        ("corrected",),
        ("background_model",),
    ),
    Stage(
        "corrected_table",
        ("corrected",),
        ("cimages.tbl",),
        ("background_correction",),
    ),
    Stage(
        "coadd",
        ("corrected", "cimages.tbl", "region.hdr"),
        ("mosaic.fits", "mosaic_area.fits"),
        ("corrected_table",),
    ),
    Stage("render", ("mosaic.fits",), ("mosaic.png",), ("coadd",)),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--size-deg", type=float, required=True)
    parser.add_argument("--location", default="M 17")
    parser.add_argument("--execution-units-log", type=Path)
    parser.add_argument("--offline", action="store_true")
    return parser


def _status_ok(result: object) -> bool:
    return isinstance(result, dict) and str(result.get("status", "")) == "0"


def _output_exists(work_dir: Path, name: str) -> bool:
    path = work_dir / name
    if path.is_dir():
        return any(item.is_file() for item in path.rglob("*"))
    return path.is_file() and path.stat().st_size > 0


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_pipeline(
    work_dir: Path,
    size_deg: float,
    location: str,
    execution_units_log: Path,
    *,
    offline: bool,
    functions: dict[str, Callable[..., Any]] | None = None,
) -> int:
    if size_deg <= 0:
        raise ValueError("--size-deg must be positive")
    raw_dir = work_dir / "raw"
    if not raw_dir.is_dir() or not any(raw_dir.glob("*.fits")):
        raise FileNotFoundError(f"fixed FITS input is missing or empty: {raw_dir}")
    for name in ("projected", "diffs", "corrected"):
        path = work_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to reuse workflow output: {path}")
        path.mkdir()

    if functions is None:
        from MontagePy.main import (  # type: ignore[import-not-found]
            mAdd,
            mBgExec,
            mBgModel,
            mDiffFitExec,
            mHdr,
            mImgtbl,
            mOverlaps,
            mProjExec,
            mViewer,
        )

        functions = {
            "header": lambda: mHdr(location, size_deg, size_deg, "region.hdr"),
            "raw_table": lambda: mImgtbl("raw", "rimages.tbl"),
            "projection": lambda: mProjExec(
                "raw", "rimages.tbl", "region.hdr", projdir="projected", quickMode=True
            ),
            "projected_table": lambda: mImgtbl("projected", "pimages.tbl"),
            "overlaps": lambda: mOverlaps("pimages.tbl", "diffs.tbl"),
            "difference_fitting": lambda: mDiffFitExec(
                "projected", "diffs.tbl", "region.hdr", "diffs", "fits.tbl"
            ),
            "background_model": lambda: mBgModel(
                "pimages.tbl", "fits.tbl", "corrections.tbl"
            ),
            "background_correction": lambda: mBgExec(
                "projected", "pimages.tbl", "corrections.tbl", "corrected"
            ),
            "corrected_table": lambda: mImgtbl("corrected", "cimages.tbl"),
            "coadd": lambda: mAdd(
                "corrected", "cimages.tbl", "region.hdr", "mosaic.fits"
            ),
            "render": lambda: mViewer(
                "-ct 1 -gray mosaic.fits -2s max gaussian-log -out mosaic.png",
                "",
                mode=2,
            ),
        }

    units: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    completed: list[str] = []
    pid = os.getpid()
    previous_cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        for stage in STAGES:
            start_ts_ms = time.time() * 1000.0
            print(f"[montage-driver] started {stage.name}", flush=True)
            error = ""
            result: object = {}
            try:
                result = functions[stage.name]()
                if not _status_ok(result):
                    error = f"Montage returned failure: {result!r}"
                missing = [name for name in stage.outputs if not _output_exists(work_dir, name)]
                if missing:
                    error = f"declared output(s) missing or empty: {', '.join(missing)}"
            except Exception as exc:  # keep the stage record before terminating
                error = str(exc)
            end_ts_ms = time.time() * 1000.0
            units.append(
                {
                    "execution_unit_id": f"montage_{stage.name}",
                    "task_id": f"montage_{stage.name}",
                    "stage": stage.name,
                    "pid": pid,
                    "start_ts_ms": start_ts_ms,
                    "end_ts_ms": end_ts_ms,
                    "status": "failed" if error else "completed",
                    "returncode": 1 if error else 0,
                    "dependencies": [f"montage_{name}" for name in stage.dependencies],
                    "inputs": list(stage.inputs),
                    "outputs": list(stage.outputs),
                }
            )
            _write_jsonl(execution_units_log, units)
            if error:
                failures.append({"task": stage.name, "error": error})
                print(f"[montage-driver] FAILED {stage.name}: {error}", file=sys.stderr)
                break
            completed.append(stage.name)
            print(f"[montage-driver] completed {stage.name}", flush=True)
    finally:
        os.chdir(previous_cwd)

    summary = {
        "offline": offline,
        "location": location,
        "size_deg": size_deg,
        "tasks_total": len(STAGES),
        "tasks_completed": len(completed),
        "tasks_not_run": [stage.name for stage in STAGES if stage.name not in completed],
        "failures": failures,
        "input_fits_count": len(list(raw_dir.glob("*.fits"))),
    }
    (work_dir / "montage_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_dir = args.work_dir.expanduser().resolve()
    units_path = (
        args.execution_units_log.expanduser().resolve()
        if args.execution_units_log
        else work_dir / "execution_units.jsonl"
    )
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        return run_pipeline(
            work_dir,
            args.size_deg,
            args.location,
            units_path,
            offline=args.offline,
        )
    except (OSError, ValueError) as exc:
        print(f"[montage-driver] setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
