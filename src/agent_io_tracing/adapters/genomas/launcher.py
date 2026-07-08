#!/usr/bin/env python3
"""
Run GenoMAS (https://github.com/Liu-Hy/GenoMAS) in-process under our
genomas_tool_logger.

Mirrors analyze_codebase_sragent.py / analyze_codebase_scilink.py for the
GenoMAS target.  Two big differences from SRAgent and SciLink:

  1. Logger is hand-rolled (genomas_tool_logger) — GenoMAS goes through
     neither LangChain nor litellm; it monkey-patches its own LLMClient
     subclasses.  See module docstring of genomas_tool_logger.py.

  2. GenoMAS hardcodes `task_info_file = './metadata/task_info.json'` and
     iterates over EVERY (trait, condition) pair in it.  For a smoke /
     small-scale run we slice task_info.json down to only the traits whose
     data lives under `<data_root>/GEO/` (or `<data_root>/TCGA/`), and drop
     all conditions so the matrix collapses to one task per trait.  The
     original task_info.json is backed up to task_info.json.full and
     restored in `finally`.

Outputs in <log_dir>:
  - tool_calls.log              parse_ebpf.py format-compatible
  - tool_calls.log.system_prompt system prompt capture
  - pi_events.jsonl             summarize_pi_events.py format-compatible
  - subagent_calls.log          empty placeholder (Phase 2 MVP)
  - genomas.stdout / .stderr    captured GenoMAS console output
  - sliced_traits.json          which traits we kept for this run

Usage:
    python analyze_codebase_genomas.py <work_dir> <log_dir> \
        --data-root /mnt/lustrefs/genomas_data \
        [--model gpt-5-mini-2025-08-07] [--quick-test] \
        [--smoke-traits trait1,trait2] \
        [--keep-conditions] \
        -- <extra_genomas_args>

The text after `--` is forwarded verbatim to GenoMAS's `main.py` argparse.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1:]
    return argv, []


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run GenoMAS under the pi-compatible LLM-call logger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("work_dir", type=Path,
                   help="Working dir for GenoMAS (it cd's here).")
    p.add_argument("log_dir", type=Path,
                   help="Dir for tool_calls.log, pi_events.jsonl, etc.")
    p.add_argument("--data-root", type=Path, required=True,
                   help="Root with GEO/ and TCGA/ subdirs (Lustre path).")
    p.add_argument("--model", default="gpt-5-mini-2025-08-07",
                   help="Default --model passed to main.py (default: gpt-5-mini).")
    p.add_argument("--api", type=int, default=1,
                   help="Default --api index passed to main.py (default: 1).")
    p.add_argument("--version", default="smoke",
                   help="Default --version passed to main.py (default: smoke).")
    p.add_argument("--quick-test", action="store_true",
                   help="Forward --quick-test to main.py (skip statistician).")
    p.add_argument("--smoke-traits", default=None,
                   help="Comma-separated trait names to keep.  Default: auto-"
                        "detect every trait that has a dir under "
                        "<data-root>/GEO/ or <data-root>/TCGA/.")
    p.add_argument("--keep-conditions", action="store_true",
                   help="By default the slicer drops all conditions to keep "
                        "the smoke matrix at len(traits).  Pass this to "
                        "preserve the full per-trait condition list.")
    p.add_argument("--parallel-mode", choices=["none", "cohorts"], default="none",
                   help="Forward to main.py (default: none).")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Forward to main.py (default: 1).")
    return p


# ---------------------------------------------------------------------------
# task_info.json slicer
# ---------------------------------------------------------------------------


def _detect_traits_from_data_root(data_root: Path) -> list[str]:
    """Return the union of subdir names under data_root/GEO and data_root/TCGA.
    These are the traits we have local data for; everything else would fail
    at file-not-found and waste budget.
    """
    traits: set[str] = set()
    for sub in ("GEO", "TCGA"):
        d = data_root / sub
        if not d.is_dir():
            continue
        for child in d.iterdir():
            if child.is_dir():
                traits.add(child.name)
    return sorted(traits)


def slice_task_info(
    task_info_path: Path,
    keep_traits: list[str],
    keep_conditions: bool,
) -> tuple[dict, dict]:
    """Read task_info.json, return (sliced, original) dicts.
    Caller writes sliced back to disk and stores original for restore.
    """
    with task_info_path.open() as f:
        original = json.load(f)

    sliced: dict = {}
    for trait in keep_traits:
        if trait not in original:
            print(f"[analyze_codebase_genomas] WARNING: trait {trait!r} "
                  f"present in data_root but missing from task_info.json; "
                  f"skipping", file=sys.stderr, flush=True)
            continue
        entry = dict(original[trait])
        if not keep_conditions:
            entry["conditions"] = []
        sliced[trait] = entry
    return sliced, original


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ours, genomas_args = _split_argv(sys.argv[1:])
    args = build_arg_parser().parse_args(ours)

    work_dir = args.work_dir.resolve()
    log_dir = args.log_dir.resolve()
    data_root = args.data_root.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.is_dir():
        print(f"[analyze_codebase_genomas] ERROR: --data-root {data_root} "
              f"does not exist", file=sys.stderr)
        return 2
    # GenoMAS main.py unconditionally lists both subdirectories.  A GEO-only
    # smoke run is valid for this wrapper, so create the empty sibling if the
    # downloaded fixture only populated one side.
    for child in ("GEO", "TCGA"):
        (data_root / child).mkdir(exist_ok=True)

    # --- locate the GenoMAS repo via sys.path -----------------------------
    # The launcher is in pi-ebpf-tracing-handoff/.  We rely on the harness
    # invoker setting CWD to the GenoMAS repo root (so `metadata/task_info.json`
    # is a valid relative path) OR explicitly passing GENOMAS_REPO env var.
    genomas_repo = Path(os.environ.get("GENOMAS_REPO", os.getcwd())).resolve()
    task_info_path = genomas_repo / "metadata" / "task_info.json"
    if not task_info_path.is_file():
        print(f"[analyze_codebase_genomas] ERROR: cannot find {task_info_path}.  "
              f"Set GENOMAS_REPO env var or cd into the GenoMAS repo before "
              f"running.", file=sys.stderr)
        return 2
    sys.path.insert(0, str(genomas_repo))

    # --- decide which traits to keep --------------------------------------
    if args.smoke_traits:
        keep_traits = [t.strip() for t in args.smoke_traits.split(",") if t.strip()]
        print(f"[analyze_codebase_genomas] traits from --smoke-traits: "
              f"{keep_traits}", file=sys.stderr)
    else:
        keep_traits = _detect_traits_from_data_root(data_root)
        print(f"[analyze_codebase_genomas] traits auto-detected from "
              f"{data_root}: {keep_traits}", file=sys.stderr)

    if not keep_traits:
        print(f"[analyze_codebase_genomas] ERROR: no traits found under "
              f"{data_root}/{{GEO,TCGA}}.  Populate the data dir first.",
              file=sys.stderr)
        return 2

    # --- slice task_info.json ---------------------------------------------
    sliced, original = slice_task_info(
        task_info_path, keep_traits, args.keep_conditions
    )
    if not sliced:
        print(f"[analyze_codebase_genomas] ERROR: after slicing to "
              f"{keep_traits}, no traits remain.  Check that the trait "
              f"names match keys in metadata/task_info.json.",
              file=sys.stderr)
        return 2

    backup_path = task_info_path.with_suffix(".json.full")
    sliced_record = log_dir / "sliced_traits.json"
    sliced_record.write_text(json.dumps(
        {"kept_traits": list(sliced.keys()),
         "n_traits": len(sliced),
         "keep_conditions": args.keep_conditions,
         "original_size_bytes": task_info_path.stat().st_size},
        indent=2,
    ))

    print(f"[analyze_codebase_genomas] slicing task_info.json: "
          f"{len(original)} traits → {len(sliced)} traits "
          f"(conditions {'kept' if args.keep_conditions else 'dropped'})",
          file=sys.stderr)

    # Atomic-ish backup-and-replace, restore in finally.
    shutil.copy2(task_info_path, backup_path)
    task_info_path.write_text(json.dumps(sliced, indent=2))

    # --- install the logger BEFORE importing GenoMAS ----------------------
    # genomas_tool_logger lives in this directory (the harness dir),
    # which is NOT on sys.path automatically.
    harness_dir = Path(__file__).resolve().parent
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))

    from agent_io_tracing.adapters.genomas.logger import GenoMASToolLogger, install_global

    handler = GenoMASToolLogger(log_dir=log_dir)

    # cd into the GenoMAS repo so `./metadata/task_info.json` and
    # `./tools/preprocess.py` resolve correctly inside main.py.  We cd
    # AFTER backing up task_info.json (path was absolute then) so the
    # restore step still finds the right file.
    os.chdir(genomas_repo)
    patched = install_global(handler)
    if not patched:
        print(f"[analyze_codebase_genomas] WARNING: no LLM clients patched; "
              f"the run will proceed but pi_events.jsonl will be empty",
              file=sys.stderr)

    # --- build argv for main.py and run -----------------------------------
    # main.py uses argparse with --version REQUIRED and --model REQUIRED,
    # so we always pass them.  Any extra forwarded args via -- override.
    forwarded = [
        "--version", args.version,
        "--model", args.model,
        "--api", str(args.api),
        "--data-root", str(data_root),
        "--parallel-mode", args.parallel_mode,
        "--max-workers", str(args.max_workers),
    ]
    if args.quick_test:
        forwarded.append("--quick-test")
    forwarded.extend(genomas_args)

    sys.argv = ["main.py", *forwarded]
    print(f"[analyze_codebase_genomas] sys.argv={sys.argv}", file=sys.stderr,
          flush=True)

    exit_code = 0
    try:
        # Import is deferred so install_global() runs first.
        from main import main as genomas_main  # type: ignore

        # main() in GenoMAS is `async def`.  Spin a fresh event loop.
        try:
            asyncio.run(genomas_main())
        except SystemExit as se:
            exit_code = (
                int(se.code) if isinstance(se.code, int)
                else (0 if se.code is None else 1)
            )
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        # Restore original task_info.json *always*, even on crash.
        try:
            shutil.move(str(backup_path), str(task_info_path))
            print(f"[analyze_codebase_genomas] restored {task_info_path} "
                  f"from {backup_path.name}", file=sys.stderr)
        except Exception as e:
            print(f"[analyze_codebase_genomas] ERROR restoring task_info.json: "
                  f"{e!r}.  Original backed up at {backup_path}.",
                  file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
