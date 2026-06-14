#!/usr/bin/env python3
"""
Run a SRAgent subcommand in-process under our LangChain tool/event logger.

Mirrors analyze_codebase_pi.py for the SRAgent target.  The key difference:
SRAgent has no first-class extension API (unlike pi's tool_call_logger.ts),
so we install a global LangChain callback handler before importing SRAgent,
which causes every Runnable inside SRAgent to inherit our handler.

Outputs in <log_dir>:
  - tool_calls.log              (parse_ebpf.py format-compatible)
  - tool_calls.log.system_prompt (system prompt capture)
  - pi_events.jsonl             (summarize_pi_events.py format-compatible)

Usage:
    python analyze_codebase_sragent.py <work_dir> <log_dir> <subcommand> -- <sragent_args>...

Examples:
    # entrez (free-text input)
    python analyze_codebase_sragent.py /tmp/sr-work /tmp/sr-log entrez -- "find SRRs for GSE125970"

    # metadata (structured input)
    python analyze_codebase_sragent.py /tmp/sr-work /tmp/sr-log metadata -- --accessions SRX19162512

    # papers (writes PDFs into work_dir)
    python analyze_codebase_sragent.py /tmp/sr-work /tmp/sr-log papers -- --accessions SRX19162512

The text after `--` is forwarded verbatim to `SRAgent <subcommand>`.
"""

from __future__ import annotations

# === pysqlite3 shim ========================================================
# ChromaDB (transitively imported by SRAgent's cli/__main__) requires
# sqlite3 >= 3.35.0.  CentOS Stream 8 ships sqlite 3.26.x, which is too old.
# pysqlite3-binary bundles a newer sqlite; swap it in for the stdlib `sqlite3`
# BEFORE any SRAgent module loads.  No-op if pysqlite3 isn't installed (e.g.
# on Ubuntu 24.04 where the system sqlite is already new enough).
import sys as _sys
try:
    __import__("pysqlite3")
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass
# ===========================================================================

import argparse
import os
import sys
import traceback
from pathlib import Path


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split on the first standalone '--'; everything after goes to SRAgent."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a SRAgent subcommand under the pi-compatible tool logger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "work_dir",
        type=Path,
        help="Working directory for SRAgent (it cd's here; PDFs/cache land here).",
    )
    p.add_argument(
        "log_dir",
        type=Path,
        help="Directory for tool_calls.log, pi_events.jsonl, system prompt.",
    )
    p.add_argument(
        "subcommand",
        type=str,
        help="SRAgent subcommand: entrez | sragent | metadata | find-datasets | "
             "papers | srx-info | tissue-ontology | disease-ontology",
    )
    p.add_argument(
        "--pre",
        type=str,
        default="",
        help="Space-separated string of GLOBAL flags to insert BEFORE the "
             "subcommand (e.g. --pre='--no-summaries').  Use the `key=value` "
             "form so argparse doesn't mistake the flag for one of ours.",
    )
    return p


def main() -> int:
    ours, sragent_args = _split_argv(sys.argv[1:])
    args = build_arg_parser().parse_args(ours)

    work_dir = args.work_dir.resolve()
    log_dir = args.log_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Install the callback BEFORE importing SRAgent so every Runnable created
    # during agent construction inherits it.
    from langchain_tool_logger import LangChainToolLogger, install_global

    handler = LangChainToolLogger(log_dir=log_dir)
    install_global(handler)

    # Some pi-compatible code (and our extension) reads PI_TOOL_LOG from env.
    # Set it for parity.
    os.environ["PI_TOOL_LOG"] = str(log_dir / "tool_calls.log")

    # cd into work_dir so any local artefacts (papers PDFs, ChromaDB, etc.)
    # land somewhere we control and BCC's path filtering still works.
    os.chdir(work_dir)

    # SRAgent's dynaconf loads settings.yml by relative path, which breaks
    # once we've cd'd away from the SRAgent repo. Pin it via env vars BEFORE
    # any SRAgent import (importing SRAgent triggers dynaconf init, after
    # which these env vars are ignored).  find_spec locates the package
    # without executing __init__.py.
    import importlib.util as _ilu
    _settings_path = os.environ.get("SRAGENT_SETTINGS")
    if not _settings_path:
        _spec = _ilu.find_spec("SRAgent")
        if _spec and _spec.origin:
            _cand = Path(_spec.origin).parent / "settings.yml"
            if _cand.exists():
                _settings_path = str(_cand)
    if _settings_path:
        os.environ["SETTINGS_FILE_FOR_DYNACONF"] = _settings_path
        os.environ["ROOT_PATH_FOR_DYNACONF"] = str(Path(_settings_path).parent)
        # SRAgent's settings.yml is env-namespaced (test/prod/claude); `default`
        # has no `models:` section.  Default to `prod` unless user overrides.
        os.environ.setdefault("ENV_FOR_DYNACONF", "prod")
        print(
            f"[analyze_codebase_sragent] dynaconf settings -> {_settings_path} "
            f"(env={os.environ['ENV_FOR_DYNACONF']})",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[analyze_codebase_sragent] WARNING: could not locate SRAgent "
            "settings.yml; set $SRAGENT_SETTINGS to its absolute path",
            file=sys.stderr,
            flush=True,
        )

    # Reconstruct sys.argv as if `SRAgent [pre-args] <subcommand> <args>`
    # was invoked.  Pre-args (global flags) go BEFORE the subcommand.
    pre_argv = args.pre.split() if args.pre else []
    sys.argv = ["SRAgent", *pre_argv, args.subcommand, *sragent_args]
    print(
        f"[analyze_codebase_sragent] cwd={work_dir} log_dir={log_dir}\n"
        f"[analyze_codebase_sragent] argv={sys.argv}",
        file=sys.stderr,
        flush=True,
    )

    exit_code = 0
    try:
        # Defer this import so SRAgent's own LangChain initialization runs
        # *after* install_global().
        from SRAgent.cli.__main__ import main as sragent_main  # type: ignore
    except ImportError as e:
        print(
            f"[analyze_codebase_sragent] failed to import SRAgent: {e}\n"
            "Hint: install with `pip install SRAgent` or from "
            "https://github.com/ArcInstitute/SRAgent",
            file=sys.stderr,
        )
        return 2

    try:
        result = sragent_main()
        if isinstance(result, int):
            exit_code = result
    except SystemExit as se:
        # argparse / SRAgent often raises SystemExit; preserve its code.
        exit_code = int(se.code) if isinstance(se.code, int) else (0 if se.code is None else 1)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        handler.flush_pending()
        # Run time-interval-based subagent reclassification.  SRAgent's
        # sub_agent.invoke(...) calls don't forward RunnableConfig, so the
        # logger's run_id-tree subagent detection misses Invoke_*_agent /
        # Invoke_*_workflow tools at trace time.  This pass moves them from
        # tool_calls.log to subagent_calls.log using LLM message_start
        # timestamps from pi_events.jsonl (which are faithful regardless of
        # callback-chain breakage).  Idempotent.
        try:
            from reclassify_subagents import reclassify  # type: ignore
            reclass_result = reclassify(log_dir)
            print(
                "[analyze_codebase_sragent] reclassify_subagents: "
                f"{reclass_result}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:
            print(
                f"[analyze_codebase_sragent] reclassify failed (non-fatal): {e}",
                file=sys.stderr,
                flush=True,
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
