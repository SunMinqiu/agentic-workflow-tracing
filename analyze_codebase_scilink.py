#!/usr/bin/env python3
"""
Run a SciLink CLI invocation in-process under our litellm tool/event logger.

Mirrors analyze_codebase_sragent.py.  Two key differences from the SRAgent
version:

  1. The logger is litellm-based (litellm.success_callback + monkey-patch
     of AnalysisOrchestratorTools.execute_tool), not LangChain-based.
     `langchain_tool_logger` is irrelevant to SciLink — see comments in
     litellm_tool_logger.py.

  2. SciLink's CLI runs an interactive `input()`-based REPL ("👤 You: ...").
     Our trace runs aren't interactive, so we pre-load sys.stdin with the
     workload's prompt and rely on the loop's EOFError handler to terminate
     cleanly once the prompt has been consumed.

Outputs in <log_dir>:
  - tool_calls.log                parse_ebpf.py format-compatible
  - tool_calls.log.system_prompt  system prompt capture
  - pi_events.jsonl               summarize_pi_events.py format-compatible
  - subagent_calls.log            empty placeholder (no SciLink subagent
                                  classification yet)

Usage:
    python analyze_codebase_scilink.py <work_dir> <log_dir> <subcommand> \\
        --prompt "<prompt text>" -- <scilink_args>...

Example (eels_plasmons_demo):
    python analyze_codebase_scilink.py /tmp/work /tmp/log analyze \\
        --prompt "Find and characterize the plasmon peaks." -- \\
        --mode autonomous --model gpt-4o-mini \\
        --data examples/eels_plasmons_demo/datacube.npy \\
        --metadata examples/eels_plasmons_demo/datacube.json \\
        --session-dir /tmp/log/scilink_session

The text after `--` is forwarded verbatim to `scilink <subcommand>`.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import traceback
from pathlib import Path


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split on the first standalone '--'; everything after goes to scilink."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1:]
    return argv, []


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a SciLink subcommand under the pi-compatible litellm logger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "work_dir",
        type=Path,
        help="Working directory for SciLink (it cd's here; session artifacts land here).",
    )
    p.add_argument(
        "log_dir",
        type=Path,
        help="Directory for tool_calls.log, pi_events.jsonl, system prompt.",
    )
    p.add_argument(
        "subcommand",
        type=str,
        help="SciLink subcommand (usually 'analyze' for eels_plasmons_demo and friends).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text fed to the agent's REPL via stdin.  With --mode autonomous "
             "the agent should run to completion on this one prompt; the EOF "
             "after it makes the REPL exit cleanly.",
    )
    p.add_argument(
        "--pre",
        type=str,
        default="",
        help="Space-separated string of GLOBAL flags to insert BEFORE the "
             "subcommand.  SciLink's CLI currently has no global flags so "
             "this is normally empty; kept for parity with the SRAgent harness.",
    )
    return p


def main() -> int:
    ours, scilink_args = _split_argv(sys.argv[1:])
    args = build_arg_parser().parse_args(ours)

    work_dir = args.work_dir.resolve()
    log_dir = args.log_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Install the handler BEFORE importing scilink so the litellm callback
    # list and the monkey-patch on AnalysisOrchestratorTools.execute_tool
    # are in place before any SciLink module wires them in.
    import litellm  # type: ignore
    from litellm_tool_logger import LiteLLMToolLogger, install_global

    # SciLink passes provider/tool parameters such as `tool_choice` through
    # litellm.  Some OpenAI model aliases reject those at litellm's validation
    # layer even when they are harmless for our workload.  Dropping unsupported
    # optional params keeps model swaps from aborting before the orchestrator
    # can do any real work.
    litellm.drop_params = True
    os.environ["LITELLM_DROP_PARAMS"] = "true"

    original_completion = litellm.completion

    def completion_with_drop_params(*c_args, **c_kwargs):
        c_kwargs.setdefault("drop_params", True)
        return original_completion(*c_args, **c_kwargs)

    completion_with_drop_params._pi_drop_params_patched = True  # type: ignore[attr-defined]
    litellm.completion = completion_with_drop_params
    print(
        "[analyze_codebase_scilink] litellm.drop_params=True; "
        "completion() injects drop_params=True",
        file=sys.stderr,
        flush=True,
    )

    handler = LiteLLMToolLogger(log_dir=log_dir)
    install_global(handler)

    # Parity with pi-coding-agent and SRAgent: some downstream code (and our
    # own debug paths) reads PI_TOOL_LOG from env.
    os.environ["PI_TOOL_LOG"] = str(log_dir / "tool_calls.log")

    # cd into work_dir so any local artefacts (SciLink session dir, generated
    # python scripts written by the autonomous code-exec agent, etc.) land
    # under our control and BCC's path filtering still works.
    os.chdir(work_dir)

    # Feed the workload's prompt via stdin.  SciLink's main chat loop is
    # `while True: user_input = input(...)`; reading from a StringIO returns
    # our prompt once, then raises EOFError on the next call.  scilink/cli/
    # analyze.py catches that and exits the REPL cleanly.
    prompt_text = args.prompt.rstrip("\n") + "\n"
    sys.stdin = io.StringIO(prompt_text)

    pre_argv = args.pre.split() if args.pre else []
    sys.argv = ["scilink", *pre_argv, args.subcommand, *scilink_args]
    print(
        f"[analyze_codebase_scilink] cwd={work_dir} log_dir={log_dir}\n"
        f"[analyze_codebase_scilink] argv={sys.argv}\n"
        f"[analyze_codebase_scilink] stdin prompt={args.prompt[:120]!r}",
        file=sys.stderr,
        flush=True,
    )

    exit_code = 0
    try:
        # Defer SciLink import until after install_global() has run.
        from scilink.cli.main import main as scilink_main  # type: ignore
    except ImportError as e:
        print(
            f"[analyze_codebase_scilink] failed to import SciLink: {e}\n"
            "Hint: pip install scilink (or pip install -e . from a clone) "
            "in this venv.",
            file=sys.stderr,
        )
        return 2

    try:
        result = scilink_main()
        if isinstance(result, int):
            exit_code = result
    except SystemExit as se:
        exit_code = int(se.code) if isinstance(se.code, int) else (0 if se.code is None else 1)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        handler.flush_pending()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
