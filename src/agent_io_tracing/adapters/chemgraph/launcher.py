#!/usr/bin/env python3
"""
Run a ChemGraph XANES MCP workflow under the pi-compatible LangChain logger.

This mirrors ChemGraph's examples/xanes_mcp/mcp_stdio/run_chemgraph.py, but
lets the trace harness choose the prompt/model/output paths and writes:

  - tool_calls.log
  - tool_calls.log.system_prompt
  - pi_events.jsonl
  - subagent_calls.log

Usage:
    python analyze_codebase_chemgraph.py <work_dir> <log_dir> \
        --model gpt4o --prompt "Run a XANES calculation ..."
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import sys
import traceback
from pathlib import Path
from typing import Any


DEFAULT_MCP_SERVER_MODULE = "chemgraph.mcp.xanes_mcp_parsl"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ChemGraph XANES MCP workflow under pi-compatible logging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("work_dir", type=Path, help="Working directory for ChemGraph/FDMNES.")
    p.add_argument("log_dir", type=Path, help="Directory for trace-compatible logs.")
    p.add_argument("--model", default=os.environ.get("CHEMGRAPH_MODEL", "gpt4o"))
    p.add_argument("--prompt", required=True, help="Natural-language ChemGraph prompt.")
    p.add_argument(
        "--workflow-type",
        default="single_agent_xanes",
        help="ChemGraph workflow_type. Default matches the XANES MCP example.",
    )
    p.add_argument(
        "--mcp-server-module",
        default=os.environ.get("CHEMGRAPH_MCP_SERVER_MODULE", DEFAULT_MCP_SERVER_MODULE),
        help="Python module launched as the XANES MCP stdio server.",
    )
    p.add_argument(
        "--mcp-server-script",
        default=os.environ.get("CHEMGRAPH_MCP_SERVER_SCRIPT"),
        help="Python script launched as the XANES MCP stdio server. "
             "If set, takes precedence over --mcp-server-module.",
    )
    return p


async def _run(args: argparse.Namespace, handler: Any) -> int:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from chemgraph.agent.llm_agent import ChemGraph

    if args.mcp_server_script:
        server_args = ["-u", str(Path(args.mcp_server_script).resolve())]
        server_desc = str(Path(args.mcp_server_script).resolve())
    else:
        server_args = ["-u", "-m", args.mcp_server_module]
        server_desc = args.mcp_server_module

    client = MultiServerMCPClient(
        {
            "XANES MCP": {
                "transport": "stdio",
                "command": sys.executable,
                "args": server_args,
                "env": {**os.environ},
            },
        }
    )

    tools = await client.get_tools()
    print(
        "[analyze_codebase_chemgraph] connected to XANES MCP via stdio; "
        f"server={server_desc} tools={[t.name for t in tools]}",
        file=sys.stderr,
        flush=True,
    )

    cg = ChemGraph(
        model_name=args.model,
        workflow_type=args.workflow_type,
        structured_output=False,
        return_option="state",
        tools=tools,
    )

    print(
        f"[analyze_codebase_chemgraph] model={args.model} "
        f"workflow_type={args.workflow_type}\n"
        f"[analyze_codebase_chemgraph] prompt={args.prompt[:240]!r}",
        file=sys.stderr,
        flush=True,
    )

    try:
        sig = inspect.signature(cg.run)
        if "config" in sig.parameters:
            result = await cg.run(args.prompt, config={"callbacks": [handler]})
        else:
            result = await cg.run(args.prompt)
    except TypeError as e:
        # Some ChemGraph revisions do not expose RunnableConfig at run().
        # install_global(handler) is already active, so retry without config.
        print(
            "[analyze_codebase_chemgraph] cg.run(config=...) not accepted; "
            f"retrying without config ({e})",
            file=sys.stderr,
            flush=True,
        )
        result = await cg.run(args.prompt)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result)
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()

    work_dir = args.work_dir.resolve()
    log_dir = args.log_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    from agent_io_tracing.adapters.sragent.logger import LangChainToolLogger, install_global

    handler = LangChainToolLogger(log_dir=log_dir)
    install_global(handler)
    os.environ["PI_TOOL_LOG"] = str(log_dir / "tool_calls.log")

    exit_code = 0
    try:
        exit_code = asyncio.run(_run(args, handler))
    except SystemExit as se:
        exit_code = int(se.code) if isinstance(se.code, int) else (0 if se.code is None else 1)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        handler.flush_pending()
        try:
            from reclassify_subagents import reclassify  # type: ignore

            reclass_result = reclassify(log_dir)
            print(
                f"[analyze_codebase_chemgraph] reclassify_subagents: {reclass_result}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:
            print(
                f"[analyze_codebase_chemgraph] reclassify failed (non-fatal): {e}",
                file=sys.stderr,
                flush=True,
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
