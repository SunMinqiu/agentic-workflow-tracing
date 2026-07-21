#!/usr/bin/env python3
"""
Analyze a codebase using pi-coding-agent in JSON mode.

This script runs pi non-interactively, saves raw JSON events to pi_events.jsonl,
and relies on tool_call_logger.ts to produce tool_calls.log in the format
expected by parse_ebpf.py.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
import re
import subprocess
import sys
from pathlib import Path
"""
main_prompt = (
    "Conduct an in-depth review of the application run results in the current directory and build a new directory structure in /mnt/lus_fs/cases/data_restructured "
    "which is optimized for you to easily search in the future and get information to answer questions about the application run results such as key aggregate metrics."
)


main_prompt = (
    "Review the metadata in the current directory and verify that it is fully consistent with the application run results in /mnt/lus_fs/kaiju_out"
)


main_prompt = (
    "Your current directory contains output data from a scientific workload. Conduct a deep analysis of the output data and summarize the key metrics per dataset."
)
"""
"""
main_prompt = 
Compute the zonal-mean cross-section of atmospheric temperature from CESM output:                                                                                                                                                        
                                                                                                                                                                                                                                                    
  Using the synthetic CESM CAM history files at b.e30.B1850.synth.001.cam.h0.*.nc, perform the following analysis:                                                                                                                             

  1. Open the dataset and read the 3D temperature field T with dimensions (time, lev, lat, lon), along with the coordinate variables lat, lev, hyam, hybm, and P0.
  2. Compute the time-mean and zonal-mean temperature: average T over all 1200 time steps and over the longitude dimension, producing a 2D array with dimensions (lev, lat).
  3. Compute the approximate pressure levels at each vertical level using the hybrid coordinate formula: p(k) = hyam(k) * P0 + hybm(k) * PS_mean, where PS_mean is the global-mean surface pressure (approximately 101325 Pa). Report the pressure
  levels in hPa.
  4. Write the results to a new NetCDF file called T_zonal_mean.nc containing:
    - T_zonalmean(lev, lat) — the zonal and time mean temperature in Kelvin
    - lat(lat) — latitude in degrees
    - lev(lev) — the hybrid level index
    - p_approx(lev) — the approximate pressure at each level in hPa
    - Global attributes: source, description, units
  5. Report the following verification statistics:
    - Global mean temperature (averaged over all lev and lat, area-weighted using cos(lat)): expected ~240-260 K
    - Temperature at the equator at the lowest model level: expected ~290-305 K
    - Temperature at the equator at the highest model level: expected ~200-230 K
    - Temperature at 90S at the lowest model level: expected ~240-260 K
    - Minimum temperature in the entire cross-section and its location (lat, lev): expected minimum ~180-210 K near the tropical tropopause
    - Maximum temperature and its location: expected near the tropical surface
  6. Verify physical consistency:
    - Temperature should generally decrease with altitude (lower pressure) in the troposphere
    - Temperature should generally decrease from equator to poles at all levels
    - No values should be below 150 K or above 350 K
    - The cross-section should be approximately symmetric about the equator

  Print all statistics to stdout. If any verification check fails, report which check failed and the actual values."
"""
main_prompt = """
Scan all dataset directories in the local GFDL-AM4 repository.

For each dataset directory:
1. identify the main data variable
2. read all valid values of that variable across all timesteps and dimensions
3. compute:
   - mean_value
   - min_value
   - max_value

Return a CSV with exactly these columns:

dataset,variable,mean_value,min_value,max_value

Sort rows by dataset name ascending.
"""

"""
main_prompt = (
    "The current directory contains output data from a scientific workload and is mounted on a lustre file system. I want to enable efficient downstream processing and analysis of the data by LLM agents. "
    "The downstream tasks may be highly diverse, anything from summarizing the key metrics per dataset to restructuring the data to be fed into external tools. "
    "Conduct a deep analysis of the directory structure and run various tests to understand what may cause low data access performance of the downstream tasks. "
    "Based on your findings, generate a document to enable agents to work with the data for any task most efficiently. Name the document 'efficient_access_skills.md' and place it in the current directory."
)
"""

"""
main_prompt = (
    "The current directory contains output data from a scientific workload "
    "and is mounted on a Lustre file system. I want to enable efficient "
    "downstream processing and analysis of the data by LLM agents. "
    "The downstream tasks may be highly diverse — anything from summarizing "
    "key metrics per dataset to restructuring the data for external tools.\n\n"

    "Conduct a deep analysis in the following order:\n"
    "1. **Inventory**: Map the directory tree. Report total size, file counts "
    "per subdirectory, naming conventions, and file formats (inspect headers "
    "with `file` or format-specific tools like `ncdump -h`). Flag any "
    "anomalies (truncated files, unexpected locations, size outliers).\n"
    "2. **Metadata deep-dive**: For each dataset group, open a representative "
    "file and document dimensions, variable names, shapes, dtypes, and "
    "compression/chunking settings.\n"
    "3. **Filesystem characterization**: Run `lfs getstripe` on representative "
    "files and directories. Report stripe count, stripe size, and OST "
    "distribution. Explain implications for read parallelism.\n"
    "4. **Micro-benchmarks**: Measure and report quantitative results for:\n"
    "   - Metadata traversal (`find`)\n"
    "   - Sequential read throughput (`dd bs=4M`)\n"
    "   - Per-file read time using the native format library\n"
    "   - Random-access penalty (many small point reads vs. block reads)\n"
    "   - Multi-file parallel reads (e.g., multiprocessing with 4-16 workers)\n"
    "5. **Recommended access patterns**: Based on your findings, write "
    "actionable rules (A, B, C, ...) with Python code examples for: "
    "choosing the right file set, building a reusable file manifest, "
    "reading blocks not points, parallelizing across files, avoiding "
    "unnecessary decoding, caching static grids, and optional format "
    "conversion for heavy workloads.\n\n"

    "Format the output as a Markdown document with YAML frontmatter "
    "(`name`, `description`). Name it 'efficient_access_skills.md' and "
    "place it in the current directory."
)
"""




additional_info_2 = ""  # Keep parity with existing script's prompt handling.
PROMPT = main_prompt + additional_info_2

DEFAULT_PI_MODEL = "claude-sonnet-4-6"


def print_skill_content(skill_path: str) -> None:
    print(f"Using skill file: {skill_path}")
    print("----- BEGIN SKILL CONTENT -----")
    print(Path(skill_path).read_text(encoding="utf-8"))
    print("----- END SKILL CONTENT -----")


def build_pi_command(cwd: str, extension_path: str, skill_path: str, use_skill: bool) -> list[str]:
    cmd = [
        "pi",
        "--mode",
        "json",
        "--print",  # non-interactive: process prompt and exit (pi >= 0.7x)
        "--no-session",
        "-e",
        extension_path,
        "--no-prompt-templates",
    ]
    if use_skill:
        cmd.extend(["--skill", skill_path])

    provider = os.getenv("PI_PROVIDER", "").strip()
    if provider:
        cmd.extend(["--provider", provider])

    model = os.getenv("PI_MODEL", "").strip() or DEFAULT_PI_MODEL
    if model:
        cmd.extend(["--model", model])

    thinking = os.getenv("PI_THINKING", "").strip()
    if thinking:
        cmd.extend(["--thinking", thinking])

    # pi >= 0.7x takes the prompt positionally; the old "--" separator is
    # rejected as an unknown option.
    cmd.append(PROMPT)
    return cmd


def _is_delta_event(event: dict) -> bool:
    ae = event.get("assistantMessageEvent")
    return isinstance(ae, dict) and ae.get("type", "").endswith("_delta")


def _delta_payload_text(assistant_event: dict) -> str | None:
    for key in ("delta", "text", "content", "value"):
        value = assistant_event.get(key)
        if isinstance(value, str):
            return value
    delta = assistant_event.get("delta")
    if isinstance(delta, dict):
        for key in ("text", "content", "value"):
            value = delta.get(key)
            if isinstance(value, str):
                return value
    return None


def _estimate_delta_tokens(text: str | None) -> int:
    """Cheap tokenizer-free estimate for plotting decode intensity.

    This is deliberately named *_est in pi_events.jsonl; it is not a model
    tokenizer count.
    """
    if not text:
        return 0
    return max(1, len(re.findall(r"\S+", text)))


def _delta_event_timestamp_ms(event: dict) -> float | None:
    msg = event.get("message")
    if isinstance(msg, dict):
        ts = msg.get("timestamp")
        if isinstance(ts, (int, float)):
            return float(ts)
    ts = event.get("timestamp")
    return float(ts) if isinstance(ts, (int, float)) else None


def _extract_text_delta(event: dict) -> str | None:
    if event.get("type") != "message_update":
        return None
    msg = event.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return None

    assistant_event = event.get("assistantMessageEvent")
    if not isinstance(assistant_event, dict):
        return None
    if assistant_event.get("type") != "text_delta":
        return None

    return _delta_payload_text(assistant_event)


def _event_run_id(event: dict, active_run_id: str | None) -> str | None:
    rid = event.get("run_id")
    if isinstance(rid, str):
        return rid
    message = event.get("message")
    if isinstance(message, dict):
        for key in ("run_id", "id"):
            rid = message.get(key)
            if isinstance(rid, str):
                return rid
    assistant_event = event.get("assistantMessageEvent")
    if isinstance(assistant_event, dict):
        for key in ("run_id", "id"):
            rid = assistant_event.get(key)
            if isinstance(rid, str):
                return rid
    return active_run_id


def _thinking_delta_types() -> set[str]:
    raw = os.getenv("PI_THINKING_DELTA_TYPES", "thinking_delta,reasoning_delta")
    return {part.strip() for part in raw.split(",") if part.strip()}


def analyze(cwd: str, log_dir: str | None = None, use_skill: bool = True) -> int:
    cwd_path = Path(cwd).resolve()
    script_dir = Path(__file__).resolve().parent
    extension_path = script_dir / "tool_call_logger.ts"
    skill_path = Path("/root/.pi/skills/lustre-skill/SKILL.md")
    if use_skill and not skill_path.is_file():
        raise FileNotFoundError(f"Skill not found: {skill_path}")
    if not extension_path.is_file():
        raise FileNotFoundError(f"Extension not found: {extension_path}")

    env = os.environ.copy()

    tool_log_path: Path | None = None
    events_log_path: Path | None = None
    if log_dir:
        out_dir = Path(log_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        tool_log_path = out_dir / "tool_calls.log"
        events_log_path = out_dir / "pi_events.jsonl"
        tool_log_path.write_text("", encoding="utf-8")
        events_log_path.write_text("", encoding="utf-8")
        env["PI_TOOL_LOG"] = str(tool_log_path)

    cmd = build_pi_command(str(cwd_path), str(extension_path), str(skill_path), use_skill)
    print("Using engine: pi --mode json")
    if use_skill:
        print_skill_content(str(skill_path))
    print(f"Prompt: {PROMPT}")
    print(f"Command: {' '.join(cmd[:-1])} <PROMPT>")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    text_by_run: dict[str, list[str]] = defaultdict(list)
    thinking_by_run: dict[str, list[str]] = defaultdict(list)
    delta_type_counts_by_run: dict[str, Counter[str]] = defaultdict(Counter)
    active_run_id: str | None = None
    thinking_delta_types = _thinking_delta_types()
    log_delta_events = os.getenv("PI_LOG_DELTA_EVENTS", "").lower() in {"1", "true", "yes"}
    capture_unknown_thinking = (
        os.getenv("PI_CAPTURE_UNKNOWN_DELTAS_AS_THINKING", "").lower() in {"1", "true", "yes"}
    )

    while True:
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            break
        if not line:
            continue

        stripped = line.strip()
        if not stripped:
            continue

        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            print(f"[warn] non-JSON stdout from pi: {stripped}", file=sys.stderr)
            continue

        event_type = event.get("type")
        rid = _event_run_id(event, active_run_id)
        if event_type == "message_start" and isinstance(rid, str):
            active_run_id = rid

        assistant_event = event.get("assistantMessageEvent")
        assistant_event_type = (
            assistant_event.get("type") if isinstance(assistant_event, dict) else None
        )
        if (
            isinstance(rid, str)
            and isinstance(assistant_event_type, str)
            and assistant_event_type.endswith("_delta")
            and isinstance(assistant_event, dict)
        ):
            delta_type_counts_by_run[rid][assistant_event_type] += 1
            payload = _delta_payload_text(assistant_event)
            if events_log_path:
                compact_delta = {
                    "type": "message_delta",
                    "run_id": rid,
                    "timestamp": _delta_event_timestamp_ms(event),
                    "delta_type": assistant_event_type,
                    "delta_chars": len(payload or ""),
                    "delta_tokens_est": _estimate_delta_tokens(payload),
                }
                with events_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(compact_delta, ensure_ascii=False) + "\n")
            if payload:
                if assistant_event_type == "text_delta":
                    text_by_run[rid].append(payload)
                elif assistant_event_type in thinking_delta_types:
                    thinking_by_run[rid].append(payload)
                elif capture_unknown_thinking:
                    thinking_by_run[rid].append(payload)

        delta = _extract_text_delta(event)
        if delta:
            print(delta, end="", flush=True)

        if event_type == "message_end" and isinstance(rid, str):
            assistant_text = "".join(text_by_run.pop(rid, []))
            assistant_thinking = "".join(thinking_by_run.pop(rid, []))
            delta_type_counts = dict(delta_type_counts_by_run.pop(rid, Counter()))
            if assistant_text:
                event["assistant_text"] = assistant_text
            if assistant_thinking:
                event["assistant_thinking"] = assistant_thinking
            if delta_type_counts:
                event["assistant_delta_type_counts"] = delta_type_counts
            if active_run_id == rid:
                active_run_id = None

        if events_log_path and (log_delta_events or not _is_delta_event(event)):
            with events_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    stderr_output = proc.stderr.read()
    if stderr_output:
        print(stderr_output, end="", file=sys.stderr)

    return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a codebase using pi-coding-agent in JSON mode.",
    )
    parser.add_argument(
        "codebase_dir",
        nargs="?",
        default=os.getcwd(),
        help="Path to the codebase directory (default: current directory)",
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=None,
        help="Directory for log output (tool_calls.log, pi_events.jsonl)",
    )
    parser.set_defaults(use_skill=True)
    parser.add_argument(
        "--skill",
        dest="use_skill",
        action="store_true",
        help="Enable pi --skill (default behavior).",
    )
    parser.add_argument(
        "--no-skill",
        dest="use_skill",
        action="store_false",
        help="Disable pi --skill for this run.",
    )
    args = parser.parse_args()

    cwd = os.path.abspath(args.codebase_dir)
    log_dir = os.path.abspath(args.log_dir) if args.log_dir else None

    if not Path(cwd).is_dir():
        print(f"Error: {cwd} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    try:
        code = analyze(cwd, log_dir, use_skill=args.use_skill)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if "pi" in str(exc).lower():
            print("Hint: install with 'npm install -g @mariozechner/pi-coding-agent'", file=sys.stderr)
        sys.exit(1)

    if code != 0:
        print(f"pi exited with code {code}", file=sys.stderr)
    sys.exit(code)


if __name__ == "__main__":
    main()
