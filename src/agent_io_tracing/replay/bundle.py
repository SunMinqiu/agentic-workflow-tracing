"""Build a token-exact replay bundle from a traced workflow cell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agent_io_tracing.serving.vllm_endpoint import VLLMEndpoint


SCHEMA_VERSION = 1


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _trace_by_run(cell: Path) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    for event in _jsonl(cell / "pi_events.jsonl"):
        run_id = event.get("run_id")
        if not run_id:
            continue
        trace = traces.setdefault(str(run_id), {})
        event_type = event.get("type")
        timestamp = float(
            (event.get("message") or {}).get("timestamp")
            or event.get("wall_time_ms")
            or 0.0
        )
        if event_type in {"message_request_start", "message_start"}:
            trace.setdefault("start_ms", timestamp)
        if event_type == "message_end":
            trace["end_ms"] = timestamp
            error = event.get("error")
            if error not in (None, "", "None"):
                trace["error"] = error
                continue
            raw = (event.get("message") or {}).get("usage") or {}
            trace["usage"] = {
                "input": int(raw.get("input", 0) or 0),
                "output": int(raw.get("output", 0) or 0),
                "cache_read": int(raw.get("cacheRead", 0) or 0),
            }
    return traces


def build_bundle(
    cell: Path,
    endpoint: VLLMEndpoint,
    served_model: str,
    limit: int | None = None,
) -> dict[str, Any]:
    messages_path = cell / "messages.jsonl"
    events_path = cell / "pi_events.jsonl"
    if not messages_path.is_file() or not events_path.is_file():
        raise ValueError(f"{cell} is missing messages.jsonl or pi_events.jsonl")
    rows = sorted(
        _jsonl(messages_path),
        key=lambda row: float(row.get("timestamp", 0.0) or 0.0),
    )
    traces = _trace_by_run(cell)
    if not rows:
        raise ValueError(f"{messages_path} contains no requests")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    first_timestamp = float(rows[0].get("timestamp", 0.0) or 0.0)
    requests = []
    incomplete = set()
    for index, row in enumerate(rows):
        run_id = str(row.get("run_id") or f"request-{index}")
        request_capture = row.get("request")
        if not isinstance(request_capture, dict):
            request_capture = {}
        messages = request_capture.get("messages") or row.get("messages") or []
        if not isinstance(messages, list):
            raise ValueError(f"{run_id} messages are not a list")
        trace = traces.get(run_id, {})
        recorded_usage = trace.get("usage") or {}
        request_params = (
            request_capture.get("parameters")
            if isinstance(request_capture.get("parameters"), dict)
            else row.get("request_params")
        )
        if not isinstance(request_params, dict):
            request_params = {}
            incomplete.add("request_params")
        template_options = {
            key: request_params[key]
            for key in (
                "tools",
                "add_generation_prompt",
                "continue_final_message",
                "chat_template",
                "chat_template_kwargs",
            )
            if key in request_params
        }
        token_ids = endpoint.tokenize_chat(
            served_model,
            messages,
            **template_options,
        )
        response_capture = row.get("response")
        if not isinstance(response_capture, dict):
            response_capture = {}
        response_text = (
            response_capture.get("text")
            if "text" in response_capture else row.get("response_text")
        )
        if "response" not in row and "response_text" not in row:
            incomplete.add("response_text")
        requests.append({
            "index": index,
            "request_id": run_id,
            "role": row.get("agent_role") or row.get("genomas_role"),
            "phase": row.get("phase"),
            "source_model": row.get("model"),
            "arrival_offset_ms": max(
                float(row.get("timestamp", 0.0) or 0.0) - first_timestamp,
                0.0,
            ),
            "original_duration_ms": max(
                float(trace.get("end_ms", 0.0) or 0.0)
                - float(trace.get("start_ms", 0.0) or 0.0),
                0.0,
            ),
            "prompt_token_ids": token_ids,
            "prompt_tokens": len(token_ids),
            "sampling_params": request_params,
            "recorded_usage": recorded_usage,
            "recorded_response": response_capture,
            "recorded_response_text": response_text,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "source_cell": str(cell.resolve()),
        "served_model": served_model,
        "capture_completeness": {
            "token_ids": True,
            "missing_fields": sorted(incomplete),
        },
        "requests": requests,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a token-exact vLLM replay bundle from one trace cell.",
    )
    parser.add_argument("cell", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--url",
        default=(
            os.environ.get("VLLM_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_SERVED_MODEL"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY") or "",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Build a small validation bundle from the first N requests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.url or not args.model:
        raise SystemExit("Set the vLLM URL and served model.")
    endpoint = VLLMEndpoint(args.url, args.api_key, timeout_s=120.0)
    try:
        bundle = build_bundle(args.cell, endpoint, args.model, args.limit)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"bundle build failed: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "requests": len(bundle["requests"]),
        "prompt_tokens": sum(
            request["prompt_tokens"] for request in bundle["requests"]
        ),
        "capture_completeness": bundle["capture_completeness"],
    }, indent=2))


if __name__ == "__main__":
    main()
