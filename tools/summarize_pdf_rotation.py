"""Summarize completed GenoMAS reference arms from captured calls and metrics."""

import argparse
import json
from collections import Counter
from pathlib import Path

from agent_io_tracing.serving.vllm_endpoint import prometheus_window


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(cell):
    events = records(cell / "pi_events.jsonl")
    calls = {}
    for event in events:
        if event.get("type") in {"message_request_start", "message_first_token", "message_end"}:
            calls.setdefault(event["run_id"], {})[event["type"]] = event
    prompt = output = cached = measured = 0
    durations, ttfts, starts, ends = [], [], [], []
    server_ttfts, server_decodes = [], []
    for call in calls.values():
        end = call.get("message_end")
        if end is None:
            continue
        usage = end.get("message", {}).get("usage", {})
        timing = end.get("server_metrics") or {}
        if isinstance(timing.get("time_to_first_token_ms"), (int, float)):
            server_ttfts.append(timing["time_to_first_token_ms"] / 1000)
        if isinstance(timing.get("generation_time_ms"), (int, float)):
            server_decodes.append(timing["generation_time_ms"] / 1000)
        prompt += usage.get("input", 0)
        output += usage.get("output", 0)
        if usage.get("cacheReadAvailable"):
            cached += usage.get("cacheRead", 0)
            measured += 1
        start = call.get("message_request_start")
        first = call.get("message_first_token")
        if start:
            starts.append(start["wall_time_ms"])
            ends.append(end["wall_time_ms"])
            durations.append((end["monotonic_ns"] - start["monotonic_ns"]) / 1e9)
            if first:
                ttfts.append((first["wall_time_ms"] - start["wall_time_ms"]) / 1000)
    messages = records(cell / "messages.jsonl")
    references = [row["cohort_reference"] for row in messages if row.get("cohort_reference")]
    injected = [row for row in references if row.get("actual_tokens", 0) > 0]
    before, after = cell / "vllm_cache_before_metrics.prom", cell / "vllm_cache_after_metrics.prom"
    metrics = prometheus_window(before.read_text(), after.read_text())["metrics"] if before.exists() and after.exists() else {}

    def delta(name):
        return metrics.get("vllm:" + name, {}).get("delta")

    prefill = delta("request_prefill_time_seconds_sum")
    decode = delta("request_decode_time_seconds_sum")
    return {
        "cell": str(cell), "calls": len(messages), "completed_timing_calls": len(durations),
        "prompt_tokens": prompt, "output_tokens": output,
        "cached_tokens": cached if measured == len(messages) and messages else None,
        "cache_measured_calls": measured,
        "uncached_tokens": prompt - cached if measured == len(messages) and messages else None,
        "llm_call_seconds": sum(durations), "client_ttft_seconds": sum(ttfts),
        "mean_client_call_seconds": sum(durations) / len(durations) if durations else None,
        "mean_client_ttft_seconds": sum(ttfts) / len(ttfts) if ttfts else None,
        "mean_server_ttft_seconds": sum(server_ttfts) / len(server_ttfts) if server_ttfts else None,
        "mean_server_decode_seconds": sum(server_decodes) / len(server_decodes) if server_decodes else None,
        "server_timing_measured_calls": len(server_decodes),
        "ttft_share_of_calls": sum(ttfts) / sum(durations) if durations and sum(durations) and len(ttfts) == len(durations) else None,
        "first_request_to_last_response_seconds": (max(ends) - min(starts)) / 1000 if starts else None,
        "server_prefill_seconds": prefill, "server_decode_seconds": decode,
        "server_prefill_share_of_prefill_decode": prefill / (prefill + decode) if prefill is not None and decode is not None and prefill + decode else None,
        "server_request_count": delta("request_prefill_time_seconds_count"),
        "server_prompt_tokens": delta("prompt_tokens_total"),
        "server_preemptions": delta("num_preemptions_total"),
        "reference_calls": len(injected),
        "reference_tokens": sum(row["actual_tokens"] for row in injected),
        "reference_preparation_seconds": sum(row.get("preparation_ms", 0) for row in references) / 1000,
        "reference_skips": dict(Counter(row.get("reason") for row in references if row.get("status") == "skipped")),
        "reference_sequence": [{key:row.get(key) for key in ["cohort", "role", "call_index", "document_id", "source_start", "source_end", "actual_tokens", "position"]} for row in injected],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(cell) for cell in args.cells], indent=2, ensure_ascii=False))
