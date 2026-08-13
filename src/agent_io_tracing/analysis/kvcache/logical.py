#!/usr/bin/env python3
"""Logical vs realized KV-cache reuse, per traced cell.

Two numbers on the *same* run, both prefix-based (only the longest leading token
run that matches counts; the divergent tail is new):

  realized%  — the vendor's cache actually served this share of prompt tokens
               (``cacheRead`` from pi_events). A FLOOR: bounded by TTL, exact
               prefix, routing, cache capacity.
  logical%   — an ideal, unbounded, never-evicting global prefix cache holding
               every previously-seen prompt would serve this share. A CEILING,
               backend-independent, computed here from the raw messages.

  gap% = logical_aligned% − realized%  → reuse left on the table by the
         vendor's retention, eviction, or routing. Alignment uses the serving
         arm's prefix-matching unit.

Sub-agents / multiple roles: a real shared prefix cache is GLOBAL — any call
whose prefix matches anything cached hits, regardless of which agent issued it.
So the ceiling uses ONE global chronological trie over all calls; different
agents' distinct system prompts naturally fall into separate branches and do not
cross-reuse. We also build per-role tries and report

  cross_agent_bonus% = global_logical − Σ per-role logical

i.e. the extra reuse that comes specifically from prefixes shared across agents
(handoff, shared system preamble). It is ≥ 0 because the global trie holds a
superset of each role's prior prompts.

Inputs per cell (joined by run_id):
  messages.jsonl   — {run_id, genomas_role, timestamp, model, messages:[{role,content}]}
  pi_events.jsonl  — message_end {run_id, message.usage.{input,cacheRead}}

Tokenization is deferred to here (raw text is stored, tokenizer-agnostic): we
re-tokenize with the model's encoding so logical is in the same unit as the
vendor-reported realized. Our chat serialization is not byte-identical to the
provider's template, so absolute token totals differ slightly — we report
tok_delta_pct and compare *fractions*, and assert logical ≥ realized.
"""
from __future__ import annotations

import json
import hashlib
import math
import statistics
from pathlib import Path
from typing import Any

import tiktoken

from agent_io_tracing.analysis.kvcache import server_tokens as _server_tokens

LEGACY_MATCH_UNIT = 128
CELL_JSON = "kvcache_logical.json"
CACHE_WARMING_PNG = "kvcache_cache_warming.png"
PREFIX_LINEAGE_PNG = "kvcache_prefix_lineage.png"
PREFIX_DUMP = "kvcache_prefixes.txt"


def _encoding_for(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # dated names (gpt-4o-mini-2024-07-18) and non-OpenAI models fall here
        return tiktoken.get_encoding("o200k_base")


def _content_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def _serialize(messages: list[dict]) -> str:
    """Canonical, consistent flattening of a messages array to one string."""
    parts = []
    for m in messages or []:
        role = m.get("role") or "?"
        parts.append(f"<|{role}|>\n{_content_str(m.get('content'))}\n")
    return "".join(parts)


def _legacy_vendor(provider: Any, model: Any) -> str:
    model_name = str(model or "").lower()
    if model_name.startswith(("gpt-", "o1", "o3", "o4")):
        return "OpenAI"
    if model_name.startswith("qwen3.6-"):
        return "FreeInference"
    return f"{provider or 'unknown'} vendor not recorded"


def load_joined_calls(cell: Path) -> list[dict[str, Any]]:
    mpath = cell / "messages.jsonl"
    ppath = cell / "pi_events.jsonl"
    if not mpath.is_file() or not ppath.is_file():
        return []

    event_data: dict[str, dict[str, Any]] = {}
    for line in ppath.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = o.get("run_id")
        if not rid:
            continue
        entry = event_data.setdefault(rid, {})
        msg = o.get("message") or {}
        event_type = o.get("type")
        if event_type in {"message_start", "message_request_start"}:
            entry["start_ms"] = float(msg.get("timestamp") or o.get("wall_time_ms") or 0.0)
            entry["start_monotonic_ns"] = o.get("monotonic_ns")
        elif event_type == "message_response_headers":
            entry["headers_ms"] = float(msg.get("timestamp") or o.get("wall_time_ms") or 0.0)
        elif event_type == "message_first_token":
            entry["first_token_ms"] = float(msg.get("timestamp") or o.get("wall_time_ms") or 0.0)
        elif event_type == "message_last_token":
            entry["last_token_ms"] = float(msg.get("timestamp") or o.get("wall_time_ms") or 0.0)
        elif event_type == "message_end":
            u = msg.get("usage") or {}
            entry.update({
                "end_ms": float(msg.get("timestamp") or o.get("wall_time_ms") or 0.0),
                "end_monotonic_ns": o.get("monotonic_ns"),
                "input": int(u.get("input", 0) or 0),
                "output": int(u.get("output", 0) or 0),
                "cacheRead": int(u.get("cacheRead", 0) or 0),
                "cacheReadAvailable": u.get("cacheReadAvailable"),
                "cacheReadSource": u.get("cacheReadSource"),
                "phase": o.get("phase") or "(unknown)",
                "cache_key": o.get("cache_key"),
                "provider_request_id": o.get("provider_request_id"),
                "provider": o.get("provider"),
                "vendor": o.get("vendor"),
                "model": o.get("model"),
                "cache_config": o.get("cache_config") or {},
                "error": o.get("error"),
            })

    calls = []
    for line in mpath.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = o.get("run_id")
        u = event_data.get(rid, {})
        if not rid or not u or u.get("error"):
            continue
        start_ms = float(u.get("start_ms") or o.get("timestamp") or 0.0)
        end_ms = float(u.get("end_ms") or start_ms)
        model = o.get("model") or u.get("model") or "unknown"
        provider = o.get("provider") or u.get("provider") or "unknown"
        calls.append({
            "run_id": rid,
            "role": o.get("agent_role") or o.get("genomas_role") or "(unknown)",
            "phase": u.get("phase") or o.get("phase") or "(unknown)",
            "t": start_ms,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": max(end_ms - start_ms, 0.0),
            "headers_ms": u.get("headers_ms"),
            "first_token_ms": u.get("first_token_ms"),
            "last_token_ms": u.get("last_token_ms"),
            "provider": provider,
            "vendor": (
                o.get("vendor") or u.get("vendor")
                or _legacy_vendor(provider, model)
            ),
            "model": model,
            "cache_config": o.get("cache_config") or u.get("cache_config") or {},
            "messages": o.get("messages") or [],
            "input": int(u.get("input", 0) or 0),
            "output": int(u.get("output", 0) or 0),
            "cacheRead": int(u.get("cacheRead", 0) or 0),
            "cacheReadAvailable": u.get("cacheReadAvailable"),
            "cacheReadSource": u.get("cacheReadSource"),
            "cache_key": o.get("cache_key") or u.get("cache_key"),
            "provider_request_id": u.get("provider_request_id"),
        })
    calls.sort(key=lambda c: c["t"])
    return calls


def _insert_and_match(root: dict, tokens: list[int]) -> int:
    """Longest prefix of `tokens` already present in the trie, then insert it.

    Returns the matched prefix length (tokens). One pass: walk existing nodes
    counting the match, create nodes for the divergent tail.
    """
    node = root
    matched = 0
    still_matching = True
    for t in tokens:
        child = node.get(t)
        if child is None:
            child = {}
            node[t] = child
            still_matching = False
        elif still_matching:
            matched += 1
        node = child
    return matched


def _preview(render, lo: int, hi: int, head: int = 600, tail: int = 280) -> str:
    """Show the token range [lo, hi) as text, eliding the middle when long."""
    text = render(lo, hi) if hi > lo else ""
    if len(text) <= head + tail + 40:
        return text
    return f"{text[:head]}\n  …[{hi - lo} tokens reused total]…\n{text[-tail:]}"


def _lcp_len(left: list[int], right: list[int]) -> int:
    matched = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        matched += 1
    return matched


def _provider_aligned_tokens(
    tokens: int,
    input_tokens: int,
    our_tokens: int,
    match_unit: int,
) -> int:
    if tokens <= 0 or input_tokens <= 0 or our_tokens <= 0:
        return 0
    provider_estimate = tokens * input_tokens / our_tokens
    return int(provider_estimate // match_unit) * match_unit


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


SERVING_CONFIG = "vllm_cache_before_serving_config.json"


def read_serving_config(cell: Path) -> dict[str, Any]:
    """The server's own cache_config_info, captured beside the counters.

    Without it the geometry falls back to LEGACY_MATCH_UNIT, and a knob sweep
    cannot say which configuration produced which number.
    """
    path = cell / SERVING_CONFIG
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _cache_geometry(
    calls: list[dict[str, Any]],
    block_size: int | None,
    prefix_match_unit: int | None,
    server_config: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Resolve physical storage and prefix-match units from the arm manifest."""
    observed: set[tuple[int | None, int | None]] = set()
    for call in calls:
        serving = (call.get("cache_config") or {}).get("serving") or {}
        candidate = (
            serving.get("resolved")
            or serving.get("observed_config")
            or serving.get("observed")
            or serving
        )
        if isinstance(candidate, dict):
            geometry = (
                _positive_int(candidate.get("block_size")),
                _positive_int(candidate.get("prefix_match_unit")),
            )
            if any(value is not None for value in geometry):
                observed.add(geometry)
    if len(observed) > 1:
        raise ValueError(
            "cell contains multiple cache geometries: "
            f"{sorted(observed, key=repr)}"
        )
    observed_block, observed_match = next(iter(observed), (None, None))
    # `serving` above is rebound inside the loop; the captured server config
    # arrives under its own name so the two cannot collide.
    server_config = server_config or {}
    physical = (
        _positive_int(block_size)
        or observed_block
        or _positive_int(server_config.get("block_size"))
        or LEGACY_MATCH_UNIT
    )
    matching = (
        _positive_int(prefix_match_unit)
        or observed_match
        or _positive_int(server_config.get("prefix_match_unit"))
        or physical
    )
    return physical, matching


def _cache_block_references(tokens: list[int], block_size: int) -> list[str]:
    """Return stable identities for every complete prefix-cache block."""
    references = []
    parent = b""
    for offset in range(0, len(tokens) - block_size + 1, block_size):
        block = tokens[offset:offset + block_size]
        payload = json.dumps(block, separators=(",", ":")).encode("ascii")
        parent = hashlib.sha256(parent + b":" + payload).digest()
        references.append(parent.hex())
    return references


def _reuse_distance_summary(references: list[str]) -> dict[str, Any]:
    """Compute exact LRU stack distance with a Fenwick tree."""
    size = len(references)
    tree = [0] * (size + 1)

    def add(index: int, delta: int) -> None:
        index += 1
        while index <= size:
            tree[index] += delta
            index += index & -index

    def prefix_sum(index: int) -> int:
        total = 0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    latest: dict[str, int] = {}
    histogram: dict[int, int] = {}
    cold = 0
    for index, key in enumerate(references):
        previous = latest.get(key)
        if previous is None:
            cold += 1
        else:
            distance = prefix_sum(index) - prefix_sum(previous + 1)
            histogram[distance] = histogram.get(distance, 0) + 1
            add(previous, -1)
        add(index, 1)
        latest[key] = index
    return {
        "cumulative_unique_blocks": len(latest),
        "block_references": size,
        "cold_references": cold,
        "reference_order": "request_start_then_prefix",
        "reuse_distance_blocks": {
            str(distance): count for distance, count in sorted(histogram.items())
        },
        "peak_resident_blocks": {
            "policy": "unbounded",
            "blocks": len(latest),
        },
    }


def _candidate_class(call: dict[str, Any]) -> str:
    if call["exact_hash_source_indices"]:
        return "exact repeat"
    count = call["source_candidate_count"]
    if count == 0:
        return "no candidate"
    if count == 1:
        return "unique"
    if count == 2:
        return "2 sources"
    return "3+ sources"


def _reuse_band(call: dict[str, Any]) -> str | None:
    if call["input"] <= 0:
        return None
    realized = call["cacheRead"] / call["input"]
    if realized > 0.70:
        return "realized >70%"
    if realized > 0:
        return "realized 1–70%"
    if call["logical_aligned"] > 0:
        return "realized 0%, logical >0"
    return None


def _temporal_metrics(per_call: list[dict[str, Any]]) -> dict[str, Any]:
    reusable = [
        call for call in per_call
        if call["logical_aligned"] > 0
        and call["newest_possible_source_age_s"] is not None
    ]
    uniquely_sourced = [
        call for call in reusable if call["source_candidate_count"] == 1
    ]
    high_reuse = [
        call for call in reusable
        if call["input"] > 0 and call["cacheRead"] / call["input"] > 0.70
    ]
    binned = []
    for label, low, high in [
        ("<5s", 0.0, 5.0),
        ("5–15s", 5.0, 15.0),
        ("15–30s", 15.0, 30.0),
        ("30–60s", 30.0, 60.0),
        (">60s", 60.0, math.inf),
    ]:
        group = [
            call for call in reusable
            if low <= call["newest_possible_source_age_s"] < high
        ]
        logical_tokens = sum(call["logical_aligned"] for call in group)
        realized_tokens = sum(call["cacheRead"] for call in group)
        binned.append({
            "age": label,
            "n": len(group),
            "logical_reusable_tokens": logical_tokens,
            "realized_cache_read_tokens": realized_tokens,
        })
    return {
        "longest_unique_source_age_s": (
            round(max(
                call["newest_possible_source_age_s"]
                for call in uniquely_sourced
            ), 4)
            if uniquely_sourced else None
        ),
        "median_age_above_70pct_reuse_s": (
            round(statistics.median(
                call["newest_possible_source_age_s"]
                for call in high_reuse
            ), 4)
            if high_reuse else None
        ),
        "age_bins": binned,
    }


def _candidate_count_table(per_call: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label in ["0", "1", "2", "3+"]:
        group = [
            call for call in per_call
            if (
                str(call["source_candidate_count"])
                if call["source_candidate_count"] < 3 else "3+"
            ) == label
        ]
        rows.append({
            "possible_sources": label,
            "calls": len(group),
            "call_fraction": round(len(group) / len(per_call), 4) if per_call else None,
        })
    return rows


def read_server_prefix_cache(cell: Path, total_input_tokens: int | None = None) -> dict[str, Any] | None:
    """Server-measured realized reuse, for backends that do not report it.

    Prefer usage.prompt_tokens_details.cached_tokens whenever the response
    includes it. Some vLLM runs omit that field, so the trace scripts also
    snapshot the server's Prometheus counters around each cell. Their delta is
    the cell-level realized reuse in tokens when the interval is attributable.

    This never overrides a vendor-reported number -- callers use it only when
    cacheRead is absent.  When ``total_input_tokens`` is given it is compared
    against the queries delta: the two match exactly when this run was the
    server's only client, and a mismatch marks the reading unattributable.
    """
    path = cell / "vllm_cache_realized.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not data.get("queries"):
        return None
    queries = int(data["queries"])
    result = {
        "source": data.get("source", "vllm_prometheus_prefix_cache"),
        # Snapshots taken before the unit was verified are labelled "blocks".
        # They are not: the queries delta equals the cell's input tokens
        # exactly, which is only possible if the counter counts tokens.  Trust
        # the measurement over the stale label rather than propagating it.
        "unit": "tokens",
        "recorded_unit_label": data.get("unit"),
        "queries": queries,
        "hits": int(data.get("hits") or 0),
        "hit_rate": data.get("hit_rate"),
        "sole_client": None,
    }
    if total_input_tokens:
        result["sole_client"] = queries == int(total_input_tokens)
        result["queries_vs_input_ratio"] = round(queries / int(total_input_tokens), 4)
    return result


def _is_nocache_arm(cell: Path) -> bool:
    """True when this cell deliberately defeated the prefix cache."""
    path = cell / "messages.jsonl"
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            if json.loads(line).get("nocache_tag"):
                return True
        except json.JSONDecodeError:
            continue
    return False


def realized_provenance(
    cell: Path, per_call: list[dict[str, Any]], server_cache: dict | None,
) -> str:
    """vendor | server | nocache_arm | unmeasured.

    "unmeasured" must not render as 0%: unknown reuse and zero reuse are
    different findings.
    """
    explicit = [
        call.get("cacheReadAvailable") for call in per_call
        if call.get("cacheReadAvailable") is not None
    ]
    if explicit and len(explicit) == len(per_call) and all(explicit):
        return "response"
    if not explicit and any((call.get("cacheRead") or 0) > 0 for call in per_call):
        # Historical traces predate the availability marker.
        return "vendor"
    if server_cache and server_cache.get("hit_rate") is not None:
        return "server"
    if _is_nocache_arm(cell):
        return "nocache_arm"
    return "unmeasured"


def analyze_cell_logical(
    cell: Path,
    dump_prefixes: bool = False,
    block_size: int | None = None,
    prefix_match_unit: int | None = None,
) -> dict[str, Any] | None:
    calls = load_joined_calls(cell)
    if not calls:
        return None

    serving = read_serving_config(cell)
    physical_block_size, match_unit = _cache_geometry(
        calls,
        block_size,
        prefix_match_unit,
        serving,
    )
    enc = _encoding_for(calls[0]["model"])
    # When the serving engine tokenized these prompts for us, its counts are
    # the truth: they include the tools schema and price images as images.
    # Missing entries fall back to tiktoken per call, so a partially built
    # cache degrades one call at a time instead of failing the cell.
    server_tokens = _server_tokens.load(cell)
    server_token_strs = _server_tokens.load_token_strs(cell)
    n_from_server = 0
    prefix_lines: list[str] = []
    role_totals: dict[str, int] = {}
    role_global_calls: dict[str, list[int]] = {}
    for global_call, call in enumerate(calls):
        role = call["role"]
        role_totals[role] = role_totals.get(role, 0) + 1
        role_global_calls.setdefault(role, []).append(global_call)
    role_call_counts: dict[str, int] = {}

    global_trie: dict = {}
    role_tries: dict[str, dict] = {}

    sum_input = sum_cacheread = 0
    sum_our_tokens = 0
    sum_logical = sum_logical_aligned = 0
    sum_role_logical = 0
    per_call = []
    token_history: list[list[int]] = []
    block_references: list[str] = []
    first_start_ms = calls[0]["start_ms"]

    for call_index, c in enumerate(calls):
        tokens = server_tokens.get(c["run_id"])
        from_server = tokens is not None
        if from_server:
            n_from_server += 1
        else:
            tokens = enc.encode(_serialize(c["messages"]), disallowed_special=())
        n = len(tokens)

        # Render a token range as the text that produced it.  Server ids belong
        # to the engine's vocabulary, so tiktoken cannot decode them; the cache
        # carries their surface forms instead, which also keeps the prefix dump
        # working when the report is regenerated with no tunnel up.
        strs = server_token_strs.get(c["run_id"]) if from_server else None
        if strs is not None and len(strs) == n:
            def render(lo: int, hi: int, _s=strs) -> str:
                return _server_tokens.decode_token_strs(_s[lo:hi])
        elif from_server:
            def render(lo: int, hi: int) -> str:
                return f"<{hi - lo} tokens; rebuild server_tokens.json to see the text>"
        else:
            def render(lo: int, hi: int, _t=tokens) -> str:
                return enc.decode(_t[lo:hi])

        trie_match = _insert_and_match(global_trie, tokens)
        rt = role_tries.setdefault(c["role"], {})
        r = _insert_and_match(rt, tokens)  # per-role (subset of global)
        lcp_lengths = [_lcp_len(tokens, previous) for previous in token_history]
        g = max(lcp_lengths, default=0)
        if g != trie_match:
            raise AssertionError(f"prefix trie mismatch at call {call_index}: {trie_match} != {g}")
        logical_match_aligned = (g // match_unit) * match_unit
        logical_candidates = [
            i for i, matched in enumerate(lcp_lengths) if g > 0 and matched == g
        ]
        realized_candidates = [
            i
            for i, matched in enumerate(lcp_lengths)
            if c["cacheRead"] > 0
            and _provider_aligned_tokens(
                matched,
                c["input"],
                n,
                match_unit,
            ) >= c["cacheRead"]
        ]
        exact_hash_sources = [
            i
            for i, previous in enumerate(calls[:call_index])
            if c["cache_key"] is not None and previous["cache_key"] == c["cache_key"]
        ]
        opportunity_candidates = realized_candidates if c["cacheRead"] > 0 else logical_candidates
        latest_candidate = max(opportunity_candidates, default=None)
        earliest_candidate = min(opportunity_candidates, default=None)
        completed_candidates = [
            i for i in opportunity_candidates
            if calls[i]["end_ms"] <= c["start_ms"]
        ]
        latest_completed_candidate = (
            max(completed_candidates, key=lambda i: calls[i]["end_ms"])
            if completed_candidates else None
        )
        candidate_ages = [
            max((c["start_ms"] - calls[i]["start_ms"]) / 1000.0, 0.0)
            for i in opportunity_candidates
        ]
        latest_start_age = (
            max((c["start_ms"] - calls[latest_candidate]["start_ms"]) / 1000.0, 0.0)
            if latest_candidate is not None else None
        )
        latest_completion_gap = (
            (c["start_ms"] - calls[latest_completed_candidate]["end_ms"]) / 1000.0
            if latest_completed_candidate is not None else None
        )
        logical_aligned = _provider_aligned_tokens(
            g,
            c["input"],
            n,
            match_unit,
        )
        if dump_prefixes:
            role_call_counts[c["role"]] = role_call_counts.get(c["role"], 0) + 1
            new_head = render(g, min(g + 60, n)) if g < n else ""
            prefix_lines.append(
                f"===== global_call={len(per_call)}  role={c['role']}  "
                f"role_call={role_call_counts[c['role']]}/{role_totals[c['role']]}  "
                f"reused={g} tok  new={n - g} tok  "
                f"logical_sources={logical_candidates}  "
                f"realized_sources={realized_candidates}  "
                f"exact_hash_sources={exact_hash_sources}  "
                f"latest_source_age_s={latest_start_age}  "
                f"newest_possible_source_age_s={latest_completion_gap} =====\n"
                f"--- REUSED PREFIX (served from cache) ---\n{_preview(render, 0, g)}\n"
                f"--- DIVERGES HERE → NEW TAIL BEGINS ---\n{new_head!r}\n"
            )

        sum_input += c["input"]
        sum_cacheread += c["cacheRead"]
        sum_our_tokens += n
        sum_logical += g
        sum_logical_aligned += logical_match_aligned
        sum_role_logical += r
        call_record = {
            "run_id": c["run_id"], "role": c["role"], "our_tokens": n,
            "phase": c["phase"],
            "input": c["input"], "cacheRead": c["cacheRead"],
            "cacheReadAvailable": c.get("cacheReadAvailable"),
            "cacheReadSource": c.get("cacheReadSource"),
            "output": c["output"],
            "logical": g, "role_logical": r,
            "logical_aligned": logical_aligned,
            "start_ms": c["start_ms"],
            "end_ms": c["end_ms"],
            "elapsed_s": round((c["start_ms"] - first_start_ms) / 1000.0, 4),
            "duration_ms": round(c["duration_ms"], 4),
            "fresh_input": max(c["input"] - c["cacheRead"], 0),
            "headers_ms": c["headers_ms"],
            "first_token_ms": c["first_token_ms"],
            "last_token_ms": c["last_token_ms"],
            "logical_source_candidates": logical_candidates,
            "realized_source_candidates": realized_candidates,
            "exact_hash_source_indices": exact_hash_sources,
            "source_candidate_count": len(opportunity_candidates),
            "latest_candidate_index": latest_candidate,
            "earliest_candidate_index": earliest_candidate,
            "latest_completed_candidate_index": latest_completed_candidate,
            "latest_candidate_start_age_s": (
                round(latest_start_age, 4) if latest_start_age is not None else None
            ),
            "latest_candidate_completion_gap_s": (
                round(latest_completion_gap, 4)
                if latest_completion_gap is not None else None
            ),
            "newest_possible_source_age_s": (
                round(latest_completion_gap, 4)
                if latest_completion_gap is not None else None
            ),
            "candidate_age_min_s": round(min(candidate_ages), 4) if candidate_ages else None,
            "candidate_age_max_s": round(max(candidate_ages), 4) if candidate_ages else None,
        }
        call_record["source_class"] = _candidate_class(call_record)
        call_record["reuse_band"] = _reuse_band(call_record)
        per_call.append(call_record)
        token_history.append(tokens)
        block_references.extend(_cache_block_references(tokens, physical_block_size))

    denom = sum_our_tokens or 1
    in_denom = sum_input or 1

    summary = {
        "cell": cell.name,
        "tokenizer": (
            f"{_server_tokens.SOURCE}+{enc.name}"
            if 0 < n_from_server < len(calls)
            else _server_tokens.SOURCE if n_from_server
            else enc.name
        ),
        # Which numbers came from the engine's own tokenizer.  A reader must be
        # able to tell a corrected cell from an uncorrected one without
        # guessing from the workflow name.
        "token_source": {
            "server_tokenized_calls": n_from_server,
            "tiktoken_calls": len(calls) - n_from_server,
            **_server_tokens.load_meta(cell),
        },
        "n_calls": len(calls),
        "our_input_tokens": sum_our_tokens,
        "openai_input_tokens": sum_input,
        "tok_delta_pct": round(100 * (sum_our_tokens - sum_input) / in_denom, 1),
        # fractions (the comparable numbers)
        "realized_frac": round(sum_cacheread / in_denom, 4),
        "logical_frac": round(sum_logical / denom, 4),
        "logical_aligned_frac": round(sum_logical_aligned / denom, 4),
        "logical_128_frac": round(sum_logical_aligned / denom, 4),
        "gap_frac": round(sum_logical_aligned / denom - sum_cacheread / in_denom, 4),
        "cross_agent_bonus_frac": round((sum_logical - sum_role_logical) / denom, 4),
        # absolute token totals (context)
        "cacheread_tokens": sum_cacheread,
        "logical_tokens": sum_logical,
        "logical_aligned_tokens": sum_logical_aligned,
        "cache_geometry": {
            "block_size": physical_block_size,
            "prefix_match_unit": match_unit,
            "source": "server" if serving else "trace/legacy",
        },
        # The server's own cache_config_info at cell start: which knobs this
        # cell actually ran under.
        "serving_config": serving,
        "block_demand": _reuse_distance_summary(block_references),
        "source_class_counts": {
            source_class: sum(1 for c in per_call if c["source_class"] == source_class)
            for source_class in [
                "unique", "2 sources", "3+ sources",
                "no candidate", "exact repeat",
            ]
        },
        "temporal_metrics": _temporal_metrics(per_call),
        "candidate_count_table": _candidate_count_table(per_call),
        "runtime": {
            "vendors": sorted({str(c["vendor"]) for c in calls}),
            "providers": sorted({str(c["provider"]) for c in calls}),
            "models": sorted({str(c["model"]) for c in calls}),
            "cache_configs": [
                json.loads(value)
                for value in sorted({
                    json.dumps(c["cache_config"], sort_keys=True, default=str)
                    for c in calls
                })
            ],
        },
        "has_stream_timing": any(c["first_token_ms"] is not None for c in per_call),
        "per_call": per_call,
    }
    if dump_prefixes:
        index_lines = ["===== AGENT CALL INDEX ====="]
        for role, global_calls in role_global_calls.items():
            calls_str = ", ".join(str(i) for i in global_calls)
            index_lines.append(
                f"{role}: {len(global_calls)} calls | global_calls=[{calls_str}]"
            )
        index_lines.append("===== CHRONOLOGICAL CALL DETAILS =====")
        (cell / PREFIX_DUMP).write_text(
            "\n".join(index_lines + prefix_lines),
            encoding="utf-8",
        )

    # sanity: an ideal cache cannot serve less than the real one
    if summary["logical_frac"] + 0.02 < summary["realized_frac"]:
        summary["WARNING"] = (
            f"logical {summary['logical_frac']:.2%} < realized "
            f"{summary['realized_frac']:.2%} — tokenizer likely misaligned "
            f"(tok_delta {summary['tok_delta_pct']}%)"
        )
    return summary


def _role_colors(roles: list[str]) -> dict[str, str]:
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    return {role: palette[i % len(palette)] for i, role in enumerate(roles)}


def _lineage_parent_index(call: dict[str, Any]) -> int | None:
    """Return the latest source sharing the call's longest logical prefix.

    Only calls whose shared prefix survives block alignment get an edge.  A
    prompt can share two or twenty leading tokens with an earlier one -- the
    opening words of a system prompt -- and that is a textual coincidence, not
    reuse: a KV cache matches whole blocks, so anything under one block yields
    nothing.  Drawing those edges filled the graph with links that no cache
    could ever serve.  ``logical_aligned`` is exactly the post-alignment
    figure, so requiring it to be positive keeps the lineage to reuse that is
    physically realizable.
    """
    if int(call.get("logical_aligned", 0) or 0) <= 0:
        return None
    candidates = call.get("logical_source_candidates") or []
    return max(candidates) if candidates else None


def _logical_reuse_fraction(call: dict[str, Any]) -> float:
    """Return aligned logical prefix reuse divided by input tokens."""
    input_tokens = int(call.get("input", 0) or 0)
    if input_tokens <= 0:
        return 0.0
    logical_tokens = int(call.get("logical_aligned", 0) or 0)
    return min(max(logical_tokens / input_tokens, 0.0), 1.0)


def _lineage_marker_size(reuse_fraction: float) -> float:
    """Map a logical reuse fraction to a Matplotlib scatter area."""
    return 90.0 + 320.0 * min(max(reuse_fraction, 0.0), 1.0)


def plot_prefix_lineage(summary: dict, out_png: Path) -> None:
    """Plot logical longest-prefix lineage across all reusable calls."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D

    pc = summary["per_call"]
    parents = {}
    for i, call in enumerate(pc):
        parent = _lineage_parent_index(call)
        if parent is not None:
            parents[i] = parent
    visible = set(parents)
    visible.update(parents.values())
    roles = list(dict.fromkeys(pc[i]["role"] for i in sorted(visible)))
    colors = _role_colors(roles)
    fig, ax = plt.subplots(figsize=(14, 7.2))

    children: dict[int, list[int]] = {i: [] for i in visible}
    for child, parent in parents.items():
        children[parent].append(child)

    roots = [
        node for node in sorted(visible)
        if node not in parents
    ]
    y_positions: dict[int, float] = {}
    next_leaf = 0.0

    def place(node: int) -> float:
        nonlocal next_leaf
        if not children[node]:
            y_positions[node] = next_leaf
            next_leaf += 1.0
        else:
            child_y = [place(child) for child in children[node]]
            y_positions[node] = sum(child_y) / len(child_y)
        return y_positions[node]

    for root in roots:
        place(root)
        next_leaf += 0.8

    for child, parent in parents.items():
        ax.plot(
            [parent, child],
            [y_positions[parent], y_positions[child]],
            color="#8C8C8C", lw=1.15, alpha=0.7, zorder=1,
        )

    if visible:
        ordered = sorted(visible)
        sizes = {
            i: _lineage_marker_size(_logical_reuse_fraction(pc[i]))
            for i in ordered
        }
        ax.scatter(
            ordered,
            [y_positions[i] for i in ordered],
            s=[sizes[i] for i in ordered],
            c=[colors[pc[i]["role"]] for i in ordered],
            edgecolors="#555555",
            linewidths=1.2,
            zorder=3,
        )
    else:
        sizes = {}
        ax.text(
            0.5, 0.5, "No logical prefix reuse",
            transform=ax.transAxes, ha="center", va="center",
            color="#555555", fontsize=12,
        )

    for i in sorted(visible):
        call = pc[i]
        ax.text(
            i, y_positions[i], str(i),
            color="white", fontsize=6.5, fontweight="bold",
            ha="center", va="center", zorder=4,
        )
        extra = max(len(call.get("logical_source_candidates") or []) - 1, 0)
        if extra > 0:
            ax.annotate(
                f"+{extra}", (i, y_positions[i]),
                xytext=(0, 9), textcoords="offset points",
                fontsize=6.5, ha="center", color="#333333",
            )
        if call["exact_hash_source_indices"]:
            ax.scatter(
                [i], [y_positions[i]], s=sizes[i] + 75,
                facecolors="none", edgecolors="#111111",
                linewidths=1.1, zorder=2,
            )

    role_legend = [
        Line2D(
            [0], [0], marker="o", color="none", label=role,
            markerfacecolor=colors[role], markeredgecolor="#555555",
            markersize=8,
        )
        for role in roles
    ]
    if role_legend:
        role_artist = ax.legend(
            handles=role_legend,
            title="agent role", fontsize=8, ncol=1, loc="upper left",
            bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
        )
        ax.add_artist(role_artist)
    size_legend = [
        Line2D(
            [0], [0], marker="o", color="none", label=f"{fraction:.2%}",
            markerfacecolor="#A7A7A7", markeredgecolor="#555555",
            markersize=_lineage_marker_size(fraction) ** 0.5,
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    ax.legend(
        handles=size_legend,
        title="logical reuse / input",
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0,
    )
    ax.set_xlim(-1, max(len(pc), 1))
    ax.set_yticks([])
    ax.set_xlabel("global LLM call index")
    # a call index is a count, so fractional ticks are meaningless
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("logical prefix branches")
    ax.set_title("Logical KV-cache lineage")
    ax.grid(True, axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _per_call_reuse(summary: dict) -> tuple[list[int], list[float], list[float]]:
    """Return realized and aligned logical reuse fractions for every call."""
    per_call = summary["per_call"]
    x = list(range(len(per_call)))
    realized = [
        call["cacheRead"] / call["input"] if call["input"] else 0.0
        for call in per_call
    ]
    logical = [
        call["logical_aligned"] / call["input"] if call["input"] else 0.0
        for call in per_call
    ]
    return x, realized, logical


def plot_cache_warming(summary: dict, out_png: Path) -> None:
    """Plot realized and aligned logical reuse for every LLM call."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    x, realized, logical = _per_call_reuse(summary)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(
        x,
        realized,
        logical,
        color="#999999",
        alpha=0.22,
        label="reuse gap",
    )
    ax.plot(
        x,
        realized,
        color="#E69F00",
        lw=1.8,
        marker="o",
        markersize=2.8,
        label="realized (cacheRead / input)",
    )
    ax.plot(
        x,
        logical,
        color="#0072B2",
        lw=1.8,
        marker="o",
        markersize=2.8,
        label="logical aligned / input",
    )
    server = summary.get("server_prefix_cache")
    if server and server.get("hit_rate") is not None:
        # The server exposes one counter for the whole cell, not per call, so
        # this is a level rather than a curve.  Same unit and denominator as
        # the orange line -- tokens served over tokens submitted -- so the two
        # are directly comparable; it is dashed because it cannot be resolved
        # per call.
        label = f"server-measured realized ({server['hit_rate']:.2%}, cell-level)"
        if server.get("sole_client") is False:
            label += " — server had other clients"
        ax.axhline(
            server["hit_rate"], color="#D55E00", lw=1.6, ls="--", label=label,
        )
    ax.set_xlabel("LLM call index (chronological)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("per-call reuse fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Per-call prefix reuse")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
