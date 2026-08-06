#!/usr/bin/env python3
"""Contents of an unbounded (never-evicting) logical KV cache, segment by segment.

logical.py answers "how much of each call's prompt could an ideal cache have
served". This module answers the complementary question: *what is actually
sitting in that ideal cache*.

Every call's prompt is inserted into one global radix trie over tokens, in
chronological order. Each trie edge is a SEGMENT: a maximal run of tokens that
is shared by the same set of calls, cut only where some call diverged. Segments
therefore never overlap, and

    sum(segment.tokens) == how big a never-evicting prefix cache has to be

The per-segment facts we care about:

  first_call    the call that created the segment (its text entered the cache here)
  n_calls       how many calls' prompts traverse it
  reuse_tokens  tokens * (n_calls - 1), i.e. tokens this segment served from cache
  content_tag   what the text in the segment is: system prompt, history dialog,
                code, or raw data

Every segment carries a content_tag, so the cache can be costed by what is in it
rather than by which prompt produced it. The headline that falls out is the
cache space each kind of content occupies against the hits it actually returns.

    PYTHONPATH=src python3 -m agent_io_tracing.analysis.kvcache.segments <cell_dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import tiktoken

from agent_io_tracing.analysis.kvcache.logical import (
    _encoding_for, _serialize, load_joined_calls,
)

SEGMENTS_CSV = "kvcache_segments.csv"
SEGMENTS_MD = "kvcache_segments.md"
SEGMENTS_JSON = "kvcache_segments.json"

# --------------------------------------------------------------------------
# radix trie over token sequences
# --------------------------------------------------------------------------

class _Node:
    __slots__ = ("edge", "children", "calls", "first_call", "terminal_calls")

    def __init__(self, edge: list[int], first_call: int | None) -> None:
        self.edge = edge
        self.children: dict[int, _Node] = {}
        self.calls: set[int] = set()
        self.terminal_calls: set[int] = set()
        self.first_call = first_call


def _lcp(a: list[int], b: list[int], b_offset: int) -> int:
    limit = min(len(a), len(b) - b_offset)
    i = 0
    while i < limit and a[i] == b[b_offset + i]:
        i += 1
    return i


def build_radix_trie(token_sequences: list[list[int]]) -> _Node:
    """Insert every call's tokens in chronological order; return the root."""
    root = _Node([], None)
    for index, tokens in enumerate(token_sequences):
        node = root
        node.calls.add(index)
        pos = 0
        while True:
            if pos == len(tokens):
                node.terminal_calls.add(index)
                break
            child = node.children.get(tokens[pos])
            if child is None:
                leaf = _Node(tokens[pos:], index)
                leaf.calls.add(index)
                leaf.terminal_calls.add(index)
                node.children[tokens[pos]] = leaf
                break
            shared = _lcp(child.edge, tokens, pos)
            if shared == len(child.edge):
                pos += shared
                node = child
                node.calls.add(index)
                continue
            # the new call diverges inside this edge: split it
            middle = _Node(child.edge[:shared], child.first_call)
            middle.calls = set(child.calls)
            middle.calls.add(index)
            child.edge = child.edge[shared:]
            middle.children[child.edge[0]] = child
            node.children[tokens[pos]] = middle
            pos += shared
            if pos == len(tokens):
                middle.terminal_calls.add(index)
            else:
                leaf = _Node(tokens[pos:], index)
                leaf.calls.add(index)
                leaf.terminal_calls.add(index)
                middle.children[tokens[pos]] = leaf
            break
    return root


# --------------------------------------------------------------------------
# what the text in a segment actually is
# --------------------------------------------------------------------------

# The four kinds of text a prompt is made of, ordered most-volatile first. A
# segment takes the tag of the most volatile thing it contains, because that is
# what decides whether it can ever be reused: one execution output landing in an
# otherwise fixed block makes the whole block unrepeatable. So "raw data" means
# "a block containing execution output", not "pure numbers", and "system prompt"
# means the fixed preamble — system message, task guidelines, tool and
# environment docs, response-format instructions — with nothing volatile in it.
#
# The markers are GenoMAS prompt conventions. On a corpus that does not use them
# everything lands in "boilerplate", which is why the summary also reports the
# share that matched no marker at all — if that is ~100%, this cut says nothing.
CONTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("raw data", re.compile(r"\[Output\]:|Execution Output")),
    ("code", re.compile(r"\[Code\]:|```python")),
    ("history dialog", re.compile(r"STEP \d|Task History")),
]
DEFAULT_CONTENT_TAG = "system prompt"
CONTENT_TAGS = [name for name, _ in CONTENT_PATTERNS] + [DEFAULT_CONTENT_TAG]

# Literal measurement values: floats with real precision, GEO sample ids, Affy
# probe ids. Deliberately narrow — a step number or a year must not count.
RAW_DATA_TOKENS = re.compile(
    r"-?\d+\.\d{3,}|GSM\d{5,}|'\d+_(?:at|s_at|x_at|a_at)'"
)


def classify_content(text: str) -> tuple[str, float]:
    """Return the segment's most volatile content type and its raw-data share."""
    tag = DEFAULT_CONTENT_TAG
    for name, pattern in CONTENT_PATTERNS:
        if pattern.search(text):
            tag = name
            break
    matched = sum(len(m.group()) for m in RAW_DATA_TOKENS.finditer(text))
    return tag, round(matched / len(text), 4) if text else 0.0


def _content_breakdown(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Resident cost against realized benefit, per content type."""
    cache_total = sum(s["tokens"] for s in segments) or 1
    realized_total = sum(s["realized_tokens"] for s in segments) or 1
    by_tag = {}
    for tag in CONTENT_TAGS:
        group = [s for s in segments if s["content_tag"] == tag]
        occupied = sum(s["tokens"] for s in group)
        realized = sum(s["realized_tokens"] for s in group)
        by_tag[tag] = {
            "n_segments": len(group),
            "cache_size_tokens": occupied,
            "cache_share": round(occupied / cache_total, 4),
            "realized_tokens": realized,
            "realized_share": round(realized / realized_total, 4),
            # >1 means this content pays back more than the space it occupies
            "leverage": (
                round((realized / realized_total) / (occupied / cache_total), 2)
                if occupied else None
            ),
        }
    return {
        "by_tag": by_tag,
        "unmatched_share": by_tag[DEFAULT_CONTENT_TAG]["cache_share"],
    }


def _excerpt(enc: tiktoken.Encoding, tokens: list[int], head: int = 160, tail: int = 90) -> str:
    text = enc.decode(tokens) if tokens else ""
    text = text.replace("\n", "\\n")
    if len(text) <= head + tail + 20:
        return text
    return f"{text[:head]} … {text[-tail:]}"


def collect_segments(
    root: _Node,
    enc: tiktoken.Encoding,
    roles: list[str],
) -> list[dict[str, Any]]:
    """Flatten the trie into one row per edge, ordered by creation time."""
    rows: list[dict[str, Any]] = []

    def walk(node: _Node, depth: int, path: str) -> None:
        for key in sorted(node.children):
            child = node.children[key]
            call_indices = sorted(child.calls)
            tokens = len(child.edge)
            text = enc.decode(child.edge) if child.edge else ""
            content_tag, raw_data_share = classify_content(text)
            rows.append({
                "content_tag": content_tag,
                "raw_data_share": raw_data_share,
                "path": f"{path}/{key}",
                "content_sha": hashlib.sha1(
                    json.dumps(child.edge, separators=(",", ":")).encode("ascii")
                ).hexdigest()[:12],
                "start_offset": depth,
                "tokens": tokens,
                "first_call": child.first_call,
                "last_call": max(call_indices),
                "n_calls": len(call_indices),
                "reuse_tokens": tokens * (len(call_indices) - 1),
                "roles": sorted({roles[i] for i in call_indices}),
                "call_indices": call_indices,
                "is_prompt_end": bool(child.terminal_calls),
                "excerpt": _excerpt(enc, child.edge),
            })
            walk(child, depth + tokens, f"{path}/{key}")

    walk(root, 0, "")
    rows.sort(key=lambda r: (r["first_call"], r["start_offset"]))
    for rank, row in enumerate(rows):
        row["segment"] = f"S{rank}"
    return rows


# --------------------------------------------------------------------------
# analysis entry point
# --------------------------------------------------------------------------

def attach_realized(
    segments: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    token_sequences: list[list[int]],
) -> dict[str, Any]:
    """Mark, per segment, which traversals the real cache actually served.

    ``cached_tokens`` is a prefix length, so a segment was really served to call
    i exactly when it lies inside that call's first ``cached_tokens`` tokens. The
    provider counts in its own tokenizer and we count in ours, so the offset is
    rescaled by this call's token ratio; that rescaling is the accuracy limit of
    the comparison, and ``attribution_residual`` measures what it costs.
    """
    cut = []
    for index, call in enumerate(calls):
        provider_input = call["input"]
        our_tokens = len(token_sequences[index])
        if provider_input <= 0 or our_tokens <= 0:
            cut.append(0.0)
            continue
        cut.append(call["cacheRead"] * our_tokens / provider_input)

    for s in segments:
        start = s["start_offset"]
        # the creating call prefilled this text, so only later traversals can be
        # cache hits — the same convention reuse_tokens uses
        reusers = s["call_indices"][1:]
        # a realized prefix can stop in the middle of a segment, so count the
        # covered part rather than requiring the whole segment to fit
        covered = {
            i: max(0.0, min(cut[i] - start, float(s["tokens"])))
            for i in reusers
        }
        s["realized_calls"] = [i for i, v in covered.items() if v > 0]
        s["realized_hits"] = len(s["realized_calls"])
        s["realized_tokens"] = round(sum(covered.values()))
        s["gap_tokens"] = max(s["reuse_tokens"] - s["realized_tokens"], 0)

    logical_reuse = sum(s["reuse_tokens"] for s in segments)
    realized = sum(s["realized_tokens"] for s in segments)
    scale_deltas = [
        abs(len(token_sequences[i]) - c["input"]) / c["input"]
        for i, c in enumerate(calls) if c["input"] > 0
    ]
    return {
        "logical_reuse_tokens": logical_reuse,
        "realized_reuse_tokens": realized,
        "gap_tokens": max(logical_reuse - realized, 0),
        "capture_rate": round(realized / logical_reuse, 4) if logical_reuse else None,
        # every call's realized prefix is partitioned by the segments on its
        # path, so the attribution must add back up to the reported hits
        "reported_hit_tokens_rescaled": round(sum(cut)),
        "attribution_residual": round(sum(cut)) - realized,
        "tokenizer_scale_error_pct": (
            round(100 * sum(scale_deltas) / len(scale_deltas), 2) if scale_deltas else None
        ),
    }


def analyze_cell_segments(cell: Path) -> dict[str, Any]:
    calls = load_joined_calls(cell)
    if not calls:
        return {"cell": cell.name, "n_calls": 0, "segments": []}

    enc = _encoding_for(calls[0]["model"])
    token_sequences = [
        enc.encode(_serialize(c["messages"]), disallowed_special=())
        for c in calls
    ]
    roles = [c["role"] for c in calls]

    root = build_radix_trie(token_sequences)
    segments = collect_segments(root, enc, roles)

    realized = attach_realized(segments, calls, token_sequences)
    resident = sum(s["tokens"] for s in segments)
    written_total = sum(len(t) for t in token_sequences)

    return {
        "cell": cell.name,
        "tokenizer": enc.name,
        "n_calls": len(calls),
        "prompt_tokens_total": written_total,
        "cache_size_tokens": resident,
        "resend_ratio": round(written_total / resident, 3) if resident else None,
        "n_segments": len(segments),
        "content_breakdown": _content_breakdown(segments),
        "realized_vs_logical": realized,
        "vendors": sorted({str(c["vendor"]) for c in calls}),
        "models": sorted({str(c["model"]) for c in calls}),
        "segments": segments,
    }


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def write_tables(summary: dict[str, Any], cell: Path) -> None:
    segments = summary["segments"]
    with (cell / SEGMENTS_CSV).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "segment", "tokens", "start_offset", "first_call", "last_call",
            "n_calls", "reuse_tokens", "roles",
            "content_tag", "raw_data_share", "is_prompt_end",
            "realized_hits", "realized_tokens", "gap_tokens",
            "call_indices", "content_sha", "excerpt",
        ])
        for s in segments:
            writer.writerow([
                s["segment"], s["tokens"], s["start_offset"], s["first_call"],
                s["last_call"], s["n_calls"], s["reuse_tokens"],
                "|".join(s["roles"]),
                s["content_tag"], s["raw_data_share"], int(s["is_prompt_end"]),
                s["realized_hits"], s["realized_tokens"], s["gap_tokens"],
                " ".join(str(i) for i in s["call_indices"]),
                s["content_sha"],
                s["excerpt"],
            ])

    (cell / SEGMENTS_JSON).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def write_markdown(summary: dict[str, Any], cell: Path, top: int = 60) -> None:
    cb = summary["content_breakdown"]

    L: list[str] = [f"# KV cache contents — `{summary['cell']}`", ""]
    L.append("| content | segments | cache space | real hits | hits ÷ space |")
    L.append("|---|---:|---:|---:|---:|")
    for tag in CONTENT_TAGS:
        e = cb["by_tag"][tag]
        if not e["n_segments"]:
            continue
        L.append(
            f"| {tag} | {e['n_segments']} | {e['cache_size_tokens']:,} "
            f"({e['cache_share']:.1%}) | {e['realized_tokens']:,} "
            f"({e['realized_share']:.1%}) | {e['leverage']} |"
        )
    L.append("")
    if cb["unmatched_share"] > 0.95:
        L.append("**Caution:** no content markers matched; the table above is meaningless here.")
        L.append("")

    losses = [
        s for s in sorted(summary["segments"], key=lambda s: -s["gap_tokens"])[:10]
        if s["gap_tokens"] > 0
    ]
    if losses:
        L.append("**Losing the most to the real cache**")
        L.append("")
        L.append("| seg | tokens | content | traversals | real hits | tokens lost | excerpt |")
        L.append("|---|---:|---|---:|---:|---:|---|")
        for s in losses:
            L.append(
                f"| {s['segment']} | {s['tokens']:,} | {s['content_tag']} | "
                f"{s['n_calls'] - 1} | {s['realized_hits']} | {s['gap_tokens']:,} | "
                f"`{_md_escape(s['excerpt'])}` |"
            )
        L.append("")

    segments = sorted(summary["segments"], key=lambda s: -s["tokens"])[:top]
    L.append(f"**Largest {len(segments)} segments**")
    L.append("")
    L.append("| seg | tokens | content | first call | calls | reuse tokens | excerpt |")
    L.append("|---|---:|---|---:|---:|---:|---|")
    for s in segments:
        L.append(
            f"| {s['segment']} | {s['tokens']:,} | {s['content_tag']} | "
            f"{s['first_call']} | {s['n_calls']} | {s['reuse_tokens']:,} | "
            f"`{_md_escape(s['excerpt'])}` |"
        )
    L.append("")
    (cell / SEGMENTS_MD).write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="+", type=Path, help="cell directories")
    parser.add_argument("--top", type=int, default=60, help="segments listed in markdown")
    args = parser.parse_args()

    for cell in args.cells:
        summary = analyze_cell_segments(cell)
        if not summary["n_calls"]:
            print(f"{cell}: no joined calls")
            continue
        write_tables(summary, cell)
        write_markdown(summary, cell, top=args.top)
        print(
            f"{cell}: {summary['n_calls']} calls, "
            f"{summary['prompt_tokens_total']:,} prompt tokens → "
            f"{summary['cache_size_tokens']:,} cache tokens "
            f"({summary['resend_ratio']}x resend)"
        )


if __name__ == "__main__":
    main()
