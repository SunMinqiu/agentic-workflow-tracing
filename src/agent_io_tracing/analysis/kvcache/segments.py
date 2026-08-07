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
  hit_by_tag    the tokens the real cache served out of it, split by what the
                text is — see sections.py for how a prompt is cut into
                instructions, code, raw data and history

Splitting per segment rather than labelling each segment once is what makes the
headline answerable: of everything the cache actually served, how much of it was
each kind of text.

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
from agent_io_tracing.analysis.kvcache.sections import (
    UNLABELED, describe, detect_grammar, overlap_by_tag, section_token_ranges,
    widest_span,
)

SEGMENTS_CSV = "kvcache_segments.csv"
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

# Two numbers per content type, with two different denominators:
#
#   tokens served   an exact split of the hit tokens. A hit crossing from one
#                   section into the next gives tokens to both, so these add
#                   back up to the total.
#   hit segments    how many hit segments contained any of this type. One
#                   segment spanning code and output counts under both, so
#                   these deliberately sum past the number of segments.
def _content_breakdown(
    segments: list[dict[str, Any]], tags: list[str]
) -> dict[str, Any]:
    """What the real cache served, per content type."""
    realized_total = sum(s["realized_tokens"] for s in segments)
    served = [s for s in segments if s["realized_tokens"] > 0]
    by_tag = {}
    for tag in tags:
        tokens = sum(s["hit_by_tag"].get(tag, 0) for s in segments)
        n = sum(1 for s in served if s["hit_by_tag"].get(tag, 0) > 0)
        by_tag[tag] = {
            "realized_tokens": tokens,
            "realized_share": (
                round(tokens / realized_total, 4) if realized_total else 0.0
            ),
            "n_segments": n,
            "segment_share": round(n / len(served), 4) if served else 0.0,
        }
    return {
        "by_tag": by_tag,
        "tags": tags,
        "realized_tokens_total": realized_total,
        "n_served_segments": len(served),
        "unlabeled_share": by_tag.get(UNLABELED, {}).get("realized_share", 0.0),
    }


def examples_by_tag(
    segments: list[dict[str, Any]], tags: list[str], per_tag: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """A few served segments per content type, biggest contribution first.

    A segment that served two types appears under both, showing each type's own
    text, so every example matches the label above it.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for tag in tags:
        group = sorted(
            (s for s in segments if s["hit_by_tag"].get(tag, 0) > 0),
            key=lambda s: -s["hit_by_tag"][tag],
        )[:per_tag]
        out[tag] = [
            {
                "tokens": s["tokens"],
                "first_call": s["first_call"],
                "realized_hits": s["realized_hits"],
                "tag_tokens": s["hit_by_tag"][tag],
                "n_calls": s["n_calls"],
                "section": s["hit_samples"][tag]["section"],
                "hit_chars": s["hit_samples"][tag]["chars"],
                "hit_sample": s["hit_samples"][tag]["sample"],
            }
            for s in group
        ]
    return out


def _sample(text: str, head: int = 420, tail: int = 220) -> str:
    """Head and tail of the text, elided in between when it is long."""
    if len(text) <= head + tail + 20:
        return text
    return f"{text[:head]}\n  …\n{text[-tail:]}"


def one_line(text: str, limit: int = 150) -> str:
    """The same text squeezed onto one row of a table."""
    flat = text.replace("\n", "\\n")
    return flat[:limit]


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
            rows.append({
                "n_chars": len(text),
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
                "sample": _sample(text),
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
        # whole tokens per call, not a rounded sum over calls: the per-content
        # split below is computed call by call and has to add back up to this
        covered = {
            i: int(max(0.0, min(cut[i] - start, float(s["tokens"]))))
            for i in reusers
        }
        covered = {i: v for i, v in covered.items() if v > 0}
        s["covered_tokens"] = covered
        s["realized_calls"] = sorted(covered)
        s["realized_hits"] = len(covered)
        s["realized_tokens"] = sum(covered.values())
        # a hit always starts at the segment's first token and stops wherever
        # the reported prefix ran out, so the served text is a prefix of the
        # segment — this is the longest one any call actually got
        s["hit_tokens"] = max(covered.values(), default=0)
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
        # every call's realized prefix is partitioned by the segments on its
        # path, so the attribution must add back up to the reported hits
        "reported_hit_tokens_rescaled": round(sum(cut)),
        "attribution_residual": round(sum(cut)) - realized,
        "tokenizer_scale_error_pct": (
            round(100 * sum(scale_deltas) / len(scale_deltas), 2) if scale_deltas else None
        ),
    }


def attach_content(
    segments: list[dict[str, Any]],
    token_sequences: list[list[int]],
    ranges_per_call: list[list[tuple[int, int, str, str]]],
    enc: tiktoken.Encoding,
) -> None:
    """Split each segment's served tokens across the content types it covers.

    A hit is a prefix of the segment, and the segment sits at a fixed offset in
    every call that traverses it, so the sections of the call that created it
    describe the text for all of them. Each call's own hit length is intersected
    with those sections separately, which is why the parts add back up to
    ``realized_tokens`` exactly rather than approximately.
    """
    for s in segments:
        start = s["start_offset"]
        deep_end = start + s["hit_tokens"]
        near = [
            r for r in ranges_per_call[s["first_call"]]
            if r[0] < deep_end and r[1] > start
        ]
        totals: dict[str, int] = {}
        for covered in s["covered_tokens"].values():
            for tag, width in overlap_by_tag(near, start, start + covered).items():
                totals[tag] = totals.get(tag, 0) + width
        s["hit_by_tag"] = totals
        # one sample per type, cut from that type's own longest section inside
        # the hit: a segment that served code and output shows the code under
        # code and the output under raw data, not one of them under both
        s["hit_samples"] = {}
        for tag in totals:
            lo, hi, name = widest_span(near, start, deep_end, tag)
            served = enc.decode(token_sequences[s["first_call"]][lo:hi])
            s["hit_samples"][tag] = {
                "section": name,
                "chars": len(served),
                "sample": _sample(served),
            }
        s["hit_tag"] = max(totals, key=lambda t: (totals[t], t)) if totals else None


def analyze_cell_segments(cell: Path) -> dict[str, Any]:
    calls = load_joined_calls(cell)
    if not calls:
        return {"cell": cell.name, "n_calls": 0, "segments": []}

    enc = _encoding_for(calls[0]["model"])
    texts = [_serialize(c["messages"]) for c in calls]
    token_sequences = [enc.encode(t, disallowed_special=()) for t in texts]
    roles = [c["role"] for c in calls]

    root = build_radix_trie(token_sequences)
    segments = collect_segments(root, enc, roles)

    realized = attach_realized(segments, calls, token_sequences)
    grammar = detect_grammar([c["messages"] for c in calls])
    ranges = [
        section_token_ranges(c["messages"], token_sequences[i], enc, grammar, texts[i])
        for i, c in enumerate(calls)
    ]
    attach_content(segments, token_sequences, ranges, enc)
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
        "sections": describe(grammar, ranges),
        "content_breakdown": _content_breakdown(segments, list(grammar.tags)),
        "content_examples": examples_by_tag(segments, list(grammar.tags)),
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
            "hit_tag", "hit_by_tag", "is_prompt_end",
            "realized_hits", "realized_tokens", "gap_tokens",
            "call_indices", "content_sha", "sample",
        ])
        for s in segments:
            writer.writerow([
                s["segment"], s["tokens"], s["start_offset"], s["first_call"],
                s["last_call"], s["n_calls"], s["reuse_tokens"],
                "|".join(s["roles"]),
                s["hit_tag"] or "",
                " ".join(f"{k}={v}" for k, v in sorted(s["hit_by_tag"].items())),
                int(s["is_prompt_end"]),
                s["realized_hits"], s["realized_tokens"], s["gap_tokens"],
                " ".join(str(i) for i in s["call_indices"]),
                s["content_sha"],
                one_line(s["sample"], 400),
            ])

    (cell / SEGMENTS_JSON).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="+", type=Path, help="cell directories")
    args = parser.parse_args()

    for cell in args.cells:
        summary = analyze_cell_segments(cell)
        if not summary["n_calls"]:
            print(f"{cell}: no joined calls")
            continue
        write_tables(summary, cell)
        print(
            f"{cell}: {summary['n_calls']} calls, "
            f"{summary['prompt_tokens_total']:,} prompt tokens → "
            f"{summary['cache_size_tokens']:,} cache tokens "
            f"({summary['resend_ratio']}x resend)"
        )


if __name__ == "__main__":
    main()
