#!/usr/bin/env python3
"""Cut a prompt into labelled sections, so cache hits can be costed by content.

A prompt is not one blob. GenoMAS assembles it by concatenating parts under
headers its own prompt-building code writes: ``[Code]:``, ``**Task History**``,
``[Output]:``. Parsing those headers replaces "which single label fits this
whole block" — a guess that is wrong whenever a block spans two kinds of text —
with an exact partition: every character belongs to exactly one section, so the
tokens a cache served can be split by content and still add back up.

The headers belong to the workflow that wrote them, so grammars are per
workflow and chosen by looking for their own headers in the corpus. A corpus
that matches none falls back to one section per message keyed on role, which is
all any OpenAI-format prompt guarantees.

Two rules keep the parse honest:

  * headers are an allowlist. ``**Key Suggestions**`` appears once, inside a
    model-written review, and must not become a section — only headers the
    template emits count.
  * a header labels only itself. ``Output Paths`` is a configuration block and
    stays instructions; only ``[Output]:`` is raw data.
  * ``unlabeled`` is reserved for a message no header touched at all. That
    number is the coverage check: if it is large, the grammar is missing
    something.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from agent_io_tracing.analysis.kvcache.logical import _content_str

INSTRUCTIONS = "instructions"
CODE = "code"
RAW_DATA = "raw data"
HISTORY = "history dialog"
SYSTEM = "system message"
DOCUMENT = "document text"
UNLABELED = "unlabeled"


@dataclass(frozen=True)
class Section:
    """One labelled run of characters inside the serialized prompt."""

    start: int
    end: int
    name: str
    tag: str


@dataclass(frozen=True)
class Grammar:
    name: str
    tags: tuple[str, ...]
    role_tags: dict[str, str]
    header: re.Pattern | None = None
    tag_of: dict[str, str] = field(default_factory=dict)
    default_tag: str = INSTRUCTIONS
    parsed_roles: frozenset[str] = frozenset({"user"})

    def tag_for_header(self, name: str) -> str:
        # exact only: "Output Paths" is a configuration block, not an [Output]
        return self.tag_of.get(name, self.default_tag)


# GenoMAS writes three header styles. Everything listed here is emitted by the
# template; nothing a model can produce on its own is.
_GENOMAS_HEADER = re.compile(
    r"^(?:"
    r"\[(?P<bracket>Instruction|Code|Output of a previous step|Output|"
    r"Chosen action unit)\]:?"
    r"|\*\*(?P<bold>Function Tools|Programming Setups|General Guidelines|"
    r"Task History|Task|TO DO:[^*\n]{0,40})\*\*:?"
    r"|(?P<plain>Instruction|Task History|Available Action Units|"
    r"Current Context|Context Variables|Input Paths|Output Paths|Tools|"
    r"Programming Environment Setup|NOTE|FORMAT):"
    r")",
    re.M,
)

GENOMAS = Grammar(
    name="genomas",
    tags=(INSTRUCTIONS, CODE, RAW_DATA, HISTORY, SYSTEM, UNLABELED),
    role_tags={"system": SYSTEM, "assistant": HISTORY, "tool": RAW_DATA},
    header=_GENOMAS_HEADER,
    tag_of={
        "Code": CODE,
        "Output": RAW_DATA,
        "Output of a previous step": RAW_DATA,
        "Task History": HISTORY,
    },
    default_tag=INSTRUCTIONS,
)

# No template headers to find: the user message is the document that was fed in.
FALLBACK = Grammar(
    name="role only",
    tags=(DOCUMENT, SYSTEM, HISTORY, RAW_DATA, UNLABELED),
    role_tags={
        "system": SYSTEM,
        "user": DOCUMENT,
        "assistant": HISTORY,
        "tool": RAW_DATA,
    },
    header=None,
    parsed_roles=frozenset(),
)

GRAMMARS: tuple[Grammar, ...] = (GENOMAS,)

# A grammar has to earn the corpus: one stray line that looks like a header is
# not evidence, a header on most calls is.
MIN_HEADERS_PER_CALL = 2.0


def detect_grammar(message_lists: list[list[dict]]) -> Grammar:
    """Pick the grammar whose own headers are actually in this corpus."""
    if not message_lists:
        return FALLBACK
    best, best_rate = FALLBACK, 0.0
    for grammar in GRAMMARS:
        if grammar.header is None:
            continue
        hits = 0
        for messages in message_lists:
            for m in messages or []:
                if (m.get("role") or "?") in grammar.parsed_roles:
                    hits += len(grammar.header.findall(_content_str(m.get("content"))))
        rate = hits / len(message_lists)
        if rate >= MIN_HEADERS_PER_CALL and rate > best_rate:
            best, best_rate = grammar, rate
    return best


def parse_sections(messages: list[dict], grammar: Grammar) -> list[Section]:
    """Partition the serialized prompt: contiguous, non-overlapping, complete.

    Character offsets match ``logical._serialize`` exactly, including its
    ``<|role|>`` delimiters, which are charged to the section that follows them.
    """
    out: list[Section] = []
    pos = 0
    for m in messages or []:
        role = m.get("role") or "?"
        content = _content_str(m.get("content"))
        # mirror _serialize: f"<|{role}|>\n{content}\n"
        body_start = pos + len(f"<|{role}|>\n")
        body_end = body_start + len(content)
        message_end = body_end + 1
        role_tag = grammar.role_tags.get(role, grammar.default_tag)

        if role not in grammar.parsed_roles or grammar.header is None:
            out.append(Section(pos, message_end, f"<{role}>", role_tag))
            pos = message_end
            continue

        marks = list(grammar.header.finditer(content))
        if not marks:
            out.append(Section(pos, message_end, f"<{role}>", UNLABELED))
            pos = message_end
            continue

        first = body_start + marks[0].start()
        if first > pos:
            # text above the first header is still the message's own kind — for
            # a user prompt that is the standing role description. Only a
            # message where no header matched at all is genuinely unlabeled.
            out.append(Section(pos, first, "preamble", role_tag))
        for i, mark in enumerate(marks):
            name = mark.group("bracket") or mark.group("bold") or mark.group("plain")
            if name.startswith("TO DO:"):
                name = "TO DO"
            start = body_start + mark.start()
            end = (
                body_start + marks[i + 1].start()
                if i + 1 < len(marks)
                else message_end
            )
            out.append(Section(start, end, name, grammar.tag_for_header(name)))
        pos = message_end
    return out


def byte_offsets(enc: tiktoken.Encoding, tokens: list[int]) -> list[int]:
    """Byte offset where each token starts, plus the total as a final entry.

    Bytes, not characters: a token can hold half a multi-byte character, and
    only the byte lengths of the pieces are guaranteed to add up.
    """
    offsets = [0]
    total = 0
    for token in tokens:
        total += len(enc.decode_single_token_bytes(token))
        offsets.append(total)
    return offsets


def section_token_ranges(
    messages: list[dict],
    tokens: list[int],
    enc: tiktoken.Encoding,
    grammar: Grammar,
    text: str,
) -> list[tuple[int, int, str, str]]:
    """The same partition expressed in token offsets: (start, end, name, tag).

    A section boundary can land inside a token. It is resolved to the token
    that contains it, the same way at both ends, so one section's end stays
    equal to the next one's start and the partition survives the conversion.
    """
    offsets = byte_offsets(enc, tokens)
    prefix_bytes = _char_to_byte(text)

    def token_at(char: int) -> int:
        return max(0, bisect.bisect_right(offsets, prefix_bytes[char]) - 1)

    ranges: list[tuple[int, int, str, str]] = []
    n = len(tokens)
    for s in parse_sections(messages, grammar):
        start = token_at(min(s.start, len(text)))
        end = n if s.end >= len(text) else token_at(min(s.end, len(text)))
        if end > start:
            ranges.append((start, end, s.name, s.tag))
    return ranges


def _char_to_byte(text: str) -> list[int]:
    """Byte offset of every character position, and of the end."""
    out = [0]
    total = 0
    for ch in text:
        total += len(ch.encode("utf-8"))
        out.append(total)
    return out


def overlap_by_tag(
    ranges: list[tuple[int, int, str, str]], start: int, end: int
) -> dict[str, int]:
    """Token count per content tag inside [start, end)."""
    totals: dict[str, int] = {}
    if end <= start:
        return totals
    for a, b, _name, tag in ranges:
        width = min(b, end) - max(a, start)
        if width > 0:
            totals[tag] = totals.get(tag, 0) + width
    return totals


def widest_span(
    ranges: list[tuple[int, int, str, str]], start: int, end: int, tag: str
) -> tuple[int, int, str]:
    """The longest single section of ``tag`` inside [start, end), and its name.

    Examples are shown from a section boundary rather than from the start of
    the segment, so what a reader sees is the labelled thing itself.
    """
    best = (start, start, "")
    for a, b, name, section_tag in ranges:
        if section_tag != tag or a >= end or b <= start:
            continue
        lo, hi = max(a, start), min(b, end)
        if hi - lo > best[1] - best[0]:
            best = (lo, hi, name)
    return best


def describe(grammar: Grammar, ranges_per_call: list[Any]) -> dict[str, Any]:
    return {
        "grammar": grammar.name,
        "tags": list(grammar.tags),
        "sections_per_call": (
            round(sum(len(r) for r in ranges_per_call) / len(ranges_per_call), 1)
            if ranges_per_call else 0.0
        ),
    }
