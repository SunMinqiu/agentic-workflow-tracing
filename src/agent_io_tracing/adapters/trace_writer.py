#!/usr/bin/env python3
"""The one place a workflow logger writes its per-cell trace files.

Every adapter emits the same set of files, so the writing lives here and a new
workflow inherits it instead of copying it:

    tool_calls.log                  free text, one tool call per block
    tool_calls.log.system_prompt    free text
    pi_events.jsonl                 one JSON object per line
    generated_code.jsonl            one JSON object per line
    messages.jsonl                  one JSON object per line

A subclass supplies ``self._lock`` and the path attributes; the append helpers
below are the only writers. Adding a file means adding a method here, not a
sixth copy of ``open(..., "a")`` in a new adapter.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path


class TraceFileWriter:
    """Serialized appends to the trace files an adapter owns.

    Subclasses must set ``_lock`` and whichever path attributes they write to.
    A method whose path attribute is unset raises AttributeError at call time,
    which is the correct failure: the adapter asked to write a file it never
    declared.
    """

    _lock: threading.RLock
    _tool_log: Path
    _system_prompt_log: Path
    _events_log: Path
    _generated_code_log: Path
    _messages_log: Path

    def _append_text(self, path: Path, text: str) -> None:
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(text)

    def _append_json(self, path: Path, record: dict) -> None:
        self._append_text(
            path, json.dumps(record, ensure_ascii=False, default=str) + "\n"
        )

    def _append_tool_log(self, line: str) -> None:
        self._append_text(self._tool_log, line)

    def _append_system_prompt(self, entry: str) -> None:
        self._append_text(self._system_prompt_log, entry)

    def _append_event(self, event: dict) -> None:
        self._append_json(self._events_log, event)

    def _append_generated_code(self, record: dict) -> None:
        self._append_json(self._generated_code_log, record)

    def _append_messages(self, record: dict) -> None:
        self._append_json(self._messages_log, record)


def normalize_tool_name(name: str | None) -> str:
    """Tool name as it appears in tool_calls.log.

    GenoMAS keeps its own variant because an unnamed call there is an LLM call,
    not a tool call.
    """
    if not name:
        return "Tool"
    if name.lower() == "bash":
        return "Bash"
    return name[0].upper() + name[1:]
