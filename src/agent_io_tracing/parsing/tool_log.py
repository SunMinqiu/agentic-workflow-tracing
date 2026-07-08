"""Shared parser for ``tool_calls.log`` lines."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent_io_tracing.parsing._ebpf_impl import parse_tool_calls as parse_ebpf_tool_calls


TOOL_CALL_PATTERN = re.compile(
    r"\[(\d{2}:\d{2}:\d{2}\.\d+)\s*->\s*(\d{2}:\d{2}:\d{2}\.\d+)\]\s*"
    r"\([\d.]+ms\)\s*"
    r"(\w+)\s*"
    r"\(id=([^)]+)\)\s*"
    r"(?:container=\S+\s*)?"
    r"input=(.+)$"
)


@dataclass
class ToolCall:
    """One tool call parsed from the common ``tool_calls.log`` format."""

    start_time: datetime
    end_time: datetime
    tool_name: str
    tool_id: str
    input_params: dict

    def contains_timestamp(self, ts: datetime) -> bool:
        return self.start_time <= ts <= self.end_time

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "input_params": self.input_params,
        }


def parse_time(time_str: str) -> datetime:
    """Parse ``HH:MM:SS.ffffff`` using today's date as the anchor."""

    parts = time_str.split(".")
    time_part = parts[0]
    microseconds = parts[1] if len(parts) > 1 else "0"
    microseconds = microseconds[:6].ljust(6, "0")

    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    h, m, s = map(int, time_part.split(":"))
    return base.replace(hour=h, minute=m, second=s, microsecond=int(microseconds))


def parse_tool_calls_log(filepath: Path) -> list[ToolCall]:
    """Parse ``tool_calls.log`` into ``ToolCall`` objects."""

    tool_calls: list[ToolCall] = []
    with filepath.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            match = TOOL_CALL_PATTERN.match(line)
            if not match:
                print(
                    f"Warning: Could not parse line {line_num}: {line[:80]}...",
                    file=sys.stderr,
                )
                continue

            start_str, end_str, tool_name, tool_id, input_str = match.groups()
            try:
                input_params = ast.literal_eval(input_str)
            except (ValueError, SyntaxError):
                input_params = {"raw": input_str}

            tool_calls.append(
                ToolCall(
                    start_time=parse_time(start_str),
                    end_time=parse_time(end_str),
                    tool_name=tool_name,
                    tool_id=tool_id,
                    input_params=input_params,
                )
            )

    return tool_calls

__all__ = [
    "ToolCall",
    "parse_time",
    "parse_tool_calls_log",
    "parse_ebpf_tool_calls",
]
