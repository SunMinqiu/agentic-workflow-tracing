#!/usr/bin/env python3
"""Timestamp parsing shared by the log parsers.

A leaf module on purpose: `tool_log`, `_ebpf_impl`, and `analysis.summary` all
need it, and any of them owning it creates an import cycle.
"""
from __future__ import annotations

from datetime import datetime


def parse_time(time_str: str) -> datetime:
    """Parse ``HH:MM:SS.ffffff`` using today's date as the anchor."""
    parts = time_str.split(".")
    time_part = parts[0]
    microseconds = parts[1] if len(parts) > 1 else "0"
    microseconds = microseconds[:6].ljust(6, "0")

    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    h, m, s = map(int, time_part.split(":"))
    return base.replace(hour=h, minute=m, second=s, microsecond=int(microseconds))
