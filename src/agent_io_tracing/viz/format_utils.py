#!/usr/bin/env python3
"""Formatting and loading helpers shared by every HTML/plot builder.

These were copied into `_trace_impl`, `fanout_index`, `fanout_plot`, and
`fanout_input_sizes` independently. A new page builder imports them from here.
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path


def jload(path: Path) -> dict:
    """Parse a JSON file, or an empty dict when it is missing or malformed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt_bytes(value) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(x) < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} TB"


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def datetime_from_ms(ms: float) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0)
