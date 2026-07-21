"""Shared phase/role labels for per-tool and per-entry I/O attribution."""
from __future__ import annotations

from typing import Any


def _tool_input(tool_call: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        return {}
    inp = tool_call.get("input_params") or tool_call.get("args") or {}
    return inp if isinstance(inp, dict) else {}


def role_for_tool_call(tool_id: str | None,
                       tool_call: dict[str, Any] | None,
                       phases: dict[str, dict[str, Any]]) -> str:
    """Return the agent role label used by both phase metrics and lineage."""
    inp = _tool_input(tool_call)
    phase_rec = phases.get(tool_id or "", {})
    role = (
        inp.get("role")
        or inp.get("genomas_role")
        or phase_rec.get("genomas_role")
        or (tool_call or {}).get("role")
    )
    return str(role) if role else "(unattributed)"


def phase_label_for_tool_call(tool_id: str | None,
                              tool_call: dict[str, Any] | None,
                              phases: dict[str, dict[str, Any]]) -> str:
    """Return the phase label used for both I/O attribution and wall time.

    Labels intentionally match the existing ``phase:role`` shape so older
    dashboards remain comparable while the denominator now comes from the same
    tool-call label derivation as the numerator.
    """
    phase_rec = phases.get(tool_id or "", {})
    if phase_rec.get("phase"):
        role = phase_rec.get("genomas_role")
        return f"{phase_rec['phase']}:{role}" if role else str(phase_rec["phase"])

    inp = _tool_input(tool_call)
    phase = inp.get("phase") or (tool_call or {}).get("tool_name") or (tool_call or {}).get("name")
    if phase:
        role = inp.get("role") or inp.get("genomas_role") or phase_rec.get("genomas_role")
        return f"{phase}:{role}" if role else str(phase)

    return "uncategorized_tool" if tool_id else "orchestration"


def role_for_entry(entry: dict[str, Any],
                   tool_calls: dict[str, dict[str, Any]],
                   phases: dict[str, dict[str, Any]]) -> str:
    tid = entry.get("matched_tool_call") or entry.get("tool_call_id")
    if not isinstance(tid, str) or not tid:
        return "(unattributed)"
    return role_for_tool_call(tid, tool_calls.get(tid), phases)
