#!/usr/bin/env python3
from __future__ import annotations
"""
Extract execution phases from Claude agent tool call logs.

Identifies the alternating pattern of:
- Tool execution phases: when tools are actively running (potentially in parallel)
- Model completion phases: gaps where the LLM generates its next response

Usage:
    python extract_phases.py <trace_dir>
    python extract_phases.py <trace_dir>/tool_calls.log
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import parsing utilities from parse_strace
from parse_strace import ToolCall, parse_tool_calls_log, parse_time


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ToolBatch:
    """A batch of tool calls that were dispatched together (in parallel)."""
    tool_calls: list[ToolCall]
    batch_index: int
    
    @property
    def start_time(self) -> datetime:
        """Earliest start time in the batch."""
        return min(tc.start_time for tc in self.tool_calls)
    
    @property
    def end_time(self) -> datetime:
        """Latest end time in the batch."""
        return max(tc.end_time for tc in self.tool_calls)
    
    @property
    def duration_ms(self) -> float:
        """Total duration of the batch in milliseconds."""
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    @property
    def tool_count(self) -> int:
        """Number of tool calls in this batch."""
        return len(self.tool_calls)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "batch_index": self.batch_index,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_ms": self.duration_ms,
            "tool_count": self.tool_count,
            "tools": [
                {
                    "tool_name": tc.tool_name,
                    "tool_id": tc.tool_id,
                    "start_time": tc.start_time.isoformat(),
                    "end_time": tc.end_time.isoformat(),
                    "duration_ms": (tc.end_time - tc.start_time).total_seconds() * 1000,
                    "input_params": tc.input_params,
                }
                for tc in self.tool_calls
            ],
        }


@dataclass
class Phase:
    """Represents a single execution phase (either tool execution or model completion)."""
    phase_type: str  # "tool_execution" or "model_completion"
    start_time: datetime
    end_time: datetime
    phase_index: int
    # For tool_execution phases only:
    batch: ToolBatch | None = None
    
    @property
    def duration_ms(self) -> float:
        """Duration of this phase in milliseconds."""
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {
            "phase_type": self.phase_type,
            "phase_index": self.phase_index,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_ms": self.duration_ms,
        }
        if self.batch:
            result["tool_count"] = self.batch.tool_count
            result["batch"] = self.batch.to_dict()
        return result


@dataclass
class PhaseAnalysis:
    """Complete phase analysis results."""
    phases: list[Phase] = field(default_factory=list)
    batches: list[ToolBatch] = field(default_factory=list)
    
    # Computed statistics
    total_duration_ms: float = 0.0
    total_tool_execution_ms: float = 0.0
    total_model_completion_ms: float = 0.0
    
    @property
    def tool_execution_phases(self) -> list[Phase]:
        """All tool execution phases."""
        return [p for p in self.phases if p.phase_type == "tool_execution"]
    
    @property
    def model_completion_phases(self) -> list[Phase]:
        """All model completion phases."""
        return [p for p in self.phases if p.phase_type == "model_completion"]
    
    @property
    def tool_execution_pct(self) -> float:
        """Percentage of time spent in tool execution."""
        if self.total_duration_ms == 0:
            return 0.0
        return (self.total_tool_execution_ms / self.total_duration_ms) * 100
    
    @property
    def model_completion_pct(self) -> float:
        """Percentage of time spent in model completion."""
        if self.total_duration_ms == 0:
            return 0.0
        return (self.total_model_completion_ms / self.total_duration_ms) * 100
    
    @property
    def batch_count(self) -> int:
        """Number of tool batches."""
        return len(self.batches)
    
    @property
    def avg_tools_per_batch(self) -> float:
        """Average number of tools per batch."""
        if not self.batches:
            return 0.0
        return sum(b.tool_count for b in self.batches) / len(self.batches)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        tool_exec_durations = [p.duration_ms for p in self.tool_execution_phases]
        model_comp_durations = [p.duration_ms for p in self.model_completion_phases]
        
        return {
            "phases": [p.to_dict() for p in self.phases],
            "summary": {
                "total_duration_ms": self.total_duration_ms,
                "total_tool_execution_ms": self.total_tool_execution_ms,
                "total_model_completion_ms": self.total_model_completion_ms,
                "tool_execution_pct": round(self.tool_execution_pct, 2),
                "model_completion_pct": round(self.model_completion_pct, 2),
                "batch_count": self.batch_count,
                "avg_tools_per_batch": round(self.avg_tools_per_batch, 2),
                "total_tool_calls": sum(b.tool_count for b in self.batches),
                "tool_execution_stats": {
                    "count": len(tool_exec_durations),
                    "min_ms": round(min(tool_exec_durations), 2) if tool_exec_durations else 0,
                    "max_ms": round(max(tool_exec_durations), 2) if tool_exec_durations else 0,
                    "avg_ms": round(sum(tool_exec_durations) / len(tool_exec_durations), 2) if tool_exec_durations else 0,
                },
                "model_completion_stats": {
                    "count": len(model_comp_durations),
                    "min_ms": round(min(model_comp_durations), 2) if model_comp_durations else 0,
                    "max_ms": round(max(model_comp_durations), 2) if model_comp_durations else 0,
                    "avg_ms": round(sum(model_comp_durations) / len(model_comp_durations), 2) if model_comp_durations else 0,
                },
            },
        }


# =============================================================================
# Phase Extraction Logic
# =============================================================================

def group_into_batches(tool_calls: list[ToolCall], threshold_ms: float = 100.0) -> list[ToolBatch]:
    """
    Group tool calls into batches based on start time proximity.
    
    Tool calls that start within `threshold_ms` of each other are considered
    to be part of the same batch (dispatched in parallel by the model).
    
    Args:
        tool_calls: List of tool calls sorted by start time
        threshold_ms: Maximum gap between start times to be in same batch
        
    Returns:
        List of ToolBatch objects
    """
    if not tool_calls:
        return []
    
    # Sort by start time
    sorted_calls = sorted(tool_calls, key=lambda tc: tc.start_time)
    
    batches = []
    current_batch_calls = [sorted_calls[0]]
    batch_index = 0
    
    for tc in sorted_calls[1:]:
        # Calculate gap from the first call in the current batch
        gap_ms = (tc.start_time - current_batch_calls[0].start_time).total_seconds() * 1000
        
        if gap_ms <= threshold_ms:
            # Same batch - tools started close together
            current_batch_calls.append(tc)
        else:
            # New batch - finalize current batch
            batches.append(ToolBatch(
                tool_calls=current_batch_calls,
                batch_index=batch_index,
            ))
            batch_index += 1
            current_batch_calls = [tc]
    
    # Don't forget the last batch
    if current_batch_calls:
        batches.append(ToolBatch(
            tool_calls=current_batch_calls,
            batch_index=batch_index,
        ))
    
    return batches


def extract_phases(
    tool_calls: list[ToolCall],
    threshold_ms: float = 100.0,
    include_initial_model_phase: bool = False,
) -> PhaseAnalysis:
    """
    Extract execution phases from tool calls.
    
    Phases alternate between:
    - tool_execution: when tools are actively running
    - model_completion: gaps between batches where the model generates responses
    
    Args:
        tool_calls: List of tool calls (will be sorted internally)
        threshold_ms: Threshold for grouping tool calls into batches
        include_initial_model_phase: If True, include model completion phase before first batch
        
    Returns:
        PhaseAnalysis containing all phases and statistics
    """
    # Filter out Task tool calls (used for overall task tracking)
    filtered_calls = [tc for tc in tool_calls if tc.tool_name != "Task"]
    
    if not filtered_calls:
        return PhaseAnalysis()
    
    # Group into batches
    batches = group_into_batches(filtered_calls, threshold_ms)
    
    if not batches:
        return PhaseAnalysis()
    
    phases: list[Phase] = []
    phase_index = 0
    
    # Process each batch and the gaps between them
    for i, batch in enumerate(batches):
        # Add model completion phase before this batch (if there was a previous batch)
        if i > 0:
            prev_batch = batches[i - 1]
            gap_start = prev_batch.end_time
            gap_end = batch.start_time
            
            # Only add if there's actually a gap
            if gap_end > gap_start:
                phases.append(Phase(
                    phase_type="model_completion",
                    start_time=gap_start,
                    end_time=gap_end,
                    phase_index=phase_index,
                ))
                phase_index += 1
        
        # Add tool execution phase for this batch
        phases.append(Phase(
            phase_type="tool_execution",
            start_time=batch.start_time,
            end_time=batch.end_time,
            phase_index=phase_index,
            batch=batch,
        ))
        phase_index += 1
    
    # Calculate statistics
    analysis = PhaseAnalysis(phases=phases, batches=batches)
    
    if phases:
        analysis.total_duration_ms = (phases[-1].end_time - phases[0].start_time).total_seconds() * 1000
    
    analysis.total_tool_execution_ms = sum(
        p.duration_ms for p in phases if p.phase_type == "tool_execution"
    )
    analysis.total_model_completion_ms = sum(
        p.duration_ms for p in phases if p.phase_type == "model_completion"
    )
    
    return analysis


# =============================================================================
# File Loading
# =============================================================================

def load_from_tool_calls_log(filepath: Path) -> PhaseAnalysis:
    """Load and analyze phases from a tool_calls.log file."""
    tool_calls = parse_tool_calls_log(filepath)
    return extract_phases(tool_calls)


def load_from_parsed_json(filepath: Path) -> PhaseAnalysis:
    """Load and analyze phases from a parsed.json file."""
    with open(filepath) as f:
        data = json.load(f)
    
    tool_calls = []
    for tc_dict in data.get("tool_calls", []):
        tool_calls.append(ToolCall(
            start_time=datetime.fromisoformat(tc_dict["start_time"]),
            end_time=datetime.fromisoformat(tc_dict["end_time"]),
            tool_name=tc_dict["tool_name"],
            tool_id=tc_dict["tool_id"],
            input_params=tc_dict.get("input_params", {}),
        ))
    
    return extract_phases(tool_calls)


def load_phases(trace_dir: Path) -> PhaseAnalysis:
    """
    Load phase analysis from a trace directory.
    
    Tries tool_calls.log first, then parsed.json.
    """
    tool_log = trace_dir / "tool_calls.log"
    parsed_json = trace_dir / "parsed.json"
    
    if tool_log.exists():
        return load_from_tool_calls_log(tool_log)
    elif parsed_json.exists():
        return load_from_parsed_json(parsed_json)
    else:
        raise FileNotFoundError(
            f"Neither tool_calls.log nor parsed.json found in {trace_dir}"
        )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract execution phases from Claude agent tool call logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s traces/20260116_134709/d-copy-nmd-mul/
  %(prog)s traces/20260116_134709/d-copy-nmd-mul/tool_calls.log
  %(prog)s traces/20260116_134709/d-copy-nmd-mul/ --threshold 200
        """
    )
    
    parser.add_argument(
        "path",
        type=Path,
        help="Trace directory or tool_calls.log file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Threshold in ms for grouping tool calls into batches (default: 100)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (default: <trace_dir>/phases.json)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics to stderr"
    )
    
    args = parser.parse_args()
    
    # Determine input path
    input_path = args.path
    if input_path.is_file():
        if input_path.name == "tool_calls.log":
            analysis = load_from_tool_calls_log(input_path)
            trace_dir = input_path.parent
        elif input_path.name == "parsed.json":
            analysis = load_from_parsed_json(input_path)
            trace_dir = input_path.parent
        else:
            print(f"Error: Unsupported file type: {input_path.name}", file=sys.stderr)
            sys.exit(1)
    elif input_path.is_dir():
        analysis = load_phases(input_path)
        trace_dir = input_path
    else:
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    output_path = args.output or (trace_dir / "phases.json")
    
    # Print summary if requested
    if args.summary:
        summary = analysis.to_dict()["summary"]
        print(f"Phase Analysis Summary:", file=sys.stderr)
        print(f"  Total duration: {summary['total_duration_ms']:.1f}ms", file=sys.stderr)
        print(f"  Tool execution: {summary['total_tool_execution_ms']:.1f}ms ({summary['tool_execution_pct']:.1f}%)", file=sys.stderr)
        print(f"  Model completion: {summary['total_model_completion_ms']:.1f}ms ({summary['model_completion_pct']:.1f}%)", file=sys.stderr)
        print(f"  Batches: {summary['batch_count']}", file=sys.stderr)
        print(f"  Avg tools/batch: {summary['avg_tools_per_batch']:.2f}", file=sys.stderr)
        print(f"  Total phases: {len(analysis.phases)}", file=sys.stderr)
    
    # Write output
    with open(output_path, 'w') as f:
        json.dump(analysis.to_dict(), f, indent=2)
    
    print(f"Phases written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

