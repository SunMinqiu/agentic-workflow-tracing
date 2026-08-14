from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_io_tracing.analysis.phase1_metrics import (
    compute_byte_normalized_summary,
    compute_inter_arrival,
    compute_request_size_cdf,
    _run_timeline_ms,
)
from agent_io_tracing.analysis.execution_units import annotate_parsed_execution_units
from agent_io_tracing.analysis.workload_paths import workload_path_index
from agent_io_tracing.analysis.per_run_io_char import _entry_bytes
from agent_io_tracing.lineage._analyzer_impl import (
    compute_generations,
    is_execution_unit_relative_artifact,
)


class ComparisonMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = [
            {"path": "/work/data.bin", "n_reads": "2", "n_writes": "2"},
        ]
        self.parsed = {
            "fs_entries": [
                {
                    "syscall": "read",
                    "path": "/work/data.bin",
                    "requested_size": 4096,
                    "bytes_transferred": 100,
                    "ts_ms": 1000,
                },
                {
                    "syscall": "read",
                    "path": "/work/data.bin",
                    "requested_size": 8192,
                    "bytes_transferred": 300,
                    "ts_ms": 2000,
                },
                {
                    "syscall": "write",
                    "path": "/work/data.bin",
                    "requested_size": 200,
                    "bytes_transferred": 200,
                    "ts_ms": 3000,
                },
                {
                    "syscall": "write",
                    "path": "/work/data.bin",
                    "requested_size": 600,
                    "bytes_transferred": 600,
                    "ts_ms": 5000,
                },
                {
                    "syscall": "read",
                    "path": "/usr/lib/noise.so",
                    "requested_size": 1_000_000,
                    "bytes_transferred": 1_000_000,
                    "ts_ms": 6000,
                },
            ]
        }

    def test_request_sizes_are_split_and_workload_scoped(self) -> None:
        result = compute_request_size_cdf(self.parsed, self.artifacts)

        self.assertEqual(result["read"]["count"], 2)
        self.assertEqual(result["write"]["count"], 2)
        self.assertEqual(result["read"]["p50_bytes"], 6144)
        self.assertEqual(result["write"]["p50_bytes"], 400)

    def test_per_run_volume_prefers_completed_bytes(self) -> None:
        self.assertEqual(
            _entry_bytes({
                "requested_size": 4 * 1024 * 1024,
                "actual_size": 4 * 1024 * 1024,
                "bytes_transferred": 133,
            }),
            133,
        )

    def test_counts_use_actual_bytes_as_denominators(self) -> None:
        self.parsed["fs_entries"].append(
            {
                "syscall": "read",
                "path": "/work/data.bin",
                "requested_size": 1_000_000,
                "bytes_transferred": 0,
                "ts_ms": 7000,
            }
        )
        result = compute_byte_normalized_summary(
            self.parsed,
            self.artifacts,
            {"storage_metadata_ops": 3},
            {"total_scans": 2},
        )

        self.assertEqual(result["denominators"]["read_bytes"], 400)
        self.assertEqual(result["denominators"]["write_bytes"], 800)
        self.assertEqual(result["absolute"]["read_ops"], 2)
        self.assertEqual(result["absolute"]["write_ops"], 2)

    def test_inter_arrival_is_split_by_direction(self) -> None:
        result = compute_inter_arrival(self.parsed, self.artifacts)

        self.assertEqual(result["read"]["n_intervals"], 1)
        self.assertEqual(result["read"]["p50_s"], 1.0)
        self.assertEqual(result["write"]["n_intervals"], 1)
        self.assertEqual(result["write"]["p50_s"], 2.0)

    def test_runtime_syscalls_do_not_expand_semantic_wall_clock(self) -> None:
        events = {
            "llm": SimpleNamespace(
                start_ms=1_000.0, end_ms=2_000.0, kind="llm", run_id="llm",
            ),
            "tool": SimpleNamespace(
                start_ms=2_000.0, end_ms=3_000.0, kind="tool", run_id="tool",
                name="Read", role=None, args={},
            ),
        }
        parsed = {
            "fs_entries": [
                {"ts_ms": 100.0, "duration": 0.0},
                {"ts_ms": 9_000.0, "duration": 0.0},
            ]
        }
        with patch(
            "agent_io_tracing.analysis.parallelism.load_events",
            return_value=events,
        ):
            timeline = _run_timeline_ms(Path("/missing"), parsed, {}, {})

        self.assertEqual(timeline["wall_start_ms"], 1_000.0)
        self.assertEqual(timeline["wall_end_ms"], 3_000.0)
        self.assertEqual(timeline["wall_s"], 2.0)

    def test_classic_pid_interval_becomes_execution_unit(self) -> None:
        parsed = {
            "tool_calls": [],
            "fs_entries": [{"pid": 42, "ts_ms": 1500, "syscall": "read"}],
        }
        units = [{
            "execution_unit_id": "task-1",
            "task_id": "task-1",
            "stage": "sifting",
            "pid": 42,
            "start_ts_ms": 1000,
            "end_ts_ms": 2000,
        }]

        self.assertEqual(annotate_parsed_execution_units(parsed, units), 1)
        self.assertEqual(parsed["fs_entries"][0]["matched_tool_call"], "task-1")
        self.assertEqual(parsed["fs_entries"][0]["execution_stage"], "sifting")
        self.assertEqual(parsed["tool_calls"][0]["input_params"]["phase"], "sifting")

    def test_workload_path_index_matches_files_and_ancestor_directories(self) -> None:
        index = workload_path_index([
            {"path": "/work/data/chr1/input.vcf"},
            {"path": "/work/results/output.csv"},
            {"path": "./chr1-ALL/output_no_sift/result.txt"},
        ])

        self.assertTrue(index.is_file("/work/data/chr1/input.vcf"))
        self.assertTrue(index.is_directory("/work/data/chr1"))
        self.assertTrue(index.is_directory("/work"))
        self.assertTrue(index.is_directory("/"))
        self.assertTrue(index.is_directory("./chr1-ALL/output_no_sift/"))
        self.assertFalse(index.is_directory("/worker"))
        self.assertFalse(index.contains(""))

    def test_classic_execution_unit_uses_parsed_trace_clock(self) -> None:
        start_ms = 10_000_000.0
        end_ms = start_ms + 2_000.0
        trace_clock_start = (
            datetime.fromtimestamp(start_ms / 1000.0) - timedelta(hours=1)
        )
        parsed = {
            "tool_calls": [],
            "fs_entries": [{
                "pid": 42,
                "ts_ms": start_ms + 500.0,
                "timestamp": (
                    datetime.fromtimestamp((start_ms + 500.0) / 1000.0)
                    - timedelta(hours=1)
                ).isoformat(),
                "syscall": "read",
            }],
        }
        units = [{
            "execution_unit_id": "task-1",
            "task_id": "task-1",
            "stage": "frequency",
            "pid": 42,
            "start_ts_ms": start_ms,
            "end_ts_ms": end_ms,
        }]

        annotate_parsed_execution_units(parsed, units)

        tool = parsed["tool_calls"][0]
        self.assertEqual(datetime.fromisoformat(tool["start_time"]), trace_clock_start)
        self.assertEqual(
            datetime.fromisoformat(tool["end_time"]),
            trace_clock_start + timedelta(seconds=2),
        )

    def test_relative_path_requires_classic_execution_unit_attribution(self) -> None:
        self.assertTrue(
            is_execution_unit_relative_artifact(
                {"execution_unit_id": "task-1"}, "chr1n.tar.gz"
            )
        )
        self.assertFalse(is_execution_unit_relative_artifact({}, "chr1n.tar.gz"))
        self.assertFalse(
            is_execution_unit_relative_artifact(
                {"execution_unit_id": "task-1"}, "/usr/lib/noise.so"
            )
        )

    def test_lineage_generation_handles_path_cycles(self) -> None:
        artifacts = {
            "input": {
                "reader_tool_ids": {"task-a"},
                "writer_tool_ids": set(),
            },
            "cycle-a": {
                "reader_tool_ids": {"task-b"},
                "writer_tool_ids": {"task-a"},
            },
            "cycle-b": {
                "reader_tool_ids": {"task-a"},
                "writer_tool_ids": {"task-b"},
            },
        }
        compute_generations(artifacts)
        self.assertEqual(artifacts["input"]["generation"], 0)
        self.assertEqual(artifacts["cycle-a"]["generation"], 1)
        self.assertEqual(artifacts["cycle-b"]["generation"], 1)


if __name__ == "__main__":
    unittest.main()
