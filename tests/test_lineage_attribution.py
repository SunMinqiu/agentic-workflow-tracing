from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from agent_io_tracing.lineage._analyzer_impl import (
    load_codeexec_index,
    write_csvs,
)


class ToolCallAttributionTests(unittest.TestCase):
    def test_parsed_tool_calls_are_exported_without_start_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_dir = Path(temporary)
            (trace_dir / "parsed.json").write_text(
                json.dumps(
                    {
                        "tool_calls": [
                            {
                                "tool_id": "inspect-1",
                                "tool_name": "Examine_data",
                                "start_time": "2026-01-01T00:00:00",
                                "end_time": "2026-01-01T00:00:01",
                                "input_params": {},
                            },
                            {
                                "tool_id": "script-1",
                                "tool_name": "ScriptExec",
                                "start_time": "2026-01-01T00:00:02",
                                "end_time": "2026-01-01T00:00:03",
                                "input_params": {"script_len": 42},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (trace_dir / "pi_events.jsonl").write_text(
                json.dumps(
                    {
                        "type": "tool_execution_end",
                        "toolCallId": "inspect-1",
                        "result": {"content": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (trace_dir / "generated_code.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "script-1",
                        "role": "scilink",
                        "phase": "script_executor",
                        "code_len": 42,
                        "io_layers": ["stdio"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            index = load_codeexec_index(trace_dir)

            self.assertEqual(set(index), {"inspect-1", "script-1"})
            self.assertEqual(index["inspect-1"]["tool_name"], "Examine_data")
            self.assertEqual(index["script-1"]["role"], "scilink")
            self.assertEqual(index["script-1"]["phase"], "script_executor")
            self.assertEqual(index["script-1"]["io_layers"], ["stdio"])

            output_dir = trace_dir / "lineage"
            write_csvs(
                {},
                index,
                output_dir,
                io_events=[
                    {
                        "tool_call_id": "inspect-1",
                        "kind": "R",
                        "size": 4096,
                    }
                ],
                parsed_entries=[],
            )
            with (output_dir / "tool_call_attribution.csv").open(newline="") as f:
                rows = {row["tool_call_id"]: row for row in csv.DictReader(f)}
            self.assertEqual(set(rows), {"inspect-1", "script-1"})
            self.assertEqual(rows["inspect-1"]["read_bytes"], "4096")


if __name__ == "__main__":
    unittest.main()
