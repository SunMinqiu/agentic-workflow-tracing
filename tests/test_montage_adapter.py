from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_io_tracing.adapters.montage import prepare_input, run_montage


class MontageDriverTests(unittest.TestCase):
    def test_existing_fixed_input_is_verified_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            raw = output / "raw"
            raw.mkdir()
            fits = raw / "input.fits"
            fits.write_bytes(b"fits")
            (output / "input_manifest.sha256").write_text(
                f"{prepare_input._sha256(fits)}  raw/input.fits\n", encoding="utf-8"
            )
            self.assertEqual(
                prepare_input.main(
                    ["--output", str(output), "--size-deg", "0.10"]
                ),
                0,
            )
            fits.write_bytes(b"changed")
            self.assertEqual(
                prepare_input.main(
                    ["--output", str(output), "--size-deg", "0.10"]
                ),
                1,
            )

    def test_fixed_input_pipeline_writes_execution_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            raw = work / "raw"
            raw.mkdir()
            (raw / "input.fits").write_bytes(b"fits")
            units = work / "execution_units.jsonl"

            functions = {}
            for stage in run_montage.STAGES:
                def function(stage: run_montage.Stage = stage) -> dict[str, str]:
                    for output in stage.outputs:
                        path = work / output
                        if path.suffix:
                            path.write_text(stage.name, encoding="utf-8")
                        else:
                            path.mkdir(exist_ok=True)
                            (path / f"{stage.name}.fits").write_bytes(b"fits")
                    return {"status": "0"}

                functions[stage.name] = function

            rc = run_montage.run_pipeline(
                work,
                0.1,
                "M 17",
                units,
                offline=True,
                functions=functions,
            )
            self.assertEqual(rc, 0)
            rows = [json.loads(line) for line in units.read_text().splitlines()]
            self.assertEqual(len(rows), len(run_montage.STAGES))
            self.assertTrue(all(row["status"] == "completed" for row in rows))
            self.assertTrue(all(row["pid"] for row in rows))
            summary = json.loads((work / "montage_run_summary.json").read_text())
            self.assertEqual(summary["tasks_completed"], len(run_montage.STAGES))
            self.assertEqual(summary["input_fits_count"], 1)


if __name__ == "__main__":
    unittest.main()
