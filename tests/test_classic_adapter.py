from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from agent_io_tracing.adapters.classic import launcher, run_1000genome


FAKE_TASK = r'''#!/usr/bin/env python3
import sys
from pathlib import Path

name = Path(__file__).stem
args = sys.argv[1:]
if name == "individuals":
    source, chrom, start, stop, total = args
    assert Path(source).is_file() and Path("columns.txt").is_file()
    output = f"chr{chrom}n-{start}-{stop}.tar.gz"
elif name == "individuals_merge":
    chrom, *chunks = args
    assert chunks and all(Path(chunk).is_file() for chunk in chunks)
    output = f"chr{chrom}n.tar.gz"
elif name == "sifting":
    source, chrom = args
    assert Path(source).is_file()
    output = f"sifted.SIFT.chr{chrom}.txt"
elif name in {"mutation_overlap", "frequency"}:
    chrom = args[args.index("-c") + 1]
    population = args[args.index("-pop") + 1]
    assert Path(f"chr{chrom}n.tar.gz").is_file()
    assert Path(f"sifted.SIFT.chr{chrom}.txt").is_file()
    assert Path(population).is_file() and Path("columns.txt").is_file()
    suffix = "-freq" if name == "frequency" else ""
    output = f"chr{chrom}-{population}{suffix}.tar.gz"
else:
    raise AssertionError(name)
Path(output).write_text(name)
'''


class ClassicLauncherTests(unittest.TestCase):
    def test_launcher_stages_input_and_writes_stubs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.txt"
            source.write_text("fixture", encoding="utf-8")
            work = root / "work"
            logs = root / "logs"
            rc = launcher.main(
                [
                    str(work),
                    str(logs),
                    "--cmd",
                    "true",
                    "--input",
                    f"{source}:renamed.txt",
                    "--no-self-stop",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((work / "renamed.txt").is_symlink())
            self.assertEqual((logs / "pi_events.jsonl").read_text(), "")
            self.assertEqual((logs / "tool_calls.log").read_text(), "")

    @unittest.skipUnless(hasattr(os, "WUNTRACED"), "requires POSIX job control")
    def test_launcher_self_stops_before_exec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "agent_io_tracing.adapters.classic.launcher",
                    str(root / "work"),
                    str(root / "logs"),
                    "--cmd",
                    "true",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stopped = False
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                waited_pid, status = os.waitpid(
                    process.pid, os.WUNTRACED | os.WNOHANG
                )
                if waited_pid and os.WIFSTOPPED(status):
                    stopped = True
                    break
                time.sleep(0.01)
            if not stopped:
                process.terminate()
                process.wait(timeout=5)
            self.assertTrue(stopped, "launcher never reached SIGSTOP boundary")
            os.kill(process.pid, signal.SIGCONT)
            _stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr)


class GenomeDriverTests(unittest.TestCase):
    def test_direct_driver_runs_dag_in_task_sandboxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            bin_dir = repo / "bin"
            bin_dir.mkdir(parents=True)
            for name in (
                "individuals",
                "individuals_merge",
                "sifting",
                "mutation_overlap",
                "frequency",
            ):
                (bin_dir / f"{name}.py").write_text(FAKE_TASK, encoding="utf-8")

            work = root / "work"
            work.mkdir()
            staged = (
                "columns.txt",
                "ALL",
                run_1000genome.MAIN_VCF.format(chromosome="1"),
                run_1000genome.ANNOTATION_VCF.format(chromosome="1"),
            )
            for name in staged:
                (work / name).write_text("fixture", encoding="utf-8")

            rc = run_1000genome.main(
                [
                    "--repo",
                    str(repo),
                    "--work-dir",
                    str(work),
                    "--chromosomes",
                    "1",
                    "--individual-jobs",
                    "2",
                    "--max-workers",
                    "2",
                    "--rows-per-chromosome",
                    "4",
                    "--offline",
                    "--python",
                    sys.executable,
                ]
            )
            self.assertEqual(rc, 0)
            summary = json.loads((work / "classic_run_summary.json").read_text())
            self.assertTrue(summary["offline"])
            self.assertEqual(summary["tasks_total"], 6)
            self.assertEqual(summary["tasks_completed"], 6)
            self.assertEqual(summary["failures"], [])
            self.assertEqual(len(list((work / "tasks").iterdir())), 6)
            self.assertTrue((work / "artifacts" / "chr1n.tar.gz").is_symlink())
            self.assertTrue((work / "artifacts" / "chr1-ALL.tar.gz").is_symlink())
            self.assertTrue((work / "artifacts" / "chr1-ALL-freq.tar.gz").is_symlink())


if __name__ == "__main__":
    unittest.main()
