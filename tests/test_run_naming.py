"""Run directories are named <Workflow>_<what ran>_<timestamp>, in one place.

The old names were pasted per config, so GenoMAS runs stayed stamped "phase4"
long after that phase ended and SciLink runs carried no workflow name at all.
These tests fail if a script goes back to writing its own prefix.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIGS = ROOT / "config"

WORKFLOW_OF = {
    "trace_script_bcc_genomas.sh": "GenoMAS",
    "trace_script_bcc_genomas_fanout.sh": "GenoMAS",
    "trace_script_bcc_scilink.sh": "SciLink",
    "trace_script_bcc_sragent.sh": "SRAgent",
    "trace_script_bcc_chemgraph.sh": "ChemGraph",
    "trace_script_bcc_montage.sh": "Montage",
    "trace_script_bcc_1000genome.sh": "1000Genome",
    "trace_script_bcc_pi.sh": "Pi",
}


def _bash(body: str) -> str:
    script = f"set -u\nRUN_STAMP=20260806_101500\nsource {SCRIPTS}/lib_results.sh\n{body}"
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, cwd=ROOT
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_one_task_run_is_named_after_that_task():
    name = _bash(
        'WORKLOADS=("A_c2_w1|1|1|x")\n'
        'run_dir_name GenoMAS $(selected_task_names)'
    )
    assert name == "GenoMAS_A_c2_w1_20260806_101500"


def test_many_task_run_is_named_by_count():
    name = _bash(
        'WORKLOADS=("A_c1_w1|1|1|x" "A_c2_w1|1|1|x" "A_c4_w2|2|1|x")\n'
        'run_dir_name GenoMAS $(selected_task_names)'
    )
    assert name == "GenoMAS_3tasks_20260806_101500"


def test_the_name_follows_the_run_workloads_filter():
    """The directory must not claim tasks the run skipped."""
    name = _bash(
        'WORKLOADS=("A_c1_w1|1|1|x" "A_c2_w1|1|1|x" "A_c4_w2|2|1|x")\n'
        'RUN_WORKLOADS="A_c2_w1"\n'
        'run_dir_name GenoMAS $(selected_task_names)'
    )
    assert name == "GenoMAS_A_c2_w1_20260806_101500"


def test_a_study_can_name_itself():
    name = _bash('RUN_LABEL=fanout; WORKLOADS=(); run_dir_name GenoMAS $(selected_task_names)')
    assert name == "GenoMAS_fanout_20260806_101500"


def test_no_workloads_at_all_still_names_the_workflow():
    assert _bash("WORKLOADS=(); run_dir_name SciLink") == "SciLink_20260806_101500"


def test_no_script_pastes_its_own_run_prefix():
    """Every trace script must delegate the name to run_dir_name."""
    offenders = []
    for path in sorted(SCRIPTS.glob("trace_script_bcc_*.sh")):
        for line in path.read_text().splitlines():
            if "default_lustre_results_root" in line and "run_dir_name" not in line:
                offenders.append(f"{path.name}: {line}")
    assert not offenders, "\n".join(offenders)


def test_no_config_pastes_its_own_run_prefix():
    offenders = [
        f"{p.name}: {line}"
        for p in sorted(CONFIGS.glob("*.env"))
        for line in p.read_text().splitlines()
        if line.startswith("BASE_OUT=")
    ]
    assert not offenders, "\n".join(offenders)


def _config_body(path: pathlib.Path) -> str:
    """Everything the config sets except its workload list."""
    return re.sub(r"^WORKLOADS=\(.*?^\)$", "", path.read_text(), flags=re.M | re.S)


def test_a_workflow_has_one_config_not_one_per_experiment():
    """A new experiment is a WORKLOADS entry, never a forked copy of the file.

    config_sragent_paper.env and config_sragent_all_readme.env were byte-identical
    to config_sragent.env apart from their workload lists, so every fix had to be
    made three times. RUN_WORKLOADS selects a subset; RUN_LABEL names the run.
    """
    bodies: dict[str, list[str]] = {}
    for path in sorted(CONFIGS.glob("*.env")):
        body = _config_body(path)
        stripped = "\n".join(
            line for line in body.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
            and not line.startswith("RUN_LABEL=")
        )
        bodies.setdefault(stripped, []).append(path.name)
    clashes = [names for names in bodies.values() if len(names) > 1]
    assert not clashes, (
        "configs differing only in WORKLOADS — merge them and select with "
        f"RUN_WORKLOADS:\n{clashes}"
    )


def test_every_trace_script_names_a_workflow():
    for path in sorted(SCRIPTS.glob("trace_script_bcc_*.sh")):
        expected = WORKFLOW_OF.get(path.name)
        assert expected, f"{path.name} has no workflow name registered in this test"
        line = next(
            (l for l in path.read_text().splitlines() if l.startswith("BASE_OUT=")), ""
        )
        assert re.search(rf"run_dir_name {expected}\b", line), f"{path.name}: {line}"
