from pathlib import Path

import pytest

from agent_io_tracing.adapters.genomas.launcher import _selected_geo_inputs


def _cohorts(root: Path, trait: str, names: list[str]) -> None:
    for name in names:
        (root / "GEO" / trait / name).mkdir(parents=True, exist_ok=True)


def test_selection_is_sorted_gse_directories_only(tmp_path: Path) -> None:
    _cohorts(tmp_path, "Trait", ["GSE9", "GSE10", "GSE2", "not-a-cohort"])
    (tmp_path / "GEO" / "Trait" / ".DS_Store").write_text("metadata")

    selected = _selected_geo_inputs(tmp_path, ["Trait"], 2)

    assert selected == {"Trait": ["GSE10", "GSE2"]}


def test_selection_rejects_an_unsatisfied_cohort_count(tmp_path: Path) -> None:
    _cohorts(tmp_path, "Trait", ["GSE1"])

    with pytest.raises(ValueError, match="requested 2 GEO cohorts"):
        _selected_geo_inputs(tmp_path, ["Trait"], 2)
