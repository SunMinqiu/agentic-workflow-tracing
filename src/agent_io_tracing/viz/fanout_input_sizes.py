#!/usr/bin/env python3
"""Summarize staged GEO input file sizes for a fanout run.

Outputs:
  <run>/figures/input_files_tidy.csv
  <run>/figures/input_size_distribution.png

The script uses each cell's stage_manifest.json and stats the real source
cohort directories under src_root/GEO/<trait>/<cohort>. It intentionally
measures experiment inputs, not generated artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXIS1 = [("base", 1), ("a1_c2", 2), ("a1_c4", 4), ("a1_c8", 8)]
AXIS2 = [("base", 1), ("a2_t2", 2), ("a2_t4", 4), ("a2_t8", 8)]


def jload(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def cell_axis_param(cell: str) -> tuple[str, int]:
    if cell == "base":
        return "shared", 1
    if cell.startswith("a1_c"):
        return "cohort", int(cell.removeprefix("a1_c"))
    if cell.startswith("a2_t"):
        return "trait", int(cell.removeprefix("a2_t"))
    return "unknown", 0


def iter_cell_input_files(cell_dir: Path) -> list[dict]:
    manifest = jload(cell_dir / "stage_manifest.json")
    src_root = Path(manifest.get("src_root") or "")
    if not src_root:
        return []
    axis, param = cell_axis_param(cell_dir.name)
    rows: list[dict] = []
    for trait_rec in manifest.get("traits") or []:
        trait = trait_rec.get("trait")
        if not trait:
            continue
        for cohort in trait_rec.get("cohorts") or []:
            cohort_dir = src_root / "GEO" / trait / cohort
            if not cohort_dir.is_dir():
                continue
            for path in sorted(cohort_dir.rglob("*")):
                if not path.is_file():
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                rows.append({
                    "cell": cell_dir.name,
                    "axis": axis,
                    "param": param,
                    "trait": trait,
                    "cohort": cohort,
                    "kind": "GEO",
                    "path": str(path),
                    "rel_path": str(path.relative_to(src_root)),
                    "size_bytes": size,
                    "size_mb": size / (1024 * 1024),
                    "suffix": path.suffix.lower() or "(none)",
                })
    return rows


def collect(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir() or d.name == "figures":
            continue
        if not (d / "stage_manifest.json").is_file():
            continue
        rows.extend(iter_cell_input_files(d))
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    cols = [
        "cell", "axis", "param", "trait", "cohort", "kind",
        "path", "rel_path", "size_bytes", "size_mb", "suffix",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _axis_rows(rows: list[dict], axis_spec: list[tuple[str, int]]) -> list[tuple[str, int, list[float]]]:
    out = []
    for cell, x in axis_spec:
        sizes = [float(r["size_bytes"]) for r in rows if r["cell"] == cell and r["size_bytes"] > 0]
        out.append((cell, x, sizes))
    return out


def _ecdf(ax, axis_rows: list[tuple[str, int, list[float]]], title: str) -> None:
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(1, len(axis_rows))))
    any_data = False
    for (cell, x, sizes), color in zip(axis_rows, colors):
        if not sizes:
            continue
        any_data = True
        vals = np.sort(np.array(sizes, dtype=float))
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", label=f"{cell} (n={len(vals)})", color=color)
    ax.set_xscale("log")
    ax.set_xlabel("input file size (bytes, log)")
    ax.set_ylabel("ECDF")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, which="both")
    if any_data:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no input files", transform=ax.transAxes,
                ha="center", va="center", color="#7f8c8d")


def _totals(ax, axis_rows: list[tuple[str, int, list[float]]], title: str, xlabel: str) -> None:
    xs = [x for _, x, _ in axis_rows]
    totals_mb = [sum(sizes) / (1024 * 1024) for _, _, sizes in axis_rows]
    counts = [len(sizes) for _, _, sizes in axis_rows]
    ax.plot(xs, totals_mb, "o-", color="#1f77b4", lw=2)
    for x, y, n in zip(xs, totals_mb, counts):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color="#52606d")
    ax.set_xticks(xs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("total input MB")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def make_plot(rows: list[dict], out: Path) -> None:
    axis1 = _axis_rows(rows, AXIS1)
    axis2 = _axis_rows(rows, AXIS2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Input Data Size Distribution", fontsize=13)
    _ecdf(axes[0, 0], axis1, "Axis 1 — cohort scaling")
    _ecdf(axes[0, 1], axis2, "Axis 2 — trait fanout")
    _totals(axes[1, 0], axis1, "Axis 1 — total staged input", "C = #cohorts")
    _totals(axes[1, 1], axis2, "Axis 2 — total staged input", "T = #traits")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)
    rows = collect(run_dir)
    write_csv(rows, figs / "input_files_tidy.csv")
    make_plot(rows, figs / "input_size_distribution.png")
    print(f"Wrote {figs / 'input_files_tidy.csv'} ({len(rows)} rows)")
    print(f"Wrote {figs / 'input_size_distribution.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
