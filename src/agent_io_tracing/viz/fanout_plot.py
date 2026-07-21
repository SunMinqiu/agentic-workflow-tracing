#!/usr/bin/env python3
"""
Plot the GEO-only fanout study (Axis 1 cohort + Axis 2 trait).

Reads each cell's analysis JSONs directly from the run dir, so it works on a
partial run (missing cells are simply skipped) and does not depend on the
harness's summary CSV.

Per cell it pulls:
  phase1_metrics.json   -> metadata_data_ratio.{storage_metadata_ops,data_ops,
                           storage_metadata_to_data_ops},
                           analytical_optimum_amplification.{actual_generated_files,
                           actual_write_bytes}
  lineage/io_summary.json -> workload.{distinct_files,read_bytes,write_bytes}
  parallelism_summary.json -> wall_clock_s, total_self_time_s (LLM)

Axes (shared origin = base, the 1-trait x 1-cohort run):
  Axis 1 cohort : x = C in {1,2,4,8}  cells base, a1_c2, a1_c4, a1_c8
  Axis 2 trait  : x = T in {1,2,4,8}  cells base, a2_t2, a2_t4, a2_t8

Outputs into <run_dir>/figures/:
  axis1_cohort.png, axis2_trait.png, fanout_tidy.csv

Usage:  python plot_fanout.py <run_dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXIS1 = [("base", 1), ("a1_c2", 2), ("a1_c4", 4), ("a1_c8", 8)]  # x = C cohorts
AXIS2 = [("base", 1), ("a2_t2", 2), ("a2_t4", 4), ("a2_t8", 8)]  # x = T traits


def jload(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _dataset_io(cell_dir: Path) -> tuple[float, float]:
    """(unique_bytes, total_read_bytes) for input-DATASET files, from
    artifacts.csv. unique = Σ each distinct file's true size (counted once);
    total_read = Σ bytes actually read (with re-read amplification). The ratio
    total/unique is the read-amplification factor.
    """
    f = cell_dir / "lineage" / "artifacts.csv"
    uniq = total = 0.0
    try:
        for r in csv.DictReader(f.open()):
            if r.get("category") == "dataset":
                # true_size = real on-disk file size (counted once); size_bytes
                # falls back to total read for un-statted inputs, so use true_size.
                uniq += float(r.get("true_size") or 0)
                total += float(r.get("total_read_bytes") or 0)
    except Exception:
        pass
    return uniq, total


def cell_metrics(run: Path, cell: str) -> dict | None:
    d = run / cell
    p1 = jload(d / "phase1_metrics.json")
    lin = jload(d / "lineage" / "io_summary.json")
    par = jload(d / "parallelism_summary.json")
    if not p1 and not lin:
        return None
    wl = lin.get("workload", {})
    r = p1.get("metadata_data_ratio", {})
    o = p1.get("analytical_optimum_amplification", {})
    im = p1.get("interface_mix", {}) or {}
    uniq_in, total_in = _dataset_io(d)
    return {
        "generated_files": o.get("actual_generated_files"),
        "distinct_files": wl.get("distinct_files"),
        "metadata_ops": r.get("storage_metadata_ops"),
        "data_ops": r.get("data_ops"),
        "metadata_to_data": r.get("storage_metadata_to_data_ops"),
        # Input data, split: unique footprint (should grow monotonically with
        # C/T) vs total read (= unique × re-read amplification).
        "unique_input_MB": uniq_in / 1e6,
        "input_read_MB": total_in / 1e6,
        "read_amp": (total_in / uniq_in) if uniq_in else None,
        "read_MB": (wl.get("read_bytes", 0) or 0) / 1e6,      # all-category total
        "write_MB": (wl.get("write_bytes", 0) or 0) / 1e6,
        # Two times: wall (elapsed) and total_work (ΣLLM+ΣTool+residual, the
        # worker-robust "total time" — same number as the donut center).
        "wall_s": par.get("wall_clock_s"),
        "total_work_s": par.get("total_work_s", par.get("total_self_time_s")),
        # I/O-abstraction mix (H1): how many code-exec calls did file I/O, and
        # what fraction stayed in stdio-only vs. reached a structured format.
        "code_execs": im.get("total_execs"),
        "io_execs": im.get("execs_with_file_io"),
        "pct_stdio_only": im.get("pct_stdio_only"),
        "pct_structured_any": im.get("pct_structured_any"),
    }


def collect(run: Path, axis: list[tuple[str, int]]) -> tuple[list[int], dict[str, list]]:
    xs: list[int] = []
    series: dict[str, list] = {k: [] for k in
                               ("generated_files", "distinct_files", "metadata_ops",
                                "data_ops", "metadata_to_data",
                                "unique_input_MB", "input_read_MB", "read_amp",
                                "read_MB", "write_MB",
                                "wall_s", "total_work_s",
                                "code_execs", "io_execs",
                                "pct_stdio_only", "pct_structured_any")}
    for cell, x in axis:
        m = cell_metrics(run, cell)
        if m is None:
            continue
        xs.append(x)
        for k in series:
            series[k].append(m[k])
    return xs, series


def _panel(ax, xs, ys, title, ylabel, ref_linear=False):
    pts = [(x, y) for x, y in zip(xs, ys) if isinstance(y, (int, float))]
    if not pts:
        ax.set_title(f"{title}\n(no data)")
        return
    px, py = zip(*pts)
    ax.plot(px, py, "o-", color="#1f77b4", lw=2, ms=7)
    if ref_linear and py[0] is not None and px[0]:
        # dashed line = perfectly linear scaling from the first point (slope = y0/x0)
        slope = py[0] / px[0]
        ax.plot(px, [slope * x for x in px], "--", color="gray", lw=1,
                label="linear ∝ x")
        ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(px)
    ax.grid(alpha=0.3)


def make_figure(run: Path, axis_name: str, xlabel: str,
                axis: list[tuple[str, int]], out: Path):
    xs, s = collect(run, axis)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"{axis_name}  (GEO-only, gpt-5-nano, quick-test)", fontsize=12)
    _panel(axes[0, 0], xs, s["generated_files"],
           "Generated files", "files", ref_linear=True)
    _panel(axes[0, 1], xs, s["unique_input_MB"],
           "Unique input data (footprint)", "MB", ref_linear=True)
    _panel(axes[1, 0], xs, s["metadata_ops"],
           "Storage metadata ops", "ops", ref_linear=True)
    _panel(axes[1, 1], xs, s["read_amp"],
           "Input read amplification (total / unique)", "×")
    for ax in axes.flat:
        ax.set_xlabel(xlabel, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return xs, s


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python plot_fanout.py <run_dir>", file=sys.stderr)
        return 2
    run = Path(sys.argv[1]).resolve()
    figs = run / "figures"
    figs.mkdir(exist_ok=True)

    x1, s1 = make_figure(run, "Axis 1 — cohort scaling", "C = #cohorts (1 trait)",
                         AXIS1, figs / "axis1_cohort.png")
    x2, s2 = make_figure(run, "Axis 2 — trait fanout", "T = #single-cohort traits",
                         AXIS2, figs / "axis2_trait.png")

    # Tidy long CSV (one row per axis-point).
    tidy = figs / "fanout_tidy.csv"
    cols = ["axis", "x"] + list(s1.keys())
    with tidy.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, xs, s in (("cohort", x1, s1), ("trait", x2, s2)):
            for i, x in enumerate(xs):
                w.writerow([name, x] + [s[k][i] for k in s])

    print(f"Wrote {figs}/axis1_cohort.png")
    print(f"Wrote {figs}/axis2_trait.png")
    print(f"Wrote {tidy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
