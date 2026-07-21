#!/usr/bin/env python3
"""Per-run I/O characterization (paper Fig 2 / Fig 3 analogs, scoped to one run).

For each traced cell (one ``parsed.json`` == one run) this emits two figures
into that cell's ``visualizations/`` dir, computed from *that run's own data*:

  file_access_volume.png  — per-file access frequency (# open/openat on the file
                            within the run) x data volume per file  (~= paper Fig 2)
  rw_asymmetry.png        — per-file read-vs-write heatmap (RH/WH/RW) plus the
                            per-file |R-W|/(R+W) distribution  (~= paper Fig 3 a+b)

The paper's cross-system axes (number of *runs* per file; % of *runs*) degenerate
for a single run, so they are replaced by their in-run analogs: number of *opens*
per file, and the distribution over *files*.

File scoping matches the per-run index / phase1 metrics exactly: the set of paths
in the cell's ``lineage/artifacts.csv`` (make_workload_filter). Byte accounting is
kernel-level (POSIX data syscalls) to avoid double-counting the STDIO uprobes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from agent_io_tracing.analysis.phase1_metrics import (
    DATA_SYSCALLS,
    WRITE_SYSCALLS_STRICT,
    _is_python_import_probe,
    is_storage_file_io,
    make_workload_filter,
    read_artifacts,
)
from agent_io_tracing.analysis.size_bins import DARSHAN_SIZE_BINS, DARSHAN_SIZE_LABELS

FREQ_PNG = "file_access_volume.png"
RW_PNG = "rw_asymmetry.png"
CMAP = "Reds"
OPEN_SYSCALLS = {"open", "openat"}
OPEN_CAP = 10  # ">=10" top bin for the per-file open-count axis

_SYS_PREFIXES = ("/dev/", "/proc/", "/sys/", "/run/", "/etc/", "/usr/", "/bin/", "/lib/")
_INTERP_MARKERS = ("/site-packages/", "/.venv/", "/lib/python", "/dist-packages/")


def _entry_bytes(entry: dict[str, Any]) -> int:
    size = entry.get("actual_size") or entry.get("requested_size") or entry.get("bytes_transferred") or 0
    return int(size) if isinstance(size, (int, float)) and size > 0 else 0


def _fallback_ok(path: str | None) -> bool:
    """Coarse guard, applied only when a cell has no artifacts.csv and the
    workload filter degrades to a permissive predicate."""
    if not path or not path.startswith("/"):
        return False
    if path.startswith(_SYS_PREFIXES):
        return False
    if any(m in path for m in _INTERP_MARKERS):
        return False
    if _is_python_import_probe(path):
        return False
    return True


def discover_cells(results_dir: Path, run_roots: list[str] | None) -> list[Path]:
    cells: list[Path] = []
    bases = [results_dir / root for root in run_roots] if run_roots else [results_dir]
    for base in bases:
        if base.is_file() and base.name == "parsed.json":
            cells.append(base.parent)
        elif base.is_dir() and (base / "parsed.json").is_file():
            cells.append(base)
        elif base.exists():
            for parsed in sorted(base.rglob("parsed.json")):
                cells.append(parsed.parent)
    return cells


def aggregate_cell(cell: Path) -> dict[str, dict[str, Any]] | None:
    """Per-file {read_bytes, write_bytes, opens} for one run, workload-scoped."""
    try:
        parsed = json.loads((cell / "parsed.json").read_text(encoding="utf-8"))
    except Exception:
        return None

    artifacts = read_artifacts(cell)
    wl = make_workload_filter(artifacts)
    has_artifacts = bool(artifacts)

    def keep(path: str | None) -> bool:
        if not path or not wl(path):
            return False
        if not has_artifacts and not _fallback_ok(path):
            return False
        return True

    per_file: dict[str, dict[str, Any]] = {}
    for e in parsed.get("fs_entries", []):
        syscall = str(e.get("syscall"))
        path = e.get("path")
        if syscall in OPEN_SYSCALLS:
            if not keep(path):
                continue
            rec = per_file.setdefault(path, {"read_bytes": 0, "write_bytes": 0, "opens": 0})
            rec["opens"] += 1
            continue
        if syscall not in DATA_SYSCALLS or not is_storage_file_io(e):
            continue
        if not keep(path):
            continue
        size = _entry_bytes(e)
        if size <= 0:
            continue
        rec = per_file.setdefault(path, {"read_bytes": 0, "write_bytes": 0, "opens": 0})
        if syscall in WRITE_SYSCALLS_STRICT:
            rec["write_bytes"] += size
        else:
            rec["read_bytes"] += size

    # Only keep files that actually moved bytes (matches the index's I/O-bytes set).
    return {p: r for p, r in per_file.items() if r["read_bytes"] + r["write_bytes"] > 0}


# --------------------------------------------------------------------------- #

def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if n >= 10 or unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.0f}TB"


def _empty(ax_or_fig, out_png: Path, msg: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.set_axis_off()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _size_bin_index(value: float) -> int:
    for idx, (lo, hi) in enumerate(zip(DARSHAN_SIZE_BINS, DARSHAN_SIZE_BINS[1:])):
        if lo <= value < hi:
            return idx
    return len(DARSHAN_SIZE_LABELS) - 1


def plot_file_access_volume(per_file: dict[str, dict[str, Any]], out_png: Path) -> None:
    """~= paper Fig 2, per run: opens on a file (y) x data volume (x)."""
    totals = np.array([r["read_bytes"] + r["write_bytes"] for r in per_file.values()], dtype=float)
    opens = np.array([max(r["opens"], 1) for r in per_file.values()], dtype=int)
    if totals.size == 0:
        _empty(None, out_png, "no workload file I/O in this run")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # Cap the open-count axis at OPEN_CAP with a ">=cap" top bin (mirrors the
    # paper's >=10 row); some workload files (e.g. an appended run log) are
    # reopened thousands of times and would otherwise flatten the whole axis.
    max_op = min(int(opens.max()), OPEN_CAP)
    opens_capped = np.minimum(opens, OPEN_CAP)
    counts = np.zeros((max_op, len(DARSHAN_SIZE_LABELS)), dtype=int)
    for total, op in zip(totals, opens_capped):
        counts[int(op) - 1, _size_bin_index(float(total))] += 1
    masked = np.ma.masked_equal(counts, 0)
    mesh = ax.imshow(masked, origin="lower", aspect="auto", cmap=CMAP,
                     norm=LogNorm(vmin=1, vmax=max(int(counts.max()), 1)))
    ax.set_xticks(np.arange(len(DARSHAN_SIZE_LABELS)))
    ax.set_xticklabels(DARSHAN_SIZE_LABELS, rotation=35, ha="right")
    yticks = list(range(1, max_op + 1))
    ax.set_yticks(np.arange(max_op))
    if max_op == OPEN_CAP:
        labels = [str(t) for t in yticks]
        labels[-1] = f"≥{OPEN_CAP}"
        ax.set_yticklabels(labels)
    else:
        ax.set_yticklabels([str(t) for t in yticks])
    ax.set_xlabel("Data volume per file (read + write)")
    ax.set_ylabel("Times the file was opened in this run")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Number of files")

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rw_asymmetry(per_file: dict[str, dict[str, Any]], out_png: Path) -> None:
    """~= paper Fig 3, per run: (a) read vs write heatmap, (b) per-file
    |R-W|/(R+W) distribution."""
    reads = np.array([r["read_bytes"] for r in per_file.values()], dtype=float)
    writes = np.array([r["write_bytes"] for r in per_file.values()], dtype=float)
    if reads.size == 0:
        _empty(None, out_png, "no workload file I/O in this run")
        return

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    counts = np.zeros((len(DARSHAN_SIZE_LABELS), len(DARSHAN_SIZE_LABELS)), dtype=int)
    for r, w in zip(reads, writes):
        counts[_size_bin_index(float(w)), _size_bin_index(float(r))] += 1
    masked = np.ma.masked_equal(counts, 0)
    mesh = axa.imshow(masked, origin="lower", aspect="auto", cmap=CMAP,
                      norm=LogNorm(vmin=1, vmax=max(int(counts.max()), 1)))
    axa.plot([0, len(DARSHAN_SIZE_LABELS) - 1], [0, len(DARSHAN_SIZE_LABELS) - 1],
             "--", color="0.6", lw=1)
    cbar = fig.colorbar(mesh, ax=axa)
    cbar.set_label("Number of files")
    axa.set_xticks(np.arange(len(DARSHAN_SIZE_LABELS)))
    axa.set_xticklabels(DARSHAN_SIZE_LABELS, rotation=35, ha="right")
    axa.set_yticks(np.arange(len(DARSHAN_SIZE_LABELS)))
    axa.set_yticklabels(DARSHAN_SIZE_LABELS)
    axa.set_xlabel("Data read per file")
    axa.set_ylabel("Data written per file")

    # (b) per-file asymmetry distribution.
    tot = reads + writes
    asym = np.abs(reads - writes)[tot > 0] / tot[tot > 0]
    bins = np.linspace(0, 1, 21)
    weights = np.ones(asym.size) / asym.size * 100.0
    axb.hist(asym, bins=bins, weights=weights, color="#c62828", edgecolor="white", lw=0.4)
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 100)
    axb.set_xlabel(r"$|Read - Write| / (Read + Write)$ per file")
    axb.set_ylabel("PMF (% of files)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run/cell roots under --results. Omit to scan all parsed.json cells.",
    )
    args = ap.parse_args()

    results_dir = args.results.resolve()
    cells = discover_cells(results_dir, args.runs)
    if not cells:
        print(f"no parsed.json found under {results_dir} for runs={args.runs}")
        return 1

    done = skipped = 0
    for cell in cells:
        viz = cell / "visualizations"
        viz.mkdir(parents=True, exist_ok=True)
        per_file = aggregate_cell(cell)
        if per_file is None:
            skipped += 1
            continue
        plot_file_access_volume(per_file, viz / FREQ_PNG)
        plot_rw_asymmetry(per_file, viz / RW_PNG)
        (viz / "per_run_io_char.json").write_text(json.dumps(per_file, indent=1), encoding="utf-8")
        rd = sum(r["read_bytes"] for r in per_file.values())
        wr = sum(r["write_bytes"] for r in per_file.values())
        print(f"  {cell.relative_to(results_dir)}: {len(per_file)} files, "
              f"read={_fmt_bytes(rd)} write={_fmt_bytes(wr)}")
        done += 1

    print(f"per-run figures written for {done} cells ({skipped} skipped, no visualizations/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
