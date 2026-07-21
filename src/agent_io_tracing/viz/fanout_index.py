#!/usr/bin/env python3
"""Build a SINGLE self-contained run-level index.html for a fanout run.

One page: cross-cell axis figures + summary table, then one collapsible
(<details>) section per cell that inlines that cell's figure thumbnails. No
need to open a separate per-cell index — expand a cell and its figures are
right there (click a thumbnail for the full PNG / interactive HTML).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


# Display names for known figure files, in preferred display order.
# (filename, display name). PNG -> thumbnail card; .html -> link button.
FIGURE_ORDER = [
    # storage / I/O shape (lineage/)
    ("lineage/fig0_io_volume_summary.png", "I/O Volume Summary"),
    ("lineage/fig1_size_distribution.png", "File and Request Size"),
    ("lineage/fig2_fanout.png", "Reader & Writer Fan-out"),
    ("lineage/fig3_staleness_cdf.png", "Write→Read Gap"),
    ("lineage/fig4_lifecycle.png", "Artifact Lifecycle"),
    # time & attribution (visualizations/ + call_dag)
    ("visualizations/agent_timeline.png", "Agent Timeline"),
    ("visualizations/phase_breakdown.png", "Time Accounting"),
    ("visualizations/measured_interface_layers.png", "Measured I/O Interface Mix"),
    ("visualizations/directory_scan.png", "Directory Re-scans (getdents64)"),
    ("visualizations/inter_arrival_cdf.png", "Inter-arrival Histogram"),
    ("visualizations/reread_attribution.png", "Reread Attribution"),
    ("visualizations/access_pattern.png", "Access Pattern"),
    ("visualizations/io_rate.png", "I/O Rate Over Time"),
    ("visualizations/effective_bandwidth.png", "Effective BW by Phase"),
    ("visualizations/io_autocorrelation.png", "I/O Autocorrelation"),
    ("call_dag.html", "Call DAG with I/O"),
]

DATA_LINKS = [
    "phase1_metrics.json", "lineage/io_summary.json",
    "lineage/artifacts.csv", "lineage/tool_call_attribution.csv",
    "generated_code.jsonl", "manifest.json", "parallelism_summary.json",
]


def jload(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt_num(value, digits: int = 2) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (int, float)):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def fmt_bytes(value) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(x) < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} TB"


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def cell_metrics(cell_dir: Path) -> dict:
    p1 = jload(cell_dir / "phase1_metrics.json")
    lin = jload(cell_dir / "lineage" / "io_summary.json")
    par = jload(cell_dir / "parallelism_summary.json")
    ratios = p1.get("metadata_data_ratio") or {}
    opt = p1.get("analytical_optimum_amplification") or {}
    wl = lin.get("workload") or {}
    return {
        "generated_files": opt.get("actual_generated_files"),
        "metadata_to_data": ratios.get("storage_metadata_to_data_ops"),
        "read_bytes": wl.get("read_bytes"),
        "wall_s": par.get("wall_clock_s"),
        "total_work_s": par.get("total_work_s", par.get("total_self_time_s")),
    }


def tidy_table(path: Path) -> str:
    if not path.is_file():
        return "<p>No figures/fanout_tidy.csv found.</p>"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return "<p>figures/fanout_tidy.csv is empty.</p>"
    def _fmt3(v: str) -> str:
        # Round decimals to 3 places; leave integers and non-numbers as-is.
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v
        return str(int(f)) if f == int(f) else f"{f:.3f}"

    head, body = rows[0], rows[1:]
    thead = "".join(f"<th>{esc(col)}</th>" for col in head)
    trs = []
    for row in body:
        trs.append("<tr>" + "".join(f"<td>{esc(_fmt3(col))}</td>" for col in row) + "</tr>")
    return (f"<table><thead><tr>{thead}</tr></thead>"
            f"<tbody>{''.join(trs)}</tbody></table>")


def cell_section(run_dir: Path, d: Path) -> str:
    """One collapsible <details> per cell with inline figure thumbnails."""
    m = cell_metrics(d)
    summary = (
        f"<summary><b>{esc(d.name)}</b>"
        f"<span class='kv'>files {fmt_num(m['generated_files'], 0)}</span>"
        f"<span class='kv'>meta/data {fmt_num(m['metadata_to_data'])}</span>"
        f"<span class='kv'>read {fmt_bytes(m['read_bytes'])}</span>"
        f"<span class='kv'>wall {fmt_num(m['wall_s'])}s</span>"
        f"<span class='kv'>total {fmt_num(m['total_work_s'])}s</span>"
        "</summary>"
    )

    img_cards, link_btns = [], []
    for rel, title in FIGURE_ORDER:
        fpath = d / rel
        if not fpath.exists():
            continue
        href = f"{d.name}/{rel}"
        if rel.endswith(".png"):
            img_cards.append(
                f"<figure class='fig'><a href='{href}'>"
                f"<img loading='lazy' src='{href}' alt='{esc(title)}'></a>"
                f"<figcaption>{esc(title)}</figcaption></figure>"
            )
        else:  # interactive HTML -> link button
            link_btns.append(f"<a href='{href}'>{esc(title)}</a>")

    # Drill-down data links for this cell.
    for rel in DATA_LINKS:
        if (d / rel).exists():
            link_btns.append(f"<a class='data' href='{d.name}/{rel}'>{esc(rel)}</a>")

    grid = ("<div class='figgrid'>" + "".join(img_cards) + "</div>") if img_cards else \
           "<p class='muted'>No figures yet — run visualize_strace / lineage_analyzer.</p>"
    links = ("<div class='links'>" + "".join(link_btns) + "</div>") if link_btns else ""
    return f"<details class='cell'>{summary}{grid}{links}</details>"


def build_index(run_dir: Path) -> str:
    axis_cards = []
    for fname, title in [
        ("axis1_cohort.png", "Axis 1 — Cohort Scaling"),
        ("axis2_trait.png", "Axis 2 — Trait Fanout"),
    ]:
        rel = f"figures/{fname}"
        if (run_dir / rel).exists():
            axis_cards.append(
                f"<figure class='fig big'><a href='{rel}'>"
                f"<img src='{rel}' alt='{esc(title)}'></a>"
                f"<figcaption>{esc(title)}</figcaption></figure>"
            )

    input_rel = "figures/input_size_distribution.png"
    input_card = ""
    if (run_dir / input_rel).exists():
        input_card = (
            f"<figure class='fig input'><a href='{input_rel}'>"
            f"<img src='{input_rel}' alt='Input Data Size Distribution'></a>"
            f"<figcaption>Input Data Size Distribution</figcaption></figure>"
        )

    cells = [d for d in sorted(run_dir.iterdir())
             if d.is_dir() and d.name != "figures"
             and (d / "phase1_metrics.json").exists()]
    cell_sections = "".join(cell_section(run_dir, d) for d in cells) or \
        "<p class='muted'>No completed cells found.</p>"

    data_links = []
    for rel in ("fanout_summary.csv", "figures/fanout_tidy.csv", "figures/input_files_tidy.csv"):
        if (run_dir / rel).exists():
            data_links.append(f"<a href='{rel}'>{esc(rel)}</a>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GEO Fanout — {esc(run_dir.name)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
            background:#f6f8fb; color:#1f2933; }}
    header {{ padding:24px 32px; background:#fff; border-bottom:1px solid #d9e2ec; }}
    h1 {{ margin:0; font-size:24px; }}
    h2 {{ margin:26px 0 12px; font-size:18px; }}
    main {{ max-width:1380px; margin:0 auto; padding:24px; }}
    .figgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }}
    .fig {{ margin:0; background:#fff; border:1px solid #d9e2ec; border-radius:8px; overflow:hidden; }}
    .fig img {{ width:100%; height:240px; object-fit:contain; background:#fff;
               border-bottom:1px solid #edf2f7; display:block; }}
    .fig.big img {{ height:360px; }}
    .fig.input {{ margin-top:14px; }}
    .fig.input img {{ height:430px; }}
    .fig figcaption {{ padding:8px 10px; font-size:13px; font-weight:600; }}
    details.cell {{ background:#fff; border:1px solid #d9e2ec; border-radius:8px;
                    margin:12px 0; padding:8px 14px; }}
    details.cell > summary {{ cursor:pointer; font-size:16px; padding:6px 0;
                              list-style:none; display:flex; align-items:center; gap:14px; flex-wrap:wrap; }}
    details.cell > summary::-webkit-details-marker {{ display:none; }}
    details.cell[open] > summary {{ border-bottom:1px solid #edf2f7; margin-bottom:12px; }}
    .kv {{ font-size:12px; font-weight:500; color:#52606d; background:#f0f4f8;
           padding:2px 8px; border-radius:10px; }}
    .links {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
    a {{ display:inline-block; padding:6px 10px; border-radius:6px; background:#e6f0ff;
         color:#0b5cad; text-decoration:none; font-size:13px; font-weight:600; }}
    a.data {{ background:#eef1f4; color:#52606d; font-weight:500; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid #d9e2ec;
             border-radius:8px; overflow:hidden; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid #edf2f7; text-align:left; font-size:13px; }}
    th {{ color:#52606d; background:#f0f4f8; }}
    .muted {{ color:#7b8794; }}
  </style>
</head>
<body>
  <header><h1>GEO Fanout — {esc(run_dir.name)}</h1></header>
  <main>
    <h2>Cross-cell scaling</h2>
    <div class="figgrid">{''.join(axis_cards) or "<p class='muted'>No axis figures found.</p>"}</div>
    {input_card}
    <h2>Cross-cell summary</h2>
    {tidy_table(run_dir / "figures" / "fanout_tidy.csv")}
    <h2>Per-cell figures <span class="muted" style="font-size:13px;font-weight:400">(click a cell to expand)</span></h2>
    {cell_sections}
    <h2>Data</h2>
    <div class="links">{''.join(data_links) or "<p class='muted'>No run-level data links.</p>"}</div>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build single-page fanout index.html")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out = run_dir / "index.html"
    out.write_text(build_index(run_dir), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
