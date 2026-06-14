"""Auto-generate a per-phase index.html from a trace cell directory.

Looks for the standard outputs from:
  - visualize_strace.py    → cell/visualizations/*.png|*.html
  - lineage_analyzer.py    → cell/lineage/fig*.png + artifacts.csv
  - custom hand-made       → cell/figures/*.png

And produces cell_parent/index.html with consistent navigation + sections.

Usage:
    python3 build_phase_index.py <phase_dir> [<cell_subdir>]

Example:
    python3 build_phase_index.py GenoMAS_traces/phase6_<ts>/ p6_mw2_full

If <cell_subdir> is omitted, the script auto-picks the only sub-directory
under <phase_dir>.  This matches the Phase-5/6 single-cell convention.
"""
from __future__ import annotations
import sys
from pathlib import Path
from textwrap import dedent


# ----- per-figure descriptions (override per phase by editing the dict) -----

LINEAGE_FIG_INFO = {
    "fig1_size_distribution.png": (
        "File size distribution",
        "Two panels: per-syscall I/O request size (top, for block-size/read-ahead) and "
        "per-FILE artifact size colored by category (bottom, for capacity/tier). Raw "
        "inputs are the 10–100 MB files; intermediates/code/scratch are KB-scale.",
    ),
    "fig2_reader_fanout.png": (
        "Reader fan-out",
        "One bar per named artifact = # distinct CodeExec calls that read it, colored by "
        "category. High fan-out concentrates on shared intermediates (e.g. cohort_info.json), "
        "not raw inputs — those are the caching candidates.",
    ),
    "fig3_staleness_cdf.png": (
        "Write→first-read staleness",
        "One dot per named artifact: gap from first write to first read, log-x, by category. "
        "Coordination files are read back sub-ms (pipeline-tight); data CSVs go cold for "
        "~100s before reuse (tiering candidates).",
    ),
    "fig4_lifecycle.png": (
        "Artifact lifecycle / reclaimable window",
        "One row per artifact (top-N by file size, size labeled on each row), x = time. Solid "
        "bar = still needed (create→last-read); gray hatch = dead weight, reclaimable; ▽ = "
        "write-once leaf, reclaimable on write. Long gray tails on big files are the cleanup targets.",
    ),
    "fig5_artifact_lifecycle.png": (
        "Top-12 artifact lifecycles",
        "▲ = write, ● = read, one row per artifact. Compresses all 4 metrics into a "
        "temporal view of artifact use.",
    ),
}

TOOL_FIG_INFO = {
    "phase_breakdown.png": (
        "Phase breakdown (donut)",
        "LLM time vs tool (CodeExec) time vs unaccounted gap as fraction of e2e wall clock.",
    ),
    "agent_timeline.png": (
        "Agent timeline (fragmented)",
        "Native visualize_strace.py Gantt: one row per CodeExec call with duration labels.",
    ),
    "timeline.png": (
        "Global timeline",
        "All events on one shared time axis.",
    ),
    "process_timeline.png": (
        "Process timeline (by PID)",
        "Per-PID swimlanes from BCC. Visualizes multiprocessing fork pattern.",
    ),
    "io_rate.png": (
        "I/O rate over time",
        "Read/write bytes per time bucket, from BCC syscall trace.",
    ),
}

CUSTOM_FIG_INFO = {
    "fig_per_role_breakdown.png": (
        "Per-role time profile",
        "Stacked LLM (green) + CodeExec (red) per agent role. Reveals two agent classes.",
    ),
    "fig_clean_timeline.png": (
        "Clean 2-lane timeline",
        "All LLM and all CodeExec events aggregated into 2 lanes, colored by role. "
        "Replaces visualize_strace's fragmented per-call rows.",
    ),
}


def render(phase_title: str, cell_subdir: str, meta: str, findings_html: str,
           lineage_cards: str, custom_cards: str, tool_cards: str,
           tool_html_only_cards: str, raw_cards: str,
           accent: str = "#2ca02c") -> str:
    return dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>{phase_title}</title>
            <style>
                * {{ margin:0; padding:0; box-sizing:border-box; }}
                body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                       background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
                       min-height:100vh; color:#eee; }}
                .nav {{ padding:1rem 2rem; background:rgba(0,0,0,0.3); border-bottom:1px solid rgba(255,255,255,0.1); }}
                .nav a {{ color:#7aa2f7; text-decoration:none; margin-right:1rem; }}
                .nav a:hover {{ color:#fff; }}
                .header {{ background:rgba(0,0,0,0.3); padding:2rem; text-align:center; border-bottom:1px solid rgba(255,255,255,0.1); }}
                .header h1 {{ font-size:1.6rem; font-weight:300; color:#fff; }}
                .header p {{ margin-top:0.4rem; color:#aaa; font-size:0.9rem; }}
                .container {{ max-width:1400px; margin:0 auto; padding:2rem; }}
                .findings {{ background:rgba(255,255,255,0.04); border-left:4px solid {accent}; padding:1.2rem 1.5rem; border-radius:0 8px 8px 0; margin-bottom:2rem; }}
                .findings h2 {{ font-size:1.1rem; font-weight:500; color:#fff; margin-bottom:0.7rem; }}
                .findings ul {{ margin-top:0.6rem; padding-left:1.2rem; }}
                .findings li {{ color:#ccc; font-size:0.9rem; margin-bottom:0.3rem; }}
                .findings .number {{ color:{accent}; font-weight:600; font-family:ui-monospace,monospace; }}
                .section {{ margin-bottom:3rem; }}
                .section-title {{ font-size:1.3rem; font-weight:400; margin-bottom:0.4rem; color:#fff; border-left:4px solid #7aa2f7; padding-left:0.8rem; }}
                .section-subtitle {{ font-size:0.85rem; color:#999; margin-bottom:1.2rem; margin-left:1.05rem; }}
                .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(380px,1fr)); gap:1.5rem; }}
                .card {{ background:rgba(255,255,255,0.05); border-radius:12px; overflow:hidden; border:1px solid rgba(255,255,255,0.1); transition:transform 0.2s; }}
                .card:hover {{ transform:translateY(-3px); }}
                .card img {{ width:100%; height:220px; object-fit:contain; background:#fff; border-bottom:1px solid rgba(255,255,255,0.1); }}
                .card.html-only {{ padding:1rem; }}
                .card-body {{ padding:1.1rem 1.25rem; }}
                .card-body h3 {{ font-size:1rem; font-weight:500; margin-bottom:0.5rem; color:#fff; }}
                .card-body p {{ font-size:0.8rem; color:#aaa; margin-bottom:0.75rem; line-height:1.4; }}
                .card-body .links a {{ display:inline-block; padding:0.35rem 0.75rem; background:rgba(122,162,247,0.18); color:#7aa2f7; text-decoration:none; border-radius:6px; font-size:0.78rem; border:1px solid rgba(122,162,247,0.3); margin-right:0.4rem; }}
                .badge {{ display:inline-block; padding:0.15rem 0.5rem; font-size:0.7rem; border-radius:4px; margin-left:0.4rem; vertical-align:middle; }}
                .badge.tool {{ background:#4a4a6a; color:#ccc; }}
                .badge.custom {{ background:{accent}; color:#fff; }}
                code {{ font-family:ui-monospace,monospace; color:#f7768e; background:rgba(0,0,0,0.3); padding:0.1rem 0.3rem; border-radius:3px; font-size:0.85em; }}
            </style>
        </head>
        <body>
            <div class="nav">
                <a href="../index.html">← All experiments</a>
                <a href="#lineage">Storage/lineage</a>
                <a href="#custom-figures">Custom timeline/role</a>
                <a href="#tool-figures">Tool-generated</a>
                <a href="#raw">Raw</a>
            </div>
            <div class="header">
                <h1>{phase_title}</h1>
                <p>{meta}</p>
            </div>
            <div class="container">
                {findings_html}
                <div class="section" id="lineage">
                    <div class="section-title">Storage / lineage characterization</div>
                    <div class="section-subtitle">From <code>lineage_analyzer.py</code> applied to <code>parsed.json</code> + <code>pi_events.jsonl</code>.</div>
                    <div class="grid">{lineage_cards}</div>
                </div>
                <div class="section" id="custom-figures">
                    <div class="section-title">Custom timeline & role figures</div>
                    <div class="section-subtitle">Hand-made for thesis-level findings.</div>
                    <div class="grid">{custom_cards}</div>
                </div>
                <div class="section" id="tool-figures">
                    <div class="section-title">Tool-generated (visualize_strace.py)</div>
                    <div class="section-subtitle">Auto-produced. Same set across phases that ran BCC.</div>
                    <div class="grid">{tool_cards}{tool_html_only_cards}</div>
                </div>
                <div class="section" id="raw">
                    <div class="section-title">Raw artifacts</div>
                    <div class="grid">{raw_cards}</div>
                </div>
            </div>
        </body>
        </html>
    """)


def card_with_img(rel_path: str, title: str, desc: str, badge_kind: str = "custom") -> str:
    return dedent(f"""\
        <div class="card">
            <img src="{rel_path}" alt="{title}">
            <div class="card-body">
                <h3>{title}<span class="badge {badge_kind}">{badge_kind}</span></h3>
                <p>{desc}</p>
                <div class="links"><a href="{rel_path}">PNG</a></div>
            </div>
        </div>
    """)


def card_html_only(title: str, desc: str, rel_path: str, badge_kind: str = "tool") -> str:
    return dedent(f"""\
        <div class="card html-only">
            <div class="card-body">
                <h3>{title}<span class="badge {badge_kind}">{badge_kind}</span></h3>
                <p>{desc}</p>
                <div class="links"><a href="{rel_path}">Open</a></div>
            </div>
        </div>
    """)


def build_section_cards(cell_subdir: str, subdir: str, info: dict, badge_kind: str) -> str:
    """Build cards for every figure in `info` that actually exists on disk."""
    parts = []
    base = Path(cell_subdir) / subdir
    for fname, (title, desc) in info.items():
        full = base / fname
        if full.exists():
            parts.append(card_with_img(f"{cell_subdir}/{subdir}/{fname}", title, desc, badge_kind))
    return "\n".join(parts)


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    phase_dir = Path(sys.argv[1]).resolve()
    cells = [d for d in phase_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    cells = [d for d in cells if d.name not in ("figures",)]
    if len(sys.argv) >= 3:
        cell = phase_dir / sys.argv[2]
    elif len(cells) == 1:
        cell = cells[0]
    else:
        print(f"ERROR: {phase_dir} has multiple cells; specify which", file=sys.stderr)
        for c in cells:
            print(f"  - {c.name}")
        sys.exit(2)

    cell_subdir = cell.name
    phase_title = f"{phase_dir.name} — auto-built dashboard"
    meta = f"cell: <code>{cell_subdir}</code>"

    # Build each section's cards by checking disk
    lineage = build_section_cards(cell_subdir, "lineage", LINEAGE_FIG_INFO, "custom")
    custom = build_section_cards(cell_subdir, "figures", CUSTOM_FIG_INFO, "custom")
    tool = build_section_cards(cell_subdir, "visualizations", TOOL_FIG_INFO, "tool")

    # HTML-only tool outputs (no PNG counterpart)
    tool_html_only = ""
    for fname, title, desc in [
        ("tool_syscalls.html", "Tool–syscall correlation",
         "Interactive HTML only. Hover-able mapping of each tool call → its syscall fingerprint."),
        ("tool_syscall_durations.html", "Tool–syscall durations",
         "Interactive HTML only. Distribution of per-tool syscall durations."),
        ("index.html", "visualize_strace native dashboard",
         "Native dashboard linking all of visualize_strace's outputs."),
    ]:
        if (cell / "visualizations" / fname).exists():
            tool_html_only += card_html_only(title, desc,
                f"{cell_subdir}/visualizations/{fname}", badge_kind="tool")

    # Raw artifacts
    raw = ""
    for fname, title, desc in [
        ("pi_events.jsonl", "pi_events.jsonl", "LLM + CodeExec event log."),
        ("tool_calls.log", "tool_calls.log", "One line per CodeExec call."),
        ("ebpf_events.log", "ebpf_events.log", "Raw BCC syscall trace."),
        ("parallelism_summary.json", "parallelism_summary.json",
         "compute_parallelism.py output."),
        ("lineage/artifacts.csv", "artifacts.csv", "Per-artifact storage/lineage summary."),
        ("lineage/tool_call_attribution.csv", "tool_call_attribution.csv",
         "Per-CodeExec role + duration."),
    ]:
        if (cell / fname).exists():
            raw += card_html_only(title, desc, f"{cell_subdir}/{fname}", badge_kind="tool")

    findings_html = dedent("""\
        <div class="findings">
            <h2>Findings</h2>
            <ul>
                <li><i>Edit this section by hand after build (or pass --findings).</i></li>
            </ul>
        </div>
    """)

    html = render(phase_title, cell_subdir, meta, findings_html,
                  lineage, custom, tool, tool_html_only, raw)
    out_path = phase_dir / "index.html"
    out_path.write_text(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
