#!/usr/bin/env python3
"""
Build a *symlinked* GEO data-root "view" for one fanout experiment cell.

Both experiment axes reduce to the same operation: present GenoMAS a
``--data-root`` whose ``GEO/<trait>/`` subtree contains exactly the cohorts we
want it to process.  GenoMAS discovers cohorts from the filesystem (it globs
``GEO/<trait>/GSE*``), so capping the number of cohort dirs caps the work — no
GenoMAS code change, no task_info.json cohort surgery.

  * Axis 1 (cohort scaling):  one trait, N cohorts        -> "Type_1_Diabetes:8"
  * Axis 2 (trait/fanout):    T traits, 1 cohort each      -> "A:1,B:1,C:1,..."

Cohorts are chosen as the **sorted prefix** of the available GSE dirs, so the
C=1 view is a subset of C=2 is a subset of C=4 ... (nested).  That makes the
"files ~ 3C" / "input read bytes ~ C" scaling purely additive and clean.

The view is built from absolute symlinks, so it costs ~no bytes and the real
data tree on Lustre is never copied or mutated.

Usage:
    python stage_geo_view.py \
        --src-root /mnt/lustrefs/genomas_data \
        --dest     /tmp/fanout_views/a1_c8 \
        --geo      "Type_1_Diabetes:8"

    python stage_geo_view.py --src-root ... --dest ... \
        --geo "Type_1_Diabetes:1,Asthma:1,Obesity:1,Epilepsy:1"

Exit codes: 0 ok; 2 a requested trait/cohort count is not satisfiable.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_geo_spec(spec: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad geo spec part {part!r} (want 'Trait:N')")
        trait, n = part.rsplit(":", 1)
        out.append((trait.strip(), int(n)))
    return out


def cohorts_for_trait(src_geo: Path, trait: str) -> list[Path]:
    d = src_geo / trait
    if not d.is_dir():
        return []
    return sorted([c for c in d.iterdir() if c.is_dir() and c.name.startswith("GSE")])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", type=Path, required=True,
                   help="Real data root containing GEO/ (and optionally TCGA/).")
    p.add_argument("--dest", type=Path, required=True,
                   help="View root to (re)create. Wiped if it exists.")
    p.add_argument("--geo", required=True,
                   help="Comma list of Trait:N (N = number of cohorts to expose).")
    args = p.parse_args()

    src_geo = (args.src_root / "GEO").resolve()
    if not src_geo.is_dir():
        print(f"[stage_geo_view] ERROR: {src_geo} not found", file=sys.stderr)
        return 2

    try:
        wanted = parse_geo_spec(args.geo)
    except ValueError as e:
        print(f"[stage_geo_view] ERROR: {e}", file=sys.stderr)
        return 2

    # Fresh view every time (idempotent re-runs).
    dest = args.dest.resolve()
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "GEO").mkdir(parents=True)
    # GenoMAS main.py unconditionally does os.listdir(<root>/TCGA); for a
    # GEO-only view we still need the dir to exist (empty => no TCGA work).
    (dest / "TCGA").mkdir(parents=True)

    manifest: dict = {"src_root": str(args.src_root.resolve()),
                      "dest": str(dest), "traits": []}
    unsatisfiable = []
    for trait, n in wanted:
        avail = cohorts_for_trait(src_geo, trait)
        if len(avail) < n:
            unsatisfiable.append(
                f"{trait}: requested {n} cohorts, only {len(avail)} available")
        chosen = avail[:n]
        tdir = dest / "GEO" / trait
        tdir.mkdir(parents=True, exist_ok=True)
        for cohort in chosen:
            (tdir / cohort.name).symlink_to(cohort)
        manifest["traits"].append(
            {"trait": trait, "requested": n, "staged": len(chosen),
             "cohorts": [c.name for c in chosen]})

    (dest / "stage_manifest.json").write_text(json.dumps(manifest, indent=2))

    n_traits = len(manifest["traits"])
    n_cohorts = sum(t["staged"] for t in manifest["traits"])
    print(f"[stage_geo_view] staged {n_traits} trait(s), {n_cohorts} cohort(s) "
          f"-> {dest}", file=sys.stderr)
    for t in manifest["traits"]:
        print(f"  {t['trait']}: {t['staged']}/{t['requested']} "
              f"({', '.join(t['cohorts']) or 'NONE'})", file=sys.stderr)

    if unsatisfiable:
        print("[stage_geo_view] ERROR: data prerequisites not met:",
              file=sys.stderr)
        for u in unsatisfiable:
            print(f"  - {u}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
