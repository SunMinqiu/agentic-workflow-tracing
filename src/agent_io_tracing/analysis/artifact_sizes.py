#!/usr/bin/env python3
"""Generate artifact_sizes.json for a trace by statting the real files.

Run this ON THE NODE where the workload's data still lives (i.e. right after a
trace finishes, before the CloudLab allocation is torn down). It reads the
trace's parsed.json, collects every absolute file path that had a read/write,
stats the ones that still exist, and writes <trace_dir>/artifact_sizes.json
(a {path: size_bytes} map).

lineage_analyzer.py picks this sidecar up automatically and uses the true
sizes for read-amplification / working-set reuse. Without it, read-only inputs
fall back to count-based reuse only.

Only ABSOLUTE paths are statted: generated files (relative ./output/...) already
have a reliable size = bytes written, so they never need a stat and we avoid
any cwd ambiguity.

Usage (on the node):
    python3 make_artifact_sizes.py <trace_dir>
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

READ_WRITE = {
    "read", "pread64", "readv", "preadv",
    "write", "pwrite64", "writev", "pwritev",
}


def main():
    if len(sys.argv) != 2:
        print("usage: make_artifact_sizes.py <trace_dir>", file=sys.stderr)
        sys.exit(2)
    trace_dir = Path(sys.argv[1]).resolve()
    parsed = trace_dir / "parsed.json"
    if not parsed.is_file():
        print(f"ERROR: {parsed} not found", file=sys.stderr)
        sys.exit(2)

    data = json.load(parsed.open())
    paths = set()
    for e in data.get("fs_entries", []):
        if e.get("syscall") not in READ_WRITE:
            continue
        p = e.get("path") or ""
        if p.startswith("/"):  # absolute only
            paths.add(p)

    sizes = {}
    missing = 0
    for p in sorted(paths):
        try:
            sizes[p] = os.path.getsize(p)
        except OSError:
            missing += 1

    out = trace_dir / "artifact_sizes.json"
    with out.open("w") as f:
        json.dump(sizes, f, indent=2)
    print(f"Wrote {out}")
    print(f"  statted {len(sizes)} file(s); {missing} absolute path(s) no longer exist")


if __name__ == "__main__":
    main()
