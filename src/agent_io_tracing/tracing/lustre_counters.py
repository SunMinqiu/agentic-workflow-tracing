#!/usr/bin/env python3
"""Sample Lustre lctl counters to JSONL.

This is intentionally best-effort: missing lctl, unavailable server-side
params, or permission errors are recorded in the output instead of failing the
workload. Use it beside eBPF to localize H3 metadata storms to llite/mdc/osc
client counters and, when visible, MDT/OST server counters.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path


DEFAULT_PARAMS = [
    "llite.*.stats",
    "llite.*.read_ahead_stats",
    "mdc.*.stats",
    "mdc.*.rpc_stats",
    "osc.*.stats",
    "osc.*.rpc_stats",
    "mdt.*.md_stats",
    "mdt.*.job_stats",
    "obdfilter.*.stats",
    "ost.*.ost_io.stats",
]


def get_param(pattern: str, timeout: float) -> dict:
    try:
        proc = subprocess.run(
            ["lctl", "get_param", pattern],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except FileNotFoundError as e:
        return {"ok": False, "error": str(e), "missing_lctl": True}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=0.0,
                   help="0 means run until SIGINT/SIGTERM.")
    p.add_argument("--param", action="append", default=[],
                   help="lctl get_param pattern; may be repeated.")
    p.add_argument("--timeout", type=float, default=2.0)
    args = p.parse_args()

    params = args.param or DEFAULT_PARAMS
    running = True

    def stop(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with args.output.open("w", encoding="utf-8") as f:
        while running:
            now = time.time()
            rec = {
                "timestamp_unix": now,
                "elapsed_s": now - start,
                "params": {pat: get_param(pat, args.timeout) for pat in params},
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if args.duration and now - start >= args.duration:
                break
            time.sleep(max(args.interval, 0.1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
