#!/usr/bin/env python3
"""Write a structured manifest.json for one trace run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


def sh(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout
        )
    except Exception:
        return None


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lustre_snapshot(mount_path: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": sh(["which", "lctl"]) is not None,
        "mount_path": mount_path,
    }
    if not out["available"]:
        return out
    if mount_path:
        out["stripe"] = sh(["lfs", "getstripe", mount_path])
    params = [
        "llite.*.max_read_ahead_mb",
        "llite.*.statahead_max",
        "mdc.*.stats",
        "osc.*.stats",
    ]
    out["params"] = {
        p: sh(["lctl", "get_param", p], timeout=3.0)
        for p in params
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--task-id", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--api", default=None)
    p.add_argument("--temperature", default=os.environ.get("GENOMAS_TEMPERATURE"))
    p.add_argument("--seed", default=os.environ.get("GENOMAS_SEED"))
    p.add_argument("--replay-mode", default=os.environ.get("GENOMAS_LLM_REPLAY", "0"))
    p.add_argument("--llm-cache-path", default=os.environ.get("GENOMAS_LLM_CACHE_PATH"))
    p.add_argument("--agent-count", type=int, default=None)
    p.add_argument("--client-node", default=socket.gethostname())
    p.add_argument("--pid", type=int, default=None)
    p.add_argument("--genomas-repo", default=os.environ.get("GENOMAS_REPO"))
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    p.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cache-state", default=os.environ.get("CACHE_STATE", "unspecified"))
    p.add_argument("--instrumentation", default=os.environ.get("INSTRUMENTATION_LEVEL", "ebpf"))
    p.add_argument("--prompt-template", type=Path, default=None)
    p.add_argument("--extra-json", default=None,
                   help="Optional JSON object merged into the manifest.")
    args = p.parse_args()

    replay_flag = str(args.replay_mode).lower() in {"1", "true", "yes", "replay"}
    cache_hash = file_sha256(Path(args.llm_cache_path)) if args.llm_cache_path else None
    prompt_hash = file_sha256(args.prompt_template) if args.prompt_template else None

    manifest: dict[str, Any] = {
        "created_at_unix": time.time(),
        "workload": args.workload,
        "task_id": args.task_id,
        "model": args.model,
        "api": args.api,
        "temperature": args.temperature,
        "seed": args.seed,
        "replay_mode": "replay" if replay_flag else "live",
        "llm_cache_path": args.llm_cache_path,
        "cached_response_hash": cache_hash,
        "prompt_template_hash": prompt_hash,
        "agent_count": args.agent_count,
        "client_node": args.client_node,
        "pid": args.pid,
        "genomas_repo": args.genomas_repo,
        "data_dir": args.data_dir,
        "work_dir": args.work_dir,
        "output_dir": args.output_dir,
        "cache_state": args.cache_state,
        "instrumentation_level": args.instrumentation,
        "lustre": lustre_snapshot(args.data_dir),
        "env": {
            "USER": os.environ.get("USER"),
            "SUDO_USER": os.environ.get("SUDO_USER"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
    }
    if args.extra_json:
        try:
            manifest.update(json.loads(args.extra_json))
        except json.JSONDecodeError:
            manifest["extra_json_parse_error"] = args.extra_json

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
