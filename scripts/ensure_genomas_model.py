#!/usr/bin/env python3
"""Make sure a model name is registered in GenoMAS's MODEL_INFO registry.

GenoMAS validates ``--model`` against a hard-coded dict in ``utils/llm.py`` and
raises "Could not detect a provider" for anything absent from it.  A locally
served vLLM model is never in that dict, so it has to be inserted.

Two facts make this fragile if done at deploy time only:

  * the registry lives inside GenoMAS's own checkout, and a re-clone silently
    reverts the insertion along with every other local patch;
  * the served model name comes from ``VLLM_SERVED_MODEL`` and is therefore only
    known when a run starts, not when the client is deployed.

So the trace scripts call this helper during preflight, and the deploy script
calls it too.  It is idempotent: already-registered models are a no-op.

Every write is verified by re-parsing the file and confirming the key really
landed under ``MODEL_INFO['openai']``.  If anything looks wrong the original
file is restored, so a failed edit can never leave GenoMAS unimportable.

Usage:  ensure_genomas_model.py --repo ~/GenoMAS --model Qwen3.6-27B
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ANCHOR = "    'openai': {\n"
PREFIX = "[genomas-model]"


def _model_info(source: str) -> dict[str, list[str]]:
    """Return {provider: [model names]} as literally written in utils/llm.py."""
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "MODEL_INFO" not in names or not isinstance(node.value, ast.Dict):
            continue
        providers: dict[str, list[str]] = {}
        for provider_key, provider_value in zip(node.value.keys, node.value.values):
            if not isinstance(provider_key, ast.Constant) or not isinstance(provider_value, ast.Dict):
                continue
            providers[str(provider_key.value)] = [
                str(k.value) for k in provider_value.keys if isinstance(k, ast.Constant)
            ]
        return providers
    raise ValueError("no MODEL_INFO assignment found")


def ensure(repo: Path, model: str) -> int:
    path = repo / "utils" / "llm.py"
    if not path.is_file():
        print(f"{PREFIX} ERROR: {path} not found (is GENOMAS_REPO correct?)", file=sys.stderr)
        return 1

    original = path.read_text(encoding="utf-8")
    try:
        providers = _model_info(original)
    except (SyntaxError, ValueError) as exc:
        print(f"{PREFIX} ERROR: cannot read MODEL_INFO from {path}: {exc}", file=sys.stderr)
        print(f"{PREFIX} the file is corrupt; restore it with: git -C {repo} checkout utils/llm.py", file=sys.stderr)
        return 1

    holders = [p for p, models in providers.items() if model in models]
    if holders:
        print(f"{PREFIX} {model} already registered under {', '.join(holders)}")
        return 0

    if ANCHOR not in original:
        print(f"{PREFIX} ERROR: MODEL_INFO['openai'] anchor not found in {path}", file=sys.stderr)
        print(f"{PREFIX} GenoMAS upstream changed shape; update ANCHOR in this script", file=sys.stderr)
        return 1

    entry = (
        f"        '{model}': {{'input_price': 0.0, 'output_price': 0.0}},"
        f"  # registered by ensure_genomas_model.py (OpenAI-compatible endpoint)\n"
    )
    patched = original.replace(ANCHOR, ANCHOR + entry, 1)

    # Verify before committing the write: the file must still parse, and the key
    # must be reachable under 'openai'.  This is what stops a bad edit from
    # taking GenoMAS down.
    try:
        if model not in _model_info(patched).get("openai", []):
            raise ValueError("key did not land under MODEL_INFO['openai']")
    except (SyntaxError, ValueError) as exc:
        print(f"{PREFIX} ERROR: refusing to write, patched file is invalid: {exc}", file=sys.stderr)
        return 1

    path.write_text(patched, encoding="utf-8")
    reread = path.read_text(encoding="utf-8")
    try:
        if model not in _model_info(reread).get("openai", []):
            raise ValueError("post-write check failed")
    except (SyntaxError, ValueError) as exc:
        path.write_text(original, encoding="utf-8")
        print(f"{PREFIX} ERROR: {exc}; original file restored", file=sys.stderr)
        return 1

    print(f"{PREFIX} registered {model} under MODEL_INFO['openai']")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GenoMAS checkout (GENOMAS_REPO)")
    parser.add_argument("--model", required=True, help="model name passed to main.py --model")
    args = parser.parse_args()
    model = args.model.strip()
    if not model:
        print(f"{PREFIX} ERROR: --model is empty", file=sys.stderr)
        return 1
    return ensure(Path(args.repo).expanduser(), model)


if __name__ == "__main__":
    raise SystemExit(main())
