#!/usr/bin/env python3
"""Download and freeze Montage FITS input before tracing starts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path


def _as_status_dict(result: object) -> object:
    """MontagePy bindings return either a dict or its string form (JSON with
    double quotes, or a Python ``str(dict)`` with single quotes).  Normalize to
    a dict so the ``status`` check works regardless of binding style."""
    if isinstance(result, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                return parse(result)
            except (ValueError, SyntaxError, TypeError):
                continue
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(root: Path, manifest: Path) -> int:
    failures = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = _sha256(path) if path.is_file() else "missing"
        if actual != expected:
            print(f"FAILED {relative}", file=sys.stderr)
            failures += 1
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size-deg", type=float, required=True)
    parser.add_argument("--location", default="M 17")
    parser.add_argument("--survey", default="2MASS J")
    args = parser.parse_args(argv)

    output = args.output.expanduser().resolve()
    manifest = output / "input_manifest.sha256"
    if manifest.is_file():
        failures = _verify(output, manifest)
        print(f"Existing fixed input: {output}")
        return 1 if failures else 0
    if output.exists() and any(output.iterdir()):
        print(f"Refusing to replace non-empty input directory: {output}", file=sys.stderr)
        return 2

    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    from MontagePy.archive import mArchiveDownload  # type: ignore[import-not-found]

    result = _as_status_dict(
        mArchiveDownload(args.survey, args.location, args.size_deg, str(raw))
    )
    if not isinstance(result, dict) or str(result.get("status", "")) != "0":
        print(f"Montage download failed: {result!r}", file=sys.stderr)
        return 1
    files = sorted(raw.glob("*.fits"))
    if not files:
        print(f"No FITS files downloaded to {raw}", file=sys.stderr)
        return 1
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(output)}\n" for path in files),
        encoding="utf-8",
    )
    if _verify(output, manifest):
        return 1
    print(f"FIXED_INPUT={output}")
    print(f"FILES={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

