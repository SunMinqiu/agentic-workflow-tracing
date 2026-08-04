"""Fast, shared lookup tables for workload files and their directories."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable


@dataclass(frozen=True)
class WorkloadPathIndex:
    files: frozenset[str]
    directories: frozenset[str]

    def is_file(self, path: str) -> bool:
        return path in self.files

    def is_directory(self, path: str) -> bool:
        if not path:
            return False
        normalized = path.rstrip("/") or "/"
        return normalized in self.directories or normalized in self.files

    def contains(self, path: str) -> bool:
        return self.is_file(path) or self.is_directory(path)


@lru_cache(maxsize=32)
def _index_for_paths(paths: tuple[str, ...]) -> WorkloadPathIndex:
    files = frozenset(paths)
    directories: set[str] = set()
    for path in files:
        # Keep lexical spellings such as ``./results``. Trace paths and lineage
        # paths use the same spelling, while PurePosixPath would discard ``./``
        # and silently change the old prefix-matching semantics.
        current = path.rstrip("/")
        while "/" in current:
            parent = current.rsplit("/", 1)[0]
            if not parent and current.startswith("/"):
                parent = "/"
            if parent:
                directories.add(parent)
            if parent in {"", "/"}:
                break
            current = parent
    return WorkloadPathIndex(files=files, directories=frozenset(directories))


def workload_path_index(artifacts: Iterable[dict[str, Any]]) -> WorkloadPathIndex:
    """Build or reuse a lexical path index from lineage artifact rows."""
    paths = tuple(sorted({
        str(row["path"])
        for row in artifacts
        if row.get("path")
    }))
    return _index_for_paths(paths)
