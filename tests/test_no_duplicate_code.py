"""Fail the build when a new copy-paste or a new orphan lands in src/.

Two guards, both with an explicit allowlist. Adding a workflow should mean
importing the shared helper, not pasting it — and if pasting is genuinely the
right call, the allowlist entry makes that a deliberate, reviewable line.
"""
from __future__ import annotations

import ast
import hashlib
import pathlib
import re
from collections import defaultdict

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
TESTS = pathlib.Path(__file__).resolve().parent
SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"

MIN_NODES = 18

# Structurally identical functions that are allowed to coexist. Each entry is a
# frozenset of "module:function" and needs a reason.
ALLOWED_DUPLICATES = {
    # dataclass accessors: identical shape, unrelated types
    frozenset({
        "agent_io_tracing/parsing/_ebpf_impl.py:to_dict",
        "agent_io_tracing/parsing/tool_log.py:to_dict",
    }),
    frozenset({
        "agent_io_tracing/analysis/parallelism.py:duration_ms",
        "agent_io_tracing/analysis/summary.py:duration_ms",
    }),
}

# Definitions kept on purpose despite having no caller yet.
ALLOWED_UNUSED = {
    "agent_io_tracing/analysis/phase1_metrics.py:_interval_overlap_ms",
    "agent_io_tracing/analysis/phase1_metrics.py:_llm_intervals_ms",
    "agent_io_tracing/parsing/_ebpf_impl.py:PURE_IO_TOOL_NAMES",
}


def _rel(path: pathlib.Path) -> str:
    return str(path.relative_to(SRC))


class _Anonymize(ast.NodeTransformer):
    """Erase identifiers and literals so only control and data shape remain."""

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_arg(self, node):
        node.arg = "_"
        node.annotation = None
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node.attr = "_"
        return node

    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(value=0), node)

    def visit_keyword(self, node):
        self.generic_visit(node)
        node.arg = "_"
        return node


def _shape(node: ast.AST) -> str:
    anonymous = _Anonymize().visit(ast.parse(ast.unparse(node)))
    return ast.dump(anonymous, annotate_fields=False)


def _source_files() -> list[pathlib.Path]:
    return sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


def test_no_new_structurally_identical_functions():
    groups: dict[str, set[str]] = defaultdict(set)
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if sum(1 for _ in ast.walk(node)) < MIN_NODES:
                continue
            try:
                key = hashlib.sha1(_shape(node).encode()).hexdigest()
            except Exception:
                continue
            groups[key].add(f"{_rel(path)}:{node.name}")

    offenders = sorted(
        sorted(members)
        for members in groups.values()
        if len(members) > 1 and frozenset(members) not in ALLOWED_DUPLICATES
    )
    assert not offenders, (
        "duplicated implementations — extract a shared helper, or add the group "
        f"to ALLOWED_DUPLICATES with a reason:\n" + "\n".join(map(str, offenders))
    )


def test_no_new_definitions_without_a_caller():
    text = "\n".join(
        p.read_text(errors="replace")
        for root in (SRC, TESTS, SCRIPTS)
        if root.exists()
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".py", ".sh", ""}
    )

    orphans = []
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [
                    t.id for t in node.targets
                    if isinstance(t, ast.Name) and t.id.isupper()
                ]
            for name in names:
                if name.startswith("__") or name == "main":
                    continue
                qualified = f"{_rel(path)}:{name}"
                if qualified in ALLOWED_UNUSED:
                    continue
                if len(re.findall(rf"\b{re.escape(name)}\b", text)) <= 1:
                    orphans.append(qualified)

    assert not orphans, (
        "defined but never referenced — delete it, or add it to ALLOWED_UNUSED "
        "with a reason:\n" + "\n".join(sorted(orphans))
    )
