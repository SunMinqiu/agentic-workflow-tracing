#!/usr/bin/env python3
"""I/O abstraction classification for agent-generated code (Phase 1 §3.3).

This is the premise-critical piece behind hypothesis **H1 (interface choice)**:
does the agent reach for STDIO / per-file text I/O where a structured format
(HDF5/AnnData/Parquet/Zarr) or MPI-IO would be the HPC-appropriate choice?

Given a Python code string (the snippet GenoMAS hands to `CodeExecutor.execute`),
we classify the I/O *layer(s)* it uses into the taxonomy from the plan:

  - ``stdio``        : buffered Python file I/O / text tabular formats
                       (``open()``, ``json``, ``csv``, ``pandas.read_csv/to_csv``)
  - ``posix_raw``    : unbuffered POSIX syscall I/O (``os.open/os.read/os.write``)
                       or shell file tools (``cp/cat/grep/awk/sed`` via subprocess)
                       — NOTE: this is *not* O_DIRECT; "direct I/O" means O_DIRECT
                       specifically, which we do not claim here.
  - ``structured``   : HDF5/AnnData/netCDF/Zarr/Parquet/sqlite/duckdb/.npy
  - ``mpiio``        : ``mpi4py`` ``MPI.File`` collective/independent I/O
  - ``vector_index`` : FAISS / Chroma / Qdrant vector stores

A single snippet can touch several layers; we report all detected, each with the
concrete *signals* (the call/import that triggered it) so a human can audit.

Primary detection is AST-based (robust to formatting and resolves import aliases).
If the snippet does not parse (agent code is often a fragment), we fall back to a
regex scan. Shell commands embedded as string arguments to ``subprocess`` /
``os.system`` are scanned for the classic POSIX file tools.

Usage as a library::

    from io_api_classifier import classify_code
    result = classify_code(code_string)
    # result["layers"] -> ["stdio", "structured"]
    # result["signals"] -> {"stdio": ["open()", "json.load"], ...}

Usage as a CLI (aggregate a whole run)::

    python io_api_classifier.py <trace_dir>/generated_code.jsonl
    python io_api_classifier.py <dir_of_.py_files>
    python io_api_classifier.py --self-test
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# --- Layer constants ------------------------------------------------------

STDIO = "stdio"
POSIX_RAW = "posix_raw"
STRUCTURED = "structured"
MPIIO = "mpiio"
VECTOR_INDEX = "vector_index"

LAYER_ORDER = (STDIO, POSIX_RAW, STRUCTURED, MPIIO, VECTOR_INDEX)

# --- Rule tables ----------------------------------------------------------
# Module-level intent. Some modules are unambiguous (h5py -> structured), others
# (pandas, numpy, os) depend on which call is used, so they are resolved per-call
# in CALL_RULES below rather than here.

MODULE_LAYER: dict[str, str] = {
    # structured formats
    "h5py": STRUCTURED,
    "tables": STRUCTURED,        # PyTables
    "anndata": STRUCTURED,
    "scanpy": STRUCTURED,
    "netCDF4": STRUCTURED,
    "zarr": STRUCTURED,
    "pyarrow": STRUCTURED,
    "fastparquet": STRUCTURED,
    "duckdb": STRUCTURED,
    "sqlite3": STRUCTURED,
    "sqlalchemy": STRUCTURED,
    # MPI-IO
    "mpi4py": MPIIO,
    # vector / index stores
    "faiss": VECTOR_INDEX,
    "chromadb": VECTOR_INDEX,
    "qdrant_client": VECTOR_INDEX,
    "pinecone": VECTOR_INDEX,
    "weaviate": VECTOR_INDEX,
    # text / stdio formats
    "csv": STDIO,
    "json": STDIO,
    "yaml": STDIO,
    "configparser": STDIO,
}

# Per-(root_module, method) rules. root_module is resolved through the import
# alias map; method is the final attribute in the call chain. None root_module
# means "bare call" (e.g. builtin open()).
CALL_RULES: dict[tuple[str | None, str], str] = {
    # builtin
    (None, "open"): STDIO,
    # pandas — text vs structured split (this is the H1 money rule)
    ("pandas", "read_csv"): STDIO,
    ("pandas", "to_csv"): STDIO,
    ("pandas", "read_table"): STDIO,
    ("pandas", "read_json"): STDIO,
    ("pandas", "to_json"): STDIO,
    ("pandas", "read_fwf"): STDIO,
    ("pandas", "read_parquet"): STRUCTURED,
    ("pandas", "to_parquet"): STRUCTURED,
    ("pandas", "read_hdf"): STRUCTURED,
    ("pandas", "to_hdf"): STRUCTURED,
    ("pandas", "HDFStore"): STRUCTURED,
    ("pandas", "read_feather"): STRUCTURED,
    ("pandas", "to_feather"): STRUCTURED,
    ("pandas", "read_orc"): STRUCTURED,
    ("pandas", "read_sql"): STRUCTURED,
    ("pandas", "to_sql"): STRUCTURED,
    ("pandas", "read_excel"): STRUCTURED,   # binary container
    ("pandas", "to_excel"): STRUCTURED,
    # numpy — binary array formats are "structured"; raw bytes too
    ("numpy", "save"): STRUCTURED,
    ("numpy", "savez"): STRUCTURED,
    ("numpy", "savez_compressed"): STRUCTURED,
    ("numpy", "load"): STRUCTURED,
    ("numpy", "memmap"): STRUCTURED,
    ("numpy", "fromfile"): STRUCTURED,
    ("numpy", "tofile"): STRUCTURED,
    ("numpy", "savetxt"): STDIO,            # text output
    ("numpy", "loadtxt"): STDIO,
    ("numpy", "genfromtxt"): STDIO,
    # os — unbuffered raw syscalls only (os.path.* etc. are NOT I/O)
    ("os", "open"): POSIX_RAW,
    ("os", "read"): POSIX_RAW,
    ("os", "write"): POSIX_RAW,
    ("os", "pread"): POSIX_RAW,
    ("os", "pwrite"): POSIX_RAW,
    ("os", "sendfile"): POSIX_RAW,
    ("os", "system"): POSIX_RAW,            # shell out (args scanned separately)
}

# Methods that, regardless of receiver, signal a layer. Used when the receiver is
# an object we cannot statically resolve to a module (e.g. `store = anndata...;
# store.write_h5ad(...)`). Kept deliberately specific to avoid false positives.
METHOD_SIGNALS: dict[str, str] = {
    "read_h5ad": STRUCTURED,
    "write_h5ad": STRUCTURED,
    "read_10x_h5": STRUCTURED,
    "read_10x_mtx": STRUCTURED,
    "to_parquet": STRUCTURED,
    "to_hdf": STRUCTURED,
    "create_dataset": STRUCTURED,   # h5py group.create_dataset
    "create_group": STRUCTURED,
}

# Known workflow-library wrapper functions -> the I/O layer(s) they perform
# *inside their own body*, keyed by bare function name.
#
# Why this table exists: GenoMAS's action-unit prompts deliberately tell the
# agent to call these high-level functions (from tools/preprocess.py and
# tools/statistics.py) instead of writing raw I/O itself. The AST visitor only
# sees the agent's own generated snippet, so a call like
# `geo_get_relevant_filepaths(in_cohort_dir)` looks like "no I/O" even though
# the function does a real `os.listdir()` underneath. On a real CloudLab run
# (2026-06-30) this made `interface_mix` report only 1 of 12 code-exec
# snippets as touching the filesystem, when in fact ~11 delegated I/O through
# these wrappers. This table was built by reading each function body once
# (not inferred at runtime) and records the layer(s) each one actually uses.
# Functions with no I/O of their own (pure dataframe/array manipulation, e.g.
# `handle_missing_values`, `tune_hyperparameters`, `detect_batch_effect`) are
# intentionally absent — they should continue to classify as "no file I/O".
LIBRARY_FUNCS: dict[str, tuple[str, ...]] = {
    # tools/preprocess.py
    "geo_get_relevant_filepaths": (POSIX_RAW,),      # os.listdir
    "tcga_get_relevant_filepaths": (POSIX_RAW,),     # os.listdir
    "line_generator": (STDIO,),                      # gzip.open + line iteration
    "filter_content_by_prefix": (STDIO,),             # delegates to line_generator;
                                                      # internal pd.read_csv reads an
                                                      # in-memory io.StringIO, not disk
    "get_background_and_clinical_data": (STDIO,),     # delegates to filter_content_by_prefix
    "get_gene_annotation": (STDIO,),                  # delegates to filter_content_by_prefix
    "get_genetic_data": (STDIO,),                     # gzip.open scan + pd.read_csv(file_path)
    "normalize_gene_symbols_in_index": (STDIO,),      # open("./metadata/gene_synonym.json")
    "validate_and_save_cohort_info": (STDIO, POSIX_RAW),  # open/json.dump + os.replace/
                                                           # os.makedirs/fcntl.flock
    # tools/statistics.py
    "read_json_to_dataframe": (STDIO,),               # open + json.load
    "filter_and_rank_cohorts": (STDIO,),               # delegates to read_json_to_dataframe
    "select_and_load_cohort": (STDIO,),                # delegates + direct pd.read_csv(path)
    "get_known_related_genes": (STDIO,),               # open + json.load
    "get_gene_regressors": (STDIO,),                   # delegates to get_known_related_genes
    "save_result": (STDIO, POSIX_RAW),                 # os.makedirs + open/json.dump
}

# Shell file tools — if any appears as the program token of a subprocess/os.system
# command string, we mark posix_raw.
SHELL_FILE_TOOLS = {
    "cp", "mv", "rm", "cat", "tac", "head", "tail", "grep", "egrep", "fgrep",
    "awk", "gawk", "sed", "cut", "sort", "uniq", "tr", "tee", "split", "wc",
    "ls", "find", "touch", "mkdir", "rmdir", "ln", "dd", "gzip", "gunzip",
    "tar", "zcat",
}

SUBPROCESS_FUNCS = {"run", "Popen", "call", "check_call", "check_output"}

# --- Regex fallback patterns (only used when AST parse fails) --------------

_REGEX_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bh5py\b"), STRUCTURED, "h5py"),
    (re.compile(r"\b(anndata|scanpy|sc)\.(read|write)_h5ad\b"), STRUCTURED, "h5ad"),
    (re.compile(r"\.h5ad\b"), STRUCTURED, ".h5ad literal"),
    (re.compile(r"\bto_parquet\b|\bread_parquet\b"), STRUCTURED, "parquet"),
    (re.compile(r"\bto_hdf\b|\bread_hdf\b|\bHDFStore\b"), STRUCTURED, "hdf"),
    (re.compile(r"\bnetCDF4\b|\bzarr\b|\bpyarrow\b|\bduckdb\b"), STRUCTURED, "structured-mod"),
    (re.compile(r"\bmpi4py\b|\bMPI\.File\b"), MPIIO, "mpi-io"),
    (re.compile(r"\bfaiss\b|\bchromadb\b|\bqdrant\b"), VECTOR_INDEX, "vector-store"),
    (re.compile(r"\bos\.(open|read|write|pread|pwrite)\b"), POSIX_RAW, "os raw syscall"),
    (re.compile(r"\b(read_csv|to_csv|read_table)\b"), STDIO, "pandas csv"),
    (re.compile(r"\bjson\.(load|dump)s?\b"), STDIO, "json"),
    (re.compile(r"\bcsv\.(reader|writer|DictReader|DictWriter)\b"), STDIO, "csv module"),
    (re.compile(r"(?<![\w.])open\s*\("), STDIO, "open()"),
]

# Regex fallback for the LIBRARY_FUNCS table above (same rationale, used only
# when the snippet fails to AST-parse).
_LIBRARY_FUNC_REGEX_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b" + re.escape(name) + r"\s*\("), layer, f"{name}() [library]")
    for name, layers in LIBRARY_FUNCS.items()
    for layer in layers
]


# --- AST visitor ----------------------------------------------------------


def _build_alias_map(tree: ast.AST) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """Resolve import aliases.

    Returns:
      module_alias: local name -> canonical top-level module
                    (``import pandas as pd`` -> {"pd": "pandas"};
                     ``import os`` -> {"os": "os"}).
      name_binding: local name -> (module, original_name) for from-imports
                    (``from anndata import read_h5ad`` ->
                     {"read_h5ad": ("anndata", "read_h5ad")}).
    """
    module_alias: dict[str, str] = {}
    name_binding: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                local = alias.asname or alias.name.split(".")[0]
                module_alias[local] = top
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            for alias in node.names:
                local = alias.asname or alias.name
                name_binding[local] = (mod, alias.name)
    return module_alias, name_binding


def _dotted_path(func: ast.AST) -> list[str]:
    """Flatten an attribute/name call target into a dotted path list.

    ``pd.read_csv`` -> ["pd", "read_csv"]; ``np.lib.npyio.save`` ->
    ["np","lib","npyio","save"]; ``open`` -> ["open"].
    """
    parts: list[str] = []
    cur = func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    return parts


def _scan_shell_string(s: str) -> bool:
    """True if a command string's first token (after env-assignments / pipes)
    is a known POSIX file tool."""
    # Split on shell operators to inspect each segment's program token.
    for segment in re.split(r"[|;&]+|\$\(|\)|`", s):
        toks = segment.strip().split()
        i = 0
        # skip leading VAR=val assignments
        while i < len(toks) and re.match(r"^\w+=", toks[i]):
            i += 1
        if i < len(toks):
            prog = toks[i].split("/")[-1]   # strip path, e.g. /bin/cat -> cat
            if prog in SHELL_FILE_TOOLS:
                return True
    return False


class _IOVisitor(ast.NodeVisitor):
    def __init__(self, module_alias: dict[str, str],
                 name_binding: dict[str, tuple[str, str]]):
        self.module_alias = module_alias
        self.name_binding = name_binding
        self.signals: dict[str, list[str]] = defaultdict(list)

    def _add(self, layer: str, signal: str) -> None:
        if signal not in self.signals[layer]:
            self.signals[layer].append(signal)

    def visit_Call(self, node: ast.Call) -> None:
        path = _dotted_path(node.func)
        if path:
            self._classify_call(path, node)
        # subprocess.run([...]) / os.system("...") shell scanning
        self._maybe_shell(path, node)
        self.generic_visit(node)

    def _classify_call(self, path: list[str], node: ast.Call) -> None:
        method = path[-1]
        head = path[0]

        # Resolve head through alias map.
        if head in self.name_binding:
            # from-imported name, e.g. read_h5ad bound to anndata
            mod, orig = self.name_binding[head]
            root = mod
            method = path[-1] if len(path) > 1 else orig
        else:
            root = self.module_alias.get(head, head)

        # 1) bare builtin (open)
        if len(path) == 1:
            layer = CALL_RULES.get((None, method))
            if layer:
                self._add(layer, f"{method}()")
                return

        # 2) (root_module, method) rule
        layer = CALL_RULES.get((root, method))
        if layer:
            self._add(layer, f"{root}.{method}")
            return

        # 3) module-level intent (any call on a structured/vector/mpi module)
        mod_layer = MODULE_LAYER.get(root)
        if mod_layer and mod_layer not in (STDIO,):
            # Only escalate for non-stdio modules; stdio modules need a concrete
            # call rule (handled above) to avoid e.g. json.dumps-to-string noise.
            self._add(mod_layer, f"{root}.{method}")
            return
        if mod_layer == STDIO and method in {"load", "loads", "dump", "dumps",
                                             "reader", "writer", "DictReader",
                                             "DictWriter"}:
            self._add(STDIO, f"{root}.{method}")
            return

        # 4) receiver-agnostic method signal (e.g. obj.write_h5ad)
        sig = METHOD_SIGNALS.get(method)
        if sig:
            self._add(sig, f".{method}")
            return

        # 5) known workflow-library wrapper (e.g. geo_get_relevant_filepaths()).
        # Matched by bare function name regardless of how it was imported/called,
        # since these are almost always invoked bare in generated code. This
        # attributes I/O the call *delegates to* inside the library, which the
        # AST visitor cannot see by construction (it only visits this snippet).
        lib_layers = LIBRARY_FUNCS.get(method)
        if lib_layers:
            for layer in lib_layers:
                self._add(layer, f"{method}() [library]")

    def _maybe_shell(self, path: list[str], node: ast.Call) -> None:
        method = path[-1] if path else ""
        head = path[0] if path else ""
        root = self.module_alias.get(head, head)
        is_subprocess = (root == "subprocess" and method in SUBPROCESS_FUNCS)
        is_os_system = (root == "os" and method == "system")
        if not (is_subprocess or is_os_system):
            return
        for arg in node.args:
            for s in _string_literals(arg):
                if _scan_shell_string(s):
                    self._add(POSIX_RAW, f"shell:{s.strip().split()[0][:24]}")
                    return


def _string_literals(node: ast.AST) -> Iterable[str]:
    """Yield string literals from a node: a bare Constant, or the elements of a
    list/tuple (subprocess arg vectors)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, (ast.List, ast.Tuple)):
        toks = [e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if toks:
            yield " ".join(toks)


# --- Public API -----------------------------------------------------------


def _regex_fallback(code: str) -> dict[str, list[str]]:
    signals: dict[str, list[str]] = defaultdict(list)
    for pat, layer, label in _REGEX_RULES + _LIBRARY_FUNC_REGEX_RULES:
        if pat.search(code):
            if label not in signals[layer]:
                signals[layer].append(label)
    return signals


def classify_code(code: str) -> dict:
    """Classify a code string into I/O layers.

    Returns a dict with:
      code_sha256 : hash of the snippet (join key / dedup)
      imports     : sorted list of top-level modules imported
      layers      : sorted list of detected layers (LAYER_ORDER)
      signals     : {layer: [concrete signals]}
      parsed      : True if AST parse succeeded, False if regex fallback used
    """
    if not isinstance(code, str):
        code = "" if code is None else str(code)
    sha = hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()

    parsed = True
    signals: dict[str, list[str]]
    imports: list[str] = []
    try:
        tree = ast.parse(code)
        module_alias, name_binding = _build_alias_map(tree)
        imports = sorted(set(module_alias.values()) |
                         {m for m, _ in name_binding.values() if m})
        visitor = _IOVisitor(module_alias, name_binding)
        visitor.visit(tree)
        signals = dict(visitor.signals)
    except SyntaxError:
        parsed = False
        signals = _regex_fallback(code)
        # best-effort import sniff
        for m in re.findall(r"^\s*(?:import|from)\s+([\w.]+)", code, re.MULTILINE):
            top = m.split(".")[0]
            if top not in imports:
                imports.append(top)

    layers = [l for l in LAYER_ORDER if l in signals and signals[l]]
    return {
        "code_sha256": sha,
        "imports": imports,
        "layers": layers,
        "signals": {l: signals[l] for l in layers},
        "parsed": parsed,
    }


# --- Aggregation / CLI ----------------------------------------------------


def _iter_code_records(target: Path) -> Iterable[tuple[str, str]]:
    """Yield (id, code) from a generated_code.jsonl, a single .py, or a dir."""
    if target.is_dir():
        for p in sorted(target.glob("*.py")):
            yield p.name, p.read_text(encoding="utf-8", errors="replace")
        for jf in sorted(target.glob("generated_code.jsonl")):
            yield from _iter_code_records(jf)
        return
    if target.suffix == ".jsonl":
        for i, line in enumerate(target.read_text(encoding="utf-8",
                                                  errors="replace").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = rec.get("code")
            if code is None:
                continue
            yield rec.get("run_id", f"line{i}"), code
        return
    # single file
    yield target.name, target.read_text(encoding="utf-8", errors="replace")


def aggregate(target: Path) -> dict:
    """Roll up classification across a run → the H1 headline numbers."""
    total = 0
    layer_exec_counts: Counter[str] = Counter()   # # execs touching each layer
    combo_counts: Counter[str] = Counter()        # exact layer-set per exec
    stdio_only = 0
    structured_any = 0
    no_io = 0
    parse_failures = 0
    per_role: dict[str, Counter] = defaultdict(Counter)

    for _id, code in _iter_code_records(target):
        total += 1
        res = classify_code(code)
        if not res["parsed"]:
            parse_failures += 1
        layers = res["layers"]
        if not layers:
            no_io += 1
            combo_counts["(no file I/O)"] += 1
            continue
        for l in layers:
            layer_exec_counts[l] += 1
        combo_counts["+".join(layers)] += 1
        if layers == [STDIO]:
            stdio_only += 1
        if STRUCTURED in layers:
            structured_any += 1

    io_execs = total - no_io
    return {
        "total_execs": total,
        "execs_with_file_io": io_execs,
        "execs_no_file_io": no_io,
        "parse_failures": parse_failures,
        "layer_exec_counts": dict(layer_exec_counts),
        "combo_counts": dict(combo_counts),
        # H1 headline ratios (computed over execs that do file I/O)
        "pct_stdio_only": round(100.0 * stdio_only / io_execs, 1) if io_execs else None,
        "pct_structured_any": round(100.0 * structured_any / io_execs, 1) if io_execs else None,
    }


def _print_report(target: Path) -> None:
    agg = aggregate(target)
    print(f"=== I/O abstraction classification: {target} ===")
    print(f"  total code-exec snippets      : {agg['total_execs']}")
    print(f"  with file I/O                 : {agg['execs_with_file_io']}")
    print(f"  no file I/O                   : {agg['execs_no_file_io']}")
    print(f"  AST parse failures (regex fb) : {agg['parse_failures']}")
    print("  per-layer exec counts (an exec may hit several layers):")
    for l in LAYER_ORDER:
        c = agg["layer_exec_counts"].get(l, 0)
        if c:
            print(f"      {l:<13}: {c}")
    print("  layer-combination per exec:")
    for combo, c in sorted(agg["combo_counts"].items(), key=lambda kv: -kv[1]):
        print(f"      {combo:<28}: {c}")
    print("  --- H1 headline (over execs that do file I/O) ---")
    print(f"      STDIO-only execs   : {agg['pct_stdio_only']}%")
    print(f"      structured-any     : {agg['pct_structured_any']}%")


# --- Self-test ------------------------------------------------------------

_SELF_TESTS = [
    ("import pandas as pd\npd.read_csv('a.csv')\npd.to_csv('b.csv')", [STDIO]),
    ("import pandas as pd\npd.read_parquet('a.parquet')", [STRUCTURED]),
    ("import h5py\nf = h5py.File('x.h5','w')\nf.create_dataset('d', data=[1])", [STRUCTURED]),
    ("import anndata as ad\nad.read_h5ad('x.h5ad')", [STRUCTURED]),
    ("import os\nfd = os.open('f', os.O_RDONLY)\nos.read(fd, 10)", [POSIX_RAW]),
    ("import subprocess\nsubprocess.run(['cat','file.txt'])", [POSIX_RAW]),
    ("import os\nos.system('grep foo bar.txt > out.txt')", [POSIX_RAW]),
    ("import json\nwith open('f.json') as fh:\n    json.load(fh)", [STDIO]),
    ("from mpi4py import MPI\nf = MPI.File.Open(MPI.COMM_WORLD, 'd')", [MPIIO]),
    ("import faiss\nidx = faiss.read_index('i.faiss')", [VECTOR_INDEX]),
    ("import pandas as pd\npd.read_csv('a.csv')\nimport h5py\nh5py.File('b.h5')", [STDIO, STRUCTURED]),
    ("x = 1 + 2\nprint(x)", []),
    ("def broken(:\n  pd.read_csv('a.csv')", [STDIO]),  # syntax error -> regex fb
    # --- library-wrapper delegation (GenoMAS tools/preprocess.py, tools/statistics.py) ---
    ("soft_file_path, matrix_file_path = geo_get_relevant_filepaths(in_cohort_dir)", [POSIX_RAW]),
    ("clinical_file_path, genetic_file_path = tcga_get_relevant_filepaths(cohort_dir)", [POSIX_RAW]),
    ("background_info, clinical_data = get_background_and_clinical_data(matrix_file_path)", [STDIO]),
    ("gene_data = get_genetic_data(matrix_file_path)", [STDIO]),
    ("gene_data = normalize_gene_symbols_in_index(gene_data)", [STDIO]),
    ("validate_and_save_cohort_info(True, cohort, json_path, True, True, is_biased, df)",
     [STDIO, POSIX_RAW]),
    ("save_result(significant_genes, performance, output_root, trait, condition)",
     [STDIO, POSIX_RAW]),
    ("trait_data, condition_data, regs = select_and_load_cohort(data_root, trait)", [STDIO]),
    # pure-compute library calls should NOT be classified as I/O
    ("best_config, best_perf = tune_hyperparameters(Lasso, param_values, X, Y, names, trait, gp)", []),
    ("has_batch_effect = detect_batch_effect(X)", []),
]


def _self_test() -> int:
    failures = 0
    for code, expected in _SELF_TESTS:
        got = classify_code(code)["layers"]
        ok = got == expected
        if not ok:
            failures += 1
        print(f"[{'OK ' if ok else 'FAIL'}] expected={expected} got={got}  :: "
              f"{code.splitlines()[0][:40]!r}")
    print(f"\n{len(_SELF_TESTS) - failures}/{len(_SELF_TESTS)} passed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--self-test":
        return _self_test()
    target = Path(argv[0])
    if not target.exists():
        print(f"no such path: {target}", file=sys.stderr)
        return 2
    _print_report(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
