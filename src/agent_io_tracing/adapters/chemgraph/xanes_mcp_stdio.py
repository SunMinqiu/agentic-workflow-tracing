#!/usr/bin/env python3
"""Local stdio MCP server shim for ChemGraph XANES tools.

ChemGraph commit e9e83bc documents ``chemgraph.mcp.xanes_mcp`` in its stdio
example, but only ships a Parsl variant.  For CloudLab tracing we want the
single-structure FDMNES path without Parsl startup noise on stdout, because
stdio MCP requires stdout to contain JSONRPC only.
"""

import asyncio
import json
import os
import pickle
import shlex
import shutil
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from chemgraph.mcp.server_utils import run_mcp_server
from chemgraph.schemas.xanes_schema import (
    mp_query_schema,
    xanes_input_schema,
    xanes_input_schema_ensemble,
)


class xanes_input_schema_ensemble_local(xanes_input_schema_ensemble):
    """CloudLab-friendly ensemble schema with trace/output controls."""

    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory for ensemble outputs. Defaults to a directory next to the "
            "input structures."
        ),
    )
    max_parallel: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of concurrent FDMNES subprocesses. Defaults to "
            "CHEMGRAPH_XANES_MAX_PARALLEL or the local CPU count."
        ),
    )


mcp = FastMCP(
    name="ChemGraph XANES Tools",
    instructions="""
        You expose tools for running XANES/FDMNES simulations.
        Use run_xanes_single for one local structure file, fetch_mp_structures
        for Materials Project retrieval, run_xanes_ensemble for multiple local
        structures, and plot_xanes for completed runs. Keep responses compact
        and return absolute output paths.
    """,
)


@mcp.tool(
    name="run_xanes_single",
    description="Run a single XANES/FDMNES calculation for one input structure.",
)
def run_xanes_single(params: xanes_input_schema):
    from chemgraph.tools.xanes_core import run_xanes_core

    return run_xanes_core(params)


def _collect_structure_files(input_structures: str | list[str]) -> list[Path]:
    if isinstance(input_structures, list):
        paths = [Path(item).expanduser().resolve() for item in input_structures]
    else:
        input_path = Path(input_structures).expanduser().resolve()
        if input_path.is_dir():
            suffixes = {".cif", ".xyz", ".vasp", ".poscar"}
            paths = sorted(
                p
                for p in input_path.iterdir()
                if p.is_file()
                and (
                    p.suffix.lower() in suffixes
                    or p.name.upper().startswith("POSCAR")
                )
            )
        else:
            paths = [input_path]

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Structure file(s) not found: {missing}")
    if not paths:
        raise ValueError("No structure files found for ensemble input.")
    return paths


def _default_ensemble_output_dir(paths: list[Path]) -> Path:
    if len(paths) == 1:
        return paths[0].parent / f"xanes_ensemble_{paths[0].stem}"
    parents = {p.parent for p in paths}
    if len(parents) == 1:
        return next(iter(parents)) / "xanes_ensemble_output"
    return Path.cwd() / "xanes_ensemble_output"


@mcp.tool(
    name="run_xanes_ensemble",
    description=(
        "Run XANES/FDMNES calculations for multiple structures concurrently "
        "on the local node. This CloudLab shim avoids Parsl so process-tree "
        "tracing captures every FDMNES subprocess."
    ),
)
async def run_xanes_ensemble(params: xanes_input_schema_ensemble_local):
    from ase.io import read as ase_read
    from chemgraph.tools.xanes_core import extract_conv, write_fdmnes_input

    structure_paths = _collect_structure_files(params.input_structures)
    output_dir = (
        Path(params.output_dir).expanduser().resolve()
        if params.output_dir
        else _default_ensemble_output_dir(structure_paths).resolve()
    )
    runs_dir = output_dir / "fdmnes_batch_runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    fdmnes_exe = params.fdmnes_exe or os.environ.get("FDMNES_EXE") or "fdmnes"
    env_parallel = int(os.environ.get("CHEMGRAPH_XANES_MAX_PARALLEL", "0") or "0")
    max_parallel = params.max_parallel or env_parallel or (os.cpu_count() or 1)
    max_parallel = max(1, min(max_parallel, len(structure_paths)))
    semaphore = asyncio.Semaphore(max_parallel)

    atoms_for_db = []
    results = []

    async def run_one(index: int, structure_path: Path) -> dict:
        run_dir = runs_dir / f"run_{index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        atoms = ase_read(str(structure_path))
        atoms.info.setdefault("source_file", str(structure_path))
        atoms_for_db.append(atoms)

        z_abs = params.z_absorber or int(max(atoms.get_atomic_numbers()))
        write_fdmnes_input(
            ase_atoms=atoms,
            z_absorber=z_abs,
            input_file_dir=run_dir,
            radius=params.radius,
            magnetism=params.magnetism,
        )

        formula = atoms.get_chemical_formula()
        mp_id = atoms.info.get("MP-id", "local")
        with open(run_dir / f"Z{z_abs}_{mp_id}_{formula}.pkl", "wb") as f:
            pickle.dump(atoms, f)

        async with semaphore:
            with (
                open(run_dir / "fdmnes_stdout.txt", "w") as fp_out,
                open(run_dir / "fdmnes_stderr.txt", "w") as fp_err,
            ):
                proc = await asyncio.create_subprocess_shell(
                    shlex.quote(fdmnes_exe),
                    cwd=str(run_dir),
                    stdout=fp_out,
                    stderr=fp_err,
                )
                return_code = await proc.wait()

        conv_data = extract_conv(run_dir) if return_code == 0 else {}
        result = {
            "index": index,
            "status": "success" if conv_data else "failure",
            "input_structure": str(structure_path),
            "output_dir": str(run_dir),
            "return_code": return_code,
            "n_conv_files": len(conv_data),
            "conv_files": [str(p) for p in sorted(run_dir.glob("*conv.txt"))],
        }
        if return_code != 0:
            result["error"] = f"FDMNES exited with return code {return_code}"
        elif not conv_data:
            result["error"] = "No *conv.txt output files found after FDMNES execution."
        return result

    results = await asyncio.gather(
        *(run_one(index, path) for index, path in enumerate(structure_paths))
    )

    with open(output_dir / "atoms_db.pkl", "wb") as f:
        pickle.dump(atoms_for_db, f)

    expanded_atoms = []
    for result in results:
        pkl_files = list(Path(result["output_dir"]).glob("*.pkl"))
        if not pkl_files:
            continue
        with open(pkl_files[0], "rb") as f:
            atoms = pickle.load(f)
        atoms.info.update({"FDMNES-xanes": extract_conv(result["output_dir"])})
        expanded_atoms.append(atoms)
    with open(output_dir / "atoms_db_expanded.pkl", "wb") as f:
        pickle.dump(expanded_atoms, f)

    result_path = output_dir / "xanes_results.jsonl"
    with open(result_path, "w") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True) + "\n")

    n_success = sum(1 for result in results if result["status"] == "success")
    return {
        "status": "success" if n_success == len(results) else "partial_failure",
        "n_structures": len(results),
        "n_success": n_success,
        "max_parallel": max_parallel,
        "output_dir": str(output_dir),
        "runs_dir": str(runs_dir),
        "results_file": str(result_path),
        "results": results,
    }


@mcp.tool(
    name="fetch_mp_structures",
    description="Fetch optimized structures from Materials Project.",
)
def fetch_mp_structures(params: mp_query_schema):
    from chemgraph.tools.xanes_core import fetch_materials_project_data, _get_data_dir

    data_dir = _get_data_dir()
    result = fetch_materials_project_data(params, data_dir)
    return {
        "status": "success",
        "n_structures": result["n_structures"],
        "chemsys": params.chemsys,
        "output_dir": str(data_dir),
        "structure_files": result["structure_files"],
        "pickle_file": result["pickle_file"],
    }


@mcp.tool(
    name="plot_xanes",
    description="Generate normalized XANES plots for completed FDMNES calculations.",
)
def plot_xanes(runs_dir: str):
    from chemgraph.tools.xanes_core import plot_xanes_results, _get_data_dir

    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        raise ValueError(f"'{runs_dir}' is not a valid directory.")

    result = plot_xanes_results(_get_data_dir(), runs_path)
    return {
        "status": "success",
        "n_plots": result["n_plots"],
        "n_failed": result["n_failed"],
        "plot_files": result["plot_files"],
        "failed": result["failed"],
    }


def main() -> None:
    run_mcp_server(mcp, default_port=9007)


if __name__ == "__main__":
    main()
