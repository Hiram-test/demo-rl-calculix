"""CalculiX interface: CPS3 (2-D plane stress) / C3D4 (3-D) decks."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import Problem
from .mesher import Mesh


@dataclass
class CcxResult:
    u: np.ndarray            # (n, 3) nodal displacements
    n_equations: int
    wall_s: float
    workdir: str
    jobname: str


def assemble_nodal_forces(mesh: Mesh, problem: Problem) -> np.ndarray:
    """Consistent nodal forces from boundary tractions.

    force = t * facet_measure * (thickness in 2-D), split equally over the
    facet nodes (exact lumping for linear facets).
    """

    F = np.zeros((mesh.n_nodes, 3))
    scale = problem.material.thickness if problem.dim == 2 else 1.0
    bf = mesh.boundary_facets
    mids = mesh.facet_centroids
    meas = mesh.facet_measures
    n_facet_nodes = bf.shape[1]
    for spec in problem.tractions:
        mask = spec.facet_predicate(mids)
        t = np.asarray(spec.value, dtype=float)
        for f_idx in np.nonzero(mask)[0]:
            f_total = t * meas[f_idx] * scale
            for node in bf[f_idx]:
                F[node] += f_total / n_facet_nodes
    return F


def write_inp(path: Path, mesh: Mesh, problem: Problem, heading: str) -> None:
    mat = problem.material
    F = assemble_nodal_forces(mesh, problem)
    loaded = np.nonzero(np.abs(F).sum(axis=1) > 0)[0]
    if len(loaded) == 0:
        raise ValueError("no loaded nodes found")

    etype = "CPS3" if problem.dim == 2 else "C3D4"
    lines: list[str] = []
    lines.append("*HEADING")
    lines.append(heading)
    lines.append("*NODE, NSET=NALL")
    for i, (x, y, z) in enumerate(mesh.nodes, start=1):
        lines.append(f"{i}, {x:.9g}, {y:.9g}, {z:.9g}")
    lines.append(f"*ELEMENT, TYPE={etype}, ELSET=EALL")
    for i, conn in enumerate(mesh.cells + 1, start=1):
        lines.append(f"{i}, " + ", ".join(str(c) for c in conn))
    lines.append("*MATERIAL, NAME=MAT")
    lines.append("*ELASTIC")
    lines.append(f"{mat.E:.6g}, {mat.nu:.6g}")
    lines.append("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT")
    if problem.dim == 2:
        lines.append(f"{mat.thickness:.6g}")
    lines.append("*BOUNDARY")
    any_fixed = False
    for k, con in enumerate(problem.constraints):
        nodes = np.nonzero(con.node_predicate(mesh.nodes))[0]
        if len(nodes) == 0:
            raise ValueError(f"constraint '{con.name}' matched no nodes")
        any_fixed = True
        for n in nodes:
            for dof in con.dofs:
                lines.append(f"{n + 1}, {dof}, {dof}")
    if not any_fixed:
        raise ValueError("no constrained nodes")
    lines.append("*STEP")
    lines.append("*STATIC")
    lines.append("*CLOAD")
    for n in loaded:
        for dof in range(3 if problem.dim == 3 else 2):
            if abs(F[n, dof]) > 0:
                lines.append(f"{n + 1}, {dof + 1}, {F[n, dof]:.9g}")
    lines.append("*NODE FILE")
    lines.append("U")
    lines.append("*END STEP")
    Path(path).write_text("\n".join(lines) + "\n")


_EQ_RE = re.compile(r"number of equations\s*\n?\s*(\d+)", re.IGNORECASE)


def default_ccx_cmd() -> str:
    """Prefer CCX_CMD, then the in-repo wrapper, then PATH."""

    env = os.environ.get("CCX_CMD")
    if env:
        return env
    bundled = Path(__file__).resolve().parents[1] / "tools" / "ccx"
    if bundled.exists():
        return str(bundled)
    return "ccx"


def run_ccx(
    workdir: Path,
    jobname: str,
    *,
    ccx_cmd: str | None = None,
    timeout: float = 600.0,
) -> tuple[str, int]:
    """Run CalculiX; return (stdout, n_equations)."""

    ccx = ccx_cmd or default_ccx_cmd()
    t0 = time.perf_counter()
    proc = subprocess.run(
        [ccx, "-i", jobname],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2")},
    )
    wall = time.perf_counter() - t0
    out = proc.stdout + proc.stderr
    frd = Path(workdir) / f"{jobname}.frd"
    if proc.returncode != 0 or not frd.exists():
        raise RuntimeError(
            f"ccx failed (rc={proc.returncode}, {wall:.1f}s) in {workdir}:\n{out[-2000:]}"
        )
    m = _EQ_RE.search(out)
    n_eq = int(m.group(1)) if m else -1
    (Path(workdir) / f"{jobname}.log").write_text(out)
    return out, n_eq


def read_frd_displacements(frd_path: Path, n_nodes: int) -> np.ndarray:
    """Parse the last DISP dataset of a CalculiX FRD file (fixed-width)."""

    u = np.zeros((n_nodes, 3))
    in_disp = False
    found = False
    with open(frd_path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith(" -4") and "DISP" in line:
                in_disp = True
                found = True
                u[:] = 0.0
                continue
            if not in_disp:
                continue
            if line.startswith(" -5"):
                continue
            if line.startswith(" -3"):
                in_disp = False
                continue
            if line.startswith(" -1"):
                try:
                    node = int(line[3:13])
                    vals = [
                        float(line[13 + 12 * k : 25 + 12 * k]) for k in range(3)
                    ]
                except ValueError:
                    parts = line.split()
                    node = int(parts[1])
                    vals = [float(v) for v in parts[2:5]]
                if 1 <= node <= n_nodes:
                    u[node - 1] = vals
    if not found:
        raise RuntimeError(f"no DISP dataset in {frd_path}")
    return u


def solve(
    mesh: Mesh,
    problem: Problem,
    workdir: Path,
    jobname: str,
    *,
    heading: str = "",
    ccx_cmd: str | None = None,
    timeout: float = 600.0,
) -> CcxResult:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    write_inp(workdir / f"{jobname}.inp", mesh, problem, heading or jobname)
    t0 = time.perf_counter()
    _, n_eq = run_ccx(workdir, jobname, ccx_cmd=ccx_cmd, timeout=timeout)
    wall = time.perf_counter() - t0
    u = read_frd_displacements(workdir / f"{jobname}.frd", mesh.n_nodes)
    return CcxResult(u=u, n_equations=n_eq, wall_s=wall, workdir=str(workdir), jobname=jobname)
