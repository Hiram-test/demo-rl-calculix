"""CalculiX interface: plane-stress deck writer, runner, FRD parser."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import Problem
from .mesher import TriMesh


@dataclass
class CcxResult:
    u: np.ndarray            # (n, 2) nodal displacements
    n_equations: int
    wall_s: float
    workdir: str
    jobname: str


def assemble_nodal_forces(mesh: TriMesh, problem: Problem) -> np.ndarray:
    """Consistent nodal forces from boundary tractions (force = t * L * thk)."""

    F = np.zeros((mesh.n_nodes, 2))
    thk = problem.material.thickness
    be = mesh.boundary_edges
    mids = 0.5 * (mesh.nodes[be[:, 0]] + mesh.nodes[be[:, 1]])
    lengths = np.linalg.norm(mesh.nodes[be[:, 0]] - mesh.nodes[be[:, 1]], axis=1)
    for spec in problem.tractions:
        mask = spec.edge_predicate(mids)
        t = np.asarray(spec.value, dtype=float)
        for e_idx in np.nonzero(mask)[0]:
            f_edge = t * lengths[e_idx] * thk
            F[be[e_idx, 0]] += 0.5 * f_edge
            F[be[e_idx, 1]] += 0.5 * f_edge
    return F


def write_inp(path: Path, mesh: TriMesh, problem: Problem, heading: str) -> None:
    mat = problem.material
    clamp = np.nonzero(problem.clamp_predicate(mesh.nodes))[0]
    if len(clamp) == 0:
        raise ValueError("no clamped nodes found")
    F = assemble_nodal_forces(mesh, problem)
    loaded = np.nonzero(np.abs(F).sum(axis=1) > 0)[0]
    if len(loaded) == 0:
        raise ValueError("no loaded nodes found")

    lines: list[str] = []
    lines.append("*HEADING")
    lines.append(heading)
    lines.append("*NODE, NSET=NALL")
    for i, (x, y) in enumerate(mesh.nodes, start=1):
        lines.append(f"{i}, {x:.9g}, {y:.9g}, 0.0")
    lines.append("*ELEMENT, TYPE=CPS3, ELSET=EALL")
    for i, (a, b, c) in enumerate(mesh.tris + 1, start=1):
        lines.append(f"{i}, {a}, {b}, {c}")
    lines.append("*NSET, NSET=CLAMP")
    for i in range(0, len(clamp), 8):
        lines.append(", ".join(str(n + 1) for n in clamp[i : i + 8]))
    lines.append("*MATERIAL, NAME=MAT")
    lines.append("*ELASTIC")
    lines.append(f"{mat.E:.6g}, {mat.nu:.6g}")
    lines.append("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT")
    lines.append(f"{mat.thickness:.6g}")
    lines.append("*BOUNDARY")
    lines.append("CLAMP, 1, 2")
    lines.append("*STEP")
    lines.append("*STATIC")
    lines.append("*CLOAD")
    for n in loaded:
        if abs(F[n, 0]) > 0:
            lines.append(f"{n + 1}, 1, {F[n, 0]:.9g}")
        if abs(F[n, 1]) > 0:
            lines.append(f"{n + 1}, 2, {F[n, 1]:.9g}")
    lines.append("*NODE FILE")
    lines.append("U")
    lines.append("*END STEP")
    Path(path).write_text("\n".join(lines) + "\n")


_EQ_RE = re.compile(r"number of equations\s*\n?\s*(\d+)", re.IGNORECASE)


def run_ccx(
    workdir: Path,
    jobname: str,
    *,
    ccx_cmd: str | None = None,
    timeout: float = 240.0,
) -> tuple[str, int]:
    """Run CalculiX; return (stdout, n_equations)."""

    ccx = ccx_cmd or os.environ.get("CCX_CMD", "ccx")
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
    return u[:, :2]


def solve(
    mesh: TriMesh,
    problem: Problem,
    workdir: Path,
    jobname: str,
    *,
    heading: str = "",
    ccx_cmd: str | None = None,
    timeout: float = 240.0,
) -> CcxResult:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    write_inp(workdir / f"{jobname}.inp", mesh, problem, heading or jobname)
    t0 = time.perf_counter()
    _, n_eq = run_ccx(workdir, jobname, ccx_cmd=ccx_cmd, timeout=timeout)
    wall = time.perf_counter() - t0
    u = read_frd_displacements(workdir / f"{jobname}.frd", mesh.n_nodes)
    return CcxResult(u=u, n_equations=n_eq, wall_s=wall, workdir=str(workdir), jobname=jobname)
