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


class CalculiXExecutionError(RuntimeError):  # Expose one typed numerical-backend failure to protocol runners.
    """Carry the native return code and retained-log provenance for one failed solve."""  # Document the audit contract.

    def __init__(self, message: str, *, returncode: int | None, wall_s: float, log_path: Path, workdir: Path) -> None:  # Require all available native evidence at construction.
        super().__init__(message)  # Preserve the conventional exception message and traceback behavior.
        self.returncode = None if returncode is None else int(returncode)  # Retain an explicit absent code for launch timeouts.
        self.wall_s = float(wall_s)  # Retain measured native wall time up to failure.
        self.log_path = str(log_path)  # Retain the exact already-written combined stdout/stderr log.
        self.workdir = str(workdir)  # Retain the isolated CalculiX working directory.


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
    log_path = Path(workdir) / f"{jobname}.log"  # Resolve the durable combined native log before launching the process.
    try:  # Convert only native launch timeout into the typed numerical-backend contract.
        proc = subprocess.run(  # Launch exactly one isolated native CalculiX process and wait for its terminal result.
            [ccx, "-i", jobname],  # Pass the resolved executable, input selector, and deterministic job name without a shell.
            cwd=str(workdir),  # Keep all solver files inside the pre-resolved per-attempt evidence directory.
            capture_output=True,  # Retain both native output streams for the durable combined solver log.
            text=True,  # Decode native output as text for equation parsing and bounded diagnostics.
            timeout=timeout,  # Apply only the caller's operational wall-clock limit to this native process.
            env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2")},  # Preserve the environment while bounding default solver threads.
        )  # Return the completed process receipt or raise the explicitly handled timeout exception.
    except subprocess.TimeoutExpired as error:  # Retain partial output and elapsed time for an interrupted native solve.
        wall = time.perf_counter() - t0  # Measure the complete time spent before timeout propagation.
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")  # Normalize captured standard output safely.
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")  # Normalize captured standard error safely.
        out = str(stdout) + str(stderr)  # Preserve the same combined-log convention as completed processes.
        log_path.write_text(out)  # Persist partial native output before raising retained numerical evidence.
        raise CalculiXExecutionError(f"ccx timed out after {wall:.1f}s in {workdir}", returncode=None, wall_s=wall, log_path=log_path, workdir=Path(workdir)) from error  # Surface one typed failure without hiding its cause.
    wall = time.perf_counter() - t0
    out = proc.stdout + proc.stderr
    frd = Path(workdir) / f"{jobname}.frd"
    log_path.write_text(out)  # Persist stdout and stderr before any failure is raised so numerical failures remain auditable.
    if proc.returncode != 0 or not frd.exists():
        raise CalculiXExecutionError(f"ccx failed (rc={proc.returncode}, {wall:.1f}s) in {workdir}:\n{out[-2000:]}", returncode=proc.returncode, wall_s=wall, log_path=log_path, workdir=Path(workdir))  # Surface only this native failure as finite benchmark evidence.
    m = _EQ_RE.search(out)
    n_eq = int(m.group(1)) if m else -1
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
    try:  # Reclassify a malformed native result file as a typed CalculiX execution failure.
        u = read_frd_displacements(workdir / f"{jobname}.frd", mesh.n_nodes)  # Parse the completed native displacement dataset.
    except RuntimeError as error:  # Catch only the parser's explicit missing-result failure.
        log_path = workdir / f"{jobname}.log"  # Reuse the combined native log written before result parsing.
        raise CalculiXExecutionError(str(error), returncode=0, wall_s=wall, log_path=log_path, workdir=workdir) from error  # Retain rc=0 while reporting an unusable native result.
    return CcxResult(u=u, n_equations=n_eq, wall_s=wall, workdir=str(workdir), jobname=jobname)
