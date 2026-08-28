"""Experiment orchestration: solve accounting, reference caching, metrics.

Every CalculiX invocation made by any method goes through ``FemRunner``,
so the "number of global solves" axis in the paper is an honest count.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import calculix
from .fem_post import PostState, compute_post
from .geometry import Problem
from .mesher import TriMesh, generate_mesh, generate_uniform


@dataclass
class SolveRecord:
    method: str
    stage: str
    solve_index: int          # 1-based, cumulative per method run
    n_nodes: int
    n_elems: int
    n_equations: int
    U_total: float
    qoi: float
    wall_s: float
    h_min: float
    h_max: float
    e_energy: float | None = None   # relative energy-norm error vs reference
    e_qoi: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Reference:
    U_total: float
    qoi: float
    n_equations: int
    n_elems: int
    h_ref: float


def reference_size_fn(problem: Problem):
    """Strongly graded reference field: background h_ref, power-law grading
    into corner singularities (r^(1-lambda), lambda ~ 0.545 for the 270-deg
    elastic corner) and towards hole edges.  The reference resolves every
    method mesh by a wide margin near the error-dominating features, so the
    Galerkin energy gap stays positive.
    """

    import math

    xmin, ymin, xmax, ymax = problem.bbox
    diam = math.hypot(xmax - xmin, ymax - ymin)
    d_grade = 0.3 * diam
    corner_floor = 1.0 / 48.0
    holes = [f for f in problem.features if f.kind == "hole" and f.r > 0]

    def size(x: float, y: float) -> float:
        h = problem.h_ref
        for sx, sy in problem.singular_points:
            d = math.hypot(x - sx, y - sy)
            h = min(h, problem.h_ref * max((d / d_grade) ** 0.55, corner_floor))
        for f in holes:
            d_edge = abs(math.hypot(x - f.x, y - f.y) - f.r)
            h = min(h, problem.h_ref * max((d_edge / (1.5 * f.r)) ** 0.7, 0.2))
        return h

    return size


class FemRunner:
    """Solves meshes for one problem instance and records every solve."""

    def __init__(
        self,
        problem: Problem,
        workdir: Path,
        *,
        keep_files: bool = False,
        ccx_timeout: float = 240.0,
    ) -> None:
        self.problem = problem
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.keep_files = keep_files
        self.ccx_timeout = ccx_timeout
        self.records: list[SolveRecord] = []
        self._counter = 0
        self.reference: Reference | None = None

    # ------------------------------------------------------------------
    def ensure_reference(self) -> Reference:
        """Solve (or load) the fine uniform reference for this instance."""

        ref_path = self.workdir / "reference.json"
        if self.reference is not None:
            return self.reference
        if ref_path.exists():
            data = json.loads(ref_path.read_text())
            self.reference = Reference(**data)
            return self.reference
        mesh = generate_mesh(self.problem, reference_size_fn(self.problem))
        post, rec = self._solve(mesh, method="reference", stage="reference", count=False)
        self.reference = Reference(
            U_total=post.U_total,
            qoi=post.qoi,
            n_equations=rec.n_equations,
            n_elems=mesh.n_tris,
            h_ref=self.problem.h_ref,
        )
        ref_path.write_text(json.dumps(asdict(self.reference), indent=1))
        return self.reference

    # ------------------------------------------------------------------
    def solve_mesh(
        self, mesh: TriMesh, *, method: str, stage: str, extra: dict | None = None
    ) -> tuple[PostState, SolveRecord]:
        return self._solve(mesh, method=method, stage=stage, count=True, extra=extra)

    def _solve(
        self,
        mesh: TriMesh,
        *,
        method: str,
        stage: str,
        count: bool,
        extra: dict | None = None,
    ) -> tuple[PostState, SolveRecord]:
        if count:
            self._counter += 1
        idx = self._counter if count else 0
        jobname = f"{method}_{idx:03d}_{stage}"[:60].replace("/", "_")
        jobdir = self.workdir / "solves" / jobname
        res = calculix.solve(
            mesh,
            self.problem,
            jobdir,
            "model",
            heading=f"{self.problem.instance_id} {method} {stage}",
            timeout=self.ccx_timeout,
        )
        post = compute_post(mesh, self.problem, res.u)
        sizes = mesh.tri_sizes
        rec = SolveRecord(
            method=method,
            stage=stage,
            solve_index=idx,
            n_nodes=mesh.n_nodes,
            n_elems=mesh.n_tris,
            n_equations=res.n_equations,
            U_total=post.U_total,
            qoi=post.qoi,
            wall_s=res.wall_s,
            h_min=float(sizes.min()),
            h_max=float(sizes.max()),
            extra=extra or {},
        )
        if self.reference is not None:
            rec.e_energy = self.energy_error(post.U_total)
            rec.e_qoi = self.qoi_error(post.qoi)
            if post.U_total > self.reference.U_total:
                rec.extra["above_reference"] = True
        if count:
            self.records.append(rec)
        if not self.keep_files:
            for f in jobdir.glob("*"):
                if f.suffix not in (".log",):
                    f.unlink(missing_ok=True)
        return post, rec

    # ------------------------------------------------------------------
    def energy_error(self, U: float) -> float:
        """Relative energy-norm error: ||u-u_h||_E / ||u||_E = sqrt(1 - U_h/U_ref).

        Valid for conforming FE with fixed traction loading (Galerkin
        orthogonality); the reference is the fine uniform mesh.
        """

        ref = self.ensure_reference()
        gap = max(ref.U_total - U, 0.0)
        return float(np.sqrt(gap / ref.U_total))

    def qoi_error(self, qoi: float) -> float:
        ref = self.ensure_reference()
        return float(abs(qoi - ref.qoi) / abs(ref.qoi))

    # ------------------------------------------------------------------
    def reset_counter(self) -> None:
        self._counter = 0

    def dump(self, path: Path | None = None) -> Path:
        path = path or (self.workdir / "records.json")
        payload = {
            "problem": self.problem.instance_id,
            "params": self.problem.params,
            "reference": asdict(self.reference) if self.reference else None,
            "records": [asdict(r) for r in self.records],
        }
        path.write_text(json.dumps(payload, indent=1))
        return path


def initial_mesh(problem: Problem) -> TriMesh:
    return generate_uniform(problem, problem.h0)
