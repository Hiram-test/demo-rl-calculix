"""Local size prediction: element-wise one-shot multi-level refinement.

This is the classic engineering adaptive-remeshing procedure of
Zienkiewicz & Zhu (1987, section on adaptive refinement) with the mesh
optimality criterion refined by Li & Bettess (1995):

1. solve once, compute eta_K per element (ZZ recovery);
2. equidistribute: choose the admissible per-element error so that the
   predicted new mesh hits either a global accuracy target or a DOF
   budget, using the a priori local rate  eta_K ~ C h_K^p  (p = element
   order = 1 here);
3. every element gets its own predicted target size

       h_new,K = h_K * (eta_allow / eta_K)^(1/p),

   which may jump several refinement levels at once;
4. Gmsh regenerates the mesh from the size map; typically 1-3 rounds
   suffice (each round is one global solve).

Unlike Doerfler marking this predicts the whole size distribution per
round rather than marking a bulk subset, so it reaches the target in far
fewer remeshing cycles -- the classic "few-shot" competitor.
"""

from __future__ import annotations

import numpy as np

from ..experiment import FemRunner, initial_mesh
from ..indicators import zz_indicator
from ..mesher import TriMesh, generate_mesh
from ..sizefield import NodalSizeField


def predicted_sizes(
    mesh: TriMesh,
    eta2: np.ndarray,
    *,
    p: float = 1.0,
    n_target: int | None = None,
    e_target: float | None = None,
    ratio_bounds: tuple[float, float] = (1.0 / 6.0, 3.0),
) -> np.ndarray:
    """Per-element predicted size from error equidistribution.

    Exactly one of ``n_target`` (element budget) or ``e_target``
    (target total indicator, sqrt(sum eta^2)) must be given.
    """

    eta = np.sqrt(np.maximum(eta2, 1e-30))
    h = mesh.tri_sizes
    if (n_target is None) == (e_target is None):
        raise ValueError("give exactly one of n_target / e_target")

    if e_target is None:
        # find eta_allow such that predicted element count matches budget:
        # each old element K becomes (h_K/h_new,K)^2 new elements
        def count(eta_allow: float) -> float:
            ratio = np.clip((eta_allow / eta) ** (1.0 / p), *ratio_bounds)
            return float(np.sum(ratio ** (-2.0)))

        lo, hi = 1e-12, float(eta.max()) * 10.0
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            if count(mid) > n_target:
                lo = mid
            else:
                hi = mid
        eta_allow = np.sqrt(lo * hi)
    else:
        # uniform admissible error per element of the *new* mesh
        # (fixed point on the predicted element count)
        m_new = float(len(eta))
        for _ in range(50):
            eta_allow = e_target / np.sqrt(m_new)
            ratio = np.clip((eta_allow / eta) ** (1.0 / p), *ratio_bounds)
            m_pred = float(np.sum(ratio ** (-2.0)))
            if abs(m_pred - m_new) < 0.5:
                break
            m_new = m_pred

    ratio = np.clip((eta_allow / eta) ** (1.0 / p), *ratio_bounds)
    return h * ratio


def element_to_node_sizes(mesh: TriMesh, h_elem: np.ndarray) -> np.ndarray:
    """Min incident-element target per node (conservative)."""

    h_node = np.full(mesh.n_nodes, np.inf)
    for k in range(3):
        np.minimum.at(h_node, mesh.tris[:, k], h_elem)
    h_node[np.isinf(h_node)] = np.median(h_elem)
    return h_node


def run_local_prediction(
    runner: FemRunner,
    *,
    budgets: list[int],
    rounds: int = 2,
    gradation: float = 0.9,
    method: str = "local_prediction",
) -> None:
    """Few-shot predicted-size remeshing at each element budget.

    For every budget: one probe solve on the initial mesh, then
    ``rounds`` predicted remeshes (the second round corrects the first
    prediction with fresh indicators at the same budget).  Every solve
    is counted; the whole run for one budget costs ``rounds + 1``
    global solves.
    """

    problem = runner.problem
    runner.ensure_reference()
    for budget in budgets:
        mesh = initial_mesh(problem)
        for r in range(rounds + 1):
            post, rec = runner.solve_mesh(
                mesh, method=method, stage=f"b{budget}_round{r}", extra={"budget": budget}
            )
            eta2 = zz_indicator(problem, post)
            rec.extra["sum_eta2"] = float(eta2.sum())
            if r >= rounds:
                break
            h_elem = predicted_sizes(mesh, eta2, n_target=budget)
            target = element_to_node_sizes(mesh, h_elem)
            field = NodalSizeField(
                mesh, target, gradation=gradation, h_min=problem.h_min, h_max=problem.h0
            )
            mesh = generate_mesh(problem, field)
