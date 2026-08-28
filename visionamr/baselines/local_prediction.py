"""Local size prediction: element-wise one-shot multi-level refinement.

The classic engineering adaptive-remeshing procedure of Zienkiewicz &
Zhu (1987) with the mesh optimality criterion of Li & Bettess (1995):
one solve gives eta_K per element; error equidistribution predicts every
element's target size (possibly jumping several levels at once); Gmsh
regenerates the mesh.  Typically probe + 2 predicted remeshes per
resource budget.  This is the strong classical few-shot competitor.
"""

from __future__ import annotations

import numpy as np

from ..experiment import FemRunner, initial_mesh
from ..indicators import zz_indicator
from ..mesher import Mesh, generate_mesh
from ..sizefield import NodalSizeField, element_to_node_sizes


def predicted_sizes(
    mesh: Mesh,
    eta2: np.ndarray,
    *,
    p: float | None = None,
    n_target: float | None = None,
    e_target: float | None = None,
    ratio_bounds: tuple[float, float] = (1.0 / 6.0, 1.8),
    d: int | None = None,
) -> np.ndarray:
    """Per-element predicted size from error equidistribution.

    Exactly one of ``n_target`` (element budget) or ``e_target``
    (target total indicator, sqrt(sum eta^2)) must be given.
    Element counts scale as (h_old/h_new)^d; the local indicator of a
    linear simplex scales as eta_K ~ h^((d+2)/2) (energy norm with the
    volume factor), which is the default prediction exponent ``p``.
    The coarsening bound is deliberately tighter than the refinement
    bound: one-shot equidistribution otherwise destabilizes in 3-D by
    trading bulk resolution for singular lines.
    """

    eta = np.sqrt(np.maximum(eta2, 1e-30))
    h = mesh.cell_sizes
    dd = float(d if d is not None else mesh.dim)
    if p is None:
        p = 0.5 * (dd + 2.0)
    if (n_target is None) == (e_target is None):
        raise ValueError("give exactly one of n_target / e_target")

    if e_target is None:
        def count(eta_allow: float) -> float:
            ratio = np.clip((eta_allow / eta) ** (1.0 / p), *ratio_bounds)
            return float(np.sum(ratio ** (-dd)))

        lo, hi = 1e-12, float(eta.max()) * 10.0
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            if count(mid) > n_target:
                lo = mid
            else:
                hi = mid
        eta_allow = np.sqrt(lo * hi)
    else:
        m_new = float(len(eta))
        for _ in range(50):
            eta_allow = e_target / np.sqrt(m_new)
            ratio = np.clip((eta_allow / eta) ** (1.0 / p), *ratio_bounds)
            m_pred = float(np.sum(ratio ** (-dd)))
            if abs(m_pred - m_new) < 0.5:
                break
            m_new = m_pred

    ratio = np.clip((eta_allow / eta) ** (1.0 / p), *ratio_bounds)
    return h * ratio


def run_local_prediction(
    runner: FemRunner,
    *,
    budgets: list[int],
    rounds: int = 2,
    gradation: float = 0.9,
    method: str = "local_prediction",
) -> None:
    """Few-shot predicted-size remeshing at each element budget.

    For every budget: one probe solve, then ``rounds`` predicted
    remeshes (the second corrects the first with fresh indicators at the
    same budget).  Every solve is counted: rounds+1 per budget.
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
