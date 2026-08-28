"""Uniform refinement ladder: the "just add DOF" control."""

from __future__ import annotations

from ..experiment import FemRunner
from ..mesher import generate_uniform


def run_uniform_ladder(
    runner: FemRunner,
    *,
    n_steps: int = 10,
    ratio: float | None = None,
    n_eq_cap: int | None = None,
    method: str = "uniform",
) -> None:
    """Solve a geometric ladder h0, h0/r, h0/r^2, ...

    The default ratio doubles the element count per step in any
    dimension (sqrt(2) in 2-D, 2^(1/3) in 3-D); the ladder stops once
    the DOF cap is crossed.
    """

    problem = runner.problem
    runner.ensure_reference()
    if ratio is None:
        ratio = 2.0 ** (1.0 / problem.dim)
    h = problem.h0
    for k in range(n_steps):
        mesh = generate_uniform(problem, h)
        _, rec = runner.solve_mesh(mesh, method=method, stage=f"h{k}", extra={"h": h})
        h /= ratio
        if h < 1.5 * problem.h_min:
            break
        if n_eq_cap is not None and rec.n_equations >= n_eq_cap:
            break
