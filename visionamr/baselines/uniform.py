"""Uniform refinement ladder: the "just add DOF" control."""

from __future__ import annotations

import numpy as np

from ..experiment import FemRunner
from ..mesher import generate_uniform


def run_uniform_ladder(
    runner: FemRunner,
    *,
    n_steps: int = 6,
    ratio: float = 1.4142135623730951,
    method: str = "uniform",
) -> None:
    """Solve a geometric ladder h0, h0/r, h0/r^2, ... (r = sqrt(2))."""

    problem = runner.problem
    runner.ensure_reference()
    h = problem.h0
    for k in range(n_steps):
        mesh = generate_uniform(problem, h)
        runner.solve_mesh(mesh, method=method, stage=f"h{k}", extra={"h": h})
        h /= ratio
        if h < 1.5 * problem.h_min:
            break
