"""Zienkiewicz-Zhu (1987) recovery-based element error indicator.

sigma_h is the piecewise-constant CST stress; sigma* is the area-weighted
nodal average interpolated linearly.  The indicator is the energy-norm of
the recovery difference,

    eta_K^2 = t * integral_K (sigma* - sigma_h) : C^{-1} : (sigma* - sigma_h) dA,

integrated exactly with the three-midpoint rule.  This is the simple ZZ
estimator (not SPR).
"""

from __future__ import annotations

import numpy as np

from .fem_post import PostState, plane_stress_C
from .geometry import Problem


def zz_indicator(problem: Problem, post: PostState) -> np.ndarray:
    mesh = post.mesh
    C = plane_stress_C(problem.material)
    Cinv = np.linalg.inv(C)
    t = problem.material.thickness

    # area-weighted nodal stress recovery
    sig_node = np.zeros((mesh.n_nodes, 3))
    wsum = np.zeros(mesh.n_nodes)
    for k in range(3):
        np.add.at(sig_node, mesh.tris[:, k], post.stress * mesh.areas[:, None])
        np.add.at(wsum, mesh.tris[:, k], mesh.areas)
    wsum[wsum == 0] = 1.0
    sig_node /= wsum[:, None]

    # nodal recovery difference per element corner: e_i = sigma*(x_i) - sigma_h
    e_corner = sig_node[mesh.tris] - post.stress[:, None, :]  # (m, 3, 3)

    # exact integral of quadratic e:Cinv:e with 3-midpoint rule
    m01 = 0.5 * (e_corner[:, 0] + e_corner[:, 1])
    m12 = 0.5 * (e_corner[:, 1] + e_corner[:, 2])
    m20 = 0.5 * (e_corner[:, 2] + e_corner[:, 0])
    quad = np.zeros(len(mesh.tris))
    for mm in (m01, m12, m20):
        quad += np.einsum("mi,ij,mj->m", mm, Cinv, mm)
    eta2 = (mesh.areas / 3.0) * quad * t
    return eta2
