"""Zienkiewicz-Zhu (1987) recovery-based element error indicator (2-D/3-D).

sigma_h is the piecewise-constant simplex stress; sigma* is the
measure-weighted nodal average interpolated linearly.  The indicator is

    eta_K^2 = scale * integral_K (sigma*-sigma_h) : C^{-1} : (sigma*-sigma_h),

integrated exactly for the quadratic integrand: three edge midpoints in
2-D, the degree-2 four-point rule in 3-D.  Simple ZZ, not SPR.
"""

from __future__ import annotations

import numpy as np

from .fem_post import PostState, elastic_C
from .geometry import Problem

# degree-2 exact quadrature on the reference tet (barycentric, weight 1/4)
_TET_A = 0.5854101966249685
_TET_B = 0.13819660112501053
_TET_POINTS = np.array(
    [
        [_TET_A, _TET_B, _TET_B, _TET_B],
        [_TET_B, _TET_A, _TET_B, _TET_B],
        [_TET_B, _TET_B, _TET_A, _TET_B],
        [_TET_B, _TET_B, _TET_B, _TET_A],
    ]
)

_TRI_POINTS = np.array(
    [
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
    ]
)


def zz_indicator(problem: Problem, post: PostState) -> np.ndarray:
    mesh = post.mesh
    d = mesh.dim
    Cinv = np.linalg.inv(elastic_C(problem.material, d))
    scale = problem.material.thickness if d == 2 else 1.0

    # measure-weighted nodal stress recovery
    nv = post.stress.shape[1]
    sig_node = np.zeros((mesh.n_nodes, nv))
    wsum = np.zeros(mesh.n_nodes)
    for k in range(d + 1):
        np.add.at(sig_node, mesh.cells[:, k], post.stress * mesh.measures[:, None])
        np.add.at(wsum, mesh.cells[:, k], mesh.measures)
    wsum[wsum == 0] = 1.0
    sig_node /= wsum[:, None]

    # recovery difference at element corners: (m, d+1, nv)
    e_corner = sig_node[mesh.cells] - post.stress[:, None, :]

    pts = _TRI_POINTS if d == 2 else _TET_POINTS
    quad = np.zeros(mesh.n_cells)
    for bary in pts:
        e_pt = np.einsum("k,mkv->mv", bary, e_corner)
        quad += np.einsum("mi,ij,mj->m", e_pt, Cinv, e_pt)
    quad /= len(pts)
    return mesh.measures * quad * scale
