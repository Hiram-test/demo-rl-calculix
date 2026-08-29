"""Size-field construction.

Every mesh decision becomes a nodal target-size map on the previous mesh
(per-element AFEM maps and per-region partition sizes alike), smoothed by
Lipschitz gradation and interpolated for Gmsh's size callback.  Boxes or
other geometric primitives are deliberately absent: region shapes live in
``vla.regions.Partition`` as element sets, not geometry.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .mesher import Mesh


def lipschitz_smooth(
    nodes: np.ndarray,
    edges: np.ndarray,
    h: np.ndarray,
    gradation: float = 1.0,
    max_sweeps: int = 100,
) -> np.ndarray:
    """Enforce |h_i - h_j| <= gradation * |x_i - x_j| over the mesh graph."""

    h = h.astype(float).copy()
    d = nodes[edges[:, 0]] - nodes[edges[:, 1]]
    L = np.linalg.norm(d, axis=1) * float(gradation)
    a, b = edges[:, 0], edges[:, 1]
    for _ in range(max_sweeps):
        cap_b = h[a] + L
        cap_a = h[b] + L
        changed = False
        upd_b = h[b] > cap_b + 1e-12
        if upd_b.any():
            np.minimum.at(h, b[upd_b], cap_b[upd_b])
            changed = True
        upd_a = h[a] > cap_a + 1e-12
        if upd_a.any():
            np.minimum.at(h, a[upd_a], cap_a[upd_a])
            changed = True
        if not changed:
            break
    return h


class NodalSizeField:
    """Interpolates a nodal target-size map defined on an existing mesh."""

    def __init__(
        self,
        mesh: Mesh,
        target_h: np.ndarray,
        *,
        gradation: float = 1.0,
        h_min: float = 1e-6,
        h_max: float | None = None,
    ) -> None:
        target_h = np.clip(np.asarray(target_h, dtype=float), h_min, h_max)
        target_h = lipschitz_smooth(mesh.nodes, mesh.edges, target_h, gradation)
        self._tree = cKDTree(mesh.nodes)
        self._h = target_h
        self._k = min(4 if mesh.dim == 2 else 6, len(mesh.nodes))

    def __call__(self, x: float, y: float, z: float = 0.0) -> float:
        dist, idx = self._tree.query([x, y, z], k=self._k)
        dist = np.atleast_1d(dist)
        idx = np.atleast_1d(idx)
        if dist[0] < 1e-12:
            return float(self._h[idx[0]])
        w = 1.0 / np.maximum(dist, 1e-12) ** 2
        return float(np.sum(w * self._h[idx]) / np.sum(w))


def element_to_node_sizes(mesh: Mesh, h_elem: np.ndarray) -> np.ndarray:
    """Min incident-element target per node (conservative)."""

    h_node = np.full(mesh.n_nodes, np.inf)
    for k in range(mesh.dim + 1):
        np.minimum.at(h_node, mesh.cells[:, k], h_elem)
    h_node[np.isinf(h_node)] = float(np.median(h_elem))
    return h_node
