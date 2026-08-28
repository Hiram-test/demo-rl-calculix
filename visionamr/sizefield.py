"""Size-field construction.

Two kinds of decisions become Gmsh size fields here:

* per-region sizes (vision / RL / supervised region methods) via
  ``RegionSizeField`` -- "change the size of a region and let Gmsh remesh";
* per-element target sizes (classic AFEM: Doerfler marking, local size
  prediction) via ``NodalSizeField`` built on the previous mesh.

Both apply Lipschitz gradation so element size varies smoothly, which is
standard practice for remeshing-based adaptivity (Borouchaki et al.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .mesher import TriMesh


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
    L = np.hypot(d[:, 0], d[:, 1]) * float(gradation)
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
        mesh: TriMesh,
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
        # inverse-distance over a few neighbours gives a smooth field and
        # is robust outside the old mesh (holes, corners)
        self._k = min(4, len(mesh.nodes))

    def __call__(self, x: float, y: float) -> float:
        dist, idx = self._tree.query([x, y], k=self._k)
        dist = np.atleast_1d(dist)
        idx = np.atleast_1d(idx)
        if dist[0] < 1e-12:
            return float(self._h[idx[0]])
        w = 1.0 / np.maximum(dist, 1e-12) ** 2
        return float(np.sum(w * self._h[idx]) / np.sum(w))


@dataclass(frozen=True)
class Region:
    """Axis-aligned box region with one mesh size (the region's decision)."""

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.xmin + self.xmax), 0.5 * (self.ymin + self.ymax))

    @property
    def area(self) -> float:
        return max(self.xmax - self.xmin, 0.0) * max(self.ymax - self.ymin, 0.0)

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx = np.maximum(np.maximum(self.xmin - x, x - self.xmax), 0.0)
        dy = np.maximum(np.maximum(self.ymin - y, y - self.ymax), 0.0)
        return np.hypot(dx, dy)

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (
            (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)
        )

    def with_h(self, h: float) -> "Region":
        return Region(self.name, self.xmin, self.ymin, self.xmax, self.ymax, float(h))


@dataclass
class RegionSizeField:
    """size(x) = min(h_bg, min_i (h_i + g * dist(x, region_i)))."""

    regions: Sequence[Region]
    h_background: float
    gradation: float = 0.9

    def __call__(self, x: float, y: float) -> float:
        xa = np.asarray([x], dtype=float)
        ya = np.asarray([y], dtype=float)
        best = float(self.h_background)
        for r in self.regions:
            v = float(r.h + self.gradation * r.distance(xa, ya)[0])
            if v < best:
                best = v
        return best

    def sizes_vector(self) -> np.ndarray:
        return np.array([r.h for r in self.regions], dtype=float)

    def with_sizes(self, sizes: Sequence[float]) -> "RegionSizeField":
        regs = [r.with_h(h) for r, h in zip(self.regions, sizes)]
        return RegionSizeField(regs, self.h_background, self.gradation)
