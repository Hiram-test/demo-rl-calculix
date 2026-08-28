"""Region graph: the shared decision object of the VLA and RL methods.

A region is an axis-aligned box with one mesh size (see
``sizefield.Region``).  The graph connects spatially adjacent regions;
features aggregate the last solve's indicator and resource distribution
over regions plus the background complement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..fem_post import PostState
from ..geometry import Problem
from ..mesher import TriMesh
from ..sizefield import Region, RegionSizeField


@dataclass
class RegionFeatures:
    err_sum: np.ndarray     # (R,) sum of eta^2 over elements in region
    elems: np.ndarray       # (R,) element counts
    vm_max: np.ndarray      # (R,)
    vm_mean: np.ndarray     # (R,)
    h_meas: np.ndarray      # (R,) measured mean element size
    area: np.ndarray        # (R,)
    bg_err: float
    bg_elems: int
    bg_h_meas: float
    total_err: float
    total_elems: int

    @property
    def err_share(self) -> np.ndarray:
        return self.err_sum / max(self.total_err, 1e-30)

    @property
    def elem_share(self) -> np.ndarray:
        return self.elems / max(self.total_elems, 1)


@dataclass
class RegionGraph:
    regions: list[Region]
    h_background: float
    problem: Problem
    adjacency: list[set[int]] = field(default_factory=list)
    gradation: float = 0.9

    # ------------------------------------------------------------------
    @staticmethod
    def build(
        regions: list[Region],
        h_background: float,
        problem: Problem,
        *,
        pad_frac: float = 0.04,
        gradation: float = 0.9,
    ) -> "RegionGraph":
        xmin, ymin, xmax, ymax = problem.bbox
        pad = pad_frac * float(np.hypot(xmax - xmin, ymax - ymin))
        n = len(regions)
        adj: list[set[int]] = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                a, b = regions[i], regions[j]
                if (
                    a.xmin - pad <= b.xmax
                    and b.xmin - pad <= a.xmax
                    and a.ymin - pad <= b.ymax
                    and b.ymin - pad <= a.ymax
                ):
                    adj[i].add(j)
                    adj[j].add(i)
        # connect isolated regions to their nearest neighbour
        centers = np.array([r.center for r in regions]) if regions else np.zeros((0, 2))
        for i in range(n):
            if not adj[i] and n > 1:
                d = np.linalg.norm(centers - centers[i], axis=1)
                d[i] = np.inf
                j = int(np.argmin(d))
                adj[i].add(j)
                adj[j].add(i)
        return RegionGraph(list(regions), h_background, problem, adj, gradation)

    # ------------------------------------------------------------------
    def assign_elements(self, mesh: TriMesh) -> np.ndarray:
        """Region index per element (-1 = background).

        Overlapping regions: the smallest-area region wins, so nested
        hotspot boxes keep their identity.
        """

        cen = mesh.centroids
        owner = np.full(mesh.n_tris, -1, dtype=np.int64)
        order = np.argsort([-r.area for r in self.regions])  # big first, small overwrite
        for i in order:
            r = self.regions[i]
            inside = r.contains(cen[:, 0], cen[:, 1])
            owner[inside] = i
        return owner

    def features(self, post: PostState, eta2: np.ndarray) -> RegionFeatures:
        mesh = post.mesh
        owner = self.assign_elements(mesh)
        R = len(self.regions)
        err = np.zeros(R)
        cnt = np.zeros(R, dtype=np.int64)
        vmx = np.zeros(R)
        vmm = np.zeros(R)
        hme = np.zeros(R)
        area = np.array([r.area for r in self.regions], dtype=float)
        sizes = mesh.tri_sizes
        for i in range(R):
            m = owner == i
            cnt[i] = int(m.sum())
            if cnt[i]:
                err[i] = float(eta2[m].sum())
                vmx[i] = float(post.vm_elem[m].max())
                vmm[i] = float(post.vm_elem[m].mean())
                hme[i] = float(sizes[m].mean())
            else:
                hme[i] = self.regions[i].h
        bg = owner == -1
        return RegionFeatures(
            err_sum=err,
            elems=cnt,
            vm_max=vmx,
            vm_mean=vmm,
            h_meas=hme,
            area=area,
            bg_err=float(eta2[bg].sum()),
            bg_elems=int(bg.sum()),
            bg_h_meas=float(sizes[bg].mean()) if bg.any() else self.h_background,
            total_err=float(eta2.sum()),
            total_elems=mesh.n_tris,
        )

    # ------------------------------------------------------------------
    def size_field(self) -> RegionSizeField:
        return RegionSizeField(list(self.regions), self.h_background, self.gradation)

    def with_sizes(self, sizes: np.ndarray, h_background: float | None = None) -> "RegionGraph":
        regs = [r.with_h(float(h)) for r, h in zip(self.regions, sizes)]
        return RegionGraph(
            regs,
            float(h_background if h_background is not None else self.h_background),
            self.problem,
            [set(s) for s in self.adjacency],
            self.gradation,
        )

    def sizes(self) -> np.ndarray:
        return np.array([r.h for r in self.regions], dtype=float)

    def adjacency_matrix(self) -> np.ndarray:
        n = len(self.regions)
        A = np.zeros((n, n))
        for i, nb in enumerate(self.adjacency):
            for j in nb:
                A[i, j] = 1.0
        return A
