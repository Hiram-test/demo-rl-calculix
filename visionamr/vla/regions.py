"""Human-like partitions: seed-grown geodesic regions (never boxes).

A vision head proposes named *seeds* (structural anchor points with a
fineness each).  The partition of any mesh is the weighted-geodesic
Voronoi decomposition of its elements: every element joins the seed with
the smallest graph distance over the element-adjacency graph.  Region
shapes therefore hug the geometry the way a human's marker stroke does
(they grow around corners, along edges, through thin members) and the
whole domain is covered -- there is no "background" pseudo-region.

Each region still carries exactly one mesh size; region-level grading is
achieved by *splitting* (a region whose internal residual is too
concentrated spawns a child seed at its hotspot), mirroring how a person
redraws a finer partition where needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from ..fem_post import PostState
from ..geometry import Problem
from ..mesher import Mesh
from ..sizefield import NodalSizeField, element_to_node_sizes


@dataclass(frozen=True)
class Seed:
    """One region: a named anchor point and its single mesh size."""

    name: str
    xyz: tuple[float, float, float]
    h: float
    origin: str = "vision"   # "vision" | "coarse" | "split"

    def point(self) -> np.ndarray:
        return np.asarray(self.xyz, dtype=float)


@dataclass
class RegionFeatures:
    err_sum: np.ndarray     # (R,) sum of eta^2 per region
    elems: np.ndarray       # (R,)
    vm_max: np.ndarray
    vm_mean: np.ndarray
    h_meas: np.ndarray      # measured mean element size
    volume: np.ndarray      # region measure (area / volume)
    total_err: float
    total_elems: int

    @property
    def err_share(self) -> np.ndarray:
        return self.err_sum / max(self.total_err, 1e-30)

    @property
    def elem_share(self) -> np.ndarray:
        return self.elems / max(self.total_elems, 1)


@dataclass
class Partition:
    seeds: list[Seed]
    problem: Problem
    gradation: float = 0.9
    assign_mode: str = "geodesic"  # "geodesic" | "linf_box" (AB2)

    # ------------------------------------------------------------------
    def sizes(self) -> np.ndarray:
        return np.array([s.h for s in self.seeds], dtype=float)

    def with_sizes(self, sizes: np.ndarray) -> "Partition":
        seeds = [replace(s, h=float(h)) for s, h in zip(self.seeds, sizes)]
        return Partition(seeds, self.problem, self.gradation, self.assign_mode)

    def add_seed(self, seed: Seed) -> "Partition":
        return Partition(list(self.seeds) + [seed], self.problem, self.gradation, self.assign_mode)

    # ------------------------------------------------------------------
    def assign(self, mesh: Mesh) -> np.ndarray:
        """Element labels: geodesic Voronoi, or L∞ boxes for the AB2 ablation."""

        if self.assign_mode == "linf_box":
            return self._assign_linf_box(mesh)
        if self.assign_mode != "geodesic":
            raise ValueError(f"unknown assign_mode {self.assign_mode!r}")
        return self._assign_geodesic(mesh)

    def _assign_linf_box(self, mesh: Mesh) -> np.ndarray:
        """Axis-aligned (Chebyshev) Voronoi: the v1-style box ablation."""

        seed_pts = np.array([s.point() for s in self.seeds])
        d = np.max(np.abs(mesh.centroids[:, None, :] - seed_pts[None, :, :]), axis=2)
        return np.asarray(np.argmin(d, axis=1), dtype=np.int64)

    def _assign_geodesic(self, mesh: Mesh) -> np.ndarray:
        """Element labels by geodesic Voronoi over the cell-adjacency graph."""

        cen = mesh.centroids
        tree = cKDTree(cen)
        src = np.array([tree.query(s.point())[1] for s in self.seeds])

        pairs, _ = mesh.cell_adjacency
        w = np.linalg.norm(cen[pairs[:, 0]] - cen[pairs[:, 1]], axis=1)
        m = mesh.n_cells
        g = coo_matrix(
            (np.concatenate([w, w]),
             (np.concatenate([pairs[:, 0], pairs[:, 1]]),
              np.concatenate([pairs[:, 1], pairs[:, 0]]))),
            shape=(m, m),
        ).tocsr()
        dist = dijkstra(g, directed=False, indices=src)
        labels = np.asarray(np.argmin(dist, axis=0), dtype=np.int64)
        # disconnected leftovers (should not happen): nearest seed by euclid
        bad = ~np.isfinite(dist[labels, np.arange(m)])
        if bad.any():
            seed_pts = np.array([s.point() for s in self.seeds])
            d_euc = np.linalg.norm(cen[bad, None, :] - seed_pts[None, :, :], axis=2)
            labels[bad] = np.argmin(d_euc, axis=1)
        return labels

    def adjacency(self, mesh: Mesh, labels: np.ndarray) -> list[set[int]]:
        R = len(self.seeds)
        adj: list[set[int]] = [set() for _ in range(R)]
        pairs, _ = mesh.cell_adjacency
        la, lb = labels[pairs[:, 0]], labels[pairs[:, 1]]
        cross = la != lb
        for a, b in zip(la[cross], lb[cross]):
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
        return adj

    def adjacency_matrix(self, mesh: Mesh, labels: np.ndarray) -> np.ndarray:
        R = len(self.seeds)
        A = np.zeros((R, R))
        for i, nbs in enumerate(self.adjacency(mesh, labels)):
            for j in nbs:
                A[i, j] = 1.0
        return A

    # ------------------------------------------------------------------
    def features(self, post: PostState, eta2: np.ndarray, labels: np.ndarray) -> RegionFeatures:
        mesh = post.mesh
        R = len(self.seeds)
        err = np.zeros(R)
        cnt = np.zeros(R, dtype=np.int64)
        vmx = np.zeros(R)
        vmm = np.zeros(R)
        hme = np.zeros(R)
        vol = np.zeros(R)
        sizes = mesh.cell_sizes
        for i in range(R):
            m = labels == i
            cnt[i] = int(m.sum())
            if cnt[i]:
                err[i] = float(eta2[m].sum())
                vmx[i] = float(post.vm_elem[m].max())
                vmm[i] = float(post.vm_elem[m].mean())
                hme[i] = float(sizes[m].mean())
                vol[i] = float(mesh.measures[m].sum())
            else:
                hme[i] = self.seeds[i].h
        return RegionFeatures(
            err_sum=err,
            elems=cnt,
            vm_max=vmx,
            vm_mean=vmm,
            h_meas=hme,
            volume=vol,
            total_err=float(eta2.sum()),
            total_elems=mesh.n_cells,
        )

    # ------------------------------------------------------------------
    def size_field(self, mesh: Mesh, labels: np.ndarray) -> NodalSizeField:
        """Element target = its region's size; node-min; graded interpolant."""

        h_elem = self.sizes()[labels]
        target = element_to_node_sizes(mesh, h_elem)
        return NodalSizeField(
            mesh,
            target,
            gradation=self.gradation,
            h_min=self.problem.h_min,
            h_max=self.problem.h0,
        )

    # ------------------------------------------------------------------
    def split_concentrated(
        self,
        post: PostState,
        eta2: np.ndarray,
        labels: np.ndarray,
        *,
        top_frac: float = 0.10,
        conc_threshold: float = 0.55,
        min_elems: int = 30,
        max_new: int = 2,
        max_seeds: int = 14,
        child_h_factor: float = 0.55,
    ) -> "Partition":
        """Spawn child seeds inside regions whose residual is concentrated.

        The human analogue: where one stroke turns out to cover both a hot
        spot and calm material, the engineer redraws a smaller patch.
        """

        mesh = post.mesh
        scores: list[tuple[float, int, int]] = []
        for i in range(len(self.seeds)):
            idx = np.nonzero(labels == i)[0]
            if len(idx) < min_elems:
                continue
            e = eta2[idx]
            k = max(int(np.ceil(top_frac * len(idx))), 1)
            top = np.sort(e)[-k:]
            conc = float(top.sum() / max(e.sum(), 1e-30))
            if conc > conc_threshold:
                peak = idx[int(np.argmax(e))]
                scores.append((conc, i, peak))
        scores.sort(reverse=True)
        part = self
        for conc, i, peak in scores[:max_new]:
            if len(part.seeds) >= max_seeds:
                break
            parent = part.seeds[i]
            child = Seed(
                name=f"{parent.name}_hot",
                xyz=tuple(mesh.centroids[peak]),
                h=max(child_h_factor * parent.h, self.problem.h_min),
                origin="split",
            )
            part = part.add_seed(child)
        return part
