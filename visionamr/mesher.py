"""Gmsh meshing driver.

This is the single place where meshes are created.  Every method in this
framework expresses its decision as a size field (callable ``size(x, y)``)
and Gmsh regenerates the mesh.  No code in this repository ever edits
nodes or elements by hand.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .geometry import Problem

_GMSH_INITIALIZED = False


def _ensure_gmsh() -> None:
    global _GMSH_INITIALIZED
    import gmsh

    if not _GMSH_INITIALIZED or not gmsh.isInitialized():
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 2)
        _GMSH_INITIALIZED = True


@dataclass
class TriMesh:
    nodes: np.ndarray  # (n, 2) float
    tris: np.ndarray   # (m, 3) int, zero-based

    @cached_property
    def areas(self) -> np.ndarray:
        p = self.nodes[self.tris]
        return 0.5 * np.abs(
            (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
            - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
        )

    @cached_property
    def centroids(self) -> np.ndarray:
        return self.nodes[self.tris].mean(axis=1)

    @cached_property
    def edges(self) -> np.ndarray:
        """Unique undirected edges (k, 2), sorted node indices."""
        e = np.vstack(
            [self.tris[:, [0, 1]], self.tris[:, [1, 2]], self.tris[:, [2, 0]]]
        )
        e.sort(axis=1)
        return np.unique(e, axis=0)

    @cached_property
    def boundary_edges(self) -> np.ndarray:
        """Edges belonging to exactly one triangle (k, 2)."""
        e = np.vstack(
            [self.tris[:, [0, 1]], self.tris[:, [1, 2]], self.tris[:, [2, 0]]]
        )
        e.sort(axis=1)
        uniq, counts = np.unique(e, axis=0, return_counts=True)
        return uniq[counts == 1]

    @cached_property
    def edge_lengths(self) -> np.ndarray:
        d = self.nodes[self.edges[:, 0]] - self.nodes[self.edges[:, 1]]
        return np.hypot(d[:, 0], d[:, 1])

    @cached_property
    def node_sizes(self) -> np.ndarray:
        """Local mesh size per node: mean length of incident edges."""
        n = len(self.nodes)
        acc = np.zeros(n)
        cnt = np.zeros(n)
        L = self.edge_lengths
        for col in (0, 1):
            np.add.at(acc, self.edges[:, col], L)
            np.add.at(cnt, self.edges[:, col], 1.0)
        cnt[cnt == 0] = 1.0
        return acc / cnt

    @cached_property
    def tri_sizes(self) -> np.ndarray:
        """Local mesh size per element: mean of its three edge lengths."""
        p = self.nodes[self.tris]
        l0 = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
        l1 = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
        l2 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
        return (l0 + l1 + l2) / 3.0

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_tris(self) -> int:
        return len(self.tris)

    def sha(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.nodes).tobytes())
        h.update(np.ascontiguousarray(self.tris).tobytes())
        return h.hexdigest()[:16]


def generate_mesh(problem: Problem, size_fn, *, model_name: str = "model") -> TriMesh:
    """Build the problem geometry and mesh it with the given size field.

    ``size_fn(x, y) -> float`` is evaluated by Gmsh's size callback; the
    returned mesh is a linear triangle mesh.
    """

    import gmsh

    _ensure_gmsh()
    gmsh.model.add(model_name)
    try:
        problem.build_geometry()
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Optimize", 1)

        h_floor = float(problem.h_min)

        def cb(dim, tag, x, y, z, lc):
            return max(float(size_fn(x, y)), h_floor)

        gmsh.model.mesh.setSizeCallback(cb)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.removeSizeCallback()

        tags, coords, _ = gmsh.model.mesh.getNodes()
        tags = np.asarray(tags, dtype=np.int64)
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]
        order = np.argsort(tags)
        tags = tags[order]
        coords = coords[order]
        remap = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
        remap[tags] = np.arange(len(tags))

        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        tris = []
        for etype, conn in zip(etypes, enodes):
            if etype == 2:  # 3-node triangle
                tris.append(np.asarray(conn, dtype=np.int64).reshape(-1, 3))
        if not tris:
            raise RuntimeError("Gmsh produced no linear triangles")
        tri = remap[np.vstack(tris)]

        mesh = TriMesh(nodes=coords, tris=tri)
        # drop unused nodes (gmsh may report embedded geometry vertices)
        used = np.zeros(mesh.n_nodes, dtype=bool)
        used[tri.ravel()] = True
        if not used.all():
            new_index = np.cumsum(used) - 1
            mesh = TriMesh(nodes=coords[used], tris=new_index[tri])
        return mesh
    finally:
        gmsh.model.remove()


def generate_uniform(problem: Problem, h: float) -> TriMesh:
    return generate_mesh(problem, lambda x, y: h)
