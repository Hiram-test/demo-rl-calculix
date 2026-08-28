"""Gmsh meshing driver (2-D triangles and 3-D tetrahedra).

This is the single place where meshes are created.  Every method in this
framework expresses its decision as a size field (callable
``size(x, y, z)``) and Gmsh regenerates the mesh.  No code in this
repository ever edits nodes or elements by hand.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property

import numpy as np

_GMSH_INITIALIZED = False

_EDGE_LOCAL = {
    2: [(0, 1), (1, 2), (2, 0)],
    3: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
}
_FACET_LOCAL = {
    2: [(0, 1), (1, 2), (2, 0)],
    3: [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
}


def _ensure_gmsh() -> None:
    global _GMSH_INITIALIZED
    import gmsh

    if not _GMSH_INITIALIZED or not gmsh.isInitialized():
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 2)
        # single-threaded meshing: bitwise-reproducible meshes (HXT is
        # otherwise nondeterministic run-to-run)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)
        _GMSH_INITIALIZED = True


@dataclass
class Mesh:
    """Simplex mesh: triangles (dim=2, z column zero) or tets (dim=3)."""

    nodes: np.ndarray  # (n, 3) float
    cells: np.ndarray  # (m, dim+1) int, zero-based
    dim: int

    @cached_property
    def measures(self) -> np.ndarray:
        """Element areas (2-D) or volumes (3-D)."""
        p = self.nodes[self.cells]
        if self.dim == 2:
            a = p[:, 1, :2] - p[:, 0, :2]
            b = p[:, 2, :2] - p[:, 0, :2]
            return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        a = p[:, 1] - p[:, 0]
        b = p[:, 2] - p[:, 0]
        c = p[:, 3] - p[:, 0]
        return np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0

    @cached_property
    def centroids(self) -> np.ndarray:
        return self.nodes[self.cells].mean(axis=1)

    @cached_property
    def edges(self) -> np.ndarray:
        """Unique undirected node-pair edges (k, 2)."""
        pairs = np.vstack(
            [self.cells[:, list(pl)] for pl in _EDGE_LOCAL[self.dim]]
        )
        pairs.sort(axis=1)
        return np.unique(pairs, axis=0)

    @cached_property
    def _boundary(self) -> tuple[np.ndarray, np.ndarray]:
        f = np.vstack([self.cells[:, list(fl)] for fl in _FACET_LOCAL[self.dim]])
        owner = np.tile(np.arange(len(self.cells)), len(_FACET_LOCAL[self.dim]))
        fs = np.sort(f, axis=1)
        uniq, idx, counts = np.unique(fs, axis=0, return_index=True, return_counts=True)
        sel = idx[counts == 1]
        return f[sel], owner[sel]

    @cached_property
    def boundary_facets(self) -> np.ndarray:
        """Facets on the boundary: edges (2-D) or triangles (3-D)."""
        return self._boundary[0]

    @cached_property
    def boundary_facet_owners(self) -> np.ndarray:
        """Owning cell index of each boundary facet."""
        return self._boundary[1]

    @cached_property
    def facet_measures(self) -> np.ndarray:
        """Length (2-D) or area (3-D) of each boundary facet."""
        bf = self.boundary_facets
        if self.dim == 2:
            d = self.nodes[bf[:, 0]] - self.nodes[bf[:, 1]]
            return np.linalg.norm(d, axis=1)
        a = self.nodes[bf[:, 1]] - self.nodes[bf[:, 0]]
        b = self.nodes[bf[:, 2]] - self.nodes[bf[:, 0]]
        return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)

    @cached_property
    def facet_centroids(self) -> np.ndarray:
        return self.nodes[self.boundary_facets].mean(axis=1)

    @cached_property
    def edge_lengths(self) -> np.ndarray:
        d = self.nodes[self.edges[:, 0]] - self.nodes[self.edges[:, 1]]
        return np.linalg.norm(d, axis=1)

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
    def cell_sizes(self) -> np.ndarray:
        """Local mesh size per element: mean of its edge lengths."""
        p = self.nodes[self.cells]
        acc = np.zeros(len(self.cells))
        pairs = _EDGE_LOCAL[self.dim]
        for i, j in pairs:
            acc += np.linalg.norm(p[:, i] - p[:, j], axis=1)
        return acc / len(pairs)

    @cached_property
    def cell_adjacency(self) -> tuple[np.ndarray, np.ndarray]:
        """Pairs of face-adjacent cells (k, 2) and the facet measure between."""
        f = np.vstack([self.cells[:, list(fl)] for fl in _FACET_LOCAL[self.dim]])
        owner = np.tile(np.arange(len(self.cells)), len(_FACET_LOCAL[self.dim]))
        fs = np.sort(f, axis=1)
        order = np.lexsort(fs.T)
        fs, owner = fs[order], owner[order]
        same = np.all(fs[1:] == fs[:-1], axis=1)
        a = owner[:-1][same]
        b = owner[1:][same]
        return np.column_stack([a, b]), fs[:-1][same]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def sha(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.nodes).tobytes())
        h.update(np.ascontiguousarray(self.cells).tobytes())
        return h.hexdigest()[:16]


def generate_mesh(problem, size_fn, *, model_name: str = "model") -> Mesh:
    """Build the problem geometry and mesh it with the given size field.

    ``size_fn(x, y, z) -> float`` is evaluated by Gmsh's size callback.
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
        if problem.dim == 3:
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)

        h_floor = float(problem.h_min)

        def cb(dim, tag, x, y, z, lc):
            return max(float(size_fn(x, y, z)), h_floor)

        gmsh.model.mesh.setSizeCallback(cb)
        gmsh.model.mesh.generate(problem.dim)
        gmsh.model.mesh.removeSizeCallback()

        tags, coords, _ = gmsh.model.mesh.getNodes()
        tags = np.asarray(tags, dtype=np.int64)
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        order = np.argsort(tags)
        tags = tags[order]
        coords = coords[order]
        remap = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
        remap[tags] = np.arange(len(tags))

        want_type = 2 if problem.dim == 2 else 4  # tri3 / tet4
        etypes, _, enodes = gmsh.model.mesh.getElements(problem.dim)
        blocks = []
        for etype, conn in zip(etypes, enodes):
            if etype == want_type:
                blocks.append(
                    np.asarray(conn, dtype=np.int64).reshape(-1, problem.dim + 1)
                )
        if not blocks:
            raise RuntimeError("Gmsh produced no linear simplices")
        cells = remap[np.vstack(blocks)]

        used = np.zeros(len(coords), dtype=bool)
        used[cells.ravel()] = True
        if not used.all():
            new_index = np.cumsum(used) - 1
            coords, cells = coords[used], new_index[cells]
        if problem.dim == 2:
            coords = coords.copy()
            coords[:, 2] = 0.0
        return Mesh(nodes=coords, cells=cells, dim=problem.dim)
    finally:
        gmsh.model.remove()


def generate_uniform(problem, h: float) -> Mesh:
    return generate_mesh(problem, lambda x, y, z: h)
