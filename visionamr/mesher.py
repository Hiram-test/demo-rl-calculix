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
_MIN_TETRA_SCALED_JACOBIAN = 1.0e-12  # Reject only numerically collapsed tetrahedra through a method-independent geometry-quality floor.


class GmshMeshingError(RuntimeError):  # Expose an explicitly identified native meshing failure to protocol runners.
    """Report a Gmsh numerical/materialization failure with no usable simplex mesh."""  # Document the narrow retained-failure category.

_EDGE_LOCAL = {
    2: [(0, 1), (1, 2), (2, 0)],
    3: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
}
_FACET_LOCAL = {
    2: [(0, 1), (1, 2), (2, 0)],
    3: [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
}


def _minimum_tetra_scaled_jacobian(nodes: np.ndarray, cells: np.ndarray) -> float:  # Measure the signed determinant relative to the longest-edge cube for every tetrahedron.
    points = np.asarray(nodes, dtype=float)  # Normalize node coordinates without mutating the generated mesh.
    tetrahedra = np.asarray(cells, dtype=np.int64)  # Normalize zero-based tetrahedral connectivity for indexed geometry operations.
    if points.ndim != 2 or points.shape[1] != 3 or tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4 or len(tetrahedra) == 0:  # Require one nonempty three-dimensional linear-tetrahedron mesh.
        return float("-inf")  # Force deterministic fallback or a typed failure for an unusable mesh.
    element_points = points[tetrahedra]  # Gather the four coordinates belonging to every tetrahedron.
    jacobians = np.stack((element_points[:, 1] - element_points[:, 0], element_points[:, 2] - element_points[:, 0], element_points[:, 3] - element_points[:, 0]), axis=2)  # Build signed C3D4 Jacobian matrices in connectivity order.
    determinants = np.linalg.det(jacobians)  # Compute signed six-times-volume values before text serialization.
    edge_lengths = np.column_stack([np.linalg.norm(element_points[:, left] - element_points[:, right], axis=1) for left, right in _EDGE_LOCAL[3]])  # Measure all six tetrahedral edges under one scale convention.
    longest_edges = np.max(edge_lengths, axis=1)  # Select the local length scale without using physics, errors, budgets, or method identity.
    denominators = np.maximum(longest_edges**3, np.finfo(float).tiny)  # Protect the dimensionless ratio from a literal zero-length denominator.
    qualities = determinants / denominators  # Retain orientation while normalizing away absolute geometry scale.
    if np.any(~np.isfinite(qualities)):  # Treat NaN or infinity as invalid native geometry evidence.
        return float("-inf")  # Trigger the same deterministic fallback used for a collapsed signed Jacobian.
    return float(np.min(qualities))  # Report the worst element so no isolated sliver is hidden by an aggregate statistic.


def _current_gmsh_tetra_quality(gmsh_module) -> float:  # Inspect the active Gmsh model before exporting its connectivity to CalculiX.
    tags, coordinates, _ = gmsh_module.model.mesh.getNodes()  # Read all generated node tags and double-precision coordinates.
    node_tags = np.asarray(tags, dtype=np.int64)  # Normalize opaque one-based Gmsh node identifiers.
    node_coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 3)  # Restore the three-coordinate node matrix.
    if node_tags.size == 0:  # Reject an empty generated node collection explicitly.
        return float("-inf")  # Trigger fallback before downstream tag remapping can fail.
    order = np.argsort(node_tags)  # Establish the same deterministic node order used by the returned Mesh.
    node_tags = node_tags[order]  # Sort identifiers before building a dense remapping table.
    node_coordinates = node_coordinates[order]  # Keep coordinates aligned with their sorted identifiers.
    remap = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)  # Allocate a dense tag-to-row mapping for generated connectivity.
    remap[node_tags] = np.arange(len(node_tags), dtype=np.int64)  # Map every present one-based tag to its sorted zero-based coordinate row.
    element_types, _, element_nodes = gmsh_module.model.mesh.getElements(3)  # Read only volume-element blocks from the active model.
    blocks = [np.asarray(connectivity, dtype=np.int64).reshape(-1, 4) for element_type, connectivity in zip(element_types, element_nodes) if int(element_type) == 4]  # Select every linear four-node tetrahedron block without accepting another element family.
    if not blocks:  # Reject a volume mesh with no supported C3D4 elements.
        return float("-inf")  # Trigger fallback before the public empty-simplex error boundary.
    tetrahedra = remap[np.vstack(blocks)]  # Convert Gmsh tags to the deterministic zero-based Mesh convention.
    if np.any(tetrahedra < 0):  # Reject connectivity that references a node absent from the generated node table.
        return float("-inf")  # Trigger the common fallback instead of indexing an unrelated row.
    return _minimum_tetra_scaled_jacobian(node_coordinates, tetrahedra)  # Apply the pure signed and scale-normalized quality metric.


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


def generate_mesh(
    problem, size_fn, *, model_name: str = "model", h_floor: float | None = None
) -> Mesh:
    """Build the problem geometry and mesh it with the given size field.

    ``size_fn(x, y, z) -> float`` is evaluated by Gmsh's size callback.
    ``h_floor`` overrides ``problem.h_min`` (used by the graded reference,
    whose singular-line floor is h_ref/8 and may be finer than the method h_min).
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

        h_floor = float(problem.h_min if h_floor is None else h_floor)

        def cb(dim, tag, x, y, z, lc):
            return max(float(size_fn(x, y, z)), h_floor)

        gmsh.model.mesh.setSizeCallback(cb)
        try:  # Guarantee callback cleanup after successful generation, native failure, or rejected fallback geometry.
            gmsh.model.mesh.generate(problem.dim)  # Generate the requested-dimensional mesh with the frozen primary algorithm and size callback.
            if problem.dim == 3 and _current_gmsh_tetra_quality(gmsh) <= _MIN_TETRA_SCALED_JACOBIAN:  # Apply one common geometry-only quality rule before any solver, estimator, or method can inspect the mesh.
                gmsh.model.mesh.clear()  # Discard the complete invalid HXT mesh without deleting selected elements or creating internal cracks.
                gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Retry deterministically with Gmsh Delaunay under the unchanged size callback and geometry.
                gmsh.model.mesh.generate(problem.dim)  # Regenerate the entire conforming volume mesh through the shared fallback path.
                if _current_gmsh_tetra_quality(gmsh) <= _MIN_TETRA_SCALED_JACOBIAN:  # Require the fallback mesh to satisfy the identical signed quality floor.
                    raise GmshMeshingError("Gmsh produced collapsed tetrahedra after deterministic Delaunay fallback")  # Preserve a typed native failure instead of passing invalid physics to CalculiX.
        finally:  # Clear the model-owned callback even when quality validation raises.
            gmsh.model.mesh.removeSizeCallback()  # Prevent a failed model from leaking its Python callback into the next Gmsh model.

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
            raise GmshMeshingError("Gmsh produced no linear simplices")  # Preserve this native empty-mesh outcome as typed numerical evidence.
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
