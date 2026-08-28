import numpy as np

from visionamr.mesher import TriMesh
from visionamr.sizefield import (
    NodalSizeField,
    Region,
    RegionSizeField,
    lipschitz_smooth,
)


def unit_square_mesh() -> TriMesh:
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float)
    tris = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return TriMesh(nodes=nodes, tris=tris)


def test_lipschitz_smooth_bounds_gradient():
    mesh = unit_square_mesh()
    h = np.array([1.0, 1.0, 1.0, 1.0, 0.01])
    hs = lipschitz_smooth(mesh.nodes, mesh.edges, h, gradation=0.5)
    d = mesh.nodes[mesh.edges[:, 0]] - mesh.nodes[mesh.edges[:, 1]]
    L = np.hypot(d[:, 0], d[:, 1])
    diff = np.abs(hs[mesh.edges[:, 0]] - hs[mesh.edges[:, 1]])
    assert np.all(diff <= 0.5 * L + 1e-9)
    assert hs[4] == 0.01  # smoothing only shrinks, never grows the minimum


def test_nodal_size_field_interpolates():
    mesh = unit_square_mesh()
    h = np.array([1.0, 1.0, 1.0, 1.0, 0.2])
    f = NodalSizeField(mesh, h, gradation=10.0)  # high gradation: no change
    assert abs(f(0.5, 0.5) - 0.2) < 1e-6
    assert 0.2 < f(0.25, 0.25) < 1.0


def test_region_size_field_min_and_gradation():
    r1 = Region("hot", 0.0, 0.0, 1.0, 1.0, h=0.1)
    f = RegionSizeField([r1], h_background=1.0, gradation=0.5)
    assert f(0.5, 0.5) == 0.1          # inside
    assert abs(f(2.0, 0.5) - 0.6) < 1e-9  # 0.1 + 0.5*1.0
    assert f(9.0, 0.5) == 1.0          # background wins far away


def test_region_geometry_helpers():
    r = Region("a", 0.0, 0.0, 2.0, 1.0, h=0.5)
    assert r.area == 2.0
    assert r.center == (1.0, 0.5)
    assert bool(r.contains(np.array([1.0]), np.array([0.5]))[0])
    assert r.distance(np.array([3.0]), np.array([0.5]))[0] == 1.0
    assert r.with_h(0.25).h == 0.25


def test_boundary_edges_of_square():
    mesh = unit_square_mesh()
    assert len(mesh.boundary_edges) == 4
    assert len(mesh.edges) == 8
