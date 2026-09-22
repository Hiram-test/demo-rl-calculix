import numpy as np

from visionamr.mesher import Mesh
from visionamr.sizefield import NodalSizeField, element_to_node_sizes, lipschitz_smooth


def unit_square_mesh() -> Mesh:
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]], dtype=float
    )
    cells = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return Mesh(nodes=nodes, cells=cells, dim=2)


def single_tet_mesh() -> Mesh:
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    return Mesh(nodes=nodes, cells=np.array([[0, 1, 2, 3]]), dim=3)


def test_lipschitz_smooth_bounds_gradient():
    mesh = unit_square_mesh()
    h = np.array([1.0, 1.0, 1.0, 1.0, 0.01])
    hs = lipschitz_smooth(mesh.nodes, mesh.edges, h, gradation=0.5)
    d = mesh.nodes[mesh.edges[:, 0]] - mesh.nodes[mesh.edges[:, 1]]
    L = np.linalg.norm(d, axis=1)
    diff = np.abs(hs[mesh.edges[:, 0]] - hs[mesh.edges[:, 1]])
    assert np.all(diff <= 0.5 * L + 1e-9)
    assert hs[4] == 0.01  # smoothing only shrinks, never grows the minimum


def test_nodal_size_field_interpolates():
    mesh = unit_square_mesh()
    h = np.array([1.0, 1.0, 1.0, 1.0, 0.2])
    f = NodalSizeField(mesh, h, gradation=10.0)  # high gradation: no change
    assert abs(f(0.5, 0.5) - 0.2) < 1e-6
    assert 0.2 < f(0.25, 0.25) < 1.0


def test_element_to_node_sizes_takes_min():
    mesh = unit_square_mesh()
    h_elem = np.array([1.0, 0.1, 1.0, 1.0])
    h_node = element_to_node_sizes(mesh, h_elem)
    for node in mesh.cells[1]:
        assert h_node[node] == 0.1


def test_mesh_geometry_2d():
    mesh = unit_square_mesh()
    assert len(mesh.boundary_facets) == 4
    assert len(mesh.edges) == 8
    assert np.isclose(mesh.measures.sum(), 1.0)


def test_mesh_geometry_3d():
    mesh = single_tet_mesh()
    assert np.isclose(mesh.measures[0], 1.0 / 6.0)
    assert len(mesh.boundary_facets) == 4
    assert len(mesh.edges) == 6
    assert np.isclose(mesh.facet_measures.min(), 0.5)  # three unit right faces
