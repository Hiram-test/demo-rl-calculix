import numpy as np

from visionamr.baselines.local_prediction import (
    element_to_node_sizes,
    predicted_sizes,
)
from visionamr.mesher import TriMesh


def strip_mesh(n=8) -> TriMesh:
    """A strip of 2n right triangles on [0,n]x[0,1]."""

    nodes, tris = [], []
    for i in range(n + 1):
        nodes.append([i, 0.0])
        nodes.append([i, 1.0])
    for i in range(n):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        tris.append([a, c, b])
        tris.append([b, c, d])
    return TriMesh(nodes=np.array(nodes, float), tris=np.array(tris))


def test_budget_mode_hits_predicted_count():
    mesh = strip_mesh()
    rng = np.random.default_rng(1)
    eta2 = rng.uniform(0.1, 2.0, size=mesh.n_tris)
    h = predicted_sizes(mesh, eta2, n_target=200)
    ratio = h / mesh.tri_sizes
    predicted_elems = np.sum(ratio ** -2.0)
    assert abs(predicted_elems - 200) / 200 < 0.15


def test_high_error_elements_get_smaller_sizes():
    mesh = strip_mesh()
    eta2 = np.ones(mesh.n_tris)
    eta2[0] = 100.0  # one dominant element
    h = predicted_sizes(mesh, eta2, n_target=100)
    assert h[0] < h[5]


def test_multi_level_jump_allowed_in_one_shot():
    """The marked element may refine by more than one halving at once."""

    mesh = strip_mesh()
    eta2 = np.full(mesh.n_tris, 1e-6)
    eta2[0] = 10.0
    h = predicted_sizes(mesh, eta2, n_target=400, ratio_bounds=(1 / 8, 3.0))
    assert h[0] < 0.3 * mesh.tri_sizes[0]  # jumped past a single 1/2 level


def test_error_target_mode_monotone():
    mesh = strip_mesh()
    eta2 = np.linspace(0.1, 1.0, mesh.n_tris)
    # current total sqrt(sum eta2) ~ 2.97: one mild and one tight target
    # (a target far below the per-round clip saturates both, by design)
    h_tight = predicted_sizes(mesh, eta2, e_target=0.5)
    h_loose = predicted_sizes(mesh, eta2, e_target=2.5)
    assert h_tight.mean() < h_loose.mean()


def test_element_to_node_sizes_takes_min():
    mesh = strip_mesh(2)
    h_elem = np.array([1.0, 0.1, 1.0, 1.0])
    h_node = element_to_node_sizes(mesh, h_elem)
    for node in mesh.tris[1]:
        assert h_node[node] == 0.1
