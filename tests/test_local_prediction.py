import numpy as np

from visionamr.baselines.local_prediction import predicted_sizes
from visionamr.mesher import Mesh


def strip_mesh(n=8) -> Mesh:
    """A strip of 2n right triangles on [0,n]x[0,1]."""

    nodes, cells = [], []
    for i in range(n + 1):
        nodes.append([i, 0.0, 0.0])
        nodes.append([i, 1.0, 0.0])
    for i in range(n):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        cells.append([a, c, b])
        cells.append([b, c, d])
    return Mesh(nodes=np.array(nodes, float), cells=np.array(cells), dim=2)


def test_budget_mode_hits_predicted_count():
    mesh = strip_mesh()
    rng = np.random.default_rng(1)
    eta2 = rng.uniform(0.1, 2.0, size=mesh.n_cells)
    h = predicted_sizes(mesh, eta2, n_target=200)
    ratio = h / mesh.cell_sizes
    predicted_elems = np.sum(ratio ** -2.0)
    assert abs(predicted_elems - 200) / 200 < 0.15


def test_budget_mode_dimension_exponent():
    """In 3-D accounting, the same size ratios buy fewer elements."""

    mesh = strip_mesh()
    eta2 = np.linspace(0.5, 1.5, mesh.n_cells)
    h2 = predicted_sizes(mesh, eta2, n_target=300, d=2)
    h3 = predicted_sizes(mesh, eta2, n_target=300, d=3)
    # to reach the same element budget, 3-D needs milder refinement
    assert h3.mean() > h2.mean()


def test_high_error_elements_get_smaller_sizes():
    mesh = strip_mesh()
    eta2 = np.ones(mesh.n_cells)
    eta2[0] = 100.0
    h = predicted_sizes(mesh, eta2, n_target=100)
    assert h[0] < h[5]


def test_multi_level_jump_allowed_in_one_shot():
    mesh = strip_mesh()
    eta2 = np.full(mesh.n_cells, 1e-6)
    eta2[0] = 10.0
    h = predicted_sizes(mesh, eta2, n_target=400, ratio_bounds=(1 / 8, 3.0))
    assert h[0] < 0.3 * mesh.cell_sizes[0]


def test_error_target_mode_monotone():
    mesh = strip_mesh()
    eta2 = np.linspace(0.1, 1.0, mesh.n_cells)
    h_tight = predicted_sizes(mesh, eta2, e_target=0.5)
    h_loose = predicted_sizes(mesh, eta2, e_target=2.5)
    assert h_tight.mean() < h_loose.mean()
