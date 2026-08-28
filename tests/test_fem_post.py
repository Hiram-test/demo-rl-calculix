import numpy as np

from visionamr.fem_post import compute_post, element_B, plane_stress_C
from visionamr.geometry import make_plate_holes
from visionamr.indicators import zz_indicator
from visionamr.mesher import TriMesh


def patch_mesh() -> TriMesh:
    nodes = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [1, 0.5]], dtype=float)
    tris = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return TriMesh(nodes=nodes, tris=tris)


def problem_for(mesh_width=2.0):
    return make_plate_holes(width=mesh_width, height=1.0, holes=(), tension=100.0)


def test_uniform_tension_patch_field():
    """Patch test: linear displacement field reproduces constant stress."""

    problem = problem_for()
    mesh = patch_mesh()
    E, nu = problem.material.E, problem.material.nu
    sx = 100.0
    ux = sx / E * mesh.nodes[:, 0]
    uy = -nu * sx / E * mesh.nodes[:, 1]
    u = np.column_stack([ux, uy])
    post = compute_post(mesh, problem, u)
    assert np.allclose(post.stress[:, 0], sx, rtol=1e-9)
    assert np.allclose(post.stress[:, 1], 0.0, atol=1e-6)
    assert np.allclose(post.stress[:, 2], 0.0, atol=1e-6)
    # strain energy = 1/2 * sigma^2/E * volume
    area = mesh.areas.sum()
    assert np.isclose(post.U_total, 0.5 * sx**2 / E * area, rtol=1e-9)


def test_zz_indicator_zero_for_constant_stress():
    problem = problem_for()
    mesh = patch_mesh()
    sx = 50.0
    E, nu = problem.material.E, problem.material.nu
    u = np.column_stack([sx / E * mesh.nodes[:, 0], -nu * sx / E * mesh.nodes[:, 1]])
    post = compute_post(mesh, problem, u)
    eta2 = zz_indicator(problem, post)
    assert np.all(eta2 < 1e-16)


def test_zz_indicator_positive_for_nonsmooth_field():
    problem = problem_for()
    mesh = patch_mesh()
    rng = np.random.default_rng(0)
    u = rng.normal(scale=1e-3, size=(mesh.n_nodes, 2))
    post = compute_post(mesh, problem, u)
    eta2 = zz_indicator(problem, post)
    assert np.all(eta2 >= 0.0)
    assert eta2.sum() > 0.0


def test_element_B_shapes():
    mesh = patch_mesh()
    B = element_B(mesh)
    assert B.shape == (4, 3, 6)
    C = plane_stress_C(problem_for().material)
    assert np.allclose(C, C.T)
