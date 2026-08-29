import numpy as np

from visionamr.fem_post import compute_post, elastic_C, element_B
from visionamr.geometry import make_bearing_block, make_plate_holes
from visionamr.indicators import zz_indicator
from visionamr.mesher import Mesh


def patch_mesh_2d() -> Mesh:
    nodes = np.array(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0.5, 0]], dtype=float
    )
    cells = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return Mesh(nodes=nodes, cells=cells, dim=2)


def patch_mesh_3d() -> Mesh:
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [0.5, 0.5, 0.5],
        ],
        dtype=float,
    )
    faces = [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4),
    ]
    cells = []
    for a, b, c, d in faces:
        cells.append([a, b, c, 8])
        cells.append([a, c, d, 8])
    return Mesh(nodes=nodes, cells=np.array(cells), dim=3)


def test_2d_uniform_tension_patch_field():
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)
    mesh = patch_mesh_2d()
    E, nu = problem.material.E, problem.material.nu
    sx = 100.0
    u = np.column_stack(
        [sx / E * mesh.nodes[:, 0], -nu * sx / E * mesh.nodes[:, 1], np.zeros(5)]
    )
    post = compute_post(mesh, problem, u)
    assert np.allclose(post.stress[:, 0], sx, rtol=1e-9)
    assert np.allclose(post.stress[:, 1], 0.0, atol=1e-6)
    area = mesh.measures.sum()
    assert np.isclose(post.U_total, 0.5 * sx**2 / E * area, rtol=1e-9)


def test_3d_uniaxial_patch_field():
    """3-D patch test: uniaxial stress state reproduced exactly."""

    problem = make_bearing_block()
    mesh = patch_mesh_3d()
    E, nu = problem.material.E, problem.material.nu
    sz = -12.0
    u = np.column_stack(
        [
            -nu * sz / E * mesh.nodes[:, 0],
            -nu * sz / E * mesh.nodes[:, 1],
            sz / E * mesh.nodes[:, 2],
        ]
    )
    post = compute_post(mesh, problem, u)
    assert np.allclose(post.stress[:, 2], sz, rtol=1e-8)
    assert np.allclose(post.stress[:, 0], 0.0, atol=1e-6)
    assert np.allclose(post.stress[:, 3:], 0.0, atol=1e-6)
    vol = mesh.measures.sum()
    assert np.isclose(post.U_total, 0.5 * sz**2 / E * vol, rtol=1e-8)
    assert np.allclose(post.vm_elem, abs(sz), rtol=1e-8)


def test_zz_indicator_zero_for_constant_stress_2d_and_3d():
    p2 = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)
    m2 = patch_mesh_2d()
    E, nu = p2.material.E, p2.material.nu
    u2 = np.column_stack(
        [50.0 / E * m2.nodes[:, 0], -nu * 50.0 / E * m2.nodes[:, 1], np.zeros(5)]
    )
    eta2 = zz_indicator(p2, compute_post(m2, p2, u2))
    assert np.all(eta2 < 1e-16)

    p3 = make_bearing_block()
    m3 = patch_mesh_3d()
    E3, nu3 = p3.material.E, p3.material.nu
    u3 = np.column_stack(
        [
            -nu3 * -10.0 / E3 * m3.nodes[:, 0],
            -nu3 * -10.0 / E3 * m3.nodes[:, 1],
            -10.0 / E3 * m3.nodes[:, 2],
        ]
    )
    eta3 = zz_indicator(p3, compute_post(m3, p3, u3))
    assert np.all(eta3 < 1e-16)


def test_zz_indicator_positive_for_nonsmooth_field():
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)
    mesh = patch_mesh_2d()
    rng = np.random.default_rng(0)
    u = np.zeros((mesh.n_nodes, 3))
    u[:, :2] = rng.normal(scale=1e-3, size=(mesh.n_nodes, 2))
    post = compute_post(mesh, problem, u)
    eta2 = zz_indicator(problem, post)
    assert np.all(eta2 >= 0.0)
    assert eta2.sum() > 0.0


def test_element_B_shapes():
    assert element_B(patch_mesh_2d()).shape == (4, 3, 6)
    assert element_B(patch_mesh_3d()).shape == (12, 6, 12)
    C3 = elastic_C(make_bearing_block().material, 3)
    assert np.allclose(C3, C3.T)
