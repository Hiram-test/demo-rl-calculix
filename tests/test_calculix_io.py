import numpy as np

from visionamr.calculix import (
    assemble_nodal_forces,
    read_frd_displacements,
    write_inp,
)
from visionamr.geometry import make_bearing_block, make_plate_holes
from visionamr.mesher import Mesh


def small_mesh_2d() -> Mesh:
    nodes = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=float)
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    return Mesh(nodes=nodes, cells=cells, dim=2)


def test_nodal_forces_sum_to_traction_resultant_2d():
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)
    mesh = small_mesh_2d()
    F = assemble_nodal_forces(mesh, problem)
    assert np.isclose(F[:, 0].sum(), 100.0)  # edge length 1 x thickness 1 x 100
    assert np.isclose(F[:, 1].sum(), 0.0)
    loaded = np.nonzero(np.abs(F).sum(axis=1) > 0)[0]
    assert set(loaded) == {1, 2}


def _cube_mesh(W=400.0, D=400.0, H=120.0) -> Mesh:
    nodes = np.array(
        [
            [0, 0, 0], [W, 0, 0], [W, D, 0], [0, D, 0],
            [0, 0, H], [W, 0, H], [W, D, H], [0, D, H],
            [W / 2, D / 2, H / 2],
        ]
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


def test_nodal_forces_patch_resultant_3d():
    mesh = _cube_mesh()
    # full-cover patch: resultant = -p * top area
    full = make_bearing_block(patch=(400.0, 400.0), offset=(0.0, 0.0), pressure=10.0)
    F = assemble_nodal_forces(mesh, full)
    assert np.isclose(F[:, 2].sum(), -10.0 * 400.0 * 400.0)
    assert np.isclose(np.abs(F[:, :2]).sum(), 0.0)
    # corner patch that contains no top-facet centroid: no load picked up
    corner = make_bearing_block(patch=(60.0, 60.0), offset=(-170.0, -170.0), pressure=10.0)
    F2 = assemble_nodal_forces(mesh, corner)
    assert np.isclose(np.abs(F2).sum(), 0.0)


def test_write_inp_2d_structure(tmp_path):
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)
    mesh = small_mesh_2d()
    path = tmp_path / "model.inp"
    write_inp(path, mesh, problem, "unit test deck")
    text = path.read_text()
    for card in (
        "*NODE",
        "*ELEMENT, TYPE=CPS3",
        "*ELASTIC",
        "*SOLID SECTION",
        "*BOUNDARY",
        "*STATIC",
        "*CLOAD",
        "*NODE FILE",
        "*END STEP",
    ):
        assert card in text, card


def test_frd_parser_fixed_width(tmp_path):
    frd = tmp_path / "model.frd"
    lines = [
        "    1CHEADER",
        "  100CL  101 1.00000E+00           4                     0    1",
        " -4  DISP        4    1",
        " -5  D1          1    2    1    0",
        " -5  D2          1    2    2    0",
        " -5  D3          1    2    3    0",
        " -1         1 1.00000E+00-2.00000E+00 3.00000E+00",
        " -1         2 5.00000E-01 2.50000E-01 0.00000E+00",
        " -3",
    ]
    frd.write_text("\n".join(lines) + "\n")
    u = read_frd_displacements(frd, 2)
    assert np.isclose(u[0, 0], 1.0)
    assert np.isclose(u[0, 1], -2.0)  # adjacent signed values without space
    assert np.isclose(u[0, 2], 3.0)
    assert np.isclose(u[1, 1], 0.25)
