import numpy as np

from visionamr.calculix import (
    assemble_nodal_forces,
    read_frd_displacements,
    write_inp,
)
from visionamr.geometry import make_plate_holes
from visionamr.mesher import TriMesh


def small_mesh() -> TriMesh:
    nodes = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    return TriMesh(nodes=nodes, tris=tris)


def test_nodal_forces_sum_to_traction_resultant():
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)
    mesh = small_mesh()
    F = assemble_nodal_forces(mesh, problem)
    # right edge length 1, thickness 1, traction 100 -> resultant (100, 0)
    assert np.isclose(F[:, 0].sum(), 100.0)
    assert np.isclose(F[:, 1].sum(), 0.0)
    # only the two right-edge nodes carry load
    loaded = np.nonzero(np.abs(F).sum(axis=1) > 0)[0]
    assert set(loaded) == {1, 2}


def test_write_inp_structure(tmp_path):
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)
    mesh = small_mesh()
    path = tmp_path / "model.inp"
    write_inp(path, mesh, problem, "unit test deck")
    text = path.read_text()
    for card in (
        "*NODE",
        "*ELEMENT, TYPE=CPS3",
        "*NSET, NSET=CLAMP",
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
        " -1         1 1.00000E+00-2.00000E+00 0.00000E+00",
        " -1         2 5.00000E-01 2.50000E-01 0.00000E+00",
        " -3",
    ]
    frd.write_text("\n".join(lines) + "\n")
    u = read_frd_displacements(frd, 2)
    assert np.isclose(u[0, 0], 1.0)
    assert np.isclose(u[0, 1], -2.0)  # adjacent signed values without space
    assert np.isclose(u[1, 0], 0.5)
    assert np.isclose(u[1, 1], 0.25)
