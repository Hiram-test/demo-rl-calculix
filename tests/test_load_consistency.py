"""Gate G7: imprinted traction resultant equals p×A on any mesh."""

import numpy as np
import pytest

from visionamr.calculix import assemble_nodal_forces
from visionamr.geometry import (
    analytic_load_resultant,
    make_bearing_block,
    make_deck_panel,
    make_lbracket,
    make_plate_holes,
)
from visionamr.mesher import generate_uniform


@pytest.mark.parametrize(
    "factory",
    [make_lbracket, make_plate_holes, make_bearing_block],
)
def test_resultant_mesh_independent(factory):
    problem = factory()
    target = analytic_load_resultant(problem)
    rels = []
    for h in (problem.h0, problem.h0 / 1.5):
        mesh = generate_uniform(problem, h)
        F = assemble_nodal_forces(mesh, problem).sum(axis=0)
        rel = np.linalg.norm(F - target) / max(np.linalg.norm(target), 1e-30)
        rels.append(rel)
        assert rel < 1e-6, (problem.name, h, F, target, rel)
    # two meshes, same resultant
    assert max(rels) < 1e-6


def test_analytic_bearing_patch_area():
    p = make_bearing_block(patch=(140.0, 140.0), pressure=12.0)
    t = analytic_load_resultant(p)
    assert np.isclose(t[2], -12.0 * 140.0 * 140.0)
