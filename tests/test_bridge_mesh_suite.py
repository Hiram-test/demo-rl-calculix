from __future__ import annotations

import unittest

import numpy as np

from bridge_mesh_suite.fem import edge_traction_load, solve_linear_plane_stress
from bridge_mesh_suite.meshes import nodes_on_coordinate, rectangle_mesh
from bridge_mesh_suite.scenarios import (
    run_all_scenarios,
    run_circular_opening_case,
    run_crack_case,
    run_load_introduction_case,
)


class BridgeMeshSuiteTests(unittest.TestCase):
    def test_plane_stress_patch(self) -> None:
        mesh = rectangle_mesh(100.0, 40.0, 10, 4)
        young, poisson, thickness, sigma = 210000.0, 0.3, 10.0, 75.0
        f = edge_traction_load(mesh, mesh.edge_sets["right"], (sigma, 0.0), thickness)
        left = nodes_on_coordinate(mesh, x=0.0)
        constraints = {2 * int(n): 0.0 for n in left}
        constraints[2 * int(left[0]) + 1] = 0.0
        sol = solve_linear_plane_stress(mesh, young, poisson, thickness, f, constraints)
        self.assertLess(sol.energy_balance_rel, 1e-8)
        self.assertAlmostEqual(float(np.median(sol.element_stress[:, 0])), sigma, delta=0.02 * sigma)

    def test_circular_hole_matches_kirsch(self) -> None:
        run = run_circular_opening_case()
        self.assertLess(abs(run.level_rows[-1]["peak_stress"] / 300.0 - 1.0), 0.06)
        self.assertLess(run.level_rows[-1]["energy_balance_rel"], 1e-8)

    def test_point_load_peak_grows_but_physical_qoi_stabilizes(self) -> None:
        run = run_load_introduction_case()
        self.assertGreater(run.level_rows[-1]["peak_stress"], run.level_rows[-2]["peak_stress"])
        self.assertLess(
            abs(run.level_rows[-1]["fixed_qoi"] / run.level_rows[-2]["fixed_qoi"] - 1.0),
            0.05,
        )
        self.assertLess(run.metadata["variant_row"]["peak_stress"], run.level_rows[-1]["peak_stress"])

    def test_crack_uses_energy_release_not_tip_stress(self) -> None:
        run = run_crack_case()
        self.assertGreater(run.level_rows[-1]["peak_stress"], run.level_rows[-2]["peak_stress"])
        self.assertLess(
            abs(run.level_rows[-1]["G_numeric"] / run.level_rows[-2]["G_numeric"] - 1.0),
            0.03,
        )
        theory = run.diagnostic.evidence["G_theory"]
        self.assertLess(abs(run.level_rows[-1]["G_numeric"] / theory - 1.0), 0.15)

    def test_full_suite_executes_energy_skill_for_every_scenario(self) -> None:
        runs = run_all_scenarios()
        self.assertGreaterEqual(len(runs), 4)
        for run in runs:
            energy = [s for s in run.diagnostic.skill_trace if s.name == "energy_consistency"]
            self.assertEqual(len(energy), 1)
            self.assertTrue(energy[0].passed)
            self.assertGreaterEqual(len(run.level_rows), 3)
            self.assertTrue(run.diagnostic.applied_plan)


if __name__ == "__main__":
    unittest.main()
