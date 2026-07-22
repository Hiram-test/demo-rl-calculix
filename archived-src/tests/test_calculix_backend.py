from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from calculix_backend import (
    CalculixRunResult,
    Msh2Mesh,
    PlateConfig,
    StateAwareCalculixEnv,
    parse_frd_displacements,
    parse_msh2,
    triangle_response,
)
from mesh_goal import GoalCondition
from state_aware_dqn_agent import REFINE


MSH_TEXT = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 10 0 0
3 10 4 0
4 0 4 0
$EndNodes
$Elements
6
1 1 2 1 1 1 2
2 1 2 1 1 2 3
3 1 2 1 1 3 4
4 1 2 1 1 4 1
5 2 2 1 1 1 2 3
6 2 2 1 1 1 3 4
$EndElements
"""

FRD_TEXT = """    1Cexample
    2C
 -1    1 0.0 0.0 0.0
 -1    2 10.0 0.0 0.0
 -1    3 10.0 4.0 0.0
 -1    4 0.0 4.0 0.0
 -3
  100CL101
 -4  DISP        3    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -1    1 0.0 0.0 0.0
 -1    2 0.0 -1.0E-2 0.0
 -1    3 0.0 -1.0E-2 0.0
 -1    4 0.0 0.0 0.0
 -3
 9999
"""

FRD_FIXED_WIDTH_TEXT = """    1Cexample
  100CL101
 -4  DISP        4    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -1         1 9.06566E-05-9.63721E-02 0.00000E+00
 -1         2 0.00000E+00 0.00000E+00 0.00000E+00
 -1         3-8.01312E-02-3.04778E-01 0.00000E+00
 -3
 9999
"""


class CalculixIOTests(unittest.TestCase):
    def test_parse_msh2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.msh"
            path.write_text(MSH_TEXT, encoding="utf-8")
            mesh = parse_msh2(path)
        self.assertEqual(len(mesh.nodes), 4)
        self.assertEqual(mesh.triangles[5], (1, 2, 3))
        self.assertEqual(len(mesh.triangles), 2)

    def test_parse_last_frd_displacement_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.frd"
            path.write_text(FRD_TEXT, encoding="utf-8")
            values = parse_frd_displacements(path)
        self.assertAlmostEqual(values[2][1], -0.01)
        self.assertEqual(len(values), 4)

    def test_parse_native_fixed_width_frd_without_spaces_between_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.frd"
            path.write_text(FRD_FIXED_WIDTH_TEXT, encoding="utf-8")
            values = parse_frd_displacements(path)
        self.assertAlmostEqual(values[1][0], 9.06566e-05)
        self.assertAlmostEqual(values[1][1], -9.63721e-02)
        self.assertAlmostEqual(values[3][0], -8.01312e-02)
        self.assertAlmostEqual(values[3][1], -3.04778e-01)

    def test_triangle_response_zero_for_rigid_translation(self):
        mises, energy, strain = triangle_response(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            [(0.1, -0.2), (0.1, -0.2), (0.1, -0.2)],
            210_000.0,
            0.3,
            1.0,
        )
        self.assertAlmostEqual(mises, 0.0, places=8)
        self.assertAlmostEqual(energy, 0.0, places=8)
        np.testing.assert_allclose(strain, np.zeros(3), atol=1.0e-12)

    def test_pipeline_with_fake_gmsh_and_ccx(self):
        def fake_runner(command, cwd, timeout):
            executable = Path(command[0]).name
            if executable == "fake-gmsh":
                output_name = command[command.index("-o") + 1]
                (cwd / output_name).write_text(MSH_TEXT, encoding="utf-8")
            elif executable == "fake-ccx":
                (cwd / "model.frd").write_text(FRD_TEXT, encoding="utf-8")
            else:
                raise AssertionError(command)
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        plate = PlateConfig(
            hole_radius=0.0,
            cells_x=2,
            cells_y=1,
        )
        env = StateAwareCalculixEnv(
            plate=plate,
            gmsh_cmd=["fake-gmsh"],
            ccx_cmd=["fake-ccx"],
            min_elements=1,
            max_elements=100,
            global_mesh_size=1.0,
            cell_min_mesh_size=0.2,
            cell_max_mesh_size=2.0,
            command_runner=fake_runner,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = env._run_analysis(Path(directory), env.cell_mesh_density)
            deck = (Path(directory) / "model.inp").read_text(encoding="utf-8")
            geo = (Path(directory) / "model.geo").read_text(encoding="utf-8")
        self.assertGreater(result.qoi, 0.0)
        self.assertEqual(result.element_count, 2)
        self.assertIn("*ELEMENT, TYPE=CPE3", deck)
        self.assertIn("FIXED, 1, 2, 0.0", deck)
        self.assertIn("Field[1] = Box", geo)
        self.assertEqual(len(result.cell_features[1]), len(env.BASE_CELL_FEATURE_NAMES))


class FakeStateAwareCalculixEnv(StateAwareCalculixEnv):
    def _run_analysis(self, workdir, mesh_sizes):
        workdir.mkdir(parents=True, exist_ok=True)
        sizes = {int(key): float(value) for key, value in mesh_sizes.items()}
        element_count = int(round(sum(20.0 / (value * value) for value in sizes.values())))
        qoi = 1.0 + 0.05 * float(np.mean([value * value for value in sizes.values()]))
        cell_to_elements = {}
        cell_energy = {}
        cell_features = {}
        next_element = 1
        for cell_id in sorted(self.virtual_cells):
            count = max(1, int(round(20.0 / (sizes[cell_id] * sizes[cell_id]))))
            ids = list(range(next_element, next_element + count))
            next_element += count
            cell_to_elements[cell_id] = ids
            cell_energy[cell_id] = qoi / len(self.virtual_cells)
            row = [0.0] * len(self.BASE_CELL_FEATURE_NAMES)
            row[-1] = 1.0
            cell_features[cell_id] = row
        mesh = Msh2Mesh(nodes={1: (0.0, 0.0, 0.0)}, triangles={1: (1, 1, 1)})
        return CalculixRunResult(
            qoi=qoi,
            element_count=element_count,
            node_count=1,
            displacements={1: (0.0, 0.0, 0.0)},
            cell_to_elements=cell_to_elements,
            cell_energy=cell_energy,
            cell_features=cell_features,
            mesh_signature=(element_count, tuple((key, len(value)) for key, value in sorted(cell_to_elements.items()))),
            workdir=str(workdir),
            mesh=mesh,
        )


class CalculixEnvironmentTests(unittest.TestCase):
    def make_env(self, root):
        return FakeStateAwareCalculixEnv(
            plate=PlateConfig(hole_radius=0.0, cells_x=2, cells_y=1),
            simulations_root=str(root),
            gmsh_cmd=["fake-gmsh"],
            ccx_cmd=["fake-ccx"],
            global_mesh_size=1.0,
            cell_min_mesh_size=0.3,
            cell_max_mesh_size=2.0,
            max_elements=500,
            min_elements=1,
            refine_step_size=0.2,
            coarsen_step_size=0.2,
        )

    def test_state_changes_after_refinement(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_env(Path(directory))
            env.compute_baseline(
                cache_dir=str(Path(directory) / "cache"),
                use_cache=False,
                baseline_mesh_size=0.4,
            )
            env.reset("test")
            goal = GoalCondition().normalized()
            before = env.build_state(goal, max_steps=10)
            _, reward, done, info = env.step({1: REFINE})
            after = env.build_state(goal, max_steps=10)
        self.assertFalse(done)
        self.assertFalse(info["state_rollback"])
        self.assertNotEqual(before.node_features[0, -9].item(), after.node_features[0, -9].item())
        self.assertFalse(np.allclose(before.global_features.numpy(), after.global_features.numpy()))
        self.assertIsInstance(reward, float)

    def test_refine_bound_is_hard_masked(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_env(Path(directory))
            env.reset("test")
            env.cell_mesh_density[1] = 0.31
            mask = env.get_action_mask([1], GoalCondition())
        self.assertFalse(mask[1][REFINE])

    def test_goal_json_normalizes_priorities(self):
        goal = GoalCondition.from_mapping(
            {
                "accuracy_priority": 6,
                "resource_priority": 3,
                "localization_priority": 1,
            }
        )
        self.assertAlmostEqual(
            goal.accuracy_priority + goal.resource_priority + goal.localization_priority,
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
