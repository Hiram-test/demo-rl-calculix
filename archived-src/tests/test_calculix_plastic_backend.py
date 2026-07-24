from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from calculix_plastic_backend import (
    PlasticPlateConfig,
    StateAwareCalculixPlasticEnv,
    parse_calculix_plastic_dat,
)
from mesh_goal import GoalCondition
from state_aware_dqn_agent import KEEP, REFINE


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


def make_frd(displacement: float) -> str:
    return f"""    1Cexample
  100CL101
 -4  DISP        4    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -1         1 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2 {displacement:12.5E} 0.00000E+00 0.00000E+00
 -1         3 {displacement:12.5E} 0.00000E+00 0.00000E+00
 -1         4 0.00000E+00 0.00000E+00 0.00000E+00
 -3
 9999
"""


def make_dat(displacement: float) -> str:
    reaction = 10000.0 * displacement
    peeq = max(0.0, displacement - 0.002)
    stress = 200.0 + 1000.0 * peeq
    return f"""
 forces (fx,fy,fz) for set RIGHT and time  0.1000000E+01

         2  {reaction / 2: .6E}  0.000000E+00  0.000000E+00
         3  {reaction / 2: .6E}  0.000000E+00  0.000000E+00

 stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EALL and time  0.1000000E+01

         5   1  {stress: .6E}  0.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00
         5   2  {stress + 2: .6E}  0.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00
         6   1  {stress + 4: .6E}  0.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00
         6   2  {stress + 6: .6E}  0.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00

 equivalent plastic strain (elem, integ.pnt.,pe)for set EALL and time  0.1000000E+01

         5   1  {peeq: .6E}
         5   2  {peeq: .6E}
         6   1  {peeq * 2: .6E}
         6   2  {peeq * 2: .6E}
"""


class PlasticDatParserTests(unittest.TestCase):
    def test_last_blocks_are_averaged_per_element(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.dat"
            path.write_text(make_dat(0.01), encoding="utf-8")
            frame = parse_calculix_plastic_dat(path)
        self.assertAlmostEqual(frame.total_reaction[0], 100.0)
        self.assertAlmostEqual(frame.stresses[5][0], 209.0)
        self.assertAlmostEqual(frame.peeq[6], 0.016)


class PlasticBackendTests(unittest.TestCase):
    def make_env(self, directory: Path) -> StateAwareCalculixPlasticEnv:
        def fake_runner(command, cwd, timeout):
            executable = Path(command[0]).name
            if executable == "fake-gmsh":
                output_name = command[command.index("-o") + 1]
                (cwd / output_name).write_text(MSH_TEXT, encoding="utf-8")
            elif executable == "fake-ccx":
                deck = (cwd / "model.inp").read_text(encoding="utf-8")
                match = re.search(r"RIGHT,\s*1,\s*1,\s*([-+0-9.Ee]+)", deck)
                if match is None:
                    raise AssertionError("Prescribed displacement missing")
                displacement = float(match.group(1))
                (cwd / "model.frd").write_text(make_frd(displacement), encoding="utf-8")
                (cwd / "model.dat").write_text(make_dat(displacement), encoding="utf-8")
            else:
                raise AssertionError(command)
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        plate = PlasticPlateConfig(
            hole_radius=0.0,
            cells_x=2,
            cells_y=1,
            load_steps=4,
            target_displacement_x=0.01,
            max_neighbor_size_ratio=3.0,
        )
        return StateAwareCalculixPlasticEnv(
            plate=plate,
            simulations_root=str(directory / "sims"),
            gmsh_cmd=["fake-gmsh"],
            ccx_cmd=["fake-ccx"],
            global_mesh_size=1.0,
            cell_min_mesh_size=0.2,
            cell_max_mesh_size=2.0,
            min_elements=1,
            max_elements=100,
            command_runner=fake_runner,
        )

    def test_deck_uses_plane_stress_plasticity_and_displacement_control(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_env(Path(directory))
            result = env._run_analysis(Path(directory) / "run", env.cell_mesh_density, 4)
            deck = (Path(directory) / "run" / "model.inp").read_text(encoding="utf-8")
        self.assertIn("*ELEMENT, TYPE=CPS3", deck)
        self.assertIn("*PLASTIC, HARDENING=ISOTROPIC", deck)
        self.assertIn("LEFT, 1, 1, 0.0", deck)
        self.assertIn("ANCHOR, 2, 2, 0.0", deck)
        self.assertIn("RIGHT, 1, 1, 0.01", deck)
        self.assertGreater(result.reaction_force_x, 0.0)
        self.assertGreater(result.max_peeq, 0.0)

    def test_load_path_advances_and_terminates(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_env(Path(directory))
            env.compute_baseline(
                cache_dir=str(Path(directory) / "cache"),
                use_cache=False,
                baseline_mesh_size=0.5,
            )
            env.reset("episode")
            self.assertEqual(env.load_step_index, 1)
            done = False
            while not done:
                mask = env.get_action_mask(goal=GoalCondition())
                keep_candidates = [
                    cell_id for cell_id, row in mask.items() if row[KEEP]
                ]
                self.assertEqual(len(keep_candidates), 1)
                _, _, done, info = env.step({keep_candidates[0]: KEEP})
            self.assertEqual(env.load_step_index, env.plate.load_steps)
            self.assertTrue(info["load_path_complete"])
            self.assertAlmostEqual(env.load_fraction, 1.0)

    def test_state_contains_physics_but_not_reference_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_env(Path(directory))
            env.compute_baseline(
                cache_dir=str(Path(directory) / "cache"),
                use_cache=False,
                baseline_mesh_size=0.5,
            )
            env.reset("episode")
            state = env.build_state(GoalCondition(), max_steps=100)
        names = " ".join(env.BASE_CELL_FEATURE_NAMES).lower()
        self.assertNotIn("reference", names)
        self.assertNotIn("relative_cell_error", names)
        self.assertEqual(state.node_features.shape[1], env.CELL_FEATURE_DIM)
        self.assertEqual(state.global_features.shape[1], env.global_feature_dim)
        self.assertTrue(bool(state.action_mask.any()))
        self.assertEqual(state.action_mask.shape[1], 3)
        self.assertEqual(int(state.action_mask[:, KEEP].sum().item()), 1)


    def test_keep_advances_load_without_mesh_penalty(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_env(Path(directory))
            env.compute_baseline(
                cache_dir=str(Path(directory) / "cache"),
                use_cache=False,
                baseline_mesh_size=0.5,
            )
            env.reset("episode")
            sizes_before = dict(env.cell_mesh_density)
            mask = env.get_action_mask(goal=GoalCondition())
            keep_cells = [cell_id for cell_id, row in mask.items() if row[KEEP]]
            self.assertEqual(len(keep_cells), 1)
            _, _, done, info = env.step({keep_cells[0]: KEEP})
        self.assertFalse(done)
        self.assertEqual(env.load_step_index, 2)
        self.assertEqual(env.cell_mesh_density, sizes_before)
        self.assertTrue(info["intentional_keep"])
        self.assertEqual(info["reward_components"]["ineffective_penalty"], 0.0)

    def test_neighbor_gradation_is_hard_masked(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_env(Path(directory))
            env.reset("episode")
            env.cell_mesh_density[1] = 0.4
            env.cell_mesh_density[2] = 1.0
            mask = env.get_action_mask([1], GoalCondition())
        self.assertFalse(mask[1][REFINE])


if __name__ == "__main__":
    unittest.main()
