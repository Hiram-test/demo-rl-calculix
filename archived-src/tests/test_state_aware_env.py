from __future__ import annotations

import sys
import types
import unittest


# Unit-test the wrapper without launching Abaqus.  The production repository has
# the full archived abaqus_env.py beside state_aware_env.py.
fake_module = types.ModuleType("abaqus_env")


class FakeAbaqusEnv:
    def __init__(self, *args, **kwargs):
        self.global_mesh_size = float(kwargs.get("global_mesh_size", 300.0))
        self.max_elements = int(kwargs.get("max_elements", 1000))
        self.min_elements = int(kwargs.get("min_elements", 10))
        self.cell_min_mesh_size = kwargs.get("cell_min_mesh_size", 100.0)
        self.cell_max_mesh_size = kwargs.get("cell_max_mesh_size", 400.0)
        self.refine_step_size = float(kwargs.get("refine_step_size", 0.05))
        self.coarsen_step_size = float(kwargs.get("coarsen_step_size", 0.05))
        self.accuracy_weight = float(kwargs.get("accuracy_weight", 1.0))
        self.resource_weight = float(kwargs.get("resource_weight", 1.0))
        self.step_index = 0
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        self.baseline_allse = 100.0
        self.initial_allse = 120.0
        self.baseline_cell_strain_energy = {1: 10.0, 2: 20.0}
        self.cell_mesh_density = {1: 100.0, 2: 300.0}
        self.cell_adjacency = {1: [2], 2: [1]}
        self.cell_to_elements_map = {1: [1, 2], 2: [3, 4]}
        self._last_obs = {
            "last_reward": 0.0,
            "resource_usage": 0.5,
            "cell_features": {1: [0.0] * 52, 2: [1.0] * 52},
        }

    def reset(self, run_id=None):
        self.step_index = 0
        return self._last_obs

    def step(self, action_params):
        self.step_index += 1
        return self._last_obs, -2.0, False, {"state_rollback": True}

    def get_cell_observations(self):
        result = {}
        for raw_cell_id, values in self._last_obs.get("cell_features", {}).items():
            cell_id = int(raw_cell_id)
            result[cell_id] = {
                "self": list(values),
                "neighbors": [
                    {"cell_id": neighbor, "features": []}
                    for neighbor in self.cell_adjacency.get(cell_id, [])
                ],
            }
        return result


fake_module.AbaqusEnv = FakeAbaqusEnv
sys.modules["abaqus_env"] = fake_module

from state_aware_env import GoalCondition, StateAwareAbaqusEnv  # noqa: E402


class StateAwareEnvironmentTests(unittest.TestCase):
    def make_env(self):
        return StateAwareAbaqusEnv(
            global_mesh_size=300.0,
            max_elements=1000,
            min_elements=10,
            cell_min_mesh_size=100.0,
            cell_max_mesh_size=400.0,
            refine_step_size=0.05,
            coarsen_step_size=0.05,
        )

    def test_mesh_density_is_explicit_state_feature(self):
        env = self.make_env()
        before = env.get_augmented_cell_observations()[2]["self"]
        env.cell_mesh_density[2] = 240.0
        after = env.get_augmented_cell_observations()[2]["self"]
        self.assertEqual(len(before), env.CELL_FEATURE_DIM)
        self.assertNotEqual(before[env.BASE_CELL_FEATURE_DIM], after[env.BASE_CELL_FEATURE_DIM])

    def test_action_mask_enforces_mesh_bounds(self):
        env = self.make_env()
        env.cell_mesh_density = {1: 100.0, 2: 400.0}
        mask = env.get_action_mask([1, 2])
        self.assertEqual(mask[1], [False, True])
        self.assertEqual(mask[2], [True, False])

    def test_resource_reserve_blocks_refinement(self):
        env = self.make_env()
        env._last_obs["resource_usage"] = 0.98
        goal = GoalCondition(reserve_budget_fraction=0.05)
        mask = env.get_action_mask([1, 2], goal)
        self.assertFalse(mask[1][0])
        self.assertFalse(mask[2][0])
        self.assertTrue(mask[1][1])

    def test_failed_pair_is_blocked_for_next_decision(self):
        env = self.make_env()
        env.cell_mesh_density[2] = 300.0
        _, reward, _, info = env.step({2: 0})
        self.assertLess(reward, 0.0)
        self.assertTrue(info["state_rollback"])
        mask = env.get_action_mask([2])
        self.assertFalse(mask[2][0])
        self.assertTrue(mask[2][1])

    def test_built_state_has_fixed_schema_and_global_context(self):
        env = self.make_env()
        state = env.build_state(GoalCondition(), max_steps=20)
        self.assertEqual(state.node_features.shape, (2, env.CELL_FEATURE_DIM))
        self.assertEqual(state.global_features.shape, (1, env.global_feature_dim))
        self.assertEqual(state.action_mask.shape, (2, 2))
        self.assertEqual(state.cell_ids, (1, 2))


if __name__ == "__main__":
    unittest.main()
