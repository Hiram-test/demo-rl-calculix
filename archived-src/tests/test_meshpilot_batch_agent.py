import json
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import tempfile
import types
import unittest


# Keep the contract/scope tests dependency-light. The real repository provides
# calculix_backend; this stub supplies only the schema needed by these tests.
backend = types.ModuleType("calculix_backend")


@dataclass(frozen=True)
class PlateConfig:
    length: float = 10.0
    height: float = 4.0
    thickness: float = 1.0
    young_modulus: float = 210000.0
    poisson_ratio: float = 0.3
    load_x: float = 0.0
    load_y: float = -1000.0
    hole_center_x: float = 4.0
    hole_center_y: float = 2.0
    hole_radius: float = 0.75
    cells_x: int = 8
    cells_y: int = 4
    gmsh_algorithm: int = 6

    def validated(self):
        if self.length <= 0 or self.height <= 0 or self.thickness <= 0:
            raise ValueError
        return self


class StateAwareCalculixEnv:
    BASE_CELL_FEATURE_NAMES = (
        "mean_mises",
        "std_mises",
        "max_mises",
    ) + tuple(f"unused_{index}" for index in range(17))


backend.PlateConfig = PlateConfig
backend.StateAwareCalculixEnv = StateAwareCalculixEnv
sys.modules.setdefault("calculix_backend", backend)

import meshpilot_batch_agent as batch_module

from meshpilot_batch_agent import (
    BatchRequest,
    expand_cases,
    hotspot_candidates,
    order_cases,
    warm_start_is_acceptable,
)
from meshpilot_pso import ObjectiveValue


class MeshPilotBatchAgentTests(unittest.TestCase):
    def _request(self, directory: str) -> BatchRequest:
        payload = {
            "request_id": "test",
            "user_role": "bridge_detail_engineer",
            "intent": "batch",
            "base_case": {
                "length": 10.0,
                "height": 4.0,
                "thickness": 1.0,
                "young_modulus": 210000.0,
                "poisson_ratio": 0.3,
                "load_x": 0.0,
                "hole_center_y": 2.0,
                "cells_x": 4,
                "cells_y": 2,
                "gmsh_algorithm": 6,
            },
            "sweep_mode": "zip",
            "sweep": {
                "hole_radius": [0.6, 1.0],
                "hole_center_x": [4.0, 4.6],
                "load_y": [-900.0, -1200.0],
            },
            "batch_order": "nearest_path",
            "allow_extrapolation": True,
            "scope": {
                "calibration": {
                    "hole_radius": [0.5, 0.9],
                    "hole_center_x": [3.5, 4.4],
                    "load_y": [-1300.0, -700.0],
                },
                "hard": {
                    "hole_radius": [0.3, 1.2],
                    "hole_center_x": [3.0, 5.0],
                    "load_y": [-1600.0, -500.0],
                },
            },
            "mesh": {
                "reference_size": 0.4,
                "level_sizes": [1.2, 0.9, 0.6, 0.4],
                "hotspot_count": 2,
                "element_budget": 1000,
            },
            "pso": {
                "particles": 4,
                "iterations": 3,
                "max_level": 3,
                "max_unique_evaluations": 10,
            },
            "transfer_min_evaluations": 4,
            "transfer_guard_ratio": 1.05,
        }
        path = Path(directory) / "request.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return BatchRequest.from_json(path)

    def test_expand_and_scope_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self._request(directory)
            cases = expand_cases(request)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].scope_status, "interpolation")
        self.assertFalse(cases[0].review_required)
        self.assertEqual(cases[1].scope_status, "extrapolation")
        self.assertTrue(cases[1].review_required)
        self.assertEqual(request.pso.max_unique_evaluations, 10)
        self.assertEqual(request.batch_order, "nearest_path")

    def test_hotspot_screening_is_simple_max_stress_ranking(self):
        result = types.SimpleNamespace(
            cell_features={
                1: [0.0, 0.0, 2.0] + [0.0] * 17,
                2: [0.0, 0.0, 5.0] + [0.0] * 17,
                3: [0.0, 0.0, 3.0] + [0.0] * 17,
            },
            cell_to_elements={1: [1], 2: [2], 3: [3]},
        )
        selected = hotspot_candidates(result, hotspot_count=2, mandatory_cells=())
        self.assertEqual(selected, (2, 3))

    def test_nearest_path_starts_in_middle_and_walks_to_neighbours(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self._request(directory)
            base_cases = expand_cases(request)
        left = replace(base_cases[0], case_id="left", index=1, descriptor=(0.0, 0.0))
        middle = replace(base_cases[0], case_id="middle", index=2, descriptor=(0.2, 0.0))
        right = replace(base_cases[0], case_id="right", index=3, descriptor=(1.0, 0.0))
        ordered = order_cases([right, left, middle], "nearest_path")
        self.assertEqual([case.case_id for case in ordered], ["middle", "left", "right"])

    def test_warm_guard_rejects_infeasible_probe_against_feasible_coarse(self):
        coarse = ObjectiveValue(
            position=(0,),
            objective=0.1,
            relative_error=0.1,
            element_count=100,
            feasible=True,
        )
        warm = ObjectiveValue(
            position=(2,),
            objective=0.01,
            relative_error=0.01,
            element_count=120,
            feasible=False,
            constraint_violation=0.2,
        )
        self.assertFalse(warm_start_is_acceptable(warm, coarse, 1.05))

    def test_warm_guard_accepts_small_objective_tolerance(self):
        coarse = ObjectiveValue(
            position=(0,),
            objective=0.100,
            relative_error=0.1,
            element_count=100,
            feasible=True,
        )
        warm = ObjectiveValue(
            position=(2,),
            objective=0.104,
            relative_error=0.09,
            element_count=100,
            feasible=True,
        )
        self.assertTrue(warm_start_is_acceptable(warm, coarse, 1.05))

    def test_fake_batch_runs_with_order_guard_and_direct_budgets(self):
        class FakeEnv:
            BASE_CELL_FEATURE_NAMES = StateAwareCalculixEnv.BASE_CELL_FEATURE_NAMES

            def __init__(self, plate, **kwargs):
                self.plate = plate
                self.virtual_cells = {1: object(), 2: object(), 3: object()}

            def _run_analysis(self, workdir, mesh_sizes):
                mean_size = sum(mesh_sizes.values()) / len(mesh_sizes)
                qoi = 1.0 + 0.2 * mean_size + abs(self.plate.load_y) / 1.0e6
                element_count = int(80 + sum(1.0 / value for value in mesh_sizes.values()) * 10)
                features = {
                    1: [0.0, 0.0, 5.0] + [0.0] * 17,
                    2: [0.0, 0.0, 3.0] + [0.0] * 17,
                    3: [0.0, 0.0, 1.0] + [0.0] * 17,
                }
                signature = (
                    element_count,
                    tuple(sorted((int(key), int(round(value * 1000))) for key, value in mesh_sizes.items())),
                )
                return types.SimpleNamespace(
                    qoi=qoi,
                    element_count=element_count,
                    cell_features=features,
                    cell_to_elements={1: [1], 2: [2], 3: [3]},
                    mesh_signature=signature,
                    workdir=str(workdir),
                )

        with tempfile.TemporaryDirectory() as directory:
            request = self._request(directory)
            original = batch_module.StateAwareCalculixEnv
            batch_module.StateAwareCalculixEnv = FakeEnv
            try:
                payload = batch_module.run_batch(
                    request,
                    Path(directory) / "out",
                    gmsh_cmd="fake-gmsh",
                    ccx_cmd="fake-ccx",
                )
            finally:
                batch_module.StateAwareCalculixEnv = original

        self.assertEqual(payload["summary"]["completed_cases"], 2)
        self.assertEqual([case["execution_index"] for case in payload["cases"]], [1, 2])
        self.assertTrue(all(case["cold"]["best_feasible"] for case in payload["cases"]))
        self.assertTrue(
            all(case["cold"]["unique_fe_evaluations"] <= 10 for case in payload["cases"])
        )
        self.assertIn(payload["cases"][1]["transfer_guard_status"], {
            "accepted",
            "accepted_same_as_coarse",
            "fallback_full_budget",
        })


if __name__ == "__main__":
    unittest.main()
