import json
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import types
import unittest


# Keep the contract/scope tests dependency-light.  The real repository provides
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

from meshpilot_batch_agent import BatchRequest, expand_cases, hotspot_candidates


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
            "allow_extrapolation": True,
            "scope": {
                "calibration": {
                    "hole_radius": [0.5, 0.9],
                    "hole_center_x": [3.5, 4.4],
                },
                "hard": {
                    "hole_radius": [0.3, 1.2],
                    "hole_center_x": [3.0, 5.0],
                },
            },
            "mesh": {
                "reference_size": 0.4,
                "level_sizes": [1.2, 0.9, 0.6, 0.4],
                "hotspot_count": 2,
                "element_budget": 1000,
            },
            "pso": {"particles": 4, "iterations": 3, "max_level": 3},
        }
        path = Path(directory) / "request.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return BatchRequest.from_json(path)

    def test_expand_and_scope_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = expand_cases(self._request(directory))
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].scope_status, "interpolation")
        self.assertFalse(cases[0].review_required)
        self.assertEqual(cases[1].scope_status, "extrapolation")
        self.assertTrue(cases[1].review_required)

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


if __name__ == "__main__":
    unittest.main()
