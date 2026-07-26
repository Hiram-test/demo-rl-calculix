import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mesh_need import diagnose_question


class MeshNeedDiagnosisTest(unittest.TestCase):
    def test_fatigue_peak_is_not_routed_to_hotspot_pso(self) -> None:
        path = REPO_ROOT / "examples/mesh_need/fatigue_weld_hotspot_question.json"
        case = json.loads(path.read_text(encoding="utf-8"))

        result = diagnose_question(case)

        self.assertEqual(result["recommended_skill"], "qoi_and_singularity_guard")
        self.assertIn("hotspot_pso", result["blocked_skills"])
        self.assertEqual(result["hotspot_class"], "singular_or_non_actionable_peak")
        reference_evidence = next(
            item
            for item in result["evidence"]
            if item["observation"] == "fixed_reference_qoi_history"
        )
        self.assertLessEqual(
            reference_evidence["last_relative_change"],
            case["acceptance"]["path_relative_change_max"],
        )

    def test_solver_style_point_load_history_is_guarded(self) -> None:
        case = {
            "question": "集中力加载点最大应力持续升高，是否交给热点PSO？",
            "intended_use": "远离加载点的位移判断",
            "qoi": "固定位置竖向位移",
            "mesh_history": [
                {"mesh_size": 5.0, "peak_stress": 1200.0, "reference_qoi": -0.74},
                {"mesh_size": 2.5, "peak_stress": 1400.0, "reference_qoi": -0.767},
                {"mesh_size": 1.25, "peak_stress": 1600.0, "reference_qoi": -0.773},
            ],
            "acceptance": {"reference_relative_change_max": 0.03},
        }

        result = diagnose_question(case)

        self.assertEqual(result["recommended_skill"], "qoi_and_singularity_guard")
        self.assertIn("peak_stress_hotspot_refinement", result["blocked_skills"])
        self.assertIn("hotspot_pso", result["blocked_skills"])


if __name__ == "__main__":
    unittest.main()
