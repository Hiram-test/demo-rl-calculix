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

        self.assertEqual(
            result["recommended_skill"],
            "fatigue_qoi_and_singularity_guard",
        )
        self.assertIn("hotspot_pso", result["blocked_skills"])
        self.assertEqual(result["hotspot_class"], "singular_or_non_actionable_peak")
        path_evidence = next(
            item for item in result["evidence"] if item["observation"] == "path_stress_history"
        )
        self.assertLessEqual(
            path_evidence["last_relative_change"],
            case["acceptance"]["path_relative_change_max"],
        )


if __name__ == "__main__":
    unittest.main()
