import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mesh_need import build_analysis_packet, validate_ai_analysis


class MeshNeedAIContractTest(unittest.TestCase):
    def test_packet_preserves_question_and_raw_evidence_without_routing(self) -> None:
        path = REPO_ROOT / "examples/mesh_need/fatigue_weld_hotspot_question.json"
        case = json.loads(path.read_text(encoding="utf-8"))

        packet = build_analysis_packet(case)

        self.assertEqual(packet["question"], case["question"])
        self.assertEqual(packet["solver_evidence"], case["mesh_history"])
        self.assertEqual(packet["status"], "awaiting_ai_analysis")
        self.assertNotIn("recommended_skill", packet)
        self.assertNotIn("hotspot_class", packet)
        self.assertNotIn("need_class", packet)

    def test_ai_response_validation_checks_structure_not_physics(self) -> None:
        valid = {
            "problem_restatement": "Determine whether the current local result is useful for the stated decision.",
            "competing_hypotheses": [
                {"hypothesis": "local discretization error", "evidence_refs": ["mesh_history"]},
                {"hypothesis": "load-introduction idealization", "evidence_refs": ["model_context"]},
            ],
            "evidence_assessment": [
                {"status": "unknown", "statement": "The present evidence does not isolate the mechanism."}
            ],
            "recommended_next_action": "Run one discriminating counterfactual model.",
            "uncertainties": ["physical load-transfer area is not specified"],
        }
        self.assertEqual(validate_ai_analysis(valid), [])

        invalid = {"problem_restatement": "too incomplete"}
        errors = validate_ai_analysis(invalid)
        self.assertTrue(any("competing_hypotheses" in item for item in errors))
        self.assertTrue(any("uncertainties" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
