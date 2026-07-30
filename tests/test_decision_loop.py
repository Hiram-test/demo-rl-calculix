import json
import tempfile
import unittest
from pathlib import Path

from engineering_agent.decision_loop import (
    DecisionAction,
    DecisionLoop,
    ModelResult,
    TraceRecorder,
)
from engineering_agent.problem_manifest import ProblemManifest
from engineering_agent.skill_contract import EngineeringSkill, SkillLibrary


class FakeModel:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    def complete_json(self, messages):
        self.messages = messages
        return ModelResult(
            payload=self.payload,
            raw_response={"ok": True},
            provider="fake",
            model="fake-v1",
        )


class FailingModel:
    def complete_json(self, messages):
        raise RuntimeError("temporary provider failure")


class DecisionLoopTests(unittest.TestCase):
    def _library(self):
        library = SkillLibrary()
        library.register(
            EngineeringSkill.from_mapping(
                {
                    "skill_id": "problem-definition-source-audit",
                    "title": "Problem definition and source audit",
                    "purpose": (
                        "Establish model facts and their sources before analysis."
                    ),
                    "procedure": [
                        "List current evidence.",
                        "Separate missing facts from assumptions.",
                    ],
                }
            )
        )
        return library

    def test_missing_inputs_can_produce_ask_user_without_default_values(self):
        manifest = ProblemManifest(
            task_id="new-case",
            user_goal="prepare an engineering model",
        )
        manifest.require_fact(
            "loads.primary",
            reason="No load is defined",
            question="Provide the load and source",
        )
        model = FakeModel(
            {
                "action": "ask_user",
                "rationale": "The physical load is absent.",
                "selected_skill_ids": [
                    "problem-definition-source-audit"
                ],
                "questions_for_user": [
                    "What load should be applied and where is it documented?"
                ],
                "files": [],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            loop = DecisionLoop(
                model=model,
                manifest=manifest,
                skills=self._library(),
                trace=TraceRecorder(Path(tmp)),
            )
            decision = loop.decide_next()
            self.assertEqual(decision.action, DecisionAction.ASK_USER)
            prompt = model.messages[1]["content"]
            self.assertNotIn("required_artifacts", prompt)
            self.assertNotIn("finish_requirements", prompt)
            self.assertNotIn("candidate_bearing_width_mm", prompt)

    def test_generated_code_is_current_decision_output_not_skill_callable(self):
        model = FakeModel(
            {
                "action": "write_code",
                "rationale": "Generate a parser for the current input file.",
                "selected_skill_ids": [
                    "problem-definition-source-audit"
                ],
                "files": [
                    {
                        "path": "work/parse_input.py",
                        "content": "print('parse current input')",
                        "purpose": "current task parser",
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            loop = DecisionLoop(
                model=model,
                manifest=ProblemManifest(task_id="case"),
                skills=self._library(),
                trace=TraceRecorder(Path(tmp)),
            )
            decision = loop.decide_next()
            self.assertEqual(
                decision.files[0].path,
                "work/parse_input.py",
            )
            skill_payload = self._library().prompt_catalog()[0]
            self.assertNotIn("function", skill_payload)
            self.assertNotIn("builder", skill_payload)

    def test_provider_error_is_traced_and_not_raised_as_a_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = DecisionLoop(
                model=FailingModel(),
                manifest=ProblemManifest(task_id="case"),
                skills=self._library(),
                trace=TraceRecorder(root),
            )
            decision = loop.decide_next()
            self.assertEqual(
                decision.action,
                DecisionAction.RECORD_FINDING,
            )
            self.assertTrue(decision.issues)
            self.assertTrue((root / "agent_trace.json").exists())
            response = json.loads(
                (
                    root
                    / "model_io"
                    / "iteration_0001_response.json"
                ).read_text()
            )
            self.assertEqual(response["error"], "RuntimeError")

    def test_skill_rejects_case_execution_keys(self):
        with self.assertRaises(ValueError):
            EngineeringSkill.from_mapping(
                {
                    "skill_id": "bad",
                    "title": "bad",
                    "purpose": "bad",
                    "procedure": ["bad"],
                    "fixed_parameters": {"width_mm": 40},
                }
            )


if __name__ == "__main__":
    unittest.main()
