import unittest

from engineering_agent.problem_manifest import (
    FactRecord,
    ProblemManifest,
    ProvenanceKind,
)


class ProblemManifestTests(unittest.TestCase):
    def test_missing_fact_becomes_user_question_without_default(self) -> None:
        manifest = ProblemManifest(task_id="case-1", user_goal="build current model")
        manifest.require_fact(
            "geometry.bearing_width_mm",
            reason="The load footprint cannot be established from current inputs.",
            question="What is the bearing/contact width, and where is it documented?",
            acceptable_sources=("drawing", "CAD", "user measurement"),
        )

        self.assertNotIn("geometry.bearing_width_mm", manifest.facts)
        self.assertEqual(
            manifest.questions_for_user()[0]["field"],
            "geometry.bearing_width_mm",
        )

    def test_current_user_value_removes_missing_entry(self) -> None:
        manifest = ProblemManifest(task_id="case-2")
        manifest.require_fact(
            "material.elastic_modulus_mpa",
            reason="Material stiffness is required.",
            question="Provide the elastic modulus and its source.",
        )
        manifest.add_fact(
            FactRecord(
                path="material.elastic_modulus_mpa",
                value=210000,
                provenance=ProvenanceKind.PROVIDED_BY_USER,
                source_ref="user message 7",
                user_confirmed=True,
            )
        )
        self.assertEqual(manifest.missing_facts, [])
        self.assertEqual(manifest.provenance_report()["issues"], [])

    def test_agent_derived_value_requires_derivation(self) -> None:
        manifest = ProblemManifest(task_id="case-3")
        manifest.add_fact(
            FactRecord(
                path="mesh.local_size_mm",
                value=8.0,
                provenance=ProvenanceKind.DERIVED_BY_AGENT,
            )
        )
        report = manifest.provenance_report()
        self.assertIn(
            "mesh.local_size_mm: derivation is required for agent-derived values",
            report["issues"],
        )

    def test_algorithm_settings_are_separate_from_physical_facts(self) -> None:
        manifest = ProblemManifest(task_id="case-4")
        manifest.add_algorithm_configuration(
            FactRecord(
                path="optimizer.pso.iterations",
                value=5,
                provenance=ProvenanceKind.ALGORITHM_CONFIGURATION,
                notes="Current run configuration, not a model fact or quality gate.",
            )
        )
        self.assertNotIn("optimizer.pso.iterations", manifest.facts)
        self.assertIn("optimizer.pso.iterations", manifest.algorithm_configuration)

    def test_legacy_fixture_is_visible_but_not_hidden(self) -> None:
        manifest = ProblemManifest(task_id="regression-only")
        manifest.add_fact(
            FactRecord(
                path="geometry.length_mm",
                value=1000.0,
                provenance=ProvenanceKind.LEGACY_FIXTURE,
                source_ref="legacy V4 synthetic benchmark",
            )
        )
        self.assertTrue(manifest.provenance_report()["has_legacy_fixture_values"])


if __name__ == "__main__":
    unittest.main()
