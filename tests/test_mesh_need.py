from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mesh_need.core import (
    analyze_mesh_series,
    apply_guards,
    build_evidence_ledger,
    classify_question,
    compare_replay_manifests,
    evaluate_energy,
    evaluate_force_moment,
    inspect_calculix_inp,
    run_pipeline,
)


class RoutingTests(unittest.TestCase):
    def test_01_singular_route(self):
        self.assertEqual(classify_question("裂纹尖端应力无限增大")["selected_skill"], "singularity_guard")

    def test_02_topology_route(self):
        self.assertEqual(classify_question("接口共同节点是否连接")["selected_skill"], "topology_alignment")

    def test_03_geometry_route(self):
        self.assertEqual(classify_question("短边导致网格生成失败")["selected_skill"], "geometry_mesh_repair")

    def test_04_model_route(self):
        self.assertEqual(classify_question("梁壳实体怎样切换")["selected_skill"], "model_fidelity_switch")

    def test_05_replay_route(self):
        self.assertEqual(classify_question("重网格后载荷丢失")["selected_skill"], "mesh_replay_guard")

    def test_06_qoi_route(self):
        self.assertEqual(classify_question("固定 QoI 是否收敛")["selected_skill"], "qoi_guided_refinement")

    def test_07_bounded_route(self):
        self.assertEqual(classify_question("有限圆角附近局部高应力")["selected_skill"], "bounded_hotspot_refinement")

    def test_08_unknown_route(self):
        diagnosis = classify_question("帮我看看这个模型")
        self.assertEqual(diagnosis["hotspot_class"], "none_or_unknown")
        self.assertIsNone(diagnosis["final_verdict"])


class TrendAndBalanceTests(unittest.TestCase):
    def test_09_singular_trend(self):
        trend = analyze_mesh_series([
            {"h": 4, "peak": 100, "qoi": 80}, {"h": 2, "peak": 145, "qoi": 81}, {"h": 1, "peak": 220, "qoi": 81.5}
        ], 0.02)
        self.assertEqual(trend["status"], "singular_peak_with_stable_qoi")

    def test_10_bounded_trend(self):
        trend = analyze_mesh_series([
            {"h": 4, "peak": 90, "qoi": 80}, {"h": 2, "peak": 91, "qoi": 81}, {"h": 1, "peak": 92, "qoi": 81.5}
        ], 0.02)
        self.assertEqual(trend["status"], "qoi_and_peak_converged")

    def test_11_unconverged_qoi(self):
        trend = analyze_mesh_series([{"h": 2, "peak": 10, "qoi": 10}, {"h": 1, "peak": 11, "qoi": 13}], 0.02)
        self.assertEqual(trend["status"], "qoi_not_converged")

    def test_12_insufficient_series(self):
        self.assertEqual(analyze_mesh_series([{"h": 1, "peak": 1, "qoi": 1}])["status"], "insufficient")

    def test_13_force_vectors_pass(self):
        result = evaluate_force_moment({"external_force": [100, 0, 0], "reaction_force": [-99, 0, 0], "equilibrium_tolerance": 0.02})
        self.assertEqual(result["status"], "within_tolerance")

    def test_14_force_vectors_fail(self):
        result = evaluate_force_moment({"external_force": [100, 0, 0], "reaction_force": [-80, 0, 0], "equilibrium_tolerance": 0.02})
        self.assertEqual(result["status"], "outside_tolerance")

    def test_15_energy_pass(self):
        result = evaluate_energy({"energy_history": [{"external_work": 100, "internal_energy": 99, "artificial_energy": 0.2}]})
        self.assertEqual(result["status"], "within_tolerance")

    def test_16_energy_fail(self):
        result = evaluate_energy({"energy_history": [{"external_work": 100, "internal_energy": 70, "artificial_energy": 20}]})
        self.assertEqual(result["status"], "outside_tolerance")


class ModelGuardTests(unittest.TestCase):
    def test_17_replay_drift(self):
        result = compare_replay_manifests({"loads": [1]}, {"loads": []})
        self.assertEqual(result["status"], "drift_observed")

    def test_18_replay_stable(self):
        manifest = {"sets": ["A"], "loads": ["L"]}
        self.assertEqual(compare_replay_manifests(manifest, manifest)["status"], "stable")

    def test_19_inp_disconnected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "case.inp"
            path.write_text("*NODE\n1,0,0\n2,1,0\n3,10,0\n4,11,0\n*ELEMENT,TYPE=T2D2\n1,1,2\n2,3,4\n", encoding="utf-8")
            result = inspect_calculix_inp(path)
            self.assertEqual(result["component_count"], 2)
            self.assertEqual(result["status"], "risks_observed")

    def test_20_inp_missing(self):
        self.assertEqual(inspect_calculix_inp(None)["status"], "not_observed")

    def test_21_singular_guard_overrides_ai(self):
        base = classify_question("有限圆角热点")
        trend = {"status": "singular_peak_with_stable_qoi"}
        result = apply_guards(base, {"qoi": {"name": "x", "location": "y", "extraction_method": "z", "tolerance": 0.02}}, {"status": "not_observed"}, trend)
        self.assertEqual(result["selected_skill"], "singularity_guard")

    def test_22_incomplete_qoi_blocks_refinement(self):
        base = classify_question("有限圆角热点")
        result = apply_guards(base, {}, {"status": "not_observed"}, {"status": "not_observed"})
        self.assertIn("bounded_hotspot_refinement", result["blocked_skills"])


class PipelineTests(unittest.TestCase):
    def complete_case(self):
        return {
            "case_id": "test-case",
            "question": "有限圆角附近局部高应力",
            "intended_use": "design comparison",
            "current_claim": "mesh is adequate",
            "qoi": {"name": "stress", "location": "fixed path", "extraction_method": "average", "tolerance": 0.02},
            "mesh_series": [{"h": 2, "peak": 90, "qoi": 80}, {"h": 1, "peak": 91, "qoi": 81}],
            "hotspots": [{"shape": "box", "xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}],
        }

    def test_23_pipeline_writes_contracts(self):
        with tempfile.TemporaryDirectory() as temp:
            run_pipeline(self.complete_case(), temp)
            for name in ("case.json", "diagnosis.json", "ai_prompt.json", "skill_result.json", "evidence_ledger.json", "pipeline-summary.json"):
                self.assertTrue((Path(temp) / name).is_file())

    def test_24_unsafe_ai_is_overridden(self):
        case = self.complete_case()
        case["question"] = "裂纹尖端"
        case["mesh_series"] = [{"h": 4, "peak": 100, "qoi": 80}, {"h": 2, "peak": 150, "qoi": 81}, {"h": 1, "peak": 230, "qoi": 81.3}]
        proposal = {"need_family": "resolution_convergence_budget", "hotspot_class": "bounded_response_hotspot", "selected_skill": "bounded_hotspot_refinement", "action": "refine", "confidence": 0.99}
        with tempfile.TemporaryDirectory() as temp:
            run_pipeline(case, temp, proposal)
            diagnosis = json.loads((Path(temp) / "diagnosis.json").read_text(encoding="utf-8"))
            self.assertEqual(diagnosis["selected_skill"], "singularity_guard")

    def test_25_bounded_field_generated(self):
        with tempfile.TemporaryDirectory() as temp:
            run_pipeline(self.complete_case(), temp)
            self.assertIn("Field[1] = Box", (Path(temp) / "bounded_hotspot_field.geo").read_text(encoding="utf-8"))

    def test_26_qoi_postview_generated(self):
        case = self.complete_case()
        case["question"] = "QoI 收敛和目标导向细化"
        case["qoi_indicator_points"] = [{"x": 0, "y": 0, "z": 0, "indicator": 4, "target_size": 0.5}]
        with tempfile.TemporaryDirectory() as temp:
            run_pipeline(case, temp)
            self.assertTrue((Path(temp) / "qoi_target_size.pos").is_file())
            self.assertTrue((Path(temp) / "qoi_postview_field.geo").is_file())

    def test_27_pso_export_is_guarded(self):
        case = self.complete_case()
        case["hotspots"].append({"shape": "box", "xmin": 2, "xmax": 3, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1})
        case["budget_conflict"] = True
        case["region_interaction"] = "measured_strong"
        with tempfile.TemporaryDirectory() as temp:
            run_pipeline(case, temp)
            job = json.loads((Path(temp) / "external_hotspot_pso_job.json").read_text(encoding="utf-8"))
            self.assertEqual(job["optimizer"], "external_multi_hotspot_pso")

    def test_28_ledger_has_no_final_judge(self):
        case = self.complete_case()
        diagnosis = classify_question(case["question"])
        ledger = build_evidence_ledger(case, diagnosis, {"status": "not_observed"}, {"status": "not_observed"}, {"status": "not_observed"}, {"status": "not_observed"})
        self.assertIsNone(ledger["final_verdict"])
        self.assertEqual(len(ledger["items"]), 6)
        self.assertEqual(sum(ledger["state_counts"].values()), 6)


if __name__ == "__main__":
    unittest.main()
