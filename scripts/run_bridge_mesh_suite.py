from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bridge_mesh_suite.calculix_crosscheck import run_calculix_crosschecks
from bridge_mesh_suite.deepseek_review import review_with_deepseek
from bridge_mesh_suite.report import build_docx_report, build_figures, build_markdown_report, build_pdf_report
from bridge_mesh_suite.scenarios import ScenarioRun, run_all_scenarios


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def save_raw_artifacts(runs: list[ScenarioRun], output_dir: Path) -> None:
    raw = output_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for run in runs:
        case_dir = raw / run.scenario_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "summary.json").write_text(
            json.dumps(run.summary_dict(), ensure_ascii=False, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        for key, mesh in run.meshes.items():
            sol = run.solutions[key]
            np.savez_compressed(
                case_dir / f"{key}.npz",
                nodes=mesh.nodes,
                elements=mesh.elements,
                displacements=sol.displacements,
                reactions=sol.reactions,
                element_centers=sol.element_centers,
                element_stress=sol.element_stress,
                element_von_mises=sol.element_von_mises,
                nodal_stress=sol.nodal_stress,
                nodal_von_mises=sol.nodal_von_mises,
                load_vector=sol.load_vector,
                strain_energy=np.array([sol.strain_energy]),
                external_half_work=np.array([sol.external_half_work]),
            )


def build_skill_trace(runs: list[ScenarioRun]) -> dict[str, Any]:
    events = []
    for run in runs:
        for index, skill in enumerate(run.diagnostic.skill_trace, start=1):
            events.append({
                "scenario_id": run.scenario_id,
                "sequence": index,
                "skill": skill.name,
                "purpose": skill.purpose,
                "inputs": skill.inputs,
                "outputs": skill.outputs,
                "passed": skill.passed,
            })
    return {
        "schema_version": "bridge-mesh-skill-trace/1.0",
        "scenario_count": len(runs),
        "skill_call_count": len(events),
        "energy_skill_call_count": sum(e["skill"] == "energy_consistency" for e in events),
        "events": events,
    }


def validate_full_run(
    runs: list[ScenarioRun],
    output_dir: Path,
    deepseek_review: dict[str, Any] | None,
    require_deepseek: bool,
    calculix_receipt: dict[str, Any] | None,
    require_calculix: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(runs) < 4:
        errors.append(f"expected at least 4 component scenarios, got {len(runs)}")
    for run in runs:
        if len(run.level_rows) < 3:
            errors.append(f"{run.scenario_id}: fewer than 3 mesh levels")
        energy = [s for s in run.diagnostic.skill_trace if s.name == "energy_consistency"]
        if len(energy) != 1:
            errors.append(f"{run.scenario_id}: energy Skill was not executed exactly once")
        elif not energy[0].passed:
            errors.append(f"{run.scenario_id}: energy Skill failed")
        if not run.diagnostic.applied_plan:
            errors.append(f"{run.scenario_id}: no implemented refinement plan")
        if not run.meshes or not run.solutions:
            errors.append(f"{run.scenario_id}: raw model/solution artifacts missing")
    if require_calculix:
        if calculix_receipt is None:
            errors.append("required CalculiX cross-check missing")
        elif not calculix_receipt.get("valid"):
            errors.append("CalculiX cross-check failed")
    if require_deepseek:
        if deepseek_review is None:
            errors.append("required DeepSeek review missing")
        elif len(deepseek_review.get("scenario_reviews", [])) != len(runs):
            errors.append("DeepSeek did not review every scenario")
    required_files = [
        "BRIDGE_COMPONENT_MESH_REPORT.md",
        "Bridge_Component_Mesh_Report.pdf",
        "Bridge_Component_Mesh_Report.docx",
        "suite_summary.json",
        "skill_trace.json",
    ]
    for name in required_files:
        if not (output_dir / name).exists():
            errors.append(f"missing report artifact: {name}")
    return {
        "valid": not errors,
        "errors": errors,
        "scenario_count": len(runs),
        "mesh_level_count": sum(len(r.level_rows) for r in runs),
        "solver_run_count": sum(len(r.solutions) for r in runs),
        "skill_call_count": sum(len(r.diagnostic.skill_trace) for r in runs),
        "energy_skill_call_count": sum(s.name == "energy_consistency" for r in runs for s in r.diagnostic.skill_trace),
        "deepseek_reviewed": deepseek_review is not None,
        "calculix_crosschecked": calculix_receipt is not None,
        "calculix_valid": bool(calculix_receipt and calculix_receipt.get("valid")),
        "report_files": required_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full bridge-component mesh evidence suite")
    parser.add_argument("--output-dir", default="artifacts/bridge-component-mesh-suite")
    parser.add_argument("--no-deepseek", action="store_true", help="local developer mode only")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--require-calculix", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    runs = run_all_scenarios()
    save_raw_artifacts(runs, output_dir)
    summary = {
        "schema_version": "bridge-component-mesh-suite/1.0",
        "purpose": "普通桥梁工程人员的网格递增问题诊断、物理融合方案落实与图文报告",
        "scenarios": [run.summary_dict() for run in runs],
    }
    (output_dir / "suite_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    trace = build_skill_trace(runs)
    (output_dir / "skill_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    calculix_receipt = None
    if args.require_calculix:
        calculix_receipt = run_calculix_crosschecks(runs, output_dir)

    deepseek_review = None
    if not args.no_deepseek:
        deepseek_review = review_with_deepseek(summary, model=args.model)
        (output_dir / "deepseek_user_review.json").write_text(json.dumps(deepseek_review, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    figures = build_figures(runs, figures_dir)
    build_markdown_report(runs, figures, output_dir / "BRIDGE_COMPONENT_MESH_REPORT.md", deepseek_review)
    build_pdf_report(runs, figures, output_dir / "Bridge_Component_Mesh_Report.pdf", deepseek_review)
    build_docx_report(runs, figures, output_dir / "Bridge_Component_Mesh_Report.docx", deepseek_review)

    receipt = validate_full_run(
        runs, output_dir, deepseek_review, require_deepseek=not args.no_deepseek,
        calculix_receipt=calculix_receipt, require_calculix=args.require_calculix,
    )
    (output_dir / "validation_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
