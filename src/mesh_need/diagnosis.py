"""Deterministic first-pass mesh-need diagnosis.

This module does not certify a finite-element model. It turns a user question
and mesh-history evidence into a transparent initial routing decision that
later numerical checks can challenge.
"""

from __future__ import annotations

from typing import Any


def _relative_change(previous: float, current: float) -> float:
    scale = max(abs(previous), 1.0e-12)
    return abs(current - previous) / scale


def _is_strictly_increasing(values: list[float]) -> bool:
    return len(values) >= 2 and all(b > a for a, b in zip(values, values[1:]))


def diagnose_question(case: dict[str, Any]) -> dict[str, Any]:
    """Return a transparent, non-final diagnosis for one engineering case.

    Expected optional fields:
      - question: natural-language problem statement
      - intended_use: decision supported by the model
      - qoi: quantity of interest description
      - mesh_history: rows with mesh_size, peak_stress and either
        reference_qoi or the legacy path_stress field
      - acceptance.reference_relative_change_max or the legacy
        path_relative_change_max field
    """

    question = str(case.get("question", "")).lower()
    intended_use = str(case.get("intended_use", "")).lower()
    qoi = str(case.get("qoi", "")).lower()
    combined = " ".join((question, intended_use, qoi))

    history = list(case.get("mesh_history", []))
    peak_values = [float(row["peak_stress"]) for row in history if "peak_stress" in row]
    reference_values = [
        float(row.get("reference_qoi", row.get("path_stress")))
        for row in history
        if "reference_qoi" in row or "path_stress" in row
    ]

    peak_increases = _is_strictly_increasing(peak_values)
    reference_last_change = None
    if len(reference_values) >= 2:
        reference_last_change = _relative_change(reference_values[-2], reference_values[-1])

    acceptance = case.get("acceptance", {})
    tolerance = float(
        acceptance.get(
            "reference_relative_change_max",
            acceptance.get("path_relative_change_max", 0.03),
        )
    )
    reference_is_stable = (
        reference_last_change is not None and reference_last_change <= tolerance
    )

    fatigue_context = any(token in combined for token in ("fatigue", "疲劳", "焊", "weld"))
    singularity_context = any(
        token in combined
        for token in (
            "crack",
            "裂纹",
            "裂尖",
            "point load",
            "concentrated load",
            "集中力",
            "点载荷",
            "point support",
            "点支承",
            "sharp corner",
            "尖角",
        )
    )
    topology_context = any(
        token in combined
        for token in ("断开", "不连接", "悬空", "共同节点", "接触", "interface", "contact")
    )

    evidence: list[dict[str, Any]] = []
    if peak_values:
        evidence.append(
            {
                "observation": "peak_stress_history",
                "values": peak_values,
                "interpretation": (
                    "peak increases as the mesh is refined"
                    if peak_increases
                    else "peak does not show monotonic growth"
                ),
            }
        )
    if reference_values:
        evidence.append(
            {
                "observation": "fixed_reference_qoi_history",
                "values": reference_values,
                "last_relative_change": reference_last_change,
                "interpretation": (
                    "fixed reference quantity is stable under the stated tolerance"
                    if reference_is_stable
                    else "fixed reference quantity is not yet stable"
                ),
            }
        )

    if topology_context:
        return {
            "status": "non_final_hypothesis",
            "need_class": "topology_interface_and_load_transfer",
            "hotspot_class": "topology_or_geometry_event",
            "recommended_skill": "topology_alignment",
            "blocked_skills": ["hotspot_pso"],
            "reason": "Connectivity or interface integrity must be examined before size optimization.",
            "evidence": evidence,
            "open_questions": ["Does the intended load path cross the detected interface?"],
        }

    if (singularity_context or fatigue_context) and peak_increases and reference_is_stable:
        return {
            "status": "non_final_hypothesis",
            "need_class": "result_validity_and_extraction",
            "hotspot_class": "singular_or_non_actionable_peak",
            "recommended_skill": "qoi_and_singularity_guard",
            "blocked_skills": ["peak_stress_hotspot_refinement", "hotspot_pso"],
            "reason": (
                "The solver-derived raw peak grows under refinement while a fixed, "
                "physically separate reference quantity stabilizes. The peak is not "
                "currently a valid optimization target."
            ),
            "evidence": evidence,
            "open_questions": [
                "Is the selected reference quantity the quantity required by the engineering decision?",
                "Is its physical location and extraction method identical across mesh levels?",
            ],
        }

    if peak_values and not peak_increases:
        return {
            "status": "non_final_hypothesis",
            "need_class": "resolution_convergence_and_budget",
            "hotspot_class": "bounded_response_hotspot",
            "recommended_skill": "bounded_hotspot_refinement",
            "blocked_skills": [],
            "reason": "The available peak history does not show persistent divergence.",
            "evidence": evidence,
            "open_questions": ["What engineering quantity and tolerance define completion?"],
        }

    return {
        "status": "insufficient_evidence",
        "need_class": "problem_formulation_required",
        "hotspot_class": "unknown",
        "recommended_skill": "clarify_qoi_and_collect_mesh_history",
        "blocked_skills": ["hotspot_pso"],
        "reason": "The question does not yet contain enough evidence to define a valid refinement target.",
        "evidence": evidence,
        "open_questions": [
            "What decision will this analysis support?",
            "Which quantity, location, extraction method and tolerance are fixed?",
        ],
    }
