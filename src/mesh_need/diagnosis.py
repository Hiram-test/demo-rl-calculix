"""Deterministic first-pass mesh-need diagnosis.

This module does not certify a finite-element model. It turns a user question
and a small amount of mesh-history evidence into a transparent initial routing
decision that later skills can challenge.
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
      - intended_use: e.g. fatigue assessment
      - qoi: quantity of interest description
      - mesh_history: rows with mesh_size, peak_stress and/or path_stress
      - acceptance.path_relative_change_max
    """

    question = str(case.get("question", "")).lower()
    intended_use = str(case.get("intended_use", "")).lower()
    qoi = str(case.get("qoi", "")).lower()
    combined = " ".join((question, intended_use, qoi))

    history = list(case.get("mesh_history", []))
    peak_values = [float(row["peak_stress"]) for row in history if "peak_stress" in row]
    path_values = [float(row["path_stress"]) for row in history if "path_stress" in row]

    peak_increases = _is_strictly_increasing(peak_values)
    path_last_change = None
    if len(path_values) >= 2:
        path_last_change = _relative_change(path_values[-2], path_values[-1])

    tolerance = float(
        case.get("acceptance", {}).get("path_relative_change_max", 0.03)
    )
    path_is_stable = path_last_change is not None and path_last_change <= tolerance

    fatigue_context = any(token in combined for token in ("fatigue", "疲劳", "焊", "weld"))
    crack_context = any(token in combined for token in ("crack", "裂纹", "裂尖"))
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
    if path_values:
        evidence.append(
            {
                "observation": "path_stress_history",
                "values": path_values,
                "last_relative_change": path_last_change,
                "interpretation": (
                    "fixed-path quantity is stable under the stated tolerance"
                    if path_is_stable
                    else "fixed-path quantity is not yet stable"
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

    if crack_context or (fatigue_context and peak_increases and path_is_stable):
        reason = (
            "A growing raw peak and a stable fixed-path fatigue quantity indicate that "
            "the peak should not be used as the optimization target."
        )
        return {
            "status": "non_final_hypothesis",
            "need_class": "result_validity_and_extraction",
            "hotspot_class": "singular_or_non_actionable_peak",
            "recommended_skill": "fatigue_qoi_and_singularity_guard",
            "blocked_skills": ["peak_stress_hotspot_refinement", "hotspot_pso"],
            "reason": reason,
            "evidence": evidence,
            "open_questions": [
                "Is the fatigue assessment based on a specified structural-stress or path method?",
                "Is the extraction location identical across all mesh levels?",
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
