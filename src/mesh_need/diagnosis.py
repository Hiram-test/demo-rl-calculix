"""AI analysis contract for finite-element modeling evidence.

This module deliberately contains no physical classification rules. Numerical
code may collect and summarize solver evidence, but a language model must form
and compare physical hypotheses from the question, model description and raw
evidence. The deterministic layer only packages inputs and validates the shape
of the model response.
"""

from __future__ import annotations

from typing import Any


AI_REQUIRED_FIELDS = (
    "problem_restatement",
    "competing_hypotheses",
    "evidence_assessment",
    "recommended_next_action",
    "uncertainties",
)


def build_analysis_packet(
    case: dict[str, Any],
    evidence: dict[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    """Build the complete evidence packet that an AI model must interpret.

    No need class, hotspot class, physical mechanism or skill is inferred here.
    The packet preserves the user's intent and the solver-derived observations,
    and asks the model to make an explicit, evidence-linked argument.
    """

    return {
        "status": "awaiting_ai_analysis",
        "question": case.get("question", ""),
        "intended_use": case.get("intended_use", ""),
        "qoi": case.get("qoi", ""),
        "model_context": case.get("model_context", case.get("model", {})),
        "acceptance_context": case.get("acceptance", {}),
        "solver_evidence": evidence if evidence is not None else case.get("mesh_history", []),
        "analysis_instructions": [
            "Restate the engineering decision before discussing mesh size.",
            "Form at least two plausible physical or numerical hypotheses when evidence permits.",
            "Distinguish observations from interpretation and cite the supplied evidence fields.",
            "Explain what evidence supports, contradicts, or fails to distinguish each hypothesis.",
            "Propose the smallest next calculation or model change that best separates the hypotheses.",
            "Do not certify the model and do not claim a unique final answer when evidence is incomplete.",
        ],
        "required_ai_output": {
            "problem_restatement": "string",
            "competing_hypotheses": "non-empty list of hypotheses with evidence links",
            "evidence_assessment": "list separating support, contradiction and unknowns",
            "recommended_next_action": "one evidence-generating action, not an unsupported verdict",
            "uncertainties": "non-empty list",
            "optional_skill": "a tool or workflow selected by the AI, with justification",
        },
    }


def validate_ai_analysis(analysis: dict[str, Any]) -> list[str]:
    """Return structural validation errors for an AI response.

    Validation is intentionally limited to completeness and traceability. It
    does not decide whether the model's physics is correct.
    """

    errors: list[str] = []
    if not isinstance(analysis, dict):
        return ["AI analysis must be a JSON object"]

    for field in AI_REQUIRED_FIELDS:
        if field not in analysis:
            errors.append(f"missing required field: {field}")

    hypotheses = analysis.get("competing_hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        errors.append("competing_hypotheses must be a non-empty list")

    uncertainties = analysis.get("uncertainties")
    if not isinstance(uncertainties, list) or not uncertainties:
        errors.append("uncertainties must be a non-empty list")

    return errors
