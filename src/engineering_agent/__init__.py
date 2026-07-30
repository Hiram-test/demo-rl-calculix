"""Core contracts for the code-generating engineering agent.

This package deliberately contains no case-specific geometry, material, load,
boundary-condition, mesh, or optimization values.
"""

from .decision_loop import (
    AgentDecision,
    DecisionAction,
    DecisionLoop,
    GeneratedFile,
    ModelResult,
    TraceRecorder,
)
from .problem_manifest import (
    FactRecord,
    MissingFact,
    ProblemManifest,
    ProvenanceKind,
)
from .skill_contract import (
    EngineeringSkill,
    EvidenceRequirement,
    SkillLibrary,
)

__all__ = [
    "AgentDecision",
    "DecisionAction",
    "DecisionLoop",
    "EngineeringSkill",
    "EvidenceRequirement",
    "FactRecord",
    "GeneratedFile",
    "MissingFact",
    "ModelResult",
    "ProblemManifest",
    "ProvenanceKind",
    "SkillLibrary",
    "TraceRecorder",
]
