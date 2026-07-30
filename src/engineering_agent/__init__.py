"""Core contracts for the code-generating engineering agent.

This package deliberately contains no case-specific geometry, material, load,
boundary-condition, mesh, or optimization values.
"""

from .problem_manifest import (
    FactRecord,
    MissingFact,
    ProblemManifest,
    ProvenanceKind,
)

__all__ = [
    "FactRecord",
    "MissingFact",
    "ProblemManifest",
    "ProvenanceKind",
]
