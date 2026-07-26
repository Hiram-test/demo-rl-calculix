"""Model-aware FEA mesh-need diagnosis MVP."""

from .core import (
    analyze_mesh_series,
    build_evidence_ledger,
    classify_question,
    inspect_calculix_inp,
    run_pipeline,
)

__all__ = [
    "analyze_mesh_series",
    "build_evidence_ledger",
    "classify_question",
    "inspect_calculix_inp",
    "run_pipeline",
]
