#!/usr/bin/env python3  # Execute the deterministic sealed-aggregate renderer with the active Python interpreter.
"""Generate audited WMVLA-4WAY-P1 PNG/SVG figures without opening raw blind evidence."""  # Describe the CLI's strict post-analysis boundary.
from __future__ import annotations  # Postpone annotation evaluation for supported runtimes.

import argparse  # Require the caller to select one explicit campaign root.
import json  # Emit strict machine-readable completion or failure records.
from pathlib import Path  # Resolve this checkout and caller-selected campaign paths portably.
import sys  # Import the checked-out package and return explicit process status.

ROOT = Path(__file__).resolve().parents[1]  # Locate the repository independently from the invocation directory.
sys.path.insert(0, str(ROOT))  # Import this exact checkout without requiring a mutable installation.

from visionamr.vla.four_way_figures import FigureEvidenceError, generate_four_way_figures  # Import the sole sealed-aggregate render boundary.


def _parser() -> argparse.ArgumentParser:  # Build the minimal no-solver command contract.
    parser = argparse.ArgumentParser(description="Render deterministic WMVLA-4WAY-P1 figures from a complete hash-verified aggregate; never reads the raw test tree.")  # Create a self-describing parser.
    parser.add_argument("--root", type=Path, required=True, help="Explicit campaign root containing aggregate/artifact_index.json and no pre-existing figures directory.")  # Avoid an implicit formal-campaign read or accidental overwrite.
    return parser  # Return the complete parser.


def main() -> int:  # Validate sealed inputs, publish one immutable figure set, and report the result.
    args = _parser().parse_args()  # Parse the sole explicit campaign location.
    try:  # Convert all expected evidence and filesystem failures into a concise nonzero result.
        result = generate_four_way_figures(args.root)  # Rehash the aggregate and render all three fixed PNG/SVG pairs atomically.
    except (FigureEvidenceError, FileNotFoundError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exception:  # Catch only configuration, schema, hash, serialization, and render-input failures.
        payload = {"schema": "wmvla-four-way-figures-v1", "protocol_id": "WMVLA-4WAY-P1", "completed": False, "error_type": type(exception).__name__, "error": str(exception)}  # Preserve a bounded machine-readable failure without claiming figures.
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False), file=sys.stderr)  # Emit strict JSON for CI and review.
        return 2  # Signal invalid, incomplete, stale, or unrenderable aggregate evidence.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit full index identity and output cardinality.
    return 0  # Signal an atomically completed audited render.


if __name__ == "__main__":  # Execute only when launched as a command rather than imported by tests.
    raise SystemExit(main())  # Propagate the explicit process status to CI or the shell.
