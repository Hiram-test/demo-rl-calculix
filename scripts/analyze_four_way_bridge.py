#!/usr/bin/env python3  # Execute deterministic four-way aggregation with the active Python interpreter.
"""Analyze the complete WMVLA-4WAY-P1 raw evidence and publish fixed gates."""  # Describe the command's sole responsibility.
from __future__ import annotations  # Postpone annotation evaluation for supported runtimes.
import argparse  # Parse explicit campaign, manifest, and diagnostic-incomplete controls.
import json  # Emit strict machine-readable terminal results.
from pathlib import Path  # Resolve repository and campaign paths portably.
import sys  # Import the checked-out package and return explicit exit statuses.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository independently from the invocation directory.
sys.path.insert(0, str(ROOT))  # Import this exact checked-out implementation without another installed version.
from visionamr.vla.four_way_analysis import IncompleteEvidenceError, analyze_four_way  # Import the sole deterministic analysis boundary.

def _parser() -> argparse.ArgumentParser:  # Build the complete post-execution command-line contract.
    parser = argparse.ArgumentParser(description="Aggregate a manifest-bound WMVLA-4WAY-P1 campaign into fixed machine gates and report artifacts.")  # Create a self-describing parser.
    parser.add_argument("--root", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1", help="Campaign root containing test, ablations, and aggregate outputs.")  # Permit an isolated campaign location without changing statistics.
    parser.add_argument("--manifest", type=Path, default=None, help="Authenticated case_manifest.json; defaults to <root>/protocol/case_manifest.json.")  # Permit explicit immutable manifest selection.
    parser.add_argument("--allow-incomplete", action="store_true", help="Publish explicitly fail-closed diagnostic artifacts for an invalid or incomplete campaign; never claims a scientific win.")  # Provide recovery evidence without selective scoring.
    parser.add_argument("--allow-unqualified-references", action="store_true", help="Analyze operational non-qualified Reference B only when TEST_STARTED already disclosed the frozen waiver.")  # Require explicit analysis-time acknowledgement without calling the reference converged.
    return parser  # Return the complete parser.

def main() -> int:  # Parse, aggregate, publish artifacts, and return a machine-meaningful exit code.
    args = _parser().parse_args()  # Parse caller-supplied paths and diagnostic mode.
    campaign = args.root.resolve()  # Normalize the campaign root for unambiguous raw evidence.
    manifest = (args.manifest or campaign / "protocol" / "case_manifest.json").resolve()  # Resolve the exact authenticated manifest input.
    try:  # Convert invalid evidence into concise terminal diagnostics without hiding a traceback in artifacts.
        result = analyze_four_way(campaign, manifest, allow_incomplete=bool(args.allow_incomplete), allow_unqualified_references=bool(args.allow_unqualified_references))  # Execute aggregation under explicit completeness and reference-qualification controls.
    except (IncompleteEvidenceError, FileNotFoundError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exception:  # Catch schema, coverage, filesystem, and numerical-analysis input errors.
        payload = {"protocol_id": "WMVLA-4WAY-P1", "completed": False, "error_type": type(exception).__name__, "error": str(exception)}  # Build one bounded explicit failure response.
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)  # Report failure without publishing or claiming a gate.
        return 2  # Signal invalid or incomplete analysis evidence distinctly.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit concise paths, coverage, and fixed gate results.
    if not bool(result.get("complete")):  # Distinguish fail-closed diagnostic output from a complete scientific aggregate.
        return 3  # Signal incomplete evidence even though diagnostic artifacts were published successfully.
    return 0 if bool(result.get("final_gate", {}).get("OVERALL_WIN")) else 4  # Distinguish a complete pass from a complete non-winning scientific result.

if __name__ == "__main__":  # Execute only when launched as a command.
    raise SystemExit(main())  # Propagate the explicit status to CI or the shell.
