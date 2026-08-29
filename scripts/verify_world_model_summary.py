#!/usr/bin/env python3  # Execute evidence verification with the active Python interpreter.
"""Verify structural and empirical world-model VLA evidence gates."""  # Describe the script purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
import argparse  # Import command-line parsing.
import json  # Import structured evidence parsing.
from pathlib import Path  # Import filesystem path handling.

def main() -> int:  # Parse one summary and enforce the requested evidence gates.
    parser = argparse.ArgumentParser()  # Create the command-line interface.
    parser.add_argument("summary", type=Path)  # Require the generated summary path.
    parser.add_argument("--require-structural", action="store_true")  # Require all method-contract invariants.
    parser.add_argument("--require-not-weaker", action="store_true")  # Require the empirical equal-solve Dörfler floor.
    parser.add_argument("--require-strict-advantage", action="store_true")  # Require at least one post-probe strict advantage.
    args = parser.parse_args()  # Parse all command-line arguments.
    data = json.loads(args.summary.read_text(encoding="utf-8"))  # Read the complete machine-generated evidence.
    failures: list[str] = []  # Allocate the complete verification report.
    contract = data.get("contract", {})  # Read the structural method contract.
    comparison = data.get("comparison")  # Read the optional empirical comparison.
    if args.require_structural:  # Enforce the deterministic architecture and action guarantees.
        if contract.get("common_probe") != "uniform_h0":  # Require the same initial mesh as Dörfler.
            failures.append("common_probe_is_not_uniform_h0")  # Record a fairness failure.
        if contract.get("world_model_reads_local_prediction") is not False:  # Require clean method independence.
            failures.append("local_prediction_dependency_detected")  # Record a purity failure.
        if contract.get("dorfler_candidate_always_present") is not True:  # Require the permanent safety candidate.
            failures.append("dorfler_candidate_missing")  # Record a policy-floor failure.
        if contract.get("nodewise_target_never_coarser_than_dorfler") is not True:  # Require exact materialized target dominance.
            failures.append("nodewise_dorfler_dominance_failed")  # Record a parameter-layer failure.
    if args.require_not_weaker:  # Enforce the real equal-solve numerical comparison.
        if not comparison:  # Require an independently executed Dörfler trajectory.
            failures.append("empirical_comparison_missing")  # Record missing evidence.
        elif comparison.get("not_weaker_at_equal_solves") is not True:  # Require every shared solve index to meet the tolerance.
            failures.append("empirical_dorfler_floor_failed")  # Record a real numerical regression.
    if args.require_strict_advantage:  # Enforce a genuine post-probe improvement when requested.
        if not comparison:  # Require an independently executed Dörfler trajectory.
            failures.append("empirical_comparison_missing_for_advantage")  # Record missing evidence.
        elif comparison.get("strict_world_model_advantage") is not True:  # Require at least one strict post-probe gain.
            failures.append("strict_world_model_advantage_missing")  # Record absence of measured advantage.
    report = {"summary": str(args.summary), "accepted": not failures, "failures": failures, "structural_checked": args.require_structural, "not_weaker_checked": args.require_not_weaker, "strict_advantage_checked": args.require_strict_advantage, "comparison": comparison}  # Assemble a transparent verification result.
    print(json.dumps(report, indent=1))  # Emit the complete evidence verdict.
    return 0 if not failures else 2  # Return a distinct nonzero code on any failed gate.

if __name__ == "__main__":  # Execute only when launched as a script.
    raise SystemExit(main())  # Propagate the verification verdict to CI or the shell.
