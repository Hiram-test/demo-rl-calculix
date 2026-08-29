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
    parser.add_argument("--require-not-weaker", action="store_true")  # Require equal-solve and equation-budget Dörfler floors.
    parser.add_argument("--require-strict-advantage", action="store_true")  # Require a Dörfler advantage beyond the declared tolerance.
    parser.add_argument("--require-local-advantage", action="store_true")  # Require an equation-budget advantage over local prediction.
    args = parser.parse_args()  # Parse all command-line arguments.
    data = json.loads(args.summary.read_text(encoding="utf-8"))  # Read the complete machine-generated evidence.
    failures: list[str] = []  # Allocate the complete verification report.
    contract = data.get("contract", {})  # Read the structural method contract.
    comparisons = data.get("comparisons", {})  # Read all independently executed comparators.
    dorfler_bundle = comparisons.get("dorfler", {})  # Read the complete Dörfler evidence bundle.
    equal_comparison = dorfler_bundle.get("equal_solves") or data.get("comparison")  # Preserve compatibility with earlier summaries.
    budget_comparison = dorfler_bundle.get("equation_budget")  # Read the best-in-equation-budget Dörfler comparison.
    local_bundle = comparisons.get("local_prediction", {})  # Read the independent local-prediction evidence bundle.
    local_budget = local_bundle.get("equation_budget")  # Read the best-in-equation-budget local comparison.
    if args.require_structural:  # Enforce the deterministic architecture and action guarantees.
        if contract.get("common_probe") != "uniform_h0":  # Require the same initial mesh as Dörfler.
            failures.append("common_probe_is_not_uniform_h0")  # Record a fairness failure.
        if contract.get("world_model_reads_local_prediction") is not False:  # Require clean method independence.
            failures.append("local_prediction_dependency_detected")  # Record a purity failure.
        if contract.get("dorfler_candidate_always_present") is not True:  # Require the permanent safety candidate.
            failures.append("dorfler_candidate_missing")  # Record a policy-floor failure.
        if contract.get("nodewise_target_never_coarser_than_dorfler") is not True:  # Require exact materialized target dominance.
            failures.append("nodewise_dorfler_dominance_failed")  # Record a parameter-layer failure.
    if args.require_not_weaker:  # Enforce both real Dörfler comparison axes.
        if not equal_comparison:  # Require an independently executed equal-solve trajectory.
            failures.append("equal_solve_dorfler_comparison_missing")  # Record missing evidence.
        elif equal_comparison.get("not_weaker_at_equal_solves") is not True:  # Require every shared solve index to meet the tolerance.
            failures.append("equal_solve_dorfler_floor_failed")  # Record a real numerical regression.
        if not budget_comparison:  # Require a best-in-equation-budget comparison.
            failures.append("equation_budget_dorfler_comparison_missing")  # Record missing resource evidence.
        elif budget_comparison.get("not_weaker") is not True:  # Require the best feasible world-model result to meet the Dörfler floor.
            failures.append("equation_budget_dorfler_floor_failed")  # Record a resource-aligned numerical regression.
    if args.require_strict_advantage:  # Enforce a genuine Dörfler improvement beyond tolerance.
        equal_strict = bool(equal_comparison and equal_comparison.get("strict_world_model_advantage"))  # Read equal-solve strict advantage.
        budget_strict = bool(budget_comparison and budget_comparison.get("strictly_better"))  # Read equation-budget strict advantage.
        if not equal_strict and not budget_strict:  # Accept either independently meaningful comparison axis.
            failures.append("strict_dorfler_advantage_missing")  # Record absence of a robust measured advantage.
    if args.require_local_advantage:  # Enforce the independent local-prediction target.
        if not local_budget:  # Require a completed local-prediction comparison.
            failures.append("local_prediction_comparison_missing")  # Record missing evidence.
        elif local_budget.get("strictly_better") is not True:  # Require an advantage beyond the declared tolerance under the equation cap.
            failures.append("local_prediction_budget_advantage_missing")  # Record failure to beat the strong local predictor.
    report = {"summary": str(args.summary), "accepted": not failures, "failures": failures, "structural_checked": args.require_structural, "dorfler_floor_checked": args.require_not_weaker, "dorfler_advantage_checked": args.require_strict_advantage, "local_advantage_checked": args.require_local_advantage, "dorfler_equal_solves": equal_comparison, "dorfler_equation_budget": budget_comparison, "local_equation_budget": local_budget}  # Assemble a transparent verification result.
    print(json.dumps(report, indent=1))  # Emit the complete evidence verdict.
    return 0 if not failures else 2  # Return a distinct nonzero code on any failed gate.

if __name__ == "__main__":  # Execute only when launched as a script.
    raise SystemExit(main())  # Propagate the verification verdict to CI or the shell.
