#!/usr/bin/env python3  # Execute the paired bridge benchmark with the active Python interpreter.
"""Run the medium-complexity bridge-diaphragm world-model VLA benchmark."""  # Describe the command-line entry point.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
import argparse  # Import command-line parsing.
import json  # Import concise terminal-result serialization.
import os  # Import deterministic native-thread environment settings.
from pathlib import Path  # Import portable output paths.
import sys  # Import repository path injection for direct script execution.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository root from this script.
sys.path.insert(0, str(ROOT))  # Make the local visionamr package importable without installation.
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Bound native solver threads for reproducible smoke execution.
from visionamr.vla.world.benchmark import run_bridge_benchmark  # Import the paired real-solver benchmark.

def _parser() -> argparse.ArgumentParser:  # Build the explicit benchmark command-line contract.
    parser = argparse.ArgumentParser(description="Run exact Dörfler and world-model VLA on a three-dimensional box-girder diaphragm.")  # Create the command-line parser.
    parser.add_argument("--output", type=Path, default=Path("runs/world_model_bridge"), help="Benchmark output directory.")  # Select the audit directory.
    parser.add_argument("--smoke", action="store_true", help="Use the smaller CI geometry.")  # Enable the reduced but topologically identical case.
    parser.add_argument("--max-solves", type=int, default=7, help="Maximum real CalculiX solves per method.")  # Set the genuine multi-step horizon.
    parser.add_argument("--n-equation-cap", type=int, default=120000, help="Active displacement-equation cap.")  # Set the common resource budget.
    parser.add_argument("--theta", type=float, default=0.5, help="Common Dörfler bulk parameter.")  # Set the shared marking parameter.
    parser.add_argument("--refine-factor", type=float, default=0.5, help="Common local refinement factor.")  # Set the shared refinement factor.
    parser.add_argument("--noninferiority-tolerance", type=float, default=0.03, help="Allowed numerical tolerance around the equal-solve Dörfler floor.")  # Set the predeclared comparison tolerance.
    parser.add_argument("--no-reference", action="store_true", help="Skip the expensive reference solve and compare the shared ZZ estimator norm.")  # Enable an estimator-only native smoke run.
    parser.add_argument("--strict-safety", action="store_true", help="Return non-zero when exact inclusion or equal-solve non-inferiority fails.")  # Enable the implementation safety gate.
    parser.add_argument("--require-budget-noninferiority", action="store_true", help="Return non-zero unless the common-budget Pareto gate passes.")  # Enable the stronger research resource gate separately.
    parser.add_argument("--require-advantage", action="store_true", help="Return non-zero unless a measurable terminal advantage is observed.")  # Enable the research-campaign superiority gate.
    return parser  # Return the complete parser.

def main() -> int:  # Execute the paired benchmark and enforce requested gates.
    args = _parser().parse_args()  # Parse user-supplied benchmark settings.
    result = run_bridge_benchmark(args.output, smoke=args.smoke, max_solves=args.max_solves, n_equation_cap=args.n_equation_cap, theta=args.theta, refine_factor=args.refine_factor, noninferiority_tolerance=args.noninferiority_tolerance, require_reference=not args.no_reference)  # Run both methods under one explicit reference contract.
    summary = {"output": str(args.output), "dorfler_solves": len(result.dorfler.records), "world_vla_solves": len(result.world_vla.records), "world_actions": result.world_action_count, "dorfler_inclusion_all": result.dorfler_inclusion_all, "solvewise_noninferior": result.solvewise_noninferior, "budgetwise_noninferior": result.budgetwise_noninferior, "terminal_advantage": result.terminal_advantage, "solvewise_ratios": list(result.solvewise_ratios), "budgetwise_ratios": list(result.budgetwise_ratios), "world_vla_timing_s": result.world_vla.timing_s}  # Build a concise result and separated-timing summary.
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # Display measured results rather than narrative claims.
    safety_pass = result.dorfler_inclusion_all and result.solvewise_noninferior  # Combine exact target inclusion with the equal-real-solve empirical floor.
    if args.strict_safety and not safety_pass:  # Enforce the implementation safety floor only when explicitly requested.
        return 2  # Signal a Dörfler-floor regression.
    if args.require_budget_noninferiority and not result.budgetwise_noninferior:  # Enforce the common-budget gate only in a dedicated campaign.
        return 3  # Signal a Pareto-envelope regression.
    if args.require_advantage and not result.terminal_advantage:  # Enforce measurable superiority only in a dedicated research run.
        return 4  # Signal absence of terminal advantage without conflating it with implementation failure.
    return 0  # Signal successful benchmark execution.

if __name__ == "__main__":  # Execute only when invoked as a script.
    raise SystemExit(main())  # Return the benchmark gate status to the shell.
