#!/usr/bin/env python3  # Execute the paired bridge benchmark with the active Python interpreter.
"""Run the medium-complexity bridge diaphragm world-model VLA benchmark."""  # Describe the command-line entry point.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
import argparse  # Import command-line parsing.
import json  # Import concise terminal-result serialization.
from pathlib import Path  # Import portable output paths.
from visionamr.vla.world.benchmark import run_bridge_benchmark  # Import the paired real-solver benchmark.

def _parser() -> argparse.ArgumentParser:  # Build the explicit benchmark command-line contract.
    parser = argparse.ArgumentParser(description="Run exact Dörfler and world-model VLA on a 3D box-girder diaphragm.")  # Create the command-line parser.
    parser.add_argument("--output", type=Path, default=Path("runs/world_model_bridge"), help="Benchmark output directory.")  # Select the audit directory.
    parser.add_argument("--smoke", action="store_true", help="Use the smaller CI geometry.")  # Enable the reduced but topologically identical case.
    parser.add_argument("--max-solves", type=int, default=7, help="Maximum real CalculiX solves per method.")  # Set the genuine multi-step horizon.
    parser.add_argument("--n-equation-cap", type=int, default=120000, help="Active displacement-equation cap.")  # Set the common resource budget.
    parser.add_argument("--theta", type=float, default=0.5, help="Common Dörfler bulk parameter.")  # Set the shared marking parameter.
    parser.add_argument("--refine-factor", type=float, default=0.5, help="Common local refinement factor.")  # Set the shared refinement factor.
    parser.add_argument("--noninferiority-tolerance", type=float, default=0.03, help="Allowed numerical tolerance around the Dörfler floor.")  # Set the predeclared comparison tolerance.
    parser.add_argument("--strict-safety", action="store_true", help="Return non-zero when either Dörfler floor gate fails.")  # Enable CI safety enforcement.
    parser.add_argument("--require-advantage", action="store_true", help="Return non-zero unless a measurable terminal advantage is observed.")  # Enable research-campaign superiority enforcement.
    return parser  # Return the complete parser.

def main() -> int:  # Execute the paired benchmark and enforce requested gates.
    args = _parser().parse_args()  # Parse user-supplied benchmark settings.
    result = run_bridge_benchmark(args.output, smoke=args.smoke, max_solves=args.max_solves, n_equation_cap=args.n_equation_cap, theta=args.theta, refine_factor=args.refine_factor, noninferiority_tolerance=args.noninferiority_tolerance)  # Run both methods under one protocol.
    summary = {"output": str(args.output), "dorfler_solves": len(result.dorfler.records), "world_vla_solves": len(result.world_vla.records), "world_actions": result.world_action_count, "dorfler_inclusion_all": result.dorfler_inclusion_all, "solvewise_noninferior": result.solvewise_noninferior, "budgetwise_noninferior": result.budgetwise_noninferior, "terminal_advantage": result.terminal_advantage, "solvewise_ratios": list(result.solvewise_ratios), "budgetwise_ratios": list(result.budgetwise_ratios)}  # Build a concise terminal summary.
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # Display measured results rather than narrative claims.
    safety_pass = result.dorfler_inclusion_all and result.solvewise_noninferior and result.budgetwise_noninferior  # Combine exact and empirical Dörfler-floor gates.
    if args.strict_safety and not safety_pass:  # Enforce the safety floor only when explicitly requested.
        return 2  # Signal a Dörfler-floor regression.
    if args.require_advantage and not result.terminal_advantage:  # Enforce measurable superiority only in a dedicated research run.
        return 3  # Signal absence of terminal advantage without conflating it with implementation failure.
    return 0  # Signal successful benchmark execution.

if __name__ == "__main__":  # Execute only when invoked as a script.
    raise SystemExit(main())  # Return the benchmark gate status to the shell.
