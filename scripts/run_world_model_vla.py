#!/usr/bin/env python3  # Execute the benchmark with the active Python interpreter.
"""Run and compare the three-dimensional bridge world-model VLA trajectory."""  # Describe the script purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
import argparse  # Import command-line parsing.
from dataclasses import asdict, replace  # Import record serialization and immutable config updates.
import json  # Import structured result output.
import os  # Import deterministic thread environment settings.
from pathlib import Path  # Import filesystem path handling.
import sys  # Import repository path injection and exit codes.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository root.
sys.path.insert(0, str(ROOT))  # Make the local package importable without installation.
os.environ.setdefault("OMP_NUM_THREADS", "2")  # Bound native linear-algebra threads for reproducibility.
from visionamr.baselines.dorfler import run_dorfler  # Import the exact classical comparator.
from visionamr.bridge_cases import make_box_girder_diaphragm  # Import the medium-complexity bridge component.
from visionamr.experiment import FemRunner  # Import honest solve accounting.
from visionamr.vla.partition import ScriptedVisionPartitioner  # Import the solve-free visual-semantic partition.
from visionamr.vla.planner import PlannerConfig  # Import finite-horizon planner settings.
from visionamr.vla.world_model import ResidualWorldModel  # Import transition-library persistence.
from visionamr.vla.world_pipeline import WorldVLAConfig, run_world_model_vla  # Import the real multi-step trajectory.

def _records(runner: FemRunner, method: str) -> list:  # Select one method's counted records.
    return [record for record in runner.records if record.method == method]  # Preserve solve order exactly.

def _metric(record) -> tuple[str, float]:  # Select the strongest common accuracy metric available.
    if record.e_energy is not None:  # Prefer the independent reference energy error.
        return "e_energy", float(record.e_energy)  # Return the reference-based metric.
    return "sum_eta2", float(record.extra.get("sum_eta2", float("inf")))  # Fall back to the shared estimator for smoke runs.

def _compare(world_records: list, dorfler_records: list, tolerance: float) -> dict:  # Compare trajectories at equal counted solve indices.
    count = min(len(world_records), len(dorfler_records))  # Restrict comparison to shared solve indices.
    rows: list[dict] = []  # Allocate per-step comparison rows.
    for index in range(count):  # Compare every common real-solve position.
        world_name, world_value = _metric(world_records[index])  # Read the world-model metric.
        dorfler_name, dorfler_value = _metric(dorfler_records[index])  # Read the Dörfler metric.
        if world_name != dorfler_name:  # Reject inconsistent reference availability.
            raise RuntimeError("comparison metrics are inconsistent")  # Stop before reporting a mixed metric.
        ratio = world_value / max(dorfler_value, 1.0e-30)  # Compute the accuracy ratio at equal solve count.
        rows.append({"solve": index + 1, "metric": world_name, "world_model": world_value, "dorfler": dorfler_value, "ratio": ratio, "not_weaker": ratio <= 1.0 + tolerance, "strictly_better": ratio < 1.0 - tolerance})  # Store the auditable comparison.
    comparable = bool(rows)  # Record whether any common solve position exists.
    not_weaker = comparable and all(row["not_weaker"] for row in rows)  # Require the empirical floor at every shared solve.
    strictly_better = comparable and any(row["strictly_better"] for row in rows[1:])  # Require a post-probe strict advantage somewhere.
    return {"comparable_solves": count, "tolerance": tolerance, "not_weaker_at_equal_solves": not_weaker, "strict_world_model_advantage": strictly_better, "rows": rows}  # Return the trajectory comparison.

def main() -> int:  # Parse inputs, execute both methods, and write evidence.
    parser = argparse.ArgumentParser()  # Create the command-line interface.
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "world_model_vla" / "box_girder_diaphragm")  # Set the evidence directory.
    parser.add_argument("--budget", type=int, default=60000)  # Set the common equation cap.
    parser.add_argument("--max-solves", type=int, default=6)  # Set the real solve count.
    parser.add_argument("--horizon", type=int, default=4)  # Set the world-model rollout horizon.
    parser.add_argument("--beam-width", type=int, default=18)  # Set the bounded rollout beam.
    parser.add_argument("--theta", type=float, default=0.5)  # Set the common Dörfler bulk parameter.
    parser.add_argument("--no-reference", action="store_true")  # Disable the independent reference for fast smoke runs.
    parser.add_argument("--compare-dorfler", action="store_true")  # Run an independent exact Dörfler trajectory.
    parser.add_argument("--enforce-dominance", action="store_true")  # Fail the process if equal-solve empirical dominance is not met.
    parser.add_argument("--tolerance", type=float, default=0.01)  # Set the empirical not-weaker tolerance.
    parser.add_argument("--transition-library", type=Path, default=None)  # Load and update a reusable world-model snapshot.
    args = parser.parse_args()  # Parse all command-line arguments.
    args.output.mkdir(parents=True, exist_ok=True)  # Create the evidence directory.
    problem = make_box_girder_diaphragm()  # Instantiate the canonical bridge segment.
    model = ResidualWorldModel.load(args.transition_library) if args.transition_library and args.transition_library.exists() else ResidualWorldModel()  # Load prior transitions or start from the physics prior.
    planner = PlannerConfig(horizon=args.horizon, beam_width=args.beam_width, theta=args.theta)  # Configure finite-horizon search.
    config = WorldVLAConfig(n_eq_budget=args.budget, max_solves=args.max_solves, theta=args.theta, require_reference=not args.no_reference, planner=planner, model=model.config)  # Configure the real adaptive loop.
    world_runner = FemRunner(problem, args.output / "world_model", ccx_timeout=1800.0)  # Create isolated world-model solve accounting.
    world_result = run_world_model_vla(world_runner, ScriptedVisionPartitioner(), config, model=model)  # Execute the multi-step world-model VLA.
    world_runner.dump(args.output / "world_model_records.json")  # Persist every counted world-model solve.
    model_path = args.transition_library or (args.output / "transition_library.json")  # Select the updated transition-library destination.
    model.save(model_path)  # Persist all newly observed real transitions.
    payload = {"case": {"name": problem.name, "instance_id": problem.instance_id, "parameters": problem.params}, "contract": {"common_probe": "uniform_h0", "world_model_reads_local_prediction": False, "dorfler_candidate_always_present": True, "nodewise_target_never_coarser_than_dorfler": all(item["target_dominance_verified"] for item in world_result.action_log), "max_real_solves": args.max_solves, "equation_budget": args.budget, "theta": args.theta}, "world_model": world_result.to_dict(), "world_model_records": [asdict(record) for record in _records(world_runner, "world_model_vla")], "transition_library": str(model_path)}  # Assemble the primary evidence payload.
    comparison = None  # Initialize the optional comparator result.
    if args.compare_dorfler:  # Run the independent classical trajectory when requested.
        dorfler_runner = FemRunner(problem, args.output / "dorfler", ccx_timeout=1800.0)  # Create isolated Dörfler solve accounting.
        if not args.no_reference:  # Build the same independent reference before the comparator.
            dorfler_runner.ensure_reference()  # Preserve reference-based comparability.
        run_dorfler(dorfler_runner, theta=args.theta, max_rounds=max(args.max_solves - 1, 0), n_eq_cap=args.budget, require_reference=not args.no_reference)  # Execute the exact baseline under the same solve and equation caps.
        dorfler_runner.dump(args.output / "dorfler_records.json")  # Persist every counted comparator solve.
        comparison = _compare(_records(world_runner, "world_model_vla"), _records(dorfler_runner, "dorfler_zz"), args.tolerance)  # Compare equal-solve trajectories.
        payload["dorfler_records"] = [asdict(record) for record in _records(dorfler_runner, "dorfler_zz")]  # Attach the comparator records.
        payload["comparison"] = comparison  # Attach the empirical dominance gate.
    summary_path = args.output / "summary.json"  # Select the final evidence path.
    summary_path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")  # Write a transparent machine-readable summary.
    print(json.dumps({"summary": str(summary_path), "world_model": world_result.to_dict(), "comparison": comparison}, indent=1, default=str))  # Report only the final compact result to the terminal.
    if args.enforce_dominance and (comparison is None or not comparison["not_weaker_at_equal_solves"]):  # Enforce the empirical gate only when explicitly requested.
        return 2  # Return a distinct validation failure code.
    return 0  # Report successful execution.

if __name__ == "__main__":  # Execute only when the file is used as a script.
    raise SystemExit(main())  # Propagate the benchmark status to CI or the shell.
