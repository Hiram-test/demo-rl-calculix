#!/usr/bin/env python3  # Execute the benchmark with the active Python interpreter.
"""Run and compare the three-dimensional bridge world-model VLA trajectory."""  # Describe the script purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
import argparse  # Import command-line parsing.
from dataclasses import asdict  # Import record serialization.
import json  # Import structured result output.
import os  # Import deterministic thread environment settings.
from pathlib import Path  # Import filesystem path handling.
import sys  # Import repository path injection and exit codes.
import time  # Import monotonic benchmark timing.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository root.
sys.path.insert(0, str(ROOT))  # Make the local package importable without installation.
os.environ.setdefault("OMP_NUM_THREADS", "2")  # Bound native linear-algebra threads for reproducibility.
from visionamr.baselines.dorfler import run_dorfler  # Import the exact classical safety comparator.
from visionamr.baselines.local_prediction import run_local_prediction  # Import the independent element-wise prediction comparator.
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

def _record_view(record) -> dict:  # Build a compact accuracy-resource snapshot.
    metric_name, metric_value = _metric(record)  # Select the common accuracy metric.
    return {"solve": int(record.solve_index), "metric": metric_name, "value": metric_value, "n_equations": int(record.n_equations), "n_elements": int(record.n_elems), "wall_s": float(record.wall_s), "e_energy": None if record.e_energy is None else float(record.e_energy), "e_qoi": None if record.e_qoi is None else float(record.e_qoi), "sum_eta2": None if "sum_eta2" not in record.extra else float(record.extra["sum_eta2"])}  # Return only auditable primitive values.

def _compare_equal_solves(world_records: list, baseline_records: list, baseline_name: str, tolerance: float) -> dict:  # Compare trajectories at equal counted solve indices.
    count = min(len(world_records), len(baseline_records))  # Restrict comparison to shared solve indices.
    rows: list[dict] = []  # Allocate per-step comparison rows.
    for index in range(count):  # Compare every common real-solve position.
        world_name, world_value = _metric(world_records[index])  # Read the world-model metric.
        baseline_metric, baseline_value = _metric(baseline_records[index])  # Read the comparator metric.
        if world_name != baseline_metric:  # Reject inconsistent reference availability.
            raise RuntimeError("comparison metrics are inconsistent")  # Stop before reporting a mixed metric.
        ratio = world_value / max(baseline_value, 1.0e-30)  # Compute the accuracy ratio at equal solve count.
        rows.append({"solve": index + 1, "metric": world_name, "world_model": world_value, baseline_name: baseline_value, "ratio": ratio, "not_weaker": ratio <= 1.0 + tolerance, "measured_improvement": ratio < 1.0, "strictly_better": ratio < 1.0 - tolerance, "world_n_equations": int(world_records[index].n_equations), f"{baseline_name}_n_equations": int(baseline_records[index].n_equations)})  # Store accuracy and resource context together.
    comparable = bool(rows)  # Record whether any common solve position exists.
    not_weaker = comparable and all(row["not_weaker"] for row in rows)  # Require the empirical floor at every shared solve.
    measured_improvement = comparable and any(row["measured_improvement"] for row in rows[1:])  # Detect any post-probe measured gain.
    strict_advantage = comparable and any(row["strictly_better"] for row in rows[1:])  # Require a gain beyond the declared tolerance.
    return {"baseline": baseline_name, "comparable_solves": count, "tolerance": tolerance, "not_weaker_at_equal_solves": not_weaker, "measured_world_model_improvement": measured_improvement, "strict_world_model_advantage": strict_advantage, "rows": rows}  # Return the complete equal-solve comparison.

def _best_in_budget(records: list, budget: int) -> object | None:  # Select one method's best accuracy point under the shared equation cap.
    eligible = [record for record in records if record.n_equations <= budget]  # Retain only resource-feasible real solves.
    if not eligible:  # Handle a probe that already exceeds the resource cap.
        return None  # Report absence of a feasible point explicitly.
    return min(eligible, key=lambda record: _metric(record)[1])  # Select the smallest common accuracy metric.

def _compare_equation_budget(world_records: list, baseline_records: list, baseline_name: str, budget: int, tolerance: float) -> dict:  # Compare the best deliverable points under one equation cap.
    world_best = _best_in_budget(world_records, budget)  # Select the world-model budget-feasible point.
    baseline_best = _best_in_budget(baseline_records, budget)  # Select the comparator budget-feasible point.
    if world_best is None or baseline_best is None:  # Require both methods to produce a feasible deliverable.
        return {"baseline": baseline_name, "budget": budget, "comparable": False, "not_weaker": False, "reason": "missing_in_budget_point", "world_model": None if world_best is None else _record_view(world_best), baseline_name: None if baseline_best is None else _record_view(baseline_best)}  # Return the incomplete comparison transparently.
    world_name, world_value = _metric(world_best)  # Read the selected world-model accuracy.
    baseline_metric, baseline_value = _metric(baseline_best)  # Read the selected comparator accuracy.
    if world_name != baseline_metric:  # Reject inconsistent reference availability.
        raise RuntimeError("budget comparison metrics are inconsistent")  # Stop before reporting a mixed metric.
    ratio = world_value / max(baseline_value, 1.0e-30)  # Compute the best-in-budget accuracy ratio.
    return {"baseline": baseline_name, "budget": budget, "comparable": True, "metric": world_name, "ratio": ratio, "not_weaker": ratio <= 1.0 + tolerance, "measured_improvement": ratio < 1.0, "strictly_better": ratio < 1.0 - tolerance, "world_model": _record_view(world_best), baseline_name: _record_view(baseline_best)}  # Return the complete budget comparison.

def _method_timing(records: list, external_total_s: float) -> dict:  # Summarize counted solver and total method wall time.
    solver_s = float(sum(record.wall_s for record in records))  # Sum the recorded CalculiX solve times.
    return {"calculix_s": solver_s, "external_total_s": float(external_total_s), "non_calculix_s": max(float(external_total_s) - solver_s, 0.0), "solves": len(records)}  # Separate solver and orchestration costs.

def main() -> int:  # Parse inputs, execute methods, and write evidence.
    parser = argparse.ArgumentParser()  # Create the command-line interface.
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "world_model_vla" / "box_girder_diaphragm")  # Set the evidence directory.
    parser.add_argument("--budget", type=int, default=60000)  # Set the common equation cap.
    parser.add_argument("--max-solves", type=int, default=6)  # Set the real solve count.
    parser.add_argument("--horizon", type=int, default=4)  # Set the world-model rollout horizon.
    parser.add_argument("--beam-width", type=int, default=18)  # Set the bounded rollout beam.
    parser.add_argument("--theta", type=float, default=0.5)  # Set the common Dörfler bulk parameter.
    parser.add_argument("--no-reference", action="store_true")  # Disable the independent reference for fast smoke runs.
    parser.add_argument("--compare-dorfler", action="store_true")  # Run an independent exact Dörfler trajectory.
    parser.add_argument("--compare-local-prediction", action="store_true")  # Run the independent element-wise local prediction trajectory.
    parser.add_argument("--local-element-budget", type=int, default=None)  # Override the equation-derived local-prediction element target.
    parser.add_argument("--enforce-dominance", action="store_true")  # Fail when the equal-solve Dörfler floor is violated.
    parser.add_argument("--enforce-local-advantage", action="store_true")  # Fail unless world-model VLA beats local prediction under the equation cap.
    parser.add_argument("--tolerance", type=float, default=0.01)  # Set the empirical not-weaker tolerance.
    parser.add_argument("--transition-library", type=Path, default=None)  # Load and update a reusable world-model snapshot.
    args = parser.parse_args()  # Parse all command-line arguments.
    if args.max_solves < 1:  # Require at least the common probe solve.
        parser.error("--max-solves must be positive")  # Reject an empty comparison.
    if args.budget <= 0:  # Require a positive equation cap.
        parser.error("--budget must be positive")  # Reject an invalid resource contract.
    args.output.mkdir(parents=True, exist_ok=True)  # Create the evidence directory.
    problem = make_box_girder_diaphragm()  # Instantiate the canonical bridge segment.
    model = ResidualWorldModel.load(args.transition_library) if args.transition_library and args.transition_library.exists() else ResidualWorldModel()  # Load prior transitions or start from the physics prior.
    planner = PlannerConfig(horizon=args.horizon, beam_width=args.beam_width, theta=args.theta)  # Configure finite-horizon search.
    config = WorldVLAConfig(n_eq_budget=args.budget, max_solves=args.max_solves, theta=args.theta, require_reference=not args.no_reference, planner=planner, model=model.config)  # Configure the real adaptive loop.
    world_runner = FemRunner(problem, args.output / "world_model", ccx_timeout=1800.0)  # Create isolated world-model solve accounting.
    world_started = time.perf_counter()  # Start end-to-end world-model timing after runner construction.
    world_result = run_world_model_vla(world_runner, ScriptedVisionPartitioner(), config, model=model)  # Execute the multi-step world-model VLA.
    world_total_s = time.perf_counter() - world_started  # Measure the complete world-model trajectory.
    world_records = _records(world_runner, "world_model_vla")  # Cache the counted world-model solves.
    world_runner.dump(args.output / "world_model_records.json")  # Persist every counted world-model solve.
    model_path = args.transition_library or (args.output / "transition_library.json")  # Select the updated transition-library destination.
    model.save(model_path)  # Persist all newly observed real transitions.
    payload = {"case": {"name": problem.name, "instance_id": problem.instance_id, "parameters": problem.params}, "contract": {"common_probe": "uniform_h0", "world_model_reads_local_prediction": False, "dorfler_candidate_always_present": True, "nodewise_target_never_coarser_than_dorfler": all(item["target_dominance_verified"] for item in world_result.action_log), "max_real_solves": args.max_solves, "equation_budget": args.budget, "theta": args.theta}, "world_model": world_result.to_dict(), "world_model_timing": _method_timing(world_records, world_total_s), "world_model_records": [asdict(record) for record in world_records], "transition_library": str(model_path), "comparisons": {}}  # Assemble the primary evidence payload.
    dorfler_comparison = None  # Initialize the optional Dörfler result.
    if args.compare_dorfler:  # Run the independent classical trajectory when requested.
        dorfler_runner = FemRunner(problem, args.output / "dorfler", ccx_timeout=1800.0)  # Create isolated Dörfler solve accounting.
        if not args.no_reference:  # Build the same independent reference before comparator timing.
            dorfler_runner.ensure_reference()  # Preserve reference-based comparability.
        dorfler_started = time.perf_counter()  # Start the comparator's adaptive-trajectory timer.
        run_dorfler(dorfler_runner, theta=args.theta, max_rounds=max(args.max_solves - 1, 0), n_eq_cap=args.budget, require_reference=False)  # Execute the exact baseline under the same solve and equation caps.
        dorfler_total_s = time.perf_counter() - dorfler_started  # Measure the complete Dörfler trajectory after reference setup.
        dorfler_records = _records(dorfler_runner, "dorfler_zz")  # Cache counted comparator solves.
        dorfler_runner.dump(args.output / "dorfler_records.json")  # Persist every counted comparator solve.
        equal_solves = _compare_equal_solves(world_records, dorfler_records, "dorfler", args.tolerance)  # Compare trajectories at equal real solve counts.
        equation_budget = _compare_equation_budget(world_records, dorfler_records, "dorfler", args.budget, args.tolerance)  # Compare best deliverables under the shared equation cap.
        dorfler_comparison = {"equal_solves": equal_solves, "equation_budget": equation_budget, "timing": {"world_model": _method_timing(world_records, world_total_s), "dorfler": _method_timing(dorfler_records, dorfler_total_s)}}  # Assemble the complete Dörfler comparison.
        payload["dorfler_records"] = [asdict(record) for record in dorfler_records]  # Attach the comparator records.
        payload["comparisons"]["dorfler"] = dorfler_comparison  # Attach all Dörfler evidence.
        payload["comparison"] = equal_solves  # Preserve the original verifier-compatible equal-solve field.
    local_comparison = None  # Initialize the optional local-prediction result.
    if args.compare_local_prediction:  # Run the independent local size-prediction trajectory when requested.
        probe = world_records[0]  # Use only common-probe resource scaling to derive the target element count.
        equations_per_element = probe.n_equations / max(probe.n_elems, 1)  # Estimate the current equation-to-element conversion.
        derived_element_budget = max(int(round(args.budget / max(equations_per_element, 1.0e-9))), 1)  # Translate the equation cap without reading any world-model action.
        local_element_budget = args.local_element_budget or derived_element_budget  # Use an explicit target when supplied.
        local_runner = FemRunner(problem, args.output / "local_prediction", ccx_timeout=1800.0)  # Create isolated local-prediction solve accounting.
        if not args.no_reference:  # Build the same independent reference before comparator timing.
            local_runner.ensure_reference()  # Preserve reference-based comparability.
        local_started = time.perf_counter()  # Start the local-prediction adaptive-trajectory timer.
        run_local_prediction(local_runner, budgets=[local_element_budget], rounds=max(args.max_solves - 1, 0), method="local_prediction", require_reference=False)  # Execute the independent element-wise predictor for the same counted solve depth.
        local_total_s = time.perf_counter() - local_started  # Measure the complete local-prediction trajectory after reference setup.
        local_records = _records(local_runner, "local_prediction")  # Cache counted local-prediction solves.
        local_runner.dump(args.output / "local_prediction_records.json")  # Persist every counted local-prediction solve.
        equal_solves = _compare_equal_solves(world_records, local_records, "local_prediction", args.tolerance)  # Compare trajectories at equal real solve counts.
        equation_budget = _compare_equation_budget(world_records, local_records, "local_prediction", args.budget, args.tolerance)  # Compare best deliverables under the shared equation cap.
        local_comparison = {"target_element_budget": int(local_element_budget), "derived_element_budget": int(derived_element_budget), "equal_solves": equal_solves, "equation_budget": equation_budget, "timing": {"world_model": _method_timing(world_records, world_total_s), "local_prediction": _method_timing(local_records, local_total_s)}}  # Assemble the complete local-prediction comparison.
        payload["local_prediction_records"] = [asdict(record) for record in local_records]  # Attach the independent comparator records.
        payload["comparisons"]["local_prediction"] = local_comparison  # Attach all local-prediction evidence.
    summary_path = args.output / "summary.json"  # Select the final evidence path.
    summary_path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")  # Write a transparent machine-readable summary.
    print(json.dumps({"summary": str(summary_path), "world_model": world_result.to_dict(), "comparisons": payload["comparisons"]}, indent=1, default=str))  # Report only the final compact result to the terminal.
    if args.enforce_dominance and (dorfler_comparison is None or not dorfler_comparison["equal_solves"]["not_weaker_at_equal_solves"] or not dorfler_comparison["equation_budget"]["not_weaker"]):  # Enforce both equal-solve and equation-budget Dörfler floors.
        return 2  # Return a distinct Dörfler validation failure code.
    if args.enforce_local_advantage and (local_comparison is None or not local_comparison["equation_budget"].get("strictly_better", False)):  # Require a real budget-aligned local-prediction advantage only when requested.
        return 3  # Return a distinct local-prediction validation failure code.
    return 0  # Report successful execution.

if __name__ == "__main__":  # Execute only when the file is used as a script.
    raise SystemExit(main())  # Propagate the benchmark status to CI or the shell.
