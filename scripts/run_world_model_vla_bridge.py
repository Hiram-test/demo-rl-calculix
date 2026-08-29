#!/usr/bin/env python3  # Execute this campaign with the repository Python environment.
"""Run independent Dörfler and multi-step world-model VLA bridge campaigns."""  # Describe the executable scientific comparison.

from __future__ import annotations  # Enable postponed annotation evaluation.

import argparse  # Import command-line argument parsing.
import importlib  # Import dynamic discovery of the repository vision head.
import inspect  # Import constructor signature inspection for repository compatibility.
import json  # Import structured result serialization.
import math  # Import finite-value and square-root utilities.
import sys  # Import deterministic campaign failure exit codes.
from dataclasses import asdict  # Import dataclass serialization for campaign metadata.
from dataclasses import replace  # Import immutable configuration specialization.
from pathlib import Path  # Import portable output-directory handling.
from typing import Any  # Import structural typing for repository runtime objects.

import numpy as np  # Import deterministic random sampling and numerical comparisons.

from visionamr.baselines.dorfler import refine_size_map  # Import the exact baseline refinement atom.
from visionamr.bridge_diaphragm import sample_steel_box_diaphragm  # Import the medium-complexity bridge family.
from visionamr.experiment import FemRunner  # Import the audited CalculiX execution gateway.
from visionamr.experiment import initial_mesh  # Import the common uniform probe mesh.
from visionamr.indicators import zz_indicator  # Import the shared element-wise ZZ estimator.
from visionamr.marking import dorfler_mark  # Import exact element-level Dörfler bulk marking.
from visionamr.mesher import generate_mesh  # Import the common Gmsh remeshing gateway.
from visionamr.sizefield import NodalSizeField  # Import deterministic nodal target interpolation.
from visionamr.vla.world_controller import RegionalWorldModel  # Import the action-conditioned regional world model.
from visionamr.vla.world_controller import WorldControllerConfig  # Import the immutable method configuration.
from visionamr.vla.world_controller import estimate_free_equations  # Import exact generated-mesh free-DOF certification.
from visionamr.vla.world_controller import run_world_model_vla  # Import the real multi-step controller.


def _make_runner(problem: Any, directory: Path) -> FemRunner:  # Instantiate the repository runner across supported constructor spellings.
    directory.mkdir(parents=True, exist_ok=True)  # Materialize the isolated method run directory.
    parameters = inspect.signature(FemRunner).parameters  # Inspect the installed repository runner contract.
    for keyword in ("workdir", "run_dir", "root", "out_dir", "output_dir"):  # Try known explicit directory parameter names.
        if keyword in parameters:  # Select the first supported keyword.
            return FemRunner(problem, **{keyword: directory})  # Construct the runner with an isolated deterministic workspace.
    return FemRunner(problem, directory)  # Fall back to the historical positional workspace contract.


def _make_partitioner() -> Any:  # Discover the existing scripted vision-region head without coupling to one legacy filename.
    module_names = ("visionamr.vla.vision", "visionamr.vla.eye", "visionamr.vla.heads", "visionamr.vla.pipeline")  # Enumerate repository locations used across VLA revisions.
    preferred_names = ("ScriptedVisionPartitioner", "ScriptedVisionHead", "ScriptedEye", "ScriptedVision", "VisionPartitioner")  # Enumerate explicit deterministic vision-head names first.
    for module_name in module_names:  # Search each possible repository module.
        try:  # Isolate optional module-layout differences.
            module = importlib.import_module(module_name)  # Import the current VLA module.
        except ImportError:  # Ignore layouts not present in this branch.
            continue  # Search the next module.
        for name in preferred_names:  # Try explicit stable class names first.
            candidate = getattr(module, name, None)  # Read the named candidate safely.
            if inspect.isclass(candidate):  # Require a constructible class.
                for kwargs in ({}, {"seed": 0}):  # Try no-argument and deterministic-seed constructors.
                    try:  # Isolate constructor signature differences.
                        instance = candidate(**kwargs)  # Construct the scripted vision head.
                    except TypeError:  # Ignore an incompatible constructor variant.
                        continue  # Try the next constructor form.
                    if callable(getattr(instance, "propose", None)):  # Require the semantic-region proposal contract.
                        return instance  # Return the compatible repository vision head.
        for name, candidate in inspect.getmembers(module, inspect.isclass):  # Fall back to structural discovery within the VLA namespace.
            if "script" not in name.lower() and "vision" not in name.lower():  # Exclude unrelated classes.
                continue  # Search the next class.
            try:  # Attempt the safest no-argument construction.
                instance = candidate()  # Construct the discovered class.
            except TypeError:  # Ignore classes requiring unrelated dependencies.
                continue  # Search the next class.
            if callable(getattr(instance, "propose", None)):  # Require the semantic-region proposal contract.
                return instance  # Return the structurally compatible vision head.
    raise RuntimeError("no scripted VLA vision partitioner with propose(problem) was found")  # Fail explicitly instead of silently removing the VLA component.


def _record_equations(record: Any) -> int:  # Read the actual equation count across repository record revisions.
    for name in ("n_equations", "n_eq", "equations"):  # Enumerate supported field spellings.
        value = getattr(record, name, None)  # Read one candidate value.
        if value is not None:  # Accept the first materialized count.
            return int(value)  # Return a normalized integer.
    raise AttributeError("solve record contains no equation count")  # Reject unauditable resource accounting.


def _record_wall(record: Any) -> float:  # Read total wall time across repository record revisions.
    for name in ("wall_s", "wall_seconds", "wall_time", "elapsed_s"):  # Enumerate supported field spellings.
        value = getattr(record, name, None)  # Read one candidate value.
        if value is not None:  # Accept the first materialized duration.
            return float(value)  # Return a normalized floating-point duration.
    return float("nan")  # Preserve missing timing as an explicit non-finite value.


def _record_metric(record: Any) -> tuple[str, float]:  # Select a non-oracular controller-independent evaluation metric.
    for name in ("e_energy", "energy_error", "rel_energy_error"):  # Prefer a common reference-based energy error when available.
        value = getattr(record, name, None)  # Read one reference metric candidate.
        if value is not None and math.isfinite(float(value)):  # Require a finite realized error.
            return "energy_error", float(value)  # Return the authoritative reference metric.
    extra = getattr(record, "extra", {}) or {}  # Read the structured solve diagnostics.
    if "sum_eta2" in extra and math.isfinite(float(extra["sum_eta2"])):  # Fall back to the shared ZZ estimator mass.
        return "zz_estimator", float(math.sqrt(max(float(extra["sum_eta2"]), 0.0)))  # Return the estimator norm rather than its squared mass.
    raise ValueError("solve record contains neither reference error nor ZZ estimator")  # Reject an unscorable campaign record.


def _records_for(runner: FemRunner, method: str) -> list[Any]:  # Isolate one method's counted CalculiX solves.
    return [record for record in runner.records if getattr(record, "method", None) == method]  # Preserve original chronological order.


def run_certified_dorfler(runner: FemRunner, n_eq_cap: int, config: WorldControllerConfig, method: str = "dorfler_certified") -> dict[str, Any]:  # Run the exact baseline with the same pre-mesh budget certification as WM-VLA.
    if config.require_reference:  # Build or load common reference evidence when requested.
        runner.ensure_reference()  # Materialize the shared reference outside counted method solves.
    problem = runner.problem  # Read the runner-owned finite-element problem.
    mesh = initial_mesh(problem)  # Start from the same uniform probe as the scientific WM-VLA comparison.
    stopped_by = "solve_cap"  # Set the default terminal reason.
    for step in range(config.max_solves):  # Execute the same maximum number of sequential physical feedback states.
        post, record = runner.solve_mesh(mesh, method=method, stage=f"cycle{step}")  # Execute one real baseline CalculiX solve.
        eta2 = zz_indicator(problem, post)  # Compute the shared element-wise ZZ indicator.
        marked = dorfler_mark(eta2, config.theta)  # Select the exact bulk-marking support.
        record.extra.update(sum_eta2=float(np.sum(eta2)), n_marked=int(len(marked)), controller="certified_exact_dorfler", local_prediction_imported=False)  # Attach common evaluation evidence.
        if len(marked) == 0:  # Stop when the estimator produces no refinement support.
            stopped_by = "empty_marking"  # Record the physical terminal condition.
            record.extra["stop"] = stopped_by  # Preserve the condition in the solve record.
            break  # End the adaptive loop.
        if step + 1 >= config.max_solves:  # Stop after the configured number of real solves.
            stopped_by = "solve_cap"  # Record the configured terminal condition.
            record.extra["stop"] = stopped_by  # Preserve the condition in the solve record.
            break  # End the adaptive loop.
        if _record_equations(record) >= n_eq_cap:  # Stop after reaching the same hard equation budget.
            stopped_by = "equation_cap"  # Record the resource terminal condition.
            record.extra["stop"] = stopped_by  # Preserve the condition in the solve record.
            break  # End the adaptive loop.
        target = refine_size_map(mesh, marked, factor=config.refine_factor)  # Build the exact Dörfler nodal target field.
        field = NodalSizeField(mesh, target, gradation=config.gradation, h_min=problem.h_min, h_max=problem.h0)  # Apply the same deterministic gradation and bounds as WM-VLA.
        next_mesh = generate_mesh(problem, field)  # Materialize the next Gmsh mesh before launching a solve.
        estimated = estimate_free_equations(next_mesh, problem)  # Compute exact free equations on the generated mesh.
        record.extra.update(next_estimated_equations=int(estimated), target_kind="exact_dorfler")  # Preserve pre-solve budget evidence.
        if estimated > int(math.floor(config.budget_safety * n_eq_cap)):  # Apply the identical safety-adjusted budget gate.
            stopped_by = "next_dorfler_mesh_over_budget"  # Record the fair pre-solve resource stop.
            record.extra["stop"] = stopped_by  # Preserve the condition in the solve record.
            break  # End the adaptive loop before an over-budget solve.
        mesh = next_mesh  # Advance to the certified exact-Dörfler mesh.
    return {"solves": len(_records_for(runner, method)), "stopped_by": stopped_by}  # Return the baseline run summary.


def _point(record: Any, solve_index: int) -> dict[str, Any]:  # Convert one solve record to a compact comparison point.
    metric_name, metric = _record_metric(record)  # Read the common error or estimator metric.
    return {"solve": int(solve_index), "n_equations": _record_equations(record), "wall_s": _record_wall(record), "metric_name": metric_name, "metric": metric, "qoi": getattr(record, "qoi", None), "qoi_error": getattr(record, "e_qoi", getattr(record, "qoi_error", None))}  # Return the normalized scientific point.


def _method_points(runner: FemRunner, method: str) -> list[dict[str, Any]]:  # Convert one method trajectory to normalized comparison points.
    return [_point(record, index + 1) for index, record in enumerate(_records_for(runner, method))]  # Preserve solve-index chronology.


def compare_to_dorfler(dorfler: list[dict[str, Any]], world: list[dict[str, Any]], tolerance: float = 1.0e-8) -> dict[str, Any]:  # Test the same-solves and same-budget Dörfler performance floor.
    if not dorfler or not world:  # Reject an empty physical trajectory.
        return {"not_weaker": False, "reason": "empty_trajectory", "checks": [], "best_gain": float("nan"), "solve_saving": 0}  # Report a failed comparison contract.
    metric_names = {point["metric_name"] for point in dorfler + world}  # Read all available comparison metric types.
    if len(metric_names) != 1:  # Require the same reference or estimator metric across both methods.
        return {"not_weaker": False, "reason": "metric_mismatch", "checks": [], "best_gain": float("nan"), "solve_saving": 0}  # Reject mixed evidence levels.
    checks: list[dict[str, Any]] = []  # Allocate pointwise Pareto-envelope checks.
    for world_point in world:  # Test every realized world-model point against reachable Dörfler evidence.
        admissible = [point for point in dorfler if point["solve"] <= world_point["solve"] and point["n_equations"] <= world_point["n_equations"]]  # Match no more sequential solves and no more equations.
        if not admissible:  # Skip only when Dörfler has no point inside the world point's resource rectangle.
            checks.append({"world": world_point, "dorfler_best": None, "pass": True, "reason": "no_dorfler_point_in_resource_rectangle"})  # Record the vacuous resource comparison explicitly.
            continue  # Test the next world point.
        dorfler_best = min(admissible, key=lambda point: point["metric"])  # Select the Dörfler Pareto-envelope metric inside the resource rectangle.
        passed = world_point["metric"] <= dorfler_best["metric"] * (1.0 + tolerance)  # Enforce the non-weaker performance floor.
        checks.append({"world": world_point, "dorfler_best": dorfler_best, "pass": bool(passed), "ratio": float(world_point["metric"] / max(dorfler_best["metric"], 1.0e-30))})  # Preserve the quantitative floor check.
    dorfler_final = min(point["metric"] for point in dorfler)  # Read the best baseline metric under the common campaign cap.
    world_final = min(point["metric"] for point in world)  # Read the best world-model metric under the common campaign cap.
    best_gain = (dorfler_final - world_final) / max(dorfler_final, 1.0e-30)  # Compute the relative final best-so-far improvement.
    world_target_solve = min((point["solve"] for point in world if point["metric"] <= dorfler_final * (1.0 + tolerance)), default=len(world) + 1)  # Find when WM-VLA first reaches the baseline's final quality.
    dorfler_target_solve = min((point["solve"] for point in dorfler if point["metric"] <= dorfler_final * (1.0 + tolerance)), default=len(dorfler))  # Find when Dörfler reaches its own final quality.
    solve_saving = max(int(dorfler_target_solve - world_target_solve), 0)  # Quantify reduced sequential physical feedback rounds.
    not_weaker = all(bool(check["pass"]) for check in checks) and world_final <= dorfler_final * (1.0 + tolerance)  # Require both the Pareto envelope and final best point.
    return {"not_weaker": bool(not_weaker), "reason": "ok" if not_weaker else "dorfler_floor_failed", "checks": checks, "best_gain": float(best_gain), "solve_saving": int(solve_saving), "world_advantage": bool(not_weaker and (best_gain > 0.02 or solve_saving >= 1))}  # Return the complete scientific comparison.


def _case_problem(rng: np.random.Generator, smoke: bool) -> Any:  # Draw a training or test member of the bridge-component family.
    problem = sample_steel_box_diaphragm(rng)  # Draw geometry, load footprint, and pressure independently of local prediction.
    if smoke:  # Reduce only mesh density and geometry scale for continuous integration.
        parameters = dict(problem.params)  # Read the sampled physical parameters.
        from visionamr.bridge_diaphragm import make_steel_box_diaphragm  # Import the deterministic benchmark constructor locally.

        return make_steel_box_diaphragm(length=1400.0, width=800.0, height=600.0, plate_t=48.0, diaphragm_t=58.0, hole_radius=min(float(parameters["hole_radius"]) * 0.58, 145.0), rib_t=36.0, rib_h=105.0, wheel=(min(float(parameters["wheel"][0]) * 0.65, 300.0), min(float(parameters["wheel"][1]) * 0.65, 210.0)), wheel_offset=(float(parameters["wheel_offset"][0]) * 0.45, float(parameters["wheel_offset"][1]) * 0.45), pressure=float(parameters["pressure"]), support_length=210.0, support_width=300.0, h0=190.0, h_ref=100.0, h_min=55.0)  # Return a topologically identical coarse smoke case.
    return problem  # Return the full medium-complexity case.


def parse_args() -> argparse.Namespace:  # Define the reproducible campaign command-line interface.
    parser = argparse.ArgumentParser(description=__doc__)  # Create the argument parser from the module purpose.
    parser.add_argument("--output", type=Path, default=Path("artifacts/world_model_vla_bridge"))  # Select the isolated campaign artifact directory.
    parser.add_argument("--seed", type=int, default=20260830)  # Select the deterministic family-sampling seed.
    parser.add_argument("--train-cases", type=int, default=4)  # Select the number of independent transition-collection cases.
    parser.add_argument("--test-cases", type=int, default=3)  # Select the number of held-out bridge cases.
    parser.add_argument("--max-solves", type=int, default=8)  # Select the maximum real CalculiX feedback states per method.
    parser.add_argument("--horizon", type=int, default=4)  # Select the finite world-model planning horizon.
    parser.add_argument("--n-eq-cap", type=int, default=180000)  # Select the common hard equation budget.
    parser.add_argument("--smoke", action="store_true")  # Select a topologically identical coarse continuous-integration campaign.
    parser.add_argument("--no-reference", action="store_true")  # Use the ZZ estimator when a full reference campaign is intentionally skipped.
    parser.add_argument("--allow-weaker", action="store_true")  # Permit exploratory artifacts without enforcing the Dörfler performance gate.
    return parser.parse_args()  # Return the parsed immutable command-line namespace.


def main() -> int:  # Execute transition collection, held-out comparison, and the hard Dörfler gate.
    args = parse_args()  # Read the campaign configuration.
    args.output.mkdir(parents=True, exist_ok=True)  # Materialize the campaign artifact root.
    rng = np.random.default_rng(args.seed)  # Create the deterministic family sampler.
    if args.smoke:  # Bound continuous-integration workload explicitly.
        args.train_cases = min(args.train_cases, 1)  # Use one independent transition-collection case.
        args.test_cases = min(args.test_cases, 1)  # Use one held-out physical comparison case.
        args.max_solves = min(args.max_solves, 3)  # Use three real states while preserving multi-step behaviour.
    config = WorldControllerConfig(max_solves=args.max_solves, planning_horizon=args.horizon, require_reference=not args.no_reference and not args.smoke)  # Build the common immutable method contract.
    training_config = replace(config, min_predicted_gain=0.0, max_log_error_sigma=0.80, budget_safety=1.0, require_reference=False)  # Permit safe independent action exploration during transition collection.
    world = RegionalWorldModel(config)  # Initialize one cross-case regional transition library.
    training: list[dict[str, Any]] = []  # Allocate transition-collection summaries.
    for index in range(args.train_cases):  # Collect real action-conditioned transitions on independent family members.
        problem = _case_problem(rng, args.smoke)  # Draw one training geometry and load state.
        runner = _make_runner(problem, args.output / "train" / f"case_{index:02d}")  # Isolate all CalculiX and Gmsh artifacts.
        result = run_world_model_vla(runner, _make_partitioner(), args.n_eq_cap, config=training_config, model=world, method="wm_train")  # Collect online residual evidence without local prediction.
        training.append({"index": index, "problem": problem.params, "result": asdict(result), "points": _method_points(runner, "wm_train")})  # Preserve complete training provenance.
    model_path = world.save(args.output / "world_model_transitions.json")  # Persist the independent transition library before held-out testing.
    tests: list[dict[str, Any]] = []  # Allocate held-out method-comparison results.
    all_floor_pass = True  # Initialize the aggregate Dörfler floor gate.
    any_world_advantage = False  # Initialize the aggregate world-model advantage flag.
    for index in range(args.test_cases):  # Evaluate distinct held-out bridge-component variants.
        problem = _case_problem(rng, args.smoke)  # Draw one test geometry and load state not used in transition collection.
        dorfler_runner = _make_runner(problem, args.output / "test" / f"case_{index:02d}" / "dorfler")  # Isolate the baseline execution.
        dorfler_summary = run_certified_dorfler(dorfler_runner, args.n_eq_cap, config, method="dorfler_certified")  # Run the exact certified baseline.
        world_runner = _make_runner(problem, args.output / "test" / f"case_{index:02d}" / "world")  # Isolate the world-model execution.
        world_summary = run_world_model_vla(world_runner, _make_partitioner(), args.n_eq_cap, config=config, model=world, method="wm_vla")  # Run the held-out multi-step controller.
        dorfler_points = _method_points(dorfler_runner, "dorfler_certified")  # Normalize the baseline trajectory.
        world_points = _method_points(world_runner, "wm_vla")  # Normalize the world-model trajectory.
        comparison = compare_to_dorfler(dorfler_points, world_points)  # Enforce the same-solves and same-equations performance floor.
        all_floor_pass = all_floor_pass and bool(comparison["not_weaker"])  # Accumulate the hard Dörfler gate.
        any_world_advantage = any_world_advantage or bool(comparison.get("world_advantage", False))  # Record whether a held-out case demonstrates world-model value.
        tests.append({"index": index, "problem": problem.params, "dorfler": dorfler_summary, "world": asdict(world_summary), "dorfler_points": dorfler_points, "world_points": world_points, "comparison": comparison})  # Preserve the complete held-out audit record.
    payload = {"schema": 1, "seed": args.seed, "config": asdict(config), "n_eq_cap": args.n_eq_cap, "training": training, "tests": tests, "world_model_rows": world.sample_count, "world_model_snapshot": str(model_path), "dorfler_floor_pass": bool(all_floor_pass), "world_model_advantage_observed": bool(any_world_advantage), "local_prediction_used": False}  # Build the final machine-readable campaign result.
    result_path = args.output / "campaign.json"  # Select the canonical result filename.
    result_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")  # Write the complete campaign evidence.
    print(json.dumps({"result": str(result_path), "dorfler_floor_pass": all_floor_pass, "world_model_advantage_observed": any_world_advantage, "world_model_rows": world.sample_count}, indent=1))  # Print only the concise final gate summary.
    if not all_floor_pass and not args.allow_weaker:  # Enforce the user's non-weaker requirement by default.
        return 2  # Fail the campaign when WM-VLA falls below the Dörfler Pareto envelope.
    return 0  # Report successful execution and gate completion.


if __name__ == "__main__":  # Execute only when the module is launched as a script.
    sys.exit(main())  # Return the deterministic campaign status to the shell or CI.
