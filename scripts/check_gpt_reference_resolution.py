#!/usr/bin/env python3  # Reevaluate frozen candidate fields against exactly three finer native references.
"""Check whether GPT's small held-out advantages survive a common h0/4 reference."""  # State the narrow numerical purpose.
from __future__ import annotations  # Support current type annotations without changing runtime evaluation.
import argparse  # Expose explicit artifact locations for reproducible replay.
import hashlib  # Record the unchanged source decisions and numerical fields.
import json  # Read completed experiment records and save the resolution report.
import os  # Preserve deterministic native threading.
import sys  # Locate the existing project modules.
import time  # Report actual new-reference and reevaluation effort.
from dataclasses import asdict  # Preserve the genuine native solver receipt.
from pathlib import Path  # Resolve input and output artifacts without changing source data.
import numpy as np  # Restore saved finite-element arrays and compute ratios.
ROOT = Path(__file__).resolve().parents[1]  # Resolve this script's repository root.
sys.path.insert(0, str(ROOT))  # Import the current repository through its normal package path.
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Keep the same native thread convention as the original experiment.
from visionamr.experiment import FemRunner  # Retain native CalculiX solve accounting and raw solver files.
from visionamr.fem_post import compute_post  # Recompute stresses from each unchanged cached displacement field.
from visionamr.mesher import Mesh, generate_uniform  # Use the common authentic Gmsh mesh realization.
from scripts.run_visual_world_experiment import case_problem, reference_error, write_json  # Reuse the exact geometry factory and original error integration.
SEEDS = (901, 902, 903)  # Fix the three already evaluated held-out cases without selecting favorable instances.
REFERENCE_FACTOR = 4.0  # Fix the requested finer reference before reevaluating any candidate.
def sha256(path):  # Fingerprint an existing artifact without changing its contents.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()  # Bind evidence to the exact numerical or decision bytes.
def restore_post(path, problem):  # Restore a cached solution without rerunning its candidate solver.
    with np.load(path, allow_pickle=False) as saved:  # Read only plain numerical arrays from the saved artifact.
        mesh = Mesh(saved["nodes"].copy(), saved["cells"].copy(), 3)  # Restore the original physical tetrahedral mesh.
        displacement = saved["u"].copy()  # Preserve the native displacement solution exactly as saved.
    return compute_post(mesh, problem, displacement)  # Recompute the original stress and energy fields from displacement.
def summarize_choice(row, comparison):  # Compare one frozen method with one explicitly identified baseline.
    old_ratio = row["old_reference_error"] / comparison["old_reference_error"]  # Form the original-reference error ratio.
    new_ratio = row["new_reference_error"] / comparison["new_reference_error"]  # Form the finer-reference error ratio.
    return {"method": row["name"], "baseline": comparison["name"], "old_ratio": old_ratio, "new_ratio": new_ratio, "old_gain_percent": 100.0 * (1.0 - old_ratio), "new_gain_percent": 100.0 * (1.0 - new_ratio), "gain_change_percentage_points": 100.0 * (old_ratio - new_ratio), "old_advantage": old_ratio < 1.0, "new_advantage": new_ratio < 1.0}  # Preserve both the magnitude and any ranking reversal.
def check_case(args, seed):  # Perform one new reference solve and reevaluate all cached old and GPT candidates.
    case_dir = args.experiment / f"test_bearing_{seed}"  # Locate the original common-state experiment.
    gpt_dir = Path(str(args.gpt_prefix) + f"_{seed}")  # Locate the completed GPT replay for the same case.
    output = args.output / f"seed_{seed}"  # Keep the new reference and all reevaluation evidence together.
    output.mkdir(parents=True, exist_ok=True)  # Prepare the isolated resolution-check directory.
    case = json.loads((case_dir / "case.json").read_text())  # Read the completed original candidates only for evaluation.
    gpt = json.loads((gpt_dir / "summary.json").read_text())  # Read the already frozen GPT and hybrid selections.
    frozen_path = gpt_dir / "predictions_before_solves.json"  # Identify the pre-solve decision evidence to preserve.
    source_paths = [case_dir / "case.json", case_dir / "reference.npz", gpt_dir / "summary.json", gpt_dir / "originaldecision.json", frozen_path]  # List existing scientific inputs whose byte identities will be recorded.
    old_entries = list(case["entries"])  # Preserve the full twelve-action original comparison without pruning.
    gpt_entries = list(gpt["candidates"])  # Preserve all three GPT alternatives without altering their choices.
    tagged = [("original", entry, case_dir / f"post_{entry['name']}.npz") for entry in old_entries] + [("gpt", entry, gpt_dir / entry["post_artifact"]) for entry in gpt_entries]  # Map all fifteen cached solutions to their recorded provenance.
    source_paths.extend(path for _, _, path in tagged)  # Include every reused native candidate field in the immutable input receipt.
    hashes_before = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}  # Record input bytes before the new reference solve.
    problem = case_problem("bearing", seed)  # Recreate exactly the original seeded physical problem.
    old_reference = restore_post(case_dir / "reference.npz", problem)  # Rebuild the old h0/3 reference from its cached native displacement.
    start = time.perf_counter()  # Count one case's new reference solve and post hoc comparison cost.
    runner = FemRunner(problem, output, keep_files=True)  # Retain raw INP, FRD, LOG, and native numerical records.
    fine_mesh = generate_uniform(problem, problem.h0 / REFERENCE_FACTOR)  # Generate exactly the requested method-independent h0/4 reference.
    fine_post, fine_record = runner.solve_mesh(fine_mesh, method="reference_h0_over_4", stage="fine")  # Perform the sole new native solve for this case.
    fine_path = output / "reference_h0_over_4.npz"  # Name the retained finer reference-field artifact.
    np.savez_compressed(fine_path, nodes=fine_mesh.nodes, cells=fine_mesh.cells, u=fine_post.u, stress=fine_post.stress, strain=fine_post.strain, vm=fine_post.vm_elem, energy=fine_post.energy_elem)  # Preserve authentic mesh and field arrays for independent verification.
    runner.dump()  # Save the single actual native solver record separately from candidate reevaluation.
    old_to_new, old_to_new_miss = reference_error(problem, old_reference, fine_post)  # Measure reference-field change on the same new integration mesh.
    rows = []  # Collect matched old and new errors for every unchanged cached candidate.
    for origin, entry, path in tagged:  # Evaluate all existing alternatives without generating or solving another candidate.
        post = restore_post(path, problem)  # Recompute the candidate field solely from its saved native displacement.
        old_error, old_miss = reference_error(problem, post, old_reference)  # Reproduce the original-reference metric through the identical integration function.
        new_error, new_miss = reference_error(problem, post, fine_post)  # Use the same finer reference and quadrature for every candidate.
        recorded = float(entry["reference_error"])  # Preserve the previously reported metric for consistency evidence.
        rows.append({"name": entry["name"], "origin": origin, "n_equations": entry["record"]["n_equations"], "old_reference_error_recorded": recorded, "old_reference_error": old_error, "old_metric_reproduction_absolute_delta": abs(recorded - old_error), "new_reference_error": new_error, "new_to_old_error_ratio": new_error / old_error, "old_mapping_miss": old_miss, "new_mapping_miss": new_miss, "cached_post": str(path.relative_to(ROOT))})  # Store measured sensitivity and all point-location approximation rates.
    by_name = {row["name"]: row for row in rows}  # Resolve frozen method names without redefining either GPT selection.
    old_best_dorfler = min((row for row in rows if row["name"].startswith("dorfler_")), key=lambda row: row["old_reference_error"])  # Keep the old-reference best Dörfler choice fixed for the requested comparison.
    new_best_dorfler = min((row for row in rows if row["name"].startswith("dorfler_")), key=lambda row: row["new_reference_error"])  # Report the stronger new-reference Dörfler envelope as a separately labeled sensitivity result.
    focus = {"gpt_primary": by_name[gpt["primary_gpt"]], "gpt_hybrid": by_name[gpt["hybrid_world_model"]], "old_best_dorfler": old_best_dorfler, "analytic_density": by_name["analytic_density"]}  # Retain the four specifically requested methods and frozen identities.
    comparisons = {}  # Report ratios against both requested classical comparators and the updated Dörfler envelope.
    for label in ("gpt_primary", "gpt_hybrid"):  # Evaluate the two already sealed deployment decisions without selecting a new GPT winner.
        comparisons[label] = {"versus_old_best_dorfler": summarize_choice(focus[label], old_best_dorfler), "versus_new_best_dorfler": summarize_choice(focus[label], new_best_dorfler), "versus_analytic_density": summarize_choice(focus[label], focus["analytic_density"])}  # Preserve all meaningful matched-reference comparisons.
    hashes_after = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}  # Check that all reused source decisions and candidate fields remain byte-identical.
    result = {"seed": seed, "family": "bearing", "new_solver_calls": len(runner.records), "candidate_solver_calls": 0, "old_reference_factor": 3.0, "new_reference_factor": REFERENCE_FACTOR, "new_reference_h": problem.h0 / REFERENCE_FACTOR, "old_reference": case["reference"], "new_reference": asdict(fine_record), "reference_field_change_error": old_to_new, "reference_field_change_mapping_miss": old_to_new_miss, "primary_gpt_frozen": gpt["primary_gpt"], "hybrid_world_model_frozen": gpt["hybrid_world_model"], "old_best_dorfler": old_best_dorfler["name"], "new_best_dorfler": new_best_dorfler["name"], "focus_methods": focus, "comparisons": comparisons, "all_methods": rows, "inputs_unchanged": hashes_before == hashes_after, "source_hashes": hashes_before, "fine_reference_sha256": sha256(fine_path), "wall_s": time.perf_counter() - start}  # Preserve complete numerical evidence without claiming exact-reference convergence.
    write_json(output / "report.json", result)  # Save the complete per-case resolution check.
    print(json.dumps({"seed": seed, "new_reference_equations": fine_record.n_equations, "reference_field_change_error": old_to_new, "comparisons": comparisons}), flush=True)  # Stream concise measured findings immediately after each of the three cases.
    return result  # Return this case for the final fixed-sample report.
def main():  # Execute only the three authorized new reference solves.
    parser = argparse.ArgumentParser(description=__doc__)  # Provide reproducible input and output artifact controls.
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/visual_wm_probe")  # Reuse the completed original twelve-method cases.
    parser.add_argument("--gpt-prefix", type=Path, default=ROOT / "runs/gpt_direct")  # Reuse the completed three-candidate GPT cases.
    parser.add_argument("--output", type=Path, default=ROOT / "runs/gpt_reference_check")  # Store all new evidence in a separate output tree.
    args = parser.parse_args()  # Parse only artifact locations without exposing method or decision tuning.
    args.experiment, args.gpt_prefix, args.output = args.experiment.resolve(), args.gpt_prefix.resolve(), args.output.resolve()  # Normalize all artifact paths once.
    cases = []  # Accumulate completed case receipts without altering cached candidates.
    report = {"scope": "fixed GPT and hybrid decisions reevaluated against h0/4 common references", "seeds": list(SEEDS), "old_reference_factor": 3.0, "new_reference_factor": REFERENCE_FACTOR, "new_solver_calls": 0, "candidate_solver_calls": 0, "method_choices_changed": False, "metric": "unchanged reference_error fine-cell-centroid stress-energy integration", "interpretation_limit": "one reference-resolution sensitivity step; no exact-reference convergence claim", "cases": cases}  # State the scientific scope before the new solves begin.
    write_json(args.output / "report.json", report)  # Save the fixed reevaluation protocol before obtaining finer results.
    for seed in SEEDS:  # Run exactly the three existing held-out cases in their original order.
        cases.append(check_case(args, seed))  # Add one new reference solve and reuse all fifteen cached candidate fields.
        report["new_solver_calls"] = sum(case["new_solver_calls"] for case in cases)  # Count genuine additional native solves explicitly.
        write_json(args.output / "report.json", report)  # Checkpoint the complete report after each costly new reference.
    report["summary"] = {label: {baseline: {"old_wins": sum(case["comparisons"][label][baseline]["old_advantage"] for case in cases), "new_wins": sum(case["comparisons"][label][baseline]["new_advantage"] for case in cases), "old_mean_gain_percent": float(np.mean([case["comparisons"][label][baseline]["old_gain_percent"] for case in cases])), "new_mean_gain_percent": float(np.mean([case["comparisons"][label][baseline]["new_gain_percent"] for case in cases]))} for baseline in ("versus_old_best_dorfler", "versus_new_best_dorfler", "versus_analytic_density")} for label in ("gpt_primary", "gpt_hybrid")}  # Summarize the fixed three-case sample while preserving per-case results above.
    report["all_source_inputs_unchanged"] = all(case["inputs_unchanged"] for case in cases)  # Record that the sensitivity study did not modify candidate choices or numerical fields.
    write_json(args.output / "report.json", report)  # Save the completed exact-three-solve comparison.
    print(json.dumps({"report": str(args.output / "report.json"), "new_solver_calls": report["new_solver_calls"], "summary": report["summary"]}, indent=2), flush=True)  # Return the final concise scientific findings.
if __name__ == "__main__":  # Execute only when explicitly launched as this script.
    main()  # Run the fixed reference-resolution sensitivity check.
