from __future__ import annotations  # Enable compact annotations in the guarded bridge benchmark.
import argparse  # Expose cases, budgets, solve caps, output, and strict gating.
import json  # Read frozen references and write complete scientific evidence.
import shutil  # Reset only the explicitly requested scratch directory.
import sys  # Add the repository root for direct script execution.
from dataclasses import asdict  # Serialize solver and controller records.
from pathlib import Path  # Resolve repository and evidence paths portably.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Import this checkout rather than an installed package.
from visionamr.baselines.dorfler import run_dorfler  # Run the faithful standalone AFEM comparator.
from visionamr.experiment import FemRunner, Reference  # Reuse honest real-solve accounting and reference metrics.
from visionamr.geometry import PROBLEM_FACTORIES  # Select canonical medium-complexity three-dimensional bridge components.
from visionamr.vla.guarded_pipeline import run_guarded_world_vla  # Run the Dörfler-protected world-model controller.
from visionamr.vla.partition import ScriptedVisionPartitioner  # Provide reproducible geometry-only semantic drawings in CI.
from visionamr.vla.world_pipeline import WorldVLAConfig  # Configure common budgets and real-solve caps.
_DEFAULT_BUDGETS = {"bearing_hole": 8000, "deck_opening": 12000}  # Preserve established moderate free-equation budgets.
def _load_reference(repo: Path, case: str) -> Reference | None:  # Reuse frozen high-resolution evidence when available.
    path = repo / "results" / "campaign" / case / "canonical" / "reference.json"  # Follow the repository's canonical evidence layout.
    if not path.exists():  # Permit a new case to compute its own reference.
        return None  # Signal absence without inventing evidence.
    return Reference(**json.loads(path.read_text()))  # Reconstruct the exact recorded reference.
def _best(records: list, budget: int, tolerance: float = 1.02):  # Select the best actually budget-feasible offline evaluation point.
    feasible = [record for record in records if record.e_energy is not None and record.n_equations <= tolerance * budget]  # Enforce actual solver equations.
    pool = feasible if feasible else [record for record in records if record.e_energy is not None]  # Remain diagnostic when all points violate the cap.
    return min(pool, key=lambda record: (float(record.e_energy), float(record.e_qoi), int(record.n_equations)))  # Prioritize energy, then QoI and resource.
def _prefix(records: list, budget: int) -> list[dict]:  # Report the best-so-far curve at every real solve count.
    points = []  # Collect one point per prefix.
    for count in range(1, len(records) + 1):  # Traverse the observed solve trajectory.
        best = _best(records[:count], budget)  # Select the honest budget-feasible prefix point.
        points.append({"k": int(count), "e_energy": float(best.e_energy), "e_qoi": float(best.e_qoi), "n_equations": int(best.n_equations)})  # Store the matched-solve comparison point.
    return points  # Return the full trajectory.
def _beneficial_guarded_actions(records: list) -> list[dict]:  # Identify accepted world bonuses that improved both deployable and offline metrics.
    beneficial = []  # Collect transparent action evidence.
    for previous, current in zip(records, records[1:]):  # Compare each real transition with its parent.
        audit = current.extra.get("tool_audit", {})  # Read exact numerical execution evidence.
        plan = current.extra.get("world_plan", {})  # Read planner provenance.
        if not bool(audit.get("world_bonus_accepted", False)):  # Exclude guarded budget fallbacks.
            continue  # Preserve only real model interventions.
        eta_improved = float(current.extra.get("sum_eta2", float("inf"))) < float(previous.extra.get("sum_eta2", float("inf")))  # Require deployable estimator reduction.
        energy_improved = float(current.e_energy) < float(previous.e_energy)  # Require independent reference-metric improvement.
        if eta_improved and energy_improved:  # Accept only dual-confirmed gains.
            beneficial.append({"solve": int(current.solve_index), "action_id": plan.get("action_id"), "selected_by": plan.get("selected_by"), "energy_before": float(previous.e_energy), "energy_after": float(current.e_energy), "eta2_before": float(previous.extra.get("sum_eta2")), "eta2_after": float(current.extra.get("sum_eta2")), "accepted_bonus_factor": audit.get("accepted_bonus_factor"), "bonus_regions": audit.get("bonus_regions", [])})  # Preserve the full evidence.
    return beneficial  # Return all genuinely useful world-model actions.
def _run_case(repo: Path, root: Path, case: str, budget: int, max_solves: int) -> dict:  # Execute one guarded VLA versus standalone Dörfler comparison.
    if case not in PROBLEM_FACTORIES:  # Reject unsupported or misspelled cases before filesystem mutation.
        raise KeyError(f"unknown problem {case!r}")  # Fail with an exact case error.
    problem = PROBLEM_FACTORIES[case]()  # Build the canonical bridge component.
    case_root = root / case  # Isolate scratch and evidence by geometry.
    if case_root.exists():  # Remove only the explicitly selected output directory.
        shutil.rmtree(case_root)  # Prevent stale records from contaminating the gate.
    case_root.mkdir(parents=True, exist_ok=True)  # Create a clean evidence root.
    reference = _load_reference(repo, case)  # Load the frozen reference when present.
    world_runner = FemRunner(problem, case_root / "world", ccx_timeout=1800.0)  # Create an independently counted world-model runner.
    if reference is not None:  # Reuse frozen evidence without a new reference solve.
        world_runner.reference = reference  # Attach the exact objective definition.
    world_config = WorldVLAConfig(n_eq_budget=int(budget), max_solves=int(max_solves), min_solves=min(3, int(max_solves)), early_stop=False, allow_split=False)  # Disable oracle stopping and unstable graph growth.
    world_result = run_guarded_world_vla(world_runner, ScriptedVisionPartitioner(), world_config, method="world_vla_guarded")  # Execute protected multi-step control.
    world_records = [record for record in world_runner.records if record.method == "world_vla_guarded"]  # Isolate this method's real solves.
    dorfler_runner = FemRunner(problem, case_root / "dorfler", ccx_timeout=1800.0)  # Create an independently counted AFEM runner.
    dorfler_runner.reference = world_runner.reference  # Use the identical frozen or freshly computed reference.
    run_dorfler(dorfler_runner, theta=0.50, max_rounds=max(int(max_solves) - 1, 0), n_eq_cap=int(budget), gradation=0.90, method="dorfler_zz", require_reference=True)  # Run faithful iterative Dörfler under the same solve cap.
    dorfler_records = [record for record in dorfler_runner.records if record.method == "dorfler_zz"]  # Isolate baseline solves.
    world_best = _best(world_records, budget)  # Select the guarded VLA deliverable.
    dorfler_best = _best(dorfler_records, budget)  # Select the standalone AFEM deliverable.
    beneficial = _beneficial_guarded_actions(world_records)  # Verify that a model-selected bonus actually helped.
    energy_ratio = float(world_best.e_energy) / max(float(dorfler_best.e_energy), 1.0e-12)  # Measure primary noninferiority.
    qoi_ratio = float(world_best.e_qoi) / max(float(dorfler_best.e_qoi), 1.0e-12)  # Measure engineering-response protection.
    gates = {"world_not_weaker_energy": bool(energy_ratio <= 1.00 + 1.0e-12), "qoi_protected": bool(qoi_ratio <= 1.10 or float(world_best.e_qoi) <= 0.01), "budget_ok": bool(world_best.n_equations <= 1.02 * budget), "guarded_world_action_used": bool(world_result.world_actions > 0), "guarded_world_action_beneficial": bool(len(beneficial) > 0), "no_reference_error_in_control": bool(all(record.extra.get("control_uses_reference_error") is False for record in world_records))}  # Define independent scientific release gates.
    payload = {"case": case, "problem": problem.instance_id, "budget": int(budget), "max_solves": int(max_solves), "world_result": asdict(world_result), "world_best": asdict(world_best), "world_last": asdict(world_records[-1]), "dorfler_best": asdict(dorfler_best), "dorfler_last": asdict(dorfler_records[-1]), "world_prefix": _prefix(world_records, budget), "dorfler_prefix": _prefix(dorfler_records, budget), "beneficial_guarded_actions": beneficial, "energy_ratio_world_over_dorfler": energy_ratio, "qoi_ratio_world_over_dorfler": qoi_ratio, "cumulative_equations_world": int(sum(record.n_equations for record in world_records)), "cumulative_equations_dorfler": int(sum(record.n_equations for record in dorfler_records)), "solver_wall_world": float(sum(record.wall_s for record in world_records)), "solver_wall_dorfler": float(sum(record.wall_s for record in dorfler_records)), "gates": gates}  # Assemble complete case evidence.
    (case_root / "comparison.json").write_text(json.dumps(payload, indent=2, default=str))  # Persist the case verdict and trace.
    world_runner.dump(case_root / "world_records.json")  # Persist every protected VLA solve.
    dorfler_runner.dump(case_root / "dorfler_records.json")  # Persist every standalone baseline solve.
    print(f"{case}: guarded WM e_E={world_best.e_energy:.6f} N={world_best.n_equations} | Dörfler e_E={dorfler_best.e_energy:.6f} N={dorfler_best.n_equations} | ratio={energy_ratio:.4f} | accepted={world_result.world_actions} beneficial={len(beneficial)}", flush=True)  # Print the decisive result promptly.
    return payload  # Return case evidence to the suite.
def main() -> int:  # Parse the command line and execute the strict bridge gate.
    parser = argparse.ArgumentParser()  # Create the CLI.
    parser.add_argument("--cases", default="bearing_hole,deck_opening")  # Select medium-complexity topology-sensitive bridge components.
    parser.add_argument("--budget", type=int, default=0)  # Override all per-case equation budgets when positive.
    parser.add_argument("--max-solves", type=int, default=6)  # Allow genuine multi-step control under a matched cap.
    parser.add_argument("--out", default="/tmp/guarded_world_model_vla_bridge")  # Select scratch and evidence output.
    parser.add_argument("--strict", action="store_true")  # Fail CI when any preregistered scientific gate fails.
    args = parser.parse_args()  # Parse user input.
    repo = Path(__file__).resolve().parents[1]  # Resolve the checkout root.
    root = Path(args.out)  # Resolve the requested evidence root.
    root.mkdir(parents=True, exist_ok=True)  # Create the suite directory.
    cases = [item.strip() for item in args.cases.split(",") if item.strip()]  # Parse an ordered case list.
    results = []  # Collect complete case evidence.
    for case in cases:  # Execute each requested bridge component.
        budget = int(args.budget) if int(args.budget) > 0 else int(_DEFAULT_BUDGETS.get(case, 8000))  # Resolve the fair equation cap.
        results.append(_run_case(repo, root, case, budget, int(args.max_solves)))  # Run the full matched-solve comparison.
    required_case_gates = ("world_not_weaker_energy", "qoi_protected", "budget_ok", "guarded_world_action_used", "no_reference_error_in_control")  # Define per-case requirements.
    all_cases = all(all(bool(result["gates"][name]) for name in required_case_gates) for result in results)  # Require noninferiority and real protected action use in every case.
    any_beneficial = any(bool(result["gates"]["guarded_world_action_beneficial"]) for result in results)  # Require at least one independently confirmed world-model benefit.
    summary = {"protocol": "dorfler-backbone-world-model-vla-3d-v2", "cases": results, "suite_gates": {"all_cases_not_weaker_than_dorfler": bool(all_cases), "at_least_one_beneficial_guarded_world_action": bool(any_beneficial), "pass": bool(all_cases and any_beneficial)}}  # Assemble the strict suite verdict.
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))  # Persist all release evidence.
    print(json.dumps(summary["suite_gates"], indent=2), flush=True)  # Print the compact verdict.
    return 1 if args.strict and not summary["suite_gates"]["pass"] else 0  # Return an honest shell status.
if __name__ == "__main__":  # Execute only when called as a script.
    raise SystemExit(main())  # Propagate the strict gate status.
