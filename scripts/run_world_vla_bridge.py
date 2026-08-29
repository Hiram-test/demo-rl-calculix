from __future__ import annotations  # Enable compact annotations in the executable benchmark.
import argparse  # Expose cases, budgets, solve caps, output, and strict gating.
import json  # Read cached references and write auditable comparison evidence.
import shutil  # Reset only the requested scratch directory.
import sys  # Add the repository root when the script is executed directly.
from dataclasses import asdict  # Serialize result and solve records.
from pathlib import Path  # Resolve repository and output paths portably.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Import the checked-out repository rather than an installed copy.
from visionamr.baselines.dorfler import run_dorfler  # Run the faithful element-wise AFEM baseline.
from visionamr.experiment import FemRunner, Reference  # Reuse honest solve accounting and cached reference records.
from visionamr.geometry import PROBLEM_FACTORIES  # Select existing medium-complexity three-dimensional bridge components.
from visionamr.vla.partition import ScriptedVisionPartitioner  # Provide reproducible geometry-only semantic regions in CI.
from visionamr.vla.world_pipeline import WorldVLAConfig, run_world_vla  # Run the new multi-step world-model controller.
_DEFAULT_BUDGETS = {"bearing_hole": 8000, "deck_opening": 12000}  # Use established moderate three-dimensional equation budgets.
def _load_reference(repo: Path, case: str) -> Reference | None:  # Reuse frozen high-resolution evidence when available.
    path = repo / "results" / "campaign" / case / "canonical" / "reference.json"  # Follow the repository's canonical evidence layout.
    if not path.exists():  # Permit new cases to compute a reference normally.
        return None  # Signal absence without inventing data.
    return Reference(**json.loads(path.read_text()))  # Reconstruct the exact recorded reference.
def _best(records: list, budget: int, tolerance: float = 1.02):  # Select the best actually budget-feasible solve.
    feasible = [record for record in records if record.e_energy is not None and record.n_equations <= tolerance * budget]  # Filter by actual solver count.
    pool = feasible if feasible else [record for record in records if record.e_energy is not None]  # Remain diagnostic if every record violates the cap.
    return min(pool, key=lambda record: (float(record.e_energy), float(record.e_qoi), int(record.n_equations)))  # Use energy, QoI, then size.
def _prefix(records: list, budget: int) -> list[dict]:  # Report best-so-far accuracy at every real solve count.
    out = []  # Collect one point per prefix.
    for count in range(1, len(records) + 1):  # Traverse available solve counts.
        best = _best(records[:count], budget)  # Select the honest prefix deliverable.
        out.append({"k": int(count), "e_energy": float(best.e_energy), "e_qoi": float(best.e_qoi), "n_equations": int(best.n_equations)})  # Store the comparison point.
    return out  # Return the trajectory.
def _case(repo: Path, root: Path, case: str, budget: int, max_solves: int) -> dict:  # Run one world-model versus Dörfler experiment.
    if case not in PROBLEM_FACTORIES:  # Reject misspelled or unsupported problems.
        raise KeyError(f"unknown problem {case!r}")  # Fail before deleting or solving anything.
    problem = PROBLEM_FACTORIES[case]()  # Build the canonical bridge component.
    case_root = root / case  # Isolate all scratch and evidence by case.
    if case_root.exists():  # Remove only an explicitly chosen output directory.
        shutil.rmtree(case_root)  # Prevent stale records from contaminating the comparison.
    case_root.mkdir(parents=True, exist_ok=True)  # Create a clean evidence root.
    reference = _load_reference(repo, case)  # Prefer the repository's frozen reference.
    world_runner = FemRunner(problem, case_root / "world", ccx_timeout=1800.0)  # Create an independently counted world-model runner.
    if reference is not None:  # Reuse frozen evidence without a new reference solve.
        world_runner.reference = reference  # Attach the exact same objective definition.
    world_config = WorldVLAConfig(n_eq_budget=int(budget), max_solves=int(max_solves), min_solves=min(3, int(max_solves)))  # Configure genuine multi-step execution.
    world_result = run_world_vla(world_runner, ScriptedVisionPartitioner(), world_config, method="world_vla")  # Execute the semantic world-model loop.
    world_records = [record for record in world_runner.records if record.method == "world_vla"]  # Isolate method solves.
    dorfler_runner = FemRunner(problem, case_root / "dorfler", ccx_timeout=1800.0)  # Create an independently counted AFEM runner.
    dorfler_runner.reference = world_runner.reference  # Use the identical frozen or freshly computed reference.
    run_dorfler(dorfler_runner, theta=0.50, max_rounds=max(int(max_solves) - 1, 0), n_eq_cap=int(budget), gradation=0.90, method="dorfler_zz", require_reference=True)  # Run faithful iterative Dörfler under the same solve cap.
    dorfler_records = [record for record in dorfler_runner.records if record.method == "dorfler_zz"]  # Isolate baseline solves.
    world_best = _best(world_records, budget)  # Select the world-model deliverable.
    dorfler_best = _best(dorfler_records, budget)  # Select the baseline deliverable.
    energy_ratio = float(world_best.e_energy) / max(float(dorfler_best.e_energy), 1.0e-12)  # Measure final energy competitiveness.
    qoi_ratio = float(world_best.e_qoi) / max(float(dorfler_best.e_qoi), 1.0e-12)  # Measure final engineering-QoI competitiveness.
    world_not_weaker = bool(energy_ratio <= 1.00 + 1.0e-12)  # Enforce the requested Dörfler floor on the primary metric.
    qoi_protected = bool(qoi_ratio <= 1.10 or float(world_best.e_qoi) <= 0.01)  # Prevent a primary-metric win that damages engineering response.
    budget_ok = bool(world_best.n_equations <= 1.02 * budget)  # Enforce the actual solver count.
    payload = {  # Assemble one complete case result.
        "case": case,  # Record the geometry family.
        "problem": problem.instance_id,  # Record the exact parameter hash.
        "budget": int(budget),  # Record the hard cap.
        "max_solves": int(max_solves),  # Record the matched solve cap.
        "world_result": asdict(world_result),  # Record controller behavior.
        "world_best": asdict(world_best),  # Record the selected world-model solve.
        "dorfler_best": asdict(dorfler_best),  # Record the selected AFEM solve.
        "world_prefix": _prefix(world_records, budget),  # Record best-so-far world-model accuracy.
        "dorfler_prefix": _prefix(dorfler_records, budget),  # Record best-so-far Dörfler accuracy.
        "energy_ratio_world_over_dorfler": energy_ratio,  # Record the primary comparison.
        "qoi_ratio_world_over_dorfler": qoi_ratio,  # Record the QoI comparison.
        "cumulative_equations_world": int(sum(record.n_equations for record in world_records)),  # Report total linear-system scale.
        "cumulative_equations_dorfler": int(sum(record.n_equations for record in dorfler_records)),  # Report baseline total linear-system scale.
        "solver_wall_world": float(sum(record.wall_s for record in world_records)),  # Report actual CalculiX wall time.
        "solver_wall_dorfler": float(sum(record.wall_s for record in dorfler_records)),  # Report baseline CalculiX wall time.
        "gates": {"world_not_weaker_energy": world_not_weaker, "qoi_protected": qoi_protected, "budget_ok": budget_ok, "world_model_action_used": bool(world_result.world_actions > 0)},  # Record independent scientific gates.
    }  # Close the case payload.
    (case_root / "comparison.json").write_text(json.dumps(payload, indent=2, default=str))  # Persist the case evidence.
    world_runner.dump(case_root / "world_records.json")  # Persist every world-model solve.
    dorfler_runner.dump(case_root / "dorfler_records.json")  # Persist every baseline solve.
    print(f"{case}: WM e_E={world_best.e_energy:.6f} N={world_best.n_equations} | Dörfler e_E={dorfler_best.e_energy:.6f} N={dorfler_best.n_equations} | ratio={energy_ratio:.4f} | world_actions={world_result.world_actions} dorfler_fallbacks={world_result.dorfler_actions}", flush=True)  # Print the decisive result.
    return payload  # Return the case evidence to the suite.
def main() -> int:  # Parse the command line and execute the bridge gate.
    parser = argparse.ArgumentParser()  # Create the CLI.
    parser.add_argument("--cases", default="bearing_hole,deck_opening")  # Select medium-complexity topology-sensitive bridge components.
    parser.add_argument("--budget", type=int, default=0)  # Override all per-case defaults when positive.
    parser.add_argument("--max-solves", type=int, default=6)  # Match genuine multi-step solve caps.
    parser.add_argument("--out", default="/tmp/world_model_vla_bridge")  # Select scratch and evidence output.
    parser.add_argument("--strict", action="store_true")  # Fail CI when the preregistered release gates fail.
    args = parser.parse_args()  # Parse user input.
    repo = Path(__file__).resolve().parents[1]  # Resolve the checkout root.
    root = Path(args.out)  # Resolve the requested evidence directory.
    root.mkdir(parents=True, exist_ok=True)  # Create the suite directory.
    cases = [item.strip() for item in args.cases.split(",") if item.strip()]  # Parse a deterministic ordered case list.
    results = []  # Collect case evidence.
    for case in cases:  # Run every selected bridge component.
        budget = int(args.budget) if int(args.budget) > 0 else int(_DEFAULT_BUDGETS.get(case, 8000))  # Resolve the fair equation cap.
        results.append(_case(repo, root, case, budget, int(args.max_solves)))  # Execute the complete comparison.
    all_primary = all(bool(item["gates"]["world_not_weaker_energy"] and item["gates"]["qoi_protected"] and item["gates"]["budget_ok"]) for item in results)  # Require per-case Dörfler competitiveness.
    any_world_action = any(bool(item["gates"]["world_model_action_used"]) for item in results)  # Require actual use of the learned planner somewhere.
    summary = {"protocol": "world-model-vla-versus-dorfler-3d-v1", "cases": results, "suite_gates": {"all_cases_not_weaker_than_dorfler": all_primary, "world_model_advantage_path_exercised": any_world_action, "pass": bool(all_primary and any_world_action)}}  # Assemble the release verdict.
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))  # Persist the complete gate.
    print(json.dumps(summary["suite_gates"], indent=2), flush=True)  # Print the compact verdict.
    return 1 if args.strict and not summary["suite_gates"]["pass"] else 0  # Make strict CI fail honestly on scientific regression.
if __name__ == "__main__":  # Execute only when called as a script.
    raise SystemExit(main())  # Return a meaningful shell status.
