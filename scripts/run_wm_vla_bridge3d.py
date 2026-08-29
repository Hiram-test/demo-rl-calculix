#!/usr/bin/env python3  # Execute the benchmark with the active Python interpreter.
# Run independent WM-VLA, exact Dörfler, and local-prediction trajectories on a 3-D bridge pier cap.  # Script purpose.
from __future__ import annotations  # Enable postponed annotations for helper return types.
import argparse  # Parse a reproducible command-line experiment contract.
import json  # Write complete machine-readable records and release gates.
import shutil  # Reuse one independently computed reference across method work directories.
import sys  # Add the repository root when the script is executed directly.
from dataclasses import asdict  # Serialize typed result and gate configurations.
from pathlib import Path  # Manage deterministic experiment artifacts.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Make local repository modules importable without installation.

from visionamr.baselines.dorfler import run_dorfler  # Run the exact element-level classical AFEM baseline independently.
from visionamr.baselines.local_prediction import run_local_prediction  # Run the independent strong few-shot classical comparator.
from visionamr.bridge_scenarios import make_bridge_pier_cap  # Build the medium-complexity three-dimensional bridge component.
from visionamr.experiment import FemRunner  # Route every CalculiX call through honest solve accounting.
from visionamr.vla.dominance import DominanceConfig, evaluate_dorfler_floor  # Enforce pre-registered non-inferiority gates.
from visionamr.vla.partition import ScriptedVisionPartitioner  # Provide a deterministic solve-free stand-in for one VLM drawing.
from visionamr.vla.pipeline_world import WorldVLAConfig, run_world_vla  # Run the new multi-step world-model-guided controller.
from visionamr.vla.world_model import WorldPlannerConfig  # Configure finite-horizon counterfactual search.


_EQ_PER_ELEM_3D = 0.62  # Preserve the repository's observed three-dimensional equation-to-element conversion seed.


def _record_dict(record) -> dict:  # Serialize one SolveRecord with only JSON-compatible values.
    payload = asdict(record)  # Convert the typed record recursively.
    payload["method"] = str(record.method)  # Preserve exact method provenance explicitly.
    return payload  # Return the complete record.


def _copy_reference(source: Path, destination: Path) -> None:  # Reuse an identical reference without mixing method trajectories.
    destination.mkdir(parents=True, exist_ok=True)  # Create the independent method work directory.
    shutil.copy2(source, destination / "reference.json")  # Copy the exact reference artifact and metadata.


def _print_table(records: list, budget: int) -> None:  # Print a compact real-solve comparison for human inspection.
    header = f"{'method':24s} {'k':>2s} {'stage':18s} {'N_eq':>9s} {'N/B':>6s} {'e_E':>9s} {'e_Q':>9s} {'sum_eta2':>12s}"  # Build a stable table header.
    print(header, flush=True)  # Emit the header immediately in CI logs.
    for record in records:  # Traverse every real solve in independent method order.
        energy = float("nan") if record.e_energy is None else float(record.e_energy)  # Normalize unavailable energy error.
        qoi = float("nan") if record.e_qoi is None else float(record.e_qoi)  # Normalize unavailable QoI error.
        eta = float(record.extra.get("sum_eta2", float("nan")))  # Read the measured estimator total.
        print(f"{record.method:24s} {record.solve_index:2d} {record.stage:18s} {record.n_equations:9d} {record.n_equations / max(float(budget), 1.0):6.2f} {energy:9.5f} {qoi:9.5f} {eta:12.5g}", flush=True)  # Emit one measured row.


def main() -> int:  # Execute the complete independent benchmark and release gate.
    parser = argparse.ArgumentParser(description="World-model-guided VLA on a medium 3-D bridge pier cap")  # Define the experiment interface.
    parser.add_argument("--out", default="artifacts/wm_vla_bridge3d", help="artifact directory")  # Select the output root.
    parser.add_argument("--budget", type=int, default=18000, help="free-equation hard cap")  # Set the shared method resource budget.
    parser.add_argument("--max-solves", type=int, default=6, help="real CalculiX solves per adaptive method")  # Set the common real-solve horizon.
    parser.add_argument("--horizon", type=int, default=3, help="world-model counterfactual planning horizon")  # Set cheap internal lookahead depth.
    parser.add_argument("--beam-width", type=int, default=24, help="world-model beam width")  # Set cheap internal search breadth.
    parser.add_argument("--guard", choices=("off", "dorfler_region_candidate"), default="off", help="disclosed WM action guard mode")  # Select pure or guarded action space.
    parser.add_argument("--theta", type=float, default=0.50, help="exact Dörfler bulk parameter")  # Set the classical baseline parameter.
    parser.add_argument("--skip-local-prediction", action="store_true", help="omit the independent LP comparator")  # Allow a faster WM-versus-Dörfler run.
    parser.add_argument("--strict-gate", action="store_true", help="return exit status 2 when the Dörfler floor fails")  # Make scientific release gating machine enforceable.
    parser.add_argument("--keep-files", action="store_true", help="retain CalculiX input and result files")  # Preserve solver files for deep debugging.
    args = parser.parse_args()  # Parse the complete reproducible experiment contract.
    output = Path(args.out)  # Normalize the output root.
    output.mkdir(parents=True, exist_ok=True)  # Create the artifact directory.
    problem = make_bridge_pier_cap()  # Build the canonical medium-complexity three-dimensional bridge component.
    shared = output / "reference"  # Isolate the independently computed reference solution.
    reference_runner = FemRunner(problem, shared, keep_files=bool(args.keep_files), ccx_timeout=2400.0)  # Create the reference runner.
    reference = reference_runner.ensure_reference()  # Solve or load the graded independent reference exactly once.
    reference_path = shared / "reference.json"  # Identify the reusable reference artifact.
    wm_work = output / "wm_vla"  # Isolate the pure or guarded WM-VLA trajectory.
    dorfler_work = output / "dorfler"  # Isolate the exact Dörfler trajectory.
    lp_work = output / "local_prediction"  # Isolate the local-prediction trajectory.
    _copy_reference(reference_path, wm_work)  # Reuse the identical reference for WM-VLA.
    _copy_reference(reference_path, dorfler_work)  # Reuse the identical reference for Dörfler.
    if not args.skip_local_prediction:  # Prepare the independent LP comparator only when requested.
        _copy_reference(reference_path, lp_work)  # Reuse the identical reference for LP.
    planner = WorldPlannerConfig(horizon=int(args.horizon), beam_width=int(args.beam_width), dorfler_theta=float(args.theta))  # Configure bounded multi-step counterfactual planning.
    wm_config = WorldVLAConfig(n_eq_budget=int(args.budget), max_solves=int(args.max_solves), guard_mode=str(args.guard), planner=planner, model_checkpoint_out=str(output / "world_model.json"))  # Configure the accountable real feedback loop.
    wm_runner = FemRunner(problem, wm_work, keep_files=bool(args.keep_files), ccx_timeout=2400.0)  # Create the independent WM-VLA runner.
    wm_result = run_world_vla(wm_runner, ScriptedVisionPartitioner(), wm_config, method="wm_vla" if args.guard == "off" else "wm_vla_guarded")  # Execute one visual call plus multi-step model-based control.
    dorfler_runner = FemRunner(problem, dorfler_work, keep_files=bool(args.keep_files), ccx_timeout=2400.0)  # Create the independent exact Dörfler runner.
    run_dorfler(dorfler_runner, theta=float(args.theta), max_rounds=max(int(args.max_solves) - 1, 0), n_eq_cap=int(args.budget), method="dorfler_zz")  # Execute the exact element-level classical loop.
    all_records = list(wm_runner.records) + list(dorfler_runner.records)  # Combine independent real records only for reporting.
    lp_result = None  # Initialize the optional LP result summary.
    if not args.skip_local_prediction:  # Execute LP independently when requested.
        lp_runner = FemRunner(problem, lp_work, keep_files=bool(args.keep_files), ccx_timeout=2400.0)  # Create the independent LP runner.
        element_budget = max(int(round(float(args.budget) / _EQ_PER_ELEM_3D)), 1)  # Convert the equation budget to LP's element target.
        run_local_prediction(lp_runner, budgets=[element_budget], rounds=max(int(args.max_solves) - 1, 0), method="local_prediction")  # Execute repeated fresh-indicator LP corrections.
        all_records.extend(lp_runner.records)  # Add LP records only after its independent run completes.
        lp_result = {"element_budget": element_budget, "solves": len(lp_runner.records)}  # Preserve LP protocol details.
    wm_method = "wm_vla" if args.guard == "off" else "wm_vla_guarded"  # Preserve the exact method label used above.
    gate_config = DominanceConfig()  # Freeze the pre-registered Dörfler-floor tolerances.
    gate = evaluate_dorfler_floor(all_records, wm_method, "dorfler_zz", int(args.budget), gate_config)  # Evaluate matched-solve non-inferiority.
    _print_table(all_records, int(args.budget))  # Print all measured trajectories.
    print(json.dumps({"dorfler_floor": gate}, indent=2), flush=True)  # Print the complete release gate in CI logs.
    payload = {  # Build the complete experiment artifact.
        "protocol": "independent_wm_vla_vs_dorfler_vs_local_prediction",  # State the scientific comparison protocol.
        "problem": problem.instance_id,  # Preserve exact scenario parameters through the instance hash.
        "problem_params": problem.params,  # Preserve the full human-readable scenario definition.
        "reference": asdict(reference),  # Preserve independent reference metadata.
        "budget": int(args.budget),  # Preserve the common hard equation cap.
        "max_solves": int(args.max_solves),  # Preserve the common real-solve horizon.
        "world_planning": {"horizon": int(args.horizon), "beam_width": int(args.beam_width), "guard": str(args.guard)},  # Preserve cheap internal planning limits.
        "wm_result": asdict(wm_result),  # Preserve the complete WM-VLA control trace.
        "local_prediction": lp_result,  # Preserve the optional LP protocol summary.
        "dorfler_floor": gate,  # Preserve the pre-registered release decision.
        "records": [_record_dict(record) for record in all_records],  # Preserve every measured real solve.
        "purity": {"wm_imports_local_prediction": False, "methods_share_state": False, "shared_object_only": "reference solution"},  # Lock method independence explicitly.
    }  # Finish the complete artifact payload.
    result_path = output / "comparison.json"  # Select the canonical artifact path.
    result_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")  # Write the human-auditable experiment record.
    print(f"wrote {result_path}", flush=True)  # Report the exact artifact location.
    if bool(args.strict_gate) and not bool(gate["pass"]):  # Enforce the scientific release criterion when requested.
        return 2  # Return a distinct non-inferiority failure status.
    return 0  # Return success when execution completed and any requested gate passed.


if __name__ == "__main__":  # Execute only when invoked as a script.
    raise SystemExit(main())  # Propagate the machine-readable benchmark exit status.
