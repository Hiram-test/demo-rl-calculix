#!/usr/bin/env python3  # Execute the transition campaign with the active Python interpreter.
"""Build a reusable world-model transition library from bridge-family solves."""  # Describe the script purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
import argparse  # Import command-line parsing.
from dataclasses import asdict  # Import solve-record serialization.
import json  # Import transparent campaign output.
import os  # Import deterministic thread environment settings.
from pathlib import Path  # Import filesystem path handling.
import sys  # Import repository path injection and exit codes.
import numpy as np  # Import deterministic random-number generation.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository root.
sys.path.insert(0, str(ROOT))  # Make the local package importable without installation.
os.environ.setdefault("OMP_NUM_THREADS", "2")  # Bound native solver threads for reproducibility.
from visionamr.bridge_cases import sample_box_girder_diaphragm  # Import the bridge-family sampler.
from visionamr.experiment import FemRunner  # Import honest solve accounting.
from visionamr.vla.partition import ScriptedVisionPartitioner  # Import the solve-free semantic visual head.
from visionamr.vla.planner import PlannerConfig  # Import the finite-horizon exploration settings.
from visionamr.vla.world_model import ResidualWorldModel, WorldModelConfig  # Import the reusable transition model.
from visionamr.vla.world_pipeline import WorldVLAConfig, run_world_model_vla  # Import the real multi-step collector.

def main() -> int:  # Parse the campaign, run real transitions, and persist the model.
    parser = argparse.ArgumentParser()  # Create the command-line interface.
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "world_model_library")  # Set the campaign evidence directory.
    parser.add_argument("--library", type=Path, default=None)  # Set an optional existing or target model snapshot.
    parser.add_argument("--instances", type=int, default=6)  # Set the number of randomized bridge components.
    parser.add_argument("--seed-start", type=int, default=12000)  # Set the first deterministic family seed.
    parser.add_argument("--solves-per-instance", type=int, default=4)  # Set the real transition depth per component.
    parser.add_argument("--budget", type=int, default=120000)  # Set the equation cap during safe transition acquisition.
    parser.add_argument("--horizon", type=int, default=3)  # Set the acquisition planner horizon.
    parser.add_argument("--beam-width", type=int, default=16)  # Set the bounded acquisition beam.
    parser.add_argument("--theta", type=float, default=0.5)  # Set the exact Dörfler bulk parameter.
    args = parser.parse_args()  # Parse all command-line arguments.
    if args.instances <= 0:  # Require at least one bridge component.
        parser.error("--instances must be positive")  # Reject an empty transition campaign.
    if args.solves_per_instance < 2:  # Require at least one observed transition after the common probe.
        parser.error("--solves-per-instance must be at least two")  # Reject a campaign with no learnable transition.
    args.output.mkdir(parents=True, exist_ok=True)  # Create the campaign evidence directory.
    library_path = args.library or (args.output / "transition_library.json")  # Select the reusable transition-library path.
    model = ResidualWorldModel.load(library_path) if library_path.exists() else ResidualWorldModel(WorldModelConfig())  # Continue an existing library or start from the physics prior.
    initial_samples = model.sample_count  # Record the pre-campaign evidence count.
    rows: list[dict] = []  # Allocate one transparent summary row per bridge instance.
    for offset in range(args.instances):  # Execute the requested randomized bridge components.
        seed = args.seed_start + offset  # Derive the reproducible family seed.
        rng = np.random.default_rng(seed)  # Create the deterministic sampler.
        problem = sample_box_girder_diaphragm(rng)  # Draw one medium-complexity bridge component.
        workdir = args.output / f"train_{seed}"  # Isolate all solver files for the instance.
        runner = FemRunner(problem, workdir, ccx_timeout=1800.0)  # Create honest real-solve accounting.
        planner = PlannerConfig(horizon=args.horizon, beam_width=args.beam_width, theta=args.theta, uncertainty_limit=1.0, failure_limit=0.95, min_relative_gain=-1.0, resource_weight=0.08, budget_safety=0.95)  # Permit bounded proactive exploration while retaining Dörfler dominance and resource certification.
        config = WorldVLAConfig(n_eq_budget=args.budget, max_solves=args.solves_per_instance, theta=args.theta, require_reference=False, early_stop=False, planner=planner, model=model.config)  # Configure a transition-acquisition trajectory without an expensive reference solve.
        before = model.sample_count  # Record the model evidence before this component.
        result = run_world_model_vla(runner, ScriptedVisionPartitioner(), config, model=model, method="world_model_library")  # Execute the real action-conditioned acquisition loop.
        after = model.sample_count  # Record the evidence gained from this component.
        records = [record for record in runner.records if record.method == "world_model_library"]  # Isolate counted acquisition solves.
        runner.dump(workdir / "records.json")  # Persist every real solve and action audit.
        model.save(library_path)  # Checkpoint the reusable transition library after every component.
        rows.append({"seed": seed, "instance_id": problem.instance_id, "parameters": problem.params, "solves": result.solves, "proactive_actions": result.proactive_actions, "dorfler_fallbacks": result.dorfler_fallbacks, "samples_before": before, "samples_after": after, "samples_added": after - before, "stopped_by": result.stopped_by, "best_estimator": result.best_estimator, "records": [asdict(record) for record in records]})  # Store a complete per-instance campaign record.
        print(f"[{offset + 1}/{args.instances}] seed={seed} solves={result.solves} proactive={result.proactive_actions} samples={after}")  # Report bounded progress to the terminal.
    payload = {"schema": "visionamr-world-model-library-campaign-v1", "family": "box_girder_diaphragm", "scientific_boundary": {"reads_local_prediction": False, "all_actions_start_from_exact_dorfler": True, "all_proactive_targets_are_nodewise_no_coarser_than_dorfler": all(item["target_dominance_verified"] for row in rows for record in row["records"] for item in ([record.get("extra", {}).get("world_model_plan")] if record.get("extra", {}).get("world_model_plan") else []))}, "configuration": {"instances": args.instances, "seed_start": args.seed_start, "solves_per_instance": args.solves_per_instance, "equation_budget": args.budget, "horizon": args.horizon, "beam_width": args.beam_width, "theta": args.theta}, "library": str(library_path), "samples_before": initial_samples, "samples_after": model.sample_count, "samples_added": model.sample_count - initial_samples, "instances": rows}  # Assemble the complete transition-campaign evidence.
    summary_path = args.output / "campaign_summary.json"  # Select the campaign summary path.
    summary_path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")  # Persist the transparent campaign summary.
    print(json.dumps({"summary": str(summary_path), "library": str(library_path), "samples_added": model.sample_count - initial_samples}, indent=1))  # Report the final model artifact and evidence count.
    return 0  # Report successful campaign completion.

if __name__ == "__main__":  # Execute only when launched as a script.
    raise SystemExit(main())  # Propagate the campaign status to CI or the shell.
