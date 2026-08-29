#!/usr/bin/env python3  # Execute the native closure check with the active Python interpreter.
# Run two real Gmsh-CalculiX solves without constructing the expensive benchmark reference.  # Script purpose.
from __future__ import annotations  # Enable postponed annotations for the smoke-only runner override.
import argparse  # Parse an explicit resource and artifact contract.
import json  # Write a machine-readable native closure artifact.
import sys  # Make the repository package importable when the script is run directly.
from dataclasses import asdict  # Serialize the typed controller result and solve records.
from pathlib import Path  # Manage deterministic output directories.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Add the repository root without installing the package.

from visionamr.bridge_scenarios import make_bridge_pier_cap  # Build the canonical medium-complexity bridge component.
from visionamr.experiment import FemRunner  # Reuse the accountable CalculiX solve path.
from visionamr.vla.bridge_partition import BridgePierCapVisionPartitioner  # Supply true three-dimensional section-aware geometry markup.
from visionamr.vla.pipeline_world import WorldVLAConfig, run_world_vla  # Execute the real multi-step WM-VLA loop.
from visionamr.vla.world_model import WorldPlannerConfig  # Configure the bounded counterfactual planner.


class SmokeFemRunner(FemRunner):  # Disable only reference construction while preserving every native solve operation.
    def ensure_reference(self):  # Satisfy the pipeline hook without fabricating reference-relative errors.
        return None  # Leave self.reference unset so measured records expose e_energy and e_qoi as unavailable.


def main() -> int:  # Execute the native two-solve closure test.
    parser = argparse.ArgumentParser(description="Native WM-VLA bridge-component smoke test")  # Define the smoke interface.
    parser.add_argument("--out", default="artifacts/wm_vla_bridge3d_smoke", help="artifact directory")  # Select the artifact root.
    parser.add_argument("--budget", type=int, default=9000, help="free-equation hard cap")  # Set the exact mesh budget.
    parser.add_argument("--max-solves", type=int, default=2, help="real CalculiX solve cap")  # Keep the smoke test short but action conditioned.
    parser.add_argument("--horizon", type=int, default=2, help="counterfactual planning horizon")  # Exercise multi-step prediction cheaply.
    parser.add_argument("--beam-width", type=int, default=8, help="counterfactual beam width")  # Bound the smoke search breadth.
    args = parser.parse_args()  # Parse the complete deterministic contract.
    output = Path(args.out)  # Normalize the artifact path.
    output.mkdir(parents=True, exist_ok=True)  # Create the artifact directory before solver execution.
    problem = make_bridge_pier_cap()  # Build the three-dimensional cap-column-bearing-duct geometry.
    runner = SmokeFemRunner(problem, output / "run", keep_files=False, ccx_timeout=900.0)  # Create an accountable native runner without a reference solve.
    planner = WorldPlannerConfig(horizon=int(args.horizon), beam_width=int(args.beam_width))  # Configure bounded model-based lookahead.
    config = WorldVLAConfig(n_eq_budget=int(args.budget), max_solves=int(args.max_solves), early_stop=False, guard_mode="off", planner=planner, model_checkpoint_out=str(output / "world_model.json"))  # Require the requested number of real feedback states.
    result = run_world_vla(runner, BridgePierCapVisionPartitioner(), config, method="wm_vla_native_smoke")  # Execute the pure WM-VLA closure on a true three-dimensional fixed partition.
    payload = {  # Assemble the complete native smoke artifact.
        "problem": problem.instance_id,  # Preserve the exact parameterized scenario identity.
        "budget": int(args.budget),  # Preserve the hard free-equation cap.
        "result": asdict(result),  # Preserve actions, states, stop reason, and measured estimator trajectory.
        "records": [asdict(record) for record in runner.records],  # Preserve every real CalculiX solve record.
        "reference_constructed": False,  # State explicitly why benchmark-relative errors are absent.
        "local_prediction_used": False,  # Lock the scientific separation from LP.
        "dorfler_used": False,  # Lock the pure controller path for this closure check.
        "partition": "bridge_pier_cap_section_aware",  # Disclose the true three-dimensional geometry partition.
    }  # Finish the artifact payload.
    path = output / "smoke.json"  # Select the canonical native closure artifact.
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")  # Persist the complete machine-readable evidence.
    print(json.dumps({"solves": len(runner.records), "equations": [record.n_equations for record in runner.records], "estimator": [record.extra.get("sum_eta2") for record in runner.records], "stopped_reason": result.stopped_reason}, indent=2), flush=True)  # Expose the measured closure summary in CI logs.
    print(f"wrote {path}", flush=True)  # Report the exact artifact location.
    return 0 if len(runner.records) == int(args.max_solves) and all(record.n_equations <= int(args.budget) for record in runner.records) else 3  # Fail if native feedback or budget enforcement is incomplete.


if __name__ == "__main__":  # Execute only when invoked as a script.
    raise SystemExit(main())  # Propagate the machine-readable closure status.
