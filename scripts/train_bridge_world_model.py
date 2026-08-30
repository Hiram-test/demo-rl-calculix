#!/usr/bin/env python3  # Execute the frozen world-model acquisition campaign with the active repository interpreter.
"""Plan or run the train-only WMVLA-4WAY-P1 transition-library campaign."""  # Describe the command's strict pre-blind responsibility.
from __future__ import annotations  # Postpone annotation evaluation for compatible repository runtimes.
import argparse  # Parse only artifact locations, operational timeout, and solve-free dry-run selection.
import json  # Print strict machine-readable plan or completion evidence.
import os  # Freeze native thread counts before importing solver-facing modules.
from pathlib import Path  # Resolve repository and campaign paths portably.
import sys  # Import the checked-out package and propagate the command status.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository root independently from the launch directory.
sys.path.insert(0, str(ROOT))  # Import this checkout without requiring installation.
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Pin OpenMP-backed native work for reproducible acquisition costs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # Pin OpenBLAS-backed residual fitting for reproducibility.
os.environ.setdefault("MKL_NUM_THREADS", "1")  # Pin MKL-backed residual fitting for reproducibility.
from visionamr.vla.four_way_world_training import WorldTrainingConfig  # Import the immutable scientific acquisition configuration.
from visionamr.vla.four_way_world_training import build_training_plan  # Import the authenticated solve-free plan builder.
from visionamr.vla.four_way_world_training import train_world_model_transition_library  # Import the sole train-only execution entry point.

CAMPAIGN_ROOT = ROOT / "results" / "wm_vla_four_way_p1"  # Locate the frozen experiment result tree.
DEFAULT_MANIFEST = CAMPAIGN_ROOT / "protocol" / "case_manifest.json"  # Select the checksummed case design.
DEFAULT_PARTITIONS = CAMPAIGN_ROOT / "protocol" / "partitions"  # Select the shared WM/RL frozen partition registry.
DEFAULT_OUTPUT = CAMPAIGN_ROOT / "training" / "world_model"  # Isolate transition evidence and the final V0 model snapshot.

def _parser() -> argparse.ArgumentParser:  # Build a location-and-operation-only command contract.
    parser = argparse.ArgumentParser(description="Build the frozen WMVLA-4WAY-P1 world-model transition library.")  # Create a self-describing CLI.
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Checksummed frozen case_manifest.json.")  # Permit manifest relocation without changing its content.
    parser.add_argument("--partition-root", type=Path, default=DEFAULT_PARTITIONS, help="Indexed protocol/partitions registry shared with RL.")  # Require the same frozen semantic definitions used at deployment.
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New empty world-model training artifact directory.")  # Prevent implicit resume or checkpoint replacement.
    parser.add_argument("--ccx-timeout", type=float, default=1800.0, help="Operational timeout per real CalculiX solve in seconds.")  # Expose only a non-scientific process bound.
    parser.add_argument("--dry-run", action="store_true", help="Print the authenticated 24x6 plan without partitions, Gmsh, CalculiX, or writes.")  # Support pre-execution review and focused tests.
    return parser  # Return the complete immutable scientific interface.

def main() -> int:  # Dispatch solve-free planning or the complete train-only acquisition campaign.
    args = _parser().parse_args()  # Parse paths, timeout, and the explicit no-write mode.
    config = WorldTrainingConfig(ccx_timeout=float(args.ccx_timeout))  # Freeze the sole allowed operational override with all scientific defaults explicit.
    if args.dry_run:  # Avoid partition access, geometry generation, solver work, and filesystem writes.
        payload = build_training_plan(args.manifest, args.partition_root, args.output, config)  # Authenticate the manifest and materialize only train identities.
    else:  # Execute all 24 ordered train trajectories and persist their complete evidence.
        payload = train_world_model_transition_library(args.manifest, args.partition_root, args.output, config=config)  # Build and hash the current public ResidualWorldModel snapshot.
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit one strict finite automation record.
    return 0  # Signal successful plan or retained campaign completion.

if __name__ == "__main__":  # Execute only when launched as a command rather than imported.
    raise SystemExit(main())  # Propagate the explicit command status to the shell or CI.
