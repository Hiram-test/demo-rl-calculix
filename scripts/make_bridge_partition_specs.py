#!/usr/bin/env python3  # Execute shared partition generation with the active repository interpreter.
"""Plan or generate indexed per-case WM/RL frozen partition specifications."""  # Describe the no-CalculiX protocol helper.
from __future__ import annotations  # Postpone annotation evaluation for compatible repository runtimes.
import argparse  # Parse only explicit case split, artifact locations, and dry-run mode.
import json  # Print strict machine-readable plan or registry evidence.
import os  # Freeze native mesher thread count before importing Gmsh-facing modules.
from pathlib import Path  # Resolve repository and campaign paths portably.
import sys  # Import the checked-out package and propagate command status.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository root independently from launch directory.
sys.path.insert(0, str(ROOT))  # Import this checkout without requiring installation.
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Pin native meshing work for deterministic resource use.
os.environ.setdefault("GMSH_NUM_THREADS", "1")  # Pin Gmsh's explicit thread control when supported.
from visionamr.vla.four_way_world_training import build_partition_plan  # Import the authenticated no-Gmsh dry-run builder.
from visionamr.vla.four_way_world_training import generate_partition_specs  # Import indexed shared partition generation.

CAMPAIGN_ROOT = ROOT / "results" / "wm_vla_four_way_p1"  # Locate the frozen experiment result tree.
DEFAULT_MANIFEST = CAMPAIGN_ROOT / "protocol" / "case_manifest.json"  # Select the checksummed case design.
DEFAULT_OUTPUT = CAMPAIGN_ROOT / "protocol" / "partitions"  # Keep shared semantics among immutable protocol inputs.

def _parser() -> argparse.ArgumentParser:  # Build the explicit split-aware no-solver command contract.
    parser = argparse.ArgumentParser(description="Generate WMVLA-4WAY-P1 shared WM/RL partition specifications without CalculiX.")  # Create a self-describing CLI.
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Checksummed frozen case_manifest.json.")  # Permit manifest relocation without content changes.
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT, help="protocol/partitions registry receiving per-case specs and the index.")  # Permit only output-root relocation.
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="all", help="Explicit complete manifest split to generate; all is freeze-ready.")  # Prevent implicit or partial case selection.
    parser.add_argument("--dry-run", action="store_true", help="Authenticate and print the exact plan without Gmsh or writes.")  # Support pre-generation review.
    return parser  # Return the complete no-CalculiX interface.

def main() -> int:  # Dispatch authenticated planning or exact shared partition generation.
    args = _parser().parse_args()  # Parse the explicit manifest, output root, split, and dry-run boundary.
    if args.dry_run:  # Avoid geometry reconstruction, Gmsh, and filesystem writes entirely.
        payload = build_partition_plan(args.manifest, args.output_root, args.split)  # Materialize only authenticated case identities.
    else:  # Generate or exactly verify every selected per-case frozen specification.
        payload = generate_partition_specs(args.manifest, args.output_root, args.split)  # Use the common uniform probe and publish file/body hash inventory.
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit one strict finite automation record.
    return 0  # Signal successful plan or no-CalculiX registry generation.

if __name__ == "__main__":  # Execute only when launched as a command rather than imported.
    raise SystemExit(main())  # Propagate the explicit command status to the shell or CI.
