#!/usr/bin/env python3  # Execute the frozen bridge RL campaign with the active Python interpreter.
"""Plan, train, freeze, or blind-test the protocol region-graph Double DQN."""  # Describe the command's complete responsibility.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
import argparse  # Expose only phase and artifact locations as command-line choices.
import json  # Print strict machine-readable completion or dry-run records.
import os  # Freeze native thread counts before importing numerical or solver modules.
from pathlib import Path  # Resolve repository defaults and caller-selected artifact roots portably.
import sys  # Inject the checked-out package path and propagate explicit process status.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository independently from the launch directory.
sys.path.insert(0, str(ROOT))  # Import this checkout without requiring a mutable installation step.
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Pin OpenMP-backed native work to one thread for reproducibility.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # Pin OpenBLAS-backed operations to one thread for reproducibility.
os.environ.setdefault("MKL_NUM_THREADS", "1")  # Pin MKL-backed operations to one thread for reproducibility.
from visionamr.baselines.bridge_rl import RL_SEEDS, assemble_bridge_rl, build_training_plan, build_training_reference_policy, test_bridge_rl, train_bridge_rl, train_bridge_rl_seed  # Import the sole protocol-bound planning, policy authorization, sequential, sharded, assembly, and test entry points.
from visionamr.bridge_case_manifest import load_case_manifest  # Authenticate the frozen 48-case manifest before a dry run.

CAMPAIGN_ROOT = ROOT / "results" / "wm_vla_four_way_p1"  # Locate the protocol's canonical result tree.
DEFAULT_MANIFEST = CAMPAIGN_ROOT / "protocol" / "case_manifest.json"  # Select the checksummed frozen manifest by default.
DEFAULT_PARTITIONS = CAMPAIGN_ROOT / "protocol" / "partitions"  # Select the canonical protocol-owned shared WM/RL partition registry by default.
DEFAULT_REFERENCES = CAMPAIGN_ROOT / "references"  # Select the completed per-case Reference-B registry by default.
DEFAULT_MODEL_OUTPUT = CAMPAIGN_ROOT / "models" / "rl"  # Keep RL model artifacts isolated from other frozen methods.
DEFAULT_TEST_OUTPUT = CAMPAIGN_ROOT / "methods" / "rl_dqn"  # Keep all raw and median RL blind-test evidence in the method tree.

def _common_paths(parser: argparse.ArgumentParser) -> None:  # Add immutable campaign input locations shared by all executable phases.
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Checksummed frozen case_manifest.json.")  # Permit relocation without changing case identities or sampling.
    parser.add_argument("--partition-root", type=Path, default=DEFAULT_PARTITIONS, help="Per-case shared partition_spec.json root.")  # Require the same frozen registry consumed by WM-VLA.
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCES, help="Completed per-case common Reference-B root.")  # Require prebuilt validation or blind-test references.

def _training_reference_policy_args(parser: argparse.ArgumentParser) -> None:  # Add the strict-default checkpoint-denominator authorization pair to planning and training only.
    parser.add_argument("--allow-unqualified-references", action="store_true", help="Explicitly permit only the fixed amended complete_unqualified validation references.")  # Require a conspicuous opt-in rather than silently accepting shortened Reference B evidence.
    parser.add_argument("--expedited-reference-levels", type=int, choices=(2,), default=None, help="Fixed amended Reference B ladder depth; requires --allow-unqualified-references.")  # Expose only the user-authorized two-level prefix and no tunable scientific choice.

def _parser() -> argparse.ArgumentParser:  # Build the explicit plan, train, and one-way blind-test command contract.
    parser = argparse.ArgumentParser(description="Run frozen WMVLA-4WAY-P1 region-graph Double-DQN phases.")  # Create a self-describing top-level parser.
    subparsers = parser.add_subparsers(dest="phase", required=True)  # Require the caller to name the authorization boundary explicitly.
    plan_parser = subparsers.add_parser("plan", help="Print the three-seed schedule without native solves or file writes.")  # Provide a safe dry-run phase for CI and review.
    plan_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Checksummed frozen case_manifest.json.")  # Read only the manifest needed to materialize train and validation identities.
    _training_reference_policy_args(plan_parser)  # Make exceptional reference intent visible and amendment-checked even in the printed plan.
    train_parser = subparsers.add_parser("train", help="Train 3x300 episodes, validate every 25, and freeze three models.")  # Expose the complete pre-test model-building phase.
    _common_paths(train_parser)  # Add the manifest, partition, and reference roots required for training and validation.
    _training_reference_policy_args(train_parser)  # Keep strict qualification by default and require both fixed amendment switches together.
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_OUTPUT, help="New empty directory receiving checkpoints and freeze index.")  # Refuse overwrite or implicit resume in the implementation.
    train_parser.add_argument("--seed", type=int, choices=RL_SEEDS, default=None, help="Train one fixed seed into an independent shard for parallel execution.")  # Permit only the three frozen seeds and never a seed search.
    train_parser.add_argument("--dry-run", action="store_true", help="Print the exact schedule without touching models, partitions, references, or solvers.")  # Permit a zero-cost contract check before starting 900 episodes.
    assemble_parser = subparsers.add_parser("assemble", help="Verify and merge exactly three complete fixed-seed shards without best-seed selection.")  # Expose the no-solver parallel-results assembly phase.
    assemble_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Checksummed frozen case_manifest.json used by every shard.")  # Bind all shard schedule checks to the same exact manifest bytes.
    assemble_parser.add_argument("--seed-shard", dest="seed_shards", type=Path, action="append", required=True, help="Independent shard directory containing rl_seed_shard.json; pass exactly three times.")  # Require explicit source directories without directory scanning or candidate search.
    assemble_parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_OUTPUT, help="New empty directory receiving three verified models and unified freeze index.")  # Keep assembly append-free and immutable.
    test_parser = subparsers.add_parser("test", help="Run the three frozen greedy policies once on 16 sorted blind cases.")  # Expose the separate post-freeze authorization boundary.
    _common_paths(test_parser)  # Add the exact manifest, partition, and common-reference roots used by blind execution.
    test_parser.add_argument("--freeze-index", type=Path, default=DEFAULT_MODEL_OUTPUT / "rl_freeze_index.json", help="Pre-test hashed model index with TEST_NOT_RUN=true.")  # Bind testing to reviewed selected weights and configuration.
    test_parser.add_argument("--output-dir", type=Path, default=DEFAULT_TEST_OUTPUT, help="New empty one-way blind-test evidence directory.")  # Refuse resume, rerun, or overwrite in the implementation.
    return parser  # Return the complete phase-separated command contract.

def _print(payload: dict) -> None:  # Emit one strict structured terminal record for automation.
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Refuse nonfinite completion metadata and preserve stable key order.

def main() -> int:  # Dispatch the caller's explicitly authorized RL campaign phase.
    args = _parser().parse_args()  # Parse phase and artifact paths without exposing scientific hyperparameters.
    if args.phase == "plan" or bool(getattr(args, "dry_run", False)):  # Keep plan and training dry-run entirely free of native work and writes.
        manifest = load_case_manifest(args.manifest, verify_checksum=True)  # Authenticate the frozen manifest and its exact-byte sidecar.
        reference_policy = build_training_reference_policy(args.manifest, allow_unqualified_references=bool(args.allow_unqualified_references), expedited_reference_levels=args.expedited_reference_levels)  # Authenticate strict intent or the exact canonical amendment before printing a plan.
        plan = build_training_plan(manifest, reference_policy)  # Build the complete reviewed schedule with its explicit validation-reference contract.
        selected_seed = getattr(args, "seed", None)  # Read an optional fixed-seed parallel shard without inventing a new seed.
        if selected_seed is not None:  # Restrict only execution scope while preserving every scientific hyperparameter.
            plan = {**plan, "phase": "seed_training_plan", "shard_seed": int(selected_seed), "policies": [policy for policy in plan["policies"] if int(policy["seed"]) == int(selected_seed)]}  # Print one exact shard schedule with no cross-seed selection.
        _print(plan)  # Emit the selected zero-solve schedule.
        return 0  # Signal successful zero-solve planning.
    if args.phase == "train":  # Execute only the authorized train and validation splits.
        reference_options = {"allow_unqualified_references": bool(args.allow_unqualified_references), "expedited_reference_levels": args.expedited_reference_levels}  # Forward the caller's exact paired choice without an implicit waiver.
        result = train_bridge_rl(args.manifest, args.partition_root, args.reference_root, args.output_dir, **reference_options) if args.seed is None else train_bridge_rl_seed(args.manifest, args.partition_root, args.reference_root, args.output_dir, args.seed, **reference_options)  # Run all seeds sequentially or one fixed-seed shard under the explicit strict or amended policy.
        _print(result)  # Report the immutable model index, hash, cost file, and TEST_NOT_RUN state.
        return 0  # Signal successful completion of the requested three-hundred- or nine-hundred-episode training scope.
    if args.phase == "assemble":  # Merge only three already completed and independently sealed fixed-seed shards.
        result = assemble_bridge_rl(args.manifest, args.seed_shards, args.output_dir)  # Recompute schedules, validation selections, report hashes, and model hashes without any solver call.
        _print(result)  # Report the unified three-model freeze index and aggregate training cost.
        return 0  # Signal successful no-best-seed assembly.
    result = test_bridge_rl(args.manifest, args.partition_root, args.reference_root, args.freeze_index, args.output_dir)  # Execute the authenticated frozen policies once on sorted blind cases.
    _print(result)  # Report raw, pointwise-median, and completion-receipt paths and counts.
    return 0  # Signal successful completion of the one-way RL blind test.

if __name__ == "__main__":  # Execute only when launched as a command rather than imported for tests.
    raise SystemExit(main())  # Propagate the explicit phase status to CI or the shell.
