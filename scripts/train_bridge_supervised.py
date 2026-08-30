#!/usr/bin/env python3  # Execute the frozen supervised training pipeline with the active repository interpreter.
"""Train, validate, select, hash, and freeze the four-way supervised baseline."""  # Describe the command's complete pre-test responsibility.
from __future__ import annotations  # Postpone annotation evaluation for broad interpreter compatibility.
import argparse  # Parse only artifact locations, native timeout, and solve-free dry-run selection.
import hashlib  # Bind the training plan to exact manifest bytes.
import json  # Print and persist transparent finite machine-readable summaries.
from pathlib import Path  # Resolve repository-relative manifest, reference, and output locations portably.
import sys  # Add the checked-out package to the import path and propagate process status.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository root independently of the launch directory.
sys.path.insert(0, str(ROOT))  # Import the checked-out implementation without requiring installation.
from visionamr.baselines.bridge_supervised import BridgeSupervisedConfig  # Import the immutable scientific configuration.
from visionamr.baselines.bridge_supervised import build_training_summary  # Import mandatory offline-cost and hash reporting.
from visionamr.baselines.bridge_supervised import cases_for_split  # Import the train/validation-only manifest boundary.
from visionamr.baselines.bridge_supervised import generate_bridge_expert_dataset  # Import train-only exact-Dörfler expert generation.
from visionamr.baselines.bridge_supervised import supervised_config_payload  # Import the complete pre-test scientific snapshot.
from visionamr.baselines.bridge_supervised import train_candidate_networks  # Import exact three-seed network fitting and hashing.
from visionamr.baselines.bridge_supervised import validate_candidate_networks  # Import reference-B validation and frozen checkpoint selection.
from visionamr.baselines.bridge_supervised import write_json  # Import strict atomic JSON artifact persistence.
from visionamr.bridge_case_manifest import load_case_manifest  # Import checksum, schema, split, geometry, and identity verification.
from visionamr.bridge_case_manifest import problem_from_case  # Reconstruct only authorized validation problems for reference preflight.
from visionamr.vla.four_way_references import load_reference_b  # Authenticate common independently converged validation Reference B caches.
from visionamr.vla.four_way_references import verify_reference_cache  # Preserve strict or explicitly amended qualification evidence in the training plan.

def _parser() -> argparse.ArgumentParser:  # Build a CLI that exposes no scientific hyperparameter or seed tuning knobs.
    parser = argparse.ArgumentParser(description="Train and freeze the WMVLA-4WAY-P1 supervised bridge baseline.")  # Create a self-describing command parser.
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1" / "protocol" / "case_manifest.json", help="Validated frozen case_manifest.json path.")  # Permit only selection of the frozen manifest artifact location.
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1" / "training" / "supervised", help="Directory receiving expert, model, validation, and freeze artifacts.")  # Permit only supervised artifact-root relocation.
    parser.add_argument("--reference-root", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1" / "references", help="Directory containing authenticated per-validation-case Reference B caches.")  # Point to the independently generated common reference registry.
    parser.add_argument("--ccx-timeout", type=float, default=1800.0, help="Operational per-CalculiX timeout in seconds; it does not alter policy or labels before a timeout.")  # Expose an environment-only native process limit.
    parser.add_argument("--allow-unqualified-references", action="store_true", help="Explicitly permit integrity-verified complete_unqualified validation references under the reviewed amendment.")  # Keep exceptional operational denominators opt-in and machine-readable.
    parser.add_argument("--expedited-reference-levels", type=int, choices=(2,), default=None, help="Fixed two-level amended reference depth; valid only with --allow-unqualified-references.")  # Bind the exceptional opt-in to the exact reviewed fast-execution schedule.
    parser.add_argument("--dry-run", action="store_true", help="Validate the manifest and print the exact solve-free training plan without creating training artifacts.")  # Support unit and freeze review before expensive native execution.
    return parser  # Return the complete location-and-operation-only command contract.

def main() -> int:  # Execute manifest validation, train-only labels, three seeds, validation selection, and final freeze evidence.
    args = _parser().parse_args()  # Parse operational artifact locations and dry-run mode.
    if not math_is_positive(args.ccx_timeout):  # Require a usable finite native timeout without changing scientific scoring.
        raise SystemExit("--ccx-timeout must be finite and positive")  # Stop before any geometry, model, or solver work.
    if bool(args.allow_unqualified_references) != (args.expedited_reference_levels is not None):  # Require both exceptional controls together rather than interpreting an ambiguous partial opt-in.
        raise SystemExit("--allow-unqualified-references and --expedited-reference-levels must be specified together")  # Stop before manifest, reference, mesh, model, or solver access.
    manifest = load_case_manifest(args.manifest, verify_checksum=True)  # Authenticate the exact frozen 48-case manifest and sidecar.
    train_cases = cases_for_split(manifest, "train")  # Select exactly 24 training records without returning blind-test records.
    validation_cases = cases_for_split(manifest, "validation")  # Select exactly eight validation records without returning blind-test records.
    config = BridgeSupervisedConfig()  # Instantiate the immutable pre-registered scientific configuration.
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()  # Bind all downstream evidence to exact manifest bytes.
    plan = {"schema": "wmvla-four-way-supervised-training-plan-v1", "protocol_id": "WMVLA-4WAY-P1", "dry_run": bool(args.dry_run), "manifest_path": str(args.manifest), "manifest_sha256": manifest_sha, "output": str(args.output), "reference_root": str(args.reference_root), "allow_unqualified_references": bool(args.allow_unqualified_references), "expedited_reference_levels": args.expedited_reference_levels, "training_case_ids": [str(case["case_id"]) for case in train_cases], "validation_case_ids": [str(case["case_id"]) for case in validation_cases], "training_case_count": len(train_cases), "validation_case_count": len(validation_cases), "test_case_count_executed": 0, "test_split_accessed_for_training_or_selection": False, "scientific_config": supervised_config_payload(config)}  # Assemble the exact pre-execution development-only plan including the explicit denominator policy.
    if args.dry_run:  # Return after validation without creating artifacts or invoking Gmsh, CalculiX, or Torch.
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit the complete solve-free plan for freeze review.
        return 0  # Signal successful dry-run validation.
    reference_receipts = []  # Accumulate exact common-reference readiness evidence before the first expert solve.
    for case in validation_cases:  # Authenticate all eight validation denominators without inspecting any test cache.
        problem = problem_from_case(case)  # Reconstruct this manifest-authorized validation problem for cache-identity verification.
        verified = verify_reference_cache(args.reference_root, case_id=str(case["case_id"]), problem=problem, allow_unqualified=bool(args.allow_unqualified_references), expedited_levels=args.expedited_reference_levels)  # Authenticate integrity and qualification under the exact explicit strict or amended policy.
        reference = load_reference_b(args.reference_root, case_id=str(case["case_id"]), problem=problem, verify=True, allow_unqualified=bool(args.allow_unqualified_references), expedited_levels=args.expedited_reference_levels)  # Fail closed on reference identity, evidence, schedule, or unauthorized qualification state.
        reference_receipts.append({"case_id": str(case["case_id"]), "status": str(verified["status"]), "qualification": bool(verified["qualification"]), "authorization": verified.get("authorization"), "original_convergence_gate": dict(verified["original_convergence_gate"]), "n_equations": int(reference.n_equations), "n_elements": int(reference.n_elems), "h_ref": float(reference.h_ref)})  # Preserve finite readiness and original-gate evidence without copying physical outputs into model inputs.
    args.output.mkdir(parents=True, exist_ok=True)  # Create the requested artifact root only after manifest and split validation.
    plan["validation_reference_preflight"] = reference_receipts  # Attach complete common-reference readiness to the immutable execution plan.
    write_json(args.output / "training_plan.json", plan)  # Persist the exact manifest-bound plan before the first expensive solve.
    dataset_path, expert_summary = generate_bridge_expert_dataset(train_cases, args.output, config, ccx_timeout=float(args.ccx_timeout))  # Generate labels only from 24 exact-Dörfler training trajectories.
    candidates, network_summary = train_candidate_networks(dataset_path, args.output, config)  # Fit and hash exactly the three frozen network seeds.
    selection, _validation_rows = validate_candidate_networks(validation_cases, candidates, args.output, args.reference_root, config, ccx_timeout=float(args.ccx_timeout), allow_unqualified_references=bool(args.allow_unqualified_references), expedited_reference_levels=args.expedited_reference_levels)  # Select one checkpoint only from eight validation cases and three budgets under the predeclared denominator policy.
    summary = build_training_summary(expert_summary, network_summary, selection, config)  # Assemble mandatory solve, time, sample, parameter, config, and model evidence.
    summary["manifest_sha256"] = manifest_sha  # Bind the final training receipt to exact manifest bytes.
    summary["training_plan_sha256"] = hashlib.sha256((args.output / "training_plan.json").read_bytes()).hexdigest()  # Bind completion to the exact pre-execution plan artifact.
    summary_path = write_json(args.output / "training_summary.json", summary)  # Persist the final finite pre-test supervised receipt.
    print(json.dumps({"training_summary": str(summary_path), **summary}, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Report exact artifacts, costs, selected seed, and hashes to automation.
    return 0  # Signal successful complete supervised training and validation freeze.

def math_is_positive(value: float) -> bool:  # Validate an operational timeout without importing numerical dependencies into the CLI.
    try:  # Normalize integer and floating command-line values safely.
        numeric = float(value)  # Convert the parsed value to a standard Python float.
    except (TypeError, ValueError):  # Reject unexpected nonnumeric programmatic invocation.
        return False  # Report invalidity to the caller's single error path.
    return numeric > 0.0 and numeric < float("inf")  # Require positive finite duration while allowing practical large timeouts.

if __name__ == "__main__":  # Execute only when launched as the training command.
    raise SystemExit(main())  # Propagate the explicit process status to CI or the shell.
