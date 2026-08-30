#!/usr/bin/env python3  # Execute transparent operational sharding without changing the frozen supervised validation mechanics.
"""Run one supervised validation seed or assemble three independently completed seed grids."""  # Describe the auditable operational-only responsibility.
from __future__ import annotations  # Postpone annotation evaluation for the repository runtime.

import argparse  # Parse one explicit validation or assembly operation.
import hashlib  # Bind the final receipt to exact manifest, plan, model, score, and row bytes.
import json  # Read strict production artifacts and print a finite terminal receipt.
from pathlib import Path  # Resolve formal and external shard locations without shell-dependent spelling.
import shutil  # Copy completed disjoint seed directories and the selected model without reserialization.
import sys  # Import the checked-out repository package and return explicit process status.
from typing import Any  # Annotate finite heterogeneous JSON records.

ROOT = Path(__file__).resolve().parents[4]  # Recover the repository root from the retained campaign-evidence location.
sys.path.insert(0, str(ROOT))  # Import this exact checked-out implementation without installing another revision.

import visionamr.baselines.bridge_supervised as bridge  # Reuse the production deployment, score, selection, hashing, and JSON boundaries.
from visionamr.bridge_case_manifest import load_case_manifest  # Authenticate the exact frozen manifest before selecting validation cases.

PROTOCOL_ID = "WMVLA-4WAY-P1"  # Bind every operational receipt to the sole formal campaign.
SEEDS = (20260831, 20260832, 20260833)  # Preserve the preregistered supervised candidate set exactly.


def _sha256(path: Path) -> str:  # Hash one finite artifact using the same complete-byte identity convention.
    return hashlib.sha256(path.read_bytes()).hexdigest()  # Return the full lowercase digest after reading the bounded structured artifact.


def _read_json(path: Path) -> dict[str, Any]:  # Load one strict top-level JSON object from a production artifact.
    payload = json.loads(path.read_text(encoding="utf-8"))  # Decode the complete UTF-8 bytes without partial recovery.
    if not isinstance(payload, dict):  # Require named fields at this evidence boundary.
        raise ValueError(f"JSON artifact is not an object: {path}")  # Reject arrays or scalars before orchestration.
    return payload  # Return the authenticated-by-context mapping for semantic validation.


def _single_score(scores: list[dict[str, Any]]) -> dict[str, Any]:  # Select the only score inside an operational one-seed shard.
    if len(scores) != 1:  # Refuse an accidental partial or multi-seed shard.
        raise ValueError("operational validation shard must produce exactly one seed score")  # Keep shard identity explicit.
    return dict(scores[0])  # Return a copy compatible with the production finalization path.


def _candidate(network_summary: dict[str, Any], seed: int) -> dict[str, Any]:  # Resolve one exact trained candidate by preregistered seed.
    matches = [dict(item) for item in network_summary.get("candidates", []) if int(item.get("seed", -1)) == seed]  # Match only the requested seed without ranking.
    if len(matches) != 1:  # Require one unambiguous trained checkpoint.
        raise ValueError(f"network summary does not contain exactly one candidate for seed {seed}")  # Reject missing or duplicate candidates.
    path = Path(str(matches[0]["model_path"]))  # Resolve the production-recorded candidate path.
    if not path.is_file() or bridge.file_sha256(path) != str(matches[0]["model_sha256"]):  # Recompute the model identity before any real validation solve.
        raise ValueError(f"candidate model hash mismatch for seed {seed}")  # Stop before spending validation truth on altered weights.
    return matches[0]  # Return the exact verified candidate receipt.


def validate_seed(seed: int, output: Path, formal_root: Path, reference_root: Path, manifest_path: Path) -> dict[str, Any]:  # Execute one complete 8-case-by-3-budget candidate grid.
    if seed not in SEEDS:  # Restrict operational shards to the frozen candidate set.
        raise ValueError(f"seed must be one of {SEEDS}")  # Reject any post-registration candidate.
    if output.exists():  # Preserve one-shot evidence for this independent shard.
        raise FileExistsError(f"validation shard output already exists: {output}")  # Refuse overwrite or implicit resume.
    manifest = load_case_manifest(manifest_path, verify_checksum=True)  # Authenticate exact manifest bytes before split access.
    validation_cases = bridge.cases_for_split(manifest, "validation")  # Select exactly the eight authorized development cases.
    network_summary_path = formal_root / "network_training_summary.json"  # Resolve the completed three-candidate training receipt.
    network_summary = _read_json(network_summary_path)  # Load candidate identities without touching test artifacts.
    candidate = _candidate(network_summary, seed)  # Verify the sole shard model before native execution.
    original_seeds = bridge.NETWORK_SEEDS  # Preserve the reviewed module constant for explicit restoration.
    original_selector = bridge.select_validation_checkpoint  # Preserve the reviewed three-seed selector for explicit restoration.
    bridge.NETWORK_SEEDS = (seed,)  # Limit only orchestration to one disjoint candidate while retaining every point-level production function.
    bridge.select_validation_checkpoint = _single_score  # Finalize the one-seed shard without pretending it is the cross-seed selection.
    try:  # Restore module globals even if a native or schema failure escapes.
        selection, rows = bridge.validate_candidate_networks(validation_cases, [candidate], output, reference_root, bridge.BridgeSupervisedConfig(), allow_unqualified_references=True, expedited_reference_levels=2)  # Execute unchanged production validation mechanics over the complete grid.
    finally:  # Reestablish the reviewed process state after the shard call.
        bridge.NETWORK_SEEDS = original_seeds  # Restore the exact frozen candidate set.
        bridge.select_validation_checkpoint = original_selector  # Restore the exact frozen three-seed selector.
    receipt = {"schema": "wmvla-four-way-supervised-validation-shard-v1", "protocol_id": PROTOCOL_ID, "execution_mode": "parallel_disjoint_seed_grid", "seed": seed, "TEST_NOT_RUN": True, "validation_case_count": 8, "validation_point_count": len(rows), "model_sha256": str(candidate["model_sha256"]), "score_sha256": _sha256(output / "validation" / f"seed_{seed}" / "score.json"), "validation_rows_sha256": _sha256(output / "validation_rows.json"), "validation_wall_s": float(selection["validation_wall_s"]), "allow_unqualified_references": True, "expedited_reference_levels": 2, "test_split_accessed": False}  # Bind the complete independent seed execution and anti-leakage declarations.
    receipt_path = bridge.write_json(output / "PARALLEL_SHARD_RECEIPT.json", receipt)  # Publish the shard receipt only after all twenty-four points and score exist.
    return {**receipt, "receipt_path": str(receipt_path)}  # Return exact terminal evidence to automation.


def _validation_rows(seed_directory: Path, seed: int) -> list[dict[str, Any]]:  # Load and validate one complete seed grid from immediate point receipts.
    paths = sorted(seed_directory.glob("BGD-*/B*/validation_result.json"))  # Enumerate only canonical case-budget result files under this seed.
    rows = [_read_json(path) for path in paths]  # Decode every retained success or typed numerical failure.
    if len(rows) != 24 or {int(row.get("seed", -1)) for row in rows} != {seed}:  # Require the exact 8-by-3 grid and one seed identity.
        raise ValueError(f"seed {seed} does not contain exactly 24 validation results")  # Reject omission, duplication, or cross-seed contamination.
    recomputed = bridge.validation_score(rows, expected_cases=8, budgets=bridge.VALIDATION_BUDGETS, failure_error=bridge.VALIDATION_FAILURE_ERROR)  # Recompute the preregistered lexicographic score from point evidence.
    score_path = seed_directory / "score.json"  # Resolve the production-written candidate score.
    score = _read_json(score_path)  # Load the exact candidate-model binding and selection key.
    if score.get("selection_key") != recomputed.get("selection_key") or int(score.get("seed", -1)) != seed:  # Require exact scoring agreement before cross-seed comparison.
        raise ValueError(f"seed {seed} score differs from recomputed validation rows")  # Reject a stale or altered score receipt.
    return rows  # Return the complete verified candidate grid.


def _copy_seed(source_root: Path, formal_validation: Path, seed: int) -> Path:  # Publish one completed external seed grid into the formal campaign without altering bytes.
    source = source_root / "validation" / f"seed_{seed}"  # Resolve the sole production-created seed directory inside its isolated shard.
    source_receipt = source_root / "PARALLEL_SHARD_RECEIPT.json"  # Resolve the terminal receipt that authenticates this complete external shard.
    destination = formal_validation / f"seed_{seed}"  # Resolve the canonical formal validation location.
    if not source.is_dir() or not source_receipt.is_file() or destination.exists():  # Require a completed source, its receipt, and a pristine destination.
        raise FileExistsError(f"cannot publish seed {seed} from {source} to {destination}")  # Refuse missing, overwrite, or race conditions.
    receipt = _read_json(source_receipt)  # Decode the terminal helper receipt before publishing any external point evidence.
    if receipt.get("schema") != "wmvla-four-way-supervised-validation-shard-v1" or int(receipt.get("seed", -1)) != seed or int(receipt.get("validation_point_count", -1)) != 24 or receipt.get("TEST_NOT_RUN") is not True:  # Require exact shard identity, coverage, and anti-leakage declarations.
        raise ValueError(f"supervised validation shard receipt is invalid for seed {seed}")  # Reject a partial, foreign, or test-exposed helper grid before copy.
    shutil.copytree(source, destination, copy_function=shutil.copy2)  # Copy exact files and timestamps for reproducible operational receipts.
    shutil.copy2(source_receipt, destination / "PARALLEL_SHARD_RECEIPT.json")  # Preserve the original shard-level terminal identity beside its point evidence.
    return destination  # Return the canonical completed seed directory.


def assemble(formal_root: Path, shard_root: Path, manifest_path: Path) -> dict[str, Any]:  # Assemble three complete seed grids through the production selector and summary builder.
    formal_validation = formal_root / "validation"  # Resolve the canonical development-selection evidence root.
    _copy_seed(shard_root / "seed_20260832", formal_validation, 20260832)  # Publish the independently completed second candidate grid.
    _copy_seed(shard_root / "seed_20260833", formal_validation, 20260833)  # Publish the independently completed third candidate grid.
    all_rows: list[dict[str, Any]] = []  # Accumulate every one of the seventy-two immediate point receipts.
    scores: list[dict[str, Any]] = []  # Accumulate all and only the three preregistered candidate scores.
    seed_wall: dict[str, float] = {}  # Record honest per-seed wall evidence from formal or isolated execution receipts.
    for seed in SEEDS:  # Process candidates in the preregistered order.
        seed_directory = formal_validation / f"seed_{seed}"  # Resolve this candidate's canonical complete grid.
        rows = _validation_rows(seed_directory, seed)  # Authenticate exact coverage and recomputed score.
        all_rows.extend(rows)  # Retain every point for the global validation table.
        scores.append(_read_json(seed_directory / "score.json"))  # Retain the exact model-bound selection record.
        point_wall = sum(float(row.get("online_wall_s", 0.0)) for row in rows)  # Sum measured online point costs without hiding parallel compute consumption.
        seed_wall[str(seed)] = point_wall  # Preserve a conservative comparable per-seed cost measure.
    selected = bridge.select_validation_checkpoint(scores)  # Apply the unchanged frozen three-seed lexicographic selector.
    network_summary_path = formal_root / "network_training_summary.json"  # Resolve the completed candidate-training inventory.
    network_summary = _read_json(network_summary_path)  # Load all exact candidate model identities.
    selected_candidate = _candidate(network_summary, int(selected["seed"]))  # Recover and rehash the selected candidate bytes.
    selected_path = formal_root / "selected_model.pt"  # Resolve the canonical deployment checkpoint path.
    if selected_path.exists():  # Preserve one-shot selection publication.
        raise FileExistsError(f"selected supervised model already exists: {selected_path}")  # Refuse overwrite or second selection.
    shutil.copyfile(Path(str(selected_candidate["model_path"])), selected_path)  # Copy exact selected bytes without retraining or serialization.
    selected_hash = bridge.file_sha256(selected_path)  # Recompute the deployment checkpoint identity after publication.
    if selected_hash != str(selected_candidate["model_sha256"]):  # Require byte identity with the validation-selected candidate.
        raise RuntimeError("selected supervised model differs from its candidate bytes")  # Stop before emitting an inconsistent index.
    ordered_rows = sorted(all_rows, key=lambda row: (int(row["seed"]), str(row["case_id"]), int(row["equation_budget"])))  # Make aggregate evidence independent of shard completion timing.
    qualifications = {str(case_id): bool(next(row["reference_qualification"] for row in ordered_rows if str(row["case_id"]) == str(case_id))) for case_id in sorted({str(row["case_id"]) for row in ordered_rows})}  # Preserve each denominator's original qualification outcome.
    selection = {"schema": bridge.SUPERVISED_SCHEMA, "protocol_id": PROTOCOL_ID, "selection_rule": ["failure_point_count", "energy_error_log_mean", "qoi_error_log_mean", "budget_violation_count", "seed"], "selected_seed": int(selected["seed"]), "selected_model_path": str(selected_path), "selected_model_sha256": selected_hash, "config_sha256": str(selected_candidate["config_sha256"]), "validation_case_count": 8, "validation_point_count_per_seed": 24, "validation_wall_s": float(sum(seed_wall.values())), "validation_parallel_makespan_upper_bound_s": float(max(seed_wall.values())), "validation_execution_mode": "three_disjoint_seed_grids_parallel", "validation_seed_online_wall_s": seed_wall, "allow_unqualified_references": True, "expedited_reference_levels": 2, "reference_qualification_by_case": qualifications, "scores": scores}  # Assemble transparent selection and conservative summed compute cost without test access.
    bridge.write_json(formal_root / "validation_rows.json", {"schema": bridge.SUPERVISED_SCHEMA, "protocol_id": PROTOCOL_ID, "rows": ordered_rows})  # Publish all seventy-two exact point outcomes before selection evidence.
    bridge.write_json(formal_root / "selected_model.json", selection)  # Publish the sole selected deployment identity and complete score table.
    expert_summary = _read_json(formal_root / "expert_dataset_metadata.json")  # Recover train-only solve and dataset cost evidence.
    summary = bridge.build_training_summary(expert_summary, network_summary, selection, bridge.BridgeSupervisedConfig())  # Build the standard production training-cost and model summary.
    summary["manifest_sha256"] = _sha256(manifest_path)  # Bind completion to exact frozen manifest bytes.
    summary["training_plan_sha256"] = _sha256(formal_root / "training_plan.json")  # Bind completion to exact pre-execution intent.
    summary["validation_execution_mode"] = "three_disjoint_seed_grids_parallel"  # Disclose the operational acceleration explicitly.
    summary["validation_parallel_makespan_upper_bound_s"] = float(max(seed_wall.values()))  # Report the conservative parallel makespan separately from total compute.
    summary_path = bridge.write_json(formal_root / "training_summary.json", summary)  # Publish the standard final supervised freeze receipt once.
    receipt = {"schema": "wmvla-four-way-supervised-validation-assembly-v1", "protocol_id": PROTOCOL_ID, "TEST_NOT_RUN": True, "execution_mode": "three_disjoint_seed_grids_parallel", "seeds": list(SEEDS), "validation_point_count": len(ordered_rows), "selected_seed": int(selected["seed"]), "selected_model_sha256": selected_hash, "validation_rows_sha256": _sha256(formal_root / "validation_rows.json"), "selected_model_receipt_sha256": _sha256(formal_root / "selected_model.json"), "training_summary_sha256": _sha256(summary_path), "test_split_accessed": False}  # Bind exact assembly outputs and anti-leakage state.
    receipt_path = bridge.write_json(formal_root / "PARALLEL_VALIDATION_ASSEMBLY.json", receipt)  # Publish the operational assembly receipt last.
    return {**receipt, "receipt_path": str(receipt_path), "training_summary": str(summary_path)}  # Return concise exact terminal evidence.


def _parser() -> argparse.ArgumentParser:  # Build the location-only operational command surface.
    parser = argparse.ArgumentParser(description="Shard or assemble frozen supervised validation without changing point mechanics.")  # Create a self-describing parser.
    subparsers = parser.add_subparsers(dest="command", required=True)  # Require an explicit validate or assemble operation.
    validate = subparsers.add_parser("validate")  # Define one independent candidate-grid execution.
    validate.add_argument("--seed", type=int, choices=SEEDS, required=True)  # Select only a preregistered candidate.
    validate.add_argument("--output", type=Path, required=True)  # Select a pristine external shard root.
    validate.add_argument("--formal-root", type=Path, required=True)  # Locate completed expert and candidate artifacts.
    validate.add_argument("--reference-root", type=Path, required=True)  # Locate authenticated validation Reference B caches.
    validate.add_argument("--manifest", type=Path, required=True)  # Select the exact checksummed manifest.
    assemble_parser = subparsers.add_parser("assemble")  # Define exact three-grid publication and checkpoint selection.
    assemble_parser.add_argument("--formal-root", type=Path, required=True)  # Select the canonical supervised artifact root.
    assemble_parser.add_argument("--shard-root", type=Path, required=True)  # Locate two completed external seed roots.
    assemble_parser.add_argument("--manifest", type=Path, required=True)  # Select the exact checksummed manifest for the final receipt.
    return parser  # Return the complete operation-and-location-only parser.


def main() -> int:  # Execute one requested operational phase and report strict JSON.
    args = _parser().parse_args()  # Parse explicit operation and artifact locations.
    if args.command == "validate":  # Dispatch one complete independent seed grid.
        result = validate_seed(args.seed, args.output.resolve(), args.formal_root.resolve(), args.reference_root.resolve(), args.manifest.resolve())  # Execute unchanged point mechanics under disjoint output.
    else:  # Dispatch exact three-grid assembly after all shards are complete.
        result = assemble(args.formal_root.resolve(), args.shard_root.resolve(), args.manifest.resolve())  # Recompute scores, select, copy, and summarize once.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit the finite terminal receipt for automation.
    return 0  # Signal successful complete sharding or assembly.


if __name__ == "__main__":  # Execute only when launched as the retained operational artifact.
    raise SystemExit(main())  # Propagate the explicit status to the shell.
