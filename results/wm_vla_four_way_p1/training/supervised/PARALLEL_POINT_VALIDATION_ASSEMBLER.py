#!/usr/bin/env python3
"""Assemble one supervised seed from complete sequential points and pristine point shards."""  # State the retained publication-only responsibility precisely.
from __future__ import annotations  # Postpone annotation evaluation for the repository runtime.

import argparse  # Parse one explicit immutable seed-assembly assignment.
import json  # Read strict point evidence and print finite plans or receipts.
from pathlib import Path  # Resolve canonical and external evidence locations portably.
import shutil  # Copy complete point directories without changing their bytes.
import sys  # Import this exact checkout and return explicit process status.
from typing import Any  # Annotate bounded heterogeneous JSON records.

ROOT = Path(__file__).resolve().parents[4]  # Recover the repository root from this retained campaign-evidence location.
sys.path.insert(0, str(ROOT))  # Import the exact checked-out implementation without another installation.

import visionamr.baselines.bridge_supervised as bridge  # Reuse the production score, hashes, and atomic JSON boundary unchanged.
from visionamr.bridge_case_manifest import load_case_manifest  # Authenticate the exact frozen manifest before defining the grid.

PROTOCOL_ID = "WMVLA-4WAY-P1"  # Bind every plan and receipt to the sole formal campaign.
SEEDS = (20260831, 20260832, 20260833)  # Restrict assembly to the three preregistered supervised candidates.
BUDGETS = (30000, 60000, 120000)  # Restrict assembly to the three preregistered equation budgets.
POINT_RECEIPT_SCHEMA = "wmvla-four-way-supervised-validation-point-shard-v1"  # Require the exact worker receipt contract before copying a point.


def _read_json(path: Path) -> dict[str, Any]:  # Load one strict top-level JSON object from bounded evidence.
    payload = json.loads(path.read_text(encoding="utf-8"))  # Decode the complete UTF-8 file without partial recovery.
    if not isinstance(payload, dict):  # Require named fields at every assembly boundary.
        raise ValueError(f"JSON artifact is not an object: {path}")  # Reject arrays or scalars before publication.
    return payload  # Return the validated top-level mapping.


def _candidate(formal_root: Path, seed: int) -> dict[str, Any]:  # Resolve and rehash one exact preregistered candidate.
    summary_path = formal_root / "network_training_summary.json"  # Resolve the completed three-candidate training inventory.
    summary = _read_json(summary_path)  # Load candidate identities without touching validation or test artifacts.
    matches = [dict(item) for item in summary.get("candidates", []) if isinstance(item, dict) and int(item.get("seed", -1)) == seed]  # Match only the assigned seed without ranking.
    if len(matches) != 1:  # Require one unambiguous candidate receipt.
        raise ValueError(f"network summary does not contain exactly one candidate for seed {seed}")  # Reject missing or duplicate candidate identities.
    candidate = matches[0]  # Read the sole inventory candidate.
    model_path = Path(str(candidate["model_path"]))  # Resolve the exact production-recorded checkpoint path.
    if not model_path.is_file() or bridge.file_sha256(model_path) != str(candidate["model_sha256"]):  # Recompute checkpoint byte identity before assembly.
        raise ValueError(f"candidate model hash mismatch for seed {seed}")  # Reject evidence bound to altered weights.
    if str(candidate.get("protocol_id")) != PROTOCOL_ID:  # Require this campaign's model inventory.
        raise ValueError(f"candidate protocol mismatch for seed {seed}")  # Reject a checkpoint imported from another experiment.
    return candidate  # Return the exact verified candidate receipt.


def _validation_case_ids(manifest_path: Path) -> tuple[str, ...]:  # Recover the exact eight-case development grid from the checksummed manifest.
    manifest = load_case_manifest(manifest_path, verify_checksum=True)  # Authenticate exact manifest bytes and its sidecar checksum.
    cases = manifest.get("cases")  # Read the validated manifest case container once.
    if not isinstance(cases, list):  # Require the expected ordered collection shape.
        raise ValueError("manifest cases must be a list")  # Reject malformed or partial manifests.
    case_ids = tuple(sorted(str(case["case_id"]) for case in cases if isinstance(case, dict) and case.get("split") == "validation"))  # Select only development-validation identifiers in stable order.
    if len(case_ids) != 8 or len(set(case_ids)) != 8:  # Require the exact preregistered validation cardinality without duplicates.
        raise ValueError("manifest must contain exactly eight unique validation cases")  # Stop before any incomplete-grid assembly.
    return case_ids  # Return the exact immutable case-ID tuple.


def _point_key(case_id: str, budget: int) -> tuple[str, int]:  # Normalize one case-budget identity for set and map operations.
    return str(case_id), int(budget)  # Return the immutable two-field key.


def _directory_inventory(point_dir: Path) -> list[dict[str, Any]]:  # Hash every regular file in one bounded point directory for exact copy verification.
    inventory: list[dict[str, Any]] = []  # Accumulate stable relative paths, sizes, and complete-byte identities.
    for path in sorted(candidate for candidate in point_dir.rglob("*") if candidate.is_file() or candidate.is_symlink()):  # Enumerate every file-like entry without following an unreviewed link silently.
        if path.is_symlink() or not path.is_file():  # Require ordinary files throughout retained point evidence.
            raise ValueError(f"validation point contains a non-regular file: {path}")  # Reject link substitution or another unsupported filesystem entry.
        inventory.append({"path": path.relative_to(point_dir).as_posix(), "size_bytes": path.stat().st_size, "sha256": bridge.file_sha256(path)})  # Bind the exact portable relative name, length, and bytes.
    return inventory  # Return the complete stable directory manifest for receipt and destination comparison.


def _validate_point_directory(point_dir: Path, seed: int, case_id: str, budget: int, model_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:  # Authenticate one complete production point directory.
    result_path = point_dir / "validation_result.json"  # Resolve the atomic production point commit marker.
    records_path = point_dir / "records.json"  # Resolve the counted-solve evidence written before the commit marker.
    if not point_dir.is_dir() or not result_path.is_file() or not records_path.is_file():  # Require the complete directory and both mandatory immediate artifacts.
        raise ValueError(f"partial validation point directory: {point_dir}")  # Reject any interrupted point rather than treating it as missing or failed.
    row = _read_json(result_path)  # Load the exact success or retained typed-failure result.
    records = _read_json(records_path)  # Load the complete counted-solve container.
    if int(row.get("seed", -1)) != seed or str(row.get("case_id")) != case_id or int(row.get("equation_budget", -1)) != budget or row.get("split") != "validation":  # Require exact tuple and split identity.
        raise ValueError(f"validation point identity mismatch: {result_path}")  # Reject cross-seed, cross-case, cross-budget, or blind evidence.
    if str(row.get("model_sha256")) != model_sha256:  # Require the exact assigned checkpoint on every point.
        raise ValueError(f"validation point model mismatch: {result_path}")  # Reject mixed checkpoint grids.
    if row.get("status") not in ("ok", "failed"):  # Accept only production success or retained numerical failure.
        raise ValueError(f"validation point has invalid status: {result_path}")  # Reject incomplete or invented terminal states.
    if int(row.get("expedited_reference_levels", -1)) != 2:  # Require the sole user-authorized nonblocking reference depth.
        raise ValueError(f"validation point reference policy mismatch: {result_path}")  # Reject denominator-policy drift within the seed grid.
    record_rows = records.get("records")  # Read the production counted-solve list once.
    if not isinstance(record_rows, list) or len(record_rows) != int(row.get("real_solve_count", -1)):  # Require exact agreement between immediate row and counted-solve evidence.
        raise ValueError(f"validation point solve count mismatch: {records_path}")  # Reject truncated or cross-point records.
    if row.get("status") == "ok" and len(record_rows) != 2:  # Enforce the exact successful probe-plus-one-deployment contract.
        raise ValueError(f"successful validation point does not contain two solves: {records_path}")  # Reject scientifically incomplete successful rows.
    evidence = {"validation_result_sha256": bridge.file_sha256(result_path), "records_sha256": bridge.file_sha256(records_path), "point_tree_sha256": bridge.canonical_json_sha256(_directory_inventory(point_dir)), "status": str(row["status"]), "online_wall_s": float(row.get("online_wall_s", 0.0))}  # Bind the complete directory, two immediate artifacts, and bounded cost fields.
    return row, evidence  # Return the validated row and exact file identities.


def _reject_partial_or_foreign_canonical(canonical: Path, expected_keys: set[tuple[str, int]]) -> set[tuple[str, int]]:  # Reject unregistered partials while identifying solver-scratch-only interrupted locations explicitly.
    stale_scratch_keys: set[tuple[str, int]] = set()  # Track exact registered points containing no terminal evidence and only ignored solver scratch.
    if not canonical.is_dir():  # Require the existing-root layout promised by the caller.
        raise FileNotFoundError(f"canonical seed directory is missing: {canonical}")  # Stop instead of silently creating an ambiguous source root.
    if any(canonical.rglob("*.tmp")):  # Treat any interrupted atomic-write temporary as partial evidence.
        raise ValueError(f"canonical seed directory contains temporary files: {canonical}")  # Require explicit quarantine before assembly.
    expected_case_ids = {case_id for case_id, _budget in expected_keys}  # Recover the exact registered case directory set.
    for case_dir in sorted(path for path in canonical.glob("BGD-*") if path.is_dir()):  # Inspect every visible validation-case directory.
        case_id = case_dir.name  # Recover the immutable case identifier from canonical layout.
        if case_id not in expected_case_ids:  # Reject even an empty foreign case directory.
            raise ValueError(f"foreign case directory in canonical seed: {case_dir}")  # Require explicit quarantine of unregistered evidence.
        for budget_dir in sorted(path for path in case_dir.glob("B*") if path.is_dir()):  # Inspect every visible equation-budget directory.
            try:  # Parse only the canonical B<integer> naming convention.
                budget = int(budget_dir.name[1:])  # Recover the exact integer budget from the directory name.
            except ValueError as error:  # Surface malformed directories without ignoring them.
                raise ValueError(f"foreign budget directory in canonical seed: {budget_dir}") from error  # Require explicit quarantine of unregistered evidence.
            if _point_key(case_id, budget) not in expected_keys:  # Reject any case or budget outside the frozen validation product.
                raise ValueError(f"foreign point directory in canonical seed: {budget_dir}")  # Prevent extra-point selection or ambiguous provenance.
    for case_id, budget in sorted(expected_keys):  # Inspect every registered location for an interrupted point.
        point_dir = canonical / case_id / f"B{budget}"  # Resolve the sole canonical directory for this tuple.
        if point_dir.exists() and (not (point_dir / "validation_result.json").is_file() or not (point_dir / "records.json").is_file()):  # Distinguish a partial directory from a genuinely absent point.
            entries = tuple(path for path in point_dir.rglob("*") if path.is_file() or path.is_symlink())  # Inventory every file-like remnant before classifying interrupted scratch.
            if any(path.is_symlink() or path.relative_to(point_dir).parts[0] != "solves" for path in entries):  # Allow only ordinary files under the ignored solver-scratch subtree.
                raise ValueError(f"canonical seed contains a non-scratch partial point: {point_dir}")  # Keep arbitrary or terminal-looking partial evidence fail-closed.
            stale_scratch_keys.add(_point_key(case_id, budget))  # Disclose this exact synchronized interrupted-solver remnant for verified replacement.
    return stale_scratch_keys  # Return only registered scratch-only locations that require exact source-tree replacement.


def _existing_points(canonical: Path, seed: int, expected_keys: set[tuple[str, int]], model_sha256: str, stale_scratch_keys: set[tuple[str, int]]) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:  # Inventory all complete pre-cutover canonical points while excluding disclosed scratch-only remnants.
    rows: dict[tuple[str, int], dict[str, Any]] = {}  # Index validated production rows by immutable point key.
    sources: dict[tuple[str, int], dict[str, Any]] = {}  # Index exact sequential-or-helper source identities by the same key.
    for case_id, budget in sorted(expected_keys):  # Inspect the full registered grid without result-dependent selection.
        point_dir = canonical / case_id / f"B{budget}"  # Resolve the sole canonical location for this tuple.
        if _point_key(case_id, budget) in stale_scratch_keys:  # Treat only preauthenticated solver-scratch remnants as scientifically absent points.
            continue  # Require the receipt-bound external production point to replace this incomplete location.
        if not point_dir.exists():  # Leave genuinely absent points for external shard fulfillment.
            continue  # Preserve missing status without creating any file or directory.
        row, evidence = _validate_point_directory(point_dir, seed, case_id, budget, model_sha256)  # Authenticate the complete existing point.
        key = _point_key(case_id, budget)  # Normalize the exact existing tuple once.
        rows[key] = row  # Preserve the authoritative pre-cutover production result.
        sources[key] = {"source_kind": "existing_sequential_or_seed_helper", "source_point_dir": str(point_dir), **evidence}  # Bind its original canonical path and exact hashes.
    return rows, sources  # Return complete existing rows and their source inventory.


def _point_shards(point_shard_root: Path, seed: int, manifest_sha256: str, model_sha256: str, expected_keys: set[tuple[str, int]], occupied_keys: set[tuple[str, int]]) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:  # Authenticate all and only the missing assigned point shards.
    if not point_shard_root.is_dir():  # Require an explicit external shard collection root.
        raise FileNotFoundError(f"point shard root is missing: {point_shard_root}")  # Stop before an incomplete-grid assembly.
    rows: dict[tuple[str, int], dict[str, Any]] = {}  # Index validated external rows by immutable point key.
    sources: dict[tuple[str, int], dict[str, Any]] = {}  # Index exact receipt-bound external sources by the same key.
    for receipt_path in sorted(point_shard_root.rglob("POINT_SHARD_RECEIPT.json")):  # Discover only terminal worker receipts recursively.
        receipt = _read_json(receipt_path)  # Load the complete worker receipt before considering its point.
        if int(receipt.get("seed", -1)) != seed:  # Ignore receipts belonging to another preregistered seed collection.
            continue  # Preserve per-seed assembly isolation without mutating other evidence.
        case_id = str(receipt.get("case_id"))  # Recover the immutable case assignment from the receipt.
        budget = int(receipt.get("equation_budget", -1))  # Recover the immutable budget assignment from the receipt.
        key = _point_key(case_id, budget)  # Normalize the complete point identity once.
        if key not in expected_keys:  # Reject any post-registration case or budget for this seed.
            raise ValueError(f"point receipt is outside the frozen grid: {receipt_path}")  # Prevent extra-point selection.
        if key in occupied_keys:  # Refuse even byte-identical duplication of an existing complete point.
            raise ValueError(f"point receipt overlaps an existing complete point: {receipt_path}")  # Preserve predeclared source authority without choosing among outcomes.
        if key in rows:  # Require exactly one terminal receipt for each missing tuple.
            raise ValueError(f"duplicate point receipts for {key}")  # Refuse result-dependent duplicate selection.
        if receipt.get("schema") != POINT_RECEIPT_SCHEMA or receipt.get("protocol_id") != PROTOCOL_ID or receipt.get("TEST_NOT_RUN") is not True:  # Require the exact worker and anti-leakage contract.
            raise ValueError(f"point receipt contract mismatch: {receipt_path}")  # Reject foreign, stale, or test-exposed shards.
        if str(receipt.get("manifest_sha256")) != manifest_sha256 or str(receipt.get("model_sha256")) != model_sha256:  # Require exact common manifest and checkpoint identities.
            raise ValueError(f"point receipt input hash mismatch: {receipt_path}")  # Reject a mixed-input validation grid.
        policy = receipt.get("reference_policy")  # Read the explicit denominator authorization once.
        if not isinstance(policy, dict) or policy.get("allow_unqualified_references") is not True or int(policy.get("expedited_reference_levels", -1)) != 2:  # Require the sole user-authorized nonblocking policy.
            raise ValueError(f"point receipt reference policy mismatch: {receipt_path}")  # Reject denominator-policy drift.
        shard_root = receipt_path.parent.resolve()  # Resolve the pristine output root that owns this terminal receipt.
        if Path(str(receipt.get("output"))).resolve() != shard_root:  # Require receipt path and declared output identity to agree.
            raise ValueError(f"point receipt output path mismatch: {receipt_path}")  # Reject moved or spliced receipt evidence.
        point_dir = shard_root / "validation" / f"seed_{seed}" / case_id / f"B{budget}"  # Resolve the sole production point inside this shard.
        row, evidence = _validate_point_directory(point_dir, seed, case_id, budget, model_sha256)  # Authenticate immediate production artifacts and row semantics.
        if str(receipt.get("validation_result_sha256")) != str(evidence["validation_result_sha256"]) or str(receipt.get("records_sha256")) != str(evidence["records_sha256"]):  # Recompute both receipt-bound output identities.
            raise ValueError(f"point receipt output hash mismatch: {receipt_path}")  # Reject altered or cross-shard point evidence.
        rows[key] = row  # Preserve the sole authenticated external production result.
        sources[key] = {"source_kind": "pristine_parallel_point_shard", "source_point_dir": str(point_dir), "point_receipt_path": str(receipt_path), "point_receipt_sha256": bridge.file_sha256(receipt_path), **evidence}  # Bind exact source paths, receipt, outputs, and cost.
    return rows, sources  # Return all authenticated external rows and source identities for this seed.


def _ordered_sources(source_map: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:  # Serialize source provenance independently of discovery timing.
    return [{"case_id": case_id, "equation_budget": budget, **source_map[(case_id, budget)]} for case_id, budget in sorted(source_map)]  # Return stable complete point provenance in case-budget order.


def assemble_seed(seed: int, existing_root: Path, point_shard_root: Path, formal_root: Path, manifest_path: Path, dry_run: bool) -> dict[str, Any]:  # Validate or publish one exact mixed-source seed grid.
    if seed not in SEEDS:  # Restrict assembly to the preregistered candidate set.
        raise ValueError(f"seed must be one of {SEEDS}")  # Reject any post-registration candidate.
    if point_shard_root == existing_root or point_shard_root.is_relative_to(existing_root):  # Keep external shards disjoint from the canonical publication root.
        raise ValueError("point shard root must be external to the existing canonical root")  # Prevent recursive receipt discovery and source overwrite.
    candidate = _candidate(formal_root, seed)  # Rehash the exact candidate before accepting point evidence.
    model_sha256 = str(candidate["model_sha256"])  # Freeze the common checkpoint identity for all subsequent guards.
    case_ids = _validation_case_ids(manifest_path)  # Recover the exact eight-case development grid.
    expected_keys = {_point_key(case_id, budget) for case_id in case_ids for budget in BUDGETS}  # Build the exact preregistered 8-by-3 Cartesian product.
    manifest_sha256 = bridge.file_sha256(manifest_path)  # Bind every point receipt to exact manifest bytes.
    canonical = existing_root / "validation" / f"seed_{seed}"  # Resolve the sole canonical seed evidence directory.
    score_path = canonical / "score.json"  # Resolve the sole canonical production-compatible candidate score.
    seed_rows_path = canonical / "validation_rows.json"  # Resolve the stable per-seed aggregate used by the shard receipt.
    shard_receipt_path = existing_root / "PARALLEL_SHARD_RECEIPT.json"  # Resolve the terminal helper-root receipt consumed by later publication.
    if score_path.exists() or seed_rows_path.exists() or shard_receipt_path.exists():  # Preserve one-shot score, aggregate, and receipt publication.
        raise FileExistsError(f"seed score, rows, or shard receipt already exists under {existing_root}")  # Refuse overwrite or second assembly.
    stale_scratch_keys = _reject_partial_or_foreign_canonical(canonical, expected_keys)  # Reject foreign partials and disclose exact synchronized solver-scratch-only remnants.
    existing_rows, existing_sources = _existing_points(canonical, seed, expected_keys, model_sha256, stale_scratch_keys)  # Authenticate every complete pre-cutover point while excluding disclosed scratch remnants.
    shard_rows, shard_sources = _point_shards(point_shard_root, seed, manifest_sha256, model_sha256, expected_keys, set(existing_rows))  # Authenticate unique receipts for only missing tuples.
    if not existing_rows or not shard_rows:  # Require the execution-mode declaration to describe genuinely mixed provenance.
        raise ValueError("mixed assembly requires at least one existing point and one pristine point shard")  # Route pure sequential or pure point execution through a semantically accurate path.
    combined_rows = {**existing_rows, **shard_rows}  # Combine disjoint authoritative sources without overwriting any point.
    combined_sources = {**existing_sources, **shard_sources}  # Combine matching exact provenance under the same disjointness guarantee.
    if set(combined_rows) != expected_keys or set(combined_sources) != expected_keys:  # Require exactly all and only the twenty-four registered points.
        missing = sorted(expected_keys - set(combined_rows))  # Compute the exact absent tuples for a finite diagnostic.
        raise ValueError(f"seed grid is incomplete; missing point receipts: {missing}")  # Reject favorable omission or partial publication.
    ordered_rows = [combined_rows[key] for key in sorted(expected_keys)]  # Make scoring independent of process completion and filesystem discovery order.
    score = bridge.validation_score(ordered_rows, expected_cases=8, budgets=BUDGETS, failure_error=bridge.VALIDATION_FAILURE_ERROR)  # Apply the unchanged preregistered complete-grid score.
    score["model_path"] = str(candidate["model_path"])  # Bind the score to the exact trained checkpoint path.
    score["model_sha256"] = model_sha256  # Bind the score to the exact trained checkpoint bytes.
    source_rows = _ordered_sources(combined_sources)  # Serialize all twenty-four source identities in stable point order.
    validation_rows_payload = {"schema": bridge.SUPERVISED_SCHEMA, "protocol_id": PROTOCOL_ID, "seed": seed, "rows": ordered_rows}  # Assemble the production-compatible per-seed row evidence.
    validation_online_compute_wall_s = float(sum(float(row.get("online_wall_s", 0.0)) for row in ordered_rows))  # Sum exact measured deployment costs without hiding parallel compute consumption.
    plan = {"schema": "wmvla-four-way-supervised-validation-mixed-seed-assembly-plan-v1", "protocol_id": PROTOCOL_ID, "dry_run": True, "execution_mode": "mixed_sequential_and_parallel_points", "TEST_NOT_RUN": True, "seed": seed, "validation_case_count": 8, "validation_point_count": len(ordered_rows), "existing_point_count": len(existing_rows), "parallel_point_count": len(shard_rows), "stale_solver_scratch_points": [{"case_id": key[0], "equation_budget": key[1]} for key in sorted(stale_scratch_keys)], "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256, "model_path": str(candidate["model_path"]), "model_sha256": model_sha256, "reference_policy": {"allow_unqualified_references": True, "expedited_reference_levels": 2}, "validation_online_compute_wall_s": validation_online_compute_wall_s, "score": score, "sources": source_rows, "source_inventory_sha256": bridge.canonical_json_sha256(source_rows), "would_copy": [str(shard_sources[key]["source_point_dir"]) for key in sorted(shard_sources)], "canonical_seed_directory": str(canonical), "test_split_accessed": False}  # Bind the complete no-write publication decision, scratch disclosure, and anti-leakage state.
    if dry_run:  # Stop after complete source, grid, hash, and score verification when explicitly requested.
        return plan  # Return the finite no-write assembly plan.
    for key in sorted(shard_sources):  # Publish each preverified missing point in stable case-budget order.
        source_dir = Path(str(shard_sources[key]["source_point_dir"]))  # Resolve the exact receipt-bound external source directory.
        destination = canonical / key[0] / f"B{key[1]}"  # Resolve the sole absent canonical destination for this tuple.
        if destination.exists() and key not in stale_scratch_keys:  # Recheck no-overwrite immediately before each copy while allowing only disclosed scratch replacement.
            raise FileExistsError(f"canonical point appeared during assembly: {destination}")  # Refuse concurrent or duplicate publication.
        shutil.copytree(source_dir, destination, copy_function=shutil.copy2, dirs_exist_ok=key in stale_scratch_keys)  # Copy the complete production point, merging only over disclosed nonterminal solver scratch.
        if _directory_inventory(destination) != _directory_inventory(source_dir):  # Require exact full-tree equality after a pristine copy or disclosed scratch replacement.
            raise RuntimeError(f"canonical point tree differs from its receipt-bound source: {destination}")  # Stop before scoring if any stale or synchronized file survived publication.
    final_rows: list[dict[str, Any]] = []  # Rebuild the complete grid only from canonical post-copy evidence.
    for case_id, budget in sorted(expected_keys):  # Reauthenticate every canonical point after all copies.
        row, _evidence = _validate_point_directory(canonical / case_id / f"B{budget}", seed, case_id, budget, model_sha256)  # Verify exact tuple, model, policy, records, and file completeness again.
        final_rows.append(row)  # Retain the canonical row for exact score agreement.
    final_score = bridge.validation_score(final_rows, expected_cases=8, budgets=BUDGETS, failure_error=bridge.VALIDATION_FAILURE_ERROR)  # Recompute the preregistered score solely from canonical evidence.
    if final_score.get("selection_key") != score.get("selection_key") or int(final_score.get("seed", -1)) != seed:  # Require pre-copy and post-copy score identity.
        raise RuntimeError("canonical post-copy score differs from the preverified assembly score")  # Stop before score publication on any source mutation or copy corruption.
    bridge.write_json(seed_rows_path, validation_rows_payload)  # Publish the stable complete per-seed row aggregate atomically.
    bridge.write_json(score_path, score)  # Publish the production-compatible candidate score atomically after all points are canonical.
    receipt = {"schema": "wmvla-four-way-supervised-validation-shard-v1", "protocol_id": PROTOCOL_ID, "execution_mode": "mixed_sequential_and_parallel_points", "seed": seed, "TEST_NOT_RUN": True, "validation_case_count": 8, "validation_point_count": len(final_rows), "existing_point_count": len(existing_rows), "parallel_point_count": len(shard_rows), "stale_solver_scratch_points_replaced": [{"case_id": key[0], "equation_budget": key[1]} for key in sorted(stale_scratch_keys)], "model_sha256": model_sha256, "manifest_sha256": manifest_sha256, "score_sha256": bridge.file_sha256(score_path), "validation_rows_sha256": bridge.file_sha256(seed_rows_path), "source_inventory_sha256": bridge.canonical_json_sha256(source_rows), "sources": source_rows, "validation_wall_s": validation_online_compute_wall_s, "validation_online_compute_wall_s": validation_online_compute_wall_s, "allow_unqualified_references": True, "expedited_reference_levels": 2, "test_split_accessed": False, "test_results_used": False}  # Bind exact mixed provenance, replaced scratch disclosure, complete-grid outputs, cost, denominator policy, and anti-leakage declarations.
    receipt_path = bridge.write_json(shard_receipt_path, receipt)  # Publish the terminal helper-root receipt last as the sole assembly completion marker.
    return {**receipt, "receipt_path": str(receipt_path), "score_path": str(score_path)}  # Return concise exact terminal evidence to the coordinator.


def _parser() -> argparse.ArgumentParser:  # Build the explicit one-seed publication command surface.
    parser = argparse.ArgumentParser(description="Assemble one supervised seed from complete canonical points and unique pristine point shards.")  # Create a self-describing campaign-specific CLI.
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)  # Select one preregistered candidate grid.
    parser.add_argument("--existing-root", type=Path, required=True)  # Locate the helper or formal root containing pre-cutover canonical points.
    parser.add_argument("--point-shard-root", type=Path, required=True)  # Locate unique terminal receipts for every missing point.
    parser.add_argument("--formal-root", type=Path, required=True)  # Locate the completed candidate-training inventory and checkpoints.
    parser.add_argument("--manifest", type=Path, required=True)  # Select the exact checksummed frozen case manifest.
    parser.add_argument("--dry-run", action="store_true")  # Validate all sources and recompute the score without copying or writing.
    return parser  # Return the complete location-and-seed-only parser.


def main() -> int:  # Validate or publish one requested mixed-source seed grid.
    args = _parser().parse_args()  # Parse every explicit seed and artifact location.
    result = assemble_seed(args.seed, args.existing_root.resolve(), args.point_shard_root.resolve(), args.formal_root.resolve(), args.manifest.resolve(), bool(args.dry_run))  # Authenticate and dispatch exactly one requested seed assembly.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit a finite machine-readable plan or terminal receipt.
    return 0  # Signal successful no-write verification or completed seed publication.


if __name__ == "__main__":  # Execute only when launched as the retained campaign assembler.
    raise SystemExit(main())  # Propagate the explicit status to the shell.
