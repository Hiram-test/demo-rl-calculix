#!/usr/bin/env python3
"""Execute one pristine supervised validation point through the unchanged production loop."""  # State the retained operational-only responsibility precisely.
from __future__ import annotations  # Postpone annotation evaluation for the repository runtime.

import argparse  # Parse one explicit immutable validation-point assignment.
import json  # Read strict production inventories and print finite receipts.
import os  # Bind execution to the campaign's single-threaded native environment.
from pathlib import Path  # Resolve formal inputs and pristine external shard roots portably.
import sys  # Import this exact checkout and return explicit process status.
from typing import Any  # Annotate bounded heterogeneous JSON objects.

ROOT = Path(__file__).resolve().parents[4]  # Recover the repository root from this retained campaign-evidence location.
sys.path.insert(0, str(ROOT))  # Import the exact checked-out implementation without another installation.

import visionamr.baselines.bridge_supervised as bridge  # Reuse the unchanged production validation loop and all point mechanics.
from visionamr.bridge_case_manifest import load_case_manifest, problem_from_case  # Authenticate and reconstruct only one manifest-authorized validation case.
from visionamr.vla.four_way_references import verify_reference_cache  # Reverify the exact common validation denominator without generating a new reference.

PROTOCOL_ID = "WMVLA-4WAY-P1"  # Bind every plan and receipt to the sole formal campaign.
SEEDS = (20260831, 20260832, 20260833)  # Restrict assignments to the three preregistered supervised candidates.
BUDGETS = (30000, 60000, 120000)  # Restrict assignments to the three preregistered equation budgets.
THREAD_ENV_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GMSH_NUM_THREADS")  # Name every campaign-controlled native thread setting.


def _read_json(path: Path) -> dict[str, Any]:  # Load one strict top-level JSON object from a bounded production artifact.
    payload = json.loads(path.read_text(encoding="utf-8"))  # Decode the complete UTF-8 file without partial recovery.
    if not isinstance(payload, dict):  # Require named fields at every orchestration boundary.
        raise ValueError(f"JSON artifact is not an object: {path}")  # Reject arrays or scalars before native work.
    return payload  # Return the validated top-level mapping.


def _single_score(scores: list[dict[str, Any]]) -> dict[str, Any]:  # Finalize the sole operational score without cross-candidate selection.
    if len(scores) != 1:  # Require exactly one result from this one-point process.
        raise ValueError("point shard must produce exactly one operational score")  # Refuse omission or accidental multi-candidate execution.
    return dict(scores[0])  # Return a copy compatible with the unchanged production finalization path.


def _thread_environment() -> dict[str, str]:  # Authenticate the exact single-threaded campaign environment.
    values = {name: str(os.environ.get(name, "")) for name in THREAD_ENV_NAMES}  # Capture every required setting without inventing defaults.
    if any(value != "1" for value in values.values()):  # Require the same all-one policy used by formal training and later freeze operations.
        raise ValueError(f"point validation requires all thread settings equal to 1: {values}")  # Stop before Gmsh or CalculiX can execute under another resource contract.
    return values  # Return the exact environment for the dry-run plan or terminal receipt.


def _candidate(formal_root: Path, seed: int) -> dict[str, Any]:  # Resolve and rehash one exact preregistered trained candidate.
    summary_path = formal_root / "network_training_summary.json"  # Resolve the completed three-candidate training inventory.
    summary = _read_json(summary_path)  # Load candidate identities without touching validation or test artifacts.
    matches = [dict(item) for item in summary.get("candidates", []) if isinstance(item, dict) and int(item.get("seed", -1)) == seed]  # Match only the explicitly assigned seed without ranking.
    if len(matches) != 1:  # Require one unambiguous candidate receipt.
        raise ValueError(f"network summary does not contain exactly one candidate for seed {seed}")  # Reject missing or duplicate candidate identities.
    candidate = matches[0]  # Read the sole authenticated-by-inventory candidate.
    model_path = Path(str(candidate["model_path"]))  # Resolve the exact production-recorded checkpoint path.
    if not model_path.is_file() or bridge.file_sha256(model_path) != str(candidate["model_sha256"]):  # Recompute exact checkpoint byte identity before any validation action.
        raise ValueError(f"candidate model hash mismatch for seed {seed}")  # Stop before spending validation truth on altered weights.
    if str(candidate.get("protocol_id")) != PROTOCOL_ID:  # Require the model inventory to belong to this campaign.
        raise ValueError(f"candidate model protocol mismatch for seed {seed}")  # Reject a checkpoint imported from another experiment.
    return candidate  # Return the exact verified candidate receipt.


def _validation_case(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:  # Select one exact validation case without exposing train or blind cases to execution.
    cases = manifest.get("cases")  # Read the checksummed manifest case container once.
    if not isinstance(cases, list):  # Require the validated manifest's expected collection shape.
        raise ValueError("manifest cases must be a list")  # Reject malformed or partial manifests.
    matches = [dict(case) for case in cases if isinstance(case, dict) and str(case.get("case_id")) == case_id]  # Match only the explicit immutable case identifier.
    if len(matches) != 1 or matches[0].get("split") != "validation":  # Require one and only one development-validation case.
        raise ValueError(f"case_id is not exactly one validation case: {case_id}")  # Forbid train, blind-test, missing, or duplicated assignments.
    return matches[0]  # Return the sole authorized case record.


def _ensure_external_pristine(output: Path, formal_root: Path) -> None:  # Enforce independent one-shot storage before execution.
    if output.exists():  # Treat any existing file or directory as prior state.
        raise FileExistsError(f"point shard output already exists: {output}")  # Refuse overwrite, implicit resume, or result-dependent replacement.
    if output == ROOT or output.is_relative_to(ROOT):  # Keep point shards outside the repository and formal campaign tree.
        raise ValueError(f"point shard output must be external to the repository: {output}")  # Prevent concurrent workers from contaminating freeze inputs.
    if output == formal_root or output.is_relative_to(formal_root):  # Defend the formal root separately if checkout topology changes later.
        raise ValueError(f"point shard output must be external to the formal root: {output}")  # Preserve pristine formal publication for a later audited assembler.


def _prepare(seed: int, case_id: str, budget: int, output: Path, formal_root: Path, reference_root: Path, manifest_path: Path, allow_unqualified_references: bool, expedited_reference_levels: int | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:  # Authenticate every immutable point input without solving.
    if seed not in SEEDS or budget not in BUDGETS:  # Restrict execution to the preregistered Cartesian product.
        raise ValueError(f"seed and budget must be in {SEEDS} and {BUDGETS}")  # Reject any post-registration operating point.
    if bool(allow_unqualified_references) != (expedited_reference_levels is not None):  # Preserve the production reference-policy pairing rule.
        raise ValueError("allow-unqualified and expedited levels must be supplied together")  # Stop before inspecting or executing an unauthorized denominator.
    if allow_unqualified_references and expedited_reference_levels != 2:  # Match the sole user-authorized rapid-execution amendment.
        raise ValueError("expedited reference levels must equal 2 when unqualified references are allowed")  # Reject another denominator depth before execution.
    _ensure_external_pristine(output, formal_root)  # Require a unique absent external output before any later native work.
    environment = _thread_environment()  # Authenticate all campaign-controlled native thread settings.
    manifest = load_case_manifest(manifest_path, verify_checksum=True)  # Verify exact manifest bytes and its sidecar checksum.
    case = _validation_case(manifest, case_id)  # Select only the explicitly assigned development-validation case.
    candidate = _candidate(formal_root, seed)  # Rehash the exact assigned checkpoint.
    problem = problem_from_case(case)  # Reconstruct the manifest-authorized geometry for reference authentication only.
    reference_verification = verify_reference_cache(reference_root, case_id=case_id, problem=problem, allow_unqualified=bool(allow_unqualified_references), expedited_levels=expedited_reference_levels)  # Reverify cached denominator integrity under the exact explicit policy without solving.
    reference_b_path = reference_root / case_id / "reference_B.json"  # Resolve the exact denominator record verified by the production loader.
    if not reference_b_path.is_file():  # Require a concrete immutable denominator artifact for receipt binding.
        raise FileNotFoundError(f"reference_B is missing for {case_id}: {reference_b_path}")  # Stop before native execution when evidence cannot be hashed.
    plan = {"schema": "wmvla-four-way-supervised-validation-point-plan-v1", "protocol_id": PROTOCOL_ID, "dry_run": True, "execution_mode": "pristine_external_single_point_production_loop", "TEST_NOT_RUN": True, "seed": seed, "case_id": case_id, "equation_budget": budget, "output": str(output), "manifest_path": str(manifest_path), "manifest_sha256": bridge.file_sha256(manifest_path), "model_path": str(candidate["model_path"]), "model_sha256": str(candidate["model_sha256"]), "reference_root": str(reference_root), "reference_b_sha256": bridge.file_sha256(reference_b_path), "reference_policy": {"allow_unqualified_references": bool(allow_unqualified_references), "expedited_reference_levels": expedited_reference_levels}, "reference_status": str(reference_verification["status"]), "reference_qualification": bool(reference_verification["qualification"]), "reference_authorization": reference_verification.get("authorization"), "thread_environment": environment, "test_split_accessed": False}  # Bind every immutable point input and anti-leakage declaration before execution.
    return case, candidate, reference_verification, plan  # Return authenticated inputs plus the finite no-solve plan.


def validate_point(seed: int, case_id: str, budget: int, output: Path, formal_root: Path, reference_root: Path, manifest_path: Path, ccx_timeout: float, allow_unqualified_references: bool, expedited_reference_levels: int | None, dry_run: bool) -> dict[str, Any]:  # Execute or dry-validate one exact seed-case-budget assignment.
    case, candidate, reference_verification, plan = _prepare(seed, case_id, budget, output, formal_root, reference_root, manifest_path, allow_unqualified_references, expedited_reference_levels)  # Authenticate all read-only inputs first.
    if not float(ccx_timeout) > 0.0:  # Require a usable operational timeout before native execution.
        raise ValueError("ccx timeout must be positive")  # Reject an immediately failing or undefined solver contract.
    if dry_run:  # Stop after complete input, timeout, and reference-cache validation when explicitly requested.
        return plan  # Return the finite no-write, no-solver execution plan.
    output.mkdir(parents=True, exist_ok=False)  # Claim the absent external shard atomically so concurrent launchers cannot share one output.
    original_seeds = bridge.NETWORK_SEEDS  # Preserve the reviewed three-candidate module constant for guaranteed restoration.
    original_validation_count = bridge.EXPECTED_SPLIT_COUNTS["validation"]  # Preserve the reviewed eight-case cardinality for guaranteed restoration.
    original_score = bridge.validation_score  # Preserve the reviewed complete-grid scoring function for scoped delegation and restoration.
    original_selector = bridge.select_validation_checkpoint  # Preserve the reviewed three-candidate selector for guaranteed restoration.
    settings = bridge.BridgeSupervisedConfig(network_seeds=(seed,), validation_budgets=(budget,))  # Limit only this isolated operational process to its preassigned single point.
    def point_score(rows: list[dict[str, Any]], *, expected_cases: int = 1, budgets: tuple[int, ...] = (budget,), failure_error: float = bridge.VALIDATION_FAILURE_ERROR) -> dict[str, Any]:  # Adapt only final operational completeness to one preassigned point.
        del expected_cases  # Ignore the production caller's literal full-grid cardinality inside this isolated shard only.
        del budgets  # Ignore the production caller's already-singleton settings argument in favor of the immutable CLI assignment.
        return original_score(rows, expected_cases=1, budgets=(budget,), failure_error=failure_error)  # Apply the unchanged metric penalties and aggregation to the one-point operational shard.
    bridge.NETWORK_SEEDS = (seed,)  # Limit the unchanged production seed loop and candidate-map guard to the assigned checkpoint.
    bridge.EXPECTED_SPLIT_COUNTS["validation"] = 1  # Limit the unchanged production case-cardinality guard to the assigned case.
    bridge.validation_score = point_score  # Limit only operational completeness while retaining the unchanged score implementation.
    bridge.select_validation_checkpoint = _single_score  # Finalize the sole shard candidate without claiming cross-candidate selection.
    try:  # Restore every reviewed module global even if native work or schema validation raises.
        selection, rows = bridge.validate_candidate_networks((case,), (candidate,), output, reference_root, settings, ccx_timeout=float(ccx_timeout), allow_unqualified_references=bool(allow_unqualified_references), expedited_reference_levels=expedited_reference_levels)  # Execute the unchanged production loop over exactly one preassigned point.
    finally:  # Reestablish exact reviewed process state before any success or failure leaves this function.
        bridge.NETWORK_SEEDS = original_seeds  # Restore the exact preregistered candidate set.
        bridge.EXPECTED_SPLIT_COUNTS["validation"] = original_validation_count  # Restore the exact eight-case development cardinality.
        bridge.validation_score = original_score  # Restore the exact complete-grid scoring function.
        bridge.select_validation_checkpoint = original_selector  # Restore the exact three-candidate checkpoint selector.
    if len(rows) != 1 or int(selection.get("selected_seed", -1)) != seed:  # Require one and only one correctly finalized operational point.
        raise RuntimeError("production point shard returned an unexpected row or seed")  # Refuse to emit a receipt for incomplete or cross-assignment output.
    row = dict(rows[0])  # Copy the sole production result for exact identity validation.
    if int(row.get("seed", -1)) != seed or str(row.get("case_id")) != case_id or int(row.get("equation_budget", -1)) != budget or row.get("split") != "validation":  # Verify the complete immutable assignment tuple after production execution.
        raise RuntimeError("production result identity differs from its point assignment")  # Stop before receipt publication on any routing error.
    if str(row.get("reference_status")) != str(reference_verification["status"]) or bool(row.get("reference_qualification")) != bool(reference_verification["qualification"]) or row.get("reference_authorization") != reference_verification.get("authorization"):  # Require the production row to use the denominator policy authenticated immediately before execution.
        raise RuntimeError("production result reference identity differs from preflight verification")  # Reject any denominator change or policy drift during the point.
    result_path = output / "validation" / f"seed_{seed}" / case_id / f"B{budget}" / "validation_result.json"  # Resolve the atomic production point commit marker.
    records_path = result_path.with_name("records.json")  # Resolve the complete counted-solve evidence written before the point marker.
    validation_rows_path = output / "validation_rows.json"  # Resolve the production aggregate containing exactly this sole row.
    if not result_path.is_file() or not records_path.is_file() or not validation_rows_path.is_file():  # Require every immediate production artifact before terminal receipt publication.
        raise FileNotFoundError("production point shard did not publish result, records, and validation rows")  # Reject an incomplete native or serialization boundary.
    reference_b_path = reference_root / case_id / "reference_B.json"  # Resolve the same verified denominator for terminal byte binding.
    if bridge.file_sha256(manifest_path) != str(plan["manifest_sha256"]) or bridge.file_sha256(Path(str(candidate["model_path"]))) != str(plan["model_sha256"]) or bridge.file_sha256(reference_b_path) != str(plan["reference_b_sha256"]):  # Rehash every immutable input after native work to detect concurrent mutation.
        raise RuntimeError("point input bytes changed during production execution")  # Refuse a terminal receipt that cannot bind one stable input set.
    receipt = {"schema": "wmvla-four-way-supervised-validation-point-shard-v1", "protocol_id": PROTOCOL_ID, "execution_mode": "pristine_external_single_point_production_loop", "production_entrypoint": "visionamr.baselines.bridge_supervised.validate_candidate_networks", "production_monkeypatch_scope": ["NETWORK_SEEDS", "EXPECTED_SPLIT_COUNTS.validation", "validation_score.completeness", "select_validation_checkpoint.cardinality"], "TEST_NOT_RUN": True, "seed": seed, "case_id": case_id, "equation_budget": budget, "status": str(row["status"]), "output": str(output), "manifest_path": str(manifest_path), "manifest_sha256": bridge.file_sha256(manifest_path), "model_path": str(candidate["model_path"]), "model_sha256": str(candidate["model_sha256"]), "reference_root": str(reference_root), "reference_b_sha256": bridge.file_sha256(reference_b_path), "reference_policy": {"allow_unqualified_references": bool(allow_unqualified_references), "expedited_reference_levels": expedited_reference_levels}, "reference_status": str(reference_verification["status"]), "reference_qualification": bool(reference_verification["qualification"]), "reference_authorization": reference_verification.get("authorization"), "validation_result_sha256": bridge.file_sha256(result_path), "records_sha256": bridge.file_sha256(records_path), "validation_rows_sha256": bridge.file_sha256(validation_rows_path), "thread_environment": _thread_environment(), "test_split_accessed": False, "test_results_used": False}  # Bind exact immutable inputs, production outputs, operational adaptation, resource policy, and anti-leakage state.
    receipt_path = bridge.write_json(output / "POINT_SHARD_RECEIPT.json", receipt)  # Publish the terminal receipt atomically only after the production point is complete.
    return {**receipt, "receipt_path": str(receipt_path)}  # Return concise exact terminal evidence to the worker launcher.


def _parser() -> argparse.ArgumentParser:  # Build the explicit immutable one-point command surface.
    parser = argparse.ArgumentParser(description="Execute one pristine external supervised validation point through the unchanged production loop.")  # Create a self-describing campaign-specific CLI.
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)  # Select one preregistered candidate checkpoint.
    parser.add_argument("--case-id", required=True)  # Select one exact manifest validation-case identifier.
    parser.add_argument("--budget", type=int, choices=BUDGETS, required=True)  # Select one preregistered equation budget.
    parser.add_argument("--output", type=Path, required=True)  # Select one absent external shard root unique to this point.
    parser.add_argument("--formal-root", type=Path, required=True)  # Locate the completed candidate-training inventory and checkpoint.
    parser.add_argument("--reference-root", type=Path, required=True)  # Locate the authenticated common validation references.
    parser.add_argument("--manifest", type=Path, required=True)  # Select the exact checksummed frozen case manifest.
    parser.add_argument("--ccx-timeout", type=float, default=1800.0)  # Bound each unchanged production CalculiX invocation operationally.
    parser.add_argument("--allow-unqualified-references", action="store_true")  # Activate only the explicit user-authorized nonblocking denominator policy.
    parser.add_argument("--expedited-reference-levels", type=int)  # Bind the exact amended two-level reference campaign when activated.
    parser.add_argument("--dry-run", action="store_true")  # Validate all immutable inputs and reference integrity without writing or solving.
    return parser  # Return the complete location-and-assignment-only parser.


def main() -> int:  # Execute the requested no-solve validation or sole native point.
    args = _parser().parse_args()  # Parse every explicit immutable assignment and artifact location.
    result = validate_point(args.seed, str(args.case_id), args.budget, args.output.resolve(), args.formal_root.resolve(), args.reference_root.resolve(), args.manifest.resolve(), args.ccx_timeout, bool(args.allow_unqualified_references), args.expedited_reference_levels, bool(args.dry_run))  # Authenticate and dispatch exactly one requested point operation.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit a finite machine-readable plan or terminal receipt.
    return 0  # Signal successful input validation or completed point evidence.


if __name__ == "__main__":  # Execute only when launched as the retained campaign worker.
    raise SystemExit(main())  # Propagate the explicit status to the shell.
