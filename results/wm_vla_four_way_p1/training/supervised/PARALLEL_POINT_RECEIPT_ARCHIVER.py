"""Archive every pristine supervised point receipt beside its canonical validation evidence."""  # State the retained evidence-only responsibility precisely.
from __future__ import annotations  # Postpone annotation evaluation for the repository runtime.

import argparse  # Parse explicit formal, external-source, manifest, and no-write inputs.
import json  # Read strict evidence objects and print finite plans or terminal receipts.
from pathlib import Path  # Resolve external sources and canonical archive destinations portably.
import sys  # Import this exact checkout and return explicit process status.
from typing import Any  # Annotate bounded heterogeneous JSON evidence.

ROOT = Path(__file__).resolve().parents[4]  # Recover the repository root from this retained campaign-evidence location.
sys.path.insert(0, str(ROOT))  # Import the exact checked-out implementation without another installation.

import visionamr.baselines.bridge_supervised as bridge  # Reuse the campaign's complete-byte and canonical-JSON hash conventions.
from visionamr.bridge_case_manifest import load_case_manifest  # Authenticate the exact frozen manifest before accepting point tuples.

PROTOCOL_ID = "WMVLA-4WAY-P1"  # Bind every plan and archive receipt to the sole formal campaign.
SEEDS = (20260831, 20260832, 20260833)  # Restrict evidence to the three preregistered supervised candidates.
BUDGETS = (30000, 60000, 120000)  # Restrict evidence to the three preregistered equation budgets.
EXPECTED_POINT_RECEIPTS = 61  # Require the exact mixed-assembly count of pristine parallel point shards.
SHARD_RECEIPT_SCHEMA = "wmvla-four-way-supervised-validation-shard-v1"  # Require the terminal per-seed mixed-grid receipt contract.
POINT_RECEIPT_SCHEMA = "wmvla-four-way-supervised-validation-point-shard-v1"  # Require the terminal one-point worker receipt contract.


def _read_json(path: Path) -> dict[str, Any]:  # Load one strict top-level JSON object from bounded evidence.
    payload = json.loads(path.read_text(encoding="utf-8"))  # Decode the complete UTF-8 file without partial recovery.
    if not isinstance(payload, dict):  # Require named fields at every archival boundary.
        raise ValueError(f"JSON artifact is not an object: {path}")  # Reject arrays or scalars before any output operation.
    return payload  # Return the validated top-level mapping.


def _manifest_case_ids(manifest_path: Path) -> tuple[str, ...]:  # Recover the exact development-validation case set from the checksummed manifest.
    manifest = load_case_manifest(manifest_path, verify_checksum=True)  # Authenticate exact manifest bytes and its checksum sidecar.
    cases = manifest.get("cases")  # Read the validated manifest case container once.
    if not isinstance(cases, list):  # Require the expected ordered collection shape.
        raise ValueError("manifest cases must be a list")  # Reject malformed or partial manifest evidence.
    case_ids = tuple(sorted(str(case["case_id"]) for case in cases if isinstance(case, dict) and case.get("split") == "validation"))  # Select only development-validation identifiers in stable order.
    if len(case_ids) != 8 or len(set(case_ids)) != 8:  # Require the exact preregistered validation cardinality without duplicates.
        raise ValueError("manifest must contain exactly eight unique validation cases")  # Stop before accepting an incomplete or expanded grid.
    return case_ids  # Return the exact immutable validation case tuple.


def _candidate_models(formal_root: Path) -> dict[int, tuple[Path, str]]:  # Resolve and rehash all three exact preregistered candidate models.
    summary_path = formal_root / "network_training_summary.json"  # Resolve the completed candidate-training inventory.
    summary = _read_json(summary_path)  # Load candidate identities without touching blind-test artifacts.
    candidates = summary.get("candidates")  # Read the candidate collection once for strict cardinality checks.
    if summary.get("protocol_id") != PROTOCOL_ID or not isinstance(candidates, list):  # Require this campaign's named candidate inventory.
        raise ValueError(f"invalid network training summary: {summary_path}")  # Reject foreign or malformed model evidence.
    models: dict[int, tuple[Path, str]] = {}  # Index exact model paths and hashes by preregistered seed.
    for item in candidates:  # Inspect every inventory entry without selecting by validation outcome.
        if not isinstance(item, dict):  # Require named fields for every candidate record.
            raise ValueError(f"invalid candidate record in {summary_path}")  # Reject a malformed mixed-type inventory.
        seed = int(item.get("seed", -1))  # Recover the immutable training seed from the inventory.
        if seed not in SEEDS or seed in models or item.get("protocol_id") != PROTOCOL_ID:  # Require one unique record for each registered seed and this protocol.
            raise ValueError(f"invalid or duplicate candidate seed {seed}")  # Reject post-registration, duplicated, or foreign candidates.
        model_path = Path(str(item.get("model_path", ""))).resolve()  # Resolve the exact production-recorded checkpoint location.
        model_sha256 = str(item.get("model_sha256", ""))  # Recover the inventory's complete-byte identity.
        if not model_path.is_file() or bridge.file_sha256(model_path) != model_sha256:  # Recompute exact checkpoint identity before trusting any point receipt.
            raise ValueError(f"candidate model hash mismatch for seed {seed}")  # Reject evidence bound to missing or altered weights.
        models[seed] = (model_path, model_sha256)  # Retain the exact authenticated model identity.
    if set(models) != set(SEEDS):  # Require all and only the three preregistered candidate seeds.
        raise ValueError("network summary does not contain the exact supervised seed set")  # Reject incomplete or expanded model inventories.
    return models  # Return the complete authenticated seed-to-model map.


def _seed_receipt_path(formal_root: Path, seed: int) -> Path:  # Resolve the retained terminal receipt location for one completed seed grid.
    if seed == SEEDS[0]:  # Preserve the first mixed seed's formal-root assembly location.
        return formal_root / "PARALLEL_SHARD_RECEIPT.json"  # Return the seed-20260831 terminal receipt path.
    return formal_root / "validation" / f"seed_{seed}" / "PARALLEL_SHARD_RECEIPT.json"  # Return the copied helper receipt location for seed 20260832 or 20260833.


def _source_receipt_path(point_shard_root: Path, seed: int, case_id: str, budget: int) -> Path:  # Resolve one immutable external worker receipt from its registered tuple.
    return point_shard_root / f"seed_{seed}" / case_id / f"B{budget}" / "POINT_SHARD_RECEIPT.json"  # Return the sole canonical pristine-shard receipt location.


def _source_point_dir(receipt_path: Path, seed: int, case_id: str, budget: int) -> Path:  # Resolve immediate production outputs owned by one worker receipt.
    return receipt_path.parent / "validation" / f"seed_{seed}" / case_id / f"B{budget}"  # Return the sole result-and-records directory inside the external point shard.


def _archive_destination(formal_root: Path, seed: int, case_id: str, budget: int) -> Path:  # Resolve one canonical self-contained receipt destination.
    return formal_root / "point_receipts" / f"seed_{seed}" / case_id / f"B{budget}" / "POINT_SHARD_RECEIPT.json"  # Return the required seed-case-budget archive layout.


def _archive_rows_destination(formal_root: Path, seed: int, case_id: str, budget: int) -> Path:  # Resolve the canonical copy of the sole-point aggregate bound by one receipt.
    return _archive_destination(formal_root, seed, case_id, budget).with_name("POINT_VALIDATION_ROWS.json")  # Preserve the worker-bound row artifact beside its archived terminal receipt.


def _validate_point_receipt(source: dict[str, Any], seed: int, case_id: str, budget: int, point_shard_root: Path, formal_root: Path, manifest_path: Path, manifest_sha256: str, model_path: Path, model_sha256: str) -> dict[str, Any]:  # Reauthenticate one source-inventory entry and its complete external point receipt.
    receipt_path = _source_receipt_path(point_shard_root, seed, case_id, budget).resolve()  # Resolve the expected immutable external receipt path from the tuple only.
    if not receipt_path.is_file() or receipt_path.is_symlink():  # Require an ordinary terminal worker receipt at the registered external location.
        raise FileNotFoundError(f"point receipt is missing or non-regular: {receipt_path}")  # Reject absent or link-substituted evidence.
    if Path(str(source.get("point_receipt_path", ""))).resolve() != receipt_path:  # Require the seed receipt's declared source location to match the explicit shard root.
        raise ValueError(f"source inventory point receipt path mismatch: {receipt_path}")  # Reject relocated, spliced, or cross-tuple evidence.
    receipt_sha256 = bridge.file_sha256(receipt_path)  # Recompute the complete terminal receipt byte identity.
    if str(source.get("point_receipt_sha256", "")) != receipt_sha256:  # Require exact agreement with the parent source inventory.
        raise ValueError(f"source inventory point receipt hash mismatch: {receipt_path}")  # Reject altered worker receipts.
    receipt = _read_json(receipt_path)  # Decode the receipt only after its parent-bound byte identity succeeds.
    if receipt.get("schema") != POINT_RECEIPT_SCHEMA or receipt.get("protocol_id") != PROTOCOL_ID or receipt.get("TEST_NOT_RUN") is not True:  # Require the exact worker contract and anti-leakage declaration.
        raise ValueError(f"point receipt contract mismatch: {receipt_path}")  # Reject foreign, stale, or test-exposed receipts.
    if receipt.get("test_split_accessed") is not False or receipt.get("test_results_used") is not False:  # Require both explicit blind-test non-use declarations.
        raise ValueError(f"point receipt test-access declaration mismatch: {receipt_path}")  # Reject any receipt without fail-closed anti-leakage state.
    if int(receipt.get("seed", -1)) != seed or str(receipt.get("case_id")) != case_id or int(receipt.get("equation_budget", -1)) != budget:  # Require the complete immutable point tuple.
        raise ValueError(f"point receipt tuple mismatch: {receipt_path}")  # Reject cross-seed, cross-case, or cross-budget receipts.
    if str(receipt.get("manifest_sha256", "")) != manifest_sha256 or Path(str(receipt.get("manifest_path", ""))).resolve() != manifest_path:  # Require the exact frozen manifest path and bytes.
        raise ValueError(f"point receipt manifest mismatch: {receipt_path}")  # Reject a mixed or altered case definition.
    if str(receipt.get("model_sha256", "")) != model_sha256 or Path(str(receipt.get("model_path", ""))).resolve() != model_path:  # Require the exact preregistered candidate checkpoint.
        raise ValueError(f"point receipt model mismatch: {receipt_path}")  # Reject mixed candidate weights within a seed grid.
    if Path(str(receipt.get("output", ""))).resolve() != receipt_path.parent.resolve():  # Require the receipt to reside at its declared pristine output root.
        raise ValueError(f"point receipt output-root mismatch: {receipt_path}")  # Reject a moved or detached terminal receipt.
    point_dir = _source_point_dir(receipt_path, seed, case_id, budget).resolve()  # Resolve the receipt-owned immediate production point directory.
    if Path(str(source.get("source_point_dir", ""))).resolve() != point_dir:  # Require the parent inventory to name this exact immediate point directory.
        raise ValueError(f"source inventory point directory mismatch: {receipt_path}")  # Reject source splicing before output validation.
    result_path = point_dir / "validation_result.json"  # Resolve the atomic immediate production result.
    records_path = point_dir / "records.json"  # Resolve the counted-solve evidence paired with the result.
    validation_rows_path = receipt_path.parent / "validation_rows.json"  # Resolve the worker's sole-point aggregate bound by its terminal receipt.
    if not result_path.is_file() or not records_path.is_file() or not validation_rows_path.is_file():  # Require every receipt-bound immediate output.
        raise FileNotFoundError(f"point receipt outputs are incomplete: {receipt_path}")  # Reject interrupted or externally truncated shards.
    result_sha256 = bridge.file_sha256(result_path)  # Recompute the exact immediate result identity.
    records_sha256 = bridge.file_sha256(records_path)  # Recompute the exact counted-solve identity.
    validation_rows_sha256 = bridge.file_sha256(validation_rows_path)  # Recompute the exact sole-point aggregate identity.
    if str(receipt.get("validation_result_sha256", "")) != result_sha256 or str(source.get("validation_result_sha256", "")) != result_sha256:  # Require point and seed receipts to bind the same result bytes.
        raise ValueError(f"validation result hash mismatch: {receipt_path}")  # Reject altered or cross-point result evidence.
    if str(receipt.get("records_sha256", "")) != records_sha256 or str(source.get("records_sha256", "")) != records_sha256:  # Require point and seed receipts to bind the same solve-record bytes.
        raise ValueError(f"validation records hash mismatch: {receipt_path}")  # Reject altered or cross-point solve evidence.
    if str(receipt.get("validation_rows_sha256", "")) != validation_rows_sha256:  # Require the worker receipt's sole-point aggregate identity to remain intact.
        raise ValueError(f"point validation rows hash mismatch: {receipt_path}")  # Reject an altered worker aggregate even though it is not archived separately.
    row = _read_json(result_path)  # Load the exact retained success or typed numerical failure result.
    records = _read_json(records_path)  # Load the complete counted-solve container.
    if int(row.get("seed", -1)) != seed or str(row.get("case_id")) != case_id or int(row.get("equation_budget", -1)) != budget or row.get("split") != "validation":  # Revalidate the complete result tuple and development split.
        raise ValueError(f"validation result tuple mismatch: {result_path}")  # Reject a blind, cross-candidate, or cross-budget result.
    if str(row.get("model_sha256", "")) != model_sha256 or str(row.get("status", "")) != str(receipt.get("status", "")) or str(row.get("status", "")) != str(source.get("status", "")):  # Require exact checkpoint and terminal-state agreement across all three layers.
        raise ValueError(f"validation result identity mismatch: {result_path}")  # Reject model or status splicing.
    record_rows = records.get("records")  # Read the counted-solve list once for strict cardinality validation.
    if not isinstance(record_rows, list) or len(record_rows) != int(row.get("real_solve_count", -1)):  # Require result and counted-solve evidence to agree exactly.
        raise ValueError(f"validation records count mismatch: {records_path}")  # Reject truncated or cross-point records.
    canonical_point = formal_root / "validation" / f"seed_{seed}" / case_id / f"B{budget}"  # Resolve the canonical retained result-and-records directory that makes the archive self-contained.
    canonical_result = canonical_point / "validation_result.json"  # Resolve the formal immediate result paired with the archived receipt.
    canonical_records = canonical_point / "records.json"  # Resolve the formal counted-solve evidence paired with the archived receipt.
    if not canonical_result.is_file() or bridge.file_sha256(canonical_result) != result_sha256:  # Require exact source-to-formal result byte identity.
        raise ValueError(f"canonical validation result differs from point receipt: {canonical_result}")  # Reject an archive that would depend on disappearing external result evidence.
    if not canonical_records.is_file() or bridge.file_sha256(canonical_records) != records_sha256:  # Require exact source-to-formal solve-record byte identity.
        raise ValueError(f"canonical validation records differ from point receipt: {canonical_records}")  # Reject an archive that would depend on disappearing external records.
    destination = _archive_destination(formal_root, seed, case_id, budget)  # Resolve the sole canonical receipt archive destination.
    rows_destination = _archive_rows_destination(formal_root, seed, case_id, budget)  # Resolve the sole canonical copy of the receipt-bound sole-point rows.
    return {"seed": seed, "case_id": case_id, "equation_budget": budget, "source_point_receipt_path": str(receipt_path), "source_point_receipt_sha256": receipt_sha256, "source_validation_rows_path": str(validation_rows_path), "source_validation_rows_sha256": validation_rows_sha256, "destination_path": str(destination), "destination_relative_path": destination.relative_to(formal_root).as_posix(), "validation_rows_destination_path": str(rows_destination), "validation_rows_destination_relative_path": rows_destination.relative_to(formal_root).as_posix(), "manifest_sha256": manifest_sha256, "model_sha256": model_sha256, "canonical_validation_result_path": canonical_result.relative_to(formal_root).as_posix(), "canonical_validation_result_sha256": result_sha256, "canonical_records_path": canonical_records.relative_to(formal_root).as_posix(), "canonical_records_sha256": records_sha256}  # Bind the tuple, exact source receipt and rows, formal destinations, and self-contained canonical dependencies.


def _collect_entries(formal_root: Path, point_shard_root: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:  # Revalidate all three seed receipts and collect exactly sixty-one unique pristine point receipts.
    case_ids = _manifest_case_ids(manifest_path)  # Authenticate and recover the exact eight-case development set.
    manifest_sha256 = bridge.file_sha256(manifest_path)  # Bind every accepted receipt to the exact frozen manifest bytes.
    models = _candidate_models(formal_root)  # Authenticate all three preregistered candidate checkpoints once.
    expected_grid = {(case_id, budget) for case_id in case_ids for budget in BUDGETS}  # Build the exact per-seed 8-by-3 Cartesian product.
    entries: list[dict[str, Any]] = []  # Accumulate archive entries independently of source discovery timing.
    seed_receipts: list[dict[str, Any]] = []  # Accumulate the three parent receipt identities for archive binding.
    seen_parallel: set[tuple[int, str, int]] = set()  # Reject duplicate pristine point tuples across all three source inventories.
    for seed in SEEDS:  # Process every preregistered candidate in stable order.
        seed_receipt_path = _seed_receipt_path(formal_root, seed).resolve()  # Resolve the exact retained parent receipt location.
        if not seed_receipt_path.is_file() or seed_receipt_path.is_symlink():  # Require an ordinary terminal parent receipt.
            raise FileNotFoundError(f"seed shard receipt is missing or non-regular: {seed_receipt_path}")  # Reject incomplete assembly or link substitution.
        seed_receipt = _read_json(seed_receipt_path)  # Load the complete parent receipt before inspecting source provenance.
        if seed_receipt.get("schema") != SHARD_RECEIPT_SCHEMA or seed_receipt.get("protocol_id") != PROTOCOL_ID or seed_receipt.get("TEST_NOT_RUN") is not True:  # Require exact parent contract and anti-leakage state.
            raise ValueError(f"seed shard receipt contract mismatch: {seed_receipt_path}")  # Reject foreign, stale, or test-exposed parent evidence.
        if int(seed_receipt.get("seed", -1)) != seed or int(seed_receipt.get("validation_point_count", -1)) != len(expected_grid):  # Require exact candidate identity and complete 24-point coverage.
            raise ValueError(f"seed shard receipt identity mismatch: {seed_receipt_path}")  # Reject cross-seed or incomplete parent evidence.
        if seed_receipt.get("test_split_accessed") is not False or seed_receipt.get("test_results_used") is not False:  # Require explicit blind-test non-use at the seed boundary.
            raise ValueError(f"seed shard receipt test-access declaration mismatch: {seed_receipt_path}")  # Reject a parent receipt without fail-closed anti-leakage state.
        model_path, model_sha256 = models[seed]  # Recover the exact authenticated checkpoint for this seed.
        if str(seed_receipt.get("manifest_sha256", "")) != manifest_sha256 or str(seed_receipt.get("model_sha256", "")) != model_sha256:  # Require common manifest and exact candidate identities.
            raise ValueError(f"seed shard receipt input hash mismatch: {seed_receipt_path}")  # Reject a mixed-input parent inventory.
        sources = seed_receipt.get("sources")  # Read the complete exact point-source inventory once.
        if not isinstance(sources, list) or len(sources) != len(expected_grid):  # Require exactly twenty-four named source records.
            raise ValueError(f"seed source inventory is incomplete: {seed_receipt_path}")  # Reject omission or expansion before extracting pristine points.
        if bridge.canonical_json_sha256(sources) != str(seed_receipt.get("source_inventory_sha256", "")):  # Recompute the complete parent source-inventory identity.
            raise ValueError(f"seed source inventory hash mismatch: {seed_receipt_path}")  # Reject reordered, altered, or spliced source declarations.
        seen_seed: set[tuple[str, int]] = set()  # Reject duplicate point tuples within this complete parent inventory.
        for source in sources:  # Inspect every source record, including non-parallel points needed for coverage validation.
            if not isinstance(source, dict):  # Require named fields for every source record.
                raise ValueError(f"invalid source entry in {seed_receipt_path}")  # Reject a malformed mixed-type source inventory.
            case_id = str(source.get("case_id", ""))  # Recover the immutable case identifier from the parent source inventory.
            budget = int(source.get("equation_budget", -1))  # Recover the immutable equation budget from the parent source inventory.
            point_key = (case_id, budget)  # Normalize the per-seed source tuple once.
            if point_key not in expected_grid or point_key in seen_seed:  # Require exact grid membership and uniqueness.
                raise ValueError(f"invalid or duplicate source tuple {seed, case_id, budget}")  # Reject post-registration or duplicated source records.
            seen_seed.add(point_key)  # Mark this exact seed-local tuple as covered.
            if source.get("source_kind") != "pristine_parallel_point_shard":  # Retain only external point receipts while still validating full parent coverage.
                continue  # Leave sequential or whole-seed-helper points out of the point-receipt archive.
            full_key = (seed, case_id, budget)  # Normalize the global unique point-receipt tuple.
            if full_key in seen_parallel:  # Refuse any duplicate pristine point tuple across parent inventories.
                raise ValueError(f"duplicate pristine point tuple {full_key}")  # Prevent result-dependent duplicate selection.
            seen_parallel.add(full_key)  # Mark the sole authoritative pristine point receipt for this tuple.
            entries.append(_validate_point_receipt(source, seed, case_id, budget, point_shard_root, formal_root, manifest_path, manifest_sha256, model_path, model_sha256))  # Reauthenticate exact receipt bytes, inputs, outputs, and canonical retained evidence.
        if seen_seed != expected_grid:  # Require all and only the registered tuples after full source inspection.
            raise ValueError(f"seed source inventory grid mismatch: {seed_receipt_path}")  # Reject hidden omission despite nominal list cardinality.
        seed_receipts.append({"seed": seed, "path": str(seed_receipt_path), "relative_path": seed_receipt_path.relative_to(formal_root).as_posix(), "sha256": bridge.file_sha256(seed_receipt_path), "source_inventory_sha256": str(seed_receipt["source_inventory_sha256"])})  # Bind each complete parent receipt and its verified source inventory.
    entries.sort(key=lambda item: (int(item["seed"]), str(item["case_id"]), int(item["equation_budget"])))  # Make archive order independent of filesystem discovery or worker completion timing.
    if len(entries) != EXPECTED_POINT_RECEIPTS or len(seen_parallel) != EXPECTED_POINT_RECEIPTS:  # Require exactly sixty-one unique pristine external point receipts.
        raise ValueError(f"expected {EXPECTED_POINT_RECEIPTS} unique pristine point receipts, found {len(entries)}")  # Reject partial, expanded, or duplicated archive inputs.
    return entries, seed_receipts, manifest_sha256  # Return the complete verified archive inventory and parent identities.


def archive_point_receipts(formal_root: Path, point_shard_root: Path, manifest_path: Path, dry_run: bool) -> dict[str, Any]:  # Validate or publish the complete self-contained point-receipt archive.
    if point_shard_root == formal_root or point_shard_root.is_relative_to(formal_root):  # Require external pristine sources to remain disjoint from formal outputs.
        raise ValueError("point shard root must be external to the formal supervised root")  # Prevent recursive discovery or source/destination aliasing.
    archive_root = formal_root / "point_receipts"  # Resolve the sole canonical receipt archive directory.
    archive_path = formal_root / "POINT_RECEIPT_ARCHIVE.json"  # Resolve the sole terminal aggregate archive receipt.
    if archive_root.exists() or archive_path.exists():  # Preserve one-shot publication and reject even partial prior output.
        raise FileExistsError(f"point receipt archive output already exists: {archive_root} or {archive_path}")  # Refuse overwrite, implicit resume, or favorable replacement.
    entries, seed_receipts, manifest_sha256 = _collect_entries(formal_root, point_shard_root, manifest_path)  # Reauthenticate every parent, point receipt, input, and retained output before writing.
    archive_inventory_sha256 = bridge.canonical_json_sha256(entries)  # Bind the stable sixty-one-entry destination-and-hash inventory.
    common = {"protocol_id": PROTOCOL_ID, "TEST_NOT_RUN": True, "point_receipt_count": len(entries), "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256, "point_shard_root": str(point_shard_root), "destination_root": str(archive_root), "archive_path": str(archive_path), "seed_receipts": seed_receipts, "entries": entries, "archive_inventory_sha256": archive_inventory_sha256, "test_split_accessed": False, "test_results_used": False}  # Assemble exact common provenance and anti-leakage fields for plan or terminal receipt.
    if dry_run:  # Stop after complete read-only revalidation when explicitly requested.
        return {"schema": "wmvla-four-way-supervised-validation-point-receipt-archive-plan-v1", "dry_run": True, **common}  # Return the finite zero-write archive plan.
    for entry in entries:  # Copy every preverified receipt in stable seed-case-budget order.
        source = Path(str(entry["source_point_receipt_path"]))  # Resolve the exact hash-bound external receipt once.
        rows_source = Path(str(entry["source_validation_rows_path"]))  # Resolve the exact hash-bound external sole-point aggregate once.
        destination = Path(str(entry["destination_path"]))  # Resolve the sole canonical archive destination once.
        rows_destination = Path(str(entry["validation_rows_destination_path"]))  # Resolve the paired canonical sole-point aggregate destination once.
        destination.parent.mkdir(parents=True, exist_ok=False)  # Create each unique absent tuple directory without tolerating prior output.
        with destination.open("xb") as handle:  # Claim the absent receipt filename atomically and refuse concurrent overwrite.
            handle.write(source.read_bytes())  # Copy the exact complete source bytes without reserialization.
        if bridge.file_sha256(destination) != str(entry["source_point_receipt_sha256"]):  # Rehash the copied receipt before terminal archive publication.
            raise RuntimeError(f"archived point receipt differs from source: {destination}")  # Stop without publishing a misleading complete archive receipt.
        with rows_destination.open("xb") as handle:  # Claim the paired absent sole-point aggregate filename atomically.
            handle.write(rows_source.read_bytes())  # Copy the exact receipt-bound aggregate bytes without reserialization.
        if bridge.file_sha256(rows_destination) != str(entry["source_validation_rows_sha256"]):  # Rehash the copied sole-point aggregate before terminal publication.
            raise RuntimeError(f"archived point validation rows differ from source: {rows_destination}")  # Stop without publishing a receipt whose bound aggregate is unavailable.
    receipt = {"schema": "wmvla-four-way-supervised-validation-point-receipt-archive-v1", "dry_run": False, **common}  # Finalize the exact complete archive receipt only after all copies rehash successfully.
    receipt_path = bridge.write_json(archive_path, receipt)  # Publish the aggregate terminal identity last through the campaign's atomic JSON boundary.
    return {**receipt, "receipt_path": str(receipt_path)}  # Return the complete terminal receipt and its canonical path.


def _parser() -> argparse.ArgumentParser:  # Build the explicit evidence-only command surface.
    parser = argparse.ArgumentParser(description="Archive all pristine supervised point receipts beside canonical validation evidence.")  # Create a self-describing campaign-specific parser.
    parser.add_argument("--formal-root", type=Path, required=True)  # Locate the completed supervised formal artifacts and canonical results.
    parser.add_argument("--point-shard-root", type=Path, required=True)  # Locate the external pristine one-point worker receipts and immediate outputs.
    parser.add_argument("--manifest", type=Path, required=True)  # Select the exact checksummed frozen case manifest.
    parser.add_argument("--dry-run", action="store_true")  # Revalidate all sixty-one receipts and destinations without writing anything.
    return parser  # Return the complete location-and-mode parser.


def main() -> int:  # Validate or publish the requested complete receipt archive.
    args = _parser().parse_args()  # Parse every explicit evidence location and no-write mode.
    result = archive_point_receipts(args.formal_root.resolve(), args.point_shard_root.resolve(), args.manifest.resolve(), bool(args.dry_run))  # Authenticate and dispatch exactly one requested archival operation.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit a finite machine-readable plan or terminal receipt.
    return 0  # Signal successful complete verification or one-shot archival publication.


if __name__ == "__main__":  # Execute only when launched as the retained evidence archiver.
    raise SystemExit(main())  # Propagate the explicit status to the shell.
