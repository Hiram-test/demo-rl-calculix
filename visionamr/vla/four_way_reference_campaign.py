"""Run the complete split-scoped Reference A/B campaign without case selection."""  # Define the formal orchestration boundary around the scientific reference ladder.
from __future__ import annotations  # Postpone annotation evaluation for supported Python runtimes.

from dataclasses import asdict  # Convert the immutable registered schedule into canonical JSON evidence.
from datetime import datetime, timezone  # Record timezone-explicit campaign and case timestamps.
import hashlib  # Compute full SHA-256 identities for every campaign artifact and receipt.
import json  # Persist finite deterministic summaries and parse authenticated prior attempts.
import os  # Publish summary bytes atomically through same-filesystem replacements.
from pathlib import Path  # Resolve campaign, repository, cache, log, and receipt locations portably.
from typing import Any, Callable, Mapping  # Describe injected test seams without constraining real runner implementations.

from ..bridge_case_manifest import load_case_manifest, problem_from_case  # Reconstruct only checksum-verified frozen manifest cases.
from ..experiment import FemRunner  # Create one independent native runner for each exact manifest case.
from .four_way_freeze import verify_freeze  # Enforce the committed post-freeze permission boundary before blind references.
from .four_way_references import DEFAULT_REFERENCE_CONFIG, PROTOCOL_ID, UNQUALIFIED_AUTHORIZATION, ReferenceBuildError, ReferenceBuildOutcome, ensure_reference_pair, verify_reference_cache, verify_reference_failure_evidence  # Reuse the registered schedule, exact expedited authorization, builder, and strict success/failure verifiers.

CAMPAIGN_SCHEMA = "wmvla-four-way-reference-campaign-v1"  # Version the split-level execution and aggregation record.
PLAN_SCHEMA = "wmvla-four-way-reference-plan-v1"  # Version the solve-free complete-split execution plan.
VERIFICATION_SCHEMA = "wmvla-four-way-reference-campaign-verification-v1"  # Name the embedded per-case verification receipt family.
ALLOWED_SPLITS = ("train", "validation", "test")  # Expose only the three frozen manifest partitions and no ad hoc subsets.
EXPECTED_SPLIT_COUNTS = {"train": 24, "validation": 8, "test": 16}  # Require the exact preregistered complete-split cardinalities.
CCX_TIMEOUT_S = 1800.0  # Keep the existing three-dimensional native solve timeout fixed across all reference cases.
SUMMARY_FILENAME = "campaign_summary.json"  # Freeze the live and final aggregate summary filename.
SUMMARY_CHECKSUM_FILENAME = "campaign_summary.sha256"  # Freeze the exact-byte summary sidecar filename.
EXPEDITED_AMENDMENT_FILENAME = "EXPEDITED_EXECUTION_AMENDMENT.md"  # Require the disclosed user-authorized nonblocking amendment beside the original protocol artifacts.


class ReferenceCampaignError(RuntimeError):  # Distinguish orchestration and authorization failures from numerical ladder evidence.
    """Report an invalid path, plan, resume summary, or blind-reference authorization."""  # Explain the public command failure contract.


RunnerFactory = Callable[[Any, Path], Any]  # Accept the real FemRunner constructor or a solve-free focused-test double.
EnsureFunction = Callable[..., ReferenceBuildOutcome]  # Accept the registered resumable A/B builder or an injected deterministic fixture.
VerifyFunction = Callable[..., Mapping[str, Any]]  # Accept the strict cache verifier or an injected structural fixture.
FreezeGate = Callable[..., Mapping[str, Any]]  # Accept the committed freeze verifier or an injected authorization recorder.


def _utc_now() -> str:  # Produce one locale-independent audit timestamp for each state transition.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # Serialize UTC explicitly with a compact terminal designator.


def _json_ready(value: Any) -> Any:  # Normalize supported scientific and filesystem objects before strict JSON encoding.
    if isinstance(value, Mapping):  # Recurse through heterogeneous mappings while preserving named evidence.
        return {str(key): _json_ready(item) for key, item in value.items()}  # Normalize keys and values without changing their logical content.
    if isinstance(value, (list, tuple)):  # Recurse through ordered sequences while discarding tuple-only encoding.
        return [_json_ready(item) for item in value]  # Preserve order because plans and case execution are deliberately ordered.
    if isinstance(value, Path):  # Convert paths into transparent portable text for persisted receipts.
        return str(value)  # Retain the resolved path supplied by the campaign plan.
    if hasattr(value, "item") and callable(value.item):  # Convert NumPy scalar wrappers without importing NumPy into orchestration.
        return value.item()  # Return the corresponding standard Python scalar.
    return value  # Leave standard strict-JSON primitives unchanged.


def _canonical_bytes(payload: Any) -> bytes:  # Serialize one payload into a stable collision-resistant representation.
    return json.dumps(_json_ready(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")  # Exclude whitespace, key-order, locale, and non-finite ambiguity.


def _payload_sha256(payload: Any) -> str:  # Hash an in-memory authorization or verification receipt completely.
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()  # Return all sixty-four lowercase hexadecimal characters.


def _file_sha256(path: Path) -> str:  # Hash exact persisted bytes using bounded memory.
    digest = hashlib.sha256()  # Allocate one fresh collision-resistant digest state.
    with path.open("rb") as handle:  # Stream the complete file rather than assuming it is small.
        for block in iter(lambda: handle.read(1024 * 1024), b""):  # Read stable one-megabyte blocks through EOF.
            digest.update(block)  # Incorporate every byte in its original order.
    return digest.hexdigest()  # Return the full exact-byte identity without abbreviation.


def _atomic_write(path: Path, encoded: bytes) -> None:  # Publish one complete file without exposing a partial summary to resume logic.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the exact selected output directory.
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")  # Isolate an interrupted write beside its final same-filesystem target.
    temporary.write_bytes(encoded)  # Write the already validated complete byte sequence once.
    os.replace(temporary, path)  # Atomically replace the previous live summary on the same filesystem.


def _write_summary(output_directory: Path, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:  # Publish an aggregate and its standard exact-byte checksum after every state change.
    summary_path = output_directory / SUMMARY_FILENAME  # Resolve the sole live aggregate summary location.
    checksum_path = output_directory / SUMMARY_CHECKSUM_FILENAME  # Resolve the sibling sha256sum-compatible sidecar.
    encoded = json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"  # Produce finite deterministic human-reviewable UTF-8 bytes.
    digest = hashlib.sha256(encoded).hexdigest()  # Hash exactly the bytes that will be visible after publication.
    _atomic_write(summary_path, encoded)  # Publish the complete aggregate before its authentication sidecar.
    _atomic_write(checksum_path, f"{digest}  {summary_path.name}\n".encode("ascii"))  # Publish the full digest and exact protected filename atomically.
    return summary_path, checksum_path, digest  # Return all locations and the complete final identity for terminal reporting.


def _read_authenticated_summary(output_directory: Path) -> tuple[dict[str, Any], bytes, str] | None:  # Load a prior interrupted or completed attempt without trusting unauthenticated bytes.
    summary_path = output_directory / SUMMARY_FILENAME  # Resolve the expected prior aggregate.
    checksum_path = output_directory / SUMMARY_CHECKSUM_FILENAME  # Resolve its mandatory exact-byte sidecar.
    if not summary_path.exists() and not checksum_path.exists():  # Treat the total absence of prior state as a first attempt.
        return None  # Report that no resume archive is necessary.
    if not summary_path.is_file() or not checksum_path.is_file():  # Reject a half-written or directory-substituted prior attempt.
        raise ReferenceCampaignError("prior reference campaign summary or checksum is incomplete")  # Stop rather than discard ambiguous resume evidence.
    encoded = summary_path.read_bytes()  # Read the exact prior bytes once for authentication and archiving.
    fields = checksum_path.read_text(encoding="ascii").strip().split()  # Parse the standard digest and filename fields.
    digest = hashlib.sha256(encoded).hexdigest()  # Independently recompute the exact prior identity.
    if len(fields) != 2 or fields[0] != digest or fields[1] != summary_path.name:  # Require the correct full digest and unambiguous sibling filename.
        raise ReferenceCampaignError("prior reference campaign checksum does not authenticate campaign_summary.json")  # Refuse unsafe overwrite of modified or truncated evidence.
    try:  # Convert malformed JSON into the campaign's explicit resume failure contract.
        payload = json.loads(encoded.decode("utf-8"))  # Decode only authenticated complete bytes.
    except (UnicodeError, json.JSONDecodeError) as exc:  # Catch invalid encoding and syntax separately from numerical failures.
        raise ReferenceCampaignError(f"prior reference campaign summary is invalid: {exc}") from exc  # Preserve the actionable decoder cause.
    if not isinstance(payload, dict):  # Require the named aggregate structure used by every attempt.
        raise ReferenceCampaignError("prior reference campaign summary must be a JSON object")  # Reject scalar and array substitutions.
    return payload, encoded, digest  # Return authenticated content and exact bytes for append-only history preservation.


def _archive_prior_summary(output_directory: Path, plan: Mapping[str, Any]) -> tuple[int, list[dict[str, Any]]]:  # Preserve an interrupted live summary before starting an automatic resume attempt.
    prior = _read_authenticated_summary(output_directory)  # Authenticate any previous live aggregate before touching it.
    if prior is None:  # Start the initial attempt without creating an empty history tree.
        return 1, []  # Number the first execution attempt explicitly.
    payload, encoded, digest = prior  # Unpack the authenticated prior state and exact bytes.
    identity_fields = ("schema", "protocol_id", "split", "manifest_sha256", "case_ids", "reference_root", "work_root", "reference_schedule_sha256", "allow_unqualified", "expedited_levels", "authorization", "expedited_amendment")  # Name every scientific, amendment, and storage identity that must remain unchanged across automatic resume.
    for field in identity_fields:  # Compare each scientific and storage identity independently for a precise failure.
        expected = CAMPAIGN_SCHEMA if field == "schema" else PROTOCOL_ID if field == "protocol_id" else plan.get(field)  # Recover immutable expected content from schema constants or the new plan.
        if payload.get(field) != expected:  # Reject resuming a different split, manifest, cache, or evidence location.
            raise ReferenceCampaignError(f"prior reference campaign {field} differs from the current complete-split plan")  # Prevent cross-campaign evidence mixing.
    previous_attempt = int(payload.get("attempt_number", 0))  # Recover the prior monotonically increasing attempt identifier.
    if previous_attempt < 1:  # Reject an unnumbered summary that cannot be archived unambiguously.
        raise ReferenceCampaignError("prior reference campaign attempt_number is invalid")  # Preserve an append-only history namespace.
    archive_path = output_directory / "history" / f"campaign_summary_attempt_{previous_attempt:03d}.json"  # Resolve a stable archive name for the exact superseded live summary.
    archive_checksum = archive_path.with_suffix(".sha256")  # Resolve the archive's exact-byte sidecar.
    if archive_path.exists() or archive_checksum.exists():  # Permit idempotent recovery only when an existing archive matches exactly.
        if not archive_path.is_file() or not archive_checksum.is_file() or _file_sha256(archive_path) != digest:  # Reject collisions or divergent attempt history.
            raise ReferenceCampaignError(f"reference campaign history collision at attempt {previous_attempt}")  # Stop rather than overwrite prior audit evidence.
    else:  # Publish the authenticated prior live bytes into append-only history once.
        _atomic_write(archive_path, encoded)  # Preserve the exact superseded summary rather than reserializing it.
        _atomic_write(archive_checksum, f"{digest}  {archive_path.name}\n".encode("ascii"))  # Authenticate the archive with the original complete digest.
    history = list(payload.get("resume_history", [])) if isinstance(payload.get("resume_history", []), list) else []  # Retain earlier compact resume links when structurally valid.
    history.append({"attempt_number": previous_attempt, "status": payload.get("status"), "updated_utc": payload.get("updated_utc"), "archive": str(archive_path), "sha256": digest})  # Link the newly archived exact attempt without duplicating all case records.
    return previous_attempt + 1, history  # Continue with the next monotonic attempt and complete archive chain.


def _is_within(path: Path, parent: Path) -> bool:  # Test containment after resolving symlinks and relative path components.
    candidate = path.resolve()  # Resolve the selected output even when its final components do not yet exist.
    boundary = parent.resolve()  # Resolve the repository boundary once.
    return candidate == boundary or boundary in candidate.parents  # Treat the repository root itself and every descendant as contained.


def _default_evidence_paths(campaign_root: Path, repository_root: Path, split: str) -> tuple[Path, Path]:  # Select freeze-compatible native work and summary locations without hiding scientific choices.
    if split == "test":  # Keep all non-whitelisted blind-reference evidence outside the committed Git worktree.
        external_root = repository_root.resolve().parent / f"{repository_root.resolve().name}-wmvla-p1-test-reference-campaign"  # Derive a stable sibling location that survives per-case freeze preflights.
        return external_root / "native", external_root / "summary"  # Separate bulky solver work from the compact authenticated campaign aggregate.
    return campaign_root / "reference_native" / split, campaign_root / "reference_campaigns" / split  # Keep pre-freeze train and validation evidence inside the eventual frozen campaign bundle.


def build_reference_campaign_plan(campaign_root: Path | str, repository_root: Path | str, split: str, *, work_root: Path | str | None = None, summary_directory: Path | str | None = None, allow_unqualified: bool = False, expedited_levels: int | None = None) -> dict[str, Any]:  # Build the exact complete-split strict or disclosed expedited plan without a solve.
    selected_split = str(split)  # Normalize the explicit split once before membership validation.
    if selected_split not in ALLOWED_SPLITS:  # Reject aliases, combined splits, and arbitrary subsets.
        raise ReferenceCampaignError(f"split must be exactly one of {ALLOWED_SPLITS}")  # Report the complete legal command surface.
    campaign = Path(campaign_root).resolve()  # Resolve the formal result root containing the frozen protocol manifest.
    repository = Path(repository_root).resolve()  # Resolve the reviewed Git worktree used by the blind authorization gate.
    manifest_path = campaign / "protocol" / "case_manifest.json"  # Require the sole canonical manifest location below the campaign root.
    manifest = load_case_manifest(manifest_path, verify_checksum=True)  # Authenticate exact bytes and every 48-case geometry and split invariant solve-free.
    cases = sorted((dict(case) for case in manifest["cases"] if case["split"] == selected_split), key=lambda case: str(case["case_id"]))  # Select the entire declared split and impose ascending case_id execution.
    expected_count = EXPECTED_SPLIT_COUNTS[selected_split]  # Read the frozen cardinality independently from manifest content.
    if len(cases) != expected_count:  # Fail closed if selection is incomplete despite manifest validation.
        raise ReferenceCampaignError(f"{selected_split} reference campaign requires exactly {expected_count} cases")  # Prevent partial or duplicated split execution.
    case_ids = [str(case["case_id"]) for case in cases]  # Materialize the exact full ordered identity list once.
    if case_ids != sorted(case_ids) or len(set(case_ids)) != expected_count:  # Recheck ordering and uniqueness at the execution boundary.
        raise ReferenceCampaignError("reference campaign case IDs are not a unique ascending complete split")  # Reject any order or identity ambiguity.
    default_work, default_summary = _default_evidence_paths(campaign, repository, selected_split)  # Resolve protocol-safe evidence defaults before optional operational overrides.
    selected_work = Path(work_root).resolve() if work_root is not None else default_work.resolve()  # Normalize the per-case native runner root without creating it.
    selected_summary = Path(summary_directory).resolve() if summary_directory is not None else default_summary.resolve()  # Normalize the aggregate output directory without creating it.
    if selected_split == "test" and (_is_within(selected_work, repository) or _is_within(selected_summary, repository)):  # Protect the committed freeze verifier's exact two-file untracked whitelist.
        raise ReferenceCampaignError("test reference native work and summary directories must both be outside the Git repository")  # Prevent logs or summaries from invalidating the freeze tag between cases.
    manifest_sha = _file_sha256(manifest_path)  # Hash exact manifest bytes independently from the already checked sidecar.
    if expedited_levels is not None and not allow_unqualified:  # Require the nonblocking opt-in whenever a shortened fixed prefix is requested.
        raise ReferenceCampaignError("--expedited-levels requires --allow-unqualified")  # Prevent an implicit schedule amendment in API and CLI plans.
    selected_levels = len(DEFAULT_REFERENCE_CONFIG.background_scales) if expedited_levels is None else int(expedited_levels)  # Use the strict full ladder or the explicitly selected prefix depth.
    if allow_unqualified and (selected_levels < 2 or selected_levels > len(DEFAULT_REFERENCE_CONFIG.background_scales)):  # Require a valid two-to-six-level operational A/B prefix.
        raise ReferenceCampaignError("expedited_levels must be between 2 and 6")  # Reject insufficient or out-of-protocol reference depth solve-free.
    if not allow_unqualified and expedited_levels is not None:  # Retain an explicit redundant fail-closed guard for future parser changes.
        raise ReferenceCampaignError("expedited reference execution is not authorized")  # Keep strict planning independent from the amendment artifact.
    original_schedule = _json_ready(asdict(DEFAULT_REFERENCE_CONFIG))  # Preserve the exact original six-level reference configuration for reporting.
    effective_schedule = {**original_schedule, "background_scales": original_schedule["background_scales"][:selected_levels], "local_floor_scales": original_schedule["local_floor_scales"][:selected_levels]} if allow_unqualified else original_schedule  # Truncate only registered scale prefixes while retaining the unchanged 0.5-percent gate.
    amendment_record = None  # Represent strict default execution without depending on a later amendment file.
    if allow_unqualified:  # Authenticate the disclosed user authorization before planning any operational fallback.
        amendment_path = campaign / "protocol" / EXPEDITED_AMENDMENT_FILENAME  # Resolve the sole amendment artifact beside the frozen manifest.
        if not amendment_path.is_file():  # Require a durable reviewed disclosure rather than a command-line flag alone.
            raise ReferenceCampaignError(f"required expedited amendment is missing: {amendment_path}")  # Stop before blind or pre-freeze native work.
        amendment_text = amendment_path.read_text(encoding="utf-8")  # Read the transparent amendment solely to verify its exact authorization token.
        if UNQUALIFIED_AUTHORIZATION not in amendment_text:  # Require the token named by the runtime and user instruction.
            raise ReferenceCampaignError("expedited amendment does not contain the required authorization token")  # Reject an unrelated or incomplete document.
        amendment_record = {**_file_record(amendment_path), "authorization": UNQUALIFIED_AUTHORIZATION, "expedited_levels": selected_levels}  # Bind exact amendment bytes, token, and selected depth into every plan and summary.
    return {"schema": PLAN_SCHEMA, "protocol_id": PROTOCOL_ID, "split": selected_split, "expected_case_count": expected_count, "case_ids": case_ids, "cases": [{"case_id": str(case["case_id"]), "split": str(case["split"]), "geometry_hash": str(case["geometry_hash"]), "config_hash": str(case["config_hash"])} for case in cases], "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha, "reference_root": str((campaign / "references").resolve()), "work_root": str(selected_work), "summary_directory": str(selected_summary), "repository_root": str(repository), "campaign_root": str(campaign), "original_reference_schedule": original_schedule, "reference_schedule": effective_schedule, "reference_schedule_sha256": _payload_sha256(effective_schedule), "allow_unqualified": bool(allow_unqualified), "expedited_levels": selected_levels if allow_unqualified else None, "authorization": UNQUALIFIED_AUTHORIZATION if allow_unqualified else None, "expedited_amendment": amendment_record, "ccx_timeout_s": CCX_TIMEOUT_S, "execution_order": "ascending_case_id", "subset_allowed": False}  # Return complete immutable strict or expedited identities without concealing the original schedule or gate.


def _default_runner_factory(problem: Any, workdir: Path) -> FemRunner:  # Construct the real isolated native runner used by formal execution.
    return FemRunner(problem, workdir, keep_files=False, ccx_timeout=CCX_TIMEOUT_S)  # Retain native logs while deleting only reproducible bulky successful intermediates.


def _file_record(path: Path) -> dict[str, Any]:  # Describe one required campaign artifact with an unabbreviated exact-byte digest.
    return {"path": str(path.resolve()), "sha256": _file_sha256(path), "size_bytes": int(path.stat().st_size)}  # Bind location, all SHA-256 bits, and byte length together.


def _optional_file_record(path: Path) -> dict[str, Any] | None:  # Preserve a partial or terminal artifact only when it exists as a regular file.
    return _file_record(path) if path.is_file() else None  # Represent absence explicitly without fabricating content.


def _native_log_records(workdir: Path) -> list[dict[str, Any]]:  # Inventory every per-case native log retained by successful or failed solves.
    return [_file_record(path) for path in sorted(candidate for candidate in workdir.rglob("*.log") if candidate.is_file())] if workdir.exists() else []  # Return deterministic complete SHA-256 receipts without reading solver values into the aggregate.


def _test_cache_exists(reference_root: Path, case_ids: list[str]) -> bool:  # Detect whether a post-freeze resume must request the exact cache-file exception immediately.
    return any((reference_root / case_id / filename).is_file() for case_id in case_ids for filename in ("reference_ledger.json", "reference_B.json"))  # Inspect only the two permitted cache filenames for manifest-owned blind cases.


def _authorize_test_references(campaign_root: Path, repository_root: Path, allow_existing: bool, gate_fn: FreezeGate, *, allow_unqualified: bool, expedited_levels: int | None, amendment: Mapping[str, Any] | None) -> dict[str, Any]:  # Invoke and cross-check the committed-freeze permission API against the selected reference policy.
    receipt = dict(gate_fn(campaign_root, repository_root, require_committed=True, allow_postfreeze_test_references=bool(allow_existing)))  # Require the fixed tag, exact frozen bytes, clean code, environment, and optional exact cache exception.
    if receipt.get("TEST_NOT_RUN") is not True:  # Require the explicit declaration even from injected or future compatible gate implementations.
        raise ReferenceCampaignError("freeze verifier did not authorize a TEST_NOT_RUN reference campaign")  # Fail before any blind reference mesh or solve.
    if receipt.get("allow_unqualified_references") is not bool(allow_unqualified) or receipt.get("expedited_reference_levels") != expedited_levels:  # Require the frozen boolean and exact depth to match the CLI plan with no coercion.
        raise ReferenceCampaignError("freeze reference qualification policy differs from the selected test reference plan")  # Prevent post-freeze reference-policy selection.
    frozen_amendment = receipt.get("reference_execution_amendment")  # Read the exact protected human authorization record returned by the freeze verifier.
    if allow_unqualified:  # Bind active operational use to the same exact amendment bytes, token, and selected depth.
        amendment_matches = isinstance(amendment, Mapping) and isinstance(frozen_amendment, Mapping) and frozen_amendment.get("sha256") == amendment.get("sha256") and frozen_amendment.get("authorization") == UNQUALIFIED_AUTHORIZATION and frozen_amendment.get("expedited_levels") == expedited_levels  # Compare collision-resistant identity and semantic authorization fields.
        if not amendment_matches:  # Reject stale, substituted, or differently configured amendment evidence.
            raise ReferenceCampaignError("freeze reference execution amendment differs from the selected test reference plan")  # Stop before opening any blind reference value.
    elif frozen_amendment is not None and isinstance(frozen_amendment, Mapping) and frozen_amendment.get("authorization") != UNQUALIFIED_AUTHORIZATION:  # Permit protected but inactive amendment evidence while rejecting a substituted token.
        raise ReferenceCampaignError("inactive frozen amendment carries an unexpected authorization token")  # Distinguish evidence presence from activation without trusting unrelated bytes.
    return {"allow_postfreeze_test_references": bool(allow_existing), "allow_unqualified_references": bool(allow_unqualified), "expedited_reference_levels": expedited_levels, "reference_execution_amendment_sha256": frozen_amendment.get("sha256") if isinstance(frozen_amendment, Mapping) else None, "verified_utc": _utc_now(), "receipt_sha256": _payload_sha256(receipt), "freeze_git_ref": receipt.get("freeze_git_ref"), "freeze_commit_sha": receipt.get("git", {}).get("freeze_commit_sha") if isinstance(receipt.get("git"), Mapping) else None}  # Retain compact full-hash Git, cache, policy, and amendment authorization evidence.


def _update_counts(summary: dict[str, Any]) -> None:  # Recompute aggregate counts from retained per-case terminal states after each transition.
    results = summary["case_results"]  # Read the ordered live result list once.
    summary["completed_case_count"] = sum(record.get("status") == "complete" for record in results)  # Count authenticated usable Reference B caches only.
    summary["qualified_reference_count"] = sum(record.get("status") == "complete" and record.get("qualification") is True for record in results)  # Count only references that passed the unchanged original dual gate.
    summary["unqualified_reference_count"] = sum(record.get("status") == "complete" and record.get("qualification") is False for record in results)  # Count explicit user-authorized operational fallbacks separately.
    summary["contains_unqualified_references"] = summary["unqualified_reference_count"] > 0  # Make any operational evidence prominent at the aggregate top level.
    summary["failed_case_count"] = sum(record.get("status") == "failed" for record in results)  # Count retained numerical or cache failures independently.
    summary["running_case_count"] = sum(record.get("status") == "running" for record in results)  # Expose an interrupted in-flight case explicitly.
    summary["cache_reuse_count"] = sum(record.get("status") == "complete" and record.get("from_cache") is True for record in results)  # Disclose how many cases required no new native reference solve.
    summary["new_build_count"] = sum(record.get("status") == "complete" and record.get("from_cache") is False for record in results)  # Disclose how many valid pairs were constructed during this attempt.
    summary["updated_utc"] = _utc_now()  # Timestamp the exact state represented by the latest atomic summary.


def run_reference_campaign(campaign_root: Path | str, repository_root: Path | str, split: str, *, dry_run: bool = False, work_root: Path | str | None = None, summary_directory: Path | str | None = None, allow_unqualified: bool = False, expedited_levels: int | None = None, runner_factory: RunnerFactory = _default_runner_factory, ensure_fn: EnsureFunction = ensure_reference_pair, verify_fn: VerifyFunction = verify_reference_cache, freeze_gate: FreezeGate = verify_freeze) -> dict[str, Any]:  # Plan or execute one complete strict or disclosed expedited split with authenticated cache reuse.
    plan = build_reference_campaign_plan(campaign_root, repository_root, split, work_root=work_root, summary_directory=summary_directory, allow_unqualified=allow_unqualified, expedited_levels=expedited_levels)  # Validate manifest, complete case set, schedule/amendment, and evidence boundaries solve-free.
    if dry_run:  # Return before creating a runner, directory, summary, mesh, or native process.
        if plan["split"] == "test":  # Prove the current committed freeze permits the planned blind reference opening without writing cache bytes.
            reference_root = Path(plan["reference_root"])  # Resolve the fixed test cache location for existing-cache detection.
            authorization = _authorize_test_references(Path(plan["campaign_root"]), Path(plan["repository_root"]), _test_cache_exists(reference_root, plan["case_ids"]), freeze_gate, allow_unqualified=plan["allow_unqualified"], expedited_levels=plan["expedited_levels"], amendment=plan["expedited_amendment"])  # Apply strict first-open/resume and exact frozen qualification-policy permission in dry-run mode.
        else:  # Pre-freeze train and validation planning requires no blind authorization.
            authorization = None  # Represent the deliberately inapplicable gate explicitly.
        return {**plan, "dry_run": True, "status": "planned", "solve_count": 0, "freeze_authorization": authorization}  # Preserve the top-level amendment token while reporting the separate optional blind-freeze receipt without filesystem mutation.
    campaign = Path(plan["campaign_root"])  # Recover the already resolved formal campaign boundary.
    repository = Path(plan["repository_root"])  # Recover the already resolved reviewed worktree.
    reference_root = Path(plan["reference_root"])  # Recover the fixed per-case cache root below the campaign.
    native_root = Path(plan["work_root"])  # Recover the isolated per-case native evidence root.
    output_directory = Path(plan["summary_directory"])  # Recover the authenticated aggregate output directory.
    authorization_receipts: list[dict[str, Any]] = []  # Collect every compact pre- and post-case blind permission receipt in order.
    if plan["split"] == "test":  # Authenticate the fixed freeze before creating any blind evidence or even the external summary.
        authorization_receipts.append(_authorize_test_references(campaign, repository, _test_cache_exists(reference_root, plan["case_ids"]), freeze_gate, allow_unqualified=plan["allow_unqualified"], expedited_levels=plan["expedited_levels"], amendment=plan["expedited_amendment"]))  # Require clean/resume cache permission and exact frozen qualification/amendment agreement.
    attempt_number, resume_history = _archive_prior_summary(output_directory, plan)  # Preserve any interrupted or completed prior aggregate before this automatic cache-resume attempt.
    summary: dict[str, Any] = {"schema": CAMPAIGN_SCHEMA, "protocol_id": PROTOCOL_ID, "split": plan["split"], "status": "running", "qualification": None, "dry_run": False, "attempt_number": attempt_number, "started_utc": _utc_now(), "updated_utc": _utc_now(), "completed_utc": None, "manifest_path": plan["manifest_path"], "manifest_sha256": plan["manifest_sha256"], "reference_root": plan["reference_root"], "work_root": plan["work_root"], "summary_directory": plan["summary_directory"], "repository_root": plan["repository_root"], "original_reference_schedule": plan["original_reference_schedule"], "reference_schedule": plan["reference_schedule"], "reference_schedule_sha256": plan["reference_schedule_sha256"], "allow_unqualified": plan["allow_unqualified"], "expedited_levels": plan["expedited_levels"], "authorization": plan["authorization"], "expedited_amendment": plan["expedited_amendment"], "expected_case_count": plan["expected_case_count"], "case_ids": plan["case_ids"], "execution_order": plan["execution_order"], "subset_allowed": False, "case_results": [], "authorization_receipts": authorization_receipts, "resume_history": resume_history, "campaign_failures": [], "completed_case_count": 0, "qualified_reference_count": 0, "unqualified_reference_count": 0, "contains_unqualified_references": False, "failed_case_count": 0, "running_case_count": 0, "cache_reuse_count": 0, "new_build_count": 0}  # Initialize original/amended schedule, visible qualification counts, authorization, and empty evidence before the first runner.
    _write_summary(output_directory, summary)  # Publish the authenticated running state before reconstructing the first physical problem.
    manifest = load_case_manifest(Path(plan["manifest_path"]), verify_checksum=True)  # Re-authenticate exact manifest bytes immediately before execution.
    case_map = {str(case["case_id"]): dict(case) for case in manifest["cases"] if case["split"] == plan["split"]}  # Rebuild an exact identity map for the already frozen ordered case list.
    for ordinal, case_id in enumerate(plan["case_ids"], start=1):  # Execute every case exactly once per attempt in ascending case_id order.
        case = case_map[case_id]  # Recover the authenticated complete geometry parameters for this exact identity.
        case_workdir = native_root / case_id  # Give this case a runner directory disjoint from every other case and split.
        record: dict[str, Any] = {"ordinal": ordinal, "case_id": case_id, "split": plan["split"], "geometry_hash": str(case["geometry_hash"]), "config_hash": str(case["config_hash"]), "status": "running", "reference_status": None, "qualification": None, "authorization": plan["authorization"], "execution_amendment": None, "original_convergence_gate": None, "started_utc": _utc_now(), "completed_utc": None, "runner_workdir": str(case_workdir.resolve()), "from_cache": None, "a_level": None, "b_level": None, "ledger": None, "reference_B": None, "verification_receipt_sha256": None, "failure_evidence_verification_sha256": None, "native_logs": [], "failure": None}  # Publish explicit qualification and original-gate nulls before any cache, mesh, or solver operation.
        summary["case_results"].append(record)  # Retain the in-flight case in the ordered aggregate.
        _update_counts(summary)  # Reflect one running case and the latest state timestamp.
        _write_summary(output_directory, summary)  # Checkpoint intent so an external interruption leaves an authenticated resume clue.
        try:  # Convert each terminal numerical or cache failure into retained evidence while continuing the complete split.
            problem = problem_from_case(case)  # Reconstruct the canonical bridge geometry and FE configuration from manifest parameters only.
            runner = runner_factory(problem, case_workdir)  # Create exactly one independent runner and native log tree for this case.
            outcome = ensure_fn(problem, runner, reference_root, case_id=case_id, config=DEFAULT_REFERENCE_CONFIG, allow_unqualified=plan["allow_unqualified"], expedited_levels=plan["expedited_levels"])  # Build the exact strict or expedited ladder and allow operational fallback only through explicit plan fields.
            verification = dict(verify_fn(reference_root, case_id=case_id, problem=problem, config=DEFAULT_REFERENCE_CONFIG, regenerate_meshes=False, allow_unqualified=plan["allow_unqualified"], expedited_levels=plan["expedited_levels"]))  # Revalidate amendment, original gate, qualification, and compact B without an extra solve.
            if verification.get("passed") is not True:  # Require an explicit positive verifier result rather than mere absence of an exception.
                raise ReferenceCampaignError(f"reference cache verification did not pass for {case_id}")  # Refuse to count an ambiguous cache as complete.
            ledger_path = reference_root / case_id / "reference_ledger.json"  # Resolve the authoritative sealed ladder artifact.
            reference_b_path = reference_root / case_id / "reference_B.json"  # Resolve the compact sealed common-reference artifact.
            if not ledger_path.is_file() or not reference_b_path.is_file():  # Require both exact formal artifacts after builder success.
                raise ReferenceCampaignError(f"reference builder omitted required artifacts for {case_id}")  # Prevent a synthetic or incomplete success record.
            record.update({"status": "complete", "reference_status": str(verification["status"]), "qualification": bool(verification["qualification"]), "authorization": verification.get("authorization"), "execution_amendment": verification.get("execution_amendment"), "original_convergence_gate": verification.get("original_convergence_gate"), "completed_utc": _utc_now(), "from_cache": bool(outcome.from_cache), "a_level": int(outcome.a_level), "b_level": int(outcome.b_level), "ledger": _file_record(ledger_path), "reference_B": _file_record(reference_b_path), "verification_schema": str(verification.get("schema", VERIFICATION_SCHEMA)), "verification_receipt_sha256": _payload_sha256(verification), "native_logs": _native_log_records(case_workdir)})  # Bind visible qualified/unqualified status, unchanged gate, accepted levels, full artifact hashes, and logs.
        except Exception as exc:  # Retain every per-case numerical, cache, schema, or native failure without fabricating Reference B values.
            ledger_path = reference_root / case_id / "reference_ledger.json"  # Resolve any checkpointed building or terminal-failure ledger.
            reference_b_path = reference_root / case_id / "reference_B.json"  # Resolve any pre-existing compact artifact for corruption diagnosis.
            record.update({"status": "failed", "completed_utc": _utc_now(), "ledger": _optional_file_record(ledger_path), "reference_B": _optional_file_record(reference_b_path), "native_logs": _native_log_records(case_workdir), "failure": {"error_type": type(exc).__name__, "error": str(exc)}})  # Preserve exact available artifacts, full log hashes, exception type, and message with no numerical substitution.
            if isinstance(exc, ReferenceBuildError) and ledger_path.is_file():  # Authenticate portable logs and failure decks written by the formal registered builder.
                try:  # Keep an evidence-verification defect explicit without masking the original numerical failure.
                    failure_verification = verify_reference_failure_evidence(reference_root, case_id=case_id, problem=problem, config=DEFAULT_REFERENCE_CONFIG, allow_unqualified=plan["allow_unqualified"], expedited_levels=plan["expedited_levels"])  # Recompute the exact strict or expedited failed ladder and every case-local native artifact.
                    record["failure_evidence_verification_sha256"] = _payload_sha256(failure_verification)  # Bind the independently verified failure receipt with a complete digest.
                except Exception as evidence_exc:  # Retain a second protocol defect when the failure artifacts themselves are unverifiable.
                    record["failure"]["evidence_verification_error_type"] = type(evidence_exc).__name__  # Preserve the exact verifier failure category.
                    record["failure"]["evidence_verification_error"] = str(evidence_exc)  # Preserve its actionable message without replacing the original cause.
        _update_counts(summary)  # Recompute success, failure, running, reuse, and new-build counts from terminal records.
        _write_summary(output_directory, summary)  # Publish each terminal case result before any later case starts.
        if plan["split"] == "test":  # Re-authenticate the fixed freeze and exact cache-file whitelist after every blind case write.
            try:  # Preserve a post-write authorization failure in the external aggregate before aborting further disclosure.
                receipt = _authorize_test_references(campaign, repository, True, freeze_gate, allow_unqualified=plan["allow_unqualified"], expedited_levels=plan["expedited_levels"], amendment=plan["expedited_amendment"])  # Permit exact cache evidence only while rechecking frozen qualification/amendment identity.
            except Exception as exc:  # Treat tag, code, environment, unknown-file, or protected-byte drift as campaign-fatal.
                summary["status"] = "authorization_failure"  # Distinguish protocol invalidation from a numerical case failure.
                summary["campaign_failures"].append({"stage": "post_case_freeze_verification", "case_id": case_id, "error_type": type(exc).__name__, "error": str(exc)})  # Retain the exact gate failure and last disclosed case.
                summary["completed_utc"] = _utc_now()  # Timestamp the forced stop immediately.
                _update_counts(summary)  # Capture final terminal counts and update time before publication.
                _write_summary(output_directory, summary)  # Persist the invalidation outside the repository without polluting the freeze whitelist.
                raise ReferenceCampaignError(f"post-case freeze verification failed after {case_id}: {exc}") from exc  # Stop before opening the next blind case.
            summary["authorization_receipts"].append(receipt)  # Retain the successful post-case full-hash permission receipt.
            _write_summary(output_directory, summary)  # Publish the authorization chain before continuing to the next case.
    summary["status"] = "complete" if summary["failed_case_count"] == 0 else "failed"  # Require every complete-split case to have an authenticated valid Reference B for campaign success.
    summary["qualification"] = bool(summary["failed_case_count"] == 0 and summary["unqualified_reference_count"] == 0)  # Distinguish operational completion from all-case original-gate qualification.
    summary["completed_utc"] = _utc_now()  # Timestamp terminal split aggregation independently from the final case.
    _update_counts(summary)  # Recompute the final exact counts and update timestamp once more.
    summary_path, checksum_path, digest = _write_summary(output_directory, summary)  # Publish the final aggregate and exact-byte sidecar atomically.
    return {**summary, "summary_path": str(summary_path), "summary_checksum_path": str(checksum_path), "summary_sha256": digest}  # Return the persisted terminal evidence plus its full identity for CLI status.
