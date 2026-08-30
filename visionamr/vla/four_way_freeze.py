"""Create and authenticate the pre-blind-test WMVLA-4WAY-P1 freeze bundle."""  # Describe the module's protocol-boundary responsibility.
from __future__ import annotations  # Postpone annotation evaluation for supported Python runtimes.

from collections.abc import Mapping, Sequence  # Import JSON container contracts used by strict validators.
from dataclasses import asdict, fields as dataclass_fields  # Convert immutable configs and inspect every deployed dataclass field.
from datetime import datetime, timezone  # Capture an unambiguous freeze timestamp in UTC.
import hashlib  # Compute exact-file SHA-256 identities for every protected artifact.
import importlib.metadata  # Capture installed distribution versions without importing optional native packages.
import json  # Read and write transparent deterministic protocol artifacts.
import os  # Publish completed JSON files atomically and read approved reproducibility variables.
from pathlib import Path  # Resolve campaign-relative paths without launch-directory dependence.
import platform  # Capture the operating-system and Python runtime contract.
import re  # Validate complete lowercase Git and SHA-256 identifiers.
import shutil  # Resolve the active CalculiX executable without invoking a shell.
import subprocess  # Query Git and native solver versions with explicit argument vectors.
import sys  # Capture the exact Python executable and implementation version.
from typing import Any  # Annotate heterogeneous strict-JSON records.
from urllib.parse import urlsplit, urlunsplit  # Remove credentials from recorded repository remotes.

from .four_way_references import DEFAULT_REFERENCE_CONFIG, UNQUALIFIED_AUTHORIZATION  # Freeze the exact reference schedule and user-authorized nonblocking amendment token.

PROTOCOL_ID = "WMVLA-4WAY-P1"  # Bind every generated artifact to the sole frozen four-way protocol.
FREEZE_SCHEMA = "wmvla-four-way-freeze-v1"  # Version the complete immutable bundle contract.
FREEZE_GIT_REF = "refs/tags/wmvla-p1-freeze"  # Resolve the dedicated freeze commit without impossible self-referential file content.
EXPEDITED_AMENDMENT_PATH = "protocol/EXPEDITED_EXECUTION_AMENDMENT.md"  # Fix the sole human-readable post-registration authorization artifact location.
EXPEDITED_REFERENCE_LEVELS = 2  # Bind the authorized operational mode to the fixed two-level prefix stated by the amendment.
CONFIG_SCHEMA = "wmvla-four-way-frozen-config-v1"  # Version the policy and artifact configuration document.
ENVIRONMENT_SCHEMA = "wmvla-four-way-environment-v1"  # Version the reproducibility environment snapshot.
GIT_STATE_SCHEMA = "wmvla-four-way-git-state-v1"  # Version the implementation-commit provenance snapshot.
PARTITION_HASH_SCHEMA = "wmvla-four-way-partition-hashes-v1"  # Version the one-spec-per-case hash inventory.
MODEL_HASH_SCHEMA = "wmvla-four-way-model-hashes-v1"  # Version the exact deployment-model inventory.
TRAINING_COST_SCHEMA = "wmvla-four-way-training-costs-v1"  # Version the combined offline-cost evidence record.
BUDGETS = (30000, 60000, 120000)  # Freeze the three public active-equation caps.
SOLVE_LIMITS = (2, 3, 4, 6)  # Freeze the four true real-solve delivery prefixes.
RL_SEED_COUNT = 3  # Require three independently trained frozen RL policies.
MODEL_METHODS = ("world_model", "supervised", "rl")  # Name the three learned artifact families.
REQUIRED_EVIDENCE = (("world_model", "training"), ("supervised", "training"), ("supervised", "validation"), ("rl", "training"), ("rl", "validation"))  # Require pre-test training and validation receipts for every selected learned method.
SCIENTIFIC_KEYS = ("horizon", "beam_width", "max_extra_regions", "max_extra_depth", "min_relative_gain", "uncertainty_limit", "failure_limit", "budget_safety", "regression_tolerance", "ensemble_size", "ridge")  # Require every protocol-listed policy degree of freedom before freezing.
KNOWN_RESULT_FILENAMES = frozenset(("final_gate.json", "primary_results.csv", "pairwise_ratios.csv", "bootstrap.json", "prediction_calibration.csv", "failure_matrix.csv", "EXECUTION_REPORT.md"))  # Detect relocated or partly written post-test aggregate evidence.
APPROVED_ENVIRONMENT_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GMSH_NUM_THREADS")  # Record only non-secret reproducibility controls.
ENVIRONMENT_DISTRIBUTIONS = ("numpy", "scipy", "gmsh", "matplotlib", "torch", "pytest")  # Require the complete reviewed numerical, native-interface, model, plotting, and test package set.
V0_CONFIG_FIELDS = {"world_planner": frozenset(("horizon", "beam_width", "candidate_regions", "max_extra_regions", "max_extra_depth", "warmup_transitions", "discount", "resource_weight", "uncertainty_weight", "failure_weight", "uncertainty_limit", "failure_limit", "budget_safety", "min_robust_gain")), "world_model_config": frozenset(("refine_factor", "error_power", "neighbor_spill", "ensemble_size", "ridge", "min_rows", "max_log_residual", "prior_uncertainty", "uncertainty_scale", "max_rows")), "world_tool_config": frozenset(("theta", "refine_factor", "core_theta", "budget_safety", "max_extra_depth")), "world_model_runtime": frozenset(("max_solves", "n_equation_cap", "theta", "refine_factor", "core_theta", "audit_slack", "fallback_cooldown", "stagnation_tolerance", "stagnation_steps", "method_name", "artifact_dir", "require_reference"))}  # Freeze every field of the deployed V0 planner, transition model, deterministic tool gateway, and runtime dataclasses.
CODE_FILES = {"model": ("visionamr/vla/world/model.py", "visionamr/vla/world_model.py", "visionamr/vla/four_way_world_training.py", "scripts/train_bridge_world_model.py"), "planner": ("visionamr/vla/world/planner.py", "visionamr/vla/planner.py"), "gateway": ("visionamr/vla/world/tool_gateway.py", "visionamr/vla/mcp_tools.py"), "pipeline": ("visionamr/vla/world/pipeline.py", "visionamr/vla/world_pipeline.py"), "statistics": ("visionamr/vla/four_way_stats.py", "visionamr/vla/four_way_analysis.py", "visionamr/vla/four_way_ablations.py", "scripts/analyze_four_way_bridge.py"), "manifest": ("visionamr/bridge_case_manifest.py", "visionamr/bridge_cases.py", "scripts/make_bridge_case_manifest.py"), "partition": ("visionamr/vla/partition_spec.py", "visionamr/vla/world/vision_partition.py", "scripts/make_bridge_partition_specs.py"), "references": ("visionamr/vla/four_way_references.py", "visionamr/vla/four_way_reference_campaign.py", "scripts/build_bridge_references.py"), "baselines": ("visionamr/baselines/local_prediction.py", "visionamr/baselines/dorfler.py", "visionamr/baselines/supervised.py", "visionamr/baselines/rl_dqn.py", "visionamr/baselines/bridge_supervised.py", "visionamr/baselines/bridge_rl.py"), "harness": ("visionamr/vla/four_way_benchmark.py", "scripts/run_four_way_bridge_benchmark.py", "scripts/run_four_way_bridge_ablations.py", "scripts/train_bridge_supervised.py", "scripts/train_bridge_rl.py"), "freeze_guard": ("visionamr/vla/four_way_freeze.py", "scripts/freeze_four_way_protocol.py"), "fem_contract": ("visionamr/experiment.py", "visionamr/calculix.py", "visionamr/mesher.py", "visionamr/sizefield.py", "visionamr/indicators.py", "visionamr/marking.py"), "ci_environment": (".github/workflows/wm-vla-four-way-p1.yml", "requirements.txt")}  # Name every policy, trainer, score, diagnostic, data boundary, baseline, reference driver, harness, native contract, workflow, and dependency declaration whose bytes affect the frozen claim.
_SHA256_RE = re.compile(r"[0-9a-f]{64}")  # Accept only complete lowercase SHA-256 strings.
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40,64}")  # Accept complete lowercase Git object identifiers across hash algorithms.


class FreezeError(RuntimeError):  # Distinguish protocol-integrity failures from numerical benchmark failures.
    """Report an invalid, incomplete, contaminated, or mutated freeze bundle."""  # Explain the exception boundary to callers.


def _utc_now() -> str:  # Return one timezone-explicit audit timestamp.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # Serialize UTC without locale ambiguity.


def _reference_config() -> dict[str, Any]:  # Convert the immutable reference schedule to exact JSON-native values.
    return json.loads(json.dumps(asdict(DEFAULT_REFERENCE_CONFIG), allow_nan=False))  # Normalize tuple schedules to the lists recovered by later JSON decoding.


def sha256_file(path: Path | str) -> str:  # Hash exact persisted bytes using bounded memory.
    target = Path(path)  # Normalize the caller-selected file path.
    digest = hashlib.sha256()  # Allocate a new collision-resistant digest state.
    with target.open("rb") as handle:  # Stream the complete artifact without assuming its size.
        for block in iter(lambda: handle.read(1024 * 1024), b""):  # Read stable one-megabyte blocks through EOF.
            digest.update(block)  # Incorporate every byte in order.
    return digest.hexdigest()  # Return the complete lowercase hexadecimal identity.


def _json_object(path: Path, label: str) -> dict[str, Any]:  # Load one finite top-level JSON object with an actionable label.
    try:  # Convert filesystem and decoder failures into one protocol-specific error.
        payload = json.loads(path.read_text(encoding="utf-8"))  # Decode the complete UTF-8 artifact.
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:  # Catch missing, unreadable, non-UTF-8, and malformed inputs.
        raise FreezeError(f"cannot read {label} {path}: {exc}") from exc  # Preserve the original cause without accepting partial content.
    if not isinstance(payload, dict):  # Require named fields at every freeze boundary.
        raise FreezeError(f"{label} must contain a JSON object: {path}")  # Reject ambiguous arrays and scalars.
    try:  # Re-encode to reject NaN and infinity accepted by Python's permissive decoder.
        json.dumps(payload, allow_nan=False)  # Validate every nested numerical value as finite strict JSON.
    except (TypeError, ValueError) as exc:  # Catch unsupported or non-finite values explicitly.
        raise FreezeError(f"{label} is not finite strict JSON: {path}") from exc  # Prevent platform-dependent scientific evidence.
    return payload  # Return only a complete finite mapping.


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:  # Publish one deterministic strict-JSON artifact atomically.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the exact protocol-owned parent directory.
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")  # Isolate an interrupted write from the last complete artifact.
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Persist stable human-auditable bytes with one terminal newline.
    os.replace(temporary, path)  # Atomically expose the completed artifact on the same filesystem.


def _write_sidecar(path: Path) -> Path:  # Publish a standard sha256sum-compatible sidecar for one artifact.
    sidecar = path.with_suffix(path.suffix + ".sha256")  # Keep the exact protected filename visible in the checksum record.
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="ascii")  # Write the exact-byte digest and unambiguous sibling filename.
    return sidecar  # Return the written sidecar for index inclusion.


def _campaign_relative(root: Path, path: Path | str, *, must_exist: bool = True) -> tuple[Path, str]:  # Bind an input path inside the selected campaign root.
    campaign = root.resolve()  # Resolve the immutable campaign boundary once.
    candidate = Path(path)  # Normalize the configured location.
    target = candidate.resolve() if candidate.is_absolute() else (campaign / candidate).resolve()  # Resolve relative paths only beneath the campaign.
    try:  # Detect traversal or accidental artifacts from another experiment.
        relative = target.relative_to(campaign)  # Derive the portable campaign-relative identity.
    except ValueError as exc:  # Catch paths outside the campaign root.
        raise FreezeError(f"artifact lies outside campaign root: {target}") from exc  # Refuse non-self-contained freeze bundles.
    if must_exist and (not target.is_file() or target.is_symlink()):  # Require regular, non-symlink exact-byte artifacts.
        raise FreezeError(f"required regular artifact is missing: {target}")  # Reject missing, directory, and symlink substitutions.
    return target, relative.as_posix()  # Return both local access and portable persisted path.


def _git(repo: Path, *arguments: str, check: bool = True) -> str:  # Execute one read-only Git query without shell expansion.
    process = subprocess.run(("git", "-C", str(repo), *arguments), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # Capture complete deterministic query output.
    if check and process.returncode != 0:  # Convert missing repositories and invalid revisions into freeze failures.
        detail = process.stderr.strip() or process.stdout.strip() or f"exit {process.returncode}"  # Preserve the most useful Git diagnostic.
        raise FreezeError(f"git {' '.join(arguments)} failed: {detail}")  # Stop before recording unverifiable code provenance.
    return process.stdout.strip()  # Return textual output without terminal line-ending differences.


def _sanitized_remote(repo: Path) -> str | None:  # Record repository identity without leaking URL credentials.
    value = _git(repo, "config", "--get", "remote.origin.url", check=False)  # Read the optional origin configured for this checkout.
    if not value:  # Permit a local-only Git repository in controlled tests.
        return None  # Represent an absent remote explicitly.
    if "://" not in value:  # Preserve credential-free SSH scp-style remotes unchanged.
        return value  # Return the repository locator without inventing a URL transform.
    parsed = urlsplit(value)  # Parse a standard transport URL into credential-bearing components.
    hostname = parsed.hostname or ""  # Retain the host while discarding user information.
    port = f":{parsed.port}" if parsed.port is not None else ""  # Preserve an explicit transport port.
    return urlunsplit((parsed.scheme, hostname + port, parsed.path, parsed.query, parsed.fragment))  # Reassemble the same remote without username or password.


def _validate_implementation_worktree(repo: Path, root: Path, implementation_commit: str) -> dict[str, Any]:  # Prove source code is exactly the declared implementation commit before artifact freeze.
    if _GIT_SHA_RE.fullmatch(implementation_commit) is None:  # Require an unabbreviated lowercase implementation identity.
        raise FreezeError("implementation_commit must be a complete lowercase Git object ID")  # Prevent ambiguous short revisions.
    head = _git(repo, "rev-parse", "HEAD")  # Resolve the checked-out implementation commit.
    if head != implementation_commit:  # Forbid freezing code other than the explicitly reviewed implementation.
        raise FreezeError(f"HEAD {head} does not equal implementation_commit {implementation_commit}")  # Report both exact identities.
    tracked_changes = _git(repo, "status", "--porcelain=v1", "--untracked-files=no")  # Detect staged and unstaged mutations independently from generated evidence.
    if tracked_changes:  # Require every tracked source and protocol file to be committed before freezing.
        raise FreezeError("tracked worktree changes exist; commit the implementation before freezing")  # Prevent uncommitted policy or scoring drift.
    repository = repo.resolve()  # Resolve the Git worktree boundary once.
    campaign = root.resolve()  # Resolve the only allowed untracked output boundary.
    untracked_text = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")  # Enumerate untracked artifacts without quote or whitespace ambiguity.
    untracked = [value for value in untracked_text.split("\0") if value]  # Recover every repository-relative untracked path.
    outside: list[str] = []  # Collect untracked files that could hide uncommitted implementation changes.
    for value in untracked:  # Check each untracked path against the campaign output boundary.
        target = (repository / value).resolve()  # Resolve the repository-relative candidate without following it into the freeze.
        if target != campaign and campaign not in target.parents:  # Permit only explicit campaign evidence before its dedicated commit.
            outside.append(value)  # Retain the offending path for an actionable failure.
    if outside:  # Refuse hidden code or configuration outside the campaign result tree.
        raise FreezeError("untracked files outside campaign root exist: " + ", ".join(sorted(outside)))  # Report the complete deterministic violation list.
    branch = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False) or None  # Record the branch when the implementation commit is not detached.
    parent = _git(repo, "rev-parse", f"{implementation_commit}^", check=False) or None  # Record the reviewed implementation parent when one exists.
    return {"schema": GIT_STATE_SCHEMA, "protocol_id": PROTOCOL_ID, "captured_utc": _utc_now(), "implementation_commit_sha": implementation_commit, "implementation_parent_sha": parent, "implementation_tree_sha": _git(repo, "rev-parse", f"{implementation_commit}^{{tree}}"), "observed_head_sha": head, "branch": branch, "origin": _sanitized_remote(repo), "tracked_worktree_clean": True, "untracked_paths": sorted(untracked), "freeze_git_ref": FREEZE_GIT_REF, "freeze_commit_required_after_creation": True}  # Return complete pre-commit implementation provenance.


def _test_case_ids(manifest: Mapping[str, Any]) -> set[str]:  # Extract exact blind case identifiers without opening any reference or result.
    values = manifest.get("cases", [])  # Read the manifest-owned case collection.
    if not isinstance(values, list):  # Reject malformed manifests before leakage scanning.
        raise FreezeError("case_manifest.json lacks a cases list")  # Prevent an incomplete test-case boundary.
    return {str(case["case_id"]) for case in values if isinstance(case, Mapping) and case.get("split") == "test"}  # Return only manifest-declared blind identifiers.


def _permitted_postfreeze_reference_path(relative: Path, blind_ids: set[str], reference_level_count: int | None = None) -> bool:  # Recognize only authenticated-cache filenames produced by the reviewed test-reference campaign.
    parts = relative.parts  # Inspect exact path components without substring or platform-separator ambiguity.
    if len(parts) == 3 and parts[0] == "references" and parts[1] in blind_ids and parts[2] in {"reference_ledger.json", "reference_B.json"}:  # Permit the authoritative ledger and compact accepted Reference B only.
        return True  # Authorize this exact manifest-owned cache JSON path for later semantic verification.
    selected_level_count = len(DEFAULT_REFERENCE_CONFIG.background_scales) if reference_level_count is None else int(reference_level_count)  # Use the full strict ladder until an authenticated expedited depth is available.
    if selected_level_count < 1 or selected_level_count > len(DEFAULT_REFERENCE_CONFIG.background_scales):  # Bound even direct scanner callers to the registered schedule.
        raise FreezeError("postfreeze reference level count lies outside DEFAULT_REFERENCE_CONFIG")  # Prevent an oversized whitelist from authorizing invented ladder files.
    allowed_logs = {f"ref_l{level:02d}.log" for level in range(selected_level_count)}  # Derive the exact active strict or expedited ladder-log names.
    allowed_inputs = {f"ref_l{level:02d}.inp" for level in range(selected_level_count)}  # Derive the matching active failed-level CalculiX input names from the same frozen schedule.
    permitted_log = len(parts) == 4 and parts[0] == "references" and parts[1] in blind_ids and parts[2] == "solver_logs" and parts[3] in allowed_logs  # Recognize one exact retained native log per possible preregistered level.
    permitted_failure_input = len(parts) == 4 and parts[0] == "references" and parts[1] in blind_ids and parts[2] == "solver_inputs" and parts[3] in allowed_inputs  # Recognize only failed-level solver decks explicitly authenticated by the ledger.
    return permitted_log or permitted_failure_input  # Permit no other nested reference evidence, extensions, or filenames.


def scan_for_disclosed_results(root: Path | str, manifest: Mapping[str, Any] | None = None, *, allow_postfreeze_test_references: bool = False, postfreeze_reference_level_count: int | None = None) -> list[str]:  # Find any blind, ablation, aggregate, or unauthorized test-reference artifact.
    campaign = Path(root)  # Normalize the campaign root without creating it.
    if not campaign.exists():  # Treat an absent campaign as uncontaminated for solve-free planning.
        return []  # Return an empty deterministic disclosure list.
    blind_ids = _test_case_ids(manifest) if manifest is not None else set()  # Recover test-reference directory names when a manifest is available.
    disclosed: list[str] = []  # Collect every forbidden regular file instead of failing on only the first.
    for path in sorted(candidate for candidate in campaign.rglob("*") if candidate.is_file() or candidate.is_symlink()):  # Inspect all visible and hidden files below the one campaign boundary.
        relative = path.relative_to(campaign)  # Derive a stable portable disclosure identity.
        parts = relative.parts  # Read directory components without substring false positives.
        forbidden_tree = bool(parts and parts[0] in {"test", "ablations", "aggregate"})  # Reject every partial or complete primary-result tree.
        forbidden_name = path.name in KNOWN_RESULT_FILENAMES  # Reject known aggregate evidence even if relocated.
        permitted_reference = allow_postfreeze_test_references and not path.is_symlink() and _permitted_postfreeze_reference_path(relative, blind_ids, postfreeze_reference_level_count)  # Permit only exact regular cache JSON and active-schedule solver logs or failed-level input decks that the reference campaign immediately authenticates per case.
        forbidden_reference = len(parts) >= 2 and parts[0] == "references" and parts[1] in blind_ids and not permitted_reference  # Reject every pre-freeze or unknown post-freeze test-reference artifact.
        if forbidden_tree or forbidden_name or forbidden_reference:  # Retain all evidence that would invalidate TEST_NOT_RUN.
            disclosed.append(relative.as_posix())  # Preserve the precise campaign-relative path for review.
    return disclosed  # Return a sorted complete contamination report.


def _manifest_record(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:  # Authenticate the manifest and its required sha256sum sidecar.
    path, relative = _campaign_relative(root, "protocol/case_manifest.json")  # Resolve the sole protocol manifest location.
    sidecar, sidecar_relative = _campaign_relative(root, "protocol/case_manifest.sha256")  # Resolve the required checksum sidecar.
    manifest = _json_object(path, "case manifest")  # Decode the exact finite manifest before split checks.
    digest = sha256_file(path)  # Compute the persisted exact-byte identity.
    fields = sidecar.read_text(encoding="ascii").strip().split()  # Parse the standard digest and filename fields.
    if fields != [digest, path.name]:  # Require the sidecar to bind exactly this manifest's bytes and name.
        raise FreezeError("case_manifest.sha256 does not authenticate case_manifest.json")  # Stop before freezing a stale case design.
    if manifest.get("protocol_id") != PROTOCOL_ID or int(manifest.get("case_count", -1)) != 48:  # Require the sole protocol and exact total cardinality.
        raise FreezeError("case manifest must declare WMVLA-4WAY-P1 with exactly 48 cases")  # Reject unrelated or truncated designs.
    counts = {split: sum(isinstance(case, Mapping) and case.get("split") == split for case in manifest.get("cases", [])) for split in ("train", "validation", "test")}  # Recompute split cardinalities independently.
    if counts != {"train": 24, "validation": 8, "test": 16}:  # Require the preregistered 24/8/16 boundary.
        raise FreezeError(f"case manifest split counts are invalid: {counts}")  # Prevent dropped or reassigned test cases.
    record = {"path": relative, "sha256": digest, "sidecar_path": sidecar_relative, "sidecar_sha256": sha256_file(sidecar), "case_count": 48, "split_counts": counts}  # Preserve exact manifest and sidecar identities.
    return manifest, record  # Return both validated content and compact provenance.


def _partition_inventory(root: Path, manifest: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:  # Hash and bind one semantic partition specification per manifest case.
    configured_root = str(config.get("partition_root", "protocol/partitions"))  # Read the sole explicit partition registry location.
    partition_root, partition_relative = _campaign_relative(root, configured_root, must_exist=False)  # Resolve the registry directory beneath the campaign.
    if not partition_root.is_dir() or partition_root.is_symlink():  # Require a concrete read-only-compatible registry directory.
        raise FreezeError(f"partition_root is missing or invalid: {partition_root}")  # Refuse blind execution without shared WM/RL semantics.
    records: list[dict[str, Any]] = []  # Collect one exact specification identity per case.
    cases = manifest.get("cases", [])  # Read the validated ordered manifest records.
    for case in sorted(cases, key=lambda value: str(value["case_id"])):  # Freeze partitions in deterministic case-id order.
        case_id = str(case["case_id"])  # Read the manifest-bound case identifier.
        path = partition_root / case_id / "partition_spec.json"  # Resolve the unique specification filename required by the protocol.
        target, relative = _campaign_relative(root, path)  # Require a regular in-campaign persisted file.
        payload = _json_object(target, f"partition specification for {case_id}")  # Decode enough content to bind the specification to its case.
        if payload.get("protocol_id") != PROTOCOL_ID or payload.get("geometry_hash") != case.get("geometry_hash"):  # Require exact protocol and geometry identities.
            raise FreezeError(f"partition specification does not match manifest case {case_id}")  # Reject cross-case or stale semantic assignments.
        semantic_sha = str(payload.get("spec_sha256", ""))  # Read the specification's optional canonical semantic identity.
        if _SHA256_RE.fullmatch(semantic_sha) is None:  # Require the adapter's transparent non-recursive semantic seal.
            raise FreezeError(f"partition specification lacks a valid spec_sha256: {case_id}")  # Prevent exact-byte hashing from hiding malformed semantics.
        records.append({"case_id": case_id, "split": str(case["split"]), "geometry_hash": str(case["geometry_hash"]), "path": relative, "sha256": sha256_file(target), "spec_sha256": semantic_sha, "size_bytes": target.stat().st_size})  # Preserve both exact-file and semantic-body identities.
    if len(records) != 48 or len({record["case_id"] for record in records}) != 48:  # Require complete unique manifest coverage.
        raise FreezeError("partition inventory must contain exactly one specification for every one of 48 cases")  # Reject partial training-only registries.
    return {"schema": PARTITION_HASH_SCHEMA, "protocol_id": PROTOCOL_ID, "partition_root": partition_relative, "case_count": len(records), "partitions": records}  # Return the complete stable inventory.


def _entry_list(config: Mapping[str, Any], key: str) -> list[dict[str, Any]]:  # Normalize one required explicit artifact list without accepting scalar shortcuts.
    values = config.get(key)  # Read the exact source-config field.
    if not isinstance(values, list) or not values:  # Require explicit nonempty inventories before test.
        raise FreezeError(f"config source must contain a nonempty {key} list")  # Prevent filesystem globbing or implicit model selection.
    if any(not isinstance(value, Mapping) for value in values):  # Require named metadata for every artifact.
        raise FreezeError(f"every {key} entry must be an object")  # Reject ambiguous bare path values.
    return [dict(value) for value in values]  # Return independent records for normalization and hashing.


def _hash_declared_artifacts(root: Path, values: Sequence[Mapping[str, Any]], *, require_metadata: Sequence[str]) -> list[dict[str, Any]]:  # Normalize and hash one explicit artifact inventory.
    records: list[dict[str, Any]] = []  # Collect exact-byte identities without mutating the source configuration.
    seen_paths: set[str] = set()  # Reject duplicate aliases that could obscure selection logic.
    for value in values:  # Validate each explicitly declared artifact in source order.
        missing = [name for name in ("name", "path", *require_metadata) if name not in value]  # Identify all required metadata omissions together.
        if missing:  # Reject incomplete provenance before filesystem access.
            raise FreezeError(f"artifact entry lacks required fields {missing}: {value}")  # Report the exact offending record.
        target, relative = _campaign_relative(root, str(value["path"]))  # Resolve only a regular in-campaign artifact.
        if relative in seen_paths:  # Forbid the same bytes from masquerading as multiple required receipts.
            raise FreezeError(f"artifact path is declared more than once: {relative}")  # Preserve one unambiguous role per exact input.
        seen_paths.add(relative)  # Retain the normalized path identity.
        digest = sha256_file(target)  # Compute the exact persisted model or evidence digest.
        claimed = value.get("sha256")  # Read an optional upstream identity without trusting it.
        if claimed is not None and str(claimed) != digest:  # Reject stale selection indices instead of silently rewriting them.
            raise FreezeError(f"declared SHA-256 does not match {relative}")  # Surface model-copy or receipt mutation.
        record = {str(key): item for key, item in value.items() if key not in {"path", "sha256"}}  # Preserve transparent method, seed, phase, and name metadata.
        record.update({"path": relative, "sha256": digest, "size_bytes": target.stat().st_size})  # Bind metadata to exact immutable bytes.
        records.append(record)  # Retain the normalized artifact for later completeness checks.
    return records  # Return a stable explicit inventory.


def _validate_models(config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:  # Require one selected deployment model set without best-test selection.
    methods = [str(record.get("method")) for record in records]  # Read explicit learned-method identities independently from names.
    if any(method not in MODEL_METHODS for method in methods):  # Reject unrelated models from the deployment inventory.
        raise FreezeError(f"model methods must be exactly from {MODEL_METHODS}")  # Preserve the learned four-way comparator boundary.
    world = [record for record in records if record.get("method") == "world_model"]  # Isolate the sole transition-library snapshot.
    if len(world) != 1:  # Require one exact no-test-learning model initial state.
        raise FreezeError("model_artifacts must contain exactly one world_model snapshot")  # Prevent fresh or ambiguous model initialization.
    selected_supervised = config.get("selected_supervised_seed")  # Read the validation-selected supervised network identity.
    if selected_supervised is None:  # Require pre-test network selection metadata.
        raise FreezeError("selected_supervised_seed is required")  # Prevent first-file or test-informed supervised selection.
    selected_models = [record for record in records if record.get("method") == "supervised" and str(record.get("seed")) == str(selected_supervised)]  # Match the sole selected deployment network.
    if len(selected_models) != 1:  # Require exactly one checkpoint for the declared selected seed.
        raise FreezeError("model_artifacts must identify exactly one selected supervised checkpoint")  # Reject missing or duplicate selected models.
    rl_seeds = config.get("rl_seeds")  # Read the actual three independent initialization seeds.
    if not isinstance(rl_seeds, list) or len(rl_seeds) != RL_SEED_COUNT or len({int(seed) for seed in rl_seeds}) != RL_SEED_COUNT:  # Require three distinct explicit values.
        raise FreezeError("rl_seeds must contain exactly three distinct integer seeds")  # Prevent incomplete or duplicate RL policy sets.
    rl_records = [record for record in records if record.get("method") == "rl"]  # Isolate frozen greedy RL deployments.
    observed = [int(record["seed"]) for record in rl_records if "seed" in record]  # Normalize every explicitly seeded RL artifact.
    if sorted(observed) != sorted(int(seed) for seed in rl_seeds):  # Require one and only one model for each declared seed.
        raise FreezeError("model_artifacts must contain exactly one RL policy for every declared rl_seed")  # Prevent best-seed selection or omission.


def _training_costs(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # Combine explicit per-method offline-cost evidence without recomputing results.
    values = _entry_list(config, "training_cost_sources")  # Require an explicit source list rather than globbing mutable training directories.
    records = _hash_declared_artifacts(root, values, require_metadata=("method",))  # Bind every cost source to exact bytes and method metadata.
    observed = {str(record["method"]) for record in records}  # Collect learned-method cost coverage.
    if observed != set(MODEL_METHODS):  # Require costs for world acquisition, supervised labels/networks, and all RL training.
        raise FreezeError(f"training_cost_sources must cover exactly {MODEL_METHODS}")  # Reject incomplete amortization evidence.
    sources: list[dict[str, Any]] = []  # Embed validated cost payloads with their exact upstream identities.
    for record in records:  # Decode every JSON cost source after its bytes have been hashed.
        target, _relative = _campaign_relative(root, str(record["path"]))  # Resolve the normalized campaign-relative path again.
        payload = _json_object(target, f"training cost source {record['name']}")  # Require finite transparent numerical evidence.
        sources.append({**record, "payload": payload})  # Preserve exact provenance and complete source content in one aggregate.
    aggregate = {"schema": TRAINING_COST_SCHEMA, "protocol_id": PROTOCOL_ID, "TEST_NOT_RUN": True, "method_count": len(observed), "methods": list(MODEL_METHODS), "sources": sources}  # Assemble the protocol-required combined offline-cost report.
    return aggregate, records  # Return both persisted aggregate content and source protection records.


def _training_validation_inventory(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:  # Authenticate all model-training and validation-selection receipts.
    values = _entry_list(config, "training_validation_artifacts")  # Require reviewed explicit evidence rather than broad directory hashing.
    records = _hash_declared_artifacts(root, values, require_metadata=("method", "phase"))  # Bind method and phase metadata to exact bytes.
    observed = {(str(record["method"]), str(record["phase"])) for record in records}  # Collect evidence coverage independently from filenames.
    missing = sorted(set(REQUIRED_EVIDENCE) - observed)  # Identify absent pre-test training or validation families.
    if missing:  # Refuse a freeze that cannot reproduce checkpoint selection.
        raise FreezeError(f"training_validation_artifacts lack required method/phase evidence: {missing}")  # Report every missing evidence class together.
    return records  # Return the exact-byte training and validation inventory.


def _find_scientific_values(payload: Any, key: str) -> list[Any]:  # Recursively locate a required scientific setting across supported nested config blocks.
    values: list[Any] = []  # Collect every occurrence to detect contradictory duplicate settings.
    if isinstance(payload, Mapping):  # Traverse named JSON blocks recursively.
        for name, item in payload.items():  # Preserve explicit source ordering only for diagnostics.
            if name == key:  # Retain the exact required setting name.
                values.append(item)  # Record its JSON value without coercion.
            values.extend(_find_scientific_values(item, key))  # Continue into nested planner, model, and runtime blocks.
    elif isinstance(payload, list):  # Traverse arrays that may contain per-method configuration records.
        for item in payload:  # Inspect every ordered entry.
            values.extend(_find_scientific_values(item, key))  # Collect nested occurrences under the same strict rule.
    return values  # Return all occurrences for absence and conflict checks.


def _validate_scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:  # Require all preregistered WM settings and reject contradictory aliases.
    values = config.get("scientific_config")  # Read the sole protocol-level compatibility block independently from component-specific tool margins.
    if not isinstance(values, Mapping):  # Reject recursive alias discovery and implicit repository defaults.
        raise FreezeError("frozen config source must contain a scientific_config object")  # Require one reviewable protocol-level setting map.
    snapshot: dict[str, Any] = {}  # Collect one explicit value for every required setting.
    for key in SCIENTIFIC_KEYS:  # Check every protocol-listed world-model policy degree of freedom.
        if key not in values:  # Reject implicit repository defaults that could change after freezing.
            raise FreezeError(f"frozen config source lacks required scientific setting {key}")  # Require an explicit stable value.
        snapshot[key] = values[key]  # Retain the sole explicit protocol-level value in the freeze summary.
    return snapshot  # Return a compact auditable scientific configuration index.


def _allow_unqualified_references(config: Mapping[str, Any]) -> bool:  # Resolve the sole reviewed override for physically complete but threshold-unqualified references.
    value = config.get("allow_unqualified_references", False)  # Default fail-closed without forcing legacy source inventories to opt out redundantly.
    if type(value) is not bool:  # Reject truthy numbers and strings that could be interpreted differently by callers.
        raise FreezeError("allow_unqualified_references must be an explicit JSON boolean")  # Require an unambiguous human-reviewed qualification policy.
    return value  # Preserve true only when the source explicitly authorizes honest complete_unqualified use.


def _reference_amendment_record(root: Path, *, required: bool) -> dict[str, Any] | None:  # Authenticate the fixed post-registration authorization artifact when present or activated.
    path, relative = _campaign_relative(root, EXPEDITED_AMENDMENT_PATH, must_exist=False)  # Resolve only the canonical in-campaign amendment location.
    if not path.exists():  # Distinguish the default strict protocol from an activated nonblocking authorization.
        if required:  # Require durable human authorization before accepting any unqualified reference.
            raise FreezeError(f"allow_unqualified_references=true requires {EXPEDITED_AMENDMENT_PATH}")  # Refuse an unaudited boolean-only relaxation.
        return None  # Preserve explicit absence under the default fail-closed policy.
    if not path.is_file() or path.is_symlink():  # Require concrete reviewable bytes rather than a directory or redirected content.
        raise FreezeError(f"reference execution amendment must be a regular non-symlink file: {relative}")  # Prevent path substitution after authorization review.
    try:  # Decode the human-readable authorization exactly once for token validation.
        content = path.read_text(encoding="utf-8")  # Require transparent UTF-8 amendment text.
    except (OSError, UnicodeError) as exc:  # Convert unreadable or non-UTF-8 evidence into a protocol failure.
        raise FreezeError(f"cannot read reference execution amendment: {relative}") from exc  # Preserve the original filesystem cause.
    authorization_line = f"- 授权标识：`{UNQUALIFIED_AUTHORIZATION}`"  # Construct the sole accepted exact human authorization declaration.
    if authorization_line not in content.splitlines():  # Require the token in its explicit named amendment field rather than as incidental prose.
        raise FreezeError(f"reference execution amendment authorization token does not match {UNQUALIFIED_AUTHORIZATION}")  # Reject stale, copied, or differently authorized exceptions.
    return {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "authorization": UNQUALIFIED_AUTHORIZATION, "expedited_levels": EXPEDITED_REFERENCE_LEVELS}  # Bind activation to exact bytes, length, path, reviewed token, and fixed operational depth.


def _validate_complete_v0_config(config: Mapping[str, Any]) -> dict[str, Any]:  # Require exact explicit values for every deployed V0 component field.
    from .world.model import WorldModelConfig  # Import the deployed transition-model dataclass lazily for schema drift detection.
    from .world.pipeline import WorldVLAConfig  # Import the deployed runtime dataclass lazily for schema drift detection.
    from .world.planner import PlannerConfig  # Import the deployed planner dataclass lazily for schema drift detection.
    from .world.tool_gateway import ToolConfig  # Import the deployed deterministic tool dataclass lazily for schema drift detection.
    deployed_fields = {"world_planner": frozenset(field.name for field in dataclass_fields(PlannerConfig)), "world_model_config": frozenset(field.name for field in dataclass_fields(WorldModelConfig)), "world_tool_config": frozenset(field.name for field in dataclass_fields(ToolConfig)), "world_model_runtime": frozenset(field.name for field in dataclass_fields(WorldVLAConfig))}  # Derive the actual reviewed executable component schemas.
    if deployed_fields != V0_CONFIG_FIELDS:  # Require the freeze guard to be reviewed whenever an executable component adds or removes a field.
        raise FreezeError(f"freeze guard V0_CONFIG_FIELDS differs from deployed dataclasses: {deployed_fields}")  # Prevent silent omission after implementation evolution.
    snapshot: dict[str, Any] = {}  # Collect complete independent component blocks for the freeze report.
    for block, fields in V0_CONFIG_FIELDS.items():  # Validate planner, model, tool, and runtime in one explicit contract.
        values = config.get(block)  # Read the sole supported component block name.
        if not isinstance(values, Mapping):  # Reject aliases or implicit repository defaults.
            raise FreezeError(f"frozen config source must contain object {block}")  # Require a reviewable complete named block.
        missing = sorted(fields - set(values))  # Identify every dataclass field omitted from the source.
        if missing:  # Refuse a partially frozen component configuration.
            raise FreezeError(f"{block} lacks complete V0 fields: {missing}")  # Report all missing settings together.
        snapshot[block] = {str(key): values[key] for key in sorted(values)}  # Preserve required fields plus any explicit versioned extensions.
    if config.get("common_gradation") != 1.0:  # Require one shared Lipschitz slope across WM, LP, supervised, RL, and Dörfler remeshing.
        raise FreezeError("common_gradation must equal exactly 1.0")  # Reject the historical 0.9 per-method divergence.
    if "world_model_seed" not in config:  # Require deterministic residual-ensemble bootstrap behavior.
        raise FreezeError("world_model_seed is required")  # Prevent constructor-default seed drift.
    snapshot["common_gradation"] = 1.0  # Preserve the common remeshing contract explicitly.
    snapshot["world_model_seed"] = config["world_model_seed"]  # Preserve the deterministic transition-ensemble seed.
    return snapshot  # Return every V0 component setting for dry-run evidence.


def _code_inventory(repo: Path, implementation_commit: str) -> dict[str, Any]:  # Hash all claim-critical code bytes and bind them to the implementation commit tree.
    records: list[dict[str, Any]] = []  # Collect one exact worktree and Git-blob identity per required path.
    for category in sorted(CODE_FILES):  # Preserve stable semantic category order.
        for relative in CODE_FILES[category]:  # Require every declared policy, baseline, harness, and finite-element contract file.
            target = (repo / relative).resolve()  # Resolve the code path beneath the reviewed worktree.
            try:  # Reject path traversal in the fixed inventory defensively.
                target.relative_to(repo.resolve())  # Confirm the exact file remains inside the Git repository.
            except ValueError as exc:  # Catch an invalid inventory path.
                raise FreezeError(f"code inventory path leaves repository: {relative}") from exc  # Stop before hashing unrelated bytes.
            if not target.is_file() or target.is_symlink():  # Require concrete source files at the reviewed implementation commit.
                raise FreezeError(f"required claim-critical code file is missing: {relative}")  # Reject incomplete implementation freezes.
            worktree_sha = sha256_file(target)  # Compute the exact checked-out bytes.
            try:  # Read exact committed bytes without writing a temporary file.
                committed_bytes = subprocess.run(("git", "-C", str(repo), "show", f"{implementation_commit}:{relative}"), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout  # Retrieve the path from the reviewed implementation tree.
            except subprocess.CalledProcessError as exc:  # Catch an untracked or absent code file.
                raise FreezeError(f"claim-critical code file is not committed at implementation SHA: {relative}") from exc  # Prevent uncommitted executable behavior.
            committed_sha = hashlib.sha256(committed_bytes).hexdigest()  # Hash the exact Git-tree file bytes independently.
            if worktree_sha != committed_sha:  # Require the checked-out code to equal the committed implementation tree.
                raise FreezeError(f"claim-critical code differs from implementation commit: {relative}")  # Reject worktree or line-ending drift.
            blob_sha = _git(repo, "rev-parse", f"{implementation_commit}:{relative}")  # Record the native Git blob identity alongside portable SHA-256.
            records.append({"category": category, "path": relative, "sha256": worktree_sha, "git_blob_sha": blob_sha, "size_bytes": target.stat().st_size})  # Preserve complete code provenance.
    return {"schema": "wmvla-four-way-code-hashes-v1", "protocol_id": PROTOCOL_ID, "implementation_commit_sha": implementation_commit, "file_count": len(records), "files": records}  # Return the stable complete claim-critical code map.


def capture_environment() -> dict[str, Any]:  # Capture the native and Python dependency contract without reading secrets.
    versions: dict[str, str | None] = {}  # Collect installed distribution versions without importing them.
    for name in ENVIRONMENT_DISTRIBUTIONS:  # Query every registered dependency deterministically.
        try:  # Treat an absent optional distribution as explicit evidence rather than an import failure.
            versions[name] = importlib.metadata.version(name)  # Read package metadata from the active interpreter environment.
        except importlib.metadata.PackageNotFoundError:  # Catch a dependency not installed in this freeze environment.
            versions[name] = None  # Preserve the absence for later blind preflight comparison.
    ccx_path = shutil.which("ccx")  # Resolve the exact CalculiX executable used by the active process environment.
    if ccx_path is None:  # Require a runnable native solver for a scientifically executable freeze.
        raise FreezeError("CalculiX executable 'ccx' is not available on PATH")  # Stop before claiming environment readiness.
    process = subprocess.run((ccx_path, "-v"), check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)  # Capture solver version text without a shell.
    ccx_version = process.stdout.strip()  # Preserve the complete short native version output.
    if not ccx_version or "version" not in ccx_version.lower():  # Require recognizable native version output while accepting CalculiX's documented nonzero version-only exit on packaged builds.
        raise FreezeError("CalculiX version query did not identify the solver")  # Refuse an environment snapshot that cannot identify its solver.
    pip_process = subprocess.run((sys.executable, "-m", "pip", "freeze", "--all"), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)  # Capture the complete Python lock from the active interpreter.
    if pip_process.returncode != 0:  # Require dependency-lock capture to succeed.
        raise FreezeError("python -m pip freeze --all failed")  # Prevent an incomplete environment lock.
    gmsh_process = subprocess.run((sys.executable, "-c", "import gmsh; print(gmsh.__version__)  # Report the loaded native-interface runtime version."), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)  # Prove the active interpreter can load Gmsh and report its runtime version.
    if gmsh_process.returncode != 0 or not gmsh_process.stdout.strip():  # Require the native Gmsh interface rather than package metadata alone.
        raise FreezeError(f"Gmsh runtime version query failed: {gmsh_process.stderr.strip()}")  # Stop before freezing an environment that cannot mesh.
    return {"schema": ENVIRONMENT_SCHEMA, "protocol_id": PROTOCOL_ID, "captured_utc": _utc_now(), "python": {"executable": sys.executable, "implementation": platform.python_implementation(), "version": platform.python_version(), "version_info": list(sys.version_info[:5])}, "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "platform": platform.platform()}, "packages": versions, "pip_freeze": sorted(line for line in pip_process.stdout.splitlines() if line.strip()), "gmsh": {"distribution_version": versions.get("gmsh"), "runtime_version": gmsh_process.stdout.strip()}, "calculix": {"executable": str(Path(ccx_path).resolve()), "executable_sha256": sha256_file(Path(ccx_path).resolve()), "version_output": ccx_version, "version_query_returncode": int(process.returncode)}, "environment_variables": {name: os.environ.get(name) for name in APPROVED_ENVIRONMENT_VARIABLES}}  # Return complete non-secret reproducibility evidence including exact native solver bytes.


def _environment_contract(payload: Mapping[str, Any]) -> dict[str, Any]:  # Select stable live fields that must match the frozen native and Python environment exactly.
    if payload.get("schema") != ENVIRONMENT_SCHEMA or payload.get("protocol_id") != PROTOCOL_ID:  # Require the exact protocol-owned environment document before comparing selected fields.
        raise FreezeError("environment snapshot has an invalid schema or protocol_id")  # Reject unrelated or incomplete injected environment records.
    python_payload = payload.get("python", {})  # Read the runtime interpreter block.
    platform_payload = payload.get("platform", {})  # Read the operating-system and architecture block.
    gmsh_payload = payload.get("gmsh", {})  # Read the loaded native mesher interface block.
    calculix_payload = payload.get("calculix", {})  # Read the native solver block.
    python_values = dict(python_payload) if isinstance(python_payload, Mapping) else {}  # Normalize only transparent named interpreter evidence.
    platform_values = dict(platform_payload) if isinstance(platform_payload, Mapping) else {}  # Normalize only transparent named platform evidence.
    gmsh_values = dict(gmsh_payload) if isinstance(gmsh_payload, Mapping) else {}  # Normalize only transparent named mesher evidence.
    calculix_values = dict(calculix_payload) if isinstance(calculix_payload, Mapping) else {}  # Normalize only transparent named solver evidence.
    required_python = ("implementation", "version", "version_info")  # Name every interpreter identity required by the live lock.
    required_platform = ("system", "release", "machine", "platform")  # Name every operating-system identity required by the live lock.
    if any(key not in python_values for key in required_python) or any(key not in platform_values for key in required_platform):  # Reject partial interpreter or runner platform snapshots.
        raise FreezeError("environment snapshot lacks complete Python or platform identity")  # Prevent missing fields from comparing equal by omission.
    packages = payload.get("packages")  # Read the reviewed direct dependency version map.
    if not isinstance(packages, Mapping) or set(packages) != set(ENVIRONMENT_DISTRIBUTIONS) or any(not isinstance(packages[name], str) or not packages[name] for name in ENVIRONMENT_DISTRIBUTIONS):  # Require every declared package to be installed with a nonempty version.
        raise FreezeError("environment snapshot lacks complete installed dependency versions")  # Reject a freeze that cannot reproduce every required Python dependency.
    pip_freeze = payload.get("pip_freeze")  # Read the complete transitive Python environment lock.
    if not isinstance(pip_freeze, list) or not pip_freeze or any(not isinstance(line, str) or not line for line in pip_freeze):  # Require a nonempty transparent installed-distribution list.
        raise FreezeError("environment snapshot lacks a complete pip freeze lock")  # Prevent direct dependency versions from hiding transitive drift.
    if any(not isinstance(gmsh_values.get(key), str) or not gmsh_values.get(key) for key in ("distribution_version", "runtime_version")):  # Require both package and successfully loaded native-interface versions.
        raise FreezeError("environment snapshot lacks complete Gmsh identity")  # Refuse a metadata-only or unloaded Gmsh environment.
    if not isinstance(calculix_values.get("version_output"), str) or not calculix_values.get("version_output") or type(calculix_values.get("version_query_returncode")) is not int or _SHA256_RE.fullmatch(str(calculix_values.get("executable_sha256", ""))) is None:  # Require recognizable native solver output, exact executable bytes, and its exact query status.
        raise FreezeError("environment snapshot lacks complete CalculiX identity")  # Refuse an ambiguous or unqueried solver environment.
    variables = payload.get("environment_variables")  # Read only the approved deterministic thread-control variables.
    if not isinstance(variables, Mapping) or set(variables) != set(APPROVED_ENVIRONMENT_VARIABLES) or any(value is not None and not isinstance(value, str) for value in variables.values()):  # Require an exact non-secret variable set and transparent string values.
        raise FreezeError("environment snapshot lacks the exact approved thread-variable contract")  # Prevent omitted or newly introduced runtime controls.
    return {"python": {key: python_values[key] for key in required_python}, "platform": {key: platform_values[key] for key in required_platform}, "packages": {name: packages[name] for name in ENVIRONMENT_DISTRIBUTIONS}, "pip_freeze": list(pip_freeze), "gmsh": {key: gmsh_values[key] for key in ("distribution_version", "runtime_version")}, "calculix": {key: calculix_values[key] for key in ("executable_sha256", "version_output", "version_query_returncode")}, "environment_variables": {name: variables[name] for name in APPROVED_ENVIRONMENT_VARIABLES}}  # Exclude only installation paths and timestamps while retaining complete Python, Gmsh, exact CalculiX bytes, platform, dependency, and thread identities.


def _payload_sha256(payload: Mapping[str, Any]) -> str:  # Hash one finite mapping through canonical strict JSON.
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")  # Remove whitespace, locale, and key-order ambiguity.
    return hashlib.sha256(encoded).hexdigest()  # Return the complete contract identity.


def _normalize_config(root: Path, source: Mapping[str, Any], implementation_commit: str, manifest_record: Mapping[str, Any], partition_inventory: Mapping[str, Any], model_inventory: Sequence[Mapping[str, Any]], evidence_inventory: Sequence[Mapping[str, Any]], partition_hash_record: Mapping[str, Any], model_hash_record: Mapping[str, Any], training_cost_record: Mapping[str, Any], environment_record: Mapping[str, Any], git_record: Mapping[str, Any], code_record: Mapping[str, Any], code_inventory: Mapping[str, Any], amendment_record: Mapping[str, Any] | None, source_sha256: str) -> dict[str, Any]:  # Build the sole immutable benchmark configuration from validated inputs.
    config = dict(source)  # Preserve reviewed scientific and baseline settings before adding generated identities.
    for transient in ("training_cost_sources",):  # Remove source-only aggregation instructions from runtime policy input.
        config.pop(transient, None)  # Keep the frozen runtime configuration focused on immutable outputs.
    config.update({"schema": CONFIG_SCHEMA, "protocol_id": PROTOCOL_ID, "TEST_NOT_RUN": True, "frozen_utc": _utc_now(), "implementation_commit_sha": implementation_commit, "freeze_git_ref": FREEZE_GIT_REF, "config_source_sha256": source_sha256, "budgets": list(BUDGETS), "solve_limits": list(SOLVE_LIMITS), "max_solves": max(SOLVE_LIMITS), "theta": 0.5, "allow_unqualified_references": _allow_unqualified_references(source), "expedited_reference_levels": EXPEDITED_REFERENCE_LEVELS if _allow_unqualified_references(source) else None, "reference_execution_amendment": dict(amendment_record) if amendment_record is not None else None, "case_manifest_sha256": str(manifest_record["sha256"]), "case_manifest_path": str(manifest_record["path"]), "partition_root": str(partition_inventory["partition_root"]), "partition_hashes": dict(partition_hash_record), "partition_spec_sha256": {str(record["case_id"]): str(record["sha256"]) for record in partition_inventory["partitions"]}, "model_hashes": dict(model_hash_record), "model_artifacts": [dict(record) for record in model_inventory], "training_validation_artifacts": [dict(record) for record in evidence_inventory], "reference_config": _reference_config(), "scientific_config_index": _validate_scientific_config(source), "complete_v0_config": _validate_complete_v0_config(source), "training_costs": dict(training_cost_record), "environment": dict(environment_record), "git_state": dict(git_record), "code_hashes": dict(code_record), "code_sha256": {str(record["path"]): str(record["sha256"]) for record in code_inventory["files"]}})  # Inject every generated exact-byte identity and fixed execution contract.
    return config  # Return the complete pre-test runtime and provenance document.


def _protected_record(root: Path, path: Path | str, role: str) -> dict[str, Any]:  # Build one exact-byte freeze-index entry.
    target, relative = _campaign_relative(root, path)  # Resolve a regular campaign-owned artifact.
    return {"role": role, "path": relative, "sha256": sha256_file(target), "size_bytes": target.stat().st_size}  # Preserve role, portable path, full digest, and size.


def _merge_protected(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:  # Merge legitimate multi-role evidence paths without weakening exact-byte identity.
    merged: dict[str, dict[str, Any]] = {}  # Index one unique digest and size record per portable path.
    for record in records:  # Traverse all generated and source evidence roles.
        path = str(record["path"])  # Normalize the unique campaign-relative path identity.
        role = str(record["role"])  # Read the transparent evidence role.
        existing = merged.get(path)  # Inspect any prior role for the same exact file.
        if existing is None:  # Create the first identity record for this path.
            merged[path] = {"roles": [role], "path": path, "sha256": str(record["sha256"]), "size_bytes": int(record["size_bytes"])}  # Preserve exact bytes and an extensible role list.
        elif existing["sha256"] != str(record["sha256"]) or existing["size_bytes"] != int(record["size_bytes"]):  # Defend impossible inconsistent identities for one path.
            raise FreezeError(f"protected artifact identity changed while building index: {path}")  # Stop before publishing a contradictory seal.
        elif role not in existing["roles"]:  # Add a distinct legitimate training, validation, or cost role once.
            existing["roles"].append(role)  # Preserve all scientific uses of the same immutable receipt.
    for record in merged.values():  # Normalize multi-role ordering for byte-deterministic review.
        record["roles"] = sorted(record["roles"])  # Make role order independent of source inventory ordering.
    return [merged[path] for path in sorted(merged)]  # Return one exact protected entry per path in stable order.


def create_freeze(root: Path | str, config_source: Path | str, implementation_commit: str, repository_root: Path | str, *, environment: Mapping[str, Any] | None = None, dry_run: bool = False) -> dict[str, Any]:  # Validate inputs and atomically create the complete pre-blind-test bundle.
    campaign = Path(root).resolve()  # Resolve the selected campaign root for every later path check.
    repository = Path(repository_root).resolve()  # Resolve the reviewed Git worktree boundary.
    manifest, manifest_record = _manifest_record(campaign)  # Authenticate case design and split boundaries first.
    disclosed = scan_for_disclosed_results(campaign, manifest)  # Scan blind results, ablations, aggregates, and test references before reading models.
    if disclosed:  # Refuse any post-disclosure or partially executed campaign.
        raise FreezeError("TEST_NOT_RUN violation; forbidden artifacts exist: " + ", ".join(disclosed))  # Return every contaminating path together.
    source_path = Path(config_source).resolve()  # Resolve the reviewed artifact-and-policy inventory source.
    source = _json_object(source_path, "freeze config source")  # Decode complete finite source settings.
    if source.get("TEST_NOT_RUN", True) is not True:  # Reject any source already marked as test-executed.
        raise FreezeError("freeze config source must declare TEST_NOT_RUN=true or omit it")  # Prevent post-test configuration recycling.
    scientific_snapshot = _validate_scientific_config(source)  # Require every preregistered policy setting during dry-run as well as creation.
    complete_v0_snapshot = _validate_complete_v0_config(source)  # Require every component dataclass field and common gradation during rehearsal.
    allow_unqualified = _allow_unqualified_references(source)  # Resolve the explicit fail-closed reference-qualification override before hashing or publication.
    amendment_record = _reference_amendment_record(campaign, required=allow_unqualified)  # Require and hash the fixed post-registration authorization artifact whenever the override is active.
    git_payload = _validate_implementation_worktree(repository, campaign, implementation_commit)  # Prove code equals the required implementation commit.
    code_payload = _code_inventory(repository, implementation_commit)  # Bind all executable claim-critical source bytes to the same commit.
    partitions = _partition_inventory(campaign, manifest, source)  # Authenticate one exact shared partition per all 48 cases.
    model_records = _hash_declared_artifacts(campaign, _entry_list(source, "model_artifacts"), require_metadata=("method",))  # Hash every selected deployment model explicitly.
    _validate_models(source, model_records)  # Enforce one world model, one selected supervised network, and all three RL policies.
    evidence_records = _training_validation_inventory(campaign, source)  # Authenticate all training and validation-selection receipts.
    training_cost_payload, training_cost_sources = _training_costs(campaign, source)  # Combine and bind all offline cost evidence.
    environment_payload = dict(environment) if environment is not None else capture_environment()  # Capture the live runtime or use an injected finite test snapshot.
    environment_payload.setdefault("schema", ENVIRONMENT_SCHEMA)  # Require the generated schema even for injected controlled tests.
    environment_payload.setdefault("protocol_id", PROTOCOL_ID)  # Bind injected controlled snapshots to the same protocol.
    _environment_contract(environment_payload)  # Reject any incomplete interpreter, dependency, native tool, platform, or thread lock before the first freeze write.
    plan = {"schema": FREEZE_SCHEMA, "protocol_id": PROTOCOL_ID, "dry_run": bool(dry_run), "TEST_NOT_RUN": True, "implementation_commit_sha": implementation_commit, "freeze_git_ref": FREEZE_GIT_REF, "allow_unqualified_references": allow_unqualified, "expedited_reference_levels": EXPEDITED_REFERENCE_LEVELS if allow_unqualified else None, "reference_execution_amendment": dict(amendment_record) if amendment_record is not None else None, "manifest": manifest_record, "scientific_config_index": scientific_snapshot, "complete_v0_config": complete_v0_snapshot, "code_file_count": code_payload["file_count"], "partition_count": len(partitions["partitions"]), "model_count": len(model_records), "training_validation_artifact_count": len(evidence_records), "training_cost_source_count": len(training_cost_sources), "disclosed_artifacts": disclosed, "outputs": ["protocol/partition_hashes.json", "protocol/model_hashes.json", "protocol/code_hashes.json", "training/training_costs.json", "protocol/environment.json", "protocol/git_state.json", "protocol/frozen_config.json", "protocol/freeze_index.json", "protocol/freeze_index.json.sha256"]}  # Assemble a solve-free complete creation plan.
    if dry_run:  # Preserve all expensive artifact and integrity checks while making no filesystem mutation.
        return plan  # Return the exact planned outputs for CI review.
    protocol = campaign / "protocol"  # Resolve the mandatory protocol artifact directory.
    outputs = [protocol / "partition_hashes.json", protocol / "model_hashes.json", protocol / "code_hashes.json", campaign / "training" / "training_costs.json", protocol / "environment.json", protocol / "git_state.json", protocol / "frozen_config.json", protocol / "freeze_index.json", protocol / "freeze_index.json.sha256"]  # Enumerate every generated freeze file before the first write.
    existing = [str(path) for path in outputs if path.exists()]  # Detect overwrite, resume, or post-hoc replacement attempts.
    if existing:  # Require one clean one-shot freeze publication.
        raise FreezeError("freeze outputs already exist; verify instead of overwriting: " + ", ".join(existing))  # Protect prior exact bytes and review history.
    partition_path = protocol / "partition_hashes.json"  # Resolve the full per-case semantic inventory artifact.
    model_path = protocol / "model_hashes.json"  # Resolve the complete deployment-model inventory artifact.
    code_path = protocol / "code_hashes.json"  # Resolve the complete implementation source-byte inventory artifact.
    training_cost_path = campaign / "training" / "training_costs.json"  # Resolve the combined required offline-cost artifact.
    environment_path = protocol / "environment.json"  # Resolve the reproducibility lock artifact.
    git_path = protocol / "git_state.json"  # Resolve the implementation provenance artifact.
    _write_json(partition_path, partitions)  # Publish partition hashes before the config references them.
    _write_json(model_path, {"schema": MODEL_HASH_SCHEMA, "protocol_id": PROTOCOL_ID, "TEST_NOT_RUN": True, "models": model_records})  # Publish exact selected model identities.
    _write_json(code_path, code_payload)  # Publish exact SHA-256 and Git blob identities for all claim-critical code.
    _write_json(training_cost_path, training_cost_payload)  # Publish combined training-cost evidence.
    _write_json(environment_path, environment_payload)  # Publish the native and Python environment lock.
    _write_json(git_path, git_payload)  # Publish the exact implementation commit provenance.
    generated_records = {"partition_hashes": _protected_record(campaign, partition_path, "partition_hash_inventory"), "model_hashes": _protected_record(campaign, model_path, "model_hash_inventory"), "code_hashes": _protected_record(campaign, code_path, "code_hash_inventory"), "training_costs": _protected_record(campaign, training_cost_path, "training_cost_aggregate"), "environment": _protected_record(campaign, environment_path, "environment_lock"), "git_state": _protected_record(campaign, git_path, "implementation_git_state")}  # Compute immutable identities used by frozen_config.
    frozen_config = _normalize_config(campaign, source, implementation_commit, manifest_record, partitions, model_records, evidence_records, generated_records["partition_hashes"], generated_records["model_hashes"], generated_records["training_costs"], generated_records["environment"], generated_records["git_state"], generated_records["code_hashes"], code_payload, amendment_record, sha256_file(source_path))  # Build the sole runtime configuration after every dependency is persisted.
    config_path = protocol / "frozen_config.json"  # Resolve the mandatory frozen runtime configuration filename.
    _write_json(config_path, frozen_config)  # Publish the exact config consumed by blind benchmark shards.
    protected: list[dict[str, Any]] = []  # Collect every immutable dependency under one verification index.
    protected.extend((_protected_record(campaign, "protocol/case_manifest.json", "case_manifest"), _protected_record(campaign, "protocol/case_manifest.sha256", "case_manifest_sidecar")))  # Protect both case bytes and conventional checksum evidence.
    protected.extend(_protected_record(campaign, record["path"], "partition_spec") for record in partitions["partitions"])  # Protect all 48 exact partition files.
    protected.extend(_protected_record(campaign, record["path"], "deployment_model") for record in model_records)  # Protect every selected learned-model snapshot.
    protected.extend(_protected_record(campaign, record["path"], "training_validation_evidence") for record in evidence_records)  # Protect checkpoint-selection and training receipts.
    protected.extend(_protected_record(campaign, record["path"], "training_cost_source") for record in training_cost_sources)  # Protect the source cost files embedded in the aggregate.
    protected.extend(generated_records.values())  # Protect all generated inventories, costs, environment, and Git provenance.
    if amendment_record is not None:  # Protect the exact post-registration authorization whenever it exists in the freeze bundle.
        protected.append(_protected_record(campaign, amendment_record["path"], "reference_execution_amendment"))  # Bind the human authorization bytes under a transparent dedicated role.
    protected.append(_protected_record(campaign, config_path, "frozen_config"))  # Protect the exact runtime configuration after it is fully materialized.
    protected = _merge_protected(protected)  # Coalesce exact training receipts that legitimately serve cost and validation roles.
    index = {"schema": FREEZE_SCHEMA, "protocol_id": PROTOCOL_ID, "created_utc": _utc_now(), "TEST_NOT_RUN": True, "implementation_commit_sha": implementation_commit, "freeze_git_ref": FREEZE_GIT_REF, "freeze_commit_required": True, "allow_unqualified_references": allow_unqualified, "expedited_reference_levels": EXPEDITED_REFERENCE_LEVELS if allow_unqualified else None, "reference_execution_amendment": dict(amendment_record) if amendment_record is not None else None, "case_manifest_sha256": manifest_record["sha256"], "reference_config": _reference_config(), "artifact_count": len(protected), "protected_artifacts": protected}  # Assemble the non-self-referential complete freeze seal.
    index_path = protocol / "freeze_index.json"  # Resolve the complete immutable-input seal.
    _write_json(index_path, index)  # Publish the exact-byte inventory last.
    sidecar_path = _write_sidecar(index_path)  # Publish an independently checkable conventional digest.
    return {**plan, "dry_run": False, "freeze_index_sha256": sha256_file(index_path), "freeze_index": str(index_path), "sidecar": str(sidecar_path)}  # Return exact creation evidence without reading test outputs.


def _verify_sidecar(index_path: Path) -> str:  # Authenticate the freeze index before trusting any protected path.
    sidecar = index_path.with_suffix(index_path.suffix + ".sha256")  # Resolve the standard sibling checksum file.
    try:  # Convert missing and malformed sidecars into one protocol failure.
        fields = sidecar.read_text(encoding="ascii").strip().split()  # Parse the exact digest and filename fields.
    except (OSError, UnicodeError) as exc:  # Catch unavailable or non-ASCII sidecars.
        raise FreezeError(f"cannot read freeze index sidecar: {sidecar}") from exc  # Preserve the original failure cause.
    observed = sha256_file(index_path)  # Hash the exact index bytes independently.
    if fields != [observed, index_path.name]:  # Require the sidecar to name and authenticate exactly this index.
        raise FreezeError("freeze_index.json.sha256 does not authenticate freeze_index.json")  # Reject mutated or substituted indexes.
    return observed  # Return the authenticated exact-byte identity.


def _verify_config_pointer(campaign: Path, config: Mapping[str, Any], key: str, expected_path: str) -> dict[str, Any]:  # Authenticate one generated artifact pointer embedded in frozen_config.
    value = config.get(key)  # Read the hash-bearing generated-artifact record.
    if not isinstance(value, Mapping) or value.get("path") != expected_path or "sha256" not in value:  # Require the exact canonical path and full digest.
        raise FreezeError(f"frozen_config has an invalid {key} artifact pointer")  # Reject split-brain inventory locations.
    target, relative = _campaign_relative(campaign, expected_path)  # Resolve the canonical protected artifact.
    digest = sha256_file(target)  # Recompute its exact persisted identity independently.
    if digest != str(value["sha256"]) or int(value.get("size_bytes", -1)) != target.stat().st_size:  # Require both hash and length to match the config pointer.
        raise FreezeError(f"frozen_config {key} pointer does not match {relative}")  # Reject stale or substituted generated evidence.
    return dict(value)  # Return the authenticated pointer for the verification report.


def _verify_training_costs(campaign: Path) -> dict[str, Any]:  # Revalidate the combined cost report and every embedded source hash and payload.
    path, relative = _campaign_relative(campaign, "training/training_costs.json")  # Resolve the mandatory aggregate cost artifact.
    payload = _json_object(path, "training cost aggregate")  # Decode finite cost evidence after protected hash authentication.
    if payload.get("schema") != TRAINING_COST_SCHEMA or payload.get("protocol_id") != PROTOCOL_ID or payload.get("TEST_NOT_RUN") is not True:  # Require the exact pre-test cost schema.
        raise FreezeError("training_costs.json is not a frozen pre-test cost aggregate")  # Reject unrelated or post-test cost reports.
    if payload.get("methods") != list(MODEL_METHODS):  # Require world, supervised, and RL cost coverage in frozen order.
        raise FreezeError("training_costs.json does not cover all learned methods")  # Prevent missing amortization evidence.
    sources = payload.get("sources")  # Read embedded exact source receipts.
    if not isinstance(sources, list) or {str(source.get("method")) for source in sources if isinstance(source, Mapping)} != set(MODEL_METHODS):  # Require all three method families.
        raise FreezeError("training_costs.json has incomplete source coverage")  # Reject malformed or incomplete embedded sources.
    for source in sources:  # Recompute every upstream source independently from the aggregate's own protected hash.
        if not isinstance(source, Mapping) or not all(key in source for key in ("path", "sha256", "payload")):  # Require transparent provenance and embedded finite content.
            raise FreezeError("training_costs.json contains a malformed source record")  # Reject opaque cost evidence.
        target, source_relative = _campaign_relative(campaign, str(source["path"]))  # Resolve the explicit campaign-owned source file.
        if sha256_file(target) != str(source["sha256"]) or _json_object(target, f"training cost source {source_relative}") != source["payload"]:  # Bind exact source bytes and decoded values together.
            raise FreezeError(f"training cost source changed or disagrees with aggregate: {source_relative}")  # Reject altered or split-brain costs.
    return {"path": relative, "sha256": sha256_file(path), "source_count": len(sources)}  # Return concise independently verified cost provenance.


def _verify_frozen_inventories(campaign: Path, manifest: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:  # Revalidate partition, model, training, validation, and generated-index identities from frozen_config.
    pointers = {"partition_hashes": _verify_config_pointer(campaign, config, "partition_hashes", "protocol/partition_hashes.json"), "model_hashes": _verify_config_pointer(campaign, config, "model_hashes", "protocol/model_hashes.json"), "training_costs": _verify_config_pointer(campaign, config, "training_costs", "training/training_costs.json"), "environment": _verify_config_pointer(campaign, config, "environment", "protocol/environment.json"), "git_state": _verify_config_pointer(campaign, config, "git_state", "protocol/git_state.json"), "code_hashes": _verify_config_pointer(campaign, config, "code_hashes", "protocol/code_hashes.json")}  # Authenticate every generated hash-bearing pointer.
    partitions = _partition_inventory(campaign, manifest, config)  # Rebind every one of 48 semantic specs to its manifest geometry.
    expected_partition_map = {str(record["case_id"]): str(record["sha256"]) for record in partitions["partitions"]}  # Build the independent exact per-case map.
    if config.get("partition_spec_sha256") != expected_partition_map:  # Require exact case set and file hashes in the runtime config.
        raise FreezeError("partition_spec_sha256 differs from live 48-case partition inventory")  # Reject omitted or swapped partitions.
    partition_index = _json_object(campaign / "protocol" / "partition_hashes.json", "partition hash inventory")  # Decode the separately protected full inventory.
    if partition_index != partitions:  # Require semantic, geometry, path, size, and file hashes to match independently.
        raise FreezeError("partition_hashes.json differs from recomputed partition inventory")  # Reject stale generated inventories.
    models = _hash_declared_artifacts(campaign, _entry_list(config, "model_artifacts"), require_metadata=("method",))  # Rehash every configured deployment model.
    _validate_models(config, models)  # Reconfirm one world, selected supervised, and three exact RL policies.
    model_index = _json_object(campaign / "protocol" / "model_hashes.json", "model hash inventory")  # Decode the separately protected model set.
    if model_index.get("schema") != MODEL_HASH_SCHEMA or model_index.get("models") != models:  # Require the generated inventory to match runtime declarations and live bytes.
        raise FreezeError("model_hashes.json differs from recomputed deployment model inventory")  # Reject model selection or hash drift.
    evidence = _training_validation_inventory(campaign, config)  # Rehash and revalidate every required method-phase receipt.
    configured_evidence = config.get("training_validation_artifacts")  # Read frozen runtime evidence declarations.
    if configured_evidence != evidence:  # Require exact metadata, paths, hashes, and sizes.
        raise FreezeError("training_validation_artifacts differ from recomputed evidence inventory")  # Reject post-freeze checkpoint-selection evidence drift.
    costs = _verify_training_costs(campaign)  # Revalidate all three cost sources and their embedded payloads.
    return {"pointers": pointers, "partition_count": len(partitions["partitions"]), "model_count": len(models), "training_validation_artifact_count": len(evidence), "training_costs": costs}  # Return complete independently recomputed inventory evidence.


def _verify_committed_freeze(repo: Path, root: Path, implementation_commit: str, *, require_tag: bool = True, allowed_untracked: set[str] | None = None) -> dict[str, Any]:  # Require a code-clean dedicated descendant freeze commit before blind execution.
    head = _git(repo, "rev-parse", "HEAD")  # Resolve the exact commit selected by the blind workflow.
    if head == implementation_commit:  # Require a distinct post-training commit that can contain only the generated freeze evidence.
        raise FreezeError("dedicated freeze commit must differ from implementation_commit")  # Reject a tag placed directly on implementation code without the mandatory committed artifact boundary.
    parent_record = _git(repo, "rev-list", "--parents", "-n", "1", head).split()  # Enumerate the freeze commit and every direct parent without hiding merge ancestry.
    if len(parent_record) != 2:  # Require exactly one parent rather than accepting a merge commit whose first parent happens to be the implementation.
        raise FreezeError("dedicated freeze commit must be a single-parent commit")  # Preserve one linear reviewed implementation-to-freeze transition.
    freeze_parent = parent_record[1]  # Recover the sole direct parent after exact cardinality validation.
    if freeze_parent != implementation_commit:  # Forbid intermediate campaign commits or later descendants from silently replacing the reviewed one-shot freeze.
        raise FreezeError("dedicated freeze commit must have implementation_commit as its direct parent")  # Require one auditable artifact-only commit immediately after implementation review.
    ancestor = subprocess.run(("git", "-C", str(repo), "merge-base", "--is-ancestor", implementation_commit, head), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # Verify reviewed implementation ancestry without shell expansion.
    if ancestor.returncode != 0:  # Reject unrelated, rebased, or stale freeze commits.
        raise FreezeError("implementation_commit is not an ancestor of the current freeze commit")  # Preserve the reviewed code lineage.
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=no"):  # Require no staged or unstaged mutation of any tracked code or frozen artifact.
        raise FreezeError("blind preflight requires no tracked worktree changes")  # Prevent local config, model, code, or sealed-evidence drift.
    untracked_text = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")  # Enumerate untracked files without quote ambiguity.
    untracked = {value for value in untracked_text.split("\0") if value}  # Recover exact repository-relative paths.
    unexpected_untracked = sorted(untracked - set(allowed_untracked or set()))  # Permit only explicitly enumerated post-freeze test-reference cache files.
    if unexpected_untracked:  # Reject test results, unknown reference files, and any uncommitted source.
        raise FreezeError("blind preflight found unauthorized untracked files: " + ", ".join(unexpected_untracked))  # Report every code-cleanliness violation together.
    repository = repo.resolve()  # Resolve the Git worktree boundary for path comparisons.
    campaign_relative = root.resolve().relative_to(repository).as_posix()  # Derive the only allowed implementation-to-freeze diff prefix.
    changed = [value for value in _git(repo, "diff", "--name-only", f"{implementation_commit}..{head}").splitlines() if value]  # Enumerate committed changes after the implementation checkpoint.
    outside = [value for value in changed if value != campaign_relative and not value.startswith(campaign_relative + "/")]  # Detect any post-freeze policy, scoring, workflow, or source mutation.
    if outside:  # Require the dedicated freeze commit to contain campaign evidence only.
        raise FreezeError("freeze commit changes files outside the campaign root: " + ", ".join(sorted(outside)))  # Report the exact code-drift paths.
    tag_target: str | None = None  # Reserve the immutable non-self-referential freeze reference target.
    if require_tag:  # Enforce tag sealing only after the dedicated freeze commit has been published.
        tag_target = _git(repo, "rev-parse", "--verify", FREEZE_GIT_REF, check=False) or None  # Resolve the required lightweight tag without accepting an absent ref.
        if tag_target != head:  # Require the fixed freeze ref to identify exactly the selected blind HEAD.
            raise FreezeError(f"{FREEZE_GIT_REF} must resolve exactly to current HEAD {head}")  # Reject unsealed, moved-branch, or stale commits.
    return {"freeze_commit_sha": head, "freeze_parent_sha": freeze_parent, "head_sha": head, "freeze_git_ref": FREEZE_GIT_REF, "freeze_git_ref_target_sha": tag_target, "implementation_commit_sha": implementation_commit, "implementation_is_ancestor": True, "tracked_worktree_clean": True, "allowed_untracked_paths": sorted(untracked), "post_implementation_changed_paths": sorted(changed)}  # Return auditable blind Git readiness.


def verify_freeze(root: Path | str, repository_root: Path | str, *, require_committed: bool = False, live_environment: Mapping[str, Any] | None = None, allow_postfreeze_test_references: bool = False) -> dict[str, Any]:  # Authenticate the complete freeze without reading any method test result value.
    campaign = Path(root).resolve()  # Resolve the campaign boundary once.
    repository = Path(repository_root).resolve()  # Resolve the Git worktree for optional committed-freeze verification.
    manifest, manifest_record = _manifest_record(campaign)  # Re-authenticate exact case bytes and split boundaries.
    disclosed = scan_for_disclosed_results(campaign, manifest, allow_postfreeze_test_references=allow_postfreeze_test_references)  # Reconfirm no blind result, ablation, aggregate, or unauthorized test reference has appeared.
    if disclosed:  # Refuse a second or contaminated blind launch.
        raise FreezeError("TEST_NOT_RUN violation; forbidden artifacts exist: " + ", ".join(disclosed))  # Report every disclosure path together.
    index_path, index_relative = _campaign_relative(campaign, "protocol/freeze_index.json")  # Resolve the unique freeze seal.
    index_digest = _verify_sidecar(index_path)  # Authenticate index bytes before resolving its content.
    index = _json_object(index_path, "freeze index")  # Decode only the authenticated exact index.
    if index.get("schema") != FREEZE_SCHEMA or index.get("protocol_id") != PROTOCOL_ID or index.get("TEST_NOT_RUN") is not True or index.get("freeze_git_ref") != FREEZE_GIT_REF:  # Require the exact pre-test schema, declaration, and non-self-referential commit ref.
        raise FreezeError("freeze index is not an authenticated WMVLA-4WAY-P1 TEST_NOT_RUN bundle")  # Reject stale or post-test indexes.
    if index.get("reference_config") != _reference_config():  # Require the exact current preregistered reference schedule.
        raise FreezeError("freeze index reference_config differs from DEFAULT_REFERENCE_CONFIG")  # Prevent reference acceptance-rule drift.
    entries = index.get("protected_artifacts")  # Read the exact protected file inventory.
    if not isinstance(entries, list) or not entries:  # Require a nonempty explicit seal.
        raise FreezeError("freeze index lacks protected_artifacts")  # Reject vacuous integrity claims.
    verified: list[dict[str, Any]] = []  # Collect recomputed exact identities for workflow evidence.
    seen: set[str] = set()  # Reject duplicate paths in a hand-edited index.
    for entry in entries:  # Verify every protected file before any blind reference is generated.
        if not isinstance(entry, Mapping) or not all(name in entry for name in ("roles", "path", "sha256", "size_bytes")):  # Require a complete inventory record.
            raise FreezeError(f"malformed protected artifact entry: {entry}")  # Refuse partially specified seals.
        target, relative = _campaign_relative(campaign, str(entry["path"]))  # Resolve only regular in-campaign files.
        if relative in seen:  # Reject ambiguous repeated paths.
            raise FreezeError(f"duplicate protected artifact path: {relative}")  # Preserve one identity per file.
        seen.add(relative)  # Retain the verified normalized path.
        digest = sha256_file(target)  # Recompute exact bytes at blind preflight time.
        if digest != str(entry["sha256"]) or target.stat().st_size != int(entry["size_bytes"]):  # Require both digest and recorded length to match.
            raise FreezeError(f"protected artifact changed after freeze: {relative}")  # Stop before opening the blind split.
        roles = entry.get("roles")  # Read every declared scientific use of the protected bytes.
        if not isinstance(roles, list) or not roles or any(not isinstance(role, str) or not role for role in roles):  # Require a transparent nonempty role set.
            raise FreezeError(f"protected artifact has invalid roles: {relative}")  # Reject opaque or malformed evidence identities.
        verified.append({"roles": sorted(set(roles)), "path": relative, "sha256": digest, "size_bytes": target.stat().st_size})  # Preserve exact verification evidence.
    if manifest_record["sha256"] != index.get("case_manifest_sha256"):  # Cross-check manifest identity independently from its protected entry.
        raise FreezeError("freeze index case_manifest_sha256 differs from case_manifest.json")  # Reject internal cross-document inconsistency.
    config_path, _config_relative = _campaign_relative(campaign, "protocol/frozen_config.json")  # Resolve the required runtime config separately.
    config = _json_object(config_path, "frozen config")  # Decode finite fields after exact-byte authentication.
    implementation_commit = str(index.get("implementation_commit_sha", ""))  # Read the reviewed code checkpoint.
    if config.get("TEST_NOT_RUN") is not True or config.get("implementation_commit_sha") != implementation_commit or config.get("freeze_git_ref") != FREEZE_GIT_REF or config.get("reference_config") != _reference_config():  # Require core declarations to agree across the config and seal.
        raise FreezeError("frozen_config core declarations disagree with freeze_index")  # Prevent split-brain runtime behavior.
    allow_unqualified = _allow_unqualified_references(config)  # Recover the authenticated explicit reference-qualification policy without coercion.
    if type(index.get("allow_unqualified_references")) is not bool or index.get("allow_unqualified_references") is not allow_unqualified:  # Require the root seal and runtime configuration to state exactly the same reviewed qualification override.
        raise FreezeError("freeze_index and frozen_config disagree on allow_unqualified_references")  # Prevent a caller from silently treating an unqualified reference as qualified.
    expected_expedited_levels = EXPEDITED_REFERENCE_LEVELS if allow_unqualified else None  # Derive the sole fixed operational prefix from the authenticated boolean policy.
    if config.get("expedited_reference_levels") != expected_expedited_levels or index.get("expedited_reference_levels") != expected_expedited_levels:  # Require runtime config and root seal to bind exactly the disclosed two-level mode or strict null.
        raise FreezeError("freeze_index and frozen_config disagree on expedited_reference_levels")  # Prevent post-freeze schedule-depth selection.
    amendment_record = _reference_amendment_record(campaign, required=allow_unqualified)  # Recompute exact live authorization evidence before permitting unqualified reference use.
    configured_amendment = config.get("reference_execution_amendment")  # Read the protected runtime pointer to the human amendment.
    if configured_amendment != amendment_record or index.get("reference_execution_amendment") != amendment_record:  # Require live bytes, frozen runtime policy, and the sidecar-authenticated root seal to agree exactly.
        raise FreezeError("reference execution amendment differs across live file, frozen_config, and freeze_index")  # Reject a missing, changed, stale, or unrecorded authorization artifact.
    active_reference_levels = expected_expedited_levels or len(DEFAULT_REFERENCE_CONFIG.background_scales)  # Resolve the authenticated strict six-level or authorized two-level evidence boundary.
    policy_disclosed = scan_for_disclosed_results(campaign, manifest, allow_postfreeze_test_references=allow_postfreeze_test_references, postfreeze_reference_level_count=active_reference_levels)  # Re-scan after config authentication so inactive ladder-level files cannot hide behind the broad first-pass boundary.
    if policy_disclosed:  # Refuse extra solver artifacts outside the exact frozen reference depth.
        raise FreezeError("TEST_NOT_RUN violation; forbidden artifacts exist: " + ", ".join(policy_disclosed))  # Report every depth-specific contamination path before blind access.
    complete_v0 = _validate_complete_v0_config(config)  # Revalidate every planner, model, tool, runtime, seed, and gradation field from frozen bytes.
    scientific = _validate_scientific_config(config)  # Revalidate all protocol-listed cross-stack scientific settings.
    inventories = _verify_frozen_inventories(campaign, manifest, config)  # Recompute partition, model, training, validation, cost, environment, Git, and code-index pointers.
    current_code = _code_inventory(repository, implementation_commit)  # Recompute every live claim-critical source identity against the implementation commit.
    expected_code = config.get("code_sha256")  # Read the frozen path-to-SHA mapping from the authenticated runtime config.
    observed_code = {str(record["path"]): str(record["sha256"]) for record in current_code["files"]}  # Build the independently recomputed live code map.
    if not isinstance(expected_code, Mapping) or dict(expected_code) != observed_code:  # Require exact path set and bytes without missing categories.
        raise FreezeError("live claim-critical code SHA mapping differs from frozen_config")  # Reject policy, scoring, baseline, or harness drift.
    environment_path, _environment_relative = _campaign_relative(campaign, "protocol/environment.json")  # Resolve the authenticated frozen environment lock.
    frozen_environment = _json_object(environment_path, "frozen environment")  # Decode exact stored version and native solver fields.
    observed_environment = dict(live_environment) if live_environment is not None else capture_environment()  # Capture the live execution environment or controlled unit fixture.
    frozen_contract = _environment_contract(frozen_environment)  # Select stable required frozen interpreter, dependency, platform, solver, and thread fields.
    live_contract = _environment_contract(observed_environment)  # Select the same stable live fields independently.
    if live_contract != frozen_contract:  # Require exact compatibility before opening a blind reference or method result.
        raise FreezeError(f"live environment differs from frozen environment; frozen={_payload_sha256(frozen_contract)} live={_payload_sha256(live_contract)}")  # Report compact deterministic identities without noisy dependency dumps.
    repository_campaign = campaign.relative_to(repository).as_posix()  # Derive repository-relative campaign paths for the Git cleanliness exception.
    allowed_reference_paths = {f"{repository_campaign}/{path.relative_to(campaign).as_posix()}" for path in campaign.glob("references/**/*") if path.is_file() and not path.is_symlink() and _permitted_postfreeze_reference_path(path.relative_to(campaign), _test_case_ids(manifest), active_reference_levels)} if allow_postfreeze_test_references else set()  # Permit only exact existing cache JSON and active frozen-depth solver-log or failed-level input paths after freeze.
    git_evidence = _verify_committed_freeze(repository, campaign, implementation_commit, allowed_untracked=allowed_reference_paths) if require_committed else {"require_committed": False}  # Enforce dedicated tag, code cleanliness, and only explicit reference-cache exceptions for blind launch.
    if require_committed:  # Require every protected exact-byte artifact to be stored in the selected freeze commit.
        repository_resolved = repository.resolve()  # Normalize the worktree boundary once.
        untracked_protected: list[str] = []  # Collect seal entries absent from Git history.
        for record in verified:  # Check each already authenticated campaign path.
            repository_path = (campaign / record["path"]).resolve().relative_to(repository_resolved).as_posix()  # Convert to a Git worktree-relative identity.
            if subprocess.run(("git", "-C", str(repository), "ls-files", "--error-unmatch", "--", repository_path), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:  # Query exact tracking without shell expansion.
                untracked_protected.append(repository_path)  # Retain the missing committed artifact.
        if untracked_protected:  # Reject a local-only model or evidence bundle.
            raise FreezeError("protected artifacts are not committed: " + ", ".join(sorted(untracked_protected)))  # Require the dedicated freeze commit promised by the protocol.
    return {"schema": FREEZE_SCHEMA, "protocol_id": PROTOCOL_ID, "TEST_NOT_RUN": True, "allow_unqualified_references": allow_unqualified, "expedited_reference_levels": expected_expedited_levels, "reference_execution_amendment": amendment_record, "allow_postfreeze_test_references": bool(allow_postfreeze_test_references), "postfreeze_test_reference_paths": sorted(allowed_reference_paths), "freeze_index": index_relative, "freeze_index_sha256": index_digest, "protected_artifact_count": len(verified), "manifest_sha256": manifest_record["sha256"], "implementation_commit_sha": implementation_commit, "freeze_git_ref": FREEZE_GIT_REF, "scientific_config_index": scientific, "complete_v0_config": complete_v0, "inventories": inventories, "code_sha256": observed_code, "environment_contract_sha256": _payload_sha256(live_contract), "git": git_evidence, "verified_artifacts": verified}  # Return complete pre-blind evidence without method result values.


def seal_freeze_tag(root: Path | str, repository_root: Path | str, *, live_environment: Mapping[str, Any] | None = None) -> dict[str, Any]:  # Seal one committed exact freeze through the fixed non-self-referential lightweight tag.
    campaign = Path(root).resolve()  # Resolve the immutable campaign root.
    repository = Path(repository_root).resolve()  # Resolve the Git worktree containing the dedicated freeze commit.
    verified = verify_freeze(campaign, repository, require_committed=False, live_environment=live_environment)  # Authenticate all bytes and live environment before creating a Git ref.
    implementation_commit = str(verified["implementation_commit_sha"])  # Recover the reviewed implementation identity from the authenticated index.
    git_evidence = _verify_committed_freeze(repository, campaign, implementation_commit, require_tag=False)  # Require a clean tracked campaign-only descendant commit.
    existing = subprocess.run(("git", "-C", str(repository), "show-ref", "--verify", "--quiet", FREEZE_GIT_REF), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Test exact ref existence without interpreting output.
    if existing.returncode == 0:  # Forbid moving or overwriting a previously disclosed freeze identity.
        raise FreezeError(f"freeze Git ref already exists and will not be overwritten: {FREEZE_GIT_REF}")  # Preserve permanent preregistration history.
    tag_name = FREEZE_GIT_REF.removeprefix("refs/tags/")  # Convert the fixed full ref to Git's lightweight tag argument.
    process = subprocess.run(("git", "-C", str(repository), "tag", tag_name, str(git_evidence["head_sha"])), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # Create the non-self-referential exact commit ref without shell expansion.
    if process.returncode != 0:  # Report ref permission or repository failures explicitly.
        raise FreezeError(f"cannot create freeze Git ref {FREEZE_GIT_REF}: {process.stderr.strip()}")  # Stop before claiming the campaign is sealed.
    sealed_git = _verify_committed_freeze(repository, campaign, implementation_commit, require_tag=True)  # Re-resolve and require tag==HEAD immediately after creation.
    return {**verified, "sealed": True, "git": sealed_git}  # Return exact bundle, live environment, code, head, and tag evidence.


def preflight_report(root: Path | str, repository_root: Path | str, implementation_commit: str | None = None) -> dict[str, Any]:  # Produce a non-mutating CI readiness report even before training artifacts exist.
    campaign = Path(root).resolve()  # Resolve the planned campaign boundary.
    repository = Path(repository_root).resolve()  # Resolve the implementation worktree.
    manifest: dict[str, Any] | None = None  # Permit an early plan before manifest generation.
    manifest_error: str | None = None  # Preserve any missing or invalid manifest as readiness evidence.
    try:  # Attempt the full manifest boundary check when artifacts already exist.
        manifest, _record = _manifest_record(campaign)  # Authenticate the exact generated design without writing it.
    except FreezeError as exc:  # Preserve an incomplete pre-training state rather than pretending readiness.
        manifest_error = str(exc)  # Record the precise missing or invalid input.
    disclosed = scan_for_disclosed_results(campaign, manifest)  # Detect any premature blind evidence independently from training readiness.
    head = _git(repository, "rev-parse", "HEAD")  # Record the exact PR or dispatch code revision.
    requested = implementation_commit or head  # Default a PR solve-free preflight to its checked-out exact revision.
    requested_valid = _GIT_SHA_RE.fullmatch(requested) is not None  # Require any explicitly reviewed identity to be complete and unambiguous.
    return {"schema": FREEZE_SCHEMA, "protocol_id": PROTOCOL_ID, "mode": "preflight", "dry_run": True, "writes_performed": False, "TEST_NOT_RUN": not disclosed, "repository_head_sha": head, "implementation_commit_sha": requested, "implementation_commit_valid": requested_valid, "implementation_matches_head": requested_valid and requested == head, "manifest_ready": manifest is not None, "manifest_error": manifest_error, "disclosed_artifacts": disclosed, "blind_launch_ready": False, "next_required_action": "complete training/validation and run create with an explicit config/artifact inventory"}  # Return transparent non-vacuous readiness status without failing merely because training is pending.
