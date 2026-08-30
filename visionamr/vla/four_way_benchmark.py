"""Execute the frozen four-way bridge benchmark without cross-case leakage."""  # Describe the module's single scientific responsibility.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from dataclasses import asdict, dataclass, is_dataclass  # Import immutable job contracts and generic record serialization.
from datetime import datetime, timezone  # Import unambiguous UTC audit timestamps.
import hashlib  # Import SHA-256 verification for every frozen input.
import json  # Import strict machine-readable artifact persistence.
import math  # Import finite-number checks for failure-safe records.
import os  # Import atomic job-claim and append operations for controlled shards.
from pathlib import Path  # Import portable repository and result paths.
import shutil  # Import solver-log copying into the protocol directory layout.
import time  # Import monotonic online timing.
import traceback  # Import complete failure diagnostics without dropping cases.
from typing import Any, Iterable, Mapping, Sequence  # Import explicit heterogeneous JSON and collection contracts.
from ..bridge_case_manifest import PROTOCOL_ID, load_case_manifest, problem_from_case  # Reuse the only validated case source and factory gateway.
from ..experiment import FemRunner, Reference, SolveRecord  # Reuse honest real-solve accounting and reference contracts.

BUDGETS = (30000, 60000, 120000)  # Freeze the three active-equation budgets from the protocol.
SOLVE_LIMITS = (2, 3, 4, 6)  # Freeze the four real-CalculiX prefix limits from the protocol.
MAX_SOLVES = 6  # Run each independent budget trajectory once to the longest registered prefix.
BASE_METHODS = ("world_model_vla", "local_prediction", "supervised", "dorfler")  # Freeze non-RL execution labels.
RL_METHODS = ("rl_seed0", "rl_seed1", "rl_seed2")  # Freeze the three independently trained RL policies.
ALL_METHODS = ("world_model_vla", "local_prediction", "supervised", *RL_METHODS, "dorfler")  # Freeze deterministic method execution order.
RESULT_SCHEMA = "wmvla-four-way-method-result-v1"  # Identify one raw method trajectory artifact.
PLAN_SCHEMA = "wmvla-four-way-execution-plan-v1"  # Identify a solve-free dry-run plan.

@dataclass(frozen=True)  # Make each execution job immutable after preregistration checks.
class ExecutionJob:  # Identify one independent case, budget, and frozen policy trajectory.
    case_id: str  # Store the content-bound manifest case identifier.
    split: str  # Store the manifest-owned split label.
    geometry_hash: str  # Bind the job to the exact canonical geometry.
    budget: int  # Store the active-equation cap.
    method: str  # Store the exact frozen method or RL seed label.
    output_dir: Path  # Store the protocol-layout destination for raw evidence.

@dataclass(frozen=True)  # Make one full invocation contract immutable.
class BenchmarkRequest:  # Carry validated paths, filters, and execution controls.
    root: Path  # Store the campaign root containing protocol, training, references, and test outputs.
    manifest_path: Path  # Store the exact case manifest path.
    frozen_config_path: Path  # Store the exact frozen policy configuration path.
    split: str = "test"  # Default to the preregistered blind split.
    case_ids: tuple[str, ...] = ()  # Optionally restrict a controlled shard to explicit manifest cases.
    methods: tuple[str, ...] = ALL_METHODS  # Default to all primary and safety trajectories.
    budgets: tuple[int, ...] = BUDGETS  # Default to all registered active-equation budgets.
    dry_run: bool = False  # Permit complete preflight and plan output without any solver call.
    resume: bool = False  # Permit verified completed jobs to be skipped in a restarted shard.
    development_run: bool = False  # Permit non-test smoke execution without a frozen blind-test state.
    allow_unqualified_references: bool = False  # Require explicit invocation opt-in before using an operational but non-qualified Reference B.

class FrozenInputError(RuntimeError):  # Distinguish pre-solve protocol violations from numerical method failures.
    """Report an invalid or incomplete frozen-input state."""  # Explain the exception category.

def _utc_now() -> str:  # Return one timezone-explicit audit timestamp.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # Serialize UTC without locale ambiguity.

def _sha256_file(path: Path) -> str:  # Hash exact file bytes for frozen-input verification.
    digest = hashlib.sha256()  # Initialize a fresh collision-resistant digest.
    with path.open("rb") as handle:  # Stream the artifact without loading a potentially large model into memory.
        for block in iter(lambda: handle.read(1024 * 1024), b""):  # Read stable one-megabyte blocks to EOF.
            digest.update(block)  # Incorporate every exact byte in order.
    return digest.hexdigest()  # Return the complete lowercase hexadecimal identity.

def _read_json(path: Path) -> Any:  # Load one strict UTF-8 JSON artifact.
    return json.loads(path.read_text(encoding="utf-8"))  # Preserve the native mapping and list structure for validation.

def _json_safe(value: Any) -> Any:  # Convert heterogeneous audit objects into finite strict-JSON values.
    if is_dataclass(value):  # Expand immutable repository records before recursive conversion.
        return _json_safe(asdict(value))  # Convert dataclass fields through the same finite-value gate.
    if isinstance(value, Mapping):  # Normalize mapping keys and values recursively.
        return {str(key): _json_safe(item) for key, item in value.items()}  # Preserve all evidence under string keys.
    if isinstance(value, (list, tuple)):  # Normalize sequences recursively.
        return [_json_safe(item) for item in value]  # Preserve sequence order exactly.
    if isinstance(value, Path):  # Convert paths to portable JSON strings.
        return str(value)  # Preserve the caller-visible path without filesystem mutation.
    if isinstance(value, bool) or value is None or isinstance(value, str):  # Preserve primitive non-numeric JSON values.
        return value  # Return the already safe primitive unchanged.
    if isinstance(value, int):  # Preserve exact integer counters and identifiers.
        return int(value)  # Normalize third-party integer subclasses.
    if isinstance(value, float):  # Validate every floating-point audit value.
        return float(value) if math.isfinite(value) else None  # Encode unavailable non-finite values as explicit JSON null.
    if hasattr(value, "tolist") and not hasattr(value, "item"):  # Normalize non-scalar NumPy arrays without calling scalar-only item().
        return _json_safe(value.tolist())  # Recurse through the finite list representation.
    if hasattr(value, "item"):  # Normalize NumPy scalar values without importing its concrete types.
        try:  # Distinguish scalar wrappers from multi-element arrays that also expose item().
            return _json_safe(value.item())  # Reapply finite-number checks to the Python scalar.
        except ValueError:  # Fall back only when item() rejects a non-scalar array.
            return _json_safe(value.tolist())  # Preserve the complete array as nested finite JSON values.
    return str(value)  # Preserve otherwise unknown diagnostics as transparent strings.

def _write_json(path: Path, payload: Any) -> None:  # Persist one complete JSON artifact atomically.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the exact artifact directory.
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")  # Isolate an incomplete write from readers and sibling shards.
    temporary.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Serialize stable strict JSON with one terminal newline.
    os.replace(temporary, path)  # Atomically publish the completed artifact on the same filesystem.

def _write_json_exclusive(path: Path, payload: Any) -> None:  # Publish an irreversible marker exactly once without overwrite races.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the marker's exact parent directory.
    encoded = (json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")  # Build complete strict JSON bytes before claiming the path.
    try:  # Convert an existing marker into a protocol refusal rather than overwriting it.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)  # Claim the exact marker path atomically.
    except FileExistsError as exception:  # Detect a concurrent or prior campaign start.
        raise FrozenInputError(f"irreversible marker already exists: {path}") from exception  # Preserve the one-shot boundary.
    try:  # Guarantee descriptor cleanup after the complete bounded write.
        os.write(descriptor, encoded)  # Write the fully serialized marker bytes through the exclusive descriptor.
    finally:  # Close the descriptor on success or an operating-system write failure.
        os.close(descriptor)  # Release the marker file resource deterministically.

def _append_jsonl(path: Path, payload: Any) -> None:  # Append one retained ledger entry safely across controlled processes.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create the exact ledger directory if absent.
    encoded = (json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")  # Build one indivisible newline-delimited JSON record.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)  # Open the ledger in kernel append mode.
    try:  # Guarantee descriptor cleanup after the bounded write.
        os.write(descriptor, encoded)  # Append the complete small ledger record without rewriting siblings.
    finally:  # Close the descriptor on both success and failure.
        os.close(descriptor)  # Release the operating-system resource deterministically.

def load_frozen_config(path: Path) -> dict[str, Any]:  # Load and minimally validate the protocol freeze document.
    payload = _read_json(path)  # Parse the caller-selected frozen configuration.
    if not isinstance(payload, dict):  # Require a named top-level contract.
        raise FrozenInputError("frozen_config.json must contain a JSON object")  # Reject an ambiguous scalar or list configuration.
    gradation_values = [payload.get("common_gradation")]  # Read the canonical freeze-owned finite-element size-field contract first.
    if "gradation" in payload:  # Inspect the deprecated top-level alias only to reject conflicts during transition.
        gradation_values.append(payload["gradation"])  # Retain a compatibility declaration under the same exact-value gate.
    for block_name in ("supervised_config", "supervised", "local_prediction", "dorfler"):  # Inspect every method block that may redundantly declare gradation.
        block = payload.get(block_name)  # Read this optional method-specific configuration object.
        if isinstance(block, Mapping) and "gradation" in block:  # Validate only explicit aliases without inventing defaults.
            gradation_values.append(block["gradation"])  # Retain the method-specific declaration for conflict checks.
    if gradation_values[0] is None or any(float(value) != 1.0 for value in gradation_values if value is not None):  # Require the frozen PR-40 V0 common gradation everywhere it is declared.
        raise FrozenInputError("frozen_config common_gradation and every declared method gradation must equal 1.0")  # Refuse method-specific smoothing drift before any solve.
    if not isinstance(payload.get("allow_unqualified_references", False), bool):  # Require the threshold-waiver decision to be an exact frozen boolean.
        raise FrozenInputError("frozen_config allow_unqualified_references must be boolean")  # Reject truthy strings or numbers that could hide waiver intent.
    return payload  # Return the mapping for context-sensitive blind or development validation.

def _truthy_test_not_run(value: Any) -> bool:  # Normalize the explicit pre-blind-test declaration.
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")  # Accept the documented boolean or literal spelling only.

def _configured_manifest_digest(config: Mapping[str, Any]) -> str | None:  # Resolve supported explicit manifest-digest locations.
    direct = config.get("case_manifest_sha256") or config.get("manifest_sha256")  # Prefer the concise top-level freeze fields.
    protocol = config.get("protocol")  # Inspect an optional nested protocol block used by training scripts.
    nested = protocol.get("case_manifest_sha256") if isinstance(protocol, Mapping) else None  # Read the nested digest without accepting arbitrary structure.
    return str(direct or nested) if direct or nested else None  # Return one normalized digest or an explicit absence.

def _artifact_entries(config: Mapping[str, Any]) -> list[dict[str, Any]]:  # Collect every declared frozen model artifact for verification.
    source = config.get("model_artifacts", config.get("artifacts", []))  # Accept the two explicit freeze-document container names.
    if isinstance(source, Mapping):  # Expand named artifact mappings while retaining their names.
        entries: list[dict[str, Any]] = []  # Allocate normalized artifact records.
        for name, raw in source.items():  # Traverse every declared method artifact.
            values = raw if isinstance(raw, list) else [raw]  # Normalize singleton and three-seed declarations.
            for value in values:  # Expand every seed-specific artifact independently.
                item = dict(value) if isinstance(value, Mapping) else {"path": value}  # Normalize string paths into explicit records.
                item.setdefault("name", str(name))  # Preserve the method key when the record omits a name.
                entries.append(item)  # Retain the artifact for exact-byte verification.
        return entries  # Return the normalized mapping-derived entries.
    if isinstance(source, list):  # Accept a transparent flat list of artifact records.
        return [dict(value) for value in source if isinstance(value, Mapping)]  # Reject unnamed scalar entries from blind validation later.
    return []  # Treat any other structure as missing frozen artifacts.

def _resolve_under(root: Path, value: str | Path) -> Path:  # Resolve frozen relative paths against the campaign root only.
    path = Path(value)  # Normalize the configured path value.
    return path if path.is_absolute() else root / path  # Preserve explicit absolute paths or bind relative paths to the campaign.

def validate_blind_freeze(request: BenchmarkRequest, manifest_bytes_digest: str, config: Mapping[str, Any]) -> dict[str, Any]:  # Refuse blind execution until every immutable input is present and hash-correct.
    from .four_way_freeze import EXPEDITED_REFERENCE_LEVELS, FreezeError, verify_freeze  # Import the canonical seal verifier and its authenticated amendment-depth contract.
    errors: list[str] = []  # Collect all preflight failures before any solve can begin.
    canonical_config = (request.root / "protocol" / "frozen_config.json").resolve()  # Resolve the sole committed runtime configuration allowed for blind execution.
    if request.frozen_config_path.resolve() != canonical_config:  # Forbid an alternate unsealed config path even when its visible fields look equivalent.
        errors.append(f"blind frozen_config must be the canonical sealed file {canonical_config}")  # Report the exact protected input boundary.
    if config.get("protocol_id") != PROTOCOL_ID:  # Require the exact preregistered protocol identity.
        errors.append(f"protocol_id must equal {PROTOCOL_ID}")  # Report the stale or unrelated freeze document.
    if not _truthy_test_not_run(config.get("TEST_NOT_RUN", config.get("test_not_run"))):  # Require the explicit pretest declaration.
        errors.append("frozen_config must declare TEST_NOT_RUN=true")  # Refuse a post-hoc or ambiguous configuration.
    configured_digest = _configured_manifest_digest(config)  # Read the manifest identity frozen by training.
    if configured_digest != manifest_bytes_digest:  # Bind execution to the exact case bytes, not just semantic fields.
        errors.append("frozen manifest SHA-256 is absent or does not match case_manifest.json")  # Report the immutable-input mismatch.
    if tuple(config.get("budgets", ())) != BUDGETS:  # Require the exact registered resource grid and order.
        errors.append(f"budgets must be {list(BUDGETS)}")  # Report any post-hoc budget change.
    if tuple(config.get("solve_limits", config.get("k_values", ()))) != SOLVE_LIMITS:  # Require every true-prefix solve limit.
        errors.append(f"solve_limits must be {list(SOLVE_LIMITS)}")  # Report any post-hoc solve-count change.
    if int(config.get("max_solves", -1)) != MAX_SOLVES:  # Require one independent six-solve trajectory per budget.
        errors.append(f"max_solves must equal {MAX_SOLVES}")  # Report a shorter or longer deployment trajectory.
    if float(config.get("theta", float("nan"))) != 0.5:  # Require the exact common Dörfler bulk parameter.
        errors.append("theta must equal 0.5")  # Report an inconsistent safety baseline.
    if bool(request.allow_unqualified_references) is not bool(config.get("allow_unqualified_references", False)):  # Require the runtime acknowledgement to match the sealed reference execution policy exactly.
        errors.append("--allow-unqualified-references must exactly match frozen_config allow_unqualified_references")  # Prevent either post-freeze relaxation or accidental strict/amended cache mismatch.
    if request.allow_unqualified_references and not isinstance(config.get("reference_execution_amendment"), Mapping):  # Require an authenticated human amendment pointer beside the exceptional frozen boolean.
        errors.append("allow_unqualified_references requires frozen reference_execution_amendment evidence")  # Prevent a boolean-only threshold waiver.
    expedited_value = config.get("expedited_reference_levels")  # Read the hash-protected amendment depth without embedding a second runtime constant.
    if request.allow_unqualified_references and (type(expedited_value) is not int or int(expedited_value) != EXPEDITED_REFERENCE_LEVELS):  # Match the frozen module's fixed amendment depth without a duplicate magic value.
        errors.append("allow_unqualified_references requires the canonical frozen expedited_reference_levels")  # Prevent runtime depth choice or an invalid comparison prefix.
    if not request.allow_unqualified_references and expedited_value is not None:  # Require strict mode to leave amendment depth inactive.
        errors.append("strict reference mode requires expedited_reference_levels=null")  # Prevent an undisclosed shortened schedule under the default path.
    entries = _artifact_entries(config)  # Collect all model snapshot declarations.
    names = [str(entry.get("name", "")) for entry in entries]  # Collect descriptive artifact names for compatibility diagnostics.
    methods = [str(entry.get("method", "")) for entry in entries]  # Collect canonical freeze-owned method identities.
    if not any(method == "world_model" or (not method and "world" in name) for method, name in zip(methods, names, strict=True)):  # Require one frozen world-model transition library.
        errors.append("model_artifacts must include a world-model snapshot")  # Prevent a fresh or test-learned initial model.
    if not any(method == "supervised" or (not method and "supervised" in name) for method, name in zip(methods, names, strict=True)):  # Require one validation-selected supervised checkpoint.
        errors.append("model_artifacts must include a supervised checkpoint")  # Prevent an untrained supervised comparator.
    rl_seeds_value = config.get("rl_seeds", (20260901, 20260902, 20260903))  # Read the protocol's three actual initialization seeds.
    if not isinstance(rl_seeds_value, Sequence) or isinstance(rl_seeds_value, (str, bytes)) or len(rl_seeds_value) != 3:  # Require exactly three independent RL seeds.
        errors.append("rl_seeds must contain exactly three values")  # Prevent incomplete or best-seed deployment.
        rl_seeds_value = ()  # Avoid secondary matching errors for a malformed list.
    for index, seed_value in enumerate(rl_seeds_value):  # Require each actual frozen policy seed while retaining public index labels.
        seed = int(seed_value)  # Normalize JSON integers for deterministic matching.
        if not any((method == "rl" or (not method and "rl" in name)) and (str(entry.get("seed", "")) == str(seed) or f"seed{seed}" in name or f"seed{index}" in name) for name, method, entry in zip(names, methods, entries, strict=True)):  # Match canonical family plus the actual frozen seed rather than the public index alone.
            errors.append(f"model_artifacts must include RL policy index {index} with seed {seed}")  # Prevent seed substitution or omission.
    verified_artifacts: list[dict[str, Any]] = []  # Collect successful exact-byte identities for the preflight report.
    for entry in entries:  # Verify every declared artifact, including unused training evidence snapshots.
        if "path" not in entry or "sha256" not in entry:  # Require both location and frozen exact-byte digest.
            errors.append(f"artifact {entry.get('name', '<unnamed>')} must declare path and sha256")  # Report incomplete provenance.
            continue  # Avoid a secondary path-resolution error for this entry.
        artifact_path = _resolve_under(request.root, str(entry["path"]))  # Resolve the frozen artifact inside or outside the campaign explicitly.
        if not artifact_path.is_file():  # Require the exact model bytes to exist before blind execution.
            errors.append(f"frozen artifact is missing: {artifact_path}")  # Report the missing policy snapshot.
            continue  # Avoid hashing a missing path.
        digest = _sha256_file(artifact_path)  # Recompute the exact model identity from disk.
        if digest != str(entry["sha256"]):  # Reject modified or partly copied model artifacts.
            errors.append(f"frozen artifact SHA-256 mismatch: {artifact_path}")  # Report the integrity violation.
        verified_artifacts.append({"name": entry.get("name"), "seed": entry.get("seed"), "path": str(artifact_path), "sha256": digest})  # Preserve successful preflight evidence.
    if errors:  # Refuse the entire invocation before creating or claiming any test job.
        raise FrozenInputError("blind-test freeze validation failed:\n- " + "\n- ".join(errors))  # Return all actionable protocol blockers together.
    try:  # Convert the canonical freeze module's integrity category without weakening it into a numerical failure.
        canonical_evidence = verify_freeze(request.root, Path(__file__).resolve().parents[2], require_committed=True, allow_postfreeze_test_references=True)  # Verify committed code and seal while allowing only authenticated schedule-bounded test-reference ledger/B/log/input evidence.
    except FreezeError as exception:  # Preserve a concise public preflight exception type for the unified CLI.
        raise FrozenInputError(f"canonical committed freeze verification failed: {exception}") from exception  # Stop the campaign before reference or test artifact access.
    if canonical_evidence.get("manifest_sha256") != manifest_bytes_digest:  # Cross-check the plan's direct manifest bytes against the authenticated freeze index.
        raise FrozenInputError("canonical freeze manifest digest differs from the requested manifest")  # Prevent a sealed config with a substituted manifest path.
    return {**canonical_evidence, "runtime_model_artifacts": verified_artifacts}  # Return the canonical complete seal plus direct runtime-model rehash evidence.

def select_manifest_cases(manifest: Mapping[str, Any], split: str, requested_ids: Sequence[str] = ()) -> list[dict[str, Any]]:  # Select cases only from the validated manifest and sort by case_id.
    cases = [dict(case) for case in manifest["cases"] if case.get("split") == split]  # Filter only by the manifest-owned split label.
    available = {str(case["case_id"]): case for case in cases}  # Build an exact case-id lookup inside the selected split.
    if requested_ids:  # Apply a controlled shard restriction when explicitly requested.
        duplicates = sorted({case_id for case_id in requested_ids if requested_ids.count(case_id) > 1})  # Detect repeated shard identifiers.
        missing = sorted(set(requested_ids) - set(available))  # Detect unknown or cross-split case identifiers.
        if duplicates or missing:  # Reject ambiguous or invalid shard definitions.
            raise FrozenInputError(f"invalid --case-id selection; duplicates={duplicates}, missing_or_wrong_split={missing}")  # Preserve the manifest boundary.
        cases = [available[case_id] for case_id in requested_ids]  # Select exactly the requested manifest records.
    return sorted(cases, key=lambda case: str(case["case_id"]))  # Enforce the blind protocol's lexicographic case order.

def normalize_methods(methods: Sequence[str]) -> tuple[str, ...]:  # Validate and deterministically order requested execution methods.
    expanded: list[str] = []  # Collect explicit frozen method labels.
    for method in methods:  # Expand convenient aggregate labels without changing policy semantics.
        values = RL_METHODS if method in ("rl", "rl_median") else (method,)  # Expand RL to all three required frozen seeds.
        for value in values:  # Add each explicit method at most once.
            if value not in ALL_METHODS:  # Reject unregistered comparators and aliases.
                raise FrozenInputError(f"unknown method {value!r}; choose from {ALL_METHODS} or rl")  # Preserve the four-way protocol vocabulary.
            if value not in expanded:  # Deduplicate repeated CLI values while preserving frozen order later.
                expanded.append(value)  # Retain the validated method label.
    return tuple(method for method in ALL_METHODS if method in expanded)  # Return canonical execution order independent of CLI ordering.

def build_execution_jobs(request: BenchmarkRequest, manifest: Mapping[str, Any]) -> list[ExecutionJob]:  # Construct the exact independent trajectory grid without solving.
    cases = select_manifest_cases(manifest, request.split, request.case_ids)  # Select and sort cases through the manifest boundary.
    methods = normalize_methods(request.methods)  # Expand RL and freeze method order.
    budgets = tuple(sorted(set(int(value) for value in request.budgets)))  # Normalize repeated budget flags deterministically.
    if not budgets or any(value not in BUDGETS for value in budgets):  # Restrict execution to preregistered resource caps only.
        raise FrozenInputError(f"budgets must be a nonempty subset of {BUDGETS}")  # Reject unregistered post-hoc operating points.
    jobs: list[ExecutionJob] = []  # Allocate the complete selected Cartesian product.
    for case in cases:  # Preserve sorted blind-case order at the outer level.
        for budget in budgets:  # Run each registered budget independently from a fresh method state.
            for method in methods:  # Preserve canonical method order inside each case-budget block.
                output_root = request.root / "test" if request.split == "test" else request.root / "development" / request.split  # Isolate controlled development shards from the one-shot blind evidence tree.
                output = output_root / str(case["case_id"]) / str(budget) / method  # Materialize the mandated raw-evidence directory layout.
                jobs.append(ExecutionJob(str(case["case_id"]), request.split, str(case["geometry_hash"]), budget, method, output))  # Store the immutable job contract.
    return jobs  # Return a solve-free deterministic execution plan.

def build_plan(request: BenchmarkRequest) -> tuple[dict[str, Any], dict[str, Any], list[ExecutionJob]]:  # Validate inputs and construct a complete dry-run or execution plan.
    manifest_bytes = request.manifest_path.read_bytes()  # Read exact bytes before validated JSON decoding.
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()  # Bind the plan to the precise persisted manifest artifact.
    manifest = load_case_manifest(request.manifest_path, verify_checksum=True)  # Enforce schema, LHS, geometry, IDs, and sidecar integrity.
    config = load_frozen_config(request.frozen_config_path) if request.frozen_config_path.is_file() else {}  # Load the freeze document when present for either split.
    if request.split == "test":  # Apply full immutable-input validation before any blind-test job exists.
        if request.case_ids:  # Forbid even a sealed-looking case shard at the irreversible blind boundary.
            raise FrozenInputError("test split forbids --case-id; all 16 sorted blind cases must run once")  # Preserve the one-shot campaign as the only scientific test unit.
        if normalize_methods(request.methods) != ALL_METHODS:  # Require every learned and safety trajectory in one invocation.
            raise FrozenInputError("test split forbids method subsets; all frozen methods must run once")  # Prevent result-dependent comparator omission.
        if tuple(sorted(set(int(value) for value in request.budgets))) != BUDGETS:  # Require every registered active-equation cap together.
            raise FrozenInputError("test split forbids budget subsets; budgets 30000, 60000, and 120000 must run once")  # Prevent post-hoc operating-point selection.
        if request.resume:  # Treat interruption or partial evidence as evidence from this disclosed round.
            raise FrozenInputError("test split forbids --resume; an interrupted blind campaign cannot be silently continued")  # Preserve the disclosure boundary.
        freeze_evidence = validate_blind_freeze(request, manifest_digest, config)  # Refuse all partial blind execution on a stale freeze.
    elif not request.dry_run and not request.development_run:  # Require an explicit opt-in for non-blind solver smoke runs.
        raise FrozenInputError("non-test execution requires --allow-development-run; --dry-run remains solve-free")  # Prevent accidental training/validation solves.
    else:  # Record the weaker development preflight transparently.
        freeze_evidence = {"protocol_id": manifest.get("protocol_id"), "manifest_sha256": manifest_digest, "verified_artifacts": []}  # Preserve exact manifest provenance without claiming a blind freeze.
    jobs = build_execution_jobs(request, manifest)  # Construct the independent case-budget-method grid after preflight.
    selected_cases = sorted({job.case_id for job in jobs})  # Summarize the exact manifest case shard.
    if request.split == "test" and (len(selected_cases) != 16 or len(jobs) != 16 * len(BUDGETS) * len(ALL_METHODS)):  # Recompute the exact one-shot blind Cartesian product independently.
        raise FrozenInputError("test split must contain exactly 16 cases x 3 budgets x all methods")  # Refuse a malformed or truncated blind manifest.
    plan = {"schema": PLAN_SCHEMA, "protocol_id": PROTOCOL_ID, "created_utc": _utc_now(), "dry_run": request.dry_run, "split": request.split, "case_ids": selected_cases, "case_count": len(selected_cases), "methods": list(normalize_methods(request.methods)), "budgets": sorted({job.budget for job in jobs}), "max_solves_per_job": MAX_SOLVES, "job_count": len(jobs), "estimated_max_real_solves": len(jobs) * MAX_SOLVES, "manifest_path": str(request.manifest_path), "frozen_config_path": str(request.frozen_config_path), "allow_unqualified_references": bool(request.allow_unqualified_references), "expedited_reference_levels": config.get("expedited_reference_levels") if request.allow_unqualified_references else None, "reference_execution_amendment": config.get("reference_execution_amendment") if request.allow_unqualified_references else None, "freeze_evidence": freeze_evidence, "ordered_jobs": [{"case_id": job.case_id, "budget": job.budget, "method": job.method, "output_dir": str(job.output_dir)} for job in jobs]}  # Assemble a complete auditable solve-free plan using the authenticated frozen amendment depth and pointer.
    return plan, config, jobs  # Return validated configuration and jobs for optional execution.

class ReceiptFemRunner(FemRunner):  # Extend the common solver runner with immutable per-solve mesh receipts.
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # Initialize common accounting plus receipt storage.
        super().__init__(*args, **kwargs)  # Preserve the repository solver implementation unchanged.
        self.mesh_receipts: list[dict[str, Any]] = []  # Collect every counted real-solve mesh identity in execution order.
        self.last_post: Any | None = None  # Retain only the latest successful physical state for solve-free final diagnostics.
        self.final_partition: Any | None = None  # Retain an optional shared semantic assignment for the final mesh receipt.
        self.last_attempted_mesh: Any | None = None  # Retain the exact most recently submitted mesh even when its native solve fails.
        self.last_attempted_partition: Any | None = None  # Snapshot any available semantic partition alongside the attempted mesh.
    def solve_mesh(self, mesh: Any, *, method: str, stage: str, extra: dict | None = None) -> tuple[Any, SolveRecord]:  # Intercept only counted method solves.
        self.last_attempted_mesh = mesh  # Save the submitted mesh before any hashing or native execution can raise.
        self.last_attempted_partition = self.final_partition  # Bind the currently available partition to this exact attempted mesh.
        attempt_index = int(getattr(self, "_counter", len(self.records))) + 1  # Identify the native attempt before FemRunner advances its honest counter.
        mesh_sha = _full_mesh_sha256(mesh)  # Hash full little-endian coordinates and connectivity rather than the legacy short display digest.
        receipt = {"attempt_index": attempt_index, "solve_index": attempt_index, "method": str(method), "stage": str(stage), "mesh_sha256": mesh_sha, "n_nodes": int(mesh.n_nodes), "n_elements": int(mesh.n_cells), "success": False, "calculix_returncode": None, "calculix_wall_s": None, "calculix_log_path": None}  # Initialize a complete attempted-mesh receipt before native execution.
        try:  # Publish the attempted mesh on both success and the narrow typed numerical failure path.
            post, record = super().solve_mesh(mesh, method=method, stage=stage, extra=extra)  # Execute exactly one real CalculiX solve through the canonical path.
        except Exception as exception:  # Attach available typed native diagnostics before preserving the original exception.
            receipt["calculix_returncode"] = getattr(exception, "returncode", None)  # Retain an explicit native code when the typed backend supplies one.
            receipt["calculix_wall_s"] = getattr(exception, "wall_s", None)  # Retain native time up to failure when available.
            receipt["calculix_log_path"] = getattr(exception, "log_path", None)  # Retain the exact already-written native log when available.
            self.mesh_receipts.append(receipt)  # Preserve the attempted full mesh even though no SolveRecord exists.
            raise  # Re-raise unchanged so campaign-level classification remains strict.
        record.extra.setdefault("mesh_sha256", mesh_sha)  # Attach the same full identity to the raw SolveRecord for independent cross-checking.
        receipt.update({"solve_index": int(record.solve_index), "success": True, "calculix_returncode": 0, "calculix_wall_s": float(record.wall_s), "calculix_log_path": _expected_log_path(self, method, int(record.solve_index), stage), "n_equations": int(record.n_equations)})  # Complete successful resource and native-return evidence explicitly.
        self.mesh_receipts.append(receipt)  # Preserve resources and exact mesh identity for this successful real solve.
        self.last_post = post  # Retain only the latest post-state for one final solve-free ZZ recomputation.
        return post, record  # Return the unmodified physical post-state and enriched solve record.

def _full_mesh_sha256(mesh: Any) -> str:  # Hash every coordinate and connectivity byte with stable dtype and shape boundaries.
    import numpy as np  # Import numerical canonicalization only for receipt generation.
    digest = hashlib.sha256()  # Allocate a complete collision-resistant mesh identity.
    nodes = np.ascontiguousarray(np.asarray(mesh.nodes, dtype="<f8"))  # Normalize coordinates to portable little-endian float64 bytes.
    cells = np.ascontiguousarray(np.asarray(mesh.cells, dtype="<i8"))  # Normalize connectivity to portable little-endian int64 bytes.
    digest.update(json.dumps({"nodes_shape": list(nodes.shape), "cells_shape": list(cells.shape)}, sort_keys=True, separators=(",", ":")).encode("ascii"))  # Bind array shapes before raw concatenation.
    digest.update(nodes.tobytes(order="C"))  # Incorporate every coordinate byte in row-major order.
    digest.update(cells.tobytes(order="C"))  # Incorporate every connectivity byte in row-major order.
    return digest.hexdigest()  # Return the full 64-character mesh SHA-256.

def _expected_log_path(runner: ReceiptFemRunner, method: str, solve_index: int, stage: str) -> str:  # Reconstruct FemRunner's deterministic native log location.
    jobname = f"{method}_{solve_index:03d}_{stage}"[:60].replace("/", "_")  # Match the canonical bounded CalculiX job-name transform exactly.
    return str(runner.workdir / "solves" / jobname / "model.log")  # Return the combined stdout/stderr log written by run_ccx.

def _reference_payload(root: Path, case: Mapping[str, Any], *, allow_unqualified: bool = False, expedited_levels: int | None = None, amendment_record: Mapping[str, Any] | None = None) -> tuple[Reference, dict[str, Any]]:  # Load the canonical operational Reference B without exposing it to an online method runner.
    from .four_way_references import LEDGER_FILENAME, REFERENCE_B_FILENAME, load_reference_b, reference_case_dir, verify_reference_cache  # Reuse the only authenticated A/B convergence implementation.
    case_id = str(case["case_id"])  # Read the manifest-owned case identity.
    problem = problem_from_case(case)  # Reconstruct the validation problem solely from the authenticated manifest record.
    verification = verify_reference_cache(root / "references", case_id=case_id, problem=problem, regenerate_meshes=False, allow_unqualified=allow_unqualified, expedited_levels=expedited_levels)  # Authenticate signatures and reproduce the exact strict or authorized two-level execution amendment.
    reference = load_reference_b(root / "references", case_id=case_id, problem=problem, runner=None, verify=False, allow_unqualified=allow_unqualified, expedited_levels=expedited_levels)  # Load the verified denominator only under the identical qualification and schedule intent.
    case_root = reference_case_dir(root / "references", case_id)  # Resolve the canonical per-case cache directory.
    ledger_path = case_root / LEDGER_FILENAME  # Resolve the authoritative convergence ledger.
    reference_path = case_root / REFERENCE_B_FILENAME  # Resolve the authenticated compact denominator.
    receipt = {"ledger_path": str(ledger_path), "ledger_sha256": _sha256_file(ledger_path), "reference_b_path": str(reference_path), "reference_b_sha256": _sha256_file(reference_path), "verification": verification, "status": verification.get("status"), "qualification": verification.get("qualification") is True, "authorization": verification.get("authorization"), "execution_amendment": verification.get("execution_amendment"), "reference_execution_amendment_sha256": amendment_record.get("sha256") if isinstance(amendment_record, Mapping) else None, "agreement_passed": bool(verification.get("qualification", verification.get("passed", False))), "allow_unqualified": bool(allow_unqualified), "expedited_levels": expedited_levels}  # Preserve exact bytes, original threshold outcome, authorization, protected amendment hash, and effective schedule.
    return reference, receipt  # Return the shared final denominator and complete provenance.

def _preflight_references(root: Path, config: Mapping[str, Any], cases: Iterable[Mapping[str, Any]], *, allow_unqualified: bool = False) -> dict[str, dict[str, Any]]:  # Validate every selected case reference before the first blind solve.
    if bool(allow_unqualified) is not bool(config.get("allow_unqualified_references", False)):  # Reassert the frozen/runtime equality immediately at reference access.
        raise FrozenInputError("reference preflight runtime waiver differs from frozen_config")  # Stop before any denominator read under mismatched intent.
    expedited_levels = int(config["expedited_reference_levels"]) if allow_unqualified else None  # Apply only the authenticated frozen operational amendment depth.
    evidence: dict[str, dict[str, Any]] = {}  # Collect exact per-case reference provenance.
    ordered = sorted((dict(case) for case in cases), key=lambda value: str(value["case_id"]))  # Preserve blind case order while retaining full manifest evidence.
    for case in ordered:  # Verify each selected reference cache exactly once before execution.
        case_id = str(case["case_id"])  # Read the immutable identity for the report key.
        _reference, receipt = _reference_payload(root, case, allow_unqualified=allow_unqualified, expedited_levels=expedited_levels, amendment_record=config.get("reference_execution_amendment") if allow_unqualified else None)  # Authenticate the sealed ladder and protected amendment identity.
        evidence[case_id] = receipt  # Store only JSON-safe frozen provenance in the global preflight.
    return evidence  # Return complete selected-shard reference readiness.

def _matching_artifact(config: Mapping[str, Any], kind: str, seed: int | None = None) -> dict[str, Any]:  # Resolve one exact frozen model snapshot by method and optional seed.
    entries = _artifact_entries(config)  # Normalize all declared model records.
    candidates: list[dict[str, Any]] = []  # Collect semantic matches without accepting best-test selection.
    for entry in entries:  # Inspect every frozen artifact declaration.
        name = str(entry.get("name", "")).lower()  # Normalize the descriptive method key.
        declared_method = str(entry.get("method", "")).lower()  # Prefer canonical freeze metadata over descriptive filenames.
        kind_match = declared_method == kind or ((not declared_method) and ((kind == "world_model" and "world" in name) or (kind == "supervised" and "supervised" in name) or (kind == "rl" and "rl" in name)))  # Match canonical method metadata or a legacy unambiguous name only when metadata is absent.
        if not kind_match:  # Ignore unrelated training and environment artifacts.
            continue  # Continue the deterministic scan.
        if seed is not None:  # Require the exact frozen seed for RL deployment.
            declared = entry.get("seed")  # Read explicit numeric seed metadata when present.
            if not ((declared is not None and int(declared) == seed) or f"seed{seed}" in name or f"seed_{seed}" in name):  # Accept only unambiguous seed identities.
                continue  # Reject other RL seeds from this job.
        candidates.append(entry)  # Retain this exact policy-family match.
    if kind == "supervised" and len(candidates) > 1:  # Resolve three trained networks through frozen validation selection only.
        selected = config.get("selected_supervised_seed")  # Read the preregistered validation-selected network seed.
        if selected is None:  # Refuse implicit first-seed or test-informed selection.
            raise FrozenInputError("multiple supervised checkpoints require selected_supervised_seed in frozen_config")  # Preserve training/validation/test separation.
        candidates = [entry for entry in candidates if str(entry.get("seed")) == str(selected) or f"seed{selected}" in str(entry.get("name", ""))]  # Keep only the validation-selected network.
    if len(candidates) != 1:  # Require one and only one frozen deployment artifact.
        raise FrozenInputError(f"expected exactly one frozen {kind} artifact for seed={seed}, found {len(candidates)}")  # Report an incomplete or ambiguous freeze.
    return candidates[0]  # Return the unique validated declaration.

def _artifact_path(root: Path, config: Mapping[str, Any], kind: str, seed: int | None = None) -> Path:  # Resolve and reverify one method's exact model bytes at job start.
    entry = _matching_artifact(config, kind, seed)  # Select the unambiguous frozen declaration.
    path = _resolve_under(root, str(entry["path"]))  # Resolve the model path against the campaign root.
    observed = _sha256_file(path)  # Recompute exact bytes immediately before policy loading.
    if observed != str(entry["sha256"]):  # Catch any mutation after global preflight.
        raise FrozenInputError(f"frozen model changed after preflight: {path}")  # Stop the affected job without executing a stale policy.
    return path  # Return the exact verified snapshot location.

def _partition_root(root: Path, config: Mapping[str, Any]) -> Path:  # Resolve the one-spec-per-case frozen semantic partition directory.
    value = config.get("partition_root", config.get("partition_specs_root", config.get("partition_spec_root", "protocol/partitions")))  # Read the canonical freeze field before compatibility aliases.
    return _resolve_under(root, str(value))  # Bind relative registry paths to the campaign root.

def _load_shared_partitions(root: Path, config: Mapping[str, Any], case: Mapping[str, Any], problem: Any) -> tuple[Any, Any, dict[str, Any]]:  # Load one frozen spec and expose its new-WM and RL adapters.
    from .partition_spec import PartitionSpecRegistry  # Import the canonical shared partition registry only for real method execution.
    registry_root = _partition_root(root, config)  # Resolve the immutable one-file-per-case registry.
    expected_value = config.get("partition_spec_sha256", config.get("partition_spec_hashes", {}))  # Read frozen semantic-body digests keyed by case ID.
    expected = dict(expected_value) if isinstance(expected_value, Mapping) else {}  # Normalize only an explicit mapping.
    registry = PartitionSpecRegistry(registry_root, expected_sha256=expected)  # Construct a fresh registry that verifies the freeze-record digest.
    case_id = str(case["case_id"])  # Read the content-bound case identifier.
    geometry_hash = str(case["geometry_hash"])  # Read the exact geometry identity checked by the spec loader.
    shared = registry.partitioner_for(case_id, problem, geometry_hash)  # Load the one case-bound provider through complete schema, hash, geometry, and probe validation.
    world_partition = shared.partition_for_world(problem)  # Return the direct new-stack CachedVisionPartition-compatible object.
    rl_partition = shared  # Pass the same provider so RegionRefineEnv verifies the regenerated common probe before creating its RL adapter.
    spec_path = registry_root / case_id / "partition_spec.json"  # Resolve the unique protocol-required case spec path.
    receipt = {"path": str(spec_path), "sha256": _sha256_file(spec_path), "case_id": case_id, "geometry_hash": geometry_hash}  # Preserve exact shared partition provenance for both method families.
    return world_partition, rl_partition, receipt  # Return both adapters and their one common source receipt.

def _copy_solver_logs(runner: ReceiptFemRunner, destination: Path) -> list[str]:  # Copy retained CalculiX logs into the mandated explicit log directory.
    destination.mkdir(parents=True, exist_ok=True)  # Create the exact job-level solver-log directory.
    copied: list[str] = []  # Collect relative copied paths for status provenance.
    for source in sorted(runner.workdir.glob("solves/**/*.log")):  # Traverse only retained text logs from this isolated job.
        relative = source.relative_to(runner.workdir / "solves")  # Preserve solve and stage identity below solver_logs.
        target = destination / relative  # Map the source hierarchy into the protocol directory.
        target.parent.mkdir(parents=True, exist_ok=True)  # Create only the required nested solve folder.
        shutil.copy2(source, target)  # Preserve log bytes and filesystem timestamps for debugging.
        copied.append(str(target.relative_to(destination.parent)))  # Record the job-relative evidence location.
    return copied  # Return all copied log paths in deterministic source order.

def _dataclass_config(config_type: Any, values: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> Any:  # Construct a repository dataclass from only declared frozen fields.
    allowed = set(getattr(config_type, "__dataclass_fields__", {}))  # Read the exact constructor field vocabulary.
    selected = {key: value for key, value in values.items() if key in allowed}  # Ignore metadata keys while retaining all scientific settings.
    selected.update({key: value for key, value in (overrides or {}).items() if key in allowed})  # Apply job-owned budget and solve-count values last.
    return config_type(**selected)  # Delegate type and invariant checks to the repository dataclass.

def _nested_config(config: Mapping[str, Any], *names: str) -> dict[str, Any]:  # Resolve the first named configuration block as a plain mapping.
    for name in names:  # Inspect stable and compatibility block names in declared order.
        value = config.get(name)  # Read the candidate frozen configuration block.
        if isinstance(value, Mapping):  # Accept only named JSON objects.
            return dict(value)  # Return an independent shallow copy for safe constructor filtering.
    return {}  # Use repository defaults only when the freeze document omits the optional block.

def _preflight_mesh_equations(mesh: Any, equations_per_element: float, budget: int) -> float:  # Estimate a solve-free candidate's active equations from the current measured conversion.
    estimate = float(mesh.n_cells) * float(equations_per_element)  # Apply only the same-method current real probe conversion.
    return estimate / max(float(budget), 1.0)  # Return a dimensionless cap-utilization estimate.

def _run_local_prediction_dynamic(runner: ReceiptFemRunner, budget: int, max_solves: int = MAX_SOLVES) -> dict[str, Any]:  # Execute strong iterative local prediction with per-round same-method budget calibration.
    import numpy as np  # Import numerical clipping only when the baseline actually runs.
    from ..baselines.local_prediction import predicted_sizes  # Reuse the corrected three-dimensional equidistribution predictor.
    from ..experiment import initial_mesh  # Reuse the common uniform probe generator.
    from ..indicators import zz_indicator  # Reuse the common exact ZZ indicator.
    from ..mesher import generate_mesh  # Reuse the canonical Gmsh remeshing path.
    from ..sizefield import NodalSizeField, element_to_node_sizes  # Reuse nodal interpolation and gradation contracts.
    mesh = initial_mesh(runner.problem)  # Start from the identical global problem.h0 common probe.
    rounds: list[dict[str, Any]] = []  # Collect every dynamic conversion and solve-free preflight decision.
    for solve_index in range(max_solves):  # Execute one probe plus at most five fresh iterative corrections.
        post, record = runner.solve_mesh(mesh, method="local_prediction", stage="common_probe" if solve_index == 0 else f"iteration{solve_index}", extra={"equation_budget": int(budget)})  # Count one real CalculiX solve with explicit resource context.
        eta2 = zz_indicator(runner.problem, post)  # Compute the shared element-wise squared estimator.
        record.extra["sum_eta2"] = float(np.sum(eta2))  # Preserve the realized global estimator mass.
        if solve_index + 1 >= max_solves:  # Stop after the common six-solve trajectory cap.
            record.extra["stop"] = "round_cap"  # Record the exact terminal condition.
            break  # Do not construct an unused seventh mesh.
        if record.n_equations >= budget:  # Stop after reaching or overshooting the active-equation cap.
            record.extra["stop"] = "equation_cap_reached"  # Preserve the measured resource terminal condition.
            break  # Avoid intentionally executing another over-cap solve.
        equations_per_element = float(record.n_equations / max(record.n_elems, 1))  # Calibrate only from this method's latest real solve.
        target_elements = max(float(budget) / max(equations_per_element, 1.0e-12), 1.0)  # Convert the public cap dynamically at the current trajectory state.
        h_element = predicted_sizes(mesh, eta2, n_target=target_elements)  # Predict corrected element sizes with the strong 3-D exponent and bounded coarsening.
        scale = 1.0  # Initialize solve-free resource preflight without changing the predictor.
        candidate = None  # Reserve the next exact Gmsh mesh.
        utilization = float("inf")  # Initialize the estimate above every feasible cap.
        preflight_trials: list[dict[str, Any]] = []  # Preserve every actually generated scale and realized candidate size.
        accepted = False  # Advance only after a generated candidate satisfies the fixed two-percent margin.
        for attempt in range(5):  # Permit a bounded solve-free scale correction for Gmsh count mismatch.
            target = element_to_node_sizes(mesh, np.asarray(h_element, dtype=float) * scale)  # Convert scaled element predictions to the canonical nodal field.
            field = NodalSizeField(mesh, target, gradation=1.0, h_min=runner.problem.h_min, h_max=runner.problem.h0)  # Apply the frozen PR-40 V0 common gradation and physical size bounds.
            candidate = generate_mesh(runner.problem, field)  # Materialize the exact candidate without a CalculiX solve.
            utilization = _preflight_mesh_equations(candidate, equations_per_element, budget)  # Estimate equations from the same-method latest observed conversion.
            preflight_trials.append({"attempt": attempt + 1, "scale": float(scale), "candidate_elements": int(candidate.n_cells), "predicted_equation_utilization": float(utilization)})  # Bind each trial's scale to the mesh it actually generated.
            if utilization <= 0.98:  # Preserve a two-percent preflight safety margin against conversion drift.
                accepted = True  # Mark this exact generated mesh as the next real-solve candidate.
                break  # Accept this exact candidate for the next real solve.
            if attempt + 1 < 5:  # Avoid reporting an updated scale that never generated a mesh on the terminal trial.
                scale *= max((utilization / 0.98) ** (1.0 / 3.0), 1.01)  # Coarsen uniformly by the 3-D resource scaling needed to recover the margin.
        if candidate is None:  # Guard the statically impossible empty preflight loop.
            raise RuntimeError("local-prediction preflight produced no mesh")  # Retain the method failure explicitly.
        rounds.append({"after_solve": int(record.solve_index), "equations_per_element": equations_per_element, "target_elements": target_elements, "preflight_attempts": attempt + 1, "final_scale": float(scale), "predicted_equation_utilization": utilization, "candidate_elements": int(candidate.n_cells), "accepted": accepted, "trials": preflight_trials})  # Preserve every adaptive budget conversion and actually generated preflight result.
        if not accepted:  # Stop rather than knowingly submit a candidate outside the registered resource margin.
            record.extra["stop"] = "preflight_equation_margin_unmet"  # Preserve the deterministic solve-free resource stop reason.
            break  # Hold the latest real feasible solve for all later prefixes without a budget-violating attempt.
        mesh = candidate  # Advance only to the final exact solve-free-preflighted mesh.
    return {"algorithm": "iterative-zz-local-prediction-v2", "rounds": rounds, "max_solves": int(max_solves)}  # Return complete method-specific action evidence.

def _rl_seed_value(config: Mapping[str, Any], index: int) -> int:  # Resolve the actual frozen initialization seed behind an RL policy-index label.
    values = config.get("rl_seeds", (20260901, 20260902, 20260903))  # Read the freeze document or the protocol's fixed three seeds.
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 3:  # Require exactly three independent policies.
        raise FrozenInputError("rl_seeds must contain exactly three integer seeds")  # Prevent missing or best-seed reporting.
    return int(values[index])  # Return the actual seed paired with rl_seed{index}.

def _campaign_root_for_job(job: ExecutionJob) -> Path:  # Recover the campaign root from the split-specific raw output layout.
    return job.output_dir.parents[3] if job.split == "test" else job.output_dir.parents[4]  # Account for the extra development/<split> path component outside blind evidence.

def _execute_method(job: ExecutionJob, case: Mapping[str, Any], config: Mapping[str, Any], runner: ReceiptFemRunner) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:  # Dispatch one fresh frozen policy trajectory and return actions, timing, and partition provenance.
    started = time.perf_counter()  # Start complete online method time before any method-specific model or partition work and without Reference B.
    partition_receipt: dict[str, Any] | None = None  # Initialize optional shared partition provenance.
    if job.method == "world_model_vla":  # Execute the frozen new-stack world-model controller.
        from .four_way_ablations import build_ablation_runtime  # Add behavior-neutral full-controller prediction diagnostics required for later identity reuse.
        from .world.model import ResidualWorldModel  # Import the new frozen action-conditioned transition model.
        from .world.pipeline import WorldVLAConfig, run_world_model_vla  # Import the only authorized new-stack online loop.
        from .world.planner import MultiStepPlanner, PlannerConfig  # Import frozen finite-horizon planning.
        from .world.tool_gateway import MCPToolGateway, ToolConfig  # Import the frozen deterministic action compiler and Dörfler certificate gateway.
        campaign_root = _campaign_root_for_job(job)  # Recover the campaign root under either blind or isolated development layout.
        model_path = _artifact_path(campaign_root, config, "world_model")  # Reverify and locate the frozen transition library.
        problem = runner.problem  # Bind the manifest-reconstructed case.
        partition_started = time.perf_counter()  # Time only frozen semantic-spec loading and verification.
        world_partition, _rl_partitioner, partition_receipt = _load_shared_partitions(campaign_root, config, case, problem)  # Load the exact case spec shared with RL.
        runner.final_partition = world_partition  # Retain the same frozen assignment solely for solve-free final element labels.
        partition_s = time.perf_counter() - partition_started  # Record solve-free semantic cache overhead.
        model = ResidualWorldModel.load(model_path)  # Reload from frozen bytes for every case and budget to prevent cross-test learning.
        planner_values = _nested_config(config, "world_planner", "planner_config")  # Read only preregistered planner hyperparameters.
        planner = MultiStepPlanner(_dataclass_config(PlannerConfig, planner_values))  # Construct a fresh finite-horizon planner for this trajectory.
        tool_values = _nested_config(config, "world_tool_config")  # Read every canonical frozen action-materialization setting.
        gateway = MCPToolGateway(_dataclass_config(ToolConfig, tool_values, {"theta": 0.5, "max_extra_depth": planner.config.max_extra_depth}))  # Construct a fresh deterministic compiler without result-dependent state.
        world_values = _nested_config(config, "world_model_runtime", "world_vla_config")  # Read frozen online loop settings.
        world_config = _dataclass_config(WorldVLAConfig, world_values, {"max_solves": MAX_SOLVES, "n_equation_cap": job.budget, "theta": 0.5, "method_name": "world_model_vla", "artifact_dir": str(job.output_dir), "require_reference": False})  # Bind registered resources while proving the online controller cannot read Reference B.
        prediction_trace = job.output_dir / "prediction_trace.json"  # Resolve the mandatory content-bound full-WM transition diagnostic artifact.
        audited = build_ablation_runtime(job.case_id, "wm_full", model, planner, gateway, trace_path=prediction_trace)  # Wrap only solve-free logging around the exact frozen V0 components.
        runner.prediction_session = audited.diagnostics  # Retain the recorder so a typed failed next solve still publishes incomplete transition evidence.
        runner.prediction_trace_path = prediction_trace  # Retain the exact job-local trace path for common post-method finalization.
        result = run_world_model_vla(runner, partition=world_partition, config=world_config, model=audited.model, planner=audited.planner, gateway=audited.gateway)  # Execute the new controller with behavior-neutral diagnostics and no legacy partition reconstruction.
        action_payload = {"result": _json_safe(result), "actions": [list(action) for action in result.actions], "decisions": [_json_safe(value) for value in result.decisions], "certificates": [_json_safe(value) for value in result.certificates], "prediction_diagnostics": audited.diagnostics.payload(), "prediction_trace_path": str(prediction_trace), "isolation_receipt": audited.isolation_receipt, "frozen_model": str(model_path)}  # Preserve the complete controller, certification, prediction trace, and unchanged-V0 receipt.
        explicit_timing = getattr(result, "timing_s", None)  # Read separated new-stack timings when the runtime exposes them.
        timing = dict(explicit_timing) if isinstance(explicit_timing, Mapping) else {}  # Preserve exact measured components without inventing absent values.
        timing.setdefault("visual_partition_s", partition_s)  # Record the separately measured frozen semantic-cache load.
    elif job.method == "local_prediction":  # Execute the strong iterative local error-equidistribution baseline.
        action_payload = _run_local_prediction_dynamic(runner, job.budget)  # Calibrate the target per round from this method's current equations/elements.
        timing = {}  # Derive common total and solver timing below.
    elif job.method == "supervised":  # Execute the validation-selected frozen supervised size-field network.
        from ..baselines.bridge_supervised import BridgeSupervisedConfig, deploy_bridge_supervised, load_frozen_supervised_model  # Import the protocol-specific two-solve deployment and authenticated model loader.
        campaign_root = _campaign_root_for_job(job)  # Recover the campaign root under either blind or isolated development layout.
        entry = _matching_artifact(config, "supervised")  # Select the sole validation-chosen supervised checkpoint declaration.
        model_path = _artifact_path(campaign_root, config, "supervised")  # Reverify the selected checkpoint bytes immediately before loading.
        supervised_values = _nested_config(config, "supervised_config", "supervised")  # Read the complete frozen bridge-specific configuration.
        supervised_config = _dataclass_config(BridgeSupervisedConfig, supervised_values, {"gradation": 1.0})  # Reconstruct the frozen network and enforce the common PR-40 V0 gradation.
        selected_seed = int(entry.get("seed", config.get("selected_supervised_seed")))  # Bind deployment to the validation-selected network seed.
        model, model_receipt = load_frozen_supervised_model(model_path, selected_seed=selected_seed, expected_sha256=str(entry["sha256"]), config=supervised_config)  # Authenticate weights and reviewed architecture before the probe.
        deployment = deploy_bridge_supervised(runner, model, n_eq_budget=job.budget, require_reference=False, config=supervised_config, method="supervised")  # Execute exactly two real solves with exact active-DOF Gmsh preflight.
        budget_mesh = deployment.budget_mesh  # Read the deterministic unsolved preflight receipt without serializing its Mesh object.
        deployment_receipt = {"probe_record": _json_safe(deployment.probe_record), "deployed_record": _json_safe(deployment.deployed_record), "budget_mesh": {"scale": float(budget_mesh.scale), "estimated_equations": int(budget_mesh.estimated_equations), "equation_budget": int(budget_mesh.equation_budget), "target_sha256": str(budget_mesh.target_sha256), "mesh_sha256": str(budget_mesh.mesh_sha256), "trials": [_json_safe(value) for value in budget_mesh.trials]}, "real_solve_count": int(deployment.real_solve_count), "hold_last_after_solve": int(deployment.hold_last_after_solve), "online_wall_s": float(deployment.online_wall_s), "mesh_free_serialization": True}  # Preserve all scientific decisions explicitly while excluding coordinates, connectivity, and arrays.
        action_payload = {"algorithm": "supervised-two-solve-hold-last", "deployment": deployment_receipt, "model_receipt": model_receipt, "hold_last_after_solve": int(deployment.hold_last_after_solve)}  # Preserve exact preflight trials, target hash, mesh hash, and hold-last contract.
        timing = {"method_reported_online_s": float(deployment.online_wall_s)}  # Preserve the protocol-specific measured deployment wall time.
    elif job.method.startswith("rl_seed"):  # Execute one of the three independently frozen greedy policies.
        import numpy as np  # Import deterministic argmax only when an RL policy actually runs.
        from ..baselines.bridge_rl import _base_dqn_config, _environment_config, _freeze_policy_for_inference, _policy_action, _safe_estimator_from_log  # Reuse the frozen bridge RL architecture, active-budget state, finite-Q action, and safe diagnostics.
        from ..baselines.rl_dqn import DQNPolicy, RegionRefineEnv  # Reuse the shared-partition Double-DQN environment without its legacy training path.
        index = int(job.method.removeprefix("rl_seed"))  # Convert the stable policy-index label to zero-based index.
        actual_seed = _rl_seed_value(config, index)  # Resolve the protocol's actual large initialization seed.
        campaign_root = _campaign_root_for_job(job)  # Recover the campaign root under either blind or isolated development layout.
        model_path = _artifact_path(campaign_root, config, "rl", actual_seed)  # Reverify this exact independent policy checkpoint.
        problem = runner.problem  # Bind the manifest-reconstructed case.
        partition_started = time.perf_counter()  # Time only frozen semantic-spec loading and verification.
        _world_partition, rl_partitioner, partition_receipt = _load_shared_partitions(campaign_root, config, case, problem)  # Load the same spec and its RL provider.
        runner.final_partition = _world_partition  # Retain the same frozen semantic assignment solely for final-state labels.
        partition_s = time.perf_counter() - partition_started  # Record solve-free cached-partition overhead.
        base_config = _base_dqn_config(actual_seed)  # Reconstruct the sole reviewed training and architecture contract for this actual seed.
        policy = DQNPolicy(base_config)  # Construct a fresh evaluator without replay samples or gradient history.
        policy.load(model_path)  # Load the exact frozen greedy Q network.
        _freeze_policy_for_inference(policy)  # Disable gradients and training-mode behavior before any blind common probe.
        environment = RegionRefineEnv(runner, rl_partitioner, _environment_config(base_config, job.budget), method=job.method)  # Bind the public active budget and shared frozen partition to one isolated trajectory.
        state, adjacency = environment.reset()  # Execute the mandatory common uniform probe and build the frozen-graph state.
        actions: list[dict[str, Any]] = []  # Record every deterministic greedy refine or stop decision.
        done = False  # Continue until stop, resource termination, or five real refinement steps.
        while not done:  # Execute the frozen policy without replay insertion or any optimizer call.
            action = _policy_action(policy, state, adjacency, 0.0, None)  # Validate graph and finite Q values before selecting the deterministic greedy action.
            q_values = np.asarray(policy.q_values(state, adjacency), dtype=float)  # Re-evaluate the already validated deterministic Q vector for the action log.
            stage = "stop" if action == state.shape[0] else f"region_{action}"  # Give the discrete action an auditable semantic-independent label.
            (state, adjacency), reward, done, info = environment.step(action)  # Execute stop or exactly one real Gmsh-plus-CalculiX refinement.
            actions.append({"action_index": action, "action": stage, "q_values": [float(value) for value in q_values], "reward": float(reward), "done": bool(done), "info": _json_safe(info)})  # Preserve full policy choice and realized environment response.
        if policy.grad_steps != 0 or policy.replay:  # Prove this fresh evaluator performed no learning or cross-case state update.
            raise RuntimeError("RL blind evaluator acquired forbidden learning state")  # Retain the trajectory as a protocol failure.
        result = {"solves": int(environment.steps + 1), "final_n_equations": int(environment.last_rec.n_equations), "final_eta": _safe_estimator_from_log(environment.log_eta), "actions": actions}  # Summarize the completed no-learning trajectory with a finite estimator diagnostic.
        action_payload = {"result": result, "policy_index": index, "policy_seed": actual_seed, "frozen_model": str(model_path), "greedy_epsilon": 0.0, "learning_updates": 0, "cross_case_learning": False}  # Preserve seed, model, and inference-isolation evidence.
        timing = {"visual_partition_s": partition_s}  # Record frozen-spec overhead explicitly.
    elif job.method == "dorfler":  # Execute the independent exact element-wise Dörfler safety comparator.
        from ..baselines.dorfler import run_dorfler  # Import the shared theta=0.5 remeshing baseline.
        run_dorfler(runner, theta=0.5, max_rounds=MAX_SOLVES - 1, n_eq_cap=job.budget, gradation=1.0, method="dorfler", require_reference=False)  # Execute one probe plus up to five exact bulk-marking iterations under the common gradation.
        action_payload = {"algorithm": "exact-element-dorfler", "theta": 0.5, "gradation": 1.0, "max_solves": MAX_SOLVES}  # Preserve the safety comparator contract.
        timing = {}  # Derive common total and solver timing below.
    else:  # Defend direct callers that bypass method normalization.
        raise FrozenInputError(f"unsupported execution method {job.method!r}")  # Reject unregistered method dispatch.
    total_s = time.perf_counter() - started  # Stop complete online timing after all policy, meshing, and solves.
    calculix_s = float(sum(record.wall_s for record in runner.records))  # Sum backend-reported solver time independently.
    if job.method == "world_model_vla":  # Reconcile pre-pipeline model/configuration work with the new stack's internal trajectory timer.
        pipeline_total_s = float(timing.get("trajectory_total", 0.0))  # Read the complete internally timed new-stack trajectory once.
        external_partition_s = float(timing.get("visual_partition_s", 0.0))  # Read the separately timed frozen partition registry load.
        timing["harness_setup_s"] = max(float(total_s) - pipeline_total_s - external_partition_s, 0.0)  # Assign model loading, config construction, and diagnostic wrapping outside the pipeline explicitly.
    timing.update({"online_total_s": float(total_s), "calculix_s": calculix_s, "non_solver_total_s": max(float(total_s) - calculix_s, 0.0)})  # Preserve exact available timing without fabricated component allocation.
    return action_payload, timing, partition_receipt  # Return raw method evidence to the common atomic writer.

def _preflight_partitions(root: Path, config: Mapping[str, Any], cases: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:  # Validate every selected shared WM/RL partition before the first blind solve.
    from .partition_spec import PartitionSpecRegistry  # Import the canonical one-file-per-geometry registry.
    expected_value = config.get("partition_spec_sha256", config.get("partition_spec_hashes", {}))  # Read the freeze-record semantic-body digests.
    if not isinstance(expected_value, Mapping):  # Require explicit case-keyed identities for blind execution.
        raise FrozenInputError("partition_spec_sha256 must be a case-id mapping")  # Prevent unfrozen or order-dependent partition inputs.
    expected = {str(key): str(value) for key, value in expected_value.items()}  # Normalize JSON keys and digest values.
    registry = PartitionSpecRegistry(_partition_root(root, config), expected_sha256=expected)  # Bind all loads to the frozen digest mapping.
    evidence: dict[str, dict[str, Any]] = {}  # Collect exact per-case partition provenance.
    for case in sorted((dict(value) for value in cases), key=lambda value: str(value["case_id"])):  # Preserve blind case order during complete preflight.
        case_id = str(case["case_id"])  # Read the content-bound manifest identifier.
        if case_id not in expected:  # Require a freeze-record digest for every selected geometry.
            raise FrozenInputError(f"partition_spec_sha256 lacks selected case {case_id}")  # Refuse an uncommitted semantic partition.
        problem = problem_from_case(case)  # Reconstruct geometry solely through the validated manifest factory.
        spec = registry.partition_for(case_id, problem, str(case["geometry_hash"]))  # Authenticate schema, canonical body, problem fingerprint, and geometry hash.
        path = registry.path_for(case_id)  # Resolve the exact unique per-case file.
        evidence[case_id] = {"path": str(path), "file_sha256": _sha256_file(path), "spec_sha256": str(spec.spec_sha256), "geometry_hash": str(spec.geometry_hash), "probe_sha256": str(spec.probe_sha256), "region_order": list(spec.names)}  # Preserve partition, common-probe, and action-vector identities.
    return evidence  # Return complete selected-shard readiness.

def _preflight_models(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:  # Load and construct every frozen learned method without touching a blind geometry or solver.
    from ..baselines.bridge_rl import _base_dqn_config, _freeze_policy_for_inference  # Reuse the reviewed RL architecture and inference-isolation checks.
    from ..baselines.bridge_supervised import BridgeSupervisedConfig, load_frozen_supervised_model  # Reuse the authenticated selected-network loader.
    from ..baselines.rl_dqn import DQNPolicy  # Construct each frozen policy evaluator solve-free.
    from .world.model import ResidualWorldModel  # Load the new-stack action-conditioned transition library.
    from .world.pipeline import WorldVLAConfig  # Construct the exact runtime config before opening blind outputs.
    from .world.planner import MultiStepPlanner, PlannerConfig  # Construct the finite-horizon planner and validate all invariants.
    from .world.tool_gateway import MCPToolGateway, ToolConfig  # Construct the deterministic action compiler and certification gateway.
    evidence: dict[str, Any] = {}  # Collect exact solve-free model and API construction receipts.
    world_path = _artifact_path(root, config, "world_model")  # Rehash the sole frozen world-model snapshot.
    world_model = ResidualWorldModel.load(world_path)  # Deserialize the complete new-stack transition model without a test state.
    planner_values = _nested_config(config, "world_planner", "planner_config")  # Read the reviewed planner hyperparameters.
    planner = MultiStepPlanner(_dataclass_config(PlannerConfig, planner_values))  # Validate planner dimensions and action bounds solve-free.
    world_values = _nested_config(config, "world_model_runtime", "world_vla_config")  # Read the reviewed online-controller configuration.
    world_config = _dataclass_config(WorldVLAConfig, world_values, {"max_solves": MAX_SOLVES, "n_equation_cap": max(BUDGETS), "theta": 0.5, "require_reference": False})  # Validate the largest registered resource contract without constructing a runner.
    gateway = MCPToolGateway(_dataclass_config(ToolConfig, _nested_config(config, "world_tool_config"), {"theta": 0.5, "max_extra_depth": planner.config.max_extra_depth}))  # Validate canonical deterministic tool parameters and common Dörfler settings.
    evidence["world_model"] = {"path": str(world_path), "sha256": _sha256_file(world_path), "schema_version": str(getattr(world_model, "schema_version", "")), "planner": _json_safe(planner.config), "runtime": _json_safe(world_config), "tool": _json_safe(gateway.config)}  # Preserve complete solve-free construction evidence.
    supervised_entry = _matching_artifact(config, "supervised")  # Resolve the validation-selected supervised checkpoint declaration.
    supervised_path = _artifact_path(root, config, "supervised")  # Rehash the exact selected network bytes.
    supervised_values = _nested_config(config, "supervised_config", "supervised")  # Read the reviewed network and deployment configuration.
    supervised_config = _dataclass_config(BridgeSupervisedConfig, supervised_values, {"gradation": 1.0})  # Enforce the common frozen gradation during solve-free loading.
    selected_seed = int(supervised_entry.get("seed", config.get("selected_supervised_seed")))  # Bind the loader to the validation-selected seed.
    _supervised_model, supervised_receipt = load_frozen_supervised_model(supervised_path, selected_seed=selected_seed, expected_sha256=str(supervised_entry["sha256"]), config=supervised_config)  # Authenticate architecture, seed, config, and weights before TEST_STARTED.
    evidence["supervised"] = {"path": str(supervised_path), "sha256": _sha256_file(supervised_path), "selected_seed": selected_seed, "config": _json_safe(supervised_config), "loader_receipt": _json_safe(supervised_receipt)}  # Preserve solve-free selected-network readiness.
    rl_rows: list[dict[str, Any]] = []  # Collect one fresh evaluator construction receipt per actual frozen seed.
    for index in range(len(RL_METHODS)):  # Preload all three policies before opening the one-shot test boundary.
        actual_seed = _rl_seed_value(config, index)  # Resolve the real training seed behind this stable public index.
        rl_path = _artifact_path(root, config, "rl", actual_seed)  # Rehash this exact independent policy snapshot.
        base_config = _base_dqn_config(actual_seed)  # Reconstruct the reviewed graph architecture and inference settings.
        policy = DQNPolicy(base_config)  # Allocate a new solve-free evaluator for deserialization validation.
        policy.load(rl_path)  # Deserialize the exact greedy Q-network checkpoint.
        _freeze_policy_for_inference(policy)  # Prove inference mode, disabled gradients, and empty online-learning state.
        if policy.grad_steps != 0 or policy.replay:  # Reject any checkpoint loader that restores forbidden online state.
            raise FrozenInputError(f"RL policy seed {actual_seed} contains online learning state")  # Stop before blind disclosure rather than scoring a programming error.
        rl_rows.append({"policy_index": index, "seed": actual_seed, "path": str(rl_path), "sha256": _sha256_file(rl_path), "config": _json_safe(base_config)})  # Preserve exact seed-to-public-index provenance.
    evidence["rl"] = rl_rows  # Attach all three independently constructed frozen policy receipts.
    return evidence  # Return complete learned-method readiness with no mesh or solver call.

def _attach_posthoc_reference_metrics(records: Sequence[SolveRecord], reference: Reference, reference_receipt: Mapping[str, Any]) -> None:  # Compute sealed Reference-B errors only after every online action and mesh is fixed.
    if not math.isfinite(float(reference.U_total)) or float(reference.U_total) <= 0.0:  # Require a valid energy denominator before scoring raw records.
        raise FrozenInputError("Reference B U_total must be finite and positive for posthoc scoring")  # Treat denominator corruption as campaign-invalid evidence.
    if not math.isfinite(float(reference.qoi)) or abs(float(reference.qoi)) <= 1.0e-30:  # Require a stable nonzero QoI denominator.
        raise FrozenInputError("Reference B qoi must be finite and nonzero for posthoc scoring")  # Prevent hidden infinite or undefined QoI errors.
    source = {"mode": "posthoc_only", "used_online": False, "reference_b_sha256": str(reference_receipt.get("reference_b_sha256", "")), "formula": {"energy": "sqrt(max(U_B-U_h,0)/U_B)", "qoi": "abs(Q_h-Q_B)/abs(Q_B)"}}  # Name exact formulas and prove the online runner never received B.
    for record in records:  # Score every successful real solve after the trajectory terminates or numerically fails.
        energy_gap = max(float(reference.U_total) - float(record.U_total), 0.0)  # Apply the common Galerkin-gap non-negativity convention.
        record.e_energy = float(math.sqrt(energy_gap / float(reference.U_total)))  # Compute the sealed common relative energy error.
        record.e_qoi = float(abs(float(record.qoi) - float(reference.qoi)) / abs(float(reference.qoi)))  # Compute the sealed common relative QoI error independently.
        record.extra["posthoc_reference_b"] = dict(source)  # Mark provenance on each raw SolveRecord without affecting any online state.

def _write_final_state(path: Path, runner: ReceiptFemRunner) -> dict[str, Any]:  # Persist the final solved state or the exact first-failure attempted mesh without another solve.
    import numpy as np  # Import compact numerical persistence only when completing a trajectory artifact.
    available = runner.last_post is not None and runner.last_mesh is not None  # Determine whether at least one real solve completed successfully.
    failed_attempt_available = not available and runner.last_attempted_mesh is not None  # Detect a submitted mesh when the first native solve produced no post-state.
    source = "successful_solve" if available else "failed_attempt" if failed_attempt_available else "unavailable"  # Publish an unambiguous state provenance in both NPZ and status.
    state_mesh = runner.last_mesh if available else runner.last_attempted_mesh if failed_attempt_available else None  # Select only a successful mesh unless no solve ever succeeded.
    state_partition = runner.final_partition if available else runner.last_attempted_partition if failed_attempt_available else None  # Use the partition captured for the selected mesh provenance.
    if available:  # Recompute diagnostics from the retained last post-state without another CalculiX solve.
        from ..indicators import zz_indicator  # Reuse the exact common ZZ implementation.
        eta2 = np.asarray(zz_indicator(runner.problem, runner.last_post), dtype="<f8")  # Recompute the exact current elementwise squared indicator.
        labels = np.asarray(state_partition.assign(state_mesh), dtype="<i8") if state_partition is not None and hasattr(state_partition, "assign") else np.empty((0,), dtype="<i8")  # Assign shared semantic regions for the final successful mesh when available.
        if labels.size not in (0, int(state_mesh.n_cells)):  # Reject malformed semantic evidence rather than publishing misaligned labels.
            raise ValueError("final region labels do not align with final mesh elements")  # Treat assignment API drift as a fatal programming error.
        nodes = np.asarray(state_mesh.nodes, dtype="<f8")  # Normalize every final coordinate to portable little-endian storage.
        cells = np.asarray(state_mesh.cells, dtype="<i8")  # Normalize every final simplex connectivity index to portable little-endian storage.
    elif failed_attempt_available:  # Preserve the exact submitted mesh after a typed first-solve native failure.
        nodes = np.asarray(state_mesh.nodes, dtype="<f8")  # Normalize every failed-attempt coordinate to portable little-endian storage.
        cells = np.asarray(state_mesh.cells, dtype="<i8")  # Normalize every failed-attempt simplex connectivity index to portable little-endian storage.
        eta2 = np.empty((0,), dtype="<f8")  # Keep the estimator empty because no physical post-state exists for the failed attempt.
        labels = np.asarray(state_partition.assign(state_mesh), dtype="<i8") if state_partition is not None and hasattr(state_partition, "assign") else np.empty((0,), dtype="<i8")  # Preserve available solve-free semantic labels for the attempted mesh.
        if labels.size not in (0, int(state_mesh.n_cells)):  # Reject a partition assignment that does not align with the failed attempted mesh.
            raise ValueError("failed-attempt region labels do not align with attempted mesh elements")  # Keep malformed semantic evidence campaign-fatal.
    else:  # Emit an explicit empty state for a numerical failure before the first successful solve.
        nodes = np.empty((0, 3), dtype="<f8")  # Preserve a typed empty coordinate array.
        cells = np.empty((0, int(getattr(runner.problem, "dim", 3)) + 1), dtype="<i8")  # Preserve a dimension-consistent empty simplex array.
        eta2 = np.empty((0,), dtype="<f8")  # Preserve an empty estimator vector instead of a fabricated value.
        labels = np.empty((0,), dtype="<i8")  # Preserve explicit unavailable semantic labels.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the exact job artifact directory.
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")  # Select a same-directory atomic temporary with an explicit npz suffix.
    np.savez_compressed(temporary, nodes=nodes, cells=cells, eta2=eta2, region_labels=labels, available=np.asarray([available], dtype=np.bool_), source=np.asarray([source], dtype="<U32"))  # Persist arrays plus explicit successful, failed-attempt, or unavailable provenance.
    os.replace(temporary, path)  # Publish the complete compressed final state atomically.
    return {"path": path.name, "available": bool(available), "source": source, "mesh_sha256": None if state_mesh is None else _full_mesh_sha256(state_mesh), "n_nodes": int(nodes.shape[0]), "n_elements": int(cells.shape[0]), "eta2_count": int(eta2.size), "region_label_count": int(labels.size), "sha256": _sha256_file(path)}  # Return exact status provenance including a failed submitted mesh when present.

def _numerical_failure_payload(exception: BaseException) -> dict[str, Any] | None:  # Classify only explicit native CalculiX or Gmsh failures as retained finite outcomes.
    from ..calculix import CalculiXExecutionError  # Import the typed native solver failure contract.
    from ..mesher import GmshMeshingError  # Import the narrow typed empty/native mesh failure contract.
    if isinstance(exception, CalculiXExecutionError):  # Retain only a typed failed or unusable native CalculiX execution.
        return {"category": "calculix_numerical", "calculix_returncode": exception.returncode, "calculix_wall_s": float(exception.wall_s), "calculix_log_path": str(exception.log_path), "calculix_workdir": str(exception.workdir)}  # Preserve complete native diagnostics.
    if isinstance(exception, GmshMeshingError):  # Retain only a typed Gmsh numerical materialization failure.
        return {"category": "gmsh_numerical"}  # Preserve the explicit narrow failure family.
    return None  # Make integrity, reference, API, serialization, and programming failures fatal to the campaign.

def _assert_pristine_test_output(root: Path) -> None:  # Enforce the irreversible one-shot blind output boundary before TEST_STARTED.
    test_root = root / "test"  # Resolve the sole protocol-owned primary evidence directory.
    if test_root.exists() and (not test_root.is_dir() or any(test_root.iterdir())):  # Reject files, empty precreated shards, status, claims, raw records, or an earlier start marker.
        entries = sorted(str(path.relative_to(root)) for path in test_root.rglob("*") if path.is_file() or path.is_symlink())  # Collect visible material evidence for a bounded diagnostic.
        raise FrozenInputError("blind test output is not pristine; refusing rerun or resume: " + ", ".join(entries[:50] or ["test/<existing-directory-entry>"]))  # Preserve interrupted and disclosed evidence exactly as found.

def _blind_start_payload(request: BenchmarkRequest, plan: Mapping[str, Any], jobs: Sequence[ExecutionJob]) -> dict[str, Any]:  # Bind the one-shot start marker to exact code, config, cases, and execution order.
    freeze = plan.get("freeze_evidence", {})  # Read canonical committed-freeze evidence already verified solve-free.
    git = freeze.get("git", {}) if isinstance(freeze, Mapping) else {}  # Read the dedicated freeze commit evidence.
    case_order = sorted({job.case_id for job in jobs})  # Recompute the immutable lexicographic blind order.
    reference_preflight = plan.get("reference_preflight", {}) if isinstance(plan, Mapping) else {}  # Read the complete solve-free reference qualification receipts.
    reference_qualified = isinstance(reference_preflight, Mapping) and len(reference_preflight) == len(case_order) and all(isinstance(value, Mapping) and value.get("qualification") is True for value in reference_preflight.values())  # Preserve one campaign-level original threshold result without treating authorization as convergence.
    amendment = plan.get("reference_execution_amendment") if request.allow_unqualified_references else None  # Read the protected human amendment pointer already verified solve-free.
    return {"schema": "wmvla-four-way-test-started-v1", "protocol_id": PROTOCOL_ID, "started_utc": _utc_now(), "one_shot": True, "resume_allowed": False, "freeze_commit_sha": git.get("freeze_commit_sha"), "implementation_commit_sha": freeze.get("implementation_commit_sha") if isinstance(freeze, Mapping) else None, "freeze_index_sha256": freeze.get("freeze_index_sha256") if isinstance(freeze, Mapping) else None, "frozen_config_path": str(request.frozen_config_path), "frozen_config_sha256": _sha256_file(request.frozen_config_path), "manifest_path": str(request.manifest_path), "manifest_sha256": _sha256_file(request.manifest_path), "case_order": case_order, "case_count": len(case_order), "budgets": list(BUDGETS), "methods": list(ALL_METHODS), "allow_unqualified_references": bool(request.allow_unqualified_references), "expedited_reference_levels": plan.get("expedited_reference_levels"), "reference_execution_amendment_sha256": amendment.get("sha256") if isinstance(amendment, Mapping) else None, "REFERENCE_QUALIFIED": bool(reference_qualified), "reference_unqualified_case_ids": sorted(str(case_id) for case_id, value in reference_preflight.items() if isinstance(value, Mapping) and value.get("qualification") is not True), "ordered_jobs": [{"case_id": job.case_id, "equation_budget": int(job.budget), "method": job.method} for job in jobs]}  # Preserve every coordinate, frozen amendment identity, and reference qualification before the first solve.

def build_diagnostic_plan(case_ids: Sequence[str]) -> dict[str, Any]:  # Preregister all six mandatory B=60000, K=6 mechanism diagnostics without executing them.
    from .four_way_ablations import ABLATION_BUDGET, ABLATION_MAX_SOLVES, ALL_VARIANTS, RANDOM_SAFE_SEEDS  # Reuse the isolated diagnostic module's frozen vocabulary and resources.
    jobs: list[dict[str, Any]] = []  # Collect deterministic variants and every independent random-safe seed explicitly.
    for case_id in sorted(case_ids):  # Preserve the same blind case order as the primary campaign.
        for variant in ALL_VARIANTS:  # Register all six diagnostic labels before any result exists.
            if variant == "wm_full":  # Reuse the content-bound primary B60000/K6 trajectory rather than spending hidden extra solves.
                jobs.append({"case_id": case_id, "variant": variant, "seed": None, "mode": "reuse_primary_wm_full", "equation_budget": int(ABLATION_BUDGET), "max_solves": int(ABLATION_MAX_SOLVES)})  # Declare the no-rerun reuse adapter explicitly.
            elif variant == "random_safe_extra":  # Expand the preregistered five-seed control without best-seed selection.
                jobs.extend({"case_id": case_id, "variant": variant, "seed": int(seed), "mode": "isolated_ablation_runtime", "equation_budget": int(ABLATION_BUDGET), "max_solves": int(ABLATION_MAX_SOLVES)} for seed in RANDOM_SAFE_SEEDS)  # Register all five random controls.
            else:  # Register each deterministic deployable or oracle diagnostic once.
                jobs.append({"case_id": case_id, "variant": variant, "seed": None, "mode": "isolated_ablation_runtime", "equation_budget": int(ABLATION_BUDGET), "max_solves": int(ABLATION_MAX_SOLVES)})  # Preserve exact resources and isolation mode.
    return {"schema": "wmvla-four-way-ablation-plan-v1", "protocol_id": PROTOCOL_ID, "case_order": sorted(case_ids), "variants": list(ALL_VARIANTS), "random_safe_seeds": list(RANDOM_SAFE_SEEDS), "runtime_builder": "visionamr.vla.four_way_ablations.build_ablation_runtime", "primary_reuse": "visionamr.vla.four_way_ablations.reuse_primary_wm_full", "case_summary_builder": "visionamr.vla.four_way_ablations.build_ablation_case_summary", "campaign_summary_builder": "visionamr.vla.four_way_ablations.build_ablation_campaign_summary", "jobs": jobs}  # Return a CLI-serializable adapter plan with exact API provenance.

def _job_identity(job: ExecutionJob) -> dict[str, Any]:  # Serialize the immutable coordinates of one independent trajectory.
    return {"case_id": job.case_id, "split": job.split, "geometry_hash": job.geometry_hash, "equation_budget": int(job.budget), "method": job.method, "max_solves": MAX_SOLVES}  # Return only preregistered job coordinates.

def _valid_metric(value: Any) -> float | None:  # Validate one stored relative error for prefix delivery.
    if value is None:  # Treat explicit no-reference or failed values as unavailable.
        return None  # Preserve the missing metric for finite failure scoring downstream.
    numeric = float(value)  # Normalize NumPy and Python scalar wrappers.
    return numeric if math.isfinite(numeric) and numeric >= 0.0 else None  # Accept only finite physically meaningful relative errors.

def _wm_prefix_safety(action_payload: Mapping[str, Any], solve_limit: int) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:  # Extract proactive and target-dominance evidence available by one real-solve prefix.
    certificate_values = action_payload.get("certificates", [])  # Read exact materialization receipts from the new world stack.
    certificates = [value for value in certificate_values if isinstance(value, Mapping)][: max(int(solve_limit) - 1, 0)]  # Pair each pre-next-solve action with prefixes that actually include its resulting solve.
    def compiled_passed(value: Mapping[str, Any]) -> bool:  # Validate the v2 compiled-field Dörfler certificate used by the real Gmsh action.
        proactive_value = str(value.get("source", "")) == "world_model" and any(int(depth) > 0 for depth in value.get("executed_action", []))  # Determine whether a proactive compiled-field hash is mandatory.
        base_hash = str(value.get("base_compiled_field_sha256", ""))  # Read the always-required compiled Dörfler field identity.
        world_hash = str(value.get("world_compiled_field_sha256", ""))  # Read the proactive compiled target identity when applicable.
        return str(value.get("schema_version", "")) == "wmvla.mcp-tool.v2" and bool(value.get("base_target_included")) and bool(value.get("no_coarsening")) and bool(value.get("compiled_dorfler_included")) and float(value.get("compiled_max_dorfler_violation", float("inf"))) <= 1.0e-12 and float(value.get("compiled_field_gradation", float("nan"))) == 1.0 and len(base_hash) == 64 and (not proactive_value or len(world_hash) == 64)  # Require exact compiled dominance, hashes, tolerance, schema, and common gradation.
    dominance = [{"action_id": str(value.get("target_sha256", f"action_{index + 1}")), "passed": compiled_passed(value), "source": str(value.get("source", "")), "executed_action": list(value.get("executed_action", [])), "schema_version": value.get("schema_version"), "base_compiled_field_sha256": value.get("base_compiled_field_sha256"), "world_compiled_field_sha256": value.get("world_compiled_field_sha256"), "compiled_field_gradation": value.get("compiled_field_gradation"), "compiled_max_dorfler_violation": value.get("compiled_max_dorfler_violation")} for index, value in enumerate(certificates)]  # Preserve full compiled-field non-coarsening evidence and exact identities.
    proactive = any(bool(value.get("accepted")) and str(value.get("source")) == "world_model" and any(int(depth) > 0 for depth in value.get("executed_action", [])) and compiled_passed(value) for value in certificates)  # Require a v2-certified nonzero action whose resulting solve is in the prefix.
    fallback = [{"trigger": str(value.get("reason", "")), "dorfler_executed": not any(int(depth) > 0 for depth in value.get("executed_action", [])), "source": str(value.get("source", ""))} for value in certificates if str(value.get("source", "")) != "world_model" or not bool(value.get("accepted"))]  # Preserve every rejection or resource fallback and whether exact Dörfler executed.
    return proactive, dominance, fallback  # Return complete structural and mechanism evidence for this prefix.

def derive_prefix_rows(job: ExecutionJob, records: Sequence[SolveRecord], completed: bool, failure: Mapping[str, Any] | None, action_payload: Mapping[str, Any]) -> list[dict[str, Any]]:  # Derive all four registered K values from one actual independent budget trajectory.
    failure_at = None if failure is None else int(failure.get("failure_at_solve", 1))  # Read the first unsuccessful solver or method step.
    rows: list[dict[str, Any]] = []  # Collect the complete four-point true-prefix grid for this trajectory.
    for solve_limit in SOLVE_LIMITS:  # Derive only the preregistered real-solve limits.
        inspected = [record for record in records if int(record.solve_index) <= solve_limit]  # Restrict evidence to actually completed solves in this prefix.
        feasible = [record for record in inspected if int(record.n_equations) <= job.budget]  # Exclude over-budget solves without substituting their errors.
        failure_affects = failure_at is not None and failure_at <= solve_limit  # Retain earlier unaffected prefixes while failing this and later prefixes.
        energy_values = [] if failure_affects else [(value, record) for record in feasible if (value := _valid_metric(record.e_energy)) is not None]  # Collect valid Reference-B energy errors only from successful feasible solves.
        qoi_values = [] if failure_affects else [(value, record) for record in feasible if (value := _valid_metric(record.e_qoi)) is not None]  # Apply the independent best-prefix rule to QoI.
        energy_best = min(energy_values, key=lambda item: item[0]) if energy_values else None  # Select the smallest feasible energy error and its real solve.
        qoi_best = min(qoi_values, key=lambda item: item[0]) if qoi_values else None  # Select the smallest feasible QoI error independently.
        proactive, dominance, fallback = _wm_prefix_safety(action_payload, solve_limit) if job.method == "world_model_vla" else (False, [], [])  # Attach structural evidence only to the world-model trajectory.
        rows.append({"case_id": job.case_id, "method": job.method, "solves": int(solve_limit), "equation_budget": int(job.budget), "energy_error": None if energy_best is None else float(energy_best[0]), "qoi_error": None if qoi_best is None else float(qoi_best[0]), "energy_ok": energy_best is not None, "qoi_ok": qoi_best is not None, "energy_best_solve": None if energy_best is None else int(energy_best[1].solve_index), "qoi_best_solve": None if qoi_best is None else int(qoi_best[1].solve_index), "budget_violation": any(int(record.n_equations) > job.budget for record in inspected), "failure_affects_prefix": failure_affects, "successful_solves_available": len(inspected), "hold_last_after_stop": len(inspected) < solve_limit and completed, "certified_proactive_action": proactive, "target_dominance_checks": dominance, "fallback_events": fallback})  # Preserve accuracy, feasibility, failure, hold-last, and safety semantics in one public-grid row.
    return rows  # Return all four K values generated from real trajectory prefixes without rerunning the method.

def _completed_status(path: Path, job: ExecutionJob) -> dict[str, Any] | None:  # Recognize only a complete status marker for this exact job.
    if not path.is_file():  # Treat absence as an unstarted job.
        return None  # Return an explicit no-status sentinel.
    payload = _read_json(path)  # Parse the atomic completion marker.
    if not isinstance(payload, Mapping) or payload.get("schema") != RESULT_SCHEMA or payload.get("job") != _job_identity(job):  # Reject stale, corrupt, or cross-job markers.
        raise FrozenInputError(f"existing status marker does not match job: {path}")  # Prevent accidental overwrite of unrelated evidence.
    return dict(payload)  # Return an independent completed status record.

def execute_job(request: BenchmarkRequest, job: ExecutionJob, case: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:  # Execute one isolated trajectory while retaining every numerical failure.
    status_path = job.output_dir / "status.json"  # Use the atomic status marker as the sole completion signal.
    existing = _completed_status(status_path, job)  # Inspect only the exact job's completion state.
    if existing is not None:  # Handle a previously completed or failed job without destructive overwrite.
        if request.resume and existing.get("completed") is True:  # Skip only a fully persisted prior trajectory during controlled resume.
            return {"job": _job_identity(job), "status": "skipped_completed", "completed": True, "status_path": str(status_path)}  # Preserve transparent resume accounting.
        raise FrozenInputError(f"job already has a status marker; refusing overwrite: {status_path}")  # Require explicit evidence review before any retry.
    job.output_dir.mkdir(parents=True, exist_ok=True)  # Create the exact method evidence directory after global preflight.
    claim_path = job.output_dir / ".running.claim"  # Reserve one atomic process-local claim file for controlled shards.
    try:  # Convert duplicate concurrent claims into an explicit protocol error.
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)  # Claim this exact case-budget-method trajectory once.
    except FileExistsError as exc:  # Detect a running or interrupted sibling process.
        raise FrozenInputError(f"job is already claimed: {claim_path}") from exc  # Prevent duplicate blind-test execution.
    os.write(descriptor, f"pid={os.getpid()} started_utc={_utc_now()}\n".encode("ascii"))  # Record bounded non-secret claim diagnostics.
    os.close(descriptor)  # Release the claim descriptor while retaining the exclusive file.
    started_utc = _utc_now()  # Record the job start after successful exclusive claim.
    runner = ReceiptFemRunner(problem_from_case(case), job.output_dir, keep_files=False, ccx_timeout=float(config.get("ccx_timeout_s", 1800.0)))  # Create isolated honest solve accounting for this manifest case.
    action_payload: dict[str, Any] = {}  # Reserve method-specific actions or diagnostics for both success and partial failure.
    timing: dict[str, Any] = {}  # Reserve measured online timing for both success and partial failure.
    partition_receipt: dict[str, Any] | None = None  # Reserve shared semantic partition provenance when applicable.
    failure: dict[str, Any] | None = None  # Initialize the retained failure ledger entry.
    completed = False  # Mark completion only after the method returns without a numerical exception.
    method_started = time.perf_counter()  # Measure partial native-failure trajectories even when method dispatch does not return timing.
    try:  # Retain only explicitly typed native numerical failures rather than swallowing protocol or programming defects.
        if runner.reference is not None:  # Prove the online runner did not acquire truth through construction or a stale helper.
            raise FrozenInputError("online method runner unexpectedly contains Reference B before execution")  # Stop the entire campaign before a leaked action.
        action_payload, timing, partition_receipt = _execute_method(job, case, config, runner)  # Execute the fresh frozen method trajectory.
        if runner.reference is not None:  # Detect any method that tried to bind or build reference truth during online execution.
            raise FrozenInputError(f"online method {job.method} bound a forbidden reference")  # Invalidate the round instead of posthoc-scoring leaked decisions.
        completed = True  # Mark numerical completion after the method returns all raw evidence.
    except Exception as exception:  # Classify an exception without downgrading integrity, API, or programming failures.
        native = _numerical_failure_payload(exception)  # Accept only explicit typed CalculiX or Gmsh numerical failures.
        if native is None:  # Make every unregistered failure campaign-fatal by default.
            raise  # Preserve its original type and complete traceback for TEST_INVALID evidence.
        counter = int(getattr(runner, "_counter", len(runner.records)))  # Read the honest invocation counter, which advances before a native solve attempt.
        failure_at_solve = counter if counter > len(runner.records) else len(runner.records) + 1  # Identify the failed current solve or the next method stage after completed records.
        failure = {**native, "exception_type": type(exception).__name__, "message": str(exception).replace("\x00", " ")[:1000], "successful_solve_count": len(runner.records), "failure_at_solve": int(max(failure_at_solve, 1)), "traceback": traceback.format_exc(limit=40)}  # Preserve bounded typed-native diagnostics without NaN or dropped observations.
        ledger_root = request.root / "test" if request.split == "test" else request.root / "development" / request.split  # Keep development failures outside the irreversible blind tree.
        _append_jsonl(ledger_root / "failure_ledger.jsonl", {"schema": "wmvla-four-way-failure-ledger-v1", "protocol_id": PROTOCOL_ID, "recorded_utc": _utc_now(), "job": _job_identity(job), "failure": failure})  # Append one finite retained native-failure row for aggregate scoring.
        timing = {"online_total_s": float(time.perf_counter() - method_started), "calculix_s": float(sum(record.wall_s for record in runner.records)), "partial_trajectory": True}  # Preserve transparent partial timing without inventing unavailable components.
    if runner.reference is not None:  # Reassert truth isolation after a partial native failure as well as normal completion.
        raise FrozenInputError(f"online method {job.method} acquired forbidden Reference B")  # Make any leakage fatal before metric attachment.
    reference, reference_receipt = _reference_payload(request.root, case, allow_unqualified=request.allow_unqualified_references, expedited_levels=int(config["expedited_reference_levels"]) if request.allow_unqualified_references else None, amendment_record=config.get("reference_execution_amendment") if request.allow_unqualified_references else None)  # Load the denominator under the authenticated frozen schedule and amendment identity only after every online decision is fixed.
    _attach_posthoc_reference_metrics(runner.records, reference, reference_receipt)  # Compute common B-relative errors only after actions, meshes, and stopping are irrevocably fixed.
    prediction_session = getattr(runner, "prediction_session", None)  # Read optional full-WM diagnostics retained across a typed native failure.
    prediction_trace_path = getattr(runner, "prediction_trace_path", None)  # Read the exact optional trace artifact destination.
    if prediction_session is not None and prediction_trace_path is not None:  # Publish even a zero-transition or interrupted full-WM trace explicitly.
        _write_json(Path(prediction_trace_path), prediction_session.payload())  # Persist completed and incomplete prediction evidence without another solve.
    copied_logs = _copy_solver_logs(runner, job.output_dir / "solver_logs")  # Preserve every surviving native solver log on both success and failure.
    final_state_receipt = _write_final_state(job.output_dir / "final_state.npz", runner)  # Persist the final successful mesh and current ZZ state without another solve.
    records_payload = {"schema": RESULT_SCHEMA, "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "case": {"case_id": case["case_id"], "parameters": case["parameters"], "config_hash": case["config_hash"], "geometry_hash": case["geometry_hash"]}, "reference_b": {**dict(reference_receipt), "usage": "posthoc_only", "used_online": False}, "completed": completed, "failure": failure, "records": [_json_safe(record) for record in runner.records]}  # Assemble every successful raw SolveRecord and mark sealed-truth usage explicitly.
    _write_json(job.output_dir / "records.json", records_payload)  # Persist the complete method trajectory before the completion marker.
    _write_json(job.output_dir / "mesh_receipts.json", {"schema": "wmvla-four-way-mesh-receipts-v1", "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "receipts": runner.mesh_receipts})  # Persist each real solve's exact mesh and resource identity.
    _write_json(job.output_dir / "action_log.json", {"schema": "wmvla-four-way-action-log-v1", "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "partition_spec": partition_receipt, "actions": action_payload, "failure": failure})  # Persist policy actions, world certificates, or baseline adaptation decisions.
    _write_json(job.output_dir / "timing.json", {"schema": "wmvla-four-way-timing-v1", "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "timing_s": timing, "solver_logs": copied_logs, "completed": completed})  # Persist measured timing separately from accuracy records.
    prefix_rows = derive_prefix_rows(job, runner.records, completed, failure, action_payload)  # Construct every public K value from this one real trajectory.
    _write_json(job.output_dir / "prefix_results.json", {"schema": "wmvla-four-way-prefix-results-v1", "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "derivation": "best_feasible_actual_prefix", "rows": prefix_rows})  # Persist the complete four-prefix grid for aggregation and independent audit.
    artifacts = ["records.json", "mesh_receipts.json", "action_log.json", "timing.json", "prefix_results.json", "final_state.npz"] + (["prediction_trace.json"] if (job.output_dir / "prediction_trace.json").is_file() else [])  # Enumerate every required common artifact and optional full-WM diagnostic trace.
    status = {"schema": RESULT_SCHEMA, "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "started_utc": started_utc, "finished_utc": _utc_now(), "completed": completed, "successful_solve_count": len(runner.records), "failure": failure, "final_state": final_state_receipt, "artifacts": artifacts}  # Build the atomic final job status after all required raw artifacts exist.
    _write_json(status_path, status)  # Publish the completion marker last so resume never accepts partial evidence.
    claim_path.unlink(missing_ok=True)  # Remove only the disposable exact-job claim after durable status publication.
    return {"job": _job_identity(job), "status": "completed" if completed else "failed", "completed": completed, "status_path": str(status_path), "successful_solve_count": len(runner.records)}  # Return a concise invocation-level ledger row.

def run_benchmark(request: BenchmarkRequest) -> dict[str, Any]:  # Preflight the full selected shard and optionally execute every job without early stopping.
    plan, config, jobs = build_plan(request)  # Validate the manifest, freeze document, filters, and exact job order.
    manifest = load_case_manifest(request.manifest_path, verify_checksum=True)  # Reload the authenticated manifest for case reconstruction during execution.
    selected_ids = sorted({job.case_id for job in jobs})  # Collect exact selected case identities.
    by_id = {str(case["case_id"]): dict(case) for case in manifest["cases"] if str(case["case_id"]) in selected_ids}  # Restrict runtime case access to the selected shard.
    if request.split == "test":  # Complete all read-only blind prerequisites before the first native solve.
        _assert_pristine_test_output(request.root)  # Refuse any earlier start, raw, status, claim, or partial directory before reading scientific evidence.
        plan["reference_preflight"] = _preflight_references(request.root, config, (by_id[case_id] for case_id in selected_ids), allow_unqualified=request.allow_unqualified_references)  # Authenticate every selected A/B ladder under the frozen strict or authorized two-level policy before TEST_STARTED.
        if any(method in normalize_methods(request.methods) for method in ("world_model_vla", *RL_METHODS)):  # Require shared semantic inputs only when WM or RL is selected.
            plan["partition_preflight"] = _preflight_partitions(request.root, config, (by_id[case_id] for case_id in selected_ids))  # Authenticate every selected shared assignment and fixed graph.
        plan["model_preflight"] = _preflight_models(request.root, config)  # Load and construct every frozen model and runtime API solve-free before the start marker.
        plan["diagnostic_plan"] = build_diagnostic_plan(selected_ids)  # Preregister every mandatory mechanism diagnostic and random seed before any result exists.
    if request.dry_run:  # Stop before directories, claims, Gmsh, or CalculiX execution.
        return plan  # Return the complete validated solve-free plan to the caller.
    if request.split == "test":  # Open the irreversible blind campaign only after every global preflight succeeds.
        test_root = request.root / "test"  # Resolve the one-shot evidence root after the final pristine check.
        test_root.mkdir(parents=True, exist_ok=True)  # Create or reuse only an already verified empty blind directory.
        _assert_pristine_test_output(request.root)  # Close the preflight-to-start race against any concurrently created raw evidence.
        _write_json_exclusive(test_root / "TEST_STARTED.json", _blind_start_payload(request, plan, jobs))  # Publish freeze/head/config/case order exactly once before the first native solve.
    outcomes: list[dict[str, Any]] = []  # Collect every job result, including failures and resume skips.
    try:  # Invalidate and stop the round immediately on any non-numerical protocol, API, or programming failure.
        for job in jobs:  # Preserve sorted case, budget, and canonical method order without result-dependent interruption.
            outcomes.append(execute_job(request, job, by_id[job.case_id], config))  # Execute or retain this exact independent trajectory.
    except Exception as exception:  # Publish a bounded invalid-campaign marker without converting the failure into a scored method point.
        if request.split == "test":  # Preserve irreversible blind evidence only for the opened test campaign.
            _write_json(request.root / "test" / "TEST_INVALID.json", {"schema": "wmvla-four-way-test-invalid-v1", "protocol_id": PROTOCOL_ID, "invalidated_utc": _utc_now(), "error_type": type(exception).__name__, "error": str(exception).replace("\x00", " ")[:2000], "traceback": traceback.format_exc(limit=80), "completed_job_count": len(outcomes), "next_job": None if len(outcomes) >= len(jobs) else _job_identity(jobs[len(outcomes)]), "rerun_allowed": False})  # Retain the exact stop boundary and forbid silent continuation.
        raise  # Preserve the original fatal exception for the CLI exit status and audit traceback.
    terminal_statuses = ("completed", "failed", "skipped_completed")  # Define every durable per-job terminal state, including a retained typed native failure.
    terminal_count = sum(outcome.get("status") in terminal_statuses for outcome in outcomes)  # Count all jobs with a durable successful, failed, or verified-skipped terminal marker.
    successful_count = sum(outcome.get("status") in ("completed", "skipped_completed") for outcome in outcomes)  # Count successful fresh trajectories and verified prior successes separately.
    failed_count = sum(outcome.get("status") == "failed" for outcome in outcomes)  # Count retained numerical failures without dropping points.
    all_jobs_completed = len(outcomes) == len(jobs) and terminal_count == len(outcomes)  # Require one recognized terminal outcome for every planned job independent of scientific success.
    summary = {"schema": "wmvla-four-way-execution-summary-v1", "protocol_id": PROTOCOL_ID, "finished_utc": _utc_now(), "plan": plan, "job_outcomes": outcomes, "completed_job_count": successful_count, "terminal_job_count": terminal_count, "successful_job_count": successful_count, "failed_job_count": failed_count, "all_jobs_completed": all_jobs_completed}  # Preserve completed-job compatibility while separating every terminal outcome from scientific success.
    summary_root = request.root / "test" if request.split == "test" else request.root / "development" / request.split  # Keep development summaries outside blind evidence.
    summary_path = summary_root / ("EXECUTION_SUMMARY.json" if request.split == "test" else f"execution_summaries/execution_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.json")  # Use one fixed blind summary or collision-resistant development summary.
    _write_json(summary_path, summary)  # Persist the complete invocation result for later aggregation.
    summary["summary_path"] = str(summary_path)  # Expose the exact durable summary location to the CLI.
    return summary  # Return the complete result without interpreting scientific gates.
