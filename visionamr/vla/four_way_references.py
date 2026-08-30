"""Build, cache, verify, and bind the frozen two-level bridge references."""  # State the module's complete protocol responsibility.
from __future__ import annotations  # Postpone annotation evaluation for stable imports.

from dataclasses import asdict, dataclass  # Import immutable configuration and JSON record helpers.
from datetime import datetime, timezone  # Import an explicit UTC clock for audit timestamps.
import hashlib  # Import SHA-256 for cache and field identities.
import json  # Import deterministic JSON persistence for the reference ledger.
import math  # Import finite-value checks for solver outputs and configuration values.
from pathlib import Path  # Import portable paths for per-case reference caches.
import time  # Import a monotonic timer for mesh and total build durations.
from typing import Any, Callable  # Import the heterogeneous ledger and injected mesh-factory annotations.

import numpy as np  # Import numerical scalars and realized mesh-size summaries.

from ..experiment import FemRunner, Reference, reference_floor, reference_size_fn  # Reuse the frozen strong reference and shared runner contract.
from ..geometry import Problem  # Import the manifest-reconstructed bridge problem contract.
from ..mesher import Mesh, generate_mesh  # Reuse the deterministic Gmsh path and mesh identity implementation.

PROTOCOL_ID = "WMVLA-4WAY-P1"  # Bind every reference artifact to the frozen four-way experiment.
REFERENCE_SCHEMA = "wmvla-four-way-reference-ladder-v3"  # Version the ledger after adding explicit nonblocking operational-reference qualification.
REFERENCE_ARTIFACT_SCHEMA = "wmvla-four-way-reference-b-v3"  # Bind the compact common reference to qualification and amendment metadata.
LEDGER_FILENAME = "reference_ledger.json"  # Freeze the per-case audit-ledger filename.
REFERENCE_B_FILENAME = "reference_B.json"  # Freeze the per-case common-reference filename.
SOLVER_LOG_DIRECTORY = "solver_logs"  # Freeze the portable per-case native-log directory below each reference cache.
SOLVER_INPUT_DIRECTORY = "solver_inputs"  # Freeze the failure-only native input directory below each reference cache.
UNQUALIFIED_AUTHORIZATION = "user_authorized_nonblocking_2026-08-30"  # Record the user's exact expedited nonblocking authorization without rewriting the original convergence gate.
DEFAULT_BACKGROUND_SCALES = (1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768)  # Pre-register successively finer background targets without inspecting method meshes.
DEFAULT_LOCAL_FLOOR_SCALES = (1.0, 0.7, 0.49, 0.343, 0.2401, 0.16807)  # Pre-register successively finer local floors independently of method outcomes.


class ReferenceBuildError(RuntimeError):  # Identify numerical, cache, and schedule failures without fabricating a reference.
    """Raised when a valid common Reference B cannot be produced or verified."""  # Describe the public failure contract.


@dataclass(frozen=True)  # Prevent post-registration mutation of the convergence schedule.
class ReferenceScheduleConfig:  # Define every choice that may affect reference construction or acceptance.
    agreement_rtol: float = 0.005  # Require at most one-half-percent relative change in both energy and QoI.
    background_scales: tuple[float, ...] = DEFAULT_BACKGROUND_SCALES  # Freeze the analytical-field background sequence.
    local_floor_scales: tuple[float, ...] = DEFAULT_LOCAL_FLOOR_SCALES  # Freeze the analytical-field local-floor sequence.
    zero_guard: float = 1.0e-30  # Treat effectively zero denominators as invalid convergence evidence.

    def __post_init__(self) -> None:  # Validate the complete schedule before any geometry or solver work starts.
        if not math.isfinite(self.agreement_rtol) or self.agreement_rtol <= 0.0:  # Reject missing, infinite, or nonpositive tolerances.
            raise ValueError("agreement_rtol must be finite and positive")  # Report the invalid convergence threshold directly.
        if not math.isfinite(self.zero_guard) or self.zero_guard <= 0.0:  # Reject a guard that cannot protect relative differences.
            raise ValueError("zero_guard must be finite and positive")  # Report the invalid denominator guard directly.
        if len(self.background_scales) != len(self.local_floor_scales) or len(self.background_scales) < 2:  # Require a paired A/B schedule with no missing scale.
            raise ValueError("reference schedules must have equal length of at least two")  # Reject incomplete or unpaired schedules.
        for values, name in ((self.background_scales, "background"), (self.local_floor_scales, "local floor")):  # Validate both scale sequences under the same rules.
            if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):  # Require every registered scale to be finite and positive.
                raise ValueError(f"{name} scales must be finite and positive")  # Identify the invalid schedule family.
            if not math.isclose(float(values[0]), 1.0, rel_tol=0.0, abs_tol=0.0):  # Require level zero to reproduce the existing strong reference exactly.
                raise ValueError(f"the first {name} scale must equal one")  # Protect the definition of Reference A.
            if any(float(newer) >= float(older) for older, newer in zip(values, values[1:])):  # Require strict refinement at every adjacent candidate level.
                raise ValueError(f"{name} scales must be strictly decreasing")  # Reject stalled or coarsening reference schedules.


DEFAULT_REFERENCE_CONFIG = ReferenceScheduleConfig()  # Publish one immutable preregistered default configuration.


@dataclass(frozen=True)  # Make the returned accepted pair immutable for downstream consumers.
class ReferenceBuildOutcome:  # Return both converged levels while exposing the shared final Reference B.
    reference_a: Reference  # Store the last coarser converged reference level.
    reference_b: Reference  # Store the final finer shared reference used by every method.
    a_level: int  # Record the zero-based ladder index promoted to final Reference A.
    b_level: int  # Record the zero-based ladder index accepted as final Reference B.
    ledger_path: Path  # Expose the complete per-level audit ledger.
    from_cache: bool  # Disclose whether this call solved or loaded verified results.
    qualification: bool = True  # Distinguish a dual-gate-qualified Reference B from an explicitly operational unqualified fallback.
    authorization: str | None = None  # Preserve the exact nonblocking authorization only for unqualified operational use.
    execution_amendment: dict[str, Any] | None = None  # Disclose any expedited schedule prefix even when it happened to qualify.


def _json_ready(value: Any) -> Any:  # Normalize common scientific values into canonical JSON-compatible primitives.
    if isinstance(value, dict):  # Recurse through heterogeneous mappings while normalizing keys.
        return {str(key): _json_ready(item) for key, item in value.items()}  # Preserve mapping content with stable string keys.
    if isinstance(value, (list, tuple)):  # Recurse through ordered sequences without retaining tuple-specific encoding.
        return [_json_ready(item) for item in value]  # Preserve sequence order in JSON form.
    if isinstance(value, np.generic):  # Convert NumPy scalar wrappers before the standard encoder sees them.
        return value.item()  # Return the corresponding Python scalar.
    if isinstance(value, Path):  # Normalize filesystem paths for portable ledger storage.
        return str(value)  # Store the path text without platform-specific object encoding.
    return value  # Leave standard JSON primitives unchanged.


def _canonical_bytes(payload: Any) -> bytes:  # Serialize one payload into a stable collision-resistant byte representation.
    return json.dumps(_json_ready(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")  # Exclude whitespace, key-order, locale, and NaN ambiguity.


def _payload_sha256(payload: Any) -> str:  # Hash a canonical scientific payload with full SHA-256.
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()  # Return the complete lowercase digest.


def _mesh_sha256(mesh: Mesh) -> str:  # Hash exact mesh arrays with a full collision-resistant digest for protocol evidence.
    nodes = np.asarray(mesh.nodes, dtype="<f8")  # Normalize every coordinate to portable little-endian float64 bytes.
    cells = np.asarray(mesh.cells, dtype="<i8")  # Normalize every connectivity index to portable little-endian int64 bytes.
    digest = hashlib.sha256()  # Allocate one full SHA-256 state rather than the repository's legacy 16-character display hash.
    digest.update(_canonical_bytes({"nodes_shape": list(nodes.shape), "cells_shape": list(cells.shape), "dim": int(mesh.dim)}))  # Bind shapes and spatial dimension before raw arrays.
    digest.update(np.ascontiguousarray(nodes).tobytes())  # Bind all node coordinates in canonical row-major order.
    digest.update(np.ascontiguousarray(cells).tobytes())  # Bind all simplex connectivity in canonical row-major order.
    return digest.hexdigest()  # Return all 64 lowercase hexadecimal characters for durable reference provenance.


def _file_sha256(path: Path) -> str:  # Hash a retained native evidence file completely using bounded memory.
    digest = hashlib.sha256()  # Allocate one collision-resistant state for this exact persisted file.
    with path.open("rb") as handle:  # Stream the complete evidence without assuming a small native log.
        for block in iter(lambda: handle.read(1024 * 1024), b""):  # Read deterministic one-megabyte blocks through EOF.
            digest.update(block)  # Incorporate every retained byte in its original order.
    return digest.hexdigest()  # Return all sixty-four lowercase hexadecimal characters.


def _evidence_record(case_dir: Path, path: Path, *, source_count: int = 1) -> dict[str, Any]:  # Describe one case-local native artifact with a portable path and exact identity.
    relative = path.resolve().relative_to(case_dir.resolve()).as_posix()  # Store only a cache-relative path after proving containment.
    return {"path": relative, "sha256": _file_sha256(path), "size_bytes": int(path.stat().st_size), "source_count": int(source_count)}  # Bind portable location, complete digest, byte length, and aggregation count.


def _combine_native_files(sources: list[Path]) -> bytes:  # Preserve every discovered native log byte in one stable level-scoped portable file.
    if len(sources) == 1:  # Avoid altering the ordinary single combined CalculiX log in any way.
        return sources[0].read_bytes()  # Copy the exact backend log bytes unchanged.
    chunks: list[bytes] = []  # Collect deterministic labelled sections only for unusual multi-log backends.
    for index, source in enumerate(sources):  # Preserve every unique source in sorted order without omission.
        header = f"===== SOURCE {index:03d} {source.name} =====\n".encode("utf-8")  # Delimit sources using only a portable basename rather than an absolute runner path.
        chunks.extend((header, source.read_bytes(), b"\n"))  # Retain the complete original bytes between explicit deterministic boundaries.
    return b"".join(chunks)  # Return one auditable combined payload for the registered ladder level.


def _persist_level_log(case_dir: Path, level_index: int, sources: list[Path], *, required: bool) -> list[dict[str, Any]]:  # Copy discovered native logs into the formal case cache and authenticate them.
    unique = sorted({source.resolve() for source in sources if source.is_file()}, key=lambda path: str(path))  # Deduplicate explicit and discovered paths while retaining only existing regular files.
    target = case_dir / SOLVER_LOG_DIRECTORY / f"ref_l{level_index:02d}.log"  # Resolve the sole portable log filename for this registered level.
    if not unique:  # Handle pre-native failures separately from successful solve evidence.
        if not required and target.is_file():  # Recover an already copied log only while preserving a failure that occurred after native success.
            return [_evidence_record(case_dir, target)]  # Authenticate the retained exact bytes rather than discarding them.
        if required:  # A successful solve must always leave independently verifiable native evidence.
            raise ReferenceBuildError(f"reference level {level_index} produced no native solver log")  # Refuse a success record backed only by in-memory values.
        return []  # Represent a genuine pre-native failure without fabricating a log.
    target.parent.mkdir(parents=True, exist_ok=True)  # Create only the fixed portable evidence directory for this case.
    payload = _combine_native_files(unique)  # Materialize the exact single or deterministically combined native evidence before publication.
    if required and not payload:  # A successful native solve must have at least one retained log byte.
        raise ReferenceBuildError(f"reference level {level_index} produced an empty native solver log")  # Refuse a vacuous path-and-hash success receipt.
    target.write_bytes(payload)  # Publish directly to the registered filename so an interruption leaves a recognizable resumable partial path.
    return [_evidence_record(case_dir, target, source_count=len(unique))]  # Return a portable full-SHA receipt for ledger inclusion.


def _persist_failure_input(case_dir: Path, level_index: int, sources: list[Path]) -> list[dict[str, Any]]:  # Retain any already-generated CalculiX input when the native solve fails.
    unique = sorted({source.resolve() for source in sources if source.is_file()}, key=lambda path: str(path))  # Deduplicate candidate input paths and ignore unavailable pre-native products.
    if not unique:  # Permit Gmsh, launch, or pre-deck failures with no input artifact.
        return []  # Represent unavailable input explicitly without a placeholder deck.
    target = case_dir / SOLVER_INPUT_DIRECTORY / f"ref_l{level_index:02d}.inp"  # Resolve the sole failure-only portable deck filename.
    target.parent.mkdir(parents=True, exist_ok=True)  # Create only the fixed failure-input evidence directory.
    target.write_bytes(_combine_native_files(unique))  # Preserve every already-generated input byte under one deterministic level path.
    return [_evidence_record(case_dir, target, source_count=len(unique))]  # Return a portable full-SHA receipt for the failure ledger.


def _verify_evidence_record(case_dir: Path, record: Any, expected_path: str) -> dict[str, Any]:  # Authenticate one ledger-declared native artifact without permitting path traversal or symlinks.
    if not isinstance(record, dict) or record.get("path") != expected_path:  # Require a named mapping and the exact registered relative location.
        raise ReferenceBuildError(f"native evidence path must equal {expected_path}")  # Reject absolute, renamed, nested, or cross-level substitutions.
    claimed_sha = record.get("sha256")  # Read the complete persisted digest before accessing the file.
    claimed_size = record.get("size_bytes")  # Read the persisted byte count independently from filesystem metadata.
    if not isinstance(claimed_sha, str) or len(claimed_sha) != 64 or any(character not in "0123456789abcdef" for character in claimed_sha):  # Require an unabbreviated lowercase SHA-256 value.
        raise ReferenceBuildError(f"native evidence SHA-256 is invalid for {expected_path}")  # Reject missing, uppercase, truncated, and nonhex identities.
    if not isinstance(claimed_size, int) or claimed_size < 0:  # Require a concrete nonnegative exact byte length.
        raise ReferenceBuildError(f"native evidence size is invalid for {expected_path}")  # Reject strings, nulls, and negative sizes.
    target = (case_dir / str(record["path"])).resolve()  # Resolve traversal and symlinks before opening any declared evidence.
    try:  # Convert paths outside the exact case cache into one protocol-specific failure.
        target.relative_to(case_dir.resolve())  # Prove the resolved artifact remains below this manifest case directory.
    except ValueError as exc:  # Catch parent traversal and intermediate symlink escape.
        raise ReferenceBuildError(f"native evidence escapes reference case directory: {expected_path}") from exc  # Stop before reading cross-case or external content.
    declared_path = case_dir / str(record["path"])  # Recover the unresolved declared path for final symlink rejection.
    if declared_path.is_symlink() or not target.is_file():  # Require one existing regular non-symlink artifact.
        raise ReferenceBuildError(f"native evidence file is missing or symlinked: {expected_path}")  # Reject mutable indirection and absent evidence.
    observed_sha = _file_sha256(target)  # Recompute all bytes independently from the authenticated ledger.
    observed_size = int(target.stat().st_size)  # Recompute the exact file length independently.
    if observed_sha != claimed_sha or observed_size != claimed_size:  # Require both collision-resistant identity and length to agree.
        raise ReferenceBuildError(f"native evidence integrity mismatch: {expected_path}")  # Detect truncation, mutation, or file replacement before cache use.
    return {"path": expected_path, "sha256": observed_sha, "size_bytes": observed_size, "passed": True}  # Return a compact independently recomputed verification receipt.


def _seal(payload: dict[str, Any]) -> dict[str, Any]:  # Attach an integrity digest that excludes only its own field.
    body = {key: value for key, value in payload.items() if key != "integrity_sha256"}  # Remove any stale seal before recomputing integrity.
    body["integrity_sha256"] = _payload_sha256(body)  # Bind all remaining ledger content to one digest.
    return body  # Return a fresh sealed mapping without mutating caller-owned data.


def _write_sealed(path: Path, payload: dict[str, Any]) -> dict[str, Any]:  # Persist one sealed JSON artifact through an atomic same-directory replacement.
    sealed = _seal(payload)  # Compute integrity after the latest ledger mutation.
    temporary = path.with_suffix(path.suffix + ".tmp")  # Keep an interrupted write separate from the last complete artifact.
    temporary.write_text(json.dumps(sealed, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")  # Write human-readable deterministic finite JSON.
    temporary.replace(path)  # Atomically publish the complete artifact on the same filesystem.
    return sealed  # Return the exact sealed content that reached disk.


def _read_sealed(path: Path) -> dict[str, Any]:  # Load JSON while rejecting truncation, mutation, and non-object payloads.
    try:  # Convert filesystem and JSON errors into one protocol-specific failure type.
        payload = json.loads(path.read_text(encoding="utf-8"))  # Decode the complete persisted artifact.
    except (OSError, json.JSONDecodeError) as exc:  # Catch missing, unreadable, and malformed cache artifacts.
        raise ReferenceBuildError(f"cannot read sealed reference artifact {path}: {exc}") from exc  # Preserve the original cause without accepting partial data.
    if not isinstance(payload, dict):  # Require the object shape used by the integrity contract.
        raise ReferenceBuildError(f"sealed reference artifact {path} is not a JSON object")  # Reject arrays and scalar substitutions.
    observed = payload.get("integrity_sha256")  # Read the digest claimed by the artifact.
    expected = _payload_sha256({key: value for key, value in payload.items() if key != "integrity_sha256"})  # Recompute integrity from every substantive field.
    if observed != expected:  # Reject even semantically small cache edits before loading a reference.
        raise ReferenceBuildError(f"integrity mismatch for reference artifact {path}")  # Surface the exact corrupted path.
    return payload  # Return only a successfully authenticated JSON mapping.


def _config_snapshot(config: ReferenceScheduleConfig) -> dict[str, Any]:  # Convert the frozen schedule into its canonical persisted representation.
    return _json_ready(asdict(config))  # Preserve every acceptance and refinement parameter.


def _effective_config(config: ReferenceScheduleConfig, *, allow_unqualified: bool, expedited_levels: int | None) -> tuple[ReferenceScheduleConfig, dict[str, Any] | None]:  # Derive an explicitly authorized prefix without altering the registered default object.
    if expedited_levels is not None and not allow_unqualified:  # Forbid a shortened ladder unless the caller opts into the named nonblocking amendment.
        raise ValueError("expedited_levels requires allow_unqualified=True")  # Prevent an implicit protocol schedule change.
    if not allow_unqualified:  # Preserve the exact original strict behavior and complete caller-supplied schedule by default.
        return config, None  # Return the unchanged registered configuration with no amendment metadata.
    if config != DEFAULT_REFERENCE_CONFIG:  # Restrict operational fallback to the exact preregistered six-level schedule rather than arbitrary tuning.
        raise ValueError("allow_unqualified requires DEFAULT_REFERENCE_CONFIG")  # Prevent post-hoc custom reference ladders from using the user authorization.
    selected_levels = len(config.background_scales) if expedited_levels is None else int(expedited_levels)  # Use all six levels unless the caller explicitly selects an expedited prefix.
    if selected_levels < 2 or selected_levels > len(config.background_scales):  # Require enough consecutive successful levels for an A/B pair and never extend the frozen ladder.
        raise ValueError(f"expedited_levels must be between 2 and {len(config.background_scales)}")  # Report the exact fixed operational range.
    effective = ReferenceScheduleConfig(agreement_rtol=config.agreement_rtol, background_scales=tuple(config.background_scales[:selected_levels]), local_floor_scales=tuple(config.local_floor_scales[:selected_levels]), zero_guard=config.zero_guard)  # Truncate only the pre-existing scale prefix while preserving the original one-half-percent gate.
    amendment = {"authorization": UNQUALIFIED_AUTHORIZATION, "reason": UNQUALIFIED_AUTHORIZATION, "allow_unqualified": True, "expedited_levels": selected_levels, "original_schedule_levels": len(config.background_scales), "original_config": _config_snapshot(config), "original_config_sha256": _payload_sha256(_config_snapshot(config)), "effective_config": _config_snapshot(effective), "effective_config_sha256": _payload_sha256(_config_snapshot(effective)), "agreement_rtol_unchanged": True, "native_failure_fallback_allowed": False}  # Preserve both original and effective schedules, exact identities, unchanged qualification math, and the strict native-failure boundary.
    return effective, amendment  # Return the explicit operational configuration and immutable amendment receipt.


def _problem_snapshot(problem: Problem) -> dict[str, Any]:  # Capture all inputs that define the manifest-reconstructed finite-element problem.
    return {"instance_id": problem.instance_id, "name": problem.name, "dim": int(problem.dim), "params": _json_ready(problem.params), "h0": float(problem.h0), "h_ref": float(problem.h_ref), "h_min": float(problem.h_min), "bbox": [float(value) for value in problem.bbox], "material": _json_ready(asdict(problem.material))}  # Bind geometry, load, mesh scales, domain, and material without serializing callables.


def _problem_signature(problem: Problem) -> str:  # Derive the exact reference-cache identity for one reconstructed case.
    return _payload_sha256(_problem_snapshot(problem))  # Hash the complete JSON-safe problem snapshot.


def _validated_case_id(case_id: str) -> str:  # Prevent path traversal and accidental nested reference roots.
    candidate = str(case_id).strip()  # Normalize harmless surrounding whitespace before validation.
    if not candidate or candidate in {".", ".."} or Path(candidate).name != candidate:  # Reject empty, parent, and multi-component identifiers.
        raise ValueError("case_id must be one nonempty path component")  # Keep every case cache below the declared reference root.
    return candidate  # Return the safe manifest identifier unchanged.


def reference_case_dir(reference_root: Path | str, case_id: str) -> Path:  # Resolve the frozen per-case cache layout shared by training and benchmark drivers.
    return Path(reference_root) / _validated_case_id(case_id)  # Keep one auditable directory per manifest case identifier.


def _scaled_reference_size_fn(problem: Problem, background_scale: float, local_floor_scale: float) -> Callable[[float, float, float], float]:  # Build one analytical refinement field without accepting a method mesh.
    base_fn = reference_size_fn(problem)  # Reuse the exact current strong graded-reference shape for Reference A.
    base_background = float(problem.h_ref)  # Read the existing strong-reference far-field target.
    base_local_floor = float(reference_floor(problem))  # Read the existing strong-reference local minimum target.
    target_background = base_background * float(background_scale)  # Apply the preregistered background refinement for this level.
    target_local_floor = base_local_floor * float(local_floor_scale)  # Apply the preregistered local refinement for this level.
    span = max(base_background - base_local_floor, np.finfo(float).eps)  # Protect normalization for a degenerate uniform reference scale.

    def size_fn(x: float, y: float, z: float = 0.0) -> float:  # Map the existing analytical grading shape onto this level's two frozen endpoints.
        normalized = min(max((float(base_fn(x, y, z)) - base_local_floor) / span, 0.0), 1.0)  # Preserve grading order while clipping roundoff beyond its declared range.
        return float(target_local_floor + normalized * (target_background - target_local_floor))  # Return a scale that is independent of every compared method's mesh or error.

    return size_fn  # Return the deterministic Gmsh size callback for this reference level.


def _reference_from_level(level: dict[str, Any]) -> Reference:  # Convert one authenticated successful ledger level into the shared experiment contract.
    return Reference(U_total=float(level["U_total"]), qoi=float(level["qoi"]), n_equations=int(level["n_equations"]), n_elems=int(level["n_elems"]), h_ref=float(level["h_background_target"]))  # Preserve final B values and its actual registered background target.


def _relative_difference(coarser: float, finer: float, zero_guard: float) -> float:  # Compute the protocol's finer-denominator relative difference safely.
    denominator = abs(float(finer))  # Use the absolute final-reference value exactly as the frozen equations require.
    if not math.isfinite(denominator) or denominator <= float(zero_guard):  # Reject zero or invalid denominators as non-evidence.
        return math.inf  # Force the convergence gate to fail without inventing a finite ratio.
    return float(abs(float(finer) - float(coarser)) / denominator)  # Return the unrounded relative change used by the hard gate.


def _runner_case_guard(problem: Problem, runner: Any) -> None:  # Prevent accidental reference injection into a runner for another manifest case.
    runner_problem = getattr(runner, "problem", None)  # Read the optional FemRunner problem binding without constraining synthetic test doubles.
    if runner_problem is not None and _problem_signature(runner_problem) != _problem_signature(problem):  # Compare full geometry, load, material, and scale identities.
        raise ValueError("runner problem does not match the requested reference case")  # Reject cross-case cache contamination before a solve.
    if not callable(getattr(runner, "_solve", None)):  # Require the non-counting solve hook used by the existing FemRunner reference path.
        raise TypeError("runner must provide FemRunner-compatible _solve")  # Keep online solve accounting separate from reference construction.


def bind_reference_b(runner: Any, reference: Reference, problem: Problem | None = None) -> Reference:  # Inject one verified Reference B into a shared method runner.
    if problem is not None:  # Apply the stronger case-identity guard whenever the caller supplies its reconstructed problem.
        _runner_case_guard(problem, runner)  # Refuse to bind a reference across manifest cases.
    runner.reference = reference  # Reuse the existing FemRunner energy and QoI error implementation unchanged.
    return reference  # Return the bound object for fluent benchmark setup.


def _discover_solver_logs(runner: Any, counter_before: int, stage: str, record: Any) -> list[Path]:  # Collect native solver-log sources without persisting runner-specific absolute paths.
    explicit = getattr(record, "extra", {}).get("solver_logs", []) if isinstance(getattr(record, "extra", {}), dict) else []  # Prefer backend-provided log paths when available.
    logs = [Path(item).resolve() for item in explicit]  # Normalize explicit paths only for immediate case-local copying.
    workdir = getattr(runner, "workdir", None)  # Read the standard FemRunner work directory when present.
    if workdir is not None:  # Reconstruct the private non-counting job path used by the current FemRunner implementation.
        jobname = f"reference_{counter_before:03d}_{stage}"[:60].replace("/", "_")  # Match FemRunner's deterministic solver job naming exactly.
        jobdir = Path(workdir) / "solves" / jobname  # Resolve the corresponding solver-output directory.
        logs.extend(path.resolve() for path in sorted(jobdir.glob("*.log")))  # Discover every surviving combined native log for immediate copying.
    return sorted(set(logs), key=lambda path: str(path))  # Deduplicate explicit and discovered paths while preserving deterministic order.


def _failed_native_sources(runner: Any, level_index: int, exc: Exception) -> tuple[list[Path], list[Path]]:  # Discover already-produced native logs and decks after an exception without assuming one backend layout.
    stage = f"ref_l{level_index:02d}"  # Reconstruct the exact stable level label used by the failed solve.
    counter = int(getattr(runner, "_counter", 0))  # Recover the non-counting reference job index used by FemRunner.
    expected_job = f"reference_{counter:03d}_{stage}"[:60].replace("/", "_")  # Match FemRunner's deterministic private job directory exactly.
    runner_workdir = getattr(runner, "workdir", None)  # Read the optional runner boundary without defaulting to the process working directory.
    expected_dir = Path(runner_workdir) / "solves" / expected_job if runner_workdir is not None else None  # Resolve the ordinary runner location only when explicitly available.
    exception_workdir = getattr(exc, "workdir", None)  # Read the typed CalculiX failure's exact native work directory when available.
    directories = {path for path in (expected_dir, Path(exception_workdir) if exception_workdir is not None else None) if path is not None}  # Build a finite explicit discovery set with no ambient-directory fallback.
    log_sources = [Path(getattr(exc, "log_path"))] if getattr(exc, "log_path", None) is not None else []  # Retain the typed backend log path before directory discovery.
    log_sources.extend(path for directory in directories for path in sorted(directory.glob("*.log")))  # Discover any persisted combined logs even for validation or untyped backend failures.
    input_sources = [path for directory in directories for path in sorted(directory.glob("*.inp"))]  # Discover every already-generated CalculiX deck for failure reproduction.
    return log_sources, input_sources  # Return raw sources for immediate portable cache copying.


def _solve_level(problem: Problem, runner: Any, mesh: Mesh, case_dir: Path, level_index: int, background_scale: float, local_floor_scale: float) -> tuple[Any, Any, list[dict[str, Any]]]:  # Execute one reference solve and retain portable evidence outside every method's online counter.
    stage = f"ref_l{level_index:02d}"  # Give every preregistered level a stable solver-job label.
    counter_before = int(getattr(runner, "_counter", 0))  # Capture the non-counting FemRunner job index used for log discovery.
    post, record = runner._solve(mesh, method="reference", stage=stage, count=False, extra={"protocol_id": PROTOCOL_ID, "reference_level": level_index, "background_scale": float(background_scale), "local_floor_scale": float(local_floor_scale), "mesh_source": "preregistered_analytical_reference_field"})  # Reuse the native solve/post path while explicitly excluding method-mesh inputs and online solve counts.
    log_sources = _discover_solver_logs(runner, counter_before, stage, record)  # Locate runner-specific sources immediately after a successful native solve.
    logs = _persist_level_log(case_dir, level_index, log_sources, required=True)  # Copy exact log bytes into the portable per-case cache and compute full identities.
    return post, record, logs  # Return numerical outputs and portable authenticated evidence without mutating runner.reference.


def _level_record(problem: Problem, runner: Any, case_dir: Path, level_index: int, config: ReferenceScheduleConfig, mesh_factory: Callable[..., Mesh]) -> dict[str, Any]:  # Mesh and solve one independently generated preregistered reference level with portable native evidence.
    background_scale = float(config.background_scales[level_index])  # Read the frozen level-specific far-field multiplier.
    local_floor_scale = float(config.local_floor_scales[level_index])  # Read the frozen level-specific local-minimum multiplier.
    background_target = float(problem.h_ref) * background_scale  # Convert the registered multiplier to the physical background size.
    local_floor_target = float(reference_floor(problem)) * local_floor_scale  # Convert the registered multiplier to the physical local floor.
    field = _scaled_reference_size_fn(problem, background_scale, local_floor_scale)  # Construct the analytical field without any compared method's mesh.
    model_name = f"reference_l{level_index:02d}_{problem.instance_id[-8:]}"  # Give deterministic Gmsh models distinct audit-friendly names.
    build_started = time.perf_counter()  # Start total per-level timing before mesh generation.
    mesh_started = time.perf_counter()  # Start the separately reported meshing duration.
    mesh = mesh_factory(problem, field, model_name=model_name, h_floor=local_floor_target)  # Generate this level directly from geometry and its preregistered size field.
    mesh_wall_s = time.perf_counter() - mesh_started  # Stop the monotonic meshing timer immediately after generation.
    mesh_sizes = np.asarray(mesh.cell_sizes, dtype=float)  # Materialize realized element sizes once for finite validation and the ledger.
    post, solve_record, solver_logs = _solve_level(problem, runner, mesh, case_dir, level_index, background_scale, local_floor_scale)  # Execute the independent native solve and persist its portable authenticated log.
    total_wall_s = time.perf_counter() - build_started  # Record the complete mesh-plus-solve duration for this level.
    U_total = float(post.U_total)  # Normalize total strain energy to a portable scalar.
    qoi = float(post.qoi)  # Normalize the displacement QoI to a portable scalar.
    n_equations = int(solve_record.n_equations)  # Read the actual solver equation count rather than estimating it from mesh size.
    solver_wall_s = float(solve_record.wall_s)  # Preserve the backend-reported solver wall time independently from orchestration overhead.
    if mesh.n_cells <= 0 or mesh.n_nodes <= 0 or mesh_sizes.size == 0:  # Reject empty geometry outputs before writing a success record.
        raise ReferenceBuildError(f"reference level {level_index} produced an empty mesh")  # Preserve an explicit numerical failure instead of fabricating counts.
    if not np.all(np.isfinite(mesh_sizes)) or float(mesh_sizes.min()) <= 0.0:  # Reject invalid realized element scales.
        raise ReferenceBuildError(f"reference level {level_index} produced invalid mesh sizes")  # Prevent a malformed mesh from entering convergence checks.
    if not math.isfinite(U_total) or U_total <= config.zero_guard:  # Require positive finite strain energy for the relative energy gate.
        raise ReferenceBuildError(f"reference level {level_index} produced invalid U_total={U_total}")  # Preserve the exact invalid solver value in the failure message.
    if not math.isfinite(qoi) or abs(qoi) <= config.zero_guard:  # Require a finite nonzero QoI for its relative convergence gate.
        raise ReferenceBuildError(f"reference level {level_index} produced invalid qoi={qoi}")  # Preserve the exact invalid post-processing value.
    if n_equations <= 0:  # Require the actual equation parser to succeed for a valid reference solve.
        raise ReferenceBuildError(f"reference level {level_index} has invalid n_equations={n_equations}")  # Refuse to replace missing counts with an estimate.
    if not math.isfinite(solver_wall_s) or solver_wall_s < 0.0:  # Require a valid nonnegative backend duration.
        raise ReferenceBuildError(f"reference level {level_index} has invalid solver wall time")  # Keep time accounting honest and JSON-safe.
    return {"level_index": level_index, "role": "reference_A_initial" if level_index == 0 else "reference_B_candidate", "status": "success", "background_scale": background_scale, "local_floor_scale": local_floor_scale, "h_background_target": background_target, "h_local_floor_target": local_floor_target, "h_realized_min": float(mesh_sizes.min()), "h_realized_max": float(mesh_sizes.max()), "U_total": U_total, "qoi": qoi, "n_equations": n_equations, "n_nodes": int(mesh.n_nodes), "n_elems": int(mesh.n_cells), "mesh_sha": _mesh_sha256(mesh), "mesh_source": "preregistered_analytical_reference_field", "mesh_wall_s": float(mesh_wall_s), "solver_wall_s": solver_wall_s, "total_wall_s": float(total_wall_s), "solver_logs": solver_logs, "solver_inputs": []}  # Return all physical sizes, outputs, complete identities, portable logs, and explicit success-only input omission required by the protocol.


def _new_ledger(problem: Problem, case_id: str, config: ReferenceScheduleConfig, execution_amendment: dict[str, Any] | None) -> dict[str, Any]:  # Initialize strict or explicitly amended intent before any potentially failing native operation.
    config_snapshot = _config_snapshot(config)  # Materialize the exact preregistered schedule once.
    return {"schema": REFERENCE_SCHEMA, "protocol_id": PROTOCOL_ID, "case_id": case_id, "problem_signature": _problem_signature(problem), "problem": _problem_snapshot(problem), "config_signature": _payload_sha256(config_snapshot), "config": config_snapshot, "execution_amendment": execution_amendment, "status": "building", "qualification": None, "authorization": execution_amendment.get("authorization") if execution_amendment is not None else None, "created_utc": datetime.now(timezone.utc).isoformat(), "completed_utc": None, "source_contract": {"reference_A": "existing_strong_graded_reference", "reference_B_and_upgrades": "same_analytical_grading_shape_with_preregistered_finer_background_and_local_floor", "method_mesh_input_allowed": False, "online_solve_counted": False}, "levels": [], "upgrade_history": [], "final_pair": None, "reference_b": None, "failure": None, "reference_b_artifact": REFERENCE_B_FILENAME}  # Preserve exact effective/original amendment identity, qualification state, independence, and empty evidence before the first mesh.


def _write_reference_artifact(case_dir: Path, ledger: dict[str, Any], reference_b: Reference) -> dict[str, Any]:  # Persist the small shared-reference object independently from the full audit ledger.
    payload = {"schema": REFERENCE_ARTIFACT_SCHEMA, "protocol_id": PROTOCOL_ID, "case_id": ledger["case_id"], "problem_signature": ledger["problem_signature"], "config_signature": ledger["config_signature"], "status": ledger["status"], "qualification": ledger["qualification"], "authorization": ledger["authorization"], "execution_amendment": ledger["execution_amendment"], "reference": _json_ready(asdict(reference_b)), "ledger_filename": LEDGER_FILENAME}  # Bind final numbers to case, effective schedule, explicit qualification, and authorization without calling unqualified evidence converged.
    return _write_sealed(case_dir / REFERENCE_B_FILENAME, payload)  # Publish an integrity-protected common Reference B cache.


def _cached_outcome(case_dir: Path, ledger: dict[str, Any], reference_b: Reference) -> ReferenceBuildOutcome:  # Reconstruct both accepted reference objects from a verified complete ledger.
    final_pair = ledger["final_pair"]  # Read the authenticated final consecutive ladder indices.
    a_level = int(final_pair["a_level"])  # Normalize the promoted Reference A index.
    b_level = int(final_pair["b_level"])  # Normalize the accepted Reference B index.
    levels = ledger["levels"]  # Read the complete authenticated level sequence.
    return ReferenceBuildOutcome(reference_a=_reference_from_level(levels[a_level]), reference_b=reference_b, a_level=a_level, b_level=b_level, ledger_path=case_dir / LEDGER_FILENAME, from_cache=True, qualification=bool(ledger["qualification"]), authorization=ledger.get("authorization"), execution_amendment=ledger.get("execution_amendment"))  # Return a no-solve outcome with identical values and explicit qualification/amendment semantics.


def _validate_building_ledger(ledger: dict[str, Any], problem: Problem, case_id: str, case_dir: Path, config: ReferenceScheduleConfig, execution_amendment: dict[str, Any] | None) -> int:  # Authenticate an interrupted ladder, amendment, and portable logs before resuming its next unsolved level.
    if ledger.get("schema") != REFERENCE_SCHEMA or ledger.get("protocol_id") != PROTOCOL_ID or ledger.get("status") != "building":  # Require the exact live-build protocol state rather than a terminal failure masquerading as resumable work.
        raise ReferenceBuildError(f"existing reference ledger is not resumable: {ledger.get('status')}")  # Preserve failed and exhausted ladders for review without overwriting them.
    if ledger.get("case_id") != case_id or ledger.get("problem_signature") != _problem_signature(problem):  # Bind the partial ladder to this exact manifest case and reconstructed physical problem.
        raise ReferenceBuildError("building reference ledger problem or case identity mismatch")  # Prevent cross-case continuation before any new mesh is generated.
    snapshot = _config_snapshot(config)  # Reconstruct every registered reference schedule and acceptance setting.
    if ledger.get("config") != snapshot or ledger.get("config_signature") != _payload_sha256(snapshot):  # Require readable and hashed schedule identity together.
        raise ReferenceBuildError("building reference ledger configuration mismatch")  # Refuse to continue an interrupted ladder under changed refinement choices.
    if ledger.get("execution_amendment") != execution_amendment or ledger.get("authorization") != (execution_amendment.get("authorization") if execution_amendment is not None else None):  # Require exact strict or expedited authorization identity across resume.
        raise ReferenceBuildError("building reference ledger execution amendment mismatch")  # Prevent switching qualification policy or expedited depth after observing partial results.
    levels = ledger.get("levels")  # Read the completed append-only prefix once for structural validation.
    upgrades = ledger.get("upgrade_history")  # Read the completed failed-pair decisions that justify every additional level.
    if not isinstance(levels, list) or not isinstance(upgrades, list) or len(levels) >= len(config.background_scales):  # Require a proper nonterminal prefix with at least one registered level remaining.
        raise ReferenceBuildError("building reference ledger has an invalid or exhausted level prefix")  # Stop rather than silently append beyond the preregistered schedule.
    if len(upgrades) != max(0, len(levels) - 1):  # Require one failed comparison for every completed finer candidate in a still-building ladder.
        raise ReferenceBuildError("building reference ledger upgrade history is incomplete")  # Reject an ambiguous promotion history after interruption.
    for level_index, level in enumerate(levels):  # Validate every checkpointed successful level before trusting it as the next Reference A.
        expected_background = float(config.background_scales[level_index])  # Recover the exact registered background multiplier for this prefix position.
        expected_floor = float(config.local_floor_scales[level_index])  # Recover the exact registered local-floor multiplier for this prefix position.
        if not isinstance(level, dict) or level.get("status") != "success" or int(level.get("level_index", -1)) != level_index:  # Require ordered successful records without gaps or failed entries.
            raise ReferenceBuildError(f"building reference ledger level {level_index} is invalid")  # Identify the first unusable checkpoint directly.
        if float(level.get("background_scale", math.nan)) != expected_background or float(level.get("local_floor_scale", math.nan)) != expected_floor:  # Require exact schedule identity at each completed level.
            raise ReferenceBuildError(f"building reference ledger level {level_index} scale mismatch")  # Prevent continuation after a post-hoc field change.
        if not isinstance(level.get("mesh_sha"), str) or len(level["mesh_sha"]) != 64:  # Require full mesh provenance for every reusable completed level.
            raise ReferenceBuildError(f"building reference ledger level {level_index} mesh SHA-256 is invalid")  # Refuse a truncated or absent checkpoint identity.
        logs = level.get("solver_logs")  # Read the mandatory portable native evidence for this reusable successful level.
        if not isinstance(logs, list) or len(logs) != 1 or not isinstance(logs[0], dict) or not isinstance(logs[0].get("size_bytes"), int) or logs[0]["size_bytes"] <= 0:  # Require exactly one nonempty registered level-scoped combined log receipt.
            raise ReferenceBuildError(f"building reference ledger level {level_index} solver log evidence is incomplete")  # Refuse an in-memory-only or ambiguous completed solve.
        _verify_evidence_record(case_dir, logs[0], f"{SOLVER_LOG_DIRECTORY}/ref_l{level_index:02d}.log")  # Recompute exact bytes and reject path traversal before trusting the checkpoint.
        if any(not math.isfinite(float(level.get(name, math.nan))) for name in ("U_total", "qoi")):  # Require finite physical outputs before using this level as a promoted Reference A.
            raise ReferenceBuildError(f"building reference ledger level {level_index} result is invalid")  # Stop before forming a comparison against corrupt numerical evidence.
        if level_index == 0:  # The initial strong reference has no preceding convergence comparison.
            continue  # Move to the first candidate or finish validating a one-level prefix.
        previous = levels[level_index - 1]  # Recover the immediately coarser successful level used by the registered promotion decision.
        energy_difference = _relative_difference(float(previous["U_total"]), float(level["U_total"]), config.zero_guard)  # Recompute the failed energy agreement exactly.
        qoi_difference = _relative_difference(float(previous["qoi"]), float(level["qoi"]), config.zero_guard)  # Recompute the failed QoI agreement exactly.
        upgrade = upgrades[level_index - 1]  # Read the checkpointed decision paired with this candidate.
        comparison = level.get("comparison_to_previous")  # Read the redundant candidate-local copy used for human audit.
        if not isinstance(upgrade, dict) or not isinstance(comparison, dict):  # Require both audit copies to be named mappings before reading their fields.
            raise ReferenceBuildError(f"building reference ledger upgrade {level_index - 1} is malformed")  # Reject a scalar or missing promotion record explicitly.
        identities_match = int(upgrade.get("a_level", -1)) == level_index - 1 and int(upgrade.get("b_level", -1)) == level_index and upgrade.get("reference_A_mesh_sha") == previous["mesh_sha"] and upgrade.get("reference_B_mesh_sha") == level["mesh_sha"]  # Bind the promotion to the exact consecutive mesh pair.
        decisions_match = comparison.get("passed") is False and upgrade.get("passed") is False and upgrade.get("decision") == "promote_B_to_A_and_continue_preregistered_refinement"  # Require a failed dual gate and the sole permitted reference-only response.
        values_match = math.isclose(float(upgrade.get("energy_relative_difference", math.nan)), energy_difference, rel_tol=1.0e-12, abs_tol=1.0e-15) and math.isclose(float(upgrade.get("qoi_relative_difference", math.nan)), qoi_difference, rel_tol=1.0e-12, abs_tol=1.0e-15)  # Recompute both stored ratios independently.
        if not identities_match or not decisions_match or not values_match or (energy_difference <= config.agreement_rtol and qoi_difference <= config.agreement_rtol):  # Reject an inconsistent or already-converged prefix that should have terminated.
            raise ReferenceBuildError(f"building reference ledger upgrade {level_index - 1} is inconsistent")  # Identify the exact unsafe continuation point.
    return len(levels)  # Resume only at the first level absent from the authenticated completed prefix.


def ensure_reference_pair(problem: Problem, runner: FemRunner | Any, reference_root: Path | str, *, case_id: str | None = None, config: ReferenceScheduleConfig = DEFAULT_REFERENCE_CONFIG, mesh_factory: Callable[..., Mesh] = generate_mesh, allow_unqualified: bool = False, expedited_levels: int | None = None) -> ReferenceBuildOutcome:  # Build or explicitly load a qualified or user-authorized operational A/B pair.
    selected_case_id = _validated_case_id(case_id or problem.instance_id)  # Prefer the manifest case ID while supporting a stable problem-derived fallback.
    registered_config = config  # Preserve the caller's original exact schedule for explicit expedited verification and loading.
    config, execution_amendment = _effective_config(registered_config, allow_unqualified=bool(allow_unqualified), expedited_levels=expedited_levels)  # Derive only an explicitly authorized prefix while keeping the one-half-percent gate unchanged.
    _runner_case_guard(problem, runner)  # Reject a runner bound to another reconstructed case before cache access.
    case_dir = reference_case_dir(reference_root, selected_case_id)  # Resolve the shared per-case cache directory.
    ledger_path = case_dir / LEDGER_FILENAME  # Resolve the authoritative audit ledger path.
    start_level = 0  # Start a new ladder at Reference A unless an authenticated interrupted prefix exists.
    if ledger_path.exists():  # Reuse a complete cache or safely continue only an append-only building checkpoint.
        ledger = _read_sealed(ledger_path)  # Authenticate every persisted byte before interpreting cache state.
        if ledger.get("status") in {"complete", "complete_unqualified"}:  # Reuse a terminal cache only through the caller's explicit qualification policy.
            reference_b = load_reference_b(reference_root, case_id=selected_case_id, problem=problem, runner=runner, config=registered_config, verify=True, mesh_factory=mesh_factory, regenerate_meshes=False, allow_unqualified=allow_unqualified, expedited_levels=expedited_levels)  # Authenticate schedule, amendment, qualification, and exact native evidence without a solver call.
            return _cached_outcome(case_dir, ledger, reference_b)  # Return the accepted cached pair with explicit provenance.
        start_level = _validate_building_ledger(ledger, problem, selected_case_id, case_dir, config, execution_amendment)  # Resume only after authenticating the exact amendment, prefix, and portable logs.
    else:  # Create a new append-only ladder only when no prior evidence exists.
        case_dir.mkdir(parents=True, exist_ok=True)  # Create the exact per-case directory below the declared reference root.
        ledger = _new_ledger(problem, selected_case_id, config, execution_amendment)  # Create the pre-solve audit state with strict or explicitly amended choices.
        ledger = _write_sealed(ledger_path, ledger)  # Persist intent before Gmsh or CalculiX can fail.
    for level_index in range(start_level, len(config.background_scales)):  # Execute only the missing predeclared suffix and stop on the dual convergence gate.
        level_started = time.perf_counter()  # Start an outer timer that also covers exceptions before a normal level record exists.
        try:  # Preserve every mesh or solver failure in the ledger before propagating it.
            level = _level_record(problem, runner, case_dir, level_index, config, mesh_factory)  # Build and solve one method-independent level while retaining portable native evidence.
        except Exception as exc:  # Catch native, meshing, post-processing, and validation failures uniformly.
            log_sources, input_sources = _failed_native_sources(runner, level_index, exc)  # Discover every already-produced native failure artifact before external runner cleanup.
            solver_logs = _persist_level_log(case_dir, level_index, log_sources, required=False)  # Preserve any available combined log under the fixed portable path.
            solver_inputs = _persist_failure_input(case_dir, level_index, input_sources)  # Preserve any available CalculiX input deck for deterministic failure reproduction.
            failed_level = {"level_index": level_index, "role": "reference_A_initial" if level_index == 0 else "reference_B_candidate", "status": "failed", "background_scale": float(config.background_scales[level_index]), "local_floor_scale": float(config.local_floor_scales[level_index]), "h_background_target": float(problem.h_ref) * float(config.background_scales[level_index]), "h_local_floor_target": float(reference_floor(problem)) * float(config.local_floor_scales[level_index]), "h_realized_min": None, "h_realized_max": None, "U_total": None, "qoi": None, "n_equations": None, "n_nodes": None, "n_elems": None, "mesh_sha": None, "mesh_source": "preregistered_analytical_reference_field", "mesh_wall_s": None, "solver_wall_s": float(getattr(exc, "wall_s")) if getattr(exc, "wall_s", None) is not None else None, "total_wall_s": float(time.perf_counter() - level_started), "solver_logs": solver_logs, "solver_inputs": solver_inputs, "native_returncode": getattr(exc, "returncode", None), "error_type": type(exc).__name__, "error": str(exc)}  # Retain targets, explicit unavailable values, typed backend metadata, portable full-hash evidence, and exact exception without fake numerical outputs.
            ledger["levels"].append(failed_level)  # Keep the failed case and level in the immutable experiment history.
            ledger["status"] = "numerical_failure"  # Mark the cache unusable for downstream error calculations.
            ledger["completed_utc"] = datetime.now(timezone.utc).isoformat()  # Timestamp terminal failure explicitly.
            ledger["failure"] = {"kind": "reference_level_failure", "level_index": level_index, "error_type": type(exc).__name__, "error": str(exc)}  # Publish a compact failure ledger entry for aggregation.
            _write_sealed(ledger_path, ledger)  # Persist the failure before raising to the campaign driver.
            raise ReferenceBuildError(f"reference construction failed for {selected_case_id} at level {level_index}: {exc}") from exc  # Stop all compared methods because no valid common reference exists.
        ledger["levels"].append(level)  # Add the complete successful level evidence in execution order.
        if level_index == 0:  # Reference A alone cannot establish a two-level convergence gate.
            ledger = _write_sealed(ledger_path, ledger)  # Checkpoint the strong initial reference before attempting finer work.
            continue  # Proceed to the first preregistered independent B candidate.
        previous = ledger["levels"][level_index - 1]  # Treat the immediately preceding successful level as the current promoted A.
        energy_relative_difference = _relative_difference(float(previous["U_total"]), float(level["U_total"]), config.zero_guard)  # Evaluate the exact finer-denominator strain-energy criterion.
        qoi_relative_difference = _relative_difference(float(previous["qoi"]), float(level["qoi"]), config.zero_guard)  # Evaluate the exact finer-denominator QoI criterion.
        converged = bool(energy_relative_difference <= config.agreement_rtol and qoi_relative_difference <= config.agreement_rtol)  # Require both unrounded one-half-percent gates simultaneously.
        comparison = {"a_level": level_index - 1, "b_level": level_index, "reference_A_mesh_sha": previous["mesh_sha"], "reference_B_mesh_sha": level["mesh_sha"], "energy_relative_difference": energy_relative_difference, "qoi_relative_difference": qoi_relative_difference, "agreement_rtol": float(config.agreement_rtol), "passed": converged}  # Record hashes and physical differences before any promotion decision.
        level["comparison_to_previous"] = comparison  # Keep each candidate's acceptance evidence adjacent to its numerical outputs.
        if converged:  # Accept only a consecutive independently generated A/B pair passing both conditions.
            reference_a = _reference_from_level(previous)  # Convert the final promoted coarser level into the experiment contract.
            reference_b = _reference_from_level(level)  # Convert the accepted finer level into the single shared error reference.
            ledger["status"] = "complete"  # Mark the cache qualified by the unchanged original dual convergence gate.
            ledger["qualification"] = True  # Disclose successful one-half-percent qualification independently from expedited depth.
            ledger["completed_utc"] = datetime.now(timezone.utc).isoformat()  # Timestamp successful convergence separately from creation.
            ledger["final_pair"] = {**comparison, "qualification": True, "decision": "dual_convergence_qualified"}  # Preserve exact hashes and unrounded differences while naming the qualified decision accurately.
            ledger["reference_b"] = _json_ready(asdict(reference_b))  # Duplicate the compact final values inside the authoritative ledger.
            ledger = _write_sealed(ledger_path, ledger)  # Publish the complete converged audit history atomically.
            _write_reference_artifact(case_dir, ledger, reference_b)  # Publish the common small Reference B cache for all method runners.
            bind_reference_b(runner, reference_b, problem)  # Inject exactly the verified final B after all persistence succeeds.
            return ReferenceBuildOutcome(reference_a=reference_a, reference_b=reference_b, a_level=level_index - 1, b_level=level_index, ledger_path=ledger_path, from_cache=False, qualification=True, authorization=ledger.get("authorization"), execution_amendment=execution_amendment)  # Return the qualified pair while disclosing any expedited schedule amendment.
        if level_index < len(config.background_scales) - 1:  # Promote only when another preregistered or explicitly expedited level remains to be attempted.
            ledger["upgrade_history"].append({**comparison, "decision": "promote_B_to_A_and_continue_preregistered_refinement"})  # Record the failed pair's hashes and the justified reference-only continuation.
            ledger = _write_sealed(ledger_path, ledger)  # Checkpoint every unsuccessful intermediate comparison before the next finer level.
            continue  # Execute the next already selected refinement level without changing any compared method.
        ledger = _write_sealed(ledger_path, ledger)  # Checkpoint the final successful level and failed original gate before terminal strict or operational handling.
    if execution_amendment is not None and len(ledger["levels"]) == len(config.background_scales) and all(level.get("status") == "success" for level in ledger["levels"]):  # Permit fallback only after every selected native level succeeds and the unchanged gate remains unmet.
        b_level = len(ledger["levels"]) - 1  # Select the finest successful explicitly attempted level as operational Reference B.
        a_level = b_level - 1  # Select its immediately preceding successful level as the auditable comparison A.
        reference_a = _reference_from_level(ledger["levels"][a_level])  # Convert the operational comparison A without claiming convergence.
        reference_b = _reference_from_level(ledger["levels"][b_level])  # Convert the finest successful level into the explicitly operational common B.
        final_comparison = dict(ledger["levels"][b_level]["comparison_to_previous"])  # Preserve the original unrounded one-half-percent gate calculation exactly.
        ledger["status"] = "complete_unqualified"  # Name the usable but nonconverged operational state without ambiguity.
        ledger["qualification"] = False  # Make failure of the original convergence qualification machine-readable.
        ledger["authorization"] = UNQUALIFIED_AUTHORIZATION  # Bind use to the user's exact nonblocking authorization.
        ledger["completed_utc"] = datetime.now(timezone.utc).isoformat()  # Timestamp operational publication separately from native completion.
        ledger["final_pair"] = {**final_comparison, "qualification": False, "authorization": UNQUALIFIED_AUTHORIZATION, "decision": "publish_finest_successful_as_operational_reference_B", "converged": False}  # Retain the failed gate and explicitly avoid any converged label.
        ledger["reference_b"] = _json_ready(asdict(reference_b))  # Publish the compact operational values while qualification remains adjacent in the ledger and artifact.
        ledger["failure"] = {"kind": "reference_qualification_not_met", "attempted_levels": len(config.background_scales), "agreement_rtol": float(config.agreement_rtol), "authorization": UNQUALIFIED_AUTHORIZATION, "nonblocking": True}  # Preserve the original qualification failure even though execution may continue operationally.
        ledger = _write_sealed(ledger_path, ledger)  # Publish the complete unqualified evidence atomically before the compact artifact.
        _write_reference_artifact(case_dir, ledger, reference_b)  # Publish the operational B with explicit false qualification and authorization metadata.
        bind_reference_b(runner, reference_b, problem)  # Bind the operational reference only because this call explicitly opted in.
        return ReferenceBuildOutcome(reference_a=reference_a, reference_b=reference_b, a_level=a_level, b_level=b_level, ledger_path=ledger_path, from_cache=False, qualification=False, authorization=UNQUALIFIED_AUTHORIZATION, execution_amendment=execution_amendment)  # Return usable values without representing them as converged or qualified.
    ledger["status"] = "schedule_exhausted"  # Mark that every preregistered refinement was attempted without convergence.
    ledger["qualification"] = False  # Record the failed original gate even though strict callers receive no usable Reference B.
    ledger["completed_utc"] = datetime.now(timezone.utc).isoformat()  # Timestamp the terminal nonconvergence state.
    ledger["failure"] = {"kind": "reference_convergence_failure", "attempted_levels": len(config.background_scales), "agreement_rtol": float(config.agreement_rtol)}  # Preserve a finite explicit failure ledger instead of selecting the finest unconverged result.
    _write_sealed(ledger_path, ledger)  # Persist terminal nonconvergence before stopping the case.
    raise ReferenceBuildError(f"reference schedule exhausted without dual convergence for {selected_case_id}")  # Prevent every method from reporting errors against an invalid Reference B.


def _validate_complete_ledger(ledger: dict[str, Any], problem: Problem, case_dir: Path, config: ReferenceScheduleConfig, execution_amendment: dict[str, Any] | None, *, allow_unqualified: bool) -> tuple[int, int, bool]:  # Enforce cache identity, amendment, native evidence, and qualified or explicit operational semantics.
    if ledger.get("schema") != REFERENCE_SCHEMA or ledger.get("protocol_id") != PROTOCOL_ID:  # Reject unrelated or stale reference artifacts.
        raise ReferenceBuildError("reference ledger schema or protocol_id mismatch")  # Report protocol substitution directly.
    if ledger.get("problem_signature") != _problem_signature(problem):  # Require the exact manifest-reconstructed geometry, load, and FE settings.
        raise ReferenceBuildError("reference ledger problem signature mismatch")  # Prevent cross-case or changed-factory cache reuse.
    config_snapshot = _config_snapshot(config)  # Recreate the caller's fully registered convergence schedule.
    if ledger.get("config_signature") != _payload_sha256(config_snapshot) or ledger.get("config") != config_snapshot:  # Require both the digest and readable configuration to match.
        raise ReferenceBuildError("reference ledger configuration mismatch")  # Prevent silent threshold or schedule changes.
    if ledger.get("execution_amendment") != execution_amendment:  # Require callers to reproduce the exact strict or expedited execution choice.
        raise ReferenceBuildError("reference ledger execution amendment mismatch")  # Prevent loading an expedited cache through an implicit default schedule.
    status = ledger.get("status")  # Read the explicit qualified or operational terminal state once.
    if status == "complete_unqualified" and not allow_unqualified:  # Keep every default verifier and loader fail-closed on operational evidence.
        raise ReferenceBuildError("reference ledger is complete_unqualified; explicit allow_unqualified=True is required")  # Report the exact opt-in needed without calling it converged.
    if status not in ({"complete", "complete_unqualified"} if allow_unqualified else {"complete"}):  # Refuse building, native failure, and strict exhaustion as common references.
        raise ReferenceBuildError(f"reference ledger is not an accepted complete cache: {status}")  # Preserve the explicit terminal state for diagnosis.
    qualification = status == "complete"  # Derive accepted scientific qualification solely from the unambiguous terminal status.
    if ledger.get("qualification") is not qualification:  # Require redundant top-level qualification to agree exactly with status.
        raise ReferenceBuildError("reference ledger status and qualification disagree")  # Reject a relabelled operational cache.
    expected_authorization = execution_amendment.get("authorization") if execution_amendment is not None else None  # Recover the exact amendment authorization when present.
    if ledger.get("authorization") != expected_authorization:  # Require strict null or exact authorized amendment identity.
        raise ReferenceBuildError("reference ledger authorization differs from the execution amendment")  # Reject missing or substituted user authorization.
    source_contract = ledger.get("source_contract")  # Read the explicit scientific provenance contract before accepting any numerical level.
    if not isinstance(source_contract, dict) or source_contract.get("method_mesh_input_allowed") is not False or source_contract.get("online_solve_counted") is not False:  # Require analytical method-independent meshes and separate reference accounting.
        raise ReferenceBuildError("reference ledger source contract is invalid")  # Reject a cache whose construction could depend on compared method grids.
    levels = ledger.get("levels")  # Read the authenticated ladder records once.
    final_pair = ledger.get("final_pair")  # Read the authenticated accepted comparison once.
    if not isinstance(levels, list) or not isinstance(final_pair, dict) or len(levels) < 2:  # Require enough evidence for an actual A/B comparison.
        raise ReferenceBuildError("reference ledger lacks a complete two-level history")  # Reject truncated complete-looking caches.
    a_level = int(final_pair.get("a_level", -1))  # Normalize the accepted Reference A index.
    b_level = int(final_pair.get("b_level", -1))  # Normalize the accepted Reference B index.
    if a_level < 0 or b_level != a_level + 1 or b_level >= len(levels):  # Require the final pair to be consecutive successful ladder levels.
        raise ReferenceBuildError("reference ledger final pair indices are invalid")  # Reject reordered or post-hoc level selection.
    for level_index, level in enumerate(levels):  # Validate every retained successful level up through the accepted B.
        if not isinstance(level, dict) or level.get("status") != "success" or int(level.get("level_index", -1)) != level_index:  # Require ordered complete records without omitted failures.
            raise ReferenceBuildError(f"reference ledger level {level_index} is invalid")  # Identify the first malformed retained level.
        if level_index >= len(config.background_scales):  # Prevent a cache from appending unregistered refinement levels.
            raise ReferenceBuildError("reference ledger contains an unregistered level")  # Preserve schedule pre-registration.
        if float(level.get("background_scale", math.nan)) != float(config.background_scales[level_index]) or float(level.get("local_floor_scale", math.nan)) != float(config.local_floor_scales[level_index]):  # Require exact registered scale identity.
            raise ReferenceBuildError(f"reference ledger level {level_index} scale mismatch")  # Reject post-hoc field edits.
        expected_background_target = float(problem.h_ref) * float(config.background_scales[level_index])  # Reconstruct the physical background target from authenticated problem and schedule data.
        expected_local_floor_target = float(reference_floor(problem)) * float(config.local_floor_scales[level_index])  # Reconstruct the physical local minimum from authenticated problem and schedule data.
        if not math.isclose(float(level.get("h_background_target", math.nan)), expected_background_target, rel_tol=0.0, abs_tol=1.0e-14) or not math.isclose(float(level.get("h_local_floor_target", math.nan)), expected_local_floor_target, rel_tol=0.0, abs_tol=1.0e-14):  # Require both persisted physical targets to match their preregistered multipliers.
            raise ReferenceBuildError(f"reference ledger level {level_index} target h mismatch")  # Reject a field whose readable h values disagree with its registered schedule.
        numeric_values = (level.get("U_total"), level.get("qoi"), level.get("h_background_target"), level.get("h_local_floor_target"), level.get("h_realized_min"), level.get("h_realized_max"), level.get("mesh_wall_s"), level.get("solver_wall_s"), level.get("total_wall_s"))  # Collect every required finite scalar once.
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in numeric_values):  # Reject missing, string, NaN, and infinite numerical evidence.
            raise ReferenceBuildError(f"reference ledger level {level_index} has invalid numerical fields")  # Prevent malformed records from reaching error calculations.
        if float(level["U_total"]) <= config.zero_guard or abs(float(level["qoi"])) <= config.zero_guard or int(level.get("n_equations", 0)) <= 0 or int(level.get("n_elems", 0)) <= 0 or int(level.get("n_nodes", 0)) <= 0:  # Require physically usable values and actual positive counts.
            raise ReferenceBuildError(f"reference ledger level {level_index} has nonphysical results or counts")  # Reject fake zero results and missing solver metadata.
        if not isinstance(level.get("mesh_sha"), str) or len(level["mesh_sha"]) != 64:  # Require the protocol's complete SHA-256 mesh identity.
            raise ReferenceBuildError(f"reference ledger level {level_index} has invalid mesh SHA-256")  # Reject absent, truncated, or malformed mesh identity evidence.
        if level.get("mesh_source") != "preregistered_analytical_reference_field":  # Require explicit independence from compared method meshes.
            raise ReferenceBuildError(f"reference ledger level {level_index} has invalid mesh provenance")  # Reject a method-derived reference cache.
        solver_logs = level.get("solver_logs")  # Read the mandatory portable native-log receipt for this successful solve.
        if not isinstance(solver_logs, list) or len(solver_logs) != 1 or not isinstance(solver_logs[0], dict) or not isinstance(solver_logs[0].get("size_bytes"), int) or solver_logs[0]["size_bytes"] <= 0:  # Require exactly one nonempty fixed level-scoped combined native log.
            raise ReferenceBuildError(f"reference ledger level {level_index} has invalid solver log ledger")  # Reject empty, duplicated, and structurally ambiguous log evidence.
        _verify_evidence_record(case_dir, solver_logs[0], f"{SOLVER_LOG_DIRECTORY}/ref_l{level_index:02d}.log")  # Prevent traversal and recompute the complete log digest and size.
        if level.get("solver_inputs") != []:  # Successful levels do not need a retained deck and must not smuggle an unrelated file into the cache.
            raise ReferenceBuildError(f"reference ledger level {level_index} has unexpected success input evidence")  # Keep failure-only native inputs semantically explicit.
    previous = levels[a_level]  # Read the authenticated final Reference A record.
    final = levels[b_level]  # Read the authenticated final Reference B record.
    energy_relative_difference = _relative_difference(float(previous["U_total"]), float(final["U_total"]), config.zero_guard)  # Recompute the energy criterion rather than trusting its stored boolean.
    qoi_relative_difference = _relative_difference(float(previous["qoi"]), float(final["qoi"]), config.zero_guard)  # Recompute the QoI criterion rather than trusting its stored boolean.
    gate_passed = bool(energy_relative_difference <= config.agreement_rtol and qoi_relative_difference <= config.agreement_rtol)  # Recompute the unchanged original dual gate without trusting status.
    if qualification and not gate_passed:  # A qualified cache must pass both original one-half-percent criteria.
        raise ReferenceBuildError("qualified reference ledger final A/B pair does not satisfy convergence")  # Reject a forged qualified label or altered output.
    if not qualification and gate_passed:  # An operational fallback must not suppress a qualifying result that should have terminated normally.
        raise ReferenceBuildError("complete_unqualified reference ledger unexpectedly satisfies convergence")  # Reject internally contradictory operational labelling.
    for name, observed in (("energy_relative_difference", energy_relative_difference), ("qoi_relative_difference", qoi_relative_difference)):  # Compare stored audit values with independent recomputation.
        if not math.isclose(float(final_pair.get(name, math.nan)), observed, rel_tol=1.0e-12, abs_tol=1.0e-15):  # Allow only floating serialization roundoff.
            raise ReferenceBuildError(f"reference ledger stored {name} is inconsistent")  # Reject internally contradictory convergence evidence.
    expected_decision = "dual_convergence_qualified" if qualification else "publish_finest_successful_as_operational_reference_B"  # Name the only legal terminal decision for this qualification state.
    decision_matches = final_pair.get("passed") is qualification and final_pair.get("qualification") is qualification and final_pair.get("decision") == expected_decision  # Require the original gate boolean, explicit qualification, and accurate decision label together.
    if not decision_matches or final_pair.get("reference_A_mesh_sha") != previous["mesh_sha"] or final_pair.get("reference_B_mesh_sha") != final["mesh_sha"]:  # Require decision semantics and both exact mesh identities to match.
        raise ReferenceBuildError("reference ledger final pair hash or decision mismatch")  # Reject a post-hoc selected or incorrectly labelled pair.
    if not qualification:  # Apply additional nonblocking amendment constraints without weakening qualified cache validation.
        operational_shape = execution_amendment is not None and b_level == len(levels) - 1 and len(levels) == len(config.background_scales) and final_pair.get("authorization") == UNQUALIFIED_AUTHORIZATION and final_pair.get("converged") is False  # Require the finest selected level, every native success, exact authorization, and explicit nonconvergence.
        if not operational_shape or ledger.get("failure", {}).get("kind") != "reference_qualification_not_met":  # Require complete operational provenance and retained qualification failure.
            raise ReferenceBuildError("complete_unqualified operational fallback provenance is invalid")  # Reject early selection, incomplete schedules, or erased gate failure.
    upgrades = ledger.get("upgrade_history")  # Read the retained failed comparisons that promoted earlier B candidates to A.
    if not isinstance(upgrades, list) or len(upgrades) != a_level:  # Require exactly one failed adjacent comparison before each promoted final-A level.
        raise ReferenceBuildError("reference ledger upgrade history is incomplete")  # Reject missing or surplus refinement decisions.
    for upgrade_index, upgrade in enumerate(upgrades):  # Recompute every failed convergence decision from its adjacent physical levels.
        upgrade_a = levels[upgrade_index]  # Read the coarser level used by this unsuccessful comparison.
        upgrade_b = levels[upgrade_index + 1]  # Read the independently finer candidate that was promoted afterward.
        expected_energy_difference = _relative_difference(float(upgrade_a["U_total"]), float(upgrade_b["U_total"]), config.zero_guard)  # Recompute the failed energy criterion exactly.
        expected_qoi_difference = _relative_difference(float(upgrade_a["qoi"]), float(upgrade_b["qoi"]), config.zero_guard)  # Recompute the failed QoI criterion exactly.
        identities_match = int(upgrade.get("a_level", -1)) == upgrade_index and int(upgrade.get("b_level", -1)) == upgrade_index + 1 and upgrade.get("reference_A_mesh_sha") == upgrade_a["mesh_sha"] and upgrade.get("reference_B_mesh_sha") == upgrade_b["mesh_sha"]  # Bind the decision to the exact consecutive mesh pair.
        differences_match = math.isclose(float(upgrade.get("energy_relative_difference", math.nan)), expected_energy_difference, rel_tol=1.0e-12, abs_tol=1.0e-15) and math.isclose(float(upgrade.get("qoi_relative_difference", math.nan)), expected_qoi_difference, rel_tol=1.0e-12, abs_tol=1.0e-15)  # Require stored differences to match physical outputs.
        actually_failed = expected_energy_difference > config.agreement_rtol or expected_qoi_difference > config.agreement_rtol  # Require at least one hard gate to justify continued reference refinement.
        if not identities_match or not differences_match or not actually_failed or upgrade.get("passed") is not False or upgrade.get("decision") != "promote_B_to_A_and_continue_preregistered_refinement":  # Require complete internally consistent reference-only upgrade evidence.
            raise ReferenceBuildError(f"reference ledger upgrade {upgrade_index} is inconsistent")  # Identify the first unverifiable promotion decision.
    expected_reference_b = _json_ready(asdict(_reference_from_level(final)))  # Reconstruct the compact final reference directly from the accepted finer level.
    if ledger.get("reference_b") != expected_reference_b:  # Require the authoritative compact copy to match the final level exactly.
        raise ReferenceBuildError("reference ledger compact Reference B is inconsistent")  # Reject altered downstream error-reference values.
    return a_level, b_level, qualification  # Return authenticated pair indices and explicit scientific qualification for downstream reporting.


def _load_reference_artifact(case_dir: Path, ledger: dict[str, Any]) -> Reference:  # Authenticate the compact common Reference B against the complete ledger.
    artifact = _read_sealed(case_dir / REFERENCE_B_FILENAME)  # Validate exact-byte content integrity before reading numerical values.
    if artifact.get("schema") != REFERENCE_ARTIFACT_SCHEMA or artifact.get("protocol_id") != PROTOCOL_ID:  # Require the exact shared-reference schema and protocol.
        raise ReferenceBuildError("Reference B artifact schema or protocol mismatch")  # Reject unrelated small JSON caches.
    for key in ("case_id", "problem_signature", "config_signature", "status", "qualification", "authorization", "execution_amendment"):  # Bind compact values and all qualification metadata to the authoritative ledger.
        if artifact.get(key) != ledger.get(key):  # Detect cross-case copies and schedule substitutions.
            raise ReferenceBuildError(f"Reference B artifact {key} mismatch")  # Identify the inconsistent binding field.
    reference_data = artifact.get("reference")  # Read the compact experiment.Reference representation.
    if not isinstance(reference_data, dict) or reference_data != ledger.get("reference_b"):  # Require exact numerical agreement with the authoritative ledger copy.
        raise ReferenceBuildError("Reference B artifact does not match the ledger")  # Reject independently altered final numbers.
    try:  # Convert schema or type errors into the module's explicit failure contract.
        return Reference(**reference_data)  # Reconstruct the shared object used by FemRunner error calculations.
    except (TypeError, ValueError) as exc:  # Catch missing, extra, and invalid dataclass values.
        raise ReferenceBuildError(f"invalid Reference B payload: {exc}") from exc  # Preserve the malformed artifact cause.


def verify_reference_cache(reference_root: Path | str, *, case_id: str, problem: Problem, config: ReferenceScheduleConfig = DEFAULT_REFERENCE_CONFIG, mesh_factory: Callable[..., Mesh] = generate_mesh, regenerate_meshes: bool = False, allow_unqualified: bool = False, expedited_levels: int | None = None) -> dict[str, Any]:  # Verify one strict or explicitly authorized operational cache and optionally regenerate analytical meshes.
    selected_case_id = _validated_case_id(case_id)  # Validate the manifest identifier before constructing paths.
    effective_config, execution_amendment = _effective_config(config, allow_unqualified=bool(allow_unqualified), expedited_levels=expedited_levels)  # Reproduce the exact strict or expedited schedule before reading cache values.
    case_dir = reference_case_dir(reference_root, selected_case_id)  # Resolve the expected immutable per-case cache location.
    ledger = _read_sealed(case_dir / LEDGER_FILENAME)  # Authenticate the complete audit ledger before interpreting it.
    if ledger.get("case_id") != selected_case_id:  # Prevent a renamed directory from masquerading as another case.
        raise ReferenceBuildError("reference ledger case_id does not match its cache directory")  # Surface directory-to-artifact identity drift.
    a_level, b_level, qualification = _validate_complete_ledger(ledger, problem, case_dir, effective_config, execution_amendment, allow_unqualified=bool(allow_unqualified))  # Enforce identity, amendment, evidence, original gate, and opt-in qualification semantics.
    reference_b = _load_reference_artifact(case_dir, ledger)  # Authenticate the compact final reference and exact ledger agreement.
    regenerated = []  # Collect optional deterministic mesh-verification receipts without invoking CalculiX.
    if regenerate_meshes:  # Perform the expensive strongest mesh-provenance check only when explicitly requested.
        for level in ledger["levels"]:  # Rebuild every successful level directly from the registered analytical field.
            level_index = int(level["level_index"])  # Read the stable ladder position used by the original Gmsh model.
            background_scale = float(effective_config.background_scales[level_index])  # Reuse the authenticated strict or expedited background multiplier.
            local_floor_scale = float(effective_config.local_floor_scales[level_index])  # Reuse the authenticated strict or expedited local-floor multiplier.
            field = _scaled_reference_size_fn(problem, background_scale, local_floor_scale)  # Reconstruct the field without any stored or method mesh.
            model_name = f"reference_l{level_index:02d}_{problem.instance_id[-8:]}"  # Match the original deterministic Gmsh model name.
            mesh = mesh_factory(problem, field, model_name=model_name, h_floor=float(reference_floor(problem)) * local_floor_scale)  # Regenerate geometry and mesh from registered inputs only.
            observed_sha = _mesh_sha256(mesh)  # Hash exact node coordinates, connectivity, shapes, and dimension with full SHA-256.
            matched = bool(observed_sha == level["mesh_sha"] and int(mesh.n_cells) == int(level["n_elems"]) and int(mesh.n_nodes) == int(level["n_nodes"]))  # Require identity and both primary mesh cardinalities.
            regenerated.append({"level_index": level_index, "mesh_sha": observed_sha, "expected_mesh_sha": level["mesh_sha"], "n_nodes": int(mesh.n_nodes), "n_elems": int(mesh.n_cells), "passed": matched})  # Preserve a complete strict-verification receipt.
            if not matched:  # Reject environment or geometry drift before using cached solver values.
                raise ReferenceBuildError(f"regenerated mesh mismatch at reference level {level_index}")  # Identify the first non-reproducible ladder level.
    verified_logs = [_verify_evidence_record(case_dir, level["solver_logs"][0], f"{SOLVER_LOG_DIRECTORY}/ref_l{int(level['level_index']):02d}.log") for level in ledger["levels"]]  # Return independently recomputed per-level portable log receipts in ladder order.
    final_pair = ledger["final_pair"]  # Read the already authenticated terminal comparison for transparent unchanged-gate reporting.
    return {"schema": "wmvla-four-way-reference-verification-v3", "protocol_id": PROTOCOL_ID, "case_id": selected_case_id, "status": ledger["status"], "qualification": qualification, "authorization": ledger.get("authorization"), "execution_amendment": execution_amendment, "problem_signature": ledger["problem_signature"], "config_signature": ledger["config_signature"], "a_level": a_level, "b_level": b_level, "reference_b": _json_ready(asdict(reference_b)), "original_convergence_gate": {"agreement_rtol": float(effective_config.agreement_rtol), "energy_relative_difference": float(final_pair["energy_relative_difference"]), "qoi_relative_difference": float(final_pair["qoi_relative_difference"]), "passed": bool(final_pair["passed"])}, "integrity_verified": True, "solver_log_verification": verified_logs, "mesh_regeneration_requested": bool(regenerate_meshes), "mesh_regeneration": regenerated, "passed": True}  # Return usable-cache verification while clearly separating protocol qualification from verifier integrity success.


def verify_reference_failure_evidence(reference_root: Path | str, *, case_id: str, problem: Problem, config: ReferenceScheduleConfig = DEFAULT_REFERENCE_CONFIG, allow_unqualified: bool = False, expedited_levels: int | None = None) -> dict[str, Any]:  # Authenticate a strict or expedited terminal failure and every native artifact it claims.
    selected_case_id = _validated_case_id(case_id)  # Validate the manifest identifier before resolving any cache path.
    effective_config, execution_amendment = _effective_config(config, allow_unqualified=bool(allow_unqualified), expedited_levels=expedited_levels)  # Reproduce the exact attempted schedule and authorization without accepting a Reference B.
    case_dir = reference_case_dir(reference_root, selected_case_id)  # Resolve the exact per-case cache boundary for containment checks.
    ledger = _read_sealed(case_dir / LEDGER_FILENAME)  # Authenticate every terminal ledger byte before interpreting failure evidence.
    if ledger.get("schema") != REFERENCE_SCHEMA or ledger.get("protocol_id") != PROTOCOL_ID or ledger.get("case_id") != selected_case_id:  # Require the upgraded exact protocol and directory identity.
        raise ReferenceBuildError("reference failure ledger schema, protocol, or case identity mismatch")  # Reject unrelated or renamed failure evidence.
    if ledger.get("problem_signature") != _problem_signature(problem):  # Bind the failure to the exact geometry, load, material, and FE settings.
        raise ReferenceBuildError("reference failure ledger problem signature mismatch")  # Prevent cross-case native evidence reuse.
    snapshot = _config_snapshot(effective_config)  # Reconstruct the exact strict or expedited attempted ladder and unchanged threshold.
    if ledger.get("config") != snapshot or ledger.get("config_signature") != _payload_sha256(snapshot):  # Require readable and hashed schedule identity together.
        raise ReferenceBuildError("reference failure ledger configuration mismatch")  # Reject post-failure schedule edits.
    if ledger.get("execution_amendment") != execution_amendment:  # Require exact strict or expedited intent even for unusable native evidence.
        raise ReferenceBuildError("reference failure ledger execution amendment mismatch")  # Reject changing operational policy after a native failure.
    status = ledger.get("status")  # Read the terminal status after identity authentication.
    if status not in {"numerical_failure", "schedule_exhausted"}:  # Exclude building and successful caches from the terminal-failure verifier.
        raise ReferenceBuildError(f"reference ledger is not a terminal failure: {status}")  # Keep each public verifier's semantics unambiguous.
    levels = ledger.get("levels")  # Read the complete attempted level history once.
    if not isinstance(levels, list) or not levels or len(levels) > len(effective_config.background_scales):  # Require a nonempty bounded attempted prefix.
        raise ReferenceBuildError("reference failure ledger has an invalid level history")  # Reject empty, scalar, and over-schedule failure evidence.
    log_receipts: list[dict[str, Any]] = []  # Collect independently recomputed log identities for all successful and failed native attempts.
    input_receipts: list[dict[str, Any]] = []  # Collect independently recomputed failure-only deck identities.
    failed_count = 0  # Require exactly one final failed level only for numerical_failure status.
    for level_index, level in enumerate(levels):  # Verify ordered structure and portable evidence for every attempted registered level.
        if not isinstance(level, dict) or int(level.get("level_index", -1)) != level_index:  # Require a gap-free ordered mapping sequence.
            raise ReferenceBuildError(f"reference failure ledger level {level_index} is malformed")  # Identify the first ambiguous attempt record.
        level_status = level.get("status")  # Read success or failure only after structural validation.
        logs = level.get("solver_logs")  # Read the possibly absent pre-native or mandatory post-native portable logs.
        inputs = level.get("solver_inputs")  # Read the failure-only portable input decks.
        if level_status == "success":  # Every completed prefix level must retain one authentic native log.
            if not isinstance(logs, list) or len(logs) != 1 or not isinstance(logs[0], dict) or not isinstance(logs[0].get("size_bytes"), int) or logs[0]["size_bytes"] <= 0 or inputs != []:  # Require one nonempty log and no success-only deck smuggling.
                raise ReferenceBuildError(f"reference failure ledger successful level {level_index} evidence is invalid")  # Refuse an unauditable promoted reference prefix.
            log_receipts.append(_verify_evidence_record(case_dir, logs[0], f"{SOLVER_LOG_DIRECTORY}/ref_l{level_index:02d}.log"))  # Recompute the exact successful native log bytes.
            continue  # Proceed to a later success or the sole terminal failure.
        if level_status != "failed" or level_index != len(levels) - 1:  # Permit one failed record only at the terminal attempted level.
            raise ReferenceBuildError(f"reference failure ledger level {level_index} status is invalid")  # Reject continued execution after failure or unknown states.
        failed_count += 1  # Count the sole permitted numerical failure record.
        if not isinstance(logs, list) or len(logs) > 1 or not isinstance(inputs, list) or len(inputs) > 1:  # Permit absent pre-native products or one fixed copied artifact of each type.
            raise ReferenceBuildError(f"reference failure ledger failed level {level_index} evidence is invalid")  # Reject duplicated or structurally malformed failure evidence.
        if logs:  # Authenticate an already-produced native failure log when present.
            log_receipts.append(_verify_evidence_record(case_dir, logs[0], f"{SOLVER_LOG_DIRECTORY}/ref_l{level_index:02d}.log"))  # Prevent traversal and recompute its full identity.
        if inputs:  # Authenticate an already-generated CalculiX deck when present.
            input_receipts.append(_verify_evidence_record(case_dir, inputs[0], f"{SOLVER_INPUT_DIRECTORY}/ref_l{level_index:02d}.inp"))  # Prevent traversal and recompute its full identity.
    if status == "numerical_failure" and failed_count != 1:  # Require one explicit failed final attempt for a numerical terminal state.
        raise ReferenceBuildError("numerical failure ledger lacks exactly one terminal failed level")  # Reject failure labels unsupported by retained attempt evidence.
    if status == "schedule_exhausted" and failed_count != 0:  # Exhaustion consists only of successful but unconverged registered solves.
        raise ReferenceBuildError("schedule-exhausted ledger contains a failed level")  # Keep nonconvergence distinct from numerical backend failure.
    return {"schema": "wmvla-four-way-reference-failure-verification-v3", "protocol_id": PROTOCOL_ID, "case_id": selected_case_id, "status": status, "authorization": ledger.get("authorization"), "execution_amendment": execution_amendment, "solver_log_verification": log_receipts, "solver_input_verification": input_receipts, "integrity_verified": True, "passed": True}  # Return complete strict or expedited native-failure evidence without inventing a Reference B.


def load_reference_b(reference_root: Path | str, *, case_id: str, problem: Problem, runner: FemRunner | Any | None = None, config: ReferenceScheduleConfig = DEFAULT_REFERENCE_CONFIG, verify: bool = True, mesh_factory: Callable[..., Mesh] = generate_mesh, regenerate_meshes: bool = False, allow_unqualified: bool = False, expedited_levels: int | None = None) -> Reference:  # Load a qualified or explicitly authorized operational B and optionally inject it into a runner.
    selected_case_id = _validated_case_id(case_id)  # Validate the manifest identifier before any filesystem access.
    _effective_schedule, execution_amendment = _effective_config(config, allow_unqualified=bool(allow_unqualified), expedited_levels=expedited_levels)  # Validate and reproduce the caller's exact strict or expedited intent before cache access.
    case_dir = reference_case_dir(reference_root, selected_case_id)  # Resolve the exact shared per-case cache location.
    ledger = _read_sealed(case_dir / LEDGER_FILENAME)  # Always enforce artifact integrity even when full validation is explicitly disabled.
    if ledger.get("status") == "complete_unqualified" and not allow_unqualified:  # Keep even the reduced-verification load path fail-closed by default.
        raise ReferenceBuildError("Reference B is complete_unqualified; explicit allow_unqualified=True is required")  # Prevent accidental operational use without the named opt-in.
    if ledger.get("status") not in ({"complete", "complete_unqualified"} if allow_unqualified else {"complete"}):  # Reject building, failed, and exhausted caches regardless of the verify flag.
        raise ReferenceBuildError(f"Reference B ledger is not loadable: {ledger.get('status')}")  # Preserve the exact unusable terminal state.
    if ledger.get("execution_amendment") != execution_amendment:  # Require callers to name the exact expedited depth even when full mesh verification is disabled.
        raise ReferenceBuildError("Reference B execution amendment mismatch")  # Prevent implicit loading under a different effective schedule.
    if verify:  # Apply full identity, schedule, convergence, and optional regenerated-mesh checks by default.
        verify_reference_cache(reference_root, case_id=selected_case_id, problem=problem, config=config, mesh_factory=mesh_factory, regenerate_meshes=regenerate_meshes, allow_unqualified=allow_unqualified, expedited_levels=expedited_levels)  # Fail closed on identity, amendment, qualification, evidence, and optional regenerated meshes.
    reference_b = _load_reference_artifact(case_dir, ledger)  # Authenticate the compact final reference against the ledger.
    if runner is not None:  # Support direct setup of WM, LP, supervised, RL, and Dörfler runners.
        bind_reference_b(runner, reference_b, problem)  # Inject the same verified object under the exact case guard.
    return reference_b  # Return the common final B for validation metrics or explicit caller binding.
