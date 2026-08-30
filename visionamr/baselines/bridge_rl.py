"""Frozen bridge-family Double-DQN training, selection, and blind-test driver."""  # State the module's protocol-bound responsibility.
from __future__ import annotations  # Postpone annotation evaluation for runtime compatibility.
from collections.abc import Mapping, Sequence  # Type read-only protocol inputs without permitting mutation assumptions.
from dataclasses import asdict, dataclass, replace  # Serialize records and derive per-budget immutable configurations.
import hashlib  # Hash every frozen model and index artifact before blind testing.
import json  # Persist strict machine-readable histories, selections, and test evidence.
import math  # Validate metrics and compute the preregistered log-mean selection scores.
from pathlib import Path  # Build portable manifest, partition, reference, model, and evidence paths.
import random  # Drive the existing epsilon-greedy replay learner from one explicit policy seed.
import shutil  # Copy the selected checkpoint into its immutable frozen-model location.
import statistics  # Report validation-score dispersion across the three independent policy seeds.
import time  # Measure training, validation, and test wall-clock costs monotonically.
from typing import Any  # Describe strict JSON payloads and solver records at the integration boundary.
import numpy as np  # Represent graph states and select greedy Q actions deterministically.
from .rl_dqn import DQNConfig, DQNPolicy, RegionRefineEnv, Transition  # Reuse the repository's region-graph Double-DQN implementation.
from ..bridge_case_manifest import load_case_manifest, problem_from_case  # Reconstruct only manifest-authorized bridge instances.
from ..calculix import CalculiXExecutionError  # Retain only explicitly typed native CalculiX numerical failures as failed samples.
from ..experiment import FemRunner  # Count every real Gmsh-plus-CalculiX trajectory solve through the common runner.
from ..mesher import GmshMeshingError  # Retain only explicitly typed native Gmsh materialization failures as failed samples.
from ..vla.four_way_references import UNQUALIFIED_AUTHORIZATION  # Reuse the sole fixed user amendment token for operational Reference B use.
from ..vla.partition_spec import COMMON_NODAL_GRADATION  # Reuse the single shared PR-40 V0 nodal size-field gradation contract.

PROTOCOL_ID = "WMVLA-4WAY-P1"  # Bind every artifact to the frozen four-way protocol.
RL_SCHEMA = "wmvla.bridge-rl.v1"  # Version the training, freeze, and test evidence contract.
RL_SEEDS = (20260901, 20260902, 20260903)  # Freeze the three independent Double-DQN initialization and exploration seeds.
EQUATION_BUDGETS = (30000, 60000, 120000)  # Freeze the three public effective-equation budgets.
SOLVE_PREFIXES = (2, 3, 4, 6)  # Freeze the four real-solve prefixes retained from each six-solve trajectory.
TRAIN_EPISODES = 300  # Require exactly three hundred attempted complete episodes per policy seed.
VALIDATION_INTERVAL = 25  # Evaluate every policy checkpoint after each twenty-five training episodes.
VALIDATION_EPISODES = tuple(range(VALIDATION_INTERVAL, TRAIN_EPISODES + 1, VALIDATION_INTERVAL))  # Freeze all twelve eligible checkpoint episodes.
TRAIN_CASE_COUNT = 24  # Require the complete manifest training split and no other cases.
VALIDATION_CASE_COUNT = 8  # Require the complete manifest validation split at every checkpoint.
TEST_CASE_COUNT = 16  # Require the complete sorted blind-test split during the separate test phase.
MAX_REAL_SOLVES = 6  # Limit each online trajectory to the common probe plus at most five region refinements.
MAX_REGION_REFINEMENTS = MAX_REAL_SOLVES - 1  # Translate the total-solve contract into the environment's refinement-step limit.
VALIDATION_FAILURE_ERROR = 10.0  # Score each missing, failed, invalid, or nonfinite validation metric with one fixed finite error.
ERROR_FLOOR = 1.0e-300  # Keep exact-zero successful validation errors finite in logarithmic selection metrics.
MODEL_FILENAME_TEMPLATE = "rl_seed{seed}.pt"  # Give each selected budget-conditioned policy one stable filename.
PARTITION_FILENAME = "partition_spec.json"  # Read the exact shared WM/RL partition artifact for every case.
REFERENCE_LEDGER_FILENAME = "reference_ledger.json"  # Preflight the common validated Reference-B ledger without building references during training.
EXPEDITED_REFERENCE_LEVELS = 2  # Bind the exceptional training-validation path to the exact user-authorized two-level reference prefix.
REFERENCE_AMENDMENT_FILENAME = "EXPEDITED_EXECUTION_AMENDMENT.md"  # Require the canonical pre-freeze human amendment beside the manifest.
RETAINED_NUMERICAL_FAILURE_TYPES = (CalculiXExecutionError, GmshMeshingError)  # Exclude schema, hash, configuration, policy, and serialization errors from sample-level failure scoring.

@dataclass(frozen=True)  # Prevent result-dependent mutation of a validation operating-point observation.
class ValidationObservation:  # Represent one validation case at one equation budget for one checkpoint.
    case_id: str  # Identify the manifest validation case.
    equation_budget: int  # Identify the budget-conditioned environment used for this rollout.
    energy_error: float | None  # Store the best six-solve budget-feasible Reference-B energy error.
    qoi_error: float | None  # Store the best six-solve budget-feasible Reference-B QoI error.
    ok: bool  # Declare whether the complete greedy trajectory remained numerically valid.
    budget_violation: bool  # Retain any attempted solve above the public equation budget.
    solve_attempts: int  # Count successful and explicitly failed real solver invocations for cost reporting.
    energy_ok: bool = True  # Preserve energy-metric validity independently when the trajectory itself completed.
    qoi_ok: bool = True  # Preserve QoI-metric validity independently when the trajectory itself completed.
    failure: Mapping[str, Any] | None = None  # Preserve a sanitized numerical-failure receipt instead of dropping the point.

def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:  # Encode artifacts without permitting NaN, infinity, or platform-dependent key order.
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"  # Produce stable reviewable UTF-8 JSON.
    return text.encode("utf-8")  # Return the exact bytes covered by artifact hashes.

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:  # Persist one strict JSON artifact after creating only its direct parent directory.
    encoded = _strict_json_bytes(payload)  # Validate and serialize the complete artifact before touching its destination.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create the specific evidence directory after serialization succeeds.
    path.write_bytes(encoded)  # Write the already validated deterministic bytes in one operation.

def _sha256_file(path: Path) -> str:  # Hash one immutable model or index file using bounded streaming memory.
    digest = hashlib.sha256()  # Initialize the protocol-required SHA-256 digest.
    with path.open("rb") as stream:  # Read the exact persisted bytes without decoding binary checkpoints.
        for block in iter(lambda: stream.read(1024 * 1024), b""):  # Process one-megabyte blocks until end of file.
            digest.update(block)  # Accumulate every byte into the content identity.
    return digest.hexdigest()  # Return the lowercase standard hexadecimal digest.

def _strict_reference_policy() -> dict[str, Any]:  # Construct the default fail-closed validation-reference policy without filesystem access.
    return {"allow_unqualified_references": False, "expedited_reference_levels": None, "authorization": None, "reference_execution_amendment": None}  # Make strict qualification and absence of an amendment explicit in every artifact.

def _normalize_reference_policy(value: Mapping[str, Any] | None) -> dict[str, Any]:  # Validate one serialized strict or explicitly amended reference policy exactly.
    if value is None:  # Preserve backward-compatible call syntax while keeping runtime behavior strict.
        return _strict_reference_policy()  # Return an explicit default policy rather than an implicit truthy fallback.
    if not isinstance(value, Mapping):  # Reject scalar or sequence policy substitutions.
        raise ValueError("RL validation reference policy must be an object")  # Keep qualification intent machine-readable.
    expected_keys = {"allow_unqualified_references", "expedited_reference_levels", "authorization", "reference_execution_amendment"}  # Freeze the complete policy vocabulary.
    if set(value) != expected_keys or type(value.get("allow_unqualified_references")) is not bool:  # Reject missing, extra, or truthy non-boolean configuration.
        raise ValueError("RL validation reference policy has an incompatible schema")  # Prevent hidden or ambiguous waiver metadata.
    allow_unqualified = value["allow_unqualified_references"]  # Read the exact exceptional-policy boolean.
    levels = value.get("expedited_reference_levels")  # Read the exact optional ladder depth.
    authorization = value.get("authorization")  # Read the fixed user amendment token.
    amendment = value.get("reference_execution_amendment")  # Read the full-hash human amendment receipt.
    if not allow_unqualified:  # Enforce the ordinary fully qualified Reference B path.
        if levels is not None or authorization is not None or amendment is not None:  # Forbid inactive amendment metadata from changing strict semantics.
            raise ValueError("strict RL validation reference policy must not activate amendment fields")  # Reject an undisclosed shortened schedule.
        return _strict_reference_policy()  # Return a fresh canonical strict policy.
    if type(levels) is not int or levels != EXPEDITED_REFERENCE_LEVELS or authorization != UNQUALIFIED_AUTHORIZATION or not isinstance(amendment, Mapping):  # Require the sole authorized two-level mode and token.
        raise ValueError("unqualified RL validation references require the fixed two-level user amendment")  # Reject caller-selected depths or missing authorization.
    amendment_keys = {"path", "sha256", "size_bytes", "authorization", "expedited_levels"}  # Freeze the exact pre-freeze amendment receipt fields.
    digest = amendment.get("sha256")  # Read the complete content identity once.
    size = amendment.get("size_bytes")  # Read the exact byte count once.
    valid_digest = isinstance(digest, str) and len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)  # Require all SHA-256 bits in canonical lowercase hexadecimal.
    if set(amendment) != amendment_keys or amendment.get("path") != f"protocol/{REFERENCE_AMENDMENT_FILENAME}" or not valid_digest or type(size) is not int or size <= 0 or amendment.get("authorization") != UNQUALIFIED_AUTHORIZATION or amendment.get("expedited_levels") != EXPEDITED_REFERENCE_LEVELS:  # Require canonical path, exact bytes, token, and depth.
        raise ValueError("RL reference execution amendment receipt is invalid")  # Reject stale, relocated, truncated, or differently authorized evidence.
    return {"allow_unqualified_references": True, "expedited_reference_levels": EXPEDITED_REFERENCE_LEVELS, "authorization": UNQUALIFIED_AUTHORIZATION, "reference_execution_amendment": dict(amendment)}  # Return a canonical copy safe for artifact serialization.

def build_training_reference_policy(manifest_path: Path | str, *, allow_unqualified_references: bool = False, expedited_reference_levels: int | None = None) -> dict[str, Any]:  # Authenticate the strict default or fixed pre-freeze user amendment before RL validation.
    if type(allow_unqualified_references) is not bool:  # Reject truthy strings and integers at the Python API boundary.
        raise ValueError("allow_unqualified_references must be an explicit boolean")  # Preserve the same semantics as the CLI flag.
    if not allow_unqualified_references:  # Keep strict scientific qualification as the default.
        if expedited_reference_levels is not None:  # Forbid a shortened schedule without the unmistakable waiver flag.
            raise ValueError("--expedited-reference-levels requires --allow-unqualified-references")  # Fail before reading any reference cache.
        return _strict_reference_policy()  # Return the explicit strict policy without requiring an amendment file.
    if type(expedited_reference_levels) is not int or expedited_reference_levels != EXPEDITED_REFERENCE_LEVELS:  # Bind exceptional use to the fixed two-level prefix only.
        raise ValueError(f"--allow-unqualified-references requires --expedited-reference-levels {EXPEDITED_REFERENCE_LEVELS}")  # Reject missing or caller-selected ladder depths.
    manifest_file = Path(manifest_path).resolve()  # Normalize the canonical manifest location before resolving human authorization.
    protocol_directory = manifest_file.parent  # Require the amendment beside the exact manifest and checksum artifacts.
    if protocol_directory.name != "protocol":  # Preserve the freeze workflow's sole canonical amendment path.
        raise ValueError("amended RL training requires manifest beneath a protocol directory")  # Prevent an unreviewed relocated amendment.
    amendment_path = protocol_directory / REFERENCE_AMENDMENT_FILENAME  # Resolve the fixed pre-freeze user authorization file.
    if not amendment_path.is_file() or amendment_path.is_symlink():  # Require concrete durable bytes and forbid redirected content.
        raise FileNotFoundError(f"required RL reference execution amendment is missing or invalid: {amendment_path}")  # Stop before training or validation solves.
    try:  # Decode the reviewable authorization bytes exactly once.
        amendment_bytes = amendment_path.read_bytes()  # Capture one coherent byte snapshot for token, size, and digest verification.
        content = amendment_bytes.decode("utf-8")  # Require transparent UTF-8 human evidence without a second filesystem read.
    except (OSError, UnicodeError) as exception:  # Convert unreadable or non-UTF-8 evidence into a configuration failure.
        raise ValueError(f"cannot read RL reference execution amendment: {amendment_path}") from exception  # Preserve the exact failing path.
    authorization_line = f"- 授权标识：`{UNQUALIFIED_AUTHORIZATION}`"  # Construct the sole accepted named authorization declaration.
    if authorization_line not in content.splitlines():  # Require the token in its exact dedicated field rather than incidental prose.
        raise ValueError(f"RL reference execution amendment lacks authorization token {UNQUALIFIED_AUTHORIZATION}")  # Reject stale or unrelated documents.
    amendment = {"path": f"protocol/{REFERENCE_AMENDMENT_FILENAME}", "sha256": hashlib.sha256(amendment_bytes).hexdigest(), "size_bytes": len(amendment_bytes), "authorization": UNQUALIFIED_AUTHORIZATION, "expedited_levels": EXPEDITED_REFERENCE_LEVELS}  # Bind the coherent human byte snapshot, length, token, and fixed depth for later freeze comparison.
    return _normalize_reference_policy({"allow_unqualified_references": True, "expedited_reference_levels": EXPEDITED_REFERENCE_LEVELS, "authorization": UNQUALIFIED_AUTHORIZATION, "reference_execution_amendment": amendment})  # Revalidate and return the canonical exceptional policy.

def _verify_live_reference_policy(manifest_path: Path | str, reference_policy: Mapping[str, Any] | None) -> dict[str, Any]:  # Reauthenticate serialized policy intent against canonical amendment bytes immediately before freezing.
    normalized_reference_policy = _normalize_reference_policy(reference_policy)  # Reject malformed strict or amended metadata before deriving operational switches.
    live_reference_policy = build_training_reference_policy(manifest_path, allow_unqualified_references=normalized_reference_policy["allow_unqualified_references"], expedited_reference_levels=normalized_reference_policy["expedited_reference_levels"])  # Recompute the sole accepted policy and amendment receipt from live bytes.
    if normalized_reference_policy != live_reference_policy:  # Require the training-time and freeze-time authorization evidence to remain byte-identical.
        raise ValueError("RL validation reference amendment no longer matches the canonical file")  # Invalidate the campaign rather than freezing under stale human authorization.
    return normalized_reference_policy  # Return the canonical verified policy for immutable artifact propagation.

def _finite_nonnegative(value: float | None) -> float | None:  # Validate one relative error without silently repairing malformed evidence.
    if value is None:  # Treat an absent metric as a failed validation point.
        return None  # Preserve the failure classification for the fixed penalty rule.
    numeric = float(value)  # Normalize NumPy and Python scalar values into one representation.
    if not math.isfinite(numeric) or numeric < 0.0:  # Reject NaN, infinity, and physically invalid negative errors.
        return None  # Return failure rather than leaking a nonstandard JSON number.
    return numeric  # Preserve every finite nonnegative value, including exact zero.

def training_assignment(episode: int, training_cases: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], int]:  # Select one manifest case and the fixed cyclic budget for a one-based episode.
    if int(episode) < 1 or int(episode) > TRAIN_EPISODES:  # Restrict callers to the frozen three-hundred-episode schedule.
        raise ValueError("training episode must be in the frozen range 1..300")  # Reject resuming or extending a policy under the same identity.
    if len(training_cases) != TRAIN_CASE_COUNT:  # Require all and only the twenty-four manifest training cases.
        raise ValueError("RL training requires exactly 24 manifest training cases")  # Prevent a silently reduced or contaminated training set.
    zero_based = int(episode) - 1  # Convert the public one-based episode number into deterministic indices.
    block = zero_based // TRAIN_CASE_COUNT  # Count complete passes through the training split.
    slot = zero_based % TRAIN_CASE_COUNT  # Locate the current slot within this complete split pass.
    case_index = (slot + block) % TRAIN_CASE_COUNT  # Rotate each pass so every case experiences all three budgets over the first 288 episodes.
    budget = EQUATION_BUDGETS[zero_based % len(EQUATION_BUDGETS)]  # Apply the literal preregistered 30k, 60k, 120k episode cycle.
    return training_cases[case_index], int(budget)  # Return one authorized case and its explicit budget-conditioned state contract.

def manifest_cases(manifest: Mapping[str, Any], split: str) -> tuple[Mapping[str, Any], ...]:  # Select one exact manifest split without exposing another split's parameters to a phase runner.
    expected_counts = {"train": TRAIN_CASE_COUNT, "validation": VALIDATION_CASE_COUNT, "test": TEST_CASE_COUNT}  # Freeze accepted split names and cardinalities.
    if split not in expected_counts:  # Reject aliases that could accidentally mix validation and blind cases.
        raise ValueError("split must be train, validation, or test")  # Keep the data boundary explicit.
    cases_value = manifest.get("cases")  # Read the already validated manifest case container once.
    if not isinstance(cases_value, list):  # Defend direct callers that bypass load_case_manifest validation.
        raise ValueError("manifest cases must be a list")  # Reject structurally invalid campaign inputs.
    selected = tuple(case for case in cases_value if isinstance(case, Mapping) and case.get("split") == split)  # Retain only the explicitly requested split in frozen manifest order.
    if len(selected) != expected_counts[split]:  # Require the complete split rather than allowing favorable subsets.
        raise ValueError(f"manifest {split} split must contain exactly {expected_counts[split]} cases")  # Report the precise cardinality contract.
    if split == "test":  # Apply the protocol's explicit blind execution ordering only in the test phase.
        selected = tuple(sorted(selected, key=lambda case: str(case["case_id"])))  # Sort by immutable case_id before any blind solve starts.
    return selected  # Return an immutable split view for the authorized phase.

def build_training_plan(manifest: Mapping[str, Any], reference_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:  # Materialize the complete three-policy schedule and explicit validation-reference policy without native work.
    training_cases = manifest_cases(manifest, "train")  # Validate and isolate the authorized twenty-four training cases.
    validation_cases = manifest_cases(manifest, "validation")  # Validate and isolate the authorized eight checkpoint-selection cases.
    normalized_reference_policy = _normalize_reference_policy(reference_policy)  # Make strict default or fixed amendment metadata explicit before schedule materialization.
    assignments = [training_assignment(episode, training_cases) for episode in range(1, TRAIN_EPISODES + 1)]  # Expand the unique budget-conditioned episode schedule once.
    budget_counts = {str(budget): sum(1 for _case, assigned_budget in assignments if assigned_budget == budget) for budget in EQUATION_BUDGETS}  # Prove each budget occurs exactly one hundred times per seed.
    case_counts = {str(case["case_id"]): sum(1 for assigned_case, _budget in assignments if assigned_case["case_id"] == case["case_id"]) for case in training_cases}  # Disclose deterministic per-case exposure without random resampling.
    policies = [{"seed": seed, "episodes": TRAIN_EPISODES, "budget_counts": budget_counts, "checkpoint_episodes": list(VALIDATION_EPISODES), "validation_points_per_checkpoint": VALIDATION_CASE_COUNT * len(EQUATION_BUDGETS), "model_file": MODEL_FILENAME_TEMPLATE.format(seed=seed), "validation_reference_policy": dict(normalized_reference_policy)} for seed in RL_SEEDS]  # Describe the three independent policies and their identical explicit checkpoint-reference policy.
    return {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "training_plan", "training_split_case_ids": [str(case["case_id"]) for case in training_cases], "validation_split_case_ids": [str(case["case_id"]) for case in validation_cases], "test_split_access": False, "seeds": list(RL_SEEDS), "equation_budgets": list(EQUATION_BUDGETS), "budget_conditioning": "state_n_equations_over_active_budget", "episode_budget_schedule": "literal_cycle_30000_60000_120000", "episode_case_schedule": "manifest_order_rotated_by_completed_24_case_blocks", "training_case_episode_counts": case_counts, "validation_reference_policy": normalized_reference_policy, "policies": policies, "selection_order": ["failure_points", "finite_penalty_energy_log_mean", "finite_penalty_qoi_log_mean", "budget_violations", "checkpoint_episode"], "validation_failure_error": VALIDATION_FAILURE_ERROR, "error_floor": ERROR_FLOOR, "max_real_solves": MAX_REAL_SOLVES, "nodal_gradation": COMMON_NODAL_GRADATION, "nodal_gradation_source": "PR40_V0_default_tool_behavior"}  # Return a complete preregistration record with no blind-test parameters or results and an auditable reference qualification choice.

def validation_selection_key(observations: Sequence[ValidationObservation], checkpoint_episode: int, expected_case_ids: Sequence[str]) -> tuple[int, float, float, int, int]:  # Compute the frozen lexicographic checkpoint-selection key.
    if int(checkpoint_episode) not in VALIDATION_EPISODES:  # Accept only checkpoints produced at the twelve preregistered intervals.
        raise ValueError("checkpoint episode is not on the frozen 25-episode schedule")  # Prevent post-hoc checkpoint insertion.
    expected_ids = tuple(sorted(str(case_id) for case_id in expected_case_ids))  # Normalize the complete validation case identity set.
    if len(expected_ids) != VALIDATION_CASE_COUNT or len(set(expected_ids)) != VALIDATION_CASE_COUNT:  # Require exactly eight unique validation cases.
        raise ValueError("checkpoint selection requires exactly 8 unique validation case ids")  # Reject incomplete or duplicated validation evidence.
    expected_grid = {(case_id, budget) for case_id in expected_ids for budget in EQUATION_BUDGETS}  # Build the exact eight-by-three validation Cartesian product.
    observed_grid = {(str(item.case_id), int(item.equation_budget)) for item in observations}  # Normalize all supplied validation operating-point keys.
    if len(observations) != len(expected_grid) or observed_grid != expected_grid:  # Require every point once and no unauthorized budget or case.
        raise ValueError("validation observations must cover the exact 8 case by 3 budget grid")  # Prevent selective checkpoint scoring.
    failure_points = 0  # Count each invalid case-budget point once even when both metrics fail.
    budget_violations = 0  # Count every trajectory that attempted an over-budget solve.
    energy_logs: list[float] = []  # Accumulate fixed-penalty finite energy-error logarithms.
    qoi_logs: list[float] = []  # Accumulate fixed-penalty finite QoI-error logarithms.
    for item in sorted(observations, key=lambda value: (str(value.case_id), int(value.equation_budget))):  # Make the key independent of execution or serialization order.
        energy = _finite_nonnegative(item.energy_error) if item.ok and item.energy_ok else None  # Retain energy only from a complete rollout with a valid energy metric.
        qoi = _finite_nonnegative(item.qoi_error) if item.ok and item.qoi_ok else None  # Retain QoI only from a complete rollout with a valid QoI metric.
        failed = energy is None or qoi is None  # Classify the whole validation operating point conservatively.
        failure_points += int(failed)  # Retain every failed point in the first lexicographic component.
        budget_violations += int(bool(item.budget_violation))  # Retain every actual budget overshoot independently from metric validity.
        scored_energy = VALIDATION_FAILURE_ERROR if energy is None else energy  # Substitute the fixed finite energy penalty without infinity or NaN.
        scored_qoi = VALIDATION_FAILURE_ERROR if qoi is None else qoi  # Substitute the same fixed finite QoI penalty.
        energy_logs.append(math.log(max(scored_energy, ERROR_FLOOR)))  # Transform the finite nonnegative energy score safely into log space.
        qoi_logs.append(math.log(max(scored_qoi, ERROR_FLOOR)))  # Transform the finite nonnegative QoI score safely into log space.
    energy_log_mean = float(sum(energy_logs) / len(energy_logs))  # Give every validation case-budget point equal multiplicative weight.
    qoi_log_mean = float(sum(qoi_logs) / len(qoi_logs))  # Apply the identical aggregation rule to QoI error.
    if not math.isfinite(energy_log_mean) or not math.isfinite(qoi_log_mean):  # Defend the checkpoint artifact against any unrepresentable score.
        raise ValueError("validation selection metrics must be finite")  # Reject a malformed checkpoint instead of silently ranking it.
    return int(failure_points), energy_log_mean, qoi_log_mean, int(budget_violations), int(checkpoint_episode)  # Return the exact ascending lexicographic key.

def select_validation_checkpoint(reports: Sequence[Mapping[str, Any]], expected_case_ids: Sequence[str]) -> Mapping[str, Any]:  # Select one checkpoint after validating all twelve preregistered reports.
    if len(reports) != len(VALIDATION_EPISODES):  # Require every scheduled validation event before selection.
        raise ValueError("checkpoint selection requires all 12 validation reports")  # Prevent selection from a favorable partial training history.
    by_episode = {int(report["checkpoint_episode"]): report for report in reports}  # Index each report by its immutable training episode.
    if set(by_episode) != set(VALIDATION_EPISODES) or len(by_episode) != len(reports):  # Reject missing, duplicated, or off-schedule checkpoint episodes.
        raise ValueError("validation reports do not match the frozen checkpoint schedule")  # Preserve the pre-registered candidate set.
    ranked: list[tuple[tuple[int, float, float, int, int], Mapping[str, Any]]] = []  # Pair each validated report with its exact selection key.
    for episode in VALIDATION_EPISODES:  # Recompute every key from raw observations rather than trusting stored summaries.
        report = by_episode[episode]  # Read the unique report for this checkpoint.
        raw_observations = report.get("observations")  # Read the transparent per-case and per-budget validation evidence.
        if not isinstance(raw_observations, list):  # Require complete raw evidence for independent re-scoring.
            raise ValueError("validation report observations must be a list")  # Reject opaque checkpoint scores.
        observations = [ValidationObservation(**dict(item)) for item in raw_observations]  # Reconstruct immutable typed observations from strict JSON records.
        key = validation_selection_key(observations, episode, expected_case_ids)  # Recompute the finite lexicographic selection metric.
        ranked.append((key, report))  # Retain the auditable key beside its source report.
    selected_key, selected_report = min(ranked, key=lambda pair: pair[0])  # Apply ascending lexicographic minimization with earlier episode as the final tie-break.
    return {"checkpoint_episode": int(selected_report["checkpoint_episode"]), "checkpoint_file": str(selected_report["checkpoint_file"]), "selection_key": {"failure_points": selected_key[0], "finite_penalty_energy_log_mean": selected_key[1], "finite_penalty_qoi_log_mean": selected_key[2], "budget_violations": selected_key[3], "checkpoint_episode": selected_key[4]}}  # Return the sole selected checkpoint and named key components.

def _partition_path(partition_root: Path, case_id: str) -> Path:  # Resolve one shared partition spec without scanning unrelated or blind directories.
    return Path(partition_root) / str(case_id) / PARTITION_FILENAME  # Use the fixed per-case audit layout shared by WM and RL.

def _reference_ledger_path(reference_root: Path, case_id: str) -> Path:  # Resolve one common-reference ledger without inspecting other cases.
    return Path(reference_root) / str(case_id) / REFERENCE_LEDGER_FILENAME  # Use the frozen per-case reference cache layout.

def preflight_case_artifacts(cases: Sequence[Mapping[str, Any]], partition_root: Path, reference_root: Path | None = None) -> None:  # Fail before a phase starts if any authorized case artifact is absent.
    missing: list[str] = []  # Collect all missing paths so one preflight reports the complete repair list.
    for case in cases:  # Inspect only the split explicitly supplied by the phase caller.
        case_id = str(case["case_id"])  # Read the immutable manifest identity without opening another split.
        partition_path = _partition_path(partition_root, case_id)  # Resolve this case's exact shared WM/RL partition spec.
        if not partition_path.is_file():  # Require partition freezing before training, validation, or test execution.
            missing.append(str(partition_path))  # Retain the exact missing partition artifact.
        if reference_root is not None:  # Require Reference B only for validation or blind-test rollouts.
            ledger_path = _reference_ledger_path(reference_root, case_id)  # Resolve this case's validated common-reference ledger.
            if not ledger_path.is_file():  # Refuse to build or tune a reference inside RL model selection or testing.
                missing.append(str(ledger_path))  # Retain the exact missing reference artifact.
    if missing:  # Stop before the first real solver call when protocol artifacts are incomplete.
        raise FileNotFoundError("missing frozen RL input artifacts:\n" + "\n".join(sorted(missing)))  # Report every authorized missing file deterministically.

def _verify_phase_inputs(cases: Sequence[Mapping[str, Any]], partition_root: Path, reference_root: Path | None = None, reference_policy: Mapping[str, Any] | None = None) -> None:  # Authenticate partitions and optional references under an explicit qualification policy before a long phase.
    preflight_case_artifacts(cases, partition_root, reference_root)  # Report all missing authorized files before parsing individual artifacts.
    normalized_reference_policy = _normalize_reference_policy(reference_policy)  # Preserve strict default behavior unless the fixed amendment was explicitly authenticated.
    for case in cases:  # Verify cases only from the explicitly authorized phase split.
        problem, _partitioner = _load_shared_partitioner(case, partition_root)  # Authenticate geometry identity, assignment, order, and fixed adjacency without solving.
        if reference_root is not None:  # Validate a common Reference B only for selection or blind-test cases.
            from ..vla.four_way_references import load_reference_b  # Import the strict cache verifier lazily at the data boundary.
            load_reference_b(reference_root, case_id=str(case["case_id"]), problem=problem, runner=None, verify=True, regenerate_meshes=False, allow_unqualified=normalized_reference_policy["allow_unqualified_references"], expedited_levels=normalized_reference_policy["expedited_reference_levels"])  # Authenticate the ledger under exactly the strict or amended schedule without native work.

def _load_shared_partitioner(case: Mapping[str, Any], partition_root: Path) -> tuple[Any, Any]:  # Reconstruct one Problem and its exact shared frozen region graph.
    from ..vla.partition_spec import PartitionSpecRegistry  # Import the sole canonical per-case WM/RL partition registry when a real rollout starts.
    problem = problem_from_case(case)  # Reconstruct the canonical manifest-bound bridge geometry and loading.
    registry = PartitionSpecRegistry(partition_root)  # Bind resolution to the configured frozen partition root without directory scanning.
    shared = registry.partitioner_for(str(case["case_id"]), problem, str(case["geometry_hash"]))  # Verify geometry identity, region order, assignment, probe contract, and fixed adjacency before use.
    return problem, shared  # Return the problem and the exact adapter shared with WM-VLA.

def _load_reference_b(reference_root: Path, case: Mapping[str, Any], problem: Any, runner: FemRunner, reference_policy: Mapping[str, Any] | None = None) -> Any:  # Load and inject Reference B only under an explicit strict or amended qualification policy.
    from ..vla.four_way_references import load_reference_b  # Import the canonical paired-reference verifier only for validation or testing.
    normalized_reference_policy = _normalize_reference_policy(reference_policy)  # Default every omitted caller to strict qualified evidence.
    return load_reference_b(reference_root, case_id=str(case["case_id"]), problem=problem, runner=runner, allow_unqualified=normalized_reference_policy["allow_unqualified_references"], expedited_levels=normalized_reference_policy["expedited_reference_levels"])  # Verify amendment, status, and ledger bytes before binding the common denominator.

def _sanitized_failure(exception: BaseException, stage: str, failure_at_solve: int) -> dict[str, Any]:  # Preserve an explicit finite numerical-failure receipt without a traceback or secret-bearing environment dump.
    message = str(exception).replace("\x00", " ")[:500]  # Bound and sanitize the direct exception message for machine-readable evidence.
    native_receipt: dict[str, Any] = {}  # Collect only the typed backend's bounded non-secret provenance fields.
    if isinstance(exception, CalculiXExecutionError):  # Preserve the explicit CalculiX failure contract without introspecting arbitrary exceptions.
        native_receipt = {"returncode": exception.returncode, "native_wall_s": float(exception.wall_s), "log_path": str(exception.log_path), "workdir": str(exception.workdir)}  # Retain native status, elapsed time, and durable log location.
    return {"stage": str(stage), "exception_type": type(exception).__name__, "message": message, "failure_at_solve": int(max(failure_at_solve, 1)), "native_receipt": native_receipt}  # Return only the auditable failure classification, location, and typed native provenance.

def _runner_solve_attempts(runner: FemRunner) -> int:  # Count successful records plus an invocation that failed after FemRunner incremented its counter.
    counter = int(getattr(runner, "_counter", len(runner.records)))  # Read the common runner's honest invocation counter when available.
    return max(counter, len(runner.records))  # Never undercount successfully persisted solve records.

def _environment_config(base: DQNConfig, budget: int) -> DQNConfig:  # Derive one active-budget environment while preserving every learned hyperparameter.
    if int(budget) not in EQUATION_BUDGETS:  # Restrict state normalization and stopping penalties to frozen public budgets.
        raise ValueError("RL environment budget is not in the frozen budget set")  # Reject hidden train or test budgets.
    return replace(base, n_eq_budget=int(budget), max_steps=MAX_REGION_REFINEMENTS)  # Encode budget utilization explicitly and cap total real solves at six.

def _epsilon(episode: int, config: DQNConfig) -> float:  # Compute the existing linear exploration schedule from the global one-based episode index.
    fraction = min((int(episode) - 1) / max(float(config.eps_decay_frac) * TRAIN_EPISODES, 1.0), 1.0)  # Decay once across the complete mixed-budget training stream.
    return float(config.eps_start + (config.eps_end - config.eps_start) * fraction)  # Return the deterministic epsilon used for every action in this episode.

def _validate_graph_state(state: np.ndarray, adjacency: np.ndarray) -> None:  # Reject malformed or nonfinite region-graph states as retained numerical failures.
    if state.ndim != 2 or adjacency.shape != (state.shape[0], state.shape[0]) or state.shape[0] < 1:  # Require one feature row per region and one square graph in the same order.
        raise FloatingPointError("RL graph state or adjacency has an invalid shape")  # Preserve action-vector ambiguity as an explicit numerical failure.
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(adjacency)):  # Prevent NaN or infinity from reaching Q values or replay.
        raise FloatingPointError("RL graph state or adjacency contains a nonfinite value")  # Retain the affected episode or rollout instead of silently choosing an action.

def _policy_action(policy: DQNPolicy, state: np.ndarray, adjacency: np.ndarray, epsilon: float, rng: random.Random | None = None) -> int:  # Select an action while explicitly rejecting nonfinite Q vectors.
    _validate_graph_state(state, adjacency)  # Validate all graph features before exploration or inference.
    if rng is not None and rng.random() < float(epsilon):  # Apply the existing epsilon-greedy random branch from the policy-local RNG.
        return int(rng.randrange(state.shape[0] + 1))  # Sample one refine-region or stop action uniformly.
    q_values = np.asarray(policy.q_values(state, adjacency), dtype=float)  # Evaluate the frozen or online Q network without gradients.
    if q_values.shape != (state.shape[0] + 1,) or not np.all(np.isfinite(q_values)):  # Require one finite Q value per region plus stop.
        raise FloatingPointError("RL policy produced an invalid or nonfinite Q vector")  # Retain numerical network failure rather than letting argmax hide it.
    return int(np.argmax(q_values))  # Use NumPy's deterministic first-index tie break for greedy selection.

def _safe_estimator_from_log(log_eta: Any) -> float | None:  # Convert a diagnostic log estimator without allowing overflow or nonfinite JSON.
    if log_eta is None:  # Preserve absence when the mandatory probe did not complete.
        return None  # Return an explicit missing diagnostic.
    numeric = float(log_eta)  # Normalize supported scalar wrappers to a Python float.
    if not math.isfinite(numeric):  # Reject NaN and infinity before exponentiation.
        return None  # Preserve the invalid estimator as missing rather than fabricating a score.
    try:  # Catch floating-point overflow from an otherwise finite extreme logarithm.
        value = float(math.exp(numeric))  # Convert the valid log estimator into its positive scale.
    except OverflowError:  # Treat unrepresentable values as numerical failure diagnostics.
        return None  # Keep strict JSON output finite.
    return value if math.isfinite(value) else None  # Return only a representable finite estimator.

def _run_training_episode(policy: DQNPolicy, rng: random.Random, base_config: DQNConfig, case: Mapping[str, Any], budget: int, partition_root: Path, workdir: Path, episode: int) -> dict[str, Any]:  # Execute one complete or explicitly failed real-solver training episode.
    problem, partitioner = _load_shared_partitioner(case, partition_root)  # Bind the manifest geometry to the exact frozen WM/RL region graph.
    runner = FemRunner(problem, workdir, keep_files=False, ccx_timeout=1800.0)  # Isolate all real solves for honest per-episode accounting.
    environment = RegionRefineEnv(runner, partitioner, _environment_config(base_config, budget), method="rl_dqn_train")  # Create one budget-conditioned fixed-partition Markov episode.
    epsilon = _epsilon(episode, base_config)  # Freeze one exploration probability for the full episode.
    reward_total = 0.0  # Accumulate rewards only from transitions actually completed.
    updates_before = int(policy.grad_steps)  # Snapshot the optimizer update count for exact episode attribution.
    failure: dict[str, Any] | None = None  # Initialize the retained numerical-failure receipt.
    stage = "probe"  # Identify the next solver-bearing operation for failure reporting.
    try:  # Retain numerical failures and continue the fixed three-hundred-attempt schedule.
        policy.q.train()  # Restore training mode after any prior checkpoint validation.
        state, adjacency = environment.reset()  # Execute the mandatory uniform-probe Gmsh mesh and real CalculiX solve.
        done = False  # Continue until stop, budget termination, or five completed region refinements.
        while not done:  # Finish the complete environment episode rather than truncating at logging intervals.
            action = _policy_action(policy, state, adjacency, epsilon, rng)  # Select one validated epsilon-greedy region-refine or stop action.
            stage = "stop" if action == state.shape[0] else f"region_refine_{environment.steps + 1}"  # Name the next action before any real remesh and solve.
            (next_state, next_adjacency), reward, done, _info = environment.step(action)  # Execute stop or one real Gmsh-plus-CalculiX region refinement.
            policy.push(Transition(state, adjacency, action, reward, next_state, next_adjacency, done))  # Retain the completed transition in the bounded replay buffer.
            loss = policy.learn(rng)  # Perform at most one Double-DQN gradient update from this completed transition.
            if loss is not None and not math.isfinite(float(loss)):  # Detect a completed but numerically invalid optimizer update explicitly.
                raise FloatingPointError("RL Double-DQN update produced a nonfinite loss")  # Retain this scheduled episode as a numerical failure without replacement.
            state, adjacency = next_state, next_adjacency  # Advance the graph state only after a completed environment transition.
            reward_total += float(reward)  # Add the finite realized reward for cost and stability diagnostics.
    except RETAINED_NUMERICAL_FAILURE_TYPES as exception:  # Preserve only explicitly typed native solver or mesher numerical failures without replacing the scheduled case or budget.
        failure = _sanitized_failure(exception, stage, _runner_solve_attempts(runner) or 1)  # Record the failed invocation location and bounded message.
    solve_attempts = _runner_solve_attempts(runner)  # Count all successful and explicitly failed solver invocations.
    final_record = runner.records[-1] if runner.records else None  # Read final resource metadata only when at least one solve completed.
    log_eta = getattr(environment, "log_eta", None)  # Read the estimator state only when the probe or a refinement completed.
    final_eta = _safe_estimator_from_log(log_eta)  # Convert only a representable finite estimator diagnostic.
    return {"episode": int(episode), "case_id": str(case["case_id"]), "equation_budget": int(budget), "epsilon": epsilon, "status": "ok" if failure is None else "numerical_failure", "failure": failure, "reward": float(reward_total), "final_eta": final_eta, "final_n_equations": None if final_record is None else int(final_record.n_equations), "solve_attempts": int(solve_attempts), "successful_solves": len(runner.records), "gradient_updates": int(policy.grad_steps) - updates_before}  # Return a strict finite episode-cost and outcome record.

def _freeze_policy_for_inference(policy: DQNPolicy) -> None:  # Disable gradient state changes before checkpoint validation or blind testing.
    policy.q.eval()  # Put the online Q network into deterministic inference mode.
    policy.q_target.eval()  # Keep the target network aligned with inference-only execution.
    for parameter in policy.q.parameters():  # Visit every online-network parameter exactly once.
        parameter.requires_grad_(False)  # Prevent accidental backward updates during greedy evaluation.
    for parameter in policy.q_target.parameters():  # Visit every target-network parameter exactly once.
        parameter.requires_grad_(False)  # Prevent accidental target-network updates during greedy evaluation.

def _unfreeze_policy_for_training(policy: DQNPolicy) -> None:  # Restore the online and target parameters after validation without changing their values.
    for parameter in policy.q.parameters():  # Visit every online-network parameter exactly once.
        parameter.requires_grad_(True)  # Re-enable gradients for subsequent scheduled training episodes.
    for parameter in policy.q_target.parameters():  # Visit every target-network parameter exactly once.
        parameter.requires_grad_(True)  # Preserve the existing learner's target-sync behavior.
    policy.q.train()  # Return the online network to training mode.

def _run_greedy_trajectory(policy: DQNPolicy, base_config: DQNConfig, case: Mapping[str, Any], budget: int, partition_root: Path, reference_root: Path, workdir: Path, method: str, reference_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:  # Run one frozen no-learning trajectory under an explicit reference policy and retain every prefix.
    runner: FemRunner | None = None  # Initialize the runner so pre-solve failures can still produce a complete result row.
    failure: dict[str, Any] | None = None  # Initialize the retained numerical-failure receipt.
    stage = "load_partition"  # Identify the next integration boundary before the common probe.
    started = time.perf_counter()  # Measure partition loading, reference verification, meshing, solving, and policy inference together.
    try:  # Retain each numerical failure instead of aborting validation or selecting a replacement blind case.
        problem, partitioner = _load_shared_partitioner(case, partition_root)  # Bind the exact manifest geometry and shared region graph.
        runner = FemRunner(problem, workdir, keep_files=False, ccx_timeout=1800.0)  # Isolate honest solve accounting for this case, seed, and budget.
        stage = "load_reference_b"  # Identify common-reference verification separately from method execution.
        _load_reference_b(reference_root, case, problem, runner, reference_policy)  # Inject only a qualified or explicitly amended authenticated Reference B without a solve.
        environment = RegionRefineEnv(runner, partitioner, _environment_config(base_config, budget), method=method)  # Create the fixed-partition active-budget deployment environment.
        stage = "probe"  # Identify the mandatory first real Gmsh-plus-CalculiX solve.
        state, adjacency = environment.reset()  # Execute the common uniform probe and initialize graph features.
        done = False  # Continue until greedy stop, budget termination, or the common six-solve cap.
        while not done:  # Run the frozen greedy policy without replay insertion or gradient updates.
            action = _policy_action(policy, state, adjacency, 0.0, None)  # Choose the validated deterministic first maximum-Q action with epsilon exactly zero.
            stage = "stop" if action == state.shape[0] else f"region_refine_{environment.steps + 1}"  # Name the next action before any real solve attempt.
            (state, adjacency), _reward, done, _info = environment.step(action)  # Execute stop or exactly one real Gmsh remesh plus CalculiX solve.
    except RETAINED_NUMERICAL_FAILURE_TYPES as exception:  # Preserve only explicit native Gmsh or CalculiX numerical failures while continuing fixed validation or test order.
        attempts = 0 if runner is None else _runner_solve_attempts(runner)  # Read any solver invocation already attempted before failure.
        failure = _sanitized_failure(exception, stage, attempts or 1)  # Record a bounded auditable failure receipt.
    records = [] if runner is None else list(runner.records)  # Preserve only successfully completed common-runner records in solve order.
    solve_attempts = 0 if runner is None else _runner_solve_attempts(runner)  # Count a failed solver invocation even when it produced no record.
    return {"case_id": str(case["case_id"]), "equation_budget": int(budget), "status": "ok" if failure is None else "numerical_failure", "failure": failure, "failure_at_solve": None if failure is None else int(failure["failure_at_solve"]), "solve_attempts": int(solve_attempts), "successful_solves": len(records), "records": records, "wall_s": float(time.perf_counter() - started), "learning_updates": 0, "cross_case_learning": False}  # Return the raw trajectory needed for validation and every test prefix.

def _best_prefix_metric(records: Sequence[Any], attribute: str, solves: int, budget: int) -> float | None:  # Select the best valid Reference-B error in one real-solve and equation-budget prefix.
    eligible: list[float] = []  # Collect finite nonnegative metric values only from genuinely feasible solves.
    for record in records:  # Inspect every successful real solve in its persisted trajectory order.
        if int(record.solve_index) > int(solves) or int(record.n_equations) > int(budget):  # Exclude later prefixes and over-budget solves without substitution.
            continue  # Advance to the next completed solve record.
        value = _finite_nonnegative(getattr(record, attribute, None))  # Validate the requested Reference-B metric.
        if value is not None:  # Retain only finite nonnegative successful errors.
            eligible.append(value)  # Preserve this feasible deliverable candidate.
    return None if not eligible else float(min(eligible))  # Apply the protocol's best-feasible-prefix definition independently per metric.

def trajectory_prefix_result(trajectory: Mapping[str, Any], solves: int) -> dict[str, Any]:  # Convert one raw greedy trajectory into one auditable public operating-point result.
    if int(solves) not in SOLVE_PREFIXES:  # Restrict derived results to the frozen four-prefix grid.
        raise ValueError("RL solve prefix is not in the frozen set 2,3,4,6")  # Reject post-hoc solve-count selection.
    budget = int(trajectory["equation_budget"])  # Read the active budget used in state normalization and termination.
    records = trajectory.get("records")  # Read successful common-runner solve records without copying another method's results.
    if not isinstance(records, list):  # Require the raw trajectory representation produced by the frozen rollout.
        raise ValueError("RL trajectory records must be a list")  # Reject opaque or malformed prefix evidence.
    failure_at = trajectory.get("failure_at_solve")  # Read the first explicitly failed real-solve attempt when present.
    failure_affects_prefix = failure_at is not None and int(failure_at) <= int(solves)  # Invalidate only prefixes that reach an actual numerical failure.
    energy = None if failure_affects_prefix else _best_prefix_metric(records, "e_energy", solves, budget)  # Preserve completed earlier prefixes while penalizing affected later prefixes.
    qoi = None if failure_affects_prefix else _best_prefix_metric(records, "e_qoi", solves, budget)  # Apply identical failure and feasibility semantics to QoI.
    inspected_records = [record for record in records if int(record.solve_index) <= int(solves)]  # Restrict resource diagnostics to the requested real-solve prefix.
    budget_violation = any(int(record.n_equations) > budget for record in inspected_records)  # Disclose every actual attempted over-budget solve in this prefix.
    energy_ok = energy is not None  # Publish a separate machine-readable energy validity flag.
    qoi_ok = qoi is not None  # Publish a separate machine-readable QoI validity flag.
    return {"case_id": str(trajectory["case_id"]), "solves": int(solves), "equation_budget": budget, "energy_error": energy, "qoi_error": qoi, "energy_ok": bool(energy_ok), "qoi_ok": bool(qoi_ok), "budget_violation": bool(budget_violation), "failure_affects_prefix": bool(failure_affects_prefix), "successful_solves_available": len(inspected_records), "hold_last_after_stop": len(records) < int(solves) and trajectory.get("status") == "ok"}  # Return all scoring and provenance fields for one operating point.

def _validation_observation(trajectory: Mapping[str, Any]) -> ValidationObservation:  # Score one validation trajectory at the sole checkpoint-selection prefix K equals six.
    prefix = trajectory_prefix_result(trajectory, MAX_REAL_SOLVES)  # Apply the public best-feasible six-solve prefix rule.
    complete_ok = trajectory.get("status") == "ok"  # Distinguish complete numerical execution from metric-specific availability.
    failure_value = trajectory.get("failure")  # Read the sanitized rollout failure receipt when present.
    return ValidationObservation(case_id=str(trajectory["case_id"]), equation_budget=int(trajectory["equation_budget"]), energy_error=prefix["energy_error"], qoi_error=prefix["qoi_error"], ok=bool(complete_ok), budget_violation=bool(prefix["budget_violation"]), solve_attempts=int(trajectory["solve_attempts"]), energy_ok=bool(prefix["energy_ok"]), qoi_ok=bool(prefix["qoi_ok"]), failure=None if failure_value is None else dict(failure_value))  # Return the exact typed input to lexicographic checkpoint selection.

def _validate_policy(policy: DQNPolicy, base_config: DQNConfig, validation_cases: Sequence[Mapping[str, Any]], partition_root: Path, reference_root: Path, workdir: Path, checkpoint_episode: int, reference_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:  # Evaluate one checkpoint on the full grid under an explicit validation-reference policy.
    if len(validation_cases) != VALIDATION_CASE_COUNT:  # Require the full validation split before any checkpoint scoring.
        raise ValueError("RL checkpoint validation requires exactly 8 manifest validation cases")  # Prevent favorable validation subsets.
    _freeze_policy_for_inference(policy)  # Disable gradients and stochastic training behavior for all twenty-four greedy rollouts.
    normalized_reference_policy = _normalize_reference_policy(reference_policy)  # Canonicalize strict or amended intent once for all twenty-four points.
    observations: list[ValidationObservation] = []  # Accumulate one retained point for every case and budget, including failures.
    started = time.perf_counter()  # Measure the complete checkpoint-validation cost separately from training.
    for case in validation_cases:  # Preserve frozen manifest validation order.
        for budget in EQUATION_BUDGETS:  # Evaluate all three budgets rather than selecting a favorable operating resource.
            trajectory_dir = workdir / str(case["case_id"]) / f"budget_{budget}"  # Isolate solver files by validation case and budget.
            trajectory = _run_greedy_trajectory(policy, base_config, case, budget, partition_root, reference_root, trajectory_dir, "rl_dqn_validation", normalized_reference_policy)  # Run a fresh no-learning greedy environment with the identical audited denominator policy.
            observations.append(_validation_observation(trajectory))  # Retain a valid point or fixed-penalty failure input.
    expected_ids = [str(case["case_id"]) for case in validation_cases]  # Preserve the exact authorized validation identity set for key verification.
    selection_key = validation_selection_key(observations, checkpoint_episode, expected_ids)  # Compute the preregistered finite lexicographic score.
    return {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "checkpoint_validation", "checkpoint_episode": int(checkpoint_episode), "validation_reference_policy": normalized_reference_policy, "observations": [asdict(item) for item in observations], "selection_key": {"failure_points": selection_key[0], "finite_penalty_energy_log_mean": selection_key[1], "finite_penalty_qoi_log_mean": selection_key[2], "budget_violations": selection_key[3], "checkpoint_episode": selection_key[4]}, "real_solve_attempts": int(sum(item.solve_attempts for item in observations)), "wall_s": float(time.perf_counter() - started), "learning_updates": 0, "test_split_access": False}  # Return raw observations, exact reference intent, cost, and named selection fields.

def _base_dqn_config(seed: int) -> DQNConfig:  # Construct the sole preregistered region-graph Double-DQN hyperparameter set for one seed.
    if int(seed) not in RL_SEEDS:  # Restrict training and model loading to the three frozen independent seeds.
        raise ValueError("RL seed is not in the frozen seed set")  # Reject hidden seed search.
    return DQNConfig(gamma_refine=0.7, max_steps=MAX_REGION_REFINEMENTS, n_eq_budget=max(EQUATION_BUDGETS), discount=0.95, lr=1.0e-3, batch_size=32, replay_size=4000, target_sync=200, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.6, hidden=32, reward_error_scale=10.0, reward_solve_cost=0.15, reward_budget_penalty=2.0, reward_stop_bonus=1.0, seed=int(seed))  # Preserve the existing learner while fixing the common six-solve and budget-conditioned contracts.

def _assert_empty_training_output(output_dir: Path) -> None:  # Prevent model replacement after any checkpoint or validation evidence exists.
    if output_dir.exists():  # Treat even an empty pre-existing directory as a prior or ambiguous training attempt.
        raise FileExistsError(f"RL training output already exists: {output_dir}")  # Refuse an implicit resume, overwrite, or result-dependent retraining.

def _train_one_seed(seed: int, training_cases: Sequence[Mapping[str, Any]], validation_cases: Sequence[Mapping[str, Any]], partition_root: Path, reference_root: Path, output_dir: Path, reference_policy: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:  # Train one policy and freeze its checkpoint under an explicit validation-reference policy.
    import torch  # Import Torch only when actual model training is explicitly requested.
    base_config = _base_dqn_config(seed)  # Build the unique hyperparameter record for this independent seed.
    normalized_reference_policy = _normalize_reference_policy(reference_policy)  # Freeze the identical strict or amended validation denominator policy for this seed.
    random.seed(seed)  # Seed Python's process-global generator for repository code that may consult it.
    np.random.seed(seed)  # Seed NumPy's legacy process-global generator for repository code that may consult it.
    torch.manual_seed(seed)  # Seed Torch before constructing the online and target graph networks.
    rng = random.Random(seed)  # Isolate epsilon actions and replay sampling in one auditable policy-local stream.
    policy = DQNPolicy(base_config)  # Construct the existing region-graph GCN Double-DQN learner.
    seed_dir = output_dir / f"seed_{seed}"  # Isolate all training and validation evidence for this policy seed.
    checkpoints_dir = seed_dir / "checkpoints"  # Retain every one of the twelve eligible pre-selection model states.
    checkpoints_dir.mkdir(parents=True, exist_ok=False)  # Create a new immutable checkpoint directory before episode one.
    episode_history: list[dict[str, Any]] = []  # Retain all three hundred attempted complete episodes, including numerical failures.
    validation_reports: list[dict[str, Any]] = []  # Retain every scheduled eight-case-by-three-budget validation event.
    training_started = time.perf_counter()  # Measure training plus scheduled validation while separately reporting each component.
    validation_wall_s = 0.0  # Accumulate checkpoint-validation wall time independently from optimizer training.
    validation_solves = 0  # Accumulate all real validation solver attempts independently from training episodes.
    for episode in range(1, TRAIN_EPISODES + 1):  # Attempt exactly three hundred episodes without replacing failed cases.
        case, budget = training_assignment(episode, training_cases)  # Apply the fixed case rotation and literal budget cycle.
        episode_dir = seed_dir / "episodes" / f"episode_{episode:04d}"  # Isolate real solver calls for this scheduled episode.
        episode_record = _run_training_episode(policy, rng, base_config, case, budget, partition_root, episode_dir, episode)  # Complete the episode or retain its explicit numerical failure.
        episode_history.append(episode_record)  # Preserve the scheduled episode regardless of outcome.
        if episode in VALIDATION_EPISODES:  # Validate after every twenty-five attempted complete episodes.
            checkpoint_path = checkpoints_dir / f"episode_{episode:04d}.pt"  # Name the exact pre-validation policy state.
            policy.save(checkpoint_path)  # Persist the model before validation so the evidence can be independently reproduced.
            report = _validate_policy(policy, base_config, validation_cases, partition_root, reference_root, seed_dir / "validation" / f"episode_{episode:04d}", episode, normalized_reference_policy)  # Evaluate all twenty-four points greedily under the exact audited denominator policy.
            report["checkpoint_file"] = str(checkpoint_path.relative_to(output_dir))  # Bind validation evidence to its portable checkpoint path.
            report["checkpoint_sha256"] = _sha256_file(checkpoint_path)  # Bind validation evidence to the exact binary model bytes.
            validation_reports.append(report)  # Retain this eligible checkpoint's complete transparent score.
            validation_wall_s += float(report["wall_s"])  # Accumulate reference verification, meshing, solving, and inference time.
            validation_solves += int(report["real_solve_attempts"])  # Count every successful and explicitly failed validation solver invocation.
            _write_json(seed_dir / "validation" / f"episode_{episode:04d}.json", report)  # Persist checkpoint evidence before later training can continue.
            _unfreeze_policy_for_training(policy)  # Re-enable gradients without altering any learned parameter values.
    training_total_wall_s = float(time.perf_counter() - training_started)  # Measure the complete scheduled seed run once.
    expected_ids = [str(case["case_id"]) for case in validation_cases]  # Preserve the exact validation identity set for independent key recomputation.
    selected = dict(select_validation_checkpoint(validation_reports, expected_ids))  # Select one and only one checkpoint through the frozen lexicographic rule.
    selected_checkpoint = output_dir / str(selected["checkpoint_file"])  # Resolve the selected portable checkpoint path inside this new run.
    selected["checkpoint_sha256"] = _sha256_file(selected_checkpoint)  # Bind selection metadata to the exact candidate checkpoint bytes.
    frozen_model = output_dir / MODEL_FILENAME_TEMPLATE.format(seed=seed)  # Resolve this seed's stable frozen policy artifact.
    shutil.copy2(selected_checkpoint, frozen_model)  # Copy exact selected bytes without retraining or reserialization.
    selected["model_file"] = frozen_model.name  # Record a path relative to the freeze index for portable verification.
    selected["model_sha256"] = _sha256_file(frozen_model)  # Bind blind testing to the exact selected model bytes.
    selected["seed"] = int(seed)  # Bind model identity to its sole independent training seed.
    selected["budget_conditioned"] = True  # Declare that this single policy is evaluated under all three active budget states.
    selected["validation_reference_policy"] = dict(normalized_reference_policy)  # Bind model selection to the exact qualified or amended denominator evidence.
    training_solves = int(sum(int(item["solve_attempts"]) for item in episode_history))  # Count successful and explicitly failed real training solver calls.
    gradient_updates = int(sum(int(item["gradient_updates"]) for item in episode_history))  # Count all completed optimizer updates from actual transitions.
    failure_count = int(sum(item["status"] != "ok" for item in episode_history))  # Retain the number of explicit numerical training failures.
    budget_counts = {str(budget): sum(int(item["equation_budget"]) == budget for item in episode_history) for budget in EQUATION_BUDGETS}  # Reconfirm exact one-hundred-episode exposure per budget from executed history.
    if len(episode_history) != TRAIN_EPISODES or any(count != 100 for count in budget_counts.values()):  # Refuse to freeze an incomplete or schedule-drifted policy.
        raise RuntimeError("RL episode history violates the frozen 300 episode and 100 per budget contract")  # Stop before producing a model index usable by blind testing.
    history_payload = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "training_history", "seed": int(seed), "validation_reference_policy": normalized_reference_policy, "episodes_requested": TRAIN_EPISODES, "episodes_attempted": len(episode_history), "numeric_failure_episodes": failure_count, "budget_counts": budget_counts, "episodes": episode_history}  # Assemble complete outcomes and the separate checkpoint denominator policy.
    _write_json(seed_dir / "training_history.json", history_payload)  # Persist all attempted episodes independently from summary aggregation.
    _write_json(seed_dir / "validation_history.json", {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "validation_history", "seed": int(seed), "validation_reference_policy": normalized_reference_policy, "reports": validation_reports, "selected": selected})  # Persist every candidate, selection, and exact denominator qualification policy.
    parameter_count = int(sum(parameter.numel() for parameter in policy.q.parameters()))  # Count trainable architecture scalars independently from the selected weight values.
    cost = {"seed": int(seed), "validation_reference_policy": normalized_reference_policy, "episodes_attempted": len(episode_history), "numeric_failure_episodes": failure_count, "training_real_solve_attempts": training_solves, "validation_real_solve_attempts": validation_solves, "gradient_updates": gradient_updates, "training_and_validation_wall_s": training_total_wall_s, "validation_wall_s": float(validation_wall_s), "optimizer_training_wall_s": max(training_total_wall_s - validation_wall_s, 0.0), "budget_episode_counts": budget_counts, "model_parameter_count": parameter_count, "selected_checkpoint_episode": int(selected["checkpoint_episode"]), "selected_validation_key": dict(selected["selection_key"])}  # Return all offline costs beside the exact validation-reference policy.
    return selected, cost  # Return the frozen model identity and its independently auditable cost record.

def _three_seed_dispersion(costs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:  # Summarize validation-selected variability while retaining all three independent policies.
    if len(costs) != len(RL_SEEDS) or {int(cost["seed"]) for cost in costs} != set(RL_SEEDS):  # Require exact seed coverage before any dispersion or freeze computation.
        raise ValueError("RL cost dispersion requires exactly the three frozen seeds")  # Prevent incomplete or best-seed summaries.
    ordered = sorted(costs, key=lambda cost: int(cost["seed"]))  # Make dispersion independent of sequential or parallel completion order.
    selected_energy_scores = [float(cost["selected_validation_key"]["finite_penalty_energy_log_mean"]) for cost in ordered]  # Collect all three selected energy log means.
    selected_qoi_scores = [float(cost["selected_validation_key"]["finite_penalty_qoi_log_mean"]) for cost in ordered]  # Collect all three selected QoI log means.
    if any(not math.isfinite(value) for value in selected_energy_scores + selected_qoi_scores):  # Require finite values from the fixed-penalty selection contract.
        raise ValueError("RL selected validation dispersion inputs must be finite")  # Reject malformed shard metadata before freezing.
    return {"selected_energy_log_mean_population_std": float(statistics.pstdev(selected_energy_scores)), "selected_energy_log_mean_min": float(min(selected_energy_scores)), "selected_energy_log_mean_max": float(max(selected_energy_scores)), "selected_qoi_log_mean_population_std": float(statistics.pstdev(selected_qoi_scores)), "selected_qoi_log_mean_min": float(min(selected_qoi_scores)), "selected_qoi_log_mean_max": float(max(selected_qoi_scores)), "seed_count": len(RL_SEEDS)}  # Report spread without selecting a best seed.

def _publish_model_set(manifest_file: Path, destination: Path, selections: Sequence[Mapping[str, Any]], costs: Sequence[Mapping[str, Any]], campaign_wall_s: float, execution_mode: str, source_shards: Sequence[Mapping[str, Any]] = (), reference_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:  # Publish one unified three-model report with exact validation-reference intent.
    if len(selections) != len(RL_SEEDS) or {int(item["seed"]) for item in selections} != set(RL_SEEDS):  # Require one selected checkpoint from every frozen seed.
        raise ValueError("RL model-set publication requires exactly the three frozen seeds")  # Prevent best-seed or duplicated-seed freezes.
    ordered_selections = [dict(item) for item in sorted(selections, key=lambda item: int(item["seed"]))]  # Freeze model order independently from shard completion timing.
    ordered_costs = [dict(item) for item in sorted(costs, key=lambda item: int(item["seed"]))]  # Freeze cost order independently from shard completion timing.
    normalized_reference_policy = _verify_live_reference_policy(manifest_file, reference_policy)  # Rehash any amendment immediately before unified model and cost freeze publication.
    if any(item.get("validation_reference_policy") != normalized_reference_policy for item in (*ordered_selections, *ordered_costs)):  # Require every selected model and cost summary to disclose identical reference intent.
        raise ValueError("RL model selections or costs disagree on validation reference policy")  # Reject mixed strict/amended seed sets before freeze publication.
    dispersion = _three_seed_dispersion(ordered_costs)  # Compute variability across all and only the three frozen policies.
    sum_seed_wall_s = float(sum(float(cost["training_and_validation_wall_s"]) for cost in ordered_costs))  # Report total consumed seed wall time for cost accounting.
    max_seed_wall_s = float(max(float(cost["training_and_validation_wall_s"]) for cost in ordered_costs))  # Report the lower-bound makespan attainable under full three-way parallelism.
    cost_payload = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "training_cost", "execution_mode": str(execution_mode), "validation_reference_policy": normalized_reference_policy, "episodes_per_seed": TRAIN_EPISODES, "total_episode_attempts": int(sum(int(cost["episodes_attempted"]) for cost in ordered_costs)), "total_numeric_failure_episodes": int(sum(int(cost["numeric_failure_episodes"]) for cost in ordered_costs)), "total_training_real_solve_attempts": int(sum(int(cost["training_real_solve_attempts"]) for cost in ordered_costs)), "total_validation_real_solve_attempts": int(sum(int(cost["validation_real_solve_attempts"]) for cost in ordered_costs)), "total_gradient_updates": int(sum(int(cost["gradient_updates"]) for cost in ordered_costs)), "campaign_wall_s": float(campaign_wall_s), "sum_seed_training_and_validation_wall_s": sum_seed_wall_s, "parallel_seed_makespan_lower_bound_s": max_seed_wall_s, "per_seed": ordered_costs, "three_seed_dispersion": dispersion, "source_shards": [dict(item) for item in source_shards]}  # Assemble all costs and the exact shared validation-reference policy.
    cost_path = destination / "rl_training_cost.json"  # Resolve the protocol-required aggregate offline-cost artifact.
    _write_json(cost_path, cost_payload)  # Persist costs before publishing the blind-test model index.
    base_configs = {str(seed): asdict(_base_dqn_config(seed)) for seed in RL_SEEDS}  # Freeze every hyperparameter including seed and maximum state budget.
    freeze_index = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "model_freeze", "TEST_NOT_RUN": True, "manifest_file": manifest_file.name, "manifest_sha256": _sha256_file(manifest_file), "training_split_count": TRAIN_CASE_COUNT, "validation_split_count": VALIDATION_CASE_COUNT, "test_split_access": False, "validation_reference_policy": normalized_reference_policy, "models": ordered_selections, "seeds": list(RL_SEEDS), "equation_budgets": list(EQUATION_BUDGETS), "solve_prefixes": list(SOLVE_PREFIXES), "max_real_solves": MAX_REAL_SOLVES, "policies": "three independent budget-conditioned region-graph Double-DQN models", "seed_merge_rule": "retain_all_three_without_cross_seed_selection", "finite_element_contract": {"nodal_gradation": COMMON_NODAL_GRADATION, "source": "PR40_V0_default_tool_behavior"}, "episode_contract": {"episodes_per_seed": TRAIN_EPISODES, "budget_schedule": "literal_cycle_30000_60000_120000", "episodes_per_budget_per_seed": 100, "numeric_failures_retained_without_replacement": True}, "checkpoint_contract": {"interval": VALIDATION_INTERVAL, "eligible_episodes": list(VALIDATION_EPISODES), "validation_grid": "8 manifest validation cases x 3 equation budgets", "ascending_lexicographic_order": ["failure_points", "finite_penalty_energy_log_mean", "finite_penalty_qoi_log_mean", "budget_violations", "checkpoint_episode"], "failure_error_penalty": VALIDATION_FAILURE_ERROR, "error_floor": ERROR_FLOOR}, "test_contract": {"case_count": TEST_CASE_COUNT, "case_order": "ascending_case_id", "greedy_epsilon": 0.0, "learning_updates": 0, "cross_case_learning": False, "one_trajectory_per_case_seed_budget": True, "prefix_delivery": "independent_best_feasible_energy_and_qoi", "three_seed_aggregation": "pointwise_median_per_case_K_B_with_failures_ranked_after_finite_results"}, "dqn_configs": base_configs, "training_cost_file": cost_path.name, "training_cost_sha256": _sha256_file(cost_path), "three_seed_dispersion": dispersion, "source_shards": [dict(item) for item in source_shards]}  # Freeze model, reference, finite-element, merge, selection, budget, test, and aggregation rules before blind execution.
    index_path = destination / "rl_freeze_index.json"  # Resolve the sole model-set identity consumed by blind testing.
    _write_json(index_path, freeze_index)  # Publish model hashes and TEST_NOT_RUN only after all three complete seeds verify.
    index_digest = _sha256_file(index_path)  # Hash the exact freeze-index bytes independently from its model hashes.
    index_path.with_suffix(".sha256").write_text(f"{index_digest}  {index_path.name}\n", encoding="ascii")  # Publish a standard sha256sum-compatible sidecar.
    return {"freeze_index": str(index_path), "freeze_index_sha256": index_digest, "model_count": len(ordered_selections), "training_cost": str(cost_path), "validation_reference_policy": normalized_reference_policy, "TEST_NOT_RUN": True}  # Return concise automation paths and exact reference intent without opening the blind split.

def train_bridge_rl(manifest_path: Path | str, partition_root: Path | str, reference_root: Path | str, output_dir: Path | str, *, allow_unqualified_references: bool = False, expedited_reference_levels: int | None = None) -> dict[str, Any]:  # Train three policies under a strict-default or fixed amended validation-reference policy.
    manifest_file = Path(manifest_path)  # Normalize the immutable manifest path for checksum verification and hashing.
    reference_policy = build_training_reference_policy(manifest_file, allow_unqualified_references=allow_unqualified_references, expedited_reference_levels=expedited_reference_levels)  # Authenticate explicit waiver intent and human amendment before cache access.
    partitions = Path(partition_root)  # Normalize the shared partition root used by both WM-VLA and RL.
    references = Path(reference_root)  # Normalize the already completed common Reference-B root.
    destination = Path(output_dir)  # Normalize the new model and audit-artifact directory.
    manifest = load_case_manifest(manifest_file, verify_checksum=True)  # Authenticate the complete frozen manifest before selecting authorized splits.
    training_cases = manifest_cases(manifest, "train")  # Isolate the complete twenty-four-case training split.
    validation_cases = manifest_cases(manifest, "validation")  # Isolate the complete eight-case validation split.
    _verify_phase_inputs(training_cases, partitions)  # Authenticate every training shared partition before episode one.
    _verify_phase_inputs(validation_cases, partitions, references, reference_policy)  # Authenticate every validation reference under the exact strict or amended schedule before episode one.
    _assert_empty_training_output(destination)  # Prevent overwrite, continuation, or post-result policy replacement.
    destination.mkdir(parents=True, exist_ok=False)  # Create one new immutable training evidence directory.
    plan = build_training_plan(manifest, reference_policy)  # Materialize the exact schedule and reference qualification choice without blind access.
    _write_json(destination / "rl_training_plan.json", plan)  # Persist pre-solve intent before the first Gmsh or CalculiX call.
    selections: list[dict[str, Any]] = []  # Accumulate one selected model identity per frozen seed.
    costs: list[dict[str, Any]] = []  # Accumulate complete offline-cost evidence per frozen seed.
    run_started = time.perf_counter()  # Measure the complete three-policy training campaign.
    for seed in RL_SEEDS:  # Train the exact three independent seeds in preregistered order.
        selected, cost = _train_one_seed(seed, training_cases, validation_cases, partitions, references, destination, reference_policy)  # Execute one complete history under the identical audited validation denominator policy.
        selections.append(selected)  # Retain the sole lexicographically selected model for this seed.
        costs.append(cost)  # Retain training solves, updates, timing, failures, and validation cost.
    return _publish_model_set(manifest_file, destination, selections, costs, float(time.perf_counter() - run_started), "sequential_three_seed_training", reference_policy=reference_policy)  # Publish models, costs, and reference intent without cross-seed selection.

def _write_sha256_sidecar(path: Path) -> str:  # Publish a standard exact-byte digest beside one JSON index artifact.
    digest = _sha256_file(path)  # Recompute the exact persisted bytes independently from any payload field.
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")  # Write a sha256sum-compatible digest and unambiguous local filename.
    return digest  # Return the frozen identity for parent receipts.

def train_bridge_rl_seed(manifest_path: Path | str, partition_root: Path | str, reference_root: Path | str, output_dir: Path | str, seed: int, *, allow_unqualified_references: bool = False, expedited_reference_levels: int | None = None) -> dict[str, Any]:  # Train one shard under a strict-default or fixed amended validation-reference policy.
    if int(seed) not in RL_SEEDS:  # Accept only the three preregistered independent seeds.
        raise ValueError(f"--seed must be one of {list(RL_SEEDS)}")  # Reject seed search before reading model inputs.
    manifest_file = Path(manifest_path)  # Normalize the immutable checksummed manifest path.
    reference_policy = build_training_reference_policy(manifest_file, allow_unqualified_references=allow_unqualified_references, expedited_reference_levels=expedited_reference_levels)  # Authenticate the fixed human amendment before any validation cache read.
    partitions = Path(partition_root)  # Normalize the exact shared WM/RL partition registry.
    references = Path(reference_root)  # Normalize the completed common Reference-B registry.
    destination = Path(output_dir)  # Normalize this seed's new independent shard directory.
    manifest = load_case_manifest(manifest_file, verify_checksum=True)  # Authenticate the frozen design before selecting authorized splits.
    training_cases = manifest_cases(manifest, "train")  # Isolate the complete twenty-four-case training split.
    validation_cases = manifest_cases(manifest, "validation")  # Isolate the complete eight-case validation split.
    _verify_phase_inputs(training_cases, partitions)  # Authenticate every training shared partition before episode one.
    _verify_phase_inputs(validation_cases, partitions, references, reference_policy)  # Authenticate every validation reference under the exact selected policy before episode one.
    _assert_empty_training_output(destination)  # Require a brand-new shard without resume or overwrite.
    destination.mkdir(parents=True, exist_ok=False)  # Create only this fixed-seed shard directory.
    plan = build_training_plan(manifest, reference_policy)  # Materialize the reviewed schedule and exact reference qualification choice.
    shard_plan = {**plan, "phase": "seed_training_plan", "shard_seed": int(seed), "policies": [policy for policy in plan["policies"] if int(policy["seed"]) == int(seed)]}  # Restrict execution intent to one frozen seed while preserving the common contract.
    _write_json(destination / "rl_training_plan.json", shard_plan)  # Persist shard intent before the first real solver call.
    started = time.perf_counter()  # Measure this complete three-hundred-episode plus validation shard.
    selected, cost = _train_one_seed(int(seed), training_cases, validation_cases, partitions, references, destination, reference_policy)  # Execute exactly one complete history under the audited denominator policy.
    reference_policy = _verify_live_reference_policy(manifest_file, reference_policy)  # Rehash the canonical amendment after training and before sealing an independently transferable shard.
    shard_wall_s = float(time.perf_counter() - started)  # Measure the complete shard independently from later assembly.
    cost_payload = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "seed_training_cost", "seed": int(seed), "validation_reference_policy": reference_policy, "shard_wall_s": shard_wall_s, "cost": cost}  # Preserve every cost component beside the exact validation-reference policy.
    cost_path = destination / "rl_training_cost.json"  # Resolve the shard-local cost artifact.
    _write_json(cost_path, cost_payload)  # Persist cost evidence before sealing the shard index.
    seed_root = destination / f"seed_{seed}"  # Resolve the shard-local detailed training and validation history directory.
    training_history_path = seed_root / "training_history.json"  # Resolve all three hundred retained episode outcomes.
    validation_history_path = seed_root / "validation_history.json"  # Resolve all twelve raw validation grids and selected checkpoint.
    model_path = destination / MODEL_FILENAME_TEMPLATE.format(seed=seed)  # Resolve the selected budget-conditioned model bytes.
    artifacts = {"model": {"path": model_path.name, "sha256": _sha256_file(model_path)}, "training_history": {"path": str(training_history_path.relative_to(destination)), "sha256": _sha256_file(training_history_path)}, "validation_history": {"path": str(validation_history_path.relative_to(destination)), "sha256": _sha256_file(validation_history_path)}, "training_cost": {"path": cost_path.name, "sha256": _sha256_file(cost_path)}}  # Bind every required shard artifact to exact bytes.
    shard_index = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "seed_shard", "TEST_NOT_RUN": True, "test_split_access": False, "seed": int(seed), "manifest_file": manifest_file.name, "manifest_sha256": _sha256_file(manifest_file), "validation_reference_policy": reference_policy, "dqn_config": asdict(_base_dqn_config(int(seed))), "finite_element_contract": {"nodal_gradation": COMMON_NODAL_GRADATION, "source": "PR40_V0_default_tool_behavior"}, "episode_contract": {"episodes": TRAIN_EPISODES, "budgets": list(EQUATION_BUDGETS), "episodes_per_budget": 100}, "checkpoint_contract": {"episodes": list(VALIDATION_EPISODES), "validation_case_count": VALIDATION_CASE_COUNT, "budgets": list(EQUATION_BUDGETS), "selection_order": ["failure_points", "finite_penalty_energy_log_mean", "finite_penalty_qoi_log_mean", "budget_violations", "checkpoint_episode"]}, "selected": selected, "cost": cost, "artifacts": artifacts}  # Seal model, schedule, cost, and exact validation-reference evidence for assembly.
    index_path = destination / "rl_seed_shard.json"  # Resolve the sole shard completion and integrity index.
    _write_json(index_path, shard_index)  # Publish the complete shard only after histories, cost, and model exist.
    digest = _write_sha256_sidecar(index_path)  # Publish and return the exact shard-index identity.
    return {"seed_shard": str(index_path), "seed_shard_sha256": digest, "seed": int(seed), "model": str(model_path), "episodes_attempted": int(cost["episodes_attempted"]), "validation_reference_policy": reference_policy, "TEST_NOT_RUN": True}  # Return a concise completion receipt with the exact pre-freeze amendment identity.

def _read_json_mapping(path: Path) -> dict[str, Any]:  # Load one strict JSON object for shard or freeze verification.
    payload = json.loads(path.read_text(encoding="utf-8"))  # Decode the transparent UTF-8 artifact.
    if not isinstance(payload, dict):  # Require named schema fields at the top level.
        raise ValueError(f"JSON artifact must contain an object: {path}")  # Reject scalar or list substitutions.
    return payload  # Return the mutable local mapping for validation only.

def _verify_index_sidecar(path: Path) -> str:  # Authenticate one shard or freeze index against its standard sidecar.
    observed = _sha256_file(path)  # Hash the exact current index bytes.
    fields = path.with_suffix(".sha256").read_text(encoding="ascii").strip().split()  # Parse the adjacent digest and filename.
    if len(fields) != 2 or fields[1] != path.name or fields[0] != observed:  # Require an exact unambiguous sidecar match.
        raise ValueError(f"invalid SHA-256 sidecar for {path}")  # Reject incomplete, renamed, or altered shard metadata.
    return observed  # Return the independently verified index identity.

def _resolve_shard_artifact(shard_root: Path, value: Any) -> Path:  # Resolve one declared shard artifact while forbidding traversal or absolute paths.
    relative = Path(str(value))  # Normalize the transparent relative artifact path.
    if relative.is_absolute() or ".." in relative.parts:  # Forbid leaving the reviewed shard tree.
        raise ValueError("RL shard artifact path must remain inside its shard directory")  # Reject external or parent traversal references.
    candidate = shard_root / relative  # Resolve the artifact below the supplied shard root.
    try:  # Confirm the normalized path remains a descendant even with redundant separators.
        candidate.resolve().relative_to(shard_root.resolve())  # Raise when resolution escapes the shard directory.
    except ValueError as exception:  # Convert traversal into the explicit shard-validation contract.
        raise ValueError("RL shard artifact resolves outside its shard directory") from exception  # Reject the unsafe declaration.
    return candidate  # Return the verified descendant path.

def _verify_seed_shard(shard_root: Path, manifest_file: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:  # Recompute one seed shard's model, report, schedule, and selection integrity before assembly.
    root = Path(shard_root)  # Normalize the caller-supplied independent shard directory.
    index_path = root / "rl_seed_shard.json"  # Resolve the required shard completion index.
    index_sha = _verify_index_sidecar(index_path)  # Authenticate exact shard metadata before interpreting paths.
    index = _read_json_mapping(index_path)  # Decode the authenticated transparent shard contract.
    if index.get("schema") != RL_SCHEMA or index.get("protocol_id") != PROTOCOL_ID or index.get("phase") != "seed_shard":  # Require the exact seed-shard schema and protocol.
        raise ValueError("RL seed shard schema or protocol is invalid")  # Reject unrelated or stale training outputs.
    if index.get("TEST_NOT_RUN") is not True or index.get("test_split_access") is not False:  # Require a genuine pre-blind-test shard.
        raise ValueError("RL seed shard is not a pre-test artifact")  # Prevent post-test retraining or selection evidence.
    seed = int(index.get("seed", -1))  # Normalize the shard's sole independent training seed.
    if seed not in RL_SEEDS or index.get("dqn_config") != asdict(_base_dqn_config(seed)):  # Require an exact frozen seed and executable hyperparameters.
        raise ValueError("RL seed shard seed or DQN configuration is invalid")  # Reject seed search or configuration drift.
    if index.get("finite_element_contract") != {"nodal_gradation": COMMON_NODAL_GRADATION, "source": "PR40_V0_default_tool_behavior"}:  # Require the shared V0 nodal size-field smoothing behavior.
        raise ValueError("RL seed shard nodal gradation differs from the frozen common value")  # Reject method- or shard-specific gradation drift.
    expected_episode_contract = {"episodes": TRAIN_EPISODES, "budgets": list(EQUATION_BUDGETS), "episodes_per_budget": 100}  # Rebuild the sole accepted shard episode contract.
    expected_checkpoint_contract = {"episodes": list(VALIDATION_EPISODES), "validation_case_count": VALIDATION_CASE_COUNT, "budgets": list(EQUATION_BUDGETS), "selection_order": ["failure_points", "finite_penalty_energy_log_mean", "finite_penalty_qoi_log_mean", "budget_violations", "checkpoint_episode"]}  # Rebuild the sole accepted checkpoint-selection contract.
    if index.get("episode_contract") != expected_episode_contract or index.get("checkpoint_contract") != expected_checkpoint_contract:  # Require exact schedule and selection metadata independently from raw cardinality checks.
        raise ValueError("RL seed shard episode or checkpoint contract differs from preregistration")  # Reject altered shard intent even when reports look complete.
    if str(index.get("manifest_sha256")) != _sha256_file(manifest_file):  # Bind the shard to the exact same manifest bytes used by assembly.
        raise ValueError("RL seed shard manifest SHA-256 mismatch")  # Reject resampled or reordered training inputs.
    try:  # Add shard context while preserving the shared exact-byte reauthentication contract.
        declared_reference_policy = _verify_live_reference_policy(manifest_file, index.get("validation_reference_policy"))  # Rehash the canonical amendment before trusting an exceptional shard.
    except ValueError as exception:  # Convert stale or malformed policy evidence into an explicit shard-validation failure.
        raise ValueError("RL seed shard validation reference amendment no longer matches the canonical file") from exception  # Reject stale, replaced, partially copied, or malformed user authorization.
    artifact_values = index.get("artifacts")  # Read exact-byte artifact declarations once.
    required_artifacts = ("model", "training_history", "validation_history", "training_cost")  # Freeze the minimum self-auditing shard content.
    if not isinstance(artifact_values, Mapping) or any(name not in artifact_values for name in required_artifacts):  # Require every model and report declaration.
        raise ValueError("RL seed shard lacks required model or report artifacts")  # Reject incomplete shard uploads.
    artifact_paths: dict[str, Path] = {}  # Resolve and authenticate all required files by semantic name.
    for name in required_artifacts:  # Verify exact bytes for the selected model and each required report.
        record = artifact_values[name]  # Read this artifact's path and hash declaration.
        if not isinstance(record, Mapping) or "path" not in record or "sha256" not in record:  # Require both provenance fields.
            raise ValueError(f"RL seed shard artifact {name} is malformed")  # Reject opaque artifact declarations.
        artifact_path = _resolve_shard_artifact(root, record["path"])  # Constrain the artifact to this independent shard.
        if not artifact_path.is_file() or _sha256_file(artifact_path) != str(record["sha256"]):  # Recompute exact bytes rather than trusting the index.
            raise ValueError(f"RL seed shard artifact hash mismatch: {name}")  # Reject missing or altered model or report data.
        artifact_paths[name] = artifact_path  # Retain the authenticated local path for semantic validation.
    training_history = _read_json_mapping(artifact_paths["training_history"])  # Load all scheduled training episode outcomes.
    if training_history.get("schema") != RL_SCHEMA or training_history.get("protocol_id") != PROTOCOL_ID or training_history.get("phase") != "training_history" or int(training_history.get("seed", -1)) != seed:  # Require the exact seed-bound history schema.
        raise ValueError("RL shard training history identity is invalid")  # Reject cross-seed report substitution.
    if training_history.get("validation_reference_policy") != declared_reference_policy:  # Require raw episode evidence to disclose the same checkpoint denominator contract.
        raise ValueError("RL shard training history validation reference policy mismatch")  # Reject metadata stripped or changed after training.
    episodes_value = training_history.get("episodes")  # Read transparent per-episode records for full schedule recomputation.
    if not isinstance(episodes_value, list) or len(episodes_value) != TRAIN_EPISODES:  # Require exactly three hundred retained attempts.
        raise ValueError("RL shard must retain exactly 300 training episodes")  # Reject truncated or extended training.
    training_cases = manifest_cases(manifest, "train")  # Reconstruct the sole authorized training split from the authenticated manifest.
    for expected_episode, episode_row in enumerate(episodes_value, start=1):  # Verify every scheduled case and literal budget without sampling replacements.
        if not isinstance(episode_row, Mapping) or int(episode_row.get("episode", -1)) != expected_episode:  # Require unique contiguous one-based episode order.
            raise ValueError("RL shard training episode order or identity is invalid")  # Reject omitted, duplicated, or reordered attempts.
        expected_case, expected_budget = training_assignment(expected_episode, training_cases)  # Recompute the sole preregistered assignment for this episode.
        if str(episode_row.get("case_id")) != str(expected_case["case_id"]) or int(episode_row.get("equation_budget", -1)) != expected_budget:  # Require exact case and budget execution.
            raise ValueError("RL shard training case or budget schedule differs from preregistration")  # Reject case replacement after a numerical failure.
        if episode_row.get("status") not in ("ok", "numerical_failure") or int(episode_row.get("solve_attempts", -1)) < 0 or int(episode_row.get("gradient_updates", -1)) < 0:  # Require explicit retained outcomes and nonnegative costs.
            raise ValueError("RL shard training episode outcome or cost is invalid")  # Reject dropped failures or malformed accounting.
        failure_value = episode_row.get("failure")  # Read the explicit numerical-failure receipt when the episode did not complete.
        if (episode_row.get("status") == "numerical_failure") != isinstance(failure_value, Mapping):  # Require exactly failed episodes to retain structured failure evidence.
            raise ValueError("RL shard numerical-failure receipt is inconsistent with episode status")  # Reject failures hidden behind successful status or dropped diagnostics.
    executed_budget_counts = {str(budget): sum(int(row["equation_budget"]) == budget for row in episodes_value) for budget in EQUATION_BUDGETS}  # Recompute exact one-hundred-episode exposure per budget.
    if executed_budget_counts != {str(budget): 100 for budget in EQUATION_BUDGETS} or training_history.get("budget_counts") != executed_budget_counts:  # Require both raw and summary schedule agreement.
        raise ValueError("RL shard budget cardinality is not exactly 100 per budget")  # Reject an incomplete budget-conditioned policy.
    validation_history = _read_json_mapping(artifact_paths["validation_history"])  # Load all twelve raw checkpoint-validation grids.
    if validation_history.get("schema") != RL_SCHEMA or validation_history.get("protocol_id") != PROTOCOL_ID or validation_history.get("phase") != "validation_history" or int(validation_history.get("seed", -1)) != seed:  # Require exact seed-bound validation identity.
        raise ValueError("RL shard validation history identity is invalid")  # Reject cross-seed checkpoint reports.
    if validation_history.get("validation_reference_policy") != declared_reference_policy:  # Bind the complete checkpoint history to the shard's authenticated qualification choice.
        raise ValueError("RL shard validation history reference policy mismatch")  # Reject a selected model whose denominators came from different evidence.
    reports = validation_history.get("reports")  # Read the complete eligible checkpoint candidate set.
    if not isinstance(reports, list):  # Require transparent raw observations for recomputation.
        raise ValueError("RL shard validation reports must be a list")  # Reject opaque summary-only selection.
    validation_ids = [str(case["case_id"]) for case in manifest_cases(manifest, "validation")]  # Reconstruct the sole authorized eight-case validation set.
    recomputed = dict(select_validation_checkpoint(reports, validation_ids))  # Recompute all twelve finite lexicographic keys and the selected episode.
    recorded_selected = validation_history.get("selected")  # Read the model-selection record persisted by the shard.
    if not isinstance(recorded_selected, Mapping):  # Require named selected checkpoint and model fields.
        raise ValueError("RL shard selected checkpoint record is missing")  # Reject incomplete model provenance.
    if index.get("selected") != recorded_selected:  # Require the sealed shard index and independently hashed validation history to name the same selected model.
        raise ValueError("RL shard index and validation report disagree on selected checkpoint")  # Reject post-report model substitution.
    if recorded_selected.get("validation_reference_policy") != declared_reference_policy:  # Require the selected checkpoint receipt to retain the same explicit denominator contract.
        raise ValueError("RL shard selected checkpoint reference policy mismatch")  # Reject a model record detached from its qualification evidence.
    for key in ("checkpoint_episode", "checkpoint_file", "selection_key"):  # Compare every scientifically relevant recomputed selection component.
        if recorded_selected.get(key) != recomputed.get(key):  # Require exact no-test checkpoint selection.
            raise ValueError("RL shard selected checkpoint does not match recomputed validation ordering")  # Reject post-hoc or best-seed checkpoint substitution.
    for report in reports:  # Authenticate every eligible checkpoint model declared in validation history.
        if not isinstance(report, Mapping) or report.get("validation_reference_policy") != declared_reference_policy:  # Require every one of twelve validation grids to state the identical policy.
            raise ValueError("RL shard checkpoint report reference policy mismatch")  # Reject mixed qualified and amended candidates within one seed.
        checkpoint_path = _resolve_shard_artifact(root, report["checkpoint_file"])  # Resolve the model inside the independent shard only.
        if not checkpoint_path.is_file() or _sha256_file(checkpoint_path) != str(report.get("checkpoint_sha256")):  # Recompute every candidate's exact model hash.
            raise ValueError("RL shard validation checkpoint hash mismatch")  # Reject altered or missing selection candidates.
    selected_checkpoint_path = _resolve_shard_artifact(root, recomputed["checkpoint_file"])  # Resolve the uniquely selected no-test checkpoint.
    model_sha = _sha256_file(artifact_paths["model"])  # Recompute the shard's deployment model identity.
    if _sha256_file(selected_checkpoint_path) != model_sha or str(recorded_selected.get("checkpoint_sha256")) != model_sha or str(recorded_selected.get("model_sha256")) != model_sha or int(recorded_selected.get("seed", -1)) != seed:  # Require byte-for-byte selected-checkpoint copying and seed identity.
        raise ValueError("RL shard frozen model is not the recomputed selected checkpoint")  # Reject reserialization or cross-seed model replacement.
    if str(recorded_selected.get("model_file")) != str(artifact_values["model"]["path"]):  # Require selected metadata to name the authenticated deployment artifact.
        raise ValueError("RL shard selected model path differs from its artifact declaration")  # Reject path-level model substitution.
    _load_inference_policy(seed, artifact_paths["model"])  # Verify the selected bytes load into the exact frozen architecture before assembly.
    training_cost = _read_json_mapping(artifact_paths["training_cost"])  # Load the seed-level aggregate cost record.
    if training_cost.get("schema") != RL_SCHEMA or training_cost.get("protocol_id") != PROTOCOL_ID or training_cost.get("phase") != "seed_training_cost" or int(training_cost.get("seed", -1)) != seed:  # Require exact seed-cost identity.
        raise ValueError("RL shard training cost identity is invalid")  # Reject unrelated cost metadata.
    if training_cost.get("validation_reference_policy") != declared_reference_policy:  # Bind the independently hashed cost envelope to the same denominator policy.
        raise ValueError("RL shard training cost reference policy mismatch")  # Reject a cost report copied from another qualification mode.
    cost = training_cost.get("cost")  # Read the complete per-seed cost fields used by unified reporting.
    if not isinstance(cost, Mapping) or index.get("cost") != cost:  # Require index and independently hashed cost artifact agreement.
        raise ValueError("RL shard cost index and report differ")  # Reject an altered summary after shard sealing.
    if cost.get("validation_reference_policy") != declared_reference_policy:  # Require the per-seed scientific cost record to carry the same exact policy.
        raise ValueError("RL shard per-seed cost reference policy mismatch")  # Reject incomplete audit propagation before aggregation.
    expected_cost_values = {"episodes_attempted": TRAIN_EPISODES, "numeric_failure_episodes": sum(row["status"] != "ok" for row in episodes_value), "training_real_solve_attempts": sum(int(row["solve_attempts"]) for row in episodes_value), "validation_real_solve_attempts": sum(int(report["real_solve_attempts"]) for report in reports), "gradient_updates": sum(int(row["gradient_updates"]) for row in episodes_value), "budget_episode_counts": executed_budget_counts, "selected_checkpoint_episode": int(recomputed["checkpoint_episode"]), "selected_validation_key": dict(recomputed["selection_key"])}  # Recompute every cardinality, cost counter, and selection field available from raw reports.
    if any(cost.get(key) != value for key, value in expected_cost_values.items()):  # Require all recomputable cost fields to match raw evidence exactly.
        raise ValueError("RL shard cost counters or selected validation metadata are inconsistent")  # Reject a stale or edited summary.
    wall_fields = (cost.get("training_and_validation_wall_s"), cost.get("validation_wall_s"), cost.get("optimizer_training_wall_s"), training_cost.get("shard_wall_s"))  # Collect non-recomputable but required finite wall measurements.
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0 for value in wall_fields):  # Require finite nonnegative wall-clock evidence.
        raise ValueError("RL shard wall-clock cost fields are invalid")  # Reject NaN, infinity, negative, or missing timing.
    selected = dict(recorded_selected)  # Copy the fully verified selected model record for unified publication.
    return {"seed": seed, "root": root, "index_path": index_path, "index_sha256": index_sha, "model_path": artifact_paths["model"], "selected": selected, "cost": dict(cost), "reference_policy": declared_reference_policy, "receipt": {"seed": seed, "seed_shard_directory": root.name, "seed_shard_index": index_path.name, "seed_shard_sha256": index_sha, "model_sha256": model_sha, "training_history_sha256": str(artifact_values["training_history"]["sha256"]), "validation_history_sha256": str(artifact_values["validation_history"]["sha256"]), "training_cost_sha256": str(artifact_values["training_cost"]["sha256"]), "validation_reference_policy": declared_reference_policy}}  # Return internal source paths plus portable verified hashes and the live-audited policy without selecting among seeds.

def assemble_bridge_rl(manifest_path: Path | str, seed_shard_dirs: Sequence[Path | str], output_dir: Path | str) -> dict[str, Any]:  # Merge exactly three independently complete fixed-seed shards without best-seed selection.
    if len(seed_shard_dirs) != len(RL_SEEDS):  # Require one supplied directory per frozen seed.
        raise ValueError("RL assembly requires exactly three --seed-shard directories")  # Reject incomplete or additional candidate seed search.
    manifest_file = Path(manifest_path)  # Normalize and authenticate the same frozen manifest used by all shards.
    manifest = load_case_manifest(manifest_file, verify_checksum=True)  # Validate exact schema, cases, hashes, and sidecar before shard inspection.
    destination = Path(output_dir)  # Normalize the brand-new unified model-set directory.
    _assert_empty_training_output(destination)  # Refuse overwrite or reassembly into a prior freeze directory.
    assembly_started = time.perf_counter()  # Measure read-only verification, model copying, and unified publication.
    verified = [_verify_seed_shard(Path(shard), manifest_file, manifest) for shard in seed_shard_dirs]  # Recompute every shard's schedule, validation selection, hashes, and model loadability.
    if {int(item["seed"]) for item in verified} != set(RL_SEEDS):  # Require exact unique coverage of the three preregistered seeds.
        raise ValueError("RL assembly shards do not contain exactly the three frozen seeds")  # Reject duplicated, missing, or extra seed candidates without best-seed choice.
    reference_policy = dict(verified[0]["reference_policy"])  # Select the first fully authenticated policy only as the equality reference.
    if any(item["reference_policy"] != reference_policy for item in verified[1:]):  # Require all three fixed seeds to use the identical qualification mode and amendment bytes.
        raise ValueError("RL assembly cannot mix validation reference policies across seed shards")  # Reject cross-seed denominator drift rather than choosing a favorable policy.
    destination.mkdir(parents=True, exist_ok=False)  # Create the unified directory only after all shards pass complete verification.
    selections: list[dict[str, Any]] = []  # Accumulate all three selected models in frozen seed order.
    costs: list[dict[str, Any]] = []  # Accumulate all three independently recomputed seed costs.
    receipts: list[dict[str, Any]] = []  # Accumulate exact source-shard hashes for provenance.
    for item in sorted(verified, key=lambda value: int(value["seed"])):  # Copy all three policies in fixed seed order without ranking their scores.
        seed = int(item["seed"])  # Read this verified independent seed.
        target_model = destination / MODEL_FILENAME_TEMPLATE.format(seed=seed)  # Resolve the sole unified filename for this seed.
        shutil.copy2(item["model_path"], target_model)  # Copy exact selected bytes without reserialization or optimizer state.
        if _sha256_file(target_model) != str(item["selected"]["model_sha256"]):  # Recompute the copied deployment identity.
            raise RuntimeError("RL assembled model changed during copy")  # Stop before publishing a freeze index if byte identity changed.
        selected = dict(item["selected"])  # Copy the verified per-seed checkpoint-selection evidence.
        selected["model_file"] = target_model.name  # Bind the unified index to its local deployment artifact.
        selected["source_checkpoint_file"] = str(selected["checkpoint_file"])  # Preserve the original shard-local checkpoint path for provenance.
        selected["checkpoint_file"] = target_model.name  # Make the assembled selected-checkpoint path resolve to the identical local frozen bytes.
        selected["checkpoint_sha256"] = _sha256_file(target_model)  # Bind the assembled selected checkpoint to exact local model bytes.
        selections.append(selected)  # Retain this seed without cross-seed comparison or elimination.
        costs.append(dict(item["cost"]))  # Retain this seed's complete recomputed offline cost.
        receipts.append(dict(item["receipt"]))  # Retain source index and report hashes for independent audit.
    plan = build_training_plan(manifest, reference_policy)  # Recreate the common three-seed preregistration with the live-audited denominator contract.
    _write_json(destination / "rl_training_plan.json", {**plan, "phase": "assembled_training_plan", "execution_mode": "parallel_fixed_seed_shards", "source_seed_order": list(RL_SEEDS)})  # Preserve the unchanged scientific contract beside assembled models.
    receipts_path = destination / "rl_shard_receipts.json"  # Resolve the transparent merge-provenance artifact.
    _write_json(receipts_path, {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "seed_shard_receipts", "merge_rule": "retain_all_three_without_cross_seed_selection", "validation_reference_policy": reference_policy, "receipts": receipts})  # Publish every source shard, report hash, and common denominator policy without ranking seeds.
    assembly_wall_s = float(time.perf_counter() - assembly_started)  # Measure complete verification and assembly work.
    parallel_makespan = max(float(cost["training_and_validation_wall_s"]) for cost in costs) + assembly_wall_s  # Report the reproducible parallel seed makespan plus serial assembly.
    source_receipts = [{**receipt, "receipt_file": receipts_path.name, "receipt_file_sha256": _sha256_file(receipts_path)} for receipt in receipts]  # Bind each source record to the unified receipt bytes.
    return _publish_model_set(manifest_file, destination, selections, costs, parallel_makespan, "parallel_fixed_seed_shards", source_receipts, reference_policy)  # Publish all three models, costs, and exact shared denominator contract with no best-seed merge operation.

def _load_frozen_index(index_path: Path) -> tuple[dict[str, Any], dict[int, Path]]:  # Authenticate the model-set index and all three selected policy files before blind solving.
    encoded = index_path.read_bytes()  # Read exact freeze-index bytes before JSON decoding.
    sidecar_path = index_path.with_suffix(".sha256")  # Resolve the standard digest sidecar beside the index.
    sidecar_fields = sidecar_path.read_text(encoding="ascii").strip().split()  # Parse the required digest and filename fields.
    observed_index_sha = hashlib.sha256(encoded).hexdigest()  # Hash the exact bytes independently from the sidecar.
    if len(sidecar_fields) != 2 or sidecar_fields[1] != index_path.name or sidecar_fields[0] != observed_index_sha:  # Require a correct unambiguous index checksum.
        raise ValueError("RL freeze index SHA-256 sidecar is invalid")  # Stop before reading any blind reference or result.
    payload = json.loads(encoded.decode("utf-8"))  # Decode the authenticated transparent index.
    if not isinstance(payload, dict) or payload.get("schema") != RL_SCHEMA or payload.get("protocol_id") != PROTOCOL_ID or payload.get("phase") != "model_freeze":  # Require the exact model-freeze schema and protocol.
        raise ValueError("RL freeze index schema or protocol is invalid")  # Reject unrelated or stale model collections.
    if payload.get("TEST_NOT_RUN") is not True or payload.get("test_split_access") is not False:  # Require a genuine pre-test freeze record.
        raise ValueError("RL freeze index is not a pre-test TEST_NOT_RUN artifact")  # Prevent model selection from a post-test index.
    if tuple(payload.get("seeds", ())) != RL_SEEDS or tuple(payload.get("equation_budgets", ())) != EQUATION_BUDGETS or tuple(payload.get("solve_prefixes", ())) != SOLVE_PREFIXES:  # Require exact seed, budget, and prefix sets.
        raise ValueError("RL freeze index seed, budget, or solve-prefix contract differs from the protocol")  # Reject hidden policy or operating-point changes.
    if payload.get("finite_element_contract") != {"nodal_gradation": COMMON_NODAL_GRADATION, "source": "PR40_V0_default_tool_behavior"}:  # Require exact common size-field smoothing across training and deployment.
        raise ValueError("RL freeze index nodal gradation differs from the frozen common value")  # Reject method-specific or post-freeze gradation drift.
    reference_policy = _normalize_reference_policy(payload.get("validation_reference_policy"))  # Authenticate the strict or explicitly amended training-validation policy before model use.
    expected_configs = {str(seed): asdict(_base_dqn_config(seed)) for seed in RL_SEEDS}  # Rebuild the reviewed code's exact current hyperparameter contract.
    if payload.get("dqn_configs") != expected_configs:  # Require model metadata and executable configuration to remain identical.
        raise ValueError("RL executable DQN configuration differs from the frozen index")  # Reject code or hyperparameter drift after checkpoint selection.
    model_values = payload.get("models")  # Read the three selected model records once.
    if not isinstance(model_values, list) or len(model_values) != len(RL_SEEDS):  # Require one and only one budget-conditioned model per seed.
        raise ValueError("RL freeze index must contain exactly three models")  # Reject missing, duplicated, or per-budget best-model selection.
    model_paths: dict[int, Path] = {}  # Resolve authenticated model paths by frozen seed.
    for model in model_values:  # Verify every selected model identity before any test case is opened.
        if not isinstance(model, Mapping) or model.get("validation_reference_policy") != reference_policy:  # Require each selected policy to retain the freeze index's exact qualification evidence.
            raise ValueError("RL frozen model validation reference policy mismatch")  # Reject model metadata detached from checkpoint-selection denominators.
        seed = int(model["seed"])  # Normalize the model's declared independent seed.
        if seed not in RL_SEEDS or seed in model_paths:  # Require the exact unique three-seed set.
            raise ValueError("RL freeze index contains an invalid or duplicate seed")  # Reject seed search or duplication.
        model_name = str(model["model_file"])  # Read the portable model filename relative to the index directory.
        if Path(model_name).name != model_name:  # Forbid absolute paths and traversal outside the frozen model directory.
            raise ValueError("RL frozen model path must be one local filename")  # Keep model identity bound to the reviewed freeze directory.
        model_path = index_path.parent / model_name  # Resolve the frozen checkpoint beside its index.
        if not model_path.is_file() or _sha256_file(model_path) != str(model["model_sha256"]):  # Authenticate exact selected checkpoint bytes.
            raise ValueError(f"RL frozen model hash mismatch for seed {seed}")  # Stop before blind execution when any model differs.
        model_paths[seed] = model_path  # Retain the authenticated model path for fresh no-learning deployment.
    if set(model_paths) != set(RL_SEEDS):  # Require complete exact seed coverage after model verification.
        raise ValueError("RL frozen model seed set is incomplete")  # Reject an incomplete three-policy median basis.
    return payload, model_paths  # Return the authenticated index and exact model paths.

def _load_inference_policy(seed: int, model_path: Path) -> DQNPolicy:  # Construct one frozen greedy policy without replay or optimizer updates from authenticated bytes.
    policy = DQNPolicy(_base_dqn_config(seed))  # Recreate the exact reviewed region-graph architecture and state normalization contract.
    policy.load(model_path)  # Load only the selected online-network weights with Torch's weights-only boundary.
    _freeze_policy_for_inference(policy)  # Disable gradients and training-mode behavior before any case execution.
    if policy.grad_steps != 0 or policy.replay:  # Require a fresh evaluator with no optimizer or cross-case experience state.
        raise RuntimeError("fresh RL inference policy unexpectedly contains learning state")  # Stop before blind testing if evaluator isolation fails.
    return policy  # Return the deterministic epsilon-zero model used for every budget-conditioned environment.

def _assert_empty_test_output(output_dir: Path) -> None:  # Enforce one-way blind execution into a brand-new evidence directory.
    if output_dir.exists():  # Treat even an empty pre-existing directory as an ambiguous prior test attempt.
        raise FileExistsError(f"RL blind-test output already exists: {output_dir}")  # Refuse rerun, overwrite, or selective continuation under the same evidence identity.

def _verify_test_inputs(test_cases: Sequence[Mapping[str, Any]], partition_root: Path, reference_root: Path) -> None:  # Authenticate all test partitions and references before the one-time start marker and first method solve.
    if len(test_cases) != TEST_CASE_COUNT:  # Require all sixteen blind cases in sorted identity order.
        raise ValueError("RL blind test requires exactly 16 manifest test cases")  # Reject subsets and duplicates before any result exists.
    _verify_phase_inputs(test_cases, partition_root, reference_root)  # Authenticate all exact test artifacts in ascending case-id order without a solver call.

def _serialized_trajectory(trajectory: Mapping[str, Any], seed: int) -> dict[str, Any]:  # Convert common-runner dataclasses into one strict JSON raw-trajectory record.
    records_value = trajectory.get("records")  # Read the internal successful solve records once.
    if not isinstance(records_value, list):  # Require the exact trajectory representation returned by the frozen runner.
        raise ValueError("RL trajectory records must be a list before serialization")  # Reject malformed integration output.
    records = [asdict(record) for record in records_value]  # Preserve every solve count, equation count, error, timing, and stage field.
    return {"case_id": str(trajectory["case_id"]), "seed": int(seed), "equation_budget": int(trajectory["equation_budget"]), "status": str(trajectory["status"]), "failure": trajectory.get("failure"), "failure_at_solve": trajectory.get("failure_at_solve"), "solve_attempts": int(trajectory["solve_attempts"]), "successful_solves": int(trajectory["successful_solves"]), "wall_s": float(trajectory["wall_s"]), "learning_updates": int(trajectory["learning_updates"]), "cross_case_learning": bool(trajectory["cross_case_learning"]), "records": records}  # Return the complete raw no-learning trajectory evidence.

def _median_rows(prefix_rows: Sequence[Mapping[str, Any]], test_case_ids: Sequence[str]) -> list[dict[str, Any]]:  # Compute the required pointwise three-seed median for every blind case, K, and B.
    from ..vla.four_way_stats import failure_aware_median_error  # Reuse the single frozen failure-ranking and pointwise-median implementation.
    expected_keys = {(str(case_id), solves, budget) for case_id in test_case_ids for budget in EQUATION_BUDGETS for solves in SOLVE_PREFIXES}  # Build the exact sixteen-by-three-by-four public grid.
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}  # Group transparent seed outcomes without selecting any seed globally.
    for row in prefix_rows:  # Inspect every per-seed prefix result once.
        key = (str(row["case_id"]), int(row["solves"]), int(row["equation_budget"]))  # Normalize the pointwise median key.
        grouped.setdefault(key, []).append(row)  # Retain all three seed outcomes at this exact operating point.
    if set(grouped) != expected_keys:  # Require complete case, prefix, and budget coverage.
        raise ValueError("RL prefix rows do not cover the exact blind-test operating grid")  # Reject missing or unauthorized pointwise groups.
    medians: list[dict[str, Any]] = []  # Accumulate one deterministic median result per public group.
    for key in sorted(expected_keys, key=lambda value: (value[0], value[2], value[1])):  # Emit ascending case ID, budget, and solve-prefix order.
        outcomes = sorted(grouped[key], key=lambda row: int(row["seed"]))  # Put the exact three independent seeds into frozen identity order.
        if [int(row["seed"]) for row in outcomes] != list(RL_SEEDS):  # Require exactly one result from every frozen seed.
            raise ValueError(f"RL median group {key} does not contain exactly the three frozen seeds")  # Reject best-seed selection, duplication, or omission.
        energy = failure_aware_median_error([row.get("energy_error") for row in outcomes], [bool(row.get("energy_ok")) for row in outcomes])  # Rank failed seeds after all finite energy outcomes and select the middle policy pointwise.
        qoi = failure_aware_median_error([row.get("qoi_error") for row in outcomes], [bool(row.get("qoi_ok")) for row in outcomes])  # Apply the identical independent pointwise median rule to QoI.
        case_id, solves, budget = key  # Unpack the public operating-point identity after median selection.
        medians.append({"case_id": case_id, "method": "rl_median", "solves": int(solves), "equation_budget": int(budget), "energy_error": energy, "qoi_error": qoi, "energy_ok": energy is not None, "qoi_ok": qoi is not None, "seed_order": list(RL_SEEDS), "seed_energy_outcomes": [{"seed": int(row["seed"]), "value": row.get("energy_error"), "ok": bool(row.get("energy_ok"))} for row in outcomes], "seed_qoi_outcomes": [{"seed": int(row["seed"]), "value": row.get("qoi_error"), "ok": bool(row.get("qoi_ok"))} for row in outcomes], "seed_budget_violation_count": int(sum(bool(row.get("budget_violation")) for row in outcomes)), "aggregation": "pointwise_three_seed_median_failures_rank_after_finite"})  # Retain median values and all source seed outcomes for audit.
    return medians  # Return the complete ordered RL-median grid.

def test_bridge_rl(manifest_path: Path | str, partition_root: Path | str, reference_root: Path | str, freeze_index_path: Path | str, output_dir: Path | str) -> dict[str, Any]:  # Execute the three frozen greedy policies once on all sixteen sorted blind cases.
    manifest_file = Path(manifest_path)  # Normalize the same checksummed manifest used before model training.
    partitions = Path(partition_root)  # Normalize the exact shared WM/RL partition root.
    references = Path(reference_root)  # Normalize the completed common Reference-B root.
    index_path = Path(freeze_index_path)  # Normalize the immutable pre-test model index.
    destination = Path(output_dir)  # Normalize the brand-new blind-test evidence directory.
    frozen_index, model_paths = _load_frozen_index(index_path)  # Authenticate index, code configuration, and all model bytes before opening the blind split.
    if _sha256_file(manifest_file) != str(frozen_index["manifest_sha256"]):  # Require the exact same manifest bytes used for training and checkpoint selection.
        raise ValueError("blind-test manifest does not match the RL freeze index")  # Reject resampling, reordering, or case mutation.
    _assert_empty_test_output(destination)  # Refuse any rerun or overwrite before opening the manifest's blind split.
    manifest = load_case_manifest(manifest_file, verify_checksum=True)  # Authenticate manifest structure and sidecar after matching frozen identity.
    test_cases = manifest_cases(manifest, "test")  # Isolate and sort the sixteen blind cases only in the authorized test phase.
    _verify_test_inputs(test_cases, partitions, references)  # Authenticate every partition and common reference without a solver call.
    policies = {seed: _load_inference_policy(seed, model_paths[seed]) for seed in RL_SEEDS}  # Preload exactly three fresh epsilon-zero evaluators with no replay or gradient state.
    destination.mkdir(parents=True, exist_ok=False)  # Create a new one-way blind-test evidence directory.
    index_sha = _sha256_file(index_path)  # Bind the test start marker to the exact pre-test model collection.
    start_payload = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "blind_test_started", "freeze_index": index_path.name, "freeze_index_sha256": index_sha, "manifest_sha256": _sha256_file(manifest_file), "case_count": TEST_CASE_COUNT, "case_order": [str(case["case_id"]) for case in test_cases], "seeds": list(RL_SEEDS), "equation_budgets": list(EQUATION_BUDGETS), "solve_prefixes": list(SOLVE_PREFIXES), "greedy_epsilon": 0.0, "learning_updates_allowed": 0, "resume_or_rerun_allowed": False}  # Persist exact intent before the first blind method solve.
    _write_json(destination / "TEST_STARTED.json", start_payload)  # Make any partial execution visible and prevent silent repetition.
    raw_trajectories: list[dict[str, Any]] = []  # Accumulate all sixteen-by-three-by-three raw no-learning trajectories.
    prefix_rows: list[dict[str, Any]] = []  # Accumulate all sixteen-by-three-by-three-by-four public operating points.
    campaign_started = time.perf_counter()  # Measure the complete RL blind-test wall time.
    for case in test_cases:  # Execute cases strictly in ascending immutable case_id order.
        case_id = str(case["case_id"])  # Read the current frozen blind identity once.
        for budget in EQUATION_BUDGETS:  # Run one trajectory per active equation budget without reusing budget state.
            for seed in RL_SEEDS:  # Execute every frozen seed rather than selecting a favorable policy.
                trajectory_dir = destination / "trajectories" / case_id / f"budget_{budget}" / f"seed_{seed}"  # Isolate every solver ledger by the full no-learning trajectory identity.
                trajectory = _run_greedy_trajectory(policies[seed], _base_dqn_config(seed), case, budget, partitions, references, trajectory_dir, "rl_dqn_test")  # Execute one frozen greedy trajectory with all numerical failures retained.
                raw_trajectories.append(_serialized_trajectory(trajectory, seed))  # Preserve every common-runner solve and failure receipt.
                for solves in SOLVE_PREFIXES:  # Derive all four public prefixes from this single actual six-solve-capped trajectory.
                    prefix = trajectory_prefix_result(trajectory, solves)  # Apply independent best-feasible energy and QoI delivery rules.
                    prefix_rows.append({**prefix, "method": "rl_dqn", "seed": int(seed), "policy_model_sha256": str(next(model["model_sha256"] for model in frozen_index["models"] if int(model["seed"]) == seed)), "greedy": True, "learning_updates": 0, "cross_case_learning": False})  # Bind every prefix to its immutable policy and no-learning contract.
    expected_trajectory_count = TEST_CASE_COUNT * len(EQUATION_BUDGETS) * len(RL_SEEDS)  # Compute the exact retained raw-trajectory cardinality.
    expected_prefix_count = expected_trajectory_count * len(SOLVE_PREFIXES)  # Compute the exact per-seed public-grid cardinality.
    if len(raw_trajectories) != expected_trajectory_count or len(prefix_rows) != expected_prefix_count:  # Reject any accidental early stop or duplicated execution.
        raise RuntimeError("RL blind-test execution did not produce the exact frozen result cardinality")  # Stop before a misleading complete artifact is published.
    test_ids = [str(case["case_id"]) for case in test_cases]  # Preserve the sorted case set for exact pointwise grouping.
    median_rows = _median_rows(prefix_rows, test_ids)  # Compute the protocol-required per-case, per-K, per-B three-seed median without best-seed selection.
    total_wall_s = float(time.perf_counter() - campaign_started)  # Measure the completed blind-test campaign once.
    raw_payload = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "blind_test_complete", "freeze_index_sha256": index_sha, "manifest_sha256": _sha256_file(manifest_file), "case_order": test_ids, "execution_order": "case_id_then_budget_then_seed", "greedy_epsilon": 0.0, "learning_updates": 0, "cross_case_learning": False, "trajectory_count": len(raw_trajectories), "prefix_row_count": len(prefix_rows), "real_solve_attempts": int(sum(int(row["solve_attempts"]) for row in raw_trajectories)), "numeric_failure_trajectories": int(sum(row["status"] != "ok" for row in raw_trajectories)), "wall_s": total_wall_s, "trajectories": raw_trajectories, "prefix_rows": prefix_rows}  # Assemble complete raw and derived evidence without dropping failed trajectories.
    median_payload = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "rl_pointwise_median", "method": "rl_median", "group_keys": ["case_id", "solves", "equation_budget"], "seed_order": list(RL_SEEDS), "failure_rule": "failed outcomes rank after all finite nonnegative outcomes; median fails when at least two seeds fail", "best_seed_selection": False, "expected_row_count": TEST_CASE_COUNT * len(EQUATION_BUDGETS) * len(SOLVE_PREFIXES), "rows": median_rows}  # Assemble the transparent denominator consumed by four-way aggregation.
    raw_path = destination / "rl_test_results.json"  # Resolve the complete per-seed trajectory and prefix artifact.
    median_path = destination / "rl_median_results.json"  # Resolve the pointwise three-seed denominator artifact.
    _write_json(raw_path, raw_payload)  # Publish every counted solve and retained failure.
    _write_json(median_path, median_payload)  # Publish the exact pointwise median plus all three source outcomes.
    complete_payload = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "blind_test_complete_receipt", "TEST_NOT_RUN": False, "raw_results": raw_path.name, "raw_results_sha256": _sha256_file(raw_path), "median_results": median_path.name, "median_results_sha256": _sha256_file(median_path), "trajectory_count": len(raw_trajectories), "median_row_count": len(median_rows), "numeric_failure_trajectories": int(raw_payload["numeric_failure_trajectories"]), "real_solve_attempts": int(raw_payload["real_solve_attempts"]), "wall_s": total_wall_s}  # Bind completion to both immutable result files without modifying the pre-test freeze index.
    complete_path = destination / "TEST_COMPLETE.json"  # Resolve the final one-way completion receipt.
    _write_json(complete_path, complete_payload)  # Publish completion only after both result artifacts are valid and hashed.
    return {"raw_results": str(raw_path), "median_results": str(median_path), "completion_receipt": str(complete_path), "trajectory_count": len(raw_trajectories), "median_row_count": len(median_rows), "TEST_NOT_RUN": False}  # Return concise machine-readable delivery paths and counts.
