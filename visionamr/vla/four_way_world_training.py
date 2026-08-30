"""Build the frozen train-only world-model transition library and shared partitions."""  # Describe the protocol-bound pre-blind responsibilities implemented here.
from __future__ import annotations  # Postpone annotation evaluation for compatible repository runtimes.
from collections.abc import Callable, Mapping  # Import explicit callback and immutable JSON container contracts.
from dataclasses import asdict, dataclass, is_dataclass  # Import frozen configuration and audit serialization helpers.
import hashlib  # Compute exact-file and deterministic mesh identities.
import json  # Persist finite human-auditable training and partition evidence.
import math  # Compute regional logarithmic transition diagnostics safely.
import os  # Publish completed JSON artifacts atomically.
from pathlib import Path  # Resolve campaign and per-case artifact paths portably.
import time  # Measure complete offline acquisition costs without changing policy behavior.
from typing import Any  # Annotate repository solver, mesh, partition, and record objects.
import numpy as np  # Normalize numerical state fields and transition diagnostics.
from ..bridge_case_manifest import load_case_manifest, problem_from_case  # Authenticate the sole manifest and reconstruct only authorized cases.
from ..calculix import CalculiXExecutionError  # Classify only an evidenced native solver failure as a retainable training outcome.
from ..experiment import FemRunner, initial_mesh  # Reuse honest counted solves and the common uniform probe mesh.
from ..indicators import zz_indicator  # Reuse the exact repository ZZ squared-error indicator.
from ..mesher import GmshMeshingError  # Classify only an evidenced native meshing failure as a retainable training outcome.
from .partition_spec import PartitionSpecRegistry, generate_partition_spec, probe_mesh_sha256  # Reuse frozen shared partition generation and verification.
from .world.model import ResidualWorldModel, WorldModelConfig, WorldPrediction, WorldState  # Reuse the unchanged V0 transition model public API.
from .world.planner import MultiStepPlanner, PlannerConfig  # Reuse the unchanged finite-horizon V0 decision implementation.
from .world.tool_gateway import MCPToolGateway, ToolConfig  # Reuse exact Dörfler-safe action materialization and budget certification.

PROTOCOL_ID = "WMVLA-4WAY-P1"  # Bind every generated artifact to the sole frozen four-way protocol.
TRAINING_SCHEMA = "wmvla-four-way-world-training-v1"  # Version the train-only transition-library evidence contract.
PARTITION_INDEX_SCHEMA = "wmvla-four-way-partition-index-v1"  # Version the shared partition registry index contract.
TRAIN_CASE_COUNT = 24  # Require the exact preregistered train split cardinality.
SOLVES_PER_CASE = 6  # Require the exact real-solve acquisition horizon for every successful training case.
EQUATION_BUDGET = 120000  # Freeze the sole active-equation budget used for transition acquisition.
MODEL_FILENAME = "world_model_v0.json"  # Freeze the deployment snapshot filename consumed by the later freeze bundle.
MODEL_SIDECAR_FILENAME = "world_model_v0.sha256"  # Freeze the conventional exact-byte checksum filename.
PARTITION_INDEX_FILENAME = "partition_spec_index.json"  # Freeze the registry-wide partition identity filename.
CaseRunner = Callable[[Mapping[str, Any], Any, Any, ResidualWorldModel, "WorldTrainingConfig", Path], Mapping[str, Any]]  # Define the injectable per-case acquisition callback used by focused tests.
ProblemFactory = Callable[[Mapping[str, Any]], Any]  # Define the authorized manifest-case reconstruction callback.
PartitionLoader = Callable[[Mapping[str, Any], Any, Path, str], tuple[Any, Mapping[str, Any]]]  # Define the authenticated shared-partition loading callback.

class WorldTrainingNumericalError(RuntimeError):  # Identify a measured nonfinite estimator outcome without hiding configuration or API errors.
    """Report a solver-derived numerical label that cannot train the frozen world model."""  # Document the narrow non-native retained-failure category.

@dataclass(frozen=True)  # Make all acquisition settings immutable and hashable before the first solve.
class WorldTrainingConfig:  # Freeze the current safe exploration settings without modifying V0 model or planner logic.
    equation_budget: int = EQUATION_BUDGET  # Use only the protocol's 120000-equation acquisition budget.
    solves_per_case: int = SOLVES_PER_CASE  # Spend exactly six successful real solves on every complete train case.
    theta: float = 0.5  # Use the shared exact Dörfler bulk parameter.
    refine_factor: float = 0.5  # Use the repository's common refinement atom.
    core_theta: float = 0.72  # Use the current gateway's concentrated semantic-core boundary.
    nodal_gradation: float = 1.0  # Freeze the PR-number-40 V0 nodal target interpolation gradation explicitly.
    audit_slack: float = 0.08  # Use the current V0 prediction-audit tolerance.
    fallback_cooldown: int = 1  # Force one exact-Dörfler recovery action after an underperforming proactive transition.
    horizon: int = 3  # Use the existing bounded acquisition rollout depth.
    beam_width: int = 16  # Use the existing bounded acquisition beam width.
    candidate_regions: int = 5  # Retain the current V0 regional candidate bound.
    max_extra_regions: int = 2  # Retain the current sparse proactive action bound.
    max_extra_depth: int = 2  # Retain the current bounded future-hit depth.
    warmup_transitions: int = 1  # Require one real transition before proactive control.
    discount: float = 0.84  # Retain the current V0 future-cost discount.
    resource_weight: float = 0.08  # Use the existing acquisition-specific resource weight.
    uncertainty_weight: float = 0.75  # Retain the current V0 uncertainty penalty.
    failure_weight: float = 1.10  # Retain the current V0 failure-risk penalty.
    uncertainty_limit: float = 1.0  # Permit the existing safe acquisition exploration envelope.
    failure_limit: float = 0.95  # Permit the existing safe acquisition exploration envelope.
    budget_safety: float = 0.95  # Preserve the existing predicted-resource acquisition margin.
    min_robust_gain: float = -1.0  # Permit safe informative acquisition actions already allowed by the earlier library builder.
    model_seed: int = 271828  # Freeze the current new-stack residual-ensemble bootstrap seed explicitly.
    ccx_timeout: float = 1800.0  # Bound only the operational native process duration.
    def __post_init__(self) -> None:  # Reject accidental policy drift or a campaign incapable of satisfying the protocol.
        if self.equation_budget != EQUATION_BUDGET or self.solves_per_case != SOLVES_PER_CASE:  # Require the literal protocol budget and solve horizon.
            raise ValueError("world training requires exactly six solves per case at budget 120000")  # Refuse a scientifically different transition campaign.
        if self.nodal_gradation != 1.0:  # Require the exact current gateway field behavior without introducing a new tuning axis.
            raise ValueError("world training nodal_gradation must remain frozen at 1.0")  # Refuse hidden mesh-gradation drift.
        if not math.isfinite(self.ccx_timeout) or self.ccx_timeout <= 0.0:  # Require a usable operational solver timeout.
            raise ValueError("ccx_timeout must be finite and positive")  # Stop before any native work starts.

def sha256_file(path: Path | str) -> str:  # Hash exact artifact bytes with bounded memory.
    digest = hashlib.sha256()  # Allocate one collision-resistant digest state.
    with Path(path).open("rb") as handle:  # Stream the complete file without assuming its size.
        for block in iter(lambda: handle.read(1024 * 1024), b""):  # Consume stable one-megabyte blocks through EOF.
            digest.update(block)  # Incorporate every byte in exact order.
    return digest.hexdigest()  # Return the complete lowercase exact-file identity.

def _json_ready(value: Any) -> Any:  # Convert repository and numerical values into finite strict-JSON primitives.
    if is_dataclass(value) and not isinstance(value, type):  # Expand immutable repository records before recursive normalization.
        return _json_ready(asdict(value))  # Normalize every dataclass field through the same finite boundary.
    if isinstance(value, np.ndarray):  # Convert numerical arrays without losing ordering.
        return _json_ready(value.tolist())  # Normalize every scalar recursively after list conversion.
    if isinstance(value, np.generic):  # Convert NumPy scalar wrappers to native values.
        return _json_ready(value.item())  # Re-enter normalization with the underlying scalar.
    if isinstance(value, Path):  # Convert portable artifact paths explicitly.
        return str(value)  # Preserve the caller-visible path spelling.
    if isinstance(value, Mapping):  # Normalize arbitrary mapping implementations.
        return {str(key): _json_ready(item) for key, item in value.items()}  # Preserve names while normalizing nested values.
    if isinstance(value, (list, tuple)):  # Normalize ordered heterogeneous containers.
        return [_json_ready(item) for item in value]  # Preserve exact sequence order.
    if isinstance(value, float):  # Enforce finite scientific evidence at the serialization boundary.
        if not math.isfinite(value):  # Reject NaN and infinity instead of silently emitting non-standard JSON.
            raise ValueError("training evidence contains a non-finite float")  # Surface the malformed evidence before publication.
        return float(value)  # Return a standard finite Python float.
    if value is None or isinstance(value, (str, int, bool)):  # Accept native JSON scalar types unchanged.
        return value  # Preserve the original finite scalar.
    if hasattr(value, "__dict__"):  # Support simple injected or repository record objects transparently.
        return _json_ready(vars(value))  # Normalize their public instance attributes recursively.
    return str(value)  # Preserve an audit-friendly representation for an otherwise opaque diagnostic value.

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:  # Publish one complete deterministic strict-JSON artifact atomically.
    normalized = _json_ready(payload)  # Normalize and validate the complete evidence tree before filesystem mutation.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the exact requested parent directory.
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")  # Isolate interrupted writes from the last complete artifact.
    temporary.write_text(json.dumps(normalized, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Persist stable human-auditable bytes.
    os.replace(temporary, path)  # Atomically expose the completed document on the same filesystem.

def _config_payload(config: WorldTrainingConfig) -> dict[str, Any]:  # Serialize the complete acquisition policy and model settings.
    model = asdict(WorldModelConfig(refine_factor=config.refine_factor))  # Capture the unchanged public residual-model defaults explicitly.
    planner = {"horizon": config.horizon, "beam_width": config.beam_width, "candidate_regions": config.candidate_regions, "max_extra_regions": config.max_extra_regions, "max_extra_depth": config.max_extra_depth, "warmup_transitions": config.warmup_transitions, "discount": config.discount, "resource_weight": config.resource_weight, "uncertainty_weight": config.uncertainty_weight, "failure_weight": config.failure_weight, "uncertainty_limit": config.uncertainty_limit, "failure_limit": config.failure_limit, "budget_safety": config.budget_safety, "min_robust_gain": config.min_robust_gain, "min_relative_gain": config.min_robust_gain}  # Preserve current names plus the protocol's compatibility alias.
    tool = {"theta": config.theta, "refine_factor": config.refine_factor, "core_theta": config.core_theta, "nodal_gradation": config.nodal_gradation, "equation_cap_fraction": 1.0, "max_extra_depth": config.max_extra_depth}  # Capture exact deterministic action compilation, interpolation, and literal cap enforcement without conflicting aliases.
    return {"runtime": {"equation_budget": config.equation_budget, "solves_per_case": config.solves_per_case, "audit_slack": config.audit_slack, "regression_tolerance": config.audit_slack, "fallback_cooldown": config.fallback_cooldown, "ccx_timeout": config.ccx_timeout}, "acquisition_planner": planner, "tool_gateway": tool, "world_model": {**model, "seed": config.model_seed}, "deployment_policy_source": "unchanged visionamr.vla.world.model/planner defaults; acquisition envelope is training-only"}  # Return one complete training-specific freeze-ready snapshot without implying deployment-policy mutation.

def training_cases(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:  # Select only the authorized training records without returning another split's parameters.
    values = manifest.get("cases")  # Read the authenticated top-level case collection once.
    if not isinstance(values, list):  # Require the validated manifest structure defensively.
        raise ValueError("case manifest lacks its case list")  # Refuse an unusable training boundary.
    selected = tuple(sorted((dict(case) for case in values if isinstance(case, Mapping) and case.get("split") == "train"), key=lambda case: str(case["case_id"])))  # Copy only train mappings and sort them independently of execution history.
    if len(selected) != TRAIN_CASE_COUNT or any(case.get("split") != "train" for case in selected):  # Require exactly the frozen 24-case train split.
        raise ValueError("world-model acquisition requires exactly 24 train cases")  # Stop before geometry, partition, or solver access.
    return selected  # Return no validation or blind-test parameter mapping.

def build_training_plan(manifest_path: Path | str, partition_root: Path | str, output_dir: Path | str, config: WorldTrainingConfig | None = None) -> dict[str, Any]:  # Authenticate the manifest and build a solve-free train-only acquisition plan.
    settings = config or WorldTrainingConfig()  # Instantiate the immutable preregistered acquisition settings.
    manifest_file = Path(manifest_path)  # Normalize the sole allowed manifest location.
    manifest = load_case_manifest(manifest_file, verify_checksum=True)  # Authenticate schema, geometry, split, and exact-byte sidecar.
    cases = training_cases(manifest)  # Copy only the 24 authorized training records.
    return {"schema": TRAINING_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "train_plan", "manifest_path": str(manifest_file), "manifest_sha256": sha256_file(manifest_file), "partition_root": str(Path(partition_root)), "output_dir": str(Path(output_dir)), "training_case_ids": [str(case["case_id"]) for case in cases], "training_case_count": len(cases), "planned_real_solve_count": len(cases) * settings.solves_per_case, "planned_transition_count": len(cases) * (settings.solves_per_case - 1), "validation_split_accessed": False, "test_split_accessed": False, "reads_local_prediction": False, "reads_supervised_labels": False, "reads_rl_labels": False, "reads_reference_errors": False, "TEST_NOT_RUN": True, "scientific_config": _config_payload(settings)}  # Return a complete plan without exposing non-train identities or parameters.

def _state_payload(state: WorldState) -> dict[str, Any]:  # Serialize one measured or predicted state without losing regional ordering.
    return {"names": list(state.names), "err_sum": state.err_sum.tolist(), "elems": state.elems.tolist(), "sizes": state.sizes.tolist(), "vm_max": state.vm_max.tolist(), "volume": state.volume.tolist(), "adjacency": state.adjacency.tolist(), "dorfler_error_fraction": state.dorfler_error_fraction.tolist(), "dorfler_element_fraction": state.dorfler_element_fraction.tolist(), "hit_count": state.hit_count.tolist(), "n_equations": int(state.n_equations), "eq_per_elem": float(state.eq_per_elem), "h_min": float(state.h_min), "h0": float(state.h0), "dim": int(state.dim), "step": int(state.step), "total_error": float(state.total_error), "total_elements": float(state.total_elements)}  # Preserve every public state field plus transparent aggregates.

def _prediction_payload(prediction: WorldPrediction) -> dict[str, Any]:  # Serialize every executed-transition prediction required for calibration.
    return {"next_state": _state_payload(prediction.next_state), "uncertainty": float(prediction.uncertainty), "failure_risk": float(prediction.failure_risk), "error_ratio_mean": float(prediction.error_ratio_mean), "error_ratio_upper": float(prediction.error_ratio_upper), "equation_ratio_mean": float(prediction.equation_ratio_mean), "equation_ratio_upper": float(prediction.equation_ratio_upper)}  # Preserve the complete public prediction object.

def _record_payload(record: Any) -> dict[str, Any]:  # Serialize one successful real-solver record without requiring a specific mutable schema.
    normalized = _json_ready(record)  # Expand dataclass or simple-object fields through the finite JSON boundary.
    if not isinstance(normalized, dict):  # Require a named solver receipt rather than an opaque scalar.
        return {"value": normalized}  # Wrap the unusual record without discarding it.
    return normalized  # Return the complete repository record mapping.

def _failure_payload(error: BaseException, stage: str, solve_attempt: int) -> dict[str, Any]:  # Retain a bounded numerical or native failure without secrets or tracebacks.
    return {"stage": str(stage), "solve_attempt": int(solve_attempt), "exception_type": type(error).__name__, "message": str(error)[-2000:]}  # Preserve actionable diagnostics and the exact acquisition boundary.

def _actual_payload(previous: WorldState, observed: WorldState) -> dict[str, Any]:  # Compute all regional and global realized transition diagnostics.
    if previous.names != observed.names:  # Require the frozen semantic ordering across a real transition.
        raise ValueError("actual transition changed frozen region ordering")  # Refuse scientifically unalignable labels.
    delta_error = np.log(np.maximum(observed.err_sum, 1.0e-30) / np.maximum(previous.err_sum, 1.0e-30))  # Measure regional realized log indicator changes.
    delta_elements = np.log(np.maximum(observed.elems, 1.0) / np.maximum(previous.elems, 1.0))  # Measure regional realized log resource changes.
    return {"state": _state_payload(observed), "regional_delta_log_eta2": delta_error.tolist(), "regional_delta_log_elements": delta_elements.tolist(), "total_error_ratio": float(observed.total_error / max(previous.total_error, 1.0e-30)), "equation_ratio": float(observed.n_equations / max(previous.n_equations, 1)), "equation_delta": int(observed.n_equations - previous.n_equations)}  # Return complete calibration targets without reference truth.

def _planner(config: WorldTrainingConfig) -> MultiStepPlanner:  # Construct the unchanged V0 planner with the frozen acquisition envelope.
    settings = PlannerConfig(horizon=config.horizon, beam_width=config.beam_width, candidate_regions=config.candidate_regions, max_extra_regions=config.max_extra_regions, max_extra_depth=config.max_extra_depth, warmup_transitions=config.warmup_transitions, discount=config.discount, resource_weight=config.resource_weight, uncertainty_weight=config.uncertainty_weight, failure_weight=config.failure_weight, uncertainty_limit=config.uncertainty_limit, failure_limit=config.failure_limit, budget_safety=config.budget_safety, min_robust_gain=config.min_robust_gain)  # Bind every public planner field explicitly without modifying decision logic.
    return MultiStepPlanner(settings)  # Return the current public planner implementation.

def _gateway(config: WorldTrainingConfig) -> MCPToolGateway:  # Construct the exact safe action compiler used by V0.
    settings = ToolConfig(theta=config.theta, refine_factor=config.refine_factor, core_theta=config.core_theta, budget_safety=1.0, max_extra_depth=config.max_extra_depth)  # Enforce the full literal acquisition cap after predicted safety margin.
    return MCPToolGateway(settings)  # Return the current public deterministic gateway implementation.

def _solver_logs(case_root: Path) -> list[dict[str, Any]]:  # Inventory every retained solver log after one case attempt.
    return [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in sorted(case_root.rglob("*.log")) if path.is_file()]  # Preserve exact bytes and paths without interpreting solver text.

def _default_case_runner(case: Mapping[str, Any], problem: Any, partition: Any, model: ResidualWorldModel, config: WorldTrainingConfig, case_root: Path) -> Mapping[str, Any]:  # Execute one six-solve reference-free safe acquisition trajectory.
    case_id = str(case["case_id"])  # Read the already authenticated train-case identity.
    runner = FemRunner(problem, case_root, keep_files=False, ccx_timeout=config.ccx_timeout)  # Retain required logs and JSON records while deleting reproducible bulky native intermediates.
    planner = _planner(config)  # Instantiate the frozen safe exploration policy for this case.
    gateway = _gateway(config)  # Instantiate deterministic Dörfler/world compilation and exact preflight.
    started = time.perf_counter()  # Start complete offline acquisition timing before the common probe mesh.
    initial_mesh_started = time.perf_counter()  # Start the first exact Gmsh mesh timing independently.
    mesh = initial_mesh(problem)  # Generate the common global-uniform problem.h0 probe.
    initial_mesh_s = float(time.perf_counter() - initial_mesh_started)  # Retain common-probe Gmsh wall time.
    partition.verify(problem=problem, expected_geometry_hash=str(case["geometry_hash"]), probe_mesh=mesh)  # Bind this exact common probe to the frozen shared partition.
    mesh_hash = probe_mesh_sha256(mesh)  # Hash the exact initial coordinates and connectivity.
    solve_receipts: list[dict[str, Any]] = []  # Retain every successful real solve and measured world state.
    action_receipts: list[dict[str, Any]] = []  # Retain every requested, executed, certified, and timed action.
    transition_receipts: list[dict[str, Any]] = []  # Retain every prediction paired with its realized next solve.
    failures: list[dict[str, Any]] = []  # Retain every numerical failure instead of deleting the case.
    hit_count: np.ndarray | None = None  # Initialize per-case semantic recurrence history.
    pending: dict[str, Any] | None = None  # Reserve one executed action awaiting its next real observation.
    cooldown = 0  # Initialize the V0 audit-triggered exact-Dörfler recovery counter.
    planning_s = 0.0  # Accumulate only world-model planner wall time.
    parameter_tools_s = 0.0  # Accumulate deterministic parameter and certification work excluding Gmsh.
    gmsh_s = initial_mesh_s  # Include the common probe in complete acquisition meshing cost.
    solve_attempt_count = 0  # Count every entered CalculiX invocation including a retained numerical failure.
    for step in range(config.solves_per_case):  # Attempt the exact six-solve acquisition schedule in fixed order.
        solve_attempt_count += 1  # Charge this scheduled native invocation before it can fail.
        try:  # Retain a native solve failure with all earlier valid evidence.
            post, record = runner.solve_mesh(mesh, method="world_model_training", stage=f"cycle{step}", extra={"case_id": case_id, "split": "train", "equation_budget": config.equation_budget, "mesh_sha256": mesh_hash})  # Execute one counted reference-free CalculiX solve.
        except CalculiXExecutionError as error:  # Convert only an evidenced CalculiX backend failure into retained finite evidence.
            failures.append(_failure_payload(error, "calculix_solve", step + 1))  # Record the exact failed attempt boundary.
            break  # Stop because no actual successor state exists for further adaptive decisions.
        eta2 = np.asarray(zz_indicator(problem, post), dtype=float).reshape(-1)  # Evaluate and normalize the exact repository squared indicator.
        if np.any(~np.isfinite(eta2)) or np.any(eta2 < -1.0e-12):  # Detect only a solver-derived unusable or physically invalid transition label.
            error = WorldTrainingNumericalError("zz_indicator returned non-finite or negative squared contributions")  # Construct a narrow retained numerical-failure receipt.
            solve_receipts.append({"solve_index": step + 1, "mesh_sha256": mesh_hash, "solver_record": _record_payload(record), "state": None, "indicator_sum": None})  # Retain all valid solver evidence without fabricating state values.
            failures.append(_failure_payload(error, "indicator_or_observation", step + 1))  # Record the exact post-solve label failure.
            break  # Stop because planning and learning require a valid state.
        eta2 = np.maximum(eta2, 0.0)  # Remove only roundoff-scale negative contributions after the explicit numerical gate.
        observation = gateway.observe_solve(problem, partition, post, record, eta2, hit_count, step)  # Build the complete measured frozen-region world state and propagate any schema or API error.
        hit_count = observation.state.hit_count.copy()  # Carry only this case's realized semantic history forward.
        solver_record = _record_payload(record)  # Freeze the complete successful solve receipt before later mutations.
        solve_receipt = {"solve_index": step + 1, "mesh_sha256": mesh_hash, "indicator_sum": float(np.sum(eta2)), "state": _state_payload(observation.state), "solver_record": solver_record, "budget_violation": bool(int(record.n_equations) > config.equation_budget)}  # Bind solver, mesh, estimator, state, and measured resource evidence.
        solve_receipts.append(solve_receipt)  # Retain this successful counted solve in exact trajectory order.
        if pending is not None:  # Pair the last executed action with this newly realized successor state.
            previous_state = pending["state"]  # Recover the measured pre-action state object.
            executed_action = pending["action"]  # Recover the exact post-gateway executed action object.
            prediction = pending["prediction"]  # Recover the prediction made before this real solve.
            actual = _actual_payload(previous_state, observation.state)  # Compute all regional and global realized changes while propagating state-contract errors.
            model.observe(previous_state, executed_action, observation.state)  # Fit through the public API and invalidate the campaign on model/schema failures.
            underperformed = bool(actual["total_error_ratio"] > prediction.error_ratio_upper * (1.0 + config.audit_slack) and actual["total_error_ratio"] > prediction.error_ratio_mean * (1.0 + config.audit_slack))  # Apply the unchanged V0 trust audit to the completed proactive transition.
            if underperformed and not executed_action.is_dorfler_only:  # Trigger recovery only after a genuinely proactive underperformance.
                cooldown = max(cooldown, config.fallback_cooldown)  # Require the frozen number of exact-Dörfler recovery actions.
            transition_receipts.append({"transition_index": len(transition_receipts) + 1, "source_solve_index": int(pending["source_solve_index"]), "actual_solve_index": step + 1, "source_mesh_sha256": str(pending["source_mesh_sha256"]), "actual_mesh_sha256": mesh_hash, "previous_state": _state_payload(previous_state), "requested_action": list(pending["decision"].action.extra_depth), "executed_action": list(executed_action.extra_depth), "decision": asdict(pending["decision"]), "certificate": asdict(pending["certificate"]), "prediction": _prediction_payload(prediction), "actual": actual, "underperformed": underperformed, "action_timing_s": dict(pending["timing_s"]), "source_solver_record": pending["source_solver_record"], "actual_solver_record": solver_record})  # Persist every executed prediction, action, certificate, timing, mesh, state, and solver pair.
        if solve_receipt["budget_violation"]:  # Treat any measured cap breach as an incomplete acquisition case.
            failures.append({"stage": "measured_budget", "solve_attempt": step + 1, "exception_type": "BudgetViolation", "message": f"measured equations {record.n_equations} exceed {config.equation_budget}"})  # Preserve the precise resource failure.
            break  # Stop before spending another out-of-contract solve.
        if step + 1 >= config.solves_per_case:  # Finish immediately after the sixth successful real solve.
            break  # Avoid proposing an unused seventh action.
        try:  # Retain planning or exact premesh failures without fabricating a transition.
            force_dorfler = cooldown > 0  # Convert the current audit cooldown to the planner's public safety gate.
            planning_started = time.perf_counter()  # Start local finite-horizon inference timing.
            decision = planner.plan(observation.state, model, config.equation_budget, force_dorfler=force_dorfler)  # Select the next safe first action through unchanged V0 logic.
            planning_elapsed = float(time.perf_counter() - planning_started)  # Measure only the current public planning call.
            planning_s += planning_elapsed  # Accumulate complete world-model inference cost.
            if cooldown > 0:  # Consume one scheduled exact-Dörfler recovery step.
                cooldown -= 1  # Update only the per-case audit state.
            materialized = gateway.materialize_action(observation, decision.action, config.equation_budget)  # Compile, premesh, count active equations, certify, or fall back exactly.
            parameter_tools_s += float(materialized.timing_s["parameter_tools"])  # Accumulate non-Gmsh deterministic tool time.
            gmsh_s += float(materialized.timing_s["gmsh_remeshing"])  # Accumulate both Dörfler and optional proactive candidate remeshing.
            action_receipt = {"action_index": len(action_receipts) + 1, "source_solve_index": step + 1, "requested_action": list(decision.action.extra_depth), "executed_action": list(materialized.action.extra_depth), "decision": asdict(decision), "certificate": asdict(materialized.certificate), "timing_s": {"world_model_planning": planning_elapsed, **materialized.timing_s}, "candidate_mesh_sha256": None if materialized.mesh is None else probe_mesh_sha256(materialized.mesh)}  # Bind the complete action and candidate evidence before execution.
            action_receipts.append(action_receipt)  # Retain accepted, fallback, and stop certificates alike.
            if materialized.mesh is None:  # Detect the exact certified inability to remain inside the equation cap.
                failures.append({"stage": "action_materialization", "solve_attempt": step + 2, "exception_type": "NoFeasibleMesh", "message": materialized.certificate.reason})  # Retain the deterministic incomplete-case reason.
                break  # Stop rather than executing an over-budget or fabricated solve.
            prediction = model.predict(observation.state, materialized.action)  # Predict the exact actually executed action through the current public model API.
            action_receipt["prediction"] = _prediction_payload(prediction)  # Preserve the prediction even if the subsequent native solve fails before an actual state exists.
            pending = {"state": observation.state, "action": materialized.action, "prediction": prediction, "decision": decision, "certificate": materialized.certificate, "timing_s": {"world_model_planning": planning_elapsed, **materialized.timing_s}, "source_solve_index": step + 1, "source_mesh_sha256": mesh_hash, "source_solver_record": solver_record}  # Hold complete evidence until the next real observation arrives.
            mesh = materialized.mesh  # Advance only to the exact preflighted candidate mesh.
            mesh_hash = str(action_receipt["candidate_mesh_sha256"])  # Carry the precomputed exact candidate identity into its solver record.
        except GmshMeshingError as error:  # Convert only an evidenced Gmsh candidate failure into a retained incomplete case.
            failures.append(_failure_payload(error, "planning_or_materialization", step + 2))  # Preserve the next attempted transition boundary.
            break  # Stop because no certified successor mesh exists.
    runner.dump(case_root / "solver_records.json")  # Persist every counted repository record including attached training metadata.
    complete = len(solve_receipts) == config.solves_per_case and len(transition_receipts) == config.solves_per_case - 1 and not failures  # Require the literal six solves and five fully paired transitions.
    timing = {"gmsh_remeshing": float(gmsh_s), "world_model_planning": float(planning_s), "parameter_tools": float(parameter_tools_s), "calculix": float(sum(float(getattr(record, "wall_s", 0.0)) for record in runner.records)), "offline_total": float(time.perf_counter() - started)}  # Separate mandatory offline cost categories transparently.
    return {"schema": TRAINING_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "case_acquisition", "case_id": case_id, "split": "train", "geometry_hash": str(case["geometry_hash"]), "partition_spec_sha256": str(partition.spec_sha256), "status": "ok" if complete else "failed", "complete": complete, "equation_budget": config.equation_budget, "planned_real_solve_count": config.solves_per_case, "real_solve_attempt_count": solve_attempt_count, "real_solve_count": len(solve_receipts), "transition_count": len(transition_receipts), "solves": solve_receipts, "actions": action_receipts, "transitions": transition_receipts, "failures": failures, "solver_logs": _solver_logs(case_root), "timing_s": timing, "scientific_boundary": {"reference_solve_count": 0, "reads_local_prediction": False, "reads_supervised_labels": False, "reads_rl_labels": False}}  # Return every required successful or failed acquisition artifact.

def _partition_index(path: Path, manifest_sha256: str) -> dict[str, Any]:  # Load and authenticate the registry index before training setup.
    payload = json.loads(path.read_text(encoding="utf-8"))  # Decode the transparent finite partition inventory.
    if not isinstance(payload, dict) or payload.get("schema") != PARTITION_INDEX_SCHEMA or payload.get("protocol_id") != PROTOCOL_ID:  # Require the exact index schema and protocol.
        raise ValueError("partition_spec_index.json is not a WMVLA-4WAY-P1 registry")  # Reject unrelated or stale partition roots.
    if payload.get("manifest_sha256") != manifest_sha256:  # Bind the registry to exact manifest bytes.
        raise ValueError("partition index manifest SHA-256 mismatch")  # Prevent cross-design partition reuse.
    if not isinstance(payload.get("partitions"), list):  # Require explicit per-case entries rather than filesystem discovery.
        raise ValueError("partition index lacks its partitions list")  # Refuse an ambiguous registry.
    json.dumps(payload, allow_nan=False)  # Reject permissively decoded NaN or infinity anywhere in the frozen index.
    return payload  # Return the authenticated top-level partition inventory.

def _default_partition_loader(case: Mapping[str, Any], problem: Any, root: Path, manifest_sha256: str) -> tuple[Any, Mapping[str, Any]]:  # Load one exact shared train partition through its registry index.
    index = _partition_index(root / PARTITION_INDEX_FILENAME, manifest_sha256)  # Authenticate the registry before resolving this case.
    matches = [dict(item) for item in index["partitions"] if isinstance(item, Mapping) and str(item.get("case_id")) == str(case["case_id"])]  # Select only the manifest-authorized train-case entry.
    if len(matches) != 1:  # Require one unambiguous indexed specification.
        raise ValueError(f"partition index must contain exactly one entry for {case['case_id']}")  # Reject missing or duplicate case semantics.
    receipt = matches[0]  # Read the sole authenticated index entry.
    if receipt.get("geometry_hash") != case.get("geometry_hash"):  # Require the same manifest geometry identity.
        raise ValueError(f"partition index geometry mismatch for {case['case_id']}")  # Reject stale or cross-case partition evidence.
    registry = PartitionSpecRegistry(root, expected_sha256={str(case["case_id"]): str(receipt["file_sha256"])})  # Bind exact file bytes before model acquisition.
    partition = registry.partition_for(str(case["case_id"]), problem, str(case["geometry_hash"]))  # Load and verify schema, file hash, body hash, and runtime geometry.
    if str(partition.spec_sha256) != str(receipt["spec_sha256"]):  # Recheck the independent semantic-body identity from the index.
        raise ValueError(f"partition semantic SHA-256 mismatch for {case['case_id']}")  # Reject a body/file index inconsistency.
    return partition, receipt  # Return the one shared WM/RL specification and its authenticated receipt.

def _validated_case_result(result: Mapping[str, Any], case_id: str, config: WorldTrainingConfig, transitions_added: int) -> dict[str, Any]:  # Validate injected or native per-case evidence before publication.
    payload = dict(_json_ready(result))  # Normalize the complete callback evidence through strict JSON.
    if str(payload.get("case_id")) != case_id or payload.get("split") != "train":  # Require exact train-case identity from the callback.
        raise ValueError(f"case runner returned the wrong identity for {case_id}")  # Prevent mixed or substituted training evidence.
    solves = int(payload.get("real_solve_count", -1))  # Read the explicit counted solve total.
    transitions = int(payload.get("transition_count", -1))  # Read the explicit completed transition total.
    complete = bool(payload.get("complete", payload.get("status") == "ok"))  # Normalize native and injected completion declarations.
    if complete and (solves != config.solves_per_case or transitions != config.solves_per_case - 1 or transitions_added != transitions):  # Require six solves, five receipts, and five public model updates.
        raise ValueError(f"complete case {case_id} lacks the exact six-solve five-transition contract")  # Reject optimistic or unfitted case evidence.
    if transitions_added != transitions:  # Require the recorded transitions to equal public ResidualWorldModel.observe calls even on failure.
        raise ValueError(f"case {case_id} transition evidence does not match model updates")  # Prevent detached or fabricated training labels.
    payload["complete"] = complete  # Preserve the normalized completion Boolean explicitly.
    payload["status"] = "ok" if complete else "failed"  # Normalize status for aggregate failure accounting.
    return payload  # Return finite validated per-case evidence.

def train_world_model_transition_library(manifest_path: Path | str, partition_root: Path | str, output_dir: Path | str, *, config: WorldTrainingConfig | None = None, case_runner: CaseRunner | None = None, problem_factory: ProblemFactory = problem_from_case, partition_loader: PartitionLoader = _default_partition_loader) -> dict[str, Any]:  # Execute all 24 train-only acquisition cases and freeze one model snapshot.
    settings = config or WorldTrainingConfig()  # Instantiate the immutable preregistered acquisition configuration.
    manifest_file = Path(manifest_path)  # Normalize the sole authenticated data-design artifact.
    manifest = load_case_manifest(manifest_file, verify_checksum=True)  # Validate exact bytes, schema, geometry, split, IDs, and LHS before writes.
    cases = training_cases(manifest)  # Copy only the exact 24 train mappings.
    manifest_sha = sha256_file(manifest_file)  # Bind every downstream artifact to exact manifest bytes.
    root = Path(output_dir)  # Normalize the isolated world-model training artifact root.
    if root.exists() and any(root.iterdir()):  # Refuse resume, overwrite, or model replacement after partial review.
        raise FileExistsError("world-model training output directory must be absent or empty")  # Preserve a one-shot pre-blind acquisition boundary.
    partition_path = Path(partition_root)  # Normalize the frozen shared partition registry location.
    setups: list[tuple[dict[str, Any], Any, Any, Mapping[str, Any]]] = []  # Preflight every train geometry and partition before the first real solve.
    for case in cases:  # Touch only authorized train parameters and their shared partition artifacts.
        problem = problem_factory(case)  # Reconstruct this train geometry from its authenticated manifest row.
        partition, receipt = partition_loader(case, problem, partition_path, manifest_sha)  # Authenticate its indexed shared semantic specification.
        setups.append((case, problem, partition, receipt))  # Retain the complete preflighted case setup in sorted order.
    plan = build_training_plan(manifest_file, partition_path, root, settings)  # Rebuild the exact solve-free plan from authenticated bytes.
    plan["partition_receipts"] = [_json_ready(receipt) for _case, _problem, _partition, receipt in setups]  # Bind all 24 preflighted train partition identities before solving.
    root.mkdir(parents=True, exist_ok=True)  # Create the new empty artifact root only after complete input preflight.
    _write_json(root / "training_plan.json", plan)  # Persist the exact immutable plan before the first expensive solve.
    model = ResidualWorldModel(WorldModelConfig(refine_factor=settings.refine_factor), seed=settings.model_seed)  # Initialize the current public residual model with explicit frozen settings.
    runner = case_runner or _default_case_runner  # Select the native acquisition path unless a focused test injects a behaviorally complete fake.
    case_results: list[dict[str, Any]] = []  # Accumulate all 24 retained success or failure receipts.
    campaign_started = time.perf_counter()  # Measure complete ordered acquisition wall time.
    for case, problem, partition, _receipt in setups:  # Execute every train case once in ascending case-ID order.
        case_id = str(case["case_id"])  # Read the immutable case identity for paths and evidence checks.
        case_root = root / "cases" / case_id  # Isolate all solver and transition artifacts by manifest case.
        transitions_before = int(model.transition_count)  # Record the public model's completed-transition count before this case.
        try:  # Retain only typed native failures escaping setup while continuing the fixed train schedule.
            raw_result = runner(case, problem, partition, model, settings, case_root)  # Execute or inject one complete reference-free acquisition trajectory.
        except (CalculiXExecutionError, GmshMeshingError, WorldTrainingNumericalError) as error:  # Preserve only explicit numerical/native failures and propagate code or artifact defects.
            raw_result = {"schema": TRAINING_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "case_acquisition", "case_id": case_id, "split": "train", "status": "failed", "complete": False, "equation_budget": settings.equation_budget, "planned_real_solve_count": settings.solves_per_case, "real_solve_attempt_count": 0, "real_solve_count": 0, "transition_count": int(model.transition_count) - transitions_before, "solves": [], "actions": [], "transitions": [], "failures": [_failure_payload(error, "case_runner", 1)], "solver_logs": [], "timing_s": {"offline_total": 0.0}}  # Preserve the exact failed case instead of omitting it.
        transitions_added = int(model.transition_count) - transitions_before  # Measure public ResidualWorldModel.observe updates independently.
        result = _validated_case_result(raw_result, case_id, settings, transitions_added)  # Require callback evidence and learned transition counts to agree.
        _write_json(case_root / "case_training.json", result)  # Persist this complete case immediately before advancing.
        case_results.append(result)  # Retain every fixed-schedule outcome for aggregate cost accounting.
        model.save(root / MODEL_FILENAME)  # Checkpoint all valid transitions after every case through the current public save API.
    model_path = root / MODEL_FILENAME  # Resolve the final shared transition-library snapshot.
    model.save(model_path)  # Persist the complete final current-model state even when a retained case failed.
    model_sha = sha256_file(model_path)  # Compute the exact deployment snapshot identity.
    (root / MODEL_SIDECAR_FILENAME).write_text(f"{model_sha}  {MODEL_FILENAME}\n", encoding="ascii")  # Publish a conventional exact-byte checksum sidecar.
    completed = [result for result in case_results if result["complete"]]  # Count only literal six-solve five-transition cases.
    successful_solves = sum(int(result["real_solve_count"]) for result in case_results)  # Count every retained successful real solve.
    total_solves = sum(int(result.get("real_solve_attempt_count", result["real_solve_count"])) for result in case_results)  # Charge every entered native solve including retained failures.
    total_transitions = sum(int(result["transition_count"]) for result in case_results)  # Count every fitted completed real transition.
    eligible = len(completed) == TRAIN_CASE_COUNT and successful_solves == TRAIN_CASE_COUNT * settings.solves_per_case and total_transitions == TRAIN_CASE_COUNT * (settings.solves_per_case - 1)  # Require the entire protocol acquisition grid before freeze eligibility.
    costs = {"schema": TRAINING_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "training_cost", "TEST_NOT_RUN": True, "training_case_count": len(cases), "complete_case_count": len(completed), "failed_case_count": len(case_results) - len(completed), "planned_real_solve_count": len(cases) * settings.solves_per_case, "real_training_solve_count": total_solves, "successful_real_training_solve_count": successful_solves, "planned_transition_count": len(cases) * (settings.solves_per_case - 1), "fitted_transition_count": total_transitions, "regional_training_row_count": len(model.snapshot().get("x", [])), "training_wall_s": float(time.perf_counter() - campaign_started), "case_timing_s": {str(result["case_id"]): result.get("timing_s", {}) for result in case_results}, "model_sha256": model_sha, "training_complete": eligible}  # Report all required offline costs and completeness without reference metrics.
    _write_json(root / "training_costs.json", costs)  # Persist the world-model cost source expected by the aggregate freeze workflow.
    summary = {"schema": TRAINING_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "model_freeze", "TEST_NOT_RUN": True, "eligible_for_freeze": eligible, "manifest_path": str(manifest_file), "manifest_sha256": manifest_sha, "partition_root": str(partition_path), "training_plan_path": str(root / "training_plan.json"), "training_plan_sha256": sha256_file(root / "training_plan.json"), "model_path": str(model_path), "model_sha256": model_sha, "model_sidecar_path": str(root / MODEL_SIDECAR_FILENAME), "scientific_config": _config_payload(settings), "training_case_count": len(cases), "complete_case_count": len(completed), "real_training_solve_count": total_solves, "successful_real_training_solve_count": successful_solves, "fitted_transition_count": total_transitions, "test_split_accessed": False, "validation_split_accessed": False, "reference_errors_accessed": False, "external_labels_accessed": {"local_prediction": False, "supervised": False, "rl": False}, "failures": [{"case_id": str(result["case_id"]), "failures": result.get("failures", [])} for result in case_results if not result["complete"]], "training_cost_path": str(root / "training_costs.json"), "training_cost_sha256": sha256_file(root / "training_costs.json")}  # Assemble the sole world-model pretest freeze receipt.
    _write_json(root / "training_summary.json", summary)  # Persist exact model, cost, configuration, and anti-leakage evidence.
    return summary  # Return the complete finite pre-blind training result to the CLI or tests.

def partition_cases(manifest: Mapping[str, Any], split: str) -> tuple[dict[str, Any], ...]:  # Select an explicit manifest split or all cases for solve-free partition generation.
    allowed = {"train", "validation", "test", "all"}  # Enumerate every supported explicit selection mode.
    if split not in allowed:  # Reject implicit or misspelled data boundaries.
        raise ValueError(f"partition split must be one of {sorted(allowed)}")  # Surface the valid choices before geometry work.
    values = manifest.get("cases")  # Read the already authenticated case collection.
    if not isinstance(values, list):  # Require the canonical manifest structure.
        raise ValueError("case manifest lacks its case list")  # Refuse an ambiguous generation plan.
    selected = tuple(sorted((dict(case) for case in values if isinstance(case, Mapping) and (split == "all" or case.get("split") == split)), key=lambda case: str(case["case_id"])))  # Copy only the explicitly selected mappings in stable order.
    expected = {"train": 24, "validation": 8, "test": 16, "all": 48}[split]  # Recover the exact frozen cardinality for the requested selection.
    if len(selected) != expected:  # Require the complete split without omissions.
        raise ValueError(f"partition generation expected {expected} cases for split {split}")  # Stop before partial registry creation.
    return selected  # Return the explicit geometry set.

def build_partition_plan(manifest_path: Path | str, output_root: Path | str, split: str = "all") -> dict[str, Any]:  # Authenticate the manifest and build a no-Gmsh dry-run partition plan.
    manifest_file = Path(manifest_path)  # Normalize the sole case-design artifact.
    manifest = load_case_manifest(manifest_file, verify_checksum=True)  # Authenticate exact manifest bytes and complete geometry records.
    cases = partition_cases(manifest, split)  # Select only the explicitly authorized split.
    return {"schema": PARTITION_INDEX_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "partition_plan", "split": split, "manifest_path": str(manifest_file), "manifest_sha256": sha256_file(manifest_file), "output_root": str(Path(output_root)), "case_count": len(cases), "case_ids": [str(case["case_id"]) for case in cases], "calculix_solve_count": 0, "writes_performed": False}  # Return a complete solve-free and write-free plan.

def generate_partition_specs(manifest_path: Path | str, output_root: Path | str, split: str = "all", *, problem_factory: ProblemFactory = problem_from_case, spec_generator: Callable[[Any, str], Any] = generate_partition_spec) -> dict[str, Any]:  # Generate or exactly verify shared partition specs without CalculiX solves.
    manifest_file = Path(manifest_path)  # Normalize the authenticated case-design artifact.
    manifest = load_case_manifest(manifest_file, verify_checksum=True)  # Validate sidecar, schema, split, hashes, and geometry feasibility.
    manifest_sha = sha256_file(manifest_file)  # Bind the registry index to exact manifest bytes.
    cases = partition_cases(manifest, split)  # Select the caller's explicit complete case subset.
    root = Path(output_root)  # Normalize the shared protocol partition root.
    index_path = root / PARTITION_INDEX_FILENAME  # Resolve the sole registry inventory filename.
    existing: dict[str, dict[str, Any]] = {}  # Preserve exact entries from an earlier explicit split generation.
    if index_path.exists():  # Support idempotent staged train, validation, test, or all generation.
        payload = _partition_index(index_path, manifest_sha)  # Authenticate the prior registry before adding anything.
        existing = {str(item["case_id"]): dict(item) for item in payload["partitions"] if isinstance(item, Mapping)}  # Index every prior exact entry by manifest case ID.
        if len(existing) != len(payload["partitions"]):  # Reject duplicate prior case identities.
            raise ValueError("partition index contains duplicate case entries")  # Prevent ambiguous semantic artifacts.
    generated_started = time.perf_counter()  # Measure only partition geometry and Gmsh work.
    selected_receipts: list[dict[str, Any]] = []  # Retain every generated or verified selected specification.
    for case in cases:  # Process the explicit split in stable case-ID order.
        case_id = str(case["case_id"])  # Read the manifest-bound safe path component.
        problem = problem_factory(case)  # Reconstruct only this explicitly selected geometry.
        path = root / case_id / "partition_spec.json"  # Resolve the one-spec-per-geometry protocol layout.
        if path.exists():  # Verify exact existing bytes rather than replacing a frozen input.
            expected = existing.get(case_id, {})  # Read any previously indexed exact-file digest.
            registry = PartitionSpecRegistry(root, expected_sha256={case_id: str(expected["file_sha256"])} if "file_sha256" in expected else None)  # Bind prior indexed bytes when available.
            spec = registry.partition_for(case_id, problem, str(case["geometry_hash"]))  # Revalidate the complete persisted specification.
        else:  # Generate the deterministic shared semantic spec from the common uniform probe.
            spec = spec_generator(problem, str(case["geometry_hash"]))  # Invoke Gmsh only; no FemRunner or CalculiX call is reachable here.
            spec.save(path)  # Persist one deterministic transparent partition document.
        loaded = PartitionSpecRegistry(root, expected_sha256={case_id: sha256_file(path)}).partition_for(case_id, problem, str(case["geometry_hash"]))  # Reopen and verify exact persisted bytes independently.
        receipt = {"case_id": case_id, "split": str(case["split"]), "geometry_hash": str(case["geometry_hash"]), "path": str(path.relative_to(root)), "file_sha256": sha256_file(path), "spec_sha256": str(loaded.spec_sha256), "probe_mesh_sha256": str(loaded.probe_sha256), "probe_node_count": int(loaded.probe_node_count), "probe_cell_count": int(loaded.probe_cell_count), "region_order": list(loaded.names)}  # Bind file, semantic body, common probe, graph ordering, and manifest identity.
        if case_id in existing and existing[case_id] != receipt:  # Reject any attempt to mutate a previously indexed frozen partition.
            raise ValueError(f"existing partition index entry changed for {case_id}")  # Preserve append-only staged registry generation.
        existing[case_id] = receipt  # Add or confirm the exact case identity in the registry.
        selected_receipts.append(receipt)  # Retain selected-split completion evidence.
    covered_splits = sorted({str(receipt["split"]) for receipt in existing.values()})  # Derive stable registry coverage solely from indexed case content.
    index = {"schema": PARTITION_INDEX_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "partition_registry", "manifest_sha256": manifest_sha, "case_count": len(existing), "covered_splits": covered_splits, "calculix_solve_count": 0, "partitions": [existing[case_id] for case_id in sorted(existing)]}  # Assemble a deterministic append-only shared registry inventory without timestamps or host paths.
    _write_json(index_path, index)  # Publish the index only after every selected specification has been independently reloaded.
    return {"index_path": str(index_path), "index_sha256": sha256_file(index_path), "case_count": len(existing), "selected_case_count": len(selected_receipts), "split": split, "calculix_solve_count": 0, "generation_wall_s": float(time.perf_counter() - generated_started)}  # Return concise no-solver generation evidence while keeping operational timing outside frozen index bytes.
