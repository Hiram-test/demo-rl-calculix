"""Multi-step world-model VLA loop anchored to exact Dörfler actions."""  # Describe the module purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
from dataclasses import dataclass, field, replace  # Import configuration and decision-update helpers.
import time  # Import a monotonic timer for planning and meshing overhead.
import numpy as np  # Import numerical arrays for state construction.
from ..experiment import FemRunner, initial_mesh  # Reuse the common solve accounting and common probe.
from ..indicators import zz_indicator  # Reuse the shared ZZ estimator.
from ..marking import dorfler_mark  # Reuse the exact element-wise Dörfler marking.
from ..mesher import generate_mesh  # Reuse the only repository meshing gateway.
from .mcp_tools import MCPMeshGateway  # Import deterministic parameter certification and materialization.
from .planner import PlanDecision, PlannerConfig, WorldModelPlanner, exact_region_exposure  # Import multi-step planning contracts.
from .regions import Partition  # Import semantic region aggregation.
from .world_model import RegionAction, ResidualWorldModel, WorldModelConfig, WorldState  # Import action-conditioned world-state contracts.

@dataclass(frozen=True)  # Keep the real adaptive-loop contract explicit.
class WorldVLAConfig:  # Configure multi-step real solves and safety gates.
    n_eq_budget: int = 60000  # Set the shared equation-count cap for the bridge component.
    max_solves: int = 6  # Permit a genuine multi-step adaptive trajectory.
    theta: float = 0.5  # Match the reference Dörfler bulk-marking parameter.
    gradation: float = 0.9  # Match the common Gmsh Lipschitz size smoothing.
    partition_mode: str = "drawn"  # Use visual drawings with geodesic fallback.
    require_reference: bool = True  # Compute reference errors unless a smoke run disables them.
    early_stop: bool = True  # Permit estimator-based convergence before the solve cap.
    estimator_target_ratio: float = 0.12  # Stop after a substantial reduction from the common probe.
    min_budget_use: float = 0.45  # Avoid declaring convergence on an accidentally tiny mesh.
    regression_tolerance: float = 0.03  # Force Dörfler after a proactive estimator regression.
    planner: PlannerConfig = field(default_factory=PlannerConfig)  # Configure finite-horizon search.
    model: WorldModelConfig = field(default_factory=WorldModelConfig)  # Configure the transition ensemble.

@dataclass(frozen=True)  # Make the completed adaptive trajectory easy to audit.
class WorldVLAResult:  # Summarize the real multi-step run.
    solves: int  # Store the number of counted CalculiX solves.
    proactive_actions: int  # Store the number of accepted world-model actions.
    dorfler_fallbacks: int  # Store the number of exact Dörfler actions.
    stopped_by: str  # Store the terminal condition.
    model_samples: int  # Store the final number of real regional transition rows.
    best_solve_index: int  # Store the best in-budget solve index.
    best_energy_error: float | None  # Store the best reference energy error when available.
    best_estimator: float  # Store the best in-budget ZZ estimator mass.
    visual_partition_s: float  # Store the one-time visual-semantic partition overhead.
    planning_s: float  # Store the cumulative local world-model search overhead.
    parameter_tools_s: float  # Store the cumulative deterministic MCP-compatible tool overhead.
    remeshing_s: float  # Store the cumulative Gmsh remeshing overhead.
    calculix_s: float  # Store the cumulative counted CalculiX wall time.
    action_log: tuple[dict, ...]  # Store every planned and certified action.
    def to_dict(self) -> dict:  # Convert the result to a JSON-safe payload.
        return {"solves": self.solves, "proactive_actions": self.proactive_actions, "dorfler_fallbacks": self.dorfler_fallbacks, "stopped_by": self.stopped_by, "model_samples": self.model_samples, "best_solve_index": self.best_solve_index, "best_energy_error": self.best_energy_error, "best_estimator": self.best_estimator, "timing_s": {"visual_partition": self.visual_partition_s, "world_model_planning": self.planning_s, "parameter_tools": self.parameter_tools_s, "gmsh_remeshing": self.remeshing_s, "calculix": self.calculix_s, "non_solver_total": self.visual_partition_s + self.planning_s + self.parameter_tools_s + self.remeshing_s, "trajectory_total": self.visual_partition_s + self.planning_s + self.parameter_tools_s + self.remeshing_s + self.calculix_s}, "action_log": list(self.action_log)}  # Return only primitive fields.

def _state_from_solve(partition: Partition, post, eta2: np.ndarray, labels: np.ndarray, marked: np.ndarray, hit_count: np.ndarray, record, step: int) -> tuple[WorldState, np.ndarray]:  # Convert one real finite-element solve into a regional world state.
    features = partition.features(post, eta2, labels)  # Aggregate estimator, stress, size, and volume by semantic region.
    adjacency = partition.adjacency_matrix(post.mesh, labels)  # Build the current regional interaction graph.
    error_fraction, element_fraction = exact_region_exposure(eta2, labels, marked, len(partition.seeds))  # Aggregate the exact current Dörfler support.
    prior_hits = np.asarray(hit_count, dtype=float).copy()  # Preserve the number of completed earlier Dörfler hits.
    updated_hits = prior_hits + (error_fraction > 1.0e-9).astype(float)  # Prepare the cumulative count for the next real solve.
    eq_per_elem = float(record.n_equations / max(record.n_elems, 1))  # Measure the current resource conversion exactly.
    state = WorldState(names=tuple(seed.name for seed in partition.seeds), err_sum=features.err_sum, elems=features.elems, sizes=features.h_meas, vm_max=features.vm_max, volume=features.volume, adjacency=adjacency, dorfler_error_fraction=error_fraction, dorfler_element_fraction=element_fraction, hit_count=prior_hits, n_equations=float(record.n_equations), eq_per_elem=max(eq_per_elem, 1.0e-9), h_min=partition.problem.h_min, h0=partition.problem.h0, dim=partition.problem.dim, step=step)  # Assemble the validated decision state without double-counting the current marking.
    return state, updated_hits  # Return the current state and counts prepared for the next observed state.

def _dorfler_decision(model: ResidualWorldModel, state: WorldState, template: PlanDecision | None, reason: str, horizon: int) -> PlanDecision:  # Build an exact Dörfler decision after any downstream gate.
    action = RegionAction(tuple(0 for _ in range(state.n_regions)), source="dorfler_fallback")  # Select no proactive regional depth.
    prediction = model.predict(state, action)  # Produce a consistent one-step audit forecast.
    baseline_score = 0.0 if template is None else float(template.baseline_score)  # Preserve the planner's baseline rollout score when available.
    evaluated = 1 if template is None else int(template.sequences_evaluated)  # Preserve the planner's evaluation accounting.
    sequence = tuple(tuple(0 for _ in range(state.n_regions)) for _ in range(max(horizon, 1)))  # Record the complete pure-Dörfler horizon.
    return PlanDecision(action=action, source="dorfler_fallback", reason=reason, horizon=horizon, baseline_score=baseline_score, selected_score=baseline_score, predicted_error=float(prediction.state.total_error), predicted_equations=float(prediction.state.n_equations), log_error_std=float(prediction.log_error_std), log_resource_std=float(prediction.log_resource_std), failure_probability=float(prediction.failure_probability), sequences_evaluated=evaluated, sequence=sequence)  # Return the common fallback record.

def run_world_model_vla(runner: FemRunner, partitioner, config: WorldVLAConfig | None = None, *, model: ResidualWorldModel | None = None, method: str = "world_model_vla") -> WorldVLAResult:  # Execute a common-probe, multi-step, Dörfler-anchored world-model trajectory.
    cfg = config or WorldVLAConfig()  # Bind explicit settings or safe defaults.
    if cfg.max_solves < 1:  # Require at least the common probe solve.
        raise ValueError("max_solves must be positive")  # Reject an empty experiment.
    if cfg.n_eq_budget <= 0:  # Require a positive resource cap.
        raise ValueError("n_eq_budget must be positive")  # Reject an invalid experiment budget.
    problem = runner.problem  # Read the shared finite-element problem.
    if cfg.require_reference:  # Build or load the independent reference only when requested.
        runner.ensure_reference()  # Preserve reference-based error comparability.
    visual_started = time.perf_counter()  # Start the one-time visual-semantic overhead timer.
    seeds = partitioner.propose(problem)  # Let the visual head define semantic regions before seeing a solve.
    drawings = list(getattr(partitioner, "last_drawings", []) or [])  # Read the corresponding visual outlines.
    partition = Partition(list(seeds), problem, gradation=cfg.gradation, assign_mode=cfg.partition_mode, drawings=drawings)  # Build the fixed semantic aggregation layer.
    visual_partition_s = time.perf_counter() - visual_started  # Measure the complete one-time partition overhead.
    transition_model = model or ResidualWorldModel(cfg.model)  # Reuse a supplied transition library or initialize one.
    planner_cfg = replace(cfg.planner, theta=cfg.theta)  # Keep the planner's future marking consistent with the real loop.
    planner = WorldModelPlanner(transition_model, planner_cfg)  # Construct the finite-horizon planner.
    gateway = MCPMeshGateway(refine_factor=transition_model.config.refine_factor, max_extra_regions=planner_cfg.max_extra_regions, max_extra_depth=planner_cfg.max_extra_depth, proactive_budget_safety=planner_cfg.budget_safety)  # Construct the exact parameter gateway.
    mesh = initial_mesh(problem)  # Use the same uniform common probe as the classical Dörfler baseline.
    hit_count = np.zeros(len(partition.seeds), dtype=float)  # Initialize completed real persistent-hotspot counts.
    previous_state: WorldState | None = None  # Reserve the last real state for transition learning.
    previous_action: RegionAction | None = None  # Reserve the executed action paired with the last state.
    previous_prediction: dict | None = None  # Reserve the last forecast for calibration diagnostics.
    initial_estimator: float | None = None  # Reserve the common-probe estimator mass.
    action_log: list[dict] = []  # Allocate the complete planning and tool audit trail.
    stopped_by = "round_cap"  # Set the default terminal condition.
    force_dorfler_once = False  # Initialize the observed-regression safety latch.
    planning_s = 0.0  # Initialize cumulative local world-model search time.
    parameter_tools_s = 0.0  # Initialize cumulative deterministic parameter-tool time.
    remeshing_s = 0.0  # Initialize cumulative Gmsh remeshing time.
    for solve_number in range(1, cfg.max_solves + 1):  # Execute the allowed real CalculiX solves.
        stage = "common_probe" if solve_number == 1 else f"cycle{solve_number - 1}"  # Name the common probe and later adaptive cycles.
        post, record = runner.solve_mesh(mesh, method=method, stage=stage)  # Run and count one real finite-element solve.
        eta2 = zz_indicator(problem, post)  # Compute the shared element-wise ZZ estimator.
        marked = dorfler_mark(eta2, cfg.theta)  # Compute the exact Dörfler safety action.
        labels = partition.assign(mesh)  # Aggregate the solved mesh into the fixed semantic regions.
        state, hit_count = _state_from_solve(partition, post, eta2, labels, marked, hit_count, record, solve_number - 1)  # Build the real post-solve world state.
        if previous_state is not None and previous_action is not None:  # Learn only from an executed real transition.
            transition_model.observe(previous_state, previous_action, state)  # Append action-conditioned residual evidence.
            if previous_prediction is not None:  # Compare the last forecast with the realized transition.
                predicted_error = max(float(previous_prediction["predicted_error"]), 1.0e-30)  # Read the prior total-error forecast.
                predicted_equations = max(float(previous_prediction["predicted_equations"]), 1.0)  # Read the prior resource forecast.
                record.extra["world_model_calibration"] = {"error_log_residual": float(np.log(max(state.total_error, 1.0e-30) / predicted_error)), "equation_log_residual": float(np.log(max(state.n_equations, 1.0) / predicted_equations))}  # Record calibration residuals without hiding errors.
            if np.any(previous_action.array(state.n_regions) > 0) and state.total_error > (1.0 + cfg.regression_tolerance) * previous_state.total_error:  # Detect a realized proactive estimator regression.
                force_dorfler_once = True  # Force the next real action back to exact Dörfler.
                record.extra["world_model_regression"] = True  # Preserve the triggering observation.
        if initial_estimator is None:  # Capture the common-probe estimator once.
            initial_estimator = state.total_error  # Establish the trajectory normalization.
        record.extra.update({"sum_eta2": state.total_error, "mesh_sha": mesh.sha(), "world_model_samples": transition_model.sample_count, "region_names": list(state.names), "region_error_share": [float(value) for value in state.error_share], "region_element_share": [float(value) for value in state.element_share], "region_sizes": [float(value) for value in state.sizes], "dorfler_error_fraction": [float(value) for value in state.dorfler_error_fraction], "dorfler_element_fraction": [float(value) for value in state.dorfler_element_fraction], "completed_hit_count": [float(value) for value in state.hit_count], "next_hit_count": [float(value) for value in hit_count], "dorfler_floor": "exact_element_marking_plus_nodewise_noncoarsening", "timing_s": {"visual_partition": visual_partition_s if solve_number == 1 else 0.0, "calculix": float(record.wall_s)}})  # Store every decision-relevant observation and separated overhead.
        if solve_number >= cfg.max_solves:  # Stop after the configured real-solve cap.
            stopped_by = "round_cap"  # Record the solve-cap terminal condition.
            record.extra["stop"] = stopped_by  # Store the stop reason on the final record.
            break  # Finish the trajectory.
        if record.n_equations >= cfg.n_eq_budget:  # Match the classical Dörfler pre-action resource stop.
            stopped_by = "dof_cap"  # Record the equation-cap terminal condition.
            record.extra["stop"] = stopped_by  # Store the stop reason on the final record.
            break  # Finish the trajectory.
        if len(marked) == 0:  # Stop when the exact Dörfler set is empty.
            stopped_by = "empty_marking"  # Record estimator exhaustion.
            record.extra["stop"] = stopped_by  # Store the stop reason on the final record.
            break  # Finish the trajectory.
        if cfg.early_stop and initial_estimator is not None and state.total_error <= cfg.estimator_target_ratio * initial_estimator and record.n_equations >= cfg.min_budget_use * cfg.n_eq_budget:  # Apply a conservative accuracy-and-resource stop.
            stopped_by = "estimator_target"  # Record successful estimator reduction.
            record.extra["stop"] = stopped_by  # Store the stop reason on the final record.
            break  # Finish the trajectory.
        planning_started = time.perf_counter()  # Start the local world-model planning timer.
        decision = planner.plan(state, cfg.n_eq_budget)  # Search multiple future adaptive actions before selecting the current one.
        planning_elapsed = time.perf_counter() - planning_started  # Measure the complete finite-horizon search overhead.
        planning_s += planning_elapsed  # Accumulate local planning overhead across the trajectory.
        if force_dorfler_once:  # Apply the observed-regression latch after planning for full diagnostics.
            decision = _dorfler_decision(transition_model, state, decision, "observed_proactive_regression", planner_cfg.horizon)  # Replace the executable action with exact Dörfler.
            force_dorfler_once = False  # Consume the one-round safety latch.
        tools_started = time.perf_counter()  # Start deterministic action certification and materialization timing.
        certificate = gateway.certify_action(state, decision.action, decision.predicted_equations, cfg.n_eq_budget, mesh.sha())  # Validate dimensions, depths, sparsity, and resource margin through the tool layer.
        if not certificate.accepted:  # Never pass an uncertified proactive action to Gmsh.
            decision = _dorfler_decision(transition_model, state, decision, "mcp_certificate_failed:" + ",".join(certificate.reasons), planner_cfg.horizon)  # Replace the action with exact Dörfler.
            certificate = gateway.certify_action(state, decision.action, decision.predicted_equations, cfg.n_eq_budget, mesh.sha())  # Re-certify the deterministic fallback.
        materialized = gateway.materialize(mesh, problem, labels, marked, decision.action, cfg.gradation, certificate.action_id)  # Convert the discrete action into exact nodal targets.
        tools_elapsed = time.perf_counter() - tools_started  # Measure deterministic parameter-tool overhead.
        parameter_tools_s += tools_elapsed  # Accumulate parameter-tool overhead across the trajectory.
        remesh_started = time.perf_counter()  # Start the Gmsh remeshing timer.
        next_mesh = generate_mesh(problem, materialized.field)  # Regenerate the complete mesh through the common Gmsh path.
        remesh_elapsed = time.perf_counter() - remesh_started  # Measure the complete Gmsh remeshing overhead.
        remeshing_s += remesh_elapsed  # Accumulate Gmsh remeshing overhead across the trajectory.
        audit = {"solve": solve_number, "decision": decision.to_dict(), "certificate": certificate.to_dict(), "target_sha256": materialized.target_sha256, "target_dominance_verified": materialized.dominance_verified, "dorfler_target_min": float(np.min(materialized.dorfler_h)), "selected_target_min": float(np.min(materialized.target_h)), "selected_target_max": float(np.max(materialized.target_h)), "timing_s": {"world_model_planning": planning_elapsed, "parameter_tools": tools_elapsed, "gmsh_remeshing": remesh_elapsed}}  # Assemble the complete action and overhead audit record.
        action_log.append(audit)  # Append the action before executing its next real solve.
        record.extra["world_model_plan"] = audit  # Attach the action to the solve that generated it.
        record.extra["timing_s"].update({"world_model_planning": planning_elapsed, "parameter_tools": tools_elapsed, "gmsh_remeshing": remesh_elapsed})  # Store separated non-solver overhead on the parent solve.
        previous_state = state  # Store the exact pre-action world state for online learning.
        previous_action = decision.action  # Store the exact executed discrete action.
        previous_prediction = {"predicted_error": decision.predicted_error, "predicted_equations": decision.predicted_equations}  # Store the forecast paired with the action.
        mesh = next_mesh  # Advance the real environment only after all contracts and timing records succeed.
    records = [record for record in runner.records if record.method == method]  # Isolate this method's counted solves.
    in_budget = [record for record in records if record.n_equations <= cfg.n_eq_budget and "sum_eta2" in record.extra]  # Select certifiable in-budget outcomes.
    candidates = in_budget or [record for record in records if "sum_eta2" in record.extra]  # Retain a fallback when the first mesh already exceeds the cap.
    best = min(candidates, key=lambda item: item.extra["sum_eta2"])  # Choose the smallest realized estimator mass.
    best.extra["certified_pick"] = True  # Mark the selected trajectory point.
    proactive_actions = sum(item["decision"]["source"] == "world_model" for item in action_log)  # Count accepted proactive actions.
    dorfler_fallbacks = len(action_log) - proactive_actions  # Count pure-Dörfler actions.
    calculix_s = float(sum(record.wall_s for record in records))  # Sum all counted real solver wall times.
    return WorldVLAResult(solves=len(records), proactive_actions=proactive_actions, dorfler_fallbacks=dorfler_fallbacks, stopped_by=stopped_by, model_samples=transition_model.sample_count, best_solve_index=int(best.solve_index), best_energy_error=None if best.e_energy is None else float(best.e_energy), best_estimator=float(best.extra["sum_eta2"]), visual_partition_s=float(visual_partition_s), planning_s=float(planning_s), parameter_tools_s=float(parameter_tools_s), remeshing_s=float(remeshing_s), calculix_s=calculix_s, action_log=tuple(action_log))  # Return the complete trajectory and separated cost summary.
