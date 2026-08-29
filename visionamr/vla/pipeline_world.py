# Multi-step world-model-guided VLA pipeline with deterministic mesh tools and honest solve accounting.  # Module purpose.
from __future__ import annotations  # Enable postponed annotations for compact typed records.
from dataclasses import dataclass, field  # Provide configuration and result containers.
from pathlib import Path  # Save optional transferable world-model checkpoints.
from typing import Any  # Type JSON-compatible diagnostic dictionaries.
import numpy as np  # Perform stable regional and prediction-error calculations.
from ..experiment import FemRunner, SolveRecord  # Reuse the single accountable CalculiX execution path.
from ..indicators import zz_indicator  # Measure real post-solve regional error evidence.
from .grades import grade_from_frac  # Recover ordinal state when a partitioner omits cached grades.
from .regions import Partition  # Hold the fixed one-shot visual region graph.
from .budget_certificate import certify_action_mesh_targeted  # Use bidirectional exact Gmsh budget targeting before every real solve.
from .tool_contract import MaterializedAction, MeshAction, MeshCertificate  # Enforce deterministic action and certificate ownership.
from .world_model import HybridGraphWorldModel, PlanResult, WorldModelConfig, WorldPlanner, WorldPlannerConfig, WorldState, build_world_state, dorfler_region_action  # Close the model-based control loop.


@dataclass  # Configure the independent WM-VLA method.
class WorldVLAConfig:  # Keep experimental limits and controller behavior explicit.
    n_eq_budget: int = 18000  # Set the hard free-equation cap for the medium 3-D bridge component.
    max_solves: int = 6  # Permit a genuine multi-step real feedback loop.
    target_error_ratio: float = 0.32  # Stop after reducing the first-state estimator to this fraction.
    min_budget_use: float = 0.55  # Prevent declaring success on an unnecessarily coarse mesh.
    exact_budget_safety: float = 0.985  # Reserve a narrow hard-cap margin during exact Gmsh certification.
    max_mesh_attempts: int = 6  # Bound bidirectional exact Gmsh certification calls per executed action.
    early_stop: bool = True  # Permit evidence-based termination before the real solve cap.
    guard_mode: str = "off"  # Choose "off" for pure WM-VLA or "dorfler_region_candidate" for a disclosed guard action.
    model_checkpoint_in: str | None = None  # Optionally load transferable transition memory from earlier instances.
    model_checkpoint_out: str | None = None  # Optionally persist all real transitions after the run.
    model: WorldModelConfig = field(default_factory=WorldModelConfig)  # Configure the hybrid graph ensemble.
    planner: WorldPlannerConfig = field(default_factory=WorldPlannerConfig)  # Configure finite-horizon counterfactual search.

    def validate(self) -> "WorldVLAConfig":  # Reject scientifically ambiguous controller settings early.
        if int(self.max_solves) < 1:  # Require at least the initial real state observation.
            raise ValueError("max_solves must be at least one")  # Report the invalid solve contract.
        if int(self.n_eq_budget) <= 0:  # Require a positive hard resource cap.
            raise ValueError("n_eq_budget must be positive")  # Report the invalid resource contract.
        if self.guard_mode not in ("off", "dorfler_region_candidate"):  # Restrict guard provenance to implemented modes.
            raise ValueError("guard_mode must be 'off' or 'dorfler_region_candidate'")  # Prevent hidden method mixtures.
        if not 0.0 < float(self.target_error_ratio) < 1.0:  # Require a meaningful convergence target.
            raise ValueError("target_error_ratio must lie strictly between zero and one")  # Report the invalid target.
        return self  # Return the verified configuration for fluent use.


@dataclass  # Return all scientifically relevant WM-VLA outcomes.
class WorldVLAResult:  # Separate the delivered iterate from the complete real trajectory.
    method: str  # Preserve the exact method label used in SolveRecord entries.
    solves: int  # Count expensive CalculiX solves consumed by this method.
    mesh_certifications: int  # Count Gmsh-only certification attempts separately.
    stopped_reason: str  # Explain convergence, planner termination, budget failure, or solve cap.
    best_solve_index: int  # Identify the best feasible delivered real iterate.
    initial_error: float  # Preserve the first measured global estimator total.
    final_error: float  # Preserve the last measured global estimator total.
    best_error: float  # Preserve the best feasible measured estimator total.
    model_transitions: int  # Count real transitions learned by the world model.
    actions: list[dict[str, Any]]  # Preserve all executed discrete actions and tool certificates.
    states: list[dict[str, Any]]  # Preserve all measured compact world states.
    info: dict[str, Any]  # Preserve purity, guard, and checkpoint diagnostics.


@dataclass  # Hold one measured solve and the regional data needed for the next decision.
class _MeasuredStep:  # Keep internal loop state aligned and explicit.
    state: WorldState  # Store the compact measured world state.
    post: Any  # Store the full post-processing object for estimator and features.
    record: SolveRecord  # Store the accountable real solve record.
    eta2: np.ndarray  # Store elementwise ZZ eta-squared values.
    labels: np.ndarray  # Store the current mesh-to-region assignment.


def _partition_grades(partitioner: Any, partition: Partition) -> np.ndarray:  # Read one-shot visual grades in stable region order.
    grade_map = dict(getattr(partitioner, "last_grades", {}) or {})  # Copy cached visual judgments without mutating the head.
    output: list[int] = []  # Accumulate one ordinal grade per region.
    for seed in partition.seeds:  # Traverse the authoritative region order.
        if seed.name in grade_map:  # Prefer the explicit visual grade when available.
            output.append(int(grade_map[seed.name]))  # Preserve the model's ordinal decision.
        elif seed.origin == "coarse":  # Handle an unpainted remainder explicitly.
            output.append(int(grade_map.get("field", 5)))  # Default only the remainder to the coarsest ordinal level.
        else:  # Recover a grade from an already validated visual fraction as a compatibility path.
            output.append(int(grade_from_frac(float(seed.h) / max(float(partition.problem.h0), 1.0e-12))))  # Convert size fraction to ordinal state.
    return np.asarray(output, dtype=int)  # Return the aligned grade vector.


def _initial_materialization(partition: Partition, grades: np.ndarray) -> MaterializedAction:  # Wrap the visual prior in the deterministic tool record.
    action = MeshAction("vision_initial", tuple(0 for _ in partition.seeds), source="vision_once", stop=False).validate(len(partition.seeds))  # Create a traceable zero-delta initialization.
    return MaterializedAction(action, np.asarray(grades, dtype=int).copy(), np.asarray(partition.sizes(), dtype=float).copy(), 0, 0.0, 1.0, True, "vision_initial")  # Preserve numerical ownership explicitly.


def _measure_mesh(runner: FemRunner, partition: Partition, grades: np.ndarray, mesh_certificate: MeshCertificate, method: str, stage: str, budget: int, extra: dict[str, Any] | None = None) -> _MeasuredStep:  # Execute one certified mesh and build the next world state.
    post, record = runner.solve_mesh(mesh_certificate.mesh, method=method, stage=stage, extra=dict(extra or {}))  # Route every expensive solve through FemRunner.
    eta2 = zz_indicator(runner.problem, post)  # Compute fresh elementwise error evidence after the actual solve.
    labels = partition.assign(mesh_certificate.mesh)  # Reassign the new mesh to the fixed visual region graph.
    features = partition.features(post, eta2, labels)  # Aggregate measured physics and resources by visual region.
    adjacency = partition.adjacency_matrix(mesh_certificate.mesh, labels)  # Recompute graph adjacency on the actual remesh.
    state = build_world_state(partition, features, adjacency, grades, record, runner.problem, budget)  # Construct the compact action-conditioned state.
    record.extra.update(  # Attach all decision-relevant evidence to the accountable solve record.
        sum_eta2=float(np.sum(eta2)),  # Preserve the global estimator total.
        regions={seed.name: float(size) for seed, size in zip(partition.seeds, partition.sizes())},  # Preserve executed region sizes.
        grades=[int(value) for value in grades],  # Preserve ordinal action coordinates.
        world_state=state.to_dict(),  # Preserve the measured compact state.
        mesh_certificate=mesh_certificate.to_dict(),  # Preserve exact pre-solve budget evidence.
        certificate_equation_delta=int(record.n_equations - mesh_certificate.n_equations),  # Audit the pre-solve equation count against CalculiX.
        method_purity={"uses_local_prediction": False, "llm_calls_inside_loop": False},  # Lock the clean scientific boundary explicitly.
    )  # Finish record enrichment.
    return _MeasuredStep(state, post, record, eta2, labels)  # Return all aligned measured data.


def _best_feasible_record(records: list[SolveRecord], budget: int) -> SolveRecord:  # Select the delivered real iterate without inventing a prediction-based result.
    feasible = [record for record in records if int(record.n_equations) <= int(budget)]  # Restrict delivery to the hard resource cap.
    candidates = feasible or list(records)  # Preserve a transparent fallback if every generated mesh violated the cap.
    with_reference = [record for record in candidates if record.e_energy is not None]  # Prefer the independently computed true benchmark metric.
    if with_reference:  # Select by actual reference-relative energy error when available.
        return min(with_reference, key=lambda record: float(record.e_energy))  # Return the best measured feasible iterate.
    return min(candidates, key=lambda record: float(record.extra.get("sum_eta2", float("inf"))))  # Fall back to the measured estimator only.


def _prediction_residual(plan: PlanResult, measured: WorldState) -> dict[str, float | None]:  # Quantify one-step world-model accuracy after real feedback.
    first = plan.sequence[0] if plan.sequence else {}  # Read the first imagined transition if one exists.
    predicted_error = first.get("predicted_total_error")  # Read the predicted global estimator total.
    predicted_equations = first.get("predicted_n_equations")  # Read the predicted resource total.
    error_log = None if predicted_error is None else float(np.log(max(float(measured.total_error), 1.0e-30) / max(float(predicted_error), 1.0e-30)))  # Compute signed log error residual.
    equation_relative = None if predicted_equations is None else float((int(measured.n_equations) - int(predicted_equations)) / max(int(measured.n_equations), 1))  # Compute signed relative resource residual.
    return {"error_log": error_log, "equation_relative": equation_relative}  # Return compact calibration evidence.


def run_world_vla(runner: FemRunner, partitioner: Any, config: WorldVLAConfig | None = None, world_model: HybridGraphWorldModel | None = None, *, method: str = "wm_vla") -> WorldVLAResult:  # Execute one independent multi-step WM-VLA trajectory.
    cfg = (config or WorldVLAConfig()).validate()  # Normalize and verify the controller contract.
    problem = runner.problem  # Read the immutable finite-element problem.
    runner.ensure_reference()  # Ensure all real solves receive independent energy and QoI errors.
    seeds = partitioner.propose(problem)  # Call the visual judgment exactly once before any solve.
    drawings = list(getattr(partitioner, "last_drawings", []) or [])  # Preserve the one-shot visual region geometry.
    if not drawings:  # Reject an unexecutable fixed-region world model explicitly.
        raise ValueError("WM-VLA requires named visual drawings; geodesic-only fallback is not enabled")  # Avoid silently collapsing all region actions into one remainder size.
    partition = Partition(seeds, problem, gradation=0.9, assign_mode="drawn", drawings=drawings)  # Freeze the visual region graph for online identification.
    grades = _partition_grades(partitioner, partition)  # Read the one-shot ordinal visual judgment.
    initial = _initial_materialization(partition, grades)  # Bind the visual prior to the deterministic tool contract.
    certificate = certify_action_mesh_targeted(problem, partition, drawings, initial, cfg.n_eq_budget, cfg.exact_budget_safety, cfg.max_mesh_attempts)  # Generate and exact-certify the initial mesh without CalculiX.
    mesh_certifications = int(certificate.attempts)  # Account for initial Gmsh-only work separately.
    if not certificate.budget_ok:  # Refuse to start from a mesh that violates the hard resource contract.
        raise RuntimeError(f"initial visual mesh could not meet equation budget {cfg.n_eq_budget}")  # Fail transparently before an expensive solve.
    partition = partition.with_sizes(certificate.sizes)  # Make the certified numerical field the authoritative current state.
    first = _measure_mesh(runner, partition, grades, certificate, method, "world_probe", cfg.n_eq_budget, {"visual_calls": 1, "world_step": 0})  # Obtain the first real physics observation.
    if world_model is None and cfg.model_checkpoint_in:  # Reuse transferable real transitions when requested.
        model = HybridGraphWorldModel.load(cfg.model_checkpoint_in)  # Load the compact human-auditable checkpoint.
        if int(model.problem_dim) != int(problem.dim):  # Prevent cross-dimensional resource-model corruption.
            raise ValueError("world-model checkpoint dimension does not match the problem")  # Stop unsafe transfer.
    else:  # Use the supplied model or initialize a fresh mechanics-informed ensemble.
        model = world_model or HybridGraphWorldModel(problem.dim, cfg.model)  # Preserve external pretraining when supplied.
    planner_cfg = WorldPlannerConfig(**dict(cfg.planner.__dict__))  # Copy planner settings so the caller's object is not mutated.
    planner_cfg.target_error_ratio = float(cfg.target_error_ratio)  # Align lookahead diagnostics with the real stopping target.
    planner_cfg.min_budget_use = float(cfg.min_budget_use)  # Align resource use between planner and executor.
    planner = WorldPlanner(problem, model, planner_cfg)  # Bind the receding-horizon controller.
    measured = first  # Start the feedback loop from the first real state.
    initial_error = float(first.state.total_error)  # Freeze the convergence reference before online updates.
    states = [first.state.to_dict()]  # Preserve every measured state in order.
    actions: list[dict[str, Any]] = []  # Preserve every executed action and certification.
    stopped_reason = "solve_cap"  # Default to the explicit hard solve limit.
    consecutive_non_improving = 0  # Detect persistent realized model failures without hiding them.
    while len([record for record in runner.records if record.method == method]) < int(cfg.max_solves):  # Respect the method-specific real solve cap.
        error_ratio = float(measured.state.total_error) / max(initial_error, 1.0e-30)  # Measure realized convergence from the first state.
        if cfg.early_stop and error_ratio <= float(cfg.target_error_ratio) and measured.state.budget_use >= float(cfg.min_budget_use):  # Apply the declared evidence-based stopping rule.
            stopped_reason = "target_reached"  # Disclose successful early convergence.
            measured.record.extra["world_stop"] = stopped_reason  # Preserve the stopping decision on the delivered trajectory.
            break  # Leave the expensive feedback loop.
        guard_action = None  # Default to a scientifically pure world-model action set.
        if cfg.guard_mode == "dorfler_region_candidate":  # Add a disclosed classical projection only in guarded mode.
            guard_action = dorfler_region_action(measured.state, measured.labels, measured.eta2, planner_cfg.dorfler_theta)  # Build the region-level guard candidate.
        eq_per_elem = float(measured.record.n_equations) / max(float(measured.record.n_elems), 1.0)  # Measure the current solver-specific resource conversion.
        plan = planner.plan(measured.state, eq_per_elem, guard_action, error_reference=initial_error)  # Perform bounded multi-step counterfactual search.
        measured.record.extra["world_plan"] = {"action": plan.action.to_dict(), "materialized": plan.materialized.to_dict(), "sequence": plan.sequence, "score": float(plan.score), "predicted_gain": float(plan.predicted_gain), "uncertainty": float(plan.uncertainty), "diagnostics": plan.diagnostics}  # Preserve the complete decision trace.
        if plan.action.stop:  # Respect an explicit risk-sensitive terminal action.
            stopped_reason = str(plan.diagnostics.get("reason", "planner_stop"))  # Preserve the planner's exact rationale.
            measured.record.extra["world_stop"] = stopped_reason  # Attach the terminal decision to the last real state.
            break  # Avoid a pointless remesh and solve.
        next_certificate = certify_action_mesh_targeted(problem, partition, drawings, plan.materialized, cfg.n_eq_budget, cfg.exact_budget_safety, cfg.max_mesh_attempts)  # Exact-certify only the selected first action.
        mesh_certifications += int(next_certificate.attempts)  # Account for all Gmsh-only correction attempts.
        action_log = {"step": int(measured.state.solve_index + 1), "action": plan.action.to_dict(), "materialized": plan.materialized.to_dict(), "certificate": next_certificate.to_dict(), "plan_score": float(plan.score), "predicted_gain": float(plan.predicted_gain), "uncertainty": float(plan.uncertainty)}  # Build the executable action audit record.
        actions.append(action_log)  # Preserve the selected action before execution.
        if not next_certificate.budget_ok:  # Refuse to solve an uncertified over-budget mesh.
            stopped_reason = "mesh_budget_failure"  # Disclose deterministic-tool failure.
            measured.record.extra["world_stop"] = stopped_reason  # Preserve failure at the last valid real state.
            break  # Leave the expensive feedback loop.
        previous = measured  # Freeze the current state for transition identification.
        partition_next = partition.with_sizes(next_certificate.sizes)  # Make the exact-certified action field authoritative.
        grades_next = np.asarray(plan.materialized.grades, dtype=int).copy()  # Apply the selected ordinal action coordinates.
        stage = f"world_step_{int(previous.state.solve_index)}"  # Build a stable human-readable stage label.
        measured = _measure_mesh(runner, partition_next, grades_next, next_certificate, method, stage, cfg.n_eq_budget, {"visual_calls": 1, "world_step": int(previous.state.solve_index), "world_action": plan.action.to_dict(), "world_prediction": plan.sequence[0] if plan.sequence else None})  # Execute exactly one selected action.
        update_info = model.update(previous.state, measured.state, next_certificate.sizes, weight=1.0)  # Learn immediately from the real transition.
        residual = _prediction_residual(plan, measured.state)  # Measure one-step model calibration.
        measured.record.extra["world_update"] = update_info  # Preserve online-learning evidence.
        measured.record.extra["world_prediction_residual"] = residual  # Preserve prediction-versus-reality evidence.
        actions[-1]["realized"] = {"total_error": float(measured.state.total_error), "n_equations": int(measured.state.n_equations), "e_energy": measured.record.e_energy, "e_qoi": measured.record.e_qoi, "prediction_residual": residual, "model_update": update_info}  # Close the selected action record with reality.
        realized_gain = float((previous.state.total_error - measured.state.total_error) / max(previous.state.total_error, 1.0e-30))  # Measure actual estimator improvement.
        consecutive_non_improving = consecutive_non_improving + 1 if realized_gain <= 0.0 else 0  # Track persistent adverse transitions.
        measured.record.extra["realized_gain"] = realized_gain  # Preserve the actual control reward.
        partition = partition_next  # Advance the authoritative region size field.
        grades = grades_next  # Advance the authoritative ordinal state.
        states.append(measured.state.to_dict())  # Preserve the new measured state.
        if consecutive_non_improving >= 2:  # Terminate after repeated real degradation rather than trusting extrapolation.
            stopped_reason = "two_non_improving_real_steps"  # Disclose failed model guidance explicitly.
            measured.record.extra["world_stop"] = stopped_reason  # Preserve failure on the real trajectory.
            break  # Leave the expensive loop.
    method_records = [record for record in runner.records if record.method == method]  # Isolate this independent method trajectory.
    best = _best_feasible_record(method_records, cfg.n_eq_budget)  # Select the best actual feasible deliverable.
    best.extra["certified_pick"] = True  # Mark the delivered record explicitly.
    best.extra["world_stop_reason"] = stopped_reason  # Preserve the complete run outcome on the deliverable.
    if cfg.model_checkpoint_out:  # Persist transferable real evidence when requested.
        checkpoint_path = model.save(Path(cfg.model_checkpoint_out))  # Save a human-auditable transition checkpoint.
        checkpoint_value = str(checkpoint_path)  # Preserve the exact path in result diagnostics.
    else:  # Disclose that no persistence was requested.
        checkpoint_value = None  # Preserve a JSON-compatible null.
    best_error = float(best.extra.get("sum_eta2", float("inf")))  # Read the measured estimator of the delivered iterate.
    return WorldVLAResult(  # Return the complete independent method result.
        method=method,  # Preserve the exact method label.
        solves=len(method_records),  # Report expensive CalculiX solves only.
        mesh_certifications=int(mesh_certifications),  # Report Gmsh-only certifications separately.
        stopped_reason=stopped_reason,  # Report the explicit terminal condition.
        best_solve_index=int(best.solve_index),  # Identify the delivered measured iterate.
        initial_error=float(initial_error),  # Preserve the first-state estimator.
        final_error=float(measured.state.total_error),  # Preserve the last-state estimator.
        best_error=best_error,  # Preserve the delivered-state estimator.
        model_transitions=int(model.n_transitions),  # Report real transition learning depth.
        actions=actions,  # Preserve the complete action trace.
        states=states,  # Preserve the complete measured-state trace.
        info={"uses_local_prediction": False, "visual_calls": 1, "llm_calls_inside_loop": 0, "guard_mode": cfg.guard_mode, "checkpoint": checkpoint_value, "region_names": list(first.state.names)},  # Preserve scientific method boundaries.
    )  # Finish WM-VLA result construction.
