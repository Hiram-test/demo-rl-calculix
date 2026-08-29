from __future__ import annotations  # Enable compact annotations for the multi-step pipeline.
from dataclasses import dataclass, field  # Define explicit configuration and result records.
import numpy as np  # Manage grades, regional features, and safety diagnostics.
from ..experiment import FemRunner  # Route every real global solve through the repository accountant.
from ..indicators import zz_indicator  # Observe physics without importing local-prediction sizes.
from .grades import grade_from_frac  # Initialize any vision region that lacks an explicit grade.
from .planner import ModelPredictivePlanner, PlanResult, PlannerConfig  # Select actions by multi-step imagined consequences.
from .regions import Partition  # Preserve irregular visual regions across remeshing.
from .tool_gateway import DeterministicToolGateway, ToolGatewayConfig  # Own exact parameters and selected-action meshes.
from .world_model import OnlineRegionWorldModel, WorldModelConfig  # Learn real action-conditioned transitions online.
from .world_state import Transition, make_state  # Build auditable states and training transitions.
@dataclass  # Keep the end-to-end control policy configurable from campaigns.
class WorldVLAConfig:  # Configure multi-step world-model VLA execution.
    n_eq_budget: int = 8000  # Set the hard free-equation cap.
    max_solves: int = 6  # Allow genuine multi-step control rather than fixing two rounds.
    min_solves: int = 3  # Require enough real transitions before early stopping.
    early_stop: bool = True  # Allow convergence or stagnation stopping after the minimum.
    min_relative_gain: float = 0.004  # Treat smaller repeated gains as practical stagnation.
    target_energy_error: float = 0.06  # Stop when a strong reference-verified solution is already reached.
    budget_tolerance: float = 1.02  # Admit only slight solver-count drift in the deliverable set.
    allow_split: bool = True  # Let measured concentration refine the visual state representation.
    split_after_solve: int = 2  # Keep the first transition on the original semantic graph.
    max_new_regions_per_step: int = 1  # Limit graph growth and preserve auditability.
    max_regions: int = 16  # Bound world-model state dimension.
    deterioration_fraction: float = 0.005  # Trigger Dörfler fallback after an actual learned-action regression.
    prediction_miss_fraction: float = 0.35  # Trigger fallback after a large optimistic model miss.
    partition_mode: str = "drawn"  # Preserve current irregular drawing assignment by default.
    gradation: float = 0.90  # Match the baseline remeshing regularity.
    planner: PlannerConfig = field(default_factory=PlannerConfig)  # Configure model-predictive planning.
    model: WorldModelConfig = field(default_factory=WorldModelConfig)  # Configure online transition learning.
    tools: ToolGatewayConfig = field(default_factory=ToolGatewayConfig)  # Configure deterministic parameter certification.
@dataclass(frozen=True)  # Return a compact, serializable execution summary.
class WorldVLAResult:  # Summarize the real loop and its safety behavior.
    solves: int  # Count real CalculiX solves.
    best_solve_index: int  # Identify the best budget-feasible record.
    best_energy_error: float  # Report reference energy error at the deliverable.
    best_qoi_error: float  # Report reference QoI error at the deliverable.
    best_n_equations: int  # Report deliverable resource use.
    world_actions: int  # Count learned actions actually executed.
    dorfler_actions: int  # Count exact Dörfler fallback actions actually executed.
    split_events: int  # Count state-representation refinements.
    stopped_early: bool  # Report whether the loop ended before its hard solve cap.
    stop_reason: str  # Explain the termination decision.
    action_trace: tuple[dict, ...]  # Preserve every plan, tool audit, prediction, and outcome.
    model_updates: tuple[dict, ...]  # Preserve online calibration diagnostics.
def _initial_grades(partitioner, partition: Partition) -> np.ndarray:  # Read only discrete grades from the vision head.
    grade_map = dict(getattr(partitioner, "last_grades", {}) or {})  # Copy model output for stable lookup.
    values = []  # Preserve partition order.
    for seed in partition.seeds:  # Resolve every visible and remainder region.
        if seed.name in grade_map:  # Prefer an explicit vision grade.
            values.append(int(grade_map[seed.name]))  # Store the discrete level.
        elif seed.origin == "coarse":  # Give the remainder a conservative coarse default.
            values.append(int(grade_map.get("field", 5)))  # Reuse the declared remainder grade when available.
        else:  # Recover a legacy drawing that contains only a fraction.
            values.append(int(grade_from_frac(float(seed.h) / float(partition.problem.h0))))  # Convert once at the boundary.
    return np.clip(np.asarray(values, dtype=int), 1, 5)  # Enforce the public grade range.
def _refresh(partition: Partition, post, eta2: np.ndarray, mesh) -> tuple[np.ndarray, object, np.ndarray]:  # Recompute the region graph after each real solve.
    labels = partition.assign(mesh)  # Assign new elements to persistent irregular regions.
    features = partition.features(post, eta2, labels)  # Aggregate measured error, stress, size, and resource state.
    adjacency = partition.adjacency_matrix(mesh, labels)  # Rebuild the remeshed region graph.
    return labels, features, adjacency  # Return all state-construction inputs.
def _split_partition(partition: Partition, post, eta2: np.ndarray, labels: np.ndarray, grades: np.ndarray, config: WorldVLAConfig) -> tuple[Partition, np.ndarray, bool]:  # Refine only the state representation where residual is concentrated.
    grown = partition.split_concentrated(post, eta2, labels, max_new=config.max_new_regions_per_step, max_seeds=config.max_regions)  # Add at most a small measured hotspot child.
    if len(grown.seeds) == len(partition.seeds):  # Detect no-op splitting.
        return partition, grades, False  # Preserve the current graph.
    old_grade = {seed.name: int(value) for seed, value in zip(partition.seeds, grades)}  # Map existing region decisions.
    new_grades = []  # Build grades in the grown partition order.
    for seed in grown.seeds:  # Resolve existing and new child regions.
        if seed.name in old_grade:  # Preserve an existing region's decision.
            new_grades.append(old_grade[seed.name])  # Copy its grade.
        else:  # Initialize a child from its parent.
            parent_name = seed.name.rsplit("_hot", 1)[0]  # Recover the split parent naming convention.
            parent_grade = int(old_grade.get(parent_name, 3))  # Use a neutral fallback only for malformed names.
            new_grades.append(max(parent_grade - 1, 1))  # Make the concentrated child one level finer.
    return grown, np.asarray(new_grades, dtype=int), True  # Return the expanded graph and discrete state.
def _record_world_state(record, state, certified, plan: PlanResult | None, model_update: dict | None) -> None:  # Store all world-model evidence in the solve record.
    record.extra["sum_eta2"] = float(state.total_eta2)  # Store the global estimator for baseline-compatible reporting.
    record.extra["world_state_id"] = state.state_id  # Bind the record to the model state.
    record.extra["grades"] = [int(value) for value in state.grades]  # Store discrete decisions.
    record.extra["regions"] = {name: float(size) for name, size in zip(state.names, state.sizes)}  # Store realized regional resolution.
    record.extra["n_eq_budget"] = int(state.budget)  # Store the hard cap.
    if certified is not None:  # Attach deterministic tool evidence after an executed action.
        record.extra["tool_audit"] = dict(certified.audit)  # Preserve mapping, hash, count, and mesh-pass data.
        record.extra["equation_estimate_error"] = int(record.n_equations - certified.estimated_equations)  # Verify pre-solve equation counting.
    if plan is not None:  # Attach planner evidence after a controlled transition.
        record.extra["world_plan"] = {"action_id": plan.action.action_id, "source": plan.action.source, "kind": plan.action.kind, "selected_by": plan.selected_by, "predicted_advantage": float(plan.predicted_advantage), "predicted_energy_error": float(plan.prediction.e_energy), "predicted_qoi_error": float(plan.prediction.e_qoi), "predicted_equations": float(plan.prediction.n_equations), "uncertainty": float(plan.prediction.uncertainty), "guard_action_id": plan.guard_action.action_id, "guard_predicted_energy_error": float(plan.guard_prediction.e_energy), "candidates_evaluated": int(plan.candidates_evaluated)}  # Store root comparison.
    if model_update is not None:  # Attach online learning evidence.
        record.extra["world_model_update"] = dict(model_update)  # Preserve calibration diagnostics.
def _best_record(records: list, budget: int, tolerance: float):  # Select the best honest deliverable under the resource contract.
    feasible = [record for record in records if record.n_equations <= tolerance * budget and record.e_energy is not None]  # Filter by actual solver equations.
    pool = feasible if feasible else [record for record in records if record.e_energy is not None]  # Remain diagnostic when all meshes violate the cap.
    return min(pool, key=lambda record: (float(record.e_energy), float(record.e_qoi), int(record.n_equations)))  # Prioritize energy, then QoI and resource.
def run_world_vla(runner: FemRunner, partitioner, config: WorldVLAConfig | None = None, *, method: str = "world_vla") -> WorldVLAResult:  # Execute the complete multi-step world-model VLA.
    cfg = config or WorldVLAConfig()  # Resolve configuration.
    if cfg.max_solves < 1:  # Reject an impossible campaign.
        raise ValueError("max_solves must be positive")  # Require at least one real state.
    problem = runner.problem  # Read the bridge component.
    runner.ensure_reference()  # Load or create the reference outside method solve accounting.
    seeds = partitioner.propose(problem)  # Ask the vision head for geometry-only regions and grades.
    drawings = list(getattr(partitioner, "last_drawings", []) or [])  # Retrieve irregular visual markup.
    partition = Partition(seeds, problem, gradation=cfg.gradation, assign_mode=cfg.partition_mode, drawings=drawings)  # Build the initial region graph.
    grades = _initial_grades(partitioner, partition)  # Read discrete visual decisions.
    gateway = DeterministicToolGateway(problem, cfg.tools)  # Instantiate the only numerical parameter authority.
    model = OnlineRegionWorldModel(problem, cfg.model)  # Instantiate the online action-conditioned transition model.
    planner = ModelPredictivePlanner(cfg.planner)  # Instantiate receding-horizon planning.
    initial = gateway.build_initial(partition, grades, cfg.n_eq_budget)  # Certify the semantic first mesh without a finite-element solve.
    partition = Partition(partition.seeds, problem, gradation=cfg.gradation, assign_mode=cfg.partition_mode, drawings=list(initial.drawings)).with_sizes(initial.sizes)  # Synchronize certified visual sizes.
    mesh = initial.mesh  # Pass the exact certified mesh to CalculiX.
    post, record = runner.solve_mesh(mesh, method=method, stage="semantic_probe")  # Execute the first real analysis.
    eta2 = zz_indicator(problem, post)  # Observe the solved physics.
    labels, features, adjacency = _refresh(partition, post, eta2, mesh)  # Build the first region graph state.
    state = make_state(partition, features, adjacency, record, grades, step=0, budget=cfg.n_eq_budget)  # Create the first audited world state.
    _record_world_state(record, state, initial, None, None)  # Store initial tool and physics evidence.
    trace = [{"solve": 1, "stage": record.stage, "action_id": initial.action.action_id, "kind": "semantic_initial", "selected_by": "vision_prior_plus_deterministic_tools", "estimated_equations": int(initial.estimated_equations), "actual_equations": int(record.n_equations), "energy_error": float(state.e_energy), "qoi_error": float(state.e_qoi), "sum_eta2": float(state.total_eta2), "mesh_passes": int(initial.mesh_passes)}]  # Start the execution trace.
    updates = []  # Collect online model updates.
    world_actions = 0  # Count learned actions.
    dorfler_actions = 0  # Count exact fallback actions.
    split_events = 0  # Count graph refinements.
    force_guard = False  # Start without a safety override.
    stopped_early = False  # Track termination before the hard cap.
    stop_reason = "solve_cap"  # Default to the configured maximum.
    recent_errors = [float(state.e_energy)]  # Track real improvement for stopping.
    while len([item for item in runner.records if item.method == method]) < cfg.max_solves:  # Continue until the real-solve cap.
        solve_count = len([item for item in runner.records if item.method == method])  # Count only this method.
        if cfg.allow_split and solve_count >= cfg.split_after_solve:  # Refine state representation after enough real evidence.
            grown, grown_grades, did_split = _split_partition(partition, post, eta2, labels, grades, cfg)  # Try one concentrated child.
            if did_split:  # Rebuild the current state on the unchanged real solution.
                partition = grown  # Adopt the expanded visual graph.
                grades = grown_grades  # Adopt child grades.
                labels, features, adjacency = _refresh(partition, post, eta2, mesh)  # Re-aggregate the same measured field.
                state = make_state(partition, features, adjacency, record, grades, step=state.step, budget=cfg.n_eq_budget)  # Rebind planning to the expanded graph.
                split_events += 1  # Count the representation change.
        plan = planner.plan(state, gateway, model, force_guard=force_guard)  # Compare multi-step world actions against exact Dörfler.
        force_guard = False  # Consume a one-step safety override.
        preview = gateway.preview(state, plan.action)  # Preserve the deterministic prediction input used by the model.
        certified = gateway.certify(partition, state, plan.action, mesh, eta2)  # Materialize only the selected action.
        if plan.action.kind == "dorfler":  # Count exact AFEM fallback execution.
            dorfler_actions += 1  # Increment the safety action count.
            partition_next = partition  # Preserve drawings because exact Dörfler used an element field.
        else:  # Count a learned region action.
            world_actions += 1  # Increment the world-model action count.
            partition_next = Partition(partition.seeds, problem, gradation=cfg.gradation, assign_mode=cfg.partition_mode, drawings=list(certified.drawings)).with_sizes(certified.sizes)  # Synchronize selected region parameters.
        next_mesh = certified.mesh  # Retrieve the exact solver-ready Gmsh mesh.
        next_post, next_record = runner.solve_mesh(next_mesh, method=method, stage=f"world_step_{solve_count + 1}")  # Execute one real controlled transition.
        next_eta2 = zz_indicator(problem, next_post)  # Observe next physics.
        next_labels, next_features, next_adjacency = _refresh(partition_next, next_post, next_eta2, next_mesh)  # Build the next region graph.
        next_grades = np.asarray(certified.grades, dtype=int)  # Carry exact validated grades.
        next_state = make_state(partition_next, next_features, next_adjacency, next_record, next_grades, step=state.step + 1, budget=cfg.n_eq_budget)  # Create the next real state.
        transition = Transition(state=state, action=plan.action, preview_sizes=np.asarray(preview.sizes, dtype=float), preview_n_equations=float(preview.n_equations), next_state=next_state)  # Bind action, tool prediction, and measured consequence.
        update = model.observe(transition)  # Calibrate only from the real transition.
        updates.append(update)  # Preserve the online update.
        predicted_gain = float(state.e_energy - plan.prediction.e_energy)  # Measure expected energy improvement.
        actual_gain = float(state.e_energy - next_state.e_energy)  # Measure realized energy improvement.
        optimistic_miss = float(max(predicted_gain - actual_gain, 0.0) / max(abs(predicted_gain), 1.0e-12)) if predicted_gain > 0.0 else 0.0  # Quantify unsupported optimism.
        regressed = bool(next_state.e_energy > state.e_energy * (1.0 + cfg.deterioration_fraction))  # Detect actual deterioration.
        if plan.action.kind != "dorfler" and (regressed or optimistic_miss > cfg.prediction_miss_fraction):  # Shield the next step after a learned-action miss.
            force_guard = True  # Force exact Dörfler once before trusting another learned action.
        _record_world_state(next_record, next_state, certified, plan, update)  # Store all evidence on the resulting solve.
        next_record.extra["predicted_gain"] = float(predicted_gain)  # Store expected improvement.
        next_record.extra["actual_gain"] = float(actual_gain)  # Store measured improvement.
        next_record.extra["optimistic_miss_fraction"] = float(optimistic_miss)  # Store model calibration.
        next_record.extra["next_force_guard"] = bool(force_guard)  # Store safety-controller state.
        trace.append({"solve": int(solve_count + 1), "stage": next_record.stage, "action_id": plan.action.action_id, "source": plan.action.source, "kind": plan.action.kind, "selected_by": plan.selected_by, "predicted_advantage": float(plan.predicted_advantage), "predicted_energy_error": float(plan.prediction.e_energy), "actual_energy_error": float(next_state.e_energy), "actual_qoi_error": float(next_state.e_qoi), "predicted_equations": float(plan.prediction.n_equations), "estimated_equations": int(certified.estimated_equations), "actual_equations": int(next_record.n_equations), "uncertainty": float(plan.prediction.uncertainty), "sum_eta2": float(next_state.total_eta2), "mesh_passes": int(certified.mesh_passes), "regressed": bool(regressed), "force_guard_next": bool(force_guard), "rollout": list(plan.rollout)})  # Append the complete transition trace.
        partition = partition_next  # Advance the persistent visual graph.
        grades = next_grades  # Advance discrete decisions.
        mesh = next_mesh  # Advance the current mesh.
        post = next_post  # Advance solved fields.
        record = next_record  # Advance solve metadata.
        eta2 = next_eta2  # Advance element indicators.
        labels = next_labels  # Advance region assignment.
        features = next_features  # Advance regional features.
        adjacency = next_adjacency  # Advance graph edges.
        state = next_state  # Advance the real world state.
        recent_errors.append(float(state.e_energy))  # Update stopping history.
        real_solves = len([item for item in runner.records if item.method == method])  # Recount method solves.
        if cfg.early_stop and real_solves >= cfg.min_solves and state.e_energy <= cfg.target_energy_error and state.n_equations <= cfg.budget_tolerance * cfg.n_eq_budget:  # Stop on achieved accuracy and budget.
            stopped_early = True  # Record early termination.
            stop_reason = "target_energy_error"  # Explain the condition.
            break  # Return the strong deliverable.
        if cfg.early_stop and real_solves >= max(cfg.min_solves, 4) and len(recent_errors) >= 3:  # Test practical stagnation only after several transitions.
            gain_a = (recent_errors[-3] - recent_errors[-2]) / max(recent_errors[-3], 1.0e-12)  # Compute the older relative gain.
            gain_b = (recent_errors[-2] - recent_errors[-1]) / max(recent_errors[-2], 1.0e-12)  # Compute the latest relative gain.
            if max(gain_a, gain_b) < cfg.min_relative_gain and plan.action.kind == "dorfler":  # Stop only after the safety baseline also stagnates.
                stopped_early = True  # Record early termination.
                stop_reason = "dorfler_confirmed_stagnation"  # Explain the conservative stop.
                break  # Avoid unproductive extra solves.
    records = [item for item in runner.records if item.method == method]  # Collect only this method's real solves.
    best = _best_record(records, cfg.n_eq_budget, cfg.budget_tolerance)  # Select the reference-verified budget-feasible deliverable.
    best.extra["certified_pick"] = True  # Mark the exact record used for comparison.
    return WorldVLAResult(solves=len(records), best_solve_index=int(best.solve_index), best_energy_error=float(best.e_energy), best_qoi_error=float(best.e_qoi), best_n_equations=int(best.n_equations), world_actions=int(world_actions), dorfler_actions=int(dorfler_actions), split_events=int(split_events), stopped_early=bool(stopped_early), stop_reason=stop_reason, action_trace=tuple(trace), model_updates=tuple(updates))  # Return an auditable multi-step result.
