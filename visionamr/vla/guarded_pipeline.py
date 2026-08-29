from __future__ import annotations  # Enable compact annotations for the protected multi-step loop.
import numpy as np  # Compare predicted and measured estimator transitions.
from ..experiment import FemRunner  # Route every real solve through the repository accountant.
from ..indicators import zz_indicator  # Observe deployable finite-element error evidence.
from .guarded_gateway import GuardedToolGateway  # Materialize exact Dörfler backbones plus model-selected bonuses.
from .guarded_planner import GuardedModelPredictivePlanner, GuardedPlannerConfig  # Select protected multi-step actions.
from .regions import Partition  # Preserve the geometry-only semantic region graph.
from .tool_gateway import ToolGatewayConfig, ToolPreview  # Build exact executed-action evidence for online calibration.
from .world_model import OnlineRegionWorldModel, WorldModelConfig  # Learn action-conditioned transition residuals online.
from .world_pipeline import WorldVLAConfig, WorldVLAResult, _best_record, _initial_grades, _record_world_state, _refresh  # Reuse audited setup and reporting helpers.
from .world_state import Transition, make_state  # Bind real states, actions, and consequences.
def run_guarded_world_vla(runner: FemRunner, partitioner, config: WorldVLAConfig | None = None, *, planner_config: GuardedPlannerConfig | None = None, method: str = "world_vla_guarded") -> WorldVLAResult:  # Execute deployable Dörfler-protected world-model control.
    cfg = config or WorldVLAConfig(max_solves=6, min_solves=3, early_stop=False, allow_split=False)  # Resolve a genuine multi-step default without oracle stopping.
    if cfg.max_solves < 2:  # Require a probe and at least one controlled transition.
        raise ValueError("guarded world VLA requires at least two real solves")  # Prevent a meaningless world-model run.
    problem = runner.problem  # Read the immutable bridge component.
    runner.ensure_reference()  # Load reference data only for offline evaluation, never for action scoring.
    seeds = partitioner.propose(problem)  # Obtain geometry-only semantic regions and discrete grades.
    drawings = list(getattr(partitioner, "last_drawings", []) or [])  # Preserve irregular drawing geometry.
    partition = Partition(seeds, problem, gradation=cfg.gradation, assign_mode=cfg.partition_mode, drawings=drawings)  # Build the persistent region graph.
    grades = _initial_grades(partitioner, partition)  # Read only discrete vision output.
    tool_policy = cfg.tools if isinstance(cfg.tools, ToolGatewayConfig) else ToolGatewayConfig()  # Resolve deterministic numerical policy.
    gateway = GuardedToolGateway(problem, tool_policy)  # Make the tool layer the sole owner of continuous mesh parameters.
    model_policy = cfg.model if isinstance(cfg.model, WorldModelConfig) else WorldModelConfig()  # Resolve online model policy.
    model = OnlineRegionWorldModel(problem, model_policy)  # Instantiate the action-conditioned transition model.
    planner = GuardedModelPredictivePlanner(planner_config or GuardedPlannerConfig(horizon=max(int(cfg.planner.horizon), 1), beam_width=max(int(cfg.planner.beam_width), 1), max_bonus_regions=min(int(cfg.tools.max_changed_regions), 3), dorfler_theta=float(cfg.tools.dorfler_theta)))  # Match the existing high-level configuration.
    initial = gateway.build_initial(partition, grades, cfg.n_eq_budget)  # Certify the semantic first mesh with mesh-only budget correction.
    partition = Partition(partition.seeds, problem, gradation=cfg.gradation, assign_mode=cfg.partition_mode, drawings=list(initial.drawings)).with_sizes(initial.sizes)  # Synchronize initial drawing metadata.
    mesh = initial.mesh  # Carry the exact certified mesh into CalculiX.
    post, record = runner.solve_mesh(mesh, method=method, stage="semantic_probe")  # Execute the first real global solve.
    eta2 = zz_indicator(problem, post)  # Compute deployable element-level error evidence.
    labels, features, adjacency = _refresh(partition, post, eta2, mesh)  # Aggregate the first region graph state.
    state = make_state(partition, features, adjacency, record, grades, step=0, budget=cfg.n_eq_budget)  # Freeze the first audited state.
    _record_world_state(record, state, initial, None, None)  # Persist the semantic probe and tool certificate.
    record.extra["control_uses_reference_error"] = False  # State the scientific boundary explicitly.
    trace = [{"solve": 1, "stage": record.stage, "action_id": initial.action.action_id, "kind": "semantic_initial", "selected_by": "vision_prior_plus_deterministic_tools", "estimated_equations": int(initial.estimated_equations), "actual_equations": int(record.n_equations), "energy_error_evaluation_only": float(state.e_energy), "qoi_error_evaluation_only": float(state.e_qoi), "sum_eta2_control": float(state.total_eta2), "mesh_passes": int(initial.mesh_passes), "world_bonus_accepted": False}]  # Start the complete execution trace.
    updates = []  # Preserve online model calibration summaries.
    world_actions = 0  # Count only accepted Dörfler-plus-world bonuses.
    dorfler_actions = 0  # Count exact Dörfler executions and guarded budget fallbacks.
    force_guard = False  # Start without a safety override.
    for transition_index in range(1, int(cfg.max_solves)):  # Execute a bounded number of real transitions.
        plan = planner.plan(state, gateway, model, force_guard=force_guard)  # Compare only exact Dörfler-preserving actions.
        force_guard = False  # Consume any one-step recovery request.
        certified = gateway.certify(partition, state, plan.action, mesh, eta2)  # Materialize only the selected action through Gmsh.
        bonus_accepted = bool(certified.audit.get("world_bonus_accepted", False))  # Read the exact tool execution result.
        if plan.action.kind == "guarded" and bonus_accepted:  # Count a real world-model allocation only when it reached the mesh.
            world_actions += 1  # Record one protected model intervention.
        else:  # Treat exact guards and guarded budget fallbacks as Dörfler executions.
            dorfler_actions += 1  # Record one safety-floor action.
        executed_preview = ToolPreview(action=plan.action, grades=np.asarray(certified.grades, dtype=int), sizes=np.asarray(certified.sizes, dtype=float), n_equations=float(certified.estimated_equations), audit=dict(certified.audit))  # Rebind prediction to the parameters and count that actually reached CalculiX.
        executed_prediction = model.predict(state, executed_preview)  # Predict the certified transition rather than the cheap planning preview.
        next_mesh = certified.mesh  # Retrieve the exact solver-ready mesh.
        next_post, next_record = runner.solve_mesh(next_mesh, method=method, stage=f"guarded_world_step_{transition_index + 1}")  # Execute one real controlled transition.
        next_eta2 = zz_indicator(problem, next_post)  # Observe the resulting estimator field.
        next_labels, next_features, next_adjacency = _refresh(partition, next_post, next_eta2, next_mesh)  # Rebuild the persistent region graph.
        next_grades = np.asarray(certified.grades, dtype=int)  # Carry validated discrete decisions.
        next_state = make_state(partition, next_features, next_adjacency, next_record, next_grades, step=state.step + 1, budget=cfg.n_eq_budget)  # Freeze the measured successor.
        transition = Transition(state=state, action=plan.action, preview_sizes=np.asarray(certified.sizes, dtype=float), preview_n_equations=float(certified.estimated_equations), next_state=next_state)  # Pair exact action evidence with the real consequence.
        update = model.observe(transition)  # Learn only from the completed real transition.
        updates.append(update)  # Preserve calibration evidence.
        predicted_gain = float(state.total_eta2 - executed_prediction.residual_total_eta2)  # Measure expected deployable estimator reduction.
        actual_gain = float(state.total_eta2 - next_state.total_eta2)  # Measure real deployable estimator reduction.
        optimistic_miss = float(max(predicted_gain - actual_gain, 0.0) / max(abs(predicted_gain), 1.0e-12)) if predicted_gain > 0.0 else 0.0  # Quantify unsupported optimism.
        regressed = bool(next_state.total_eta2 > state.total_eta2 * (1.0 + cfg.deterioration_fraction))  # Detect actual estimator deterioration without a reference solution.
        if plan.action.kind == "guarded" and bonus_accepted and (regressed or optimistic_miss > cfg.prediction_miss_fraction):  # Shield the next state after a protected model miss.
            force_guard = True  # Force exact Dörfler once before another semantic bonus.
        _record_world_state(next_record, next_state, certified, plan, update)  # Preserve common solver, model, and tool evidence.
        next_record.extra["control_uses_reference_error"] = False  # Declare that reference metrics remained offline.
        next_record.extra["predicted_sum_eta2"] = float(executed_prediction.residual_total_eta2)  # Store the certified action prediction.
        next_record.extra["predicted_eta_ratio"] = float(np.sqrt(max(executed_prediction.residual_total_eta2, 1.0e-30) / max(state.total_eta2, 1.0e-30)))  # Store the deployable relative objective.
        next_record.extra["actual_eta_ratio"] = float(np.sqrt(max(next_state.total_eta2, 1.0e-30) / max(state.total_eta2, 1.0e-30)))  # Store the measured relative objective.
        next_record.extra["predicted_gain_eta2"] = float(predicted_gain)  # Store expected improvement.
        next_record.extra["actual_gain_eta2"] = float(actual_gain)  # Store measured improvement.
        next_record.extra["optimistic_miss_fraction"] = float(optimistic_miss)  # Store online calibration quality.
        next_record.extra["next_force_guard"] = bool(force_guard)  # Store safety-controller state.
        trace.append({"solve": int(transition_index + 1), "stage": next_record.stage, "action_id": plan.action.action_id, "source": plan.action.source, "kind": plan.action.kind, "selected_by": plan.selected_by, "action_deltas": [int(value) for value in plan.action.deltas], "rationale": plan.action.rationale, "predicted_advantage": float(plan.predicted_advantage), "predicted_sum_eta2": float(executed_prediction.residual_total_eta2), "actual_sum_eta2": float(next_state.total_eta2), "predicted_eta_ratio": float(np.sqrt(max(executed_prediction.residual_total_eta2, 1.0e-30) / max(state.total_eta2, 1.0e-30))), "actual_eta_ratio": float(np.sqrt(max(next_state.total_eta2, 1.0e-30) / max(state.total_eta2, 1.0e-30))), "energy_error_evaluation_only": float(next_state.e_energy), "qoi_error_evaluation_only": float(next_state.e_qoi), "estimated_equations": int(certified.estimated_equations), "actual_equations": int(next_record.n_equations), "uncertainty": float(executed_prediction.uncertainty), "mesh_passes": int(certified.mesh_passes), "world_bonus_accepted": bool(bonus_accepted), "accepted_bonus_factor": certified.audit.get("accepted_bonus_factor"), "regressed_eta2": bool(regressed), "force_guard_next": bool(force_guard), "rollout": list(plan.rollout)})  # Append full action, prediction, tool, and outcome evidence.
        mesh = next_mesh  # Advance the real mesh.
        post = next_post  # Advance solved physical fields.
        record = next_record  # Advance solve metadata.
        eta2 = next_eta2  # Advance element indicators.
        labels = next_labels  # Advance region assignment.
        features = next_features  # Advance regional features.
        adjacency = next_adjacency  # Advance graph edges.
        grades = next_grades  # Advance discrete decisions.
        state = next_state  # Advance the audited world state.
    records = [item for item in runner.records if item.method == method]  # Collect only this controller's real solves.
    best = _best_record(records, cfg.n_eq_budget, cfg.budget_tolerance)  # Select the best budget-feasible offline evaluation record.
    best.extra["evaluation_best_pick"] = True  # Mark the evaluation-only best prefix explicitly.
    return WorldVLAResult(solves=len(records), best_solve_index=int(best.solve_index), best_energy_error=float(best.e_energy), best_qoi_error=float(best.e_qoi), best_n_equations=int(best.n_equations), world_actions=int(world_actions), dorfler_actions=int(dorfler_actions), split_events=0, stopped_early=False, stop_reason="solve_cap", action_trace=tuple(trace), model_updates=tuple(updates))  # Return a complete protected multi-step result.
