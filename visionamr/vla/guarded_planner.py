from __future__ import annotations  # Enable compact annotations for guarded receding-horizon planning.
import hashlib  # Create deterministic action identifiers from exact states and regions.
from dataclasses import dataclass  # Define explicit guarded planning policy.
import numpy as np  # Rank semantic regions and evaluate compact imagined rollouts.
from .guarded_gateway import GuardedToolGateway  # Resolve every protected action through deterministic numerical tools.
from .planner import PlanResult, PlannerConfig, dorfler_guard_action  # Reuse the common decision audit and exact safety action.
from .world_model import OnlineRegionWorldModel, WorldPrediction  # Predict action-conditioned regional consequences.
from .world_state import WorldAction, WorldState  # Operate only on immutable audited states and discrete actions.
@dataclass(frozen=True)  # Keep the protected policy explicit in records and tests.
class GuardedPlannerConfig:  # Configure model-predictive allocation above the Dörfler floor.
    horizon: int = 3  # Evaluate several future mesh reallocations without extra global solves.
    beam_width: int = 5  # Retain a small set of plausible imagined futures.
    max_bonus_regions: int = 3  # Bound each semantic intervention while permitting repeated steps.
    candidate_limit: int = 10  # Bound model evaluations per imagined state.
    dorfler_theta: float = 0.50  # Match exact baseline bulk marking.
    discount: float = 0.82  # Prefer near-term reductions while preserving multi-step value.
    uncertainty_weight: float = 0.12  # Penalize unsupported predictions without blocking protected actions.
    resource_over_weight: float = 40.0  # Reject previews that cannot plausibly fit the hard budget.
    action_weight: float = 0.002  # Prefer smaller semantic bonuses when consequences are equal.
    minimum_guarded_advantage: float = 0.004  # Require a positive model-predicted gain over exact Dörfler.
    warmup_transitions: int = 1  # Collect one exact Dörfler transition before allocating semantic bonus resource.
def _guarded_action_id(state: WorldState, deltas: np.ndarray) -> str:  # Bind one semantic bonus choice to an exact world state.
    digest = hashlib.sha256()  # Start a cryptographic audit digest.
    digest.update(state.state_id.encode("ascii"))  # Bind the action to one observed or imagined state.
    digest.update(np.ascontiguousarray(deltas, dtype=np.int8).tobytes())  # Bind the exact selected region set.
    return "world_guarded-" + digest.hexdigest()[:10]  # Return a compact traceable id.
def _semantic_score(state: WorldState) -> np.ndarray:  # Rank features whose future error may be under-observed by a coarse probe.
    roles = state.roles  # Read fixed geometry-visible semantic channels.
    topology = 1.45 * roles[:, 2] + 0.85 * roles[:, 3] + 0.75 * roles[:, 0] + 0.35 * roles[:, 1] + 1.10 * roles[:, 5] - 0.25 * roles[:, 4]  # Prioritize openings, singular edges, loads, supports, and measured split children.
    physics = 0.55 * state.err_share / max(float(np.max(state.err_share)), 1.0e-12)  # Add measured error concentration without replacing semantic anticipation.
    efficiency = 0.25 * (state.err_share / np.maximum(state.elem_share, 1.0e-12))  # Reward regions with high error per allocated element.
    efficiency /= max(float(np.max(efficiency)), 1.0e-12)  # Normalize marginal-efficiency evidence.
    return topology + physics + efficiency  # Return one interpretable priority score per region.
def _guarded_candidates(state: WorldState, config: GuardedPlannerConfig) -> list[WorldAction]:  # Generate only exact Dörfler or Dörfler-plus-semantic actions.
    guard = dorfler_guard_action(state, config.dorfler_theta)  # Preserve faithful element-level AFEM as the first candidate.
    actions = [guard]  # Start the action set with the safety floor.
    score = _semantic_score(state)  # Rank future-relevant bridge features.
    order = [int(index) for index in np.argsort(-score) if score[index] > 0.0]  # Remove generic low-value regions.
    for count in range(1, min(config.max_bonus_regions, len(order)) + 1):  # Offer nested semantic bonus sets.
        selected = order[:count]  # Select the strongest visible mechanisms.
        deltas = np.zeros(state.n_regions, dtype=int)  # Allocate an all-hold discrete action.
        deltas[selected] = -1  # Request one adjacent refinement decision for each bonus region.
        names = ", ".join(state.names[index] for index in selected)  # Preserve a human-readable decision explanation.
        actions.append(WorldAction(action_id=_guarded_action_id(state, deltas), deltas=tuple(int(value) for value in deltas), kind="guarded", source="world_model_guarded", rationale=f"retain exact Dörfler and spend feasible residual budget on {names}"))  # Add one protected world action.
    error_order = [int(index) for index in np.argsort(-state.err_share) if state.origins[index] != "coarse"]  # Add a measured-but-nonbackground alternative.
    if error_order:  # Create one physics-focused protected candidate.
        selected = error_order[: min(2, config.max_bonus_regions, len(error_order))]  # Limit its scope.
        deltas = np.zeros(state.n_regions, dtype=int)  # Allocate the discrete action.
        deltas[selected] = -1  # Add refinement only above the Dörfler floor.
        actions.append(WorldAction(action_id=_guarded_action_id(state, deltas), deltas=tuple(int(value) for value in deltas), kind="guarded", source="world_model_guarded", rationale="retain exact Dörfler and reinforce high-error nonbackground regions"))  # Add the protected measured alternative.
    unique = []  # Deduplicate numerically identical bonus sets.
    seen = set()  # Track kind and delta signatures.
    for action in actions:  # Preserve deterministic proposal order.
        key = (action.kind, action.deltas)  # Ignore narrative differences for execution.
        if key in seen:  # Skip duplicate target fields.
            continue  # Continue to the next proposal.
        seen.add(key)  # Record the signature.
        unique.append(action)  # Preserve the unique candidate.
        if len(unique) >= config.candidate_limit:  # Enforce the branching cap.
            break  # Stop bounded generation.
    return unique  # Return only Dörfler-preserving actions.
def _cost(state: WorldState, action: WorldAction, prediction: WorldPrediction, config: GuardedPlannerConfig) -> float:  # Score deployable estimator consequences without reference-error leakage.
    eta_ratio = float(np.sqrt(max(float(prediction.err_sum.sum()), 1.0e-30) / max(float(state.total_eta2), 1.0e-30)))  # Normalize the predicted ZZ indicator to the current real state.
    budget_ratio = float(prediction.n_equations) / max(float(state.budget), 1.0)  # Normalize predicted resource use.
    over = max(budget_ratio - 1.0, 0.0)  # Measure hard-budget risk.
    return float(eta_ratio + config.uncertainty_weight * prediction.uncertainty + config.resource_over_weight * over**2 + config.action_weight * action.n_changed)  # Combine deployable accuracy, uncertainty, resource, and intervention cost.
class GuardedModelPredictivePlanner:  # Direct Dörfler-safe multi-step VLA using an action-conditioned world model.
    def __init__(self, config: GuardedPlannerConfig | None = None) -> None:  # Store one reproducible protected planning policy.
        self.config = config or GuardedPlannerConfig()  # Resolve explicit defaults.
    def plan(self, state: WorldState, gateway: GuardedToolGateway, model: OnlineRegionWorldModel, force_guard: bool = False) -> PlanResult:  # Select one executable root action.
        guard_action = dorfler_guard_action(state, self.config.dorfler_theta)  # Build the exact element-level safety token.
        guard_prediction = model.predict(state, gateway.preview(state, guard_action))  # Predict its deployable estimator consequence.
        guard_cost = _cost(state, guard_action, guard_prediction, self.config)  # Score the safety floor.
        if force_guard or len(model.transitions) < self.config.warmup_transitions:  # Collect real transition evidence or recover after a miss.
            reason = "forced_dorfler_after_model_miss" if force_guard else "dorfler_world_model_warmup"  # Explain the conservative decision.
            return PlanResult(action=guard_action, prediction=guard_prediction, guard_action=guard_action, guard_prediction=guard_prediction, selected_by=reason, predicted_advantage=0.0, rollout=({"depth": 0, "action": guard_action.action_id, "source": guard_action.source, "kind": guard_action.kind, "cost": float(guard_cost)},), candidates_evaluated=1)  # Return the faithful guard immediately.
        beam = [(0.0, state, None, tuple())]  # Initialize cumulative cost, imagined state, root action, and trace.
        evaluated = 0  # Count world-model consequence queries.
        for depth in range(max(int(self.config.horizon), 1)):  # Expand a finite protected rollout tree.
            expanded = []  # Collect successors at this depth.
            for cumulative, node_state, first_action, trace in beam:  # Expand every retained imagined future.
                for action in _guarded_candidates(node_state, self.config):  # Enumerate only Dörfler-preserving choices.
                    prediction = model.predict(node_state, gateway.preview(node_state, action))  # Predict action-conditioned consequences.
                    step_cost = _cost(node_state, action, prediction, self.config)  # Score the deployable proxy objective.
                    total_cost = float(cumulative + self.config.discount**depth * step_cost)  # Accumulate discounted rollout cost.
                    root_action = action if first_action is None else first_action  # Preserve the executable first action.
                    eta_ratio = float(np.sqrt(max(float(prediction.err_sum.sum()), 1.0e-30) / max(float(node_state.total_eta2), 1.0e-30)))  # Record the estimator consequence.
                    next_trace = trace + ({"depth": int(depth), "action": action.action_id, "source": action.source, "kind": action.kind, "cost": float(step_cost), "eta_ratio": eta_ratio, "predicted_equations": float(prediction.n_equations), "uncertainty": float(prediction.uncertainty)},)  # Append a transparent imagined transition.
                    expanded.append((total_cost, prediction.imagined_state(node_state), root_action, next_trace))  # Add the successor to the next beam.
                    evaluated += 1  # Count one world-model evaluation.
            expanded.sort(key=lambda item: (item[0], item[2].action_id if item[2] is not None else ""))  # Rank deterministically.
            beam = expanded[: max(int(self.config.beam_width), 1)]  # Retain a bounded set of futures.
            if not beam:  # Detect an impossible empty action set.
                break  # Fall back to exact Dörfler below.
        if not beam or beam[0][2] is None:  # Handle an empty planner result safely.
            return PlanResult(action=guard_action, prediction=guard_prediction, guard_action=guard_action, guard_prediction=guard_prediction, selected_by="dorfler_empty_guarded_plan", predicted_advantage=0.0, rollout=tuple(), candidates_evaluated=int(evaluated))  # Return the safety floor.
        best_action = beam[0][2]  # Read the root action from the best protected rollout.
        best_prediction = model.predict(state, gateway.preview(state, best_action))  # Re-evaluate the root on the exact current real state.
        best_cost = _cost(state, best_action, best_prediction, self.config)  # Compare root actions at one common state.
        advantage = float((guard_cost - best_cost) / max(abs(guard_cost), 1.0e-12))  # Normalize predicted gain over exact Dörfler.
        if best_action.kind != "guarded" or advantage < self.config.minimum_guarded_advantage:  # Reject a weak or already-baseline result.
            return PlanResult(action=guard_action, prediction=guard_prediction, guard_action=guard_action, guard_prediction=guard_prediction, selected_by="dorfler_no_guarded_advantage", predicted_advantage=0.0, rollout=tuple(beam[0][3]), candidates_evaluated=int(evaluated))  # Preserve the exact baseline action.
        return PlanResult(action=best_action, prediction=best_prediction, guard_action=guard_action, guard_prediction=guard_prediction, selected_by="world_model_guarded_advantage", predicted_advantage=float(advantage), rollout=tuple(beam[0][3]), candidates_evaluated=int(evaluated))  # Execute the Dörfler-dominating world action.
