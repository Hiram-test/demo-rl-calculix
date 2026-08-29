"""Risk-aware multi-step planner with exact Dörfler fallback."""  # Describe the module purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
from dataclasses import dataclass  # Import immutable planning contracts.
from itertools import combinations  # Import bounded regional action combinations.
import math  # Import logarithms for scale-invariant planning scores.
import numpy as np  # Import numerical arrays for regional ranking.
from .world_model import RegionAction, ResidualWorldModel, WorldPrediction, WorldState, semantic_persistence  # Import the world-model contracts.

@dataclass(frozen=True)  # Keep planning settings explicit and reproducible.
class PlannerConfig:  # Configure finite-horizon world-model search.
    horizon: int = 4  # Roll several adaptive steps ahead before choosing the current action.
    beam_width: int = 18  # Retain only the strongest partial trajectories.
    candidate_regions: int = 5  # Consider a small number of physically ranked regions per rollout.
    max_extra_regions: int = 2  # Limit simultaneous advance investment to avoid diffuse refinement.
    max_extra_depth: int = 2  # Allow at most two delegated regional refinement levels.
    theta: float = 0.5  # Match the exact Dörfler bulk parameter used by the safety policy.
    discount: float = 0.82  # Discount distant rollout costs while retaining multi-step effects.
    resource_weight: float = 0.16  # Penalize equation growth in the rollout objective.
    uncertainty_weight: float = 0.55  # Penalize epistemically uncertain world-model trajectories.
    failure_weight: float = 1.25  # Penalize predicted transition failure risk.
    over_budget_weight: float = 12.0  # Strongly penalize trajectories above the hard resource cap.
    budget_safety: float = 0.98  # Require a small predicted equation-count margin for proactive actions.
    max_budget_ratio: float = 1.10  # Reject severe projected overshoots even during beam search.
    uncertainty_limit: float = 0.42  # Reject a proactive first action whose error uncertainty is too high.
    failure_limit: float = 0.38  # Reject a proactive first action whose model-risk score is too high.
    min_relative_gain: float = 0.01  # Require a robust score improvement over pure Dörfler.
    persistence_weight: float = 0.16  # Rank repeat-hit mechanisms for action enumeration.

@dataclass(frozen=True)  # Make the planner decision fully auditable.
class PlanDecision:  # Store the selected immediate action and comparison against Dörfler.
    action: RegionAction  # Store the immediate regional macro-action.
    source: str  # Record world-model selection or a specific fallback route.
    reason: str  # Explain the acceptance or fallback gate.
    horizon: int  # Record the rollout horizon used for this decision.
    baseline_score: float  # Store the pure-Dörfler rollout score.
    selected_score: float  # Store the accepted rollout score.
    predicted_error: float  # Store the selected first-step total estimator prediction.
    predicted_equations: float  # Store the selected first-step equation prediction.
    log_error_std: float  # Store first-step epistemic error uncertainty.
    log_resource_std: float  # Store first-step epistemic resource uncertainty.
    failure_probability: float  # Store the first-step model-risk score.
    sequences_evaluated: int  # Store the number of rollout branches evaluated.
    sequence: tuple[tuple[int, ...], ...]  # Store the complete selected rollout action sequence.
    def to_dict(self) -> dict:  # Convert the decision to a JSON-safe experiment record.
        return {"action": list(self.action.extra_depth), "source": self.source, "reason": self.reason, "horizon": self.horizon, "baseline_score": self.baseline_score, "selected_score": self.selected_score, "predicted_error": self.predicted_error, "predicted_equations": self.predicted_equations, "log_error_std": self.log_error_std, "log_resource_std": self.log_resource_std, "failure_probability": self.failure_probability, "sequences_evaluated": self.sequences_evaluated, "sequence": [list(values) for values in self.sequence]}  # Return only primitive values.

@dataclass(frozen=True)  # Store one internal beam-search node.
class _BeamNode:  # Carry a predicted world and its action history.
    state: WorldState  # Store the predicted state at the current rollout depth.
    score: float  # Store the cumulative discounted robust cost.
    actions: tuple[RegionAction, ...]  # Store the complete action prefix.
    first_prediction: WorldPrediction  # Store the real next-step prediction for gating.
    last_prediction: WorldPrediction  # Store the most recent transition prediction.

def exact_region_exposure(eta2: np.ndarray, labels: np.ndarray, marked: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:  # Aggregate exact element-wise Dörfler support by semantic region.
    values = np.asarray(eta2, dtype=float).reshape(-1)  # Normalize the estimator vector.
    region_labels = np.asarray(labels, dtype=int).reshape(-1)  # Normalize the element labels.
    marked_mask = np.zeros(len(values), dtype=bool)  # Allocate the exact marked-element mask.
    marked_mask[np.asarray(marked, dtype=int)] = True  # Activate the Dörfler-selected elements.
    error_fraction = np.zeros(count, dtype=float)  # Allocate marked estimator fractions.
    element_fraction = np.zeros(count, dtype=float)  # Allocate marked element fractions.
    for region in range(count):  # Aggregate each semantic region independently.
        inside = region_labels == region  # Select all elements in the region.
        region_count = int(np.sum(inside))  # Count the regional elements.
        if region_count == 0:  # Preserve zeros for empty visual regions.
            continue  # Skip undefined regional ratios.
        region_error = float(np.sum(values[inside]))  # Measure the regional estimator mass.
        error_fraction[region] = float(np.sum(values[inside & marked_mask]) / max(region_error, 1.0e-30))  # Measure the marked error fraction.
        element_fraction[region] = float(np.sum(inside & marked_mask) / region_count)  # Measure the marked element fraction.
    return np.clip(error_fraction, 0.0, 1.0), np.clip(element_fraction, 0.0, 1.0)  # Return physically bounded exact exposures.

def regional_dorfler_proxy(state: WorldState, theta: float) -> WorldState:  # Approximate a future element-wise Dörfler set on predicted regional masses.
    if not 0.0 < theta <= 1.0:  # Validate the bulk parameter.
        raise ValueError("theta must lie in (0, 1]")  # Reject an invalid planning contract.
    order = np.argsort(state.err_sum)[::-1]  # Rank predicted regional estimator masses.
    remaining = theta * state.total_error  # Set the future Dörfler bulk target.
    error_fraction = np.zeros(state.n_regions, dtype=float)  # Allocate future marked error fractions.
    element_fraction = np.zeros(state.n_regions, dtype=float)  # Allocate future marked element fractions.
    for region in order:  # Fill the bulk target greedily by predicted region.
        if remaining <= 0.0:  # Stop after the target estimator mass is represented.
            break  # Leave all lower-ranked regions unmarked.
        available = float(state.err_sum[region])  # Read the predicted regional estimator mass.
        take = min(available, remaining)  # Take only the mass required to finish the bulk set.
        fraction = take / max(available, 1.0e-30)  # Convert the selected mass to a regional fraction.
        error_fraction[region] = fraction  # Store the proxy marked error fraction.
        element_fraction[region] = math.sqrt(fraction)  # Use a conservative wider support proxy for element count.
        remaining -= take  # Reduce the unresolved bulk target.
    return state.with_dorfler(error_fraction, np.clip(element_fraction, 0.0, 1.0))  # Return the predicted state prepared for another action.

def _rank_regions(state: WorldState, config: PlannerConfig) -> np.ndarray:  # Rank regions by physical persistence and current marginal importance.
    semantic = np.asarray([semantic_persistence(name) for name in state.names], dtype=float)  # Compute bounded semantic persistence cues.
    neighbour_error = state.adjacency @ state.error_share / np.maximum(np.sum(state.adjacency, axis=1), 1.0)  # Aggregate adjacent estimator shares.
    persistence = np.log1p(state.hit_count) + state.dorfler_error_fraction + semantic  # Combine observed repeat hits, current marking, and mechanism cues.
    score = state.error_share + 0.22 * neighbour_error + config.persistence_weight * persistence  # Form the regional action-enumeration score.
    return np.argsort(score)[::-1]  # Return descending regional indices.

def enumerate_actions(state: WorldState, config: PlannerConfig) -> list[RegionAction]:  # Enumerate a bounded deterministic discrete action set.
    count = state.n_regions  # Read the regional action dimension.
    zero = tuple(0 for _ in range(count))  # Construct the exact pure-Dörfler action.
    actions: list[RegionAction] = [RegionAction(zero, source="dorfler")]  # Keep pure Dörfler as the first and permanent candidate.
    ranked = _rank_regions(state, config)[: min(config.candidate_regions, count)]  # Restrict branching to the strongest physical candidates.
    for region in ranked:  # Add one-region advance actions.
        depth_one = [0] * count  # Allocate a sparse regional action.
        depth_one[int(region)] = 1  # Extend standard Dörfler over the selected region.
        actions.append(RegionAction(tuple(depth_one)))  # Store the one-level action.
        if config.max_extra_depth >= 2 and (state.hit_count[region] >= 1.0 or state.dorfler_error_fraction[region] >= 0.25):  # Reserve deeper delegation for persistent evidence.
            depth_two = [0] * count  # Allocate a second sparse action.
            depth_two[int(region)] = 2  # Delegate one additional future refinement level.
            actions.append(RegionAction(tuple(depth_two)))  # Store the two-level action.
    if config.max_extra_regions >= 2:  # Add bounded two-region resource-allocation actions.
        pair_pool = list(ranked[: min(4, len(ranked))])  # Keep pair enumeration computationally small.
        for first, second in combinations(pair_pool, 2):  # Enumerate unique high-value pairs.
            paired = [0] * count  # Allocate a two-region sparse action.
            paired[int(first)] = 1  # Advance the first persistent region.
            paired[int(second)] = 1  # Advance the second persistent region.
            actions.append(RegionAction(tuple(paired)))  # Store the paired action.
    unique: dict[tuple[int, ...], RegionAction] = {}  # Allocate a deterministic deduplication map.
    for action in actions:  # Deduplicate equivalent sparse actions.
        values = action.array(count)  # Validate the action against the state.
        if int(np.count_nonzero(values)) <= config.max_extra_regions and int(np.max(values, initial=0)) <= config.max_extra_depth:  # Enforce the action-space contract.
            unique.setdefault(tuple(int(value) for value in values), action)  # Preserve the first deterministic label.
    return list(unique.values())  # Return pure Dörfler plus all legal proactive actions.

def _robust_cost(prediction: WorldPrediction, root_error: float, root_equations: float, budget: float, config: PlannerConfig) -> float:  # Score one predicted transition on a dimensionless scale.
    error_ratio = prediction.state.total_error / max(root_error, 1.0e-30)  # Normalize predicted estimator mass by the current real solve.
    resource_ratio = prediction.state.n_equations / max(root_equations, 1.0)  # Normalize predicted equations by the current real solve.
    budget_ratio = prediction.state.n_equations / max(budget, 1.0)  # Normalize predicted equations by the hard cap.
    over_budget = max(budget_ratio - 1.0, 0.0)  # Isolate only hard-cap violation.
    return math.log(max(error_ratio, 1.0e-12)) + config.resource_weight * math.log(max(resource_ratio, 1.0e-12)) + config.uncertainty_weight * (prediction.log_error_std + prediction.log_resource_std) + config.failure_weight * prediction.failure_probability + config.over_budget_weight * over_budget ** 2  # Combine accuracy, resource, uncertainty, and failure risk.

def _rollout_zero(model: ResidualWorldModel, state: WorldState, budget: float, config: PlannerConfig) -> tuple[float, tuple[RegionAction, ...], WorldPrediction]:  # Compute the pure-Dörfler comparison trajectory.
    current = state  # Start from the current real state with exact Dörfler exposure.
    score = 0.0  # Initialize the discounted rollout cost.
    actions: list[RegionAction] = []  # Store the baseline action sequence.
    first_prediction: WorldPrediction | None = None  # Reserve the first-step prediction for common gating.
    for depth in range(max(config.horizon, 1)):  # Roll pure Dörfler through the requested horizon.
        action = RegionAction(tuple(0 for _ in range(current.n_regions)), source="dorfler")  # Select no proactive regional depth.
        prediction = model.predict(current, action)  # Predict the next Dörfler state.
        if first_prediction is None:  # Capture the immediate baseline transition once.
            first_prediction = prediction  # Store the first pure-Dörfler prediction.
        score += config.discount ** depth * _robust_cost(prediction, state.total_error, state.n_equations, budget, config)  # Accumulate the discounted robust cost.
        actions.append(action)  # Append the baseline action.
        current = regional_dorfler_proxy(prediction.state, config.theta)  # Prepare the next predicted Dörfler exposure.
    if first_prediction is None:  # Guard the impossible zero-horizon case.
        raise RuntimeError("baseline rollout produced no prediction")  # Report the invalid planner configuration.
    return score, tuple(actions), first_prediction  # Return the baseline score, sequence, and immediate prediction.

class WorldModelPlanner:  # Select a proactive action only when it robustly beats the Dörfler rollout.
    def __init__(self, model: ResidualWorldModel, config: PlannerConfig | None = None) -> None:  # Initialize the planner around one transition model.
        self.model = model  # Retain the shared online world model.
        self.config = config or PlannerConfig()  # Use explicit caller settings or safe defaults.
    def plan(self, state: WorldState, n_eq_budget: int) -> PlanDecision:  # Plan several transitions and return only the current executable action.
        config = self.config  # Bind the immutable settings locally.
        budget = float(n_eq_budget)  # Normalize the shared equation cap.
        baseline_score, baseline_sequence, baseline_first = _rollout_zero(self.model, state, budget, config)  # Establish the permanent Dörfler lower-bound candidate.
        initial_prediction = baseline_first  # Initialize fallback diagnostics with the Dörfler prediction.
        initial_node = _BeamNode(state=state, score=0.0, actions=tuple(), first_prediction=initial_prediction, last_prediction=initial_prediction)  # Seed the beam at the current real state.
        beam = [initial_node]  # Start finite-horizon search with one root node.
        sequences_evaluated = 0  # Count every action-conditioned world-model call.
        for depth in range(max(config.horizon, 1)):  # Expand the beam over multiple adaptive decisions.
            expanded: list[_BeamNode] = []  # Allocate the next-depth candidate beam.
            for node in beam:  # Expand every retained partial trajectory.
                planning_state = node.state if depth == 0 else regional_dorfler_proxy(node.state, config.theta)  # Use exact current exposure and proxy future exposures.
                for action in enumerate_actions(planning_state, config):  # Evaluate pure and proactive regional actions.
                    prediction = self.model.predict(planning_state, action)  # Roll the action through the regional world model.
                    sequences_evaluated += 1  # Record the model evaluation.
                    upper_equations = max(prediction.member_equations)  # Use the most conservative ensemble resource member.
                    if np.any(action.array(planning_state.n_regions) > 0) and upper_equations > config.max_budget_ratio * budget:  # Prune unsafe proactive overshoots.
                        continue  # Preserve resources for feasible trajectories.
                    step_cost = _robust_cost(prediction, state.total_error, state.n_equations, budget, config)  # Score the predicted transition.
                    cumulative = node.score + config.discount ** depth * step_cost  # Accumulate the discounted trajectory cost.
                    actions = node.actions + (action,)  # Extend the auditable action sequence.
                    first_prediction = prediction if depth == 0 else node.first_prediction  # Preserve the real next-step prediction.
                    expanded.append(_BeamNode(state=prediction.state, score=cumulative, actions=actions, first_prediction=first_prediction, last_prediction=prediction))  # Store the expanded trajectory.
            if not expanded:  # Guard an over-pruned search.
                break  # Fall back to the permanent pure-Dörfler trajectory.
            expanded.sort(key=lambda node: node.score)  # Rank trajectories by robust cumulative cost.
            beam = expanded[: config.beam_width]  # Retain only the strongest bounded beam.
        proactive = [node for node in beam if node.actions and np.any(node.actions[0].array(state.n_regions) > 0)]  # Isolate trajectories that change the immediate action.
        if not proactive:  # Fall back when no feasible proactive trajectory survived.
            return self._fallback(state, baseline_score, baseline_sequence, baseline_first, sequences_evaluated, "no_feasible_proactive_sequence")  # Return exact Dörfler.
        best = min(proactive, key=lambda node: node.score)  # Select the strongest proactive multi-step trajectory.
        first = best.first_prediction  # Read the immediate transition used for execution gates.
        relative_gain = (baseline_score - best.score) / max(abs(baseline_score), 1.0)  # Measure robust improvement over pure Dörfler.
        if relative_gain < config.min_relative_gain:  # Require a meaningful advantage over the fallback.
            return self._fallback(state, baseline_score, baseline_sequence, baseline_first, sequences_evaluated, "insufficient_predicted_gain")  # Reject marginal model actions.
        if first.log_error_std > config.uncertainty_limit:  # Gate epistemically uncertain error forecasts.
            return self._fallback(state, baseline_score, baseline_sequence, baseline_first, sequences_evaluated, "error_uncertainty_gate")  # Return exact Dörfler.
        if first.failure_probability > config.failure_limit:  # Gate high model-risk actions.
            return self._fallback(state, baseline_score, baseline_sequence, baseline_first, sequences_evaluated, "world_model_risk_gate")  # Return exact Dörfler.
        if max(first.member_equations) > config.budget_safety * budget:  # Require every ensemble member to respect the proactive safety margin.
            return self._fallback(state, baseline_score, baseline_sequence, baseline_first, sequences_evaluated, "resource_uncertainty_gate")  # Return exact Dörfler.
        return PlanDecision(action=RegionAction(best.actions[0].extra_depth, source="world_model"), source="world_model", reason="robust_multi_step_gain", horizon=config.horizon, baseline_score=float(baseline_score), selected_score=float(best.score), predicted_error=float(first.state.total_error), predicted_equations=float(first.state.n_equations), log_error_std=float(first.log_error_std), log_resource_std=float(first.log_resource_std), failure_probability=float(first.failure_probability), sequences_evaluated=sequences_evaluated, sequence=tuple(action.extra_depth for action in best.actions))  # Accept the first action of the best robust rollout.
    def _fallback(self, state: WorldState, baseline_score: float, baseline_sequence: tuple[RegionAction, ...], baseline_first: WorldPrediction, sequences_evaluated: int, reason: str) -> PlanDecision:  # Build a common exact-Dörfler fallback decision.
        action = RegionAction(tuple(0 for _ in range(state.n_regions)), source="dorfler_fallback")  # Select the permanent safety action.
        return PlanDecision(action=action, source="dorfler_fallback", reason=reason, horizon=self.config.horizon, baseline_score=float(baseline_score), selected_score=float(baseline_score), predicted_error=float(baseline_first.state.total_error), predicted_equations=float(baseline_first.state.n_equations), log_error_std=float(baseline_first.log_error_std), log_resource_std=float(baseline_first.log_resource_std), failure_probability=float(baseline_first.failure_probability), sequences_evaluated=sequences_evaluated, sequence=tuple(action_value.extra_depth for action_value in baseline_sequence))  # Return a complete fallback audit record.
