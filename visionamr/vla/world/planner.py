"""Finite-horizon planner that keeps exact Dörfler refinement as the safety action."""  # Describe the planning policy implemented below.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from dataclasses import dataclass  # Import immutable planning contracts.
from itertools import combinations  # Import bounded regional action combinations.
import math  # Import stable logarithmic cost terms.
import numpy as np  # Import numerical ranking operations.
from .model import RegionAction, ResidualWorldModel, WorldPrediction, WorldState, semantic_persistence  # Import the world-model state and action contracts.

@dataclass(frozen=True)
class PlannerConfig:  # Configure risk-aware receding-horizon planning.
    horizon: int = 4  # Predict several adaptive steps without adding real finite-element solves.
    beam_width: int = 20  # Bound the internal search tree.
    candidate_regions: int = 5  # Restrict actions to the most relevant measured semantic regions.
    max_extra_regions: int = 2  # Limit each action to a small number of mechanisms.
    max_extra_depth: int = 2  # Limit speculative future-hit compression per region.
    warmup_transitions: int = 1  # Require one exact-Dörfler transition before world-model control.
    discount: float = 0.84  # Discount distant internal predictions.
    resource_weight: float = 0.17  # Penalize active-equation growth.
    uncertainty_weight: float = 0.75  # Penalize epistemic uncertainty.
    failure_weight: float = 1.10  # Penalize transition-risk estimates.
    uncertainty_limit: float = 0.34  # Reject poorly supported first actions.
    failure_limit: float = 0.42  # Reject high-risk first actions.
    budget_safety: float = 0.97  # Preserve a resource margin before exact mesh certification.
    min_robust_gain: float = 0.018  # Require a measurable robust advantage over repeated Dörfler actions.

@dataclass(frozen=True)
class PlanDecision:  # Store the selected safe first action and its audit trail.
    action: RegionAction  # Store the receding-horizon first action.
    accepted: bool  # Report whether the world-model addition passed all gates.
    reason: str  # Explain acceptance or exact-Dörfler fallback.
    baseline_cost: float  # Store the repeated-Dörfler horizon cost.
    selected_cost: float  # Store the selected horizon cost.
    robust_gain: float  # Store the cost improvement over the baseline trajectory.
    predicted_equations_upper: int  # Store the conservative first-step resource prediction.
    predicted_error_ratio_upper: float  # Store the conservative first-step error prediction.
    path: tuple[tuple[int, ...], ...]  # Store the selected internal action sequence.

@dataclass(frozen=True)
class _BeamNode:  # Store one internal search node.
    state: WorldState  # Store the predicted state reached at this depth.
    cost: float  # Store accumulated discounted robust cost.
    first_action: RegionAction  # Store the action that would be executed now.
    path: tuple[RegionAction, ...]  # Store the complete internal action sequence.
    first_prediction: WorldPrediction  # Store first-step risk quantities for final certification.

class MultiStepPlanner:  # Search discrete future-hit actions while retaining Dörfler as a fixed candidate.
    def __init__(self, config: PlannerConfig | None = None) -> None:  # Initialize an immutable planning policy.
        self.config = config or PlannerConfig()  # Store planner hyperparameters.
        if self.config.horizon < 1 or self.config.beam_width < 1:  # Reject an empty search horizon or beam.
            raise ValueError("planner horizon and beam width must be positive")  # Explain the invalid planning contract.
        if self.config.max_extra_regions < 1 or self.config.max_extra_depth < 1:  # Require at least one possible world-model action.
            raise ValueError("planner must permit at least one bounded extra-depth action")  # Explain the invalid action domain.
    def _eligible(self, state: WorldState) -> list[int]:  # Rank measured persistent mechanisms for bounded action enumeration.
        ranking: list[tuple[float, int]] = []  # Collect deterministic regional ranking scores.
        for index, name in enumerate(state.names):  # Inspect each semantic region.
            persistence = semantic_persistence(name)  # Read the mechanism persistence prior.
            marked = float(state.dorfler_element_fraction[index])  # Read the current exact-Dörfler support.
            if persistence <= 0.0 or marked <= 0.0:  # Exclude generic fields and currently inactive mechanisms.
                continue  # Preserve the clean world-model action boundary.
            error_density = float(state.error_share[index] / max(state.element_share[index], 1.0e-12))  # Measure error concentration per regional resource share.
            recurrence = 1.0 + 0.22 * float(state.hit_count[index])  # Reward repeatedly selected mechanisms.
            score = error_density * (0.55 + persistence) * recurrence * (0.35 + float(state.dorfler_error_fraction[index]))  # Combine measured concentration, persistence, and recurrence.
            ranking.append((score, index))  # Store the candidate score and stable index.
        ranking.sort(key=lambda item: (-item[0], item[1]))  # Rank deterministically from strongest to weakest mechanism.
        return [index for _, index in ranking[: self.config.candidate_regions]]  # Return only the bounded candidate set.
    def enumerate_actions(self, state: WorldState) -> list[RegionAction]:  # Enumerate legal discrete first or rollout actions.
        actions: list[RegionAction] = [RegionAction.dorfler(state)]  # Insert exact Dörfler as the first immutable candidate.
        eligible = self._eligible(state)  # Identify active semantic mechanisms.
        for index in eligible:  # Enumerate bounded single-region depth actions.
            for depth in range(1, self.config.max_extra_depth + 1):  # Consider one or more compressed future hits.
                vector = [0 for _ in state.names]  # Start from the exact-Dörfler baseline vector.
                vector[index] = depth  # Add depth only to the selected persistent mechanism.
                actions.append(RegionAction(tuple(vector), source="world_model"))  # Store the legal single-region action.
        if self.config.max_extra_regions >= 2:  # Enable sparse two-mechanism coordination when configured.
            for left, right in combinations(eligible, 2):  # Enumerate unordered candidate pairs.
                vector = [0 for _ in state.names]  # Start from the exact-Dörfler baseline vector.
                vector[left] = 1  # Add one future hit to the first mechanism.
                vector[right] = 1  # Add one future hit to the second mechanism.
                actions.append(RegionAction(tuple(vector), source="world_model"))  # Store the coordinated regional action.
        return actions  # Return exact Dörfler followed by bounded world-model alternatives.
    def _stage_cost(self, current: WorldState, prediction: WorldPrediction) -> float:  # Score one predicted transition with risk and resource penalties.
        error_term = math.log(max(prediction.error_ratio_mean, 1.0e-12))  # Reward predicted global error decay.
        resource_term = self.config.resource_weight * math.log(max(prediction.equation_ratio_mean, 1.0))  # Penalize active-equation growth without rewarding artificial coarsening.
        uncertainty_term = self.config.uncertainty_weight * prediction.uncertainty  # Penalize epistemic uncertainty directly.
        failure_term = self.config.failure_weight * prediction.failure_risk  # Penalize predicted transition failure.
        persistence_term = 0.012 * max(0, prediction.next_state.step - current.step - 1)  # Keep internal rollout depth numerically well behaved.
        return float(error_term + resource_term + uncertainty_term + failure_term + persistence_term)  # Return the lower-is-better robust stage cost.
    def _within_budget(self, state: WorldState, prediction: WorldPrediction, n_equation_cap: int) -> bool:  # Check the conservative predicted resource gate.
        upper = float(state.n_equations) * float(prediction.equation_ratio_upper)  # Convert the ratio bound to active equations.
        return upper <= self.config.budget_safety * float(n_equation_cap)  # Preserve a margin for exact Gmsh preflight certification.
    def _baseline_rollout(self, state: WorldState, model: ResidualWorldModel, n_equation_cap: int) -> tuple[float, tuple[RegionAction, ...], WorldPrediction]:  # Roll out repeated exact-Dörfler actions.
        current = state  # Initialize the internal state at the latest real solve.
        total = 0.0  # Initialize discounted robust cost.
        path: list[RegionAction] = []  # Record the immutable baseline path.
        first_prediction: WorldPrediction | None = None  # Reserve the first-step prediction.
        for depth in range(self.config.horizon):  # Advance through the finite prediction horizon.
            action = RegionAction.dorfler(current)  # Apply the exact-Dörfler baseline at this predicted state.
            prediction = model.predict(current, action)  # Predict the baseline transition.
            if first_prediction is None:  # Capture first-step quantities for the final decision contract.
                first_prediction = prediction  # Store the first exact-Dörfler prediction.
            if not self._within_budget(current, prediction, n_equation_cap):  # Stop the internal baseline when the cap would be exceeded.
                total += (self.config.discount**depth) * 5.0  # Add an explicit resource-cap penalty.
                break  # End the infeasible baseline rollout.
            total += (self.config.discount**depth) * self._stage_cost(current, prediction)  # Accumulate discounted robust cost.
            path.append(action)  # Record the baseline action.
            current = prediction.next_state  # Advance to the predicted successor.
        if first_prediction is None:  # Guard the statically impossible empty-horizon state.
            first_prediction = model.predict(state, RegionAction.dorfler(state))  # Construct a defensive first prediction.
        return float(total), tuple(path), first_prediction  # Return the complete baseline audit tuple.
    def plan(self, state: WorldState, model: ResidualWorldModel, n_equation_cap: int, *, force_dorfler: bool = False) -> PlanDecision:  # Select a safe receding-horizon action.
        baseline_cost, baseline_path, baseline_prediction = self._baseline_rollout(state, model, n_equation_cap)  # Establish the fixed Dörfler floor before searching alternatives.
        baseline_action = RegionAction.dorfler(state)  # Construct the executable exact-Dörfler action.
        baseline_upper_equations = int(math.ceil(state.n_equations * baseline_prediction.equation_ratio_upper))  # Record the conservative baseline resource estimate.
        if force_dorfler:  # Honor an external audit cooldown after an underperforming transition.
            return PlanDecision(baseline_action, False, "audit_cooldown", baseline_cost, baseline_cost, 0.0, baseline_upper_equations, baseline_prediction.error_ratio_upper, tuple(action.extra_depth for action in baseline_path))  # Return the exact-Dörfler fallback with a complete audit trail.
        if model.transition_count < self.config.warmup_transitions:  # Require real transition evidence before granting world-model control.
            return PlanDecision(baseline_action, False, "world_model_warmup", baseline_cost, baseline_cost, 0.0, baseline_upper_equations, baseline_prediction.error_ratio_upper, tuple(action.extra_depth for action in baseline_path))  # Execute exact Dörfler during model warmup.
        initial_actions = self.enumerate_actions(state)  # Enumerate exact Dörfler and bounded semantic alternatives.
        beam: list[_BeamNode] = []  # Initialize the first search layer.
        for action in initial_actions:  # Evaluate every legal first action.
            prediction = model.predict(state, action)  # Predict the action-conditioned transition.
            if not self._within_budget(state, prediction, n_equation_cap):  # Reject predicted cap violations before search expansion.
                continue  # Preserve the deterministic resource safety gate.
            cost = self._stage_cost(state, prediction)  # Score the first predicted transition.
            beam.append(_BeamNode(prediction.next_state, cost, action, (action,), prediction))  # Seed one beam node per feasible first action.
        if not beam:  # Fall back safely when every predicted action violates the cap.
            return PlanDecision(baseline_action, False, "no_feasible_world_action", baseline_cost, baseline_cost, 0.0, baseline_upper_equations, baseline_prediction.error_ratio_upper, tuple(action.extra_depth for action in baseline_path))  # Return exact Dörfler.
        beam.sort(key=lambda node: (node.cost, node.first_action.extra_depth))  # Rank the first layer deterministically.
        beam = beam[: self.config.beam_width]  # Enforce the configured beam width.
        for depth in range(1, self.config.horizon):  # Expand remaining internal prediction steps.
            expanded: list[_BeamNode] = []  # Collect feasible child nodes.
            for node in beam:  # Expand each retained internal trajectory.
                for action in self.enumerate_actions(node.state):  # Enumerate bounded actions in the predicted state.
                    prediction = model.predict(node.state, action)  # Predict the child transition.
                    if not self._within_budget(node.state, prediction, n_equation_cap):  # Reject predicted cap violations.
                        continue  # Preserve the resource safety gate at every horizon step.
                    cost = node.cost + (self.config.discount**depth) * self._stage_cost(node.state, prediction)  # Accumulate discounted robust cost.
                    expanded.append(_BeamNode(prediction.next_state, cost, node.first_action, node.path + (action,), node.first_prediction))  # Preserve the executable first action while extending the internal path.
            if not expanded:  # Stop when no additional feasible internal transitions remain.
                break  # Retain the best previously feasible beam layer.
            expanded.sort(key=lambda node: (node.cost, node.first_action.extra_depth, tuple(action.extra_depth for action in node.path)))  # Rank child trajectories deterministically.
            beam = expanded[: self.config.beam_width]  # Retain only the bounded best trajectories.
        selected = min(beam, key=lambda node: (node.cost, node.first_action.extra_depth))  # Select the lowest-cost feasible internal trajectory.
        gain = float(baseline_cost - selected.cost)  # Compute robust advantage over repeated exact Dörfler actions.
        upper_equations = int(math.ceil(state.n_equations * selected.first_prediction.equation_ratio_upper))  # Compute the conservative first-step resource estimate.
        path = tuple(action.extra_depth for action in selected.path)  # Serialize the selected internal action sequence.
        if selected.first_action.is_dorfler_only:  # Retain exact Dörfler when search finds no useful semantic addition.
            return PlanDecision(baseline_action, False, "dorfler_is_optimal", baseline_cost, selected.cost, gain, upper_equations, selected.first_prediction.error_ratio_upper, path)  # Return the fixed safety action.
        if selected.first_prediction.uncertainty > self.config.uncertainty_limit:  # Reject unsupported first actions.
            return PlanDecision(baseline_action, False, "uncertainty_gate", baseline_cost, selected.cost, gain, upper_equations, selected.first_prediction.error_ratio_upper, path)  # Return exact Dörfler.
        if selected.first_prediction.failure_risk > self.config.failure_limit:  # Reject high-risk first actions.
            return PlanDecision(baseline_action, False, "failure_gate", baseline_cost, selected.cost, gain, upper_equations, selected.first_prediction.error_ratio_upper, path)  # Return exact Dörfler.
        if gain < self.config.min_robust_gain:  # Require a meaningful robust advantage over the Dörfler floor.
            return PlanDecision(baseline_action, False, "gain_gate", baseline_cost, selected.cost, gain, upper_equations, selected.first_prediction.error_ratio_upper, path)  # Return exact Dörfler.
        return PlanDecision(selected.first_action, True, "world_action_accepted", baseline_cost, selected.cost, gain, upper_equations, selected.first_prediction.error_ratio_upper, path)  # Grant the bounded world-model action.

WorldModelPlanner = MultiStepPlanner  # Preserve a descriptive compatibility alias for downstream scripts.
