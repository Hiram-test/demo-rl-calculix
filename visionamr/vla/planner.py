from __future__ import annotations  # Enable compact annotations across planning records.
import hashlib  # Create deterministic action identifiers.
from dataclasses import dataclass  # Define planner configuration and result records.
import numpy as np  # Rank regional actions and perform compact beam search.
from .tool_gateway import DeterministicToolGateway, ToolPreview  # Resolve every candidate through the numerical tool contract.
from .world_model import OnlineRegionWorldModel, WorldPrediction  # Evaluate action-conditioned future states.
from .world_state import WorldAction, WorldState  # Operate only on discrete actions and audited states.
@dataclass(frozen=True)  # Keep release-relevant planning policy explicit.
class PlannerConfig:  # Configure short-horizon model-predictive control.
    horizon: int = 3  # Look several mesh decisions ahead without extra finite-element solves.
    beam_width: int = 6  # Retain a small set of promising imagined futures.
    candidate_limit: int = 18  # Bound per-node action branching.
    max_region_changes: int = 3  # Match the deterministic tool gateway.
    dorfler_theta: float = 0.50  # Build the exact AFEM safety action from the same bulk fraction.
    discount: float = 0.82  # Prioritize near-term improvements while retaining multi-step value.
    qoi_weight: float = 0.25  # Protect the engineering quantity of interest.
    uncertainty_weight: float = 0.20  # Penalize unsupported imagined gains.
    resource_over_weight: float = 25.0  # Make predicted budget violations unacceptable.
    resource_under_weight: float = 0.05  # Mildly discourage wasting the available budget.
    action_weight: float = 0.003  # Discourage unnecessary region churn.
    minimum_world_advantage: float = 0.01  # Require a measurable predicted benefit over the Dörfler guard.
    uncertainty_guard: float = 0.42  # Fall back when ensemble spread is too high.
@dataclass(frozen=True)  # Return the complete root decision audit.
class PlanResult:  # Explain why the planner selected or rejected a learned action.
    action: WorldAction  # Return the first action to execute.
    prediction: WorldPrediction  # Return its predicted consequence.
    guard_action: WorldAction  # Return the exact Dörfler alternative.
    guard_prediction: WorldPrediction  # Return the guard's predicted consequence.
    selected_by: str  # Record world-model, Dörfler, forced fallback, or stopping logic.
    predicted_advantage: float  # Record normalized benefit relative to the guard.
    rollout: tuple[dict, ...]  # Preserve the best imagined sequence.
    candidates_evaluated: int  # Record planning work separately from solves.
def _action_id(state: WorldState, kind: str, source: str, deltas: np.ndarray) -> str:  # Create a deterministic compact action id.
    digest = hashlib.sha256()  # Start an action digest.
    digest.update(state.state_id.encode("ascii"))  # Bind the action to one state.
    digest.update(kind.encode("ascii"))  # Bind the execution kind.
    digest.update(source.encode("utf-8"))  # Bind the proposal source.
    digest.update(np.ascontiguousarray(deltas, dtype=np.int8).tobytes())  # Bind exact discrete changes.
    return source + "-" + digest.hexdigest()[:10]  # Keep traces human-readable and collision-resistant.
def _make_action(state: WorldState, deltas: np.ndarray, source: str, rationale: str, kind: str = "region") -> WorldAction:  # Build one validated high-level action.
    clipped = np.clip(np.asarray(deltas, dtype=int), -1, 1)  # Restrict changes to adjacent grades.
    return WorldAction(action_id=_action_id(state, kind, source, clipped), deltas=tuple(int(value) for value in clipped), kind=kind, source=source, rationale=rationale)  # Return an immutable action.
def dorfler_guard_action(state: WorldState, theta: float = 0.50) -> WorldAction:  # Build the region-level preview for the exact element Dörfler tool action.
    order = np.argsort(-state.err_sum)  # Sort regions by measured estimator mass.
    threshold = float(theta * max(float(state.err_sum.sum()), 1.0e-30))  # Define the bulk target.
    cumulative = 0.0  # Accumulate marked regional mass.
    marked = []  # Preserve marked region indices.
    for index in order:  # Select the smallest regional bulk set.
        marked.append(int(index))  # Mark the next highest-error region.
        cumulative += float(state.err_sum[index])  # Update the captured estimator mass.
        if cumulative >= threshold:  # Stop once the bulk condition is met.
            break  # Preserve minimality.
    deltas = np.zeros(state.n_regions, dtype=int)  # Allocate the preview change vector.
    for index in marked:  # Approximate the exact element action for imagined rollouts.
        deltas[index] = -1  # Mark the region even when its grade is already one because exact Dörfler can refine below the grade prior.
    return _make_action(state, deltas, "dorfler_guard", "exact element-wise ZZ bulk marking available as the safety action", kind="dorfler")  # Return the exact-execution guard token.
def _candidate_actions(state: WorldState, config: PlannerConfig) -> list[WorldAction]:  # Generate a bounded action set without continuous parameter search.
    actions = [dorfler_guard_action(state, config.dorfler_theta)]  # Always include faithful Dörfler fallback first.
    error_rank = np.argsort(-state.err_share)  # Rank observed error concentration.
    efficiency = state.err_share / np.maximum(state.elem_share, 1.0e-12)  # Rank observed error removed per resource share.
    efficiency_rank = np.argsort(-efficiency)  # Sort marginal efficiency.
    semantic_score = 0.70 * state.roles[:, 0] + 0.25 * state.roles[:, 1] + 1.35 * state.roles[:, 2] + 0.55 * state.roles[:, 3] - 0.20 * state.roles[:, 4] + 1.00 * state.roles[:, 5]  # Rank geometry-visible topology and transfer paths.
    semantic_rank = np.argsort(-semantic_score)  # Sort prior-relevant regions.
    semantic_delta = np.zeros(state.n_regions, dtype=int)  # Allocate the topology-anticipation action.
    semantic_selected = []  # Preserve selected semantic regions.
    for index in semantic_rank:  # Refine the strongest visible mechanisms first.
        if semantic_score[index] <= 0.0 or state.grades[index] <= 1:  # Skip irrelevant or already finest regions.
            continue  # Continue searching for legal semantic moves.
        semantic_delta[index] = -1  # Move one level finer.
        semantic_selected.append(int(index))  # Record the intervention.
        if len(semantic_selected) >= config.max_region_changes:  # Bound one-step complexity.
            break  # Let repeated MPC steps handle larger changes.
    if semantic_selected:  # Add the semantic action only when it changes the mesh.
        actions.append(_make_action(state, semantic_delta, "semantic_lookahead", "anticipate load-path and opening or duct concentrations before residual-only marking"))  # Add geometry-first planning.
    for index in error_rank[: min(5, state.n_regions)]:  # Add focused observed-error actions.
        if state.grades[index] <= 1:  # Skip a region that cannot move finer through grades.
            continue  # Preserve candidate diversity.
        delta = np.zeros(state.n_regions, dtype=int)  # Allocate a single-region action.
        delta[index] = -1  # Refine the selected error region.
        actions.append(_make_action(state, delta, "measured_focus", f"refine observed error region {state.names[index]}"))  # Add the focused action.
    pair = []  # Build one high-efficiency multi-region action.
    for index in efficiency_rank:  # Traverse marginal efficiency.
        if state.grades[index] > 1:  # Require a legal refinement.
            pair.append(int(index))  # Select the region.
        if len(pair) >= min(2, config.max_region_changes):  # Bound the pair.
            break  # Finish the candidate.
    if pair:  # Add a marginal-efficiency action when possible.
        delta = np.zeros(state.n_regions, dtype=int)  # Allocate the pair action.
        delta[pair] = -1  # Refine selected efficient regions.
        actions.append(_make_action(state, delta, "marginal_gain", "refine regions with the highest measured estimator mass per element"))  # Add the efficiency candidate.
    low_rank = np.argsort(efficiency)  # Rank inexpensive regions for possible resource release.
    transfers = 0  # Bound resource-neutral transfer candidates.
    for fine_index in efficiency_rank[: min(4, state.n_regions)]:  # Consider a few high-value destinations.
        if state.grades[fine_index] <= 1:  # Skip destinations already at the finest grade.
            continue  # Preserve legal actions.
        coarse_index = next((int(index) for index in low_rank if index != fine_index and state.grades[index] < 5 and state.roles[index, 4] > 0.0), None)  # Prefer releasing low-value field resource.
        if coarse_index is None:  # Fall back to any low-efficiency legal source.
            coarse_index = next((int(index) for index in low_rank if index != fine_index and state.grades[index] < 5), None)  # Search all regions.
        if coarse_index is None:  # Skip when no resource can be released.
            continue  # Preserve a finite candidate set.
        delta = np.zeros(state.n_regions, dtype=int)  # Allocate a transfer action.
        delta[fine_index] = -1  # Refine the high-value destination.
        delta[coarse_index] = 1  # Coarsen the low-value source.
        actions.append(_make_action(state, delta, "resource_transfer", f"transfer mesh resource from {state.names[coarse_index]} to {state.names[fine_index]}"))  # Add a resource-neutral action.
        transfers += 1  # Count transfer candidates.
        if transfers >= 3:  # Bound branching.
            break  # Finish transfer generation.
    zero = np.zeros(state.n_regions, dtype=int)  # Define an explicit no-change candidate.
    actions.append(_make_action(state, zero, "hold", "retain the current allocation while the tool verifies resource mapping"))  # Allow the world model to reject unnecessary movement.
    unique = []  # Deduplicate candidates with identical execution semantics.
    seen = set()  # Track kind and delta signatures.
    for action in actions:  # Preserve proposal order as a deterministic tie-break.
        key = (action.kind, action.deltas)  # Ignore rationale differences for numerical execution.
        if key in seen:  # Skip duplicates.
            continue  # Continue to the next proposal.
        seen.add(key)  # Record the unique signature.
        unique.append(action)  # Keep the action.
        if len(unique) >= config.candidate_limit:  # Enforce the branching cap.
            break  # Stop candidate generation.
    return unique  # Return a deterministic bounded action set.
def _prediction_cost(state: WorldState, action: WorldAction, prediction: WorldPrediction, config: PlannerConfig) -> float:  # Score one imagined consequence.
    budget_ratio = float(prediction.n_equations) / max(float(state.budget), 1.0)  # Normalize resource use.
    over = max(budget_ratio - 1.0, 0.0)  # Measure hard-cap risk.
    under = max(0.65 - budget_ratio, 0.0)  # Measure severe resource underuse.
    return float(prediction.e_energy + config.qoi_weight * prediction.e_qoi + config.uncertainty_weight * prediction.uncertainty + config.resource_over_weight * over**2 + config.resource_under_weight * under**2 + config.action_weight * action.n_changed)  # Combine accuracy, risk, resource, and action terms.
class ModelPredictivePlanner:  # Select one real action from multi-step imagined rollouts.
    def __init__(self, config: PlannerConfig | None = None) -> None:  # Store reproducible planning policy.
        self.config = config or PlannerConfig()  # Use explicit defaults when no override is supplied.
    def plan(self, state: WorldState, gateway: DeterministicToolGateway, model: OnlineRegionWorldModel, force_guard: bool = False) -> PlanResult:  # Plan one action and retain Dörfler as the safety floor.
        guard_action = dorfler_guard_action(state, self.config.dorfler_theta)  # Build the faithful safety token.
        guard_preview = gateway.preview(state, guard_action)  # Resolve guard sizes and resource prediction through the same tool.
        guard_prediction = model.predict(state, guard_preview)  # Predict the guard consequence.
        guard_cost = _prediction_cost(state, guard_action, guard_prediction, self.config)  # Score the guard.
        if force_guard:  # Honor a runtime safety request after a failed learned transition.
            return PlanResult(action=guard_action, prediction=guard_prediction, guard_action=guard_action, guard_prediction=guard_prediction, selected_by="forced_dorfler_after_model_miss", predicted_advantage=0.0, rollout=({"depth": 0, "action": guard_action.action_id, "source": guard_action.source, "cost": guard_cost},), candidates_evaluated=1)  # Return immediately with the guard.
        beam = [(0.0, state, None, tuple())]  # Initialize cumulative cost, imagined state, first action, and trace.
        evaluated = 0  # Count model predictions.
        for depth in range(max(int(self.config.horizon), 1)):  # Expand a finite receding-horizon tree.
            expanded = []  # Collect next beam candidates.
            for cumulative, node_state, first_action, trace in beam:  # Expand each retained imagined future.
                for action in _candidate_actions(node_state, self.config):  # Enumerate discrete legal actions.
                    preview = gateway.preview(node_state, action)  # Resolve numerical consequences through the deterministic tool.
                    prediction = model.predict(node_state, preview)  # Predict the action-conditioned transition.
                    step_cost = _prediction_cost(node_state, action, prediction, self.config)  # Score this transition.
                    total = float(cumulative + (self.config.discount**depth) * step_cost)  # Accumulate discounted cost.
                    first = action if first_action is None else first_action  # Preserve the executable root action.
                    next_trace = trace + ({"depth": int(depth), "action": action.action_id, "source": action.source, "kind": action.kind, "cost": float(step_cost), "e_energy": float(prediction.e_energy), "e_qoi": float(prediction.e_qoi), "n_equations": float(prediction.n_equations), "uncertainty": float(prediction.uncertainty)},)  # Append an auditable imagined step.
                    expanded.append((total, prediction.imagined_state(node_state), first, next_trace))  # Add the imagined successor.
                    evaluated += 1  # Count one world-model query.
            expanded.sort(key=lambda item: (item[0], item[2].action_id if item[2] is not None else ""))  # Rank deterministically.
            beam = expanded[: max(int(self.config.beam_width), 1)]  # Retain only the strongest futures.
            if not beam:  # Guard against an impossible empty action set.
                break  # Fall back below.
        if not beam or beam[0][2] is None:  # Fall back if planning produced no root action.
            return PlanResult(action=guard_action, prediction=guard_prediction, guard_action=guard_action, guard_prediction=guard_prediction, selected_by="dorfler_empty_plan", predicted_advantage=0.0, rollout=tuple(), candidates_evaluated=evaluated)  # Return the faithful guard.
        best_action = beam[0][2]  # Read the root action of the best rollout.
        best_preview = gateway.preview(state, best_action)  # Re-evaluate the executable root on the real current state.
        best_prediction = model.predict(state, best_preview)  # Obtain its one-step consequence and uncertainty.
        best_cost = _prediction_cost(state, best_action, best_prediction, self.config)  # Compare root actions on the same horizon-one basis.
        advantage = float((guard_cost - best_cost) / max(abs(guard_cost), 1.0e-12))  # Normalize predicted improvement over Dörfler.
        if best_action.kind == "dorfler":  # Accept the guard when the rollout itself selects it.
            selected_by = "dorfler_selected_by_world_model"  # Record endogenous safety selection.
        elif best_prediction.uncertainty > self.config.uncertainty_guard:  # Reject unsupported learned gains.
            best_action = guard_action  # Replace the action with faithful Dörfler.
            best_prediction = guard_prediction  # Replace the consequence audit.
            advantage = 0.0  # Do not claim a learned advantage.
            selected_by = "dorfler_uncertainty_guard"  # Record the safety reason.
        elif advantage < self.config.minimum_world_advantage:  # Require a real predicted margin over the baseline action.
            best_action = guard_action  # Replace a weak learned proposal.
            best_prediction = guard_prediction  # Preserve a consistent prediction.
            advantage = 0.0  # Do not claim a marginal advantage.
            selected_by = "dorfler_no_predicted_advantage"  # Record the comparison result.
        else:  # Execute a supported learned action.
            selected_by = "world_model_advantage"  # Record the model-based choice.
        return PlanResult(action=best_action, prediction=best_prediction, guard_action=guard_action, guard_prediction=guard_prediction, selected_by=selected_by, predicted_advantage=float(advantage), rollout=tuple(beam[0][3]), candidates_evaluated=int(evaluated))  # Return the complete decision audit.
