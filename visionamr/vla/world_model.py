# Action-conditioned region-graph world model and finite-horizon planner for WM-VLA.  # Module purpose.
from __future__ import annotations  # Enable postponed type annotations for recursive state records.
from dataclasses import dataclass, field, replace  # Provide typed immutable-like configuration and state containers.
from itertools import combinations  # Enumerate a bounded set of sparse region actions deterministically.
from pathlib import Path  # Persist and reload transferable transition data when requested.
from typing import Any, Iterable  # Type JSON payloads and candidate-action collections.
import json  # Serialize the learned transition memory without a framework dependency.
import numpy as np  # Perform graph features, ridge fitting, uncertainty estimates, and rollouts.
from ..geometry import Problem  # Read structural features, dimensions, and the problem resource scale.
from ..marking import dorfler_mark  # Build an explicitly labelled Dörfler-derived region candidate.
from .regions import Partition, RegionFeatures  # Reuse the fixed visual partition and measured regional physics.
from .tool_contract import MaterializedAction, MeshAction, fast_materialize_action  # Keep numerical parameters in tools.


_SEMANTIC_KINDS = ("load", "support", "clamp", "hole", "corner")  # Fix a transferable semantic feature order.


@dataclass  # Store one measured or counterfactual regional state.
class WorldState:  # Define the sufficient decision state used by the compact world model.
    names: tuple[str, ...]  # Preserve the stable region order across every real and imagined transition.
    grades: np.ndarray  # Store current ordinal coarseness levels selected through the action interface.
    sizes: np.ndarray  # Store tool-certified numerical sizes currently executed by Gmsh.
    err_sum: np.ndarray  # Store measured or predicted regional ZZ eta-squared totals.
    elems: np.ndarray  # Store measured or predicted regional element counts.
    vm_max: np.ndarray  # Store regional maximum von Mises stress as a hotspot descriptor.
    vm_mean: np.ndarray  # Store regional mean von Mises stress as a contrast descriptor.
    h_meas: np.ndarray  # Store measured regional element size after Gmsh gradation.
    volume: np.ndarray  # Store regional physical area or volume for density normalization.
    semantics: np.ndarray  # Store continuous proximity to load, support, clamp, hole, and corner features.
    adjacency: np.ndarray  # Store the region graph used to model cross-region remeshing effects.
    total_error: float  # Store the global ZZ eta-squared total used for planning and stopping.
    n_equations: int  # Store the exact current free-equation count.
    qoi_error: float  # Store the current reference-relative QoI error when a reference is available.
    solve_index: int  # Store the number of expensive real solves already consumed by this method.
    budget: int  # Store the hard free-equation budget shared by every planned action.

    def validate(self) -> "WorldState":  # Verify all state arrays before fitting or rolling out the model.
        n_regions = len(self.names)  # Establish the authoritative state width.
        vectors = (self.grades, self.sizes, self.err_sum, self.elems, self.vm_max, self.vm_mean, self.h_meas, self.volume)  # Collect aligned vectors.
        if any(np.asarray(vector).shape != (n_regions,) for vector in vectors):  # Detect region-order corruption early.
            raise ValueError("all regional state vectors must match names")  # Stop before a misleading prediction.
        if np.asarray(self.semantics).shape != (n_regions, len(_SEMANTIC_KINDS)):  # Enforce the semantic contract.
            raise ValueError("semantics must have shape (regions, 5)")  # Report the expected fixed representation.
        if np.asarray(self.adjacency).shape != (n_regions, n_regions):  # Enforce a square graph aligned with regions.
            raise ValueError("adjacency must be square and aligned with regions")  # Stop graph-message misalignment.
        return self  # Return the verified state for fluent construction.

    @property  # Expose a numerically safe regional error share.
    def err_share(self) -> np.ndarray:  # Normalize regional error for transferable action features.
        return np.asarray(self.err_sum, dtype=float) / max(float(np.sum(self.err_sum)), 1.0e-30)  # Avoid zero division.

    @property  # Expose a numerically safe regional element share.
    def elem_share(self) -> np.ndarray:  # Normalize regional resource use for transferable action features.
        return np.asarray(self.elems, dtype=float) / max(float(np.sum(self.elems)), 1.0)  # Avoid zero division.

    @property  # Expose exact current budget utilization.
    def budget_use(self) -> float:  # Measure how much of the hard equation budget is currently occupied.
        return float(self.n_equations) / max(float(self.budget), 1.0)  # Return a stable dimensionless ratio.

    @property  # Expose a semantic importance prior without hard-coding one bridge family.
    def semantic_importance(self) -> np.ndarray:  # Weight load, opening, corner, and support proximity for planning.
        weights = np.array([1.00, 0.65, 0.55, 0.90, 0.85], dtype=float)  # Encode broad mechanics relevance only.
        return np.clip(np.asarray(self.semantics, dtype=float) @ weights / float(np.sum(weights)), 0.0, 1.0)  # Aggregate safely.

    def to_dict(self) -> dict[str, Any]:  # Serialize the complete compact state for audit artifacts.
        return {  # Build a JSON-compatible state object.
            "names": list(self.names),  # Preserve exact region ordering.
            "grades": [int(value) for value in self.grades],  # Emit ordinary integers.
            "sizes": [float(value) for value in self.sizes],  # Emit ordinary floating-point values.
            "err_sum": [float(value) for value in self.err_sum],  # Preserve regional error evidence.
            "elems": [float(value) for value in self.elems],  # Preserve regional resource evidence.
            "vm_max": [float(value) for value in self.vm_max],  # Preserve hotspot stress evidence.
            "vm_mean": [float(value) for value in self.vm_mean],  # Preserve stress contrast evidence.
            "h_meas": [float(value) for value in self.h_meas],  # Preserve realized Gmsh sizes.
            "volume": [float(value) for value in self.volume],  # Preserve physical region measures.
            "semantics": np.asarray(self.semantics, dtype=float).tolist(),  # Preserve semantic graph features.
            "adjacency": np.asarray(self.adjacency, dtype=float).tolist(),  # Preserve the region graph.
            "total_error": float(self.total_error),  # Preserve the global estimator total.
            "n_equations": int(self.n_equations),  # Preserve exact resource use.
            "qoi_error": float(self.qoi_error),  # Preserve reference-relative QoI evidence.
            "solve_index": int(self.solve_index),  # Preserve real solve accounting.
            "budget": int(self.budget),  # Preserve the hard experimental budget.
        }  # Finish state serialization.


@dataclass  # Configure the compact hybrid world model.
class WorldModelConfig:  # Hold only stable learning and uncertainty parameters.
    ridge: float = 6.0  # Pull sparse online fits toward mechanics-informed prior coefficients.
    ensemble_size: int = 7  # Use a small bootstrap ensemble for epistemic uncertainty.
    bootstrap_fraction: float = 0.80  # Resample transition rows for each ensemble member.
    prior_jitter: float = 0.08  # Diversify prior members before enough real transitions exist.
    uncertainty_floor: float = 0.025  # Prevent unjustified zero uncertainty at initialization.
    error_log_clip: float = 2.5  # Bound one-step predicted regional error changes.
    element_log_clip: float = 2.5  # Bound one-step predicted regional resource changes.
    qoi_log_clip: float = 1.5  # Bound one-step predicted QoI-error changes.
    seed: int = 29  # Make fitting and rollout uncertainty reproducible.


@dataclass  # Store one real transition in transferable feature form.
class TransitionSample:  # Keep training data independent of mesh topology and region count.
    x_error: np.ndarray  # Store one feature row per region for eta-squared changes.
    y_error: np.ndarray  # Store observed log eta-squared changes per region.
    x_elems: np.ndarray  # Store one feature row per region for element-count changes.
    y_elems: np.ndarray  # Store observed log element-count changes per region.
    x_qoi: np.ndarray  # Store one global feature row for QoI-error change.
    y_qoi: float  # Store the observed global log QoI-error change.
    weight: float = 1.0  # Give real transitions explicit weight in regularized fitting.

    def to_dict(self) -> dict[str, Any]:  # Serialize a transition for reuse across nearby bridge instances.
        return {  # Build a JSON-compatible transition payload.
            "x_error": np.asarray(self.x_error, dtype=float).tolist(),  # Preserve regional error features.
            "y_error": np.asarray(self.y_error, dtype=float).tolist(),  # Preserve regional error targets.
            "x_elems": np.asarray(self.x_elems, dtype=float).tolist(),  # Preserve regional resource features.
            "y_elems": np.asarray(self.y_elems, dtype=float).tolist(),  # Preserve regional resource targets.
            "x_qoi": np.asarray(self.x_qoi, dtype=float).tolist(),  # Preserve global QoI features.
            "y_qoi": float(self.y_qoi),  # Preserve the scalar QoI target.
            "weight": float(self.weight),  # Preserve sample importance.
        }  # Finish transition serialization.

    @classmethod  # Reconstruct a transition from a saved JSON artifact.
    def from_dict(cls, payload: dict[str, Any]) -> "TransitionSample":  # Normalize persisted arrays safely.
        return cls(  # Construct the typed transition.
            x_error=np.asarray(payload["x_error"], dtype=float),  # Restore regional error features.
            y_error=np.asarray(payload["y_error"], dtype=float),  # Restore regional error targets.
            x_elems=np.asarray(payload["x_elems"], dtype=float),  # Restore regional resource features.
            y_elems=np.asarray(payload["y_elems"], dtype=float),  # Restore regional resource targets.
            x_qoi=np.asarray(payload["x_qoi"], dtype=float),  # Restore global QoI features.
            y_qoi=float(payload["y_qoi"]),  # Restore the scalar QoI target.
            weight=float(payload.get("weight", 1.0)),  # Restore or default sample importance.
        )  # Finish reconstruction.


@dataclass  # Store an ensemble counterfactual prediction.
class WorldPrediction:  # Return both the imagined next state and calibrated planning diagnostics.
    state: WorldState  # Store the ensemble-mean predicted next state.
    uncertainty: float  # Store aggregate epistemic uncertainty used by risk-sensitive planning.
    error_uncertainty: np.ndarray  # Store regional standard deviation of log error changes.
    element_uncertainty: np.ndarray  # Store regional standard deviation of log resource changes.
    qoi_uncertainty: float  # Store standard deviation of log QoI-error change.
    diagnostics: dict[str, Any]  # Preserve member totals and features for auditability.


@dataclass  # Configure bounded candidate generation and finite-horizon search.
class WorldPlannerConfig:  # Define the multi-step decision problem explicitly.
    horizon: int = 3  # Roll out several future actions without extra real solves.
    beam_width: int = 24  # Bound retained counterfactual plans at each depth.
    candidate_regions: int = 7  # Focus sparse actions on the most decision-relevant regions.
    max_refine_regions: int = 3  # Limit simultaneous refinements to preserve identifiability.
    max_transfer_pairs: int = 8  # Limit refine/coarsen resource-transfer candidates.
    min_predicted_gain: float = 0.025  # Stop when no action has a credible estimator reduction.
    target_error_ratio: float = 0.30  # Define a convergence target relative to the first real WM state.
    min_budget_use: float = 0.65  # Avoid accepting a needlessly coarse final mesh.
    budget_safety: float = 0.94  # Reserve resource headroom during counterfactual materialization.
    max_neighbor_ratio: float = 1.8  # Match the deterministic tool gradation contract.
    w_error: float = 1.00  # Prioritize global energy-estimator reduction.
    w_qoi: float = 0.30  # Retain QoI accuracy as a secondary objective.
    w_budget_over: float = 20.0  # Treat predicted budget excess as a hard planning violation.
    w_budget_under: float = 0.12  # Penalize severe budget under-use without forcing waste.
    w_uncertainty: float = 0.45  # Prefer robust plans when ensemble predictions disagree.
    w_solve: float = 0.035  # Charge each imagined expensive solve in the finite-horizon objective.
    dorfler_theta: float = 0.50  # Build an explicit Dörfler-derived candidate when guard mode is enabled.


@dataclass  # Store one planned action sequence and its first executable decision.
class PlanResult:  # Preserve the complete counterfactual search trace.
    action: MeshAction  # Return the first discrete action to execute now.
    materialized: MaterializedAction  # Return the tool-owned numerical realization of that action.
    sequence: list[dict[str, Any]]  # Preserve every imagined action in the selected horizon.
    score: float  # Store the risk-sensitive terminal plan score.
    predicted_gain: float  # Store the expected one-step global estimator reduction.
    uncertainty: float  # Store first-step epistemic uncertainty.
    diagnostics: dict[str, Any]  # Preserve candidate counts, alternatives, and stop rationale.


@dataclass  # Hold one internal beam-search node.
class _BeamNode:  # Keep search bookkeeping private to the planner.
    state: WorldState  # Store the imagined state at this node.
    sequence: list[tuple[MeshAction, MaterializedAction, WorldPrediction]]  # Store imagined transitions.
    score: float  # Store the current terminal objective value.


def _semantic_matrix(partition: Partition, problem: Problem) -> np.ndarray:  # Convert CAD features into soft region semantics.
    points = np.array([seed.point() for seed in partition.seeds], dtype=float)  # Read stable visual-region anchors.
    diameter = max(float(problem.diameter), 1.0e-9)  # Normalize distances across bridge-component scales.
    columns: list[np.ndarray] = []  # Accumulate one proximity column per semantic kind.
    for kind in _SEMANTIC_KINDS:  # Preserve the fixed transferable semantic order.
        anchors = np.array([feature.xyz for feature in problem.features if feature.kind == kind], dtype=float)  # Gather matching CAD anchors.
        if anchors.size == 0:  # Handle absent feature classes explicitly.
            columns.append(np.zeros(len(points), dtype=float))  # Emit a neutral semantic column.
            continue  # Move to the next structural feature class.
        distance = np.min(np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=2), axis=1)  # Find nearest matching anchor.
        columns.append(np.exp(-distance / (0.18 * diameter)))  # Convert distance into a smooth dimensionless relevance.
    return np.column_stack(columns)  # Return the complete region-by-semantic matrix.


def build_world_state(partition: Partition, features: RegionFeatures, adjacency: np.ndarray, grades: np.ndarray, solve_record: Any, problem: Problem, budget: int) -> WorldState:  # Build a measured decision state after one real solve.
    qoi_error = 0.0 if getattr(solve_record, "e_qoi", None) is None else float(solve_record.e_qoi)  # Normalize unavailable QoI error.
    state = WorldState(  # Construct the complete compact state.
        names=tuple(seed.name for seed in partition.seeds),  # Preserve fixed region ordering.
        grades=np.asarray(grades, dtype=int).copy(),  # Preserve current ordinal action coordinates.
        sizes=np.asarray(partition.sizes(), dtype=float).copy(),  # Preserve tool-certified current sizes.
        err_sum=np.maximum(np.asarray(features.err_sum, dtype=float), 1.0e-30),  # Preserve positive regional estimator totals.
        elems=np.maximum(np.asarray(features.elems, dtype=float), 1.0),  # Preserve positive regional resource counts.
        vm_max=np.maximum(np.asarray(features.vm_max, dtype=float), 0.0),  # Preserve nonnegative stress maxima.
        vm_mean=np.maximum(np.asarray(features.vm_mean, dtype=float), 0.0),  # Preserve nonnegative stress means.
        h_meas=np.maximum(np.asarray(features.h_meas, dtype=float), 1.0e-12),  # Preserve realized positive sizes.
        volume=np.maximum(np.asarray(features.volume, dtype=float), 1.0e-30),  # Preserve positive physical measures.
        semantics=_semantic_matrix(partition, problem),  # Attach CAD-derived semantics without solver leakage.
        adjacency=np.asarray(adjacency, dtype=float).copy(),  # Attach the measured region graph.
        total_error=max(float(features.total_err), 1.0e-30),  # Preserve the global estimator total.
        n_equations=int(solve_record.n_equations),  # Preserve exact expensive-solve resource use.
        qoi_error=max(qoi_error, 1.0e-12),  # Keep logarithmic QoI transitions numerically defined.
        solve_index=int(solve_record.solve_index),  # Preserve honest solve accounting.
        budget=int(budget),  # Preserve the hard experimental budget.
    )  # Finish measured-state construction.
    return state.validate()  # Verify alignment before exposing the state to planning.


def _neighbor_average(values: np.ndarray, adjacency: np.ndarray) -> np.ndarray:  # Aggregate one-hop graph messages safely.
    graph = np.asarray(adjacency, dtype=float)  # Normalize the adjacency matrix.
    degree = np.sum(graph, axis=1)  # Compute region degrees.
    return (graph @ np.asarray(values, dtype=float)) / np.maximum(degree, 1.0)  # Return zero-safe neighbor means.


def _transition_features(state: WorldState, target_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # Build action-conditioned graph features.
    log_h = np.log(np.maximum(np.asarray(target_sizes, dtype=float), 1.0e-12) / np.maximum(state.sizes, 1.0e-12))  # Encode signed local size changes.
    neighbor_log_h = _neighbor_average(log_h, state.adjacency)  # Encode cross-region gradation and load-transfer effects.
    error_focus = np.sqrt(np.clip(state.err_share, 0.0, 1.0))  # Temper highly concentrated error shares.
    stress_ratio = np.log1p(state.vm_max / np.maximum(state.vm_mean, 1.0e-12))  # Encode regional stress contrast robustly.
    stress_ratio = stress_ratio / max(float(np.max(stress_ratio)), 1.0)  # Normalize stress contrast to a stable range.
    semantic = state.semantic_importance  # Read CAD-derived structural relevance.
    grade_norm = (np.asarray(state.grades, dtype=float) - 3.0) / 2.0  # Normalize ordinal coarseness around zero.
    x_error = np.column_stack((np.ones(len(log_h)), log_h, neighbor_log_h, log_h * error_focus, log_h * semantic, log_h * stress_ratio, neighbor_log_h * semantic, grade_norm * log_h))  # Build regional error features.
    x_elems = np.column_stack((np.ones(len(log_h)), log_h, neighbor_log_h, log_h * state.elem_share, log_h * semantic))  # Build regional resource features.
    load_semantic = np.asarray(state.semantics[:, 0], dtype=float)  # Isolate load proximity for QoI sensitivity.
    x_qoi = np.array((1.0, float(np.sum(state.err_share * log_h)), float(np.sum(load_semantic * log_h) / max(np.sum(load_semantic), 1.0e-12)), float(np.sum(semantic * log_h) / max(np.sum(semantic), 1.0e-12)), float(state.budget_use)), dtype=float)  # Build global QoI features.
    return x_error, x_elems, x_qoi  # Return aligned model inputs.


def _ridge_with_prior(x: np.ndarray, y: np.ndarray, prior: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> np.ndarray:  # Fit a stable small linear model around physics priors.
    design = np.asarray(x, dtype=float)  # Normalize the design matrix.
    target = np.asarray(y, dtype=float)  # Normalize the target vector.
    if design.ndim != 2 or target.ndim != 1 or len(design) != len(target):  # Verify regression dimensions.
        raise ValueError("ridge inputs must be a two-dimensional design and aligned target")  # Stop malformed online updates.
    sample_weights = np.ones(len(target), dtype=float) if weights is None else np.asarray(weights, dtype=float)  # Normalize sample weights.
    root_w = np.sqrt(np.maximum(sample_weights, 1.0e-12))  # Convert weights for least-squares multiplication.
    weighted_x = design * root_w[:, None]  # Apply row weights to features.
    weighted_y = target * root_w  # Apply row weights to targets.
    regularizer = float(alpha) * np.eye(design.shape[1], dtype=float)  # Build isotropic prior regularization.
    lhs = weighted_x.T @ weighted_x + regularizer  # Build the positive-definite normal matrix.
    rhs = weighted_x.T @ weighted_y + regularizer @ np.asarray(prior, dtype=float)  # Pull the solution toward the prior.
    try:  # Prefer a direct stable solve for the tiny dense system.
        return np.linalg.solve(lhs, rhs)  # Return the regularized coefficient vector.
    except np.linalg.LinAlgError:  # Handle rare numerical degeneracy explicitly.
        return np.linalg.pinv(lhs) @ rhs  # Fall back to a deterministic pseudoinverse.


class HybridGraphWorldModel:  # Learn action-conditioned graph transitions from sparse real solves.
    def __init__(self, problem_dim: int, config: WorldModelConfig | None = None) -> None:  # Initialize mechanics priors and empty real memory.
        self.problem_dim = int(problem_dim)  # Preserve the resource-scaling dimension.
        self.config = config or WorldModelConfig()  # Use explicit or default stable settings.
        self.samples: list[TransitionSample] = []  # Store only real observed transitions.
        self._fit_counter = 0  # Make ensemble bootstraps reproducible after each update.
        self._error_prior = np.array((0.0, 1.80, 0.24, 0.75, 0.55, 0.30, 0.18, 0.10), dtype=float)  # Encode eta reducing under refinement.
        self._element_prior = np.array((0.0, -float(self.problem_dim), -0.28 * float(self.problem_dim), -0.15, -0.10), dtype=float)  # Encode N~h^-d plus graph spillover.
        self._qoi_prior = np.array((0.0, 0.70, 1.10, 0.35, 0.0), dtype=float)  # Encode load-zone refinement as QoI-improving.
        self._error_members: list[np.ndarray] = []  # Hold fitted ensemble error coefficients.
        self._element_members: list[np.ndarray] = []  # Hold fitted ensemble resource coefficients.
        self._qoi_members: list[np.ndarray] = []  # Hold fitted ensemble QoI coefficients.
        self._refit()  # Initialize a diversified prior ensemble immediately.

    @property  # Expose how many real transitions currently inform the model.
    def n_transitions(self) -> int:  # Count expensive feedback updates rather than region rows.
        return len(self.samples)  # Return the real transition count.

    def _refit(self) -> None:  # Refit the complete bootstrap ensemble after each real transition.
        cfg = self.config  # Read stable fitting parameters once.
        rng = np.random.default_rng(int(cfg.seed + 1009 * self._fit_counter))  # Derive a deterministic fit-specific generator.
        self._fit_counter += 1  # Advance the fit sequence for the next real update.
        if self.samples:  # Concatenate all transferable regional and global rows.
            x_error_all = np.vstack([sample.x_error for sample in self.samples])  # Stack regional error features.
            y_error_all = np.concatenate([sample.y_error for sample in self.samples])  # Stack regional error targets.
            x_elems_all = np.vstack([sample.x_elems for sample in self.samples])  # Stack regional resource features.
            y_elems_all = np.concatenate([sample.y_elems for sample in self.samples])  # Stack regional resource targets.
            x_qoi_all = np.vstack([sample.x_qoi for sample in self.samples])  # Stack global QoI features.
            y_qoi_all = np.array([sample.y_qoi for sample in self.samples], dtype=float)  # Stack global QoI targets.
            w_error_all = np.concatenate([np.full(len(sample.y_error), sample.weight, dtype=float) for sample in self.samples])  # Expand transition weights to regions.
            w_elems_all = np.concatenate([np.full(len(sample.y_elems), sample.weight, dtype=float) for sample in self.samples])  # Expand resource weights to regions.
            w_qoi_all = np.array([sample.weight for sample in self.samples], dtype=float)  # Preserve global transition weights.
        else:  # Create empty matrices that preserve coefficient dimensions before data exist.
            x_error_all = np.empty((0, len(self._error_prior)), dtype=float)  # Preserve error feature width.
            y_error_all = np.empty(0, dtype=float)  # Preserve empty error targets.
            x_elems_all = np.empty((0, len(self._element_prior)), dtype=float)  # Preserve resource feature width.
            y_elems_all = np.empty(0, dtype=float)  # Preserve empty resource targets.
            x_qoi_all = np.empty((0, len(self._qoi_prior)), dtype=float)  # Preserve QoI feature width.
            y_qoi_all = np.empty(0, dtype=float)  # Preserve empty QoI targets.
            w_error_all = np.empty(0, dtype=float)  # Preserve empty error weights.
            w_elems_all = np.empty(0, dtype=float)  # Preserve empty resource weights.
            w_qoi_all = np.empty(0, dtype=float)  # Preserve empty QoI weights.
        self._error_members = []  # Replace the previous ensemble atomically.
        self._element_members = []  # Replace the previous resource ensemble atomically.
        self._qoi_members = []  # Replace the previous QoI ensemble atomically.
        for _ in range(int(cfg.ensemble_size)):  # Fit each independently bootstrapped member.
            error_prior = self._error_prior * (1.0 + rng.normal(0.0, cfg.prior_jitter, size=len(self._error_prior)))  # Jitter error priors.
            element_prior = self._element_prior * (1.0 + rng.normal(0.0, cfg.prior_jitter, size=len(self._element_prior)))  # Jitter resource priors.
            qoi_prior = self._qoi_prior * (1.0 + rng.normal(0.0, cfg.prior_jitter, size=len(self._qoi_prior)))  # Jitter QoI priors.
            if len(y_error_all):  # Bootstrap regional rows when real data exist.
                count = max(int(np.ceil(cfg.bootstrap_fraction * len(y_error_all))), 1)  # Select a nonempty bootstrap size.
                index = rng.integers(0, len(y_error_all), size=count)  # Sample error rows with replacement.
                error_coef = _ridge_with_prior(x_error_all[index], y_error_all[index], error_prior, cfg.ridge, w_error_all[index])  # Fit error dynamics.
            else:  # Use the diversified prior before observing a transition.
                error_coef = error_prior  # Preserve mechanics-informed initialization.
            if len(y_elems_all):  # Bootstrap regional resource rows when available.
                count = max(int(np.ceil(cfg.bootstrap_fraction * len(y_elems_all))), 1)  # Select a nonempty bootstrap size.
                index = rng.integers(0, len(y_elems_all), size=count)  # Sample resource rows with replacement.
                element_coef = _ridge_with_prior(x_elems_all[index], y_elems_all[index], element_prior, cfg.ridge, w_elems_all[index])  # Fit resource dynamics.
            else:  # Use the diversified resource prior initially.
                element_coef = element_prior  # Preserve N~h^-d initialization.
            if len(y_qoi_all):  # Bootstrap transition-level QoI rows when available.
                count = max(int(np.ceil(cfg.bootstrap_fraction * len(y_qoi_all))), 1)  # Select a nonempty bootstrap size.
                index = rng.integers(0, len(y_qoi_all), size=count)  # Sample QoI transitions with replacement.
                qoi_coef = _ridge_with_prior(x_qoi_all[index], y_qoi_all[index], qoi_prior, cfg.ridge, w_qoi_all[index])  # Fit QoI dynamics.
            else:  # Use the diversified QoI prior initially.
                qoi_coef = qoi_prior  # Preserve load-sensitive initialization.
            error_coef[1] = max(float(error_coef[1]), 0.15)  # Enforce the physical sign that refinement should not increase modeled error locally.
            element_coef[1] = min(float(element_coef[1]), -0.25)  # Enforce the physical sign that refinement consumes elements.
            self._error_members.append(np.asarray(error_coef, dtype=float))  # Store the fitted error member.
            self._element_members.append(np.asarray(element_coef, dtype=float))  # Store the fitted resource member.
            self._qoi_members.append(np.asarray(qoi_coef, dtype=float))  # Store the fitted QoI member.

    def update(self, previous: WorldState, current: WorldState, executed_sizes: np.ndarray, weight: float = 1.0) -> dict[str, Any]:  # Learn from one real CalculiX transition.
        if previous.names != current.names:  # Keep online identification on a fixed visual region graph.
            return {"accepted": False, "reason": "region_order_changed", "n_transitions": self.n_transitions}  # Disclose skipped incompatible data.
        x_error, x_elems, x_qoi = _transition_features(previous, executed_sizes)  # Rebuild exact action-conditioned inputs.
        y_error = np.log(np.maximum(current.err_sum, 1.0e-30) / np.maximum(previous.err_sum, 1.0e-30))  # Measure regional estimator changes.
        y_elems = np.log(np.maximum(current.elems, 1.0) / np.maximum(previous.elems, 1.0))  # Measure regional resource changes.
        y_qoi = float(np.log(max(current.qoi_error, 1.0e-12) / max(previous.qoi_error, 1.0e-12)))  # Measure QoI-error change.
        finite = bool(np.all(np.isfinite(y_error)) and np.all(np.isfinite(y_elems)) and np.isfinite(y_qoi))  # Reject numerically invalid observations.
        if not finite:  # Preserve model integrity on failed or degenerate solves.
            return {"accepted": False, "reason": "nonfinite_transition", "n_transitions": self.n_transitions}  # Disclose the rejection.
        sample = TransitionSample(x_error, y_error, x_elems, y_elems, x_qoi, y_qoi, float(weight))  # Package the real transition.
        self.samples.append(sample)  # Add the expensive evidence exactly once.
        self._refit()  # Update all ensemble members immediately.
        return {  # Return transparent online-learning diagnostics.
            "accepted": True,  # Confirm that the real transition entered memory.
            "n_transitions": self.n_transitions,  # Report the expensive evidence count.
            "mean_abs_error_log": float(np.mean(np.abs(y_error))),  # Report transition magnitude.
            "mean_abs_element_log": float(np.mean(np.abs(y_elems))),  # Report resource transition magnitude.
            "qoi_log": float(y_qoi),  # Report the observed QoI transition.
        }  # Finish update diagnostics.

    def predict(self, state: WorldState, materialized: MaterializedAction) -> WorldPrediction:  # Imagine one action-conditioned next state.
        if materialized.action.stop:  # Preserve a true terminal action without fabricated dynamics.
            diagnostics = {"members": len(self._error_members), "stop": True, "n_transitions": self.n_transitions}  # Explain the no-op prediction.
            return WorldPrediction(state, 0.0, np.zeros(len(state.names)), np.zeros(len(state.names)), 0.0, diagnostics)  # Return the unchanged state.
        x_error, x_elems, x_qoi = _transition_features(state, materialized.sizes)  # Build graph and semantic action features.
        error_logs = np.vstack([np.clip(x_error @ coefficient, -self.config.error_log_clip, self.config.error_log_clip) for coefficient in self._error_members])  # Predict regional error changes.
        element_logs = np.vstack([np.clip(x_elems @ coefficient, -self.config.element_log_clip, self.config.element_log_clip) for coefficient in self._element_members])  # Predict regional resource changes.
        qoi_logs = np.array([float(np.clip(x_qoi @ coefficient, -self.config.qoi_log_clip, self.config.qoi_log_clip)) for coefficient in self._qoi_members], dtype=float)  # Predict QoI changes.
        error_mean = np.mean(error_logs, axis=0)  # Aggregate error dynamics across ensemble members.
        element_mean = np.mean(element_logs, axis=0)  # Aggregate resource dynamics across ensemble members.
        qoi_mean = float(np.mean(qoi_logs))  # Aggregate QoI dynamics across ensemble members.
        error_std = np.maximum(np.std(error_logs, axis=0), self.config.uncertainty_floor)  # Preserve epistemic uncertainty floor.
        element_std = np.maximum(np.std(element_logs, axis=0), self.config.uncertainty_floor)  # Preserve resource uncertainty floor.
        qoi_std = max(float(np.std(qoi_logs)), self.config.uncertainty_floor)  # Preserve QoI uncertainty floor.
        err_next = np.maximum(state.err_sum * np.exp(error_mean), 1.0e-30)  # Apply predicted regional estimator changes.
        elems_raw = np.maximum(state.elems * np.exp(element_mean), 1.0)  # Apply predicted regional resource changes.
        element_scale = float(materialized.predicted_elements) / max(float(np.sum(elems_raw)), 1.0)  # Anchor graph resource distribution to tool cost.
        elems_next = np.maximum(elems_raw * element_scale, 1.0)  # Preserve tool-owned total resource prediction.
        qoi_next = max(float(state.qoi_error * np.exp(qoi_mean)), 1.0e-12)  # Apply predicted QoI-error change.
        h_ratio = np.asarray(materialized.sizes, dtype=float) / np.maximum(state.sizes, 1.0e-12)  # Measure requested size changes.
        h_meas_next = np.clip(state.h_meas * h_ratio, 1.0e-12, np.inf)  # Approximate realized local sizes for rollout features.
        next_state = replace(  # Construct the ensemble-mean counterfactual state.
            state,  # Preserve names, semantics, graph, volumes, and budget.
            grades=np.asarray(materialized.grades, dtype=int).copy(),  # Apply the discrete action coordinates.
            sizes=np.asarray(materialized.sizes, dtype=float).copy(),  # Apply tool-owned numerical sizes.
            err_sum=err_next,  # Apply predicted regional errors.
            elems=elems_next,  # Apply predicted regional resource allocation.
            h_meas=h_meas_next,  # Apply approximate realized sizes.
            total_error=float(np.sum(err_next)),  # Update the global estimator total.
            n_equations=int(materialized.predicted_equations),  # Use the deterministic tool resource prediction.
            qoi_error=qoi_next,  # Update the predicted QoI error.
            solve_index=int(state.solve_index + 1),  # Charge one imagined global solve.
        )  # Finish counterfactual-state construction.
        member_totals = np.sum(state.err_sum[None, :] * np.exp(error_logs), axis=1)  # Compute member-level global error totals.
        relative_total_std = float(np.std(member_totals) / max(float(np.mean(member_totals)), 1.0e-30))  # Normalize global disagreement.
        uncertainty = float(relative_total_std + np.mean(error_std * state.err_share) + 0.25 * np.mean(element_std * state.elem_share) + 0.20 * qoi_std)  # Aggregate decision risk.
        diagnostics = {  # Preserve counterfactual evidence for audit artifacts.
            "n_transitions": self.n_transitions,  # Report real training depth.
            "member_total_error": [float(value) for value in member_totals],  # Preserve ensemble global predictions.
            "error_log_mean": [float(value) for value in error_mean],  # Preserve regional error effects.
            "element_log_mean": [float(value) for value in element_mean],  # Preserve regional resource effects.
            "qoi_log_mean": float(qoi_mean),  # Preserve global QoI effect.
        }  # Finish prediction diagnostics.
        return WorldPrediction(next_state.validate(), uncertainty, error_std, element_std, qoi_std, diagnostics)  # Return the complete imagined transition.

    def save(self, path: str | Path) -> Path:  # Persist only transferable configuration and real transitions.
        target = Path(path)  # Normalize the output path.
        target.parent.mkdir(parents=True, exist_ok=True)  # Create the artifact directory deterministically.
        payload = {  # Build a stable JSON checkpoint.
            "version": 1,  # Mark the compact transition schema version.
            "problem_dim": int(self.problem_dim),  # Preserve the resource-scaling dimension.
            "config": dict(self.config.__dict__),  # Preserve model hyperparameters explicitly.
            "samples": [sample.to_dict() for sample in self.samples],  # Preserve only real transition evidence.
        }  # Finish checkpoint payload construction.
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")  # Write a human-auditable checkpoint.
        return target  # Return the exact saved path.

    @classmethod  # Reconstruct a model and refit its ensemble from saved real transitions.
    def load(cls, path: str | Path) -> "HybridGraphWorldModel":  # Read a portable JSON checkpoint.
        payload = json.loads(Path(path).read_text(encoding="utf-8"))  # Parse the checkpoint deterministically.
        model = cls(int(payload["problem_dim"]), WorldModelConfig(**payload.get("config", {})))  # Recreate priors and configuration.
        model.samples = [TransitionSample.from_dict(item) for item in payload.get("samples", [])]  # Restore real transition memory.
        model._refit()  # Refit the ensemble from restored evidence.
        return model  # Return the reconstructed world model.


def dorfler_region_action(state: WorldState, labels: np.ndarray, eta2: np.ndarray, theta: float = 0.50) -> MeshAction:  # Project exact element marking into the fixed region action space.
    marked = dorfler_mark(np.asarray(eta2, dtype=float), float(theta))  # Apply the repository's exact element-level bulk marker.
    deltas = np.zeros(len(state.names), dtype=int)  # Start from a keep action for every visual region.
    if len(marked):  # Refine every region receiving a meaningful share of marked error.
        marked_labels = np.asarray(labels, dtype=int)[marked]  # Read visual regions of marked elements.
        marked_error = np.bincount(marked_labels, weights=np.asarray(eta2, dtype=float)[marked], minlength=len(state.names))  # Aggregate marked error by region.
        total_marked = max(float(np.sum(marked_error)), 1.0e-30)  # Normalize marked contributions safely.
        active = marked_error / total_marked >= max(0.08, 0.5 / max(len(state.names), 1))  # Reject negligible projection spillover.
        deltas[active] = -1  # Request one discrete refinement level in active marked regions.
    return MeshAction("dorfler_region", tuple(int(value) for value in deltas), source="dorfler_region_guard", stop=False).validate(len(state.names))  # Return an explicitly labelled candidate.


def _candidate_actions(state: WorldState, config: WorldPlannerConfig, guard_action: MeshAction | None = None) -> list[MeshAction]:  # Generate a bounded diverse sparse action set.
    n_regions = len(state.names)  # Read the fixed action width.
    score = state.err_share * (0.55 + 0.45 * state.semantic_importance)  # Rank regions by measured error and structural relevance.
    top = list(np.argsort(score)[::-1][: min(config.candidate_regions, n_regions)])  # Select the most decision-relevant regions.
    low_score = state.err_share / np.maximum(state.elem_share, 1.0e-12)  # Rank low-yield regions for resource release.
    low = list(np.argsort(low_score)[: min(config.candidate_regions, n_regions)])  # Select low marginal-value regions.
    actions: list[MeshAction] = []  # Accumulate candidate actions before deduplication.
    actions.append(MeshAction("stop", tuple(0 for _ in range(n_regions)), source="world", stop=True))  # Always represent explicit termination.
    for count in range(1, min(config.max_refine_regions, len(top)) + 1):  # Enumerate sparse multi-region refinements.
        for chosen in combinations(top, count):  # Enumerate deterministic subsets of high-value regions.
            deltas = np.zeros(n_regions, dtype=int)  # Start from a keep action.
            deltas[list(chosen)] = -1  # Refine the selected regions by one ordinal level.
            identifier = "refine_" + "_".join(str(index) for index in chosen)  # Build a stable trace identifier.
            actions.append(MeshAction(identifier, tuple(int(value) for value in deltas), source="world", stop=False))  # Add the candidate.
    transfers = 0  # Bound pairwise refine/coarsen reallocations explicitly.
    for high in top:  # Consider refinement of each high-value region.
        for calm in low:  # Consider financing it by coarsening a low-yield region.
            if int(high) == int(calm):  # Skip contradictory self-transfer.
                continue  # Move to the next pair.
            deltas = np.zeros(n_regions, dtype=int)  # Start from a keep action.
            deltas[int(high)] = -1  # Refine the high-value region.
            deltas[int(calm)] = +1  # Coarsen the low-yield region.
            actions.append(MeshAction(f"transfer_{int(calm)}_to_{int(high)}", tuple(int(value) for value in deltas), source="world", stop=False))  # Add the transfer candidate.
            transfers += 1  # Count the bounded transfer set.
            if transfers >= int(config.max_transfer_pairs):  # Stop after the configured diversity budget.
                break  # Leave the inner loop.
        if transfers >= int(config.max_transfer_pairs):  # Honor the same bound across high-value regions.
            break  # Leave the outer loop.
    if state.budget_use > 0.96:  # Offer direct coarsening when the current mesh is close to the hard cap.
        for calm in low[:3]:  # Restrict emergency coarsening to the lowest-yield regions.
            deltas = np.zeros(n_regions, dtype=int)  # Start from a keep action.
            deltas[int(calm)] = +1  # Coarsen one low-yield region.
            actions.append(MeshAction(f"coarsen_{int(calm)}", tuple(int(value) for value in deltas), source="world", stop=False))  # Add the resource-relief candidate.
    if guard_action is not None:  # Include the Dörfler-derived projection only when explicitly requested.
        actions.append(guard_action.validate(n_regions))  # Preserve its disclosed provenance.
    unique: dict[tuple[tuple[int, ...], bool], MeshAction] = {}  # Deduplicate equivalent vectors while preserving first provenance.
    for action in actions:  # Inspect every generated candidate.
        key = (tuple(int(value) for value in action.deltas), bool(action.stop))  # Define semantic action identity.
        unique.setdefault(key, action)  # Keep the first stable identifier for each unique action.
    return list(unique.values())  # Return the bounded unique action set.


def _plan_score(root: WorldState, predicted: WorldPrediction, depth: int, config: WorldPlannerConfig) -> float:  # Score a terminal counterfactual state.
    state = predicted.state  # Read the imagined terminal state.
    error_ratio = float(state.total_error) / max(float(root.total_error), 1.0e-30)  # Normalize estimator reduction to the current real state.
    qoi_ratio = float(state.qoi_error) / max(float(root.qoi_error), 1.0e-12)  # Normalize QoI-error change to the current real state.
    budget_use = float(state.n_equations) / max(float(root.budget), 1.0)  # Normalize predicted resource use.
    over = max(budget_use - 1.0, 0.0)  # Measure hard-cap violation.
    under = max(float(config.min_budget_use) - budget_use, 0.0)  # Measure severe resource under-use.
    return float(config.w_error * error_ratio + config.w_qoi * min(qoi_ratio, 3.0) + config.w_budget_over * over**2 + config.w_budget_under * under**2 + config.w_uncertainty * predicted.uncertainty + config.w_solve * float(depth))  # Return the risk-sensitive objective.


def _stop_prediction(state: WorldState) -> WorldPrediction:  # Represent an unchanged terminal state inside beam search.
    return WorldPrediction(state, 0.0, np.zeros(len(state.names)), np.zeros(len(state.names)), 0.0, {"stop": True})  # Return a no-op prediction.


class WorldPlanner:  # Perform bounded multi-step counterfactual search under the deterministic tool contract.
    def __init__(self, problem: Problem, model: HybridGraphWorldModel, config: WorldPlannerConfig | None = None) -> None:  # Bind problem, model, and search policy.
        self.problem = problem  # Preserve geometry and dimensional resource scaling.
        self.model = model  # Preserve the learned transition predictor.
        self.config = config or WorldPlannerConfig()  # Use explicit or default search settings.

    def plan(self, state: WorldState, eq_per_elem: float, guard_action: MeshAction | None = None, error_reference: float | None = None) -> PlanResult:  # Select the first action of the best finite-horizon plan.
        cfg = self.config  # Read stable search parameters once.
        stop_action = MeshAction("stop", tuple(0 for _ in state.names), source="world", stop=True).validate(len(state.names))  # Construct a terminal fallback.
        stop_materialized = fast_materialize_action(self.problem, state.sizes, state.grades, state.elems, state.adjacency, stop_action, state.budget, eq_per_elem, cfg.budget_safety, cfg.max_neighbor_ratio)  # Materialize the terminal action through the same tool.
        root_stop_prediction = _stop_prediction(state)  # Build the unchanged terminal prediction.
        initial_node = _BeamNode(state, [], _plan_score(state, root_stop_prediction, 0, cfg))  # Initialize the search at the real measured state.
        beam = [initial_node]  # Start with one root node.
        completed: list[_BeamNode] = []  # Accumulate terminal or horizon-complete plans.
        evaluated = 0  # Count counterfactual model evaluations explicitly.
        alternatives: list[dict[str, Any]] = []  # Preserve first-depth alternatives for interpretation.
        for depth in range(1, max(int(cfg.horizon), 1) + 1):  # Roll out the configured finite horizon.
            expanded: list[_BeamNode] = []  # Accumulate this depth's candidate nodes.
            for node in beam:  # Expand every retained partial plan.
                actions = _candidate_actions(node.state, cfg, guard_action if depth == 1 else None)  # Generate state-dependent sparse actions.
                for action in actions:  # Evaluate every legal candidate.
                    materialized = fast_materialize_action(self.problem, node.state.sizes, node.state.grades, node.state.elems, node.state.adjacency, action, node.state.budget, eq_per_elem, cfg.budget_safety, cfg.max_neighbor_ratio)  # Convert grades to safe parameters.
                    if action.stop:  # Terminate this partial plan explicitly.
                        prediction = _stop_prediction(node.state)  # Preserve the current imagined state.
                    elif not materialized.valid:  # Reject clipped no-op actions from expensive execution.
                        continue  # Move to the next candidate.
                    else:  # Predict one action-conditioned transition.
                        prediction = self.model.predict(node.state, materialized)  # Roll the world model forward once.
                        evaluated += 1  # Count the counterfactual transition.
                    sequence = list(node.sequence) + [(action, materialized, prediction)]  # Extend the partial plan.
                    score = _plan_score(state, prediction, depth, cfg)  # Score the new terminal state against the real root.
                    candidate = _BeamNode(prediction.state, sequence, score)  # Build the expanded node.
                    if depth == 1:  # Preserve interpretable first-step alternatives.
                        alternatives.append({"action": action.to_dict(), "score": float(score), "total_error": float(prediction.state.total_error), "n_equations": int(prediction.state.n_equations), "uncertainty": float(prediction.uncertainty)})  # Record compact diagnostics.
                    if action.stop:  # Do not expand terminal actions further.
                        completed.append(candidate)  # Preserve the terminal plan.
                    else:  # Retain executable partial plans for possible deeper lookahead.
                        expanded.append(candidate)  # Add the node to this depth's pool.
            if not expanded:  # Stop when no executable candidates remain.
                break  # Leave finite-horizon expansion.
            expanded.sort(key=lambda item: item.score)  # Rank partial plans by risk-sensitive objective.
            beam = expanded[: max(int(cfg.beam_width), 1)]  # Retain only the bounded best beam.
            if depth == max(int(cfg.horizon), 1):  # Treat the final retained beam as completed plans.
                completed.extend(beam)  # Add all horizon-complete alternatives.
        candidates = completed or beam  # Fall back to retained partial plans if no explicit terminal was produced.
        candidates.sort(key=lambda item: item.score)  # Rank all completed plans deterministically.
        best = candidates[0] if candidates else initial_node  # Select the best available plan safely.
        if not best.sequence:  # Return an explicit stop when no executable transition exists.
            return PlanResult(stop_action, stop_materialized, [], float(initial_node.score), 0.0, 0.0, {"reason": "no_valid_action", "evaluated": evaluated, "alternatives": alternatives})  # Preserve the failure rationale.
        first_action, first_materialized, first_prediction = best.sequence[0]  # Extract the receding-horizon action.
        predicted_gain = float((state.total_error - first_prediction.state.total_error) / max(state.total_error, 1.0e-30))  # Measure one-step predicted estimator gain.
        reference_error = float(state.total_error) if error_reference is None else max(float(error_reference), 1.0e-30)  # Read the first-real-state convergence reference when supplied.
        target_reached = float(state.total_error) <= float(cfg.target_error_ratio) * reference_error  # Evaluate the declared convergence target without resetting it each round.
        credible_gain = predicted_gain - float(cfg.w_uncertainty) * float(first_prediction.uncertainty)  # Discount gain by epistemic risk.
        if not first_action.stop and credible_gain < float(cfg.min_predicted_gain):  # Stop when no robust gain remains.
            first_action = stop_action  # Replace the risky weak action with explicit termination.
            first_materialized = stop_materialized  # Preserve tool-consistent terminal materialization.
            first_prediction = root_stop_prediction  # Preserve the unchanged real state.
            predicted_gain = 0.0  # Report no fabricated improvement.
            stop_reason = "insufficient_credible_gain"  # Explain risk-sensitive termination.
        else:  # Preserve the selected executable action.
            stop_reason = "planner_stop" if first_action.stop else "execute"  # Explain the selected branch.
        sequence_payload = [  # Serialize the selected imagined horizon.
            {"action": action.to_dict(), "materialized": materialized.to_dict(), "predicted_total_error": float(prediction.state.total_error), "predicted_n_equations": int(prediction.state.n_equations), "uncertainty": float(prediction.uncertainty)}  # Preserve each imagined transition.
            for action, materialized, prediction in best.sequence  # Traverse the selected plan in order.
        ]  # Finish sequence serialization.
        diagnostics = {  # Preserve planner workload and alternatives.
            "reason": stop_reason,  # Explain execution or termination.
            "evaluated": int(evaluated),  # Count cheap model counterfactuals.
            "beam_width": int(cfg.beam_width),  # Preserve search bounds.
            "horizon": int(cfg.horizon),  # Preserve lookahead depth.
            "model_transitions": int(self.model.n_transitions),  # Preserve real learning depth.
            "target_flag": bool(target_reached),  # Preserve the explicit convergence diagnostic.
            "alternatives": sorted(alternatives, key=lambda item: item["score"])[:12],  # Preserve the best first-step alternatives.
        }  # Finish planner diagnostics.
        return PlanResult(first_action, first_materialized, sequence_payload, float(best.score), float(predicted_gain), float(first_prediction.uncertainty), diagnostics)  # Return the receding-horizon decision.
