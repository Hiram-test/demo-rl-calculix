"""Multi-step world-model VLA with an exact Dörfler action shield."""  # State the scientific role of this implementation.

from __future__ import annotations  # Enable postponed type-annotation evaluation.

import hashlib  # Import hashing for immutable action and mesh receipts.
import json  # Import JSON support for persistent transition libraries.
import math  # Import scalar logarithms used by the planner objective.
from dataclasses import asdict  # Import dataclass serialization for audit records.
from dataclasses import dataclass  # Import typed immutable state containers.
from itertools import combinations  # Import bounded region-combination enumeration.
from pathlib import Path  # Import portable filesystem paths for model snapshots.

import numpy as np  # Import vectorized numerical operations.

from ..baselines.dorfler import refine_size_map  # Reuse the exact repository Dörfler refinement atom.
from ..experiment import FemRunner  # Import the single audited CalculiX execution gateway.
from ..experiment import initial_mesh  # Import the common uniform probe mesh constructor.
from ..fem_post import PostState  # Import the solved finite-element state contract.
from ..indicators import zz_indicator  # Import the common element-wise ZZ estimator.
from ..marking import dorfler_mark  # Import exact element-level Dörfler bulk marking.
from ..mesher import Mesh  # Import the repository simplex-mesh representation.
from ..mesher import generate_mesh  # Import the single Gmsh remeshing gateway.
from ..sizefield import NodalSizeField  # Import the deterministic nodal target-size interpolator.
from .regions import Partition  # Import the fixed semantic-region partition.
from .regions import RegionFeatures  # Import aggregated regional solve evidence.


@dataclass(frozen=True)  # Make the controller configuration immutable during one experiment.
class WorldControllerConfig:  # Collect every numerical and safety parameter in one auditable object.
    theta: float = 0.50  # Use the same Dörfler bulk parameter as the baseline.
    refine_factor: float = 0.50  # Halve targets on each exact Dörfler refinement hit.
    max_solves: int = 8  # Permit a genuine multi-step physical feedback loop.
    planning_horizon: int = 4  # Roll the learned world model several future steps ahead.
    beam_width: int = 18  # Bound the number of retained hypothetical trajectories.
    max_extra_depth: int = 3  # Bound delegated future-hit depth per semantic region.
    max_active_regions: int = 2  # Bound simultaneous advance investment to avoid diffuse over-refinement.
    candidate_regions: int = 5  # Restrict action enumeration to the strongest currently marked regions.
    min_predicted_gain: float = 0.025  # Require a risk-adjusted terminal-error gain over pure Dörfler.
    confidence_z: float = 1.96  # Convert ensemble spread to a conservative upper prediction.
    max_log_error_sigma: float = 0.45  # Reject actions whose predicted error transition is too uncertain.
    budget_safety: float = 0.98  # Reserve headroom below the hard equation cap.
    gradation: float = 0.90  # Use the same Lipschitz size-gradation contract as existing methods.
    support_hops_per_depth: int = 1  # Grow each delegated action only around its current marked support.
    error_power: float = 2.0  # Supply a weak finite-element convergence prior before residual learning.
    ridge: float = 1.0e-3  # Regularize every bootstrap residual model.
    ensemble_size: int = 5  # Fit multiple residual models for epistemic uncertainty.
    min_rows_for_learning: int = 20  # Keep the physics prior dominant until enough real transitions exist.
    random_seed: int = 20260830  # Freeze bootstrap resampling for reproducibility.
    resource_penalty: float = 0.10  # Penalize predicted use of the remaining equation budget.
    uncertainty_penalty: float = 0.35  # Penalize wide model uncertainty during finite-horizon search.
    action_penalty: float = 0.015  # Penalize unnecessary delegated refinement depth.
    require_reference: bool = True  # Compute reference-based metrics in scientific campaigns.
    common_uniform_probe: bool = True  # Use the same first mesh as Dörfler for mechanism attribution.

    def validate(self) -> None:  # Validate all configuration invariants before any solve is launched.
        if not 0.0 < self.theta <= 1.0:  # Check the mathematical Dörfler domain.
            raise ValueError("theta must lie in (0, 1]")  # Reject an invalid bulk fraction.
        if not 0.0 < self.refine_factor < 1.0:  # Check that refinement actually reduces target size.
            raise ValueError("refine_factor must lie in (0, 1)")  # Reject a non-refining action atom.
        if self.max_solves < 1 or self.planning_horizon < 1:  # Check the physical and simulated horizon lengths.
            raise ValueError("solve and planning horizons must be positive")  # Reject empty loops.
        if self.max_extra_depth < 0 or self.max_active_regions < 1:  # Check discrete action bounds.
            raise ValueError("invalid delegated-depth limits")  # Reject an impossible action space.
        if not 0.0 < self.budget_safety <= 1.0:  # Check the hard-budget safety factor.
            raise ValueError("budget_safety must lie in (0, 1]")  # Reject an invalid resource shield.


@dataclass(frozen=True)  # Keep each observed regional state immutable after it enters the transition library.
class RegionalState:  # Store only decision-relevant finite-element and semantic information.
    names: tuple[str, ...]  # Preserve stable semantic region identities across remeshing.
    err_sum: np.ndarray  # Store regional sums of the element-wise squared ZZ indicator.
    elems: np.ndarray  # Store realized regional element counts.
    sizes: np.ndarray  # Store realized mean regional element sizes.
    vm_max: np.ndarray  # Store regional maximum von Mises stress as a state cue.
    volume: np.ndarray  # Store regional material volume represented by the current mesh.
    adjacency: np.ndarray  # Store the semantic-region adjacency graph.
    hit_count: np.ndarray  # Store how often each region has been hit by exact Dörfler marking.
    marked_error_fraction: np.ndarray  # Store the fraction of each region's indicator captured this round.
    marked_element_fraction: np.ndarray  # Store the fraction of each region's elements marked this round.
    n_equations: int  # Store the actual CalculiX equation count.
    step: int  # Store the real adaptive-step index.
    h0: float  # Store the family coarse-size ceiling.
    h_min: float  # Store the family minimum admissible target size.
    dim: int  # Store the spatial dimension used by the resource prior.

    @property  # Expose the number of semantic regions without duplicating stored state.
    def n_regions(self) -> int:  # Return the regional graph order.
        return len(self.names)  # Count the stable semantic names.

    @property  # Expose the current total estimator mass.
    def total_error(self) -> float:  # Return the sum of all regional indicator masses.
        return float(np.sum(self.err_sum))  # Sum with a Python scalar result.

    @property  # Expose the current total element count.
    def total_elems(self) -> float:  # Return the sum of all regional element counts.
        return float(np.sum(self.elems))  # Sum with a Python scalar result.


@dataclass(frozen=True)  # Make actions immutable and hashable in audit trails.
class RegionAction:  # Represent only discrete future-hit delegation, never continuous mesh sizes.
    extra_depth: tuple[int, ...]  # Store additional future refinement hits beyond mandatory Dörfler.

    def array(self, n_regions: int) -> np.ndarray:  # Validate and expose the action as an integer vector.
        values = np.asarray(self.extra_depth, dtype=int)  # Convert the immutable tuple to a NumPy vector.
        if values.shape != (n_regions,):  # Require one decision per stable semantic region.
            raise ValueError("action width does not match the regional state")  # Reject misaligned action vectors.
        if np.any(values < 0):  # Forbid coarsening because the action must dominate Dörfler.
            raise ValueError("world-model actions may not contain negative depth")  # Reject a safety-violating action.
        return values  # Return the validated non-negative depth vector.

    @property  # Expose whether the action is exactly the classical fallback.
    def is_dorfler(self) -> bool:  # Test for a zero delegated-depth vector.
        return not any(int(value) > 0 for value in self.extra_depth)  # Return true only when no future hit is delegated.


@dataclass(frozen=True)  # Preserve one model prediction for later audit.
class TransitionPrediction:  # Store mean and upper-bound consequences of one discrete action.
    next_error_mean: np.ndarray  # Store predicted regional estimator masses.
    next_error_upper: np.ndarray  # Store conservative regional estimator masses.
    next_elems_mean: np.ndarray  # Store predicted regional element counts.
    next_elems_upper: np.ndarray  # Store conservative regional element counts.
    next_sizes: np.ndarray  # Store predicted regional mean sizes for multi-step rollout only.
    error_log_sigma: np.ndarray  # Store regional epistemic uncertainty in log-error change.
    resource_log_sigma: np.ndarray  # Store regional epistemic uncertainty in log-resource change.
    n_equations_mean: float  # Store predicted total equation count.
    n_equations_upper: float  # Store conservative total equation count.

    @property  # Expose the predicted total estimator mass.
    def total_error_mean(self) -> float:  # Return the mean terminal estimator prediction.
        return float(np.sum(self.next_error_mean))  # Sum the regional means.

    @property  # Expose the conservative predicted total estimator mass.
    def total_error_upper(self) -> float:  # Return the risk-adjusted terminal estimator prediction.
        return float(np.sum(self.next_error_upper))  # Sum the regional upper bounds.


@dataclass(frozen=True)  # Preserve the planner's accepted or rejected decision rationale.
class PlanDecision:  # Store the first action and its finite-horizon evidence.
    action: RegionAction  # Store the executable discrete action.
    source: str  # Record whether world-model planning or Dörfler fallback produced it.
    predicted_gain: float  # Record the risk-adjusted terminal gain over pure Dörfler.
    baseline_terminal_error: float  # Record the pure-Dörfler terminal upper prediction.
    selected_terminal_error: float  # Record the selected trajectory terminal upper prediction.
    selected_equations_upper: float  # Record the selected first transition's resource upper bound.
    selected_sigma: float  # Record the maximum first-transition log-error uncertainty.
    trajectory: tuple[tuple[int, ...], ...]  # Record the planned discrete action sequence.
    reason: str  # Record the deterministic admission or fallback reason.


@dataclass(frozen=True)  # Preserve deterministic compilation and certification evidence.
class ActionReceipt:  # Record how a discrete world-model action became an exact Gmsh target field.
    action_hash: str  # Hash the stable region names and discrete depth vector.
    base_target_hash: str  # Hash the mandatory exact-Dörfler nodal target map.
    candidate_target_hash: str  # Hash the Dörfler-dominating candidate target map.
    marked_elements: int  # Record the exact element-level Dörfler support size.
    delegated_regions: tuple[str, ...]  # Record semantic regions receiving extra future depth.
    support_elements: int  # Record the union of local support and graph-halo elements.
    dominance_pass: bool  # Record whether every mandatory Dörfler node is at least as fine.
    bounds_pass: bool  # Record whether every target lies in the admissible mesh-size interval.
    finite_pass: bool  # Record whether the compiled field is numerically finite.
    estimated_equations: int  # Record the exact free-DOF count computed on the generated Gmsh mesh.
    budget_pass: bool  # Record whether the generated mesh satisfies the equation-cap safety factor.
    fallback_used: bool  # Record whether deterministic certification replaced the world action by Dörfler.


@dataclass(frozen=True)  # Preserve the final run summary independently of mutable runner state.
class WorldVLAResult:  # Summarize a complete multi-step world-model VLA execution.
    solves: int  # Record the number of real CalculiX solves used by the method.
    stopped_by: str  # Record the physical or configured stopping condition.
    world_actions: int  # Record how many non-zero delegated actions were actually executed.
    dorfler_fallbacks: int  # Record how many planner or certification fallbacks occurred.
    transition_rows: int  # Record the resulting online world-model evidence volume.
    receipts: tuple[ActionReceipt, ...]  # Preserve every deterministic action-compilation receipt.


def semantic_persistence(name: str) -> float:  # Convert a vision-supplied structural name to a weak persistence prior.
    lowered = name.lower()  # Normalize the semantic label for stable keyword matching.
    score = 0.0  # Start from a neutral mechanism prior.
    for token, increment in (("rim", 0.45), ("hole", 0.45), ("diaphragm", 0.35), ("junction", 0.35), ("rib", 0.30), ("support", 0.25), ("bearing", 0.25), ("edge", 0.20), ("wheel", 0.15), ("load", 0.15)):  # Define bounded bridge-mechanism persistence cues.
        if token in lowered:  # Test whether the semantic region contains the current cue.
            score += increment  # Accumulate the weak prior contribution.
    return float(np.clip(score, 0.0, 1.0))  # Bound semantics so measured physics remains dominant.


def _safe_share(values: np.ndarray) -> np.ndarray:  # Normalize a non-negative vector robustly.
    clipped = np.maximum(np.asarray(values, dtype=float), 0.0)  # Remove tiny negative numerical noise.
    total = float(np.sum(clipped))  # Compute the normalization denominator.
    if total <= 1.0e-30:  # Handle a degenerate zero-information vector.
        return np.full(len(clipped), 1.0 / max(len(clipped), 1), dtype=float)  # Return a neutral uniform share.
    return clipped / total  # Return the normalized non-negative shares.


def _neighbor_average(adjacency: np.ndarray, values: np.ndarray) -> np.ndarray:  # Aggregate one-hop graph information without a graph-learning dependency.
    matrix = np.asarray(adjacency, dtype=float)  # Normalize the adjacency matrix representation.
    degree = np.sum(matrix, axis=1)  # Count each region's graph neighbours.
    return (matrix @ np.asarray(values, dtype=float)) / np.maximum(degree, 1.0)  # Return the stable neighbour average.


class RegionalWorldModel:  # Learn action-conditioned residuals around a finite-element physics prior.
    def __init__(self, config: WorldControllerConfig | None = None) -> None:  # Initialize an empty online transition library.
        self.config = config or WorldControllerConfig()  # Freeze the selected controller configuration.
        self.config.validate()  # Reject invalid model and planning parameters immediately.
        self._x_rows: list[list[float]] = []  # Store regional action-conditioned feature rows.
        self._y_rows: list[list[float]] = []  # Store paired error and resource residual targets.

    @property  # Expose the amount of real transition evidence collected so far.
    def sample_count(self) -> int:  # Return the number of regional transition rows.
        return len(self._x_rows)  # Count the stored feature rows.

    def _features(self, state: RegionalState, action: RegionAction) -> np.ndarray:  # Build action-conditioned regional world-model features.
        depth = action.array(state.n_regions).astype(float)  # Read the discrete delegated future-hit depths.
        error_share = _safe_share(state.err_sum)  # Normalize regional estimator masses.
        element_share = _safe_share(state.elems)  # Normalize regional element counts.
        density = np.maximum(state.err_sum, 1.0e-30) / np.maximum(state.elems, 1.0)  # Compute estimator mass per realized element.
        density_share = _safe_share(density)  # Normalize estimator density across semantic regions.
        volume_share = _safe_share(state.volume)  # Normalize represented material volume.
        stress_scale = max(float(np.max(state.vm_max)), 1.0e-30)  # Establish a stable stress normalization.
        degree = np.sum(state.adjacency, axis=1) / max(state.n_regions - 1, 1)  # Normalize graph degree to the unit interval.
        neighbor_error = _neighbor_average(state.adjacency, error_share)  # Compute adjacent-region error influence.
        neighbor_depth = _neighbor_average(state.adjacency, depth)  # Compute delegated-depth spill toward adjacent regions.
        semantic = np.asarray([semantic_persistence(name) for name in state.names], dtype=float)  # Convert structural names to bounded weak priors.
        columns = [np.ones(state.n_regions), np.log(np.maximum(error_share, 1.0e-12)), np.log(np.maximum(element_share, 1.0e-12)), np.log(np.maximum(density_share, 1.0e-12)), np.log(np.clip(state.sizes / state.h0, 1.0e-6, 1.0)), np.log(np.maximum(state.vm_max / stress_scale, 1.0e-12)), np.log(np.maximum(volume_share, 1.0e-12)), degree, np.log1p(state.hit_count), state.marked_error_fraction, state.marked_element_fraction, depth, neighbor_error, neighbor_depth, semantic, np.full(state.n_regions, min(state.step / 12.0, 2.0))]  # Assemble only observable, action-conditioned descriptors.
        return np.column_stack(columns)  # Return one feature row per stable semantic region.

    def _prior_deltas(self, state: RegionalState, action: RegionAction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # Predict conservative regional changes before residual learning.
        depth = action.array(state.n_regions).astype(float)  # Read non-negative delegated future-hit depths.
        semantic = np.asarray([semantic_persistence(name) for name in state.names], dtype=float)  # Read the weak bridge-mechanism prior.
        exponent = np.clip(self.config.error_power * (1.0 + 0.12 * semantic + 0.04 * np.log1p(state.hit_count)), 1.0, 3.5)  # Adapt the convergence prior mildly by persistent mechanism evidence.
        base_error_multiplier = 1.0 - np.clip(state.marked_error_fraction, 0.0, 1.0) * (1.0 - self.config.refine_factor**exponent)  # Model the mandatory exact-Dörfler error reduction.
        base_resource_multiplier = 1.0 + np.clip(state.marked_element_fraction, 0.0, 1.0) * (self.config.refine_factor ** (-state.dim) - 1.0)  # Model the mandatory exact-Dörfler resource growth.
        support_error = np.clip(state.marked_error_fraction + 0.20 * _neighbor_average(state.adjacency, state.marked_error_fraction), 0.0, 1.0)  # Estimate the local support affected by advance refinement.
        support_resource = np.clip(state.marked_element_fraction + 0.18 * _neighbor_average(state.adjacency, state.marked_element_fraction), 0.0, 1.0)  # Estimate graph-gradation resource spill.
        extra_error_multiplier = 1.0 - support_error * (1.0 - self.config.refine_factor ** (exponent * depth))  # Predict extra reduction from delegated future hits.
        extra_resource_multiplier = 1.0 + support_resource * (self.config.refine_factor ** (-state.dim * depth) - 1.0)  # Predict extra elements from delegated future hits.
        spill = _neighbor_average(state.adjacency, depth)  # Estimate additional size-gradation spill into neighbouring regions.
        resource_multiplier = base_resource_multiplier * extra_resource_multiplier * np.exp(0.10 * spill)  # Combine mandatory and delegated resource effects conservatively.
        error_multiplier = base_error_multiplier * extra_error_multiplier  # Combine mandatory and delegated estimator effects.
        error_delta = np.log(np.clip(error_multiplier, 1.0e-8, 4.0))  # Convert estimator multipliers to bounded logarithmic changes.
        resource_delta = np.log(np.clip(resource_multiplier, 0.25, 64.0))  # Convert element multipliers to bounded logarithmic changes.
        size_depth = np.clip(state.marked_element_fraction + depth, 0.0, float(self.config.max_extra_depth + 1))  # Approximate regional mean target-depth evolution for rollout.
        return error_delta, resource_delta, size_depth  # Return all physics-prior transition components.

    def _ensemble_coefficients(self, width: int) -> list[np.ndarray]:  # Fit deterministic bootstrap ridge residual models.
        if self.sample_count < self.config.min_rows_for_learning:  # Retain the physics prior while evidence is sparse.
            return []  # Signal the prior-only uncertainty path.
        x_values = np.asarray(self._x_rows, dtype=float)  # Assemble the transition design matrix.
        y_values = np.asarray(self._y_rows, dtype=float)  # Assemble paired residual targets.
        if x_values.ndim != 2 or x_values.shape[1] != width:  # Reject an incompatible persisted feature schema.
            raise ValueError("world-model snapshot feature width is incompatible")  # Prevent silent misuse of stale evidence.
        identity = np.eye(width, dtype=float)  # Build the ridge regularization matrix.
        coefficients: list[np.ndarray] = []  # Allocate the bootstrap ensemble.
        for member in range(self.config.ensemble_size):  # Fit each residual model independently.
            rng = np.random.default_rng(self.config.random_seed + 7919 * member + self.sample_count)  # Derive a deterministic bootstrap seed.
            indices = rng.integers(0, len(x_values), size=len(x_values))  # Resample real regional transitions with replacement.
            x_boot = x_values[indices]  # Select the bootstrap feature matrix.
            y_boot = y_values[indices]  # Select the paired residual targets.
            lhs = x_boot.T @ x_boot + self.config.ridge * identity  # Form the regularized normal matrix.
            rhs = x_boot.T @ y_boot  # Form the two-target normal-equation right-hand side.
            coefficients.append(np.linalg.solve(lhs, rhs))  # Solve and retain one residual-model member.
        return coefficients  # Return the complete deterministic ensemble.

    def predict(self, state: RegionalState, action: RegionAction) -> TransitionPrediction:  # Roll the regional world one adaptive step forward.
        features = self._features(state, action)  # Build the action-conditioned model input.
        prior_error, prior_resource, size_depth = self._prior_deltas(state, action)  # Evaluate the finite-element physics prior.
        coefficients = self._ensemble_coefficients(features.shape[1])  # Fit residual members when enough evidence exists.
        error_members: list[np.ndarray] = []  # Allocate per-member log-error transitions.
        resource_members: list[np.ndarray] = []  # Allocate per-member log-resource transitions.
        if coefficients:  # Apply learned residual corrections when real evidence is sufficient.
            for beta in coefficients:  # Evaluate every bootstrap member.
                error_members.append(np.clip(prior_error + features @ beta[:, 0], -8.0, 1.5))  # Predict bounded regional log-error changes.
                resource_members.append(np.clip(prior_resource + features @ beta[:, 1], -1.5, 4.5))  # Predict bounded regional log-resource changes.
        else:  # Use a deliberately uncertain physics-prior ensemble before learning.
            error_members = [prior_error.copy() for _ in range(self.config.ensemble_size)]  # Replicate the prior mean across members.
            resource_members = [prior_resource.copy() for _ in range(self.config.ensemble_size)]  # Replicate the resource prior across members.
        error_stack = np.stack(error_members, axis=0)  # Assemble the error-transition ensemble tensor.
        resource_stack = np.stack(resource_members, axis=0)  # Assemble the resource-transition ensemble tensor.
        error_mean_delta = np.mean(error_stack, axis=0)  # Compute the ensemble mean log-error change.
        resource_mean_delta = np.mean(resource_stack, axis=0)  # Compute the ensemble mean log-resource change.
        if coefficients:  # Estimate epistemic spread from learned bootstrap disagreement.
            error_sigma = np.maximum(np.std(error_stack, axis=0, ddof=0), 0.035)  # Retain a non-zero learned error floor.
            resource_sigma = np.maximum(np.std(resource_stack, axis=0, ddof=0), 0.030)  # Retain a non-zero learned resource floor.
        else:  # Encode explicit uncertainty for a prior-only action proposal.
            depth = action.array(state.n_regions).astype(float)  # Read the delegated action magnitude.
            error_sigma = 0.16 + 0.07 * depth  # Increase prior error uncertainty with delegated depth.
            resource_sigma = 0.13 + 0.06 * depth  # Increase prior resource uncertainty with delegated depth.
        error_upper_delta = error_mean_delta + self.config.confidence_z * error_sigma  # Form conservative log-error changes.
        resource_upper_delta = resource_mean_delta + self.config.confidence_z * resource_sigma  # Form conservative log-resource changes.
        next_error_mean = np.maximum(state.err_sum, 1.0e-30) * np.exp(error_mean_delta)  # Predict regional estimator means.
        next_error_upper = np.maximum(state.err_sum, 1.0e-30) * np.exp(error_upper_delta)  # Predict conservative regional estimator masses.
        next_elems_mean = np.maximum(state.elems, 1.0) * np.exp(resource_mean_delta)  # Predict regional element-count means.
        next_elems_upper = np.maximum(state.elems, 1.0) * np.exp(resource_upper_delta)  # Predict conservative regional element counts.
        element_ratio_mean = float(np.sum(next_elems_mean) / max(state.total_elems, 1.0))  # Convert regional element growth to a global ratio.
        element_ratio_upper = float(np.sum(next_elems_upper) / max(state.total_elems, 1.0))  # Convert conservative element growth to a global ratio.
        next_sizes = np.clip(state.sizes * self.config.refine_factor**size_depth, state.h_min, state.h0)  # Predict mean regional sizes for the next simulated state.
        return TransitionPrediction(next_error_mean=next_error_mean, next_error_upper=next_error_upper, next_elems_mean=next_elems_mean, next_elems_upper=next_elems_upper, next_sizes=next_sizes, error_log_sigma=error_sigma, resource_log_sigma=resource_sigma, n_equations_mean=float(state.n_equations * element_ratio_mean), n_equations_upper=float(state.n_equations * element_ratio_upper))  # Return the complete action-conditioned transition prediction.

    def observe(self, previous: RegionalState, action: RegionAction, observed: RegionalState) -> None:  # Learn residuals from one real Gmsh-plus-CalculiX transition.
        if previous.names != observed.names:  # Require stable semantic region identity across the transition.
            raise ValueError("world-model transitions require identical region names")  # Reject ambiguous regional alignment.
        features = self._features(previous, action)  # Recreate the pre-action feature rows.
        prior_error, prior_resource, _ = self._prior_deltas(previous, action)  # Evaluate the physics-prior transition.
        actual_error = np.log(np.maximum(observed.err_sum, 1.0e-30) / np.maximum(previous.err_sum, 1.0e-30))  # Measure actual regional log-error changes.
        actual_resource = np.log(np.maximum(observed.elems, 1.0) / np.maximum(previous.elems, 1.0))  # Measure actual regional log-resource changes.
        targets = np.column_stack([actual_error - prior_error, actual_resource - prior_resource])  # Isolate the learned residual corrections.
        for row, target in zip(features, targets):  # Append each region as one action-conditioned transition row.
            if np.all(np.isfinite(row)) and np.all(np.isfinite(target)):  # Retain only numerically valid evidence.
                self._x_rows.append(row.astype(float).tolist())  # Store a JSON-safe feature row.
                self._y_rows.append(target.astype(float).tolist())  # Store its paired residual targets.

    def save(self, path: str | Path) -> Path:  # Persist the independently collected world-model transition library.
        destination = Path(path)  # Normalize the snapshot destination.
        destination.parent.mkdir(parents=True, exist_ok=True)  # Create the containing directory when needed.
        payload = {"schema": 1, "config": asdict(self.config), "x_rows": self._x_rows, "y_rows": self._y_rows}  # Build an explicit versioned JSON payload.
        destination.write_text(json.dumps(payload, indent=1), encoding="utf-8")  # Write the snapshot without binary or framework-specific serialization.
        return destination  # Return the materialized snapshot path.

    @classmethod  # Allow a transition library to be restored without constructing an empty instance first.
    def load(cls, path: str | Path, config: WorldControllerConfig | None = None) -> "RegionalWorldModel":  # Restore a versioned transition snapshot.
        payload = json.loads(Path(path).read_text(encoding="utf-8"))  # Read and decode the persisted JSON evidence.
        if int(payload.get("schema", -1)) != 1:  # Require the known feature and target schema.
            raise ValueError("unsupported world-model snapshot schema")  # Reject an incompatible snapshot explicitly.
        model = cls(config or WorldControllerConfig(**payload["config"]))  # Reconstruct the configured model.
        model._x_rows = [[float(value) for value in row] for row in payload.get("x_rows", [])]  # Restore validated feature rows.
        model._y_rows = [[float(value) for value in row] for row in payload.get("y_rows", [])]  # Restore validated target rows.
        if len(model._x_rows) != len(model._y_rows):  # Verify one target row per feature row.
            raise ValueError("world-model snapshot has unpaired transition rows")  # Reject incomplete evidence.
        return model  # Return the restored regional world model.


def build_regional_state(partition: Partition, post: PostState, eta2: np.ndarray, labels: np.ndarray, marked: np.ndarray, n_equations: int, step: int, previous_hits: np.ndarray | None = None) -> tuple[RegionalState, RegionFeatures]:  # Convert one real solve into the world-model state contract.
    features = partition.features(post, eta2, labels)  # Aggregate element-wise physical evidence by stable semantic region.
    n_regions = len(partition.seeds)  # Read the semantic graph order.
    marked_mask = np.zeros(len(eta2), dtype=bool)  # Allocate an element-level exact-Dörfler mask.
    marked_mask[np.asarray(marked, dtype=int)] = True  # Mark the exact selected elements.
    marked_error = np.zeros(n_regions, dtype=float)  # Allocate regional captured estimator masses.
    marked_count = np.zeros(n_regions, dtype=float)  # Allocate regional marked-element counts.
    for region in range(n_regions):  # Aggregate exact element marking into diagnostic regional fractions.
        region_mask = labels == region  # Select all elements in the current semantic region.
        selected = region_mask & marked_mask  # Select the exact Dörfler elements inside the region.
        marked_error[region] = float(np.sum(eta2[selected]))  # Sum the captured estimator mass.
        marked_count[region] = float(np.sum(selected))  # Count the captured elements.
    marked_error_fraction = marked_error / np.maximum(features.err_sum, 1.0e-30)  # Compute the within-region captured error fraction.
    marked_element_fraction = marked_count / np.maximum(features.elems.astype(float), 1.0)  # Compute the within-region marked-element fraction.
    hits = np.zeros(n_regions, dtype=float) if previous_hits is None else np.asarray(previous_hits, dtype=float).copy()  # Carry forward previous exact-Dörfler hit counts.
    hits += (marked_count > 0.0).astype(float)  # Increment every region hit in the current real solve.
    adjacency = partition.adjacency_matrix(post.mesh, labels)  # Build the realized semantic-region adjacency graph.
    state = RegionalState(names=tuple(seed.name for seed in partition.seeds), err_sum=np.maximum(features.err_sum.astype(float), 1.0e-30), elems=np.maximum(features.elems.astype(float), 1.0), sizes=np.clip(features.h_meas.astype(float), partition.problem.h_min, partition.problem.h0), vm_max=np.maximum(features.vm_max.astype(float), 0.0), volume=np.maximum(features.volume.astype(float), 1.0e-30), adjacency=adjacency.astype(float), hit_count=hits, marked_error_fraction=np.clip(marked_error_fraction, 0.0, 1.0), marked_element_fraction=np.clip(marked_element_fraction, 0.0, 1.0), n_equations=int(n_equations), step=int(step), h0=float(partition.problem.h0), h_min=float(partition.problem.h_min), dim=int(partition.problem.dim))  # Freeze all observed decision-relevant evidence.
    return state, features  # Return both the model state and the existing regional diagnostics.


def _virtual_dorfler_fractions(error: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:  # Approximate the next element-level marking during model-only rollout.
    values = np.maximum(np.asarray(error, dtype=float), 0.0)  # Normalize predicted regional estimator masses.
    order = np.argsort(values)[::-1]  # Rank regions by predicted estimator mass.
    remaining = theta * float(np.sum(values))  # Set the predicted global bulk target.
    error_fraction = np.zeros(len(values), dtype=float)  # Allocate predicted within-region error capture.
    element_fraction = np.zeros(len(values), dtype=float)  # Allocate a conservative marked-element proxy.
    for region in order:  # Fill the bulk target region by region.
        if remaining <= 0.0:  # Stop when the predicted bulk target is met.
            break  # Leave all lower-ranked regions unmarked.
        captured = min(float(values[region]), remaining)  # Capture only the remaining required mass.
        fraction = captured / max(float(values[region]), 1.0e-30)  # Convert captured mass to a within-region fraction.
        error_fraction[region] = fraction  # Store the predicted regional error capture.
        element_fraction[region] = min(0.55, max(0.06, 0.35 * fraction))  # Use a bounded sparse element-marking proxy.
        remaining -= captured  # Reduce the outstanding global bulk target.
    return error_fraction, element_fraction  # Return the simulated marking fractions.


def _prediction_state(current: RegionalState, prediction: TransitionPrediction, action: RegionAction, theta: float) -> RegionalState:  # Convert one prediction into the next simulated planning state.
    marked_error_fraction, marked_element_fraction = _virtual_dorfler_fractions(prediction.next_error_mean, theta)  # Recompute a virtual next-step Dörfler support.
    action_values = action.array(current.n_regions)  # Read the delegated-depth vector.
    next_hits = current.hit_count + (marked_error_fraction > 0.0).astype(float)  # Advance the simulated regional hit history.
    return RegionalState(names=current.names, err_sum=np.maximum(prediction.next_error_mean, 1.0e-30), elems=np.maximum(prediction.next_elems_mean, 1.0), sizes=np.clip(prediction.next_sizes, current.h_min, current.h0), vm_max=current.vm_max, volume=current.volume, adjacency=current.adjacency, hit_count=next_hits + 0.15 * action_values, marked_error_fraction=marked_error_fraction, marked_element_fraction=marked_element_fraction, n_equations=max(int(round(prediction.n_equations_mean)), 1), step=current.step + 1, h0=current.h0, h_min=current.h_min, dim=current.dim)  # Return the next hypothetical world state.


def enumerate_actions(state: RegionalState, config: WorldControllerConfig) -> list[RegionAction]:  # Enumerate a compact, deterministic, Dörfler-subsuming action set.
    zero = tuple(0 for _ in range(state.n_regions))  # Build the mandatory pure-Dörfler fallback action.
    actions: list[RegionAction] = [RegionAction(zero)]  # Place the fallback first so it can never be omitted.
    eligible = np.nonzero(state.marked_element_fraction > 0.0)[0]  # Permit delegation only where current exact Dörfler found evidence.
    if len(eligible) == 0 or config.max_extra_depth == 0:  # Handle a terminal or deliberately disabled action space.
        return actions  # Return only the classical fallback.
    error_share = _safe_share(state.err_sum)  # Compute regional estimator importance.
    semantic = np.asarray([semantic_persistence(name) for name in state.names], dtype=float)  # Compute weak structural persistence cues.
    rank_score = error_share * (1.0 + 0.35 * semantic) * (1.0 + 0.10 * np.log1p(state.hit_count))  # Rank current hits by physics and persistence evidence.
    ranked = sorted((int(region) for region in eligible), key=lambda region: float(rank_score[region]), reverse=True)[: config.candidate_regions]  # Retain a bounded candidate set.
    for region in ranked:  # Create single-region delegated actions at all admitted depths.
        for depth in range(1, config.max_extra_depth + 1):  # Enumerate integral future-hit depth.
            values = [0 for _ in range(state.n_regions)]  # Start from exact Dörfler.
            values[region] = depth  # Delegate the selected future-hit depth to one region.
            actions.append(RegionAction(tuple(values)))  # Add the discrete action.
    for width in range(2, min(config.max_active_regions, len(ranked)) + 1):  # Create bounded multi-region coordination actions.
        for group in combinations(ranked, width):  # Enumerate stable region combinations.
            for shared_depth in range(1, min(config.max_extra_depth, 2) + 1):  # Keep coordinated actions shallow enough for safety.
                values = [0 for _ in range(state.n_regions)]  # Start from exact Dörfler.
                for region in group:  # Assign the same conservative depth to each coordinated region.
                    values[region] = shared_depth  # Record one non-negative delegated depth.
                actions.append(RegionAction(tuple(values)))  # Add the coordinated discrete action.
    unique: dict[tuple[int, ...], RegionAction] = {}  # Allocate a deterministic de-duplication map.
    for action in actions:  # Remove any duplicate action vectors.
        unique.setdefault(action.extra_depth, action)  # Preserve the first deterministic occurrence.
    return list(unique.values())  # Return pure Dörfler plus all bounded world-model candidates.


def _trajectory_score(prediction: TransitionPrediction, action: RegionAction, n_eq_cap: int, config: WorldControllerConfig) -> float:  # Score one hypothetical transition for beam search.
    error_term = math.log(max(prediction.total_error_upper, 1.0e-30))  # Prefer lower risk-adjusted estimator mass.
    resource_ratio = prediction.n_equations_upper / max(float(n_eq_cap), 1.0)  # Normalize conservative equation use.
    resource_term = config.resource_penalty * max(resource_ratio, 0.0) ** 2  # Penalize finite-budget consumption smoothly.
    uncertainty_term = config.uncertainty_penalty * float(np.max(prediction.error_log_sigma))  # Penalize epistemic uncertainty.
    action_term = config.action_penalty * float(np.sum(action.extra_depth))  # Penalize unnecessary delegated future depth.
    return float(error_term + resource_term + uncertainty_term + action_term)  # Return the scalar beam-search objective.


def _rollout_first_actions(model: RegionalWorldModel, initial: RegionalState, n_eq_cap: int, config: WorldControllerConfig) -> list[tuple[float, RegionAction, tuple[tuple[int, ...], ...], TransitionPrediction]]:  # Evaluate finite-horizon action sequences by bounded beam search.
    beam: list[tuple[float, RegionalState, RegionAction | None, tuple[tuple[int, ...], ...], TransitionPrediction | None]] = [(0.0, initial, None, tuple(), None)]  # Seed the search with the observed real state.
    for _ in range(config.planning_horizon):  # Roll the regional world forward several hypothetical adaptive steps.
        expanded: list[tuple[float, RegionalState, RegionAction, tuple[tuple[int, ...], ...], TransitionPrediction]] = []  # Allocate the next beam layer.
        for accumulated, state, first_action, trajectory, _ in beam:  # Expand every retained hypothetical world state.
            for action in enumerate_actions(state, config):  # Evaluate pure Dörfler and bounded delegated alternatives.
                prediction = model.predict(state, action)  # Predict action-conditioned error and resource consequences.
                if not action.is_dorfler and prediction.n_equations_upper > config.budget_safety * n_eq_cap:  # Reject risky delegated actions before execution.
                    continue  # Keep the exact Dörfler branch available instead.
                chosen_first = action if first_action is None else first_action  # Preserve the sequence's executable first action.
                next_trajectory = trajectory + (action.extra_depth,)  # Append the simulated discrete action.
                score = accumulated + _trajectory_score(prediction, action, n_eq_cap, config)  # Accumulate risk-adjusted trajectory cost.
                next_state = _prediction_state(state, prediction, action, config.theta)  # Form the next model-only world state.
                expanded.append((score, next_state, chosen_first, next_trajectory, prediction))  # Add the simulated child trajectory.
        if not expanded:  # Guard against an unexpectedly empty constrained action tree.
            break  # Return the best trajectories accumulated so far.
        expanded.sort(key=lambda item: item[0])  # Rank hypothetical trajectories by cumulative risk-adjusted cost.
        beam = expanded[: config.beam_width]  # Retain only the configured beam width.
    return [(score, first_action or RegionAction(tuple(0 for _ in range(initial.n_regions))), trajectory, prediction) for score, _, first_action, trajectory, prediction in beam if prediction is not None]  # Return executable first actions with terminal evidence.


def plan_action(model: RegionalWorldModel, state: RegionalState, n_eq_cap: int, config: WorldControllerConfig) -> PlanDecision:  # Admit a world-model action only when it beats a pure-Dörfler rollout.
    zero_action = RegionAction(tuple(0 for _ in range(state.n_regions)))  # Build the mandatory classical fallback.
    baseline_state = state  # Start the pure-Dörfler comparison from the same observed state.
    baseline_prediction: TransitionPrediction | None = None  # Allocate the terminal baseline prediction.
    baseline_trajectory: list[tuple[int, ...]] = []  # Allocate the audited baseline action sequence.
    for _ in range(config.planning_horizon):  # Roll pure Dörfler through the same planning horizon.
        baseline_prediction = model.predict(baseline_state, zero_action)  # Predict one exact-Dörfler transition.
        baseline_trajectory.append(zero_action.extra_depth)  # Record the fallback action.
        baseline_state = _prediction_state(baseline_state, baseline_prediction, zero_action, config.theta)  # Advance the baseline world state.
    assert baseline_prediction is not None  # Establish the positive planning-horizon invariant for static type checkers.
    baseline_terminal = baseline_prediction.total_error_upper  # Read the pure-Dörfler terminal risk bound.
    candidates = _rollout_first_actions(model, state, n_eq_cap, config)  # Evaluate all bounded multi-step candidate trajectories.
    nonzero = [candidate for candidate in candidates if not candidate[1].is_dorfler]  # Retain trajectories whose executable first action adds world-model depth.
    if not nonzero:  # Handle an empty safe world-action set.
        first_prediction = model.predict(state, zero_action)  # Compute the fallback's first-step resource evidence.
        return PlanDecision(action=zero_action, source="dorfler", predicted_gain=0.0, baseline_terminal_error=baseline_terminal, selected_terminal_error=baseline_terminal, selected_equations_upper=first_prediction.n_equations_upper, selected_sigma=float(np.max(first_prediction.error_log_sigma)), trajectory=tuple(baseline_trajectory), reason="no_safe_nonzero_trajectory")  # Return the exact Dörfler policy.
    selected = min(nonzero, key=lambda item: item[0])  # Choose the lowest-cost safe finite-horizon trajectory.
    _, action, trajectory, terminal_prediction = selected  # Unpack the selected first action and terminal evidence.
    first_prediction = model.predict(state, action)  # Re-evaluate the executable transition for admission checks.
    selected_terminal = terminal_prediction.total_error_upper  # Read the selected terminal risk bound.
    gain = (baseline_terminal - selected_terminal) / max(baseline_terminal, 1.0e-30)  # Compute the relative risk-adjusted gain over Dörfler.
    sigma = float(np.max(first_prediction.error_log_sigma))  # Read the largest regional epistemic uncertainty.
    if first_prediction.n_equations_upper > config.budget_safety * n_eq_cap:  # Enforce the conservative equation-cap shield.
        return PlanDecision(action=zero_action, source="dorfler", predicted_gain=float(gain), baseline_terminal_error=baseline_terminal, selected_terminal_error=selected_terminal, selected_equations_upper=first_prediction.n_equations_upper, selected_sigma=sigma, trajectory=tuple(baseline_trajectory), reason="resource_upper_bound_failed")  # Fall back before meshing.
    if sigma > config.max_log_error_sigma:  # Enforce the uncertainty admission shield.
        return PlanDecision(action=zero_action, source="dorfler", predicted_gain=float(gain), baseline_terminal_error=baseline_terminal, selected_terminal_error=selected_terminal, selected_equations_upper=first_prediction.n_equations_upper, selected_sigma=sigma, trajectory=tuple(baseline_trajectory), reason="uncertainty_gate_failed")  # Fall back before meshing.
    if gain < config.min_predicted_gain:  # Require a material terminal improvement over pure Dörfler.
        return PlanDecision(action=zero_action, source="dorfler", predicted_gain=float(gain), baseline_terminal_error=baseline_terminal, selected_terminal_error=selected_terminal, selected_equations_upper=first_prediction.n_equations_upper, selected_sigma=sigma, trajectory=tuple(baseline_trajectory), reason="gain_gate_failed")  # Fall back when the learned advantage is too small.
    return PlanDecision(action=action, source="world_model", predicted_gain=float(gain), baseline_terminal_error=baseline_terminal, selected_terminal_error=selected_terminal, selected_equations_upper=first_prediction.n_equations_upper, selected_sigma=sigma, trajectory=trajectory, reason="admitted")  # Admit the world-model-guided delegated action.


def _cell_halo(mesh: Mesh, seed_cells: np.ndarray, hops: int, allowed: np.ndarray) -> np.ndarray:  # Grow a local element support over the face-adjacency graph.
    active = np.zeros(mesh.n_cells, dtype=bool)  # Allocate the current support mask.
    active[np.asarray(seed_cells, dtype=int)] = True  # Seed the support with exact marked elements.
    active &= allowed  # Keep the support inside the selected semantic region.
    pairs, _ = mesh.cell_adjacency  # Read all face-adjacent element pairs.
    for _ in range(max(int(hops), 0)):  # Grow the support by the requested number of adjacency layers.
        touched = active[pairs[:, 0]] | active[pairs[:, 1]]  # Identify graph edges touching the current support.
        additions = np.zeros(mesh.n_cells, dtype=bool)  # Allocate the next-layer additions.
        additions[pairs[touched, 0]] = True  # Add the first cell of every touched graph edge.
        additions[pairs[touched, 1]] = True  # Add the second cell of every touched graph edge.
        active |= additions & allowed  # Grow only within the stable semantic region.
    return np.nonzero(active)[0]  # Return the final local support element indices.


def compile_dorfler_dominating_target(mesh: Mesh, labels: np.ndarray, marked: np.ndarray, action: RegionAction, state: RegionalState, config: WorldControllerConfig) -> tuple[np.ndarray, np.ndarray, int]:  # Compile one discrete action into mandatory and candidate nodal targets.
    depth = action.array(state.n_regions)  # Validate and read the non-negative delegated depths.
    base_target = refine_size_map(mesh, np.asarray(marked, dtype=int), factor=config.refine_factor)  # Build the exact element-level Dörfler target map.
    candidate = base_target.copy()  # Start the candidate from the complete classical action.
    support_union = np.zeros(mesh.n_cells, dtype=bool)  # Allocate the union of all delegated local supports.
    marked_values = np.asarray(marked, dtype=int)  # Normalize exact marked indices.
    for region, extra_depth in enumerate(depth):  # Compile each admitted regional future-hit delegation.
        if int(extra_depth) <= 0:  # Skip regions retaining pure Dörfler behaviour.
            continue  # Leave their exact element-level target unchanged.
        region_marked = marked_values[labels[marked_values] == region]  # Select exact Dörfler hits inside this semantic region.
        if len(region_marked) == 0:  # Reject unsupported future investment defensively.
            continue  # Do not create refinement from semantics alone.
        allowed = labels == region  # Restrict graph support to the fixed semantic region.
        hops = config.support_hops_per_depth * int(extra_depth)  # Convert future-hit depth to a bounded graph halo.
        support_cells = _cell_halo(mesh, region_marked, hops, allowed)  # Grow a local support around physically marked elements.
        support_union[support_cells] = True  # Add the regional support to the audited union.
        support_nodes = np.unique(mesh.cells[support_cells].ravel())  # Convert support elements to nodal target locations.
        requested = mesh.node_sizes[support_nodes] * config.refine_factor ** (1 + int(extra_depth))  # Advance the mandatory hit by the delegated future depth.
        candidate[support_nodes] = np.minimum(candidate[support_nodes], requested)  # Refine only; never erase or coarsen Dörfler.
    candidate = np.clip(candidate, state.h_min, state.h0)  # Enforce the family mesh-size bounds deterministically.
    base_target = np.clip(base_target, state.h_min, state.h0)  # Apply the same bounds to the mandatory baseline target.
    return base_target, candidate, int(np.sum(support_union))  # Return both target maps and delegated support size.


def estimate_free_equations(mesh: Mesh, problem) -> int:  # Compute the exact displacement-equation count implied by a generated mesh and constraints.
    constrained: set[tuple[int, int]] = set()  # Allocate unique constrained node-and-DOF pairs.
    for constraint in problem.constraints:  # Evaluate every repository boundary-condition predicate.
        mask = np.asarray(constraint.node_predicate(mesh.nodes), dtype=bool)  # Select all constrained mesh nodes geometrically.
        for node in np.nonzero(mask)[0]:  # Iterate over each selected node index.
            for dof in constraint.dofs:  # Iterate over each constrained displacement component.
                if 1 <= int(dof) <= int(problem.dim):  # Ignore non-displacement components outside this simplex formulation.
                    constrained.add((int(node), int(dof)))  # Add the unique constrained equation key.
    return max(int(problem.dim * mesh.n_nodes - len(constrained)), 1)  # Return the exact free displacement-DOF count.


def _array_hash(values: np.ndarray) -> str:  # Hash one numerical target field reproducibly.
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))  # Normalize byte order and memory layout.
    return hashlib.sha256(array.tobytes()).hexdigest()  # Return the full SHA-256 digest.


def _action_hash(names: tuple[str, ...], action: RegionAction) -> str:  # Hash stable semantic identities and discrete future depths.
    payload = json.dumps({"names": names, "extra_depth": action.extra_depth}, sort_keys=True, separators=(",", ":")).encode("utf-8")  # Build a canonical JSON action payload.
    return hashlib.sha256(payload).hexdigest()  # Return the immutable action digest.


def certify_action(base_target: np.ndarray, candidate_target: np.ndarray, mesh: Mesh, marked: np.ndarray, state: RegionalState, action: RegionAction, support_elements: int, generated_equations: int, n_eq_cap: int, config: WorldControllerConfig, fallback_used: bool = False) -> ActionReceipt:  # Certify that tools, not the language model, own every numerical parameter.
    base = np.asarray(base_target, dtype=float)  # Normalize the mandatory Dörfler target map.
    candidate = np.asarray(candidate_target, dtype=float)  # Normalize the compiled candidate target map.
    marked_nodes = np.unique(mesh.cells[np.asarray(marked, dtype=int)].ravel()) if len(marked) else np.array([], dtype=int)  # Identify all mandatory Dörfler nodes.
    finite_pass = bool(np.all(np.isfinite(candidate)))  # Verify numerical finiteness.
    bounds_pass = bool(np.all(candidate >= state.h_min - 1.0e-10) and np.all(candidate <= state.h0 + 1.0e-10))  # Verify family size bounds.
    dominance_pass = bool(np.all(candidate <= mesh.node_sizes + 1.0e-10) and np.all(candidate[marked_nodes] <= base[marked_nodes] + 1.0e-10))  # Verify no coarsening and complete Dörfler support domination.
    delegated = tuple(state.names[index] for index, depth in enumerate(action.extra_depth) if int(depth) > 0)  # Record only regions receiving world-model depth.
    budget_pass = bool(generated_equations <= int(math.floor(config.budget_safety * n_eq_cap)))  # Verify the generated Gmsh mesh against the safety-adjusted cap.
    return ActionReceipt(action_hash=_action_hash(state.names, action), base_target_hash=_array_hash(base), candidate_target_hash=_array_hash(candidate), marked_elements=int(len(marked)), delegated_regions=delegated, support_elements=int(support_elements), dominance_pass=dominance_pass, bounds_pass=bounds_pass, finite_pass=finite_pass, estimated_equations=int(generated_equations), budget_pass=budget_pass, fallback_used=bool(fallback_used))  # Return the complete deterministic receipt.


def run_world_model_vla(runner: FemRunner, partitioner, n_eq_cap: int, config: WorldControllerConfig | None = None, model: RegionalWorldModel | None = None, method: str = "wm_vla") -> WorldVLAResult:  # Execute the real multi-step VLA with online world-model correction and Dörfler fallback.
    cfg = config or WorldControllerConfig()  # Resolve the immutable controller configuration.
    cfg.validate()  # Validate all safety and planning parameters.
    if n_eq_cap <= 0:  # Validate the hard equation budget.
        raise ValueError("n_eq_cap must be positive")  # Reject an invalid finite-resource experiment.
    world = model or RegionalWorldModel(cfg)  # Reuse a cross-case transition library or create an empty model.
    problem = runner.problem  # Read the finite-element problem owned by the runner.
    if cfg.require_reference:  # Build or load reference evidence only when the experiment requests it.
        runner.ensure_reference()  # Materialize the common reference before counted method solves.
    seeds = partitioner.propose(problem)  # Ask the vision head only for named semantic regions and ordinal priors.
    drawings = list(getattr(partitioner, "last_drawings", []) or [])  # Read fixed geometric drawings without any solved-field information.
    partition = Partition(seeds, problem, gradation=cfg.gradation, assign_mode="drawn", drawings=drawings)  # Freeze the semantic partition across the adaptive trajectory.
    mesh = initial_mesh(problem)  # Use the common uniform probe so world-model gains are not conflated with a privileged first mesh.
    previous_state: RegionalState | None = None  # Allocate the previous real state for online residual learning.
    previous_action: RegionAction | None = None  # Allocate the action that produced the next real state.
    hit_count = np.zeros(len(seeds), dtype=float)  # Initialize exact-Dörfler regional hit history.
    receipts: list[ActionReceipt] = []  # Allocate deterministic action compilation receipts.
    world_actions = 0  # Count executed non-zero world-model actions.
    fallbacks = 0  # Count deterministic Dörfler fallbacks.
    stopped_by = "solve_cap"  # Set the default terminal reason.
    for step in range(cfg.max_solves):  # Execute a genuine multi-step solve-estimate-plan-remesh loop.
        post, record = runner.solve_mesh(mesh, method=method, stage=f"cycle{step}")  # Execute one real CalculiX solve through the audited runner.
        eta2 = zz_indicator(problem, post)  # Compute the common element-wise ZZ indicator.
        marked = dorfler_mark(eta2, cfg.theta)  # Compute the exact classical bulk-marking support.
        labels = partition.assign(mesh)  # Reapply the fixed semantic drawings to the realized Gmsh mesh.
        state, features = build_regional_state(partition, post, eta2, labels, marked, record.n_equations, step, hit_count)  # Build the observed regional world state.
        hit_count = state.hit_count.copy()  # Carry the updated exact-Dörfler hit history forward.
        if previous_state is not None and previous_action is not None:  # Learn only after a complete real action-conditioned transition exists.
            world.observe(previous_state, previous_action, state)  # Correct the physics prior from real Gmsh and CalculiX evidence.
        record.extra.update(sum_eta2=float(np.sum(eta2)), n_marked=int(len(marked)), region_names=list(state.names), region_error=state.err_sum.tolist(), region_elems=state.elems.tolist(), region_sizes=state.sizes.tolist(), region_hits=state.hit_count.tolist(), world_model_rows=int(world.sample_count), controller="multi_step_world_model_vla", local_prediction_imported=False)  # Attach complete scientific and purity evidence to the solve record.
        if len(marked) == 0:  # Stop when the common estimator produces no refinement support.
            stopped_by = "empty_marking"  # Record the physical terminal condition.
            record.extra["stop"] = stopped_by  # Preserve the condition in the counted solve record.
            break  # End the adaptive loop.
        if step + 1 >= cfg.max_solves:  # Stop after the configured number of real feedback states.
            stopped_by = "solve_cap"  # Record the configured terminal condition.
            record.extra["stop"] = stopped_by  # Preserve the condition in the counted solve record.
            break  # End the adaptive loop.
        if record.n_equations >= n_eq_cap:  # Stop after reaching or exceeding the same hard cap as Dörfler.
            stopped_by = "equation_cap"  # Record the resource terminal condition.
            record.extra["stop"] = stopped_by  # Preserve the condition in the counted solve record.
            break  # End the adaptive loop.
        decision = plan_action(world, state, n_eq_cap, cfg)  # Ask the finite-horizon world model whether extra future depth is justified.
        action = decision.action  # Read the admitted discrete action or exact Dörfler fallback.
        if decision.source != "world_model":  # Count planner-level classical fallback.
            fallbacks += 1  # Increment the fallback audit counter.
        base_target, candidate_target, support_elements = compile_dorfler_dominating_target(mesh, labels, marked, action, state, cfg)  # Compile the discrete action through deterministic numerical tools.
        candidate_field = NodalSizeField(mesh, candidate_target, gradation=cfg.gradation, h_min=problem.h_min, h_max=problem.h0)  # Build the deterministic Gmsh target interpolator.
        candidate_mesh = generate_mesh(problem, candidate_field)  # Materialize the exact candidate mesh before any further real solve.
        estimated_equations = estimate_free_equations(candidate_mesh, problem)  # Compute exact free displacement equations on the generated mesh.
        provisional = certify_action(base_target, candidate_target, mesh, marked, state, action, support_elements, estimated_equations, n_eq_cap, cfg, fallback_used=False)  # Certify the initially planned action.
        valid = provisional.finite_pass and provisional.bounds_pass and provisional.dominance_pass and provisional.budget_pass  # Combine all deterministic numerical gates.
        if not valid and not action.is_dorfler:  # Replace any failed world action by the exact Dörfler action.
            fallbacks += 1  # Count the certification-level fallback.
            action = RegionAction(tuple(0 for _ in range(state.n_regions)))  # Reset delegated depth to the classical policy.
            base_target, candidate_target, support_elements = compile_dorfler_dominating_target(mesh, labels, marked, action, state, cfg)  # Recompile the exact Dörfler target.
            candidate_field = NodalSizeField(mesh, candidate_target, gradation=cfg.gradation, h_min=problem.h_min, h_max=problem.h0)  # Rebuild the deterministic target interpolator.
            candidate_mesh = generate_mesh(problem, candidate_field)  # Materialize the certified fallback mesh.
            estimated_equations = estimate_free_equations(candidate_mesh, problem)  # Recompute exact free equations for the fallback mesh.
            receipt = certify_action(base_target, candidate_target, mesh, marked, state, action, support_elements, estimated_equations, n_eq_cap, cfg, fallback_used=True)  # Record the fallback receipt.
        else:  # Retain the originally planned action.
            receipt = provisional  # Preserve its complete certification evidence.
        if not (receipt.finite_pass and receipt.bounds_pass and receipt.dominance_pass):  # Refuse to execute a numerically invalid target even for the baseline.
            stopped_by = "target_certification_failed"  # Record the hard tool-contract failure.
            record.extra["stop"] = stopped_by  # Preserve the failure in the counted solve record.
            receipts.append(receipt)  # Retain the failed receipt for diagnosis.
            break  # End the adaptive loop without another solve.
        if not receipt.budget_pass and receipt.fallback_used:  # Stop when even exact Dörfler's next realized mesh would violate the safety-adjusted cap.
            stopped_by = "next_dorfler_mesh_over_budget"  # Record the fair resource stop.
            record.extra["stop"] = stopped_by  # Preserve the stop in the counted solve record.
            receipts.append(receipt)  # Retain the resource receipt.
            break  # End the adaptive loop before launching an over-budget solve.
        receipts.append(receipt)  # Preserve the accepted deterministic action receipt.
        if not action.is_dorfler:  # Count only actions that actually add world-model depth.
            world_actions += 1  # Increment the executed world-action counter.
        record.extra.update(plan=asdict(decision), action_receipt=asdict(receipt), dorfler_target_dominated=True)  # Attach finite-horizon and tool-certification evidence to the source solve.
        previous_state = state  # Retain the observed state for the next online residual update.
        previous_action = action  # Retain the executed discrete action for the next online residual update.
        mesh = candidate_mesh  # Advance to the certified Gmsh mesh for the next real CalculiX solve.
    method_records = [record for record in runner.records if record.method == method]  # Isolate counted records belonging to this controller.
    return WorldVLAResult(solves=len(method_records), stopped_by=stopped_by, world_actions=world_actions, dorfler_fallbacks=fallbacks, transition_rows=world.sample_count, receipts=tuple(receipts))  # Return the complete run summary without claiming unmeasured superiority.
