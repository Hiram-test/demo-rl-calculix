"""Action-conditioned regional world model for adaptive finite-element planning."""  # Describe the learned transition model implemented below.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from dataclasses import asdict, dataclass  # Import immutable data contracts and serialization support.
import json  # Import the snapshot interchange format.
from pathlib import Path  # Import portable snapshot paths.
from typing import Any  # Import a generic JSON-compatible value type.
import numpy as np  # Import the numerical linear-algebra backend.

def _array(value: Any, *, dtype: Any = float) -> np.ndarray:  # Normalize state fields to independent arrays.
    return np.asarray(value, dtype=dtype).copy()  # Prevent callers from mutating model state through shared storage.

def semantic_persistence(name: str) -> float:  # Estimate whether a named bridge mechanism will remain active across remeshes.
    text = str(name).lower()  # Normalize the region name for deterministic matching.
    if any(token in text for token in ("opening", "hole", "rim", "weld", "joint", "diaphragm")):  # Detect persistent geometric concentration mechanisms.
        return 1.0  # Assign the strongest persistence prior to geometry-controlled hotspots.
    if any(token in text for token in ("load", "wheel", "support", "bearing")):  # Detect persistent boundary-condition mechanisms.
        return 0.82  # Assign a strong but slightly lower persistence prior to boundary footprints.
    if any(token in text for token in ("web", "flange", "corner")):  # Detect structural load-path transition mechanisms.
        return 0.68  # Assign a moderate persistence prior to member junctions.
    if any(token in text for token in ("field", "remainder", "background", "coarse")):  # Detect generic non-mechanistic regions.
        return 0.0  # Prevent the world model from inventing a generic-field refinement policy.
    return 0.35  # Retain a weak neutral prior for otherwise valid semantic regions.

@dataclass(frozen=True)  # Make the measured world state immutable.
class WorldState:  # Store the compact decision-relevant state after one real finite-element solve.
    names: tuple[str, ...]  # Store stable semantic region identifiers.
    err_sum: np.ndarray  # Store the regional sums of squared error indicators.
    elems: np.ndarray  # Store the regional element counts.
    sizes: np.ndarray  # Store the regional measured mesh sizes.
    vm_max: np.ndarray  # Store the regional peak von Mises stresses.
    volume: np.ndarray  # Store the regional geometric volumes represented by current elements.
    adjacency: np.ndarray  # Store the normalized region-interaction graph.
    dorfler_error_fraction: np.ndarray  # Store the within-region error fraction selected by exact Dörfler marking.
    dorfler_element_fraction: np.ndarray  # Store the within-region element fraction selected by exact Dörfler marking.
    hit_count: np.ndarray  # Store how many real solves have selected each region.
    n_equations: int  # Store the measured active equation count.
    eq_per_elem: float  # Store the measured equation-to-element conversion for resource prediction.
    h_min: float  # Store the admissible minimum target size.
    h0: float  # Store the initial global target size.
    dim: int  # Store the spatial dimension.
    step: int  # Store the zero-based real-solve index.
    def __post_init__(self) -> None:  # Validate all state invariants immediately.
        count = len(self.names)  # Determine the number of semantic regions.
        vectors = (self.err_sum, self.elems, self.sizes, self.vm_max, self.volume, self.dorfler_error_fraction, self.dorfler_element_fraction, self.hit_count)  # Collect all region-indexed vectors.
        if count == 0:  # Reject an empty semantic partition.
            raise ValueError("world state requires at least one region")  # Explain the invalid state.
        if any(np.asarray(vector).shape != (count,) for vector in vectors):  # Require every regional vector to share one ordering.
            raise ValueError("all regional world-state vectors must match names")  # Report the inconsistent state layout.
        if np.asarray(self.adjacency).shape != (count, count):  # Require a square region graph.
            raise ValueError("world-state adjacency must be square")  # Report the invalid interaction graph.
        if np.any(np.asarray(self.err_sum) < 0.0) or np.any(np.asarray(self.elems) < 0.0):  # Reject negative physical measures.
            raise ValueError("regional error and element counts must be non-negative")  # Explain the physical invariant.
        if self.n_equations <= 0 or self.eq_per_elem <= 0.0:  # Require a usable resource state.
            raise ValueError("world state requires positive resource measurements")  # Explain the resource invariant.
        if self.dim not in (2, 3):  # Restrict the physics prior to supported finite-element dimensions.
            raise ValueError("world state supports only two- or three-dimensional problems")  # Explain the dimensional contract.
    @property  # Expose the global squared indicator as a read-only quantity.
    def total_error(self) -> float:  # Return the global squared-indicator sum.
        return float(np.sum(self.err_sum))  # Aggregate all regional indicator contributions.
    @property  # Expose the global element count as a read-only quantity.
    def total_elements(self) -> float:  # Return the global element count as a float for stable ratios.
        return float(np.sum(self.elems))  # Aggregate all regional element counts.
    @property  # Expose normalized regional error shares.
    def error_share(self) -> np.ndarray:  # Return normalized regional error shares.
        return _array(self.err_sum) / max(self.total_error, 1.0e-30)  # Protect normalization against a converged zero-error state.
    @property  # Expose normalized regional resource shares.
    def element_share(self) -> np.ndarray:  # Return normalized regional resource shares.
        return _array(self.elems) / max(self.total_elements, 1.0)  # Protect normalization against an empty mesh.
    def replaced(self, *, err_sum: np.ndarray, elems: np.ndarray, sizes: np.ndarray, n_equations: int, dorfler_error_fraction: np.ndarray | None = None, dorfler_element_fraction: np.ndarray | None = None, step: int | None = None) -> "WorldState":  # Build a predicted successor state without mutating observations.
        return WorldState(names=self.names, err_sum=_array(err_sum), elems=_array(elems), sizes=_array(sizes), vm_max=_array(self.vm_max), volume=_array(self.volume), adjacency=_array(self.adjacency), dorfler_error_fraction=_array(self.dorfler_error_fraction if dorfler_error_fraction is None else dorfler_error_fraction), dorfler_element_fraction=_array(self.dorfler_element_fraction if dorfler_element_fraction is None else dorfler_element_fraction), hit_count=_array(self.hit_count), n_equations=int(max(1, n_equations)), eq_per_elem=float(self.eq_per_elem), h_min=float(self.h_min), h0=float(self.h0), dim=int(self.dim), step=int(self.step + 1 if step is None else step))  # Preserve the stable semantic graph while replacing transition variables.

@dataclass(frozen=True)  # Make planner actions immutable and auditable.
class RegionAction:  # Represent one exact Dörfler step plus optional regional future-hit depth.
    extra_depth: tuple[int, ...]  # Store non-negative additional refinement depths for each semantic region.
    source: str = "world_model"  # Record whether the action came from planning or the Dörfler fallback.
    def validate(self, state: WorldState, *, max_depth: int = 3) -> None:  # Validate the discrete action against the current state.
        if len(self.extra_depth) != len(state.names):  # Require one discrete depth per region.
            raise ValueError("action depth vector must match world-state regions")  # Explain the ordering mismatch.
        if any(int(depth) != depth or depth < 0 or depth > max_depth for depth in self.extra_depth):  # Forbid coarsening and unbounded refinement.
            raise ValueError("action depths must be bounded non-negative integers")  # Explain the safe action domain.
        for name, depth in zip(state.names, self.extra_depth, strict=True):  # Inspect every named regional action.
            if semantic_persistence(name) == 0.0 and depth != 0:  # Forbid refinement of generic field regions.
                raise ValueError("generic field regions cannot receive world-model depth")  # Preserve the clean semantic mechanism boundary.
    @classmethod  # Construct baseline actions without exposing vector details to callers.
    def dorfler(cls, state: WorldState) -> "RegionAction":  # Construct the mandatory exact-Dörfler baseline action.
        return cls(extra_depth=tuple(0 for _ in state.names), source="dorfler")  # Encode zero additional depth rather than no refinement.
    @property  # Expose whether the action is the exact baseline.
    def is_dorfler_only(self) -> bool:  # Report whether the action adds no world-model depth.
        return not any(self.extra_depth)  # Treat the all-zero vector as the exact-Dörfler action.

@dataclass(frozen=True)  # Make predictions immutable after model evaluation.
class WorldPrediction:  # Store one probabilistic action-conditioned state transition.
    next_state: WorldState  # Store the predicted successor state.
    uncertainty: float  # Store the ensemble epistemic uncertainty in log-transition space.
    failure_risk: float  # Store a bounded transition-risk proxy.
    error_ratio_mean: float  # Store the predicted global error ratio.
    error_ratio_upper: float  # Store a conservative upper error-ratio estimate.
    equation_ratio_mean: float  # Store the predicted global equation ratio.
    equation_ratio_upper: float  # Store a conservative upper equation-ratio estimate.

@dataclass(frozen=True)  # Make world-model settings immutable.
class WorldModelConfig:  # Configure the compact residual ensemble.
    refine_factor: float = 0.5  # Match the repository Dörfler refinement factor.
    error_power: float = 1.65  # Encode a conservative local three-dimensional error-decay prior.
    neighbor_spill: float = 0.18  # Encode Gmsh gradation spill into adjacent semantic regions.
    ensemble_size: int = 5  # Use several bootstrap regressors for epistemic uncertainty.
    ridge: float = 1.0e-3  # Stabilize small-sample online regression.
    min_rows: int = 10  # Require enough regional transitions before applying learned residuals.
    max_log_residual: float = 0.7  # Clip learned corrections outside the observed local regime.
    prior_uncertainty: float = 0.24  # Assign a conservative uncertainty before online learning.
    uncertainty_scale: float = 1.8  # Convert ensemble spread to conservative prediction bounds.
    max_rows: int = 3000  # Bound online memory while retaining several trajectories.

class ResidualWorldModel:  # Learn deviations from an explicit Dörfler and remeshing physics prior.
    def __init__(self, config: WorldModelConfig | None = None, *, seed: int = 271828) -> None:  # Initialize a deterministic online ensemble.
        self.config = config or WorldModelConfig()  # Store an immutable model configuration.
        self.seed = int(seed)  # Store the bootstrap random seed.
        self._x: list[list[float]] = []  # Store regional action-conditioned feature rows.
        self._y: list[list[float]] = []  # Store log error and resource residual targets.
        self.transition_count = 0  # Count fully observed real-solve transitions.
    def _features(self, state: WorldState, action: RegionAction) -> np.ndarray:  # Build region-level transition features.
        action.validate(state)  # Reject unsafe actions before prediction.
        count = len(state.names)  # Determine the regional feature-row count.
        degree = np.sum(state.adjacency, axis=1)  # Measure each region's interaction degree.
        neighbor_error = state.adjacency @ state.error_share  # Aggregate adjacent-region error shares.
        neighbor_depth = state.adjacency @ np.asarray(action.extra_depth, dtype=float)  # Aggregate adjacent planned depths.
        stress_scale = max(float(np.max(state.vm_max)), 1.0e-12)  # Normalize stress without leaking absolute unit choices.
        volume_scale = max(float(np.sum(state.volume)), 1.0e-30)  # Normalize regional geometric volumes.
        rows = np.zeros((count, 15), dtype=float)  # Allocate the compact feature matrix.
        for index, name in enumerate(state.names):  # Populate one action-conditioned feature row per region.
            rows[index] = np.asarray([1.0, np.log(max(state.error_share[index], 1.0e-12)), np.log(max(state.element_share[index], 1.0e-12)), np.log(max(state.sizes[index] / state.h0, 1.0e-6)), state.vm_max[index] / stress_scale, state.volume[index] / volume_scale, degree[index], np.log1p(state.hit_count[index]), state.dorfler_error_fraction[index], state.dorfler_element_fraction[index], float(action.extra_depth[index]), neighbor_error[index], neighbor_depth[index], semantic_persistence(name), float(state.step) / 10.0], dtype=float)  # Combine physics, history, graph, and semantic persistence features.
        return rows  # Return features in stable region order.
    def _prior(self, state: WorldState, action: RegionAction) -> WorldPrediction:  # Predict one transition from explicit refinement physics only.
        action.validate(state)  # Validate the safe discrete action domain.
        factor = float(self.config.refine_factor)  # Read the common refinement factor.
        depths = 1.0 + np.asarray(action.extra_depth, dtype=float)  # Include the mandatory current Dörfler hit in every effective depth.
        marked_error = np.clip(state.dorfler_error_fraction, 0.0, 1.0)  # Bound measured marked-error fractions.
        marked_elements = np.clip(state.dorfler_element_fraction, 0.0, 1.0)  # Bound measured marked-element fractions.
        persistence = np.asarray([semantic_persistence(name) for name in state.names], dtype=float)  # Convert semantic region names to persistence priors.
        effective_error_depth = 1.0 + persistence * np.asarray(action.extra_depth, dtype=float)  # Discount speculative future hits by mechanism persistence.
        refined_error_ratio = np.power(factor, self.config.error_power * effective_error_depth)  # Predict error decay on the selected regional core.
        regional_error_ratio = 1.0 - marked_error * (1.0 - refined_error_ratio)  # Leave unmarked regional error unchanged under the prior.
        refined_resource_ratio = np.power(factor, -float(state.dim) * depths)  # Predict tetrahedral resource growth on refined elements.
        regional_resource_ratio = 1.0 + marked_elements * (refined_resource_ratio - 1.0)  # Leave unmarked regional resources unchanged under the prior.
        spill = state.adjacency @ (marked_elements * np.asarray(action.extra_depth, dtype=float))  # Predict gradation-induced neighboring refinement.
        regional_resource_ratio *= 1.0 + self.config.neighbor_spill * np.clip(spill, 0.0, 3.0)  # Add conservative adjacent-region resource spill.
        next_error = np.maximum(state.err_sum * regional_error_ratio, 1.0e-30)  # Construct positive regional error predictions.
        next_elements = np.maximum(state.elems * regional_resource_ratio, 1.0)  # Construct positive regional element predictions.
        size_exponent = marked_elements * depths + 0.25 * spill  # Approximate measured regional size reduction after remeshing.
        next_sizes = np.maximum(state.h_min, state.sizes * np.power(factor, size_exponent))  # Respect the minimum target size.
        element_delta = float(np.sum(next_elements) - np.sum(state.elems))  # Compute the predicted element-count increment.
        next_equations = int(max(1.0, round(state.n_equations + state.eq_per_elem * element_delta)))  # Convert resource growth to active equations.
        next_marked_error = np.clip(0.62 * marked_error + 0.18 * state.error_share, 0.0, 1.0)  # Propagate a decaying persistent-hotspot proxy for internal rollouts.
        next_marked_elements = np.clip(0.62 * marked_elements + 0.08 * state.element_share, 0.0, 1.0)  # Propagate a decaying marked-support proxy for internal rollouts.
        next_state = state.replaced(err_sum=next_error, elems=next_elements, sizes=next_sizes, n_equations=next_equations, dorfler_error_fraction=next_marked_error, dorfler_element_fraction=next_marked_elements)  # Build the physics-prior successor state.
        error_ratio = next_state.total_error / max(state.total_error, 1.0e-30)  # Compute the predicted global error ratio.
        equation_ratio = next_state.n_equations / max(state.n_equations, 1)  # Compute the predicted global equation ratio.
        uncertainty = float(self.config.prior_uncertainty + 0.05 * np.mean(np.asarray(action.extra_depth, dtype=float)))  # Increase prior uncertainty for deeper speculative actions.
        failure = float(np.clip(0.08 + 0.9 * max(0.0, uncertainty - 0.20), 0.0, 1.0))  # Map prior uncertainty to a bounded failure proxy.
        upper_error = float(error_ratio * np.exp(self.config.uncertainty_scale * uncertainty))  # Form a conservative upper error ratio.
        upper_equations = float(equation_ratio * np.exp(0.65 * self.config.uncertainty_scale * uncertainty))  # Form a conservative upper resource ratio.
        return WorldPrediction(next_state=next_state, uncertainty=uncertainty, failure_risk=failure, error_ratio_mean=float(error_ratio), error_ratio_upper=upper_error, equation_ratio_mean=float(equation_ratio), equation_ratio_upper=upper_equations)  # Return the complete prior prediction.
    def _ensemble(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # Predict learned log residuals and epistemic spread.
        x_train = np.asarray(self._x, dtype=float)  # Materialize the online feature matrix.
        y_train = np.asarray(self._y, dtype=float)  # Materialize the online residual matrix.
        if x_train.shape[0] < self.config.min_rows:  # Keep the model prior-only until sufficient regional transitions exist.
            return np.zeros((features.shape[0], 2), dtype=float), np.full((features.shape[0], 2), self.config.prior_uncertainty, dtype=float)  # Return zero correction and conservative uncertainty.
        mean = np.mean(x_train, axis=0)  # Compute online feature centering.
        scale = np.std(x_train, axis=0)  # Compute online feature scaling.
        scale[scale < 1.0e-8] = 1.0  # Avoid amplifying constant features.
        mean[0] = 0.0  # Preserve the existing leading-one feature as an intercept without changing stored raw rows.
        scale[0] = 1.0  # Keep the intercept equal to one in both training and query features.
        x_std = (x_train - mean) / scale  # Standardize training features.
        q_std = (features - mean) / scale  # Standardize queried features with the same transform.
        rng = np.random.default_rng(self.seed + self.transition_count)  # Build a reproducible bootstrap generator.
        members: list[np.ndarray] = []  # Collect one prediction matrix per ensemble member.
        identity = np.eye(x_std.shape[1], dtype=float)  # Construct the ridge regularizer matrix.
        identity[0, 0] = 0.0  # Leave systematic log-residual bias unpenalized while retaining ridge on feature responses.
        for _ in range(self.config.ensemble_size):  # Fit several bootstrap ridge regressors.
            indices = rng.integers(0, x_std.shape[0], size=x_std.shape[0])  # Sample regional transitions with replacement.
            xb = x_std[indices]  # Select bootstrap feature rows.
            yb = y_train[indices]  # Select matching residual targets.
            matrix = xb.T @ xb + self.config.ridge * identity  # Form the regularized normal matrix.
            weights = np.linalg.solve(matrix, xb.T @ yb)  # Solve both residual channels in one linear system.
            members.append(q_std @ weights)  # Predict queried residuals for this bootstrap member.
        stacked = np.stack(members, axis=0)  # Stack ensemble predictions for moment estimation.
        predicted = np.clip(np.mean(stacked, axis=0), -self.config.max_log_residual, self.config.max_log_residual)  # Bound the mean learned correction.
        spread = np.maximum(np.std(stacked, axis=0), 0.02)  # Retain a non-zero uncertainty floor.
        return predicted, spread  # Return learned corrections and epistemic spread.
    def predict(self, state: WorldState, action: RegionAction) -> WorldPrediction:  # Predict an action-conditioned successor with physics plus learned residuals.
        prior = self._prior(state, action)  # Compute the explicit Dörfler and remeshing prior.
        features = self._features(state, action)  # Build the matching regional feature rows.
        residual, spread = self._ensemble(features)  # Predict online residual corrections.
        next_error = np.maximum(prior.next_state.err_sum * np.exp(residual[:, 0]), 1.0e-30)  # Correct regional error transitions in log space.
        next_elements = np.maximum(prior.next_state.elems * np.exp(residual[:, 1]), 1.0)  # Correct regional resource transitions in log space.
        element_delta = float(np.sum(next_elements) - np.sum(state.elems))  # Recompute the corrected resource increment.
        next_equations = int(max(1.0, round(state.n_equations + state.eq_per_elem * element_delta)))  # Convert corrected elements to active equations.
        next_state = state.replaced(err_sum=next_error, elems=next_elements, sizes=prior.next_state.sizes, n_equations=next_equations, dorfler_error_fraction=prior.next_state.dorfler_error_fraction, dorfler_element_fraction=prior.next_state.dorfler_element_fraction)  # Build the corrected successor state.
        uncertainty = float(np.mean(np.sqrt(np.sum(np.square(spread), axis=1))))  # Aggregate two-channel ensemble spread into one transition uncertainty.
        error_ratio = next_state.total_error / max(state.total_error, 1.0e-30)  # Compute the corrected global error ratio.
        equation_ratio = next_state.n_equations / max(state.n_equations, 1)  # Compute the corrected global equation ratio.
        upper_error = float(error_ratio * np.exp(self.config.uncertainty_scale * uncertainty))  # Form a conservative global error bound.
        upper_equations = float(equation_ratio * np.exp(0.65 * self.config.uncertainty_scale * uncertainty))  # Form a conservative resource bound.
        failure = float(np.clip(0.06 + 1.2 * max(0.0, uncertainty - 0.08) + 0.08 * max(action.extra_depth, default=0), 0.0, 1.0))  # Estimate action failure risk from epistemic uncertainty and depth.
        return WorldPrediction(next_state=next_state, uncertainty=uncertainty, failure_risk=failure, error_ratio_mean=float(error_ratio), error_ratio_upper=upper_error, equation_ratio_mean=float(equation_ratio), equation_ratio_upper=upper_equations)  # Return the calibrated transition prediction.
    def observe(self, previous: WorldState, action: RegionAction, observed: WorldState) -> None:  # Learn from one completed real-solve transition.
        if previous.names != observed.names:  # Require a stable semantic partition across the trajectory.
            raise ValueError("world-model observations require stable region names")  # Report an invalid transition pairing.
        prior = self._prior(previous, action)  # Reconstruct the prior used for the executed action.
        features = self._features(previous, action)  # Reconstruct action-conditioned feature rows.
        actual_error_ratio = np.log(np.maximum(observed.err_sum, 1.0e-30) / np.maximum(previous.err_sum, 1.0e-30))  # Measure actual regional log error transitions.
        prior_error_ratio = np.log(np.maximum(prior.next_state.err_sum, 1.0e-30) / np.maximum(previous.err_sum, 1.0e-30))  # Measure prior regional log error transitions.
        actual_resource_ratio = np.log(np.maximum(observed.elems, 1.0) / np.maximum(previous.elems, 1.0))  # Measure actual regional log resource transitions.
        prior_resource_ratio = np.log(np.maximum(prior.next_state.elems, 1.0) / np.maximum(previous.elems, 1.0))  # Measure prior regional log resource transitions.
        targets = np.column_stack((actual_error_ratio - prior_error_ratio, actual_resource_ratio - prior_resource_ratio))  # Define residual-learning targets.
        self._x.extend(features.tolist())  # Append regional feature rows to online memory.
        self._y.extend(np.clip(targets, -self.config.max_log_residual, self.config.max_log_residual).tolist())  # Append bounded residual targets.
        if len(self._x) > self.config.max_rows:  # Enforce bounded online memory.
            self._x = self._x[-self.config.max_rows :]  # Retain the most recent transition features.
            self._y = self._y[-self.config.max_rows :]  # Retain the matching recent residual targets.
        self.transition_count += 1  # Count the completed real transition after all rows are stored.
    def snapshot(self) -> dict[str, Any]:  # Return a JSON-compatible model snapshot.
        return {"config": asdict(self.config), "seed": self.seed, "transition_count": self.transition_count, "x": self._x, "y": self._y}  # Preserve configuration, data, and learning progress.
    def save(self, path: str | Path) -> None:  # Persist the online world model deterministically.
        target = Path(path)  # Normalize the output path.
        target.parent.mkdir(parents=True, exist_ok=True)  # Create the snapshot directory when needed.
        target.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")  # Write a human-auditable JSON snapshot.
    @classmethod  # Restore snapshots through the class constructor.
    def load(cls, path: str | Path) -> "ResidualWorldModel":  # Restore a previously saved online world model.
        payload = json.loads(Path(path).read_text(encoding="utf-8"))  # Read and parse the snapshot.
        model = cls(WorldModelConfig(**payload["config"]), seed=int(payload["seed"]))  # Reconstruct the configured model.
        model.transition_count = int(payload.get("transition_count", 0))  # Restore completed-transition count.
        model._x = [[float(value) for value in row] for row in payload.get("x", [])]  # Restore numerical feature rows.
        model._y = [[float(value) for value in row] for row in payload.get("y", [])]  # Restore numerical residual targets.
        return model  # Return the restored online model.
