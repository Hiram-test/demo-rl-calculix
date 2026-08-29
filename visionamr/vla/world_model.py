"""Action-conditioned regional world model for adaptive finite-element planning."""  # Describe the module purpose.
from __future__ import annotations  # Enable postponed evaluation of type annotations.
from dataclasses import asdict, dataclass, replace  # Import compact immutable data contracts.
import json  # Import JSON persistence for the transition library.
from pathlib import Path  # Import path handling for model snapshots.
import numpy as np  # Import numerical arrays and linear algebra.

@dataclass(frozen=True)  # Make every observed world state immutable at the API boundary.
class WorldState:  # Store the decision-relevant regional state after one real solve.
    names: tuple[str, ...]  # Keep a stable name for every semantic region.
    err_sum: np.ndarray  # Store the current regional sums of eta squared.
    elems: np.ndarray  # Store the current regional element counts.
    sizes: np.ndarray  # Store the current measured regional mesh sizes.
    vm_max: np.ndarray  # Store the current regional peak von Mises stresses.
    volume: np.ndarray  # Store the current regional geometric measures.
    adjacency: np.ndarray  # Store the symmetric regional adjacency matrix.
    dorfler_error_fraction: np.ndarray  # Store the exact or proxy marked error fraction in each region.
    dorfler_element_fraction: np.ndarray  # Store the exact or proxy marked element fraction in each region.
    hit_count: np.ndarray  # Store how often each region has entered the Dörfler set.
    n_equations: float  # Store the current global equation count.
    eq_per_elem: float  # Store the measured equation-to-element ratio.
    h_min: float  # Store the admissible minimum target size.
    h0: float  # Store the admissible maximum target size.
    dim: int  # Store the physical mesh dimension.
    step: int = 0  # Store the completed real-solve index.
    def __post_init__(self) -> None:  # Validate and normalize every numerical field.
        count = len(self.names)  # Determine the required regional vector length.
        vector_fields = ("err_sum", "elems", "sizes", "vm_max", "volume", "dorfler_error_fraction", "dorfler_element_fraction", "hit_count")  # Enumerate all regional vectors.
        for field_name in vector_fields:  # Normalize every vector in one guarded loop.
            values = np.asarray(getattr(self, field_name), dtype=float).reshape(-1).copy()  # Convert the field to a private one-dimensional float array.
            if len(values) != count:  # Reject a state whose regions cannot be aligned.
                raise ValueError(f"{field_name} has {len(values)} entries for {count} regions")  # Explain the structural mismatch.
            if not np.all(np.isfinite(values)):  # Reject NaN or infinite observations before planning.
                raise ValueError(f"{field_name} contains non-finite values")  # Report the invalid field.
            object.__setattr__(self, field_name, values)  # Store the normalized immutable-boundary value.
        graph = np.asarray(self.adjacency, dtype=float).copy()  # Convert the graph to a numerical matrix.
        if graph.shape != (count, count):  # Require one graph row and column per region.
            raise ValueError(f"adjacency has shape {graph.shape}, expected {(count, count)}")  # Report the graph mismatch.
        if not np.all(np.isfinite(graph)):  # Reject invalid graph weights.
            raise ValueError("adjacency contains non-finite values")  # Report the invalid graph.
        graph = np.maximum(graph, graph.T)  # Enforce undirected connectivity for message aggregation.
        np.fill_diagonal(graph, 0.0)  # Remove self-edges from neighbour statistics.
        object.__setattr__(self, "adjacency", graph)  # Store the normalized graph.
        if self.dim not in (2, 3):  # Restrict the model to the repository simplex dimensions.
            raise ValueError("dim must be 2 or 3")  # Report an unsupported physical dimension.
        if self.h_min <= 0.0 or self.h0 < self.h_min:  # Check the target-size interval.
            raise ValueError("invalid mesh-size bounds")  # Reject inconsistent size limits.
        if self.eq_per_elem <= 0.0 or self.n_equations <= 0.0:  # Check the resource observations.
            raise ValueError("resource observations must be positive")  # Reject invalid resource fields.
    @property  # Expose the number of semantic regions without copying arrays.
    def n_regions(self) -> int:  # Return the regional state dimension.
        return len(self.names)  # Use the stable name tuple as the source of truth.
    @property  # Expose the current total estimator mass.
    def total_error(self) -> float:  # Return the sum of regional eta squared.
        return float(np.sum(self.err_sum))  # Sum the non-negative regional values.
    @property  # Expose the current total element count.
    def total_elems(self) -> float:  # Return the sum of regional element counts.
        return float(np.sum(self.elems))  # Sum the regional resource counts.
    @property  # Expose normalized regional estimator shares.
    def error_share(self) -> np.ndarray:  # Return a robust error-share vector.
        return self.err_sum / max(self.total_error, 1.0e-30)  # Normalize with a positive numerical floor.
    @property  # Expose normalized regional element shares.
    def element_share(self) -> np.ndarray:  # Return a robust element-share vector.
        return self.elems / max(self.total_elems, 1.0)  # Normalize with a positive resource floor.
    def with_dorfler(self, error_fraction: np.ndarray, element_fraction: np.ndarray) -> "WorldState":  # Replace the current Dörfler action descriptors.
        return replace(self, dorfler_error_fraction=np.asarray(error_fraction, dtype=float), dorfler_element_fraction=np.asarray(element_fraction, dtype=float))  # Return a new validated state.

@dataclass(frozen=True)  # Make a proposed regional macro-action immutable.
class RegionAction:  # Represent extra future Dörfler hits delegated at the current solve.
    extra_depth: tuple[int, ...]  # Store one non-negative integer advance depth per region.
    source: str = "world_model"  # Record whether the action came from planning or fallback.
    def array(self, count: int | None = None) -> np.ndarray:  # Convert the tuple to a checked numerical vector.
        values = np.asarray(self.extra_depth, dtype=int).reshape(-1)  # Build the integer action vector.
        if count is not None and len(values) != count:  # Check alignment with a supplied state.
            raise ValueError(f"action has {len(values)} entries for {count} regions")  # Report the alignment failure.
        if np.any(values < 0):  # Forbid coarsening relative to the Dörfler safety action.
            raise ValueError("extra_depth must be non-negative")  # Explain the monotonic safety rule.
        return values  # Return the validated action vector.

@dataclass(frozen=True)  # Make each prediction an auditable value object.
class WorldPrediction:  # Store the predicted next state and epistemic uncertainty.
    state: WorldState  # Store the ensemble-mean next regional state.
    log_error_std: float  # Store weighted uncertainty in the log error transition.
    log_resource_std: float  # Store weighted uncertainty in the log resource transition.
    failure_probability: float  # Store a conservative model-risk indicator.
    member_total_errors: tuple[float, ...]  # Store each ensemble member's total estimator mass.
    member_equations: tuple[float, ...]  # Store each ensemble member's equation prediction.

@dataclass(frozen=True)  # Keep world-model hyperparameters explicit and serializable.
class WorldModelConfig:  # Configure the physics prior and residual ensemble.
    refine_factor: float = 0.5  # Match the repository Dörfler refinement atom.
    error_power: float = 2.0  # Set the prior eta-squared reduction exponent.
    ridge: float = 1.0e-3  # Regularize each residual linear model.
    ensemble_size: int = 5  # Use several bootstrap members for uncertainty.
    min_rows_for_learning: int = 24  # Require enough regional transitions before trusting learned residuals.
    prior_spread: float = 0.22  # Quantify uncertainty before sufficient real transitions exist.
    max_log_error_change: float = 4.0  # Bound extrapolated error changes.
    max_log_resource_change: float = 5.0  # Bound extrapolated resource changes.
    random_seed: int = 73  # Make bootstrap models deterministic and reproducible.

def semantic_persistence(name: str) -> float:  # Encode a bounded mechanism prior without producing mesh parameters.
    lowered = name.lower()  # Normalize the region name for token matching.
    strong_tokens = ("opening", "rim", "wheel", "patch", "diaphragm", "web", "bearing", "support", "clamp", "corner", "edge", "hot")  # List mechanisms that commonly persist over several refinements.
    matched = sum(token in lowered for token in strong_tokens)  # Count independent semantic persistence cues.
    return float(np.clip(0.12 * matched, 0.0, 0.72))  # Return a weak bounded prior that cannot dominate physics.

def _neighbour_average(graph: np.ndarray, values: np.ndarray) -> np.ndarray:  # Aggregate neighbouring regional features.
    degree = np.sum(graph, axis=1)  # Compute each regional graph degree.
    return graph @ values / np.maximum(degree, 1.0)  # Average connected values with an isolated-node guard.

class ResidualWorldModel:  # Learn corrections to a conservative action-conditioned physics prior.
    def __init__(self, config: WorldModelConfig | None = None) -> None:  # Initialize an empty transition model.
        self.config = config or WorldModelConfig()  # Use explicit caller settings or safe defaults.
        self._x_rows: list[list[float]] = []  # Store standardized regional transition features.
        self._y_rows: list[list[float]] = []  # Store observed residual corrections for error and resource.
    @property  # Expose the amount of real training evidence.
    def sample_count(self) -> int:  # Return the number of stored regional transition rows.
        return len(self._x_rows)  # Count the append-only feature rows.
    def _features(self, state: WorldState, action: RegionAction) -> np.ndarray:  # Build fixed-width regional model inputs.
        depth = action.array(state.n_regions).astype(float)  # Read the delegated future-hit depths.
        err_share = np.maximum(state.error_share, 1.0e-12)  # Stabilize logarithmic estimator shares.
        elem_share = np.maximum(state.element_share, 1.0e-12)  # Stabilize logarithmic element shares.
        density = state.err_sum / np.maximum(state.elems, 1.0)  # Compute estimator mass per element.
        density = density / max(float(np.sum(state.err_sum) / max(np.sum(state.elems), 1.0)), 1.0e-30)  # Normalize density by the global mean.
        vm_scale = max(float(np.max(state.vm_max)), 1.0e-12)  # Compute a robust stress scale.
        volume_share = state.volume / max(float(np.sum(state.volume)), 1.0e-30)  # Normalize regional geometric measure.
        degree = np.sum(state.adjacency, axis=1)  # Compute graph degree for every region.
        degree = degree / max(float(np.max(degree)), 1.0)  # Normalize graph degree.
        neighbour_error = _neighbour_average(state.adjacency, err_share)  # Aggregate adjacent estimator shares.
        neighbour_depth = _neighbour_average(state.adjacency, depth)  # Aggregate adjacent delegated depths.
        semantic = np.asarray([semantic_persistence(name) for name in state.names], dtype=float)  # Compute weak mechanism priors.
        columns = [np.ones(state.n_regions), np.log(err_share), np.log(np.maximum(density, 1.0e-12)), np.log(elem_share), np.log(np.clip(state.sizes / state.h0, 1.0e-6, 1.0)), np.log(np.maximum(state.vm_max / vm_scale, 1.0e-12)), np.log(np.maximum(volume_share, 1.0e-12)), degree, np.log1p(state.hit_count), state.dorfler_error_fraction, state.dorfler_element_fraction, depth, neighbour_error, neighbour_depth, semantic, np.full(state.n_regions, min(state.step / 10.0, 2.0))]  # Assemble all decision-relevant regional descriptors.
        return np.column_stack(columns)  # Return one feature row per semantic region.
    def _prior_deltas(self, state: WorldState, action: RegionAction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # Predict a conservative transition before learning residuals.
        depth = action.array(state.n_regions).astype(float)  # Read the non-negative delegated depths.
        semantic = np.asarray([semantic_persistence(name) for name in state.names], dtype=float)  # Compute bounded persistence cues.
        exponent = np.clip(self.config.error_power * (1.0 + 0.15 * semantic + 0.05 * np.log1p(state.hit_count)), 1.0, 3.4)  # Adapt the prior convergence rate weakly by mechanism persistence.
        base_error_multiplier = 1.0 - state.dorfler_error_fraction * (1.0 - self.config.refine_factor ** exponent)  # Model exact Dörfler as partial regional refinement.
        base_resource_multiplier = 1.0 + state.dorfler_element_fraction * (self.config.refine_factor ** (-state.dim) - 1.0)  # Model exact Dörfler's local element growth.
        selected = depth > 0.0  # Identify regions receiving advance investment.
        delegated_levels = depth  # Interpret one level as extending the current Dörfler atom over the selected region.
        full_error_multiplier = self.config.refine_factor ** (exponent * delegated_levels)  # Predict full-region error reduction for delegated regions.
        full_resource_multiplier = self.config.refine_factor ** (-state.dim * delegated_levels)  # Predict full-region element growth for delegated regions.
        error_multiplier = np.where(selected, np.minimum(base_error_multiplier, full_error_multiplier), base_error_multiplier)  # Preserve at least the Dörfler error reduction.
        resource_multiplier = np.where(selected, np.maximum(base_resource_multiplier, full_resource_multiplier), base_resource_multiplier)  # Account for the extra mesh investment.
        spill = _neighbour_average(state.adjacency, depth)  # Estimate gradation-induced resource spill to neighbours.
        resource_multiplier = resource_multiplier * np.exp(0.12 * spill)  # Increase resource use conservatively near deep actions.
        error_delta = np.log(np.clip(error_multiplier, 1.0e-8, np.exp(self.config.max_log_error_change)))  # Convert error multipliers to bounded log changes.
        resource_delta = np.log(np.clip(resource_multiplier, np.exp(-self.config.max_log_resource_change), np.exp(self.config.max_log_resource_change)))  # Convert resource multipliers to bounded log changes.
        effective_levels = np.where(selected, delegated_levels, state.dorfler_element_fraction)  # Estimate the mean regional size-depth change.
        return error_delta, resource_delta, effective_levels  # Return all physics-prior transition components.
    def observe(self, previous: WorldState, action: RegionAction, observed: WorldState) -> None:  # Add one real action-conditioned transition to the residual library.
        if previous.names != observed.names:  # Require a stable semantic partition across the transition.
            raise ValueError("world-model transitions require identical region names")  # Reject ambiguous region alignment.
        features = self._features(previous, action)  # Recreate the pre-action feature rows.
        prior_error, prior_resource, _ = self._prior_deltas(previous, action)  # Evaluate the physics-prior transition.
        actual_error = np.log(np.maximum(observed.err_sum, 1.0e-30) / np.maximum(previous.err_sum, 1.0e-30))  # Measure the actual regional log error change.
        actual_resource = np.log(np.maximum(observed.elems, 1.0) / np.maximum(previous.elems, 1.0))  # Measure the actual regional log element change.
        targets = np.column_stack([actual_error - prior_error, actual_resource - prior_resource])  # Isolate the residual corrections to learn.
        for row, target in zip(features, targets):  # Append each regional transition independently.
            if np.all(np.isfinite(row)) and np.all(np.isfinite(target)):  # Retain only numerically valid evidence.
                self._x_rows.append(row.astype(float).tolist())  # Store a JSON-safe feature row.
                self._y_rows.append(target.astype(float).tolist())  # Store the paired JSON-safe residual target.
    def _ensemble_coefficients(self, width: int) -> list[np.ndarray]:  # Fit deterministic bootstrap ridge members.
        if self.sample_count < self.config.min_rows_for_learning:  # Keep the physics prior dominant before enough evidence exists.
            return []  # Signal the prior-only uncertainty path.
        x_values = np.asarray(self._x_rows, dtype=float)  # Assemble the transition design matrix.
        y_values = np.asarray(self._y_rows, dtype=float)  # Assemble the two residual targets.
        if x_values.shape[1] != width:  # Reject stale snapshots with another feature schema.
            raise ValueError("stored transition feature width is incompatible with this model")  # Report the schema mismatch.
        coefficients: list[np.ndarray] = []  # Allocate the ensemble coefficient list.
        identity = np.eye(width, dtype=float)  # Build the ridge regularization matrix.
        for member in range(self.config.ensemble_size):  # Fit each bootstrap member independently.
            rng = np.random.default_rng(self.config.random_seed + 7919 * member + self.sample_count)  # Derive a deterministic member seed.
            indices = rng.integers(0, len(x_values), size=len(x_values))  # Resample transition rows with replacement.
            x_boot = x_values[indices]  # Select the bootstrap feature matrix.
            y_boot = y_values[indices]  # Select the paired bootstrap targets.
            lhs = x_boot.T @ x_boot + self.config.ridge * identity  # Form the regularized normal matrix.
            rhs = x_boot.T @ y_boot  # Form the two-target normal-equation right side.
            coefficients.append(np.linalg.solve(lhs, rhs))  # Solve and store the member coefficients.
        return coefficients  # Return all fitted ensemble members.
    def predict(self, state: WorldState, action: RegionAction) -> WorldPrediction:  # Roll the regional world forward by one adaptive action.
        features = self._features(state, action)  # Build the action-conditioned model input.
        prior_error, prior_resource, effective_levels = self._prior_deltas(state, action)  # Evaluate the conservative physics prior.
        coefficients = self._ensemble_coefficients(features.shape[1])  # Fit or retrieve the residual ensemble.
        member_error_deltas: list[np.ndarray] = []  # Allocate per-member error transitions.
        member_resource_deltas: list[np.ndarray] = []  # Allocate per-member resource transitions.
        if coefficients:  # Use learned residual corrections when sufficient real evidence exists.
            for beta in coefficients:  # Evaluate every bootstrap member.
                member_error_deltas.append(prior_error + features @ beta[:, 0])  # Correct the regional error transition.
                member_resource_deltas.append(prior_resource + features @ beta[:, 1])  # Correct the regional resource transition.
        else:  # Quantify prior uncertainty explicitly before learning is supported.
            offsets = np.linspace(-self.config.prior_spread, self.config.prior_spread, self.config.ensemble_size)  # Create symmetric deterministic prior perturbations.
            for offset in offsets:  # Build each prior-only ensemble member.
                member_error_deltas.append(prior_error * (1.0 + offset))  # Perturb the prior error reduction magnitude.
                member_resource_deltas.append(prior_resource * (1.0 - 0.5 * offset))  # Perturb the prior resource growth in the opposite direction.
        error_stack = np.vstack([np.clip(values, -self.config.max_log_error_change, 1.0) for values in member_error_deltas])  # Bound all member error extrapolations.
        resource_stack = np.vstack([np.clip(values, -1.0, self.config.max_log_resource_change) for values in member_resource_deltas])  # Bound all member resource extrapolations.
        mean_error_delta = np.mean(error_stack, axis=0)  # Compute the ensemble-mean regional error transition.
        mean_resource_delta = np.mean(resource_stack, axis=0)  # Compute the ensemble-mean regional resource transition.
        next_error = np.maximum(state.err_sum * np.exp(mean_error_delta), 1.0e-30)  # Predict positive regional estimator masses.
        next_elems = np.maximum(state.elems * np.exp(mean_resource_delta), 1.0)  # Predict positive regional element counts.
        next_sizes = np.clip(state.sizes * self.config.refine_factor ** effective_levels, state.h_min, state.h0)  # Predict mean regional mesh sizes.
        next_vm = np.maximum(state.vm_max * np.exp(-0.08 * mean_error_delta), 0.0)  # Carry stress peaks forward with a weak resolution correction.
        next_hits = state.hit_count + (state.dorfler_error_fraction > 1.0e-9).astype(float)  # Update persistent-hotspot counts.
        next_equations = float(np.sum(next_elems) * state.eq_per_elem)  # Convert predicted elements to the shared equation budget.
        next_state = WorldState(names=state.names, err_sum=next_error, elems=next_elems, sizes=next_sizes, vm_max=next_vm, volume=state.volume, adjacency=state.adjacency, dorfler_error_fraction=np.zeros(state.n_regions), dorfler_element_fraction=np.zeros(state.n_regions), hit_count=next_hits, n_equations=max(next_equations, 1.0), eq_per_elem=state.eq_per_elem, h_min=state.h_min, h0=state.h0, dim=state.dim, step=state.step + 1)  # Assemble the predicted next decision state.
        error_std_by_region = np.std(error_stack, axis=0)  # Compute regional epistemic error uncertainty.
        resource_std_by_region = np.std(resource_stack, axis=0)  # Compute regional epistemic resource uncertainty.
        log_error_std = float(np.sqrt(np.sum(state.error_share * error_std_by_region ** 2)))  # Aggregate error uncertainty by current estimator importance.
        log_resource_std = float(np.sqrt(np.sum(state.element_share * resource_std_by_region ** 2)))  # Aggregate resource uncertainty by current mesh importance.
        member_total_errors = tuple(float(np.sum(state.err_sum * np.exp(member_delta))) for member_delta in error_stack)  # Retain total-error predictions for audit.
        member_equations = tuple(float(np.sum(state.elems * np.exp(member_delta)) * state.eq_per_elem) for member_delta in resource_stack)  # Retain equation predictions for audit.
        action_scale = float(np.mean(action.array(state.n_regions)))  # Measure how far the action extrapolates beyond standard Dörfler.
        raw_risk = 2.2 * log_error_std + 1.7 * log_resource_std + 0.18 * action_scale - 1.7  # Combine uncertainty and action depth into a conservative risk logit.
        failure_probability = float(1.0 / (1.0 + np.exp(-np.clip(raw_risk, -20.0, 20.0))))  # Map the risk logit into a bounded probability-like score.
        return WorldPrediction(state=next_state, log_error_std=log_error_std, log_resource_std=log_resource_std, failure_probability=failure_probability, member_total_errors=member_total_errors, member_equations=member_equations)  # Return the complete auditable prediction.
    def to_dict(self) -> dict:  # Serialize the transition model without pickled executable state.
        return {"schema": "visionamr-regional-world-model-v1", "config": asdict(self.config), "x_rows": self._x_rows, "y_rows": self._y_rows}  # Return a JSON-safe snapshot.
    @classmethod  # Construct a model from a validated JSON snapshot.
    def from_dict(cls, payload: dict) -> "ResidualWorldModel":  # Restore a transition library.
        if payload.get("schema") != "visionamr-regional-world-model-v1":  # Check the explicit persistence schema.
            raise ValueError("unsupported world-model snapshot schema")  # Reject silent schema drift.
        model = cls(WorldModelConfig(**payload.get("config", {})))  # Restore the configured physics prior.
        model._x_rows = [[float(value) for value in row] for row in payload.get("x_rows", [])]  # Restore numerical feature rows.
        model._y_rows = [[float(value) for value in row] for row in payload.get("y_rows", [])]  # Restore numerical residual targets.
        if len(model._x_rows) != len(model._y_rows):  # Require complete transition pairs.
            raise ValueError("world-model snapshot has unmatched feature and target rows")  # Reject a corrupt snapshot.
        return model  # Return the restored model.
    def save(self, path: Path) -> Path:  # Persist the transition library atomically enough for experiment use.
        target = Path(path)  # Normalize the destination path.
        target.parent.mkdir(parents=True, exist_ok=True)  # Create the destination directory.
        target.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")  # Write a human-auditable JSON snapshot.
        return target  # Return the written path.
    @classmethod  # Restore a model directly from a filesystem path.
    def load(cls, path: Path) -> "ResidualWorldModel":  # Read and validate a persisted transition library.
        payload = json.loads(Path(path).read_text(encoding="utf-8"))  # Parse the JSON snapshot.
        return cls.from_dict(payload)  # Delegate validation to the schema-aware constructor.
