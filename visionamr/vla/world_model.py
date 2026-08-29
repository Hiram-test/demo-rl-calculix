from __future__ import annotations  # Enable compact type references across the world-model modules.
import hashlib  # Create stable identifiers for imagined states.
from dataclasses import dataclass  # Define explicit prediction records.
import numpy as np  # Implement the small online ensemble without a heavyweight ML runtime.
from .tool_gateway import ToolPreview  # Consume only deterministic tool-owned action parameters.
from .world_state import Transition, WorldState  # Learn from real action-conditioned transitions.
@dataclass(frozen=True)  # Keep every rollout prediction immutable and auditable.
class WorldPrediction:  # Predict decision-relevant consequences rather than the full stress field.
    action_id: str  # Identify the evaluated high-level action.
    grades: np.ndarray  # Store the tool-resolved next grades.
    sizes: np.ndarray  # Store the tool-resolved next sizes.
    err_sum: np.ndarray  # Predict next regional estimator mass.
    elems: np.ndarray  # Predict next regional resource allocation.
    n_equations: float  # Predict next total equations.
    e_energy: float  # Predict next energy-norm error.
    e_qoi: float  # Predict next QoI error.
    uncertainty: float  # Quantify ensemble and structural-prior uncertainty.
    prior_total_eta2: float  # Record the physics-prior global estimator.
    residual_total_eta2: float  # Record the calibrated global estimator.
    model_samples: int  # Record the amount of online transition evidence.
    details: dict  # Preserve decomposition for trace inspection.
    def imagined_state(self, state: WorldState) -> WorldState:  # Convert the consequence prediction into the next planning node.
        digest = hashlib.sha256()  # Build a stable imagined-state id.
        digest.update(state.state_id.encode("ascii"))  # Bind the node to its parent state.
        digest.update(self.action_id.encode("utf-8"))  # Bind the node to the selected action.
        digest.update(np.ascontiguousarray(np.round(self.err_sum, 12)).tobytes())  # Bind the node to predicted physics.
        return WorldState(  # Construct a rollout-compatible state.
            step=int(state.step + 1),  # Advance one imagined transition.
            names=state.names,  # Keep the current region graph order.
            origins=state.origins,  # Keep region provenance.
            grades=np.asarray(self.grades, dtype=int).copy(),  # Carry discrete action state.
            sizes=np.asarray(self.sizes, dtype=float).copy(),  # Carry tool-owned sizes.
            err_sum=np.asarray(self.err_sum, dtype=float).copy(),  # Carry predicted regional estimator mass.
            elems=np.asarray(self.elems, dtype=float).copy(),  # Carry predicted resource allocation.
            vm_max=np.asarray(state.vm_max, dtype=float).copy(),  # Hold stress scale constant inside short rollouts.
            vm_mean=np.asarray(state.vm_mean, dtype=float).copy(),  # Hold mean stress scale constant inside short rollouts.
            volume=np.asarray(state.volume, dtype=float).copy(),  # Preserve physical region volumes.
            adjacency=np.asarray(state.adjacency, dtype=float).copy(),  # Preserve the region graph.
            roles=np.asarray(state.roles, dtype=float).copy(),  # Preserve geometry-visible semantics.
            n_equations=int(round(self.n_equations)),  # Carry predicted resource count.
            budget=int(state.budget),  # Preserve the hard budget.
            e_energy=float(self.e_energy),  # Carry predicted energy error.
            e_qoi=float(self.e_qoi),  # Carry predicted QoI error.
            total_eta2=float(np.sum(self.err_sum)),  # Carry predicted global estimator mass.
            qoi=float(state.qoi),  # Avoid inventing a signed physical QoI in the compact model.
            U_total=float(state.U_total),  # Avoid inventing absolute strain energy in the compact model.
            state_id="imagined-" + digest.hexdigest()[:12],  # Mark the state as model-generated.
        )  # Finish the imagined state.
@dataclass(frozen=True)  # Keep world-model hyperparameters explicit.
class WorldModelConfig:  # Configure the structured prior and online ensemble.
    ensemble_size: int = 5  # Use several small ridge models for epistemic spread.
    ridge: float = 1.0e-2  # Stabilize fits from only a few real transitions.
    min_rows_for_fit: int = 6  # Require enough regional observations before learning corrections.
    prior_uncertainty: float = 0.18  # Express honest uncertainty before online evidence.
    topology_uncertainty: float = 0.10  # Add uncertainty around holes and openings.
    residual_clip: float = 1.5  # Prevent one noisy transition from dominating logarithmic corrections.
    seed: int = 314159  # Make bootstrap ensembles reproducible.
class OnlineRegionWorldModel:  # Learn action-conditioned regional transitions while retaining a physics prior.
    def __init__(self, problem, config: WorldModelConfig | None = None) -> None:  # Bind the model to one bridge component.
        self.problem = problem  # Store dimension and mesh-size limits.
        self.config = config or WorldModelConfig()  # Store deterministic model policy.
        self._x_region: list[np.ndarray] = []  # Accumulate regional residual features.
        self._y_region: list[float] = []  # Accumulate regional log-error residuals.
        self._x_resource: list[np.ndarray] = []  # Accumulate global resource residual features.
        self._y_resource: list[float] = []  # Accumulate global log-count residuals.
        self.transitions: list[Transition] = []  # Preserve all real transition evidence for auditing.
    @property  # Expose sample volume to planners and logs.
    def n_samples(self) -> int:  # Return the number of regional supervision rows.
        return len(self._y_region)  # Count rows rather than transitions because each region informs the model.
    def _semantic_floor(self, state: WorldState) -> np.ndarray:  # Allocate latent error to geometry-visible but probe-blind features.
        roles = state.roles  # Read the fixed semantic channels.
        raw = 0.80 * roles[:, 0] + 0.35 * roles[:, 1] + 1.30 * roles[:, 2] + 0.70 * roles[:, 3] + 0.05 * roles[:, 4] + 1.00 * roles[:, 5]  # Weight load, support, topology, singularity, field, and split evidence.
        if float(raw.sum()) <= 1.0e-12:  # Handle a partition with no recognized semantic labels.
            raw = np.ones(state.n_regions, dtype=float)  # Spread a weak prior uniformly.
        floor_share = 0.28 * raw / max(float(raw.sum()), 1.0e-30)  # Reserve at most twenty-eight percent of current estimator mass as latent structure.
        return float(state.total_eta2) * floor_share  # Return per-region latent estimator floors.
    def _convergence_order(self, state: WorldState) -> np.ndarray:  # Encode broad finite-element convergence expectations by region type.
        roles = state.roles  # Read semantic channels.
        q = 1.45 + 0.35 * roles[:, 0] + 0.18 * roles[:, 1] + 0.30 * roles[:, 2] + 0.42 * roles[:, 3] - 0.15 * roles[:, 4] + 0.35 * roles[:, 5]  # Assign stronger returns at localized visible features.
        return np.clip(q, 0.80, 2.80)  # Keep the prior conservative for linear tetrahedra near singularities.
    def _action_features(self, state: WorldState, sizes: np.ndarray) -> np.ndarray:  # Build one feature row per region for residual learning.
        ratio = np.maximum(np.asarray(sizes, dtype=float), 1.0e-12) / np.maximum(state.sizes, 1.0e-12)  # Measure the requested local size change.
        log_ratio = np.log(ratio)  # Express multiplicative mesh changes additively.
        degree = np.maximum(state.adjacency.sum(axis=1), 1.0)  # Normalize neighbor aggregation safely.
        neighbor_change = state.adjacency @ log_ratio / degree  # Encode gradation and cross-region coupling.
        node = state.node_features()  # Reuse scale-robust physical and semantic features.
        return np.column_stack([node, log_ratio, np.abs(log_ratio), neighbor_change, np.ones(state.n_regions)])  # Append action and intercept channels.
    def _global_features(self, state: WorldState, sizes: np.ndarray) -> np.ndarray:  # Build one compact resource-correction row.
        ratio = np.maximum(np.asarray(sizes, dtype=float), 1.0e-12) / np.maximum(state.sizes, 1.0e-12)  # Measure regional changes.
        log_ratio = np.log(ratio)  # Express changes additively.
        weighted = float(np.sum(state.elem_share * log_ratio))  # Summarize mean mesh scaling.
        spread = float(np.sqrt(np.sum(state.elem_share * (log_ratio - weighted) ** 2)))  # Summarize heterogeneity that drives Gmsh drift.
        boundary = float(np.sum(np.abs(state.adjacency @ log_ratio - state.adjacency.sum(axis=1) * log_ratio))) / max(float(state.n_regions), 1.0)  # Summarize inter-region jumps.
        topology = float(np.sum(state.err_share * (state.roles[:, 2] + state.roles[:, 5])))  # Summarize topology-sensitive mass.
        return np.array([1.0, weighted, spread, boundary, topology, state.budget_use], dtype=float)  # Return one fixed-length feature vector.
    def _prior(self, state: WorldState, preview: ToolPreview) -> tuple[np.ndarray, np.ndarray, float, float, dict]:  # Predict consequences from finite-element scaling and the region graph.
        ratio = np.maximum(preview.sizes, 1.0e-12) / np.maximum(state.sizes, 1.0e-12)  # Compute local action ratios.
        log_ratio = np.log(ratio)  # Convert ratios to additive changes.
        degree = np.maximum(state.adjacency.sum(axis=1), 1.0)  # Prepare safe graph averaging.
        neighbor_log = state.adjacency @ log_ratio / degree  # Model size-gradation spill across region boundaries.
        effective_log = 0.78 * log_ratio + 0.22 * neighbor_log  # Blend direct and neighboring action effects.
        latent_floor = self._semantic_floor(state)  # Add geometry-visible unresolved risk.
        anchor_error = np.maximum(state.err_sum, latent_floor)  # Prevent a coarse probe from declaring visible topology harmless.
        q = self._convergence_order(state)  # Retrieve conservative regional convergence exponents.
        err_prior = np.maximum(anchor_error * np.exp(q * effective_log), 1.0e-30)  # Predict next squared-estimator mass.
        elem_raw = np.maximum(state.elems, 1.0) * ratio ** (-float(self.problem.dim))  # Predict regional resource redistribution.
        elem_scale = max(float(preview.n_equations), 1.0) / max(float(state.n_equations), 1.0)  # Match the tool's calibrated global count.
        elem_target = max(float(state.elems.sum()) * elem_scale, 1.0)  # Convert equation scaling to a total element target.
        elems_prior = elem_raw * elem_target / max(float(elem_raw.sum()), 1.0)  # Preserve regional scaling while matching the global tool result.
        base_total = max(float(anchor_error.sum()), 1.0e-30)  # Define the no-action latent baseline.
        error_ratio = max(float(err_prior.sum()) / base_total, 1.0e-12)  # Measure predicted error change relative to the same latent baseline.
        e_energy = float(state.e_energy * np.sqrt(error_ratio))  # Map squared-estimator change to an energy-error change.
        focus = 0.55 * state.roles[:, 0] + 0.75 * state.roles[:, 2] + 0.35 * state.roles[:, 3] + 0.10  # Weight QoI-sensitive load and topology regions.
        focus = focus / max(float(focus.sum()), 1.0e-30)  # Normalize QoI focus.
        qoi_ratio = max(float(np.sum(focus * err_prior) / np.sum(focus * anchor_error)), 1.0e-12)  # Predict a focused estimator change.
        e_qoi = float(state.e_qoi * np.sqrt(qoi_ratio))  # Map focused squared-error change to QoI error.
        details = {"latent_eta2": float(latent_floor.sum()), "base_eta2": float(base_total), "q_mean": float(np.average(q, weights=np.maximum(anchor_error, 1.0e-30))), "tool_n_equations": float(preview.n_equations)}  # Preserve prior decomposition.
        return err_prior, elems_prior, float(e_energy), float(e_qoi), details  # Return structured prior consequences.
    def _ridge_ensemble(self, x_train: list[np.ndarray], y_train: list[float], x_query: np.ndarray, min_rows: int) -> tuple[np.ndarray, np.ndarray]:  # Fit deterministic bootstrap ridge models.
        if len(y_train) < min_rows:  # Keep the correction neutral before enough evidence exists.
            shape = x_query.shape[0] if x_query.ndim == 2 else 1  # Determine query count.
            return np.zeros(shape, dtype=float), np.full(shape, self.config.prior_uncertainty, dtype=float)  # Return zero mean and honest prior spread.
        X = np.vstack(x_train).astype(float)  # Assemble training features.
        y = np.asarray(y_train, dtype=float)  # Assemble training targets.
        query = np.atleast_2d(x_query).astype(float)  # Normalize query shape.
        rng = np.random.default_rng(self.config.seed + len(y_train))  # Change deterministic bootstrap samples as evidence grows.
        predictions = []  # Collect ensemble outputs.
        eye = np.eye(X.shape[1], dtype=float)  # Prepare ridge regularization.
        for _ in range(self.config.ensemble_size):  # Fit several bootstrap views of the same online evidence.
            indices = rng.integers(0, len(y), size=len(y))  # Resample rows with replacement.
            xb = X[indices]  # Select bootstrap features.
            yb = y[indices]  # Select bootstrap targets.
            beta = np.linalg.solve(xb.T @ xb + self.config.ridge * eye, xb.T @ yb)  # Fit a stable linear residual model.
            predictions.append(query @ beta)  # Evaluate the fitted correction.
        pred = np.vstack(predictions)  # Stack ensemble members.
        return pred.mean(axis=0), pred.std(axis=0)  # Return epistemic mean and spread.
    def predict(self, state: WorldState, preview: ToolPreview) -> WorldPrediction:  # Predict one action-conditioned transition.
        err_prior, elems_prior, e_energy_prior, e_qoi_prior, details = self._prior(state, preview)  # Evaluate structured finite-element dynamics.
        x_region = self._action_features(state, preview.sizes)  # Build residual-query rows.
        mean_region, std_region = self._ridge_ensemble(self._x_region, self._y_region, x_region, self.config.min_rows_for_fit)  # Predict online corrections.
        mean_region = np.clip(mean_region, -self.config.residual_clip, self.config.residual_clip)  # Bound extrapolation.
        err_corrected = np.maximum(err_prior * np.exp(mean_region), 1.0e-30)  # Apply multiplicative residual corrections.
        x_resource = self._global_features(state, preview.sizes)  # Build the resource query.
        mean_resource, std_resource = self._ridge_ensemble(self._x_resource, self._y_resource, x_resource, 1)  # Predict Gmsh count drift.
        n_equations = float(preview.n_equations * np.exp(float(np.clip(mean_resource[0], -0.35, 0.35))))  # Correct resource prediction conservatively.
        base_prior_total = max(float(err_prior.sum()), 1.0e-30)  # Normalize calibrated physical errors.
        corrected_ratio = max(float(err_corrected.sum()) / base_prior_total, 1.0e-12)  # Measure residual correction.
        e_energy = float(e_energy_prior * np.sqrt(corrected_ratio))  # Correct energy error.
        e_qoi = float(e_qoi_prior * np.sqrt(corrected_ratio))  # Correct QoI error with the same limited evidence.
        topology_mass = float(np.sum(state.err_share * state.roles[:, 2]))  # Quantify topology-sensitive uncertainty.
        uncertainty = float(np.sqrt(float(np.mean(std_region**2)) + float(std_resource[0] ** 2)) + self.config.topology_uncertainty * topology_mass)  # Aggregate epistemic and structural spread.
        details = dict(details)  # Copy prior details before extension.
        details.update({"regional_residual_mean": float(np.mean(mean_region)), "regional_residual_std": float(np.mean(std_region)), "resource_residual_mean": float(mean_resource[0]), "resource_residual_std": float(std_resource[0])})  # Add online-model decomposition.
        return WorldPrediction(action_id=preview.action.action_id, grades=preview.grades.copy(), sizes=preview.sizes.copy(), err_sum=err_corrected, elems=elems_prior, n_equations=n_equations, e_energy=e_energy, e_qoi=e_qoi, uncertainty=uncertainty, prior_total_eta2=float(err_prior.sum()), residual_total_eta2=float(err_corrected.sum()), model_samples=self.n_samples, details=details)  # Return the complete consequence prediction.
    def observe(self, transition: Transition) -> dict:  # Learn only after a real CalculiX transition.
        state = transition.state  # Read the executed parent state.
        preview = ToolPreview(action=transition.action, grades=transition.action.next_grades(state), sizes=np.asarray(transition.preview_sizes, dtype=float), n_equations=float(transition.preview_n_equations), audit={})  # Reconstruct the deterministic tool preview.
        prior_err, _, _, _, _ = self._prior(state, preview)  # Compute the uncalibrated prediction made at decision time.
        x_region = self._action_features(state, preview.sizes)  # Build training features in parent-region order.
        next_index = {name: i for i, name in enumerate(transition.next_state.names)}  # Match regions across optional splits.
        added = 0  # Count usable regional observations.
        residuals = []  # Preserve diagnostics.
        for i, name in enumerate(state.names):  # Train on every region that survives the transition.
            if name not in next_index:  # Skip a region removed by a future partition policy.
                continue  # Preserve alignment without fabricating a target.
            j = next_index[name]  # Locate the corresponding measured region.
            target = float(np.log(max(transition.next_state.err_sum[j], 1.0e-30) / max(prior_err[i], 1.0e-30)))  # Measure multiplicative prior error.
            target = float(np.clip(target, -self.config.residual_clip, self.config.residual_clip))  # Bound noisy targets.
            self._x_region.append(x_region[i].copy())  # Store one regional action-state row.
            self._y_region.append(target)  # Store its measured residual.
            residuals.append(target)  # Preserve transition diagnostics.
            added += 1  # Count the supervision row.
        x_resource = self._global_features(state, preview.sizes)  # Build the global resource row.
        y_resource = float(np.log(max(float(transition.next_state.n_equations), 1.0) / max(float(transition.preview_n_equations), 1.0)))  # Measure deterministic count drift.
        self._x_resource.append(x_resource.copy())  # Store the resource feature row.
        self._y_resource.append(float(np.clip(y_resource, -0.50, 0.50)))  # Store a bounded drift target.
        self.transitions.append(transition)  # Preserve the complete real transition.
        return {"regional_rows_added": int(added), "regional_residual_mean": float(np.mean(residuals)) if residuals else 0.0, "regional_residual_max_abs": float(np.max(np.abs(residuals))) if residuals else 0.0, "resource_log_residual": float(y_resource), "total_transitions": int(len(self.transitions)), "total_region_samples": int(self.n_samples)}  # Return an auditable update summary.
