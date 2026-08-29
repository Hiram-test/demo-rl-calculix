from __future__ import annotations  # Enable postponed annotations for compact type hints.
import hashlib  # Hash every state and action for deterministic tool auditing.
from dataclasses import dataclass  # Define immutable state and action records.
import numpy as np  # Store region graphs and numerical features.
from .regions import Partition, RegionFeatures  # Reuse the existing geometric partition and measured features.
_ROLE_NAMES = ("load", "support", "hole", "corner", "field", "split")  # Fix the semantic channel order.
def semantic_roles(names: tuple[str, ...], origins: tuple[str, ...]) -> np.ndarray:  # Encode visible bridge semantics without solver leakage.
    roles = np.zeros((len(names), len(_ROLE_NAMES)), dtype=float)  # Allocate one multi-hot vector per region.
    for i, (name, origin) in enumerate(zip(names, origins)):  # Inspect each region name and provenance once.
        tag = name.lower()  # Normalize text matching while preserving the original name elsewhere.
        roles[i, 0] = float(any(word in tag for word in ("wheel", "patch", "load", "pressure")))  # Mark load-transfer regions.
        roles[i, 1] = float(any(word in tag for word in ("support", "girder", "strip", "bottom", "clamp", "bearing")))  # Mark supports and reaction paths.
        roles[i, 2] = float(any(word in tag for word in ("hole", "duct", "opening", "service", "rim")))  # Mark topology-visible openings and rims.
        roles[i, 3] = float(any(word in tag for word in ("corner", "edge", "reentrant", "junction")))  # Mark geometric singularity candidates.
        roles[i, 4] = float(origin == "coarse" or tag in ("field", "bulk", "remainder", "unpainted"))  # Mark the unpainted background region.
        roles[i, 5] = float(origin == "split" or "hot" in tag)  # Mark regions created from measured concentration.
        if float(roles[i].sum()) == 0.0:  # Detect names that carry no explicit semantic token.
            roles[i, 4] = 0.25  # Retain a weak generic-field prior instead of an all-zero embedding.
    return roles  # Return deterministic semantics that never use local-prediction output.
@dataclass(frozen=True)  # Keep an executed or imagined state immutable.
class WorldState:  # Represent the decision-relevant region graph after one real or imagined solve.
    step: int  # Count real or imagined transitions from the first solve.
    names: tuple[str, ...]  # Preserve stable region identifiers across remeshing.
    origins: tuple[str, ...]  # Preserve vision, coarse, or split provenance.
    grades: np.ndarray  # Store discrete model decisions where one is finest and five is coarsest.
    sizes: np.ndarray  # Store measured or tool-certified characteristic sizes.
    err_sum: np.ndarray  # Store regional sums of the squared ZZ indicator.
    elems: np.ndarray  # Store regional element counts.
    vm_max: np.ndarray  # Store regional maximum von Mises stress.
    vm_mean: np.ndarray  # Store regional mean von Mises stress.
    volume: np.ndarray  # Store regional physical volume.
    adjacency: np.ndarray  # Store the undirected region graph.
    roles: np.ndarray  # Store semantic channels derived only from geometry and names.
    n_equations: int  # Store the realized or predicted free-equation count.
    budget: int  # Store the hard equation budget.
    e_energy: float  # Store the reference energy-norm error.
    e_qoi: float  # Store the reference quantity-of-interest error.
    total_eta2: float  # Store the global squared ZZ indicator.
    qoi: float  # Store the physical quantity of interest.
    U_total: float  # Store total strain energy.
    state_id: str  # Bind every action request to an exact state snapshot.
    @property  # Expose the number of regions without duplicating metadata.
    def n_regions(self) -> int:  # Return the graph order.
        return len(self.names)  # Count stable names because every regional array follows that order.
    @property  # Normalize error allocation for planning.
    def err_share(self) -> np.ndarray:  # Return a numerically safe regional error distribution.
        return self.err_sum / max(float(self.err_sum.sum()), 1.0e-30)  # Avoid division by zero on converged states.
    @property  # Normalize resource allocation for planning.
    def elem_share(self) -> np.ndarray:  # Return the regional element distribution.
        return self.elems / max(float(self.elems.sum()), 1.0)  # Use one as the zero-mesh guard.
    @property  # Expose budget utilization for objectives and stopping.
    def budget_use(self) -> float:  # Return realized equations divided by the hard budget.
        return float(self.n_equations) / max(float(self.budget), 1.0)  # Keep the ratio finite for malformed test fixtures.
    def node_features(self) -> np.ndarray:  # Build scale-robust node features for online residual learning.
        degree = self.adjacency.sum(axis=1) / max(float(self.n_regions - 1), 1.0)  # Normalize graph degree.
        vm_scale = max(float(np.max(self.vm_max)), 1.0e-30)  # Normalize stress without importing material-specific units.
        vol_share = self.volume / max(float(self.volume.sum()), 1.0e-30)  # Normalize physical support.
        h_scale = max(float(np.max(self.sizes)), 1.0e-30)  # Normalize measured size inside the current state.
        columns = [  # Assemble only decision-relevant, dimensionless channels.
            np.log10(np.maximum(self.err_share, 1.0e-12)),  # Preserve several orders of estimator concentration.
            np.log10(np.maximum(self.elem_share, 1.0e-12)),  # Preserve several orders of resource concentration.
            self.vm_max / vm_scale,  # Encode local stress peaks.
            self.vm_mean / vm_scale,  # Encode regional stress level.
            self.sizes / h_scale,  # Encode current relative resolution.
            self.grades.astype(float) / 5.0,  # Encode discrete visual/tool state.
            vol_share,  # Encode physical region extent.
            degree,  # Encode cross-region influence opportunities.
        ]  # Close the scalar feature list.
        return np.column_stack(columns + [self.roles[:, j] for j in range(self.roles.shape[1])])  # Append semantic channels in fixed order.
@dataclass(frozen=True)  # Make actions hashable and safe to pass through a tool boundary.
class WorldAction:  # Express a high-level VLA action without any continuous mesh parameter.
    action_id: str  # Identify the action in traces and model rollouts.
    deltas: tuple[int, ...]  # Change each grade by minus one, zero, or plus one.
    kind: str = "region"  # Select region-grade, exact Dörfler, or stop execution.
    source: str = "world_model"  # Record why the planner proposed the action.
    rationale: str = ""  # Preserve an auditable natural-language explanation.
    def next_grades(self, state: WorldState) -> np.ndarray:  # Apply the discrete action deterministically.
        if len(self.deltas) != state.n_regions:  # Reject stale actions after a partition change.
            raise ValueError("action region count does not match state")  # Stop before any mesh parameter is produced.
        delta = np.asarray(self.deltas, dtype=int)  # Convert the immutable tuple to a numerical vector.
        if np.any((delta < -1) | (delta > 1)):  # Restrict one decision step to adjacent grade moves.
            raise ValueError("grade deltas must be -1, 0, or 1")  # Prevent hidden continuous tuning.
        return np.clip(state.grades + delta, 1, 5).astype(int)  # Enforce the declared grade contract.
    @property  # Count action sparsity for tool limits and planning cost.
    def n_changed(self) -> int:  # Return the number of nonzero grade decisions.
        return int(np.count_nonzero(np.asarray(self.deltas, dtype=int)))  # Count only explicit model interventions.
@dataclass(frozen=True)  # Store one real transition for online calibration.
class Transition:  # Pair a previous state, executed action, tool preview, and next real state.
    state: WorldState  # Store the state on which the action was certified.
    action: WorldAction  # Store the discrete action that was actually executed.
    preview_sizes: np.ndarray  # Store deterministic tool sizes used by the prior prediction.
    preview_n_equations: float  # Store the tool's pre-solve resource prediction.
    next_state: WorldState  # Store the next state measured by CalculiX.
def make_state(partition: Partition, features: RegionFeatures, adjacency: np.ndarray, record, grades: np.ndarray, step: int, budget: int, sizes: np.ndarray | None = None) -> WorldState:  # Convert one real solve into a world-model state.
    names = tuple(seed.name for seed in partition.seeds)  # Freeze the partition order.
    origins = tuple(seed.origin for seed in partition.seeds)  # Freeze semantic provenance.
    grades_array = np.asarray(grades, dtype=int).copy()  # Copy model decisions away from mutable caller storage.
    size_array = np.asarray(features.h_meas if sizes is None else sizes, dtype=float).copy()  # Prefer realized regional mesh sizes.
    roles = semantic_roles(names, origins)  # Derive geometry-visible semantics without reading the solution.
    digest = hashlib.sha256()  # Create an idempotency key for tool calls.
    digest.update("|".join(names).encode("utf-8"))  # Bind the identifier to exact region names and order.
    digest.update(np.ascontiguousarray(grades_array).tobytes())  # Bind the identifier to the discrete state.
    digest.update(np.ascontiguousarray(np.round(features.err_share, 10)).tobytes())  # Bind the identifier to the observed physics distribution.
    digest.update(f"{int(record.n_equations)}:{int(step)}".encode("ascii"))  # Bind the identifier to resource state and step.
    e_energy = float(record.e_energy) if record.e_energy is not None else float("inf")  # Preserve a defined objective when no reference exists.
    e_qoi = float(record.e_qoi) if record.e_qoi is not None else float("inf")  # Preserve a defined QoI objective when no reference exists.
    return WorldState(  # Construct one immutable state snapshot.
        step=int(step),  # Record the transition depth.
        names=names,  # Record stable names.
        origins=origins,  # Record provenance.
        grades=grades_array,  # Record discrete grades.
        sizes=size_array,  # Record realized or certified sizes.
        err_sum=np.asarray(features.err_sum, dtype=float).copy(),  # Copy regional estimator mass.
        elems=np.asarray(features.elems, dtype=float).copy(),  # Copy regional element counts as float for scaling.
        vm_max=np.asarray(features.vm_max, dtype=float).copy(),  # Copy peak stress.
        vm_mean=np.asarray(features.vm_mean, dtype=float).copy(),  # Copy mean stress.
        volume=np.asarray(features.volume, dtype=float).copy(),  # Copy physical volume.
        adjacency=np.asarray(adjacency, dtype=float).copy(),  # Copy the region graph.
        roles=roles,  # Store fixed semantic channels.
        n_equations=int(record.n_equations),  # Store actual equations.
        budget=int(budget),  # Store the hard cap.
        e_energy=e_energy,  # Store reference energy error.
        e_qoi=e_qoi,  # Store reference QoI error.
        total_eta2=float(features.total_err),  # Store global estimator mass.
        qoi=float(record.qoi),  # Store the measured QoI.
        U_total=float(record.U_total),  # Store measured strain energy.
        state_id=digest.hexdigest()[:20],  # Use a compact but collision-resistant audit identifier.
    )  # Finish state construction.
