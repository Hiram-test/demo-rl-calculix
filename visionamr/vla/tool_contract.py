# Deterministic mesh-action tools used directly by WM-VLA and by an optional MCP adapter.  # Module purpose.
from __future__ import annotations  # Enable postponed annotations for lightweight imports.
from dataclasses import dataclass  # Provide typed immutable action and certificate records.
from typing import Any  # Type JSON-compatible payloads without an external validation package.
import numpy as np  # Perform deterministic vector calculations for grades, sizes, and budgets.
from ..geometry import Problem  # Reuse the repository's geometry, load, and constraint contract.
from ..mesher import Mesh, generate_mesh  # Build candidate meshes and inspect their topology exactly.
from .drawing import drawings_size_fn, drawings_with_sizes  # Convert certified region sizes into Gmsh callbacks.
from .grades import GRADE_MAX, GRADE_MIN, GRADE_PRIOR  # Keep every numerical size mapping inside the tool layer.
from .regions import Partition  # Bind actions to the current named visual partition.


@dataclass(frozen=True)  # Make an action immutable after validation.
class MeshAction:  # Represent one discrete region-level control action.
    action_id: str  # Carry a stable identifier into records and benchmark artifacts.
    deltas: tuple[int, ...]  # Use -1 for refine, 0 for keep, and +1 for coarsen.
    source: str = "world"  # Disclose whether the action came from planning or a Dörfler guard candidate.
    stop: bool = False  # Allow the planner to terminate without another expensive solve.

    def validate(self, n_regions: int) -> "MeshAction":  # Enforce the complete action contract before execution.
        if not isinstance(self.action_id, str) or not self.action_id.strip():  # Reject missing trace identifiers.
            raise ValueError("action_id must be a non-empty string")  # Report a precise contract violation.
        if len(self.deltas) != int(n_regions):  # Prevent region-order or region-count mismatches.
            raise ValueError(f"action needs {n_regions} deltas, got {len(self.deltas)}")  # Stop unsafe execution.
        if any(int(value) not in (-1, 0, 1) for value in self.deltas):  # Restrict the action alphabet strictly.
            raise ValueError("every region delta must be -1, 0, or +1")  # Keep the model away from raw parameters.
        if self.stop and any(int(value) != 0 for value in self.deltas):  # Keep STOP semantically unambiguous.
            raise ValueError("a stop action cannot also change region grades")  # Reject contradictory requests.
        return self  # Return the validated immutable object for fluent use.

    def to_dict(self) -> dict[str, Any]:  # Serialize the action for SolveRecord.extra and JSON artifacts.
        return {  # Build a schema-stable JSON object.
            "action_id": self.action_id,  # Preserve the planner trace identifier.
            "deltas": [int(value) for value in self.deltas],  # Convert tuples and NumPy-like values to JSON ints.
            "source": self.source,  # Preserve scientific provenance explicitly.
            "stop": bool(self.stop),  # Preserve the terminal decision explicitly.
        }  # Finish the serialized action.


@dataclass  # Store the fast deterministic interpretation of one discrete action.
class MaterializedAction:  # Separate model-selected grades from tool-owned numerical parameters.
    action: MeshAction  # Retain the original discrete request and provenance.
    grades: np.ndarray  # Store the validated target grade vector.
    sizes: np.ndarray  # Store the numerical region sizes computed by the tool.
    predicted_equations: int  # Store the cheap resource-model prediction used during planning.
    predicted_elements: float  # Store the corresponding predicted element count.
    budget_scale: float  # Record any global coarsening applied to respect the predicted budget.
    valid: bool  # Mark whether the action changes the executable size field safely.
    reason: str  # Explain no-op, clipping, or budget projection behavior.

    def to_dict(self) -> dict[str, Any]:  # Serialize only JSON-compatible diagnostics.
        return {  # Build a stable artifact payload.
            "action": self.action.to_dict(),  # Include the original discrete action.
            "grades": [int(value) for value in self.grades],  # Emit ordinary integers.
            "sizes": [float(value) for value in self.sizes],  # Emit ordinary floating-point sizes.
            "predicted_equations": int(self.predicted_equations),  # Emit the planning resource estimate.
            "predicted_elements": float(self.predicted_elements),  # Emit the planning element estimate.
            "budget_scale": float(self.budget_scale),  # Disclose deterministic budget correction.
            "valid": bool(self.valid),  # Disclose whether execution is meaningful.
            "reason": str(self.reason),  # Preserve the validation explanation.
        }  # Finish the serialized materialization.


@dataclass  # Store the exact mesh-only certification result before a CalculiX solve.
class MeshCertificate:  # Make Gmsh and boundary-condition checks an explicit tool result.
    mesh: Mesh  # Return the actual candidate mesh to the solver pipeline.
    sizes: np.ndarray  # Return the final sizes after exact-budget correction.
    n_equations: int  # Return the exact free-displacement equation count for this mesh.
    attempts: int  # Count mesh generations because they have nonzero computational cost.
    budget_ok: bool  # State whether the certified mesh is within the hard equation cap.
    diagnostics: dict[str, Any]  # Preserve scales, hashes, and discrepancies for auditing.

    def to_dict(self) -> dict[str, Any]:  # Serialize certificate metadata without embedding the mesh arrays.
        return {  # Build a compact JSON-ready certificate.
            "sizes": [float(value) for value in self.sizes],  # Preserve the executed numerical parameters.
            "n_equations": int(self.n_equations),  # Preserve the exact pre-solve equation count.
            "attempts": int(self.attempts),  # Account for Gmsh certification cost.
            "budget_ok": bool(self.budget_ok),  # Preserve the hard-cap outcome.
            "mesh_sha": self.mesh.sha(),  # Bind the certificate to an exact topology.
            "diagnostics": dict(self.diagnostics),  # Preserve deterministic correction details.
        }  # Finish the serialized certificate.


def action_schema(n_regions: int) -> dict[str, Any]:  # Publish a JSON Schema suitable for an MCP tool wrapper.
    return {  # Return a strict schema with no undeclared parameters.
        "type": "object",  # Require an object payload.
        "additionalProperties": False,  # Reject hallucinated fields before they reach meshing code.
        "required": ["action_id", "deltas", "source", "stop"],  # Require complete provenance and intent.
        "properties": {  # Describe every accepted field precisely.
            "action_id": {"type": "string", "minLength": 1},  # Require a stable non-empty identifier.
            "deltas": {  # Describe the fixed-length discrete action vector.
                "type": "array",  # Require an array rather than free text.
                "minItems": int(n_regions),  # Require one entry for every region.
                "maxItems": int(n_regions),  # Forbid surplus entries and ordering ambiguity.
                "items": {"type": "integer", "enum": [-1, 0, 1]},  # Forbid direct continuous sizes.
            },  # Finish the delta-vector schema.
            "source": {"type": "string", "minLength": 1},  # Require scientific provenance.
            "stop": {"type": "boolean"},  # Require an explicit terminal flag.
        },  # Finish the property map.
    }  # Finish the JSON Schema.


def validate_action_payload(payload: dict[str, Any], n_regions: int) -> MeshAction:  # Convert untrusted JSON safely.
    allowed = {"action_id", "deltas", "source", "stop"}  # Define the complete accepted field set.
    extra = set(payload) - allowed  # Detect undeclared fields before type conversion.
    if extra:  # Reject any model-invented parameter names.
        raise ValueError(f"unexpected action fields: {sorted(extra)}")  # Explain the exact schema breach.
    missing = allowed - set(payload)  # Detect incomplete requests explicitly.
    if missing:  # Reject requests that omit provenance or intent.
        raise ValueError(f"missing action fields: {sorted(missing)}")  # Explain the exact missing fields.
    action = MeshAction(  # Construct the immutable typed action.
        action_id=str(payload["action_id"]),  # Normalize the identifier to text.
        deltas=tuple(int(value) for value in payload["deltas"]),  # Normalize the discrete vector to integers.
        source=str(payload["source"]),  # Normalize provenance to text.
        stop=bool(payload["stop"]),  # Normalize the terminal flag to bool.
    )  # Finish typed construction.
    return action.validate(n_regions)  # Apply semantic validation before returning.


def estimate_free_equations(problem: Problem, mesh: Mesh) -> int:  # Count displacement equations without solving.
    fixed = np.zeros((mesh.n_nodes, problem.dim), dtype=bool)  # Track constrained physical DOFs per mesh node.
    for constraint in problem.constraints:  # Apply every geometric boundary-condition predicate.
        mask = np.asarray(constraint.node_predicate(mesh.nodes), dtype=bool)  # Evaluate the predicate on this mesh.
        if mask.shape != (mesh.n_nodes,):  # Detect malformed predicates before an incorrect budget count.
            raise ValueError(f"constraint {constraint.name!r} returned shape {mask.shape}")  # Fail transparently.
        for dof in constraint.dofs:  # Apply each one-based displacement component listed by the problem.
            if 1 <= int(dof) <= problem.dim:  # Ignore components that do not exist in a lower-dimensional model.
                fixed[mask, int(dof) - 1] = True  # Mark the matching physical degree of freedom as fixed.
    return int(problem.dim * mesh.n_nodes - int(fixed.sum()))  # Return the exact free displacement count.


def _smoothed_sizes(sizes: np.ndarray, adjacency: np.ndarray, max_ratio: float) -> np.ndarray:  # Enforce region gradation.
    out = np.asarray(sizes, dtype=float).copy()  # Avoid mutating planner or state arrays in place.
    edges = np.argwhere(np.triu(np.asarray(adjacency, dtype=float), 1) > 0.0)  # Enumerate each region edge once.
    for _ in range(3):  # A few deterministic sweeps are sufficient for the small region graph.
        for left, right in edges:  # Inspect every adjacent region pair.
            i = int(left)  # Convert NumPy integer indices for stable list and array access.
            j = int(right)  # Convert NumPy integer indices for stable list and array access.
            if out[i] > max_ratio * out[j]:  # Detect an excessively coarse region next to a fine region.
                out[i] = max_ratio * out[j]  # Refine the coarse side rather than erasing the fine decision.
            if out[j] > max_ratio * out[i]:  # Check the reverse orientation after the first correction.
                out[j] = max_ratio * out[i]  # Refine the reverse coarse side deterministically.
    return out  # Return a graph-consistent size vector.


def _resource_prediction(problem: Problem, current_sizes: np.ndarray, target_sizes: np.ndarray, region_elems: np.ndarray, eq_per_elem: float) -> tuple[float, int]:  # Predict resource use cheaply.
    ratio = np.maximum(target_sizes, 1.0e-12) / np.maximum(current_sizes, 1.0e-12)  # Compute local size changes safely.
    predicted_elements = float(np.sum(np.maximum(region_elems, 1.0) * ratio ** (-float(problem.dim))))  # Apply N~h^-d.
    predicted_equations = int(round(max(eq_per_elem, 1.0e-12) * predicted_elements))  # Convert elements to equations.
    return predicted_elements, predicted_equations  # Return both levels for transparent accounting.


def fast_materialize_action(problem: Problem, current_sizes: np.ndarray, current_grades: np.ndarray, region_elems: np.ndarray, adjacency: np.ndarray, action: MeshAction, n_eq_budget: int, eq_per_elem: float, budget_safety: float = 0.95, max_neighbor_ratio: float = 1.8) -> MaterializedAction:  # Map grades to safe numerical sizes.
    n_regions = int(len(current_sizes))  # Establish the authoritative region count from the current state.
    action.validate(n_regions)  # Reject malformed or continuous actions before any calculation.
    sizes_now = np.asarray(current_sizes, dtype=float)  # Normalize current sizes to a floating-point vector.
    grades_now = np.asarray(current_grades, dtype=int)  # Normalize current grades to an integer vector.
    elems_now = np.asarray(region_elems, dtype=float)  # Normalize current regional element counts.
    if grades_now.shape != (n_regions,) or elems_now.shape != (n_regions,):  # Enforce aligned state arrays.
        raise ValueError("current sizes, grades, and region_elems must have identical lengths")  # Stop misalignment.
    if action.stop:  # Preserve a true no-solve terminal decision.
        predicted_elements = float(np.sum(np.maximum(elems_now, 1.0)))  # Report current resource use unchanged.
        predicted_equations = int(round(max(eq_per_elem, 1.0e-12) * predicted_elements))  # Report current equations.
        return MaterializedAction(action, grades_now.copy(), sizes_now.copy(), predicted_equations, predicted_elements, 1.0, True, "stop")  # Return unchanged parameters.
    deltas = np.asarray(action.deltas, dtype=int)  # Convert the validated discrete action into a vector.
    grades_new = np.clip(grades_now + deltas, GRADE_MIN, GRADE_MAX).astype(int)  # Apply grade bounds deterministically.
    prior_now = np.array([GRADE_PRIOR[int(grade)] for grade in grades_now], dtype=float)  # Read current grade priors.
    prior_new = np.array([GRADE_PRIOR[int(grade)] for grade in grades_new], dtype=float)  # Read target grade priors.
    ratio = prior_new / np.maximum(prior_now, 1.0e-12)  # Convert ordinal changes into relative size changes.
    sizes = np.clip(sizes_now * ratio, problem.h_min, problem.h0)  # Keep all parameters within problem bounds.
    sizes = _smoothed_sizes(sizes, adjacency, max_neighbor_ratio)  # Enforce region-to-region gradation.
    predicted_elements, predicted_equations = _resource_prediction(problem, sizes_now, sizes, elems_now, eq_per_elem)  # Predict cost before meshing.
    cap = max(float(budget_safety) * float(n_eq_budget), 1.0)  # Reserve deterministic headroom for Gmsh drift.
    budget_scale = 1.0  # Record no global correction unless the predicted cap is exceeded.
    if predicted_equations > cap:  # Apply a single closed-form resource projection when needed.
        budget_scale = float((predicted_equations / cap) ** (1.0 / float(problem.dim)) * 1.01)  # Compute safe coarsening.
        sizes = np.clip(sizes * budget_scale, problem.h_min, problem.h0)  # Apply the tool-owned scale within bounds.
        sizes = _smoothed_sizes(sizes, adjacency, max_neighbor_ratio)  # Recheck graph gradation after clipping.
        predicted_elements, predicted_equations = _resource_prediction(problem, sizes_now, sizes, elems_now, eq_per_elem)  # Recompute transparent cost.
    changed = bool(np.max(np.abs(np.log(np.maximum(sizes, 1.0e-12) / np.maximum(sizes_now, 1.0e-12)))) > 1.0e-5)  # Detect executable change.
    reason = "ok" if changed else "clipped_no_change"  # Explain why a nominal action may be ineffective.
    return MaterializedAction(action, grades_new, sizes, predicted_equations, predicted_elements, budget_scale, changed, reason)  # Return the safe action.


def certify_action_mesh(problem: Problem, partition: Partition, drawings: list, materialized: MaterializedAction, n_eq_budget: int, budget_safety: float = 0.98, max_attempts: int = 4) -> MeshCertificate:  # Verify the final mesh before solving.
    names = [seed.name for seed in partition.seeds]  # Fix region order from the authoritative partition.
    sizes = np.asarray(materialized.sizes, dtype=float).copy()  # Start from the planner's tool-materialized parameters.
    scales: list[float] = []  # Record every exact-budget correction for auditability.
    mesh: Mesh | None = None  # Hold the most recent generated mesh for a transparent fallback result.
    n_equations = 0  # Initialize the exact free-equation count.
    cap = max(int(np.floor(float(budget_safety) * float(n_eq_budget))), 1)  # Convert the hard safety band to an integer.
    for attempt in range(1, int(max_attempts) + 1):  # Limit mesh-only tool calls explicitly.
        sized_drawings = drawings_with_sizes(drawings, names, sizes)  # Bind numerical sizes to the visual regions.
        remainder_index = next((index for index, seed in enumerate(partition.seeds) if seed.origin == "coarse"), len(partition.seeds) - 1)  # Locate unpainted-volume size.
        remainder_h = float(sizes[int(remainder_index)])  # Read the certified remainder size.
        mesh = generate_mesh(problem, drawings_size_fn(sized_drawings, remainder_h, problem))  # Generate one deterministic candidate mesh.
        n_equations = estimate_free_equations(problem, mesh)  # Count exact free displacement equations without CCX.
        if n_equations <= cap:  # Accept the first mesh that satisfies the hard resource contract.
            diagnostics = {"cap": cap, "scales": scales, "materialized": materialized.to_dict()}  # Preserve all correction evidence.
            return MeshCertificate(mesh, sizes, n_equations, attempt, True, diagnostics)  # Return a certified executable mesh.
        scale = float((n_equations / float(cap)) ** (1.0 / float(problem.dim)) * 1.03)  # Infer a conservative correction from actual topology.
        scales.append(scale)  # Account for the correction explicitly.
        sizes = np.clip(sizes * scale, problem.h_min, problem.h0)  # Coarsen all regions without changing their ranking.
    if mesh is None:  # Guard against a nonpositive attempt count or unexpected loop bypass.
        raise RuntimeError("mesh certification generated no candidate mesh")  # Fail rather than inventing a certificate.
    diagnostics = {"cap": cap, "scales": scales, "materialized": materialized.to_dict(), "failure": "budget_not_met"}  # Preserve failure evidence.
    return MeshCertificate(mesh, sizes, n_equations, int(max_attempts), False, diagnostics)  # Return the failed certificate explicitly.
