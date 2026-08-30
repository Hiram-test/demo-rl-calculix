"""Deterministic MCP-compatible tool layer for adaptive mesh actions."""  # Describe the module purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
from dataclasses import dataclass  # Import compact tool-result contracts.
import hashlib  # Import hashing for action and mesh provenance.
import json  # Import canonical JSON encoding for provenance.
import numpy as np  # Import numerical arrays for mesh targets.
from ..baselines.dorfler import refine_size_map  # Reuse the exact repository Dörfler refinement atom.
from ..sizefield import NodalSizeField  # Reuse the common Gmsh nodal size-field adapter.
from .world_model import RegionAction, WorldState  # Import typed world-model action contracts.

@dataclass(frozen=True)  # Make a parameter certificate immutable and auditable.
class ActionCertificate:  # Store deterministic validation of one proposed regional action.
    accepted: bool  # Record whether the tool contract accepts the action.
    reasons: tuple[str, ...]  # Record every validation failure or warning.
    action_id: str  # Store a content-addressed action identifier.
    proactive: bool  # Record whether the action extends pure Dörfler.
    predicted_equations: float  # Store the planner's predicted equation count.
    budget: int  # Store the configured equation cap.
    budget_ratio: float  # Store the predicted resource ratio.
    def to_dict(self) -> dict:  # Convert the certificate to a JSON-safe payload.
        return {"accepted": self.accepted, "reasons": list(self.reasons), "action_id": self.action_id, "proactive": self.proactive, "predicted_equations": self.predicted_equations, "budget": self.budget, "budget_ratio": self.budget_ratio}  # Return primitive certificate fields.

@dataclass(frozen=True)  # Make a materialized mesh action immutable at the API boundary.
class MaterializedAction:  # Store exact nodal targets and their provenance.
    field: NodalSizeField  # Store the callable size field consumed by Gmsh.
    target_h: np.ndarray  # Store the final nodal target sizes.
    dorfler_h: np.ndarray  # Store the exact pure-Dörfler nodal target sizes.
    action_id: str  # Store the validated action identifier.
    target_sha256: str  # Store the exact nodal-target digest.
    dominance_verified: bool  # Record whether every target is no coarser than Dörfler.

def action_identifier(region_names: tuple[str, ...], action: RegionAction, mesh_sha: str = "") -> str:  # Build a stable content-addressed action identifier.
    payload = {"regions": list(region_names), "extra_depth": list(action.extra_depth), "mesh_sha": mesh_sha, "source": action.source}  # Assemble the canonical action payload.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")  # Encode the payload deterministically.
    return hashlib.sha256(encoded).hexdigest()[:20]  # Return a compact collision-resistant identifier.

def regional_level_targets(current_sizes: list[float] | np.ndarray, extra_depth: list[int] | np.ndarray, refine_factor: float, h_min: float, h_max: float) -> list[float]:  # Convert discrete depths into exact regional target sizes.
    sizes = np.asarray(current_sizes, dtype=float).reshape(-1)  # Normalize the regional sizes.
    depth = np.asarray(extra_depth, dtype=int).reshape(-1)  # Normalize the integer depths.
    if len(sizes) != len(depth):  # Require one depth per regional size.
        raise ValueError("current_sizes and extra_depth must have equal length")  # Report the structural mismatch.
    if not 0.0 < refine_factor < 1.0:  # Require a genuine refinement factor.
        raise ValueError("refine_factor must lie in (0, 1)")  # Report an invalid refinement atom.
    if h_min <= 0.0 or h_max < h_min:  # Validate the target-size interval.
        raise ValueError("invalid mesh-size bounds")  # Reject inconsistent size bounds.
    if np.any(depth < 0):  # Forbid action-level coarsening relative to Dörfler.
        raise ValueError("extra_depth must be non-negative")  # Report the monotonicity violation.
    targets = np.clip(sizes * refine_factor ** depth, h_min, h_max)  # Compute exact bounded size targets.
    return [float(value) for value in targets]  # Return a serialization-safe list.

class MCPMeshGateway:  # Centralize numerical validation and action materialization behind typed tools.
    def __init__(self, refine_factor: float = 0.5, max_extra_regions: int = 2, max_extra_depth: int = 2, proactive_budget_safety: float = 0.98) -> None:  # Configure the deterministic gateway.
        if not 0.0 < refine_factor < 1.0:  # Validate the shared refinement atom.
            raise ValueError("refine_factor must lie in (0, 1)")  # Reject an invalid atom.
        self.refine_factor = float(refine_factor)  # Store the exact Dörfler-compatible factor.
        self.max_extra_regions = int(max_extra_regions)  # Store the sparse-action limit.
        self.max_extra_depth = int(max_extra_depth)  # Store the depth limit.
        self.proactive_budget_safety = float(proactive_budget_safety)  # Store the proactive resource margin.
    def inspect_state(self, state: WorldState) -> dict:  # Return exact solver observations for an external agent or audit.
        return {"regions": list(state.names), "total_eta2": state.total_error, "n_equations": state.n_equations, "eq_per_elem": state.eq_per_elem, "h_min": state.h_min, "h_max": state.h0, "dimension": state.dim, "step": state.step, "error_share": [float(value) for value in state.error_share], "element_share": [float(value) for value in state.element_share], "dorfler_error_fraction": [float(value) for value in state.dorfler_error_fraction], "dorfler_element_fraction": [float(value) for value in state.dorfler_element_fraction], "hit_count": [float(value) for value in state.hit_count]}  # Return only measured or deterministically derived numbers.
    def certify_action(self, state: WorldState, action: RegionAction, predicted_equations: float, n_eq_budget: int, mesh_sha: str = "") -> ActionCertificate:  # Validate one planner action without changing its numerical values.
        reasons: list[str] = []  # Allocate the complete validation report.
        values = action.array(state.n_regions)  # Validate action alignment and non-negativity.
        proactive = bool(np.any(values > 0))  # Detect whether the action extends pure Dörfler.
        if int(np.count_nonzero(values)) > self.max_extra_regions:  # Enforce sparse regional investment.
            reasons.append("too_many_proactive_regions")  # Record the sparsity violation.
        if int(np.max(values, initial=0)) > self.max_extra_depth:  # Enforce the configured depth limit.
            reasons.append("extra_depth_exceeds_contract")  # Record the depth violation.
        if not np.isfinite(predicted_equations) or predicted_equations <= 0.0:  # Validate the planner's resource number.
            reasons.append("invalid_predicted_equations")  # Record an invalid resource forecast.
        budget = max(int(n_eq_budget), 1)  # Normalize the configured equation cap.
        budget_ratio = float(predicted_equations / budget) if np.isfinite(predicted_equations) else float("inf")  # Compute the auditable resource ratio.
        if proactive and budget_ratio > self.proactive_budget_safety:  # Require headroom only for proactive world-model actions.
            reasons.append("proactive_budget_margin_failed")  # Record the resource-margin failure.
        action_id = action_identifier(state.names, action, mesh_sha)  # Compute the content-addressed action identifier.
        return ActionCertificate(accepted=not reasons, reasons=tuple(reasons), action_id=action_id, proactive=proactive, predicted_equations=float(predicted_equations), budget=budget, budget_ratio=budget_ratio)  # Return the immutable certificate.
    def materialize(self, mesh, problem, labels: np.ndarray, marked: np.ndarray, action: RegionAction, gradation: float, action_id: str | None = None) -> MaterializedAction:  # Convert one certified macro-action into exact Dörfler-dominant nodal targets.
        values = action.array()  # Read the complete semantic action vector.
        if int(np.max(labels, initial=-1)) >= len(values):  # Ensure every element label maps to an action entry.
            raise ValueError("element labels exceed the action dimension")  # Reject an inconsistent partition before meshing.
        current_h = np.asarray(mesh.node_sizes, dtype=float)  # Read the current mesh sizes from the shared mesh object.
        dorfler_h = np.clip(refine_size_map(mesh, np.asarray(marked, dtype=int), factor=self.refine_factor), problem.h_min, problem.h0)  # Build the exact standard Dörfler target.
        target_h = dorfler_h.copy()  # Start every action from the exact Dörfler safety policy.
        region_labels = np.asarray(labels, dtype=int).reshape(-1)  # Normalize element-to-region labels.
        for region, depth in enumerate(values):  # Apply only validated proactive regional depths.
            if depth <= 0:  # Leave pure-Dörfler regions unchanged.
                continue  # Skip all non-proactive regions.
            region_elements = np.nonzero(region_labels == region)[0]  # Select the semantic region's elements.
            if len(region_elements) == 0:  # Ignore an empty visual region without inventing nodes.
                continue  # Preserve the exact Dörfler target.
            region_nodes = np.unique(mesh.cells[region_elements].ravel())  # Collect every node incident to the region.
            region_target = np.clip(current_h[region_nodes] * self.refine_factor ** int(depth), problem.h_min, problem.h0)  # Compute the exact discrete-depth target.
            target_h[region_nodes] = np.minimum(target_h[region_nodes], region_target)  # Extend or deepen Dörfler without ever coarsening it.
        dominance = bool(np.all(target_h <= dorfler_h + 1.0e-12))  # Verify nodal action dominance explicitly.
        if not dominance:  # Treat any numerical violation as a hard contract failure.
            raise RuntimeError("materialized target is coarser than the Dörfler fallback")  # Stop before Gmsh sees an unsafe target.
        target_h = np.clip(target_h, problem.h_min, problem.h0)  # Reassert the repository mesh-size limits.
        field = NodalSizeField(mesh, target_h, gradation=float(gradation), h_min=problem.h_min, h_max=problem.h0)  # Build the single approved Gmsh callback.
        digest = hashlib.sha256(np.ascontiguousarray(target_h).tobytes()).hexdigest()  # Hash the exact nodal action.
        identifier = action_id or action_identifier(tuple(str(index) for index in range(len(values))), action, mesh.sha())  # Preserve a prior certificate identifier or derive a local one.
        return MaterializedAction(field=field, target_h=target_h, dorfler_h=dorfler_h, action_id=identifier, target_sha256=digest, dominance_verified=dominance)  # Return the exact materialized action.
