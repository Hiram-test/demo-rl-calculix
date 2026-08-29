from __future__ import annotations  # Enable compact type annotations for the guarded numerical gateway.
import hashlib  # Bind every guarded request to its exact state and bonus regions.
from dataclasses import replace  # Reuse immutable tool records while changing only certified fields.
import numpy as np  # Build Dörfler and semantic nodal target fields deterministically.
from ..baselines.dorfler import refine_size_map  # Reuse the exact repository Dörfler refinement atom.
from ..marking import dorfler_mark  # Reuse faithful element-level bulk marking.
from ..mesher import generate_mesh  # Materialize only selected candidate meshes through Gmsh.
from ..sizefield import NodalSizeField  # Convert guarded nodal targets into the common remeshing field.
from .tool_gateway import CertifiedAction, DeterministicToolGateway, ToolGatewayConfig, ToolPreview, estimate_free_equations  # Reuse the existing validated tool contract.
from .world_state import WorldAction, WorldState  # Accept only audited states and discrete model actions.
class GuardedToolGateway(DeterministicToolGateway):  # Add world-model bonus refinement without weakening the Dörfler backbone.
    mapping_version = "wm-vla-dorfler-backbone-v2"  # Version the guarded numerical contract independently from v1.
    def __init__(self, problem, config: ToolGatewayConfig | None = None, bonus_factors: tuple[float, ...] = (0.58, 0.72, 0.86)) -> None:  # Bind deterministic bonus strengths to one problem.
        super().__init__(problem, config)  # Reuse initial semantic mesh certification and exact Dörfler execution.
        self.bonus_factors = tuple(float(value) for value in bonus_factors)  # Store tool-owned strengths from strongest to weakest.
        if not self.bonus_factors or any(value <= 0.0 or value > 1.0 for value in self.bonus_factors):  # Validate the numerical contract once.
            raise ValueError("bonus factors must lie in (0, 1]")  # Prevent hidden coarsening or invalid target sizes.
    def validate(self, state: WorldState, action: WorldAction) -> np.ndarray:  # Extend validation with one guarded execution kind.
        if action.kind == "guarded":  # Validate the world-model bonus action locally.
            grades = action.next_grades(state)  # Check vector length and adjacent discrete changes.
            if action.n_changed < 1:  # Require a real semantic allocation decision.
                raise ValueError("guarded action must select at least one bonus region")  # Exclude empty actions from advantage claims.
            if action.n_changed > self.config.max_changed_regions:  # Bound one model intervention.
                raise ValueError("guarded action changes too many regions in one step")  # Let receding-horizon control handle later regions.
            if any(value > 0 for value in action.deltas):  # Forbid semantic coarsening in the protected action.
                raise ValueError("guarded action may only add refinement to Dörfler")  # Preserve the backbone pointwise in target-size space.
            return grades  # Return validated discrete state.
        return super().validate(state, action)  # Delegate exact Dörfler, region, and stop validation to v1.
    @staticmethod  # Keep regional bulk approximation deterministic for model rollouts.
    def _regional_bulk(state: WorldState, theta: float) -> np.ndarray:  # Approximate exact element marking only inside the cheap preview.
        order = np.argsort(-state.err_sum)  # Rank regions by measured squared-estimator mass.
        selected = np.zeros(state.n_regions, dtype=bool)  # Allocate the regional preview mask.
        captured = 0.0  # Accumulate selected error mass.
        target = float(theta * max(float(state.err_sum.sum()), 1.0e-30))  # Define the bulk target.
        for index in order:  # Select the smallest regional bulk set.
            selected[int(index)] = True  # Mark the next region.
            captured += float(state.err_sum[index])  # Update captured mass.
            if captured >= target:  # Stop once the bulk condition is reached.
                break  # Preserve minimal preview support.
        return selected  # Return the cheap approximation without changing exact execution.
    def preview(self, state: WorldState, action: WorldAction) -> ToolPreview:  # Resolve guarded consequences without accepting continuous model parameters.
        if action.kind != "guarded":  # Keep existing behavior for all non-guarded methods.
            return super().preview(state, action)  # Reuse the validated v1 path.
        grades = self.validate(state, action)  # Validate the discrete bonus selection.
        bulk = self._regional_bulk(state, float(self.config.dorfler_theta))  # Approximate the Dörfler backbone in region space.
        bonus = np.asarray(action.deltas, dtype=int) < 0  # Decode only selected bonus regions.
        raw_sizes = np.asarray(state.sizes, dtype=float).copy()  # Start from the latest realized mesh instead of resetting to absolute grade priors.
        raw_sizes[bulk] *= float(self.config.dorfler_factor)  # Apply the regional preview of exact Dörfler.
        raw_sizes[bonus] *= float(self.bonus_factors[1 if len(self.bonus_factors) > 1 else 0])  # Add a moderate semantic bonus in the preview.
        raw_sizes = np.clip(raw_sizes, float(self.problem.h_min), float(self.problem.h0))  # Enforce physical mesh limits.
        ratio = np.maximum(raw_sizes, 1.0e-12) / np.maximum(state.sizes, 1.0e-12)  # Measure relative changes from the real current state.
        predicted_elems = float(np.sum(np.maximum(state.elems, 1.0) * ratio ** (-float(self.problem.dim))))  # Predict regional element redistribution.
        eq_per_elem = float(state.n_equations) / max(float(state.elems.sum()), 1.0)  # Calibrate resource scaling on the latest real mesh.
        predicted_equations = max(predicted_elems * eq_per_elem, 1.0)  # Obtain the unprojected equation prediction.
        target = float(self.config.budget_safety * state.budget)  # Define the deterministic safe budget.
        if predicted_equations > target:  # Project only the preview when the protected candidate is predicted to exceed resources.
            scale = float((predicted_equations / max(target, 1.0)) ** (1.0 / float(self.problem.dim)))  # Solve the global resource law once.
            preview_sizes = np.clip(raw_sizes * scale, float(self.problem.h_min), float(self.problem.h0))  # Produce a comparable world-model input.
            ratio = np.maximum(preview_sizes, 1.0e-12) / np.maximum(state.sizes, 1.0e-12)  # Recompute projected changes.
            predicted_elems = float(np.sum(np.maximum(state.elems, 1.0) * ratio ** (-float(self.problem.dim))))  # Recompute projected elements.
            predicted_equations = max(predicted_elems * eq_per_elem, 1.0)  # Recompute projected equations.
        else:  # Preserve all protected refinement when the preview is already feasible.
            scale = 1.0  # Record that no preview projection occurred.
            preview_sizes = raw_sizes  # Preserve the exact relative allocation.
        digest = hashlib.sha256()  # Create an idempotent guarded request id.
        digest.update(state.state_id.encode("ascii"))  # Bind the request to one state.
        digest.update(action.action_id.encode("utf-8"))  # Bind the request to one planner choice.
        digest.update(np.ascontiguousarray(bonus, dtype=np.uint8).tobytes())  # Bind the exact semantic bonus set.
        audit = {"mapping_version": self.mapping_version, "state_id": state.state_id, "action_id": action.action_id, "request_id": digest.hexdigest()[:24], "global_preview_scale": float(scale), "predicted_equations": float(predicted_equations), "continuous_input_from_model": False, "dorfler_backbone": True, "bonus_regions": [state.names[i] for i in np.flatnonzero(bonus)]}  # Emit a strict MCP-style preview response.
        return ToolPreview(action=action, grades=grades, sizes=preview_sizes, n_equations=float(predicted_equations), audit=audit)  # Return the action-conditioned world-model input.
    def certify(self, partition, state: WorldState, action: WorldAction, current_mesh, eta2: np.ndarray) -> CertifiedAction:  # Materialize an exact Dörfler backbone plus selected semantic refinement.
        if action.kind != "guarded":  # Keep exact v1 execution for all other methods.
            return super().certify(partition, state, action, current_mesh, eta2)  # Delegate validated execution.
        preview = self.preview(state, action)  # Validate and audit the selected bonus regions.
        marked = dorfler_mark(np.asarray(eta2, dtype=float), float(self.config.dorfler_theta))  # Compute exact element-level Dörfler marking.
        base_target = refine_size_map(current_mesh, marked, factor=float(self.config.dorfler_factor))  # Construct the exact nodal backbone.
        labels = partition.assign(current_mesh)  # Assign current elements to persistent semantic regions.
        bonus_region_mask = np.asarray(action.deltas, dtype=int) < 0  # Decode the selected bonus regions.
        bonus_cells = np.flatnonzero(bonus_region_mask[labels])  # Locate their current elements.
        bonus_nodes = np.unique(current_mesh.cells[bonus_cells].ravel()) if len(bonus_cells) else np.empty(0, dtype=int)  # Locate their mesh nodes.
        target_equations = float(self.config.budget_safety * state.budget)  # Define the pre-solve safety cap.
        candidates: list[tuple[int, float, object, np.ndarray]] = []  # Collect feasible protected meshes as equation count, factor, mesh, and target.
        passes = 0  # Count Gmsh work separately from real finite-element solves.
        for factor in self.bonus_factors[: self.config.max_mesh_passes]:  # Try strong-to-weak tool-owned semantic bonuses.
            target_h = np.asarray(base_target, dtype=float).copy()  # Start every trial from the same exact Dörfler backbone.
            if len(bonus_nodes):  # Add only selected semantic refinement.
                target_h[bonus_nodes] = np.minimum(target_h[bonus_nodes], float(factor) * current_mesh.node_sizes[bonus_nodes])  # Never make a Dörfler target coarser.
            field = NodalSizeField(current_mesh, target_h, gradation=float(self.config.gradation), h_min=self.problem.h_min, h_max=self.problem.h0)  # Build the common remeshing field.
            trial_mesh = generate_mesh(self.problem, field)  # Generate one protected candidate without a CalculiX call.
            passes += 1  # Account for one mesh-only candidate.
            estimated = estimate_free_equations(trial_mesh, self.problem)  # Count exact free translational equations.
            if estimated <= target_equations:  # Retain only budget-feasible protected candidates.
                candidates.append((int(estimated), float(factor), trial_mesh, target_h))  # Preserve the densest feasible candidate later.
        if candidates:  # Choose a protected candidate whenever the budget permits one.
            estimated, accepted_factor, mesh, accepted_target = max(candidates, key=lambda item: item[0])  # Use the largest feasible equation count.
            bonus_accepted = bool(len(bonus_nodes) > 0 and accepted_factor < 1.0)  # Record a real world-model intervention.
            execution = "exact_dorfler_plus_world_bonus"  # Identify the protected execution path.
        else:  # Fall back to exact Dörfler when every semantic bonus exceeds the safe budget.
            fallback_action = replace(action, kind="dorfler", source="guarded_budget_fallback")  # Preserve the planner id while declaring exact fallback execution.
            fallback = super().certify(partition, state, fallback_action, current_mesh, eta2)  # Reuse the faithful exact Dörfler certifier.
            audit = dict(preview.audit)  # Preserve the original guarded request.
            audit.update(dict(fallback.audit))  # Add exact fallback mesh evidence.
            audit.update({"execution": "exact_dorfler_budget_fallback", "world_bonus_accepted": False, "bonus_nodes": int(len(bonus_nodes)), "mesh_passes": int(passes + fallback.mesh_passes)})  # Explain why no bonus reached CalculiX.
            return CertifiedAction(action=action, grades=preview.grades, sizes=preview.sizes, drawings=tuple(partition.drawings), mesh=fallback.mesh, estimated_equations=int(fallback.estimated_equations), mesh_passes=int(passes + fallback.mesh_passes), audit=audit)  # Return a safe solver-ready fallback under the original action identity.
        measured_sizes = np.asarray(preview.sizes, dtype=float).copy()  # Retain a region-level consequence representation for online learning.
        audit = dict(preview.audit)  # Start from the strict request audit.
        audit.update({"execution": execution, "world_bonus_accepted": bool(bonus_accepted), "accepted_bonus_factor": float(accepted_factor), "bonus_nodes": int(len(bonus_nodes)), "n_marked": int(len(marked)), "estimated_equations": int(estimated), "mesh_passes": int(passes), "target_is_never_coarser_than_dorfler": True})  # Record the full numerical certificate.
        return CertifiedAction(action=action, grades=preview.grades, sizes=measured_sizes, drawings=tuple(partition.drawings), mesh=mesh, estimated_equations=int(estimated), mesh_passes=int(passes), audit=audit)  # Return the protected solver-ready mesh.
