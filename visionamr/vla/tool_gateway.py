from __future__ import annotations  # Enable compact forward type references.
import hashlib  # Bind every certified mesh to its exact state and action.
from dataclasses import dataclass  # Define explicit tool request and response records.
import numpy as np  # Perform deterministic grade, resource, and constraint calculations.
from ..baselines.dorfler import refine_size_map  # Reuse the exact element-wise Dörfler refinement atom.
from ..marking import dorfler_mark  # Reuse the repository's faithful bulk marker.
from ..mesher import Mesh, generate_mesh  # Generate every candidate through the single Gmsh path.
from ..sizefield import NodalSizeField  # Convert exact Dörfler targets into a Gmsh field.
from .drawing import DrawnRegion, drawings_size_fn, drawings_with_sizes  # Materialize region grades without LLM parameters.
from .grades import GRADE_PRIOR, MIN_STEP, parse_grade  # Map discrete grades through one versioned table.
from .regions import Partition  # Keep tool execution aligned with the visible region graph.
from .world_state import WorldAction, WorldState  # Accept only audited world states and discrete actions.
@dataclass(frozen=True)  # Keep numerical tool policy explicit and reproducible.
class ToolGatewayConfig:  # Configure deterministic mesh certification rather than language-model tuning.
    budget_safety: float = 0.94  # Leave headroom between exact free-DOF counting and the solver report.
    lower_use_target: float = 0.72  # Avoid extremely under-filled meshes when a global correction is safe.
    max_mesh_passes: int = 3  # Limit selected-action Gmsh correction passes.
    max_changed_regions: int = 3  # Bound each learned action while allowing repeated multi-step control.
    dorfler_theta: float = 0.50  # Match the main bulk-marking baseline.
    dorfler_factor: float = 0.50  # Match the main marked-node refinement ratio.
    gradation: float = 0.90  # Match the framework's size-field Lipschitz setting.
@dataclass(frozen=True)  # Return a cheap consequence input for world-model rollouts.
class ToolPreview:  # Represent deterministic grade-to-size and resource mapping without a solve.
    action: WorldAction  # Preserve the audited high-level action.
    grades: np.ndarray  # Return exact next discrete grades.
    sizes: np.ndarray  # Return exact versioned grade sizes after budget projection.
    n_equations: float  # Return the resource-model equation prediction.
    audit: dict  # Return an MCP-style structured audit payload.
@dataclass(frozen=True)  # Return the only object permitted to reach CalculiX.
class CertifiedAction:  # Bind a validated action to one concrete Gmsh mesh.
    action: WorldAction  # Preserve the selected action.
    grades: np.ndarray  # Preserve exact grades after validation.
    sizes: np.ndarray  # Preserve certified requested regional sizes.
    drawings: tuple[DrawnRegion, ...]  # Preserve updated visible-region geometry for later steps.
    mesh: Mesh  # Carry the generated mesh directly into the solver.
    estimated_equations: int  # Record exact free-DOF counting on the generated mesh.
    mesh_passes: int  # Record the number of deterministic Gmsh corrections.
    audit: dict  # Record hashes, limits, and mapping version.
def estimate_free_equations(mesh: Mesh, problem) -> int:  # Count active translational equations before invoking CalculiX.
    n_dof = int(problem.dim)  # Use two active translations for plane stress and three for solids.
    fixed = np.zeros((mesh.n_nodes, n_dof), dtype=bool)  # Allocate one flag per potentially active degree of freedom.
    for constraint in problem.constraints:  # Apply every geometric boundary condition to the generated nodes.
        node_mask = np.asarray(constraint.node_predicate(mesh.nodes), dtype=bool)  # Evaluate the same predicate used by deck writing.
        if not np.any(node_mask):  # Reject a mesh that lost a required boundary footprint.
            raise ValueError(f"constraint {constraint.name!r} matched no nodes")  # Fail before an invalid solver call.
        for dof in constraint.dofs:  # Mark each constrained translation exactly once.
            if 1 <= int(dof) <= n_dof:  # Ignore inactive out-of-plane labels in reduced-dimensional tests.
                fixed[node_mask, int(dof) - 1] = True  # Deduplicate overlaps through the Boolean mask.
    return int(n_dof * mesh.n_nodes - int(fixed.sum()))  # Return the exact free translational count for current models.
def _grade_sizes(problem, grades: np.ndarray) -> np.ndarray:  # Convert only validated discrete levels into physical sizes.
    parsed = np.array([parse_grade(int(value), "world action grade") for value in np.asarray(grades).ravel()], dtype=int)  # Validate every level.
    sizes = np.array([GRADE_PRIOR[int(value)] * float(problem.h0) for value in parsed], dtype=float)  # Apply the single versioned lookup.
    levels = sorted({int(value) for value in parsed})  # Preserve strict ordering between levels that are present.
    for fine, coarse in zip(levels, levels[1:]):  # Enforce a visible minimum separation between adjacent used grades.
        lower = float(sizes[parsed == fine][0] * MIN_STEP)  # Compute the coarser level's minimum legal size.
        sizes[parsed == coarse] = np.maximum(sizes[parsed == coarse], lower)  # Prevent grade collapse after clipping.
    return np.clip(sizes, float(problem.h_min), float(problem.h0))  # Enforce problem-level physical limits.
def _resource_prediction(state: WorldState, sizes: np.ndarray, problem) -> float:  # Predict equations using only measured resource scaling.
    anchor = np.maximum(np.asarray(state.sizes, dtype=float), 1.0e-12)  # Use realized regional sizes from the latest solve.
    ratio = np.maximum(np.asarray(sizes, dtype=float), 1.0e-12) / anchor  # Form dimensionless requested changes.
    predicted_elems = float(np.sum(np.maximum(state.elems, 1.0) * ratio ** (-float(problem.dim))))  # Apply geometric dimensional scaling.
    eq_per_elem = float(state.n_equations) / max(float(np.sum(state.elems)), 1.0)  # Calibrate the count model on the latest real mesh.
    return max(predicted_elems * eq_per_elem, 1.0)  # Return a positive resource prediction.
def _project_sizes(state: WorldState, raw_sizes: np.ndarray, problem, config: ToolGatewayConfig) -> tuple[np.ndarray, float, float]:  # Project one candidate toward the hard budget.
    target = float(config.budget_safety * state.budget)  # Reserve deterministic solver headroom.
    predicted = _resource_prediction(state, raw_sizes, problem)  # Evaluate the unprojected candidate.
    scale = float((predicted / max(target, 1.0)) ** (1.0 / float(problem.dim)))  # Solve the global resource law in closed form.
    projected = np.clip(np.asarray(raw_sizes, dtype=float) * scale, float(problem.h_min), float(problem.h0))  # Apply one global correction.
    projected_prediction = _resource_prediction(state, projected, problem)  # Re-evaluate after clipping.
    return projected, projected_prediction, scale  # Return exact sizes, revised count, and audit scale.
def _remainder_size(partition: Partition, sizes: np.ndarray) -> float:  # Select the unpainted-volume size deterministically.
    for seed, size in zip(partition.seeds, sizes):  # Search in stable partition order.
        if seed.origin == "coarse":  # Prefer the explicitly declared remainder seed.
            return float(size)  # Return its certified size.
    return float(np.max(sizes))  # Fall back to the coarsest certified region when no remainder exists.
def _audit_id(state_id: str, action: WorldAction, grades: np.ndarray) -> str:  # Build an idempotent action request hash.
    digest = hashlib.sha256()  # Start a cryptographic audit digest.
    digest.update(state_id.encode("ascii"))  # Bind the request to one state.
    digest.update(action.action_id.encode("utf-8"))  # Bind the request to one planner action.
    digest.update(action.kind.encode("ascii"))  # Bind the request to one execution path.
    digest.update(np.ascontiguousarray(grades, dtype=np.int64).tobytes())  # Bind the request to exact grades.
    return digest.hexdigest()[:24]  # Return a compact identifier for records and MCP wrappers.
class DeterministicToolGateway:  # Own every numerical parameter between the VLA planner and Gmsh.
    mapping_version = "wm-vla-grade-map-v1"  # Version the discrete-to-physical contract.
    def __init__(self, problem, config: ToolGatewayConfig | None = None) -> None:  # Bind one gateway to one immutable problem.
        self.problem = problem  # Store the geometry, limits, loads, and constraints.
        self.config = config or ToolGatewayConfig()  # Store deterministic certification policy.
    def validate(self, state: WorldState, action: WorldAction) -> np.ndarray:  # Validate one MCP-style action request.
        if action.kind not in ("region", "dorfler", "stop"):  # Restrict execution to declared tool methods.
            raise ValueError(f"unsupported world action kind {action.kind!r}")  # Reject hidden parameter channels.
        grades = action.next_grades(state)  # Validate delta shape and range through the state contract.
        if action.kind == "region" and action.n_changed > self.config.max_changed_regions:  # Bound learned action sparsity.
            raise ValueError("world action changes too many regions in one step")  # Force large changes to emerge over multiple states.
        return grades  # Return the only accepted numerical intent.
    def preview(self, state: WorldState, action: WorldAction) -> ToolPreview:  # Produce cheap deterministic inputs for world-model rollouts.
        grades = self.validate(state, action)  # Validate before any parameter mapping.
        if action.kind == "dorfler":  # Approximate the exact marked-node action for world-model comparison.
            marked = np.asarray(action.deltas, dtype=int) < 0  # Recover the region-level bulk preview mask.
            raw_sizes = np.clip(np.asarray(state.sizes, dtype=float) * np.where(marked, float(self.config.dorfler_factor), 1.0), float(self.problem.h_min), float(self.problem.h0))  # Give the safety action an optimistic faithful refinement preview.
        else:  # Resolve ordinary region actions through the versioned grade table.
            raw_sizes = _grade_sizes(self.problem, grades)  # Map levels through the versioned table.
        if action.source == "hold" or action.kind == "stop":  # Preserve an explicit no-change decision exactly.
            sizes = np.asarray(state.sizes, dtype=float).copy()  # Retain the measured current allocation.
            predicted = float(state.n_equations)  # Retain the measured current resource use.
            scale = 1.0  # Record that no numerical projection occurred.
        else:  # Project an actual redistribution toward the safe resource target.
            sizes, predicted, scale = _project_sizes(state, raw_sizes, self.problem, self.config)  # Enforce the global budget analytically.
        audit = {  # Emit a strict structured tool result.
            "mapping_version": self.mapping_version,  # Identify the numerical contract.
            "state_id": state.state_id,  # Identify the exact observed state.
            "action_id": action.action_id,  # Identify the planner action.
            "request_id": _audit_id(state.state_id, action, grades),  # Make repeated calls idempotently auditable.
            "global_scale": float(scale),  # Record the closed-form projection.
            "predicted_equations": float(predicted),  # Record the calibrated resource result.
            "continuous_input_from_model": False,  # Assert that the model supplied no mesh size.
        }  # Close the structured audit payload.
        return ToolPreview(action=action, grades=grades, sizes=sizes, n_equations=float(predicted), audit=audit)  # Return rollout input.
    def build_initial(self, partition: Partition, grades: np.ndarray, budget: int) -> CertifiedAction:  # Certify the first semantic mesh against an explicit equation cap.
        action = WorldAction(action_id="semantic_initial", deltas=tuple(0 for _ in partition.seeds), kind="region", source="vision_prior", rationale="geometry-visible bridge features")  # Create an auditable initial action.
        checked = np.array([parse_grade(int(value), "initial grade") for value in np.asarray(grades).ravel()], dtype=int)  # Validate initial levels.
        sizes = _grade_sizes(self.problem, checked)  # Convert only through the tool-owned table.
        drawings = list(drawings_with_sizes(list(partition.drawings), [seed.name for seed in partition.seeds], sizes))  # Materialize visual regions.
        remainder = _remainder_size(partition, sizes)  # Materialize the unpainted volume.
        mesh = generate_mesh(self.problem, drawings_size_fn(drawings, remainder, self.problem))  # Generate the first candidate with Gmsh.
        passes = 1  # Count the first deterministic mesh generation.
        estimated = estimate_free_equations(mesh, self.problem)  # Count exact free translational equations.
        target = float(self.config.budget_safety * max(int(budget), 1))  # Define the safe equation target.
        while passes < self.config.max_mesh_passes and (estimated > target or estimated < self.config.lower_use_target * target):  # Correct large count mismatch only.
            scale = float((estimated / max(target, 1.0)) ** (1.0 / float(self.problem.dim)))  # Compute one closed-form global size correction.
            sizes = np.clip(sizes * scale, float(self.problem.h_min), float(self.problem.h0))  # Apply physical limits.
            drawings = list(drawings_with_sizes(drawings, [seed.name for seed in partition.seeds], sizes))  # Preserve region geometry and ranking.
            remainder = _remainder_size(partition, sizes)  # Update the remainder consistently.
            mesh = generate_mesh(self.problem, drawings_size_fn(drawings, remainder, self.problem))  # Regenerate only the selected initial mesh.
            passes += 1  # Account for the deterministic Gmsh correction.
            estimated = estimate_free_equations(mesh, self.problem)  # Recount exact free equations.
        request_id = hashlib.sha256((partition.problem.instance_id + "|" + ",".join(map(str, checked.tolist()))).encode("utf-8")).hexdigest()[:24]  # Hash the initial contract.
        audit = {  # Record the initial MCP-style response.
            "mapping_version": self.mapping_version,  # Identify the grade mapping.
            "state_id": "before_first_solve",  # Declare that no solution state was consumed.
            "action_id": action.action_id,  # Identify semantic initialization.
            "request_id": request_id,  # Support idempotent replay.
            "estimated_equations": int(estimated),  # Record exact pre-solve free-DOF count.
            "mesh_passes": int(passes),  # Record Gmsh work separately from global solves.
            "continuous_input_from_model": False,  # Assert the vision head supplied grades only.
        }  # Close the audit payload.
        return CertifiedAction(action=action, grades=checked, sizes=sizes, drawings=tuple(drawings), mesh=mesh, estimated_equations=int(estimated), mesh_passes=int(passes), audit=audit)  # Return the only allowed first mesh.
    def certify(self, partition: Partition, state: WorldState, action: WorldAction, current_mesh: Mesh, eta2: np.ndarray) -> CertifiedAction:  # Materialize one selected action and no alternatives.
        preview = self.preview(state, action)  # Resolve and audit the discrete action before meshing.
        if action.kind == "stop":  # Reject attempts to execute a planning-only stop token.
            raise ValueError("stop actions do not produce a mesh")  # Keep stopping outside numerical tools.
        if action.kind == "dorfler":  # Execute the faithful element-wise safety action.
            marked = dorfler_mark(np.asarray(eta2, dtype=float), float(self.config.dorfler_theta))  # Bulk-mark current element indicators.
            target_h = refine_size_map(current_mesh, marked, factor=float(self.config.dorfler_factor))  # Halve only marked-element nodes.
            mesh = generate_mesh(self.problem, NodalSizeField(current_mesh, target_h, gradation=self.config.gradation, h_min=self.problem.h_min, h_max=self.problem.h0))  # Remesh through Gmsh.
            passes = 1  # Count the first selected-action mesh.
            estimated = estimate_free_equations(mesh, self.problem)  # Count exact equations before the solver.
            target = float(self.config.budget_safety * state.budget)  # Reserve solver headroom.
            while passes < self.config.max_mesh_passes and estimated > target:  # Project only if faithful refinement exceeds the cap.
                scale = float((estimated / max(target, 1.0)) ** (1.0 / float(self.problem.dim)))  # Compute the minimum global coarsening estimate.
                target_h = np.clip(target_h * scale, float(self.problem.h_min), float(self.problem.h0))  # Preserve Dörfler ranking while meeting resources.
                mesh = generate_mesh(self.problem, NodalSizeField(current_mesh, target_h, gradation=self.config.gradation, h_min=self.problem.h_min, h_max=self.problem.h0))  # Regenerate the same marked pattern.
                passes += 1  # Account for one deterministic correction.
                estimated = estimate_free_equations(mesh, self.problem)  # Recount the generated mesh exactly.
            audit = dict(preview.audit)  # Start from the validated request.
            audit.update({"execution": "exact_element_dorfler", "n_marked": int(len(marked)), "estimated_equations": int(estimated), "mesh_passes": int(passes)})  # Record faithful fallback details.
            return CertifiedAction(action=action, grades=preview.grades, sizes=preview.sizes, drawings=tuple(partition.drawings), mesh=mesh, estimated_equations=int(estimated), mesh_passes=int(passes), audit=audit)  # Return the exact fallback mesh.
        sizes = np.asarray(preview.sizes, dtype=float).copy()  # Start the learned action from tool-owned sizes.
        drawings = list(drawings_with_sizes(list(partition.drawings), list(state.names), sizes))  # Keep polygons fixed while changing certified sizes.
        remainder = _remainder_size(partition, sizes)  # Apply the certified background size.
        mesh = generate_mesh(self.problem, drawings_size_fn(drawings, remainder, self.problem))  # Generate only the selected learned action.
        passes = 1  # Count selected-action Gmsh work.
        estimated = estimate_free_equations(mesh, self.problem)  # Count exact free equations.
        target = float(self.config.budget_safety * state.budget)  # Define the safe resource target.
        while passes < self.config.max_mesh_passes and (estimated > target or estimated < self.config.lower_use_target * target):  # Correct only material mismatch.
            scale = float((estimated / max(target, 1.0)) ** (1.0 / float(self.problem.dim)))  # Solve the observed mesh-count mismatch in closed form.
            sizes = np.clip(sizes * scale, float(self.problem.h_min), float(self.problem.h0))  # Apply one global correction without changing ranking.
            drawings = list(drawings_with_sizes(drawings, list(state.names), sizes))  # Preserve all region boundaries.
            remainder = _remainder_size(partition, sizes)  # Update the unpainted volume.
            mesh = generate_mesh(self.problem, drawings_size_fn(drawings, remainder, self.problem))  # Regenerate the selected action.
            passes += 1  # Account for deterministic Gmsh work.
            estimated = estimate_free_equations(mesh, self.problem)  # Recount exact equations.
        audit = dict(preview.audit)  # Copy the request audit.
        audit.update({"execution": "region_grade", "estimated_equations": int(estimated), "mesh_passes": int(passes)})  # Append exact materialization results.
        return CertifiedAction(action=action, grades=preview.grades, sizes=sizes, drawings=tuple(drawings), mesh=mesh, estimated_equations=int(estimated), mesh_passes=int(passes), audit=audit)  # Return the solver-ready mesh.
