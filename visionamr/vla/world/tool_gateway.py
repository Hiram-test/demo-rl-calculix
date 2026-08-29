"""Deterministic MCP-shaped tool gateway for exact adaptive-mesh action materialization."""  # Describe the parameter-certification boundary implemented below.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from dataclasses import dataclass  # Import immutable tool contracts.
import hashlib  # Import deterministic target-field hashing.
import inspect  # Import runtime signature adaptation for repository meshing utilities.
from typing import Any  # Import generic repository object types.
import numpy as np  # Import numerical mesh operations.
from ...baselines.dorfler import refine_size_map  # Reuse the repository exact-Dörfler target-field implementation.
from ...marking import dorfler_mark  # Reuse the repository bulk-marking implementation.
from ...mesher import generate_mesh  # Reuse exact Gmsh remeshing.
from ...sizefield import NodalSizeField  # Reuse the repository nodal size-field interpolator.
from .model import RegionAction, WorldState, semantic_persistence  # Import world-model state and safe action contracts.

@dataclass(frozen=True)
class ToolConfig:  # Configure deterministic action materialization.
    theta: float = 0.5  # Use one common Dörfler bulk parameter for all compared methods.
    refine_factor: float = 0.5  # Match the repository exact-Dörfler refinement factor.
    core_theta: float = 0.72  # Restrict extra future depth to a concentrated within-region error core.
    budget_safety: float = 1.0  # Enforce the requested active-equation cap during exact preflight.
    max_extra_depth: int = 2  # Bound the discrete future-hit depth accepted from planning.

@dataclass(frozen=True)
class ToolObservation:  # Store a real-solve observation used by planning and action materialization.
    problem: Any  # Store the finite-element problem contract.
    partition: Any  # Store the fixed semantic partition.
    post: Any  # Store the real finite-element post-processing result.
    record: Any  # Store the measured solve record.
    mesh: Any  # Store the solved mesh.
    eta2: np.ndarray  # Store elementwise squared error indicators.
    labels: np.ndarray  # Store one semantic region index per element.
    marked: np.ndarray  # Store the exact Dörfler element mask.
    state: WorldState  # Store the compact world-model state.

@dataclass(frozen=True)
class MeshCertificate:  # Store deterministic evidence for one materialized adaptive action.
    schema_version: str  # Store the stable tool-contract version.
    requested_action: tuple[int, ...]  # Store the planner-requested regional depths.
    executed_action: tuple[int, ...]  # Store the actually materialized regional depths.
    source: str  # Store world-model or Dörfler execution provenance.
    target_sha256: str  # Store the exact nodal target-field hash.
    base_target_included: bool  # Certify componentwise inclusion of the Dörfler target field.
    no_coarsening: bool  # Certify that the action never coarsens relative to Dörfler.
    estimated_equations: int  # Store the exact active-DOF count on the candidate mesh.
    equation_cap: int  # Store the requested active-equation cap.
    accepted: bool  # Report whether the world-model candidate passed all deterministic checks.
    reason: str  # Explain acceptance, fallback, or stop.

@dataclass(frozen=True)
class MaterializedAction:  # Store the candidate mesh and its deterministic certificate.
    mesh: Any | None  # Store the selected remesh or None when even Dörfler exceeds the cap.
    action: RegionAction  # Store the actually executed action.
    certificate: MeshCertificate  # Store exact parameter and resource evidence.
    base_estimated_equations: int  # Store the exact-Dörfler candidate resource count.

class MCPToolGateway:  # Expose inspect, observe, materialize, and certify operations without LLM numeric tuning.
    schema_version = "wmvla.mcp-tool.v1"  # Define the structured-output contract version.
    def __init__(self, config: ToolConfig | None = None) -> None:  # Initialize deterministic tool behavior.
        self.config = config or ToolConfig()  # Store immutable tool settings.
        if not 0.0 < self.config.theta <= 1.0:  # Validate the Dörfler bulk parameter.
            raise ValueError("theta must be in (0, 1]")  # Explain the invalid marking contract.
        if not 0.0 < self.config.refine_factor < 1.0:  # Validate the refinement factor.
            raise ValueError("refine_factor must be in (0, 1)")  # Explain the invalid refinement contract.
        if not 0.0 < self.config.core_theta <= 1.0:  # Validate concentrated-core selection.
            raise ValueError("core_theta must be in (0, 1]")  # Explain the invalid core contract.
    def inspect_case(self, problem: Any) -> dict[str, Any]:  # Return a structured read-only case description suitable for MCP exposure.
        return {"schema_version": self.schema_version, "name": str(problem.name), "dimension": int(problem.dim), "bbox": [float(value) for value in problem.bbox], "h0": float(problem.h0), "h_ref": float(problem.h_ref), "h_min": float(problem.h_min), "material": {"E": float(problem.material.E), "nu": float(problem.material.nu)}, "features": [str(getattr(feature, "name", "feature")) for feature in problem.features], "parameters": dict(problem.params)}  # Return only validated repository values rather than model-generated numbers.
    def _mesh_arrays(self, mesh: Any) -> tuple[np.ndarray, np.ndarray]:  # Normalize repository mesh arrays.
        points = np.asarray(getattr(mesh, "points", getattr(mesh, "nodes", None)), dtype=float)  # Read nodal coordinates from the supported mesh attributes.
        cells = np.asarray(getattr(mesh, "cells", getattr(mesh, "elements", None)), dtype=int)  # Read element connectivity from the supported mesh attributes.
        if points.ndim != 2 or cells.ndim != 2:  # Reject unsupported mesh objects.
            raise ValueError("mesh must expose two-dimensional points and cells arrays")  # Explain the mesh contract.
        return points, cells  # Return normalized mesh arrays.
    def _record_equations(self, record: Any, mesh: Any) -> int:  # Read measured active equations with a deterministic fallback.
        for name in ("n_equations", "equations", "neq"):  # Inspect known solve-record field names.
            if hasattr(record, name):  # Use the first available measured field.
                value = int(getattr(record, name))  # Convert the measured value to an integer.
                if value > 0:  # Accept only a positive measured equation count.
                    return value  # Return the measured resource quantity.
        return self.estimate_equations(getattr(record, "problem", None), mesh)  # Fall back to exact active-DOF counting when record metadata is absent.
    def _partition_names(self, partition: Any, count: int) -> tuple[str, ...]:  # Recover stable semantic region names from repository partitions.
        if hasattr(partition, "names"):  # Prefer an explicit partition-name sequence.
            names = tuple(str(value) for value in getattr(partition, "names"))  # Normalize explicit names.
        elif hasattr(partition, "regions"):  # Fall back to region objects.
            names = tuple(str(getattr(region, "name", f"region_{index}")) for index, region in enumerate(getattr(partition, "regions")))  # Extract object names deterministically.
        else:  # Handle minimal partition implementations.
            names = tuple(f"region_{index}" for index in range(count))  # Construct stable generic identifiers.
        if len(names) < count:  # Extend incomplete name sequences defensively.
            names = names + tuple(f"region_{index}" for index in range(len(names), count))  # Preserve existing names and append deterministic identifiers.
        return names[:count]  # Return exactly one name per observed label.
    def _element_geometry(self, mesh: Any) -> tuple[np.ndarray, np.ndarray]:  # Measure element sizes and geometric volumes.
        points, cells = self._mesh_arrays(mesh)  # Read normalized mesh arrays.
        coordinates = points[cells]  # Gather nodal coordinates for every element.
        if cells.shape[1] >= 4 and points.shape[1] >= 3:  # Use tetrahedral volume and six-edge size in three dimensions.
            a = coordinates[:, 1, :3] - coordinates[:, 0, :3]  # Form the first tetrahedral edge vector.
            b = coordinates[:, 2, :3] - coordinates[:, 0, :3]  # Form the second tetrahedral edge vector.
            c = coordinates[:, 3, :3] - coordinates[:, 0, :3]  # Form the third tetrahedral edge vector.
            volumes = np.abs(np.einsum("ij,ij->i", np.cross(a, b), c)) / 6.0  # Compute tetrahedral volumes.
        else:  # Use triangular area in two dimensions.
            a = coordinates[:, 1, :2] - coordinates[:, 0, :2]  # Form the first triangle edge vector.
            b = coordinates[:, 2, :2] - coordinates[:, 0, :2]  # Form the second triangle edge vector.
            volumes = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])  # Compute triangle areas.
        edge_lengths: list[np.ndarray] = []  # Collect all pairwise element-edge lengths.
        for left in range(cells.shape[1]):  # Iterate over local nodes.
            for right in range(left + 1, cells.shape[1]):  # Iterate over unique local-node pairs.
                edge_lengths.append(np.linalg.norm(coordinates[:, left, : points.shape[1]] - coordinates[:, right, : points.shape[1]], axis=1))  # Measure one edge family.
        sizes = np.median(np.stack(edge_lengths, axis=1), axis=1)  # Use median edge length as a robust element-size measure.
        return sizes, np.maximum(volumes, 1.0e-30)  # Return positive element sizes and volumes.
    def _stress_values(self, post: Any, n_elements: int) -> np.ndarray:  # Recover an elementwise stress-severity vector.
        for name in ("von_mises", "vm", "stress_vm", "element_von_mises"):  # Inspect supported post-processing field names.
            if hasattr(post, name):  # Use the first available stress field.
                values = np.asarray(getattr(post, name), dtype=float).reshape(-1)  # Normalize the stress vector.
                if values.size == n_elements:  # Accept elementwise stress directly.
                    return values  # Return the measured stress vector.
        return np.zeros(n_elements, dtype=float)  # Preserve planner operation when stress is unavailable.
    def _adjacency(self, cells: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:  # Build a normalized semantic region graph from shared mesh nodes.
        node_regions: dict[int, set[int]] = {}  # Map each node to incident semantic regions.
        for element, region in zip(cells, labels, strict=True):  # Traverse element-region assignments.
            for node in element:  # Traverse nodes of the current element.
                node_regions.setdefault(int(node), set()).add(int(region))  # Register region incidence at this node.
        graph = np.zeros((count, count), dtype=float)  # Allocate the symmetric region graph.
        for regions in node_regions.values():  # Inspect each node's incident region set.
            ordered = sorted(regions)  # Ensure deterministic pair construction.
            for left_index, left in enumerate(ordered):  # Traverse the first region in each pair.
                for right in ordered[left_index + 1 :]:  # Traverse distinct adjacent regions.
                    graph[left, right] += 1.0  # Count one shared-node interaction.
                    graph[right, left] += 1.0  # Preserve graph symmetry.
        row_sum = np.sum(graph, axis=1)  # Compute regional interaction totals.
        normalized = np.divide(graph, row_sum[:, None], out=np.zeros_like(graph), where=row_sum[:, None] > 0.0)  # Convert counts to row-normalized coupling weights.
        return normalized  # Return the finite-element-derived semantic graph.
    def _normalize_marked(self, marked: Any, n_elements: int) -> np.ndarray:  # Normalize Dörfler output to a Boolean element mask.
        values = np.asarray(marked)  # Materialize the marker output.
        if values.dtype == bool and values.shape == (n_elements,):  # Accept an existing Boolean mask.
            return values.copy()  # Return an independent mask.
        mask = np.zeros(n_elements, dtype=bool)  # Allocate a Boolean mask for index output.
        mask[np.asarray(values, dtype=int).reshape(-1)] = True  # Mark returned element indices.
        return mask  # Return the normalized exact-Dörfler mask.
    def observe_solve(self, problem: Any, partition: Any, post: Any, record: Any, eta2: np.ndarray, hit_count: np.ndarray | None, step: int) -> ToolObservation:  # Convert one real solve into a world-model state.
        mesh = getattr(post, "mesh", getattr(record, "mesh", None))  # Recover the solved mesh from post-processing or record metadata.
        if mesh is None:  # Reject observations without a solved mesh.
            raise ValueError("post-processing result must expose its solved mesh")  # Explain the observation contract.
        points, cells = self._mesh_arrays(mesh)  # Read normalized mesh arrays.
        indicator = np.asarray(eta2, dtype=float).reshape(-1)  # Normalize the squared error indicators.
        if indicator.shape != (cells.shape[0],):  # Require one indicator per element.
            raise ValueError("eta2 must contain one value per element")  # Explain the estimator contract.
        labels = np.asarray(partition.assign(mesh), dtype=int).reshape(-1)  # Assign every current element to the fixed semantic partition.
        if labels.shape != (cells.shape[0],) or np.any(labels < 0):  # Validate the partition result.
            raise ValueError("partition must assign one non-negative region index per element")  # Explain the partition contract.
        count = int(np.max(labels)) + 1  # Determine the number of observed semantic regions.
        names = self._partition_names(partition, count)  # Recover stable names in label order.
        marked = self._normalize_marked(dorfler_mark(indicator, self.config.theta), cells.shape[0])  # Compute the mandatory exact-Dörfler element mask.
        element_sizes, element_volume = self._element_geometry(mesh)  # Measure regional geometry directly from the solved mesh.
        stress = self._stress_values(post, cells.shape[0])  # Recover an optional stress-severity vector.
        err_sum = np.zeros(count, dtype=float)  # Allocate regional squared-indicator sums.
        elems = np.zeros(count, dtype=float)  # Allocate regional element counts.
        sizes = np.zeros(count, dtype=float)  # Allocate regional measured sizes.
        vm_max = np.zeros(count, dtype=float)  # Allocate regional peak stress values.
        volume = np.zeros(count, dtype=float)  # Allocate regional geometric volumes.
        marked_error_fraction = np.zeros(count, dtype=float)  # Allocate within-region Dörfler error fractions.
        marked_element_fraction = np.zeros(count, dtype=float)  # Allocate within-region Dörfler element fractions.
        hits = np.zeros(count, dtype=float) if hit_count is None else np.asarray(hit_count, dtype=float).copy()  # Restore persistent hit counts or initialize them.
        if hits.shape != (count,):  # Reset incompatible history after an invalid partition change.
            hits = np.zeros(count, dtype=float)  # Preserve a valid state rather than misaligning histories.
        for region in range(count):  # Aggregate one semantic region at a time.
            mask = labels == region  # Select elements assigned to the region.
            region_marked = mask & marked  # Select exact-Dörfler elements inside the region.
            elems[region] = float(np.count_nonzero(mask))  # Count regional elements.
            err_sum[region] = float(np.sum(indicator[mask]))  # Sum regional squared indicators.
            sizes[region] = float(np.median(element_sizes[mask])) if np.any(mask) else float(problem.h0)  # Measure a robust regional size.
            vm_max[region] = float(np.max(stress[mask])) if np.any(mask) else 0.0  # Measure regional peak stress.
            volume[region] = float(np.sum(element_volume[mask]))  # Sum regional geometric volume.
            marked_error_fraction[region] = float(np.sum(indicator[region_marked]) / max(err_sum[region], 1.0e-30))  # Measure the selected share of regional error.
            marked_element_fraction[region] = float(np.count_nonzero(region_marked) / max(np.count_nonzero(mask), 1))  # Measure the selected share of regional elements.
            if np.any(region_marked):  # Detect an actual Dörfler hit in this real solve.
                hits[region] += 1.0  # Update recurrence evidence for the world model.
        adjacency = self._adjacency(cells, labels, count)  # Build the current semantic region graph.
        n_equations = self._record_equations(record, mesh)  # Read the measured active-equation count.
        eq_per_elem = float(n_equations / max(cells.shape[0], 1))  # Calibrate local resource conversion from the real solver.
        state = WorldState(names=names, err_sum=err_sum, elems=elems, sizes=sizes, vm_max=vm_max, volume=volume, adjacency=adjacency, dorfler_error_fraction=marked_error_fraction, dorfler_element_fraction=marked_element_fraction, hit_count=hits, n_equations=n_equations, eq_per_elem=eq_per_elem, h_min=float(problem.h_min), h0=float(problem.h0), dim=int(problem.dim), step=int(step))  # Construct the compact action-conditioned world state.
        return ToolObservation(problem=problem, partition=partition, post=post, record=record, mesh=mesh, eta2=indicator, labels=labels, marked=marked, state=state)  # Return the complete tool observation.
    def estimate_equations(self, problem: Any, mesh: Any) -> int:  # Count active displacement degrees of freedom on a candidate mesh.
        points, _ = self._mesh_arrays(mesh)  # Read candidate nodal coordinates.
        dimension = int(getattr(problem, "dim", points.shape[1]))  # Recover the supported displacement dimension.
        fixed = np.zeros((points.shape[0], dimension), dtype=bool)  # Allocate one Boolean entry per nodal displacement degree of freedom.
        for constraint in getattr(problem, "constraints", []):  # Apply every mesh-independent boundary constraint.
            mask = np.asarray(constraint.predicate(points), dtype=bool).reshape(-1)  # Evaluate the constraint on candidate nodes.
            if mask.shape != (points.shape[0],):  # Reject malformed boundary predicates.
                raise ValueError("constraint predicate must return one Boolean per node")  # Explain the boundary contract.
            for dof in constraint.dofs:  # Apply each one-based constrained displacement component.
                fixed[mask, int(dof) - 1] = True  # Mark the corresponding candidate degrees of freedom as constrained.
        return int(points.shape[0] * dimension - np.count_nonzero(fixed))  # Return the exact active displacement-DOF count.
    def _current_nodal_sizes(self, mesh: Any) -> np.ndarray:  # Recover a stable nodal size estimate for extra-depth materialization.
        points, cells = self._mesh_arrays(mesh)  # Read normalized mesh arrays.
        for name in ("node_sizes", "nodal_sizes", "sizes"):  # Inspect known repository mesh-size attributes.
            if hasattr(mesh, name):  # Use an explicit nodal size vector when available.
                values = np.asarray(getattr(mesh, name), dtype=float).reshape(-1)  # Normalize the size vector.
                if values.shape == (points.shape[0],):  # Accept only one size per node.
                    return values.copy()  # Return independent nodal sizes.
        element_sizes, _ = self._element_geometry(mesh)  # Measure current element sizes when metadata is absent.
        accumulated = np.zeros(points.shape[0], dtype=float)  # Accumulate incident element sizes at each node.
        counts = np.zeros(points.shape[0], dtype=float)  # Count incident elements at each node.
        for element, size in zip(cells, element_sizes, strict=True):  # Traverse element-size pairs.
            accumulated[element] += size  # Add the element size to all incident nodes.
            counts[element] += 1.0  # Count one incident element for all nodes.
        return np.divide(accumulated, counts, out=np.full(points.shape[0], float(np.median(element_sizes))), where=counts > 0.0)  # Return average incident element sizes.
    def _base_target(self, observation: ToolObservation) -> np.ndarray:  # Build the repository exact-Dörfler nodal target field.
        try:  # Prefer the repository helper's documented keyword call.
            target = refine_size_map(observation.mesh, observation.marked, factor=self.config.refine_factor)  # Generate the standard Dörfler target field.
        except TypeError:  # Support positional factor signatures from earlier repository commits.
            target = refine_size_map(observation.mesh, observation.marked, self.config.refine_factor)  # Generate the same standard target field positionally.
        values = np.asarray(target, dtype=float).reshape(-1)  # Normalize the returned nodal target vector.
        points, _ = self._mesh_arrays(observation.mesh)  # Read the current node count.
        if values.shape != (points.shape[0],):  # Reject non-nodal target fields.
            raise ValueError("refine_size_map must return one target size per node")  # Explain the exact-Dörfler tool contract.
        return np.maximum(float(observation.problem.h_min), values)  # Enforce the problem minimum size.
    def _world_target(self, observation: ToolObservation, action: RegionAction, base_target: np.ndarray) -> np.ndarray:  # Add bounded future-hit depth to the exact-Dörfler target.
        action.validate(observation.state, max_depth=self.config.max_extra_depth)  # Reject unsafe action vectors.
        points, cells = self._mesh_arrays(observation.mesh)  # Read current mesh arrays.
        current = self._current_nodal_sizes(observation.mesh)  # Recover current nodal sizes.
        target = base_target.copy()  # Start from the exact-Dörfler target field.
        for region, depth in enumerate(action.extra_depth):  # Materialize each bounded semantic action.
            if depth <= 0:  # Skip regions delegated entirely to Dörfler.
                continue  # Preserve the baseline target unchanged.
            if semantic_persistence(observation.state.names[region]) <= 0.0:  # Defensively reject generic field actions.
                raise ValueError("world-model depth cannot target a generic field region")  # Preserve the semantic action boundary.
            region_elements = np.flatnonzero(observation.labels == region)  # Select all elements in the semantic mechanism.
            if region_elements.size == 0:  # Skip empty partition regions.
                continue  # Preserve target-field validity.
            local_error = observation.eta2[region_elements]  # Read within-region indicator contributions.
            order = region_elements[np.argsort(-local_error, kind="mergesort")]  # Rank the regional hotspot deterministically.
            cumulative = np.cumsum(observation.eta2[order])  # Accumulate ranked regional error.
            threshold = self.config.core_theta * max(float(np.sum(local_error)), 1.0e-30)  # Define the concentrated future-hit support.
            count = int(np.searchsorted(cumulative, threshold, side="left") + 1)  # Select the smallest regional core meeting the threshold.
            core_elements = order[: min(count, order.size)]  # Extract the concentrated semantic core.
            selected_elements = np.unique(np.concatenate((core_elements, np.flatnonzero((observation.labels == region) & observation.marked))))  # Include every currently marked element in the selected mechanism.
            selected_nodes = np.unique(cells[selected_elements].reshape(-1))  # Convert selected elements to nodal targets.
            desired = current[selected_nodes] * self.config.refine_factor ** (1 + int(depth))  # Compress the current hit plus requested future-hit depth.
            target[selected_nodes] = np.minimum(target[selected_nodes], desired)  # Refine without ever coarsening the exact-Dörfler target.
        return np.maximum(float(observation.problem.h_min), target)  # Enforce the admissible minimum target size.
    def _field(self, mesh: Any, target: np.ndarray, problem: Any) -> Any:  # Construct a repository-compatible nodal size field by signature inspection.
        points, _ = self._mesh_arrays(mesh)  # Read source nodal coordinates.
        signature = inspect.signature(NodalSizeField)  # Inspect the active repository constructor.
        kwargs: dict[str, Any] = {}  # Collect supported keyword arguments.
        for name in signature.parameters:  # Map known constructor fields without guessing numeric parameters.
            if name in ("points", "nodes", "coordinates"):  # Match coordinate arguments.
                kwargs[name] = points  # Supply solved-mesh coordinates.
            elif name in ("sizes", "values", "target_sizes", "h"):  # Match nodal target arguments.
                kwargs[name] = target  # Supply certified nodal target sizes.
            elif name in ("h_min", "minimum", "min_size"):  # Match minimum-size arguments.
                kwargs[name] = float(problem.h_min)  # Supply the problem's exact minimum size.
            elif name in ("h_max", "maximum", "max_size"):  # Match maximum-size arguments.
                kwargs[name] = float(problem.h0)  # Supply the problem's exact initial size.
        try:  # Prefer keyword construction for auditable parameter binding.
            return NodalSizeField(**kwargs)  # Construct the repository size field.
        except TypeError:  # Support the common two-positional-argument implementation.
            return NodalSizeField(points, target)  # Construct the same field without model-generated parameters.
    def _generate(self, problem: Any, source_mesh: Any, target: np.ndarray) -> Any:  # Generate one exact Gmsh candidate mesh.
        field = self._field(source_mesh, target, problem)  # Construct the certified nodal field.
        try:  # Prefer the repository documented keyword interface.
            return generate_mesh(problem, size_field=field)  # Regenerate a conformal mesh with the target field.
        except TypeError:  # Support positional size-field signatures from earlier commits.
            return generate_mesh(problem, field)  # Regenerate the same conformal mesh positionally.
    def materialize_action(self, observation: ToolObservation, action: RegionAction, n_equation_cap: int) -> MaterializedAction:  # Materialize, preflight, and certify one planner action.
        action.validate(observation.state, max_depth=self.config.max_extra_depth)  # Validate the discrete semantic action.
        cap = int(self.config.budget_safety * int(n_equation_cap))  # Compute the exact deterministic resource cap.
        base_target = self._base_target(observation)  # Build the immutable exact-Dörfler target field.
        base_mesh = self._generate(observation.problem, observation.mesh, base_target)  # Generate the exact-Dörfler candidate without solving it.
        base_equations = self.estimate_equations(observation.problem, base_mesh)  # Count exact candidate active displacement equations.
        base_hash = hashlib.sha256(np.asarray(base_target, dtype="<f8").tobytes()).hexdigest()  # Hash the exact nodal target field.
        if base_equations > cap:  # Stop when even the required Dörfler action exceeds the resource budget.
            certificate = MeshCertificate(self.schema_version, action.extra_depth, RegionAction.dorfler(observation.state).extra_depth, "stop", base_hash, True, True, base_equations, cap, False, "dorfler_candidate_exceeds_cap")  # Record the deterministic stop reason.
            return MaterializedAction(None, RegionAction.dorfler(observation.state), certificate, base_equations)  # Return no candidate mesh.
        if action.is_dorfler_only:  # Execute the exact-Dörfler safety action directly.
            certificate = MeshCertificate(self.schema_version, action.extra_depth, action.extra_depth, "dorfler", base_hash, True, True, base_equations, cap, True, "exact_dorfler")  # Certify the baseline target and resource count.
            return MaterializedAction(base_mesh, action, certificate, base_equations)  # Return the exact-Dörfler candidate.
        world_target = self._world_target(observation, action, base_target)  # Materialize bounded semantic future-hit depth.
        included = bool(np.all(world_target <= base_target + 1.0e-12))  # Verify componentwise inclusion of the exact-Dörfler target.
        no_coarsening = included  # Equate smaller target size with no coarsening relative to Dörfler.
        if not included:  # Reject any implementation defect that violates the Dörfler floor.
            raise RuntimeError("world target failed exact-Dörfler inclusion check")  # Stop before meshing or solving an unsafe action.
        world_mesh = self._generate(observation.problem, observation.mesh, world_target)  # Generate the world-model candidate without a finite-element solve.
        world_equations = self.estimate_equations(observation.problem, world_mesh)  # Count exact candidate active displacement equations.
        world_hash = hashlib.sha256(np.asarray(world_target, dtype="<f8").tobytes()).hexdigest()  # Hash the world-model nodal target field.
        if world_equations > cap:  # Reject exact resource-cap violations after Gmsh behavior is known.
            certificate = MeshCertificate(self.schema_version, action.extra_depth, RegionAction.dorfler(observation.state).extra_depth, "dorfler_fallback", base_hash, True, True, base_equations, cap, False, "world_candidate_exceeds_cap")  # Record the precise fallback reason.
            return MaterializedAction(base_mesh, RegionAction.dorfler(observation.state), certificate, base_equations)  # Execute exact Dörfler instead of the rejected world action.
        certificate = MeshCertificate(self.schema_version, action.extra_depth, action.extra_depth, "world_model", world_hash, included, no_coarsening, world_equations, cap, True, "world_candidate_certified")  # Certify the accepted world-model target and mesh.
        return MaterializedAction(world_mesh, action, certificate, base_equations)  # Return the exact preflighted world-model candidate.
