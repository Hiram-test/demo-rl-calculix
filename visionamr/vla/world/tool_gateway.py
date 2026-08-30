"""Deterministic MCP-shaped gateway for exact adaptive-mesh action materialization."""  # Describe the parameter-certification boundary implemented below.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from dataclasses import dataclass  # Import immutable tool contracts.
import hashlib  # Import deterministic target-field hashing.
import time  # Import a monotonic timer for separated parameter and Gmsh costs.
from typing import Any  # Import generic repository object types.
import numpy as np  # Import numerical mesh operations.
from ...baselines.dorfler import refine_size_map  # Reuse the repository exact-Dörfler target-field implementation.
from ...marking import dorfler_mark  # Reuse the repository bulk-marking implementation.
from ...mesher import generate_mesh  # Reuse exact Gmsh remeshing.
from ...sizefield import NodalSizeField  # Reuse the repository nodal size-field interpolator.
from .model import RegionAction, WorldState, semantic_persistence  # Import world-model state and safe action contracts.

@dataclass(frozen=True)  # Make tool settings immutable.
class ToolConfig:  # Configure deterministic action materialization.
    theta: float = 0.5  # Use one common Dörfler bulk parameter for all compared methods.
    refine_factor: float = 0.5  # Match the repository exact-Dörfler refinement factor.
    core_theta: float = 0.72  # Restrict extra future depth to a concentrated within-region error core.
    budget_safety: float = 1.0  # Enforce the requested active-equation cap during exact preflight.
    max_extra_depth: int = 2  # Bound the discrete future-hit depth accepted from planning.

@dataclass(frozen=True)  # Make real-solve observations immutable.
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

@dataclass(frozen=True)  # Make certificates immutable and hashable.
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
    base_raw_target_sha256: str = ""  # Bind the complete unsmoothed Dörfler nodal target vector.
    world_raw_target_sha256: str | None = None  # Bind the complete unsmoothed proactive target vector when one was proposed.
    base_compiled_field_sha256: str = ""  # Bind source nodes, compiled Dörfler values, interpolation arity, and gradation.
    world_compiled_field_sha256: str | None = None  # Bind the equivalently compiled proactive field when one was proposed.
    compiled_field_node_count: int = 0  # Record the exact common source-node count used by both compiled fields.
    compiled_field_gradation: float = 1.0  # Record the identical Lipschitz gradation used for both field compilations.
    compiled_max_dorfler_violation: float = 0.0  # Record max(max(h_WM_compiled-h_D_compiled),0) over all source nodes.
    compiled_dorfler_included: bool = True  # Certify nodewise dominance after clipping and Lipschitz compilation.

@dataclass(frozen=True)  # Make materialized actions immutable.
class MaterializedAction:  # Store the candidate mesh and its deterministic certificate.
    mesh: Any | None  # Store the selected remesh or None when even Dörfler exceeds the cap.
    action: RegionAction  # Store the actually executed action.
    certificate: MeshCertificate  # Store exact parameter and resource evidence.
    base_estimated_equations: int  # Store the exact-Dörfler candidate resource count.
    timing_s: dict[str, float]  # Store separated deterministic parameter-tool and Gmsh wall times.

class MCPToolGateway:  # Expose inspect, observe, materialize, and certify operations without LLM numeric tuning.
    schema_version = "wmvla.mcp-tool.v2"  # Version complete raw and post-gradation nodewise dominance receipts.
    def __init__(self, config: ToolConfig | None = None) -> None:  # Initialize deterministic tool behavior.
        self.config = config or ToolConfig()  # Store immutable tool settings.
        if not 0.0 < self.config.theta <= 1.0:  # Validate the Dörfler bulk parameter.
            raise ValueError("theta must be in (0, 1]")  # Explain the invalid marking contract.
        if not 0.0 < self.config.refine_factor < 1.0:  # Validate the refinement factor.
            raise ValueError("refine_factor must be in (0, 1)")  # Explain the invalid refinement contract.
    def inspect_case(self, problem: Any) -> dict[str, Any]:  # Return a structured read-only case description suitable for MCP exposure.
        return {"schema_version": self.schema_version, "name": str(problem.name), "dimension": int(problem.dim), "bbox": [float(value) for value in problem.bbox], "h0": float(problem.h0), "h_ref": float(problem.h_ref), "h_min": float(problem.h_min), "material": {"E": float(problem.material.E), "nu": float(problem.material.nu)}, "features": [str(getattr(feature, "name", "feature")) for feature in problem.features], "parameters": dict(problem.params)}  # Return only validated repository values rather than model-generated numbers.
    def _arrays(self, mesh: Any) -> tuple[np.ndarray, np.ndarray]:  # Normalize repository mesh arrays.
        points = np.asarray(getattr(mesh, "points", getattr(mesh, "nodes", None)), dtype=float)  # Read nodal coordinates.
        cells = np.asarray(getattr(mesh, "cells", getattr(mesh, "elements", None)), dtype=int)  # Read element connectivity.
        if points.ndim != 2 or cells.ndim != 2:  # Reject unsupported mesh objects.
            raise ValueError("mesh must expose two-dimensional points and cells arrays")  # Explain the mesh contract.
        return points, cells  # Return normalized arrays.
    def _names(self, partition: Any, count: int) -> tuple[str, ...]:  # Recover stable semantic names.
        names = tuple(str(value) for value in getattr(partition, "names", ()))  # Prefer explicit partition names.
        if not names and hasattr(partition, "regions"):  # Fall back to region objects.
            names = tuple(str(getattr(region, "name", f"region_{index}")) for index, region in enumerate(partition.regions))  # Extract region names deterministically.
        names = names + tuple(f"region_{index}" for index in range(len(names), count))  # Extend incomplete sequences defensively.
        return names[:count]  # Return exactly one name per observed label.
    def _marked_mask(self, eta2: np.ndarray) -> np.ndarray:  # Normalize repository Dörfler output to a Boolean mask.
        raw = np.asarray(dorfler_mark(eta2, self.config.theta))  # Execute exact repository bulk marking.
        if raw.dtype == bool and raw.shape == eta2.shape:  # Accept an existing Boolean mask.
            return raw.copy()  # Return an independent mask.
        mask = np.zeros(eta2.size, dtype=bool)  # Allocate a Boolean mask for index output.
        mask[np.asarray(raw, dtype=int).reshape(-1)] = True  # Mark returned element indices.
        return mask  # Return the normalized mask.
    def _element_measures(self, mesh: Any) -> tuple[np.ndarray, np.ndarray]:  # Measure robust element sizes and volumes.
        points, cells = self._arrays(mesh)  # Read normalized mesh arrays.
        xyz = points[cells]  # Gather element nodal coordinates.
        edges = [np.linalg.norm(xyz[:, left, :] - xyz[:, right, :], axis=1) for left in range(cells.shape[1]) for right in range(left + 1, cells.shape[1])]  # Measure all local edge families.
        sizes = np.median(np.stack(edges, axis=1), axis=1)  # Use median edge length as a robust size.
        if cells.shape[1] >= 4 and points.shape[1] >= 3:  # Compute tetrahedral volume in three dimensions.
            a = xyz[:, 1, :3] - xyz[:, 0, :3]  # Form the first tetrahedral edge.
            b = xyz[:, 2, :3] - xyz[:, 0, :3]  # Form the second tetrahedral edge.
            c = xyz[:, 3, :3] - xyz[:, 0, :3]  # Form the third tetrahedral edge.
            volumes = np.abs(np.einsum("ij,ij->i", np.cross(a, b), c)) / 6.0  # Evaluate positive tetrahedral volumes.
        else:  # Compute triangle area in two dimensions.
            a = xyz[:, 1, :2] - xyz[:, 0, :2]  # Form the first triangle edge.
            b = xyz[:, 2, :2] - xyz[:, 0, :2]  # Form the second triangle edge.
            volumes = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])  # Evaluate positive triangle areas.
        return sizes, np.maximum(volumes, 1.0e-30)  # Return positive measures.
    def _stress(self, post: Any, count: int) -> np.ndarray:  # Recover an elementwise stress-severity vector.
        for name in ("vm_elem", "von_mises", "vm", "stress_vm", "element_von_mises"):  # Prefer the repository PostState von Mises field before compatibility aliases.
            values = getattr(post, name, None)  # Read the candidate stress field.
            if values is not None and np.asarray(values).size == count:  # Accept one value per element.
                return np.asarray(values, dtype=float).reshape(-1)  # Return normalized element stress.
        return np.zeros(count, dtype=float)  # Preserve operation when stress metadata is absent.
    def _adjacency(self, cells: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:  # Build a normalized region graph from shared nodes.
        incidence: dict[int, set[int]] = {}  # Map each node to incident regions.
        for cell, region in zip(cells, labels, strict=True):  # Traverse element-region pairs.
            for node in cell:  # Traverse local nodes.
                incidence.setdefault(int(node), set()).add(int(region))  # Register regional incidence.
        graph = np.zeros((count, count), dtype=float)  # Allocate the symmetric graph.
        for regions in incidence.values():  # Inspect each node's region set.
            ordered = sorted(regions)  # Order region identifiers deterministically.
            for left_pos, left in enumerate(ordered):  # Traverse the first region.
                for right in ordered[left_pos + 1 :]:  # Traverse distinct adjacent regions.
                    graph[left, right] += 1.0  # Count one shared-node interaction.
                    graph[right, left] += 1.0  # Preserve symmetry.
        totals = np.sum(graph, axis=1)  # Compute regional interaction totals.
        return np.divide(graph, totals[:, None], out=np.zeros_like(graph), where=totals[:, None] > 0.0)  # Return row-normalized coupling weights.
    def _partition_adjacency(self, partition: Any, mesh: Any, labels: np.ndarray, count: int) -> np.ndarray:  # Prefer a frozen partition-provided graph when the shared partition exposes one.
        adjacency_matrix = getattr(partition, "adjacency_matrix", None)  # Read the optional frozen-partition adjacency interface.
        if callable(adjacency_matrix):  # Use the preregistered shared graph instead of reconstructing method-specific topology.
            graph = np.asarray(adjacency_matrix(mesh, labels), dtype=float)  # Request the partition graph in its frozen semantic ordering.
            if graph.shape != (count, count):  # Reject a graph whose ordering cannot match the observed regional state.
                raise ValueError("partition adjacency_matrix must return one square entry per observed region")  # Explain the frozen-graph shape contract.
            return graph.copy()  # Isolate the immutable world state from partition-owned storage.
        _, cells = self._arrays(mesh)  # Read connectivity for the legacy deterministic fallback.
        return self._adjacency(cells, labels, count)  # Reconstruct shared-node coupling only when no frozen graph is available.
    def _equations_from_record(self, problem: Any, record: Any, mesh: Any) -> int:  # Recover measured equations with exact active-DOF fallback.
        for name in ("n_equations", "equations", "neq"):  # Inspect supported record fields.
            if hasattr(record, name) and int(getattr(record, name)) > 0:  # Accept a positive measured value.
                return int(getattr(record, name))  # Return measured active equations.
        return self.estimate_equations(problem, mesh)  # Count exact active candidate degrees of freedom.
    def observe_solve(self, problem: Any, partition: Any, post: Any, record: Any, eta2: np.ndarray, hit_count: np.ndarray | None, step: int) -> ToolObservation:  # Convert one real solve into a world-model state.
        mesh = getattr(post, "mesh", getattr(record, "mesh", None))  # Recover the solved mesh.
        if mesh is None:  # Reject missing mesh evidence.
            raise ValueError("post-processing result must expose its solved mesh")  # Explain the observation contract.
        _, cells = self._arrays(mesh)  # Read current element connectivity.
        indicator = np.asarray(eta2, dtype=float).reshape(-1)  # Normalize squared error indicators.
        if indicator.shape != (cells.shape[0],):  # Require one value per element.
            raise ValueError("eta2 must contain one value per element")  # Explain the estimator contract.
        labels = np.asarray(partition.assign(mesh), dtype=int).reshape(-1)  # Assign current elements to fixed semantic regions.
        if labels.shape != indicator.shape or np.any(labels < 0):  # Validate semantic assignment.
            raise ValueError("partition must assign one non-negative label per element")  # Explain the partition contract.
        declared_count = len(tuple(getattr(partition, "names", ())))  # Preserve every frozen semantic region even when one is temporarily empty after remeshing.
        observed_count = int(np.max(labels)) + 1  # Measure the label dimension represented by the current mesh.
        if declared_count and observed_count > declared_count:  # Reject assignments outside an explicitly frozen state dimension.
            raise ValueError("partition labels must remain within the declared semantic ordering")  # Explain the stable-ordering contract.
        count = declared_count or observed_count  # Use the complete frozen ordering while accepting partitions without explicit names.
        names = self._names(partition, count)  # Recover stable semantic names.
        marked = self._marked_mask(indicator)  # Compute the mandatory exact-Dörfler mask.
        element_sizes, element_volumes = self._element_measures(mesh)  # Measure current geometry.
        stress = self._stress(post, cells.shape[0])  # Recover optional stress severity.
        err_sum = np.zeros(count, dtype=float)  # Allocate regional error sums.
        elems = np.zeros(count, dtype=float)  # Allocate regional element counts.
        sizes = np.zeros(count, dtype=float)  # Allocate regional measured sizes.
        vm_max = np.zeros(count, dtype=float)  # Allocate regional peak stresses.
        volume = np.zeros(count, dtype=float)  # Allocate regional volumes.
        marked_error = np.zeros(count, dtype=float)  # Allocate regional marked-error fractions.
        marked_elements = np.zeros(count, dtype=float)  # Allocate regional marked-element fractions.
        hits = np.zeros(count, dtype=float) if hit_count is None or np.asarray(hit_count).shape != (count,) else np.asarray(hit_count, dtype=float).copy()  # Initialize or restore recurrence evidence.
        for region in range(count):  # Aggregate each semantic region.
            mask = labels == region  # Select regional elements.
            selected = mask & marked  # Select exact-Dörfler elements in the region.
            elems[region] = float(np.count_nonzero(mask))  # Count regional elements.
            err_sum[region] = float(np.sum(indicator[mask]))  # Sum regional squared indicators.
            sizes[region] = float(np.median(element_sizes[mask])) if np.any(mask) else float(problem.h0)  # Measure regional size.
            vm_max[region] = float(np.max(stress[mask])) if np.any(mask) else 0.0  # Measure regional peak stress.
            volume[region] = float(np.sum(element_volumes[mask]))  # Sum regional volume.
            marked_error[region] = float(np.sum(indicator[selected]) / max(err_sum[region], 1.0e-30))  # Measure selected regional error share.
            marked_elements[region] = float(np.count_nonzero(selected) / max(np.count_nonzero(mask), 1))  # Measure selected regional element share.
            hits[region] += float(np.any(selected))  # Count one real Dörfler hit when present.
        equations = self._equations_from_record(problem, record, mesh)  # Recover measured active equations.
        state = WorldState(names=names, err_sum=err_sum, elems=elems, sizes=sizes, vm_max=vm_max, volume=volume, adjacency=self._partition_adjacency(partition, mesh, labels, count), dorfler_error_fraction=marked_error, dorfler_element_fraction=marked_elements, hit_count=hits, n_equations=equations, eq_per_elem=float(equations / max(cells.shape[0], 1)), h_min=float(problem.h_min), h0=float(problem.h0), dim=int(problem.dim), step=int(step))  # Construct the compact measured world state with the shared frozen graph when available.
        return ToolObservation(problem, partition, post, record, mesh, indicator, labels, marked, state)  # Return the complete observation.
    def estimate_equations(self, problem: Any, mesh: Any) -> int:  # Count active displacement degrees of freedom on a candidate mesh.
        points, _ = self._arrays(mesh)  # Read candidate nodes.
        dimension = int(getattr(problem, "dim", points.shape[1]))  # Recover displacement dimension.
        fixed = np.zeros((points.shape[0], dimension), dtype=bool)  # Allocate one entry per nodal displacement degree of freedom.
        for constraint in getattr(problem, "constraints", []):  # Apply all mesh-independent constraints.
            mask = np.asarray(constraint.node_predicate(points), dtype=bool).reshape(-1)  # Evaluate the repository Constraint node predicate.
            if mask.shape != (points.shape[0],):  # Reject malformed predicates.
                raise ValueError("constraint predicate must return one Boolean per node")  # Explain the boundary contract.
            for dof in constraint.dofs:  # Apply each one-based displacement component.
                fixed[mask, int(dof) - 1] = True  # Mark constrained candidate degrees of freedom.
        return int(points.shape[0] * dimension - np.count_nonzero(fixed))  # Return exact active displacement degrees of freedom.
    def _current_nodal_sizes(self, mesh: Any) -> np.ndarray:  # Recover nodal sizes for future-hit depth.
        points, cells = self._arrays(mesh)  # Read current mesh arrays.
        for name in ("node_sizes", "nodal_sizes", "sizes"):  # Inspect supported metadata fields.
            values = getattr(mesh, name, None)  # Read the candidate nodal size vector.
            if values is not None and np.asarray(values).size == points.shape[0]:  # Accept one size per node.
                return np.asarray(values, dtype=float).reshape(-1).copy()  # Return independent nodal sizes.
        element_sizes, _ = self._element_measures(mesh)  # Measure element sizes when metadata is absent.
        sums = np.zeros(points.shape[0], dtype=float)  # Accumulate incident element sizes.
        counts = np.zeros(points.shape[0], dtype=float)  # Count incident elements.
        for cell, size in zip(cells, element_sizes, strict=True):  # Traverse element-size pairs.
            sums[cell] += size  # Add the element size to incident nodes.
            counts[cell] += 1.0  # Count one incident element.
        return np.divide(sums, counts, out=np.full(points.shape[0], float(np.median(element_sizes))), where=counts > 0.0)  # Return average incident sizes.
    def _base_target(self, observation: ToolObservation) -> np.ndarray:  # Build the repository exact-Dörfler nodal target.
        try:  # Prefer the documented keyword call.
            target = refine_size_map(observation.mesh, observation.marked, factor=self.config.refine_factor)  # Generate the standard Dörfler target.
        except TypeError:  # Support positional factor signatures.
            target = refine_size_map(observation.mesh, observation.marked, self.config.refine_factor)  # Generate the same target positionally.
        values = np.asarray(target, dtype=float).reshape(-1)  # Normalize the nodal target vector.
        points, _ = self._arrays(observation.mesh)  # Read current node count.
        if values.shape != (points.shape[0],):  # Reject non-nodal target fields.
            raise ValueError("refine_size_map must return one target size per node")  # Explain the exact-Dörfler contract.
        return np.maximum(float(observation.problem.h_min), values)  # Enforce the minimum size.
    def _world_target(self, observation: ToolObservation, action: RegionAction, base_target: np.ndarray) -> np.ndarray:  # Add bounded future-hit depth to the exact-Dörfler target.
        action.validate(observation.state, max_depth=self.config.max_extra_depth)  # Validate the safe action domain.
        _, cells = self._arrays(observation.mesh)  # Read current connectivity.
        current = self._current_nodal_sizes(observation.mesh)  # Recover current nodal sizes.
        target = base_target.copy()  # Start from exact Dörfler.
        for region, depth in enumerate(action.extra_depth):  # Materialize each semantic depth.
            if depth <= 0:  # Skip baseline-only regions.
                continue  # Preserve exact Dörfler unchanged.
            if semantic_persistence(observation.state.names[region]) <= 0.0:  # Reject generic field actions defensively.
                raise ValueError("world-model depth cannot target a generic field region")  # Preserve the clean action boundary.
            region_elements = np.flatnonzero(observation.labels == region)  # Select the semantic mechanism.
            if region_elements.size == 0:  # Skip empty regions.
                continue  # Preserve target validity.
            order = region_elements[np.argsort(-observation.eta2[region_elements], kind="mergesort")]  # Rank regional indicator contributions deterministically.
            cumulative = np.cumsum(observation.eta2[order])  # Accumulate regional error.
            threshold = self.config.core_theta * max(float(np.sum(observation.eta2[region_elements])), 1.0e-30)  # Define concentrated future-hit support.
            count = int(np.searchsorted(cumulative, threshold, side="left") + 1)  # Select the smallest core meeting the threshold.
            core = order[: min(count, order.size)]  # Extract the concentrated core.
            current_marked = np.flatnonzero((observation.labels == region) & observation.marked)  # Recover all current Dörfler hits in the region.
            selected_nodes = np.unique(cells[np.unique(np.concatenate((core, current_marked)))].reshape(-1))  # Convert the union to nodal targets.
            desired = current[selected_nodes] * self.config.refine_factor ** (1 + int(depth))  # Compress current and predicted future hits.
            target[selected_nodes] = np.minimum(target[selected_nodes], desired)  # Refine without coarsening exact Dörfler.
        return np.maximum(float(observation.problem.h_min), target)  # Enforce the admissible minimum size.
    def _field(self, mesh: Any, target: np.ndarray, problem: Any) -> Any:  # Construct a repository-compatible nodal size field.
        return NodalSizeField(mesh, target, gradation=1.0, h_min=float(problem.h_min), h_max=float(problem.h0))  # Bind the exact repository mesh, target, gradation, and admissible size bounds.
    def _compiled_field(self, mesh: Any, target: np.ndarray, problem: Any) -> tuple[Any, np.ndarray, str]:  # Compile and content-bind the exact field passed to Gmsh.
        field = self._field(mesh, target, problem)  # Apply identical clipping and gradation through the repository implementation.
        values = np.asarray(field._h, dtype=float).reshape(-1).copy()  # Read the complete compiled nodal values rather than the raw input targets.
        points, _ = self._arrays(mesh)  # Bind compiled values to their exact source-node coordinates and ordering.
        if values.shape != (points.shape[0],):  # Reject an interpolation field that cannot be checked nodewise.
            raise ValueError("compiled size field must retain one value per source node")  # Preserve the structural certificate contract.
        digest = hashlib.sha256()  # Initialize the full compiled-field identity.
        digest.update(b"wmvla.compiled-nodal-size-field.v1\x00")  # Domain-separate compiled fields from legacy raw-target hashes.
        digest.update(np.asarray(points.shape, dtype="<i8").tobytes(order="C"))  # Bind node count and coordinate dimension without ambiguous concatenation.
        digest.update(np.asarray(points, dtype="<f8").tobytes(order="C"))  # Bind every source coordinate in repository ordering.
        digest.update(np.asarray([int(field._k)], dtype="<i8").tobytes(order="C"))  # Bind the dimensional interpolation-neighbor count.
        digest.update(np.asarray([1.0], dtype="<f8").tobytes(order="C"))  # Bind the common gradation value explicitly.
        digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))  # Bind every compiled nodal target value.
        return field, values, digest.hexdigest()  # Return the exact Gmsh field, auditable values, and complete SHA-256.
    def _generate(self, problem: Any, source_mesh: Any, target: np.ndarray) -> Any:  # Generate one exact Gmsh candidate mesh.
        field = self._field(source_mesh, target, problem)  # Construct the certified size field.
        return generate_mesh(problem, field)  # Regenerate a conformal mesh through the repository positional size-function API.
    def _timing(self, started: float, gmsh_s: float) -> dict[str, float]:  # Build one non-overlapping materialization timing payload.
        total_s = float(time.perf_counter() - started)  # Measure the complete gateway call.
        parameter_s = max(total_s - float(gmsh_s), 0.0)  # Exclude Gmsh from deterministic parameter and certification work.
        return {"parameter_tools": parameter_s, "gmsh_remeshing": float(gmsh_s), "tool_total": total_s}  # Return auditable disjoint timing components and their total.
    def materialize_action(self, observation: ToolObservation, action: RegionAction, n_equation_cap: int) -> MaterializedAction:  # Materialize, preflight, and certify one planner action.
        started = time.perf_counter()  # Start complete gateway timing before action validation.
        gmsh_s = 0.0  # Initialize cumulative exact candidate-mesh generation time.
        action.validate(observation.state, max_depth=self.config.max_extra_depth)  # Validate the discrete action.
        cap = int(self.config.budget_safety * int(n_equation_cap))  # Compute the deterministic resource cap.
        base_target = self._base_target(observation)  # Build exact Dörfler.
        base_field, base_compiled, base_compiled_hash = self._compiled_field(observation.mesh, base_target, observation.problem)  # Compile and bind the exact Dörfler field before Gmsh.
        gmsh_started = time.perf_counter()  # Start exact-Dörfler candidate remeshing timing.
        base_mesh = generate_mesh(observation.problem, base_field)  # Generate the baseline candidate from the exact hashed compiled field.
        gmsh_s += time.perf_counter() - gmsh_started  # Accumulate exact-Dörfler Gmsh wall time.
        base_equations = self.estimate_equations(observation.problem, base_mesh)  # Count exact candidate active degrees of freedom.
        base_hash = hashlib.sha256(np.asarray(base_target, dtype="<f8").tobytes()).hexdigest()  # Hash the exact target field.
        baseline = RegionAction.dorfler(observation.state)  # Construct the executable baseline action.
        if base_equations > cap:  # Stop when exact Dörfler itself exceeds the cap.
            certificate = MeshCertificate(self.schema_version, action.extra_depth, baseline.extra_depth, "stop", base_hash, True, True, base_equations, cap, False, "dorfler_candidate_exceeds_cap", base_raw_target_sha256=base_hash, base_compiled_field_sha256=base_compiled_hash, compiled_field_node_count=int(base_compiled.size), compiled_field_gradation=1.0, compiled_max_dorfler_violation=0.0, compiled_dorfler_included=True)  # Record the deterministic stop with complete compiled Dörfler identity.
            return MaterializedAction(None, baseline, certificate, base_equations, self._timing(started, gmsh_s))  # Return no over-budget candidate with separated timing.
        if action.is_dorfler_only:  # Execute exact Dörfler directly.
            certificate = MeshCertificate(self.schema_version, action.extra_depth, action.extra_depth, "dorfler", base_hash, True, True, base_equations, cap, True, "exact_dorfler", base_raw_target_sha256=base_hash, base_compiled_field_sha256=base_compiled_hash, compiled_field_node_count=int(base_compiled.size), compiled_field_gradation=1.0, compiled_max_dorfler_violation=0.0, compiled_dorfler_included=True)  # Certify the baseline and its exact compiled field.
            return MaterializedAction(base_mesh, action, certificate, base_equations, self._timing(started, gmsh_s))  # Return the baseline candidate with separated timing.
        world_target = self._world_target(observation, action, base_target)  # Add bounded semantic future depth.
        raw_violation = float(max(np.max(world_target - base_target), 0.0))  # Measure raw target dominance before field compilation.
        raw_included = bool(raw_violation <= 1.0e-12)  # Verify componentwise raw Dörfler inclusion.
        if not raw_included:  # Reject implementation defects before field compilation.
            raise RuntimeError("world target failed exact-Dörfler inclusion check")  # Preserve the Dörfler floor.
        world_field, world_compiled, world_compiled_hash = self._compiled_field(observation.mesh, world_target, observation.problem)  # Compile the proactive field with the identical repository gradation path.
        compiled_violation = float(max(np.max(world_compiled - base_compiled), 0.0))  # Measure the maximum compiled nodewise coarsening relative to Dörfler.
        compiled_included = bool(compiled_violation <= 1.0e-12)  # Apply the same explicit numerical tolerance after compilation.
        if not compiled_included:  # Reject any clipping or smoothing interaction that breaks Dörfler dominance.
            raise RuntimeError("compiled world field failed exact-Dörfler inclusion check")  # Stop before Gmsh sees a structurally uncertified field.
        gmsh_started = time.perf_counter()  # Start proactive world-candidate remeshing timing.
        world_mesh = generate_mesh(observation.problem, world_field)  # Generate the world candidate from the exact hashed compiled field.
        gmsh_s += time.perf_counter() - gmsh_started  # Accumulate proactive Gmsh wall time.
        world_equations = self.estimate_equations(observation.problem, world_mesh)  # Count exact candidate active degrees of freedom.
        world_hash = hashlib.sha256(np.asarray(world_target, dtype="<f8").tobytes()).hexdigest()  # Hash the world target field.
        if world_equations > cap:  # Reject exact resource-cap violations.
            certificate = MeshCertificate(self.schema_version, action.extra_depth, baseline.extra_depth, "dorfler_fallback", base_hash, raw_included and compiled_included, compiled_included, base_equations, cap, False, "world_candidate_exceeds_cap", base_raw_target_sha256=base_hash, world_raw_target_sha256=world_hash, base_compiled_field_sha256=base_compiled_hash, world_compiled_field_sha256=world_compiled_hash, compiled_field_node_count=int(base_compiled.size), compiled_field_gradation=1.0, compiled_max_dorfler_violation=compiled_violation, compiled_dorfler_included=compiled_included)  # Record fallback plus both complete attempted field identities.
            return MaterializedAction(base_mesh, baseline, certificate, base_equations, self._timing(started, gmsh_s))  # Execute exact Dörfler instead while retaining both candidate timings.
        certificate = MeshCertificate(self.schema_version, action.extra_depth, action.extra_depth, "world_model", world_hash, raw_included and compiled_included, compiled_included, world_equations, cap, True, "world_candidate_certified", base_raw_target_sha256=base_hash, world_raw_target_sha256=world_hash, base_compiled_field_sha256=base_compiled_hash, world_compiled_field_sha256=world_compiled_hash, compiled_field_node_count=int(base_compiled.size), compiled_field_gradation=1.0, compiled_max_dorfler_violation=compiled_violation, compiled_dorfler_included=compiled_included)  # Certify both raw and compiled nodewise Dörfler dominance.
        return MaterializedAction(world_mesh, action, certificate, base_equations, self._timing(started, gmsh_s))  # Return the exact preflighted candidate with separated timing.
