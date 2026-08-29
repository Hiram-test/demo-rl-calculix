# Bidirectional exact Gmsh budget certification for MCP-controlled WM-VLA execution.  # Module purpose.
from __future__ import annotations  # Enable postponed annotations for lightweight solver imports.
from typing import Any  # Type machine-readable certification histories.
import numpy as np  # Apply deterministic global size scaling and bound checks.
from ..geometry import Problem  # Read mesh bounds and physical dimension from the finite-element problem.
from ..mesher import Mesh, generate_mesh  # Generate exact candidate topologies without running CalculiX.
from .drawing import drawings_size_fn, drawings_with_sizes  # Bind regional size vectors to the fixed visual geometry.
from .regions import Partition  # Read stable region order and the coarse remainder identity.
from .tool_contract import MaterializedAction, MeshCertificate, estimate_free_equations  # Reuse strict action and exact equation-count contracts.


def _candidate_mesh(problem: Problem, partition: Partition, drawings: list, sizes: np.ndarray) -> tuple[Mesh, int]:  # Generate one exact mesh-only resource observation.
    names = [seed.name for seed in partition.seeds]  # Preserve the authoritative fixed region order.
    sized_drawings = drawings_with_sizes(drawings, names, np.asarray(sizes, dtype=float))  # Bind the current numerical vector to all drawn regions.
    remainder_index = next((index for index, seed in enumerate(partition.seeds) if seed.origin == "coarse"), len(partition.seeds) - 1)  # Locate the unpainted-volume controller.
    remainder_h = float(np.asarray(sizes, dtype=float)[int(remainder_index)])  # Read the exact current field size.
    mesh = generate_mesh(problem, drawings_size_fn(sized_drawings, remainder_h, problem))  # Generate a deterministic Gmsh topology.
    n_equations = estimate_free_equations(problem, mesh)  # Count free displacement equations before any CalculiX solve.
    return mesh, int(n_equations)  # Return the exact topology and resource count.


def certify_action_mesh_targeted(problem: Problem, partition: Partition, drawings: list, materialized: MaterializedAction, n_eq_budget: int, budget_safety: float = 0.985, max_attempts: int = 6, target_use: float = 0.90, lower_use: float = 0.82) -> MeshCertificate:  # Fill a declared budget band while preserving regional priorities.
    if int(n_eq_budget) <= 0:  # Require a positive hard resource cap.
        raise ValueError("n_eq_budget must be positive")  # Reject an unusable certification contract.
    if not 0.0 < float(lower_use) <= float(target_use) < float(budget_safety) <= 1.0:  # Require an ordered feasible utilization band.
        raise ValueError("require 0 < lower_use <= target_use < budget_safety <= 1")  # Report the exact contract violation.
    attempts_limit = max(int(max_attempts), 1)  # Guarantee at least one exact mesh observation.
    cap = max(int(np.floor(float(budget_safety) * float(n_eq_budget))), 1)  # Convert the hard safety margin to an integer equation cap.
    target = max(int(np.floor(float(target_use) * float(n_eq_budget))), 1)  # Define the desired feasible resource level.
    lower = max(int(np.floor(float(lower_use) * float(n_eq_budget))), 1)  # Define the minimum efficient budget utilization.
    sizes = np.clip(np.asarray(materialized.sizes, dtype=float).copy(), float(problem.h_min), float(problem.h0))  # Normalize the tool-owned starting parameters.
    history: list[dict[str, Any]] = []  # Preserve every exact Gmsh observation and deterministic correction.
    best_mesh: Mesh | None = None  # Retain the finest feasible topology observed so far.
    best_sizes: np.ndarray | None = None  # Retain the numerical field that generated the finest feasible topology.
    best_equations = -1  # Initialize the feasible-resource ranking below every valid mesh.
    last_mesh: Mesh | None = None  # Preserve the final topology for transparent failure reporting.
    last_equations = 0  # Preserve the final exact resource count.
    for attempt in range(1, attempts_limit + 1):  # Bound all mesh-only MCP calls explicitly.
        mesh, n_equations = _candidate_mesh(problem, partition, drawings, sizes)  # Observe the exact Gmsh resource outcome.
        last_mesh = mesh  # Preserve the latest topology for a no-feasible-candidate result.
        last_equations = int(n_equations)  # Preserve the latest exact equation count.
        if n_equations <= cap and n_equations > best_equations:  # Prefer the highest-resolution feasible topology under a fixed relative field.
            best_mesh = mesh  # Retain the current exact feasible topology.
            best_sizes = sizes.copy()  # Retain the exact numerical parameters that generated it.
            best_equations = int(n_equations)  # Retain its resource level for subsequent comparison.
        within_band = lower <= n_equations <= cap  # Detect an efficient and hard-cap-safe candidate.
        history.append({"attempt": int(attempt), "n_equations": int(n_equations), "utilization": float(n_equations) / float(n_eq_budget), "sizes": [float(value) for value in sizes], "within_band": bool(within_band)})  # Preserve the complete exact observation.
        if within_band:  # Accept the first candidate in the predeclared efficiency band.
            diagnostics = {"cap": cap, "target": target, "lower": lower, "target_met": True, "history": history, "materialized": materialized.to_dict(), "strategy": "bidirectional_exact_gmsh"}  # Assemble auditable evidence.
            return MeshCertificate(mesh, sizes.copy(), int(n_equations), int(attempt), True, diagnostics)  # Return the exact executable topology.
        if n_equations > cap:  # Coarsen a topology that violates the hard cap.
            scale = float((float(n_equations) / float(cap)) ** (1.0 / float(problem.dim)) * 1.018)  # Infer a conservative monotone coarsening factor.
        else:  # Refine a feasible topology that wastes too much of the declared budget.
            scale = float((max(float(n_equations), 1.0) / float(target)) ** (1.0 / float(problem.dim)) * 0.992)  # Infer a restrained refinement factor toward the target.
        updated = np.clip(sizes * scale, float(problem.h_min), float(problem.h0))  # Apply one global scale while preserving all regional size ratios.
        if np.max(np.abs(np.log(np.maximum(updated, 1.0e-12) / np.maximum(sizes, 1.0e-12)))) <= 1.0e-5:  # Detect bound saturation or a numerically ineffective correction.
            history[-1]["stop"] = "size_bounds_saturated"  # Explain why further exact calls cannot alter the topology reliably.
            break  # Return the best feasible observed topology instead of looping pointlessly.
        history[-1]["next_scale"] = float(scale)  # Preserve the exact deterministic correction applied next.
        sizes = updated  # Advance to the next globally scaled but priority-preserving field.
    if best_mesh is not None and best_sizes is not None:  # Deliver the finest feasible observed topology even if the lower efficiency band was missed.
        diagnostics = {"cap": cap, "target": target, "lower": lower, "target_met": bool(best_equations >= lower), "history": history, "materialized": materialized.to_dict(), "strategy": "bidirectional_exact_gmsh", "fallback": "best_feasible"}  # Preserve all evidence and fallback semantics.
        return MeshCertificate(best_mesh, best_sizes, int(best_equations), int(len(history)), True, diagnostics)  # Return a valid hard-cap-safe certificate.
    if last_mesh is None:  # Guard against an impossible empty attempt loop.
        raise RuntimeError("targeted mesh certification generated no candidate mesh")  # Refuse to fabricate topology or resource evidence.
    diagnostics = {"cap": cap, "target": target, "lower": lower, "target_met": False, "history": history, "materialized": materialized.to_dict(), "strategy": "bidirectional_exact_gmsh", "failure": "no_feasible_mesh"}  # Preserve transparent failure evidence.
    return MeshCertificate(last_mesh, sizes.copy(), int(last_equations), int(len(history)), False, diagnostics)  # Return the failed certificate explicitly for diagnostics.
