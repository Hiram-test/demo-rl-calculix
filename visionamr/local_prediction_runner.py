"""Independent multi-step local-prediction baseline for held-out bridge evaluation."""  # State that this module is an evaluation baseline, not a WM-VLA dependency.

from __future__ import annotations  # Enable postponed annotation evaluation.

import importlib  # Import the existing repository local-prediction implementation dynamically.
import inspect  # Import signature inspection for baseline API compatibility.
import math  # Import finite-budget integer operations.
from dataclasses import dataclass  # Import an immutable baseline-result container.
from typing import Any  # Import structural typing for repository runtime objects.

import numpy as np  # Import vectorized target normalization and estimator accounting.

from .experiment import FemRunner  # Import the common audited CalculiX execution gateway.
from .experiment import initial_mesh  # Import the common uniform probe mesh.
from .indicators import zz_indicator  # Import the shared element-wise ZZ estimator.
from .mesher import Mesh  # Import the repository simplex-mesh contract.
from .mesher import generate_mesh  # Import the common Gmsh remeshing gateway.
from .sizefield import NodalSizeField  # Import deterministic nodal target interpolation.


@dataclass(frozen=True)  # Keep the independent baseline summary immutable.
class LocalPredictionResult:  # Summarize one complete local-prediction trajectory.
    solves: int  # Record the number of real CalculiX solves.
    stopped_by: str  # Record the physical, resource, or configured stopping condition.
    predicted_size_calls: int  # Record how many independent LP size fields were evaluated.


def _record_equations(record: Any) -> int:  # Read actual equation counts across repository record revisions.
    for field in ("n_equations", "n_eq", "equations"):  # Enumerate supported explicit record fields.
        value = getattr(record, field, None)  # Read one candidate field safely.
        if value is not None:  # Accept the first materialized count.
            return int(value)  # Return a normalized integer.
    extra = getattr(record, "extra", {}) or {}  # Read structured diagnostics as a final compatibility path.
    for field in ("n_equations", "n_eq", "equations"):  # Enumerate supported diagnostic keys.
        if field in extra:  # Accept an explicitly recorded equation count.
            return int(extra[field])  # Return the normalized diagnostic count.
    raise AttributeError("solve record contains no equation count")  # Reject unauditable resource accounting.


def _estimate_free_equations(mesh: Mesh, problem: Any) -> int:  # Compute the exact displacement-equation count implied by a generated mesh.
    constrained: set[tuple[int, int]] = set()  # Allocate unique constrained node-and-DOF pairs.
    for constraint in problem.constraints:  # Evaluate every repository boundary-condition predicate.
        mask = np.asarray(constraint.node_predicate(mesh.nodes), dtype=bool)  # Select all constrained mesh nodes geometrically.
        for node in np.nonzero(mask)[0]:  # Visit each constrained node index.
            for dof in constraint.dofs:  # Visit each constrained displacement component.
                if 1 <= int(dof) <= int(problem.dim):  # Retain displacement components represented by this formulation.
                    constrained.add((int(node), int(dof)))  # Add the unique constrained equation key.
    return max(int(problem.dim * mesh.n_nodes - len(constrained)), 1)  # Return the exact free displacement-DOF count.


def _prediction_callable() -> Any:  # Locate the canonical repository local-prediction size function.
    module = importlib.import_module("visionamr.baselines.local_prediction")  # Import the independent baseline implementation only inside this evaluation module.
    preferred = ("predicted_sizes", "predict_sizes", "local_predicted_sizes", "make_predicted_sizes")  # Rank explicit size-field function names.
    for name in preferred:  # Search stable function names first.
        candidate = getattr(module, name, None)  # Read one named candidate safely.
        if callable(candidate):  # Require an executable function.
            return candidate  # Return the canonical independent baseline function.
    for name, candidate in inspect.getmembers(module, callable):  # Search structurally compatible module callables as a final path.
        lowered = name.lower()  # Normalize the callable name for filtering.
        if "predict" in lowered and "size" in lowered:  # Require both prediction and mesh-size semantics.
            return candidate  # Return the first deterministic structural match.
    raise AttributeError("no local-prediction size function was found")  # Reject an unavailable baseline instead of substituting another method.


def _invoke_prediction(function: Any, problem: Any, mesh: Mesh, post: Any, eta2: np.ndarray, n_eq_cap: int, exponent: float) -> Any:  # Call the repository LP function through explicit semantic arguments.
    target_nodes = max(int(n_eq_cap // max(int(problem.dim), 1)), mesh.n_nodes)  # Convert the equation cap to a conservative target-node budget.
    target_elements = max(int(round(mesh.n_cells * target_nodes / max(mesh.n_nodes, 1))), mesh.n_cells)  # Convert the node budget to a topology-scaled element budget.
    context: dict[str, Any] = {"problem": problem, "mesh": mesh, "post": post, "state": post, "solution": post, "eta2": eta2, "indicator": eta2, "indicators": eta2, "error": eta2, "errors": eta2, "n_eq_cap": int(n_eq_cap), "equation_cap": int(n_eq_cap), "budget": int(n_eq_cap), "target_nodes": int(target_nodes), "n_target_nodes": int(target_nodes), "target_elements": int(target_elements), "n_target": int(target_elements), "n_target_elements": int(target_elements), "dim": int(problem.dim), "dimension": int(problem.dim), "p": float(exponent), "exponent": float(exponent), "h_min": float(problem.h_min), "h_max": float(problem.h0), "h0": float(problem.h0)}  # Map known LP argument names to measured baseline data.
    parameters = inspect.signature(function).parameters  # Read the installed LP function contract.
    kwargs: dict[str, Any] = {}  # Allocate accepted keyword arguments only.
    unresolved: list[str] = []  # Track unsupported required parameters explicitly.
    for name, parameter in parameters.items():  # Resolve each declared LP function parameter.
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):  # Ignore variadic compatibility escape hatches.
            continue  # Resolve explicit parameters only.
        if name in context:  # Supply a known semantic argument.
            kwargs[name] = context[name]  # Store the exact measured or configured value.
        elif parameter.default is inspect.Parameter.empty:  # Record an unsupported required argument.
            unresolved.append(name)  # Preserve the exact missing parameter name.
    if unresolved:  # Reject an ambiguous baseline API rather than guessing numerical inputs.
        raise TypeError(f"unresolved local-prediction parameters: {unresolved}")  # Surface the exact independent-baseline contract mismatch.
    return function(**kwargs)  # Execute the existing repository local-prediction function.


def _extract_array(result: Any, mesh: Mesh) -> np.ndarray:  # Extract a node- or cell-sized numerical field from common LP return contracts.
    candidates: list[Any] = []  # Allocate ordered result candidates.
    if isinstance(result, dict):  # Inspect explicit structured return values first.
        for key in ("node_sizes", "sizes", "h", "h_new", "target_sizes", "cell_sizes"):  # Rank known target-field keys.
            if key in result:  # Retain only keys present in the result.
                candidates.append(result[key])  # Add the corresponding numerical field.
    elif isinstance(result, (tuple, list)):  # Inspect tuple-based result contracts.
        candidates.extend(result)  # Preserve the repository return order.
    else:  # Treat a direct return value as the primary field candidate.
        candidates.append(result)  # Add the direct result.
    for candidate in candidates:  # Normalize each numerical candidate in stable order.
        if hasattr(candidate, "node_sizes"):  # Support an explicit nodal size-field object.
            candidate = getattr(candidate, "node_sizes")  # Read its nodal target array.
        try:  # Isolate non-numerical metadata entries.
            array = np.asarray(candidate, dtype=float).reshape(-1)  # Normalize the candidate to one finite vector.
        except (TypeError, ValueError):  # Ignore non-numerical result members.
            continue  # Search the next candidate.
        if array.shape in ((mesh.n_nodes,), (mesh.n_cells,)):  # Accept only fields aligned with the current mesh.
            return array  # Return the first valid target field.
    raise ValueError("local-prediction result contains no node- or cell-aligned size field")  # Reject an unauditable baseline result.


def _nodal_target(result: Any, mesh: Mesh, problem: Any) -> np.ndarray:  # Convert any valid LP field to the repository nodal target contract.
    values = _extract_array(result, mesh)  # Extract the aligned numerical size field.
    if values.shape == (mesh.n_nodes,):  # Preserve an explicit nodal prediction directly.
        nodal = values.copy()  # Copy the result to prevent baseline-side mutation.
    else:  # Convert an element-wise LP field conservatively to shared nodes.
        nodal = np.full(mesh.n_nodes, float(problem.h0), dtype=float)  # Initialize nodes at the family coarse-size ceiling.
        for cell_index, nodes in enumerate(mesh.cells):  # Visit every simplex element and its target size.
            nodal[np.asarray(nodes, dtype=int)] = np.minimum(nodal[np.asarray(nodes, dtype=int)], float(values[cell_index]))  # Assign the finest adjacent element target to each shared node.
    nodal = np.clip(nodal, float(problem.h_min), float(problem.h0))  # Enforce the same family mesh-size bounds used by other methods.
    if not np.all(np.isfinite(nodal)):  # Reject NaN or infinite baseline targets before meshing.
        raise ValueError("local-prediction target contains non-finite values")  # Surface the invalid independent baseline result.
    return nodal  # Return the deterministic nodal LP target field.


def run_local_prediction_multistep(runner: FemRunner, n_eq_cap: int, max_solves: int = 8, gradation: float = 0.90, budget_safety: float = 0.98, exponent: float = 2.5, require_reference: bool = True, method: str = "local_prediction") -> LocalPredictionResult:  # Execute an independent repeated LP solve-predict-remesh loop.
    if n_eq_cap <= 0 or max_solves < 1:  # Validate the finite resource and solve horizons.
        raise ValueError("n_eq_cap and max_solves must be positive")  # Reject an invalid baseline campaign.
    if not 0.0 < budget_safety <= 1.0:  # Validate the pre-solve budget safety factor.
        raise ValueError("budget_safety must lie in (0, 1]")  # Reject an invalid resource contract.
    problem = runner.problem  # Read the finite-element problem owned by the audited runner.
    if require_reference:  # Build or load reference evidence when the held-out campaign requests it.
        runner.ensure_reference()  # Materialize the common reference before counted baseline solves.
    prediction = _prediction_callable()  # Resolve the existing repository LP size-field function once.
    mesh = initial_mesh(problem)  # Start from the same common uniform probe as Dörfler and WM-VLA.
    calls = 0  # Initialize the independent LP prediction-call counter.
    stopped_by = "solve_cap"  # Set the default terminal reason.
    for step in range(int(max_solves)):  # Execute a genuine repeated physical feedback loop.
        post, record = runner.solve_mesh(mesh, method=method, stage=f"cycle{step}")  # Execute one real CalculiX baseline solve.
        eta2 = zz_indicator(problem, post)  # Compute the shared element-wise ZZ estimator for reporting and LP input.
        record.extra.update(sum_eta2=float(np.sum(eta2)), controller="independent_multi_step_local_prediction", local_prediction_baseline=True, world_model_input=False, exponent=float(exponent))  # Attach explicit method-separation evidence.
        if step + 1 >= int(max_solves):  # Stop after the configured number of real solves.
            stopped_by = "solve_cap"  # Record the configured terminal condition.
            record.extra["stop"] = stopped_by  # Preserve the condition in the counted solve record.
            break  # End the baseline loop.
        if _record_equations(record) >= int(n_eq_cap):  # Stop after reaching the same hard equation budget.
            stopped_by = "equation_cap"  # Record the resource terminal condition.
            record.extra["stop"] = stopped_by  # Preserve the condition in the counted solve record.
            break  # End the baseline loop.
        raw = _invoke_prediction(prediction, problem, mesh, post, eta2, int(n_eq_cap), float(exponent))  # Evaluate the independent LP continuous size field.
        calls += 1  # Count the baseline prediction invocation.
        target = _nodal_target(raw, mesh, problem)  # Normalize the result to the deterministic nodal mesh contract.
        field = NodalSizeField(mesh, target, gradation=float(gradation), h_min=float(problem.h_min), h_max=float(problem.h0))  # Apply the same deterministic gradation and bounds as competing methods.
        next_mesh = generate_mesh(problem, field)  # Materialize the next independent LP Gmsh mesh.
        estimated = _estimate_free_equations(next_mesh, problem)  # Compute exact free equations before another real solve.
        record.extra.update(next_estimated_equations=int(estimated), target_min=float(np.min(target)), target_max=float(np.max(target)), target_kind="independent_local_prediction")  # Preserve complete pre-solve resource and target evidence.
        if estimated > int(math.floor(float(budget_safety) * int(n_eq_cap))):  # Apply the identical safety-adjusted equation cap.
            stopped_by = "next_local_prediction_mesh_over_budget"  # Record the fair pre-solve resource stop.
            record.extra["stop"] = stopped_by  # Preserve the condition in the counted solve record.
            break  # End the baseline loop before an over-budget solve.
        mesh = next_mesh  # Advance to the certified independent LP mesh.
    records = [record for record in runner.records if getattr(record, "method", None) == method]  # Isolate counted records belonging to this baseline.
    return LocalPredictionResult(solves=len(records), stopped_by=stopped_by, predicted_size_calls=calls)  # Return the complete independent baseline summary.
