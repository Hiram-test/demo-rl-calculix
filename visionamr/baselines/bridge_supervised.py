"""Frozen supervised bridge baseline for protocol WMVLA-4WAY-P1."""  # State this module's sole scientific responsibility.
from __future__ import annotations  # Postpone annotation evaluation for broad Python compatibility.
from collections.abc import Callable, Mapping, Sequence  # Import read-only collection and callback contracts.
from dataclasses import asdict, dataclass  # Define immutable protocol configuration and serialize solver records.
import hashlib  # Compute canonical configuration, dataset, target, mesh, and model identities.
import json  # Persist transparent machine-readable training and validation evidence.
import math  # Compute deterministic budget scaling and validation log means.
from pathlib import Path  # Handle repository-relative artifact paths portably.
import shutil  # Copy the validation-selected checkpoint into its frozen deployment location.
import time  # Measure expert generation, network training, validation, and online deployment wall times.
from typing import Any  # Type repository solver and mesh objects without introducing circular imports.
import numpy as np  # Build expert arrays, numerical features, hashes, and deterministic scores.
from ..bridge_case_manifest import problem_from_case  # Reconstruct only manifest-authorized bridge cases.
from ..calculix import CalculiXExecutionError  # Retain only evidenced native solver failures during fixed validation scoring.
from ..experiment import FemRunner, Reference, initial_mesh  # Reuse counted solves, trusted references, and the common probe.
from ..indicators import zz_indicator  # Reuse the repository ZZ element indicator for expert and deployment evidence.
from ..marking import dorfler_mark  # Reuse the exact repository Dörfler bulk-marking implementation.
from ..mesher import GmshMeshingError, Mesh, generate_mesh  # Reuse conforming Gmsh remeshing and its explicit native failure category.
from ..sizefield import NodalSizeField  # Reuse the nodal target-size interpolator used by all mesh baselines.
from .dorfler import refine_size_map  # Reuse the exact repository Dörfler refinement atom.
from .supervised import N_FEATURES, SizeMLP, SupervisedConfig, node_features  # Reuse the existing feature and MLP route without altering it.

PROTOCOL_ID = "WMVLA-4WAY-P1"  # Bind every artifact emitted here to the frozen four-way protocol.
SUPERVISED_SCHEMA = "wmvla-four-way-supervised-v1"  # Version the complete supervised training and deployment contract.
NETWORK_SEEDS = (20260831, 20260832, 20260833)  # Freeze the three independently initialized network seeds before testing.
VALIDATION_BUDGETS = (30000, 60000, 120000)  # Validate one checkpoint at every equation budget used by the benchmark.
VALIDATION_FAILURE_ERROR = 10.0  # Assign failed validation metrics a finite preregistered relative-error penalty.
VALIDATION_ERROR_FLOOR = 1.0e-300  # Keep exact-zero valid errors representable in finite log aggregation.
EXPECTED_SPLIT_COUNTS = {"train": 24, "validation": 8}  # Freeze the only two manifest partitions this module may execute.

@dataclass(frozen=True)  # Prevent result-dependent mutation of the supervised scientific configuration.
class BridgeSupervisedConfig:  # Collect every training, validation, and deployment degree of freedom in one hashable record.
    expert_equation_budget: int = 120000  # Generate expert labels under the largest common equation budget.
    expert_max_solves: int = 6  # Limit each complete Dörfler expert trajectory to six real global solves.
    theta: float = 0.5  # Match the common exact Dörfler bulk parameter.
    gradation: float = 1.0  # Match the preregistered common gradation while preserving the PR-40 V0 tool behavior.
    hidden: int = 64  # Retain the existing small two-hidden-layer MLP capacity.
    learning_rate: float = 2.0e-3  # Retain the existing supervised optimizer learning rate.
    epochs: int = 300  # Retain the existing fixed full-training epoch count.
    batch_size: int = 4096  # Retain the existing fixed mini-batch size.
    network_seeds: tuple[int, int, int] = NETWORK_SEEDS  # Train exactly the three preregistered independent initializations.
    validation_budgets: tuple[int, int, int] = VALIDATION_BUDGETS  # Select one network jointly over all three budgets.
    validation_failure_error: float = VALIDATION_FAILURE_ERROR  # Keep numerical failures finite and visible during selection.
    preflight_bisections: int = 8  # Resolve the feasible budget scale with a fixed amount of deterministic Gmsh work.
    preflight_expansions: int = 10  # Bound deterministic bracketing work without using reference errors.

@dataclass(frozen=True)  # Keep one exact mesh-budget preflight result immutable after selection.
class BudgetMesh:  # Return the selected unsolved mesh and all deterministic resource evidence.
    mesh: Any  # Store the exact Gmsh candidate that will receive the second real solve.
    scale: float  # Store the single scalar applied to the network-predicted nodal sizes.
    estimated_equations: int  # Store the exact active displacement-DOF count before CalculiX.
    equation_budget: int  # Store the requested common equation cap.
    target_sha256: str  # Identify the clipped scaled nodal target field exactly.
    mesh_sha256: str  # Identify the selected candidate coordinates and connectivity exactly.
    trials: tuple[dict[str, Any], ...]  # Preserve every scale and exact preflight count in execution order.

@dataclass(frozen=True)  # Keep the two-real-solve deployment receipt immutable after execution.
class SupervisedDeployment:  # Return online records plus deterministic scaling and hold-last evidence.
    probe_record: Any  # Store the counted common uniform-probe solve record.
    deployed_record: Any  # Store the counted predicted-remesh solve record.
    budget_mesh: BudgetMesh  # Store the exact resource preflight that authorized the second solve.
    real_solve_count: int  # Prove that deployment used exactly two real global solves.
    hold_last_after_solve: int  # Declare that comparisons at K greater than two reuse the second solve.
    online_wall_s: float  # Measure the complete probe, inference, Gmsh, and deployed-solve path.

def canonical_json_sha256(payload: object) -> str:  # Hash one JSON-compatible configuration independently of formatting.
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")  # Produce one canonical byte sequence.
    return hashlib.sha256(encoded).hexdigest()  # Return the complete lowercase SHA-256 digest.

def file_sha256(path: Path | str) -> str:  # Hash an artifact's exact persisted bytes.
    digest = hashlib.sha256()  # Allocate an incremental SHA-256 state.
    with Path(path).open("rb") as handle:  # Stream the artifact without assuming it fits in memory.
        for block in iter(lambda: handle.read(1024 * 1024), b""):  # Read deterministic one-megabyte chunks until EOF.
            digest.update(block)  # Incorporate the next exact byte block.
    return digest.hexdigest()  # Return the complete artifact identity.

def write_json(path: Path | str, payload: object) -> Path:  # Persist strict finite JSON through an atomic same-directory replacement.
    output = Path(path)  # Normalize the destination path once.
    output.parent.mkdir(parents=True, exist_ok=True)  # Create only the requested artifact directory hierarchy.
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"  # Produce reviewable standards-compliant JSON.
    temporary = output.with_suffix(output.suffix + ".tmp")  # Place the incomplete file beside its final target.
    temporary.write_text(text, encoding="utf-8")  # Write the complete validated serialization before publication.
    temporary.replace(output)  # Publish the complete artifact atomically on the same filesystem.
    return output  # Return the exact persisted artifact path.

def supervised_config_payload(config: BridgeSupervisedConfig) -> dict[str, Any]:  # Convert the immutable configuration into a complete hashable record.
    return {"schema": SUPERVISED_SCHEMA, "protocol_id": PROTOCOL_ID, "feature_count": N_FEATURES, "expert": {"equation_budget": config.expert_equation_budget, "max_solves": config.expert_max_solves, "theta": config.theta, "gradation": config.gradation}, "network": {"hidden": config.hidden, "learning_rate": config.learning_rate, "epochs": config.epochs, "batch_size": config.batch_size, "seeds": list(config.network_seeds)}, "validation": {"split": "validation", "budgets": list(config.validation_budgets), "failure_error": config.validation_failure_error, "selection_order": ["failure_point_count", "energy_error_log_mean", "qoi_error_log_mean", "budget_violation_count", "seed"]}, "deployment": {"common_probe": "uniform_h0", "real_solves": 2, "hold_last_after_solve": 2, "preflight_bisections": config.preflight_bisections, "preflight_expansions": config.preflight_expansions}}  # Expose every scientific choice without runtime-only paths.

def cases_for_split(manifest: Mapping[str, Any], split: str) -> tuple[dict[str, Any], ...]:  # Select one authorized development split without returning blind-test records.
    if split not in EXPECTED_SPLIT_COUNTS:  # Forbid this training module from selecting or executing the blind split.
        raise ValueError("supervised training may access only train or validation cases")  # Stop before any test-case reconstruction or solve.
    cases_value = manifest.get("cases")  # Read the manifest case container once.
    if not isinstance(cases_value, list):  # Require the validated manifest's ordered list shape.
        raise ValueError("manifest cases must be a list")  # Reject malformed or partial manifests.
    selected = tuple(dict(case) for case in cases_value if isinstance(case, Mapping) and case.get("split") == split)  # Copy only records explicitly assigned to the requested development split.
    if len(selected) != EXPECTED_SPLIT_COUNTS[split]:  # Require the frozen 24/8 development cardinality.
        raise ValueError(f"expected {EXPECTED_SPLIT_COUNTS[split]} {split} cases, found {len(selected)}")  # Report incomplete or relabeled input.
    if any(case.get("split") != split for case in selected):  # Defend against accidental mixed-partition inputs.
        raise RuntimeError("selected supervised cases contain a foreign split")  # Stop before any geometry or solver action.
    return tuple(sorted(selected, key=lambda case: str(case["case_id"])))  # Preserve deterministic case-ID execution order.

def estimate_active_equations(problem: Any, mesh: Any) -> int:  # Count exact active displacement degrees of freedom on an unsolved candidate mesh.
    points = np.asarray(mesh.nodes, dtype=float)  # Read candidate nodal coordinates.
    dimension = int(problem.dim)  # Use the problem's displacement dimension rather than ambient coordinate padding.
    fixed = np.zeros((points.shape[0], dimension), dtype=bool)  # Allocate one flag per candidate displacement degree of freedom.
    for constraint in problem.constraints:  # Apply every mesh-independent boundary constraint.
        mask = np.asarray(constraint.node_predicate(points), dtype=bool).reshape(-1)  # Evaluate the repository's actual Constraint contract.
        if mask.shape != (points.shape[0],) or not np.any(mask):  # Match CalculiX input validation for malformed or empty boundary selections.
            raise ValueError(f"constraint {constraint.name!r} matched no valid candidate nodes")  # Reject an invalid mesh before spending a real solve.
        for dof in constraint.dofs:  # Apply each one-based constrained displacement component.
            component = int(dof) - 1  # Convert the CalculiX component number to a zero-based array column.
            if component < 0 or component >= dimension:  # Reject constraints outside the active problem dimension.
                raise ValueError(f"constraint {constraint.name!r} has invalid dof {dof}")  # Surface the exact invalid boundary contract.
            fixed[mask, component] = True  # Mark each unique constrained candidate degree of freedom once.
    return int(points.shape[0] * dimension - np.count_nonzero(fixed))  # Return the exact active nodal displacement count.

def mesh_sha256(mesh: Any) -> str:  # Hash candidate geometry and connectivity independently of Python object identity.
    digest = hashlib.sha256()  # Allocate a streaming SHA-256 state.
    digest.update(np.asarray(mesh.nodes, dtype="<f8", order="C").tobytes(order="C"))  # Hash canonical little-endian coordinates.
    digest.update(np.asarray(mesh.cells, dtype="<i8", order="C").tobytes(order="C"))  # Hash canonical little-endian connectivity.
    return digest.hexdigest()  # Return the complete deterministic mesh identity.

def _scaled_field(problem: Any, source_mesh: Any, predicted_sizes: np.ndarray, scale: float, gradation: float) -> tuple[NodalSizeField, np.ndarray]:  # Compile one scalar-scaled network output under the frozen common gradation.
    target = np.clip(float(scale) * np.asarray(predicted_sizes, dtype=float), float(problem.h_min), float(problem.h0))  # Enforce the exact admissible nodal size range.
    field = NodalSizeField(source_mesh, target, gradation=float(gradation), h_min=problem.h_min, h_max=problem.h0)  # Build the deterministic common-contract mesh-size interpolator.
    compiled_target = np.asarray(field._h, dtype=float).copy()  # Recover the exact Lipschitz-smoothed nodal values evaluated by the executable callback.
    return field, compiled_target  # Return both executable field and its exact hashable nodal target.

def _initial_budget_scale(source_mesh: Any, predicted_sizes: np.ndarray, probe_equations: int, equation_budget: int) -> float:  # Estimate a deterministic starting scalar without any reference error.
    dimension = int(source_mesh.dim)  # Read the simplex dimension used by the volume-to-count law.
    factor = 2.0 if dimension == 2 else 8.49  # Reuse the repository's calibrated triangle or tetrahedron simplex factor.
    cell_predicted = np.asarray(predicted_sizes, dtype=float)[source_mesh.cells].mean(axis=1)  # Average nodal predictions on every current simplex.
    theory_probe = float(np.sum(factor * source_mesh.measures / np.maximum(source_mesh.cell_sizes, 1.0e-30) ** dimension))  # Evaluate the same count proxy on the observed probe.
    theory_predicted = float(np.sum(factor * source_mesh.measures / np.maximum(cell_predicted, 1.0e-30) ** dimension))  # Evaluate the proxy on the network field at unit scale.
    calibrated_elements = source_mesh.n_cells * theory_predicted / max(theory_probe, 1.0e-30)  # Calibrate the proxy against the actual probe mesh count.
    equations_per_element = max(float(probe_equations) / max(source_mesh.n_cells, 1), 1.0e-12)  # Convert the common equation cap using only measured probe resources.
    target_elements = max(float(equation_budget) / equations_per_element, 1.0)  # Derive the budget-equivalent element target without reference information.
    scale = (calibrated_elements / target_elements) ** (1.0 / dimension)  # Solve the simplex count law for one global size multiplier.
    return float(np.clip(scale, 1.0 / 256.0, 256.0))  # Bound only pathological model outputs before exact Gmsh preflight.

def preflight_budget_mesh(problem: Any, source_mesh: Any, predicted_sizes: np.ndarray, probe_equations: int, equation_budget: int, *, gradation: float = 1.0, bisections: int = 8, expansions: int = 10, mesh_generator: Callable[[Any, Any], Any] = generate_mesh) -> BudgetMesh:  # Select the finest deterministically bracketed candidate under the frozen common gradation.
    if int(equation_budget) <= 0 or int(probe_equations) <= 0:  # Require usable measured and requested resource counts.
        raise ValueError("probe equations and equation budget must be positive")  # Reject undefined scaling before meshing.
    if np.asarray(predicted_sizes).shape != (source_mesh.n_nodes,):  # Require exactly one network prediction per common-probe node.
        raise ValueError("predicted_sizes must contain one value per probe node")  # Reject misaligned network output before interpolation.
    trials: list[dict[str, Any]] = []  # Accumulate every deterministic Gmsh preflight receipt.
    def evaluate(scale: float) -> tuple[float, Any, int, str, str]:  # Generate and measure one unsolved candidate at a proposed scalar.
        field, target = _scaled_field(problem, source_mesh, predicted_sizes, scale, gradation)  # Compile the clipped scaled nodal field under the registered contract.
        candidate = mesh_generator(problem, field)  # Generate the exact conforming Gmsh mesh without a CalculiX invocation.
        equations = estimate_active_equations(problem, candidate)  # Count exact active displacement degrees of freedom on the candidate.
        target_hash = hashlib.sha256(np.asarray(target, dtype="<f8").tobytes(order="C")).hexdigest()  # Identify the exact clipped nodal target.
        candidate_hash = mesh_sha256(candidate)  # Identify the exact generated coordinates and connectivity.
        trials.append({"scale": float(scale), "estimated_equations": equations, "feasible": equations <= int(equation_budget), "target_sha256": target_hash, "mesh_sha256": candidate_hash})  # Preserve the complete preflight decision input.
        return float(scale), candidate, equations, target_hash, candidate_hash  # Return the executable candidate and exact resource evidence.
    initial_scale = _initial_budget_scale(source_mesh, predicted_sizes, probe_equations, equation_budget)  # Start from a probe-calibrated error-blind estimate.
    first = evaluate(initial_scale)  # Perform the first exact Gmsh resource preflight.
    lower: tuple[float, Any, int, str, str] | None = None  # Reserve the known-over-budget side of the logarithmic bracket.
    upper: tuple[float, Any, int, str, str] | None = None  # Reserve the known-feasible side of the logarithmic bracket.
    if first[2] <= int(equation_budget):  # Search toward finer scales when the initial candidate is feasible.
        upper = first  # Retain the current finest known feasible candidate.
        scale = first[0]  # Initialize deterministic geometric expansion from the current scalar.
        for _index in range(int(expansions)):  # Bound the finer-side bracket search independently of results.
            next_scale = max(scale / 2.0, 1.0 / 256.0)  # Refine the global size scalar by an exact factor of two.
            if next_scale == scale:  # Stop when the declared pathological lower scale is reached.
                break  # Preserve the current finest feasible candidate.
            candidate = evaluate(next_scale)  # Preflight the finer deterministic candidate.
            if candidate[2] > int(equation_budget):  # Detect the first known-over-budget lower bracket endpoint.
                lower = candidate  # Retain the infeasible side for bisection.
                break  # Stop geometric expansion once the bracket is complete.
            upper = candidate  # Retain a newly verified finer feasible candidate.
            scale = next_scale  # Continue toward a finer candidate on the next fixed expansion.
    else:  # Search toward coarser scales when the initial candidate exceeds the cap.
        lower = first  # Retain the current known-over-budget candidate.
        scale = first[0]  # Initialize deterministic geometric expansion from the current scalar.
        for _index in range(int(expansions)):  # Bound the coarser-side bracket search independently of reference results.
            next_scale = min(scale * 2.0, 256.0)  # Coarsen the global scalar by an exact factor of two.
            if next_scale == scale:  # Stop when the declared pathological upper scale is reached.
                break  # Leave the bracket incomplete so the explicit budget error below is raised.
            candidate = evaluate(next_scale)  # Preflight the coarser deterministic candidate.
            if candidate[2] <= int(equation_budget):  # Detect the first exactly feasible upper bracket endpoint.
                upper = candidate  # Retain the feasible side for bisection.
                break  # Stop geometric expansion once the bracket is complete.
            lower = candidate  # Retain the latest known-over-budget candidate.
            scale = next_scale  # Continue toward a coarser candidate on the next fixed expansion.
    if upper is None:  # Reject a budget that even the admissible coarsest field cannot satisfy.
        raise RuntimeError("no admissible supervised remesh satisfies the equation budget")  # Preserve the numerical failure instead of executing an over-budget solve.
    if lower is not None:  # Refine a complete infeasible-to-feasible logarithmic bracket deterministically.
        for _index in range(int(bisections)):  # Execute the preregistered fixed number of Gmsh bisection trials.
            midpoint = math.sqrt(lower[0] * upper[0])  # Bisect multiplicatively because mesh size and element counts follow power laws.
            candidate = evaluate(midpoint)  # Generate and exactly count the midpoint candidate.
            if candidate[2] <= int(equation_budget):  # Retain a feasible candidate as the coarser bracket side.
                upper = candidate  # Move the feasible endpoint toward the resource boundary.
            else:  # Retain an over-budget candidate as the finer bracket side.
                lower = candidate  # Move the infeasible endpoint toward the resource boundary.
    return BudgetMesh(mesh=upper[1], scale=upper[0], estimated_equations=upper[2], equation_budget=int(equation_budget), target_sha256=upper[3], mesh_sha256=upper[4], trials=tuple(trials))  # Return the finest bracketed feasible candidate and every preflight receipt.

def deploy_bridge_supervised(runner: FemRunner, model: Any, *, n_eq_budget: int, require_reference: bool = False, config: BridgeSupervisedConfig | None = None, mesh_generator: Callable[[Any, Any], Any] = generate_mesh, method: str = "supervised") -> SupervisedDeployment:  # Execute exactly common probe plus one budget-preflighted predicted remesh.
    settings = config or BridgeSupervisedConfig()  # Use the immutable frozen configuration unless an explicit test supplies an equivalent record.
    started = time.perf_counter()  # Start complete online timing before any optional reference-independent work.
    if require_reference:  # Permit standalone evaluation to construct a trusted reference explicitly.
        runner.ensure_reference()  # Build or reuse the runner reference only when requested by the caller.
    start_count = len(runner.records)  # Record the counted-solve boundary before supervised deployment.
    problem = runner.problem  # Read the immutable bridge problem from the isolated runner.
    probe_mesh = initial_mesh(problem)  # Generate the identical uniform-h0 probe used by every compared method.
    probe_post, probe_record = runner.solve_mesh(probe_mesh, method=method, stage="probe", extra={"common_probe": "uniform_h0", "equation_budget": int(n_eq_budget)})  # Spend the first and only probe solve.
    probe_eta2 = zz_indicator(problem, probe_post)  # Compute model features from the already-counted probe solution.
    probe_record.extra["sum_eta2"] = float(np.sum(probe_eta2))  # Preserve a reference-free accuracy diagnostic on the common probe.
    features = node_features(problem, probe_mesh, probe_post, probe_eta2)  # Build the existing instance-normalized nine-feature input.
    log_ratio = np.asarray(model.predict(features), dtype=float).reshape(-1)  # Execute one frozen-network forward pass on every probe node.
    if log_ratio.shape != (probe_mesh.n_nodes,) or np.any(~np.isfinite(log_ratio)):  # Reject malformed or nonfinite network output before Gmsh.
        raise RuntimeError("supervised model produced invalid nodal predictions")  # Retain the method failure rather than substituting another policy.
    clipped_log_ratio = np.clip(log_ratio, math.log(problem.h_min / problem.h0), 0.0)  # Restrict predictions to refinement-only expert-label support.
    predicted_sizes = problem.h0 * np.exp(clipped_log_ratio)  # Convert normalized log labels back to physical nodal sizes.
    budget_mesh = preflight_budget_mesh(problem, probe_mesh, predicted_sizes, probe_record.n_equations, int(n_eq_budget), gradation=settings.gradation, bisections=settings.preflight_bisections, expansions=settings.preflight_expansions, mesh_generator=mesh_generator)  # Scale and exactly preflight this budget under the frozen common gradation without reference-error feedback.
    deployment_extra = {"budget_scalar": budget_mesh.scale, "equation_budget": int(n_eq_budget), "preflight_equations": budget_mesh.estimated_equations, "preflight_feasible": budget_mesh.estimated_equations <= int(n_eq_budget), "preflight_trial_count": len(budget_mesh.trials), "target_sha256": budget_mesh.target_sha256, "mesh_sha256": budget_mesh.mesh_sha256, "common_probe": "uniform_h0", "deployment_contract": "probe_plus_one_predicted_remesh", "hold_last_after_solve": 2}  # Assemble the complete second-solve audit receipt.
    deployed_post, deployed_record = runner.solve_mesh(budget_mesh.mesh, method=method, stage="deployed", extra=deployment_extra)  # Spend the second and final real global solve.
    deployed_eta2 = zz_indicator(problem, deployed_post)  # Compute a reference-free final estimator for diagnostics and recovery.
    deployed_record.extra["sum_eta2"] = float(np.sum(deployed_eta2))  # Preserve the deployed estimator without changing selection metrics.
    deployed_record.extra["measured_budget_violation"] = bool(deployed_record.n_equations > int(n_eq_budget))  # Keep any estimator-to-CalculiX count discrepancy as a hard visible event.
    real_solve_count = len(runner.records) - start_count  # Count only solves performed by this isolated deployment invocation.
    if real_solve_count != 2:  # Enforce the protocol's exact two-real-solve supervised contract.
        raise RuntimeError(f"supervised deployment used {real_solve_count} real solves instead of 2")  # Refuse to hide an accidental extra solve.
    return SupervisedDeployment(probe_record=probe_record, deployed_record=deployed_record, budget_mesh=budget_mesh, real_solve_count=real_solve_count, hold_last_after_solve=2, online_wall_s=float(time.perf_counter() - started))  # Return complete counted records and scaling evidence.

def _expert_case(case: Mapping[str, Any], workdir: Path, config: BridgeSupervisedConfig, ccx_timeout: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:  # Generate one train-only exact-Dörfler expert label set.
    if case.get("split") != "train":  # Defend the label generator itself against validation or blind-case access.
        raise ValueError("expert labels may be generated only from train cases")  # Stop before reconstructing a non-training geometry.
    case_id = str(case["case_id"])  # Read the immutable manifest case identity.
    problem = problem_from_case(case)  # Reconstruct the exact manifest-authorized bridge problem.
    runner = FemRunner(problem, workdir / case_id, ccx_timeout=float(ccx_timeout))  # Isolate every counted expert solve and its evidence.
    mesh = initial_mesh(problem)  # Generate the common uniform probe once for both features and expert trajectory.
    post, record = runner.solve_mesh(mesh, method="supervised_expert", stage="probe", extra={"theta": config.theta, "equation_budget": config.expert_equation_budget})  # Spend the first expert solve without creating a duplicate probe.
    eta2 = zz_indicator(problem, post)  # Compute the first exact elementwise ZZ indicator.
    record.extra["sum_eta2"] = float(np.sum(eta2))  # Preserve the first expert estimator sum.
    features = node_features(problem, mesh, post, eta2)  # Freeze network inputs on the common probe mesh.
    probe_mesh = mesh  # Retain the exact feature mesh for final-label interpolation.
    stop_reason = "max_solves"  # Default to the hard complete-trajectory solve cap.
    while len(runner.records) < config.expert_max_solves:  # Iterate exact solve-estimate-mark-remesh steps up to the frozen cap.
        if record.n_equations >= config.expert_equation_budget:  # Stop once the currently delivered expert mesh reaches the active cap.
            stop_reason = "equation_cap_reached"  # Record the exact resource stop.
            break  # Preserve the last feasible expert mesh as the label source.
        marked = dorfler_mark(eta2, config.theta)  # Compute the exact minimal-cardinality bulk marking on current elements.
        record.extra["n_marked"] = int(len(marked))  # Preserve the exact current marked-element count.
        if len(marked) == 0:  # Stop a converged or degenerate trajectory without fabricating refinements.
            stop_reason = "empty_marking"  # Record the exact numerical stop.
            break  # Preserve the current mesh as the expert label source.
        target = refine_size_map(mesh, marked, factor=0.5)  # Apply the same element-to-node Dörfler refinement atom used by the baseline.
        field = NodalSizeField(mesh, target, gradation=config.gradation, h_min=problem.h_min, h_max=problem.h0)  # Compile the exact Dörfler nodal target.
        candidate = generate_mesh(problem, field)  # Generate the next conforming expert mesh without spending a solve.
        candidate_equations = estimate_active_equations(problem, candidate)  # Preflight exact active equations before the next expert solve.
        record.extra["next_preflight_equations"] = candidate_equations  # Preserve the candidate resource receipt on its source solve.
        if candidate_equations > config.expert_equation_budget:  # Keep expert labels at the best actually feasible trajectory prefix.
            stop_reason = "next_mesh_exceeds_equation_cap"  # Record the exact unsolved preflight stop.
            break  # Avoid an unnecessary over-budget expert solve.
        mesh = candidate  # Advance to the preflight-certified next Dörfler mesh.
        post, record = runner.solve_mesh(mesh, method="supervised_expert", stage=f"cycle{len(runner.records)}", extra={"theta": config.theta, "equation_budget": config.expert_equation_budget, "preflight_equations": candidate_equations})  # Spend exactly one counted solve for the next expert state.
        eta2 = zz_indicator(problem, post)  # Recompute ZZ on the newly realized expert mesh.
        record.extra["sum_eta2"] = float(np.sum(eta2))  # Preserve the realized expert estimator sum.
    record.extra["stop"] = stop_reason  # Attach the terminal reason to the final delivered expert record.
    expert_mesh: Mesh = runner.last_mesh  # Read the final counted feasible Dörfler mesh.
    interpolator = NodalSizeField(expert_mesh, expert_mesh.node_sizes, gradation=1.2, h_min=problem.h_min, h_max=problem.h0)  # Build a smooth expert-size label field on the final mesh.
    expert_sizes = np.asarray([interpolator(*point) for point in probe_mesh.nodes], dtype=np.float32)  # Interpolate the final expert field onto common-probe nodes.
    labels = np.log(np.maximum(expert_sizes, problem.h_min) / problem.h0).astype(np.float32)  # Encode the existing normalized log-size regression target.
    runner.dump(workdir / case_id / "expert_records.json")  # Persist every counted expert solve for independent audit.
    metadata = {"case_id": case_id, "split": "train", "sample_count": int(labels.size), "real_solve_count": len(runner.records), "stop_reason": stop_reason, "final_equations": int(record.n_equations), "final_elements": int(expert_mesh.n_cells), "final_mesh_sha256": mesh_sha256(expert_mesh)}  # Summarize this retained training case without reference metrics.
    return features.astype(np.float32), labels, metadata  # Return common-probe features, Dörfler labels, and exact cost evidence.

def generate_bridge_expert_dataset(train_cases: Sequence[Mapping[str, Any]], output_dir: Path | str, config: BridgeSupervisedConfig | None = None, *, ccx_timeout: float = 1800.0) -> tuple[Path, dict[str, Any]]:  # Generate all 24 train-only expert trajectories and one immutable dataset artifact.
    settings = config or BridgeSupervisedConfig()  # Use the frozen supervised settings by default.
    ordered = tuple(sorted((dict(case) for case in train_cases), key=lambda case: str(case["case_id"])))  # Freeze deterministic execution order independently of caller ordering.
    if len(ordered) != EXPECTED_SPLIT_COUNTS["train"] or any(case.get("split") != "train" for case in ordered):  # Require exactly the manifest's 24 training cases and no other split.
        raise ValueError("expert dataset requires exactly 24 train cases")  # Stop before any partial or contaminated expert campaign.
    root = Path(output_dir)  # Normalize the supervised training artifact root.
    expert_root = root / "experts"  # Isolate counted expert trajectories from network artifacts.
    started = time.perf_counter()  # Measure complete solver, meshing, feature, and label generation time.
    feature_blocks: list[np.ndarray] = []  # Accumulate per-case common-probe feature matrices.
    label_blocks: list[np.ndarray] = []  # Accumulate per-case Dörfler expert label vectors.
    metadata_rows: list[dict[str, Any]] = []  # Accumulate exact per-case cost and provenance evidence.
    offsets = [0]  # Record sample boundaries for every manifest training case.
    for case in ordered:  # Execute every training case exactly once in stable case-ID order.
        features, labels, metadata = _expert_case(case, expert_root, settings, float(ccx_timeout))  # Build one exact-Dörfler expert label block.
        feature_blocks.append(features)  # Retain the common-probe inputs.
        label_blocks.append(labels)  # Retain the final-mesh expert labels.
        metadata_rows.append(metadata)  # Retain the counted solve and mesh evidence.
        offsets.append(offsets[-1] + int(labels.size))  # Close this case's half-open sample interval.
    features_all = np.vstack(feature_blocks).astype(np.float32)  # Concatenate all cases without losing feature precision.
    labels_all = np.concatenate(label_blocks).astype(np.float32)  # Concatenate all labels in identical case order.
    dataset_path = root / "expert_dataset.npz"  # Freeze the model-training array filename.
    dataset_path.parent.mkdir(parents=True, exist_ok=True)  # Create the supervised training root only after input checks.
    np.savez_compressed(dataset_path, X=features_all, y=labels_all, case_offsets=np.asarray(offsets, dtype=np.int64), case_ids=np.asarray([str(case["case_id"]) for case in ordered], dtype="U64"))  # Persist arrays plus auditable case boundaries without pickle objects.
    summary = {"schema": SUPERVISED_SCHEMA, "protocol_id": PROTOCOL_ID, "split": "train", "case_count": len(ordered), "case_ids": [str(case["case_id"]) for case in ordered], "sample_count": int(labels_all.size), "feature_count": int(features_all.shape[1]), "expert_real_solve_count": int(sum(row["real_solve_count"] for row in metadata_rows)), "expert_wall_s": float(time.perf_counter() - started), "dataset_path": str(dataset_path), "dataset_sha256": file_sha256(dataset_path), "cases": metadata_rows}  # Assemble complete expert cost and provenance evidence.
    write_json(root / "expert_dataset_metadata.json", summary)  # Persist the train-only dataset summary before network fitting.
    return dataset_path, summary  # Return the exact dataset artifact and its complete evidence.

def train_candidate_networks(dataset_path: Path | str, output_dir: Path | str, config: BridgeSupervisedConfig | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # Fit and hash exactly three independent fixed-seed networks.
    settings = config or BridgeSupervisedConfig()  # Use the immutable frozen settings by default.
    if tuple(settings.network_seeds) != NETWORK_SEEDS:  # Prevent accidental best-of-more or post-hoc seed substitution.
        raise ValueError(f"network seeds must be exactly {NETWORK_SEEDS}")  # Reject a scientifically different candidate pool.
    data = np.load(Path(dataset_path), allow_pickle=False)  # Load only numerical arrays from the train-only dataset artifact.
    features = np.asarray(data["X"], dtype=np.float32)  # Normalize the network input matrix.
    labels = np.asarray(data["y"], dtype=np.float32)  # Normalize the one-dimensional log-size target.
    if features.ndim != 2 or features.shape[1] != N_FEATURES or labels.shape != (features.shape[0],):  # Validate the exact existing MLP training contract.
        raise ValueError("expert dataset has incompatible feature or label shapes")  # Stop before fitting a malformed dataset.
    root = Path(output_dir)  # Normalize the supervised artifact root.
    candidates_root = root / "candidates"  # Isolate all three seed checkpoints for audit.
    config_payload = supervised_config_payload(settings)  # Materialize every frozen scientific setting.
    config_sha = canonical_json_sha256(config_payload)  # Bind all checkpoints to one exact supervised configuration identity.
    write_json(root / "supervised_config.json", {**config_payload, "config_sha256": config_sha})  # Persist the hash-bearing configuration before checkpoint generation.
    metadata_rows: list[dict[str, Any]] = []  # Accumulate exact model, seed, cost, and loss evidence.
    total_started = time.perf_counter()  # Measure aggregate network-only training time.
    for seed in settings.network_seeds:  # Fit exactly one independent MLP for each preregistered seed.
        model_config = SupervisedConfig(hidden=settings.hidden, lr=settings.learning_rate, epochs=settings.epochs, batch=settings.batch_size, seed=int(seed))  # Translate the frozen protocol record into the existing MLP configuration.
        model = SizeMLP(model_config)  # Initialize one independent seeded network through the existing route.
        started = time.perf_counter()  # Measure this seed's optimizer wall time independently.
        losses = model.fit(features, labels)  # Fit only the 24-case expert dataset without validation or blind labels.
        training_wall_s = float(time.perf_counter() - started)  # Close the seed-specific network training timer.
        seed_root = candidates_root / f"seed_{seed}"  # Allocate one immutable candidate checkpoint directory.
        seed_root.mkdir(parents=True, exist_ok=True)  # Create the candidate directory only after successful fitting.
        model_path = seed_root / "model.pt"  # Freeze the candidate checkpoint filename.
        model.save(model_path)  # Persist the exact PyTorch state dictionary for later validation and deployment.
        parameter_count = int(sum(parameter.numel() for parameter in model.net.parameters()))  # Count all trainable and nontrainable tensor scalars in the network.
        metadata = {"schema": SUPERVISED_SCHEMA, "protocol_id": PROTOCOL_ID, "seed": int(seed), "config_sha256": config_sha, "model_path": str(model_path), "model_sha256": file_sha256(model_path), "training_wall_s": training_wall_s, "sample_count": int(labels.size), "parameter_count": parameter_count, "epoch_count": len(losses), "initial_train_mse": float(losses[0]), "final_train_mse": float(losses[-1])}  # Assemble the exact candidate checkpoint receipt.
        write_json(seed_root / "losses.json", {"seed": int(seed), "losses": [float(value) for value in losses]})  # Persist the complete preregistered training trajectory.
        write_json(seed_root / "metadata.json", metadata)  # Persist model identity and cost beside its checkpoint.
        metadata_rows.append(metadata)  # Retain this candidate for joint validation and aggregate reporting.
    summary = {"schema": SUPERVISED_SCHEMA, "protocol_id": PROTOCOL_ID, "candidate_count": len(metadata_rows), "seeds": list(settings.network_seeds), "sample_count": int(labels.size), "parameter_count": int(metadata_rows[0]["parameter_count"]), "network_training_wall_s": float(time.perf_counter() - total_started), "config_sha256": config_sha, "dataset_sha256": file_sha256(dataset_path), "candidates": metadata_rows}  # Summarize exact three-seed offline network cost and identities.
    write_json(root / "network_training_summary.json", summary)  # Persist the complete three-candidate training evidence.
    return metadata_rows, summary  # Return candidate checkpoint records and aggregate cost evidence.

def load_frozen_supervised_model(model_path: Path | str, *, selected_seed: int, expected_sha256: str | None = None, config: BridgeSupervisedConfig | None = None) -> tuple[SizeMLP, dict[str, Any]]:  # Reconstruct and verify the sole validation-selected deployment network.
    settings = config or BridgeSupervisedConfig()  # Use the same immutable architecture record used during three-seed fitting.
    seed = int(selected_seed)  # Normalize the frozen validation-selected seed.
    if seed not in settings.network_seeds:  # Reject a seed outside the exact preregistered candidate pool.
        raise ValueError(f"selected supervised seed must be one of {settings.network_seeds}")  # Prevent a fresh or test-informed checkpoint substitution.
    path = Path(model_path)  # Normalize the exact selected checkpoint path.
    observed_sha = file_sha256(path)  # Recompute model identity immediately before loading deployment weights.
    if expected_sha256 is not None and observed_sha != str(expected_sha256):  # Reject mutation after the pre-test freeze.
        raise RuntimeError("frozen supervised model SHA-256 mismatch")  # Stop before any blind-case probe solve.
    model_config = SupervisedConfig(hidden=settings.hidden, lr=settings.learning_rate, epochs=settings.epochs, batch=settings.batch_size, seed=seed)  # Reconstruct the exact existing MLP architecture.
    model = SizeMLP(model_config)  # Allocate the architecture before loading the frozen state dictionary.
    model.load(path)  # Load only the exact selected state dictionary through the existing weights-only route.
    model.net.eval()  # Freeze inference behavior explicitly even though this MLP has no stochastic dropout layers.
    receipt = {"schema": SUPERVISED_SCHEMA, "protocol_id": PROTOCOL_ID, "selected_seed": seed, "model_path": str(path), "model_sha256": observed_sha, "config_sha256": canonical_json_sha256(supervised_config_payload(settings)), "training_split_accessed": False, "validation_split_accessed": False, "test_metrics_accessed": False}  # Return complete blind-deployment provenance and isolation declarations.
    return model, receipt  # Return the verified frozen network and JSON-safe exact artifact receipt.

def _reference_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:  # Extract a Reference-compatible mapping from common reference-B report wrappers.
    if isinstance(payload.get("reference_B"), Mapping):  # Accept a case report with an explicit reference-B member.
        return _reference_payload(payload["reference_B"])  # Recurse through one transparent wrapper layer.
    if isinstance(payload.get("reference"), Mapping):  # Accept a runner dump or reference-stage wrapper.
        return _reference_payload(payload["reference"])  # Recurse through one transparent wrapper layer.
    return payload  # Treat the current mapping as the direct Reference record.

def load_reference_b(reference_root: Path | str, case_id: str) -> Reference:  # Load one precomputed validation reference B without performing another hidden solve.
    root = Path(reference_root)  # Normalize the common reference artifact root.
    candidates = (root / case_id / "reference_B.json", root / case_id / "reference_B" / "reference.json", root / case_id / "reference_B" / "summary.json", root / f"{case_id}.reference_B.json")  # Support the repository's likely transparent per-case layouts.
    reference_path = next((path for path in candidates if path.is_file()), None)  # Select the first explicit existing reference-B artifact deterministically.
    if reference_path is None:  # Reject missing validation truth rather than choosing by test or estimator behavior.
        raise FileNotFoundError(f"reference_B not found for validation case {case_id}")  # Identify the exact missing development case.
    raw = json.loads(reference_path.read_text(encoding="utf-8"))  # Decode the trusted common reference artifact.
    if not isinstance(raw, Mapping):  # Require a transparent JSON object wrapper.
        raise ValueError(f"reference_B for {case_id} must be a JSON object")  # Reject opaque or malformed truth data.
    values = _reference_payload(raw)  # Extract the direct repository Reference fields.
    required = ("U_total", "qoi", "n_equations", "n_elems", "h_ref")  # Freeze the exact common reference fields needed by FemRunner.
    if any(name not in values for name in required):  # Reject incomplete truth that could produce incomparable metrics.
        raise ValueError(f"reference_B for {case_id} lacks required fields")  # Surface the exact development-data failure.
    return Reference(U_total=float(values["U_total"]), qoi=float(values["qoi"]), n_equations=int(values["n_equations"]), n_elems=int(values["n_elems"]), h_ref=float(values["h_ref"]))  # Reconstruct the existing immutable reference contract.

def _valid_error(value: Any) -> bool:  # Classify one relative validation error without accepting NaN, infinity, or negatives.
    return value is not None and math.isfinite(float(value)) and float(value) >= 0.0  # Preserve every finite nonnegative successful metric including zero.

def validation_score(rows: Sequence[Mapping[str, Any]], *, expected_cases: int = 8, budgets: Sequence[int] = VALIDATION_BUDGETS, failure_error: float = VALIDATION_FAILURE_ERROR) -> dict[str, Any]:  # Compute the preregistered lexicographic validation checkpoint key.
    expected_keys = {(str(case_id), int(budget)) for case_id in sorted({str(row.get("case_id")) for row in rows}) for budget in budgets}  # Build the observed-case Cartesian product with frozen budgets.
    observed_keys = {(str(row.get("case_id")), int(row.get("equation_budget", -1))) for row in rows}  # Collect unique supplied validation operating points.
    if len({str(row.get("case_id")) for row in rows}) != int(expected_cases) or len(rows) != int(expected_cases) * len(tuple(budgets)) or observed_keys != expected_keys:  # Require exactly eight cases by three budgets with no omissions or duplicates.
        raise ValueError("validation rows must form the complete 8-case by 3-budget grid")  # Prevent favorable post-hoc point dropping.
    seeds = {int(row.get("seed", -1)) for row in rows}  # Recover the candidate seed represented by this complete grid.
    if len(seeds) != 1:  # Require one scientifically coherent checkpoint per score.
        raise ValueError("validation rows must contain exactly one network seed")  # Reject accidental cross-checkpoint aggregation.
    energy_logs: list[float] = []  # Accumulate finite penalized energy log errors.
    qoi_logs: list[float] = []  # Accumulate finite penalized QoI log errors.
    failure_points = 0  # Count any operating point lacking both valid required metrics and successful execution.
    budget_violations = 0  # Count measured active-equation violations separately as the fourth tie-break.
    for row in sorted(rows, key=lambda value: (str(value["case_id"]), int(value["equation_budget"]))):  # Score in stable order independently of execution order.
        energy_ok = row.get("status") == "ok" and _valid_error(row.get("energy_error"))  # Validate the reference-B energy error and execution status.
        qoi_ok = row.get("status") == "ok" and _valid_error(row.get("qoi_error"))  # Validate the reference-B QoI error and execution status.
        if not energy_ok or not qoi_ok:  # Count a case-budget point once even if both metrics failed.
            failure_points += 1  # Preserve the primary failure count as the first lexicographic component.
        energy_value = float(row["energy_error"]) if energy_ok else float(failure_error)  # Apply the fixed finite penalty only to invalid energy results.
        qoi_value = float(row["qoi_error"]) if qoi_ok else float(failure_error)  # Apply the same fixed finite penalty to invalid QoI results.
        energy_logs.append(math.log(max(energy_value, VALIDATION_ERROR_FLOOR)))  # Aggregate equal-weight energy errors in log space.
        qoi_logs.append(math.log(max(qoi_value, VALIDATION_ERROR_FLOOR)))  # Aggregate equal-weight QoI errors in log space.
        budget_violations += int(bool(row.get("budget_violation", False)))  # Preserve every measured deployment cap breach.
    seed = next(iter(seeds))  # Read the unique validated checkpoint seed.
    energy_log_mean = float(sum(energy_logs) / len(energy_logs))  # Compute the equal-weight finite energy log mean over all 24 points.
    qoi_log_mean = float(sum(qoi_logs) / len(qoi_logs))  # Compute the equal-weight finite QoI log mean over all 24 points.
    key = [int(failure_points), energy_log_mean, qoi_log_mean, int(budget_violations), int(seed)]  # Freeze the exact lexicographic selection tuple requested before test.
    return {"seed": int(seed), "failure_point_count": int(failure_points), "energy_error_log_mean": energy_log_mean, "energy_error_geometric_mean": float(math.exp(energy_log_mean)), "qoi_error_log_mean": qoi_log_mean, "qoi_error_geometric_mean": float(math.exp(qoi_log_mean)), "budget_violation_count": int(budget_violations), "selection_key": key, "failure_error": float(failure_error), "point_count": len(rows)}  # Return finite auditable checkpoint-selection evidence.

def select_validation_checkpoint(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:  # Select exactly one frozen checkpoint by the preregistered lexicographic key.
    if len(scores) != 3 or {int(score["seed"]) for score in scores} != set(NETWORK_SEEDS):  # Require all and only the three frozen candidates.
        raise ValueError(f"checkpoint selection requires seeds {NETWORK_SEEDS}")  # Reject best-of-fewer, extra-seed, or substituted candidate pools.
    selected = min((dict(score) for score in scores), key=lambda score: tuple(score["selection_key"]))  # Apply the exact ordered failure, energy, QoI, budget, and seed tie-break.
    return selected  # Return an independent record of the uniquely selected checkpoint.

def validate_candidate_networks(validation_cases: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]], output_dir: Path | str, reference_root: Path | str, config: BridgeSupervisedConfig | None = None, *, ccx_timeout: float = 1800.0, allow_unqualified_references: bool = False, expedited_reference_levels: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # Evaluate all three candidates on validation only under one explicit strict or amended reference policy.
    settings = config or BridgeSupervisedConfig()  # Use the frozen configuration unless a test supplies an equivalent record.
    if bool(allow_unqualified_references) != (expedited_reference_levels is not None):  # Require both exceptional controls together so a truthy flag or depth cannot silently weaken reference qualification.
        raise ValueError("allow_unqualified_references and expedited_reference_levels must be specified together")  # Stop before loading any validation denominator or checkpoint.
    if allow_unqualified_references and expedited_reference_levels != 2:  # Match the sole depth authorized by the protected rapid-execution amendment and later freeze configuration.
        raise ValueError("expedited_reference_levels must equal the amended value 2")  # Reject a costly validation run that could never pass the dedicated freeze contract.
    cases = tuple(sorted((dict(case) for case in validation_cases), key=lambda case: str(case["case_id"])))  # Freeze deterministic validation execution order.
    if len(cases) != EXPECTED_SPLIT_COUNTS["validation"] or any(case.get("split") != "validation" for case in cases):  # Require exactly the manifest's eight validation cases.
        raise ValueError("checkpoint validation requires exactly 8 validation cases")  # Stop before any blind or training-case evaluation.
    candidate_map = {int(item["seed"]): dict(item) for item in candidates}  # Index the complete trained candidate set by preregistered seed.
    if set(candidate_map) != set(NETWORK_SEEDS):  # Require all and only the three frozen network candidates.
        raise ValueError(f"candidate checkpoints must use seeds {NETWORK_SEEDS}")  # Reject post-hoc checkpoint-pool changes.
    root = Path(output_dir)  # Normalize the supervised artifact root.
    validation_root = root / "validation"  # Isolate development selection evidence from training arrays and checkpoints.
    all_rows: list[dict[str, Any]] = []  # Accumulate every seed-case-budget deployment result without dropping failures.
    score_rows: list[dict[str, Any]] = []  # Accumulate one complete-grid lexicographic score per candidate.
    validation_started = time.perf_counter()  # Measure total validation-only online cost.
    for seed in NETWORK_SEEDS:  # Evaluate candidates in preregistered seed order.
        candidate = candidate_map[seed]  # Read this seed's exact model artifact receipt.
        model_config = SupervisedConfig(hidden=settings.hidden, lr=settings.learning_rate, epochs=settings.epochs, batch=settings.batch_size, seed=seed)  # Reconstruct the exact network architecture and initialization metadata.
        model = SizeMLP(model_config)  # Allocate the existing architecture before loading frozen weights.
        model.load(Path(candidate["model_path"]))  # Load the exact trained state dictionary identified by candidate SHA.
        seed_rows: list[dict[str, Any]] = []  # Accumulate the complete 8-by-3 grid for this candidate.
        for case in cases:  # Evaluate every validation geometry without touching the blind split.
            case_id = str(case["case_id"])  # Read the immutable validation case identity.
            problem = problem_from_case(case)  # Reconstruct only this authorized validation geometry.
            from ..vla.four_way_references import load_reference_b as load_verified_reference_b  # Import the authenticated common-reference loader lazily to avoid coupling basic utilities.
            from ..vla.four_way_references import verify_reference_cache as verify_common_reference  # Import the qualification-aware verifier beside the loader so every row carries exact denominator status.
            reference_receipt = verify_common_reference(reference_root, case_id=case_id, problem=problem, allow_unqualified=bool(allow_unqualified_references), expedited_levels=expedited_reference_levels)  # Recompute cache integrity and preserve the original convergence-gate outcome.
            reference = load_verified_reference_b(reference_root, case_id=case_id, problem=problem, verify=True, allow_unqualified=bool(allow_unqualified_references), expedited_levels=expedited_reference_levels)  # Load only through the caller's explicit strict or amended reference contract.
            for budget in settings.validation_budgets:  # Deploy the same frozen network with only deterministic budget scaling changed.
                run_root = validation_root / f"seed_{seed}" / case_id / f"B{budget}"  # Isolate counted solves for this exact validation point.
                runner = FemRunner(problem, run_root, ccx_timeout=float(ccx_timeout))  # Create an independent two-solve deployment runner.
                runner.reference = reference  # Attach common reference B directly without generating a method-dependent reference.
                row: dict[str, Any] = {"seed": seed, "case_id": case_id, "split": "validation", "equation_budget": int(budget), "status": "failed", "energy_error": None, "qoi_error": None, "measured_equations": None, "budget_violation": False, "real_solve_count": 0, "hold_last_after_solve": 2, "model_sha256": candidate["model_sha256"], "reference_status": str(reference_receipt["status"]), "reference_qualification": bool(reference_receipt["qualification"]), "reference_authorization": reference_receipt.get("authorization"), "expedited_reference_levels": expedited_reference_levels}  # Initialize a finite JSON-safe retained-failure record with explicit denominator qualification.
                try:  # Retain numerical failures as validation outcomes instead of deleting unfavorable points.
                    deployment = deploy_bridge_supervised(runner, model, n_eq_budget=int(budget), require_reference=False, config=settings)  # Execute exactly probe plus one preflighted predicted remesh.
                    deployed = deployment.deployed_record  # Select the second real solve as the frozen supervised deliverable.
                    row.update({"status": "ok", "energy_error": None if deployed.e_energy is None else float(deployed.e_energy), "qoi_error": None if deployed.e_qoi is None else float(deployed.e_qoi), "measured_equations": int(deployed.n_equations), "preflight_equations": int(deployment.budget_mesh.estimated_equations), "budget_scalar": float(deployment.budget_mesh.scale), "budget_violation": bool(deployed.n_equations > int(budget)), "real_solve_count": int(deployment.real_solve_count), "online_wall_s": float(deployment.online_wall_s), "target_sha256": deployment.budget_mesh.target_sha256, "mesh_sha256": deployment.budget_mesh.mesh_sha256})  # Preserve the complete successful validation receipt.
                except (CalculiXExecutionError, GmshMeshingError) as error:  # Convert only explicit native numerical failures into the fixed retained validation penalty.
                    row["failure_type"] = type(error).__name__  # Preserve the exact exception class without nonfinite sentinels.
                    row["failure_message"] = str(error)[-2000:]  # Preserve a bounded diagnostic tail for audit and recovery.
                    row["real_solve_count"] = len(runner.records)  # Count any real solves spent before the retained failure.
                runner.dump(run_root / "records.json")  # Persist all counted solves even when deployment failed partway.
                write_json(run_root / "validation_result.json", row)  # Persist this exact seed-case-budget outcome immediately.
                seed_rows.append(row)  # Retain the point for complete-grid finite scoring.
                all_rows.append(row)  # Retain the point for the global validation evidence table.
        score = validation_score(seed_rows, expected_cases=8, budgets=settings.validation_budgets, failure_error=settings.validation_failure_error)  # Compute this candidate's preregistered complete-grid key.
        score["model_path"] = candidate["model_path"]  # Bind the score to the exact checkpoint path.
        score["model_sha256"] = candidate["model_sha256"]  # Bind the score to the exact checkpoint content.
        write_json(validation_root / f"seed_{seed}" / "score.json", score)  # Persist the candidate key before cross-seed selection.
        score_rows.append(score)  # Retain this complete candidate for lexicographic comparison.
    selected = select_validation_checkpoint(score_rows)  # Select one network with the frozen failure-energy-QoI-budget-seed ordering.
    selected_candidate = candidate_map[int(selected["seed"])]  # Recover the selected checkpoint's training receipt.
    selected_path = root / "selected_model.pt"  # Freeze the one deployment checkpoint filename.
    shutil.copyfile(Path(selected_candidate["model_path"]), selected_path)  # Copy exact selected bytes without retraining or test access.
    selected_hash = file_sha256(selected_path)  # Verify the frozen copy's exact content identity.
    if selected_hash != selected_candidate["model_sha256"]:  # Guard against incomplete or altered checkpoint publication.
        raise RuntimeError("selected supervised model hash differs from its candidate checkpoint")  # Stop before exposing an inconsistent frozen model.
    selection = {"schema": SUPERVISED_SCHEMA, "protocol_id": PROTOCOL_ID, "selection_rule": ["failure_point_count", "energy_error_log_mean", "qoi_error_log_mean", "budget_violation_count", "seed"], "selected_seed": int(selected["seed"]), "selected_model_path": str(selected_path), "selected_model_sha256": selected_hash, "config_sha256": selected_candidate["config_sha256"], "validation_case_count": len(cases), "validation_point_count_per_seed": len(cases) * len(settings.validation_budgets), "validation_wall_s": float(time.perf_counter() - validation_started), "allow_unqualified_references": bool(allow_unqualified_references), "expedited_reference_levels": expedited_reference_levels, "reference_qualification_by_case": {str(case_id): bool(next(row["reference_qualification"] for row in all_rows if str(row["case_id"]) == str(case_id))) for case_id in sorted({str(row["case_id"]) for row in all_rows})}, "scores": score_rows}  # Assemble complete pre-test checkpoint-selection provenance including every denominator's qualification.
    write_json(root / "validation_rows.json", {"schema": SUPERVISED_SCHEMA, "protocol_id": PROTOCOL_ID, "rows": all_rows})  # Persist every success and retained failure used by selection.
    write_json(root / "selected_model.json", selection)  # Persist the sole frozen deployment checkpoint identity and selection evidence.
    return selection, all_rows  # Return the selected checkpoint receipt and complete validation table.

def build_training_summary(expert_summary: Mapping[str, Any], network_summary: Mapping[str, Any], selection: Mapping[str, Any], config: BridgeSupervisedConfig | None = None) -> dict[str, Any]:  # Assemble the protocol-required offline cost and frozen-artifact report.
    settings = config or BridgeSupervisedConfig()  # Use the same immutable configuration represented by the selected model.
    return {"schema": SUPERVISED_SCHEMA, "protocol_id": PROTOCOL_ID, "training_split_case_count": int(expert_summary["case_count"]), "validation_split_case_count": int(selection["validation_case_count"]), "expert_calculix_solve_count": int(expert_summary["expert_real_solve_count"]), "expert_generation_wall_s": float(expert_summary["expert_wall_s"]), "network_training_wall_s": float(network_summary["network_training_wall_s"]), "validation_wall_s": float(selection["validation_wall_s"]), "training_sample_count": int(expert_summary["sample_count"]), "model_parameter_count": int(network_summary["parameter_count"]), "dataset_sha256": str(expert_summary["dataset_sha256"]), "config_sha256": str(network_summary["config_sha256"]), "candidate_model_sha256": {str(row["seed"]): str(row["model_sha256"]) for row in network_summary["candidates"]}, "selected_seed": int(selection["selected_seed"]), "selected_model_sha256": str(selection["selected_model_sha256"]), "network_seeds": list(settings.network_seeds), "validation_budgets": list(settings.validation_budgets), "allow_unqualified_references": bool(selection.get("allow_unqualified_references", False)), "expedited_reference_levels": selection.get("expedited_reference_levels"), "reference_qualification_by_case": dict(selection.get("reference_qualification_by_case", {})), "test_split_accessed": False, "test_results_used": False}  # Return every mandatory cost, count, hash, reference-policy, and anti-leakage declaration.
