"""Paired exact-Dörfler and world-model VLA benchmark harness."""  # Describe the clean-method comparison implemented below.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from dataclasses import asdict, dataclass, is_dataclass  # Import benchmark contracts and record serialization support.
import inspect  # Import constructor signature adaptation.
import json  # Import benchmark-result serialization.
from pathlib import Path  # Import portable campaign paths.
from typing import Any  # Import generic repository runner and record types.
import numpy as np  # Import curve and Pareto-envelope calculations.
from ...bridge_cases import make_box_girder_diaphragm  # Import the canonical repository bridge-case factory.
from ...experiment import FemRunner  # Reuse the repository CalculiX experiment runner.
from .model import RegionAction, ResidualWorldModel, WorldModelConfig  # Import exact-Dörfler and learned world-model actions.
from .pipeline import WorldVLAConfig, WorldVLAResult, _RuntimeAdapter, _record_metric, run_world_model_vla  # Reuse the common real-solve runtime and metric selection.
from .planner import MultiStepPlanner, PlannerConfig  # Import finite-horizon planning.
from .tool_gateway import MCPToolGateway, MeshCertificate, ToolConfig  # Import exact action materialization and certificates.
from .vision_partition import CachedVisionPartition  # Import the fixed semantic vision output.

@dataclass(frozen=True)  # Make the exact-Dörfler trajectory immutable.
class DorflerResult:  # Store the paired exact-Dörfler baseline trajectory.
    records: tuple[Any, ...]  # Store real finite-element solve records.
    indicator_sums: tuple[float, ...]  # Store real global squared-indicator sums.
    certificates: tuple[MeshCertificate, ...]  # Store exact-Dörfler materialization evidence.
    stop_reason: str  # Explain the baseline stopping condition.

@dataclass(frozen=True)  # Make the paired benchmark result immutable.
class BenchmarkResult:  # Store paired trajectory curves and predeclared scientific gates.
    dorfler: DorflerResult  # Store the exact-Dörfler baseline.
    world_vla: WorldVLAResult  # Store the world-model VLA trajectory.
    dorfler_metrics: tuple[float, ...]  # Store best available error metrics by solve.
    world_metrics: tuple[float, ...]  # Store best available error metrics by solve.
    dorfler_equations: tuple[int, ...]  # Store active equations by solve.
    world_equations: tuple[int, ...]  # Store active equations by solve.
    solvewise_ratios: tuple[float, ...]  # Store world-to-Dörfler error ratios at equal real-solve count.
    budgetwise_ratios: tuple[float, ...]  # Store world-to-Dörfler best error ratios on common equation caps.
    world_action_count: int  # Store how many certified non-zero world actions executed.
    dorfler_inclusion_all: bool  # Store whether every world target included exact Dörfler.
    solvewise_noninferior: bool  # Store the equal-solve Dörfler-floor gate.
    budgetwise_noninferior: bool  # Store the common-budget Dörfler-floor gate.
    terminal_advantage: bool  # Store whether the world route produced a measurable terminal gain.

def _make_runner(problem: Any, workdir: Path, n_equation_cap: int) -> Any:  # Construct FemRunner across minor repository constructor revisions.
    workdir.mkdir(parents=True, exist_ok=True)  # Create the isolated method directory before runner construction.
    signature = inspect.signature(FemRunner)  # Inspect the active runner constructor.
    kwargs: dict[str, Any] = {}  # Collect supported constructor arguments.
    unmapped_required = False  # Track whether the active constructor exposes an unknown required argument.
    for name, parameter in signature.parameters.items():  # Map stable experiment concepts to current argument names.
        lower = name.lower()  # Normalize the constructor field.
        if lower in ("problem", "case"):  # Match the finite-element problem.
            kwargs[name] = problem  # Supply the benchmark problem.
        elif lower in ("workdir", "root", "output_dir", "outdir", "directory"):  # Match the experiment directory.
            kwargs[name] = workdir  # Supply the isolated method directory.
        elif lower in ("n_eq_cap", "n_equation_cap", "equation_cap", "budget"):  # Match an optional resource cap.
            kwargs[name] = int(n_equation_cap)  # Supply the benchmark resource cap.
        elif parameter.default is inspect.Parameter.empty and lower not in ("self",):  # Detect an unknown required constructor field.
            unmapped_required = True  # Request bounded positional fallback below.
    if not unmapped_required:  # Prefer auditable keyword construction when every required field is known.
        try:  # Isolate a keyword-signature mismatch.
            return FemRunner(**kwargs)  # Construct the repository runner.
        except TypeError:  # Support positional-only implementations.
            pass  # Continue to bounded common signatures.
    attempts = ((problem, workdir), (problem, str(workdir)), (problem, workdir, int(n_equation_cap)), (problem, str(workdir), int(n_equation_cap)))  # Define bounded common repository constructor patterns.
    last_error: Exception | None = None  # Preserve the final constructor error for diagnosis.
    for arguments in attempts:  # Try only stable physical argument orders.
        try:  # Isolate each constructor pattern.
            return FemRunner(*arguments)  # Construct the repository runner.
        except TypeError as error:  # Capture signature mismatches only.
            last_error = error  # Preserve the diagnostic error.
    raise TypeError("FemRunner constructor could not be adapted") from last_error  # Refuse to invent solver parameters.

def _record_equations(record: Any) -> int:  # Recover measured active equations from a solve record.
    for name in ("n_equations", "n_eq", "equations", "neq"):  # Inspect supported resource fields.
        if hasattr(record, name):  # Use the first available measured field.
            return int(getattr(record, name))  # Return the measured active equation count.
    return 0  # Preserve curve construction when legacy records omit the field.

def _record_payload(record: Any) -> dict[str, Any]:  # Serialize a solve record without assuming one implementation.
    if is_dataclass(record):  # Prefer exact dataclass serialization.
        payload = asdict(record)  # Convert the immutable record recursively.
    elif hasattr(record, "__dict__"):  # Support conventional record objects.
        payload = dict(vars(record))  # Copy the object's public state.
    else:  # Preserve opaque legacy records.
        payload = {"value": str(record)}  # Store a stable textual representation.
    def normalize(value: Any) -> Any:  # Convert numerical and path objects to JSON-compatible values.
        if isinstance(value, np.ndarray):  # Convert arrays to nested lists.
            return value.tolist()  # Return a JSON-compatible array representation.
        if isinstance(value, (np.integer, np.floating)):  # Convert NumPy scalars to native scalars.
            return value.item()  # Return the native scalar.
        if isinstance(value, Path):  # Convert paths to strings.
            return str(value)  # Return a portable path representation.
        if isinstance(value, dict):  # Normalize mapping values recursively.
            return {str(key): normalize(item) for key, item in value.items()}  # Return a JSON-compatible mapping.
        if isinstance(value, (list, tuple)):  # Normalize sequence values recursively.
            return [normalize(item) for item in value]  # Return a JSON-compatible sequence.
        return value  # Preserve already compatible values.
    return {str(key): normalize(value) for key, value in payload.items()}  # Return the normalized record payload.

def run_exact_dorfler(runner: Any, partition: CachedVisionPartition, *, max_solves: int, n_equation_cap: int, theta: float, refine_factor: float, require_reference: bool = True) -> DorflerResult:  # Execute the paired exact-Dörfler trajectory with optional reference metrics.
    adapter = _RuntimeAdapter(runner)  # Reuse the common solver and estimator adapter.
    adapter.ensure_reference(require_reference)  # Build trusted energy-error metrics only when requested.
    gateway = MCPToolGateway(ToolConfig(theta=theta, refine_factor=refine_factor, core_theta=0.72, max_extra_depth=2))  # Configure exact Dörfler materialization only.
    mesh = adapter.initial_mesh()  # Generate the same uniform initial mesh used by world VLA.
    records: list[Any] = []  # Collect real baseline solve records.
    indicator_sums: list[float] = []  # Collect real baseline estimator values.
    certificates: list[MeshCertificate] = []  # Collect exact target-field certificates.
    hit_count: np.ndarray | None = None  # Initialize consistently measured semantic recurrence without using it for actions.
    stop_reason = "max_solves"  # Default to the configured real-solve horizon.
    for step in range(max_solves):  # Execute the same maximum number of real solves as world VLA.
        post, record = adapter.solve(mesh, "dorfler", step)  # Execute one real baseline CalculiX solve.
        eta2 = adapter.indicator(post)  # Evaluate the same exact repository ZZ indicator.
        observation = gateway.observe_solve(adapter.problem, partition, post, record, eta2, hit_count, step)  # Measure the same state without granting semantic control.
        hit_count = observation.state.hit_count.copy()  # Preserve consistent measurement history.
        records.append(record)  # Store the real baseline solve record.
        indicator_sums.append(float(np.sum(eta2)))  # Store the real global squared-indicator sum.
        adapter.add_audit(record, {"wmvla_schema": gateway.schema_version, "baseline": "exact_dorfler", "theta": theta, "refine_factor": refine_factor, "indicator_sum": indicator_sums[-1]})  # Attach baseline provenance.
        if step + 1 >= max_solves:  # Stop after the common real-solve horizon.
            stop_reason = "max_solves"  # Record the horizon stop.
            break  # Return the completed baseline.
        if observation.state.n_equations >= n_equation_cap:  # Stop at the measured active-equation cap.
            stop_reason = "equation_cap_reached"  # Record the resource stop.
            break  # Preserve the last feasible baseline solve.
        action = RegionAction.dorfler(observation.state)  # Construct the exact-Dörfler action with no semantic additions.
        materialized = gateway.materialize_action(observation, action, n_equation_cap)  # Generate and exactly preflight the next Dörfler mesh.
        certificates.append(materialized.certificate)  # Store deterministic baseline evidence.
        if materialized.mesh is None:  # Stop when the next exact-Dörfler mesh exceeds the cap.
            stop_reason = materialized.certificate.reason  # Preserve the deterministic preflight reason.
            break  # End without an over-budget real solve.
        mesh = materialized.mesh  # Advance to the exact-Dörfler candidate mesh.
    return DorflerResult(records=tuple(records), indicator_sums=tuple(indicator_sums), certificates=tuple(certificates), stop_reason=stop_reason)  # Return the complete paired baseline trajectory.

def _budget_ratios(d_metrics: list[float], d_equations: list[int], w_metrics: list[float], w_equations: list[int]) -> list[float]:  # Compare Pareto envelopes on common measured resource caps.
    caps = sorted(set(d_equations + w_equations))  # Construct all observed common resource thresholds.
    ratios: list[float] = []  # Collect world-to-Dörfler best-error ratios.
    for cap in caps:  # Evaluate both trajectories at each observed active-equation cap.
        d_candidates = [metric for metric, equations in zip(d_metrics, d_equations, strict=True) if equations <= cap]  # Select feasible Dörfler solves.
        w_candidates = [metric for metric, equations in zip(w_metrics, w_equations, strict=True) if equations <= cap]  # Select feasible world-VLA solves.
        if not d_candidates or not w_candidates:  # Skip caps without both methods represented.
            continue  # Preserve a fair common-budget comparison.
        ratios.append(float(min(w_candidates) / max(min(d_candidates), 1.0e-30)))  # Compare Pareto-envelope errors at the same cap.
    return ratios  # Return common-budget error ratios.

def run_bridge_benchmark(output_dir: str | Path, *, smoke: bool = False, max_solves: int = 7, n_equation_cap: int = 120000, theta: float = 0.5, refine_factor: float = 0.5, noninferiority_tolerance: float = 0.03, require_reference: bool = True) -> BenchmarkResult:  # Run the clean paired canonical bridge-component benchmark.
    root = Path(output_dir)  # Normalize the campaign output directory.
    root.mkdir(parents=True, exist_ok=True)  # Create the campaign directory.
    problem = make_box_girder_diaphragm(length=420.0, width=300.0, height=220.0, top_thickness=22.0, bottom_thickness=18.0, web_thickness=16.0, diaphragm_thickness=26.0, opening_radius=48.0, frame_width=16.0, wheel_size=(110.0, 80.0), wheel_offset=(25.0, 18.0), pressure=3.0, support_width=55.0) if smoke else make_box_girder_diaphragm()  # Select a reduced or default instance from the canonical root factory.
    partition = CachedVisionPartition.from_problem(problem)  # Create one shared cached semantic vision output.
    partition.save(root / "shared_vision_partition.json")  # Persist identical perception for audit.
    dorfler_runner = _make_runner(problem, root / "dorfler", n_equation_cap)  # Construct the isolated exact-Dörfler runner.
    world_runner = _make_runner(problem, root / "world_vla", n_equation_cap)  # Construct the isolated world-VLA runner.
    dorfler = run_exact_dorfler(dorfler_runner, partition, max_solves=max_solves, n_equation_cap=n_equation_cap, theta=theta, refine_factor=refine_factor, require_reference=require_reference)  # Execute the exact baseline first under the selected reference contract.
    if require_reference:  # Share reference evidence only when a reference was requested.
        world_runner.reference = dorfler_runner.reference  # Reuse the exact repository Reference object without another solve.
    planner = MultiStepPlanner(PlannerConfig(horizon=4 if smoke else 5, beam_width=18 if smoke else 28, warmup_transitions=1, min_robust_gain=0.010 if smoke else 0.018))  # Configure genuine multi-step internal planning.
    world_model = ResidualWorldModel(WorldModelConfig(refine_factor=refine_factor))  # Match the world prior to the common refinement factor.
    world = run_world_model_vla(world_runner, partition=partition, config=WorldVLAConfig(max_solves=max_solves, n_equation_cap=n_equation_cap, theta=theta, refine_factor=refine_factor, artifact_dir=str(root / "world_vla"), require_reference=require_reference), model=world_model, planner=planner)  # Execute the clean world-model VLA route under the selected reference contract.
    d_metrics = [_record_metric(record, dorfler.indicator_sums[index]) for index, record in enumerate(dorfler.records)]  # Compute real baseline quality metrics.
    w_metrics = [_record_metric(record, world.indicator_sums[index]) for index, record in enumerate(world.records)]  # Compute real world-VLA quality metrics.
    d_equations = [_record_equations(record) for record in dorfler.records]  # Read measured baseline active equations.
    w_equations = [_record_equations(record) for record in world.records]  # Read measured world-VLA active equations.
    common = min(len(d_metrics), len(w_metrics))  # Determine the equal-real-solve comparison length.
    solvewise = [float(w_metrics[index] / max(d_metrics[index], 1.0e-30)) for index in range(common)]  # Compare errors at equal real-solve count.
    budgetwise = _budget_ratios(d_metrics, d_equations, w_metrics, w_equations)  # Compare best errors on common resource caps.
    action_count = sum(any(depth > 0 for depth in action) for action in world.actions)  # Count actually certified non-zero world actions.
    inclusion = all(certificate.base_target_included and certificate.no_coarsening for certificate in world.certificates)  # Verify the exact-Dörfler target floor on every action.
    solvewise_noninferior = bool(all(ratio <= 1.0 + noninferiority_tolerance for ratio in solvewise)) if solvewise else False  # Apply the equal-solve non-inferiority gate.
    budgetwise_noninferior = bool(all(ratio <= 1.0 + noninferiority_tolerance for ratio in budgetwise)) if budgetwise else False  # Apply the common-budget non-inferiority gate.
    terminal_advantage = bool(min(w_metrics, default=np.inf) < (1.0 - 0.005) * min(d_metrics, default=np.inf))  # Require a measurable best-error improvement for terminal advantage.
    result = BenchmarkResult(dorfler=dorfler, world_vla=world, dorfler_metrics=tuple(d_metrics), world_metrics=tuple(w_metrics), dorfler_equations=tuple(d_equations), world_equations=tuple(w_equations), solvewise_ratios=tuple(solvewise), budgetwise_ratios=tuple(budgetwise), world_action_count=int(action_count), dorfler_inclusion_all=bool(inclusion), solvewise_noninferior=solvewise_noninferior, budgetwise_noninferior=budgetwise_noninferior, terminal_advantage=terminal_advantage)  # Build the complete paired benchmark result.
    payload = {"configuration": {"smoke": smoke, "max_solves": max_solves, "n_equation_cap": n_equation_cap, "theta": theta, "refine_factor": refine_factor, "noninferiority_tolerance": noninferiority_tolerance, "require_reference": require_reference}, "dorfler": {"records": [_record_payload(record) for record in dorfler.records], "indicator_sums": list(dorfler.indicator_sums), "metrics": d_metrics, "equations": d_equations, "stop_reason": dorfler.stop_reason, "certificates": [asdict(certificate) for certificate in dorfler.certificates]}, "world_vla": {"records": [_record_payload(record) for record in world.records], "indicator_sums": list(world.indicator_sums), "metrics": w_metrics, "equations": w_equations, "stop_reason": world.stop_reason, "actions": [list(action) for action in world.actions], "decisions": [asdict(decision) for decision in world.decisions], "certificates": [asdict(certificate) for certificate in world.certificates], "timing_s": world.timing_s}, "gates": {"world_action_count": action_count, "dorfler_inclusion_all": inclusion, "solvewise_noninferior": solvewise_noninferior, "budgetwise_noninferior": budgetwise_noninferior, "terminal_advantage": terminal_advantage, "solvewise_ratios": solvewise, "budgetwise_ratios": budgetwise}}  # Build the complete human-auditable benchmark payload including reference and separated timing provenance.
    (root / "benchmark_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # Persist all real-solve curves and scientific gates.
    return result  # Return the paired benchmark result.
