"""Executable multi-step world-model VLA adaptive finite-element loop."""  # Describe the end-to-end runtime implemented below.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from dataclasses import asdict, dataclass  # Import immutable runtime contracts and serialization support.
import importlib  # Import estimator modules lazily across repository revisions.
import inspect  # Import runtime signature adaptation for solver and estimator utilities.
import json  # Import audit-manifest serialization.
from pathlib import Path  # Import portable artifact paths.
import pkgutil  # Import bounded discovery of repository estimator modules.
from typing import Any, Callable  # Import generic repository object and callable types.
import numpy as np  # Import numerical audit operations.
from ...mesher import generate_mesh  # Reuse exact Gmsh mesh generation.
from .model import RegionAction, ResidualWorldModel, WorldModelConfig, WorldPrediction, WorldState  # Import world-model contracts.
from .planner import MultiStepPlanner, PlanDecision, PlannerConfig  # Import receding-horizon planning.
from .tool_gateway import MCPToolGateway, MeshCertificate, ToolConfig  # Import exact action materialization.
from .vision_partition import CachedVisionPartition  # Import one-shot cached semantic perception.

@dataclass(frozen=True)  # Make runtime settings immutable.
class WorldVLAConfig:  # Configure the real-solve adaptive loop.
    max_solves: int = 7  # Permit a genuinely multi-step VLA trajectory.
    n_equation_cap: int = 120000  # Bound active displacement equations.
    theta: float = 0.5  # Use the same Dörfler marking parameter as the baseline.
    refine_factor: float = 0.5  # Use the same local refinement factor as the baseline.
    core_theta: float = 0.72  # Restrict world actions to concentrated semantic cores.
    audit_slack: float = 0.08  # Allow modest model-bound error before triggering fallback cooldown.
    fallback_cooldown: int = 1  # Execute exact Dörfler after an underperforming world transition.
    stagnation_tolerance: float = 0.002  # Stop after repeated negligible real indicator improvement.
    stagnation_steps: int = 2  # Require repeated stagnation before stopping.
    method_name: str = "world_model_vla"  # Label real solver records consistently.
    artifact_dir: str | None = None  # Optionally override the audit artifact directory.

@dataclass(frozen=True)  # Make completed trajectory results immutable.
class WorldVLAResult:  # Store the complete executable trajectory and audit evidence.
    records: tuple[Any, ...]  # Store real finite-element solve records only.
    actions: tuple[tuple[int, ...], ...]  # Store actually executed regional extra-depth vectors.
    decisions: tuple[PlanDecision, ...]  # Store planner decisions before exact tool certification.
    certificates: tuple[MeshCertificate, ...]  # Store deterministic materialization evidence.
    indicator_sums: tuple[float, ...]  # Store global squared-indicator sums after every real solve.
    stop_reason: str  # Explain why the adaptive trajectory ended.
    best_index: int  # Identify the best recorded real solve under the configured metric.
    model_snapshot: str  # Store the world-model snapshot path.
    vision_snapshot: str  # Store the cached semantic-vision path.

class _RuntimeAdapter:  # Adapt stable world-VLA logic to minor repository API variations.
    def __init__(self, runner: Any) -> None:  # Initialize around an existing finite-element runner.
        self.runner = runner  # Store the real solver runner.
        self.problem = getattr(runner, "problem", None)  # Recover the finite-element problem.
        if self.problem is None:  # Reject runners without a problem contract.
            raise ValueError("runner must expose its finite-element problem")  # Explain the runtime contract.
    def initial_mesh(self) -> Any:  # Generate the common uniform starting mesh.
        try:  # Prefer the repository no-field interface.
            return generate_mesh(self.problem)  # Generate the common initial mesh.
        except TypeError:  # Support explicit None size-field signatures.
            return generate_mesh(self.problem, size_field=None)  # Generate the same common initial mesh.
    def ensure_reference(self) -> Any:  # Build or reuse the runner's reference solution.
        if hasattr(self.runner, "ensure_reference"):  # Use the repository reference cache when available.
            return self.runner.ensure_reference()  # Return the cached or newly solved reference.
        return None  # Permit estimator-only smoke runs without a reference solution.
    def solve(self, mesh: Any, method: str, step: int) -> tuple[Any, Any]:  # Execute one real finite-element solve with signature adaptation.
        solve = getattr(self.runner, "solve_mesh")  # Recover the repository solve entry point.
        signature = inspect.signature(solve)  # Inspect accepted solve metadata fields.
        kwargs: dict[str, Any] = {}  # Collect supported keyword metadata.
        for name in signature.parameters:  # Map stable metadata concepts to active parameter names.
            if name in ("method", "method_name", "label"):  # Match the method-label parameter.
                kwargs[name] = method  # Supply the configured world-VLA label.
            elif name in ("iteration", "step", "round_index", "round_id"):  # Match the adaptive-step parameter.
                kwargs[name] = int(step)  # Supply the zero-based real-solve index.
        result = solve(mesh, **kwargs)  # Execute the real CalculiX finite-element solve.
        if isinstance(result, tuple) and len(result) == 2:  # Handle the repository post-and-record tuple.
            first, second = result  # Unpack the two returned objects.
            if hasattr(first, "mesh") and not hasattr(second, "mesh"):  # Detect post-processing followed by record.
                return first, second  # Return objects in stable post, record order.
            if hasattr(second, "mesh") and not hasattr(first, "mesh"):  # Detect record followed by post-processing.
                return second, first  # Normalize to stable post, record order.
            return first, second  # Preserve the documented tuple order when both expose mesh metadata.
        post = result  # Treat a single return value as the post-processing result.
        records = getattr(self.runner, "records", [])  # Recover runner-managed solve records.
        if not records:  # Reject a single-result API without record metadata.
            raise RuntimeError("runner.solve_mesh returned no solve record")  # Explain the missing resource evidence.
        return post, records[-1]  # Pair the post result with the newly appended record.
    def _candidate_modules(self) -> list[Any]:  # Discover only plausible repository estimator modules.
        modules: list[Any] = []  # Collect imported estimator modules.
        package = importlib.import_module("visionamr")  # Import the repository package root.
        preferred = ("visionamr.estimator", "visionamr.error_estimator", "visionamr.indicator", "visionamr.post", "visionamr.postprocess")  # List likely stable estimator module paths.
        for name in preferred:  # Try preferred module paths first.
            try:  # Isolate optional module import failures.
                modules.append(importlib.import_module(name))  # Import the candidate estimator module.
            except ImportError:  # Ignore paths absent from the active revision.
                pass  # Continue bounded discovery.
        for info in pkgutil.iter_modules(package.__path__):  # Inspect top-level repository modules only.
            if not any(token in info.name.lower() for token in ("estim", "error", "indicator", "post")):  # Exclude unrelated modules and side effects.
                continue  # Continue bounded estimator discovery.
            full_name = f"visionamr.{info.name}"  # Construct the importable module path.
            if any(getattr(module, "__name__", "") == full_name for module in modules):  # Avoid duplicate imports.
                continue  # Preserve deterministic module order.
            try:  # Isolate optional native-dependency import failures.
                modules.append(importlib.import_module(full_name))  # Import the plausible estimator module.
            except Exception:  # Ignore modules unavailable in the current execution environment.
                pass  # Continue searching for the exact repository estimator.
        return modules  # Return the bounded candidate module list.
    def _zz_function(self) -> Callable[..., Any]:  # Locate the repository exact ZZ indicator implementation.
        for name in ("zz_indicator", "error_indicator", "estimate_error"):  # Inspect supported runner estimator method names.
            function = getattr(self.runner, name, None)  # Read the candidate method.
            if callable(function):  # Accept a callable estimator.
                return function  # Return the exact runner-bound implementation.
        for module in self._candidate_modules():  # Inspect bounded repository estimator modules.
            function = getattr(module, "zz_indicator", None)  # Read the canonical ZZ indicator symbol.
            if callable(function):  # Accept the first canonical implementation.
                return function  # Return the exact repository function.
        raise ImportError("could not locate repository zz_indicator implementation")  # Refuse to substitute an unrelated indicator.
    def _post_value(self, post: Any, names: tuple[str, ...]) -> Any:  # Recover one post-processing field by stable aliases.
        for name in names:  # Inspect candidate field names in order.
            if hasattr(post, name):  # Use the first available post-processing field.
                return getattr(post, name)  # Return the measured field.
        return None  # Report absence to the signature adapter.
    def indicator(self, post: Any) -> np.ndarray:  # Evaluate the repository exact ZZ squared indicator.
        function = self._zz_function()  # Locate the exact repository estimator.
        mesh = getattr(post, "mesh", None)  # Recover the solved mesh.
        displacement = self._post_value(post, ("displacement", "u", "nodal_displacement"))  # Recover the solved displacement field.
        stress = self._post_value(post, ("stress", "element_stress", "sigma"))  # Recover the solved stress field.
        signature = inspect.signature(function)  # Inspect the exact estimator signature.
        kwargs: dict[str, Any] = {}  # Collect supported estimator arguments.
        for name in signature.parameters:  # Map physical quantities to exact parameter names.
            lower = name.lower()  # Normalize the parameter name.
            if lower in ("post", "result", "solution"):  # Match the complete post-processing result.
                kwargs[name] = post  # Supply the real result object.
            elif lower in ("mesh", "grid"):  # Match the solved mesh.
                kwargs[name] = mesh  # Supply the exact solved mesh.
            elif lower in ("u", "displacement", "disp"):  # Match the displacement field.
                kwargs[name] = displacement  # Supply the real solved displacement.
            elif lower in ("stress", "sigma", "element_stress"):  # Match the stress field.
                kwargs[name] = stress  # Supply the real solved stress.
            elif lower in ("problem", "case"):  # Match the finite-element problem.
                kwargs[name] = self.problem  # Supply the exact problem contract.
            elif lower in ("material", "mat"):  # Match the material contract.
                kwargs[name] = self.problem.material  # Supply the exact material.
            elif lower in ("e", "young", "youngs_modulus"):  # Match Young's modulus.
                kwargs[name] = float(self.problem.material.E)  # Supply the exact material modulus.
            elif lower in ("nu", "poisson", "poisson_ratio"):  # Match Poisson's ratio.
                kwargs[name] = float(self.problem.material.nu)  # Supply the exact material ratio.
        try:  # Prefer auditable keyword binding.
            values = function(**kwargs)  # Evaluate the exact ZZ estimator.
        except TypeError:  # Support common positional repository signatures.
            attempts = ((post,), (mesh, displacement, self.problem.material), (mesh, displacement, float(self.problem.material.E), float(self.problem.material.nu)), (mesh, stress, self.problem.material))  # Define bounded physical argument patterns.
            last_error: Exception | None = None  # Preserve the final signature error for diagnosis.
            for arguments in attempts:  # Try only physically meaningful exact-estimator calls.
                try:  # Isolate each signature pattern.
                    values = function(*arguments)  # Evaluate the exact ZZ estimator.
                    break  # Stop after the first successful exact call.
                except TypeError as error:  # Capture signature mismatches only.
                    last_error = error  # Preserve the diagnostic error.
            else:  # Handle exhaustion of exact physical signatures.
                raise TypeError("zz_indicator signature could not be adapted") from last_error  # Refuse to substitute a fabricated estimator.
        if isinstance(values, tuple):  # Handle estimators returning indicator plus recovered fields.
            values = values[0]  # Select the squared element indicator vector.
        eta2 = np.asarray(values, dtype=float).reshape(-1)  # Normalize the exact indicator vector.
        if np.any(eta2 < -1.0e-12):  # Reject physically invalid negative squared indicators.
            raise ValueError("zz_indicator returned negative squared contributions")  # Explain the estimator invariant.
        return np.maximum(eta2, 0.0)  # Return a non-negative elementwise squared indicator.
    def add_audit(self, record: Any, payload: dict[str, Any]) -> None:  # Attach world-VLA evidence without assuming a mutable record schema.
        for name in ("extra", "extras", "metadata"):  # Inspect common record metadata containers.
            container = getattr(record, name, None)  # Read the candidate metadata object.
            if isinstance(container, dict):  # Update a mutable audit dictionary in place.
                container.update(payload)  # Attach the complete evidence payload.
                return  # Stop after the first supported metadata container.

def _record_metric(record: Any, indicator_sum: float) -> float:  # Select the strongest available real-solve quality metric.
    for name in ("e_energy", "energy_error", "error_energy", "relative_energy_error"):  # Prefer reference-based energy error.
        if hasattr(record, name):  # Read the metric when present.
            value = float(getattr(record, name))  # Convert the metric to a comparable scalar.
            if np.isfinite(value):  # Accept only a finite quality value.
                return value  # Return the reference-based error.
    return float(np.sqrt(max(indicator_sum, 0.0)))  # Fall back to the global ZZ estimator norm.

def run_world_model_vla(runner: Any, *, partition: CachedVisionPartition | None = None, config: WorldVLAConfig | None = None, model: ResidualWorldModel | None = None, planner: MultiStepPlanner | None = None, gateway: MCPToolGateway | None = None) -> WorldVLAResult:  # Execute the complete multi-step world-model-guided VLA loop.
    settings = config or WorldVLAConfig()  # Store immutable runtime settings.
    if settings.max_solves < 1 or settings.n_equation_cap < 1:  # Reject an empty trajectory or invalid resource cap.
        raise ValueError("max_solves and n_equation_cap must be positive")  # Explain the runtime contract.
    adapter = _RuntimeAdapter(runner)  # Adapt the repository solver API.
    adapter.ensure_reference()  # Ensure energy-error metrics share one trusted reference solution.
    semantic_partition = partition or CachedVisionPartition.from_problem(adapter.problem)  # Create or reuse one geometry-level cached vision output.
    world_model = model or ResidualWorldModel(WorldModelConfig(refine_factor=settings.refine_factor))  # Initialize the online residual world model.
    world_planner = planner or MultiStepPlanner(PlannerConfig())  # Initialize finite-horizon risk-aware planning.
    tool_gateway = gateway or MCPToolGateway(ToolConfig(theta=settings.theta, refine_factor=settings.refine_factor, core_theta=settings.core_theta, max_extra_depth=world_planner.config.max_extra_depth))  # Initialize deterministic parameter materialization.
    root = Path(settings.artifact_dir or getattr(runner, "workdir", getattr(runner, "root", ".")))  # Select the trajectory artifact directory.
    root.mkdir(parents=True, exist_ok=True)  # Create the audit directory.
    vision_path = root / "world_vla_vision_partition.json"  # Define the one-shot semantic-vision cache path.
    model_path = root / "world_vla_model.json"  # Define the online world-model snapshot path.
    manifest_path = root / "world_vla_manifest.json"  # Define the complete trajectory audit path.
    semantic_partition.save(vision_path)  # Persist the fixed semantic perception before solving.
    mesh = adapter.initial_mesh()  # Generate the common uniform initial mesh shared with Dörfler.
    records: list[Any] = []  # Collect real finite-element solve records.
    actions: list[tuple[int, ...]] = []  # Collect actually executed extra-depth vectors.
    decisions: list[PlanDecision] = []  # Collect planner decision evidence.
    certificates: list[MeshCertificate] = []  # Collect exact tool certificates.
    indicator_sums: list[float] = []  # Collect real global squared-indicator sums.
    hit_count: np.ndarray | None = None  # Initialize persistent semantic-hit history.
    pending_state: WorldState | None = None  # Reserve the previous real state for transition learning.
    pending_action: RegionAction | None = None  # Reserve the executed action for transition learning.
    pending_prediction: WorldPrediction | None = None  # Reserve the pre-execution prediction for audit.
    cooldown = 0  # Initialize the exact-Dörfler audit cooldown.
    stagnant = 0  # Initialize repeated-stagnation count.
    stop_reason = "max_solves"  # Default to the configured real-solve horizon.
    for step in range(settings.max_solves):  # Execute at most the configured number of real CalculiX solves.
        post, record = adapter.solve(mesh, settings.method_name, step)  # Execute one real finite-element solve.
        eta2 = adapter.indicator(post)  # Evaluate the exact repository ZZ indicator.
        observation = tool_gateway.observe_solve(adapter.problem, semantic_partition, post, record, eta2, hit_count, step)  # Build the measured action-conditioned world state.
        hit_count = observation.state.hit_count.copy()  # Preserve semantic recurrence evidence for the next remesh.
        indicator_sum = float(np.sum(eta2))  # Compute the global squared-indicator sum.
        indicator_sums.append(indicator_sum)  # Record the real estimator trajectory.
        records.append(record)  # Record the real finite-element solve.
        audit_payload: dict[str, Any] = {"wmvla_schema": tool_gateway.schema_version, "wmvla_step": step, "wmvla_indicator_sum": indicator_sum, "wmvla_regions": list(observation.state.names), "wmvla_region_error": observation.state.err_sum.tolist(), "wmvla_region_elements": observation.state.elems.tolist(), "wmvla_dorfler_error_fraction": observation.state.dorfler_error_fraction.tolist(), "wmvla_dorfler_element_fraction": observation.state.dorfler_element_fraction.tolist(), "wmvla_model_transitions": world_model.transition_count}  # Build solve-level measured evidence.
        if pending_state is not None and pending_action is not None and pending_prediction is not None:  # Learn only after a complete real transition exists.
            actual_ratio = observation.state.total_error / max(pending_state.total_error, 1.0e-30)  # Measure the realized global indicator ratio.
            world_model.observe(pending_state, pending_action, observation.state)  # Learn residuals from the real CalculiX transition.
            bound = pending_prediction.error_ratio_upper * (1.0 + settings.audit_slack)  # Form the predeclared model-audit bound.
            underperformed = bool(actual_ratio > bound and actual_ratio > pending_prediction.error_ratio_mean * (1.0 + settings.audit_slack))  # Detect a materially worse-than-predicted transition.
            if underperformed and not pending_action.is_dorfler_only:  # Penalize only an accepted world-model addition.
                cooldown = max(cooldown, settings.fallback_cooldown)  # Force subsequent exact-Dörfler recovery steps.
            audit_payload.update({"wmvla_previous_action": list(pending_action.extra_depth), "wmvla_actual_error_ratio": actual_ratio, "wmvla_predicted_error_ratio_upper": pending_prediction.error_ratio_upper, "wmvla_underperformed": underperformed})  # Attach transition audit evidence.
        adapter.add_audit(record, audit_payload)  # Attach measured evidence to the repository record when supported.
        if step + 1 >= settings.max_solves:  # Stop after the configured real-solve horizon.
            stop_reason = "max_solves"  # Record the explicit horizon stop.
            break  # Return the completed trajectory.
        if observation.state.n_equations >= settings.n_equation_cap:  # Stop when the real solve reaches the active-equation cap.
            stop_reason = "equation_cap_reached"  # Record the measured resource stop.
            break  # Preserve the last feasible real solve.
        if not np.any(observation.marked):  # Stop when exact Dörfler finds no positive indicator contribution.
            stop_reason = "no_marked_elements"  # Record estimator convergence.
            break  # End the adaptive trajectory.
        if len(indicator_sums) >= 2:  # Evaluate measured adaptive progress after the first transition.
            improvement = (indicator_sums[-2] - indicator_sums[-1]) / max(indicator_sums[-2], 1.0e-30)  # Compute relative real estimator improvement.
            stagnant = stagnant + 1 if improvement < settings.stagnation_tolerance else 0  # Count only consecutive negligible improvements.
            if stagnant >= settings.stagnation_steps:  # Stop after repeated measured stagnation.
                stop_reason = "indicator_stagnation"  # Record the measured convergence stop.
                break  # Avoid wasting additional real solves.
        force_dorfler = cooldown > 0  # Convert the audit cooldown to a planner safety gate.
        decision = world_planner.plan(observation.state, world_model, settings.n_equation_cap, force_dorfler=force_dorfler)  # Select a finite-horizon safe action.
        if cooldown > 0:  # Consume one forced exact-Dörfler recovery step.
            cooldown -= 1  # Decrement the audit cooldown deterministically.
        materialized = tool_gateway.materialize_action(observation, decision.action, settings.n_equation_cap)  # Generate and exactly preflight Dörfler and world candidates without a real solve.
        decisions.append(decision)  # Record the planner decision before tool fallback.
        certificates.append(materialized.certificate)  # Record the exact parameter and resource certificate.
        actions.append(materialized.action.extra_depth)  # Record the actually executed regional depths.
        adapter.add_audit(record, {"wmvla_decision": asdict(decision), "wmvla_certificate": asdict(materialized.certificate)})  # Attach decision and tool evidence to the current real solve.
        if materialized.mesh is None:  # Stop when even the exact-Dörfler candidate exceeds the cap.
            stop_reason = materialized.certificate.reason  # Preserve the deterministic preflight stop reason.
            break  # End without executing an over-budget real solve.
        pending_state = observation.state  # Store the real pre-action state for online learning.
        pending_action = materialized.action  # Store the actually executed action after any tool fallback.
        pending_prediction = world_model.predict(observation.state, materialized.action)  # Store the pre-execution transition prediction for audit.
        mesh = materialized.mesh  # Advance to the exact preflighted candidate mesh.
        world_model.save(model_path)  # Persist online learning after every planned step.
    world_model.save(model_path)  # Persist the final online world model even for one-solve trajectories.
    metrics = [_record_metric(record, indicator_sums[index]) for index, record in enumerate(records)]  # Compute comparable real-solve quality metrics.
    best_index = int(np.argmin(np.asarray(metrics, dtype=float))) if metrics else -1  # Identify the best real solve under the strongest available metric.
    manifest = {"config": asdict(settings), "case": tool_gateway.inspect_case(adapter.problem), "stop_reason": stop_reason, "best_index": best_index, "metrics": metrics, "indicator_sums": indicator_sums, "actions": [list(action) for action in actions], "decisions": [asdict(decision) for decision in decisions], "certificates": [asdict(certificate) for certificate in certificates], "model_snapshot": str(model_path), "vision_snapshot": str(vision_path)}  # Build the complete trajectory audit manifest.
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")  # Persist human-auditable runtime evidence.
    return WorldVLAResult(records=tuple(records), actions=tuple(actions), decisions=tuple(decisions), certificates=tuple(certificates), indicator_sums=tuple(indicator_sums), stop_reason=stop_reason, best_index=best_index, model_snapshot=str(model_path), vision_snapshot=str(vision_path))  # Return the complete real-solve trajectory.
