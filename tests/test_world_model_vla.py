from __future__ import annotations  # Enable compact annotations in the focused unit tests.
from pathlib import Path  # Verify that the new method remains independent of local prediction.
from types import SimpleNamespace  # Build small deterministic problem and constraint fixtures.
import numpy as np  # Construct numerical world states and actions.
from visionamr.mesher import Mesh  # Test exact free-equation certification on a real mesh record.
from visionamr.vla.planner import ModelPredictivePlanner, PlannerConfig  # Test Dörfler-guarded model-predictive control.
from visionamr.vla.tool_gateway import DeterministicToolGateway, ToolGatewayConfig, estimate_free_equations  # Test deterministic parameter ownership.
from visionamr.vla.world_model import OnlineRegionWorldModel, WorldModelConfig  # Test semantic prediction and online correction.
from visionamr.vla.world_state import Transition, WorldAction, WorldState  # Build explicit action-conditioned state transitions.
def _problem():  # Build the minimum immutable geometry contract used by preview tests.
    return SimpleNamespace(dim=3, h0=50.0, h_min=2.0, instance_id="bridge-test")  # Match a three-dimensional bridge solid.
def _state(step: int = 0, err: np.ndarray | None = None, n_equations: int = 1000) -> WorldState:  # Build a four-region topology-sensitive state.
    values = np.asarray([45.0, 25.0, 5.0, 25.0] if err is None else err, dtype=float)  # Put little observed error on the probe-blind hole.
    adjacency = np.asarray([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]], dtype=float)  # Create a connected region graph.
    roles = np.asarray([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=float)  # Encode load, support, hole, and field semantics.
    return WorldState(step=step, names=("wheel_edge", "girder_support", "service_opening", "field"), origins=("vision", "vision", "vision", "coarse"), grades=np.asarray([2, 3, 2, 5], dtype=int), sizes=np.asarray([12.0, 20.0, 16.0, 36.0], dtype=float), err_sum=values, elems=np.asarray([180.0, 220.0, 80.0, 520.0], dtype=float), vm_max=np.asarray([100.0, 60.0, 35.0, 15.0], dtype=float), vm_mean=np.asarray([50.0, 30.0, 20.0, 8.0], dtype=float), volume=np.asarray([1.0, 2.0, 0.5, 8.0], dtype=float), adjacency=adjacency, roles=roles, n_equations=int(n_equations), budget=1400, e_energy=0.30, e_qoi=0.20, total_eta2=float(values.sum()), qoi=1.0, U_total=1.0, state_id=f"state-{step}-{n_equations}")  # Return the complete audited state.
def test_semantic_world_model_values_probe_blind_opening():  # Verify the key advantage mechanism over residual-only marking.
    problem = _problem()  # Build the bridge-solid contract.
    state = _state()  # Build a state where the opening has low observed residual.
    gateway = DeterministicToolGateway(problem)  # Resolve all sizes outside the model.
    model = OnlineRegionWorldModel(problem)  # Use the structured prior before online fitting.
    refine_hole = WorldAction("refine-hole", (0, 0, -1, 0), kind="region", source="semantic_lookahead")  # Refine only the visible opening.
    hold = WorldAction("hold", (0, 0, 0, 0), kind="region", source="hold")  # Preserve the current allocation.
    hole_prediction = model.predict(state, gateway.preview(state, refine_hole))  # Predict the topology-aware action.
    hold_prediction = model.predict(state, gateway.preview(state, hold))  # Predict no redistribution.
    assert hole_prediction.e_energy < hold_prediction.e_energy  # Require the world model to value visible unresolved topology.
    assert hole_prediction.details["latent_eta2"] > 0.0  # Confirm that the gain is tied to explicit latent structural risk.
def test_gateway_maps_only_discrete_actions_and_projects_budget():  # Verify that no LLM-supplied continuous size can enter execution.
    problem = _problem()  # Build the bridge-solid contract.
    state = _state()  # Build a measured resource anchor.
    gateway = DeterministicToolGateway(problem, ToolGatewayConfig(budget_safety=0.94, max_changed_regions=3))  # Configure strict mapping.
    action = WorldAction("transfer", (-1, 0, -1, 1), kind="region", source="world_model")  # Supply only adjacent grade changes.
    preview = gateway.preview(state, action)  # Resolve exact physical sizes and resource use.
    assert preview.audit["continuous_input_from_model"] is False  # Confirm parameter ownership.
    assert preview.audit["mapping_version"] == gateway.mapping_version  # Confirm versioned reproducibility.
    assert np.all((preview.grades >= 1) & (preview.grades <= 5))  # Confirm grade validity.
    assert preview.n_equations <= state.budget * 1.05  # Confirm closed-form budget control within clipping tolerance.
def test_gateway_rejects_dense_or_nonlocal_grade_action():  # Verify the high-level action boundary.
    problem = _problem()  # Build the bridge-solid contract.
    state = _state()  # Build a four-region state.
    gateway = DeterministicToolGateway(problem, ToolGatewayConfig(max_changed_regions=2))  # Permit only sparse learned moves.
    action = WorldAction("too-many", (-1, -1, -1, 0), kind="region", source="world_model")  # Change three regions.
    try:  # Exercise the rejecting tool boundary.
        gateway.preview(state, action)  # Attempt to materialize an invalid request.
    except ValueError as exc:  # Capture the expected contract failure.
        assert "too many regions" in str(exc)  # Confirm the precise guard.
    else:  # Detect an unsafe acceptance.
        raise AssertionError("dense world action was accepted")  # Fail the test.
def test_exact_equation_counter_matches_constrained_translations():  # Verify pre-solve resource certification.
    mesh = Mesh(nodes=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float), cells=np.asarray([[0, 1, 2, 3]], dtype=int), dim=3)  # Build one tetrahedron.
    constraint = SimpleNamespace(node_predicate=lambda nodes: nodes[:, 0] == 0.0, dofs=(1, 2, 3), name="x0_fix")  # Fix three nodes on x equals zero.
    problem = SimpleNamespace(dim=3, constraints=[constraint])  # Build the required constraint contract.
    assert estimate_free_equations(mesh, problem) == 3  # Twelve translations minus nine constrained equations leaves three.
def test_online_model_learns_measured_transition_residual():  # Verify that repeated VLA steps update the action-conditioned model.
    problem = _problem()  # Build the bridge-solid contract.
    state = _state()  # Build the parent state.
    gateway = DeterministicToolGateway(problem)  # Resolve tool-owned sizes.
    model = OnlineRegionWorldModel(problem, WorldModelConfig(min_rows_for_fit=2, ensemble_size=5, ridge=0.05))  # Allow one four-region transition to fit.
    action = WorldAction("refine-hole", (0, 0, -1, 0), kind="region", source="semantic_lookahead")  # Define the executed action.
    preview = gateway.preview(state, action)  # Resolve deterministic parameters.
    prior = model.predict(state, preview)  # Record the uncalibrated prediction.
    measured_err = prior.err_sum * np.exp(0.20)  # Simulate a systematic optimistic prior.
    next_state = _state(step=1, err=measured_err, n_equations=int(round(preview.n_equations * 1.08)))  # Build the measured successor.
    next_state = WorldState(step=next_state.step, names=next_state.names, origins=next_state.origins, grades=preview.grades, sizes=preview.sizes, err_sum=next_state.err_sum, elems=next_state.elems, vm_max=next_state.vm_max, vm_mean=next_state.vm_mean, volume=next_state.volume, adjacency=next_state.adjacency, roles=next_state.roles, n_equations=next_state.n_equations, budget=next_state.budget, e_energy=next_state.e_energy, e_qoi=next_state.e_qoi, total_eta2=next_state.total_eta2, qoi=next_state.qoi, U_total=next_state.U_total, state_id=next_state.state_id)  # Align the successor with the executed grades and sizes.
    update = model.observe(Transition(state=state, action=action, preview_sizes=preview.sizes, preview_n_equations=preview.n_equations, next_state=next_state))  # Fit the real residual.
    calibrated = model.predict(state, preview)  # Re-evaluate the same state-action pair.
    assert update["regional_rows_added"] == state.n_regions  # Confirm region-level supervision.
    assert calibrated.residual_total_eta2 > prior.residual_total_eta2  # Confirm correction of optimistic error prediction.
    assert calibrated.n_equations > prior.n_equations  # Confirm correction of optimistic resource prediction.
def test_planner_always_carries_exact_dorfler_guard():  # Verify that world-model control cannot delete the AFEM safety action.
    problem = _problem()  # Build the bridge-solid contract.
    state = _state()  # Build the current world state.
    gateway = DeterministicToolGateway(problem)  # Build the numerical action gateway.
    model = OnlineRegionWorldModel(problem)  # Build the structured transition model.
    planner = ModelPredictivePlanner(PlannerConfig(horizon=2, beam_width=4, candidate_limit=10))  # Build a compact multi-step planner.
    plan = planner.plan(state, gateway, model)  # Compare imagined actions.
    assert plan.guard_action.kind == "dorfler"  # Require the exact safety path.
    assert plan.guard_action.source == "dorfler_guard"  # Require explicit provenance.
    assert plan.action.kind in ("region", "dorfler")  # Permit only certified execution kinds.
    assert plan.candidates_evaluated > 1  # Confirm genuine model-based comparison rather than a fixed rule.
def test_world_model_modules_do_not_import_local_prediction():  # Preserve the clean scientific separation requested for comparison.
    repo = Path(__file__).resolve().parents[1]  # Resolve the repository root.
    paths = [repo / "visionamr" / "vla" / name for name in ("world_state.py", "world_model.py", "tool_gateway.py", "planner.py", "world_pipeline.py")]  # Enumerate the new method implementation.
    text = "\n".join(path.read_text() for path in paths)  # Read source without executing it.
    assert "baselines.local_prediction" not in text  # Forbid implementation reuse.
    assert "predicted_sizes" not in text  # Forbid the local-prediction size operator.
