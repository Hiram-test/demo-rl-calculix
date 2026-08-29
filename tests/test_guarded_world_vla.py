from __future__ import annotations  # Enable compact annotations in the guarded-controller tests.
from pathlib import Path  # Inspect source boundaries for scientific isolation.
from types import SimpleNamespace  # Build minimal deterministic problem fixtures.
import numpy as np  # Construct audited states and compare numerical previews.
from visionamr.vla.guarded_gateway import GuardedToolGateway  # Test Dörfler-backbone tool ownership.
from visionamr.vla.guarded_planner import GuardedModelPredictivePlanner, GuardedPlannerConfig, _guarded_candidates  # Test the protected action space.
from visionamr.vla.tool_gateway import ToolGatewayConfig  # Configure deterministic resource limits.
from visionamr.vla.world_model import OnlineRegionWorldModel  # Provide action-conditioned consequence predictions.
from visionamr.vla.world_state import Transition, WorldAction, WorldState  # Build exact state-action evidence.
def _problem():  # Build the minimum three-dimensional bridge-solid contract.
    return SimpleNamespace(dim=3, h0=100.0, h_min=4.0, instance_id="guarded-bridge-test")  # Provide only fields needed by previews and the world model.
def _state(step: int = 0) -> WorldState:  # Build a topology-sensitive four-region world state.
    adjacency = np.asarray([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]], dtype=float)  # Create a connected region graph.
    roles = np.asarray([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=float)  # Encode load edge, support, opening, and field semantics.
    return WorldState(step=int(step), names=("wheel_edge", "support_strip", "service_opening", "field"), origins=("vision", "vision", "vision", "coarse"), grades=np.asarray([1, 3, 2, 5], dtype=int), sizes=np.asarray([24.0, 45.0, 32.0, 85.0], dtype=float), err_sum=np.asarray([35.0, 25.0, 8.0, 32.0], dtype=float), elems=np.asarray([180.0, 230.0, 90.0, 500.0], dtype=float), vm_max=np.asarray([100.0, 70.0, 45.0, 15.0], dtype=float), vm_mean=np.asarray([55.0, 38.0, 25.0, 8.0], dtype=float), volume=np.asarray([1.0, 2.0, 0.5, 9.0], dtype=float), adjacency=adjacency, roles=roles, n_equations=6000, budget=8000, e_energy=0.4, e_qoi=0.2, total_eta2=100.0, qoi=1.0, U_total=1.0, state_id=f"guarded-state-{step}")  # Return the complete immutable state.
def test_guarded_preview_starts_from_realized_sizes_and_adds_no_coarsening():  # Verify that v2 no longer resets a Dörfler mesh to absolute grade priors.
    state = _state()  # Build a measured current mesh state.
    gateway = GuardedToolGateway(_problem(), ToolGatewayConfig(max_changed_regions=3))  # Build the deterministic numerical boundary.
    action = WorldAction("guarded-opening", (0, 0, -1, 0), kind="guarded", source="world_model_guarded")  # Select only the visible opening as a bonus.
    preview = gateway.preview(state, action)  # Resolve model-independent continuous sizes.
    assert preview.audit["continuous_input_from_model"] is False  # Confirm that the model supplied no continuous parameter.
    assert preview.audit["dorfler_backbone"] is True  # Confirm protected execution semantics.
    assert preview.audit["bonus_regions"] == ["service_opening"]  # Confirm exact discrete-to-semantic mapping.
    assert preview.sizes[2] < state.sizes[2]  # Require refinement of the selected opening.
    assert preview.sizes[3] <= state.sizes[3]  # Forbid coarsening of the background inside the protected preview.
def test_guarded_gateway_rejects_any_world_requested_coarsening():  # Verify the pointwise Dörfler floor at the action boundary.
    state = _state()  # Build an audited state.
    gateway = GuardedToolGateway(_problem())  # Build the guarded numerical boundary.
    action = WorldAction("unsafe-transfer", (-1, 0, 0, 1), kind="guarded", source="world_model_guarded")  # Attempt to trade away field resolution.
    try:  # Exercise strict request validation.
        gateway.preview(state, action)  # Attempt to map the unsafe action.
    except ValueError as exc:  # Capture the expected rejection.
        assert "only add refinement" in str(exc)  # Confirm the protected reason.
    else:  # Detect unsafe acceptance.
        raise AssertionError("guarded action accepted model-requested coarsening")  # Fail the contract test.
def test_guarded_candidate_space_contains_no_naked_region_tuning():  # Verify that every learned action retains exact Dörfler execution.
    state = _state()  # Build a bridge-component state.
    candidates = _guarded_candidates(state, GuardedPlannerConfig())  # Enumerate the complete bounded action space.
    assert candidates[0].kind == "dorfler"  # Require the faithful safety action first.
    assert all(action.kind in ("dorfler", "guarded") for action in candidates)  # Exclude naked grade remapping and resource transfer.
    assert any(action.kind == "guarded" for action in candidates)  # Preserve a real world-model advantage path.
    assert all(all(value <= 0 for value in action.deltas) for action in candidates if action.kind == "guarded")  # Forbid model-requested coarsening.
def test_guarded_planner_uses_exact_dorfler_for_world_model_warmup():  # Verify conservative collection of the first real transition.
    state = _state()  # Build the first solved state.
    gateway = GuardedToolGateway(_problem())  # Build deterministic action tools.
    model = OnlineRegionWorldModel(_problem())  # Build an uncalibrated transition model.
    planner = GuardedModelPredictivePlanner(GuardedPlannerConfig(warmup_transitions=1, horizon=2, beam_width=3))  # Require one real warmup transition.
    plan = planner.plan(state, gateway, model)  # Request the first controlled action.
    assert plan.action.kind == "dorfler"  # Require exact AFEM before a semantic bonus.
    assert plan.selected_by == "dorfler_world_model_warmup"  # Preserve explicit safety provenance.
def test_guarded_planner_never_returns_an_unprotected_action_after_warmup():  # Verify the deployed action space after online evidence exists.
    state = _state()  # Build a solved parent state.
    next_state = _state(step=1)  # Build a compatible measured successor.
    gateway = GuardedToolGateway(_problem())  # Build deterministic numerical tools.
    model = OnlineRegionWorldModel(_problem())  # Build the online transition model.
    warmup_action = WorldAction("warmup", (-1, 0, 0, 0), kind="dorfler", source="dorfler_guard")  # Represent the exact warmup action.
    model.transitions.append(Transition(state=state, action=warmup_action, preview_sizes=state.sizes.copy(), preview_n_equations=float(state.n_equations), next_state=next_state))  # Mark one real transition as available without fitting synthetic residuals.
    planner = GuardedModelPredictivePlanner(GuardedPlannerConfig(warmup_transitions=1, horizon=2, beam_width=3))  # Enable protected world planning.
    plan = planner.plan(next_state, gateway, model)  # Plan after warmup.
    assert plan.action.kind in ("dorfler", "guarded")  # Require a protected executable action.
    assert plan.guard_action.kind == "dorfler"  # Retain exact baseline provenance.
def test_guarded_control_objective_has_no_local_prediction_or_reference_error_dependency():  # Preserve method independence and deployability.
    repo = Path(__file__).resolve().parents[1]  # Resolve the repository root.
    planner_text = (repo / "visionamr" / "vla" / "guarded_planner.py").read_text()  # Read the deployed decision logic.
    pipeline_text = (repo / "visionamr" / "vla" / "guarded_pipeline.py").read_text()  # Read the deployed real loop.
    gateway_text = (repo / "visionamr" / "vla" / "guarded_gateway.py").read_text()  # Read the numerical tool boundary.
    combined = planner_text + pipeline_text + gateway_text  # Assemble the implementation for isolation checks.
    assert "baselines.local_prediction" not in combined  # Forbid implementation reuse from local prediction.
    assert "predicted_sizes" not in combined  # Forbid the local equidistribution size operator.
    assert "prediction.e_energy" not in planner_text  # Forbid reference energy error in action scoring.
    assert "prediction.e_qoi" not in planner_text  # Forbid reference QoI error in action scoring.
