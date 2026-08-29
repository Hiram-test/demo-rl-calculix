"""Unit tests for the Dörfler-anchored world-model VLA contracts."""  # Describe the test module purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
import inspect  # Import source inspection for the method-independence gate.
from types import SimpleNamespace  # Import a minimal problem stand-in for mesh-target tests.
import numpy as np  # Import numerical arrays for synthetic states and meshes.
from visionamr.bridge_cases import make_box_girder_diaphragm  # Import the bridge-component definition.
from visionamr.mesher import Mesh  # Import the repository simplex mesh contract.
from visionamr.vla.mcp_tools import MCPMeshGateway, regional_level_targets  # Import deterministic parameter tools.
from visionamr.vla.planner import PlannerConfig, WorldModelPlanner, enumerate_actions, exact_region_exposure  # Import planning and exposure functions.
from visionamr.vla.world_model import RegionAction, ResidualWorldModel, WorldModelConfig, WorldState  # Import world-model contracts.
from visionamr.vla import world_pipeline  # Import the real pipeline for the purity gate.

def _state(step: int = 0) -> WorldState:  # Build a compact synthetic three-region world state.
    return WorldState(names=("wheel_patch", "opening_rim", "field"), err_sum=np.array([6.0, 3.0, 1.0]), elems=np.array([40.0, 35.0, 25.0]), sizes=np.array([1.0, 1.2, 1.8]), vm_max=np.array([12.0, 9.0, 2.0]), volume=np.array([1.0, 1.2, 4.0]), adjacency=np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]), dorfler_error_fraction=np.array([0.75, 0.20, 0.0]), dorfler_element_fraction=np.array([0.30, 0.10, 0.0]), hit_count=np.array([2.0, 1.0, 0.0]), n_equations=300.0, eq_per_elem=3.0, h_min=0.1, h0=2.0, dim=3, step=step)  # Return the validated synthetic state.

def test_exact_region_exposure_preserves_elementwise_dorfler() -> None:  # Verify exact aggregation of element-wise marking.
    eta2 = np.array([5.0, 1.0, 3.0, 1.0])  # Define four element indicators.
    labels = np.array([0, 0, 1, 1])  # Assign two elements to each region.
    marked = np.array([0, 2])  # Mark one element in each region.
    error_fraction, element_fraction = exact_region_exposure(eta2, labels, marked, 2)  # Aggregate the exact marking.
    np.testing.assert_allclose(error_fraction, np.array([5.0 / 6.0, 3.0 / 4.0]))  # Check marked estimator fractions.
    np.testing.assert_allclose(element_fraction, np.array([0.5, 0.5]))  # Check marked element fractions.

def test_action_enumeration_always_contains_pure_dorfler() -> None:  # Verify the permanent safety candidate.
    actions = enumerate_actions(_state(), PlannerConfig())  # Enumerate the bounded discrete action set.
    assert actions[0].extra_depth == (0, 0, 0)  # Require pure Dörfler to be first.
    assert any(action.extra_depth == (0, 0, 0) for action in actions)  # Require pure Dörfler to remain present.

def test_mcp_gateway_materialization_is_nodewise_dorfler_dominant() -> None:  # Verify exact nodal non-coarsening.
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])  # Define a square mesh.
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)  # Split the square into two triangles.
    mesh = Mesh(nodes=nodes, cells=cells, dim=2)  # Build the repository mesh object.
    problem = SimpleNamespace(h_min=0.05, h0=2.0)  # Supply only required size limits.
    gateway = MCPMeshGateway(refine_factor=0.5)  # Construct the exact tool gateway.
    result = gateway.materialize(mesh, problem, np.array([0, 1]), np.array([0]), RegionAction((0, 1)), 0.9)  # Extend Dörfler over the second region.
    assert result.dominance_verified  # Require the explicit contract flag.
    assert np.all(result.target_h <= result.dorfler_h + 1.0e-12)  # Require nodewise target dominance.
    assert result.target_h[3] < mesh.node_sizes[3]  # Require proactive refinement outside the marked triangle.

def test_mcp_gateway_rejects_excessive_sparse_action() -> None:  # Verify deterministic parameter validation.
    gateway = MCPMeshGateway(max_extra_regions=1, max_extra_depth=1)  # Tighten the action contract.
    certificate = gateway.certify_action(_state(), RegionAction((1, 1, 0)), 500.0, 1000)  # Propose two proactive regions.
    assert not certificate.accepted  # Require rejection.
    assert "too_many_proactive_regions" in certificate.reasons  # Require a precise failure reason.

def test_regional_level_targets_are_exact_and_bounded() -> None:  # Verify the MCP-exposed discrete conversion.
    targets = regional_level_targets([2.0, 1.0, 0.2], [0, 1, 3], 0.5, 0.1, 2.0)  # Convert three regional levels.
    np.testing.assert_allclose(targets, [2.0, 0.5, 0.1])  # Check exact powers and lower clipping.

def test_world_model_learns_real_transition_rows_and_roundtrips(tmp_path) -> None:  # Verify online residual learning and transparent persistence.
    model = ResidualWorldModel(WorldModelConfig(min_rows_for_learning=1, ensemble_size=3))  # Permit fitting after one synthetic transition.
    previous = _state()  # Build the pre-action state.
    action = RegionAction((1, 0, 0))  # Extend refinement over the persistent wheel region.
    observed = WorldState(names=previous.names, err_sum=np.array([1.8, 2.6, 0.9]), elems=np.array([95.0, 40.0, 28.0]), sizes=np.array([0.55, 1.0, 1.7]), vm_max=np.array([13.0, 9.2, 2.1]), volume=previous.volume, adjacency=previous.adjacency, dorfler_error_fraction=np.array([0.4, 0.5, 0.0]), dorfler_element_fraction=np.array([0.2, 0.25, 0.0]), hit_count=np.array([3.0, 2.0, 0.0]), n_equations=489.0, eq_per_elem=3.0, h_min=previous.h_min, h0=previous.h0, dim=previous.dim, step=1)  # Define the realized next solve.
    model.observe(previous, action, observed)  # Add only the real action-conditioned transition.
    assert model.sample_count == previous.n_regions  # Require one residual row per region.
    prediction = model.predict(previous, action)  # Exercise the learned ensemble.
    assert prediction.state.total_error > 0.0  # Require a physically positive prediction.
    path = tmp_path / "world_model.json"  # Select a transparent snapshot path.
    model.save(path)  # Persist the transition library.
    restored = ResidualWorldModel.load(path)  # Restore the transition library.
    assert restored.sample_count == model.sample_count  # Require lossless transition accounting.

def test_planner_falls_back_when_uncertainty_gate_is_zero() -> None:  # Verify a strict safety-gate fallback.
    model = ResidualWorldModel()  # Start from prior-only uncertainty.
    config = PlannerConfig(horizon=3, uncertainty_limit=0.0, failure_limit=1.0, min_relative_gain=-1.0, resource_weight=0.0)  # Make only the uncertainty gate binding.
    decision = WorldModelPlanner(model, config).plan(_state(), 1000000)  # Plan under an ample equation budget.
    assert decision.source == "dorfler_fallback"  # Require exact Dörfler fallback.
    assert decision.action.extra_depth == (0, 0, 0)  # Require no proactive depth.

def test_planner_can_select_persistent_region_under_relaxed_safe_budget() -> None:  # Verify that multi-step planning can act rather than always fall back.
    model = ResidualWorldModel(WorldModelConfig(prior_spread=0.01))  # Use a narrow physics-prior ensemble for the synthetic test.
    config = PlannerConfig(horizon=4, beam_width=12, uncertainty_limit=1.0, failure_limit=1.0, min_relative_gain=-1.0, resource_weight=0.0, budget_safety=1.0)  # Relax gates while preserving discrete search.
    decision = WorldModelPlanner(model, config).plan(_state(), 100000000)  # Plan under a non-binding synthetic budget.
    assert decision.source == "world_model"  # Require an accepted proactive trajectory.
    assert any(value > 0 for value in decision.action.extra_depth)  # Require actual regional advance investment.
    assert len(decision.sequence) == config.horizon  # Require a true multi-step rollout.

def test_bridge_case_contains_competing_three_dimensional_mechanisms() -> None:  # Verify the selected benchmark is not a trivial block.
    problem = make_box_girder_diaphragm()  # Build only the problem metadata and closures.
    names = {feature.name for feature in problem.features}  # Collect semantic mechanism names.
    assert problem.dim == 3  # Require a three-dimensional component.
    assert problem.name == "box_girder_diaphragm"  # Require the intended bridge family.
    assert any("wheel" in name for name in names)  # Require a local deck-loading mechanism.
    assert any("opening" in name for name in names)  # Require an access-opening mechanism.
    assert any("web_diaphragm" in name for name in names)  # Require plate-intersection mechanisms.
    assert any("bearing" in name for name in names)  # Require support-reaction mechanisms.

def test_world_pipeline_has_no_local_prediction_dependency() -> None:  # Guard the clean scientific separation requested for the comparison.
    source = inspect.getsource(world_pipeline)  # Read the implemented world-model pipeline source.
    assert "local_prediction" not in source  # Forbid importing or consuming the local-prediction method.
