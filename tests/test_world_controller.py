"""Unit tests for the multi-step world-model VLA controller."""  # Describe the tested controller surface.

from __future__ import annotations  # Enable postponed annotation evaluation.

import inspect  # Import source inspection for the local-prediction independence gate.
from dataclasses import replace  # Import immutable dataclass replacement for transition fixtures.

import numpy as np  # Import vectorized numerical fixtures.

from visionamr.vla.world_controller import RegionAction  # Import the discrete Dörfler-subsuming action.
from visionamr.vla.world_controller import RegionalState  # Import the action-conditioned world state.
from visionamr.vla.world_controller import RegionalWorldModel  # Import the online residual world model.
from visionamr.vla.world_controller import WorldControllerConfig  # Import the immutable controller configuration.
from visionamr.vla.world_controller import compile_dorfler_dominating_target  # Import the deterministic action compiler.
from visionamr.vla.world_controller import enumerate_actions  # Import the bounded action enumerator.
from visionamr.vla.world_controller import plan_action  # Import the finite-horizon planner.


def _state(step: int = 0) -> RegionalState:  # Build a deterministic three-region bridge transition fixture.
    return RegionalState(names=("access_rim", "wheel_edge", "background"), err_sum=np.asarray([6.0, 3.0, 1.0]), elems=np.asarray([120.0, 180.0, 700.0]), sizes=np.asarray([0.35, 0.45, 0.80]), vm_max=np.asarray([280.0, 220.0, 80.0]), volume=np.asarray([0.12, 0.20, 0.68]), adjacency=np.asarray([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]), hit_count=np.asarray([3.0, 2.0, 0.0]), marked_error_fraction=np.asarray([0.72, 0.48, 0.0]), marked_element_fraction=np.asarray([0.16, 0.10, 0.0]), n_equations=12000, step=step, h0=1.0, h_min=0.05, dim=3)  # Return a state with two persistent exact-Dörfler hotspots.


class _FakeMesh:  # Provide only the mesh attributes required by deterministic target compilation.
    def __init__(self) -> None:  # Build two adjacent triangular cells.
        self.cells = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=int)  # Store the simplex connectivity.
        self.node_sizes = np.ones(4, dtype=float)  # Store the current nodal mesh sizes.
        self.n_cells = 2  # Store the number of cells.
        self.n_nodes = 4  # Store the number of nodes.

    @property  # Expose one face-adjacency edge in the repository-compatible format.
    def cell_adjacency(self):  # Return adjacent cell pairs and an unused face payload.
        return np.asarray([[0, 1]], dtype=int), np.asarray([[1, 2]], dtype=int)  # Connect the two fixture cells.


def test_action_space_always_contains_exact_dorfler() -> None:  # Verify that planning can never remove the classical fallback.
    config = WorldControllerConfig(max_extra_depth=2, max_active_regions=2)  # Configure a non-trivial delegated action space.
    actions = enumerate_actions(_state(), config)  # Enumerate all admissible actions.
    assert actions[0].is_dorfler  # Require pure Dörfler as the first stable candidate.
    assert any(not action.is_dorfler for action in actions)  # Require at least one world-model alternative.
    assert all(action.extra_depth[2] == 0 for action in actions)  # Forbid semantics-only refinement of an unmarked background region.


def test_planner_uses_multi_step_world_model_only_after_gain_gate() -> None:  # Verify finite-horizon admission relative to a Dörfler trajectory.
    config = WorldControllerConfig(planning_horizon=4, beam_width=24, max_extra_depth=2, max_active_regions=2, candidate_regions=2, min_predicted_gain=0.001, max_log_error_sigma=0.50, resource_penalty=0.0, uncertainty_penalty=0.0, action_penalty=0.0, require_reference=False)  # Make the prior trajectory comparison observable in a large budget.
    model = RegionalWorldModel(config)  # Start with the finite-element prior and explicit uncertainty.
    decision = plan_action(model, _state(), n_eq_cap=2_000_000, config=config)  # Plan from the same state as the Dörfler baseline.
    assert decision.baseline_terminal_error > 0.0  # Require a valid baseline rollout.
    assert decision.selected_terminal_error > 0.0  # Require a valid selected rollout.
    assert decision.action.extra_depth in decision.trajectory  # Require the executable action to be the first planned action.
    assert decision.source in {"world_model", "dorfler"}  # Restrict decisions to the two explicit scientific policies.
    if decision.source == "world_model":  # Check the quantitative admission contract for a delegated action.
        assert decision.predicted_gain >= config.min_predicted_gain  # Require the configured terminal gain over Dörfler.
        assert not decision.action.is_dorfler  # Require actual delegated future depth.


def test_small_resource_cap_forces_dorfler_fallback() -> None:  # Verify that a finite-budget violation cannot execute a world action.
    config = WorldControllerConfig(planning_horizon=3, max_extra_depth=3, min_predicted_gain=0.0, require_reference=False)  # Permit aggressive proposals before the resource shield.
    model = RegionalWorldModel(config)  # Build the prior-only world model.
    decision = plan_action(model, _state(), n_eq_cap=1000, config=config)  # Supply a cap below the current equation count.
    assert decision.action.is_dorfler  # Require the exact classical fallback.
    assert decision.source == "dorfler"  # Require an explicit fallback source label.


def test_compiled_target_dominates_exact_dorfler() -> None:  # Verify the central action-level safety invariant.
    config = WorldControllerConfig(refine_factor=0.5, support_hops_per_depth=1, require_reference=False)  # Configure one local delegated halo.
    state = RegionalState(names=("access_rim", "background"), err_sum=np.asarray([8.0, 2.0]), elems=np.asarray([1.0, 1.0]), sizes=np.asarray([1.0, 1.0]), vm_max=np.asarray([1.0, 1.0]), volume=np.asarray([1.0, 1.0]), adjacency=np.asarray([[0.0, 1.0], [1.0, 0.0]]), hit_count=np.asarray([2.0, 0.0]), marked_error_fraction=np.asarray([1.0, 0.0]), marked_element_fraction=np.asarray([1.0, 0.0]), n_equations=8, step=0, h0=1.0, h_min=0.05, dim=2)  # Build a two-region target-compilation fixture.
    mesh = _FakeMesh()  # Build the minimal adjacent-cell mesh.
    base, candidate, support = compile_dorfler_dominating_target(mesh, np.asarray([0, 1]), np.asarray([0]), RegionAction((1, 0)), state, config)  # Compile one delegated future hit above exact Dörfler.
    marked_nodes = np.unique(mesh.cells[0])  # Identify mandatory Dörfler nodes.
    assert support >= 1  # Require a non-empty physically supported delegated region.
    assert np.all(candidate <= mesh.node_sizes + 1.0e-12)  # Forbid any local coarsening.
    assert np.all(candidate[marked_nodes] <= base[marked_nodes] + 1.0e-12)  # Require complete mandatory-support domination.
    assert np.any(candidate[marked_nodes] < base[marked_nodes] - 1.0e-12)  # Require the delegated action to add real future depth.


def test_real_transition_updates_and_round_trips_world_model(tmp_path) -> None:  # Verify online residual learning and evidence persistence.
    config = WorldControllerConfig(min_rows_for_learning=1, ensemble_size=3, require_reference=False)  # Enable learning after one observed regional transition.
    model = RegionalWorldModel(config)  # Build an empty transition library.
    previous = _state(step=0)  # Build the pre-action real state.
    action = RegionAction((1, 0, 0))  # Delegate one future hit to the persistent access rim.
    observed = replace(previous, err_sum=np.asarray([1.9, 2.0, 0.95]), elems=np.asarray([320.0, 250.0, 760.0]), sizes=np.asarray([0.17, 0.35, 0.75]), n_equations=19500, step=1)  # Build a non-trivial observed Gmsh-plus-CalculiX transition.
    model.observe(previous, action, observed)  # Add real action-conditioned residual evidence.
    assert model.sample_count == previous.n_regions  # Require one residual row per stable semantic region.
    path = model.save(tmp_path / "world_model.json")  # Persist the independent transition library.
    restored = RegionalWorldModel.load(path, config=config)  # Restore the same evidence schema.
    assert restored.sample_count == model.sample_count  # Require lossless transition-row recovery.
    prediction = restored.predict(previous, action)  # Evaluate the learned ensemble after restoration.
    assert np.all(np.isfinite(prediction.next_error_mean))  # Require finite regional estimator predictions.
    assert np.all(np.isfinite(prediction.next_elems_mean))  # Require finite regional resource predictions.


def test_controller_does_not_import_local_prediction() -> None:  # Enforce the clean scientific separation requested for WM-VLA.
    import visionamr.vla.world_controller as controller  # Import the tested canonical controller module.

    source = inspect.getsource(controller)  # Read the exact controller source text.
    assert "from ..baselines.local_prediction" not in source  # Forbid a direct local-prediction baseline import.
    assert "from .local_prediction" not in source  # Forbid a relative local-prediction implementation import.
    assert "predicted_sizes(" not in source  # Forbid reuse of the local-prediction continuous size field.
