"""Temporal-state tests for regional world-model rollouts."""  # Describe the test module purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
import numpy as np  # Import numerical arrays for a synthetic regional state.
from visionamr.vla.world_model import RegionAction, ResidualWorldModel, WorldState  # Import the temporal world-model contracts.

def _state() -> WorldState:  # Build a two-region state whose hit counts exclude the current marking.
    return WorldState(names=("opening_rim", "field"), err_sum=np.array([8.0, 2.0]), elems=np.array([60.0, 40.0]), sizes=np.array([1.0, 1.5]), vm_max=np.array([10.0, 2.0]), volume=np.array([1.0, 4.0]), adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]), dorfler_error_fraction=np.array([0.7, 0.0]), dorfler_element_fraction=np.array([0.3, 0.0]), hit_count=np.array([2.0, 0.0]), n_equations=300.0, eq_per_elem=3.0, h_min=0.1, h0=2.0, dim=3, step=2)  # Return a validated pre-action state.

def test_prediction_counts_the_current_dorfler_hit_once() -> None:  # Guard against temporal information leakage or double counting.
    state = _state()  # Build the pre-action state.
    prediction = ResidualWorldModel().predict(state, RegionAction((0, 0), source="dorfler"))  # Roll one standard Dörfler transition forward.
    np.testing.assert_allclose(prediction.state.hit_count, np.array([3.0, 0.0]))  # Require exactly one new hit in the currently marked region.

def test_prediction_does_not_mutate_observed_hit_counts() -> None:  # Verify immutable temporal state semantics.
    state = _state()  # Build the pre-action state.
    original = state.hit_count.copy()  # Preserve the measured completed-hit vector.
    ResidualWorldModel().predict(state, RegionAction((1, 0)))  # Evaluate a proactive action.
    np.testing.assert_allclose(state.hit_count, original)  # Require the real observation to remain unchanged.
