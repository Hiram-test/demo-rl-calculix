"""Scientific action-boundary tests for world-model VLA planning."""  # Describe the test module purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
import numpy as np  # Import numerical arrays for synthetic regional states.
from visionamr.vla.planner import PlannerConfig, enumerate_actions  # Import the bounded action enumerator.
from visionamr.vla.world_model import WorldState  # Import the regional state contract.

def _state() -> WorldState:  # Build a state in which the generic field has the largest raw error share.
    return WorldState(names=("wheel_patch_zone", "opening_rim_zone", "field"), err_sum=np.array([2.0, 1.0, 20.0]), elems=np.array([20.0, 20.0, 60.0]), sizes=np.array([1.0, 1.0, 1.8]), vm_max=np.array([8.0, 7.0, 2.0]), volume=np.array([1.0, 1.0, 8.0]), adjacency=np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]), dorfler_error_fraction=np.array([0.2, 0.1, 0.8]), dorfler_element_fraction=np.array([0.1, 0.1, 0.4]), hit_count=np.array([1.0, 1.0, 3.0]), n_equations=300.0, eq_per_elem=3.0, h_min=0.1, h0=2.0, dim=3, step=2)  # Return the validated synthetic state.

def test_generic_field_is_never_a_proactive_action() -> None:  # Prevent broad background refinement from masquerading as semantic world-model control.
    state = _state()  # Build the adversarial regional ranking case.
    actions = enumerate_actions(state, PlannerConfig(candidate_regions=3, max_extra_regions=2))  # Enumerate every legal sparse action.
    field_index = state.names.index("field")  # Locate the generic remainder region.
    assert actions  # Require the permanent pure-Dörfler candidate.
    assert all(action.extra_depth[field_index] == 0 for action in actions)  # Require all proactive depth on the field to remain zero.

def test_named_bridge_mechanisms_remain_actionable() -> None:  # Verify that the scientific boundary does not disable the world model entirely.
    state = _state()  # Build the same adversarial regional ranking case.
    actions = enumerate_actions(state, PlannerConfig(candidate_regions=3, max_extra_regions=2))  # Enumerate every legal sparse action.
    wheel_index = state.names.index("wheel_patch_zone")  # Locate the named load-transfer mechanism.
    opening_index = state.names.index("opening_rim_zone")  # Locate the named geometric stress mechanism.
    assert any(action.extra_depth[wheel_index] > 0 for action in actions)  # Require a wheel-patch proactive candidate.
    assert any(action.extra_depth[opening_index] > 0 for action in actions)  # Require an opening-rim proactive candidate.
