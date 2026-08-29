# Unit tests for the independent world-model-guided VLA implementation.  # Test module purpose.
from __future__ import annotations  # Enable postponed annotations consistently with production modules.
from pathlib import Path  # Inspect method-purity and line-comment contracts in source files.
import numpy as np  # Build deterministic synthetic region states and actions.
import pytest  # Assert explicit contract failures.
from visionamr.bridge_scenarios import make_bridge_pier_cap  # Verify the medium-complexity bridge scenario contract.
from visionamr.geometry import make_plate_holes  # Reuse a lightweight problem contract without meshing.
from visionamr.vla.dominance import DominanceConfig, evaluate_dorfler_floor  # Verify the Dörfler release gate.
from visionamr.vla.tool_contract import MeshAction, action_schema, fast_materialize_action, validate_action_payload  # Verify deterministic tool ownership.
from visionamr.vla.world_model import HybridGraphWorldModel, WorldPlanner, WorldPlannerConfig, WorldState  # Verify action-conditioned prediction and planning.


def _state(problem, error_scale: float = 1.0) -> WorldState:  # Build a small aligned synthetic region graph.
    return WorldState(  # Construct a complete valid measured-like state.
        names=("load", "support", "field"),  # Preserve a stable three-region order.
        grades=np.array([3, 3, 5], dtype=int),  # Start with moderate critical zones and a coarse field.
        sizes=np.array([0.38 * problem.h0, 0.38 * problem.h0, 0.72 * problem.h0], dtype=float),  # Derive valid positive sizes from the problem scale.
        err_sum=np.array([7.0, 2.0, 1.0], dtype=float) * float(error_scale),  # Concentrate most error in the load region.
        elems=np.array([400.0, 320.0, 480.0], dtype=float),  # Match 1.5 equations per element to the declared current resource state.
        vm_max=np.array([12.0, 6.0, 2.0], dtype=float),  # Preserve a strong load-zone stress contrast.
        vm_mean=np.array([4.0, 3.0, 1.5], dtype=float),  # Preserve positive regional mean stresses.
        h_meas=np.array([0.40 * problem.h0, 0.40 * problem.h0, 0.75 * problem.h0], dtype=float),  # Preserve realized mesh sizes on the same physical scale.
        volume=np.array([0.20, 0.25, 0.55], dtype=float),  # Preserve positive physical measures.
        semantics=np.array([[1.0, 0.1, 0.0, 0.0, 0.2], [0.1, 1.0, 0.8, 0.0, 0.2], [0.0, 0.2, 0.1, 0.0, 0.0]], dtype=float),  # Encode load, support, and field semantics.
        adjacency=np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=float),  # Connect the small region graph.
        total_error=10.0 * float(error_scale),  # Preserve the global estimator total.
        n_equations=1800,  # Preserve exact current resource use.
        qoi_error=0.20,  # Preserve a positive reference-relative QoI error.
        solve_index=1,  # Represent the first real WM solve.
        budget=6000,  # Leave room for an informative refinement action.
    ).validate()  # Verify complete state alignment.


def test_action_contract_rejects_continuous_or_extra_parameters() -> None:  # Ensure models cannot inject raw mesh sizes.
    schema = action_schema(3)  # Build a fixed-width MCP-ready action schema.
    assert schema["properties"]["deltas"]["items"]["enum"] == [-1, 0, 1]  # Confirm the complete discrete alphabet.
    with pytest.raises(ValueError):  # Expect undeclared continuous parameters to fail.
        validate_action_payload({"action_id": "bad", "deltas": [-1, 0, 0], "source": "world", "stop": False, "h": 0.1}, 3)  # Attempt direct size injection.
    with pytest.raises(ValueError):  # Expect invalid ordinal jumps to fail.
        validate_action_payload({"action_id": "bad", "deltas": [-2, 0, 0], "source": "world", "stop": False}, 3)  # Attempt an undeclared multi-level jump.


def test_tool_materialization_refines_without_importing_local_prediction() -> None:  # Verify deterministic relative grade mapping and resource projection.
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)  # Build a lightweight valid problem contract.
    action = MeshAction("refine_load", (-1, 0, 0), source="world", stop=False)  # Request one discrete load-zone refinement.
    state = _state(problem)  # Build the current measured-like state.
    materialized = fast_materialize_action(problem, state.sizes, state.grades, state.elems, state.adjacency, action, state.budget, 1.5)  # Let the deterministic tool own numerical sizes.
    assert materialized.sizes[0] < state.sizes[0]  # Confirm the selected critical region becomes finer.
    assert materialized.grades.tolist() == [2, 3, 5]  # Confirm exactly one ordinal level changed.
    assert materialized.predicted_equations <= state.budget  # Confirm cheap resource projection respects the hard cap.


def test_hybrid_world_model_predicts_action_conditioned_graph_transition() -> None:  # Verify refinement changes predicted error and resource state coherently.
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)  # Build a lightweight valid problem contract.
    state = _state(problem)  # Build the current measured-like state.
    action = MeshAction("refine_load", (-1, 0, 0), source="world", stop=False)  # Request critical-region refinement.
    materialized = fast_materialize_action(problem, state.sizes, state.grades, state.elems, state.adjacency, action, state.budget, 1.5)  # Materialize numerical sizes deterministically.
    model = HybridGraphWorldModel(problem.dim)  # Initialize mechanics-informed dynamics without real samples.
    prediction = model.predict(state, materialized)  # Roll the compact world model forward once.
    assert prediction.state.total_error < state.total_error  # Confirm the prior predicts lower estimator error under refinement.
    assert prediction.state.n_equations > state.n_equations  # Confirm refinement consumes additional equations.
    assert prediction.uncertainty > 0.0  # Confirm initialization does not claim false certainty.


def test_world_model_updates_only_from_real_transition_evidence() -> None:  # Verify online fitting consumes measured state changes.
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)  # Build a lightweight valid problem contract.
    previous = _state(problem)  # Build the previous measured state.
    action = MeshAction("refine_load", (-1, 0, 0), source="world", stop=False)  # Define the executed discrete action.
    materialized = fast_materialize_action(problem, previous.sizes, previous.grades, previous.elems, previous.adjacency, action, previous.budget, 1.5)  # Recover exact executed sizes.
    current = WorldState(names=previous.names, grades=materialized.grades, sizes=materialized.sizes, err_sum=previous.err_sum * np.array([0.55, 0.92, 0.98]), elems=previous.elems * np.array([1.70, 1.08, 1.02]), vm_max=previous.vm_max, vm_mean=previous.vm_mean, h_meas=materialized.sizes, volume=previous.volume, semantics=previous.semantics, adjacency=previous.adjacency, total_error=float(np.sum(previous.err_sum * np.array([0.55, 0.92, 0.98]))), n_equations=2600, qoi_error=0.14, solve_index=2, budget=previous.budget).validate()  # Construct a measured-like next state.
    model = HybridGraphWorldModel(problem.dim)  # Initialize the compact ensemble.
    info = model.update(previous, current, materialized.sizes)  # Learn from the explicit real transition.
    assert info["accepted"] is True  # Confirm the compatible finite transition entered memory.
    assert model.n_transitions == 1  # Count one expensive transition rather than region rows.


def test_multistep_planner_returns_only_discrete_first_action() -> None:  # Verify receding-horizon search never emits raw numerical parameters.
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)  # Build a lightweight valid problem contract.
    state = _state(problem)  # Build a high-error current state.
    model = HybridGraphWorldModel(problem.dim)  # Initialize mechanics-informed dynamics.
    planner = WorldPlanner(problem, model, WorldPlannerConfig(horizon=3, beam_width=12, min_predicted_gain=-1.0))  # Force an executable multi-step plan for the unit test.
    plan = planner.plan(state, eq_per_elem=1.5)  # Perform bounded counterfactual search.
    assert all(value in (-1, 0, 1) for value in plan.action.deltas)  # Confirm the selected action remains discrete.
    assert len(plan.sequence) >= 1  # Confirm finite-horizon planning produced an explicit imagined sequence.
    assert plan.materialized.action.action_id == plan.action.action_id  # Confirm tool materialization preserves action provenance.


def test_bridge_pier_cap_is_medium_complexity_three_dimensional_component() -> None:  # Verify the requested bridge scenario semantics without invoking Gmsh.
    problem = make_bridge_pier_cap()  # Build the canonical component contract.
    assert problem.dim == 3  # Confirm full three-dimensional elasticity.
    assert problem.name == "bridge_pier_cap"  # Confirm the stable campaign family name.
    assert len(problem.tractions) == 2  # Confirm two competing bearing-load regions.
    assert any(feature.kind == "hole" for feature in problem.features)  # Confirm prestressing-duct semantics exist.
    assert any(feature.kind == "corner" for feature in problem.features)  # Confirm column-cap re-entrant semantics exist.
    assert len(problem.singular_segments) >= 20  # Confirm multiple independent concentration lines drive the reference.


def test_dorfler_floor_passes_and_fails_transparently() -> None:  # Verify the release gate cannot silently relabel an inferior result.
    passing = [  # Build two independent measured trajectories with WM non-inferior.
        {"method": "wm", "solve_index": 1, "n_equations": 1000, "e_energy": 0.40, "e_qoi": 0.30},  # Preserve WM probe evidence.
        {"method": "wm", "solve_index": 2, "n_equations": 1800, "e_energy": 0.20, "e_qoi": 0.10},  # Preserve WM adaptive evidence.
        {"method": "dorfler", "solve_index": 1, "n_equations": 1000, "e_energy": 0.42, "e_qoi": 0.31},  # Preserve Dörfler probe evidence.
        {"method": "dorfler", "solve_index": 2, "n_equations": 1800, "e_energy": 0.205, "e_qoi": 0.102},  # Preserve Dörfler adaptive evidence.
    ]  # Finish the passing trajectory set.
    gate_pass = evaluate_dorfler_floor(passing, "wm", "dorfler", 2000, DominanceConfig())  # Evaluate the pre-registered release gate.
    assert gate_pass["pass"] is True  # Confirm a genuinely non-inferior trajectory passes.
    failing = [dict(record) for record in passing]  # Copy the same independent measured records.
    failing[1]["e_energy"] = 0.30  # Make the final WM energy error materially inferior.
    gate_fail = evaluate_dorfler_floor(failing, "wm", "dorfler", 2000, DominanceConfig())  # Re-evaluate without changing tolerances.
    assert gate_fail["pass"] is False  # Confirm the inferior trajectory remains a failed result.
    assert gate_fail["checks"]["final_energy"] is False  # Confirm the exact failed criterion is disclosed.


def test_world_pipeline_preserves_clean_local_prediction_boundary() -> None:  # Lock the user's requested method purity in source.
    root = Path(__file__).resolve().parents[1]  # Locate the repository root.
    pipeline = (root / "visionamr" / "vla" / "pipeline_world.py").read_text(encoding="utf-8")  # Read the independent controller source.
    world = (root / "visionamr" / "vla" / "world_model.py").read_text(encoding="utf-8")  # Read the compact dynamics source.
    assert "predicted_sizes" not in pipeline  # Confirm the pipeline never calls LP target-size prediction.
    assert "predicted_sizes" not in world  # Confirm the world model never distills LP outputs.
    assert "baselines.local_prediction" not in pipeline  # Confirm no hidden LP import enters the controller.


def test_new_python_sources_keep_line_level_comments() -> None:  # Enforce the user's line-comment requirement on all newly implemented Python files.
    root = Path(__file__).resolve().parents[1]  # Locate the repository root.
    paths = [root / "visionamr" / "bridge_scenarios.py", root / "visionamr" / "vla" / "tool_contract.py", root / "visionamr" / "vla" / "world_model.py", root / "visionamr" / "vla" / "pipeline_world.py", root / "visionamr" / "vla" / "dominance.py", root / "scripts" / "run_wm_vla_bridge3d.py"]  # Enumerate the complete new implementation.
    for path in paths:  # Inspect every new Python source file.
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):  # Inspect every physical source line.
            if not line.strip():  # Permit blank separators for readability.
                continue  # Move to the next physical line.
            assert "#" in line, f"{path}:{line_number} lacks a line-level comment"  # Require an explicit comment marker on every nonblank line.
