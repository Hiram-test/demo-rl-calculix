"""Focused tests for train-only world-model acquisition and shared partition indexing."""  # Describe the no-native verification surface.
from __future__ import annotations  # Postpone annotation evaluation consistently with the implementation.
import json  # Inspect strict plan, model, summary, and registry artifacts.
from pathlib import Path  # Build isolated manifest, partition, and training roots.
import subprocess  # Exercise the actual solve-free command-line boundary.
import sys  # Reuse the active test interpreter for the CLI subprocess.
from types import SimpleNamespace  # Build minimal injected problem and partition objects.
from typing import Any  # Annotate heterogeneous fake-runner audit values.
import numpy as np  # Construct deterministic world states and a tiny unsolved mesh.
import pytest  # Assert fail-fast treatment of non-native configuration and API defects.
from visionamr.bridge_case_manifest import build_case_manifest, write_case_manifest  # Build and persist the exact authenticated 48-case design.
from visionamr.mesher import GmshMeshingError, Mesh  # Construct tiny meshes and inject the explicit native meshing failure category.
from visionamr.vla.four_way_world_training import MODEL_FILENAME  # Verify the exact deployment snapshot identity.
from visionamr.vla.four_way_world_training import PARTITION_INDEX_FILENAME  # Verify the shared registry inventory location.
from visionamr.vla.four_way_world_training import WorldTrainingConfig  # Reuse the immutable six-solve acquisition settings.
from visionamr.vla.four_way_world_training import build_partition_plan  # Verify explicit split-aware no-write planning.
from visionamr.vla.four_way_world_training import build_training_plan  # Verify the train-only authenticated plan.
from visionamr.vla.four_way_world_training import generate_partition_specs  # Verify exact file/body SHA registry publication without CalculiX.
from visionamr.vla.four_way_world_training import sha256_file  # Verify the persisted model and registry identities.
from visionamr.vla.four_way_world_training import train_world_model_transition_library  # Exercise all 24 cases with an injected fake runner.
from visionamr.vla.partition_spec import build_partition_spec  # Build valid shared specs from a tiny injected unsolved probe.
from visionamr.vla.world.model import RegionAction, WorldState  # Generate public-model transition observations in the fake runner.

def _state(step: int) -> WorldState:  # Build one valid two-region measured state for public ResidualWorldModel fitting.
    scale = float(step + 1)  # Increase resource use and reduce indicator mass deterministically.
    return WorldState(names=("inspection_opening", "field_remainder"), err_sum=np.asarray([12.0 / scale, 3.0 / scale], dtype=float), elems=np.asarray([10.0 * scale, 20.0 * scale], dtype=float), sizes=np.asarray([1.0 / scale, 1.0 / scale], dtype=float), vm_max=np.asarray([2.0, 1.0], dtype=float), volume=np.asarray([0.4, 0.6], dtype=float), adjacency=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float), dorfler_error_fraction=np.asarray([0.6, 0.1], dtype=float), dorfler_element_fraction=np.asarray([0.4, 0.1], dtype=float), hit_count=np.asarray([scale, 0.0], dtype=float), n_equations=int(1000 * scale), eq_per_elem=float(1000 * scale / (30 * scale)), h_min=0.01, h0=1.0, dim=3, step=step)  # Return a complete immutable state with stable names and finite physical values.

def _fake_problem_factory(case: dict[str, Any]) -> SimpleNamespace:  # Avoid geometry and native dependencies in the injected acquisition test.
    return SimpleNamespace(case_id=str(case["case_id"]))  # Preserve only the identity needed by the fake callbacks.

def _fake_partition_loader(case: dict[str, Any], _problem: Any, _root: Path, _manifest_sha: str) -> tuple[SimpleNamespace, dict[str, Any]]:  # Inject one authenticated-looking partition boundary per train case.
    digest = str(case["geometry_hash"])  # Reuse a complete manifest SHA solely as deterministic fake identity.
    partition = SimpleNamespace(spec_sha256=digest)  # Supply the public field inspected by training summaries.
    receipt = {"case_id": str(case["case_id"]), "geometry_hash": digest, "file_sha256": digest, "spec_sha256": digest}  # Preserve complete per-case setup evidence.
    return partition, receipt  # Return the injected shared semantic object and receipt.

def _fake_case_runner(case: dict[str, Any], _problem: Any, partition: Any, model: Any, config: WorldTrainingConfig, _case_root: Path) -> dict[str, Any]:  # Simulate exactly six solves while fitting only through the public model API.
    states = [_state(step) for step in range(config.solves_per_case)]  # Build the exact six realized fake solve states.
    transitions: list[dict[str, Any]] = []  # Retain one transparent receipt per fitted real transition.
    action = RegionAction.dorfler(states[0])  # Use the mandatory exact-Dörfler safety action throughout the injected path.
    for index in range(config.solves_per_case - 1):  # Fit exactly five completed successor observations.
        prediction = model.predict(states[index], action)  # Query the current ResidualWorldModel through its public prediction API.
        model.observe(states[index], action, states[index + 1])  # Fit the current ResidualWorldModel through its public observation API.
        transitions.append({"transition_index": index + 1, "prediction": {"error_ratio_mean": prediction.error_ratio_mean, "equation_ratio_mean": prediction.equation_ratio_mean}, "actual": {"total_error_ratio": states[index + 1].total_error / states[index].total_error}, "executed_action": list(action.extra_depth), "source_mesh_sha256": f"{index:064x}"[-64:], "actual_mesh_sha256": f"{index + 1:064x}"[-64:]})  # Preserve complete fake prediction/actual/action/mesh evidence.
    solves = [{"solve_index": index + 1, "mesh_sha256": f"{index:064x}"[-64:], "solver_record": {"n_equations": states[index].n_equations}, "state": {"step": index}} for index in range(config.solves_per_case)]  # Preserve one fake solver and mesh receipt per real solve.
    return {"schema": "wmvla-four-way-world-training-v1", "protocol_id": "WMVLA-4WAY-P1", "phase": "case_acquisition", "case_id": str(case["case_id"]), "split": "train", "geometry_hash": str(case["geometry_hash"]), "partition_spec_sha256": str(partition.spec_sha256), "status": "ok", "complete": True, "equation_budget": config.equation_budget, "planned_real_solve_count": config.solves_per_case, "real_solve_count": len(solves), "transition_count": len(transitions), "solves": solves, "actions": [{"action_index": index + 1, "executed_action": list(action.extra_depth), "certificate": {"accepted": True}, "timing_s": {"world_model_planning": 0.0}} for index in range(config.solves_per_case - 1)], "transitions": transitions, "failures": [], "solver_logs": [], "timing_s": {"offline_total": 0.01}}  # Return the complete successful injected case contract.

def test_training_plan_contains_only_the_24_train_cases(tmp_path: Path) -> None:  # Prove solve-free planning does not expose validation or blind identities and performs no writes.
    manifest_path, _sidecar, _digest = write_case_manifest(tmp_path / "protocol")  # Persist the exact checksummed manifest.
    output = tmp_path / "training"  # Select a nonexistent output that planning must not create.
    plan = build_training_plan(manifest_path, tmp_path / "protocol" / "partitions", output)  # Authenticate and materialize the train-only plan.
    manifest = build_case_manifest()  # Recover validation and blind identities only inside this isolation assertion.
    excluded_ids = {str(case["case_id"]) for case in manifest["cases"] if case["split"] != "train"}  # Collect all forbidden non-train identifiers.
    serialized = json.dumps(plan, sort_keys=True)  # Flatten the plan for complete leakage checking.
    assert plan["training_case_count"] == 24 and plan["planned_real_solve_count"] == 144 and plan["planned_transition_count"] == 120  # Require the exact 24-by-six acquisition contract.
    assert plan["scientific_config"]["tool_gateway"]["nodal_gradation"] == 1.0  # Require the frozen PR-number-40 interpolation behavior explicitly.
    assert plan["validation_split_accessed"] is False and plan["test_split_accessed"] is False and all(case_id not in serialized for case_id in excluded_ids)  # Prevent non-train identity or parameter propagation.
    assert not output.exists()  # Prove plan construction created no training artifact.

def test_injected_fake_runner_fits_and_hashes_exact_24_by_6_campaign(tmp_path: Path) -> None:  # Exercise complete orchestration without Gmsh or CalculiX.
    manifest_path, _sidecar, _digest = write_case_manifest(tmp_path / "protocol")  # Persist the exact authenticated case design.
    output = tmp_path / "training" / "world_model"  # Select a new isolated artifact root.
    summary = train_world_model_transition_library(manifest_path, tmp_path / "unused_partitions", output, case_runner=_fake_case_runner, problem_factory=_fake_problem_factory, partition_loader=_fake_partition_loader)  # Run all 24 train identities through the injected six-solve callback.
    snapshot = json.loads((output / MODEL_FILENAME).read_text(encoding="utf-8"))  # Inspect the public model snapshot persisted through ResidualWorldModel.save.
    costs = json.loads((output / "training_costs.json").read_text(encoding="utf-8"))  # Inspect mandatory offline cost accounting.
    assert summary["eligible_for_freeze"] is True and summary["TEST_NOT_RUN"] is True  # Require a complete pre-blind model receipt.
    assert summary["real_training_solve_count"] == 144 and summary["fitted_transition_count"] == 120  # Require every planned solve and transition exactly once.
    assert snapshot["transition_count"] == 120 and len(snapshot["x"]) == 240 and len(snapshot["y"]) == 240  # Require two regional public-model rows for each of 120 transitions.
    assert summary["model_sha256"] == sha256_file(output / MODEL_FILENAME) and costs["training_complete"] is True  # Bind the freeze receipt and costs to exact model bytes and completeness.
    assert len(list((output / "cases").glob("*/case_training.json"))) == 24  # Require one complete retained case receipt for every train identity.

def test_injected_case_failure_is_retained_and_blocks_freeze_eligibility(tmp_path: Path) -> None:  # Prove an unfavorable train outcome is neither dropped nor silently replaced.
    manifest_path, _sidecar, _digest = write_case_manifest(tmp_path / "protocol")  # Persist the exact authenticated case design.
    failed_id = str(build_case_manifest()["cases"][0]["case_id"])  # Identify the first sorted train case for one deterministic injected failure.
    def failing_runner(case: dict[str, Any], problem: Any, partition: Any, model: Any, config: WorldTrainingConfig, case_root: Path) -> dict[str, Any]:  # Wrap the complete fake runner with one retained callback failure.
        if str(case["case_id"]) == failed_id:  # Select exactly one manifest train identity.
            raise GmshMeshingError("injected native failure")  # Exercise only the orchestrator's typed numerical-failure conversion.
        return _fake_case_runner(case, problem, partition, model, config, case_root)  # Complete every remaining fixed-schedule train case normally.
    output = tmp_path / "training" / "failed_world_model"  # Select a new isolated artifact root.
    summary = train_world_model_transition_library(manifest_path, tmp_path / "unused_partitions", output, case_runner=failing_runner, problem_factory=_fake_problem_factory, partition_loader=_fake_partition_loader)  # Execute the full ordered schedule with one retained failure.
    failed_receipt = json.loads((output / "cases" / failed_id / "case_training.json").read_text(encoding="utf-8"))  # Inspect the exact failed-case evidence.
    assert summary["eligible_for_freeze"] is False and summary["complete_case_count"] == 23 and summary["fitted_transition_count"] == 115  # Prevent an incomplete model from entering the blind freeze.
    assert failed_receipt["status"] == "failed" and failed_receipt["failures"][0]["exception_type"] == "GmshMeshingError"  # Preserve the unfavorable typed native outcome explicitly.
    assert len(list((output / "cases").glob("*/case_training.json"))) == 24  # Require the remaining fixed training schedule to complete without dropping identities.

def test_untyped_case_runner_defect_invalidates_campaign_immediately(tmp_path: Path) -> None:  # Prove configuration or API defects cannot be mislabeled as numerical training failures.
    manifest_path, _sidecar, _digest = write_case_manifest(tmp_path / "protocol")  # Persist the exact authenticated case design used by setup validation.
    def defective_runner(case: dict[str, Any], problem: Any, partition: Any, model: Any, config: WorldTrainingConfig, case_root: Path) -> dict[str, Any]:  # Inject a non-native implementation defect at the first scheduled case.
        del case, problem, partition, model, config, case_root  # Confirm the injected failure is independent of physical outputs.
        raise RuntimeError("injected schema defect")  # Represent a programming or artifact-contract error that must propagate.
    with pytest.raises(RuntimeError, match="schema defect"):  # Require the campaign to fail rather than emitting an eligible-looking partial model.
        train_world_model_transition_library(manifest_path, tmp_path / "unused_partitions", tmp_path / "training" / "invalid", case_runner=defective_runner, problem_factory=_fake_problem_factory, partition_loader=_fake_partition_loader)  # Enter the authenticated fixed schedule without native work.

def _tiny_probe(problem: Any) -> Mesh:  # Build a valid two-tetrahedron common-probe stand-in inside one canonical bridge bounding box.
    x0, y0, z0, x1, y1, z1 = (float(value) for value in problem.bbox)  # Read the geometry envelope without invoking its Gmsh builder.
    center = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5], dtype=float)  # Place the fixture safely inside the component.
    delta = min(x1 - x0, y1 - y0, z1 - z0) * 0.01  # Choose a positive local edge scale from the bounding box.
    nodes = np.asarray([center + [-delta, -delta, -delta], center + [delta, -delta, -delta], center + [-delta, delta, -delta], center + [-delta, -delta, delta], center + [delta, delta, delta]], dtype=float)  # Define a small face-adjacent tetrahedral pair.
    cells = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=int)  # Share one complete triangular face for deterministic adjacency.
    return Mesh(nodes=nodes, cells=cells, dim=3)  # Return the unsolved simplex mesh.

def _tiny_spec_generator(problem: Any, geometry_hash: str) -> Any:  # Replace Gmsh only while retaining the real partition schema and hashing logic.
    return build_partition_spec(problem, geometry_hash, _tiny_probe(problem))  # Build a valid persisted shared semantic specification.

def test_partition_generator_indexes_file_and_body_hashes_without_calculix(tmp_path: Path) -> None:  # Verify explicit split generation and append-only registry evidence.
    manifest_path, _sidecar, _digest = write_case_manifest(tmp_path / "protocol")  # Persist the exact authenticated case design.
    root = tmp_path / "protocol" / "partitions"  # Select the literal frozen shared-partition location.
    plan = build_partition_plan(manifest_path, root, "validation")  # Build an eight-case no-write plan.
    assert plan["case_count"] == 8 and plan["calculix_solve_count"] == 0 and not root.exists()  # Require explicit split cardinality and zero mutation.
    result = generate_partition_specs(manifest_path, root, "validation", spec_generator=_tiny_spec_generator)  # Generate real hashed specs from injected unsolved probes.
    index = json.loads((root / PARTITION_INDEX_FILENAME).read_text(encoding="utf-8"))  # Inspect the published registry inventory.
    repeated = generate_partition_specs(manifest_path, root, "validation", spec_generator=_tiny_spec_generator)  # Reverify the same frozen registry without changing its content identity.
    assert result["selected_case_count"] == 8 and result["calculix_solve_count"] == 0 and index["case_count"] == 8  # Require every validation geometry and no solver accounting.
    assert repeated["index_sha256"] == result["index_sha256"]  # Require idempotent re-verification to preserve exact deterministic index bytes.
    assert all(len(item["file_sha256"]) == 64 and len(item["spec_sha256"]) == 64 for item in index["partitions"])  # Require both exact-file and semantic-body identities.
    assert all((root / item["path"]).is_file() for item in index["partitions"])  # Require one transparent persisted spec per indexed case.

def test_world_training_cli_dry_run_is_write_free(tmp_path: Path) -> None:  # Exercise the real command's authenticated no-partition no-native boundary.
    manifest_path, _sidecar, _digest = write_case_manifest(tmp_path / "protocol")  # Persist the exact checksummed manifest.
    output = tmp_path / "training" / "world_model"  # Select a nonexistent artifact root.
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_bridge_world_model.py"  # Resolve the checked-out command entry point.
    completed = subprocess.run([sys.executable, str(script), "--manifest", str(manifest_path), "--partition-root", str(tmp_path / "missing_partitions"), "--output", str(output), "--dry-run"], check=True, capture_output=True, text=True)  # Launch planning without native work or writes.
    payload = json.loads(completed.stdout)  # Parse the complete strict terminal plan.
    assert payload["training_case_count"] == 24 and payload["planned_real_solve_count"] == 144  # Require the exact frozen campaign size.
    assert payload["reads_reference_errors"] is False and payload["TEST_NOT_RUN"] is True  # Require explicit scientific isolation and pretest state.
    assert not output.exists()  # Prove the CLI dry run did not create any artifact.
