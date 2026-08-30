"""Tests for the frozen partition shared by new-stack world VLA and existing RL."""  # Describe the focused protocol contract suite.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
import hashlib  # Import SHA-256 for conventional exact-file freeze-record verification.
import json  # Import JSON for transparent tampering tests.
from types import SimpleNamespace  # Import compact synthetic problem metadata.
import numpy as np  # Import numerical arrays and assertions.
import pytest  # Import explicit validation-error assertions.
from visionamr.bridge_cases import make_box_girder_diaphragm  # Import the canonical protocol problem family.
from visionamr.geometry import FeatureAnchor  # Import exact semantic feature metadata for a small graph fixture.
from visionamr.mesher import Mesh  # Import the repository simplex mesh contract.
from visionamr.vla.partition_spec import ASSIGNMENT_VERSION, BACKGROUND_RULE, PROBE_SIZE_RULE, PartitionSpecRegistry, SharedPartitioner, build_partition_spec, load_partition_spec, probe_mesh_sha256  # Import the complete public frozen-partition surface.

def _bridge_probe() -> Mesh:  # Build a tiny unsolved mesh accepted by the canonical bridge metadata fixture.
    nodes = np.array([[270.0, 150.0, 120.0], [330.0, 150.0, 120.0], [300.0, 210.0, 120.0], [300.0, 180.0, 180.0], [360.0, 210.0, 180.0]], dtype=float)  # Place two face-adjacent tetrahedra near the diaphragm.
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=int)  # Share one triangular face for deterministic graph construction.
    return Mesh(nodes=nodes, cells=cells, dim=3)  # Return the synthetic common-probe stand-in.

def _small_problem() -> SimpleNamespace:  # Build two exactly centered semantic regions for a non-empty adjacency test.
    features = [FeatureAnchor("alpha", 0.25, 0.25, 0.25, "feature"), FeatureAnchor("beta", 0.50, 0.50, 0.50, "feature")]  # Align anchors with two face-adjacent tetrahedral centroids.
    return SimpleNamespace(name="box_girder_diaphragm", dim=3, bbox=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), h0=1.0, h_ref=0.5, h_min=0.1, params={"shape": "fixture", "pressure": 4.0}, features=features, singular_points=[], singular_segments=[])  # Supply every fingerprint and partition field without native meshing.

def _small_probe() -> Mesh:  # Build two face-adjacent tetrahedra whose centroids equal the synthetic anchors.
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=float)  # Define one unit-corner bipyramid fixture.
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=int)  # Create one exact shared triangular face.
    return Mesh(nodes=nodes, cells=cells, dim=3)  # Return the deterministic graph fixture.

def test_bridge_spec_roundtrip_and_uniform_rl_sizes(tmp_path) -> None:  # Verify canonical bridge compatibility, persistence, and no hidden size prior.
    problem = make_box_girder_diaphragm()  # Build protocol geometry metadata without invoking Gmsh.
    probe = _bridge_probe()  # Build an unsolved deterministic probe fixture.
    geometry_hash = "a" * 64  # Supply a manifest-shaped geometry identity.
    spec = build_partition_spec(problem, geometry_hash, probe)  # Freeze the semantic assignment and graph exactly once.
    assert spec.names == ("diaphragm_joints", "inspection_opening", "left_bearing", "right_bearing", "wheel_load", "field_remainder")  # Require deterministic region and background order.
    assert spec.to_dict()["assignment"]["version"] == ASSIGNMENT_VERSION  # Require explicit assignment-version evidence.
    assert spec.to_dict()["background"]["rule"] == BACKGROUND_RULE  # Require explicit background semantics.
    assert spec.to_dict()["probe"]["size_rule"] == PROBE_SIZE_RULE  # Require a global uniform probe declaration.
    assert all("h" not in region for region in spec.to_dict()["regions"])  # Forbid semantic mesh sizes in every visual region definition.
    path = tmp_path / "partition_spec.json"  # Select the protocol-required per-instance filename.
    spec.save(path)  # Persist the complete transparent specification.
    restored = load_partition_spec(path, expected_sha256=spec.spec_sha256, problem=problem, expected_geometry_hash=geometry_hash)  # Revalidate hashes and runtime geometry during load.
    assert load_partition_spec(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()).file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()  # Accept and expose the conventional exact-file freeze digest too.
    np.testing.assert_array_equal(restored.assign(probe), spec.assign(probe))  # Require assignment roundtrip identity.
    np.testing.assert_allclose(restored.adjacency_matrix(), spec.adjacency_matrix())  # Require fixed new-world graph roundtrip identity.
    shared = SharedPartitioner(restored, expected_geometry_hash=geometry_hash)  # Bind one provider to the same manifest identity.
    rl_partition = shared.partition_for_rl(problem, probe)  # Build the thin existing-RL adapter after exact probe verification.
    np.testing.assert_allclose(rl_partition.sizes(), np.full(len(spec.names), problem.h0))  # Require every initial regional size to equal the uniform probe size.
    np.testing.assert_array_equal(rl_partition.assign(probe), restored.assign(probe))  # Require exact WM/RL element labels.
    np.testing.assert_array_equal(rl_partition.adjacency_matrix(), restored.fixed_adjacency)  # Require the same persisted binary graph before DQN normalization.

def test_fixed_adjacency_comes_from_probe_once() -> None:  # Verify graph construction and remesh independence explicitly.
    problem = _small_problem()  # Build the two-region synthetic geometry metadata.
    probe = _small_probe()  # Build its common uniform probe.
    spec = build_partition_spec(problem, "b" * 64, probe)  # Freeze labels and shared-face graph.
    assert spec.names == ("alpha", "beta", "field_remainder")  # Require stable sorted semantic order plus background.
    assert spec.fixed_adjacency[0, 1] == 1 and spec.fixed_adjacency[1, 0] == 1  # Require the one observed cross-region probe interface.
    altered = Mesh(nodes=probe.nodes.copy(), cells=np.array([[0, 1, 2, 3]], dtype=int), dim=3)  # Build a later one-cell remesh with different topology.
    np.testing.assert_array_equal(spec.adjacency_matrix(altered, spec.assign(altered)), spec.adjacency_matrix())  # Require the new-world graph to remain frozen after remeshing.
    rl_partition = spec.as_rl_partition(problem)  # Build the corresponding existing-RL adapter.
    np.testing.assert_array_equal(rl_partition.adjacency_matrix(altered, spec.assign(altered)), spec.fixed_adjacency)  # Require the same fixed graph in the RL adapter.

def test_spec_rejects_tampering_and_wrong_probe(tmp_path) -> None:  # Verify self-hash, case binding, and exact probe evidence.
    problem = _small_problem()  # Build deterministic synthetic geometry metadata.
    probe = _small_probe()  # Build its exact common probe.
    spec = build_partition_spec(problem, "c" * 64, probe)  # Freeze one valid specification.
    path = tmp_path / "partition_spec.json"  # Select the persisted artifact path.
    spec.save(path)  # Write the valid source artifact.
    payload = json.loads(path.read_text(encoding="utf-8"))  # Parse the transparent JSON for controlled tampering.
    payload["regions"][0]["priority"] = 9.0  # Change one hash-covered assignment parameter.
    path.write_text(json.dumps(payload), encoding="utf-8")  # Persist the altered document without updating its digest.
    with pytest.raises(ValueError, match="spec_sha256"):  # Require explicit integrity failure.
        load_partition_spec(path)  # Reject the tampered semantic body.
    assert probe_mesh_sha256(probe) != probe_mesh_sha256(Mesh(nodes=probe.nodes + 0.01, cells=probe.cells, dim=3))  # Require coordinate changes to alter the exact probe identity.
    with pytest.raises(ValueError, match="probe mesh hash"):  # Require runtime common-probe binding.
        spec.verify(problem=problem, expected_geometry_hash="c" * 64, probe_mesh=Mesh(nodes=probe.nodes + 0.01, cells=probe.cells, dim=3))  # Reject a mismatched regenerated probe.

def test_registry_resolves_one_spec_per_case(tmp_path) -> None:  # Verify the benchmark harness's deterministic per-case loader.
    problem = _small_problem()  # Build deterministic synthetic geometry metadata.
    case_id = "BGD-001-fixture"  # Define one safe manifest-shaped case identifier.
    geometry_hash = "d" * 64  # Define its geometry identity.
    spec = build_partition_spec(problem, geometry_hash, _small_probe())  # Build the per-instance semantic specification.
    path = tmp_path / case_id / "partition_spec.json"  # Resolve the required registry layout.
    spec.save(path)  # Persist the unique case artifact.
    registry = PartitionSpecRegistry(tmp_path, expected_sha256={case_id: spec.spec_sha256})  # Freeze the expected digest by case ID.
    shared = registry.partitioner_for(case_id, problem, geometry_hash)  # Load a provider only after complete integrity checks.
    assert shared.spec.spec_sha256 == spec.spec_sha256  # Require the registry to return the intended frozen instance.
    assert registry.partition_for(case_id, problem, geometry_hash).spec_sha256 == spec.spec_sha256  # Require direct new-stack injection to resolve the same object.
    with pytest.raises(ValueError, match="safe path component"):  # Require path traversal rejection.
        registry.path_for("../escape")  # Reject an unsafe case identifier.
