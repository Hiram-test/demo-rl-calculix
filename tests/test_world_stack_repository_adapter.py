"""Focused repository-API tests for the post-V0 world stack."""  # Describe the mechanical adapter coverage below.
from __future__ import annotations  # Enable postponed evaluation of annotations.
from pathlib import Path  # Import portable temporary artifact paths.
from types import SimpleNamespace  # Build compact repository-shaped test objects.
import numpy as np  # Import numerical mesh and indicator arrays.
import pytest  # Assert hard rejection of a post-compilation structural violation.
from visionamr.bridge_cases import make_box_girder_diaphragm  # Exercise the canonical root bridge factory.
from visionamr.mesher import Mesh  # Exercise the repository mesh contract directly.
from visionamr.vla.world import pipeline as pipeline_module  # Patch only native execution boundaries in focused tests.
from visionamr.vla.world import tool_gateway as tool_gateway_module  # Patch only native Gmsh generation while checking compiled fields.
from visionamr.vla.world.pipeline import WorldVLAConfig, _RuntimeAdapter, _record_metric, run_world_model_vla  # Exercise runtime adaptation and no-reference execution.
from visionamr.vla.world.model import RegionAction  # Construct one bounded proactive action for certificate testing.
from visionamr.vla.world.tool_gateway import MCPToolGateway  # Exercise repository post, constraint, graph, and size-field adaptation.

def _mesh() -> Mesh:  # Build a small valid repository Mesh without invoking Gmsh.
    nodes = np.asarray([[72.0, 100.0, 0.0], [528.0, 100.0, 0.0], [300.0, 180.0, 100.0], [300.0, 80.0, 30.0], [350.0, 220.0, 120.0]], dtype=float)  # Include one node on each canonical support and three free nodes.
    cells = np.asarray([[0, 2, 3, 4], [1, 2, 4, 3]], dtype=int)  # Form two repository tetrahedral connectivity rows.
    return Mesh(nodes=nodes, cells=cells, dim=3)  # Return the concrete repository mesh object.

class _Runner:  # Mimic the exact FemRunner methods consumed by the runtime adapter.
    def __init__(self, problem: object, mesh: Mesh) -> None:  # Store one canonical problem and solved mesh.
        self.problem = problem  # Expose the repository problem attribute.
        self.mesh = mesh  # Preserve the synthetic solved mesh.
        self.reference = None  # Match FemRunner's unloaded-reference sentinel.
        self.reference_calls = 0  # Count attempted reference solves.
        self.solve_calls: list[dict[str, object]] = []  # Record mandatory solve metadata.
        self.records: list[object] = []  # Match FemRunner's counted-record collection.
        self.workdir = Path(".")  # Provide a default artifact root for direct adapter tests.
    def ensure_reference(self) -> object:  # Mimic an expensive reference boundary.
        self.reference_calls += 1  # Record that the runtime requested a reference.
        self.reference = object()  # Install an opaque cached reference.
        return self.reference  # Return the cached reference object.
    def solve_mesh(self, mesh: Mesh, *, method: str, stage: str, extra: dict | None = None) -> tuple[object, object]:  # Match the exact FemRunner solve signature.
        self.solve_calls.append({"mesh": mesh, "method": method, "stage": stage, "extra": extra})  # Preserve every mandatory argument.
        post = SimpleNamespace(mesh=mesh, vm_elem=np.asarray([5.0, 9.0], dtype=float))  # Build the post fields consumed by the gateway.
        record = SimpleNamespace(n_equations=11, n_elems=mesh.n_cells, e_energy=None, extra={}, wall_s=0.01)  # Build a no-reference repository-shaped solve record.
        self.records.append(record)  # Match FemRunner counted-record behavior.
        return post, record  # Return the documented post-and-record tuple.

class _FrozenPartition:  # Expose the shared frozen-partition API used by the gateway.
    names = ("inspection_opening", "wheel_load")  # Define a stable semantic ordering.
    def assign(self, mesh: Mesh) -> np.ndarray:  # Assign one tetrahedron to each frozen semantic region.
        return np.asarray([0, 1], dtype=int)  # Return one label per element.
    def adjacency_matrix(self, mesh: Mesh, labels: np.ndarray) -> np.ndarray:  # Return the preregistered shared adjacency graph.
        assert labels.tolist() == [0, 1]  # Verify the gateway preserves frozen label ordering.
        return np.asarray([[0.0, 0.25], [0.75, 0.0]], dtype=float)  # Return a deliberately distinctive normalized graph.
    def save(self, path: str | Path) -> None:  # Mimic cached semantic-partition persistence.
        Path(path).write_text("{}", encoding="utf-8")  # Materialize the expected audit artifact.

def test_runtime_adapter_uses_uniform_mesh_stage_and_optional_reference(monkeypatch) -> None:  # Cover the three FemRunner integration defects together.
    problem = make_box_girder_diaphragm()  # Build the canonical root bridge Problem without invoking geometry generation.
    mesh = _mesh()  # Build a concrete repository mesh sentinel.
    runner = _Runner(problem, mesh)  # Construct the repository-shaped runner.
    seen: list[object] = []  # Record the problem passed to the common initial-mesh helper.
    def fake_initial_mesh(candidate: object) -> Mesh:  # Replace only native Gmsh generation in this unit test.
        seen.append(candidate)  # Preserve the exact problem identity.
        return mesh  # Return the concrete synthetic uniform-mesh stand-in.
    monkeypatch.setattr(pipeline_module, "initial_mesh", fake_initial_mesh)  # Redirect the imported repository boundary.
    adapter = _RuntimeAdapter(runner)  # Construct the corrected runtime adapter.
    assert adapter.initial_mesh() is mesh  # Require the shared uniform helper rather than an invalid field-free generate_mesh call.
    assert seen == [problem]  # Require the canonical root problem to reach that helper unchanged.
    assert adapter.ensure_reference(False) is None  # Permit a no-reference smoke without triggering a solve.
    assert runner.reference_calls == 0  # Prove the expensive reference boundary was not called.
    adapter.ensure_reference(True)  # Exercise the explicit reference-enabled path.
    assert runner.reference_calls == 1  # Require one reference request only on opt-in.
    adapter.solve(mesh, "world_model_vla", 3)  # Execute the exact repository solve signature.
    assert runner.solve_calls[-1]["stage"] == "cycle3"  # Require the previously missing mandatory stage label.

def test_gateway_uses_root_constraint_post_sizefield_and_frozen_graph() -> None:  # Cover all corrected root repository field names and constructors.
    problem = make_box_girder_diaphragm()  # Build the canonical root bridge Problem.
    mesh = _mesh()  # Build the concrete repository mesh.
    gateway = MCPToolGateway()  # Use the unchanged action-policy configuration.
    assert gateway.estimate_equations(problem, mesh) == 11  # Count node_predicate-constrained active displacement equations exactly.
    stress = gateway._stress(SimpleNamespace(vm_elem=np.asarray([5.0, 9.0], dtype=float)), mesh.n_cells)  # Read the actual PostState.vm_elem field.
    np.testing.assert_allclose(stress, np.asarray([5.0, 9.0], dtype=float))  # Preserve both elementwise von Mises values.
    target = np.asarray([2.0, 80.0, 24.0, 20.0, 18.0], dtype=float)  # Include values outside both repository size bounds.
    field = gateway._field(mesh, target, problem)  # Construct the exact NodalSizeField(mesh, target, gradation, bounds) API.
    assert field._tree.n == mesh.n_nodes  # Require interpolation to be defined on the concrete source mesh rather than raw points.
    assert float(np.min(field._h)) >= problem.h_min  # Enforce the repository lower size bound.
    assert float(np.max(field._h)) <= problem.h0  # Enforce the repository upper size bound.
    post = SimpleNamespace(mesh=mesh, vm_elem=np.asarray([5.0, 9.0], dtype=float))  # Build a repository-shaped post state.
    record = SimpleNamespace(n_equations=11)  # Supply measured active-equation evidence.
    observation = gateway.observe_solve(problem, _FrozenPartition(), post, record, np.asarray([1.0, 0.5], dtype=float), None, 0)  # Build a complete measured world observation.
    np.testing.assert_allclose(observation.state.adjacency, np.asarray([[0.0, 0.25], [0.75, 0.0]], dtype=float))  # Prefer the frozen shared graph over reconstructed mesh topology.

def test_gateway_certifies_complete_post_gradation_dorfler_dominance(monkeypatch) -> None:  # Verify the exact compiled field passed to Gmsh remains nodewise Dörfler-dominant.
    problem = make_box_girder_diaphragm()  # Build the canonical root bridge problem and size bounds.
    mesh = _mesh()  # Build one concrete source mesh with a nontrivial edge graph.
    gateway = MCPToolGateway()  # Use the fixed gradation-one deterministic compiler.
    post = SimpleNamespace(mesh=mesh, vm_elem=np.asarray([5.0, 9.0], dtype=float))  # Build the repository post fields required by observation.
    record = SimpleNamespace(n_equations=11)  # Supply exact active-equation evidence for the source solve.
    observation = gateway.observe_solve(problem, _FrozenPartition(), post, record, np.asarray([1.0, 0.5], dtype=float), None, 0)  # Build a complete shared-partition world state.
    monkeypatch.setattr(tool_gateway_module, "generate_mesh", lambda candidate_problem, field: mesh)  # Replace only native remeshing while retaining the exact compiled field object.
    base_target = gateway._base_target(observation)  # Reconstruct the exact Dörfler raw target for independent audit.
    world_target = gateway._world_target(observation, RegionAction((1, 0)), base_target)  # Reconstruct one legal proactive raw target.
    _base_field, base_compiled, base_hash = gateway._compiled_field(mesh, base_target, problem)  # Compile and hash the Dörfler field independently.
    _world_field, world_compiled, world_hash = gateway._compiled_field(mesh, world_target, problem)  # Compile and hash the proactive field through the identical path.
    materialized = gateway.materialize_action(observation, RegionAction((1, 0)), 1000)  # Exercise exact certification with solve-free fake remeshing.
    certificate = materialized.certificate  # Read the immutable structural receipt.
    assert certificate.schema_version == "wmvla.mcp-tool.v2"  # Require the complete compiled-field schema.
    assert certificate.base_compiled_field_sha256 == base_hash and certificate.world_compiled_field_sha256 == world_hash  # Bind both exact compiled fields used for candidate generation.
    assert len(base_hash) == 64 and len(world_hash) == 64 and base_hash != world_hash  # Require full SHA-256 identities and a genuinely different proactive field.
    assert certificate.compiled_field_node_count == mesh.n_nodes and certificate.compiled_field_gradation == 1.0  # Bind the common source-node domain and identical gradation.
    assert certificate.compiled_dorfler_included and certificate.compiled_max_dorfler_violation == 0.0  # Require the explicit post-compilation structural gate.
    assert np.all(world_compiled <= base_compiled + 1.0e-12)  # Independently verify h_WM_compiled is never coarser than h_D_compiled.

def test_gateway_rejects_post_gradation_dominance_violation_before_world_gmsh(monkeypatch) -> None:  # Prove compiled dominance is a hard pre-meshing gate rather than metadata only.
    problem = make_box_girder_diaphragm()  # Build the canonical problem and size bounds.
    mesh = _mesh()  # Build one concrete source mesh.
    gateway = MCPToolGateway()  # Construct the exact deterministic compiler.
    post = SimpleNamespace(mesh=mesh, vm_elem=np.asarray([5.0, 9.0], dtype=float))  # Build repository-shaped post fields.
    record = SimpleNamespace(n_equations=11)  # Supply source active-equation evidence.
    observation = gateway.observe_solve(problem, _FrozenPartition(), post, record, np.asarray([1.0, 0.5], dtype=float), None, 0)  # Build the measured shared-partition state.
    original = gateway._compiled_field  # Retain the real clipping, smoothing, and hashing implementation.
    calls = 0  # Count base and proactive field compilations.
    def corrupted_compile(candidate_mesh: Mesh, target: np.ndarray, candidate_problem: object) -> tuple[object, np.ndarray, str]:  # Inject a synthetic post-gradation implementation defect only in the returned audit values.
        nonlocal calls  # Advance one local call counter without global state.
        calls += 1  # Identify the base or world compilation.
        field, values, digest = original(candidate_mesh, target, candidate_problem)  # Compile the exact real field first.
        if calls == 2:  # Corrupt only the proactive compiled values after raw dominance passed.
            values = values + 1.0e6  # Force unmistakable nodewise coarsening relative to Dörfler.
        return field, values, digest  # Return the controlled test evidence.
    monkeypatch.setattr(gateway, "_compiled_field", corrupted_compile)  # Replace only the compiled-value audit boundary.
    generated: list[object] = []  # Count exact candidate-generation calls without native Gmsh.
    def fake_generate(candidate_problem: object, field: object) -> Mesh:  # Return the source mesh while retaining call evidence.
        generated.append(field)  # Record the exact compiled field handed to the mesher boundary.
        return mesh  # Return a valid concrete candidate stand-in.
    monkeypatch.setattr(tool_gateway_module, "generate_mesh", fake_generate)  # Replace native remeshing while preserving field-call accounting.
    with pytest.raises(RuntimeError, match="compiled world field"):  # Require an explicit structural contract failure.
        gateway.materialize_action(observation, RegionAction((1, 0)), 1000)  # Attempt one proactive action whose returned compiled field violates dominance.
    assert calls == 2 and len(generated) == 1  # Require rejection after world compilation but before world Gmsh generation.

def test_no_reference_pipeline_handles_none_metric_and_reports_timing(monkeypatch, tmp_path: Path) -> None:  # Exercise a complete one-solve no-reference trajectory without native subprocesses.
    problem = make_box_girder_diaphragm()  # Build the canonical root bridge Problem.
    mesh = _mesh()  # Build the concrete repository mesh.
    runner = _Runner(problem, mesh)  # Construct the exact runner-shaped boundary.
    runner.workdir = tmp_path  # Isolate all audit artifacts in the test directory.
    monkeypatch.setattr(pipeline_module, "initial_mesh", lambda candidate: mesh)  # Replace only native Gmsh generation for the one-solve unit path.
    monkeypatch.setattr(pipeline_module, "zz_indicator", lambda candidate, post: np.asarray([1.0, 0.5], dtype=float))  # Replace only numerical estimator work with a positive exact-shape fixture.
    result = run_world_model_vla(runner, partition=_FrozenPartition(), config=WorldVLAConfig(max_solves=1, n_equation_cap=100, artifact_dir=str(tmp_path), require_reference=False))  # Execute the complete reference-free runtime path.
    assert runner.reference_calls == 0  # Prove the configuration suppresses reference generation.
    assert result.best_index == 0  # Select the sole real solve using the ZZ fallback metric.
    assert _record_metric(result.records[0], 1.5) == np.sqrt(1.5)  # Treat e_energy=None as an explicit no-reference sentinel.
    assert set(("visual_partition", "world_model_planning", "parameter_tools", "gmsh_remeshing", "calculix")) <= set(result.timing_s)  # Report every predeclared online timing component.
    assert result.timing_s["calculix"] == 0.01  # Preserve the counted solver wall time exactly.
    assert (tmp_path / "world_vla_manifest.json").is_file()  # Persist the complete result and timing manifest.
