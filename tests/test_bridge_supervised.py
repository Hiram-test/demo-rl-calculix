"""Focused contract tests for the frozen four-way supervised baseline."""  # State the suite's exact scientific scope.
from __future__ import annotations  # Postpone annotation evaluation consistently with the implementation.
from dataclasses import dataclass  # Build compact synthetic constraint and solve-record contracts.
import json  # Verify strict finite selected-checkpoint evidence.
from pathlib import Path  # Locate the solve-free training CLI for an integration dry run.
import sys  # Reuse the active test interpreter for the solve-free CLI subprocess.
from types import SimpleNamespace  # Build minimal repository-compatible mesh, problem, model, post, and record doubles.
import numpy as np  # Construct deterministic synthetic coordinates, connectivity, predictions, and indicators.
import pytest  # Assert explicit rejection of blind splits and malformed candidate pools.
from visionamr.baselines import bridge_supervised as supervised  # Import the complete frozen helper surface for isolated tests.
from visionamr.bridge_case_manifest import build_case_manifest, write_case_manifest  # Build the real deterministic split and checksum artifacts.

@dataclass(frozen=True)  # Match the repository's immutable boundary-condition shape in a tiny test double.
class _Constraint:  # Represent one named node-selection and component restriction.
    node_predicate: object  # Store the vectorized nodal predicate callable.
    dofs: tuple[int, ...]  # Store one-based constrained displacement components.
    name: str  # Store the audit-friendly boundary-condition name.

def _mesh() -> SimpleNamespace:  # Build one small tetrahedral mesh exposing every supervised resource attribute.
    nodes = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)  # Define four noncoplanar candidate nodes.
    cells = np.asarray([[0, 1, 2, 3]], dtype=int)  # Define one linear tetrahedron.
    return SimpleNamespace(nodes=nodes, cells=cells, n_nodes=4, n_cells=1, dim=3, measures=np.asarray([1.0 / 6.0]), cell_sizes=np.asarray([1.0]))  # Return the minimal mesh contract used by deployment and preflight.

def _validation_rows(seed: int, *, energy: float = 0.2, qoi: float = 0.3, failed_index: int | None = None, violation_index: int | None = None) -> list[dict]:  # Build one exact eight-case by three-budget validation grid.
    rows: list[dict] = []  # Accumulate every deterministic candidate operating point.
    index = 0  # Track a stable scalar point index for injected failures.
    for case_number in range(8):  # Cover exactly the frozen validation-case cardinality.
        for budget in supervised.VALIDATION_BUDGETS:  # Cover every frozen deployment scaling budget.
            failed = failed_index == index  # Select at most one explicit retained numerical failure.
            rows.append({"seed": seed, "case_id": f"validation_{case_number:02d}", "equation_budget": budget, "status": "failed" if failed else "ok", "energy_error": None if failed else energy, "qoi_error": None if failed else qoi, "budget_violation": violation_index == index})  # Preserve finite successful metrics and JSON-null failures.
            index += 1  # Advance to the next unique case-budget point.
    return rows  # Return the complete candidate validation evidence.

def test_manifest_boundary_returns_only_train_or_validation_cases() -> None:  # Prove the supervised pipeline cannot select the blind split.
    manifest = build_case_manifest()  # Build the exact frozen 48-case design in memory.
    train = supervised.cases_for_split(manifest, "train")  # Select only the 24 authorized expert-label cases.
    validation = supervised.cases_for_split(manifest, "validation")  # Select only the eight authorized checkpoint-selection cases.
    assert len(train) == 24 and {case["split"] for case in train} == {"train"}  # Require an uncontaminated expert split.
    assert len(validation) == 8 and {case["split"] for case in validation} == {"validation"}  # Require an uncontaminated checkpoint split.
    with pytest.raises(ValueError, match="only train or validation"):  # Require explicit rejection before any blind-case reconstruction.
        supervised.cases_for_split(manifest, "test")  # Attempt to cross the training module's split boundary.

def test_exact_active_equation_preflight_deduplicates_constraints() -> None:  # Verify candidate resource checks count unique active displacement DOFs exactly.
    mesh = _mesh()  # Build the synthetic four-node tetrahedron.
    at_origin = lambda points: np.all(np.isclose(points, 0.0), axis=1)  # Select only node zero for a three-component pin.
    on_x_zero = lambda points: np.isclose(points[:, 0], 0.0)  # Select nodes zero, two, and three for a roller component.
    problem = SimpleNamespace(dim=3, constraints=[_Constraint(at_origin, (1, 2, 3), "pin"), _Constraint(on_x_zero, (3,), "roller")])  # Define overlapping constraints whose shared uz must be counted once.
    assert supervised.estimate_active_equations(problem, mesh) == 7  # Subtract five unique fixed DOFs from twelve total candidate DOFs.

def test_validation_score_uses_finite_penalty_and_declared_lexicographic_order() -> None:  # Prove failures dominate errors and every selection value remains finite JSON.
    clean = supervised.validation_score(_validation_rows(20260831, energy=0.4, qoi=0.5))  # Score a complete valid but less accurate checkpoint.
    accurate_with_failure = supervised.validation_score(_validation_rows(20260832, energy=0.01, qoi=0.01, failed_index=0))  # Score a numerically attractive checkpoint with one retained failure.
    cleaner_but_violating = supervised.validation_score(_validation_rows(20260833, energy=0.3, qoi=0.5, violation_index=0))  # Score a valid checkpoint whose energy beats the clean seed despite one budget breach.
    selected = supervised.select_validation_checkpoint([accurate_with_failure, clean, cleaner_but_violating])  # Apply failure, energy, QoI, violation, then seed ordering exactly.
    assert accurate_with_failure["failure_point_count"] == 1 and np.isfinite(accurate_with_failure["energy_error_log_mean"])  # Retain the failure with the fixed finite error penalty.
    assert selected["seed"] == 20260833  # Let lower energy precede the later budget-violation tie-break after both have zero failures.
    json.dumps({"scores": [clean, accurate_with_failure, cleaner_but_violating], "selected": selected}, allow_nan=False)  # Require strict standards-compliant freeze evidence.

def test_selection_uses_seed_only_as_final_tie_break() -> None:  # Prove identical candidates select the smallest preregistered seed deterministically.
    scores = [supervised.validation_score(_validation_rows(seed)) for seed in supervised.NETWORK_SEEDS]  # Give all three candidates identical physical validation results.
    assert supervised.select_validation_checkpoint(list(reversed(scores)))["seed"] == 20260831  # Select independently of input order using seed last.

def test_frozen_model_loader_verifies_hash_and_selected_seed(tmp_path: Path) -> None:  # Prove blind deployment cannot load changed bytes or an unregistered checkpoint seed.
    model = supervised.SizeMLP(supervised.SupervisedConfig(seed=20260831))  # Allocate one valid existing architecture without fitting it.
    model_path = tmp_path / "selected_model.pt"  # Select a disposable frozen-checkpoint path.
    model.save(model_path)  # Persist a valid weights-only state dictionary.
    expected_sha = supervised.file_sha256(model_path)  # Freeze the exact candidate bytes before deployment loading.
    loaded, receipt = supervised.load_frozen_supervised_model(model_path, selected_seed=20260831, expected_sha256=expected_sha)  # Verify and reconstruct the selected model.
    assert receipt["model_sha256"] == expected_sha and loaded.net.training is False  # Require exact identity and explicit evaluation mode.
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):  # Reject any post-freeze byte identity change.
        supervised.load_frozen_supervised_model(model_path, selected_seed=20260831, expected_sha256="0" * 64)  # Supply a deliberately stale frozen digest.
    with pytest.raises(ValueError, match="must be one of"):  # Reject a network outside the exact three-seed candidate pool.
        supervised.load_frozen_supervised_model(model_path, selected_seed=7)  # Attempt an unregistered deployment seed.

def test_deployment_is_reference_optional_and_exactly_two_real_solves(monkeypatch: pytest.MonkeyPatch) -> None:  # Verify common probe, one remesh, and hold-last without a hidden reference solve.
    mesh = _mesh()  # Reuse one tiny candidate for both counted synthetic solves.
    problem = SimpleNamespace(dim=3, h0=2.0, h_min=0.25, constraints=[], bbox=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), diameter=1.0, features=[])  # Build the minimal model-feature problem contract.
    class _Runner:  # Provide only the counted deployment surface used by the supervised helper.
        def __init__(self) -> None:  # Initialize isolated solve evidence.
            self.problem = problem  # Bind the synthetic bridge-like problem.
            self.records: list[SimpleNamespace] = []  # Count every real synthetic solve explicitly.
            self.reference_calls = 0  # Detect any forbidden implicit reference construction.
        def ensure_reference(self) -> None:  # Record reference construction attempts without performing work.
            self.reference_calls += 1  # Count a hidden reference request as a test failure signal.
        def solve_mesh(self, candidate: object, *, method: str, stage: str, extra: dict) -> tuple[SimpleNamespace, SimpleNamespace]:  # Emulate one counted real solve.
            record = SimpleNamespace(method=method, stage=stage, solve_index=len(self.records) + 1, n_equations=12, n_elems=1, extra=dict(extra), e_energy=0.2, e_qoi=0.3)  # Build the exact fields consumed by deployment.
            self.records.append(record)  # Count this solve before returning its observation.
            return SimpleNamespace(vm_node=np.ones(4)), record  # Return a finite probe/deployed post-state.
    class _Model:  # Provide one deterministic refinement-only network output.
        def predict(self, features: np.ndarray) -> np.ndarray:  # Return one log ratio per common-probe node.
            return np.full(features.shape[0], math_log_half(), dtype=float)  # Predict a uniform half-size target without Torch.
    receipt = supervised.BudgetMesh(mesh=mesh, scale=1.0, estimated_equations=12, equation_budget=100, target_sha256="a" * 64, mesh_sha256="b" * 64, trials=())  # Build a preflight-certified synthetic remesh.
    monkeypatch.setattr(supervised, "initial_mesh", lambda _problem: mesh)  # Remove Gmsh from the unit deployment path.
    monkeypatch.setattr(supervised, "zz_indicator", lambda _problem, _post: np.asarray([1.0]))  # Supply one finite element indicator per solve.
    monkeypatch.setattr(supervised, "node_features", lambda _problem, _mesh, _post, _eta2: np.zeros((_mesh.n_nodes, supervised.N_FEATURES), dtype=np.float32))  # Supply existing-shape network features without stress recovery.
    monkeypatch.setattr(supervised, "preflight_budget_mesh", lambda *_args, **_kwargs: receipt)  # Supply one exact feasible unsolved candidate without Gmsh.
    runner = _Runner()  # Create an isolated counted synthetic deployment.
    deployment = supervised.deploy_bridge_supervised(runner, _Model(), n_eq_budget=100, require_reference=False)  # Execute the reference-free two-solve contract.
    assert deployment.real_solve_count == 2 and deployment.hold_last_after_solve == 2  # Require exact probe-plus-remesh cost and explicit K-greater-than-two semantics.
    assert [record.stage for record in runner.records] == ["probe", "deployed"]  # Require the exact common probe and one predicted remesh stages.
    assert runner.reference_calls == 0  # Prove validation or benchmark runners may inject common Reference B without another solve.

def math_log_half() -> float:  # Return a deterministic physical test prediction without importing math solely for one line.
    return float(np.log(0.5))  # Encode a half-size network prediction in the trained log-ratio space.

def test_training_cli_dry_run_validates_without_creating_training_artifacts(tmp_path: Path) -> None:  # Exercise the actual command's solve-free manifest and split plan.
    protocol_root = tmp_path / "protocol"  # Allocate one disposable manifest directory.
    manifest_path, _checksum_path, _digest = write_case_manifest(protocol_root)  # Persist the exact case manifest and checksum sidecar.
    output = tmp_path / "training"  # Select a nonexistent artifact root that dry-run must not create.
    import subprocess  # Import process execution locally to keep the test module's dependency surface explicit.
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "train_bridge_supervised.py"), "--manifest", str(manifest_path), "--output", str(output), "--dry-run"], check=True, capture_output=True, text=True)  # Launch the real CLI without native or Torch work.
    payload = json.loads(completed.stdout)  # Parse the complete printed training plan.
    assert payload["training_case_count"] == 24 and payload["validation_case_count"] == 8  # Require the exact authorized development split counts.
    assert payload["test_case_count_executed"] == 0 and payload["test_split_accessed_for_training_or_selection"] is False  # Require an explicit anti-leakage declaration.
    assert not output.exists()  # Prove dry-run creates no model, dataset, or solver artifact.

def test_training_cli_dry_run_records_explicit_unqualified_reference_policy(tmp_path: Path) -> None:  # Prove expedited operational references require and preserve both explicit command controls.
    protocol_root = tmp_path / "protocol"  # Allocate one disposable manifest directory without any solver artifacts.
    manifest_path, _checksum_path, _digest = write_case_manifest(protocol_root)  # Persist the exact case manifest and checksum sidecar.
    output = tmp_path / "training"  # Select a nonexistent artifact root that both dry runs must leave untouched.
    import subprocess  # Import process execution locally for the real command contract.
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "train_bridge_supervised.py"), "--manifest", str(manifest_path), "--output", str(output), "--allow-unqualified-references", "--expedited-reference-levels", "2", "--dry-run"]  # Declare the complete exceptional policy rather than relying on implicit truthiness.
    completed = subprocess.run(command, check=True, capture_output=True, text=True)  # Execute the solve-free authorized plan.
    payload = json.loads(completed.stdout)  # Decode the printed immutable plan.
    assert payload["allow_unqualified_references"] is True and payload["expedited_reference_levels"] == 2  # Require exact amended policy disclosure.
    incomplete = subprocess.run(command[:7] + ["--dry-run"], check=False, capture_output=True, text=True)  # Remove the paired depth and attempt an ambiguous partial opt-in.
    assert incomplete.returncode != 0 and "must be specified together" in incomplete.stderr  # Require fail-closed rejection before any artifact creation.
    assert not output.exists()  # Prove both solve-free paths leave the training root untouched.

def test_validation_api_rejects_unfrozen_expedited_depth_before_case_access(tmp_path: Path) -> None:  # Prevent a direct caller from selecting a reference prefix that the reviewed amendment and freeze cannot reproduce.
    with pytest.raises(ValueError, match="amended value 2"):  # Require the exact protected two-level exception rather than any nominally valid ladder prefix.
        supervised.validate_candidate_networks((), (), tmp_path / "output", tmp_path / "references", allow_unqualified_references=True, expedited_reference_levels=3)  # Attempt an unauthorized depth before supplying cases, models, references, meshes, or solves.
