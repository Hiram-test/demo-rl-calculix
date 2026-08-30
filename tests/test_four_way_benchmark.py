"""Test the unified four-way runner's pure protocol and failure boundaries."""  # Describe this no-solver test module.
from __future__ import annotations  # Postpone annotation evaluation consistently with production code.
from pathlib import Path  # Build isolated job paths and typed native-failure receipts.
from types import SimpleNamespace  # Construct minimal mesh and record fixtures without native solvers.
import json  # Persist minimal solve-free frozen-config fixtures.
import numpy as np  # Verify non-scalar serialization and deterministic full mesh identities.
import pytest  # Assert finite posthoc metrics and exact protocol behavior.
from visionamr.calculix import CalculiXExecutionError  # Construct the only retained native solver failure type.
from visionamr.bridge_case_manifest import write_case_manifest  # Generate an authentic manifest for solve-free blind filter refusals.
from visionamr.experiment import FemRunner, Reference, SolveRecord  # Reuse the production runner, posthoc denominator, and raw record contracts.
from visionamr.vla.four_way_benchmark import ALL_METHODS, BUDGETS, BenchmarkRequest, ExecutionJob, FrozenInputError, ReceiptFemRunner, _attach_posthoc_reference_metrics, _campaign_root_for_job, _full_mesh_sha256, _json_safe, _numerical_failure_payload, _wm_prefix_safety, _write_final_state, build_diagnostic_plan, build_execution_jobs, build_plan, derive_prefix_rows, run_benchmark  # Exercise the unified runner's pure boundaries directly.

def _record(index: int, equations: int, energy: float | None = None, qoi: float | None = None) -> SolveRecord:  # Build one transparent successful raw solve fixture.
    return SolveRecord(method="world_model_vla", stage=f"cycle{index}", solve_index=index, n_nodes=10, n_elems=5, n_equations=equations, U_total=80.0 + index, qoi=2.0 + 0.1 * index, wall_s=0.2, h_min=0.1, h_max=0.5, e_energy=energy, e_qoi=qoi)  # Supply every production field with finite values.

def test_development_jobs_sort_manifest_cases_and_keep_registered_grid(tmp_path: Path) -> None:  # Verify controlled shards preserve manifest ordering and never write below test.
    manifest = {"cases": [{"case_id": "z", "split": "train", "geometry_hash": "b" * 64}, {"case_id": "a", "split": "train", "geometry_hash": "a" * 64}]}  # Build two deliberately reversed development cases.
    request = BenchmarkRequest(root=tmp_path, manifest_path=tmp_path / "manifest.json", frozen_config_path=tmp_path / "config.json", split="train", methods=("local_prediction",), budgets=(60000,), dry_run=True)  # Select one solve-free controlled shard method and budget.
    jobs = build_execution_jobs(request, manifest)  # Construct the deterministic independent trajectory plan.
    assert [job.case_id for job in jobs] == ["a", "z"]  # Require lexicographic manifest case order independent of input serialization.
    assert all(job.output_dir.parts[-5:-3] == ("development", "train") for job in jobs)  # Keep development evidence outside the irreversible blind tree.
    assert _campaign_root_for_job(jobs[0]) == tmp_path  # Recover frozen model and partition paths above the extra development/split components.
    blind_job = ExecutionJob("case", "test", "c" * 64, 60000, "world_model_vla", tmp_path / "test" / "case" / "60000" / "world_model_vla")  # Build the shorter blind output layout without opening it.
    assert _campaign_root_for_job(blind_job) == tmp_path  # Recover the same campaign root under the one-shot blind layout.

@pytest.mark.parametrize(("overrides", "message"), [({"case_ids": ("forbidden",)}, "forbids --case-id"), ({"methods": ("world_model_vla",)}, "forbids method subsets"), ({"budgets": (60000,)}, "forbids budget subsets"), ({"resume": True}, "forbids --resume")])  # Cover every prohibited blind subset or continuation control.
def test_blind_plan_rejects_all_filters_and_resume_before_freeze_access(tmp_path: Path, overrides: dict[str, object], message: str) -> None:  # Verify the one-shot test unit cannot be split even by solve-free direct callers.
    manifest_path, _sidecar, _digest = write_case_manifest(tmp_path / "protocol")  # Persist a valid checksummed 24/8/16 manifest before request validation.
    config_path = tmp_path / "protocol" / "frozen_config.json"  # Resolve the minimal config read before subset refusal.
    config_path.write_text(json.dumps({"common_gradation": 1.0, "allow_unqualified_references": False}), encoding="utf-8")  # Satisfy only format validation without constructing a fake freeze.
    values = {"root": tmp_path, "manifest_path": manifest_path, "frozen_config_path": config_path, "split": "test", **overrides}  # Apply exactly one prohibited test control per parameterized case.
    with pytest.raises(FrozenInputError, match=message):  # Require refusal before canonical freeze or any reference/test access.
        build_plan(BenchmarkRequest(**values))  # Exercise only the solve-free request boundary and never open a blind output tree.
    assert not (tmp_path / "test").exists()  # Prove every refusal occurred before TEST_STARTED or raw directory creation.

def test_true_prefix_delivery_uses_feasible_records_and_propagates_native_failure() -> None:  # Verify actual prefixes are derived once without rerunning or hiding a failed attempt.
    job = ExecutionJob("case", "test", "a" * 64, 30000, "world_model_vla", Path("unused"))  # Build one independent public-budget trajectory identity.
    records = [_record(1, 20000, 0.8, 0.7), _record(2, 29000, 0.6, 0.65), _record(3, 31000, 0.4, 0.3)]  # Provide two feasible solves and one better-looking over-budget solve.
    rows = derive_prefix_rows(job, records, True, None, {})  # Derive all K values from the single actual trajectory.
    assert rows[0]["energy_error"] == pytest.approx(0.6) and rows[-1]["energy_error"] == pytest.approx(0.6)  # Exclude the over-budget error from every later best prefix.
    assert rows[-1]["budget_violation"] is True and rows[-1]["hold_last_after_stop"] is True  # Disclose the actual overshoot and held later prefix.
    failed = derive_prefix_rows(job, records[:2], False, {"failure_at_solve": 3}, {})  # Place a typed native failure at the third attempted solve.
    assert failed[0]["energy_ok"] is True and failed[1]["energy_ok"] is False  # Preserve K=2 while invalidating K>=3.

def test_reference_metrics_are_attached_posthoc_with_explicit_nononline_provenance() -> None:  # Verify sealed truth scoring does not require binding FemRunner.reference.
    records = [_record(1, 20000), _record(2, 28000)]  # Build an already completed online trajectory with no reference errors.
    reference = Reference(U_total=100.0, qoi=2.0, n_equations=200000, n_elems=100000, h_ref=0.01)  # Build one finite sealed Reference B denominator.
    _attach_posthoc_reference_metrics(records, reference, {"reference_b_sha256": "a" * 64})  # Score only after the raw actions and meshes are fixed.
    assert records[0].e_energy == pytest.approx(np.sqrt(19.0 / 100.0)) and records[0].e_qoi == pytest.approx(0.05)  # Apply the exact common formulas independently.
    assert all(record.extra["posthoc_reference_b"]["used_online"] is False for record in records)  # Mark truth as posthoc-only on every raw solve.

def test_only_typed_native_failures_are_finite_and_arrays_serialize_completely(tmp_path: Path) -> None:  # Verify integrity and programming errors remain campaign-fatal by default.
    native = CalculiXExecutionError("failed", returncode=7, wall_s=1.5, log_path=tmp_path / "model.log", workdir=tmp_path)  # Build a typed native failure with complete receipt fields.
    payload = _numerical_failure_payload(native)  # Classify through the production retained-failure boundary.
    assert payload is not None and payload["category"] == "calculix_numerical" and payload["calculix_returncode"] == 7  # Preserve exact native failure evidence.
    assert _numerical_failure_payload(ValueError("schema")) is None and _numerical_failure_payload(RuntimeError("model API")) is None  # Force configuration and API defects to propagate.
    assert _json_safe(np.asarray([[1.0, 2.0], [3.0, 4.0]])) == [[1.0, 2.0], [3.0, 4.0]]  # Serialize non-scalar arrays without invalid item() calls or data loss.

def test_first_solve_calculix_failure_persists_attempted_mesh_and_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # Regress the exact first-native-failure final-state evidence boundary.
    mesh = SimpleNamespace(nodes=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), cells=np.asarray([[0, 1, 2]]), n_nodes=3, n_cells=1)  # Build the exact two-dimensional mesh submitted to the fake first solve.
    partition = SimpleNamespace(assign=lambda attempted_mesh: np.asarray([7] * int(attempted_mesh.n_cells)))  # Provide one available solve-free semantic assignment for the attempted mesh.
    native = CalculiXExecutionError("first solve failed", returncode=11, wall_s=0.25, log_path=tmp_path / "model.log", workdir=tmp_path / "solve")  # Construct a typed first-solve native failure with complete provenance.
    def fail_first_solve(fake_runner: FemRunner, _mesh: object, *, method: str, stage: str, extra: dict | None = None) -> object:  # Replace only the underlying real solver call for this unit regression.
        fake_runner._counter += 1  # Match FemRunner's honest attempt counter advancement before native execution.
        raise native  # Fail before FemRunner can publish last_mesh or a physical post-state.
    monkeypatch.setattr(FemRunner, "solve_mesh", fail_first_solve)  # Route ReceiptFemRunner's super call through the deterministic fake failure.
    runner = ReceiptFemRunner(SimpleNamespace(dim=2), tmp_path / "job")  # Create isolated receipt accounting without invoking Gmsh or CalculiX.
    runner.final_partition = partition  # Make the frozen semantic partition available before the mesh submission.
    with pytest.raises(CalculiXExecutionError, match="first solve failed"):  # Require the typed native error to propagate unchanged.
        runner.solve_mesh(mesh, method="world_model_vla", stage="common_probe")  # Submit the first attempted mesh through the production receipt wrapper.
    assert runner.last_mesh is None and runner.last_post is None  # Confirm no successful physical state was fabricated by the failure path.
    assert runner.last_attempted_mesh is mesh and runner.last_attempted_partition is partition  # Preserve the exact attempted mesh and contemporaneous partition.
    assert len(runner.mesh_receipts) == 1 and runner.mesh_receipts[0]["success"] is False  # Retain the failed solve's immutable mesh receipt independently.
    state_path = tmp_path / "final_state.npz"  # Select an isolated final-state artifact path.
    state_receipt = _write_final_state(state_path, runner)  # Publish solve-free failed-attempt evidence through the production writer.
    assert state_receipt["available"] is False and state_receipt["source"] == "failed_attempt"  # Distinguish mesh evidence from an unavailable successful post-state.
    assert state_receipt["mesh_sha256"] == _full_mesh_sha256(mesh) and state_receipt["eta2_count"] == 0  # Bind status to the full attempted mesh and an honestly empty estimator.
    with np.load(state_path, allow_pickle=False) as state:  # Inspect the compact artifact without permitting object deserialization.
        assert state["source"].item() == "failed_attempt" and not bool(state["available"].item())  # Repeat provenance and physical-state availability inside the NPZ itself.
        assert np.array_equal(state["nodes"], mesh.nodes) and np.array_equal(state["cells"], mesh.cells)  # Preserve every submitted coordinate and connectivity entry exactly.
        assert state["eta2"].size == 0 and state["region_labels"].tolist() == [7]  # Keep eta2 empty while retaining the available attempted-mesh partition.

def test_execution_summary_treats_typed_failure_as_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # Regress invocation completion independently from per-method scientific success.
    import visionamr.vla.four_way_benchmark as benchmark_module  # Patch only the local orchestration seams while retaining the production summary writer.
    case = {"case_id": "train_case", "split": "train", "geometry_hash": "a" * 64}  # Provide the sole manifest identity needed by the fake executor.
    jobs = [ExecutionJob("train_case", "train", "a" * 64, 60000, method, tmp_path / "jobs" / method) for method in ("world_model_vla", "local_prediction", "supervised")]  # Register three pure fake jobs spanning all terminal classes.
    planned_outcomes = iter(("completed", "failed", "skipped_completed"))  # Include one retained typed-native failure between two successful terminal outcomes.
    def fake_execute(_request: BenchmarkRequest, job: ExecutionJob, _case: dict, _config: dict) -> dict:  # Return deterministic terminal rows without any mesh generation or solver call.
        status = next(planned_outcomes)  # Consume exactly one registered fake status per planned job.
        return {"job": {"case_id": job.case_id, "equation_budget": job.budget, "method": job.method}, "status": status, "completed": status != "failed"}  # Mimic execute_job's concise terminal contract.
    monkeypatch.setattr(benchmark_module, "build_plan", lambda _request: ({"schema": "fake-plan"}, {}, jobs))  # Bypass filesystem protocol preflight for this isolated development-only summary test.
    monkeypatch.setattr(benchmark_module, "load_case_manifest", lambda _path, verify_checksum=True: {"cases": [case]})  # Supply the one fake development manifest record without reading a formal campaign.
    monkeypatch.setattr(benchmark_module, "execute_job", fake_execute)  # Replace all trajectory execution with the finite fake outcome stream.
    request = BenchmarkRequest(root=tmp_path, manifest_path=tmp_path / "manifest.json", frozen_config_path=tmp_path / "config.json", split="train", development_run=True)  # Keep every generated summary under an isolated non-test root.
    summary = run_benchmark(request)  # Exercise the production aggregation and atomic summary persistence only.
    assert summary["all_jobs_completed"] is True and summary["terminal_job_count"] == 3  # Treat the typed failure as a durable terminal job rather than an interrupted invocation.
    assert summary["completed_job_count"] == 2 and summary["successful_job_count"] == 2 and summary["failed_job_count"] == 1  # Preserve completed-success compatibility while reporting retained failures explicitly.
    assert Path(summary["summary_path"]).is_file() and not (tmp_path / "test").exists()  # Prove the regression wrote only development evidence and never opened a blind-test tree.

def test_full_mesh_hash_and_v2_dorfler_certificate_are_strict() -> None:  # Verify receipts bind complete arrays and proactive credit requires compiled-field v2 evidence.
    mesh = SimpleNamespace(nodes=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), cells=np.asarray([[0, 1]]))  # Build a minimal deterministic mesh fixture.
    changed = SimpleNamespace(nodes=np.asarray([[0.0, 0.0, 0.0], [1.0 + 1.0e-12, 0.0, 0.0]]), cells=np.asarray([[0, 1]]))  # Change one full-precision coordinate byte.
    assert len(_full_mesh_sha256(mesh)) == 64 and _full_mesh_sha256(mesh) != _full_mesh_sha256(changed)  # Require full SHA-256 sensitivity rather than a truncated display identity.
    certificate = {"schema_version": "wmvla.mcp-tool.v2", "accepted": True, "source": "world_model", "executed_action": [1], "target_sha256": "c" * 64, "base_target_included": True, "no_coarsening": True, "compiled_dorfler_included": True, "compiled_max_dorfler_violation": 0.0, "compiled_field_gradation": 1.0, "base_compiled_field_sha256": "a" * 64, "world_compiled_field_sha256": "b" * 64}  # Build a complete executed proactive certificate.
    proactive, dominance, _fallback = _wm_prefix_safety({"certificates": [certificate]}, 2)  # Evaluate the first materialized action at K=2.
    assert proactive is True and dominance[0]["passed"] is True  # Grant proactive credit only to the complete compiled-field certificate.
    certificate["compiled_field_gradation"] = 0.9  # Reintroduce a historical inconsistent smoothing contract.
    assert _wm_prefix_safety({"certificates": [certificate]}, 2)[0] is False  # Reject proactive credit when compiled gradation is not the frozen 1.0.

def test_diagnostic_plan_registers_all_variants_and_random_seeds() -> None:  # Verify the primary runner exposes the isolated mandatory diagnostic adapter before results exist.
    plan = build_diagnostic_plan(["case_b", "case_a"])  # Build a two-case solve-free diagnostic preregistration plan.
    assert plan["case_order"] == ["case_a", "case_b"] and len(plan["variants"]) == 6  # Preserve sorted cases and the complete six-variant vocabulary.
    random_jobs = [job for job in plan["jobs"] if job["variant"] == "random_safe_extra"]  # Isolate the five-seed safe-random controls.
    assert len(random_jobs) == 10 and {job["seed"] for job in random_jobs} == {20260911, 20260912, 20260913, 20260914, 20260915}  # Require all five seeds for every case without selection.
    assert all(job["mode"] == "reuse_primary_wm_full" for job in plan["jobs"] if job["variant"] == "wm_full")  # Prevent hidden duplicate primary WM solves.
