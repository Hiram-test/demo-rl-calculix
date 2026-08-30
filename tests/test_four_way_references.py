"""Exercise the frozen two-level reference ladder without native solver cost."""  # Describe the synthetic protocol tests in this module.
from __future__ import annotations  # Postpone annotation evaluation for stable test collection.

import json  # Import direct ledger inspection for retained-failure assertions.
from pathlib import Path  # Import temporary cache path annotations.
from types import SimpleNamespace  # Import compact synthetic post and solve records.
from typing import Any  # Import heterogeneous fake-runner value annotations.

import numpy as np  # Import deterministic tetrahedron coordinates and connectivity.
import pytest  # Import explicit exception assertions for fail-closed behavior.

from visionamr.bridge_cases import make_box_girder_diaphragm  # Reuse the canonical manifest-family Problem without invoking its geometry builder.
from visionamr.mesher import Mesh  # Reuse the repository mesh identity and realized-size implementation.
from visionamr.vla.four_way_references import LEDGER_FILENAME, REFERENCE_B_FILENAME, UNQUALIFIED_AUTHORIZATION, ReferenceBuildError, ReferenceScheduleConfig, ensure_reference_pair, load_reference_b, reference_case_dir, verify_reference_cache, verify_reference_failure_evidence  # Import strict and explicit operational reference APIs plus the exact authorization token.


class SyntheticMeshFactory:  # Generate deterministic one-tetra meshes while recording analytical-field inputs.
    def __init__(self) -> None:  # Initialize an empty invocation ledger for independence checks.
        self.calls: list[dict[str, Any]] = []  # Store only immutable scalar receipts rather than any method mesh.

    def __call__(self, problem: Any, size_fn: Any, *, model_name: str, h_floor: float) -> Mesh:  # Match the production Gmsh mesh-factory signature exactly.
        sample_point = problem.singular_segments[0][0]  # Select a canonical bridge singular-line endpoint where local grading is strongest.
        field_value = float(size_fn(*sample_point))  # Evaluate the independently constructed analytical reference field.
        scale = float(h_floor)  # Use the registered local floor to make each synthetic level's mesh SHA distinct and reproducible.
        nodes = np.asarray([[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]], dtype=float)  # Build a positive-volume tetrahedron with deterministic physical scale.
        cells = np.asarray([[0, 1, 2, 3]], dtype=np.int64)  # Connect the four nodes into one valid linear tetrahedron.
        self.calls.append({"model_name": model_name, "h_floor": scale, "field_at_singular_line": field_value})  # Preserve the only inputs used by this synthetic mesh generator.
        return Mesh(nodes=nodes, cells=cells, dim=3)  # Return a repository-native mesh with a real SHA and cell sizes.


class SyntheticRunner:  # Emulate FemRunner's non-counting reference hook with prescribed independent solver outputs and native evidence.
    def __init__(self, problem: Any, values: list[Any], workdir: Path) -> None:  # Bind one case, an ordered result-or-exception sequence, and an isolated runner tree.
        self.problem = problem  # Preserve the full problem identity used by the cross-case guard.
        self.values = list(values)  # Copy the finite synthetic solve schedule to prevent caller mutation.
        self.workdir = Path(workdir)  # Expose the same portable runner-work contract used for native log discovery.
        self._counter = 0  # Match FemRunner's online solve counter without incrementing for references.
        self.reference = None  # Start without any shared reference so successful binding is observable.
        self.calls = 0  # Count actual synthetic solver invocations independently from `_counter`.

    def _solve(self, mesh: Mesh, *, method: str, stage: str, count: bool, extra: dict[str, Any]) -> tuple[Any, Any]:  # Match FemRunner's complete private reference solve signature.
        del mesh  # Confirm the fake solver does not derive its prescribed values from a compared method mesh.
        assert method == "reference" and stage.startswith("ref_l") and count is False  # Require the production non-online reference path.
        assert extra["mesh_source"] == "preregistered_analytical_reference_field"  # Require explicit method-mesh independence at the solve boundary.
        jobdir = self.workdir / "solves" / f"reference_000_{stage}"  # Reproduce FemRunner's exact non-counting per-level native job layout.
        jobdir.mkdir(parents=True, exist_ok=True)  # Create the isolated synthetic native evidence directory.
        (jobdir / "model.inp").write_text(f"synthetic input {stage}\n", encoding="utf-8")  # Leave a deterministic already-generated input for failure preservation tests.
        (jobdir / "model.log").write_text(f"synthetic log {stage}\n", encoding="utf-8")  # Leave a deterministic combined native log required by every successful level.
        value = self.values[self.calls]  # Select the output registered for this ladder level.
        self.calls += 1  # Count the attempted solver call even when it raises.
        if isinstance(value, BaseException):  # Preserve deliberate numerical failures for fail-closed tests.
            raise value  # Reproduce the backend failure without inventing U, QoI, or equation counts.
        U_total, qoi = value  # Unpack the prescribed positive physical outputs.
        post = SimpleNamespace(U_total=float(U_total), qoi=float(qoi))  # Expose only the fields consumed by reference construction.
        record = SimpleNamespace(n_equations=1200 + self.calls, wall_s=0.01 * self.calls, extra={})  # Expose actual-looking positive equation and solver-time metadata.
        return post, record  # Return the synthetic native-solve pair.


class InterruptingMeshFactory:  # Simulate an external process interruption after one successfully checkpointed reference level.
    def __init__(self) -> None:  # Initialize a deterministic delegate and invocation counter.
        self.delegate = SyntheticMeshFactory()  # Reuse the normal analytical synthetic mesh construction for the completed prefix.
        self.calls = 0  # Count attempted level meshes so the second level can emulate an abrupt stop.

    def __call__(self, problem: Any, size_fn: Any, *, model_name: str, h_floor: float) -> Mesh:  # Match the production mesh-factory signature exactly.
        self.calls += 1  # Advance the interruption schedule before invoking or aborting mesh construction.
        if self.calls == 2:  # Interrupt only after level zero has been solved and sealed to disk.
            raise KeyboardInterrupt("synthetic external interruption")  # Bypass ordinary numerical-failure capture like an externally stopped process.
        return self.delegate(problem, size_fn, model_name=model_name, h_floor=h_floor)  # Produce the first valid independently graded mesh normally.


def _problem() -> Any:  # Construct one stable canonical bridge case for every synthetic test.
    return make_box_girder_diaphragm(wheel_offset=(12.0, -8.0), opening_radius=61.0, diaphragm_thickness=31.0, pressure=4.2, support_width=72.0)  # Exercise all six manifest-varying factory inputs without Gmsh.


def _config(levels: int = 3) -> ReferenceScheduleConfig:  # Build a short test-only prefix of the preregistered geometric refinement pattern.
    return ReferenceScheduleConfig(background_scales=(1.0, 0.8, 0.64)[:levels], local_floor_scales=(1.0, 0.7, 0.49)[:levels])  # Retain exact A identity and strictly finer independent B candidates.


def test_first_a_b_pair_converges_caches_and_binds_common_reference(tmp_path: Path) -> None:  # Verify the normal two-solve path and no-solve cache reuse.
    problem = _problem()  # Reconstruct the canonical case exactly once.
    mesh_factory = SyntheticMeshFactory()  # Record every independently generated analytical reference mesh.
    runner = SyntheticRunner(problem, [(100.0, 10.0), (100.4, 10.04)], tmp_path / "runner-build")  # Prescribe dual relative changes below one-half percent with isolated native logs.
    outcome = ensure_reference_pair(problem, runner, tmp_path, case_id="BGD-SYNTH-001", config=_config(), mesh_factory=mesh_factory)  # Build, validate, persist, and bind the first A/B pair.
    assert outcome.a_level == 0 and outcome.b_level == 1 and outcome.from_cache is False  # Require direct acceptance of the first independently finer candidate.
    assert outcome.reference_a.U_total == 100.0 and outcome.reference_b.U_total == 100.4  # Preserve both physical levels rather than overwriting A with B.
    assert runner.reference is outcome.reference_b and runner._counter == 0 and runner.calls == 2  # Bind one shared B without charging either reference to online solves.
    assert mesh_factory.calls[1]["h_floor"] < mesh_factory.calls[0]["h_floor"]  # Require a strictly smaller preregistered local minimum for B.
    case_dir = reference_case_dir(tmp_path, "BGD-SYNTH-001")  # Resolve the frozen per-case cache layout through its public helper.
    assert (case_dir / LEDGER_FILENAME).is_file() and (case_dir / REFERENCE_B_FILENAME).is_file()  # Require both the full ledger and compact shared-reference artifact.
    verification = verify_reference_cache(tmp_path, case_id="BGD-SYNTH-001", problem=problem, config=_config(), mesh_factory=mesh_factory, regenerate_meshes=True)  # Rebuild analytical meshes and verify exact hashes without a solver call.
    assert verification["passed"] is True and all(item["passed"] for item in verification["mesh_regeneration"])  # Require structural, convergence, artifact, and strict mesh verification together.
    assert [item["path"] for item in verification["solver_log_verification"]] == ["solver_logs/ref_l00.log", "solver_logs/ref_l01.log"]  # Require one portable authenticated native log for every accepted ladder level.
    assert all(len(item["sha256"]) == 64 and item["size_bytes"] > 0 for item in verification["solver_log_verification"])  # Require nonempty exact bytes and complete digests rather than absolute runner paths.
    cached_runner = SyntheticRunner(problem, [], tmp_path / "runner-cache")  # Create a fresh isolated method runner that cannot perform a solver call.
    cached = ensure_reference_pair(problem, cached_runner, tmp_path, case_id="BGD-SYNTH-001", config=_config(), mesh_factory=mesh_factory)  # Load the authenticated common B through the same idempotent entry point.
    assert cached.from_cache is True and cached_runner.calls == 0 and cached_runner.reference.U_total == 100.4  # Prove cache reuse injects identical numbers without hidden native work.


def test_interrupted_build_resumes_only_missing_registered_suffix(tmp_path: Path) -> None:  # Verify an external stop does not discard or recompute an authenticated completed reference level.
    problem = _problem()  # Reconstruct one unchanged physical case across the interrupted and resumed processes.
    interrupted_runner = SyntheticRunner(problem, [(100.0, 10.0)], tmp_path / "runner-interrupted")  # Provide exactly the level-zero result and portable log completed before interruption.
    with pytest.raises(KeyboardInterrupt, match="external interruption"):  # Confirm the simulated process stop propagates rather than becoming a numerical failure.
        ensure_reference_pair(problem, interrupted_runner, tmp_path, case_id="BGD-SYNTH-RESUME", config=_config(), mesh_factory=InterruptingMeshFactory())  # Stop during construction of the first finer candidate.
    partial_path = reference_case_dir(tmp_path, "BGD-SYNTH-RESUME") / LEDGER_FILENAME  # Resolve the append-only checkpoint left by the interrupted process.
    partial = json.loads(partial_path.read_text(encoding="utf-8"))  # Inspect the sealed human-readable prefix before continuation.
    assert partial["status"] == "building" and len(partial["levels"]) == 1 and len(partial["levels"][0]["mesh_sha"]) == 64  # Require one full-SHA successful level and no fabricated candidate.
    resumed_runner = SyntheticRunner(problem, [(100.4, 10.04)], tmp_path / "runner-resumed")  # Provide only the missing candidate result and its isolated log so recomputing A would fail.
    resumed = ensure_reference_pair(problem, resumed_runner, tmp_path, case_id="BGD-SYNTH-RESUME", config=_config(), mesh_factory=SyntheticMeshFactory())  # Authenticate the prefix and execute only level one.
    assert resumed.a_level == 0 and resumed.b_level == 1 and resumed.from_cache is False  # Accept the same registered consecutive pair after continuation.
    assert resumed_runner.calls == 1 and resumed_runner._counter == 0 and resumed_runner.reference.U_total == 100.4  # Prove no completed solve was repeated or charged online.


def test_failed_first_pair_promotes_only_reference_and_records_upgrade(tmp_path: Path) -> None:  # Verify the sole permitted response to failed A/B agreement.
    problem = _problem()  # Reuse one unchanged compared-method problem throughout all reference levels.
    runner = SyntheticRunner(problem, [(100.0, 10.0), (101.0, 10.2), (101.3, 10.22)], tmp_path / "runner-upgrade")  # Make level one fail and consecutive levels one/two pass with isolated logs.
    outcome = ensure_reference_pair(problem, runner, tmp_path, case_id="BGD-SYNTH-002", config=_config(), mesh_factory=SyntheticMeshFactory())  # Execute only the predeclared reference refinements.
    assert outcome.a_level == 1 and outcome.b_level == 2  # Require promotion of the failed B candidate to the new A before comparing the next finer B.
    assert outcome.reference_a.U_total == 101.0 and outcome.reference_b.U_total == 101.3  # Use the final consecutive pair instead of comparing every upgrade to the stale initial A.
    ledger = json.loads(outcome.ledger_path.read_text(encoding="utf-8"))  # Inspect the human-readable authenticated audit history.
    assert ledger["status"] == "complete" and len(ledger["levels"]) == 3 and len(ledger["upgrade_history"]) == 1  # Retain all levels and exactly one failed convergence decision.
    upgrade = ledger["upgrade_history"][0]  # Read the preserved pre-upgrade comparison evidence.
    assert upgrade["passed"] is False and upgrade["decision"] == "promote_B_to_A_and_continue_preregistered_refinement"  # Require the protocol's reference-only response.
    assert upgrade["reference_A_mesh_sha"] != upgrade["reference_B_mesh_sha"]  # Record distinct independently generated meshes before promotion.


def test_numerical_failure_is_retained_and_never_bound_as_reference(tmp_path: Path) -> None:  # Verify native failure remains explicit rather than becoming a favorable missing result.
    problem = _problem()  # Construct the same canonical finite-element case.
    runner = SyntheticRunner(problem, [(100.0, 10.0), RuntimeError("synthetic ccx divergence")], tmp_path / "runner-failure")  # Fail the first finer reference after producing a valid A and native failure evidence.
    with pytest.raises(ReferenceBuildError, match="level 1"):  # Require an explicit terminal construction error at the exact failed level.
        ensure_reference_pair(problem, runner, tmp_path, case_id="BGD-SYNTH-003", config=_config(), mesh_factory=SyntheticMeshFactory())  # Attempt the two-level reference build.
    case_dir = reference_case_dir(tmp_path, "BGD-SYNTH-003")  # Resolve the retained failed-case directory.
    ledger = json.loads((case_dir / LEDGER_FILENAME).read_text(encoding="utf-8"))  # Read the failure ledger without relaxing its existence requirement.
    assert ledger["status"] == "numerical_failure" and ledger["failure"]["level_index"] == 1  # Preserve terminal status and exact failed level.
    assert ledger["levels"][1]["error_type"] == "RuntimeError" and "divergence" in ledger["levels"][1]["error"]  # Preserve backend type and message for failure matrices.
    assert ledger["levels"][1]["U_total"] is None and ledger["levels"][1]["mesh_sha"] is None  # Represent unavailable failure outputs explicitly without fabricated numerical or mesh values.
    assert ledger["levels"][1]["solver_logs"][0]["path"] == "solver_logs/ref_l01.log" and len(ledger["levels"][1]["solver_logs"][0]["sha256"]) == 64  # Preserve the already-produced failure log under a portable exact path and complete digest.
    assert ledger["levels"][1]["solver_inputs"][0]["path"] == "solver_inputs/ref_l01.inp" and len(ledger["levels"][1]["solver_inputs"][0]["sha256"]) == 64  # Preserve the already-produced failure deck for reproduction with complete identity.
    failed_verification = verify_reference_failure_evidence(tmp_path, case_id="BGD-SYNTH-003", problem=problem, config=_config())  # Recompute every retained success log, failure log, and failure input independently.
    assert failed_verification["passed"] is True and [item["path"] for item in failed_verification["solver_input_verification"]] == ["solver_inputs/ref_l01.inp"]  # Require explicit authenticated failure-deck evidence without a fabricated Reference B.
    assert runner.reference is None and not (case_dir / REFERENCE_B_FILENAME).exists()  # Never inject or persist an unconverged/fabricated common reference.


def test_exhausted_schedule_and_tampered_cache_fail_closed(tmp_path: Path) -> None:  # Verify nonconvergence and cache mutation cannot leak into method metrics.
    problem = _problem()  # Construct the canonical problem for both independent failure checks.
    exhausted_runner = SyntheticRunner(problem, [(100.0, 10.0), (102.0, 10.3)], tmp_path / "runner-exhausted")  # Keep both physical differences above one-half percent with retained logs.
    with pytest.raises(ReferenceBuildError, match="schedule exhausted"):  # Require terminal nonconvergence after every registered level.
        ensure_reference_pair(problem, exhausted_runner, tmp_path, case_id="BGD-SYNTH-004", config=_config(levels=2), mesh_factory=SyntheticMeshFactory())  # Execute the complete short schedule without changing compared methods.
    exhausted_ledger = json.loads((reference_case_dir(tmp_path, "BGD-SYNTH-004") / LEDGER_FILENAME).read_text(encoding="utf-8"))  # Inspect the retained terminal evidence.
    assert exhausted_ledger["status"] == "schedule_exhausted" and exhausted_runner.reference is None  # Refuse to select the finest unconverged result as B.
    valid_runner = SyntheticRunner(problem, [(100.0, 10.0), (100.4, 10.04)], tmp_path / "runner-valid")  # Build a separate valid cache with logs for exact-byte mutation testing.
    ensure_reference_pair(problem, valid_runner, tmp_path, case_id="BGD-SYNTH-005", config=_config(), mesh_factory=SyntheticMeshFactory())  # Persist one authenticated complete pair.
    artifact_path = reference_case_dir(tmp_path, "BGD-SYNTH-005") / REFERENCE_B_FILENAME  # Resolve the compact artifact selected by every method runner.
    tampered = json.loads(artifact_path.read_text(encoding="utf-8"))  # Decode the artifact while intentionally retaining its now-stale integrity seal.
    tampered["reference"]["qoi"] = 999.0  # Change one final metric without recomputing the content digest.
    artifact_path.write_text(json.dumps(tampered), encoding="utf-8")  # Simulate accidental or adversarial cache mutation on disk.
    with pytest.raises(ReferenceBuildError, match="integrity mismatch"):  # Require strict rejection before returning or binding altered numbers.
        load_reference_b(tmp_path, case_id="BGD-SYNTH-005", problem=problem, config=_config())  # Attempt to load the tampered final shared reference.


def test_schedule_validation_rejects_nonrefining_or_unpaired_levels() -> None:  # Verify all scale choices are frozen before any blind solve can inspect results.
    with pytest.raises(ValueError, match="strictly decreasing"):  # Reject a B background that is not finer than A.
        ReferenceScheduleConfig(background_scales=(1.0, 1.0), local_floor_scales=(1.0, 0.7))  # Attempt to register a stalled far-field mesh.
    with pytest.raises(ValueError, match="equal length"):  # Reject a background candidate without a corresponding local-floor target.
        ReferenceScheduleConfig(background_scales=(1.0, 0.8), local_floor_scales=(1.0,))  # Attempt to register an incomplete two-endpoint refinement field.


def test_portable_solver_log_mutation_and_symlink_escape_fail_closed(tmp_path: Path) -> None:  # Prove cache verification recomputes bytes and prevents case-directory traversal through symlinks.
    problem = _problem()  # Construct one stable exact physical case for both tamper checks.
    case_id = "BGD-SYNTH-LOG-GUARD"  # Give the guarded cache one deterministic safe path component.
    runner = SyntheticRunner(problem, [(100.0, 10.0), (100.4, 10.04)], tmp_path / "runner-log-guard")  # Build a valid two-level cache backed by isolated synthetic native logs.
    ensure_reference_pair(problem, runner, tmp_path, case_id=case_id, config=_config(), mesh_factory=SyntheticMeshFactory())  # Persist and seal the complete reference ladder first.
    case_dir = reference_case_dir(tmp_path, case_id)  # Resolve the exact cache boundary used by portable evidence validation.
    level_zero_log = case_dir / "solver_logs" / "ref_l00.log"  # Select the first ledger-authenticated native log.
    original_bytes = level_zero_log.read_bytes()  # Preserve exact valid bytes for the independent symlink-escape check.
    level_zero_log.write_bytes(original_bytes + b"tampered\n")  # Mutate only native evidence while leaving the sealed ledger unchanged.
    with pytest.raises(ReferenceBuildError, match="native evidence integrity mismatch"):  # Require byte-level recomputation rather than trusting stored SHA text.
        verify_reference_cache(tmp_path, case_id=case_id, problem=problem, config=_config())  # Attempt to use the mutated otherwise complete cache.
    level_zero_log.write_bytes(original_bytes)  # Restore exact authenticated bytes before testing path containment independently.
    external_directory = tmp_path / "outside-case-cache"  # Resolve a directory outside the manifest case cache.
    external_directory.mkdir(parents=True, exist_ok=True)  # Create the controlled traversal target.
    (external_directory / "ref_l00.log").write_bytes(original_bytes)  # Preserve matching content so only containment, not digest mismatch, decides the check.
    level_zero_log.unlink()  # Remove the regular in-cache evidence before replacing its parent with a symlink.
    (case_dir / "solver_logs" / "ref_l01.log").unlink()  # Remove the second regular log so the evidence directory can be replaced safely.
    (case_dir / "solver_logs").rmdir()  # Remove only the empty controlled evidence directory.
    (case_dir / "solver_logs").symlink_to(external_directory, target_is_directory=True)  # Redirect the exact registered relative path outside the case cache.
    with pytest.raises(ReferenceBuildError, match="escapes reference case directory"):  # Require resolved containment before reading even digest-matching bytes.
        verify_reference_cache(tmp_path, case_id=case_id, problem=problem, config=_config())  # Attempt to follow the malicious intermediate symlink.


def test_expedited_two_level_operational_b_is_unqualified_and_requires_explicit_loading(tmp_path: Path) -> None:  # Prove nonblocking execution reports the failed original gate and remains fail-closed by default.
    problem = _problem()  # Construct one stable exact physical case for build, strict rejection, and explicit operational loading.
    case_id = "BGD-SYNTH-EXPEDITED"  # Give the operational cache one deterministic safe identifier.
    runner = SyntheticRunner(problem, [(100.0, 10.0), (106.0, 10.8)], tmp_path / "runner-expedited")  # Make both native levels succeed while both original 0.5-percent comparisons fail clearly.
    outcome = ensure_reference_pair(problem, runner, tmp_path, case_id=case_id, mesh_factory=SyntheticMeshFactory(), allow_unqualified=True, expedited_levels=2)  # Invoke the explicit two-level nonblocking amendment.
    assert outcome.qualification is False and outcome.authorization == UNQUALIFIED_AUTHORIZATION and outcome.b_level == 1  # Select the finer successful operational B without claiming qualification.
    ledger = json.loads(outcome.ledger_path.read_text(encoding="utf-8"))  # Inspect the sealed transparent operational record.
    assert ledger["status"] == "complete_unqualified" and ledger["qualification"] is False  # Require the exact nonconverged usable status and machine-readable false qualification.
    assert ledger["final_pair"]["passed"] is False and ledger["final_pair"]["converged"] is False  # Preserve the original gate result and prohibit a converged label.
    assert ledger["authorization"] == UNQUALIFIED_AUTHORIZATION and ledger["execution_amendment"]["expedited_levels"] == 2  # Bind the exact user authorization and selected fixed prefix immutably.
    assert len(ledger["levels"]) == 2 and all(level["solver_logs"][0]["size_bytes"] > 0 for level in ledger["levels"])  # Require exactly two successful levels with the full nonempty log contract intact.
    with pytest.raises(ReferenceBuildError, match="complete_unqualified"):  # Keep the default loader explicitly fail-closed on operational evidence.
        load_reference_b(tmp_path, case_id=case_id, problem=problem)  # Attempt an implicit strict load without authorization or expedited depth.
    with pytest.raises(ReferenceBuildError):  # Keep the default verifier fail-closed even though exact rejection ordering may identify schedule mismatch first.
        verify_reference_cache(tmp_path, case_id=case_id, problem=problem)  # Attempt strict verification of the amended cache.
    verified = verify_reference_cache(tmp_path, case_id=case_id, problem=problem, allow_unqualified=True, expedited_levels=2)  # Explicitly reproduce the amendment and accept integrity, not qualification.
    assert verified["passed"] is True and verified["qualification"] is False and verified["status"] == "complete_unqualified"  # Separate verifier integrity success from scientific qualification failure.
    assert verified["original_convergence_gate"]["passed"] is False and verified["authorization"] == UNQUALIFIED_AUTHORIZATION  # Continue reporting the unchanged gate and exact authorization.
    loaded = load_reference_b(tmp_path, case_id=case_id, problem=problem, allow_unqualified=True, expedited_levels=2)  # Load only through the explicit operational API.
    assert loaded.U_total == pytest.approx(106.0)  # Use the finest successful selected level as operational B exactly.


def test_expedited_native_failure_with_no_success_never_publishes_operational_b(tmp_path: Path) -> None:  # Prove nonblocking qualification never weakens native success or evidence requirements.
    problem = _problem()  # Construct the same canonical physical case.
    case_id = "BGD-SYNTH-EXPEDITED-FAIL"  # Give the terminal failure one deterministic safe identifier.
    runner = SyntheticRunner(problem, [RuntimeError("synthetic level-zero failure")], tmp_path / "runner-expedited-failure")  # Fail the first native level after producing its log and input evidence.
    with pytest.raises(ReferenceBuildError, match="level 0"):  # Require the native failure to remain terminal under the expedited amendment.
        ensure_reference_pair(problem, runner, tmp_path, case_id=case_id, mesh_factory=SyntheticMeshFactory(), allow_unqualified=True, expedited_levels=2)  # Attempt operational execution without any successful reference level.
    case_dir = reference_case_dir(tmp_path, case_id)  # Resolve the retained failed cache evidence.
    ledger = json.loads((case_dir / LEDGER_FILENAME).read_text(encoding="utf-8"))  # Inspect the sealed native failure record.
    assert ledger["status"] == "numerical_failure" and ledger["qualification"] is None  # Refuse both qualified and operational completion after native failure.
    assert not (case_dir / REFERENCE_B_FILENAME).exists()  # Never fabricate or publish a compact Reference B without two successful levels.
    failed = verify_reference_failure_evidence(tmp_path, case_id=case_id, problem=problem, allow_unqualified=True, expedited_levels=2)  # Authenticate the failed amended ladder and retained native evidence.
    assert failed["passed"] is True and failed["execution_amendment"]["native_failure_fallback_allowed"] is False  # Preserve valid evidence while explicitly forbidding fallback.
