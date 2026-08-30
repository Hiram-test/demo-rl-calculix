"""Focused tests for one-shot pre-blind freezing and exact-byte verification."""  # Describe the protocol guard covered by this suite.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.

import hashlib  # Build conventional checksum sidecars independently from production code.
import json  # Write transparent finite fixture artifacts.
from pathlib import Path  # Annotate and construct isolated Git worktrees.
import subprocess  # Create exact implementation and freeze commits without a shell.
import pytest  # Assert precise protocol rejections.

from visionamr.bridge_case_manifest import write_case_manifest  # Generate the canonical 48-case protocol design for each isolated freeze.
from visionamr.vla.four_way_benchmark import load_frozen_config  # Exercise direct canonical freeze-to-blind-config schema compatibility.
from visionamr.vla.four_way_freeze import CODE_FILES, FreezeError, create_freeze, scan_for_disclosed_results, seal_freeze_tag, sha256_file, verify_freeze  # Exercise the complete public freeze boundary and fixed Git ref seal.


def _run(repo: Path, *arguments: str) -> str:  # Execute one controlled Git command in the isolated fixture worktree.
    process = subprocess.run(("git", "-C", str(repo), *arguments), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # Avoid shell parsing and capture exact output.
    return process.stdout.strip()  # Return the complete textual result without terminal whitespace.


def _write_json(path: Path, payload: object) -> None:  # Write one deterministic finite fixture artifact.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the requested fixture directory.
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Match transparent protocol JSON conventions.


def _artifact(root: Path, relative: str, payload: object) -> str:  # Create one JSON evidence file and return its campaign-relative path.
    path = root / relative  # Resolve the artifact beneath the isolated campaign.
    _write_json(path, payload)  # Persist finite transparent fixture evidence.
    return relative  # Return the portable source-config path.


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:  # Build a complete uncommitted campaign under one exact implementation commit.
    repo = tmp_path / "repo"  # Isolate Git provenance from the real shared checkout.
    repo.mkdir()  # Create the fixture worktree root.
    _run(repo, "init", "-q")  # Initialize a local-only Git repository.
    _run(repo, "config", "user.email", "freeze-test@example.invalid")  # Configure a deterministic local commit identity.
    _run(repo, "config", "user.name", "Freeze Test")  # Configure a deterministic local author name.
    marker = repo / "implementation.txt"  # Create one reviewed implementation-tree file.
    marker.write_text("reviewed implementation\n", encoding="utf-8")  # Give the implementation commit deterministic content.
    for relative in sorted({path for paths in CODE_FILES.values() for path in paths}):  # Materialize every claim-critical source path required by code-hash freezing.
        code_path = repo / relative  # Resolve the dummy reviewed code path inside the isolated worktree.
        code_path.parent.mkdir(parents=True, exist_ok=True)  # Create the exact source package or scripts directory.
        code_path.write_text(f"# reviewed fixture for {relative}\n", encoding="utf-8")  # Give every required committed path distinct deterministic bytes.
    _run(repo, "add", ".")  # Stage the reviewed implementation marker and all claim-critical code fixtures.
    _run(repo, "commit", "-q", "-m", "implementation")  # Create the required implementation checkpoint.
    implementation = _run(repo, "rev-parse", "HEAD")  # Capture its complete exact object identity.
    root = repo / "results" / "wm_vla_four_way_p1"  # Resolve the protocol-mandated campaign root.
    _manifest_path, _sidecar_path, _digest = write_case_manifest(root / "protocol")  # Persist the canonical 48-case design and sidecar.
    manifest = json.loads((root / "protocol" / "case_manifest.json").read_text(encoding="utf-8"))  # Read case IDs and geometry hashes for partition fixtures.
    for case in manifest["cases"]:  # Create exactly one semantically bound partition file per manifest case.
        payload = {"protocol_id": "WMVLA-4WAY-P1", "geometry_hash": case["geometry_hash"], "spec_sha256": hashlib.sha256(str(case["case_id"]).encode("utf-8")).hexdigest()}  # Build the minimum transparent integrity fields checked by the freeze boundary.
        _write_json(root / "protocol" / "partitions" / case["case_id"] / "partition_spec.json", payload)  # Persist the per-case shared WM/RL specification fixture.
    models = [  # Declare one world model, three supervised candidates, and three independent RL policies.
        {"name": "world_model", "method": "world_model", "path": _artifact(root, "training/world_model/model.json", {"weights": [1.0]})},  # Create the sole world-model snapshot.
        {"name": "supervised_seed_11", "method": "supervised", "seed": 11, "path": _artifact(root, "training/supervised/seed_11.pt", {"weights": [11.0]})},  # Create the validation-selected supervised checkpoint.
        {"name": "supervised_seed_12", "method": "supervised", "seed": 12, "path": _artifact(root, "training/supervised/seed_12.pt", {"weights": [12.0]})},  # Retain an unselected supervised candidate for audit.
        {"name": "supervised_seed_13", "method": "supervised", "seed": 13, "path": _artifact(root, "training/supervised/seed_13.pt", {"weights": [13.0]})},  # Retain the third supervised candidate for audit.
        {"name": "rl_seed_21", "method": "rl", "seed": 21, "path": _artifact(root, "training/rl/seed_21.pt", {"weights": [21.0]})},  # Create the first frozen RL policy.
        {"name": "rl_seed_22", "method": "rl", "seed": 22, "path": _artifact(root, "training/rl/seed_22.pt", {"weights": [22.0]})},  # Create the second frozen RL policy.
        {"name": "rl_seed_23", "method": "rl", "seed": 23, "path": _artifact(root, "training/rl/seed_23.pt", {"weights": [23.0]})},  # Create the third frozen RL policy.
    ]  # Complete the explicit deployment-model inventory.
    evidence = [  # Declare complete pre-test training and validation-selection evidence.
        {"name": "world_training", "method": "world_model", "phase": "training", "path": _artifact(root, "training/world_model/training.json", {"test_access": False})},  # Bind world transition acquisition evidence.
        {"name": "supervised_training", "method": "supervised", "phase": "training", "path": _artifact(root, "training/supervised/training.json", {"test_access": False})},  # Bind supervised expert and network training evidence.
        {"name": "supervised_validation", "method": "supervised", "phase": "validation", "path": _artifact(root, "training/supervised/validation.json", {"selected_seed": 11})},  # Bind supervised checkpoint selection evidence.
        {"name": "rl_training", "method": "rl", "phase": "training", "path": _artifact(root, "training/rl/training.json", {"episodes": 900})},  # Bind all three RL training histories.
        {"name": "rl_validation", "method": "rl", "phase": "validation", "path": _artifact(root, "training/rl/validation.json", {"selected": [21, 22, 23]})},  # Bind all RL checkpoint selection evidence.
    ]  # Complete required method-phase coverage.
    costs = [  # Declare one finite offline-cost report for each learned method family.
        {"name": "world_cost", "method": "world_model", "path": _artifact(root, "training/world_model/cost.json", {"real_solves": 144})},  # Report world transition acquisition solves.
        {"name": "supervised_cost", "method": "supervised", "path": _artifact(root, "training/supervised/cost.json", {"expert_solves": 144})},  # Report supervised expert and network cost.
        {"name": "rl_cost", "method": "rl", "path": _artifact(root, "training/rl/cost.json", {"episodes": 900})},  # Report RL episodes and cost.
    ]  # Complete all three cost families.
    scientific = {"horizon": 4, "beam_width": 20, "max_extra_regions": 2, "max_extra_depth": 2, "min_relative_gain": 0.018, "uncertainty_limit": 0.34, "failure_limit": 0.42, "budget_safety": 0.97, "regression_tolerance": 0.03, "ensemble_size": 5, "ridge": 0.001}  # Define the protocol-level compatibility settings explicitly.
    planner = {"horizon": 4, "beam_width": 20, "candidate_regions": 5, "max_extra_regions": 2, "max_extra_depth": 2, "warmup_transitions": 1, "discount": 0.84, "resource_weight": 0.17, "uncertainty_weight": 0.75, "failure_weight": 1.1, "uncertainty_limit": 0.34, "failure_limit": 0.42, "budget_safety": 0.97, "min_robust_gain": 0.018}  # Freeze every V0 planner dataclass field.
    world_model = {"refine_factor": 0.5, "error_power": 1.65, "neighbor_spill": 0.18, "ensemble_size": 5, "ridge": 0.001, "min_rows": 10, "max_log_residual": 0.7, "prior_uncertainty": 0.24, "uncertainty_scale": 1.8, "max_rows": 3000}  # Freeze every V0 transition-model dataclass field.
    tool = {"theta": 0.5, "refine_factor": 0.5, "core_theta": 0.72, "budget_safety": 1.0, "max_extra_depth": 2}  # Freeze every deterministic gateway dataclass field.
    runtime = {"max_solves": 6, "n_equation_cap": 120000, "theta": 0.5, "refine_factor": 0.5, "core_theta": 0.72, "audit_slack": 0.08, "fallback_cooldown": 1, "stagnation_tolerance": 0.002, "stagnation_steps": 2, "method_name": "world_model_vla", "artifact_dir": None, "require_reference": True}  # Freeze every V0 runtime dataclass field.
    config = {"TEST_NOT_RUN": True, "partition_root": "protocol/partitions", "common_gradation": 1.0, "world_model_seed": 271828, "scientific_config": scientific, "world_planner": planner, "world_model_config": world_model, "world_tool_config": tool, "world_model_runtime": runtime, "selected_supervised_seed": 11, "rl_seeds": [21, 22, 23], "model_artifacts": models, "training_validation_artifacts": evidence, "training_cost_sources": costs}  # Freeze complete components, common remeshing, seeds, and explicit artifact sources.
    source = root / "protocol" / "freeze_config_source.json"  # Resolve the reviewed source inventory below the campaign boundary.
    _write_json(source, config)  # Persist the source without generated hashes.
    return repo, root, source, implementation  # Return the complete creation inputs.


def _environment() -> dict[str, object]:  # Build a finite deterministic injected environment for unit isolation.
    packages = {name: "test-1.0" for name in ("numpy", "scipy", "gmsh", "matplotlib", "torch", "pytest")}  # Provide every direct dependency identity required by the production environment lock.
    variables = {name: None for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GMSH_NUM_THREADS")}  # Provide the exact approved deterministic thread-control variable set.
    return {"schema": "wmvla-four-way-environment-v1", "protocol_id": "WMVLA-4WAY-P1", "python": {"implementation": "CPython", "version": "test", "version_info": [3, 12, 0, "final", 0]}, "platform": {"system": "TestOS", "release": "1", "machine": "test64", "platform": "TestOS-1-test64"}, "packages": packages, "pip_freeze": [f"{name}=={version}" for name, version in sorted(packages.items())], "gmsh": {"distribution_version": "test-1.0", "runtime_version": "test-1.0"}, "calculix": {"executable_sha256": "a" * 64, "version_output": "This is Version test", "version_query_returncode": 201}, "environment_variables": variables}  # Avoid native execution while exercising the same complete environment contract as production.


def test_create_dry_run_writes_nothing_then_freeze_verifies_exact_bytes(tmp_path: Path) -> None:  # Prove full rehearsal, one-shot creation, mutation rejection, and committed verification.
    repo, root, source, implementation = _build_fixture(tmp_path)  # Build complete reviewed pre-test inputs.
    dry = create_freeze(root, source, implementation, repo, environment=_environment(), dry_run=True)  # Perform every input and hash check without publication.
    assert dry["dry_run"] is True and not (root / "protocol" / "frozen_config.json").exists()  # Require a complete non-mutating rehearsal.
    created = create_freeze(root, source, implementation, repo, environment=_environment())  # Publish the unique freeze bundle.
    verified = verify_freeze(root, repo, live_environment=_environment())  # Recompute every protected identity and environment before committing.
    assert created["TEST_NOT_RUN"] is True and verified["protected_artifact_count"] >= 48 + 7  # Require an explicit pre-test declaration and complete partition/model coverage.
    assert created["allow_unqualified_references"] is False and verified["allow_unqualified_references"] is False  # Require fail-closed reference qualification unless the reviewed source opts in explicitly.
    assert verified["complete_v0_config"]["common_gradation"] == 1.0 and len(verified["code_sha256"]) == sum(len(paths) for paths in CODE_FILES.values())  # Verify full V0 configuration and every required live code file.
    baseline_environment = _environment()  # Recover the same complete environment shape used by the valid freeze fixture.
    changed_environment = {**baseline_environment, "python": {**baseline_environment["python"], "version": "different"}}  # Change only the interpreter version while retaining every other required lock field.
    with pytest.raises(FreezeError, match="live environment differs"):  # Require active environment comparison beyond stored-file hashing.
        verify_freeze(root, repo, live_environment=changed_environment)  # Refuse a blind runner with different Python evidence.
    frozen = json.loads((root / "protocol" / "frozen_config.json").read_text(encoding="utf-8"))  # Inspect the exact runtime configuration.
    blind_loaded = load_frozen_config(root / "protocol" / "frozen_config.json")  # Pass the generated canonical bytes directly through the benchmark's blind loader.
    assert frozen["implementation_commit_sha"] == implementation and len(frozen["partition_spec_sha256"]) == 48  # Bind code and all per-case partitions exactly.
    assert blind_loaded["partition_root"] == "protocol/partitions" and blind_loaded["common_gradation"] == 1.0  # Prove canonical partition and common mesh settings require no compatibility rewrite.
    assert frozen["reference_config"]["agreement_rtol"] == pytest.approx(0.005) and frozen["TEST_NOT_RUN"] is True  # Include the exact preregistered reference acceptance schedule.
    _run(repo, "add", "results/wm_vla_four_way_p1")  # Stage only the campaign freeze and its explicit source artifacts.
    _run(repo, "commit", "-q", "-m", "freeze")  # Create the dedicated descendant freeze commit.
    sealed = seal_freeze_tag(root, repo, live_environment=_environment())  # Create the fixed non-self-referential tag after the dedicated freeze commit.
    assert sealed["git"]["freeze_git_ref_target_sha"] == sealed["git"]["head_sha"]  # Require the immutable tag to resolve exactly to the selected freeze commit.
    with pytest.raises(FreezeError, match="already exists"):  # Refuse moving or silently replacing an already disclosed freeze ref.
        seal_freeze_tag(root, repo, live_environment=_environment())  # Attempt to seal the same campaign a second time.
    committed = verify_freeze(root, repo, require_committed=True, live_environment=_environment())  # Enforce tag, clean Git ancestry, tracking, code, and environment before blind execution.
    assert committed["git"]["implementation_commit_sha"] == implementation and committed["git"]["freeze_commit_sha"] != implementation  # Distinguish implementation and dedicated freeze commits.
    manifest = json.loads((root / "protocol" / "case_manifest.json").read_text(encoding="utf-8"))  # Recover blind IDs for the strict post-freeze reference exception.
    test_id = next(case["case_id"] for case in manifest["cases"] if case["split"] == "test")  # Select one manifest-owned test case deterministically.
    _write_json(root / "references" / test_id / "reference_ledger.json", {"integrity_sha256": "fixture"})  # Simulate a separately built reference ledger that the benchmark will authenticate.
    _write_json(root / "references" / test_id / "reference_B.json", {"integrity_sha256": "fixture"})  # Simulate the sole permitted final reference cache file.
    solver_log = root / "references" / test_id / "solver_logs" / "ref_l00.log"  # Resolve one preregistered retained CalculiX level log beneath the exact test case.
    solver_log.parent.mkdir(parents=True, exist_ok=True)  # Create only the controlled strict solver-log directory in the isolated repository fixture.
    solver_log.write_text("CalculiX fixture\n", encoding="utf-8")  # Simulate the full-hash log later authenticated by the reference ledger verifier.
    solver_input = root / "references" / test_id / "solver_inputs" / "ref_l00.inp"  # Resolve one retained failed-level CalculiX deck beneath the exact test case.
    solver_input.parent.mkdir(parents=True, exist_ok=True)  # Create only the controlled strict solver-input directory in the isolated repository fixture.
    solver_input.write_text("*HEADING\n", encoding="utf-8")  # Simulate the full-hash failed-level input later authenticated by the reference ledger verifier.
    with pytest.raises(FreezeError, match="TEST_NOT_RUN violation"):  # Keep the default verifier strict for create and first blind opening.
        verify_freeze(root, repo, require_committed=True, live_environment=_environment())  # Reject test reference files without the explicit post-freeze mode.
    cached = verify_freeze(root, repo, require_committed=True, live_environment=_environment(), allow_postfreeze_test_references=True)  # Permit exact two-file caches before immediate benchmark-level semantic verification.
    assert cached["postfreeze_test_reference_paths"] == [f"results/wm_vla_four_way_p1/references/{test_id}/reference_B.json", f"results/wm_vla_four_way_p1/references/{test_id}/reference_ledger.json", f"results/wm_vla_four_way_p1/references/{test_id}/solver_inputs/ref_l00.inp", f"results/wm_vla_four_way_p1/references/{test_id}/solver_logs/ref_l00.log"]  # Disclose every allowed untracked cache, failed-level input, and retained solver-log path precisely.
    invalid_level_log = root / "references" / test_id / "solver_logs" / "ref_l06.log"  # Select the first level beyond the six preregistered refinement scales.
    invalid_level_log.write_text("unregistered ladder level\n", encoding="utf-8")  # Introduce a plausible-looking but unauthorized post-freeze solver log.
    with pytest.raises(FreezeError, match="ref_l06.log"):  # Require the whitelist to derive its finite level set from DEFAULT_REFERENCE_CONFIG.
        verify_freeze(root, repo, require_committed=True, live_environment=_environment(), allow_postfreeze_test_references=True)  # Reject a solver log outside the preregistered ladder.
    invalid_level_log.unlink()  # Remove only the controlled out-of-range log before checking an unknown root-level filename.
    (root / "references" / test_id / "solver.log").write_text("unknown\n", encoding="utf-8")  # Introduce one unregistered reference-directory file.
    with pytest.raises(FreezeError, match="solver.log"):  # Require strict path and filename whitelisting.
        verify_freeze(root, repo, require_committed=True, live_environment=_environment(), allow_postfreeze_test_references=True)  # Reject unknown logs or hidden evidence even in post-freeze mode.
    (root / "references" / test_id / "solver.log").unlink()  # Remove only the controlled unknown file before the model-mutation check.
    model_path = root / "training" / "world_model" / "model.json"  # Select one protected deployment snapshot.
    model_path.write_text('{"weights":[2.0]}\n', encoding="utf-8")  # Mutate exact bytes after the freeze commit.
    with pytest.raises(FreezeError, match="changed after freeze"):  # Require exact hash verification to catch model drift.
        verify_freeze(root, repo, live_environment=_environment(), allow_postfreeze_test_references=True)  # Reject the mutated bundle while retaining the already authenticated strict reference exception.


def test_freeze_refuses_primary_results_ablations_and_test_references(tmp_path: Path) -> None:  # Prove TEST_NOT_RUN is a filesystem-enforced property rather than prose.
    repo, root, source, implementation = _build_fixture(tmp_path)  # Build uncontaminated reviewed inputs.
    manifest = json.loads((root / "protocol" / "case_manifest.json").read_text(encoding="utf-8"))  # Recover the exact blind case IDs without reference values.
    test_id = next(case["case_id"] for case in manifest["cases"] if case["split"] == "test")  # Select one manifest-declared blind case deterministically.
    forbidden = root / "references" / test_id / "reference_B.json"  # Resolve a prohibited pre-freeze test-reference artifact.
    _write_json(forbidden, {"energy": 1.0})  # Simulate prior blind reference disclosure.
    assert scan_for_disclosed_results(root, manifest) == [f"references/{test_id}/reference_B.json"]  # Report the exact contaminating path.
    with pytest.raises(FreezeError, match="TEST_NOT_RUN violation"):  # Require creation to stop before any freeze output is written.
        create_freeze(root, source, implementation, repo, environment=_environment())  # Refuse a post-disclosure freeze.
    forbidden.unlink()  # Remove only the controlled fixture contamination.
    inactive_log = root / "references" / test_id / "solver_logs" / "ref_l02.log"  # Resolve a valid strict-ladder filename outside an authenticated two-level expedited prefix.
    inactive_log.parent.mkdir(parents=True, exist_ok=True)  # Create only the controlled test-reference log directory.
    inactive_log.write_text("inactive level\n", encoding="utf-8")  # Simulate an extra native log that an expedited ledger must not silently ignore.
    assert scan_for_disclosed_results(root, manifest, allow_postfreeze_test_references=True, postfreeze_reference_level_count=2) == [f"references/{test_id}/solver_logs/ref_l02.log"]  # Enforce the exact active frozen ladder depth in the post-freeze whitelist.
    inactive_log.unlink()  # Remove only the controlled inactive-level artifact before the aggregate check.
    _write_json(root / "aggregate" / "final_gate.json", {"OVERALL_WIN": True})  # Simulate a partly written post-test aggregate.
    with pytest.raises(FreezeError, match="aggregate/final_gate.json"):  # Require aggregate detection regardless of claimed outcome.
        create_freeze(root, source, implementation, repo, environment=_environment())  # Refuse post-hoc configuration freezing.


def test_committed_verifier_requires_a_distinct_freeze_commit(tmp_path: Path) -> None:  # Prove an implementation tag alone cannot impersonate the mandatory artifact-only freeze commit.
    repo, root, source, implementation = _build_fixture(tmp_path)  # Build complete inputs beneath the still-selected implementation commit.
    create_freeze(root, source, implementation, repo, environment=_environment())  # Publish valid but deliberately uncommitted freeze evidence.
    _run(repo, "tag", "wmvla-p1-freeze", implementation)  # Place the fixed ref directly on implementation HEAD to simulate an invalid shortcut.
    with pytest.raises(FreezeError, match="dedicated freeze commit must differ"):  # Require the ancestry guard to reject equality before considering artifact tracking.
        verify_freeze(root, repo, require_committed=True, live_environment=_environment())  # Refuse blind launch without a distinct descendant campaign-only commit.


def test_unqualified_reference_override_is_explicit_hashed_and_boolean(tmp_path: Path) -> None:  # Prove the exceptional qualification policy is explicit, immutable, and never inferred from truthy values.
    repo, root, source, implementation = _build_fixture(tmp_path)  # Build complete reviewed inputs with the default fail-closed policy omitted.
    config = json.loads(source.read_text(encoding="utf-8"))  # Read the controlled source inventory before any freeze artifact exists.
    config["allow_unqualified_references"] = "true"  # Introduce an ambiguous truthy string that JSON callers could otherwise coerce.
    _write_json(source, config)  # Publish the invalid non-boolean override for dry-run validation.
    with pytest.raises(FreezeError, match="explicit JSON boolean"):  # Require an exact JSON true or false token.
        create_freeze(root, source, implementation, repo, environment=_environment(), dry_run=True)  # Reject ambiguous authorization before publication.
    config["allow_unqualified_references"] = True  # Make the exceptional reviewed authorization explicit.
    _write_json(source, config)  # Publish the valid opt-in source inventory.
    with pytest.raises(FreezeError, match="requires protocol/EXPEDITED_EXECUTION_AMENDMENT.md"):  # Require durable human authorization in addition to the boolean switch.
        create_freeze(root, source, implementation, repo, environment=_environment(), dry_run=True)  # Reject an unaudited unqualified-reference override.
    amendment = root / "protocol" / "EXPEDITED_EXECUTION_AMENDMENT.md"  # Resolve the sole canonical post-registration authorization path.
    amendment.write_text("# Stale amendment\n\n- 授权标识：`different_authorization`\n", encoding="utf-8")  # Publish a plausible file carrying the wrong authorization identity.
    with pytest.raises(FreezeError, match="authorization token does not match"):  # Require exact token matching rather than mere filename presence.
        create_freeze(root, source, implementation, repo, environment=_environment(), dry_run=True)  # Reject a stale or substituted post-registration amendment.
    amendment.write_text("# Explicit post-registration amendment\n\n- 授权标识：`user_authorized_nonblocking_2026-08-30`\n", encoding="utf-8")  # Publish the exact named authorization token in a compact controlled fixture.
    created = create_freeze(root, source, implementation, repo, environment=_environment())  # Seal the explicit policy alongside every immutable input.
    verified = verify_freeze(root, repo, live_environment=_environment())  # Recompute the frozen policy and all protected hashes.
    frozen = json.loads((root / "protocol" / "frozen_config.json").read_text(encoding="utf-8"))  # Inspect the exact protected runtime policy.
    index = json.loads((root / "protocol" / "freeze_index.json").read_text(encoding="utf-8"))  # Inspect the independently sidecar-authenticated root seal.
    assert created["allow_unqualified_references"] is True and verified["allow_unqualified_references"] is True and verified["expedited_reference_levels"] == 2  # Report the exceptional authorization and fixed two-level operational depth honestly.
    assert frozen["allow_unqualified_references"] is True and index["allow_unqualified_references"] is True  # Bind the same boolean into both exact-hash protected configuration layers.
    assert frozen["reference_execution_amendment"]["sha256"] == sha256_file(amendment) and verified["reference_execution_amendment"] == frozen["reference_execution_amendment"]  # Recompute and report the exact human amendment bytes under the same authorization.


def test_freeze_requires_exact_implementation_and_complete_scientific_settings(tmp_path: Path) -> None:  # Prove code identity and explicit settings cannot fall back silently.
    repo, root, source, implementation = _build_fixture(tmp_path)  # Build complete valid inputs first.
    with pytest.raises(FreezeError, match="does not equal implementation_commit"):  # Reject a different complete object identity.
        create_freeze(root, source, "0" * len(implementation), repo, environment=_environment(), dry_run=True)  # Attempt a dry-run against nonexistent reviewed code.
    config = json.loads(source.read_text(encoding="utf-8"))  # Load the reviewed source for one controlled omission.
    del config["scientific_config"]["horizon"]  # Remove one protocol-listed planner setting.
    _write_json(source, config)  # Publish the deliberately incomplete source inventory.
    with pytest.raises(FreezeError, match="horizon"):  # Require explicit scientific settings even during rehearsal.
        create_freeze(root, source, implementation, repo, environment=_environment(), dry_run=True)  # Reject implicit repository defaults before test.
    config["scientific_config"]["horizon"] = 4  # Restore the omitted protocol-level setting for the next independent guard.
    config["common_gradation"] = 0.9  # Reintroduce the historical inconsistent per-method gradation value.
    _write_json(source, config)  # Publish the deliberately incompatible complete-component configuration.
    with pytest.raises(FreezeError, match="common_gradation"):  # Require exact unified size-field smoothing.
        create_freeze(root, source, implementation, repo, environment=_environment(), dry_run=True)  # Reject a noncanonical common gradation before test.


def test_freeze_index_sidecar_authenticates_exact_index_bytes(tmp_path: Path) -> None:  # Prove the root seal itself cannot be changed unnoticed.
    repo, root, source, implementation = _build_fixture(tmp_path)  # Build complete reviewed pre-test inputs.
    create_freeze(root, source, implementation, repo, environment=_environment())  # Publish the unique freeze index and sidecar.
    index = root / "protocol" / "freeze_index.json"  # Resolve the root immutable-input inventory.
    before = sha256_file(index)  # Capture the authenticated exact-byte digest.
    index.write_bytes(index.read_bytes() + b" ")  # Alter only formatting bytes while preserving JSON semantics.
    assert sha256_file(index) != before  # Confirm the mutation changed exact persisted bytes.
    with pytest.raises(FreezeError, match="does not authenticate"):  # Require sidecar verification before trusting index paths.
        verify_freeze(root, repo, live_environment=_environment())  # Reject semantically harmless but review-invalid index mutation.


def test_workflow_blind_is_unsplit_and_requires_long_lived_self_hosted_capacity() -> None:  # Prevent a non-resumable 2016-solve campaign from returning to an impossible hosted-run ceiling.
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "wm-vla-four-way-p1.yml"  # Resolve the exact reviewed CI contract independently from the test launch directory.
    text = workflow.read_text(encoding="utf-8")  # Inspect transparent YAML without adding a parser dependency to the repository requirements.
    blind = text.split("  dispatch-blind-once:", maxsplit=1)[1]  # Isolate the sole manually dispatched blind job after asserting its fixed marker exists.
    assert "runs-on: [self-hosted, linux, x64, wmvla-four-way-p1-blind]" in blind  # Require the explicit capacity-reviewed self-hosted label set.
    assert "runs-on: ubuntu-" not in blind  # Forbid accidental fallback to a six-hour GitHub-hosted runner.
    timeout_line = next(line for line in blind.splitlines() if line.strip().startswith("timeout-minutes:"))  # Locate the sole blind job timeout declaration.
    timeout_minutes = int(timeout_line.split(":", maxsplit=1)[1].split("#", maxsplit=1)[0].strip())  # Parse only the comment-free integer value.
    assert timeout_minutes >= 24 * 60  # Require at least one uninterrupted day without enabling resume or scientific sharding.
    assert "matrix:" not in blind and "shard" not in blind.lower() and "--resume" not in blind  # Preserve the complete ascending 16-case one-command boundary.
    assert blind.count("Execute explicit complete blind command") == 1  # Require exactly one reviewed command execution step for the full campaign.
