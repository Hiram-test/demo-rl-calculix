"""Verify complete-split Reference A/B planning, execution, resume, and blind gating."""  # Define focused coverage for the formal reference campaign boundary.
from __future__ import annotations  # Postpone annotation evaluation for supported Python runtimes.

import hashlib  # Recompute exact summary identities independently from the implementation.
import json  # Inspect persisted campaign artifacts and subprocess terminal plans.
from pathlib import Path  # Build isolated campaign, repository, work, and summary fixtures.
import subprocess  # Exercise the public CLI without importing its parser internals.
import sys  # Launch the CLI with the same interpreter running the focused tests.
from types import SimpleNamespace  # Create a minimal independent runner fixture without native solves.
from typing import Any  # Annotate heterogeneous injected callback arguments.

import pytest  # Express path rejection and exact complete-split assertions.

from visionamr.bridge_case_manifest import write_case_manifest  # Create the exact canonical 48-case manifest and checksum sidecar solve-free.
from visionamr.experiment import Reference  # Construct legitimate immutable Reference objects for injected builder outcomes.
from visionamr.vla.four_way_reference_campaign import ReferenceCampaignError, build_reference_campaign_plan, run_reference_campaign  # Exercise the public solve-free and execution APIs directly.
from visionamr.vla.four_way_references import UNQUALIFIED_AUTHORIZATION, ReferenceBuildOutcome  # Return the real result shape and exact amendment token from deterministic fixtures.

ROOT = Path(__file__).resolve().parents[1]  # Locate the repository and public CLI independently from the test launch directory.


def _campaign_fixture(tmp_path: Path) -> tuple[Path, Path]:  # Create one repository-shaped campaign containing only the authenticated manifest.
    repository = tmp_path / "repository"  # Resolve an isolated fake Git worktree boundary for path and injected gate tests.
    campaign = repository / "results" / "wm_vla_four_way_p1"  # Match the formal campaign-relative layout exactly.
    write_case_manifest(campaign / "protocol")  # Persist the frozen 48-case JSON and full SHA-256 sidecar without a solve.
    return repository, campaign  # Return both boundaries for explicit driver calls.


def _fixture_callbacks(failing_case: str | None = None, *, qualification: bool = True) -> tuple[Any, Any, Any, list[str]]:  # Build deterministic strict or operational callback seams plus call evidence.
    runner_calls: list[str] = []  # Retain the exact per-case work-directory order for isolation checks.

    def runner_factory(problem: Any, workdir: Path) -> Any:  # Replace FemRunner construction without invoking CalculiX.
        workdir.mkdir(parents=True, exist_ok=True)  # Reproduce the real runner's per-case directory creation.
        (workdir / "native.log").write_text(f"fixture {problem.instance_id}\n", encoding="utf-8")  # Leave one retained native-log analogue with deterministic bytes.
        runner_calls.append(workdir.name)  # Record the disjoint case directory in execution order.
        return SimpleNamespace(problem=problem, workdir=workdir)  # Return only the attributes needed by the injected builder.

    def ensure_fn(problem: Any, runner: Any, reference_root: Path, *, case_id: str, config: Any, allow_unqualified: bool, expedited_levels: int | None) -> ReferenceBuildOutcome:  # Simulate strict or expedited new-build and cache-reuse outcomes with formal filenames.
        del problem, runner, config, allow_unqualified, expedited_levels  # Mark scientific and amendment inputs as intentionally unused by this success fixture.
        case_root = Path(reference_root) / case_id  # Resolve the exact formal per-case cache directory.
        case_root.mkdir(parents=True, exist_ok=True)  # Reproduce the builder's atomic intent-directory creation.
        ledger_path = case_root / "reference_ledger.json"  # Resolve the formal authoritative ladder filename.
        reference_b_path = case_root / "reference_B.json"  # Resolve the formal compact common-reference filename.
        from_cache = reference_b_path.is_file()  # Model automatic complete-cache reuse on a second campaign attempt.
        if case_id == failing_case:  # Exercise terminal failure retention without fabricating Reference B.
            ledger_path.write_text(json.dumps({"case_id": case_id, "status": "numerical_failure"}) + "\n", encoding="utf-8")  # Preserve explicit failed evidence exactly where the real builder does.
            raise RuntimeError("controlled native failure")  # Surface a nonzero campaign outcome while allowing later cases to execute.
        ledger_path.write_text(json.dumps({"case_id": case_id, "status": "complete", "levels": [0, 1]}) + "\n", encoding="utf-8")  # Publish a compact deterministic successful ladder fixture.
        reference_b_path.write_text(json.dumps({"case_id": case_id, "reference": {"U_total": 2.0}}) + "\n", encoding="utf-8")  # Publish a distinct compact final-reference fixture.
        reference_a = Reference(U_total=1.9, qoi=1.0, n_equations=10, n_elems=5, h_ref=0.1)  # Construct a valid coarser accepted reference object.
        reference_b = Reference(U_total=2.0, qoi=1.0, n_equations=20, n_elems=10, h_ref=0.08)  # Construct a valid finer common Reference B object.
        return ReferenceBuildOutcome(reference_a=reference_a, reference_b=reference_b, a_level=0, b_level=1, ledger_path=ledger_path, from_cache=from_cache, qualification=qualification, authorization=UNQUALIFIED_AUTHORIZATION if not qualification else None)  # Match qualified or operational immutable outcome semantics exactly.

    def verify_fn(reference_root: Path, *, case_id: str, problem: Any, config: Any, regenerate_meshes: bool, allow_unqualified: bool, expedited_levels: int | None) -> dict[str, Any]:  # Return an explicit positive strict or amended verification receipt solve-free.
        del reference_root, problem, config, expedited_levels  # Mark path, scientific, and optional depth fixture inputs as intentionally unused.
        assert regenerate_meshes is False  # Require campaign verification to avoid extra meshing or solving.
        reference_status = "complete" if qualification else "complete_unqualified"  # Reproduce the production terminal status for this fixture's qualification state.
        return {"schema": "fixture-reference-verification-v1", "case_id": case_id, "status": reference_status, "qualification": qualification, "authorization": UNQUALIFIED_AUTHORIZATION if allow_unqualified else None, "execution_amendment": {"fixture": True} if allow_unqualified else None, "original_convergence_gate": {"agreement_rtol": 0.005, "passed": qualification}, "passed": True}  # Bind deterministic qualification and distinct verifier-integrity success to this exact case.

    return runner_factory, ensure_fn, verify_fn, runner_calls  # Return all injected behavior and observable runner isolation evidence.


def test_plan_and_dry_run_cover_exact_order_without_runner_or_writes(tmp_path: Path) -> None:  # Prove solve-free planning includes every case once and performs no runner side effects.
    repository, campaign = _campaign_fixture(tmp_path)  # Create the exact authenticated manifest fixture.
    runner_factory, ensure_fn, verify_fn, runner_calls = _fixture_callbacks()  # Create callbacks that would reveal any accidental execution.
    result = run_reference_campaign(campaign, repository, "train", dry_run=True, runner_factory=runner_factory, ensure_fn=ensure_fn, verify_fn=verify_fn)  # Rehearse the complete train split without native work.
    assert result["status"] == "planned" and result["solve_count"] == 0  # Require explicit solve-free terminal plan semantics.
    assert result["expected_case_count"] == 24 and result["case_ids"] == sorted(result["case_ids"])  # Require all 24 train cases in ascending identity order.
    assert len(set(result["case_ids"])) == 24 and result["subset_allowed"] is False  # Reject duplicates and every hidden subsetting path.
    assert runner_calls == []  # Prove the runner factory was never instantiated.
    assert not Path(result["work_root"]).exists() and not Path(result["summary_directory"]).exists()  # Prove dry-run created no native or aggregate directories.


def test_validation_campaign_aggregates_full_hashes_and_archives_resume(tmp_path: Path) -> None:  # Prove independent runners, exact checksums, cache reuse, and append-only attempt history.
    repository, campaign = _campaign_fixture(tmp_path)  # Create an isolated formal campaign layout.
    runner_factory, ensure_fn, verify_fn, runner_calls = _fixture_callbacks()  # Create deterministic success callbacks for all validation cases.
    first = run_reference_campaign(campaign, repository, "validation", runner_factory=runner_factory, ensure_fn=ensure_fn, verify_fn=verify_fn)  # Execute the complete eight-case split once.
    assert first["status"] == "complete" and first["completed_case_count"] == 8 and first["new_build_count"] == 8  # Require every case to produce a newly authenticated valid cache.
    assert runner_calls == first["case_ids"] and len(set(runner_calls)) == 8  # Require exactly one distinct per-case runner in ascending order.
    assert all(len(record["ledger"]["sha256"]) == 64 and len(record["reference_B"]["sha256"]) == 64 for record in first["case_results"])  # Require complete artifact SHA-256 values without display truncation.
    summary_path = Path(first["summary_path"])  # Resolve the persisted terminal aggregate.
    observed_digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()  # Recompute its exact-byte identity independently.
    assert observed_digest == first["summary_sha256"]  # Require terminal reporting to match persisted bytes exactly.
    assert Path(first["summary_checksum_path"]).read_text(encoding="ascii").split()[0] == observed_digest  # Require the standard checksum sidecar to authenticate the same full digest.
    runner_calls.clear()  # Isolate runner evidence for the automatic resume attempt.
    second = run_reference_campaign(campaign, repository, "validation", runner_factory=runner_factory, ensure_fn=ensure_fn, verify_fn=verify_fn)  # Re-run the complete split and require safe cache reuse.
    assert second["status"] == "complete" and second["attempt_number"] == 2 and second["cache_reuse_count"] == 8  # Require a new attempt record with all successful caches reused solve-free.
    assert Path(second["resume_history"][-1]["archive"]).is_file() and len(second["resume_history"][-1]["sha256"]) == 64  # Preserve the exact prior live summary in append-only history before replacement.


def test_campaign_retains_failure_continues_complete_split_and_returns_failed(tmp_path: Path) -> None:  # Prove one native failure cannot fabricate a reference or silently truncate later cases.
    repository, campaign = _campaign_fixture(tmp_path)  # Create an isolated exact manifest fixture.
    plan = build_reference_campaign_plan(campaign, repository, "validation")  # Recover the ascending case order solve-free.
    failing_case = plan["case_ids"][2]  # Select one deterministic interior case so both earlier and later execution are observable.
    runner_factory, ensure_fn, verify_fn, runner_calls = _fixture_callbacks(failing_case=failing_case)  # Inject a terminal failure for only that exact case.
    result = run_reference_campaign(campaign, repository, "validation", runner_factory=runner_factory, ensure_fn=ensure_fn, verify_fn=verify_fn)  # Execute the entire split despite the controlled numerical failure.
    failed = next(record for record in result["case_results"] if record["case_id"] == failing_case)  # Locate the retained failure evidence directly.
    assert result["status"] == "failed" and result["failed_case_count"] == 1 and result["completed_case_count"] == 7  # Require non-success aggregation without discarding valid peers.
    assert runner_calls == result["case_ids"]  # Require every later sorted case to execute after the retained failure.
    assert failed["reference_B"] is None and failed["ledger"] is not None  # Preserve the failed ledger while refusing to fabricate a compact Reference B.
    assert failed["failure"] == {"error_type": "RuntimeError", "error": "controlled native failure"}  # Retain the exact actionable error category and message.
    assert len(failed["native_logs"][0]["sha256"]) == 64  # Authenticate the independent per-case native failure log with full SHA-256.


def test_test_split_requires_external_evidence_and_freeze_gate_around_every_case(tmp_path: Path) -> None:  # Prove blind reference files are opened only under the committed exact-cache permission API.
    repository, campaign = _campaign_fixture(tmp_path)  # Create a repository-shaped campaign without real blind outputs.
    with pytest.raises(ReferenceCampaignError, match="outside the Git repository"):  # Reject any non-whitelisted test log or summary location in the worktree.
        build_reference_campaign_plan(campaign, repository, "test", work_root=campaign / "bad-native", summary_directory=tmp_path / "external-summary")  # Attempt to place native blind logs under the frozen repository.
    gate_calls: list[bool] = []  # Record the exact first-open and post-case permission sequence.

    def freeze_gate(root: Path, repo: Path, *, require_committed: bool, allow_postfreeze_test_references: bool) -> dict[str, Any]:  # Replace Git and environment checks while preserving the stable API contract.
        assert Path(root) == campaign.resolve() and Path(repo) == repository.resolve()  # Require the formal campaign and reviewed repository boundaries.
        assert require_committed is True  # Require every blind reference gate to demand the sealed committed freeze tag.
        gate_calls.append(bool(allow_postfreeze_test_references))  # Retain whether exact existing cache files were permitted.
        return {"TEST_NOT_RUN": True, "allow_unqualified_references": False, "expedited_reference_levels": None, "reference_execution_amendment": None, "freeze_git_ref": "refs/tags/wmvla-p1-freeze", "git": {"freeze_commit_sha": "a" * 40}}  # Return deterministic strict qualification and Git authorization evidence.

    runner_factory, ensure_fn, verify_fn, _runner_calls = _fixture_callbacks()  # Create solve-free successful blind-reference fixtures.
    external_work = tmp_path / "external-test-native"  # Resolve a native evidence root outside the fake Git worktree.
    external_summary = tmp_path / "external-test-summary"  # Resolve an aggregate evidence root outside the fake Git worktree.
    result = run_reference_campaign(campaign, repository, "test", work_root=external_work, summary_directory=external_summary, runner_factory=runner_factory, ensure_fn=ensure_fn, verify_fn=verify_fn, freeze_gate=freeze_gate)  # Exercise all 16 blind cases under injected committed-freeze authorization.
    assert result["status"] == "complete" and result["completed_case_count"] == 16  # Require the complete test split rather than a selectable subset.
    assert gate_calls == [False] + [True] * 16  # Require strict allow-false before the first cache and exact allow-true revalidation after each disclosed case.
    assert len(result["authorization_receipts"]) == 17 and all(len(receipt["receipt_sha256"]) == 64 for receipt in result["authorization_receipts"])  # Preserve every full-hash freeze permission receipt.
    assert not _is_repository_child(Path(result["summary_path"]), repository)  # Keep the aggregate outside the committed whitelist throughout blind reference generation.


def _is_repository_child(path: Path, repository: Path) -> bool:  # Independently test whether a focused-test artifact leaked into the fake worktree.
    resolved = path.resolve()  # Resolve symlinks and relative components for the candidate.
    boundary = repository.resolve()  # Resolve the fake Git boundary once.
    return resolved == boundary or boundary in resolved.parents  # Report both the root and any descendant as an invalid test evidence location.


def test_cli_dry_run_is_complete_and_rejects_case_selector(tmp_path: Path) -> None:  # Prove the public command plans solve-free and has no subset argument.
    repository, campaign = _campaign_fixture(tmp_path)  # Create the exact manifest and formal directory layout.
    command = [sys.executable, str(ROOT / "scripts" / "build_bridge_references.py"), "--split", "train", "--dry-run", "--root", str(campaign), "--repository", str(repository)]  # Build the explicit complete-split dry-run invocation.
    planned = subprocess.run(command, cwd=ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # Execute the real CLI without a native solve.
    assert planned.returncode == 0 and json.loads(planned.stdout)["expected_case_count"] == 24  # Require a valid complete train plan and successful process status.
    rejected = subprocess.run(command + ["--case-id", "BGD-001-forbidden"], cwd=ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # Attempt the intentionally unsupported case subset selector.
    assert rejected.returncode != 0 and "unrecognized arguments: --case-id" in rejected.stderr  # Require argparse to reject the subset before driver execution.


def test_expedited_campaign_requires_hashed_amendment_and_marks_every_operational_reference(tmp_path: Path) -> None:  # Prove the two-level nonblocking mode is disclosed prominently rather than masquerading as qualified.
    repository, campaign = _campaign_fixture(tmp_path)  # Create an isolated formal manifest layout first.
    with pytest.raises(ReferenceCampaignError, match="amendment is missing"):  # Require a durable disclosed authorization artifact in addition to the API flag.
        build_reference_campaign_plan(campaign, repository, "validation", allow_unqualified=True, expedited_levels=2)  # Attempt expedited planning before publishing the amendment fixture.
    amendment = campaign / "protocol" / "EXPEDITED_EXECUTION_AMENDMENT.md"  # Resolve the exact required protocol amendment filename.
    amendment.write_text(f"# Expedited execution\n\nAuthorization: `{UNQUALIFIED_AUTHORIZATION}`\n", encoding="utf-8")  # Publish the exact token in transparent deterministic fixture bytes.
    runner_factory, ensure_fn, verify_fn, _runner_calls = _fixture_callbacks(qualification=False)  # Create eight deterministic usable but unqualified validation outcomes.
    result = run_reference_campaign(campaign, repository, "validation", allow_unqualified=True, expedited_levels=2, runner_factory=runner_factory, ensure_fn=ensure_fn, verify_fn=verify_fn)  # Execute the entire explicitly amended split.
    assert result["status"] == "complete" and result["qualification"] is False  # Separate operational campaign completion from original-gate qualification.
    assert result["unqualified_reference_count"] == 8 and result["qualified_reference_count"] == 0 and result["contains_unqualified_references"] is True  # Make every operational fallback prominent in top-level aggregation.
    assert result["authorization"] == UNQUALIFIED_AUTHORIZATION and result["expedited_levels"] == 2  # Bind the exact user authorization and fixed two-level prefix in the summary.
    assert len(result["expedited_amendment"]["sha256"]) == 64 and result["expedited_amendment"]["path"] == str(amendment.resolve())  # Authenticate the exact amendment bytes and path with a full digest.
    assert all(record["reference_status"] == "complete_unqualified" and record["qualification"] is False for record in result["case_results"])  # Prevent any per-case operational record from being labelled converged or qualified.
    strict_only = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_bridge_references.py"), "--split", "validation", "--dry-run", "--root", str(campaign), "--repository", str(repository), "--expedited-levels", "2"], cwd=ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # Attempt the CLI prefix without its mandatory opt-in.
    assert strict_only.returncode != 0 and "requires --allow-unqualified" in strict_only.stderr  # Require explicit dual flags before even solve-free expedited planning.
    expedited_dry = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_bridge_references.py"), "--split", "validation", "--dry-run", "--root", str(campaign), "--repository", str(repository), "--allow-unqualified", "--expedited-levels", "2"], cwd=ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # Exercise the exact public expedited solve-free command.
    expedited_plan = json.loads(expedited_dry.stdout)  # Decode the machine-readable plan only after the command returns successfully.
    assert expedited_dry.returncode == 0 and expedited_plan["authorization"] == UNQUALIFIED_AUTHORIZATION  # Keep the exact amendment token at the top level rather than overwriting it with a null validation-only freeze receipt.
    assert expedited_plan["freeze_authorization"] is None and expedited_plan["expedited_amendment"]["authorization"] == UNQUALIFIED_AUTHORIZATION  # Report validation's inapplicable blind gate separately while retaining matching nested amendment evidence.
