"""Test deterministic four-way aggregation, failure rows, and fixed report conclusions."""  # Describe this no-solver analysis test module.
from __future__ import annotations  # Postpone annotation evaluation consistently with production code.
import json  # Inspect generated strict JSON artifacts and build synthetic status evidence.
from pathlib import Path  # Build isolated campaign trees under pytest temporary directories.
import pytest  # Assert strict reference-qualification acknowledgement failures.
from visionamr.bridge_case_manifest import write_case_manifest  # Generate the exact authenticated 24/8/16 protocol manifest.
from visionamr.vla.four_way_analysis import IncompleteEvidenceError, _cost_intersection, _reference_qualification_evidence, _report_lines, analyze_four_way, build_failure_matrix, build_rl_median_rows  # Exercise aggregation, reference acknowledgement, amortization, and fail-closed orchestration.
from visionamr.vla.four_way_benchmark import ALL_METHODS, BUDGETS  # Build the exact raw-job denominator receipt grid.
from visionamr.vla.four_way_stats import PRIMARY_COMPETITORS  # Build the exact primary machine-gate vocabulary.

def test_rl_pointwise_median_ranks_failed_seed_after_finite_values() -> None:  # Verify aggregation never selects the best global or per-point policy.
    rows = []  # Collect one complete single-case three-policy public grid.
    for budget in (30000, 60000, 120000):  # Cover every active-equation budget.
        for solves in (2, 3, 4, 6):  # Cover every true real-solve prefix.
            for method, value, ok in zip(("rl_seed0", "rl_seed1", "rl_seed2"), (0.8, 1.0, None), (True, True, False), strict=True):  # Provide two finite policies and one retained failure.
                rows.append({"case_id": "case", "equation_budget": budget, "solves": solves, "method": method, "energy_error": value, "qoi_error": value, "energy_ok": ok, "qoi_ok": ok, "budget_violation": False})  # Preserve every source policy outcome.
    medians = build_rl_median_rows(list(reversed(rows)), ["case"])  # Aggregate deliberately reversed evidence pointwise.
    assert len(medians) == 12 and all(row["energy_error"] == 1.0 and row["qoi_error"] == 1.0 for row in medians)  # Select the worse finite outcome as the middle rank at every point.
    assert all(len(row["seed_energy_outcomes"]) == 3 for row in medians)  # Preserve all three source outcomes for audit.

def test_failure_matrix_keeps_unaffected_prefix_before_typed_native_failure(tmp_path: Path) -> None:  # Verify numerical failure boundaries do not erase earlier completed K values.
    method_dir = tmp_path / "test" / "case" / "30000" / "world_model_vla"  # Reproduce one raw protocol trajectory directory.
    method_dir.mkdir(parents=True)  # Create the isolated status fixture path.
    status = {"completed": False, "failure": {"category": "calculix_numerical", "exception_type": "CalculiXExecutionError", "message": "rc=1", "calculix_returncode": 1}}  # Build one typed failure after a successful prefix.
    (method_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")  # Persist the transparent trajectory status.
    raw = [{"case_id": "case", "equation_budget": 30000, "method": "world_model_vla", "solves": 2, "energy_ok": True, "qoi_ok": True, "energy_error": 0.5, "qoi_error": 0.6, "budget_violation": False, "failure_affects_prefix": False, "successful_solves_available": 2}, {"case_id": "case", "equation_budget": 30000, "method": "world_model_vla", "solves": 3, "energy_ok": False, "qoi_ok": False, "energy_error": None, "qoi_error": None, "budget_violation": False, "failure_affects_prefix": True, "successful_solves_available": 2}]  # Provide K=2 before and K=3 at the failed attempt.
    matrix = build_failure_matrix(tmp_path, raw, [], [])  # Build the transparent public-point failure table.
    assert matrix[0]["category"] == "ok" and matrix[1]["category"] == "calculix_numerical"  # Preserve the earlier valid prefix and fail only affected later prefixes.
    assert matrix[1]["calculix_returncode"] == 1  # Retain native return-code evidence in the delivered matrix.

def test_report_starts_with_exact_seven_machine_results() -> None:  # Verify no title or qualitative prose precedes the fixed conclusion block.
    final = {"DORFLER_SAFE": True, "BEAT_LOCAL_PREDICTION": False, "BEAT_SUPERVISED": True, "BEAT_RL": False, "WORLD_MODEL_MECHANISM": True, "ONLINE_TIME_ACCEPTABLE": False, "OVERALL_WIN": False}  # Build a mixed machine decision.
    primary = {name: {"gates": {"example": name != "local_prediction"}} for name in PRIMARY_COMPETITORS}  # Build transparent atomic primary gates.
    lines = _report_lines(final, primary, {"gates": {"safe": True}}, {"gates": {"mechanism": True}}, {"gates": {"time": False}}, {"case_count": 16, "raw_prefix_row_count": 100, "expected_raw_prefix_row_count": 100, "missing_job_count": 0, "boundary_issue_count": 0})  # Render the fixed report with concrete failure details.
    assert [line.split("=")[0].strip() for line in lines[:7]] == ["DORFLER_SAFE", "BEAT_LOCAL_PREDICTION", "BEAT_SUPERVISED", "BEAT_RL", "WORLD_MODEL_MECHANISM", "ONLINE_TIME_ACCEPTABLE", "OVERALL_WIN"]  # Pin exact conclusion order.
    assert all("#" not in line for line in lines[:7]) and lines[7] == ""  # Forbid any heading before or inside the seven machine lines.
    assert lines[8].strip() == "REFERENCE_QUALIFIED     = false"  # Keep a non-qualified denominator prominent immediately after, but never inside, the fixed seven-line gate block.

def test_cost_intersection_handles_crossing_and_parallel_lines() -> None:  # Verify real and integer amortization never divide by a parallel or misclassify an equality.
    later = _cost_intersection(100.0, 1.0, 0.0, 3.0)  # Make WM recover a larger training cost at the exact real deployment n=50.
    assert later["status"] == "crosses" and later["real_intersection_n"] == 50.0  # Preserve the exact real equality classification.
    assert later["wm_no_more_expensive_integer_range"] == {"min_n": 50, "max_n": None, "kind": "lower_bounded"}  # Include the equality deployment in the minimal integer range.
    parallel_better = _cost_intersection(1.0, 2.0, 3.0, 2.0)  # Give WM a lower intercept under identical online slopes.
    parallel_worse = _cost_intersection(3.0, 2.0, 1.0, 2.0)  # Give WM a higher intercept under identical online slopes.
    assert parallel_better["parallel"] is True and parallel_better["status"] == "always"  # Report all integer deployments without a fabricated single crossing.
    assert parallel_worse["parallel"] is True and parallel_worse["status"] == "never"  # Report an empty integer advantage range without division.

def test_unqualified_reference_requires_explicit_analysis_acknowledgement(tmp_path: Path) -> None:  # Verify operational authorization never silently becomes a qualified denominator.
    test_root = tmp_path / "test"  # Resolve an isolated synthetic campaign evidence root.
    test_root.mkdir(parents=True)  # Create only the temporary blind-layout fixture.
    started = {"allow_unqualified_references": True, "expedited_reference_levels": 2, "reference_execution_amendment_sha256": "b" * 64, "REFERENCE_QUALIFIED": False}  # Disclose the frozen operational waiver, depth, and authorization bytes before all synthetic method receipts.
    (test_root / "TEST_STARTED.json").write_text(json.dumps(started), encoding="utf-8")  # Persist the minimal qualification fields consumed by the focused helper.
    for budget in BUDGETS:  # Cover every independently run public equation budget.
        for method in ALL_METHODS:  # Cover every raw WM, baseline, seed, and safety trajectory.
            method_root = test_root / "case" / str(budget) / method  # Resolve the exact protocol-layout method directory.
            method_root.mkdir(parents=True)  # Create the isolated receipt fixture path.
            reference = {"usage": "posthoc_only", "used_online": False, "qualification": False, "status": "complete_unqualified", "authorization": "user_authorized_nonblocking_2026-08-30", "expedited_levels": 2, "execution_amendment": {"threshold_unchanged": True, "expedited_levels": 2}, "reference_execution_amendment_sha256": "b" * 64, "reference_b_sha256": "a" * 64}  # Preserve failed qualification, amendment identity, and exact nonblocking schedule separately.
            (method_root / "records.json").write_text(json.dumps({"reference_b": reference}), encoding="utf-8")  # Persist one transparent posthoc-only denominator receipt.
    with pytest.raises(IncompleteEvidenceError, match="--allow-unqualified-references"):  # Keep strict qualification as the analyzer default.
        _reference_qualification_evidence(tmp_path, ["case"], allow_unqualified_references=False, allow_incomplete=False)  # Attempt to consume the operational B without acknowledgement.
    evidence = _reference_qualification_evidence(tmp_path, ["case"], allow_unqualified_references=True, allow_incomplete=False)  # Acknowledge the already disclosed frozen waiver explicitly.
    assert evidence["available"] is True and evidence["REFERENCE_QUALIFIED"] is False  # Permit analysis while refusing any convergence claim.
    assert evidence["unqualified_case_ids"] == ["case"] and evidence["job_receipt_count"] == len(BUDGETS) * len(ALL_METHODS)  # Preserve complete affected-case and raw-job coverage.
    tampered_path = test_root / "case" / str(BUDGETS[0]) / ALL_METHODS[0] / "records.json"  # Select one exact raw receipt to simulate authorization-pointer substitution.
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))  # Parse the isolated fixture without changing any other evidence.
    tampered["reference_b"]["reference_execution_amendment_sha256"] = "c" * 64  # Substitute a well-formed but different protected amendment identity.
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")  # Persist the tampered posthoc receipt for fail-closed analysis.
    with pytest.raises(IncompleteEvidenceError, match="incomplete"):  # Require exact amendment identity instead of accepting authorization text alone.
        _reference_qualification_evidence(tmp_path, ["case"], allow_unqualified_references=True, allow_incomplete=False)  # Audit the altered grid under the otherwise valid explicit acknowledgement.

def test_allow_incomplete_analysis_publishes_fail_closed_required_artifacts(tmp_path: Path) -> None:  # Verify diagnostic recovery never converts missing raw evidence into a scientific win.
    protocol_root = tmp_path / "protocol"  # Resolve the isolated campaign protocol directory.
    manifest_path, _sidecar, _digest = write_case_manifest(protocol_root)  # Persist the exact authenticated 48-case design and sidecar.
    result = analyze_four_way(tmp_path, manifest_path, allow_incomplete=True)  # Analyze an intentionally unstarted and empty campaign in fail-closed mode.
    assert result["complete"] is False and result["final_gate"]["OVERALL_WIN"] is False  # Refuse every scientific success claim on missing evidence.
    required = ("primary_results.csv", "pairwise_ratios.csv", "bootstrap.json", "prediction_calibration.csv", "failure_matrix.csv", "amortized_cost.json", "final_gate.json", "artifact_index.json")  # Name the mandatory aggregate delivery files including cost accounting.
    assert all((tmp_path / "aggregate" / name).is_file() for name in required)  # Publish every required diagnostic artifact despite incomplete raw evidence.
    first_lines = (tmp_path / "EXECUTION_REPORT.md").read_text(encoding="utf-8").splitlines()[:7]  # Read only the fixed first conclusion block.
    assert len(first_lines) == 7 and all(line.endswith("false") for line in first_lines)  # Fail all seven machine conclusions explicitly and first.
    amortized = json.loads((tmp_path / "aggregate" / "amortized_cost.json").read_text(encoding="utf-8"))  # Inspect the diagnostic-mode absence receipt.
    assert amortized["available"] is False and "FileNotFoundError" in amortized["reason"]  # Never fabricate zero training or online cost on incomplete evidence.
