"""Focused contract tests for the frozen WMVLA-4WAY-P1 statistics module."""  # State the test suite's exact scope.
from __future__ import annotations  # Permit modern annotations consistently with the implementation.
from dataclasses import replace  # Create immutable-record variants for negative gate tests.
import json  # Verify strict JSON serialization without NaN or infinity tokens.
import math  # Verify finite deterministic failure scores.
import pytest  # Assert explicit validation failures for malformed protocol evidence.
from visionamr.vla.four_way_stats import DorflerObservation  # Import independent-Dorfler paired evidence.
from visionamr.vla.four_way_stats import FallbackEvidence  # Import exact-Dorfler fallback receipts.
from visionamr.vla.four_way_stats import MechanismObservation  # Import fixed-point mechanism-ablation evidence.
from visionamr.vla.four_way_stats import PairwiseObservation  # Import primary competitor paired evidence.
from visionamr.vla.four_way_stats import PRIMARY_COMPETITORS  # Import the frozen final-formula competitor set.
from visionamr.vla.four_way_stats import PRIMARY_OPERATING_POINTS  # Import the six preregistered operating points.
from visionamr.vla.four_way_stats import TargetDominanceCheck  # Import nodewise target-dominance receipts.
from visionamr.vla.four_way_stats import TimeObservation  # Import separated online timing evidence.
from visionamr.vla.four_way_stats import build_pairwise_ratio_rows  # Import deterministic long-form ratio generation.
from visionamr.vla.four_way_stats import case_bootstrap_geometric_ci  # Import the deterministic case-level bootstrap.
from visionamr.vla.four_way_stats import evaluate_dorfler_safety  # Import the complete safety conjunction.
from visionamr.vla.four_way_stats import evaluate_final_gate  # Import the exact protocol-wide formula.
from visionamr.vla.four_way_stats import evaluate_online_time  # Import the separate engineering-efficiency gate.
from visionamr.vla.four_way_stats import evaluate_primary_competitor  # Import the seven-term primary win gate.
from visionamr.vla.four_way_stats import evaluate_world_model_mechanism  # Import the conservative attribution gate.
from visionamr.vla.four_way_stats import failure_aware_median_error  # Import the pointwise three-seed RL median helper.
from visionamr.vla.four_way_stats import flatten_csv_row  # Import deterministic nested-report CSV flattening.
from visionamr.vla.four_way_stats import json_safe  # Import strict JSON-number validation.
from visionamr.vla.four_way_stats import score_error_ratio  # Import finite deterministic failure scoring.
def _case_ids() -> tuple[str, ...]:  # Build the frozen number of synthetic blind-case identifiers.
    return tuple(f"test_{index:02d}" for index in range(16))  # Return sixteen deterministically ordered case labels.
def _primary_observations(competitor: str = "local_prediction", wm_energy: float | None = 0.90, baseline_energy: float | None = 1.0, wm_ok: bool = True, baseline_ok: bool = True) -> list[PairwiseObservation]:  # Build a complete synthetic primary Cartesian product.
    rows = []  # Accumulate one observation per case and preregistered point.
    for case_id in _case_ids():  # Cover every synthetic blind case.
        for solves, budget in PRIMARY_OPERATING_POINTS:  # Cover all six frozen primary operating points.
            rows.append(PairwiseObservation(case_id=case_id, competitor=competitor, solves=solves, equation_budget=budget, wm_energy_error=wm_energy, competitor_energy_error=baseline_energy, wm_qoi_error=0.98, competitor_qoi_error=1.0, wm_energy_ok=wm_ok, competitor_energy_ok=baseline_ok, wm_proactive_action=True))  # Add a paired passing record with certified proactive coverage.
    return rows  # Return the exact sixteen-by-six grid.
def _dorfler_observations() -> list[DorflerObservation]:  # Build complete passing independent-Dorfler evidence.
    rows = []  # Accumulate all paired safety comparisons.
    for case_id in _case_ids():  # Cover every blind case.
        for solves, budget in PRIMARY_OPERATING_POINTS:  # Cover all primary operating points.
            rows.append(DorflerObservation(case_id=case_id, solves=solves, equation_budget=budget, wm_energy_error=1.01, dorfler_energy_error=1.0))  # Stay inside aggregate and pointwise safety tolerances.
    return rows  # Return complete safety-grid evidence.
def _dominance_checks(passed: bool = True) -> list[TargetDominanceCheck]:  # Build one unique executed-target receipt per case.
    return [TargetDominanceCheck(case_id=case_id, action_id=f"{case_id}:action:1", passed=passed) for case_id in _case_ids()]  # Certify or reject all synthetic nodewise target checks together.
def _time_observations(vlm_calls: int = 1) -> list[TimeObservation]:  # Build separated passing online timing evidence.
    return [TimeObservation(case_id=case_id, vlm_calls=vlm_calls, visual_partition_s=0.10, world_model_s=0.10, parameter_tools_s=0.10, gmsh_s=0.20, calculix_s=4.50, lp_online_total_s=4.50) for case_id in _case_ids()]  # Keep non-solver share at ten percent and total ratio near 1.11.
def _mechanism_observations() -> list[MechanismObservation]:  # Build conservative passing mechanism-attribution evidence.
    return [MechanismObservation(case_id=case_id, wm_full_energy_error=0.80, wm_h1_energy_error=1.0, random_safe_median_energy_error=0.90, certified_proactive_actions=1, executed_proactive_actions=1) for case_id in _case_ids()]  # Make full multi-step planning uniformly beat both controls under matched contracts.
def test_failure_scoring_is_finite_symmetric_and_auditable() -> None:  # Verify every failure combination has a deterministic finite score.
    success = score_error_ratio(0.5, 1.0)  # Score an ordinary successful pair.
    numerator_failure = score_error_ratio(None, 1.0, False, True)  # Score a WM-only numerical failure.
    denominator_failure = score_error_ratio(0.5, None, True, False)  # Score a competitor-only numerical failure.
    shared_failure = score_error_ratio(None, None, False, False)  # Score a shared numerical failure.
    malformed = score_error_ratio(math.nan, 1.0)  # Verify nonfinite declared success is retained as failure.
    assert success.ratio == pytest.approx(0.5) and success.status == "success"  # Preserve an ordinary exact paired ratio.
    assert numerator_failure.ratio == pytest.approx(10.0) and numerator_failure.status == "numerator_failed"  # Apply the frozen tenfold loss.
    assert denominator_failure.ratio == pytest.approx(0.1) and denominator_failure.status == "denominator_failed"  # Apply the symmetric reciprocal win.
    assert shared_failure.ratio == pytest.approx(1.0) and shared_failure.status == "both_failed"  # Treat shared failure as a tie while retaining both flags.
    assert malformed.status == "numerator_failed" and math.isfinite(malformed.ratio)  # Keep malformed evidence finite and explicitly classified.
    floored = score_error_ratio(1.0, 0.0)  # Exercise the widest energy-error ratio implied by the sole preregistered successful-error floor.
    assert floored.ratio == pytest.approx(1.0e300) and floored.numerically_clipped is False  # Preserve the exact floor-derived ratio without an undeclared `1e100` clamp.
def test_rl_median_retains_seed_failures_without_best_seed_selection() -> None:  # Verify pointwise three-seed median semantics.
    assert failure_aware_median_error([0.8, 1.0, None], [True, True, False]) == pytest.approx(1.0)  # Make one failed seed rank worse than both successful policies.
    assert failure_aware_median_error([0.8, None, None], [True, False, False]) is None  # Make two failed seeds produce a failed median policy.
    with pytest.raises(ValueError, match="exactly three"):  # Require all three frozen RL seeds.
        failure_aware_median_error([0.8, 1.0])  # Reject an incomplete two-seed comparison.
def test_case_bootstrap_is_deterministic_and_resamples_case_logs() -> None:  # Verify the implementation's fixed cluster-bootstrap behavior.
    case_logs = [math.log(0.8), math.log(0.9), math.log(1.1), math.log(1.2)]  # Define four heterogeneous case-cluster effects.
    first = case_bootstrap_geometric_ci(case_logs, seed=73, replicates=256)  # Compute a small deterministic test interval.
    second = case_bootstrap_geometric_ci(case_logs, seed=73, replicates=256)  # Repeat with identical frozen inputs.
    assert first == second  # Require bitwise-stable bootstrap metadata and percentiles.
    assert first["resampling_unit"] == "case" and first["replicates"] == 256  # Disclose case-level clustering and replicate count.
    assert first["lower"] <= first["point_estimate"] <= first["upper"]  # Require a coherent percentile interval for this heterogeneous sample.
def test_primary_competitor_gate_passes_complete_strong_evidence_and_is_serializable() -> None:  # Verify all seven primary gates and strict output safety.
    observations = _primary_observations()  # Build a uniform ten-percent energy improvement over local prediction.
    result = evaluate_primary_competitor(observations, "local_prediction")  # Evaluate the complete frozen grid.
    ratio_rows = build_pairwise_ratio_rows(list(reversed(observations)), "local_prediction")  # Verify long-form output is sorted independently of input order.
    assert result["passed"] is True and all(result["gates"].values())  # Require the exact seven-term conjunction to pass.
    assert result["energy"]["geometric_mean_ratio"] == pytest.approx(0.90)  # Preserve the deterministic all-pair geometric ratio.
    assert result["energy"]["bootstrap_ci_95"]["upper"] == pytest.approx(0.90)  # Preserve the zero-heterogeneity case-bootstrap endpoint.
    assert ratio_rows[0]["case_id"] == "test_00" and ratio_rows[-1]["case_id"] == "test_15"  # Emit stable case order for CSV evidence.
    json.dumps(result, allow_nan=False)  # Require strict standards-compliant JSON serialization.
    flattened = flatten_csv_row(result)  # Flatten the nested result for csv.DictWriter use.
    assert flattened["passed"] == "true" and all(isinstance(value, str) for value in flattened.values())  # Require scalar string CSV cells only.
def test_primary_grid_rejects_missing_or_mixed_evidence() -> None:  # Verify incomplete experiments cannot pass vacuously.
    incomplete = _primary_observations()[:-1]  # Remove one preregistered case-point pair.
    with pytest.raises(ValueError, match="incomplete primary grid"):  # Require exact Cartesian-product coverage.
        evaluate_primary_competitor(incomplete, "local_prediction")  # Reject the incomplete primary evidence.
    mixed = _primary_observations()  # Build otherwise complete evidence.
    mixed[0] = replace(mixed[0], competitor="supervised")  # Contaminate one denominator label.
    with pytest.raises(ValueError, match="mixed competitor"):  # Require a single scientifically unambiguous comparator.
        evaluate_primary_competitor(mixed, "local_prediction")  # Reject mixed-method aggregation.
def test_primary_gate_retains_method_failures_and_fails() -> None:  # Verify one-sided WM failures are counted instead of dropped.
    observations = _primary_observations(wm_energy=None, wm_ok=False)  # Make WM fail at every primary point while the competitor succeeds.
    result = evaluate_primary_competitor(observations, "local_prediction")  # Score all retained failures with the frozen finite penalty.
    assert result["passed"] is False  # Prevent a numerically failed method from winning.
    assert result["energy"]["failure_counts"]["numerator_failed"] == 16 * 6  # Count all ninety-six retained WM failures.
    assert result["energy"]["geometric_mean_ratio"] == pytest.approx(10.0)  # Apply the declared finite deterministic failure loss.
def test_dorfler_safety_requires_ratios_nodewise_receipts_and_fallback_execution() -> None:  # Verify all structural and empirical safety terms.
    fallback = [FallbackEvidence(case_id="test_00", trigger="uncertainty", dorfler_executed=True)]  # Record one correctly executed required fallback.
    passing = evaluate_dorfler_safety(_dorfler_observations(), _dominance_checks(), fallback)  # Evaluate complete passing safety evidence.
    assert passing["passed"] is True and all(passing["gates"].values())  # Require all five safety terms to pass.
    failed_receipt = _dominance_checks()  # Build otherwise valid target receipts.
    failed_receipt[0] = replace(failed_receipt[0], passed=False)  # Introduce one nodewise Dorfler-dominance breach.
    failing = evaluate_dorfler_safety(_dorfler_observations(), failed_receipt, [replace(fallback[0], dorfler_executed=False)])  # Also withhold the required exact fallback.
    assert failing["passed"] is False  # Fail safety on either hard structural or fallback breach.
    assert failing["gates"]["nodewise_target_dominance"] is False  # Identify the exact structural failure term.
    assert failing["gates"]["required_fallbacks_executed"] is False  # Identify the exact fallback failure term.
def test_online_time_gate_is_separate_and_enforces_one_vlm_call() -> None:  # Verify the three frozen engineering-efficiency thresholds.
    passing = evaluate_online_time(_time_observations())  # Evaluate complete separated timing evidence.
    failing = evaluate_online_time(_time_observations(vlm_calls=2))  # Repeat with one extra semantic model call per case.
    assert passing["passed"] is True  # Pass valid one-call, low-overhead, bounded-total timings.
    assert passing["median_non_solver_share"] == pytest.approx(0.10)  # Compute the non-solver fraction from separated components.
    assert failing["passed"] is False and failing["gates"]["vlm_call_limit"] is False  # Enforce the per-case single-call contract.
def test_mechanism_gate_conservatively_requires_multistep_and_informed_superiority() -> None:  # Verify numerical attribution beyond proactive-action counting alone.
    passing = evaluate_world_model_mechanism(_mechanism_observations())  # Evaluate full versus horizon-one and random-safe controls.
    assert passing["passed"] is True and all(passing["gates"].values())  # Require proactive, certification, fairness, and both superiority proofs.
    assert passing["full_vs_h1"]["bootstrap_ci_95"]["upper"] < 1.0  # Require case-bootstrap evidence for multi-step value.
    contaminated = _mechanism_observations()  # Build otherwise passing mechanism evidence.
    contaminated[0] = replace(contaminated[0], common_uniform_probe=False)  # Introduce a first-mesh advantage in one case.
    failing = evaluate_world_model_mechanism(contaminated)  # Re-evaluate the conservative attribution conjunction.
    assert failing["passed"] is False and failing["gates"]["common_uniform_probe"] is False  # Reject benefits confounded by unequal probes.
def test_final_formula_excludes_time_but_requires_all_five_scientific_terms() -> None:  # Verify the protocol section-one formula exactly.
    primary = {name: {"passed": True} for name in PRIMARY_COMPETITORS}  # Mark all three primary competitors beaten.
    result = evaluate_final_gate(primary, {"passed": True}, {"passed": True}, {"passed": False})  # Fail only the separately reported online-time gate.
    assert result["OVERALL_WIN"] is True and result["ONLINE_TIME_ACCEPTABLE"] is False  # Keep time visible without adding an undeclared overall conjunct.
    assert result["online_time_is_overall_term"] is False  # Make the scientific boundary machine-readable.
    failed_primary = dict(primary)  # Copy the three primary machine decisions.
    failed_primary["rl_median"] = {"passed": False}  # Make the RL comparison fail.
    failed = evaluate_final_gate(failed_primary, {"passed": True}, {"passed": True}, {"passed": True})  # Re-evaluate with every other term passing.
    assert failed["OVERALL_WIN"] is False and failed["failed_overall_terms"] == ["BEAT_RL"]  # Require the exact five-term conjunction and precise failure label.
def test_json_boundary_rejects_nonfinite_numbers() -> None:  # Verify strict JSON and CSV safety never silently stringifies NaN.
    with pytest.raises(ValueError, match="nonfinite"):  # Require an explicit report-boundary error.
        json_safe({"invalid": math.inf})  # Reject infinity before serialization.
