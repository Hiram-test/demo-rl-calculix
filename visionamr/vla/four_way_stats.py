"""Frozen deterministic statistics and decision gates for protocol WMVLA-4WAY-P1."""  # State the module's single protocol-scoring responsibility.
from __future__ import annotations  # Permit modern type annotations without runtime evaluation.
from dataclasses import asdict, dataclass, is_dataclass  # Define immutable input records and safely serialize them.
import json  # Encode nested values into stable CSV cells when flattening reports.
import math  # Compute logarithmic ratios, geometric means, and finite-value checks.
from collections.abc import Mapping, Sequence  # Describe accepted read-only collections precisely.
from typing import Any  # Type the recursive JSON-safety boundary without constraining callers.
PROTOCOL_ID = "WMVLA-4WAY-P1"  # Freeze the protocol identifier before any benchmark result is inspected.
PRIMARY_OPERATING_POINTS = ((2, 30000), (2, 60000), (3, 60000), (4, 60000), (4, 120000), (6, 120000))  # Freeze the six preregistered primary operating points.
PRIMARY_COMPETITORS = ("local_prediction", "supervised", "rl_median")  # Freeze the three methods that WM-VLA must beat independently.
EXPECTED_TEST_CASES = 16  # Freeze the blind-test case count required by the protocol.
BOOTSTRAP_SEED = 20260830  # Freeze the case-bootstrap seed independently of observation order.
BOOTSTRAP_REPLICATES = 10000  # Freeze a sufficiently resolved percentile-bootstrap sample size.
BOOTSTRAP_CONFIDENCE = 0.95  # Freeze the two-sided confidence level used by every superiority gate.
FAILURE_RATIO = 10.0  # Score a one-sided numerical failure as a finite tenfold loss for log aggregation.
ERROR_FLOOR = 1.0e-300  # Keep exact-zero successful errors representable in logarithmic ratios.
PRIMARY_ENERGY_LIMIT = 0.95  # Require at least five percent aggregate energy-error improvement.
PRIMARY_BOOTSTRAP_UPPER_LIMIT = 1.0  # Require the paired case-bootstrap upper bound to show strict superiority.
PRIMARY_WIN_FRACTION_MIN = 0.60  # Require broad improvement across case and operating-point pairs.
PRIMARY_P95_LIMIT = 1.15  # Bound the upper-tail paired regression accepted by a primary win.
PRIMARY_QOI_LIMIT = 1.05  # Prevent an energy-error win from hiding material QoI degradation.
PROACTIVE_CASE_FRACTION_MIN = 0.75  # Require proactive world-model execution on at least three quarters of cases.
DORFLER_AGGREGATE_LIMIT = 1.02  # Freeze the aggregate independent-Dorfler safety regression allowance.
DORFLER_POINT_LIMIT = 1.15  # Freeze the worst primary-point independent-Dorfler regression allowance.
TIME_NON_SOLVER_SHARE_LIMIT = 0.15  # Freeze the median online non-solver overhead share allowance.
TIME_VLM_CALL_LIMIT = 1  # Freeze the maximum semantic-partition model calls per test case.
TIME_LP_RATIO_LIMIT = 1.25  # Freeze the aggregate WM-to-local-prediction online wall-clock allowance.
MECHANISM_RATIO_LIMIT = 1.0  # Require strict full-model superiority over each mechanism-control comparator.
_MASK_64 = (1 << 64) - 1  # Constrain the deterministic bootstrap mixer to unsigned 64-bit arithmetic.
@dataclass(frozen=True)  # Prevent result-dependent mutation of a primary paired observation.
class PairwiseObservation:  # Represent one competitor comparison at one preregistered case and operating point.
    case_id: str  # Identify the blind-test geometry instance.
    competitor: str  # Identify local prediction, supervised learning, or pointwise RL median.
    solves: int  # Record the real CalculiX solve-prefix limit K.
    equation_budget: int  # Record the effective-equation budget B.
    wm_energy_error: float | None  # Store WM-VLA's best budget-feasible energy error or no value on failure.
    competitor_energy_error: float | None  # Store the competitor's paired energy error or no value on failure.
    wm_qoi_error: float | None  # Store WM-VLA's paired QoI error or no value on failure.
    competitor_qoi_error: float | None  # Store the competitor's paired QoI error or no value on failure.
    wm_energy_ok: bool = True  # Mark whether WM-VLA produced a valid energy-error result.
    competitor_energy_ok: bool = True  # Mark whether the competitor produced a valid energy-error result.
    wm_qoi_ok: bool = True  # Mark whether WM-VLA produced a valid QoI-error result.
    competitor_qoi_ok: bool = True  # Mark whether the competitor produced a valid QoI-error result.
    wm_budget_violation: bool = False  # Preserve any WM-VLA budget violation as an explicit hard-gate event.
    wm_proactive_action: bool = False  # Record whether this case had an executed certified proactive action by this prefix.
@dataclass(frozen=True)  # Keep each failure-aware ratio score immutable after classification.
class RatioScore:  # Return both the finite ratio and its auditable success or failure classification.
    ratio: float  # Store the finite positive value used in aggregate log statistics.
    status: str  # Explain whether the pair succeeded, failed on one side, or failed on both sides.
    numerator_failed: bool  # Preserve whether the evaluated method failed for this metric.
    denominator_failed: bool  # Preserve whether the comparison method failed for this metric.
    numerically_clipped: bool  # Disclose rare numeric-range clipping rather than silently hiding it.
@dataclass(frozen=True)  # Prevent mutation of independent-Dorfler comparison observations.
class DorflerObservation:  # Represent one WM-versus-Dorfler energy comparison at a primary point.
    case_id: str  # Identify the blind-test geometry instance.
    solves: int  # Record the real solve-prefix limit K.
    equation_budget: int  # Record the effective-equation budget B.
    wm_energy_error: float | None  # Store WM-VLA's budget-feasible energy error or no value on failure.
    dorfler_energy_error: float | None  # Store independent Dorfler's paired energy error or no value on failure.
    wm_ok: bool = True  # Mark whether WM-VLA produced a valid energy result.
    dorfler_ok: bool = True  # Mark whether independent Dorfler produced a valid energy result.
    wm_budget_violation: bool = False  # Preserve any actual WM-VLA budget violation as a hard failure.
@dataclass(frozen=True)  # Keep every nodewise target-field receipt immutable for auditability.
class TargetDominanceCheck:  # Represent one executed WM target's structural no-coarsening certificate.
    case_id: str  # Identify the case that executed the target size field.
    action_id: str  # Identify the exact hashed or logged action receipt.
    passed: bool  # Record whether h_WM was no larger than h_D at every previous-grid node.
@dataclass(frozen=True)  # Keep fallback evidence immutable after the benchmark run.
class FallbackEvidence:  # Represent one event for which exact Dorfler fallback was required.
    case_id: str  # Identify the case that triggered the fallback.
    trigger: str  # Preserve the nonempty rejection, distrust, regression, or budget trigger.
    dorfler_executed: bool  # Record whether the executable action actually became pure Dorfler.
@dataclass(frozen=True)  # Prevent timing evidence from being adjusted after online execution.
class TimeObservation:  # Represent separated online timing for one WM and local-prediction case pair.
    case_id: str  # Identify the blind-test case.
    vlm_calls: int  # Count semantic-partition model calls made for this case.
    visual_partition_s: float  # Record visual or deterministic semantic partition time.
    world_model_s: float  # Record world-model rollout and scoring time.
    parameter_tools_s: float  # Record deterministic action-validation and parameter-tool time.
    gmsh_s: float  # Record real Gmsh remeshing and certification time.
    calculix_s: float  # Record real CalculiX solve time.
    lp_online_total_s: float  # Record the paired local-prediction total online time.
@dataclass(frozen=True)  # Prevent mechanism evidence from being changed after ablation results are known.
class MechanismObservation:  # Represent the fixed K=6 and B=60000 mechanism comparison for one case.
    case_id: str  # Identify the blind-test case.
    wm_full_energy_error: float | None  # Store complete multi-step WM-VLA energy error.
    wm_h1_energy_error: float | None  # Store otherwise-identical horizon-one energy error.
    random_safe_median_energy_error: float | None  # Store the pointwise median over five random-safe-extra seeds.
    wm_full_ok: bool = True  # Mark whether complete WM-VLA produced a valid result.
    wm_h1_ok: bool = True  # Mark whether the horizon-one ablation produced a valid result.
    random_safe_ok: bool = True  # Mark whether the random-safe median was valid.
    certified_proactive_actions: int = 0  # Count proactive executed actions with valid safety certificates.
    executed_proactive_actions: int = 0  # Count proactive actions actually materialized and solved.
    common_uniform_probe: bool = True  # Confirm that full and controls started from the identical uniform probe.
    matched_solve_budget: bool = True  # Confirm that benefit was not bought with extra solves or equations.
    competitor_isolation: bool = True  # Confirm that WM-VLA read no LP, supervised, RL, or future-reference output.
def _valid_nonnegative(value: float | None, declared_ok: bool) -> tuple[bool, float]:  # Classify an error while rejecting invalid nonfinite or negative values.
    if not declared_ok or value is None:  # Treat explicit method failure and missing values identically.
        return False, 0.0  # Return a neutral placeholder that is never scored as a success.
    numeric = float(value)  # Normalize supported numeric scalar types to a JSON-safe Python float.
    if not math.isfinite(numeric) or numeric < 0.0:  # Prevent NaN, infinity, and negative errors from entering statistics.
        return False, 0.0  # Classify malformed numeric results as retained method failures.
    return True, numeric  # Preserve every finite nonnegative successful error, including exact zero.
def score_error_ratio(numerator: float | None, denominator: float | None, numerator_ok: bool = True, denominator_ok: bool = True) -> RatioScore:  # Convert a paired metric into a finite auditable ratio.
    numerator_valid, numerator_value = _valid_nonnegative(numerator, numerator_ok)  # Validate the evaluated method's metric.
    denominator_valid, denominator_value = _valid_nonnegative(denominator, denominator_ok)  # Validate the comparator's metric.
    if not numerator_valid and not denominator_valid:  # Treat a shared failure as a tie rather than inventing an advantage.
        return RatioScore(1.0, "both_failed", True, True, False)  # Retain both failures with a neutral finite ratio.
    if not numerator_valid:  # Penalize a one-sided evaluated-method failure.
        return RatioScore(FAILURE_RATIO, "numerator_failed", True, False, False)  # Apply the preregistered finite failure loss.
    if not denominator_valid:  # Reward a valid evaluated method against a failed comparator symmetrically.
        return RatioScore(1.0 / FAILURE_RATIO, "denominator_failed", False, True, False)  # Apply the reciprocal finite failure win.
    numerator_log = math.log(max(numerator_value, ERROR_FLOOR))  # Represent zero and tiny successful numerator errors safely.
    denominator_log = math.log(max(denominator_value, ERROR_FLOOR))  # Represent zero and tiny successful denominator errors safely.
    raw_log_ratio = numerator_log - denominator_log  # Compute the ratio in log space to avoid overflow and underflow.
    try:  # Convert the exact preregistered log ratio back to its unmodified ratio without adding a scoring clamp.
        ratio = math.exp(raw_log_ratio)  # Preserve all representable consequences of the sole declared `1e-300` error floor.
    except OverflowError as error:  # Reject an unrepresentable successful ratio rather than silently altering its scientific score.
        raise ValueError("successful error ratio exceeds finite float range") from error  # Invalidate malformed evidence at the analysis boundary.
    if not math.isfinite(ratio) or ratio <= 0.0:  # Require a finite positive value for geometric means and strict JSON output.
        raise ValueError("successful error ratio is not finite and positive")  # Refuse underflow or overflow without an undeclared clamp.
    return RatioScore(float(ratio), "success", False, False, False)  # Return the exact representable paired ratio with no numerical clipping.
def failure_aware_median_error(seed_errors: Sequence[float | None], seed_ok: Sequence[bool] | None = None) -> float | None:  # Compute the frozen pointwise median across exactly three RL seeds.
    if len(seed_errors) != 3:  # Enforce the protocol's exact independent-seed count.
        raise ValueError("RL median requires exactly three seed errors")  # Reject best-seed selection and incomplete seed sets.
    flags = tuple(True for _ in seed_errors) if seed_ok is None else tuple(bool(item) for item in seed_ok)  # Default omitted status flags to declared success.
    if len(flags) != 3:  # Require one status flag per seed when explicit flags are supplied.
        raise ValueError("RL median requires exactly three seed status flags")  # Reject mismatched failure metadata.
    ranked = []  # Collect valid finite errors and infinite failure sentinels for deterministic ranking.
    for value, declared_ok in zip(seed_errors, flags, strict=True):  # Classify every seed without dropping failures.
        valid, numeric = _valid_nonnegative(value, declared_ok)  # Apply the common metric-validity rule.
        ranked.append(numeric if valid else math.inf)  # Place failures after every successful finite error.
    ranked.sort()  # Order the three seed outcomes so the middle policy is selected pointwise.
    return None if not math.isfinite(ranked[1]) else float(ranked[1])  # Return failure when at least two seeds failed, otherwise the finite median.
def _geometric_mean(values: Sequence[float]) -> float:  # Aggregate positive ratios without overweighting their raw numeric scale.
    if not values:  # Reject empty evidence rather than emitting an undefined statistic.
        raise ValueError("geometric mean requires at least one value")  # Surface incomplete aggregation inputs immediately.
    logs = [math.log(float(value)) for value in values]  # Transform every finite positive ratio into additive log space.
    if any(not math.isfinite(item) for item in logs):  # Guard the report boundary against invalid values.
        raise ValueError("geometric mean values must be finite and positive")  # Refuse NaN, infinity, zero, and negative ratios.
    return float(math.exp(sum(logs) / len(logs)))  # Return the equal-weight geometric mean required by the protocol.
def _quantile_r7(values: Sequence[float], probability: float) -> float:  # Compute the deterministic R-7 linear-interpolation quantile.
    if not values:  # Reject an undefined empty-sample quantile.
        raise ValueError("quantile requires at least one value")  # Fail before any partial report is produced.
    if probability < 0.0 or probability > 1.0:  # Validate the requested probability domain.
        raise ValueError("quantile probability must be between zero and one")  # Keep percentile semantics explicit.
    ordered = sorted(float(value) for value in values)  # Make the result independent of input ordering.
    position = (len(ordered) - 1) * probability  # Use the preregistered R-7 fractional order-statistic index.
    lower_index = int(math.floor(position))  # Locate the lower bracketing observation.
    upper_index = int(math.ceil(position))  # Locate the upper bracketing observation.
    fraction = position - lower_index  # Compute the interpolation weight within the bracket.
    return float(ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction)  # Return the linearly interpolated quantile.
def _splitmix64(value: int) -> int:  # Produce a reproducible pseudo-random unsigned integer without library-version dependence.
    mixed = (int(value) + 0x9E3779B97F4A7C15) & _MASK_64  # Apply the SplitMix64 Weyl increment in fixed-width arithmetic.
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64  # Apply the first avalanche multiplication.
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _MASK_64  # Apply the second avalanche multiplication.
    return (mixed ^ (mixed >> 31)) & _MASK_64  # Return the final deterministic 64-bit sample.
def case_bootstrap_geometric_ci(case_log_means: Sequence[float], seed: int = BOOTSTRAP_SEED, replicates: int = BOOTSTRAP_REPLICATES, confidence: float = BOOTSTRAP_CONFIDENCE) -> dict[str, Any]:  # Bootstrap whole cases while retaining all within-case operating points together.
    logs = tuple(float(value) for value in case_log_means)  # Freeze case-cluster log means in caller-provided deterministic order.
    if not logs or any(not math.isfinite(value) for value in logs):  # Require at least one finite case-cluster statistic.
        raise ValueError("case bootstrap requires finite case log means")  # Reject invalid or empty bootstrap evidence.
    if int(replicates) <= 0:  # Require a positive number of bootstrap samples.
        raise ValueError("bootstrap replicates must be positive")  # Prevent an undefined interval.
    if confidence <= 0.0 or confidence >= 1.0:  # Require a proper two-sided confidence level.
        raise ValueError("bootstrap confidence must lie strictly between zero and one")  # Keep interval semantics unambiguous.
    estimates = []  # Accumulate deterministic case-resampled geometric-mean estimates.
    for replicate in range(int(replicates)):  # Generate the frozen number of cluster-bootstrap replicates.
        sampled_sum = 0.0  # Accumulate selected case log means for this replicate.
        for slot in range(len(logs)):  # Draw exactly one case per original case slot with replacement.
            token = (int(seed) & _MASK_64) ^ ((replicate + 1) * 0xD2B74407B1CE6E93) ^ ((slot + 1) * 0xCA5A826395121157)  # Derive an order-stable sample token.
            sampled_index = _splitmix64(token) % len(logs)  # Map the deterministic token uniformly enough onto a case index.
            sampled_sum += logs[sampled_index]  # Retain the selected case's complete within-case aggregate.
        estimates.append(math.exp(sampled_sum / len(logs)))  # Convert the resampled mean log ratio back to ratio space.
    tail = (1.0 - confidence) / 2.0  # Split excluded probability equally across the two tails.
    point = math.exp(sum(logs) / len(logs))  # Compute the unresampled case-equal point estimate.
    return {"point_estimate": float(point), "lower": _quantile_r7(estimates, tail), "upper": _quantile_r7(estimates, 1.0 - tail), "confidence": float(confidence), "replicates": int(replicates), "seed": int(seed), "resampling_unit": "case"}  # Return fully auditable finite bootstrap metadata.
def _validate_case_grid(observations: Sequence[Any], expected_cases: int = EXPECTED_TEST_CASES) -> tuple[str, ...]:  # Enforce the exact blind-case by preregistered-point Cartesian product.
    if not observations:  # Reject an empty benchmark result set.
        raise ValueError("primary grid observations cannot be empty")  # Prevent vacuous gate success.
    case_ids = tuple(sorted({str(item.case_id) for item in observations}))  # Establish deterministic case ordering for aggregation and bootstrap.
    if len(case_ids) != int(expected_cases):  # Require all sixteen frozen blind-test cases.
        raise ValueError(f"expected {expected_cases} cases, found {len(case_ids)}")  # Report the exact manifest-cardinality mismatch.
    expected_points = set(PRIMARY_OPERATING_POINTS)  # Build the preregistered point set once for validation.
    seen = set()  # Track unique case and operating-point keys.
    for item in observations:  # Inspect every supplied primary observation.
        key = (str(item.case_id), int(item.solves), int(item.equation_budget))  # Normalize the primary-grid key.
        if key in seen:  # Reject duplicated evidence that could silently reweight a favorable point.
            raise ValueError(f"duplicate primary observation {key}")  # Surface the duplicate exact key.
        if (key[1], key[2]) not in expected_points:  # Exclude non-preregistered points from primary statistics.
            raise ValueError(f"non-primary operating point {(key[1], key[2])}")  # Prevent post-hoc operating-point selection.
        seen.add(key)  # Register the validated unique primary observation.
    expected = {(case_id, solves, budget) for case_id in case_ids for solves, budget in PRIMARY_OPERATING_POINTS}  # Construct the required Cartesian product.
    if seen != expected:  # Reject any missing case-point pair.
        missing = sorted(expected - seen)  # Identify absent evidence deterministically.
        raise ValueError(f"incomplete primary grid; missing {missing[:5]}")  # Report a bounded preview of missing keys.
    return case_ids  # Return sorted cases for deterministic downstream cluster aggregation.
def _failure_summary(scores: Sequence[RatioScore]) -> dict[str, int]:  # Count every retained success and failure classification.
    labels = ("success", "success_clipped", "numerator_failed", "denominator_failed", "both_failed")  # Freeze the complete status vocabulary.
    return {label: sum(score.status == label for score in scores) for label in labels}  # Return zero-inclusive deterministic status counts.
def _metric_summary(rows: Sequence[tuple[str, RatioScore]], case_ids: Sequence[str], include_tail: bool) -> dict[str, Any]:  # Aggregate paired ratios with case-cluster uncertainty.
    ratios = [score.ratio for _, score in rows]  # Preserve equal weight over all case and operating-point pairs.
    case_logs = []  # Collect one within-case mean log ratio per blind case.
    for case_id in case_ids:  # Aggregate all six operating points inside each resampling cluster.
        selected = [math.log(score.ratio) for row_case, score in rows if row_case == case_id]  # Keep the complete paired point set for this case.
        case_logs.append(sum(selected) / len(selected))  # Give each case equal weight regardless of row ordering.
    bootstrap = case_bootstrap_geometric_ci(case_logs)  # Resample entire cases under the frozen deterministic bootstrap.
    summary = {"geometric_mean_ratio": _geometric_mean(ratios), "bootstrap_ci_95": bootstrap, "failure_counts": _failure_summary([score for _, score in rows]), "numeric_clip_count": sum(score.numerically_clipped for _, score in rows)}  # Assemble common metric fields.
    if include_tail:  # Add distribution gates only for the primary energy metric.
        summary["win_fraction"] = float(sum(ratio < 1.0 for ratio in ratios) / len(ratios))  # Count strict pairwise WM wins.
        summary["p95_ratio"] = _quantile_r7(ratios, 0.95)  # Measure the preregistered upper-tail regression statistic.
    return summary  # Return JSON-safe finite aggregate statistics.
def build_pairwise_ratio_rows(observations: Sequence[PairwiseObservation], competitor: str) -> list[dict[str, Any]]:  # Build deterministic CSV-ready long-form paired-ratio evidence.
    _validate_case_grid(observations)  # Refuse incomplete or post-hoc primary data before row generation.
    if competitor not in PRIMARY_COMPETITORS:  # Restrict reports to the preregistered competitor vocabulary.
        raise ValueError(f"unknown primary competitor {competitor!r}")  # Reject accidental best-baseline or renamed-method comparisons.
    if any(item.competitor != competitor for item in observations):  # Prevent mixing multiple competitors in one aggregate.
        raise ValueError("pairwise observations contain mixed competitor labels")  # Keep every ratio's denominator scientifically unambiguous.
    rows = []  # Accumulate stable primitive ratio records.
    for item in sorted(observations, key=lambda value: (value.case_id, value.solves, value.equation_budget)):  # Emit rows in manifest and operating-point order.
        energy = score_error_ratio(item.wm_energy_error, item.competitor_energy_error, item.wm_energy_ok, item.competitor_energy_ok)  # Score the paired energy metric with retained failures.
        qoi = score_error_ratio(item.wm_qoi_error, item.competitor_qoi_error, item.wm_qoi_ok, item.competitor_qoi_ok)  # Score the paired QoI metric with retained failures.
        rows.append({"protocol_id": PROTOCOL_ID, "case_id": item.case_id, "competitor": competitor, "solves": int(item.solves), "equation_budget": int(item.equation_budget), "energy_ratio": energy.ratio, "energy_status": energy.status, "qoi_ratio": qoi.ratio, "qoi_status": qoi.status, "wm_budget_violation": bool(item.wm_budget_violation), "wm_proactive_action": bool(item.wm_proactive_action)})  # Preserve all audit-critical scalar evidence.
    return rows  # Return deterministic JSON-safe row dictionaries.
def evaluate_primary_competitor(observations: Sequence[PairwiseObservation], competitor: str) -> dict[str, Any]:  # Evaluate all seven frozen primary gates against one competitor.
    case_ids = _validate_case_grid(observations)  # Enforce all sixteen cases and all six primary points.
    if competitor not in PRIMARY_COMPETITORS:  # Restrict evaluation to the three preregistered competitors.
        raise ValueError(f"unknown primary competitor {competitor!r}")  # Reject post-hoc comparator labels.
    if any(item.competitor != competitor for item in observations):  # Prevent denominator mixing within a claimed comparison.
        raise ValueError("pairwise observations contain mixed competitor labels")  # Fail explicitly on scientifically invalid input.
    energy_rows = []  # Collect case-labelled energy ratio scores for aggregation.
    qoi_rows = []  # Collect case-labelled QoI ratio scores for aggregation.
    for item in observations:  # Score every preregistered paired observation once.
        energy_rows.append((item.case_id, score_error_ratio(item.wm_energy_error, item.competitor_energy_error, item.wm_energy_ok, item.competitor_energy_ok)))  # Retain energy failures in the primary statistic.
        qoi_rows.append((item.case_id, score_error_ratio(item.wm_qoi_error, item.competitor_qoi_error, item.wm_qoi_ok, item.competitor_qoi_ok)))  # Retain QoI failures in the guard statistic.
    energy = _metric_summary(energy_rows, case_ids, True)  # Compute geometric, bootstrap, win-rate, and tail energy statistics.
    qoi = _metric_summary(qoi_rows, case_ids, False)  # Compute the aggregate QoI guard statistic.
    budget_violations = sum(bool(item.wm_budget_violation) for item in observations)  # Count every actual primary-point budget violation.
    proactive_cases = {item.case_id for item in observations if item.wm_proactive_action}  # Collapse proactive execution to the protocol's case-level unit.
    proactive_fraction = float(len(proactive_cases) / len(case_ids))  # Compute the fraction of blind cases with certified proactive execution.
    gates = {"aggregate_energy": energy["geometric_mean_ratio"] <= PRIMARY_ENERGY_LIMIT, "bootstrap_upper": energy["bootstrap_ci_95"]["upper"] < PRIMARY_BOOTSTRAP_UPPER_LIMIT, "win_fraction": energy["win_fraction"] >= PRIMARY_WIN_FRACTION_MIN, "p95_regression": energy["p95_ratio"] <= PRIMARY_P95_LIMIT, "aggregate_qoi": qoi["geometric_mean_ratio"] <= PRIMARY_QOI_LIMIT, "zero_budget_violations": budget_violations == 0, "proactive_case_fraction": proactive_fraction >= PROACTIVE_CASE_FRACTION_MIN}  # Apply the seven preregistered primary gates without rounding.
    return json_safe({"schema": "wmvla-four-way-primary-v1", "protocol_id": PROTOCOL_ID, "competitor": competitor, "case_count": len(case_ids), "operating_point_count": len(PRIMARY_OPERATING_POINTS), "observation_count": len(observations), "energy": energy, "qoi": qoi, "wm_budget_violation_count": budget_violations, "proactive_case_fraction": proactive_fraction, "thresholds": {"aggregate_energy_max": PRIMARY_ENERGY_LIMIT, "bootstrap_upper_strict_max": PRIMARY_BOOTSTRAP_UPPER_LIMIT, "win_fraction_min": PRIMARY_WIN_FRACTION_MIN, "p95_ratio_max": PRIMARY_P95_LIMIT, "aggregate_qoi_max": PRIMARY_QOI_LIMIT, "proactive_case_fraction_min": PROACTIVE_CASE_FRACTION_MIN}, "gates": gates, "passed": all(gates.values())})  # Return only finite JSON-safe protocol evidence and the exact conjunction.
def evaluate_dorfler_safety(observations: Sequence[DorflerObservation], dominance_checks: Sequence[TargetDominanceCheck], fallback_events: Sequence[FallbackEvidence]) -> dict[str, Any]:  # Evaluate structural, empirical, budget, and fallback safety evidence.
    case_ids = _validate_case_grid(observations)  # Require the complete sixteen-case primary comparison grid.
    ratio_scores = [(item.case_id, score_error_ratio(item.wm_energy_error, item.dorfler_energy_error, item.wm_ok, item.dorfler_ok)) for item in observations]  # Score independent-Dorfler energy comparisons with retained failures.
    metric = _metric_summary(ratio_scores, case_ids, True)  # Aggregate ratios and preserve worst primary-point behavior.
    budget_violations = sum(bool(item.wm_budget_violation) for item in observations)  # Count every actual WM primary budget violation.
    dominance_ids = [(item.case_id, item.action_id) for item in dominance_checks]  # Build auditable identities for every target receipt.
    if len(dominance_ids) != len(set(dominance_ids)):  # Reject duplicate receipts that could hide an omitted executed target.
        raise ValueError("duplicate target-dominance receipt")  # Require one structural check per unique executed action.
    dominance_complete = bool(dominance_checks) and all(bool(item.passed) for item in dominance_checks)  # Require present and universally passing nodewise target evidence.
    if any(not str(item.trigger).strip() for item in fallback_events):  # Reject fallback rows without an auditable trigger.
        raise ValueError("fallback trigger must be nonempty")  # Prevent unverifiable generic fallback claims.
    fallback_complete = all(bool(item.dorfler_executed) for item in fallback_events)  # Treat no triggered fallback as vacuously safe and every triggered event as mandatory.
    gates = {"nodewise_target_dominance": dominance_complete, "aggregate_energy": metric["geometric_mean_ratio"] <= DORFLER_AGGREGATE_LIMIT, "worst_primary_point": max(score.ratio for _, score in ratio_scores) <= DORFLER_POINT_LIMIT, "zero_budget_violations": budget_violations == 0, "required_fallbacks_executed": fallback_complete}  # Apply every frozen Dorfler hard gate.
    return json_safe({"schema": "wmvla-four-way-dorfler-safety-v1", "protocol_id": PROTOCOL_ID, "case_count": len(case_ids), "energy": metric, "worst_primary_point_ratio": max(score.ratio for _, score in ratio_scores), "target_dominance_check_count": len(dominance_checks), "fallback_trigger_count": len(fallback_events), "wm_budget_violation_count": budget_violations, "thresholds": {"aggregate_energy_max": DORFLER_AGGREGATE_LIMIT, "worst_primary_point_max": DORFLER_POINT_LIMIT}, "gates": gates, "passed": all(gates.values())})  # Return finite auditable safety evidence and its exact conjunction.
def _valid_time(value: float) -> bool:  # Validate one nonnegative finite timing component.
    return math.isfinite(float(value)) and float(value) >= 0.0  # Accept zero-duration components while rejecting invalid clocks.
def evaluate_online_time(observations: Sequence[TimeObservation]) -> dict[str, Any]:  # Evaluate the separate online engineering-efficiency gate.
    case_ids = tuple(sorted({item.case_id for item in observations}))  # Establish deterministic case order for aggregation.
    if len(observations) != EXPECTED_TEST_CASES or len(case_ids) != EXPECTED_TEST_CASES:  # Require exactly one timing record per blind case.
        raise ValueError(f"time gate requires exactly {EXPECTED_TEST_CASES} unique cases")  # Reject duplicated or incomplete timing evidence.
    component_names = ("visual_partition_s", "world_model_s", "parameter_tools_s", "gmsh_s", "calculix_s", "lp_online_total_s")  # Freeze every timing component subject to validity checks.
    invalid_cases = []  # Collect cases with missing, negative, or nonfinite timing evidence.
    overhead_shares = []  # Collect case-level WM non-solver fractions.
    online_ratios = []  # Collect case-level WM-to-LP total online wall-clock ratios.
    vlm_violations = 0  # Count cases that exceed the one-call partition contract.
    for item in sorted(observations, key=lambda value: value.case_id):  # Evaluate timings in deterministic manifest order.
        valid = all(_valid_time(getattr(item, name)) for name in component_names) and int(item.vlm_calls) >= 0  # Validate all separated timing components and call counts.
        wm_non_solver = float(item.visual_partition_s + item.world_model_s + item.parameter_tools_s + item.gmsh_s)  # Compute all online work outside CalculiX.
        wm_total = wm_non_solver + float(item.calculix_s)  # Compute total online time from separated components without trusting a redundant total.
        valid = valid and wm_total > 0.0 and float(item.lp_online_total_s) > 0.0  # Require positive denominators for both time ratios.
        if not valid:  # Retain invalid timing evidence as a hard gate failure.
            invalid_cases.append(item.case_id)  # Identify the exact case with unusable time evidence.
        else:  # Aggregate only mathematically valid ratios while separately failing on omissions.
            overhead_shares.append(wm_non_solver / wm_total)  # Measure the case-level non-solver fraction.
            online_ratios.append(wm_total / float(item.lp_online_total_s))  # Measure paired total online wall-clock cost.
        vlm_violations += int(item.vlm_calls) > TIME_VLM_CALL_LIMIT  # Count every case that repeated the semantic model call.
    median_overhead = None if not overhead_shares else _quantile_r7(overhead_shares, 0.5)  # Compute the deterministic case median when valid evidence exists.
    aggregate_ratio = None if not online_ratios else _geometric_mean(online_ratios)  # Aggregate paired online ratios geometrically across cases.
    gates = {"valid_complete_timing": not invalid_cases, "vlm_call_limit": vlm_violations == 0, "median_non_solver_share": median_overhead is not None and median_overhead <= TIME_NON_SOLVER_SHARE_LIMIT, "online_time_vs_lp": aggregate_ratio is not None and aggregate_ratio <= TIME_LP_RATIO_LIMIT}  # Apply the frozen separate efficiency gates.
    return json_safe({"schema": "wmvla-four-way-online-time-v1", "protocol_id": PROTOCOL_ID, "case_count": len(case_ids), "invalid_cases": invalid_cases, "vlm_call_violation_count": vlm_violations, "median_non_solver_share": median_overhead, "aggregate_online_time_ratio_vs_lp": aggregate_ratio, "thresholds": {"vlm_calls_per_case_max": TIME_VLM_CALL_LIMIT, "median_non_solver_share_max": TIME_NON_SOLVER_SHARE_LIMIT, "aggregate_online_time_ratio_vs_lp_max": TIME_LP_RATIO_LIMIT}, "gates": gates, "passed": all(gates.values())})  # Return JSON-safe timing evidence without making it an undeclared overall term.
def _one_point_case_comparison(observations: Sequence[MechanismObservation], denominator_name: str) -> dict[str, Any]:  # Compare WM-full with one fixed mechanism control at one point per case.
    rows = []  # Collect one failure-aware paired ratio per blind case.
    for item in sorted(observations, key=lambda value: value.case_id):  # Preserve manifest order for deterministic bootstrap input.
        denominator = item.wm_h1_energy_error if denominator_name == "wm_h1" else item.random_safe_median_energy_error  # Select the preregistered control metric.
        denominator_ok = item.wm_h1_ok if denominator_name == "wm_h1" else item.random_safe_ok  # Select the matching control status flag.
        rows.append((item.case_id, score_error_ratio(item.wm_full_energy_error, denominator, item.wm_full_ok, denominator_ok)))  # Retain all one-sided and shared failures.
    ratios = [score.ratio for _, score in rows]  # Extract finite paired ratios for point aggregation.
    logs = [math.log(score.ratio) for _, score in rows]  # Build one log statistic per case for cluster bootstrap.
    bootstrap = case_bootstrap_geometric_ci(logs)  # Apply the same deterministic case-level uncertainty rule.
    return {"comparator": denominator_name, "geometric_mean_ratio": _geometric_mean(ratios), "bootstrap_ci_95": bootstrap, "failure_counts": _failure_summary([score for _, score in rows]), "strict_win_fraction": float(sum(ratio < 1.0 for ratio in ratios) / len(ratios))}  # Return complete finite mechanism-comparison evidence.
def evaluate_world_model_mechanism(observations: Sequence[MechanismObservation]) -> dict[str, Any]:  # Test whether gains are attributable to certified multi-step informed world-model actions.
    case_ids = tuple(sorted({item.case_id for item in observations}))  # Establish deterministic case ordering.
    if len(observations) != EXPECTED_TEST_CASES or len(case_ids) != EXPECTED_TEST_CASES:  # Require exactly one fixed-point mechanism record per blind case.
        raise ValueError(f"mechanism gate requires exactly {EXPECTED_TEST_CASES} unique cases")  # Reject incomplete or duplicated ablation evidence.
    full_vs_h1 = _one_point_case_comparison(observations, "wm_h1")  # Test whether multi-step rollout adds value beyond one-step planning.
    full_vs_random = _one_point_case_comparison(observations, "random_safe_median")  # Test whether informed action choice beats merely adding safe refinement.
    proactive_cases = sum(item.executed_proactive_actions > 0 for item in observations)  # Count cases that actually executed non-Dorfler world-model action.
    proactive_fraction = float(proactive_cases / len(observations))  # Convert proactive coverage to the protocol's case-level fraction.
    certified_total = sum(int(item.certified_proactive_actions) for item in observations)  # Count certified proactive action receipts.
    executed_total = sum(int(item.executed_proactive_actions) for item in observations)  # Count proactive actions that reached a real solve.
    nonnegative_action_counts = all(item.certified_proactive_actions >= 0 and item.executed_proactive_actions >= 0 for item in observations)  # Reject malformed negative action counts.
    certification_complete = nonnegative_action_counts and executed_total > 0 and all(item.certified_proactive_actions == item.executed_proactive_actions for item in observations)  # Require one certificate for every executed proactive action.
    gates = {"proactive_case_fraction": proactive_fraction >= PROACTIVE_CASE_FRACTION_MIN, "all_proactive_actions_certified": certification_complete, "common_uniform_probe": all(item.common_uniform_probe for item in observations), "matched_solve_and_budget": all(item.matched_solve_budget for item in observations), "competitor_isolation": all(item.competitor_isolation for item in observations), "multi_step_point_superiority": full_vs_h1["geometric_mean_ratio"] < MECHANISM_RATIO_LIMIT, "multi_step_bootstrap_superiority": full_vs_h1["bootstrap_ci_95"]["upper"] < MECHANISM_RATIO_LIMIT, "informed_action_point_superiority": full_vs_random["geometric_mean_ratio"] < MECHANISM_RATIO_LIMIT, "informed_action_bootstrap_superiority": full_vs_random["bootstrap_ci_95"]["upper"] < MECHANISM_RATIO_LIMIT}  # Apply the preregistered attribution conjunction without post-hoc tolerance tuning.
    return json_safe({"schema": "wmvla-four-way-mechanism-v1", "protocol_id": PROTOCOL_ID, "case_count": len(case_ids), "fixed_operating_point": {"solves": 6, "equation_budget": 60000}, "proactive_case_fraction": proactive_fraction, "certified_proactive_action_count": certified_total, "executed_proactive_action_count": executed_total, "full_vs_h1": full_vs_h1, "full_vs_random_safe_median": full_vs_random, "thresholds": {"proactive_case_fraction_min": PROACTIVE_CASE_FRACTION_MIN, "comparison_ratio_strict_max": MECHANISM_RATIO_LIMIT, "bootstrap_upper_strict_max": MECHANISM_RATIO_LIMIT}, "gates": gates, "passed": all(gates.values())})  # Return finite mechanism-attribution evidence and its exact conjunction.
def evaluate_final_gate(primary_results: Mapping[str, Mapping[str, Any]], dorfler_result: Mapping[str, Any], mechanism_result: Mapping[str, Any], time_result: Mapping[str, Any]) -> dict[str, Any]:  # Apply the protocol's exact five-term overall formula and report time separately.
    if set(primary_results) != set(PRIMARY_COMPETITORS):  # Require exactly the three preregistered pairwise decisions.
        raise ValueError(f"primary results must contain exactly {PRIMARY_COMPETITORS}")  # Reject missing, renamed, or post-hoc competitor terms.
    primary_pass = {name: bool(primary_results[name].get("passed", False)) for name in PRIMARY_COMPETITORS}  # Extract each machine gate without accepting truthy missing data.
    dorfler_safe = bool(dorfler_result.get("passed", False))  # Extract the independent-Dorfler safety conjunction.
    mechanism_pass = bool(mechanism_result.get("passed", False))  # Extract the world-model attribution conjunction.
    time_pass = bool(time_result.get("passed", False))  # Extract the separately reported engineering-efficiency conjunction.
    overall = dorfler_safe and primary_pass["local_prediction"] and primary_pass["supervised"] and primary_pass["rl_median"] and mechanism_pass  # Apply G_safe AND G_LP AND G_SUP AND G_RL AND G_WM_mechanism exactly.
    machine_gates = {"DORFLER_SAFE": dorfler_safe, "BEAT_LOCAL_PREDICTION": primary_pass["local_prediction"], "BEAT_SUPERVISED": primary_pass["supervised"], "BEAT_RL": primary_pass["rl_median"], "WORLD_MODEL_MECHANISM": mechanism_pass, "ONLINE_TIME_ACCEPTABLE": time_pass, "OVERALL_WIN": overall}  # Preserve the execution report's fixed result labels.
    overall_terms = ("DORFLER_SAFE", "BEAT_LOCAL_PREDICTION", "BEAT_SUPERVISED", "BEAT_RL", "WORLD_MODEL_MECHANISM")  # Freeze the only terms allowed to control OVERALL_WIN.
    failed_overall_terms = [name for name in overall_terms if not machine_gates[name]]  # List every failed required term without vague interpretation.
    return json_safe({"schema": "wmvla-four-way-final-gate-v1", "protocol_id": PROTOCOL_ID, **machine_gates, "overall_formula": "DORFLER_SAFE and BEAT_LOCAL_PREDICTION and BEAT_SUPERVISED and BEAT_RL and WORLD_MODEL_MECHANISM", "online_time_is_overall_term": False, "failed_overall_terms": failed_overall_terms})  # Return a machine-readable final gate with the time-boundary decision explicit.
def json_safe(value: Any) -> Any:  # Recursively convert supported report objects and reject nonstandard JSON numbers.
    if is_dataclass(value) and not isinstance(value, type):  # Convert dataclass instances before inspecting generic containers.
        return json_safe(asdict(value))  # Preserve field names and recursively validate their values.
    if isinstance(value, Mapping):  # Normalize mapping keys and validate values recursively.
        return {str(key): json_safe(item) for key, item in value.items()}  # Return an insertion-order-preserving primitive dictionary.
    if isinstance(value, (list, tuple)):  # Normalize supported ordered sequences to JSON arrays.
        return [json_safe(item) for item in value]  # Preserve sequence order during recursive validation.
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):  # Pass through standard JSON scalar types.
        return value  # Return an already safe primitive unchanged.
    if isinstance(value, float):  # Validate Python floating-point scalars explicitly.
        if not math.isfinite(value):  # Reject NaN and infinities forbidden by strict JSON.
            raise ValueError("nonfinite float is not JSON safe")  # Fail before json.dumps can emit nonstandard tokens.
        return float(value)  # Return a finite built-in float.
    raise TypeError(f"unsupported JSON-safe value type {type(value).__name__}")  # Reject implicit stringification of unknown scientific objects.
def flatten_csv_row(report: Mapping[str, Any]) -> dict[str, str]:  # Flatten a nested report into deterministic scalar CSV columns.
    safe = json_safe(report)  # Validate the entire report before producing any row cells.
    flattened: dict[str, str] = {}  # Accumulate dot-delimited stable column names and string values.
    def visit(prefix: str, item: Any) -> None:  # Recursively visit nested mappings while keeping sequences atomic.
        if isinstance(item, Mapping):  # Expand mappings into distinct deterministic columns.
            for key in sorted(item):  # Sort keys so column order is stable across runs.
                visit(f"{prefix}.{key}" if prefix else str(key), item[key])  # Recurse with a dot-delimited path.
            return  # Stop after expanding all mapping children.
        if isinstance(item, list):  # Preserve ordered evidence arrays as compact strict JSON cells.
            flattened[prefix] = json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":"))  # Encode sequences without nonstandard numbers.
            return  # Stop after assigning the atomic sequence cell.
        if item is None:  # Represent missing optional report values as an empty CSV cell.
            flattened[prefix] = ""  # Avoid writing a misleading textual null token.
            return  # Stop after assigning the missing-value cell.
        if isinstance(item, bool):  # Normalize booleans independently from integers.
            flattened[prefix] = "true" if item else "false"  # Use lowercase JSON-compatible boolean text.
            return  # Stop after assigning the boolean cell.
        flattened[prefix] = str(item)  # Convert finite numeric and textual scalar values predictably.
    visit("", safe)  # Flatten the validated report from its root.
    return flattened  # Return only string-valued cells suitable for csv.DictWriter.
