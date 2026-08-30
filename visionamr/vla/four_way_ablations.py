"""Frozen mechanism-ablation wrappers and honest world-model diagnostics for WMVLA-4WAY-P1."""  # Define the module's bounded scientific responsibility.
from __future__ import annotations  # Postpone annotation evaluation for compatibility with repository runtimes.

from collections import Counter  # Count normalized fallback causes without dropping unknown reasons.
from collections.abc import Mapping, Sequence  # Describe immutable schedule and aggregation inputs.
from dataclasses import asdict, dataclass, replace  # Define immutable evidence records and isolated configuration variants.
from datetime import datetime, timezone  # Timestamp irreversible ablation campaign boundaries in unambiguous UTC.
import hashlib  # Bind reused primary results and emitted traces to exact bytes.
import json  # Persist complete finite machine-readable diagnostic evidence.
import math  # Compute logarithmic calibration errors and rank correlations.
import os  # Publish one-shot campaign markers and atomic artifacts without overwrite races.
from pathlib import Path  # Address append-safe experiment artifacts portably.
import random  # Implement the five preregistered random-safe controls reproducibly.
import time  # Measure complete online and partial native-failure trajectory durations.
import traceback  # Retain bounded fatal and typed-native failure provenance for audit.
from typing import Any, Callable  # Type repository adapters and behavior-neutral state transforms.

import numpy as np  # Compute regional transition diagnostics without additional real solves.

from .world.model import RegionAction, ResidualWorldModel, WorldPrediction, WorldState, semantic_persistence  # Reuse the frozen V0 state, action, and transition contracts.
from .world.planner import MultiStepPlanner, PlanDecision, PlannerConfig  # Reuse the frozen V0 candidate and scoring logic.
from .world.tool_gateway import MCPToolGateway, MaterializedAction  # Reuse the exact Dörfler-dominant action compiler and certificate.

PROTOCOL_ID = "WMVLA-4WAY-P1"  # Bind every output to the preregistered protocol.
TRACE_SCHEMA = "wmvla-four-way-prediction-trace-v1"  # Version completed and incomplete transition diagnostics.
CASE_SCHEMA = "wmvla-four-way-ablation-case-v1"  # Version the fixed per-case mechanism summary consumed by analysis.
CAMPAIGN_SCHEMA = "wmvla-four-way-ablation-campaign-v1"  # Version the complete sixteen-case mechanism index.
CAMPAIGN_PLAN_SCHEMA = "wmvla-four-way-ablation-execution-plan-v1"  # Version the formal post-primary one-shot execution order.
CAMPAIGN_START_SCHEMA = "wmvla-four-way-ablation-started-v1"  # Version the irreversible formal ablation start marker.
CAMPAIGN_EXECUTION_SCHEMA = "wmvla-four-way-ablation-execution-summary-v1"  # Version the complete raw campaign outcome ledger.
ABLATION_RESULT_SCHEMA = "wmvla-four-way-ablation-result-v1"  # Version one native or primary-reuse variant result.
ORACLE_SOURCE_SCHEMA = "wmvla-four-way-oracle-source-v1"  # Version one independently executed Dörfler future source.
ABLATION_BUDGET = 60000  # Freeze the only equation budget used by mechanism ablations.
ABLATION_MAX_SOLVES = 6  # Freeze the only real-solve prefix used by mechanism ablations.
EXPECTED_TEST_CASES = 16  # Require the entire blind-test split before campaign aggregation.
RANDOM_SAFE_SEEDS = (20260911, 20260912, 20260913, 20260914, 20260915)  # Freeze the exact five random-safe repetitions.
DETERMINISTIC_VARIANTS = ("wm_full", "wm_h1", "wm_prior_only", "wm_no_history", "oracle_future_hit")  # Freeze the one-run diagnostic variants.
ALL_VARIANTS = ("wm_full", "wm_h1", "wm_prior_only", "wm_no_history", "random_safe_extra", "oracle_future_hit")  # Freeze the formal execution vocabulary so the independent source can immediately precede the final oracle phase.
_TINY = 1.0e-300  # Keep exact-zero or underflowed regional values finite in logarithmic diagnostics.


@dataclass(frozen=True)  # Keep one candidate prediction immutable after the real transition is known.
class CandidatePrediction:  # Store predicted ranking evidence and any independently realized counterfactual score.
    action: tuple[int, ...]  # Identify the exact regional future-depth vector.
    predicted_robust_cost: float  # Store the frozen planner's lower-is-better first-step robust cost.
    predicted_error_ratio: float  # Store the predicted global squared-indicator ratio.
    predicted_equation_ratio: float  # Store the predicted active-equation ratio.
    uncertainty: float  # Store the model's epistemic transition uncertainty.
    failure_probability: float  # Store the model's bounded transition-failure probability.
    predicted_budget_feasible: bool  # Record the planner's conservative equation-budget screen.
    predicted_rank: int  # Rank feasible candidates first by robust cost and stable action identity.
    actual_robust_cost: float | None = None  # Store a real counterfactual score only when that candidate was independently solved.
    actual_rank: int | None = None  # Store the real rank only when every compared candidate has real evidence.
    predicted_score_scope: str = "first_step_frozen_stage_cost"  # Disclose that ranking does not fabricate unexposed full-beam counterfactual costs.
    selected_by_planner: bool = False  # Identify the candidate requested before exact tool certification.
    executed_after_certification: bool = False  # Identify the candidate actually followed by a real solve.


@dataclass(frozen=True)  # Keep each completed real transition diagnostic immutable.
class TransitionDiagnostic:  # Store all prediction, execution, certification, and realized-transition evidence.
    case_id: str  # Identify the blind-test geometry without exposing parameters to the model.
    variant: str  # Identify the isolated full model or mechanism control.
    seed: int | None  # Identify a random-safe repetition or no seed for deterministic variants.
    step: int  # Identify the pre-action real-solve index.
    region_names: tuple[str, ...]  # Preserve the frozen shared-partition ordering.
    requested_action: tuple[int, ...]  # Store the planner-requested future-depth vector.
    executed_action: tuple[int, ...]  # Store the exact vector selected after deterministic tool certification.
    predicted_region_delta_log_eta2: tuple[float, ...]  # Predict regional changes in log squared-indicator mass.
    actual_region_delta_log_eta2: tuple[float, ...]  # Measure regional changes in log squared-indicator mass.
    predicted_region_delta_log_elements: tuple[float, ...]  # Predict regional changes in log element count.
    actual_region_delta_log_elements: tuple[float, ...]  # Measure regional changes in log element count.
    previous_total_error: float  # Store the pre-action total squared-indicator mass.
    predicted_total_error: float  # Store the model's next total squared-indicator mass.
    actual_total_error: float  # Store the realized next total squared-indicator mass.
    predicted_total_error_lower: float | None  # Preserve absence because frozen V0 emits no calibrated lower error bound.
    predicted_total_error_upper: float  # Store the frozen model's conservative upper prediction bound.
    previous_equations: int  # Store the pre-action active-equation count.
    predicted_equations: int  # Store the model's mean next active-equation count.
    actual_equations: int  # Store the realized next active-equation count.
    predicted_equations_lower: float | None  # Preserve absence because frozen V0 emits no calibrated lower resource bound.
    predicted_equations_upper: float  # Store the frozen model's conservative upper resource bound.
    uncertainty: float  # Store epistemic uncertainty before action execution.
    failure_probability: float  # Store predicted failure probability before action execution.
    candidate_ranking: tuple[CandidatePrediction, ...]  # Preserve the full solve-free predicted candidate ordering.
    planner_accepted: bool  # Record whether the planner requested a proactive action.
    proactive_executed: bool  # Record whether a non-Dörfler action was actually solved.
    proactive_certified: bool  # Record whether the executed proactive action had a valid exact certificate.
    planner_reason: str  # Preserve the frozen planner decision reason.
    execution_reason: str  # Preserve the deterministic gateway acceptance, fallback, or stop reason.
    fallback_cause: str | None  # Normalize required fallback categories while retaining raw reasons above.
    actual_improved: bool  # Report whether total squared-indicator mass decreased after an accepted action.


@dataclass(frozen=True)  # Keep an independent Dörfler step immutable before oracle construction.
class DorflerFutureStep:  # Store only the future-hit evidence needed by the oracle upper bound.
    step: int  # Identify the completed Dörfler solve index.
    region_names: tuple[str, ...]  # Preserve the shared semantic ordering used by every method.
    dorfler_error_fraction: tuple[float, ...]  # Store realized within-region marked squared-error fractions.
    dorfler_element_fraction: tuple[float, ...]  # Store realized within-region marked-element fractions.
    eligible_regions: tuple[int, ...]  # Store the frozen planner-eligible region order at this real state.


@dataclass(frozen=True)  # Prevent post-hoc mutation of the oracle's independent source trajectory.
class DorflerFutureTrajectory:  # Prove oracle actions came only from a separately completed Dörfler run.
    case_id: str  # Identify the same blind-test case used by the later oracle run.
    equation_budget: int  # Require the fixed mechanism-ablation budget.
    max_solves: int  # Require the fixed mechanism-ablation solve prefix.
    source_method: str  # Require the literal independent Dörfler source label.
    completed_independently: bool  # Certify source completion before oracle action construction.
    stop_reason: str  # Explain full-prefix completion or an earlier physical/resource stop.
    common_probe_sha256: str  # Bind the source to the shared uniform probe identity.
    steps: tuple[DorflerFutureStep, ...]  # Store the complete ordered real Dörfler hit trajectory.


@dataclass(frozen=True)  # Keep each derived oracle action and objective audit immutable.
class OracleActionChoice:  # Store the best allowed future-hit action under a transparent fixed objective.
    step: int  # Identify the oracle action's pre-action step.
    action: tuple[int, ...]  # Store the selected allowed regional depth vector.
    future_hit_score: float  # Store discounted captured future Dörfler error mass.
    objective: str  # Describe the fixed future-hit maximization rule.


@dataclass(frozen=True)  # Keep one ablation result's analysis-facing evidence immutable.
class AblationOutcome:  # Normalize deterministic and seeded diagnostic runs before case aggregation.
    case_id: str  # Identify the blind-test case.
    variant: str  # Identify one frozen diagnostic variant.
    energy_error: float | None  # Store the K=6 and B=60000 energy error or no value on failure.
    ok: bool  # Mark whether the method produced a finite nonnegative energy error within budget.
    executed_proactive_actions: int  # Count actually solved non-Dörfler actions.
    certified_proactive_actions: int  # Count those actions with exact valid certificates.
    common_probe_sha256: str  # Bind the run to the common uniform probe.
    matched_budget: bool  # Confirm K=6 and B=60000 were not exceeded.
    competitor_isolation: bool  # Confirm the run read no competitor results.
    trace_path: str  # Point to the complete prediction-calibration trace.
    result_path: str  # Point to the exact trajectory result reused or executed.
    result_sha256: str  # Bind the result path to exact bytes.
    seed: int | None = None  # Identify one random-safe run or no seed for deterministic variants.
    reused_from_primary: bool = False  # Mark WM-full identity reuse instead of a second favorable rerun.


@dataclass  # Keep mutable runtime components grouped without changing the wrapped V0 objects.
class AblationRuntime:  # Return isolated adapters plus their shared diagnostic recorder.
    variant: str  # Identify the frozen variant configured by the factory.
    seed: int | None  # Identify the random-safe seed when applicable.
    model: Any  # Expose the audited model adapter accepted by the existing pipeline.
    planner: Any  # Expose the audited planner adapter accepted by the existing pipeline.
    gateway: Any  # Expose the audited exact materialization gateway accepted by the existing pipeline.
    diagnostics: "DiagnosticSession"  # Expose completed and interrupted trace evidence to the harness.
    isolation_receipt: dict[str, Any]  # Explain exactly which behavior-neutral transforms define the variant.


@dataclass(frozen=True)  # Keep the formal campaign invocation immutable after readiness validation begins.
class AblationCampaignRequest:  # Carry the sole post-primary campaign root and frozen input paths.
    root: Path  # Store the WMVLA-4WAY-P1 campaign root containing protocol and primary test evidence.
    manifest_path: Path  # Store the canonical authenticated case manifest path.
    frozen_config_path: Path  # Store the canonical authenticated frozen runtime configuration path.
    dry_run: bool = False  # Permit full post-primary readiness validation without creating ablation artifacts or invoking native tools.
    allow_unqualified_references: bool = False  # Require explicit invocation acknowledgement matching the frozen primary Reference-B execution policy.


@dataclass(frozen=True)  # Keep one formal variant execution coordinate immutable and auditable.
class AblationCampaignJob:  # Identify one case, variant, optional seed, and exact raw-artifact directory.
    case_id: str  # Store the manifest-owned blind case identifier.
    variant: str  # Store one frozen mechanism variant label.
    seed: int | None  # Store one exact random-safe seed or no seed for deterministic variants.
    output_dir: Path  # Store the exact case-local raw artifact directory.


@dataclass(frozen=True)  # Keep validated post-primary evidence isolated from mutable campaign execution state.
class AblationReadiness:  # Return authenticated cases, frozen settings, and post-primary identity evidence.
    cases: tuple[dict[str, Any], ...]  # Store all sixteen ascending manifest test cases.
    config: dict[str, Any]  # Store the exact decoded frozen runtime configuration.
    evidence: dict[str, Any]  # Store protected hashes, primary completion, and common-probe identities.


@dataclass(frozen=True)  # Keep an independent Dörfler source completion immutable before oracle schedule derivation.
class OracleSourceExecution:  # Return source trajectory availability and complete raw result identity.
    case_id: str  # Identify the manifest case used by the independent Dörfler run.
    trajectory: DorflerFutureTrajectory | None  # Store a complete natural-stop trajectory or no usable future source after typed failure.
    failure: dict[str, Any] | None  # Store only a retained typed native failure or no failure.
    common_probe_sha256: str  # Bind the source to the exact shared uniform probe.
    result_path: str  # Point to the independently executed source raw records.
    result_sha256: str  # Bind the source records to exact bytes.
    status_path: str  # Point to the atomic source completion marker.


def _finite_nonnegative(value: float | None, ok: bool) -> bool:  # Validate scientific error values without deleting failures.
    return bool(ok and value is not None and math.isfinite(float(value)) and float(value) >= 0.0)  # Accept finite nonnegative declared successes only.


def _safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray, floor: float) -> tuple[float, ...]:  # Compute finite elementwise log changes with an explicit floor.
    left = np.maximum(np.asarray(numerator, dtype=float), float(floor))  # Bound next values away from logarithmic underflow.
    right = np.maximum(np.asarray(denominator, dtype=float), float(floor))  # Bound prior values away from logarithmic underflow.
    return tuple(float(value) for value in np.log(left / right))  # Return a JSON-safe immutable regional vector.


def strip_history(state: WorldState) -> WorldState:  # Remove only recurrence and trajectory-index history for WM-no-history.
    return WorldState(names=state.names, err_sum=np.asarray(state.err_sum, dtype=float).copy(), elems=np.asarray(state.elems, dtype=float).copy(), sizes=np.asarray(state.sizes, dtype=float).copy(), vm_max=np.asarray(state.vm_max, dtype=float).copy(), volume=np.asarray(state.volume, dtype=float).copy(), adjacency=np.asarray(state.adjacency, dtype=float).copy(), dorfler_error_fraction=np.asarray(state.dorfler_error_fraction, dtype=float).copy(), dorfler_element_fraction=np.asarray(state.dorfler_element_fraction, dtype=float).copy(), hit_count=np.zeros(len(state.names), dtype=float), n_equations=int(state.n_equations), eq_per_elem=float(state.eq_per_elem), h_min=float(state.h_min), h0=float(state.h0), dim=int(state.dim), step=0)  # Preserve current physics while deleting hit counts and elapsed-step history.


class PriorOnlyModel:  # Disable learned residual corrections while retaining the exact frozen physics prior.
    def __init__(self, base: ResidualWorldModel) -> None:  # Wrap a fresh frozen-model reload without mutating it.
        self.base = base  # Retain the authenticated model solely for its fixed prior and configuration.
        self.config = base.config  # Expose the unchanged frozen configuration expected by the pipeline.
        self._observed_transitions = 0  # Count online transitions without storing residual rows.
    @property  # Expose warmup-compatible real-transition count without enabling residual learning.
    def transition_count(self) -> int:  # Return training evidence or observed online steps, whichever is greater.
        return max(int(self.base.transition_count), int(self._observed_transitions))  # Preserve full-model decision opportunities fairly.
    def predict(self, state: WorldState, action: RegionAction) -> WorldPrediction:  # Predict solely from the explicit mesh-evolution prior.
        return self.base._prior(state, action)  # Bypass the residual ensemble without changing any prior constant.
    def observe(self, previous: WorldState, action: RegionAction, observed: WorldState) -> None:  # Count but never learn from a completed transition.
        self._observed_transitions += 1  # Preserve planner warmup semantics without adding residual rows.
    def save(self, path: str | Path) -> None:  # Persist an auditable unchanged base snapshot through the normal pipeline hook.
        self.base.save(path)  # Reuse exact repository serialization without inventing an ablation checkpoint format.


class NoHistoryModel:  # Remove online recurrence and last-transition residual state while retaining frozen residual training.
    def __init__(self, base: ResidualWorldModel) -> None:  # Wrap one fresh authenticated frozen-model reload.
        self.base = base  # Retain offline residual weights and immutable model constants.
        self.config = base.config  # Expose the unchanged frozen configuration expected by the pipeline.
        self._observed_transitions = 0  # Count completed online steps without feeding them back into residual memory.
    @property  # Expose fair action warmup while suppressing online residual feedback.
    def transition_count(self) -> int:  # Return frozen training evidence or observed transition count.
        return max(int(self.base.transition_count), int(self._observed_transitions))  # Avoid changing action opportunity solely through adapter mechanics.
    def predict(self, state: WorldState, action: RegionAction) -> WorldPrediction:  # Predict with all explicit history features zeroed.
        return self.base.predict(strip_history(state), action)  # Retain current measured physics and frozen residual ensemble only.
    def observe(self, previous: WorldState, action: RegionAction, observed: WorldState) -> None:  # Suppress last-transition residual feedback by design.
        self._observed_transitions += 1  # Count the transition without changing the frozen residual library.
    def save(self, path: str | Path) -> None:  # Persist an unchanged authenticated base snapshot for audit.
        self.base.save(path)  # Reuse exact repository serialization.


def _prediction_bounds(previous: WorldState, prediction: WorldPrediction) -> tuple[None, float, None, float]:  # Convert only the bounds the frozen model actually emits to absolute units.
    error_upper = float(previous.total_error * prediction.error_ratio_upper)  # Convert the frozen one-sided ratio bound to an absolute upper limit.
    equation_upper = float(previous.n_equations * prediction.equation_ratio_upper)  # Convert the frozen resource ratio bound to an absolute upper limit.
    return None, error_upper, None, equation_upper  # Preserve honest missing lower bounds instead of imposing post-hoc symmetry.


def _normalize_fallback(planner_reason: str, execution_reason: str, proactive_executed: bool) -> str | None:  # Map raw controller reasons to the four mandated fallback categories.
    if proactive_executed:  # Treat a certified non-Dörfler execution as no fallback.
        return None  # Return an explicit no-fallback sentinel.
    joined = f"{planner_reason}|{execution_reason}".lower()  # Normalize both independent decision layers once.
    if "uncertainty" in joined:  # Recognize the frozen epistemic gate.
        return "uncertainty"  # Count an uncertainty fallback.
    if any(token in joined for token in ("budget", "equation", "cap", "feasible")):  # Recognize predicted or exact resource rejection.
        return "budget"  # Count a budget fallback.
    if any(token in joined for token in ("gain", "optimal")):  # Recognize a small or absent robust advantage.
        return "low_gain"  # Count a low-gain fallback.
    if any(token in joined for token in ("audit", "failure", "underperform", "distrust")):  # Recognize model-risk or empirical-discredit recovery.
        return "distrust"  # Count a distrust fallback.
    return "other"  # Retain warmup, explicit Dörfler, and unclassified safe fallbacks separately.


def _candidate_predictions(planner: MultiStepPlanner, state: WorldState, model: Any, n_equation_cap: int) -> tuple[CandidatePrediction, ...]:  # Instrument the frozen first-step candidate set without real solves.
    rows: list[dict[str, Any]] = []  # Collect raw candidate predictions before stable ranking.
    for action in planner.enumerate_actions(state):  # Enumerate exactly the frozen V0 first-step actions.
        prediction = model.predict(state, action)  # Evaluate a solve-free model transition with no model update.
        feasible = bool(planner._within_budget(state, prediction, n_equation_cap))  # Apply the same conservative predicted resource screen.
        cost = float(planner._stage_cost(state, prediction))  # Apply the same frozen one-step robust cost.
        rows.append({"action": tuple(int(value) for value in action.extra_depth), "predicted_robust_cost": cost, "predicted_error_ratio": float(prediction.error_ratio_mean), "predicted_equation_ratio": float(prediction.equation_ratio_mean), "uncertainty": float(prediction.uncertainty), "failure_probability": float(prediction.failure_risk), "predicted_budget_feasible": feasible})  # Preserve every predicted ranking component.
    order = sorted(range(len(rows)), key=lambda index: (not rows[index]["predicted_budget_feasible"], rows[index]["predicted_robust_cost"], rows[index]["action"]))  # Rank feasible low-cost actions first with deterministic ties.
    ranks = {index: position + 1 for position, index in enumerate(order)}  # Convert sorted positions to one-based audit ranks.
    return tuple(CandidatePrediction(**row, predicted_rank=ranks[index]) for index, row in enumerate(rows))  # Return the complete immutable candidate set.


class DiagnosticSession:  # Coordinate planner, compiler, model, and realized-transition evidence without policy changes.
    def __init__(self, case_id: str, variant: str, *, seed: int | None = None, artifact_path: str | Path | None = None) -> None:  # Initialize one isolated case-variant trace.
        if variant not in ALL_VARIANTS:  # Reject post-hoc diagnostic labels.
            raise ValueError(f"unknown ablation variant {variant!r}")  # Surface the frozen vocabulary violation.
        self.case_id = str(case_id)  # Store the manifest case identifier.
        self.variant = str(variant)  # Store the frozen variant label.
        self.seed = None if seed is None else int(seed)  # Store the exact random-safe seed when present.
        self.artifact_path = None if artifact_path is None else Path(artifact_path)  # Normalize optional durable trace output.
        self._pending: dict[int, dict[str, Any]] = {}  # Join decision, certification, prediction, and realization by pre-action step.
        self._records: list[TransitionDiagnostic] = []  # Retain every completed real transition in execution order.
    @property  # Expose completed evidence without permitting mutation.
    def records(self) -> tuple[TransitionDiagnostic, ...]:  # Return completed real transitions.
        return tuple(self._records)  # Isolate caller views from session storage.
    def register_decision(self, state: WorldState, decision: PlanDecision, ranking: Sequence[CandidatePrediction]) -> None:  # Record the solve-free planner decision and candidate ranking.
        entry = self._pending.setdefault(int(state.step), {})  # Create or recover this pre-action join row.
        entry["decision"] = decision  # Store the immutable planner audit contract.
        entry["ranking"] = tuple(ranking)  # Store the complete immutable predicted candidate ordering.
    def register_materialization(self, state: WorldState, materialized: MaterializedAction) -> None:  # Record exact action compilation and budget certification.
        entry = self._pending.setdefault(int(state.step), {})  # Create or recover this pre-action join row.
        entry["materialized"] = materialized  # Store exact requested, executed, target, and equation evidence.
        entry["awaiting"] = tuple(int(value) for value in materialized.action.extra_depth)  # Identify the pipeline's immediate executed-action prediction call.
        self._flush()  # Persist an interrupted-transition receipt before the next real solve begins.
    def capture_prediction(self, state: WorldState, action: RegionAction, prediction: WorldPrediction) -> None:  # Capture only the post-certification prediction for the executed action.
        entry = self._pending.get(int(state.step))  # Read the pending materialized step without creating planner-free noise.
        if entry is None or entry.get("awaiting") != tuple(int(value) for value in action.extra_depth):  # Ignore internal beam-rollout prediction calls.
            return  # Preserve diagnostic identity without affecting model behavior.
        entry["prediction"] = prediction  # Store the exact prediction made immediately before real execution.
        entry["previous"] = state  # Store the matching measured pre-action state.
        entry.pop("awaiting", None)  # Ensure later model calls cannot overwrite executed-action evidence.
        self._flush()  # Persist the complete pre-execution receipt before the next solver call.
    def finalize(self, previous: WorldState, action: RegionAction, observed: WorldState) -> TransitionDiagnostic:  # Join the next real solve to its unique prior prediction.
        entry = self._pending.get(int(previous.step))  # Recover the pending pre-action evidence.
        if entry is None or "prediction" not in entry or "materialized" not in entry:  # Refuse silent partial calibration records.
            raise RuntimeError(f"missing pre-execution diagnostic evidence for step {previous.step}")  # Preserve honest trace completeness.
        prediction = entry["prediction"]  # Read the exact executed-action prediction.
        materialized = entry["materialized"]  # Read the exact action certificate.
        decision = entry.get("decision")  # Read the planner decision when the standard pipeline produced one.
        raw_ranking = tuple(entry.get("ranking", ()))  # Preserve an empty ranking only for a nonstandard injected planner.
        if previous.names != observed.names or prediction.next_state.names != previous.names:  # Require stable shared-partition ordering.
            raise ValueError("diagnostic transition region names must remain stable")  # Reject scientifically misaligned regional deltas.
        error_lower, error_upper, equation_lower, equation_upper = _prediction_bounds(previous, prediction)  # Derive transparent absolute prediction intervals.
        requested = tuple(int(value) for value in materialized.certificate.requested_action)  # Read the requested action from the exact certificate.
        executed = tuple(int(value) for value in materialized.certificate.executed_action)  # Read the executed action from the exact certificate.
        ranking = tuple(replace(candidate, selected_by_planner=candidate.action == requested, executed_after_certification=candidate.action == executed) for candidate in raw_ranking)  # Identify requested and actually solved candidates without changing their predicted ordering.
        proactive = any(executed)  # Identify an actually executed non-Dörfler action.
        compiled_hashes = (str(materialized.certificate.base_compiled_field_sha256), str(materialized.certificate.world_compiled_field_sha256 or ""))  # Read both complete post-gradation field identities.
        certified = bool(proactive and materialized.certificate.accepted and materialized.certificate.base_target_included and materialized.certificate.no_coarsening and materialized.certificate.compiled_dorfler_included and materialized.certificate.compiled_max_dorfler_violation <= 1.0e-12 and materialized.certificate.compiled_field_node_count > 0 and all(len(value) == 64 for value in compiled_hashes))  # Require every raw and compiled structural safety term for proactive certification.
        planner_reason = str(getattr(decision, "reason", "planner_reason_unavailable"))  # Preserve an explicit missing reason for injected traces.
        execution_reason = str(materialized.certificate.reason)  # Preserve the exact tool outcome.
        record = TransitionDiagnostic(case_id=self.case_id, variant=self.variant, seed=self.seed, step=int(previous.step), region_names=tuple(previous.names), requested_action=requested, executed_action=executed, predicted_region_delta_log_eta2=_safe_log_ratio(prediction.next_state.err_sum, previous.err_sum, _TINY), actual_region_delta_log_eta2=_safe_log_ratio(observed.err_sum, previous.err_sum, _TINY), predicted_region_delta_log_elements=_safe_log_ratio(prediction.next_state.elems, previous.elems, 1.0), actual_region_delta_log_elements=_safe_log_ratio(observed.elems, previous.elems, 1.0), previous_total_error=float(previous.total_error), predicted_total_error=float(prediction.next_state.total_error), actual_total_error=float(observed.total_error), predicted_total_error_lower=error_lower, predicted_total_error_upper=error_upper, previous_equations=int(previous.n_equations), predicted_equations=int(prediction.next_state.n_equations), actual_equations=int(observed.n_equations), predicted_equations_lower=equation_lower, predicted_equations_upper=equation_upper, uncertainty=float(prediction.uncertainty), failure_probability=float(prediction.failure_risk), candidate_ranking=ranking, planner_accepted=bool(getattr(decision, "accepted", False)), proactive_executed=proactive, proactive_certified=certified, planner_reason=planner_reason, execution_reason=execution_reason, fallback_cause=_normalize_fallback(planner_reason, execution_reason, proactive), actual_improved=bool(observed.total_error < previous.total_error))  # Assemble every mandatory diagnostic without counterfactual fabrication.
        self._records.append(record)  # Append the completed transition once.
        self._pending.pop(int(previous.step), None)  # Remove the joined pending row to prevent duplicate realization.
        self._flush()  # Persist the completed prediction-versus-actual trace atomically.
        return record  # Return the immutable completed diagnostic to direct callers.
    def payload(self) -> dict[str, Any]:  # Serialize completed and interrupted diagnostic evidence.
        incomplete = []  # Preserve pre-execution evidence when the next real solve fails or is never attempted.
        for step, entry in sorted(self._pending.items()):  # Emit pending actions deterministically by pre-action step.
            decision = entry.get("decision")  # Read optional planner evidence.
            materialized = entry.get("materialized")  # Read optional compiler evidence.
            prediction = entry.get("prediction")  # Read optional executed-action prediction.
            incomplete.append({"step": int(step), "status": "not_realized", "reason": "next_real_solve_missing_or_failed", "decision": None if decision is None else asdict(decision), "certificate": None if materialized is None else asdict(materialized.certificate), "prediction": None if prediction is None else {"region_error": prediction.next_state.err_sum.tolist(), "region_elements": prediction.next_state.elems.tolist(), "total_error": float(prediction.next_state.total_error), "equations": int(prediction.next_state.n_equations), "uncertainty": float(prediction.uncertainty), "failure_probability": float(prediction.failure_risk)}})  # Retain uncertainty and failure predictions even without an actual transition.
        return {"schema": TRACE_SCHEMA, "protocol_id": PROTOCOL_ID, "case_id": self.case_id, "variant": self.variant, "seed": self.seed, "transitions": [_transition_payload(item) for item in self._records], "incomplete_transitions": incomplete}  # Return one complete JSON-safe trace document.
    def _flush(self) -> None:  # Persist current evidence whenever an artifact path was requested.
        if self.artifact_path is None:  # Permit lightweight in-memory tests and callers.
            return  # Skip filesystem work without changing runtime behavior.
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)  # Create the exact case-variant artifact directory.
        temporary = self.artifact_path.with_suffix(self.artifact_path.suffix + ".tmp")  # Choose a same-directory atomic staging file.
        temporary.write_text(json.dumps(self.payload(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # Write finite human-auditable JSON bytes.
        temporary.replace(self.artifact_path)  # Publish the complete trace atomically.


class AuditedModel:  # Observe model calls without modifying prediction or update behavior.
    def __init__(self, base: Any, diagnostics: DiagnosticSession) -> None:  # Bind one isolated model adapter to one case trace.
        self.base = base  # Store the selected full or ablated model implementation.
        self.diagnostics = diagnostics  # Store the shared cross-layer recorder.
        self.config = base.config  # Expose unchanged model settings to the pipeline.
    @property  # Preserve the planner's exact warmup query.
    def transition_count(self) -> int:  # Return the selected model's transition count.
        return int(self.base.transition_count)  # Delegate without adjustment.
    def predict(self, state: WorldState, action: RegionAction) -> WorldPrediction:  # Delegate and conditionally capture the executed-action prediction.
        prediction = self.base.predict(state, action)  # Run the selected model exactly once.
        self.diagnostics.capture_prediction(state, action, prediction)  # Ignore beam calls and retain only post-certification execution evidence.
        return prediction  # Return the unchanged prediction object.
    def observe(self, previous: WorldState, action: RegionAction, observed: WorldState) -> None:  # Finalize diagnostics before applying the selected online update rule.
        self.diagnostics.finalize(previous, action, observed)  # Join prediction and actual result before model memory can change.
        self.base.observe(previous, action, observed)  # Delegate to full learning or the isolated no-learning ablation.
    def save(self, path: str | Path) -> None:  # Preserve the pipeline's normal snapshot lifecycle.
        self.base.save(path)  # Delegate exact serialization to the selected model adapter.


class AuditedPlanner:  # Add solve-free candidate ranking around an unchanged selected planner.
    def __init__(self, base: MultiStepPlanner, diagnostics: DiagnosticSession, *, state_transform: Callable[[WorldState], WorldState] | None = None) -> None:  # Configure optional WM-no-history state removal.
        self.base = base  # Store the unchanged full, horizon-one, random, or oracle planner.
        self.config = base.config  # Expose exact planner settings to the pipeline and gateway.
        self.diagnostics = diagnostics  # Store the shared recorder.
        self.state_transform = state_transform or (lambda value: value)  # Default to behavior-neutral identity state access.
    def plan(self, state: WorldState, model: Any, n_equation_cap: int, *, force_dorfler: bool = False) -> PlanDecision:  # Delegate the decision then log solve-free first-step rankings.
        viewed = self.state_transform(state)  # Remove only explicitly ablated history when requested.
        decision = self.base.plan(viewed, model, n_equation_cap, force_dorfler=force_dorfler)  # Execute the selected planner without changing constants.
        ranking = _candidate_predictions(self.base, viewed, model, n_equation_cap)  # Score the same first-step candidates without any real solve.
        self.diagnostics.register_decision(state, decision, ranking)  # Bind the original real step to decision evidence.
        return decision  # Return the unchanged planner decision.


class AuditedGateway:  # Record exact materialization evidence around the unchanged deterministic gateway.
    def __init__(self, base: MCPToolGateway, diagnostics: DiagnosticSession) -> None:  # Bind one gateway to one isolated trace.
        self.base = base  # Store the exact Dörfler-dominant compiler.
        self.diagnostics = diagnostics  # Store the shared recorder.
        self.config = base.config  # Expose unchanged tool settings.
        self.schema_version = base.schema_version  # Expose unchanged certificate schema identity.
    def inspect_case(self, problem: Any) -> dict[str, Any]:  # Delegate read-only case inspection.
        return self.base.inspect_case(problem)  # Preserve the exact structured case payload.
    def observe_solve(self, problem: Any, partition: Any, post: Any, record: Any, eta2: np.ndarray, hit_count: np.ndarray | None, step: int) -> Any:  # Delegate real-state construction.
        return self.base.observe_solve(problem, partition, post, record, eta2, hit_count, step)  # Preserve exact shared-partition observation behavior.
    def materialize_action(self, observation: Any, action: RegionAction, n_equation_cap: int) -> MaterializedAction:  # Delegate exact candidate generation and certification.
        materialized = self.base.materialize_action(observation, action, n_equation_cap)  # Generate the same Dörfler and optional proactive candidates.
        self.diagnostics.register_materialization(observation.state, materialized)  # Record the exact requested and executed action receipt.
        return materialized  # Return the unchanged compiled action.


class RandomSafePlanner(MultiStepPlanner):  # Select a random predicted-budget-feasible proactive action under frozen action constraints.
    def __init__(self, config: PlannerConfig, seed: int) -> None:  # Initialize one exact preregistered random-safe repetition.
        super().__init__(config)  # Reuse all frozen V0 action-domain and resource settings.
        if int(seed) not in RANDOM_SAFE_SEEDS:  # Reject opportunistic random repetitions.
            raise ValueError(f"random-safe seed must be one of {RANDOM_SAFE_SEEDS}")  # Preserve the preregistered seed set.
        self.seed = int(seed)  # Store the exact repetition identity.
        self._rng = random.Random(self.seed)  # Create a local deterministic generator isolated from global state.
    def plan(self, state: WorldState, model: Any, n_equation_cap: int, *, force_dorfler: bool = False) -> PlanDecision:  # Randomize only safe extra-region choice.
        baseline_cost, baseline_path, baseline_prediction = self._baseline_rollout(state, model, n_equation_cap)  # Preserve the exact Dörfler floor and audit values.
        baseline = RegionAction.dorfler(state)  # Construct the executable safety action.
        baseline_upper = int(math.ceil(state.n_equations * baseline_prediction.equation_ratio_upper))  # Preserve the conservative baseline resource estimate.
        serialized_path = tuple(action.extra_depth for action in baseline_path)  # Serialize the baseline rollout through the standard decision contract.
        if force_dorfler:  # Honor the same empirical-discredit cooldown as WM-full.
            return PlanDecision(baseline, False, "audit_cooldown", baseline_cost, baseline_cost, 0.0, baseline_upper, baseline_prediction.error_ratio_upper, serialized_path)  # Execute exact Dörfler during recovery.
        if model.transition_count < self.config.warmup_transitions:  # Preserve the same transition-evidence opportunity gate as WM-full.
            return PlanDecision(baseline, False, "world_model_warmup", baseline_cost, baseline_cost, 0.0, baseline_upper, baseline_prediction.error_ratio_upper, serialized_path)  # Execute exact Dörfler during warmup.
        candidates: list[tuple[RegionAction, WorldPrediction, float]] = []  # Collect nonzero actions that pass the same predicted resource screen.
        for action in self.enumerate_actions(state):  # Reuse the exact frozen action enumeration.
            if action.is_dorfler_only:  # Exclude the baseline from the random-extra draw.
                continue  # Keep the control focused on uninformed additional refinement.
            prediction = model.predict(state, action)  # Predict only for the shared conservative budget screen and audit quantities.
            if self._within_budget(state, prediction, n_equation_cap):  # Retain candidates inside the same predicted budget margin.
                candidates.append((action, prediction, self._stage_cost(state, prediction)))  # Preserve the selected candidate's unchanged score fields.
        if not candidates:  # Fall back when no proactive candidate passes the common screen.
            return PlanDecision(baseline, False, "no_feasible_random_safe_action", baseline_cost, baseline_cost, 0.0, baseline_upper, baseline_prediction.error_ratio_upper, serialized_path)  # Execute exact Dörfler safely.
        action, prediction, cost = candidates[self._rng.randrange(len(candidates))]  # Select one safe extra action uniformly with the local frozen RNG.
        upper = int(math.ceil(state.n_equations * prediction.equation_ratio_upper))  # Record the selected candidate's conservative resource estimate.
        gain = float(baseline_cost - cost)  # Preserve a diagnostic robust-gain value without gating random selection.
        return PlanDecision(action, True, "random_safe_selected", baseline_cost, float(cost), gain, upper, prediction.error_ratio_upper, (action.extra_depth,))  # Request the random action while leaving exact gateway certification authoritative.


class OracleFutureHitPlanner(MultiStepPlanner):  # Execute actions fixed from a separately completed Dörfler future-hit trajectory.
    def __init__(self, config: PlannerConfig, schedule: Mapping[int, OracleActionChoice]) -> None:  # Bind a post-hoc oracle schedule before the oracle run starts.
        super().__init__(config)  # Reuse all frozen V0 action-domain and resource settings.
        self.schedule = {int(step): choice for step, choice in schedule.items()}  # Copy the complete immutable future-hit schedule.
    def plan(self, state: WorldState, model: Any, n_equation_cap: int, *, force_dorfler: bool = False) -> PlanDecision:  # Select the precomputed upper-bound action under common safety screens.
        baseline_cost, baseline_path, baseline_prediction = self._baseline_rollout(state, model, n_equation_cap)  # Preserve the exact Dörfler audit floor.
        baseline = RegionAction.dorfler(state)  # Construct the executable safety action.
        baseline_upper = int(math.ceil(state.n_equations * baseline_prediction.equation_ratio_upper))  # Preserve the conservative baseline resource estimate.
        serialized_path = tuple(action.extra_depth for action in baseline_path)  # Serialize the baseline rollout through the standard contract.
        if force_dorfler:  # Honor empirical-discredit recovery consistently.
            return PlanDecision(baseline, False, "audit_cooldown", baseline_cost, baseline_cost, 0.0, baseline_upper, baseline_prediction.error_ratio_upper, serialized_path)  # Execute exact Dörfler.
        choice = self.schedule.get(int(state.step))  # Read only the precomputed action for this real step.
        if choice is None or not any(choice.action):  # Handle trajectories with no compressible future hit.
            return PlanDecision(baseline, False, "oracle_no_future_hit", baseline_cost, baseline_cost, 0.0, baseline_upper, baseline_prediction.error_ratio_upper, serialized_path)  # Execute exact Dörfler transparently.
        action = RegionAction(tuple(int(value) for value in choice.action), source="oracle_future_hit")  # Construct the allowed oracle vector without changing depths.
        action.validate(state, max_depth=self.config.max_extra_depth)  # Enforce the same bounded noncoarsening action contract.
        prediction = model.predict(state, action)  # Compute common resource and audit predictions without using them to learn the oracle.
        if not self._within_budget(state, prediction, n_equation_cap):  # Apply the same conservative predicted budget screen.
            return PlanDecision(baseline, False, "oracle_predicted_budget_fallback", baseline_cost, baseline_cost, 0.0, baseline_upper, baseline_prediction.error_ratio_upper, serialized_path)  # Execute exact Dörfler before Gmsh.
        cost = float(self._stage_cost(state, prediction))  # Preserve the frozen robust cost for audit only.
        upper = int(math.ceil(state.n_equations * prediction.equation_ratio_upper))  # Store the conservative selected resource estimate.
        return PlanDecision(action, True, "oracle_future_hit_selected", baseline_cost, cost, float(baseline_cost - cost), upper, prediction.error_ratio_upper, (action.extra_depth,))  # Request the future-informed action for exact certification.


def build_dorfler_future_trajectory(case_id: str, states: Sequence[WorldState], planner: MultiStepPlanner, common_probe_sha256: str, stop_reason: str) -> DorflerFutureTrajectory:  # Convert independently recorded Dörfler states into oracle-only future-hit evidence.
    ordered = tuple(states)  # Freeze the complete real-state trajectory before validation.
    if not ordered or len(ordered) > ABLATION_MAX_SOLVES:  # Require at least the common probe and at most the matched K=6 prefix.
        raise ValueError("independent dorfler source must contain one through six real states")  # Reject empty or over-budget future evidence.
    if tuple(int(state.step) for state in ordered) != tuple(range(len(ordered))):  # Require contiguous real-solve ordering without hidden omissions.
        raise ValueError("independent dorfler source states must be contiguous from step zero")  # Preserve exact future offsets.
    names = ordered[0].names  # Freeze the shared semantic ordering from the common probe.
    if any(state.names != names for state in ordered):  # Reject partition drift across remeshing.
        raise ValueError("independent dorfler source must retain stable region names")  # Preserve future-hit vector identity.
    if not common_probe_sha256 or not stop_reason:  # Require exact common-probe and completion evidence.
        raise ValueError("independent dorfler source requires probe hash and stop reason")  # Reject an unauditable future source.
    if len(ordered) < ABLATION_MAX_SOLVES and stop_reason not in ("equation_cap_reached", "dorfler_candidate_exceeds_cap", "no_marked_elements", "solver_failure"):  # Restrict early completion to explicit nonselective stops.
        raise ValueError("short independent dorfler source requires an explicit physical, resource, or solver stop")  # Prevent favorable manual truncation.
    steps = tuple(DorflerFutureStep(step=int(state.step), region_names=tuple(state.names), dorfler_error_fraction=tuple(float(value) for value in state.dorfler_error_fraction), dorfler_element_fraction=tuple(float(value) for value in state.dorfler_element_fraction), eligible_regions=tuple(int(value) for value in planner._eligible(state))) for state in ordered)  # Reproduce the frozen V0 eligibility order from each independently measured state.
    return DorflerFutureTrajectory(case_id=str(case_id), equation_budget=ABLATION_BUDGET, max_solves=ABLATION_MAX_SOLVES, source_method="dorfler", completed_independently=True, stop_reason=str(stop_reason), common_probe_sha256=str(common_probe_sha256), steps=steps)  # Return content-ready oracle source evidence without competitor access.


def derive_oracle_schedule(trajectory: DorflerFutureTrajectory, config: PlannerConfig) -> dict[int, OracleActionChoice]:  # Construct a fixed future-hit upper-bound schedule after independent Dörfler completion.
    if trajectory.source_method != "dorfler" or not trajectory.completed_independently:  # Require an independent source rather than peeking during oracle execution.
        raise ValueError("oracle requires an independently completed dorfler trajectory")  # Reject leakage-prone or incomplete sources.
    if trajectory.equation_budget != ABLATION_BUDGET or trajectory.max_solves != ABLATION_MAX_SOLVES:  # Require the matched diagnostic operating point.
        raise ValueError("oracle dorfler trajectory must use B=60000 and K=6")  # Reject a mismatched future signal.
    if not trajectory.steps or len(trajectory.steps) > ABLATION_MAX_SOLVES:  # Require a nonempty completed trajectory within the matched prefix.
        raise ValueError("oracle dorfler trajectory must contain one through six steps")  # Reject absent or over-budget future evidence.
    ordered = tuple(sorted(trajectory.steps, key=lambda item: item.step))  # Normalize source order without changing evidence.
    if tuple(item.step for item in ordered) != tuple(range(len(ordered))):  # Require a contiguous completed real trajectory.
        raise ValueError("oracle dorfler steps must be contiguous from zero")  # Reject missing future hits.
    if not trajectory.stop_reason or (len(ordered) < ABLATION_MAX_SOLVES and trajectory.stop_reason not in ("equation_cap_reached", "dorfler_candidate_exceeds_cap", "no_marked_elements", "solver_failure")):  # Require honest completion evidence for a short source trajectory.
        raise ValueError("oracle dorfler trajectory has an invalid early stop")  # Reject result-dependent manual truncation.
    names = ordered[0].region_names  # Freeze the common semantic ordering.
    if not trajectory.common_probe_sha256 or any(item.region_names != names for item in ordered):  # Require a bound common probe and stable partition.
        raise ValueError("oracle dorfler trajectory requires a common probe and stable regions")  # Reject incomparable future labels.
    schedule: dict[int, OracleActionChoice] = {}  # Accumulate one precomputed action per executable transition.
    objective = "maximize_discounted_realized_future_dorfler_error_fraction_within_v0_action_space"  # Freeze the transparent oracle objective label.
    for position, current in enumerate(ordered[:-1]):  # Construct actions only where a later real solve exists.
        candidates: list[tuple[int, ...]] = [tuple(0 for _ in names)]  # Retain pure Dörfler as the zero-score allowed action.
        eligible = tuple(index for index in current.eligible_regions[: config.candidate_regions] if 0 <= int(index) < len(names) and semantic_persistence(names[int(index)]) > 0.0)  # Reuse the stored frozen eligibility order and exclude generic regions.
        for index in eligible:  # Enumerate the same bounded single-region depth actions.
            for depth in range(1, config.max_extra_depth + 1):  # Consider every allowed compressed future-hit depth.
                vector = [0 for _ in names]  # Start from the exact Dörfler vector.
                vector[int(index)] = int(depth)  # Assign depth only to the eligible mechanism.
                candidates.append(tuple(vector))  # Retain the legal single-region oracle candidate.
        if config.max_extra_regions >= 2:  # Enable the same sparse paired actions as V0.
            for left_position, left in enumerate(eligible):  # Traverse the first eligible mechanism deterministically.
                for right in eligible[left_position + 1 :]:  # Traverse distinct later eligible mechanisms.
                    vector = [0 for _ in names]  # Start from the exact Dörfler vector.
                    vector[int(left)] = 1  # Add one future hit to the first mechanism.
                    vector[int(right)] = 1  # Add one future hit to the second mechanism.
                    candidates.append(tuple(vector))  # Retain the legal paired oracle candidate.
        future = ordered[position + 1 :]  # Read only later independently completed Dörfler hits.
        def score(action: tuple[int, ...]) -> float:  # Evaluate discounted captured future error for one allowed vector.
            total = 0.0  # Initialize the additive oracle objective.
            for region, depth in enumerate(action):  # Score each selected semantic mechanism independently.
                hits = [(offset, step.dorfler_error_fraction[region]) for offset, step in enumerate(future, start=1) if step.dorfler_element_fraction[region] > 0.0]  # Collect actual later hits and their realized error mass.
                total += sum((config.discount ** (offset - 1)) * float(value) for offset, value in hits[: int(depth)])  # Credit only as many earliest future hits as the requested depth compresses.
            return float(total)  # Return the fixed higher-is-better future-hit score.
        best = min(candidates, key=lambda action: (-score(action), sum(action), action))  # Maximize captured future mass, then prefer shallower lexicographic actions.
        schedule[int(current.step)] = OracleActionChoice(step=int(current.step), action=best, future_hit_score=score(best), objective=objective)  # Freeze the selected allowed action before oracle execution.
    return schedule  # Return the complete deterministic oracle schedule.


def build_ablation_runtime(case_id: str, variant: str, base_model: ResidualWorldModel, base_planner: MultiStepPlanner, base_gateway: MCPToolGateway, *, seed: int | None = None, oracle_schedule: Mapping[int, OracleActionChoice] | None = None, trace_path: str | Path | None = None) -> AblationRuntime:  # Build one isolated wrapper stack without modifying V0 constants or source logic.
    if variant not in ALL_VARIANTS:  # Reject labels outside the frozen diagnostic family.
        raise ValueError(f"unknown ablation variant {variant!r}")  # Surface the exact invalid label.
    if variant == "random_safe_extra" and seed not in RANDOM_SAFE_SEEDS:  # Require the exact seeded control set.
        raise ValueError(f"random_safe_extra requires one seed from {RANDOM_SAFE_SEEDS}")  # Reject unregistered repetitions.
    if variant != "random_safe_extra" and seed is not None:  # Prevent seed-based cherry-picking of deterministic variants.
        raise ValueError("only random_safe_extra accepts a seed")  # Preserve deterministic variant identity.
    diagnostics = DiagnosticSession(case_id, variant, seed=seed, artifact_path=trace_path)  # Create one cross-layer recorder for this isolated run.
    model_variant: Any = base_model  # Default to the exact authenticated full residual model.
    planner_variant: MultiStepPlanner = base_planner  # Default to the exact full multi-step planner.
    state_transform: Callable[[WorldState], WorldState] | None = None  # Default to the complete measured history state.
    transforms: list[str] = []  # Explain the minimal isolated differences from WM-full.
    if variant == "wm_h1":  # Change only the planning horizon for the horizon-one ablation.
        planner_variant = MultiStepPlanner(replace(base_planner.config, horizon=1))  # Preserve every other frozen planner field byte-for-value.
        transforms.append("planner_config.horizon=1_only")  # Record the sole policy change.
    elif variant == "wm_prior_only":  # Disable residual corrections without changing the analytic prior.
        model_variant = PriorOnlyModel(base_model)  # Wrap the model with prior prediction and no residual updates.
        transforms.append("predict=analytic_prior_only")  # Record residual-ensemble removal.
    elif variant == "wm_no_history":  # Remove recurrence and last-transition feedback only.
        model_variant = NoHistoryModel(base_model)  # Retain frozen residual training while suppressing online transition updates.
        state_transform = strip_history  # Remove hit counts and absolute trajectory step from planner state.
        transforms.extend(("hit_count=0", "state.step=0", "online_residual_update=disabled"))  # Record every removed history channel.
    elif variant == "random_safe_extra":  # Replace informed action choice with the preregistered random control.
        planner_variant = RandomSafePlanner(base_planner.config, int(seed))  # Retain the same candidate space and predicted resource screen.
        transforms.append(f"uniform_random_safe_candidate_seed={int(seed)}")  # Record the sole uninformed-choice transform.
    elif variant == "oracle_future_hit":  # Replace learned choice with a precomputed independent future-hit upper bound.
        if oracle_schedule is None:  # Require precommitted source-derived actions.
            raise ValueError("oracle_future_hit requires an independently derived schedule")  # Prevent live future peeking.
        planner_variant = OracleFutureHitPlanner(base_planner.config, oracle_schedule)  # Retain the same allowed action and resource contracts.
        transforms.append("choice=independent_completed_dorfler_future_hits")  # Record the nondeployable information advantage.
    else:  # Preserve WM-full as the exact identity control.
        transforms.append("identity_v0_components")  # Record behavior-neutral auditing only.
    audited_model = AuditedModel(model_variant, diagnostics)  # Add transition capture without changing selected predictions.
    audited_planner = AuditedPlanner(planner_variant, diagnostics, state_transform=state_transform)  # Add candidate logging around the selected planner.
    audited_gateway = AuditedGateway(base_gateway, diagnostics)  # Add exact certificate logging around the selected compiler.
    receipt = {"protocol_id": PROTOCOL_ID, "variant": variant, "seed": seed, "transforms": transforms, "base_model_type": type(base_model).__qualname__, "base_planner_type": type(base_planner).__qualname__, "base_gateway_type": type(base_gateway).__qualname__, "planner_config": asdict(planner_variant.config), "model_config": asdict(base_model.config), "gateway_config": asdict(base_gateway.config), "v0_constants_changed": False, "competitor_results_accessed": False}  # Prove isolation and unchanged V0 constants at construction.
    return AblationRuntime(variant=variant, seed=seed, model=audited_model, planner=audited_planner, gateway=audited_gateway, diagnostics=diagnostics, isolation_receipt=receipt)  # Return pipeline-compatible isolated components.


def _rankdata(values: Sequence[float]) -> np.ndarray:  # Compute one-based average ranks with deterministic tie handling.
    array = np.asarray(values, dtype=float)  # Normalize the finite score vector.
    order = np.argsort(array, kind="mergesort")  # Preserve deterministic ordering inside ties.
    ranks = np.empty(array.size, dtype=float)  # Allocate one average rank per original item.
    start = 0  # Initialize the first equal-value group.
    while start < array.size:  # Traverse every sorted tie group.
        stop = start + 1  # Initialize the group after its first value.
        while stop < array.size and array[order[stop]] == array[order[start]]:  # Extend across exact equal finite scores.
            stop += 1  # Advance to the next sorted position.
        average = 0.5 * ((start + 1) + stop)  # Compute the mean one-based rank across the tie group.
        ranks[order[start:stop]] = average  # Assign the same average rank to all tied items.
        start = stop  # Advance to the next group.
    return ranks  # Return ranks in original item order.


def _spearman(predicted: Sequence[float], actual: Sequence[float]) -> float | None:  # Compute rank correlation without adding an optional statistics dependency.
    if len(predicted) != len(actual) or len(predicted) < 2:  # Require a matched nontrivial counterfactual set.
        return None  # Report unavailable rather than inventing a statistic.
    left = _rankdata(predicted)  # Rank predicted lower-is-better robust costs.
    right = _rankdata(actual)  # Rank actual lower-is-better robust costs.
    left -= np.mean(left)  # Center predicted ranks.
    right -= np.mean(right)  # Center actual ranks.
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))  # Compute the Pearson rank-normalization term.
    if denominator <= 0.0:  # Reject a constant ranking on either side.
        return None  # Preserve honest non-measurability for tied-only evidence.
    return float(np.dot(left, right) / denominator)  # Return the standard Spearman rank correlation.


def _transition_payload(item: TransitionDiagnostic) -> dict[str, Any]:  # Convert an immutable transition to finite explicit JSON fields.
    payload = asdict(item)  # Expand nested candidate dataclasses recursively.
    payload["candidate_ranking_measurement"] = "real_counterfactual_solves" if all(candidate.actual_robust_cost is not None for candidate in item.candidate_ranking) and len(item.candidate_ranking) >= 2 else "predicted_only"  # Disclose whether true candidate ranking exists.
    return payload  # Return the complete JSON-safe transition record.


def transition_from_payload(payload: Mapping[str, Any]) -> TransitionDiagnostic:  # Reconstruct one persisted transition without weakening schema validation.
    value = dict(payload)  # Copy the parsed JSON row before normalization.
    value.pop("candidate_ranking_measurement", None)  # Remove the derived disclosure field rather than treating it as model evidence.
    value["region_names"] = tuple(str(item) for item in value["region_names"])  # Restore immutable shared-region ordering.
    value["requested_action"] = tuple(int(item) for item in value["requested_action"])  # Restore the immutable requested action vector.
    value["executed_action"] = tuple(int(item) for item in value["executed_action"])  # Restore the immutable executed action vector.
    value["predicted_region_delta_log_eta2"] = tuple(float(item) for item in value["predicted_region_delta_log_eta2"])  # Restore predicted regional error changes.
    value["actual_region_delta_log_eta2"] = tuple(float(item) for item in value["actual_region_delta_log_eta2"])  # Restore realized regional error changes.
    value["predicted_region_delta_log_elements"] = tuple(float(item) for item in value["predicted_region_delta_log_elements"])  # Restore predicted regional resource changes.
    value["actual_region_delta_log_elements"] = tuple(float(item) for item in value["actual_region_delta_log_elements"])  # Restore realized regional resource changes.
    value["candidate_ranking"] = tuple(CandidatePrediction(action=tuple(int(component) for component in item["action"]), predicted_robust_cost=float(item["predicted_robust_cost"]), predicted_error_ratio=float(item["predicted_error_ratio"]), predicted_equation_ratio=float(item["predicted_equation_ratio"]), uncertainty=float(item["uncertainty"]), failure_probability=float(item["failure_probability"]), predicted_budget_feasible=bool(item["predicted_budget_feasible"]), predicted_rank=int(item["predicted_rank"]), actual_robust_cost=None if item.get("actual_robust_cost") is None else float(item["actual_robust_cost"]), actual_rank=None if item.get("actual_rank") is None else int(item["actual_rank"]), predicted_score_scope=str(item.get("predicted_score_scope", "first_step_frozen_stage_cost")), selected_by_planner=bool(item.get("selected_by_planner", False)), executed_after_certification=bool(item.get("executed_after_certification", False))) for item in value["candidate_ranking"])  # Restore nested immutable candidate evidence exactly.
    record = TransitionDiagnostic(**value)  # Reconstruct the versioned immutable transition contract.
    if record.variant not in ALL_VARIANTS or len(record.region_names) != len(record.executed_action):  # Reject post-hoc labels or misaligned regional actions.
        raise ValueError("persisted transition has invalid variant or action dimension")  # Preserve scientific vector identity.
    numeric = (record.predicted_total_error, record.actual_total_error, record.predicted_total_error_upper, float(record.predicted_equations), float(record.actual_equations), record.predicted_equations_upper, record.uncertainty, record.failure_probability)  # Collect mandatory finite global diagnostics.
    if any(not math.isfinite(float(item)) for item in numeric):  # Reject NaN or infinity at the analysis boundary.
        raise ValueError("persisted transition contains nonfinite diagnostics")  # Prevent invalid aggregate arithmetic.
    return record  # Return the validated persisted transition.


def load_diagnostic_trace(path: str | Path) -> tuple[TransitionDiagnostic, ...]:  # Load one exact versioned case-variant trace for aggregation.
    source = Path(path)  # Normalize the durable trace path.
    payload = json.loads(source.read_text(encoding="utf-8"))  # Parse the complete persisted JSON document.
    if payload.get("schema") != TRACE_SCHEMA or payload.get("protocol_id") != PROTOCOL_ID:  # Require the exact frozen trace contract.
        raise ValueError(f"incompatible diagnostic trace schema: {source}")  # Reject legacy or foreign evidence explicitly.
    records = tuple(transition_from_payload(item) for item in payload.get("transitions", ()))  # Restore every completed real transition in stored order.
    if any(record.case_id != str(payload.get("case_id")) or record.variant != str(payload.get("variant")) or record.seed != payload.get("seed") for record in records):  # Require document and row identities to agree.
        raise ValueError(f"diagnostic trace row identity mismatch: {source}")  # Prevent cross-case or cross-seed mixing.
    return records  # Return completed transitions while the source document retains interrupted attempts separately.


def aggregate_diagnostic_trace_files(paths: Sequence[str | Path]) -> dict[str, Any]:  # Aggregate persisted traces while retaining exact input identities.
    sources = tuple(Path(path) for path in paths)  # Freeze supplied source order before deterministic sorting.
    if not sources:  # Reject a report without source traces.
        raise ValueError("diagnostic trace aggregation requires at least one path")  # Surface missing persisted evidence.
    ordered = tuple(sorted(sources, key=lambda path: str(path)))  # Remove invocation-order effects from provenance.
    records = tuple(record for path in ordered for record in load_diagnostic_trace(path))  # Restore all completed real transitions exactly once.
    aggregate = aggregate_world_model_diagnostics(records)  # Compute the mandatory calibration and mechanism diagnostics.
    aggregate["trace_sources"] = [{"path": str(path), "sha256": _sha256_file(path)} for path in ordered]  # Bind every aggregate input to exact bytes.
    return aggregate  # Return the content-bound aggregate report.


def aggregate_world_model_diagnostics(transitions: Sequence[TransitionDiagnostic]) -> dict[str, Any]:  # Aggregate mandatory calibration, action, and fallback diagnostics.
    rows = tuple(transitions)  # Freeze caller order before deterministic aggregation.
    if not rows:  # Reject a vacuous quality report.
        raise ValueError("world-model diagnostics require at least one completed transition")  # Surface missing evidence explicitly.
    error_log_errors = [abs(math.log(max(item.predicted_total_error, _TINY)) - math.log(max(item.actual_total_error, _TINY))) for item in rows]  # Compute global total-error log absolute errors.
    equation_mapes = [abs(float(item.predicted_equations - item.actual_equations)) / max(float(item.actual_equations), 1.0) for item in rows]  # Compute active-equation mean absolute percentage errors.
    regional_error_abs = [abs(predicted - actual) for item in rows for predicted, actual in zip(item.predicted_region_delta_log_eta2, item.actual_region_delta_log_eta2, strict=True)]  # Flatten regional log-transition absolute errors.
    regional_element_abs = [abs(predicted - actual) for item in rows for predicted, actual in zip(item.predicted_region_delta_log_elements, item.actual_region_delta_log_elements, strict=True)]  # Flatten regional log-resource absolute errors.
    error_covered = [item.actual_total_error <= item.predicted_total_error_upper for item in rows]  # Test the frozen one-sided global-error upper bounds.
    equation_covered = [item.actual_equations <= item.predicted_equations_upper for item in rows]  # Test the frozen one-sided active-equation upper bounds.
    correlations: list[dict[str, Any]] = []  # Collect only genuinely counterfactual-realized candidate rankings.
    for item in rows:  # Inspect every completed real transition.
        candidates = tuple(item.candidate_ranking)  # Read the candidate evidence once.
        if len(candidates) < 2 or any(candidate.actual_robust_cost is None for candidate in candidates):  # Require real scores for every ranked candidate.
            continue  # Do not treat one executed action as counterfactual evidence.
        value = _spearman([candidate.predicted_robust_cost for candidate in candidates], [float(candidate.actual_robust_cost) for candidate in candidates])  # Compute the matched true rank correlation.
        if value is not None:  # Retain only defined nonconstant correlations.
            correlations.append({"case_id": item.case_id, "variant": item.variant, "seed": item.seed, "step": item.step, "spearman": value, "candidate_count": len(candidates)})  # Preserve transition identity and sample size.
    if correlations:  # Summarize available independent candidate realizations.
        spearman_report: dict[str, Any] = {"status": "measured", "mean": float(np.mean([row["spearman"] for row in correlations])), "transition_count": len(correlations), "per_transition": correlations}  # Report observed correlations without pooling ranks across cases.
    else:  # Explain why the frozen benchmark does not provide counterfactual rankings.
        spearman_report = {"status": "not_measurable", "reason": "Only the selected candidate is executed; true gains for unselected candidates would require extra real CalculiX solves forbidden by the matched K=6 budget.", "transition_count": 0, "per_transition": []}  # Preserve honest non-measurability rather than fabricating actual ranks.
    proactive_decisions = [item for item in rows if item.planner_accepted]  # Select planner-accepted proactive requests.
    executed_proactive = [item for item in rows if item.proactive_executed]  # Select actions actually executed after certification.
    improved = [item for item in executed_proactive if item.actual_improved]  # Select executed proactive actions followed by lower real error.
    fallback_counts = Counter(item.fallback_cause for item in rows if item.fallback_cause is not None)  # Count all normalized safe fallbacks.
    return {"schema": "wmvla-four-way-prediction-aggregate-v1", "protocol_id": PROTOCOL_ID, "transition_count": len(rows), "case_count": len({item.case_id for item in rows}), "total_error_log_mae": float(np.mean(error_log_errors)), "equation_mape": float(np.mean(equation_mapes)), "regional_delta_log_eta2_mae": float(np.mean(regional_error_abs)), "regional_delta_log_elements_mae": float(np.mean(regional_element_abs)), "prediction_interval_coverage": {"definition": "one_sided_upper_bounds_emitted_by_frozen_v0", "total_error_upper": float(np.mean(error_covered)), "equations_upper": float(np.mean(equation_covered)), "joint_upper": float(np.mean([left and right for left, right in zip(error_covered, equation_covered, strict=True)])), "two_sided_status": "not_measurable", "two_sided_reason": "Frozen V0 emits calibrated upper bounds but no lower bounds; post-hoc symmetric intervals are not imposed."}, "candidate_ranking_spearman": spearman_report, "proactive_acceptance_rate": float(len(proactive_decisions) / len(rows)), "proactive_execution_rate": float(len(executed_proactive) / len(rows)), "accepted_real_improvement_rate": None if not executed_proactive else float(len(improved) / len(executed_proactive)), "fallback_cause_counts": {name: int(fallback_counts.get(name, 0)) for name in ("uncertainty", "budget", "low_gain", "distrust", "other")}, "mean_uncertainty": float(np.mean([item.uncertainty for item in rows])), "mean_failure_probability": float(np.mean([item.failure_probability for item in rows]))}  # Return every mandatory aggregate with explicit definitions and honest unavailable terms.


def _sha256_file(path: str | Path) -> str:  # Hash an exact result or trace artifact without interpreting it.
    digest = hashlib.sha256()  # Initialize the protocol's SHA-256 identity.
    with Path(path).open("rb") as handle:  # Read the exact existing artifact bytes.
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):  # Stream large result files in bounded chunks.
            digest.update(chunk)  # Add each exact byte range to the identity.
    return digest.hexdigest()  # Return the lowercase canonical digest.


def reuse_primary_wm_full(case_id: str, energy_error: float | None, ok: bool, executed_proactive_actions: int, certified_proactive_actions: int, common_probe_sha256: str, trace_path: str | Path, result_path: str | Path, *, competitor_isolation: bool = True) -> AblationOutcome:  # Bind WM-full to the already executed primary K6-B60000 trajectory.
    result = Path(result_path)  # Normalize the exact primary result path.
    trace = Path(trace_path)  # Normalize the exact primary prediction trace path.
    if not result.is_file() or not trace.is_file():  # Require durable primary evidence before identity reuse.
        raise FileNotFoundError("WM-full reuse requires existing primary result and diagnostic trace")  # Prevent a silent second run or dangling receipt.
    return AblationOutcome(case_id=str(case_id), variant="wm_full", energy_error=energy_error, ok=bool(ok), executed_proactive_actions=int(executed_proactive_actions), certified_proactive_actions=int(certified_proactive_actions), common_probe_sha256=str(common_probe_sha256), matched_budget=True, competitor_isolation=bool(competitor_isolation), trace_path=str(trace), result_path=str(result), result_sha256=_sha256_file(result), seed=None, reused_from_primary=True)  # Return a content-bound identity outcome rather than executing WM-full again.


def _random_safe_median(outcomes: Sequence[AblationOutcome]) -> tuple[float | None, bool]:  # Compute the five-seed failure-aware pointwise median.
    ranked = []  # Rank successful finite errors before retained failures.
    for outcome in outcomes:  # Classify every preregistered random seed.
        ranked.append(float(outcome.energy_error) if _finite_nonnegative(outcome.energy_error, outcome.ok) else math.inf)  # Place failures after every valid finite error.
    ranked.sort()  # Order all five outcomes deterministically.
    return (None, False) if not math.isfinite(ranked[2]) else (float(ranked[2]), True)  # Fail the median only when at least three seeds failed.


def build_ablation_case_summary(outcomes: Sequence[AblationOutcome], prediction_aggregate: Mapping[str, Any] | None = None) -> dict[str, Any]:  # Build the fixed analyzer-facing summary for one blind case.
    rows = tuple(outcomes)  # Freeze supplied evidence before validation.
    if not rows:  # Reject an empty per-case mechanism report.
        raise ValueError("ablation case summary requires outcomes")  # Surface incomplete evidence.
    case_ids = {item.case_id for item in rows}  # Collect manifest identities.
    if len(case_ids) != 1:  # Prevent cross-case result mixing.
        raise ValueError("ablation outcomes must belong to one case")  # Preserve the paired case aggregation unit.
    deterministic: dict[str, AblationOutcome] = {}  # Index one-run diagnostics by frozen variant.
    random_rows: dict[int, AblationOutcome] = {}  # Index random controls by exact seed.
    for item in rows:  # Validate every outcome label and seed coordinate.
        if item.variant not in ALL_VARIANTS:  # Reject post-hoc variants.
            raise ValueError(f"unknown ablation outcome {item.variant!r}")  # Surface the invalid label.
        if item.variant == "random_safe_extra":  # Validate seeded random controls separately.
            if item.seed not in RANDOM_SAFE_SEEDS or int(item.seed) in random_rows:  # Require one unique exact seed.
                raise ValueError("random-safe outcomes require each preregistered seed exactly once")  # Reject missing identity or duplicate reruns.
            random_rows[int(item.seed)] = item  # Register this exact random repetition.
        else:  # Validate deterministic variants.
            if item.seed is not None or item.variant in deterministic:  # Forbid seeded or duplicated deterministic runs.
                raise ValueError("deterministic ablation outcomes must be unique and unseeded")  # Prevent best-run selection.
            deterministic[item.variant] = item  # Register the unique variant.
    if set(deterministic) != set(DETERMINISTIC_VARIANTS) or set(random_rows) != set(RANDOM_SAFE_SEEDS):  # Require the complete six-diagnostic design.
        raise ValueError("ablation case summary requires five deterministic variants and five random-safe seeds")  # Report incomplete mandatory evidence.
    if not deterministic["wm_full"].reused_from_primary:  # Require identity reuse of the primary full-model result.
        raise ValueError("WM-full must reuse the primary K6-B60000 result")  # Prevent a favorable duplicate full-model run.
    all_rows = tuple(deterministic.values()) + tuple(random_rows.values())  # Collect every diagnostic run once.
    mechanism_rows = (deterministic["wm_full"], deterministic["wm_h1"], *(random_rows[seed] for seed in RANDOM_SAFE_SEEDS))  # Restrict the frozen mechanism gate to full, h1, and five random controls.
    mechanism_probe_hashes = {item.common_probe_sha256 for item in mechanism_rows}  # Compare exact common-probe identities only across gate-bearing controls.
    common_probe = len(mechanism_probe_hashes) == 1 and "" not in mechanism_probe_hashes  # Require one nonempty probe digest across gate-bearing controls.
    matched_budget = all(item.matched_budget for item in mechanism_rows)  # Require every gate-bearing run to respect K=6 and B=60000.
    isolated = all(item.competitor_isolation for item in mechanism_rows)  # Require every gate-bearing run to avoid competitor results.
    diagnostic_probe_hashes = {item.common_probe_sha256 for item in all_rows}  # Audit all non-gating diagnostic controls separately.
    all_diagnostics_common = len(diagnostic_probe_hashes) == 1 and "" not in diagnostic_probe_hashes  # Report complete-design probe identity without adding a post-hoc gate.
    median_error, median_ok = _random_safe_median(tuple(random_rows[seed] for seed in RANDOM_SAFE_SEEDS))  # Compute the frozen five-seed pointwise median.
    variants: dict[str, Any] = {}  # Build the exact analyzer-facing variant mapping.
    for name in DETERMINISTIC_VARIANTS:  # Emit deterministic variants in frozen order.
        item = deterministic[name]  # Read the unique validated outcome.
        variants[name] = {"energy_error": item.energy_error, "ok": bool(_finite_nonnegative(item.energy_error, item.ok)), "executed_proactive_actions": int(item.executed_proactive_actions), "certified_proactive_actions": int(item.certified_proactive_actions), "common_probe_sha256": item.common_probe_sha256, "matched_budget": bool(item.matched_budget), "competitor_isolation": bool(item.competitor_isolation), "trace_path": item.trace_path, "result_path": item.result_path, "result_sha256": item.result_sha256, "reused_from_primary": bool(item.reused_from_primary)}  # Preserve accuracy, mechanism, identity, and diagnostic evidence.
    variants["random_safe_extra"] = {"median_energy_error": median_error, "median_ok": median_ok, "seeds": {str(seed): {"energy_error": random_rows[seed].energy_error, "ok": bool(_finite_nonnegative(random_rows[seed].energy_error, random_rows[seed].ok)), "executed_proactive_actions": int(random_rows[seed].executed_proactive_actions), "certified_proactive_actions": int(random_rows[seed].certified_proactive_actions), "trace_path": random_rows[seed].trace_path, "result_path": random_rows[seed].result_path, "result_sha256": random_rows[seed].result_sha256} for seed in RANDOM_SAFE_SEEDS}, "common_probe_sha256": next(iter(mechanism_probe_hashes)) if common_probe else None, "matched_budget": all(random_rows[seed].matched_budget for seed in RANDOM_SAFE_SEEDS), "competitor_isolation": all(random_rows[seed].competitor_isolation for seed in RANDOM_SAFE_SEEDS)}  # Preserve all five repetitions and their frozen median without best-seed selection.
    trace_paths = [item.trace_path for item in all_rows]  # Index every detailed calibration trace for downstream review.
    return {"schema": CASE_SCHEMA, "protocol_id": PROTOCOL_ID, "case_id": next(iter(case_ids)), "operating_point": {"equation_budget": ABLATION_BUDGET, "max_solves": ABLATION_MAX_SOLVES}, "variants": variants, "mechanism_evidence": {"common_uniform_probe": common_probe, "common_probe_sha256": next(iter(mechanism_probe_hashes)) if common_probe else None, "matched_solve_budget": matched_budget, "competitor_isolation": isolated, "wm_full_executed_proactive_actions": int(deterministic["wm_full"].executed_proactive_actions), "wm_full_certified_proactive_actions": int(deterministic["wm_full"].certified_proactive_actions)}, "diagnostic_design_evidence": {"all_variants_common_uniform_probe": all_diagnostics_common, "all_variants_common_probe_sha256": next(iter(diagnostic_probe_hashes)) if all_diagnostics_common else None, "all_variants_matched_solve_budget": all(item.matched_budget for item in all_rows), "all_variants_competitor_isolation": all(item.competitor_isolation for item in all_rows), "not_an_additional_mechanism_gate": True}, "prediction_diagnostics": {"trace_paths": trace_paths, "aggregate": None if prediction_aggregate is None else dict(prediction_aggregate)}}  # Return the complete stable per-case summary while keeping non-gating diagnostics separate.


def build_ablation_campaign_summary(case_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:  # Validate and index all sixteen blind-case mechanism reports.
    rows = [dict(item) for item in case_summaries]  # Copy caller mappings before validation.
    if len(rows) != EXPECTED_TEST_CASES:  # Require every blind-test case exactly once.
        raise ValueError(f"ablation campaign requires exactly {EXPECTED_TEST_CASES} case summaries")  # Reject incomplete or extra evidence.
    if any(item.get("schema") != CASE_SCHEMA or item.get("protocol_id") != PROTOCOL_ID for item in rows):  # Require the exact per-case schema and protocol.
        raise ValueError("ablation campaign contains an incompatible case summary")  # Reject mixed analysis contracts.
    case_ids = [str(item.get("case_id")) for item in rows]  # Normalize case identities.
    if len(set(case_ids)) != EXPECTED_TEST_CASES:  # Reject duplicate or missing case identities.
        raise ValueError("ablation campaign case identifiers must be unique")  # Preserve equal blind-case weight.
    ordered = [item for _, item in sorted(zip(case_ids, rows, strict=True), key=lambda pair: pair[0])]  # Emit deterministic manifest-ID ordering.
    return {"schema": CAMPAIGN_SCHEMA, "protocol_id": PROTOCOL_ID, "operating_point": {"equation_budget": ABLATION_BUDGET, "max_solves": ABLATION_MAX_SOLVES}, "case_count": EXPECTED_TEST_CASES, "cases": ordered, "all_common_uniform_probe": all(bool(item["mechanism_evidence"]["common_uniform_probe"]) for item in ordered), "all_matched_solve_budget": all(bool(item["mechanism_evidence"]["matched_solve_budget"]) for item in ordered), "all_competitor_isolated": all(bool(item["mechanism_evidence"]["competitor_isolation"]) for item in ordered)}  # Return the complete analyzer-facing campaign index.


def _utc_now() -> str:  # Return one timezone-explicit audit timestamp for formal campaign markers.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # Serialize UTC without locale ambiguity.


def _read_json_object(path: Path, label: str) -> dict[str, Any]:  # Load one finite top-level JSON object with an actionable label.
    try:  # Convert missing, malformed, and non-finite artifacts into one readiness failure boundary.
        payload = json.loads(path.read_text(encoding="utf-8"))  # Decode the complete persisted UTF-8 artifact.
        json.dumps(payload, allow_nan=False)  # Re-encode strictly because Python's decoder otherwise accepts NaN and infinity.
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exception:  # Catch every incomplete strict-JSON input category.
        raise ValueError(f"cannot read finite {label} {path}: {exception}") from exception  # Preserve the exact artifact and original cause.
    if not isinstance(payload, dict):  # Require named evidence fields at every campaign boundary.
        raise ValueError(f"{label} must contain a JSON object: {path}")  # Reject ambiguous scalar or array artifacts.
    return payload  # Return only a complete finite mapping.


def _campaign_file(root: Path, relative: str) -> Path:  # Resolve one regular non-symlink artifact beneath the selected campaign root.
    campaign = root.resolve()  # Normalize the sole campaign boundary once.
    candidate = campaign / relative  # Preserve the unresolved entry so symlink substitution remains detectable.
    target = candidate.resolve()  # Resolve the requested campaign-relative path without trusting embedded paths.
    try:  # Detect traversal or a symlink escape before reading result evidence.
        target.relative_to(campaign)  # Require the resolved artifact to remain under the campaign root.
    except ValueError as exception:  # Convert an escaped path into a frozen-input violation.
        raise ValueError(f"artifact leaves campaign root: {relative}") from exception  # Report the portable offending identity.
    if candidate.is_symlink() or not target.is_file() or target.is_symlink():  # Require concrete complete artifact bytes without symlink aliases.
        raise FileNotFoundError(f"required regular campaign artifact is missing: {target}")  # Reject missing files and symlink substitution.
    return target  # Return the authenticated location for exact-byte hashing or decoding.


def _load_posthoc_reference(root: Path, config: Mapping[str, Any], case: Mapping[str, Any], *, allow_unqualified: bool) -> tuple[Any, dict[str, Any]]:  # Reuse the benchmark loader under the exact frozen strict-or-expedited reference policy after online execution ends.
    from .four_way_benchmark import _reference_payload  # Import the sole authenticated Reference-B payload loader without exposing it to the online runner.
    expedited_levels = int(config["expedited_reference_levels"]) if allow_unqualified else None  # Recover the authenticated two-level depth only when the frozen amendment is active.
    amendment_record = config.get("reference_execution_amendment") if allow_unqualified else None  # Bind posthoc scoring to the same protected human amendment used by primary preflight.
    return _reference_payload(root, case, allow_unqualified=allow_unqualified, expedited_levels=expedited_levels, amendment_record=amendment_record)  # Load and receipt the denominator under exactly the preflighted schedule intent.


def _reference_digest_identity(receipt: Mapping[str, Any], label: str) -> dict[str, str]:  # Normalize the two immutable denominator-file identities repeated by preflight and every primary trajectory.
    identity = {name: str(receipt.get(name, "")) for name in ("ledger_sha256", "reference_b_sha256")}  # Read only the canonical ledger and compact Reference-B byte digests.
    for name, digest in identity.items():  # Validate both identities before any comparison can trust them.
        try:  # Reject nonhex strings without weakening the exact lowercase digest contract.
            valid = len(digest) == 64 and digest == digest.lower() and int(digest, 16) >= 0  # Require one complete lowercase SHA-256 value.
        except ValueError:  # Convert malformed hexadecimal text into one bounded evidence failure.
            valid = False  # Preserve the common invalid-receipt path.
        if not valid:  # Refuse absent, truncated, uppercase, or nonhex denominator identities.
            raise ValueError(f"{label} has invalid {name}")  # Identify the exact malformed receipt field before any native solve.
    return identity  # Return the compact immutable denominator identity.


def _pinned_reference_identity(readiness: AblationReadiness, case_id: str) -> dict[str, str]:  # Recover the primary-campaign denominator identity already cross-checked during readiness.
    primary = readiness.evidence.get("primary")  # Read the authenticated primary evidence block.
    identities = primary.get("reference_sha256_by_case") if isinstance(primary, Mapping) else None  # Read the complete per-case primary denominator map.
    value = identities.get(case_id) if isinstance(identities, Mapping) else None  # Select only the active manifest case identity.
    if not isinstance(value, Mapping):  # Require every fresh variant and oracle source to inherit a pinned primary denominator.
        raise ValueError(f"primary reference identity is missing for {case_id}")  # Stop before output creation or a native solve.
    return _reference_digest_identity(value, f"primary reference identity for {case_id}")  # Revalidate the compact receipt at its use boundary.


def _assert_pinned_reference_files(root: Path, readiness: AblationReadiness, case_id: str) -> dict[str, str]:  # Rehash denominator bytes before a fresh ablation trajectory without parsing or exposing truth values online.
    expected = _pinned_reference_identity(readiness, case_id)  # Recover the denominator accepted by every primary trajectory.
    ledger_path = _campaign_file(root, f"references/{case_id}/reference_ledger.json")  # Resolve the canonical authoritative ledger without following an external alias.
    reference_path = _campaign_file(root, f"references/{case_id}/reference_B.json")  # Resolve the canonical compact denominator artifact.
    observed = {"ledger_sha256": _sha256_file(ledger_path), "reference_b_sha256": _sha256_file(reference_path)}  # Hash only exact bytes and never deserialize physical outputs before online execution.
    if observed != expected:  # Detect replacement between readiness and this specific native phase.
        raise ValueError(f"Reference B bytes changed after primary execution for {case_id}")  # Refuse mixed-denominator mechanism evidence before spending a new solve.
    return expected  # Return the already matched identity for the posthoc receipt check.


def _assert_reference_receipt_identity(receipt: Mapping[str, Any], expected: Mapping[str, str], case_id: str) -> None:  # Recheck posthoc loader provenance against the primary-pinned denominator.
    if _reference_digest_identity(receipt, f"posthoc reference receipt for {case_id}") != dict(expected):  # Compare both ledger and compact Reference-B bytes after authenticated loading.
        raise ValueError(f"posthoc Reference B differs from primary denominator for {case_id}")  # Invalidate any concurrent replacement rather than silently mixing errors.


def _assert_pinned_primary_wm_full_files(root: Path, readiness: AblationReadiness, case_id: str) -> dict[str, str]:  # Rehash every primary byte reused or cited by the WM-full identity phase immediately before creating ablation output.
    primary = readiness.evidence.get("primary")  # Read only the solve-free primary readiness evidence already validated across the complete grid.
    by_case = primary.get("wm_full_artifact_sha256") if isinstance(primary, Mapping) else None  # Recover the complete sixteen-case pinned raw-artifact inventory.
    expected = by_case.get(case_id) if isinstance(by_case, Mapping) else None  # Select the authenticated B60000 WM-full trajectory for this case.
    mandatory = {"records.json", "mesh_receipts.json", "action_log.json", "timing.json", "prefix_results.json", "final_state.npz", "prediction_trace.json", "status.json"}  # Name every deeply validated byte identity carried into the reuse receipt or readiness chain.
    if not isinstance(expected, Mapping) or set(str(name) for name in expected) != mandatory:  # Require the exact complete inventory rather than a favorable subset.
        raise ValueError(f"primary WM-full byte inventory is incomplete for {case_id}")  # Stop before creating an identity result from partially pinned evidence.
    base = f"test/{case_id}/{ABLATION_BUDGET}/world_model_vla"  # Derive the sole registered primary identity-control directory from immutable coordinates.
    observed = {filename: _sha256_file(_campaign_file(root, f"{base}/{filename}")) for filename in sorted(mandatory)}  # Rehash all regular in-root artifacts so readiness-to-execution replacement cannot pass silently.
    normalized_expected = {str(filename): str(digest) for filename, digest in expected.items()}  # Normalize the JSON-compatible mapping without weakening exact digest equality.
    if observed != normalized_expected:  # Compare the complete raw trajectory byte set, including trace, prefix, terminal state, and status.
        raise ValueError(f"primary WM-full bytes changed after ablation readiness for {case_id}")  # Refuse identity reuse after any post-readiness mutation.
    return observed  # Return the matched exact-byte inventory for the durable reuse receipt.


def _protected_post_primary_freeze(request: AblationCampaignRequest, started: Mapping[str, Any]) -> dict[str, Any]:  # Reauthenticate the immutable freeze after primary result disclosure without invoking the pretest-only scanner.
    from .four_way_freeze import FREEZE_GIT_REF, FREEZE_SCHEMA, _environment_contract, _git, _payload_sha256, capture_environment  # Reuse canonical freeze, Git, and live-environment contracts without its pretest-only disclosure scan.
    root = request.root.resolve()  # Normalize the selected campaign boundary.
    canonical_manifest = (root / "protocol" / "case_manifest.json").resolve()  # Resolve the sole sealed manifest admitted after primary completion.
    canonical_config = (root / "protocol" / "frozen_config.json").resolve()  # Resolve the sole sealed runtime configuration admitted after primary completion.
    if request.manifest_path.resolve() != canonical_manifest or request.frozen_config_path.resolve() != canonical_config:  # Forbid alternate copies even when their visible JSON matches.
        raise ValueError("ablation campaign requires canonical protocol/case_manifest.json and protocol/frozen_config.json")  # Preserve the original freeze identity boundary.
    index_path = _campaign_file(root, "protocol/freeze_index.json")  # Resolve the authenticated protected-artifact inventory.
    sidecar_path = _campaign_file(root, "protocol/freeze_index.json.sha256")  # Resolve its independent conventional digest.
    index_digest = _sha256_file(index_path)  # Hash exact index bytes before trusting any declared path.
    sidecar_fields = sidecar_path.read_text(encoding="ascii").strip().split()  # Decode the exact digest-and-filename sidecar contract.
    if sidecar_fields != [index_digest, index_path.name]:  # Require the sidecar to authenticate exactly this sibling index.
        raise ValueError("freeze_index.json.sha256 does not authenticate freeze_index.json")  # Reject mutated or substituted freeze inventories.
    index = _read_json_object(index_path, "freeze index")  # Decode only the authenticated index bytes.
    if index.get("schema") != FREEZE_SCHEMA or index.get("protocol_id") != PROTOCOL_ID or index.get("TEST_NOT_RUN") is not True:  # Require the original pretest freeze declaration to remain intact.
        raise ValueError("freeze index is not the authenticated WMVLA-4WAY-P1 bundle")  # Reject stale or unrelated protected inputs.
    if str(started.get("freeze_index_sha256", "")) != index_digest:  # Bind the post-primary check to the exact freeze already accepted before TEST_STARTED.
        raise ValueError("TEST_STARTED freeze index identity differs from current protected bundle")  # Prevent a second freeze substitution after result disclosure.
    entries = index.get("protected_artifacts")  # Read the explicit exact-byte protected inventory.
    if not isinstance(entries, list) or not entries:  # Require a nonvacuous seal.
        raise ValueError("freeze index lacks protected_artifacts")  # Reject an empty or malformed protected bundle.
    verified: list[dict[str, Any]] = []  # Collect independently rehashed protected evidence.
    seen: set[str] = set()  # Reject duplicate aliases that could obscure one mutated file.
    for entry in entries:  # Rehash every originally protected training, model, partition, config, and provenance artifact.
        if not isinstance(entry, Mapping) or not all(name in entry for name in ("path", "sha256", "size_bytes", "roles")):  # Require the canonical complete record shape.
            raise ValueError(f"malformed protected artifact entry: {entry}")  # Surface the exact malformed record.
        relative = str(entry["path"])  # Normalize the campaign-relative identity once.
        if relative in seen:  # Reject duplicate path declarations.
            raise ValueError(f"duplicate protected artifact path: {relative}")  # Preserve one authoritative identity per artifact.
        seen.add(relative)  # Register the unique protected path.
        target = _campaign_file(root, relative)  # Resolve only a regular file inside the campaign.
        digest = _sha256_file(target)  # Recompute the complete exact-byte identity after primary execution.
        if digest != str(entry["sha256"]) or target.stat().st_size != int(entry["size_bytes"]):  # Require both hash and length to remain frozen.
            raise ValueError(f"protected artifact changed after primary execution: {relative}")  # Stop before any ablation solve.
        verified.append({"path": relative, "sha256": digest, "size_bytes": target.stat().st_size, "roles": list(entry["roles"])})  # Preserve compact reauthentication evidence.
    config_path = _campaign_file(root, "protocol/frozen_config.json")  # Resolve the protected runtime config independently from the request object.
    manifest_path = _campaign_file(root, "protocol/case_manifest.json")  # Resolve the protected manifest independently from the request object.
    config = _read_json_object(config_path, "frozen config")  # Decode the exact protected runtime configuration.
    if _sha256_file(config_path) != str(started.get("frozen_config_sha256", "")) or _sha256_file(manifest_path) != str(started.get("manifest_sha256", "")):  # Bind both live inputs to TEST_STARTED.
        raise ValueError("TEST_STARTED manifest or frozen-config identity differs from current protected bytes")  # Refuse post-primary input substitution.
    expected_code = config.get("code_sha256")  # Read the frozen claim-critical code path map.
    if not isinstance(expected_code, Mapping) or not expected_code:  # Require a nonempty explicit code seal.
        raise ValueError("frozen_config lacks the claim-critical code SHA mapping")  # Prevent an unauthenticated ablation driver.
    repository = Path(__file__).resolve().parents[2]  # Resolve the repository root containing the frozen code inventory.
    observed_code: dict[str, str] = {}  # Collect live exact-byte identities without consulting Git result state.
    for relative, expected in sorted(expected_code.items(), key=lambda item: str(item[0])):  # Rehash every claim-critical path in stable order.
        target = (repository / str(relative)).resolve()  # Resolve the frozen repository-relative source path.
        try:  # Prevent a malicious frozen path from escaping the repository.
            target.relative_to(repository)  # Require the source to remain within the reviewed worktree.
        except ValueError as exception:  # Convert a traversal attempt into a frozen-input violation.
            raise ValueError(f"frozen code path leaves repository: {relative}") from exception  # Report the exact invalid path.
        if not target.is_file() or target.is_symlink():  # Require concrete current source bytes.
            raise FileNotFoundError(f"frozen code file is missing: {target}")  # Reject an incomplete runtime installation.
        digest = _sha256_file(target)  # Hash current source bytes independently.
        if digest != str(expected):  # Require exact equality with the freeze accepted before primary testing.
            raise ValueError(f"claim-critical code changed after freeze: {relative}")  # Stop before executing altered diagnostics.
        observed_code[str(relative)] = digest  # Retain compact current code evidence.
    environment_path = _campaign_file(root, "protocol/environment.json")  # Resolve the protected native and Python environment lock.
    frozen_environment = _read_json_object(environment_path, "frozen environment")  # Decode only its already rehashed exact bytes.
    frozen_contract = _environment_contract(frozen_environment)  # Select stable interpreter, dependency, platform, solver, and thread fields.
    live_contract = _environment_contract(capture_environment())  # Capture the same live fields immediately before ablation execution.
    if live_contract != frozen_contract:  # Require the same numerical environment accepted before primary testing.
        raise ValueError(f"live ablation environment differs from frozen contract; frozen={_payload_sha256(frozen_contract)} live={_payload_sha256(live_contract)}")  # Report compact identities without noisy package dumps.
    freeze_commit = str(started.get("freeze_commit_sha", ""))  # Read the exact dedicated commit accepted before primary disclosure.
    current_head = _git(repository, "rev-parse", "HEAD")  # Resolve the current Git commit without inspecting result values.
    freeze_tag = _git(repository, "rev-parse", "--verify", FREEZE_GIT_REF, check=False)  # Resolve the immutable non-self-referential freeze tag.
    if not freeze_commit or current_head != freeze_commit or freeze_tag != freeze_commit:  # Require the same selected commit and tag after primary execution.
        raise ValueError("current HEAD or wmvla-p1-freeze tag differs from TEST_STARTED freeze commit")  # Prevent post-primary branch or tag substitution.
    return {"schema": FREEZE_SCHEMA, "freeze_index_path": str(index_path), "freeze_index_sha256": index_digest, "protected_artifact_count": len(verified), "protected_artifacts": verified, "manifest_sha256": _sha256_file(manifest_path), "frozen_config_sha256": _sha256_file(config_path), "implementation_commit_sha": index.get("implementation_commit_sha"), "freeze_commit_sha": freeze_commit, "freeze_git_ref": FREEZE_GIT_REF, "environment_contract_sha256": _payload_sha256(live_contract), "code_sha256": observed_code}  # Return complete post-primary transitive freeze evidence.


def _expected_primary_coordinates(cases: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:  # Reconstruct the full mandatory primary grid without reading result values.
    from .four_way_benchmark import ALL_METHODS, BUDGETS, MAX_SOLVES  # Reuse the frozen benchmark coordinates accepted by TEST_STARTED.
    expected: dict[tuple[str, int, str], dict[str, Any]] = {}  # Allocate one exact job identity per primary trajectory.
    for case in cases:  # Preserve complete sixteen-case coverage.
        for budget in BUDGETS:  # Preserve all three independent equation caps.
            for method in ALL_METHODS:  # Preserve every frozen comparator and seed label.
                key = (str(case["case_id"]), int(budget), str(method))  # Form the unique primary coordinate.
                expected[key] = {"case_id": key[0], "split": "test", "geometry_hash": str(case["geometry_hash"]), "equation_budget": key[1], "method": key[2], "max_solves": int(MAX_SOLVES)}  # Reconstruct the status identity exactly.
    return expected  # Return all 16 x 3 x 7 expected trajectories.


def _validate_primary_job_artifacts(root: Path, case: Mapping[str, Any], expected_job: Mapping[str, Any], status: Mapping[str, Any], expected_reference: Mapping[str, Any], solve_limits: Sequence[int]) -> dict[str, str]:  # Deep-read one terminal primary trajectory so no malformed late-case input first fails after new ablation solves.
    case_id = str(expected_job["case_id"])  # Recover the authenticated manifest case coordinate.
    budget = int(expected_job["equation_budget"])  # Recover the registered active-equation cap.
    method = str(expected_job["method"])  # Recover the exact primary method label.
    base = f"test/{case_id}/{budget}/{method}"  # Derive every raw path solely from authenticated coordinates.
    paths = {name: _campaign_file(root, f"{base}/{name}") for name in ("records.json", "mesh_receipts.json", "action_log.json", "timing.json", "prefix_results.json", "final_state.npz")}  # Resolve all mandatory regular in-campaign artifacts before interpreting any content.
    records_payload = _read_json_object(paths["records.json"], "primary method records")  # Decode the complete trajectory and posthoc denominator receipt.
    if records_payload.get("schema") != "wmvla-four-way-method-result-v1" or records_payload.get("protocol_id") != PROTOCOL_ID or records_payload.get("job") != dict(expected_job):  # Require exact records schema and registered identity.
        raise ValueError(f"primary records identity mismatch: {(case_id, budget, method)}")  # Reject stale or cross-job raw evidence before ablation output exists.
    expected_case = {"case_id": case_id, "parameters": case["parameters"], "config_hash": str(case["config_hash"]), "geometry_hash": str(case["geometry_hash"])}  # Reconstruct the sole accepted manifest snapshot stored by the benchmark.
    if records_payload.get("case") != expected_case or records_payload.get("completed") is not status.get("completed") or records_payload.get("failure") != status.get("failure"):  # Bind physical identity and terminal state across status and records.
        raise ValueError(f"primary records content disagrees with status or manifest: {(case_id, budget, method)}")  # Stop on cross-file drift before any new solve.
    record_rows = records_payload.get("records")  # Read every successfully completed real solve row.
    if not isinstance(record_rows, list) or len(record_rows) != int(status.get("successful_solve_count", -1)):  # Require raw solve cardinality to equal the atomic terminal marker.
        raise ValueError(f"primary records count mismatch: {(case_id, budget, method)}")  # Reject truncated or padded trajectories.
    for solve_index, record in enumerate(record_rows, start=1):  # Validate the minimal fields later prefix and diagnostic consumers cast directly.
        if not isinstance(record, Mapping) or int(record.get("solve_index", -1)) != solve_index or str(record.get("method", "")) != method or int(record.get("n_equations", 0)) <= 0:  # Require contiguous positive-resource records from the registered method.
            raise ValueError(f"primary solve record is malformed: {(case_id, budget, method, solve_index)}")  # Identify the exact late-failing row early.
    reference_receipt = records_payload.get("reference_b")  # Read the posthoc truth provenance attached only after online execution.
    if not isinstance(reference_receipt, Mapping) or reference_receipt.get("usage") != "posthoc_only" or reference_receipt.get("used_online") is not False:  # Require explicit truth isolation on every one of 336 primary jobs.
        raise ValueError(f"primary reference isolation receipt is invalid: {(case_id, budget, method)}")  # Refuse online-denominator leakage before diagnostics.
    primary_receipt = {str(name): value for name, value in reference_receipt.items() if name not in {"usage", "used_online"}}  # Remove only the two benchmark-added isolation declarations before comparing the original preflight receipt.
    if primary_receipt != dict(expected_reference):  # Require every primary job for this case to use the exact predeclared ledger, compact B, amendment, and verification evidence.
        raise ValueError(f"primary Reference B receipt differs from execution-plan preflight: {(case_id, budget, method)}")  # Prevent denominator drift hidden inside otherwise valid raw results.
    _reference_digest_identity(primary_receipt, f"primary records reference receipt for {(case_id, budget, method)}")  # Independently require complete canonical ledger and B hashes.
    prefix_payload = _read_json_object(paths["prefix_results.json"], "primary prefix results")  # Decode all four registered true-prefix outcomes before ablation start.
    if prefix_payload.get("schema") != "wmvla-four-way-prefix-results-v1" or prefix_payload.get("protocol_id") != PROTOCOL_ID or prefix_payload.get("job") != dict(expected_job) or prefix_payload.get("derivation") != "best_feasible_actual_prefix":  # Require exact prefix provenance and identity.
        raise ValueError(f"primary prefix result identity mismatch: {(case_id, budget, method)}")  # Reject stale or alternate prefix derivations.
    prefix_rows = prefix_payload.get("rows")  # Read the registered K grid.
    if not isinstance(prefix_rows, list) or len(prefix_rows) != len(tuple(solve_limits)):  # Require exactly one row per registered solve prefix.
        raise ValueError(f"primary prefix row count mismatch: {(case_id, budget, method)}")  # Reject incomplete or duplicated favorable prefixes.
    observed_limits: list[int] = []  # Preserve duplicates so set equality cannot hide them.
    for row in prefix_rows:  # Validate every row before selecting the K=6 reuse point.
        if not isinstance(row, Mapping):  # Require transparent row mappings.
            raise ValueError(f"primary prefix row is malformed: {(case_id, budget, method)}")  # Reject scalar or opaque rows early.
        solve_limit = int(row.get("solves", -1))  # Normalize the registered K coordinate.
        observed_limits.append(solve_limit)  # Retain exact cardinality and ordering-independent membership evidence.
        if str(row.get("case_id", "")) != case_id or str(row.get("method", "")) != method or int(row.get("equation_budget", -1)) != budget:  # Bind each row to its parent job.
            raise ValueError(f"primary prefix row identity mismatch: {(case_id, budget, method, solve_limit)}")  # Reject cross-job row substitution.
        for metric_name, ok_name in (("energy_error", "energy_ok"), ("qoi_error", "qoi_ok")):  # Validate both common posthoc metrics under identical semantics.
            metric = row.get(metric_name)  # Read the optional finite error value.
            metric_valid = metric is not None and math.isfinite(float(metric)) and float(metric) >= 0.0  # Accept only finite nonnegative successful errors.
            if bool(row.get(ok_name)) is not metric_valid:  # Require explicit availability flags to match the stored numerical value exactly.
                raise ValueError(f"primary prefix metric validity mismatch: {(case_id, budget, method, solve_limit, metric_name)}")  # Prevent later casts or silent failure relabeling.
    if sorted(observed_limits) != sorted(int(value) for value in solve_limits) or len(set(observed_limits)) != len(observed_limits):  # Require exactly K={2,3,4,6} with no duplicate masking.
        raise ValueError(f"primary prefix solve limits mismatch: {(case_id, budget, method)}")  # Stop before any diagnostic selects K=6.
    json_contracts = {"mesh_receipts.json": "wmvla-four-way-mesh-receipts-v1", "action_log.json": "wmvla-four-way-action-log-v1", "timing.json": "wmvla-four-way-timing-v1"}  # Name every remaining primary JSON schema consumed or hashed later.
    for filename, schema in json_contracts.items():  # Deep-decode and bind the remaining common raw artifacts.
        payload = _read_json_object(paths[filename], f"primary {filename}")  # Refuse malformed JSON during solve-free readiness.
        if payload.get("schema") != schema or payload.get("protocol_id") != PROTOCOL_ID or payload.get("job") != dict(expected_job):  # Require exact schema and job identity.
            raise ValueError(f"primary {filename} identity mismatch: {(case_id, budget, method)}")  # Surface the precise malformed artifact before execution.
    final_receipt = status.get("final_state")  # Read the atomic final-state identity recorded by the method status.
    if not isinstance(final_receipt, Mapping) or final_receipt.get("path") != "final_state.npz" or str(final_receipt.get("sha256", "")) != _sha256_file(paths["final_state.npz"]):  # Rehash the compressed state before trusting it.
        raise ValueError(f"primary final-state receipt mismatch: {(case_id, budget, method)}")  # Reject a truncated or replaced NPZ before ablation start.
    try:  # Parse the complete no-pickle state archive and validate its public field vocabulary.
        with np.load(paths["final_state.npz"], allow_pickle=False) as archive:  # Prevent object deserialization while checking compressed-array integrity.
            if set(archive.files) != {"nodes", "cells", "eta2", "region_labels", "available", "source"}:  # Require the exact benchmark final-state contract.
                raise ValueError("field vocabulary mismatch")  # Route malformed archive structure through the bounded job-specific error below.
            if archive["nodes"].ndim != 2 or archive["cells"].ndim != 2 or archive["eta2"].ndim != 1 or archive["region_labels"].ndim != 1 or archive["available"].shape != (1,) or archive["source"].shape != (1,):  # Require shape-safe arrays used by later diagnostics.
                raise ValueError("array shape mismatch")  # Route inconsistent arrays through the common preflight failure.
    except (OSError, ValueError, KeyError) as exception:  # Convert corrupt zip, NumPy, key, or shape failures into one actionable readiness error.
        raise ValueError(f"primary final-state archive is invalid: {(case_id, budget, method)}") from exception  # Preserve the original parser cause without starting ablations.
    hashes = {filename: _sha256_file(path) for filename, path in paths.items()}  # Bind all deeply validated raw bytes for later no-change checks.
    if method == "world_model_vla":  # Deep-read every one of the 48 mandatory behavior-neutral WM traces.
        trace_path = _campaign_file(root, f"{base}/prediction_trace.json")  # Resolve the exact job-local trace.
        trace_payload = _read_json_object(trace_path, "primary WM-full prediction trace")  # Decode top-level identity and incomplete-transition structure.
        if trace_payload.get("schema") != TRACE_SCHEMA or trace_payload.get("protocol_id") != PROTOCOL_ID or trace_payload.get("case_id") != case_id or trace_payload.get("variant") != "wm_full" or trace_payload.get("seed") is not None or not isinstance(trace_payload.get("incomplete_transitions"), list):  # Require the exact primary identity-control trace contract.
            raise ValueError(f"primary WM-full trace identity mismatch: {(case_id, budget)}")  # Reject cross-case, seeded, or structurally incomplete traces.
        load_diagnostic_trace(trace_path)  # Reconstruct every completed transition now so row-level type or shape errors cannot surface after new solves.
        hashes["prediction_trace.json"] = _sha256_file(trace_path)  # Bind the exact fully parsed trace bytes for WM-full reuse.
    return hashes  # Return exact validated raw identities for readiness evidence and reuse checks.


def _validate_primary_completion(request: AblationCampaignRequest, cases: Sequence[Mapping[str, Any]], started: Mapping[str, Any]) -> dict[str, Any]:  # Require the one-shot primary campaign to finish every registered job before diagnostics begin.
    from .four_way_benchmark import ALL_METHODS, BUDGETS, RESULT_SCHEMA, SOLVE_LIMITS, build_diagnostic_plan  # Reuse canonical primary schemas, resource prefixes, and the preregistered diagnostic plan.
    root = request.root.resolve()  # Normalize the campaign root once.
    invalid_path = root / "test" / "TEST_INVALID.json"  # Resolve the fatal primary invalidation marker.
    if invalid_path.exists():  # Refuse diagnostics after a protocol, API, or programming failure.
        raise ValueError(f"primary test campaign is invalidated: {invalid_path}")  # Preserve the disclosed primary evidence without continuation.
    summary_path = _campaign_file(root, "test/EXECUTION_SUMMARY.json")  # Require the atomic complete primary invocation ledger.
    summary = _read_json_object(summary_path, "primary execution summary")  # Decode the complete result index.
    if summary.get("schema") != "wmvla-four-way-execution-summary-v1" or summary.get("protocol_id") != PROTOCOL_ID:  # Require the canonical primary summary schema.
        raise ValueError("primary EXECUTION_SUMMARY has an incompatible schema")  # Reject stale or foreign result trees.
    case_ids = [str(case["case_id"]) for case in cases]  # Preserve the authenticated ascending blind case order.
    if started.get("case_order") != case_ids or started.get("case_count") != EXPECTED_TEST_CASES:  # Require TEST_STARTED to bind exactly the same complete blind set.
        raise ValueError("TEST_STARTED case order is not the complete ascending 16-case manifest test split")  # Prevent post-hoc case filtering or reordering.
    if started.get("budgets") != list(BUDGETS) or started.get("methods") != list(ALL_METHODS):  # Require all registered primary resources and methods.
        raise ValueError("TEST_STARTED primary budget or method grid differs from the frozen benchmark")  # Reject a partial primary campaign.
    plan = summary.get("plan")  # Read the solve-free plan embedded before primary execution.
    if not isinstance(plan, Mapping) or plan.get("diagnostic_plan") != build_diagnostic_plan(case_ids):  # Require this exact ablation campaign to have been preregistered before results.
        raise ValueError("primary execution summary lacks the exact preregistered ablation plan")  # Reject post-hoc diagnostic selection.
    planned_references = plan.get("reference_preflight")  # Recover the exact solve-free denominator receipts embedded before TEST_STARTED.
    if not isinstance(planned_references, Mapping) or set(str(value) for value in planned_references) != set(case_ids):  # Require one and only one primary preflight receipt for every blind case.
        raise ValueError("primary execution plan lacks the complete per-case reference preflight")  # Refuse unpinned denominators before deep-reading raw jobs.
    reference_receipts: dict[str, dict[str, Any]] = {}  # Normalize complete immutable preflight receipts for all 336 job comparisons.
    reference_identities: dict[str, dict[str, str]] = {}  # Retain compact ledger/B hashes for no-value pre-trajectory checks.
    for case_id in case_ids:  # Validate every predeclared denominator receipt before trusting repeated raw-job copies.
        receipt = planned_references.get(case_id)  # Select the exact manifest-owned case receipt.
        if not isinstance(receipt, Mapping):  # Require a transparent complete benchmark preflight object.
            raise ValueError(f"primary reference preflight receipt is malformed for {case_id}")  # Stop before any raw job or ablation output is opened.
        reference_receipts[case_id] = dict(receipt)  # Preserve the full verification, amendment, qualification, path, and digest evidence.
        reference_identities[case_id] = _reference_digest_identity(receipt, f"primary execution-plan reference receipt for {case_id}")  # Validate and retain canonical byte identities.
    expected = _expected_primary_coordinates(cases)  # Reconstruct every required job identity.
    outcomes = summary.get("job_outcomes")  # Read the invocation-level completion ledger.
    if not isinstance(outcomes, list) or len(outcomes) != len(expected):  # Require one terminal row for every expected primary job.
        raise ValueError(f"primary test must finish exactly {len(expected)} registered jobs before ablations")  # Reject interrupted or duplicated evidence.
    observed: set[tuple[str, int, str]] = set()  # Track unique terminal primary coordinates.
    status_hashes: dict[str, str] = {}  # Bind every terminal status artifact to exact bytes.
    raw_artifact_hashes: dict[str, dict[str, str]] = {}  # Bind every deeply parsed primary job artifact to the readiness snapshot.
    wm_full_artifact_hashes: dict[str, dict[str, str]] = {}  # Bind the sixteen exact B60000 WM-full inputs later reused without native calls.
    cases_by_id = {str(case["case_id"]): dict(case) for case in cases}  # Index the authenticated manifest snapshot for exact records comparisons.
    observed_successes = 0  # Recompute successful terminal count rather than trusting the summary declaration.
    observed_failures = 0  # Recompute retained typed-failure count from durable statuses.
    for outcome in outcomes:  # Validate each invocation ledger row against its durable status marker.
        if not isinstance(outcome, Mapping) or not isinstance(outcome.get("job"), Mapping):  # Require an explicit job mapping.
            raise ValueError("primary execution summary contains a malformed job outcome")  # Reject opaque terminal evidence.
        job = dict(outcome["job"])  # Copy the immutable job coordinates.
        key = (str(job.get("case_id")), int(job.get("equation_budget", -1)), str(job.get("method")))  # Recover the unique primary coordinate.
        if key not in expected or key in observed or job != expected[key]:  # Require complete identity equality and no reruns.
            raise ValueError(f"primary execution summary contains an unexpected or duplicate job: {key}")  # Surface the exact bad coordinate.
        observed.add(key)  # Register this primary terminal outcome.
        if outcome.get("status") not in ("completed", "failed"):  # Test mode forbids resume and therefore has only fresh terminal statuses.
            raise ValueError(f"primary job lacks a terminal fresh status: {key}")  # Reject partial or resumed blind evidence.
        relative = f"test/{key[0]}/{key[1]}/{key[2]}/status.json"  # Derive the sole trusted status location from authenticated coordinates.
        status_path = _campaign_file(root, relative)  # Require the durable atomic marker.
        status = _read_json_object(status_path, "primary method status")  # Decode its exact terminal state.
        if status.get("schema") != RESULT_SCHEMA or status.get("protocol_id") != PROTOCOL_ID or status.get("job") != expected[key]:  # Require exact status identity.
            raise ValueError(f"primary status marker does not match registered job: {key}")  # Reject stale or cross-job evidence.
        failure = status.get("failure")  # Read an optional retained typed native failure.
        if status.get("completed") is not True and (not isinstance(failure, Mapping) or failure.get("category") not in ("calculix_numerical", "gmsh_numerical")):  # Permit only the benchmark's two typed native failure families.
            raise ValueError(f"primary non-completion is not a retained typed native failure: {key}")  # Refuse programming or integrity failures disguised as method points.
        expected_outcome_status = "completed" if status.get("completed") is True else "failed"  # Derive the sole accepted invocation-ledger label from the durable marker.
        if outcome.get("status") != expected_outcome_status or bool(outcome.get("completed")) is not bool(status.get("completed")):  # Require the execution summary to agree with the terminal job bytes.
            raise ValueError(f"primary execution outcome disagrees with status: {key}")  # Reject altered or stale aggregate claims.
        observed_successes += int(status.get("completed") is True)  # Count each successful trajectory once.
        observed_failures += int(status.get("completed") is not True)  # Count each retained typed native failure once.
        artifact_hashes = _validate_primary_job_artifacts(root, cases_by_id[key[0]], expected[key], status, reference_receipts[key[0]], SOLVE_LIMITS)  # Deep-read all raw JSON, prefixes, traces, NPZ, and denominator evidence before any new solve.
        artifact_key = "/".join((key[0], str(key[1]), key[2]))  # Form a portable stable raw-job evidence key.
        status_hashes[artifact_key] = _sha256_file(status_path)  # Bind every atomic status to exact bytes.
        raw_artifact_hashes[artifact_key] = {**artifact_hashes, "status.json": status_hashes[artifact_key]}  # Preserve the complete deep-validated primary byte snapshot.
        if key[1] == ABLATION_BUDGET and key[2] == "world_model_vla":  # Select exactly the sixteen identity-control trajectories reused later.
            wm_full_artifact_hashes[key[0]] = dict(raw_artifact_hashes[artifact_key])  # Pin every reused input byte before ablation output exists.
    if observed != set(expected):  # Recheck complete set equality independently from list length.
        raise ValueError("primary execution summary omits registered job coordinates")  # Refuse an incomplete blind campaign.
    if observed_successes + observed_failures != len(expected) or int(summary.get("completed_job_count", -1)) != observed_successes or int(summary.get("successful_job_count", -1)) != observed_successes or int(summary.get("failed_job_count", -1)) != observed_failures or int(summary.get("terminal_job_count", -1)) != len(expected) or summary.get("all_jobs_completed") is not True:  # Recompute every aggregate terminal count from durable raw statuses.
        raise ValueError("primary execution summary terminal counts disagree with raw jobs")  # Refuse a favorable or stale aggregate before diagnostics.
    if set(wm_full_artifact_hashes) != set(case_ids):  # Require one complete pinned B60000 WM-full identity for every blind case.
        raise ValueError("primary WM-full byte inventory is incomplete")  # Prevent a late missing reuse input.
    return {"execution_summary_path": str(summary_path), "execution_summary_sha256": _sha256_file(summary_path), "job_count": len(observed), "completed_job_count": observed_successes, "failed_job_count": observed_failures, "status_sha256": status_hashes, "raw_artifact_sha256": raw_artifact_hashes, "wm_full_artifact_sha256": wm_full_artifact_hashes, "reference_receipt_by_case": reference_receipts, "reference_sha256_by_case": reference_identities, "diagnostic_plan": build_diagnostic_plan(case_ids)}  # Return deep terminal primary evidence, pinned reuse bytes, and exact denominators without filtering failures.


def validate_ablation_readiness(request: AblationCampaignRequest) -> AblationReadiness:  # Validate freeze, full primary completion, references, partitions, and models before any ablation artifact is created.
    from ..bridge_case_manifest import load_case_manifest  # Reuse the canonical manifest checksum, geometry, LHS, and split validator.
    from .four_way_benchmark import _preflight_models, _preflight_partitions, _preflight_references, load_frozen_config, select_manifest_cases  # Reuse primary read-only model, partition, and Reference-B checks.
    root = request.root.resolve()  # Normalize the selected campaign root once.
    started_path = _campaign_file(root, "test/TEST_STARTED.json")  # Require the irreversible primary start marker.
    started = _read_json_object(started_path, "primary TEST_STARTED marker")  # Decode the exact primary disclosure boundary.
    if started.get("schema") != "wmvla-four-way-test-started-v1" or started.get("protocol_id") != PROTOCOL_ID or started.get("one_shot") is not True or started.get("resume_allowed") is not False:  # Require the canonical one-shot primary boundary.
        raise ValueError("primary TEST_STARTED marker is incompatible with formal ablations")  # Reject development, resumed, or foreign runs.
    freeze_evidence = _protected_post_primary_freeze(request, started)  # Reauthenticate all frozen bytes without invoking the now-inapplicable TEST_NOT_RUN scan.
    manifest = load_case_manifest(request.manifest_path, verify_checksum=True)  # Revalidate exact manifest bytes, sidecar, LHS, geometry, and hashes.
    cases = tuple(select_manifest_cases(manifest, "test"))  # Select all authenticated test cases in ascending case-id order.
    if len(cases) != EXPECTED_TEST_CASES or tuple(str(case["case_id"]) for case in cases) != tuple(sorted(str(case["case_id"]) for case in cases)):  # Require the complete sorted blind unit.
        raise ValueError("formal ablations require exactly sixteen ascending manifest test cases")  # Reject shards and reordered campaigns.
    config = load_frozen_config(request.frozen_config_path)  # Revalidate canonical common gradation and runtime config structure.
    if bool(request.allow_unqualified_references) is not bool(config.get("allow_unqualified_references", False)) or bool(request.allow_unqualified_references) is not bool(started.get("allow_unqualified_references", False)):  # Require the exact waiver policy frozen and used by the primary campaign.
        raise ValueError("ablation --allow-unqualified-references must exactly match frozen_config and TEST_STARTED")  # Prevent post-primary relaxation or silent removal of an acknowledged operational amendment.
    primary_evidence = _validate_primary_completion(request, cases, started)  # Require all primary jobs and WM-full identity traces before diagnostics.
    reference_evidence = _preflight_references(root, config, cases, allow_unqualified=request.allow_unqualified_references)  # Reauthenticate every qualified or explicitly authorized Reference B under the exact frozen strict-or-expedited schedule without binding it to an online runner.
    pinned_references = primary_evidence.get("reference_receipt_by_case")  # Recover the exact denominator receipts used by all 336 primary jobs.
    if not isinstance(pinned_references, Mapping) or dict(reference_evidence) != dict(pinned_references):  # Require current cache verification, paths, bytes, qualification, and amendment evidence to remain byte-for-byte identical to primary preflight.
        raise ValueError("current Reference B preflight differs from the complete primary campaign")  # Stop during solve-free readiness rather than mixing denominators after fresh ablation solves.
    partition_evidence = _preflight_partitions(root, config, cases)  # Reauthenticate every shared semantic partition and common uniform probe.
    model_evidence = _preflight_models(root, config)  # Reload and construct every frozen learned runtime solve-free.
    evidence = {"schema": "wmvla-four-way-ablation-readiness-v1", "protocol_id": PROTOCOL_ID, "validated_utc": _utc_now(), "test_started_path": str(started_path), "test_started_sha256": _sha256_file(started_path), "freeze": freeze_evidence, "primary": primary_evidence, "references": reference_evidence, "partitions": partition_evidence, "models": model_evidence, "allow_unqualified_references": bool(request.allow_unqualified_references), "REFERENCE_QUALIFIED": bool(started.get("REFERENCE_QUALIFIED", False)), "reference_unqualified_case_ids": list(started.get("reference_unqualified_case_ids", [])), "validated_freeze": True, "primary_test_complete": True}  # Assemble the full post-primary readiness chain and preserve qualification truth.
    return AblationReadiness(cases=cases, config=config, evidence=evidence)  # Return authenticated immutable inputs for solve-free planning or execution.


def build_ablation_campaign_jobs(root: Path, case_ids: Sequence[str]) -> tuple[AblationCampaignJob, ...]:  # Construct the exact 160 outcome coordinates in frozen per-case execution order.
    jobs: list[AblationCampaignJob] = []  # Allocate the complete ordered variant grid.
    for case_id in sorted(str(value) for value in case_ids):  # Preserve ascending manifest case order at the outer level.
        case_root = root / "ablations" / case_id  # Resolve the mandated per-case raw evidence root.
        for variant in ("wm_full", "wm_h1", "wm_prior_only", "wm_no_history"):  # Execute identity reuse and deployable deterministic controls before seeded controls.
            jobs.append(AblationCampaignJob(case_id, variant, None, case_root / variant))  # Register one unique deterministic coordinate.
        for seed in RANDOM_SAFE_SEEDS:  # Execute all five random-safe repetitions without selection.
            jobs.append(AblationCampaignJob(case_id, "random_safe_extra", int(seed), case_root / "random_safe_extra" / f"seed_{int(seed)}"))  # Register the exact seed-specific directory.
        jobs.append(AblationCampaignJob(case_id, "oracle_future_hit", None, case_root / "oracle_future_hit"))  # Register oracle execution after its separate source phase.
    return tuple(jobs)  # Return all 16 x 10 outcome jobs without invoking any solver.


def build_ablation_campaign_plan(request: AblationCampaignRequest, readiness: AblationReadiness) -> dict[str, Any]:  # Build the exact post-primary plan including independent oracle source phases.
    case_ids = [str(case["case_id"]) for case in readiness.cases]  # Preserve the authenticated ascending case order.
    jobs = build_ablation_campaign_jobs(request.root.resolve(), case_ids)  # Construct every scored outcome coordinate.
    ordered_phases: list[dict[str, Any]] = []  # Expand the oracle source as its own prior real-solve phase.
    by_case = {case_id: [job for job in jobs if job.case_id == case_id] for case_id in case_ids}  # Group stable outcome coordinates without changing order.
    for case_id in case_ids:  # Preserve ascending blind case order throughout execution.
        for job in by_case[case_id][:-1]:  # Emit full reuse, three deterministic ablations, and five seeded controls first.
            mode = "reuse_primary_identity" if job.variant == "wm_full" else "fresh_native_variant"  # Distinguish the sole no-solve identity phase.
            ordered_phases.append({"case_id": job.case_id, "variant": job.variant, "seed": job.seed, "mode": mode, "output_dir": str(job.output_dir), "equation_budget": ABLATION_BUDGET, "max_solves": ABLATION_MAX_SOLVES})  # Preserve exact resources and destination.
        source_dir = request.root.resolve() / "ablations" / case_id / "oracle_source_dorfler"  # Resolve the independent future-source directory.
        ordered_phases.append({"case_id": case_id, "variant": "oracle_source_dorfler", "seed": None, "mode": "fresh_independent_dorfler_then_freeze_future_hits", "output_dir": str(source_dir), "equation_budget": ABLATION_BUDGET, "max_solves": ABLATION_MAX_SOLVES})  # Require source completion before schedule derivation.
        oracle = by_case[case_id][-1]  # Recover the sole oracle outcome job.
        ordered_phases.append({"case_id": oracle.case_id, "variant": oracle.variant, "seed": None, "mode": "fresh_native_variant_from_completed_source", "output_dir": str(oracle.output_dir), "equation_budget": ABLATION_BUDGET, "max_solves": ABLATION_MAX_SOLVES})  # Register the post-source oracle phase.
    return {"schema": CAMPAIGN_PLAN_SCHEMA, "protocol_id": PROTOCOL_ID, "created_utc": _utc_now(), "dry_run": bool(request.dry_run), "case_order": case_ids, "case_count": len(case_ids), "equation_budget": ABLATION_BUDGET, "max_solves": ABLATION_MAX_SOLVES, "random_safe_seeds": list(RANDOM_SAFE_SEEDS), "allow_unqualified_references": bool(request.allow_unqualified_references), "REFERENCE_QUALIFIED": readiness.evidence.get("REFERENCE_QUALIFIED"), "outcome_job_count": len(jobs), "oracle_source_count": len(case_ids), "estimated_max_new_real_solves": (len(jobs) - len(case_ids)) * ABLATION_MAX_SOLVES + len(case_ids) * ABLATION_MAX_SOLVES, "wm_full_new_real_solves": 0, "wm_full_identity_source": "test/<case_id>/60000/world_model_vla", "ordered_phases": ordered_phases, "readiness": readiness.evidence}  # Return a complete solve-free plan with honest source, reference qualification, and solve bounds.


def _job_identity(job: AblationCampaignJob) -> dict[str, Any]:  # Serialize one formal ablation outcome coordinate.
    return {"case_id": job.case_id, "variant": job.variant, "seed": job.seed, "equation_budget": ABLATION_BUDGET, "max_solves": ABLATION_MAX_SOLVES}  # Return only preregistered scientific coordinates.


def _probe_sha(readiness: AblationReadiness, case_id: str) -> str:  # Recover one exact common uniform-probe identity from authenticated partition preflight.
    partitions = readiness.evidence.get("partitions")  # Read the case-keyed shared semantic evidence.
    row = partitions.get(case_id) if isinstance(partitions, Mapping) else None  # Select only this manifest case.
    digest = str(row.get("probe_sha256", "")) if isinstance(row, Mapping) else ""  # Read the complete probe mesh digest.
    if len(digest) != 64:  # Require a full SHA-256 rather than a display prefix.
        raise ValueError(f"shared partition lacks a complete common-probe identity for {case_id}")  # Stop before an incomparable run.
    return digest  # Return the authenticated common-probe identity.


def _aggregate_campaign_traces(paths: Sequence[str | Path]) -> dict[str, Any]:  # Aggregate complete traces while preserving honest all-failure or zero-transition campaigns.
    sources = tuple(sorted((Path(value) for value in paths), key=lambda value: str(value)))  # Normalize deterministic exact input order.
    if not sources:  # Require an explicit diagnostic artifact for every campaign aggregate.
        raise ValueError("campaign trace aggregate requires at least one persisted trace")  # Surface missing raw evidence.
    records = tuple(record for path in sources for record in load_diagnostic_trace(path))  # Restore every completed real transition exactly once.
    trace_sources = [{"path": str(path), "sha256": _sha256_file(path)} for path in sources]  # Bind every input to exact bytes.
    if records:  # Compute all mandatory calibration quantities when at least one prediction was realized.
        report = aggregate_world_model_diagnostics(records)  # Reuse the canonical finite diagnostic formulas.
        report["status"] = "measured"  # Disclose that aggregate quantities have completed transition support.
        report["trace_sources"] = trace_sources  # Attach exact raw provenance after arithmetic.
        return report  # Return the measured content-bound aggregate.
    return {"schema": "wmvla-four-way-prediction-aggregate-v1", "protocol_id": PROTOCOL_ID, "status": "not_measurable", "reason": "No predicted action had a successful subsequent real solve; retained native failures and natural one-solve stops provide no realized transition.", "transition_count": 0, "case_count": 0, "total_error_log_mae": None, "equation_mape": None, "regional_delta_log_eta2_mae": None, "regional_delta_log_elements_mae": None, "prediction_interval_coverage": {"definition": "one_sided_upper_bounds_emitted_by_frozen_v0", "total_error_upper": None, "equations_upper": None, "joint_upper": None, "two_sided_status": "not_measurable", "two_sided_reason": "Frozen V0 emits no lower bounds, and no completed transition is available."}, "candidate_ranking_spearman": {"status": "not_measurable", "reason": "No completed transition exists; unselected candidates would additionally require forbidden extra real solves.", "transition_count": 0, "per_transition": []}, "proactive_acceptance_rate": None, "proactive_execution_rate": None, "accepted_real_improvement_rate": None, "fallback_cause_counts": {name: 0 for name in ("uncertainty", "budget", "low_gain", "distrust", "other")}, "mean_uncertainty": None, "mean_failure_probability": None, "trace_sources": trace_sources}  # Preserve every required field with explicit unavailable values rather than fabricating zeros.


def _failure_at_solve(runner: Any) -> int:  # Locate the first unsuccessful native attempt after completed records.
    counter = int(getattr(runner, "_counter", len(runner.records)))  # Read the honest solve counter advanced before a CalculiX attempt.
    return int(max(counter if counter > len(runner.records) else len(runner.records) + 1, 1))  # Identify the current failed attempt or next failed meshing stage.


def _retained_native_failure(exception: BaseException, runner: Any) -> dict[str, Any] | None:  # Convert only benchmark-typed native failures into finite ablation evidence.
    from .four_way_benchmark import _numerical_failure_payload  # Reuse the sole benchmark-native failure classifier.
    native = _numerical_failure_payload(exception)  # Classify CalculiX and Gmsh failures without broad exception swallowing.
    if native is None:  # Keep integrity, API, reference, serialization, and programming errors fatal.
        return None  # Tell the caller to re-raise the original exception unchanged.
    return {**native, "exception_type": type(exception).__name__, "message": str(exception).replace("\x00", " ")[:1000], "successful_solve_count": len(runner.records), "failure_at_solve": _failure_at_solve(runner), "traceback": traceback.format_exc(limit=40)}  # Preserve bounded typed-native diagnostics.


def _prefix_energy(case: Mapping[str, Any], output_dir: Path, records: Sequence[Any], completed: bool, failure: Mapping[str, Any] | None, action_payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # Reuse Reference-B best-feasible K=6 semantics from the primary harness.
    from .four_way_benchmark import ExecutionJob, derive_prefix_rows  # Reuse the exact registered true-prefix implementation.
    primary_shaped = ExecutionJob(str(case["case_id"]), "test", str(case["geometry_hash"]), ABLATION_BUDGET, "world_model_vla", output_dir)  # Adapt only the method label needed for WM safety extraction.
    rows = derive_prefix_rows(primary_shaped, records, completed, failure, action_payload)  # Derive K=2,3,4,6 from the one actual trajectory without rerunning.
    selected = next((dict(row) for row in rows if int(row["solves"]) == ABLATION_MAX_SOLVES), None)  # Select the sole registered K=6 row.
    if selected is None:  # Defend a future incompatible prefix schema.
        raise ValueError("benchmark prefix derivation did not produce K=6")  # Stop rather than inventing a score.
    selected["variant_method_label"] = str(action_payload.get("variant", ""))  # Disclose that the primary-shaped adapter did not relabel the raw native run.
    return selected, rows  # Return the scored operating point and complete public prefix evidence.


class NativeAblationExecutor:  # Execute formal variants through ReceiptFemRunner, shared partitions, frozen V0 components, and posthoc Reference B.
    def __init__(self, request: AblationCampaignRequest) -> None:  # Bind one executor to the already validated campaign root.
        self.request = request  # Store canonical paths and the sole output root without opening any result.
    def reuse_primary(self, case: Mapping[str, Any], job: AblationCampaignJob, readiness: AblationReadiness) -> AblationOutcome:  # Reuse WM-full exact identity without constructing a runner or invoking native tools.
        from .four_way_benchmark import _write_json  # Reuse atomic strict-JSON persistence for identity receipts.
        case_id = str(case["case_id"])  # Read the authenticated case coordinate.
        primary = self.request.root.resolve() / "test" / case_id / str(ABLATION_BUDGET) / "world_model_vla"  # Resolve the sole primary K6-B60000 trajectory.
        prefix_path = _campaign_file(self.request.root.resolve(), f"test/{case_id}/{ABLATION_BUDGET}/world_model_vla/prefix_results.json")  # Resolve its scored true-prefix result.
        trace_path = _campaign_file(self.request.root.resolve(), f"test/{case_id}/{ABLATION_BUDGET}/world_model_vla/prediction_trace.json")  # Resolve its behavior-neutral diagnostic trace.
        records_path = _campaign_file(self.request.root.resolve(), f"test/{case_id}/{ABLATION_BUDGET}/world_model_vla/records.json")  # Resolve its complete online-isolation evidence.
        action_path = _campaign_file(self.request.root.resolve(), f"test/{case_id}/{ABLATION_BUDGET}/world_model_vla/action_log.json")  # Resolve its exact planner and certificate evidence.
        status_path = _campaign_file(self.request.root.resolve(), f"test/{case_id}/{ABLATION_BUDGET}/world_model_vla/status.json")  # Resolve its terminal atomic status.
        pinned_artifacts = _assert_pinned_primary_wm_full_files(self.request.root.resolve(), readiness, case_id)  # Rehash the entire readiness-pinned trajectory before any ablation directory or identity receipt exists.
        prefix = _read_json_object(prefix_path, "primary WM-full prefix result")  # Decode the exact K grid.
        selected = next((dict(row) for row in prefix.get("rows", ()) if isinstance(row, Mapping) and int(row.get("solves", -1)) == ABLATION_MAX_SOLVES), None)  # Select only K=6.
        if selected is None:  # Require the registered scored operating point.
            raise ValueError(f"primary WM-full prefix lacks K=6 for {case_id}")  # Refuse a substitute rerun.
        trace_payload = _read_json_object(trace_path, "primary WM-full prediction trace")  # Decode identity fields independently from row loading.
        if trace_payload.get("case_id") != case_id or trace_payload.get("variant") != "wm_full" or trace_payload.get("seed") is not None:  # Require exact case and identity-control coordinates.
            raise ValueError(f"primary WM-full trace identity mismatch for {case_id}")  # Reject cross-case or ablation trace substitution.
        transitions = load_diagnostic_trace(trace_path)  # Restore completed transition evidence for proactive counts.
        executed = sum(bool(item.proactive_executed) for item in transitions)  # Count only proactive actions followed by a successful real solve.
        certified = sum(bool(item.proactive_certified) for item in transitions)  # Count complete raw-and-compiled structural certificates.
        records_payload = _read_json_object(records_path, "primary WM-full records")  # Read sealed-truth provenance without exposing competitor trajectories.
        reference_receipt = records_payload.get("reference_b")  # Recover the posthoc-only truth-use declaration.
        isolated = bool(isinstance(reference_receipt, Mapping) and reference_receipt.get("used_online") is False and reference_receipt.get("usage") == "posthoc_only")  # Require explicit online truth isolation.
        matched = bool(not selected.get("budget_violation", True) and int(selected.get("equation_budget", -1)) == ABLATION_BUDGET and int(selected.get("solves", -1)) == ABLATION_MAX_SOLVES)  # Require the exact matched resource point.
        energy = selected.get("energy_error")  # Read the posthoc Reference-B energy result or retained failure sentinel.
        ok = bool(selected.get("energy_ok") and _finite_nonnegative(None if energy is None else float(energy), True) and matched)  # Accept only a finite nonnegative matched-budget K=6 result.
        outcome = reuse_primary_wm_full(case_id, None if energy is None else float(energy), ok, executed, certified, _probe_sha(readiness, case_id), trace_path, prefix_path, competitor_isolation=isolated)  # Bind exact primary bytes without executing WM-full again.
        outcome = replace(outcome, matched_budget=matched)  # Preserve an honest false value if primary evidence violates the matched point.
        job.output_dir.mkdir(parents=True, exist_ok=False)  # Create the identity-only raw directory once after the global one-shot marker.
        receipt = {"schema": ABLATION_RESULT_SCHEMA, "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "mode": "reuse_primary_identity_no_native_calls", "reused_from_primary": True, "new_real_solves": 0, "primary_root": str(primary), "primary_artifacts": {"prefix_results": {"path": str(prefix_path), "sha256": pinned_artifacts["prefix_results.json"]}, "prediction_trace": {"path": str(trace_path), "sha256": pinned_artifacts["prediction_trace.json"]}, "records": {"path": str(records_path), "sha256": pinned_artifacts["records.json"]}, "action_log": {"path": str(action_path), "sha256": pinned_artifacts["action_log.json"]}, "status": {"path": str(status_path), "sha256": pinned_artifacts["status.json"]}}, "readiness_pinned_artifact_sha256": pinned_artifacts, "outcome": asdict(outcome), "competitor_isolation": isolated, "common_probe_sha256": outcome.common_probe_sha256}  # Preserve complete content-bound reuse evidence and the full immediately rechecked raw-byte inventory.
        _write_json(job.output_dir / "identity_receipt.json", receipt)  # Publish the no-rerun identity receipt before its terminal marker.
        _write_json(job.output_dir / "status.json", {"schema": ABLATION_RESULT_SCHEMA, "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "completed": True, "reused_from_primary": True, "new_real_solves": 0, "artifacts": ["identity_receipt.json"], "source_result_sha256": outcome.result_sha256})  # Publish the atomic identity completion marker last.
        return outcome  # Return the exact primary result as the WM-full diagnostic point.
    def run_variant(self, case: Mapping[str, Any], job: AblationCampaignJob, readiness: AblationReadiness, *, oracle_schedule: Mapping[int, OracleActionChoice] | None = None) -> AblationOutcome:  # Execute one fresh isolated wrapper stack and retain every raw artifact.
        from .four_way_benchmark import ReceiptFemRunner, _append_jsonl, _artifact_path, _attach_posthoc_reference_metrics, _copy_solver_logs, _dataclass_config, _json_safe, _load_shared_partitions, _nested_config, _reference_payload, _write_final_state, _write_json  # Reuse the primary runner and exact artifact/reference helpers.
        from .world.model import ResidualWorldModel  # Reload the authenticated frozen transition model independently per run.
        from .world.pipeline import WorldVLAConfig, run_world_model_vla  # Reuse the unchanged V0 real-solve loop.
        from .world.planner import MultiStepPlanner, PlannerConfig  # Reconstruct the exact frozen base planner before applying isolated wrappers.
        from .world.tool_gateway import MCPToolGateway, ToolConfig  # Reconstruct the exact deterministic compiler and safety certificates.
        from ..bridge_case_manifest import problem_from_case  # Reconstruct geometry solely from the authenticated manifest row.
        case_id = str(case["case_id"])  # Read the immutable case coordinate.
        if job.case_id != case_id or job.variant == "wm_full":  # Reserve WM-full for identity reuse and reject cross-case dispatch.
            raise ValueError("native variant execution requires a matching non-full campaign job")  # Preserve the formal phase boundary.
        pinned_reference = _assert_pinned_reference_files(self.request.root.resolve(), readiness, case_id)  # Rehash only primary-pinned denominator bytes before output creation or the first fresh real solve.
        job.output_dir.mkdir(parents=True, exist_ok=False)  # Claim the exact fresh variant directory once without resume or overwrite.
        trace_path = job.output_dir / "prediction_trace.json"  # Reserve the complete cross-layer prediction-versus-actual trace.
        problem = problem_from_case(case)  # Rebuild the canonical bridge problem without reading any result value.
        runner = ReceiptFemRunner(problem, job.output_dir, keep_files=False, ccx_timeout=float(readiness.config.get("ccx_timeout_s", 1800.0)))  # Create fresh honest solve accounting with no Reference B.
        if runner.reference is not None:  # Defend unexpected constructor truth injection.
            raise ValueError("fresh ablation runner unexpectedly contains Reference B")  # Stop before the first online action.
        partition_started = time.perf_counter()  # Measure shared semantic loading separately from solver time.
        world_partition, _rl_partition, partition_receipt = _load_shared_partitions(self.request.root.resolve(), readiness.config, case, problem)  # Load the one frozen partition shared with every comparator.
        runner.final_partition = world_partition  # Retain shared labels solely for solve-free final-state persistence.
        partition_s = float(time.perf_counter() - partition_started)  # Record exact solve-free partition overhead.
        model_path = _artifact_path(self.request.root.resolve(), readiness.config, "world_model")  # Rehash the sole frozen model immediately before this isolated trajectory.
        base_model = ResidualWorldModel.load(model_path)  # Reload a fresh model so no test case or variant shares online residual state.
        planner_values = _nested_config(readiness.config, "world_planner", "planner_config")  # Read the complete reviewed V0 planner fields.
        base_planner = MultiStepPlanner(_dataclass_config(PlannerConfig, planner_values))  # Construct the unchanged V0 planner.
        tool_values = _nested_config(readiness.config, "world_tool_config")  # Read the complete reviewed deterministic tool fields.
        base_gateway = MCPToolGateway(_dataclass_config(ToolConfig, tool_values, {"theta": 0.5, "max_extra_depth": base_planner.config.max_extra_depth}))  # Bind common Dörfler settings without changing V0 logic.
        runtime = build_ablation_runtime(case_id, job.variant, base_model, base_planner, base_gateway, seed=job.seed, oracle_schedule=oracle_schedule, trace_path=trace_path)  # Apply only the declared isolated adapter or state transform.
        world_values = _nested_config(readiness.config, "world_model_runtime", "world_vla_config")  # Read the complete reviewed online-loop fields.
        settings = _dataclass_config(WorldVLAConfig, world_values, {"max_solves": ABLATION_MAX_SOLVES, "n_equation_cap": ABLATION_BUDGET, "theta": 0.5, "method_name": f"ablation_{job.variant}" if job.seed is None else f"ablation_random_{job.seed}", "artifact_dir": str(job.output_dir), "require_reference": False})  # Bind only the preregistered operating point and audit label.
        result: Any | None = None  # Reserve a complete V0 result only when the native trajectory returns normally.
        failure: dict[str, Any] | None = None  # Reserve one retained typed native failure.
        completed = False  # Mark numerical completion only after the unchanged pipeline returns.
        started_utc = _utc_now()  # Record the exact variant start boundary.
        method_started = time.perf_counter()  # Measure complete success or partial native-failure online duration.
        try:  # Retain only explicit CalculiX or Gmsh native failures.
            result = run_world_model_vla(runner, partition=world_partition, config=settings, model=runtime.model, planner=runtime.planner, gateway=runtime.gateway)  # Execute the unchanged V0 loop with isolated wrappers and no Reference B.
            completed = True  # Mark normal natural-stop or K=6 completion.
        except Exception as exception:  # Classify without weakening programming and integrity failures.
            failure = _retained_native_failure(exception, runner)  # Accept only the two benchmark-native failure families.
            if failure is None:  # Keep every other exception campaign-fatal.
                raise  # Preserve original type and traceback for ABLATION_INVALID.
            _append_jsonl(self.request.root.resolve() / "ablations" / "failure_ledger.jsonl", {"schema": "wmvla-four-way-ablation-failure-ledger-v1", "protocol_id": PROTOCOL_ID, "recorded_utc": _utc_now(), "job": _job_identity(job), "failure": failure})  # Retain the typed native failure without deleting the case.
        online_total_s = float(time.perf_counter() - method_started)  # Stop total online timing after normal return or typed failure.
        if runner.reference is not None:  # Prove no online code path acquired truth.
            raise ValueError(f"ablation variant {job.variant} acquired forbidden Reference B")  # Invalidate the campaign before posthoc scoring.
        reference, reference_receipt = _load_posthoc_reference(self.request.root.resolve(), readiness.config, case, allow_unqualified=self.request.allow_unqualified_references)  # Load the already preflighted Reference B under the identical frozen strict-or-two-level schedule only after all online actions and meshes are irrevocably fixed.
        _assert_reference_receipt_identity(reference_receipt, pinned_reference, case_id)  # Bind posthoc scoring to the exact ledger and compact B bytes used throughout the primary campaign.
        _attach_posthoc_reference_metrics(runner.records, reference, reference_receipt)  # Compute common energy and QoI errors strictly posthoc.
        if runner.reference is not None:  # Reassert that posthoc scoring did not bind truth back into the runner.
            raise ValueError(f"posthoc scoring bound Reference B to {job.variant}")  # Preserve online isolation evidence.
        _write_json(trace_path, runtime.diagnostics.payload())  # Publish an explicit trace even for zero-transition natural stops or first-attempt native failure.
        result_payload = None if result is None else _json_safe(result)  # Serialize the complete returned V0 result without inventing partial fields.
        certificates = [] if result is None else [_json_safe(value) for value in result.certificates]  # Preserve every exact compiler certificate on normal return.
        actions = [] if result is None else [list(value) for value in result.actions]  # Preserve every actually selected post-certification action on normal return.
        decisions = [] if result is None else [_json_safe(value) for value in result.decisions]  # Preserve every planner decision on normal return.
        action_payload = {"variant": job.variant, "seed": job.seed, "result": result_payload, "actions": actions, "decisions": decisions, "certificates": certificates, "isolation_receipt": runtime.isolation_receipt, "frozen_model": {"path": str(model_path), "sha256": _sha256_file(model_path)}, "prediction_trace": {"path": str(trace_path), "sha256": _sha256_file(trace_path)}}  # Assemble complete policy, isolation, and trace evidence.
        selected_prefix, prefix_rows = _prefix_energy(case, job.output_dir, runner.records, completed, failure, action_payload)  # Score the matched K=6/B60000 point with primary semantics.
        copied_logs = _copy_solver_logs(runner, job.output_dir / "solver_logs")  # Retain every surviving native log on success and typed failure.
        final_state = _write_final_state(job.output_dir / "final_state.npz", runner)  # Persist the final successful mesh and ZZ field without another solve.
        records_payload = {"schema": ABLATION_RESULT_SCHEMA, "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "case": {"case_id": case_id, "parameters": case["parameters"], "config_hash": case["config_hash"], "geometry_hash": case["geometry_hash"]}, "reference_b": {**dict(reference_receipt), "usage": "posthoc_only", "used_online": False}, "completed": completed, "failure": failure, "records": [_json_safe(record) for record in runner.records]}  # Preserve every SolveRecord and sealed-truth provenance.
        records_path = job.output_dir / "records.json"  # Select the exact raw trajectory result identity.
        _write_json(records_path, records_payload)  # Persist raw solve records before any completion marker.
        _write_json(job.output_dir / "mesh_receipts.json", {"schema": "wmvla-four-way-ablation-mesh-receipts-v1", "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "receipts": runner.mesh_receipts})  # Persist full mesh hashes, resources, return codes, and native log paths.
        _write_json(job.output_dir / "action_log.json", {"schema": "wmvla-four-way-ablation-action-log-v1", "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "partition_spec": partition_receipt, "actions": action_payload, "failure": failure})  # Persist decisions, candidate execution, certificates, and isolation transforms.
        calculix_s = float(sum(float(record.wall_s) for record in runner.records))  # Sum successful native solver durations independently.
        timing = dict(getattr(result, "timing_s", {})) if result is not None and isinstance(getattr(result, "timing_s", {}), Mapping) else {}  # Preserve separated V0 timing only when returned.
        timing.update({"online_total_s": online_total_s, "calculix_s": calculix_s, "shared_partition_s": partition_s, "partial_trajectory": not completed, "unattributed_or_failure_overhead_s": max(online_total_s - calculix_s, 0.0)})  # Preserve complete success or partial-failure timing transparently.
        _write_json(job.output_dir / "timing.json", {"schema": "wmvla-four-way-ablation-timing-v1", "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "timing_s": timing, "solver_logs": copied_logs, "completed": completed})  # Persist timing separately from accuracy.
        _write_json(job.output_dir / "prefix_results.json", {"schema": "wmvla-four-way-ablation-prefix-results-v1", "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "derivation": "benchmark_best_feasible_actual_prefix", "rows": prefix_rows})  # Persist all K prefixes derived from the one real trajectory.
        prediction_report = _aggregate_campaign_traces((trace_path,))  # Aggregate this run's realized prediction quality or explicit non-measurability.
        _write_json(job.output_dir / "prediction_diagnostics.json", prediction_report)  # Persist the complete per-run calibration and fallback report.
        matched = bool(len(runner.records) <= ABLATION_MAX_SOLVES and all(int(record.n_equations) <= ABLATION_BUDGET for record in runner.records) and not selected_prefix.get("budget_violation", True))  # Require actual solve and equation caps without substituting over-budget states.
        energy_value = selected_prefix.get("energy_error")  # Read the posthoc scored K=6 energy result.
        ok = bool(selected_prefix.get("energy_ok") and _finite_nonnegative(None if energy_value is None else float(energy_value), True) and matched)  # Accept only finite nonnegative matched-budget evidence.
        transitions = runtime.diagnostics.records  # Read only completed predicted-to-real transitions.
        executed = sum(bool(item.proactive_executed) for item in transitions)  # Count actually solved proactive actions.
        certified = sum(bool(item.proactive_certified) for item in transitions)  # Count proactive actions with complete compiled-field certificates.
        isolation = bool(runtime.isolation_receipt.get("competitor_results_accessed") is False and runner.reference is None)  # Require explicit no-competitor and no-truth state.
        outcome = AblationOutcome(case_id=case_id, variant=job.variant, energy_error=None if energy_value is None else float(energy_value), ok=ok, executed_proactive_actions=executed, certified_proactive_actions=certified, common_probe_sha256=_probe_sha(readiness, case_id), matched_budget=matched, competitor_isolation=isolation, trace_path=str(trace_path), result_path=str(records_path), result_sha256=_sha256_file(records_path), seed=job.seed, reused_from_primary=False)  # Build the analyzer-facing content-bound result.
        status = {"schema": ABLATION_RESULT_SCHEMA, "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "started_utc": started_utc, "finished_utc": _utc_now(), "completed": completed, "successful_solve_count": len(runner.records), "failure": failure, "outcome": asdict(outcome), "final_state": final_state, "artifacts": ["records.json", "mesh_receipts.json", "action_log.json", "timing.json", "prefix_results.json", "final_state.npz", "prediction_trace.json", "prediction_diagnostics.json"]}  # Assemble the terminal raw-artifact index.
        _write_json(job.output_dir / "status.json", status)  # Publish the atomic variant completion marker last.
        return outcome  # Return the retained success or typed-native failure outcome without dropping the case.
    def run_oracle_source(self, case: Mapping[str, Any], output_dir: Path, readiness: AblationReadiness) -> OracleSourceExecution:  # Complete one independent exact-Dörfler trajectory before any oracle schedule exists.
        from .four_way_benchmark import ReceiptFemRunner, _append_jsonl, _attach_posthoc_reference_metrics, _copy_solver_logs, _dataclass_config, _json_safe, _load_shared_partitions, _nested_config, _reference_payload, _write_final_state, _write_json  # Reuse the primary runner and exact raw-artifact helpers.
        from .world.planner import MultiStepPlanner, PlannerConfig  # Reconstruct frozen V0 eligibility only after source states are measured.
        from .world.tool_gateway import MCPToolGateway, ToolConfig  # Use the exact Dörfler target compiler and active-equation preflight.
        from ..bridge_case_manifest import problem_from_case  # Reconstruct the authenticated bridge geometry.
        from ..experiment import initial_mesh  # Generate the same common uniform initial mesh as every online method.
        from ..indicators import zz_indicator  # Evaluate the exact common elementwise ZZ squared indicator.
        case_id = str(case["case_id"])  # Read the immutable manifest case identity.
        pinned_reference = _assert_pinned_reference_files(self.request.root.resolve(), readiness, case_id)  # Rehash only primary-pinned denominator bytes before output creation or the first independent source solve.
        output_dir.mkdir(parents=True, exist_ok=False)  # Claim the independent source directory once before native execution.
        problem = problem_from_case(case)  # Reconstruct geometry without opening any competitor result.
        runner = ReceiptFemRunner(problem, output_dir, keep_files=False, ccx_timeout=float(readiness.config.get("ccx_timeout_s", 1800.0)))  # Create fresh honest solve accounting with no Reference B.
        if runner.reference is not None:  # Defend accidental truth injection.
            raise ValueError("fresh oracle-source runner unexpectedly contains Reference B")  # Stop before the common probe.
        partition_started = time.perf_counter()  # Measure shared semantic verification separately.
        world_partition, _rl_partition, partition_receipt = _load_shared_partitions(self.request.root.resolve(), readiness.config, case, problem)  # Load the same frozen partition used by full and every ablation.
        runner.final_partition = world_partition  # Retain shared labels solely for final-state persistence.
        partition_s = float(time.perf_counter() - partition_started)  # Record exact partition overhead.
        planner = MultiStepPlanner(_dataclass_config(PlannerConfig, _nested_config(readiness.config, "world_planner", "planner_config")))  # Reconstruct unchanged V0 eligibility and action-space settings.
        gateway = MCPToolGateway(_dataclass_config(ToolConfig, _nested_config(readiness.config, "world_tool_config"), {"theta": 0.5, "max_extra_depth": planner.config.max_extra_depth}))  # Reconstruct the exact compiled Dörfler gateway.
        states: list[WorldState] = []  # Collect only successful independently measured source states.
        certificates: list[Any] = []  # Collect exact Dörfler compiled-field and resource receipts.
        actions: list[tuple[int, ...]] = []  # Collect every source transition action.
        failure: dict[str, Any] | None = None  # Reserve one retained typed native failure.
        stop_reason = "max_solves"  # Default to the complete six-solve prefix.
        started_utc = _utc_now()  # Record the independent source start boundary.
        source_started = time.perf_counter()  # Measure the complete online source trajectory.
        try:  # Retain only typed native solver and mesher failures.
            mesh = initial_mesh(problem)  # Generate the exact common uniform probe without semantic size hints.
            hit_count: np.ndarray | None = None  # Initialize recurrence evidence independently from every other trajectory.
            for step in range(ABLATION_MAX_SOLVES):  # Execute at most the matched K=6 real solves.
                post, record = runner.solve_mesh(mesh, method="oracle_source_dorfler", stage=f"cycle{step}", extra={"theta": 0.5, "gradation": 1.0, "independent_future_source": True})  # Execute one counted real source solve.
                eta2 = np.asarray(zz_indicator(problem, post), dtype=float).reshape(-1)  # Evaluate the exact common squared estimator.
                observation = gateway.observe_solve(problem, world_partition, post, record, eta2, hit_count, step)  # Aggregate the source into the shared region ordering.
                hit_count = observation.state.hit_count.copy()  # Preserve only this source trajectory's measured Dörfler history.
                states.append(observation.state)  # Freeze the successful state before any next remesh.
                record.extra.update({"oracle_source_schema": ORACLE_SOURCE_SCHEMA, "independent_future_source": True, "wmvla_step": int(step), "wmvla_indicator_sum": float(np.sum(eta2)), "wmvla_regions": list(observation.state.names), "wmvla_region_error": observation.state.err_sum.tolist(), "wmvla_region_elements": observation.state.elems.tolist(), "wmvla_dorfler_error_fraction": observation.state.dorfler_error_fraction.tolist(), "wmvla_dorfler_element_fraction": observation.state.dorfler_element_fraction.tolist()})  # Preserve complete source observations on raw SolveRecords.
                if step + 1 >= ABLATION_MAX_SOLVES:  # Stop after the complete matched real-solve horizon.
                    stop_reason = "max_solves"  # Record full-prefix completion.
                    break  # Prevent any seventh source solve.
                if observation.state.n_equations >= ABLATION_BUDGET:  # Stop at the measured active-equation cap.
                    stop_reason = "equation_cap_reached"  # Record the natural resource stop.
                    break  # Preserve the last feasible state without another remesh.
                if not np.any(observation.marked):  # Stop when exact Dörfler has no positive contribution.
                    stop_reason = "no_marked_elements"  # Record the natural estimator stop.
                    break  # End the completed independent trajectory.
                baseline = RegionAction.dorfler(observation.state)  # Construct the exact zero-extra-depth action.
                materialized = gateway.materialize_action(observation, baseline, ABLATION_BUDGET)  # Compile and exact-preflight only the Dörfler candidate.
                certificates.append(materialized.certificate)  # Preserve the complete raw and compiled target certificate.
                actions.append(materialized.action.extra_depth)  # Preserve the actually selected zero-depth action.
                record.extra.update({"oracle_source_certificate": _json_safe(materialized.certificate), "oracle_source_timing_s": dict(materialized.timing_s)})  # Attach source resource and mesh-generation evidence to the preceding solve.
                if materialized.mesh is None:  # Stop before an over-budget next real solve.
                    stop_reason = "dorfler_candidate_exceeds_cap"  # Record exact candidate resource termination.
                    break  # Preserve strict matched-budget source evidence.
                mesh = materialized.mesh  # Advance to the exact compiled Dörfler candidate.
        except Exception as exception:  # Classify the independent source failure without broad swallowing.
            failure = _retained_native_failure(exception, runner)  # Accept only typed CalculiX or Gmsh failures.
            if failure is None:  # Keep any API, partition, indicator, or certificate defect campaign-fatal.
                raise  # Preserve the original exception for ABLATION_INVALID.
            stop_reason = "solver_failure"  # Mark the source unusable for oracle execution after any native failure.
            _append_jsonl(self.request.root.resolve() / "ablations" / "failure_ledger.jsonl", {"schema": "wmvla-four-way-ablation-failure-ledger-v1", "protocol_id": PROTOCOL_ID, "recorded_utc": _utc_now(), "job": {"case_id": case_id, "variant": "oracle_source_dorfler", "seed": None, "equation_budget": ABLATION_BUDGET, "max_solves": ABLATION_MAX_SOLVES}, "failure": failure})  # Retain the failed source without deriving a partial oracle.
        source_total_s = float(time.perf_counter() - source_started)  # Stop complete source timing after normal or typed failure termination.
        if runner.reference is not None:  # Prove the independently executed source never acquired truth.
            raise ValueError("oracle source acquired forbidden Reference B")  # Invalidate before posthoc scoring.
        reference, reference_receipt = _load_posthoc_reference(self.request.root.resolve(), readiness.config, case, allow_unqualified=self.request.allow_unqualified_references)  # Load Reference B under the identical frozen strict-or-two-level schedule only after the full source action sequence is fixed.
        _assert_reference_receipt_identity(reference_receipt, pinned_reference, case_id)  # Bind source posthoc scoring to the exact ledger and compact B bytes used throughout the primary campaign.
        _attach_posthoc_reference_metrics(runner.records, reference, reference_receipt)  # Compute common posthoc energy and QoI errors.
        probe_sha = _probe_sha(readiness, case_id)  # Bind source states to the same authenticated common uniform probe.
        trajectory = build_dorfler_future_trajectory(case_id, states, planner, probe_sha, stop_reason) if failure is None and states else None  # Freeze future hits only from a normally completed full or natural-stop source.
        source_state_payload = {"schema": ORACLE_SOURCE_SCHEMA, "protocol_id": PROTOCOL_ID, "case_id": case_id, "equation_budget": ABLATION_BUDGET, "max_solves": ABLATION_MAX_SOLVES, "source_method": "dorfler", "completed_before_oracle_derivation": failure is None, "stop_reason": stop_reason, "failure": failure, "common_probe_sha256": probe_sha, "states": [_json_safe(state) for state in states], "trajectory": None if trajectory is None else _json_safe(trajectory)}  # Assemble complete measured future-source evidence.
        _write_json(output_dir / "future_trajectory.json", source_state_payload)  # Publish the source states before any caller can derive an oracle schedule.
        copied_logs = _copy_solver_logs(runner, output_dir / "solver_logs")  # Retain every surviving CalculiX log.
        final_state = _write_final_state(output_dir / "final_state.npz", runner)  # Persist the last successful source mesh and ZZ field without another solve.
        records_path = output_dir / "records.json"  # Select the exact independent source result identity.
        _write_json(records_path, {"schema": ORACLE_SOURCE_SCHEMA, "protocol_id": PROTOCOL_ID, "case": {"case_id": case_id, "parameters": case["parameters"], "config_hash": case["config_hash"], "geometry_hash": case["geometry_hash"]}, "reference_b": {**dict(reference_receipt), "usage": "posthoc_only", "used_online": False}, "completed": failure is None, "source_usable": trajectory is not None, "failure": failure, "records": [_json_safe(record) for record in runner.records]})  # Persist every source SolveRecord and sealed-truth provenance.
        _write_json(output_dir / "mesh_receipts.json", {"schema": "wmvla-four-way-oracle-source-mesh-receipts-v1", "protocol_id": PROTOCOL_ID, "case_id": case_id, "receipts": runner.mesh_receipts})  # Persist exact attempted mesh and native-return evidence.
        _write_json(output_dir / "action_log.json", {"schema": "wmvla-four-way-oracle-source-action-log-v1", "protocol_id": PROTOCOL_ID, "case_id": case_id, "partition_spec": partition_receipt, "actions": [list(value) for value in actions], "certificates": [_json_safe(value) for value in certificates], "oracle_schedule_derived": False, "failure": failure})  # Prove source execution preceded schedule construction.
        calculix_s = float(sum(float(record.wall_s) for record in runner.records))  # Sum successful source solver durations.
        tool_s = float(sum(float(record.extra.get("oracle_source_timing_s", {}).get("tool_total", 0.0)) for record in runner.records))  # Sum recorded exact Dörfler materialization durations without inventing missing values.
        _write_json(output_dir / "timing.json", {"schema": "wmvla-four-way-oracle-source-timing-v1", "protocol_id": PROTOCOL_ID, "case_id": case_id, "timing_s": {"online_total_s": source_total_s, "calculix_s": calculix_s, "shared_partition_s": partition_s, "dorfler_materialization_s": tool_s, "partial_trajectory": failure is not None}, "solver_logs": copied_logs})  # Persist complete source timing separately from accuracy.
        status_path = output_dir / "status.json"  # Reserve the atomic source completion marker.
        _write_json(status_path, {"schema": ORACLE_SOURCE_SCHEMA, "protocol_id": PROTOCOL_ID, "case_id": case_id, "started_utc": started_utc, "finished_utc": _utc_now(), "completed": failure is None, "source_usable": trajectory is not None, "successful_solve_count": len(runner.records), "stop_reason": stop_reason, "failure": failure, "common_probe_sha256": probe_sha, "final_state": final_state, "artifacts": ["records.json", "mesh_receipts.json", "action_log.json", "timing.json", "future_trajectory.json", "final_state.npz"]})  # Publish terminal source status only after every raw artifact exists.
        return OracleSourceExecution(case_id=case_id, trajectory=trajectory, failure=failure, common_probe_sha256=probe_sha, result_path=str(records_path), result_sha256=_sha256_file(records_path), status_path=str(status_path))  # Return immutable source identity for strictly subsequent schedule derivation.


def _assert_pristine_ablation_output(root: Path) -> None:  # Enforce a one-shot no-resume boundary for all formal mechanism diagnostics.
    ablations = root.resolve() / "ablations"  # Resolve the sole protocol-owned diagnostic evidence tree.
    if ablations.is_symlink() or (ablations.exists() and (not ablations.is_dir() or any(ablations.iterdir()))):  # Reject symlink aliases plus prior markers, partial cases, summaries, or failure ledgers.
        entries = sorted(str(path.relative_to(root.resolve())) for path in ablations.rglob("*") if path.is_file() or path.is_symlink()) if ablations.is_dir() and not ablations.is_symlink() else [str(ablations)]  # Collect bounded visible evidence without traversing an attacker-controlled symlink root.
        raise ValueError("formal ablation output is not pristine; rerun and resume are forbidden: " + ", ".join(entries[:50] or ["ablations/<existing-entry>"]))  # Preserve interrupted disclosed evidence exactly as found.


def _write_unavailable_oracle(job: AblationCampaignJob, source: OracleSourceExecution) -> AblationOutcome:  # Retain a failed oracle point when its independent source cannot support a schedule.
    from .four_way_benchmark import _write_json  # Reuse atomic strict-JSON persistence.
    job.output_dir.mkdir(parents=True, exist_ok=False)  # Create the exact scored oracle directory without native execution.
    trace_path = job.output_dir / "prediction_trace.json"  # Persist an explicit empty trace rather than a dangling path.
    trace_payload = {"schema": TRACE_SCHEMA, "protocol_id": PROTOCOL_ID, "case_id": job.case_id, "variant": "oracle_future_hit", "seed": None, "transitions": [], "incomplete_transitions": [], "execution_status": "not_run_source_unavailable", "source_status_path": source.status_path, "source_failure": source.failure}  # Explain why no prediction could be realized.
    _write_json(trace_path, trace_payload)  # Publish complete unavailable diagnostic evidence.
    unavailable_path = job.output_dir / "unavailable.json"  # Select the exact failed-result identity.
    unavailable = {"schema": ABLATION_RESULT_SCHEMA, "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "completed": False, "failure": {"category": "oracle_source_unavailable", "reason": "Independent Dörfler source did not complete normally; no partial future trajectory is used.", "source_status_path": source.status_path, "source_status_sha256": _sha256_file(source.status_path), "source_result_path": source.result_path, "source_result_sha256": source.result_sha256, "source_failure": source.failure}, "new_real_solves": 0, "competitor_results_accessed": False}  # Preserve a finite non-native derived failure without attempting a favorable substitute source.
    _write_json(unavailable_path, unavailable)  # Persist the exact failed result before constructing its outcome hash.
    outcome = AblationOutcome(case_id=job.case_id, variant="oracle_future_hit", energy_error=None, ok=False, executed_proactive_actions=0, certified_proactive_actions=0, common_probe_sha256=source.common_probe_sha256, matched_budget=True, competitor_isolation=True, trace_path=str(trace_path), result_path=str(unavailable_path), result_sha256=_sha256_file(unavailable_path), seed=None, reused_from_primary=False)  # Retain the case as a failed oracle point.
    _write_json(job.output_dir / "prediction_diagnostics.json", _aggregate_campaign_traces((trace_path,)))  # Persist explicit calibration non-measurability.
    _write_json(job.output_dir / "status.json", {"schema": ABLATION_RESULT_SCHEMA, "protocol_id": PROTOCOL_ID, "job": _job_identity(job), "completed": False, "failure": unavailable["failure"], "outcome": asdict(outcome), "artifacts": ["unavailable.json", "prediction_trace.json", "prediction_diagnostics.json"]})  # Publish the terminal unavailable status last.
    return outcome  # Return the retained failed diagnostic point.


def _derive_and_persist_oracle_schedule(source: OracleSourceExecution, output_dir: Path, readiness: AblationReadiness) -> dict[int, OracleActionChoice]:  # Derive future-hit actions strictly after independent source status publication.
    from .four_way_benchmark import _dataclass_config, _nested_config, _write_json  # Reconstruct frozen planner settings and persist the derivation atomically.
    from .world.planner import PlannerConfig  # Reuse the exact V0 action-space configuration.
    if source.failure is not None or source.trajectory is None:  # Refuse every typed-failure or empty source before accessing future hits.
        raise ValueError("oracle schedule requires a normally completed independent Dörfler source")  # Prevent partial-trajectory leakage.
    status_path = Path(source.status_path)  # Resolve the already published source completion marker.
    status = _read_json_object(status_path, "oracle source status")  # Decode its temporal completion evidence.
    if status.get("completed") is not True or status.get("source_usable") is not True:  # Require normal source completion before derivation.
        raise ValueError("oracle source status is not normally complete and usable")  # Preserve strict phase ordering.
    planner_config = _dataclass_config(PlannerConfig, _nested_config(readiness.config, "world_planner", "planner_config"))  # Reconstruct unchanged V0 action bounds and discount.
    schedule = derive_oracle_schedule(source.trajectory, planner_config)  # Compute the fixed future-hit upper-bound schedule from completed real source states only.
    future_path = output_dir / "future_trajectory.json"  # Resolve the exact source payload written before status.
    payload = {"schema": "wmvla-four-way-oracle-schedule-v1", "protocol_id": PROTOCOL_ID, "case_id": source.case_id, "derived_utc": _utc_now(), "derived_after_source_completion": True, "source_status": {"path": str(status_path), "sha256": _sha256_file(status_path)}, "source_result": {"path": source.result_path, "sha256": source.result_sha256}, "source_future_trajectory": {"path": str(future_path), "sha256": _sha256_file(future_path)}, "objective": "maximize_discounted_realized_future_dorfler_error_fraction_within_v0_action_space", "schedule": {str(step): asdict(choice) for step, choice in sorted(schedule.items())}}  # Bind every derived action to exact previously completed source bytes.
    _write_json(output_dir / "oracle_schedule.json", payload)  # Publish the fixed schedule before the oracle runner is constructed.
    return schedule  # Return immutable derived choices for one later fresh oracle trajectory.


def run_ablation_campaign(request: AblationCampaignRequest, *, executor: Any | None = None) -> dict[str, Any]:  # Validate and execute the complete formal sixteen-case mechanism campaign once.
    from .four_way_benchmark import _write_json, _write_json_exclusive  # Reuse atomic and exclusive strict-JSON persistence.
    readiness = validate_ablation_readiness(request)  # Authenticate freeze, primary completion, references, partitions, and models before any output mutation.
    plan = build_ablation_campaign_plan(request, readiness)  # Construct the exact ordered 160 outcomes plus sixteen source phases.
    if request.dry_run:  # Stop before directories, markers, Gmsh, or CalculiX.
        return plan  # Return the complete post-primary solve-free plan.
    root = request.root.resolve()  # Normalize the sole campaign root.
    _assert_pristine_ablation_output(root)  # Refuse any partial or previous diagnostic evidence.
    ablation_root = root / "ablations"  # Resolve the formal evidence tree after the final pristine check.
    ablation_root.mkdir(parents=True, exist_ok=True)  # Create only the already verified empty tree.
    start_payload = {"schema": CAMPAIGN_START_SCHEMA, "protocol_id": PROTOCOL_ID, "started_utc": _utc_now(), "one_shot": True, "resume_allowed": False, "validated_freeze": True, "primary_test_complete": True, "plan": plan, "executor_type": type(executor).__qualname__ if executor is not None else NativeAblationExecutor.__qualname__}  # Bind the irreversible start to exact readiness and phase order.
    _write_json_exclusive(ablation_root / "ABLATION_STARTED.json", start_payload)  # Publish the one-shot boundary before any case artifact or native solve.
    _write_json(ablation_root / "EXECUTION_PLAN.json", plan)  # Persist a convenient standalone copy of the preregistered plan.
    active_executor = executor if executor is not None else NativeAblationExecutor(request)  # Use injected fake traces only for tests; the CLI always constructs the native executor.
    jobs = build_ablation_campaign_jobs(root, [str(case["case_id"]) for case in readiness.cases])  # Reconstruct scored jobs from the authenticated plan coordinates.
    jobs_by_case = {str(case["case_id"]): [job for job in jobs if job.case_id == str(case["case_id"])] for case in readiness.cases}  # Group jobs without reordering variants.
    phase_outcomes: list[dict[str, Any]] = []  # Retain every identity, native, source, and derived-unavailable terminal phase.
    case_summaries: list[dict[str, Any]] = []  # Collect all sixteen analyzer-facing case summaries.
    all_trace_paths: list[str] = []  # Collect every full and ablated prediction trace for global calibration.
    active_phase: dict[str, Any] | None = None  # Identify the exact fatal stop boundary if a non-native error occurs.
    try:  # Invalidate and stop immediately on integrity, API, reference, serialization, or programming failures.
        for case in readiness.cases:  # Preserve the ascending authenticated case order.
            case_id = str(case["case_id"])  # Read the immutable case coordinate.
            case_jobs = jobs_by_case[case_id]  # Recover full, three deterministic, five seeded, and oracle jobs in frozen order.
            outcomes: list[AblationOutcome] = []  # Collect exactly ten scored diagnostic outcomes for this case.
            full_job = case_jobs[0]  # Recover the sole WM-full identity coordinate.
            active_phase = {"case_id": case_id, "variant": "wm_full", "seed": None, "mode": "reuse_primary_identity"}  # Record the current no-solve boundary.
            full = active_executor.reuse_primary(case, full_job, readiness)  # Reuse exact primary bytes without constructing a real runner.
            outcomes.append(full)  # Retain the identity result once.
            phase_outcomes.append({**active_phase, "status": "reused_primary", "ok": bool(full.ok), "result_path": full.result_path, "result_sha256": full.result_sha256})  # Record the zero-new-solve terminal phase.
            for job in case_jobs[1:-1]:  # Execute WM-h1, prior-only, no-history, and all five random-safe seeds from fresh models.
                active_phase = {"case_id": case_id, "variant": job.variant, "seed": job.seed, "mode": "fresh_native_variant"}  # Record the exact active native coordinate.
                outcome = active_executor.run_variant(case, job, readiness)  # Execute the isolated wrapper through the real or injected runner.
                outcomes.append(outcome)  # Retain success or typed-native failure without dropping the case.
                phase_outcomes.append({**active_phase, "status": "terminal", "ok": bool(outcome.ok), "result_path": outcome.result_path, "result_sha256": outcome.result_sha256})  # Record the content-bound terminal result.
            source_dir = ablation_root / case_id / "oracle_source_dorfler"  # Resolve the separate independent source phase.
            active_phase = {"case_id": case_id, "variant": "oracle_source_dorfler", "seed": None, "mode": "fresh_independent_dorfler"}  # Record the active future-source coordinate.
            source = active_executor.run_oracle_source(case, source_dir, readiness)  # Complete the source before any schedule derivation.
            phase_outcomes.append({**active_phase, "status": "terminal", "source_usable": source.trajectory is not None and source.failure is None, "result_path": source.result_path, "result_sha256": source.result_sha256, "status_path": source.status_path})  # Record independent source completion or typed failure.
            oracle_job = case_jobs[-1]  # Recover the sole scored oracle coordinate.
            if source.trajectory is not None and source.failure is None:  # Derive and execute only after normal source completion.
                active_phase = {"case_id": case_id, "variant": "oracle_future_hit", "seed": None, "mode": "derive_schedule_then_fresh_native_variant"}  # Record the strictly post-source oracle boundary.
                schedule = _derive_and_persist_oracle_schedule(source, source_dir, readiness)  # Freeze future-hit choices to exact completed source bytes.
                oracle = active_executor.run_variant(case, oracle_job, readiness, oracle_schedule=schedule)  # Execute a fresh model, planner wrapper, gateway, mesh, and runner.
            else:  # Retain an oracle failure without exploiting partial future information.
                active_phase = {"case_id": case_id, "variant": "oracle_future_hit", "seed": None, "mode": "not_run_source_unavailable"}  # Record the transparent derived failure boundary.
                oracle = _write_unavailable_oracle(oracle_job, source)  # Publish a complete failed point with zero hidden solves.
            outcomes.append(oracle)  # Retain the scored oracle result once.
            phase_outcomes.append({**active_phase, "status": "terminal", "ok": bool(oracle.ok), "result_path": oracle.result_path, "result_sha256": oracle.result_sha256})  # Record the oracle terminal result.
            trace_paths = [item.trace_path for item in outcomes]  # Collect full identity and every fresh variant trace.
            prediction_report = _aggregate_campaign_traces(trace_paths)  # Compute per-case calibration, coverage, actions, and fallback causes.
            case_root = ablation_root / case_id  # Resolve the mandated per-case evidence directory.
            _write_json(case_root / "prediction_diagnostics.json", prediction_report)  # Persist the complete per-case prediction aggregate.
            _write_json(case_root / "ablation_outcomes.json", {"schema": "wmvla-four-way-ablation-outcomes-v1", "protocol_id": PROTOCOL_ID, "case_id": case_id, "outcomes": [asdict(item) for item in outcomes]})  # Persist all ten normalized scored outcomes without selection.
            case_summary = build_ablation_case_summary(outcomes, prediction_report)  # Build the fixed analyzer-facing complete mechanism record.
            case_summary["artifact_paths"] = {"case_summary": str(case_root / "ablation_case.json"), "outcomes": str(case_root / "ablation_outcomes.json"), "prediction_diagnostics": str(case_root / "prediction_diagnostics.json"), "oracle_source": str(source_dir), "trace_paths": trace_paths}  # Index all raw and aggregate evidence in stable locations.
            _write_json(case_root / "ablation_case.json", case_summary)  # Publish the analyzer's canonical per-case summary path.
            case_summaries.append(case_summary)  # Retain this complete case for the sixteen-case campaign index.
            all_trace_paths.extend(trace_paths)  # Add every case trace to global calibration without duplication.
        campaign_prediction = _aggregate_campaign_traces(all_trace_paths)  # Aggregate prediction quality and fallback causes across the complete blind unit.
        _write_json(ablation_root / "PREDICTION_DIAGNOSTICS.json", campaign_prediction)  # Persist the global mandatory calibration artifact.
        campaign_summary = build_ablation_campaign_summary(case_summaries)  # Validate exact sixteen-case uniqueness and mechanism evidence.
        campaign_summary["artifact_paths"] = {"prediction_diagnostics": str(ablation_root / "PREDICTION_DIAGNOSTICS.json"), "execution_summary": str(ablation_root / "ABLATION_EXECUTION_SUMMARY.json"), "case_summaries": [str(ablation_root / str(case["case_id"]) / "ablation_case.json") for case in readiness.cases]}  # Expose stable analyzer and audit locations.
        _write_json(ablation_root / "CAMPAIGN_SUMMARY.json", campaign_summary)  # Persist the complete sixteen-case mechanism summary.
        execution_summary = {"schema": CAMPAIGN_EXECUTION_SCHEMA, "protocol_id": PROTOCOL_ID, "finished_utc": _utc_now(), "plan_path": str(ablation_root / "EXECUTION_PLAN.json"), "plan_sha256": _sha256_file(ablation_root / "EXECUTION_PLAN.json"), "phase_count": len(phase_outcomes), "outcome_job_count": len(jobs), "oracle_source_count": EXPECTED_TEST_CASES, "phase_outcomes": phase_outcomes, "case_summary_paths": campaign_summary["artifact_paths"]["case_summaries"], "campaign_summary_path": str(ablation_root / "CAMPAIGN_SUMMARY.json"), "prediction_diagnostics_path": str(ablation_root / "PREDICTION_DIAGNOSTICS.json"), "native_or_scored_failure_count": sum(not bool(row.get("ok", row.get("source_usable", True))) for row in phase_outcomes if row.get("variant") != "wm_full"), "all_phases_terminal": len(phase_outcomes) == EXPECTED_TEST_CASES * 11}  # Assemble the complete 176-phase terminal ledger.
        _write_json(ablation_root / "ABLATION_EXECUTION_SUMMARY.json", execution_summary)  # Persist every terminal phase before the completion marker.
        complete = {"schema": "wmvla-four-way-ablation-complete-v1", "protocol_id": PROTOCOL_ID, "finished_utc": _utc_now(), "case_count": EXPECTED_TEST_CASES, "outcome_job_count": len(jobs), "oracle_source_count": EXPECTED_TEST_CASES, "campaign_summary": {"path": str(ablation_root / "CAMPAIGN_SUMMARY.json"), "sha256": _sha256_file(ablation_root / "CAMPAIGN_SUMMARY.json")}, "execution_summary": {"path": str(ablation_root / "ABLATION_EXECUTION_SUMMARY.json"), "sha256": _sha256_file(ablation_root / "ABLATION_EXECUTION_SUMMARY.json")}, "prediction_diagnostics": {"path": str(ablation_root / "PREDICTION_DIAGNOSTICS.json"), "sha256": _sha256_file(ablation_root / "PREDICTION_DIAGNOSTICS.json")}, "rerun_allowed": False}  # Build the atomic full-campaign completion identity.
        _write_json(ablation_root / "ABLATION_COMPLETE.json", complete)  # Publish completion last so partial evidence is never mistaken for a finished campaign.
        return {**execution_summary, "completion_path": str(ablation_root / "ABLATION_COMPLETE.json"), "completion_sha256": _sha256_file(ablation_root / "ABLATION_COMPLETE.json")}  # Return the durable formal execution index.
    except Exception as exception:  # Preserve fatal non-native protocol, API, serialization, reference, and programming failures.
        _write_json(ablation_root / "ABLATION_INVALID.json", {"schema": "wmvla-four-way-ablation-invalid-v1", "protocol_id": PROTOCOL_ID, "invalidated_utc": _utc_now(), "error_type": type(exception).__name__, "error": str(exception).replace("\x00", " ")[:2000], "traceback": traceback.format_exc(limit=80), "completed_phase_count": len(phase_outcomes), "active_phase": active_phase, "rerun_allowed": False})  # Publish the exact stop boundary without converting it into a method failure.
        raise  # Preserve the original fatal exception for the CLI and audit log.
