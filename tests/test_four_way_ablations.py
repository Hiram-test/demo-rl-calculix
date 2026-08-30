"""Focused fake-trace tests for frozen WMVLA-4WAY-P1 ablations and diagnostics."""  # Describe this test module's solve-free scope.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.

from dataclasses import asdict, replace  # Compare isolated configuration values and create diagnostic variants.
import json  # Verify persisted trace artifacts remain valid machine-readable evidence.
from pathlib import Path  # Address temporary fake result and trace artifacts.
from typing import Any  # Type the deliberately injected fake campaign executor compactly.

import numpy as np  # Construct compact deterministic regional states.
import pytest  # Assert explicit validation failures for leakage-prone inputs.

from visionamr.vla.four_way_ablations import ABLATION_BUDGET, ABLATION_MAX_SOLVES, RANDOM_SAFE_SEEDS  # Import fixed diagnostic coordinates.
from visionamr.vla.four_way_ablations import AblationCampaignRequest, AblationOutcome, AblationReadiness, CandidatePrediction, DiagnosticSession  # Import immutable fake-trace and formal campaign evidence contracts.
from visionamr.vla.four_way_ablations import DorflerFutureStep, DorflerFutureTrajectory, NoHistoryModel, PriorOnlyModel  # Import isolated model and oracle source adapters.
from visionamr.vla.four_way_ablations import OracleSourceExecution, TransitionDiagnostic  # Import injected source and completed-transition contracts.
from visionamr.vla.four_way_ablations import aggregate_diagnostic_trace_files, aggregate_world_model_diagnostics, build_ablation_campaign_summary, build_ablation_case_summary, load_diagnostic_trace  # Import mandatory persisted and in-memory aggregate builders.
from visionamr.vla.four_way_ablations import build_ablation_runtime, build_dorfler_future_trajectory, derive_oracle_schedule, reuse_primary_wm_full, run_ablation_campaign, strip_history  # Import isolated runtime, oracle-source, identity, and formal execution helpers.
import visionamr.vla.four_way_ablations as ablations_module  # Patch only post-primary readiness and use internal exact-byte helpers in fake integration tests.
from visionamr.vla.world.model import RegionAction, ResidualWorldModel, WorldModelConfig, WorldState  # Import frozen V0 state and model contracts.
from visionamr.vla.world.planner import MultiStepPlanner, PlanDecision, PlannerConfig  # Import frozen V0 planner contracts.
from visionamr.vla.world.tool_gateway import MCPToolGateway, MaterializedAction, MeshCertificate  # Import exact tool receipt contracts.


def _state(step: int = 0, *, error_scale: float = 1.0, equation_scale: float = 1.0) -> WorldState:  # Build one compact three-region measured state.
    return WorldState(names=("wheel_patch", "opening_rim", "field"), err_sum=np.asarray([6.0, 3.0, 1.0], dtype=float) * error_scale, elems=np.asarray([40.0, 35.0, 25.0], dtype=float) * equation_scale, sizes=np.asarray([1.0, 1.2, 1.8], dtype=float), vm_max=np.asarray([12.0, 9.0, 2.0], dtype=float), volume=np.asarray([1.0, 1.2, 4.0], dtype=float), adjacency=np.asarray([[0.0, 0.7, 0.3], [0.7, 0.0, 0.3], [0.5, 0.5, 0.0]], dtype=float), dorfler_error_fraction=np.asarray([0.75, 0.40, 0.0], dtype=float), dorfler_element_fraction=np.asarray([0.30, 0.20, 0.0], dtype=float), hit_count=np.asarray([2.0, 1.0, 0.0], dtype=float), n_equations=int(round(300 * equation_scale)), eq_per_elem=3.0, h_min=0.1, h0=2.0, dim=3, step=step)  # Return a validated shared-order state with two eligible mechanisms.


def _prediction(previous: WorldState) -> object:  # Build a deterministic prior prediction without fitting residuals.
    return ResidualWorldModel(WorldModelConfig(prior_uncertainty=0.05)).predict(previous, RegionAction((1, 0, 0)))  # Reuse the real frozen prediction contract.


def _candidate(action: tuple[int, ...], predicted_cost: float, rank: int, actual_cost: float | None = None) -> CandidatePrediction:  # Build compact ranking evidence.
    return CandidatePrediction(action=action, predicted_robust_cost=predicted_cost, predicted_error_ratio=0.7 + 0.02 * rank, predicted_equation_ratio=1.2 + 0.02 * rank, uncertainty=0.1, failure_probability=0.05, predicted_budget_feasible=True, predicted_rank=rank, actual_robust_cost=actual_cost, actual_rank=None)  # Return one finite predicted candidate and optional real counterfactual score.


def _completed_trace(tmp_path: Path) -> object:  # Construct one fully joined prediction-versus-actual fake trace.
    previous = _state(0)  # Create the pre-action measured state.
    observed = _state(1, error_scale=0.62, equation_scale=1.35)  # Create the independently realized successor.
    action = RegionAction((1, 0, 0), source="world_model")  # Request one proactive persistent-region hit.
    prediction = _prediction(previous)  # Produce an authentic V0 prediction object.
    decision = PlanDecision(action=action, accepted=True, reason="world_action_accepted", baseline_cost=0.2, selected_cost=0.1, robust_gain=0.1, predicted_equations_upper=500, predicted_error_ratio_upper=prediction.error_ratio_upper, path=(action.extra_depth,))  # Build the frozen planner audit contract.
    certificate = MeshCertificate(schema_version="wmvla.mcp-tool.v2", requested_action=action.extra_depth, executed_action=action.extra_depth, source="world_model", target_sha256="a" * 64, base_target_included=True, no_coarsening=True, estimated_equations=405, equation_cap=ABLATION_BUDGET, accepted=True, reason="world_candidate_certified", base_raw_target_sha256="b" * 64, world_raw_target_sha256="a" * 64, base_compiled_field_sha256="c" * 64, world_compiled_field_sha256="d" * 64, compiled_field_node_count=100, compiled_field_gradation=1.0, compiled_max_dorfler_violation=0.0, compiled_dorfler_included=True)  # Build a complete exact proactive safety receipt.
    materialized = MaterializedAction(mesh=object(), action=action, certificate=certificate, base_estimated_equations=360, timing_s={"parameter_tools": 0.01, "gmsh_remeshing": 0.02, "tool_total": 0.03})  # Build the compiler result without native meshing.
    trace_path = tmp_path / "prediction_trace.json"  # Select a temporary durable diagnostic path.
    session = DiagnosticSession("test_00", "wm_full", artifact_path=trace_path)  # Initialize one identity-control recorder.
    session.register_decision(previous, decision, (_candidate((0, 0, 0), 0.4, 2), _candidate((1, 0, 0), 0.2, 1)))  # Preserve two solve-free predicted candidates.
    session.register_materialization(previous, materialized)  # Register exact execution and certification.
    session.capture_prediction(previous, action, prediction)  # Register the post-certification prediction.
    record = session.finalize(previous, action, observed)  # Join the independently injected actual successor.
    assert json.loads(trace_path.read_text(encoding="utf-8"))["schema"] == "wmvla-four-way-prediction-trace-v1"  # Verify atomic persistence produced valid versioned JSON.
    return record  # Return the completed immutable diagnostic.


def test_variant_factory_changes_only_declared_ablation_channels(tmp_path: Path) -> None:  # Verify isolation wrappers leave V0 constants unchanged.
    model = ResidualWorldModel()  # Construct one fresh full V0 model.
    model.transition_count = 2  # Supply fair planner warmup evidence without fitting fake residuals.
    planner = MultiStepPlanner(PlannerConfig(horizon=4, beam_width=7))  # Construct a nondefault but frozen full planner.
    gateway = MCPToolGateway()  # Construct the exact deterministic compiler.
    full = build_ablation_runtime("test_00", "wm_full", model, planner, gateway, trace_path=tmp_path / "full.json")  # Build behavior-neutral full diagnostics.
    h1 = build_ablation_runtime("test_00", "wm_h1", model, planner, gateway)  # Build the horizon-one control.
    prior = build_ablation_runtime("test_00", "wm_prior_only", model, planner, gateway)  # Build the analytic-prior control.
    no_history = build_ablation_runtime("test_00", "wm_no_history", model, planner, gateway)  # Build the recurrence-free control.
    assert full.planner.base is planner and full.model.base is model and full.gateway.base is gateway  # Require WM-full to delegate to the exact supplied V0 identities.
    assert asdict(h1.planner.config) == {**asdict(planner.config), "horizon": 1}  # Require horizon to be the only changed planner field.
    assert isinstance(prior.model.base, PriorOnlyModel)  # Require the residual ensemble to be bypassed through an isolated adapter.
    assert isinstance(no_history.model.base, NoHistoryModel)  # Require history removal through an isolated adapter.
    assert all(not runtime.isolation_receipt["v0_constants_changed"] for runtime in (full, h1, prior, no_history))  # Require every receipt to deny frozen-constant mutation.


def test_prior_only_and_no_history_do_not_append_online_residual_rows() -> None:  # Verify both controls suppress only their declared learning channels.
    base = ResidualWorldModel()  # Construct an empty authenticated-model stand-in.
    previous = _state(0)  # Build the measured predecessor.
    observed = _state(1, error_scale=0.7, equation_scale=1.2)  # Build the realized successor.
    action = RegionAction((1, 0, 0))  # Select one legal proactive action.
    prior = PriorOnlyModel(base)  # Disable learned residual prediction and online fitting.
    np.testing.assert_allclose(prior.predict(previous, action).next_state.err_sum, base._prior(previous, action).next_state.err_sum)  # Require exact analytic-prior identity.
    prior.observe(previous, action, observed)  # Count one real transition without fitting.
    assert base._x == [] and base._y == [] and prior.transition_count == 1  # Require no residual-row mutation with fair warmup count.
    no_history = NoHistoryModel(base)  # Retain offline residual behavior while disabling online feedback.
    no_history.observe(previous, action, observed)  # Process one completed transition.
    assert base._x == [] and base._y == [] and no_history.transition_count == 1  # Require no last-transition residual feedback.
    stripped = strip_history(previous)  # Remove the explicit recurrence channels.
    assert stripped.step == 0 and np.all(stripped.hit_count == 0.0)  # Require elapsed-step and hit-count history removal.
    np.testing.assert_allclose(stripped.err_sum, previous.err_sum)  # Preserve the current measured physics state.


def test_random_safe_seed_is_reproducible_and_action_remains_bounded() -> None:  # Verify exact seed identity and legal random actions without real solves.
    config = PlannerConfig(horizon=1, beam_width=8, warmup_transitions=1, budget_safety=1.0)  # Keep the fake planning call small while preserving the V0 action domain.
    left_model = ResidualWorldModel(WorldModelConfig(prior_uncertainty=0.01))  # Construct the first deterministic prediction model.
    right_model = ResidualWorldModel(WorldModelConfig(prior_uncertainty=0.01))  # Construct an independent identical prediction model.
    left_model.transition_count = 2  # Pass the common warmup gate.
    right_model.transition_count = 2  # Pass the common warmup gate independently.
    from visionamr.vla.four_way_ablations import RandomSafePlanner  # Import the control locally to keep the module import list focused.
    left = RandomSafePlanner(config, RANDOM_SAFE_SEEDS[0]).plan(_state(), left_model, 10**9)  # Draw the first seeded safe candidate.
    right = RandomSafePlanner(config, RANDOM_SAFE_SEEDS[0]).plan(_state(), right_model, 10**9)  # Repeat from the same seed and fresh state.
    assert left.action.extra_depth == right.action.extra_depth and left.reason == "random_safe_selected"  # Require exact seed reproducibility.
    assert 1 <= sum(value > 0 for value in left.action.extra_depth) <= config.max_extra_regions  # Require the same sparse action bound.
    assert max(left.action.extra_depth) <= config.max_extra_depth and left.action.extra_depth[-1] == 0  # Require bounded depth and no generic-field action.


def test_oracle_schedule_uses_only_completed_future_dorfler_hits() -> None:  # Verify the nondeployable upper bound is fixed from independent evidence.
    names = ("wheel_patch", "opening_rim", "field")  # Define the stable shared semantic ordering.
    error_rows = ((0.6, 0.3, 0.0), (0.8, 0.0, 0.0), (0.7, 0.9, 0.0), (0.0, 0.8, 0.0), (0.4, 0.0, 0.0), (0.0, 0.0, 0.0))  # Define realized future marked-error masses.
    element_rows = tuple(tuple(1.0 if value > 0.0 else 0.0 for value in row) for row in error_rows)  # Convert positive error hits to realized element hits.
    steps = tuple(DorflerFutureStep(step=index, region_names=names, dorfler_error_fraction=error_rows[index], dorfler_element_fraction=element_rows[index], eligible_regions=(0, 1)) for index in range(ABLATION_MAX_SOLVES))  # Build a complete independent six-solve trace.
    trajectory = DorflerFutureTrajectory(case_id="test_00", equation_budget=ABLATION_BUDGET, max_solves=ABLATION_MAX_SOLVES, source_method="dorfler", completed_independently=True, stop_reason="max_solves", common_probe_sha256="b" * 64, steps=steps)  # Bind the future hits to a completed independent run.
    schedule = derive_oracle_schedule(trajectory, PlannerConfig())  # Derive the fixed allowed-action schedule.
    assert schedule[0].action == (1, 1, 0) and schedule[0].future_hit_score > 0.0  # Require the best paired future-hit action under the fixed objective.
    assert all(choice.action[-1] == 0 for choice in schedule.values())  # Forbid oracle refinement of the generic field.
    with pytest.raises(ValueError, match="independently completed"):  # Reject an online or incomplete future source.
        derive_oracle_schedule(replace(trajectory, completed_independently=False), PlannerConfig())  # Remove the anti-leakage receipt.

def test_oracle_source_builder_accepts_only_contiguous_independent_states() -> None:  # Verify live harnesses can convert shared-partition Dörfler observations without hand-built eligibility.
    planner = MultiStepPlanner(PlannerConfig())  # Reuse the frozen V0 eligibility rule.
    states = tuple(_state(step, error_scale=0.9**step, equation_scale=1.1**step) for step in range(3))  # Build a contiguous early resource-stopped source trajectory.
    trajectory = build_dorfler_future_trajectory("test_00", states, planner, "f" * 64, "equation_cap_reached")  # Convert independently measured states to future-hit evidence.
    assert len(trajectory.steps) == 3 and trajectory.steps[0].eligible_regions[:2] == (0, 1)  # Preserve all real states and frozen eligibility order.
    schedule = derive_oracle_schedule(trajectory, planner.config)  # Accept honest early completion and derive only available future actions.
    assert set(schedule) == {0, 1}  # Produce actions only for transitions with a later independently completed solve.
    with pytest.raises(ValueError, match="explicit physical"):  # Reject a favorable manual truncation without a valid stop.
        build_dorfler_future_trajectory("test_00", states, planner, "f" * 64, "manual_stop")  # Supply an unregistered early stop.


def test_trace_aggregation_reports_calibration_and_honest_spearman(tmp_path: Path) -> None:  # Verify mandatory metrics without fabricated counterfactual ranks.
    record = _completed_trace(tmp_path)  # Build one fully joined fake execution trace.
    report = aggregate_world_model_diagnostics((record,))  # Aggregate all mandatory calibration and action fields.
    selected = next(candidate for candidate in record.candidate_ranking if candidate.action == (1, 0, 0))  # Read the requested proactive candidate from the full ranking.
    assert selected.selected_by_planner and selected.executed_after_certification  # Distinguish planner selection and certified execution explicitly.
    assert selected.predicted_score_scope == "first_step_frozen_stage_cost"  # Disclose the exact solve-free ranking scope.
    assert report["total_error_log_mae"] >= 0.0 and report["equation_mape"] >= 0.0  # Require finite global prediction errors.
    assert 0.0 <= report["prediction_interval_coverage"]["joint_upper"] <= 1.0  # Require a valid transparent upper-bound coverage rate.
    assert report["prediction_interval_coverage"]["two_sided_status"] == "not_measurable"  # Refuse to invent absent lower prediction bounds.
    assert report["proactive_acceptance_rate"] == 1.0 and report["accepted_real_improvement_rate"] == 1.0  # Require correct accepted-action accounting.
    assert report["candidate_ranking_spearman"]["status"] == "not_measurable"  # Refuse to infer actual rankings from one executed candidate.
    realized = replace(record, candidate_ranking=(_candidate((0, 0, 0), 0.1, 1, 0.2), _candidate((1, 0, 0), 0.2, 2, 0.4), _candidate((0, 1, 0), 0.3, 3, 0.6)))  # Inject a complete independent counterfactual ranking.
    measured = aggregate_world_model_diagnostics((realized,))  # Reaggregate with true scores for every candidate.
    assert measured["candidate_ranking_spearman"]["status"] == "measured"  # Recognize fully realized candidate evidence.
    assert measured["candidate_ranking_spearman"]["mean"] == pytest.approx(1.0)  # Recover perfect matching rank order.
    restored = load_diagnostic_trace(tmp_path / "prediction_trace.json")  # Restore the exact persisted transition contract.
    assert restored == (record,)  # Require lossless nested candidate and regional-vector roundtripping.
    persisted = aggregate_diagnostic_trace_files((tmp_path / "prediction_trace.json",))  # Aggregate directly from durable trace evidence.
    assert persisted["transition_count"] == 1 and len(persisted["trace_sources"][0]["sha256"]) == 64  # Bind the aggregate to one exact full SHA-256 source.


def _outcome(case_id: str, variant: str, value: float | None, *, seed: int | None = None, reused: bool = False) -> AblationOutcome:  # Build one compact case-summary fixture.
    suffix = variant if seed is None else f"{variant}_{seed}"  # Give each fake result a stable unique identity.
    return AblationOutcome(case_id=case_id, variant=variant, energy_error=value, ok=value is not None, executed_proactive_actions=2, certified_proactive_actions=2, common_probe_sha256="c" * 64, matched_budget=True, competitor_isolation=True, trace_path=f"trace/{suffix}.json", result_path=f"result/{suffix}.json", result_sha256=("d" if seed is None else "e") * 64, seed=seed, reused_from_primary=reused)  # Return complete analyzer-facing evidence without filesystem access.


def test_case_and_campaign_summaries_require_complete_fixed_design(tmp_path: Path) -> None:  # Verify full identity reuse, five-seed median, and sixteen-case completeness.
    primary_result = tmp_path / "primary.json"  # Select a fake exact primary result.
    primary_trace = tmp_path / "trace.json"  # Select a fake exact primary trace.
    primary_result.write_text('{"energy":0.5}', encoding="utf-8")  # Persist deterministic bytes for identity hashing.
    primary_trace.write_text('{"trace":[]}', encoding="utf-8")  # Persist deterministic bytes for trace existence.
    full = reuse_primary_wm_full("test_00", 0.5, True, 2, 2, "c" * 64, primary_trace, primary_result)  # Bind WM-full to the existing primary result rather than rerunning it.
    deterministic = [full, _outcome("test_00", "wm_h1", 0.6), _outcome("test_00", "wm_prior_only", 0.7), _outcome("test_00", "wm_no_history", 0.65), _outcome("test_00", "oracle_future_hit", 0.4)]  # Build all one-run diagnostics.
    random_rows = [_outcome("test_00", "random_safe_extra", value, seed=seed) for seed, value in zip(RANDOM_SAFE_SEEDS, (0.8, None, 0.9, 0.7, 1.1), strict=True)]  # Build all five seeded controls with one retained failure.
    summary = build_ablation_case_summary((*deterministic, *random_rows))  # Build the fixed analyzer-facing case record.
    assert summary["variants"]["wm_full"]["reused_from_primary"] is True  # Require explicit primary identity reuse.
    assert summary["variants"]["random_safe_extra"]["median_energy_error"] == pytest.approx(0.9)  # Rank the one failure after four finite errors and select the third.
    assert summary["mechanism_evidence"] == {"common_uniform_probe": True, "common_probe_sha256": "c" * 64, "matched_solve_budget": True, "competitor_isolation": True, "wm_full_executed_proactive_actions": 2, "wm_full_certified_proactive_actions": 2}  # Preserve all mechanism-gate evidence.
    campaign_rows = [dict(summary, case_id=f"test_{index:02d}") for index in range(16)]  # Build sixteen unique schema-compatible case summaries.
    campaign = build_ablation_campaign_summary(campaign_rows)  # Validate and sort the complete blind set.
    assert campaign["case_count"] == 16 and campaign["all_common_uniform_probe"] is True  # Require complete matched mechanism evidence.
    with pytest.raises(ValueError, match="exactly 16"):  # Reject an incomplete favorable subset.
        build_ablation_campaign_summary(campaign_rows[:-1])  # Remove one blind case.


def _campaign_readiness() -> AblationReadiness:  # Build complete solve-free fake readiness for sixteen ascending cases.
    cases = tuple({"case_id": f"test_{index:02d}", "split": "test", "geometry_hash": f"{index:064x}", "config_hash": f"{index + 100:064x}", "parameters": {"fake": float(index)}} for index in range(16))  # Construct distinct manifest-shaped case rows without real geometries.
    partitions = {str(case["case_id"]): {"probe_sha256": "c" * 64, "region_order": ["wheel_patch", "opening_rim", "field"]} for case in cases}  # Bind every fake case to one common-probe identity and stable region order.
    config = {"world_planner": asdict(PlannerConfig())}  # Supply the complete planner fields needed only for post-source oracle schedule derivation.
    evidence = {"schema": "wmvla-four-way-ablation-readiness-v1", "protocol_id": "WMVLA-4WAY-P1", "validated_freeze": True, "primary_test_complete": True, "partitions": partitions}  # Declare already injected fake post-primary readiness.
    return AblationReadiness(cases=cases, config=config, evidence=evidence)  # Return immutable test-only readiness without filesystem or native access.


class _FakeCampaignExecutor:  # Inject deterministic traces and source states while exercising the complete formal orchestration layer.
    def __init__(self, template: TransitionDiagnostic) -> None:  # Store one validated transition shape for cheap per-job cloning.
        self.template = template  # Retain the finite authentic diagnostic contract.
        self.events: list[tuple[str, str, int | None]] = []  # Record exact phase order for all sixteen cases.
    def _outcome(self, job: Any, *, reused: bool) -> AblationOutcome:  # Persist one complete fake variant trace and result.
        job.output_dir.mkdir(parents=True, exist_ok=False)  # Create the same exact raw directory expected from native execution.
        record = replace(self.template, case_id=job.case_id, variant=job.variant, seed=job.seed)  # Clone finite prediction-versus-actual evidence under this exact coordinate.
        trace_path = job.output_dir / "prediction_trace.json"  # Select the mandatory raw diagnostic path.
        trace_payload = {"schema": "wmvla-four-way-prediction-trace-v1", "protocol_id": "WMVLA-4WAY-P1", "case_id": job.case_id, "variant": job.variant, "seed": job.seed, "transitions": [ablations_module._transition_payload(record)], "incomplete_transitions": []}  # Build one complete versioned injected trace.
        trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2), encoding="utf-8")  # Persist deterministic fake trace bytes only below pytest tmp_path.
        result_path = job.output_dir / "result.json"  # Select one fake raw result identity.
        result_path.write_text(json.dumps({"case_id": job.case_id, "variant": job.variant, "seed": job.seed, "energy_error": 0.5}, sort_keys=True), encoding="utf-8")  # Persist finite deterministic fake result bytes.
        outcome = AblationOutcome(case_id=job.case_id, variant=job.variant, energy_error=0.5, ok=True, executed_proactive_actions=1, certified_proactive_actions=1, common_probe_sha256="c" * 64, matched_budget=True, competitor_isolation=True, trace_path=str(trace_path), result_path=str(result_path), result_sha256=ablations_module._sha256_file(result_path), seed=job.seed, reused_from_primary=reused)  # Return complete analyzer-facing evidence.
        (job.output_dir / "status.json").write_text(json.dumps({"completed": True, "outcome": asdict(outcome)}, sort_keys=True), encoding="utf-8")  # Persist a fake terminal raw status for artifact-layout inspection.
        return outcome  # Return the deterministic successful fake point.
    def reuse_primary(self, case: dict[str, Any], job: Any, readiness: AblationReadiness) -> AblationOutcome:  # Exercise the formal no-native WM-full phase.
        assert str(case["case_id"]) == job.case_id and readiness.evidence["primary_test_complete"] is True  # Require matching authenticated coordinates.
        self.events.append((job.case_id, "wm_full_reuse", None))  # Record the identity phase once.
        return self._outcome(job, reused=True)  # Return explicit reused primary identity evidence.
    def run_variant(self, case: dict[str, Any], job: Any, readiness: AblationReadiness, *, oracle_schedule: dict[int, Any] | None = None) -> AblationOutcome:  # Exercise every fresh native-variant orchestration coordinate.
        assert str(case["case_id"]) == job.case_id and readiness.evidence["validated_freeze"] is True  # Require matching authenticated coordinates.
        if job.variant == "oracle_future_hit":  # Check the strict two-phase oracle boundary.
            assert oracle_schedule is not None and len(oracle_schedule) == ABLATION_MAX_SOLVES - 1  # Require a schedule derived only after a complete six-state source.
        else:  # Keep non-oracle variants independent from future information.
            assert oracle_schedule is None  # Reject schedule leakage into deployable or random controls.
        self.events.append((job.case_id, job.variant, job.seed))  # Record exact deterministic or seeded phase order.
        return self._outcome(job, reused=False)  # Persist and return one complete fake run.
    def run_oracle_source(self, case: dict[str, Any], output_dir: Path, readiness: AblationReadiness) -> OracleSourceExecution:  # Inject one independently completed Dörfler future trajectory.
        case_id = str(case["case_id"])  # Read the exact active case coordinate.
        output_dir.mkdir(parents=True, exist_ok=False)  # Create the mandated independent source directory.
        names = ("wheel_patch", "opening_rim", "field")  # Preserve the same stable shared region ordering as injected traces.
        steps = tuple(DorflerFutureStep(step=index, region_names=names, dorfler_error_fraction=(0.8 - 0.1 * index, 0.2 + 0.1 * index, 0.0), dorfler_element_fraction=(0.4, 0.3, 0.0), eligible_regions=(0, 1)) for index in range(ABLATION_MAX_SOLVES))  # Build a complete contiguous six-solve future source.
        trajectory = DorflerFutureTrajectory(case_id=case_id, equation_budget=ABLATION_BUDGET, max_solves=ABLATION_MAX_SOLVES, source_method="dorfler", completed_independently=True, stop_reason="max_solves", common_probe_sha256="c" * 64, steps=steps)  # Bind future hits to a normally completed independent run.
        future_path = output_dir / "future_trajectory.json"  # Select the pre-schedule source artifact.
        future_path.write_text(json.dumps({"trajectory": asdict(trajectory)}, sort_keys=True), encoding="utf-8")  # Persist source evidence before the terminal status.
        result_path = output_dir / "records.json"  # Select one fake independent source result.
        result_path.write_text(json.dumps({"case_id": case_id, "source": "dorfler", "solves": ABLATION_MAX_SOLVES}, sort_keys=True), encoding="utf-8")  # Persist finite deterministic source bytes.
        status_path = output_dir / "status.json"  # Select the temporal source-completion marker.
        status_path.write_text(json.dumps({"completed": True, "source_usable": True, "case_id": case_id}, sort_keys=True), encoding="utf-8")  # Publish source completion before orchestration derives its schedule.
        self.events.append((case_id, "oracle_source_dorfler", None))  # Record the independent source phase before oracle execution.
        return OracleSourceExecution(case_id=case_id, trajectory=trajectory, failure=None, common_probe_sha256="c" * 64, result_path=str(result_path), result_sha256=ablations_module._sha256_file(result_path), status_path=str(status_path))  # Return complete source identity and states.


def test_formal_campaign_runner_executes_all_fixed_phases_with_injected_traces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # Verify the full 16-case x 10-outcome driver and sixteen prior oracle sources without blind data.
    template_root = tmp_path / "template"  # Isolate the authentic fake transition fixture from campaign artifacts.
    template_root.mkdir()  # Create the exact fixture directory.
    template = _completed_trace(template_root)  # Build one fully joined authentic V0-shaped transition.
    readiness = _campaign_readiness()  # Build complete already-validated fake post-primary evidence.
    monkeypatch.setattr(ablations_module, "validate_ablation_readiness", lambda _request: readiness)  # Replace only expensive real freeze/reference/model checks for this integration test.
    root = tmp_path / "campaign"  # Select a fresh temporary formal campaign root.
    request = AblationCampaignRequest(root=root, manifest_path=root / "protocol" / "case_manifest.json", frozen_config_path=root / "protocol" / "frozen_config.json")  # Assemble the fixed-resource request without real input files.
    executor = _FakeCampaignExecutor(template)  # Inject deterministic fake raw traces and source states.
    result = run_ablation_campaign(request, executor=executor)  # Exercise all formal orchestration, aggregation, and atomic summary paths.
    assert result["phase_count"] == 16 * 11 and result["outcome_job_count"] == 16 * 10  # Require ten scored variants plus one independent source per case.
    assert len(executor.events) == 16 * 11  # Require every preregistered phase exactly once.
    first = executor.events[:11]  # Inspect the complete first-case phase order.
    assert [value[1] for value in first] == ["wm_full_reuse", "wm_h1", "wm_prior_only", "wm_no_history", *("random_safe_extra" for _seed in RANDOM_SAFE_SEEDS), "oracle_source_dorfler", "oracle_future_hit"]  # Require source completion strictly before oracle execution.
    assert [value[2] for value in first[4:9]] == list(RANDOM_SAFE_SEEDS)  # Require the exact five random-safe seeds in frozen order.
    case_path = root / "ablations" / "test_00" / "ablation_case.json"  # Resolve the analyzer's canonical per-case path.
    case_summary = json.loads(case_path.read_text(encoding="utf-8"))  # Decode the persisted complete case summary.
    assert case_summary["schema"] == "wmvla-four-way-ablation-case-v1" and len(case_summary["prediction_diagnostics"]["trace_paths"]) == 10  # Require all ten traces and the fixed schema.
    assert case_summary["variants"]["wm_full"]["reused_from_primary"] is True  # Require no-rerun WM-full identity evidence.
    assert (root / "ablations" / "test_00" / "oracle_source_dorfler" / "oracle_schedule.json").is_file()  # Require post-source schedule persistence.
    assert (root / "ablations" / "CAMPAIGN_SUMMARY.json").is_file() and (root / "ablations" / "PREDICTION_DIAGNOSTICS.json").is_file()  # Require global mechanism and calibration artifacts.
    assert (root / "ablations" / "ABLATION_COMPLETE.json").is_file() and not (root / "ablations" / "ABLATION_INVALID.json").exists()  # Require atomic formal completion without invalidation.


def test_formal_campaign_dry_run_writes_nothing_after_injected_readiness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # Verify post-primary readiness planning remains solve-free and mutation-free.
    readiness = _campaign_readiness()  # Build complete fake readiness without native inputs.
    monkeypatch.setattr(ablations_module, "validate_ablation_readiness", lambda _request: readiness)  # Inject only the read-only validation result.
    root = tmp_path / "dry_campaign"  # Select a fresh temporary campaign boundary.
    request = AblationCampaignRequest(root=root, manifest_path=root / "protocol" / "case_manifest.json", frozen_config_path=root / "protocol" / "frozen_config.json", dry_run=True)  # Request formal solve-free planning.
    plan = run_ablation_campaign(request)  # Build the complete phase order without constructing an executor.
    assert plan["schema"] == "wmvla-four-way-ablation-execution-plan-v1" and plan["outcome_job_count"] == 160  # Require the fixed complete scored grid.
    assert len(plan["ordered_phases"]) == 176 and plan["wm_full_new_real_solves"] == 0  # Require all source phases and exact identity reuse.
    assert not (root / "ablations").exists()  # Prove dry-run created no markers, results, summaries, or directories.


def test_posthoc_reference_loader_forwards_frozen_expedited_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # Prevent a completed native ablation trajectory from failing during authorized two-level posthoc scoring.
    import visionamr.vla.four_way_benchmark as benchmark_module  # Patch the canonical denominator loader reached by the ablation helper.
    calls: list[dict[str, Any]] = []  # Record both amended and strict forwarding without opening a reference cache.
    sentinel = object()  # Supply one opaque reference identity that the helper must return unchanged.
    def fake_reference_payload(root: Path, case: dict[str, Any], **kwargs: Any) -> tuple[object, dict[str, Any]]:  # Capture the exact schedule and amendment arguments solve-free.
        calls.append({"root": root, "case": case, **kwargs})  # Preserve every forwarded value for explicit assertions.
        return sentinel, {"status": "complete_unqualified" if kwargs["allow_unqualified"] else "complete"}  # Return a compact authenticated-loader stand-in.
    monkeypatch.setattr(benchmark_module, "_reference_payload", fake_reference_payload)  # Replace only the solve-free reference read boundary.
    amendment = {"path": "protocol/EXPEDITED_EXECUTION_AMENDMENT.md", "sha256": "a" * 64}  # Build the protected frozen amendment pointer.
    case = {"case_id": "test_00"}  # Build the minimum manifest identity consumed by the fake loader.
    reference, receipt = ablations_module._load_posthoc_reference(tmp_path, {"expedited_reference_levels": 2, "reference_execution_amendment": amendment}, case, allow_unqualified=True)  # Exercise the authorized two-level path used after each native trajectory.
    assert reference is sentinel and receipt["status"] == "complete_unqualified"  # Require transparent loader return values.
    assert calls[-1] == {"root": tmp_path, "case": case, "allow_unqualified": True, "expedited_levels": 2, "amendment_record": amendment}  # Require exact schedule identity and protected amendment forwarding.
    ablations_module._load_posthoc_reference(tmp_path, {}, case, allow_unqualified=False)  # Exercise strict mode without requiring amendment-only fields.
    assert calls[-1]["expedited_levels"] is None and calls[-1]["amendment_record"] is None  # Keep strict reference verification free of expedited intent.


def test_ablation_readiness_passes_frozen_config_to_reference_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # Catch reference-preflight signature drift before any one-shot ablation marker or native solve.
    import visionamr.bridge_case_manifest as manifest_module  # Patch only the manifest loader imported inside readiness validation.
    import visionamr.vla.four_way_benchmark as benchmark_module  # Patch only benchmark-owned solve-free preflight helpers.
    root = tmp_path / "campaign"  # Isolate the synthetic post-primary campaign below pytest storage.
    (root / "protocol").mkdir(parents=True)  # Create canonical protocol parents for request identity.
    (root / "test").mkdir()  # Create the canonical primary marker parent.
    started = {"schema": "wmvla-four-way-test-started-v1", "protocol_id": "WMVLA-4WAY-P1", "one_shot": True, "resume_allowed": False, "allow_unqualified_references": True, "REFERENCE_QUALIFIED": False, "reference_unqualified_case_ids": [f"test_{index:02d}" for index in range(16)]}  # Build the minimum authenticated one-shot primary boundary.
    (root / "test" / "TEST_STARTED.json").write_text(json.dumps(started, sort_keys=True), encoding="utf-8")  # Persist the regular canonical marker required before patched deep checks.
    cases = tuple({"case_id": f"test_{index:02d}", "split": "test", "geometry_hash": f"{index:064x}"} for index in range(16))  # Build the complete ascending blind manifest unit.
    amendment = {"path": "protocol/EXPEDITED_EXECUTION_AMENDMENT.md", "sha256": "b" * 64}  # Build the frozen two-level authorization pointer.
    config = {"allow_unqualified_references": True, "expedited_reference_levels": 2, "reference_execution_amendment": amendment}  # Supply the exact config object reference preflight must receive.
    observed: dict[str, Any] = {}  # Capture positional and keyword reference-preflight inputs.
    reference_receipts = {str(case["case_id"]): {"qualification": False} for case in cases}  # Give current and primary preflight the same immutable synthetic denominator identities.
    def fake_preflight_references(root_arg: Path, config_arg: dict[str, Any], cases_arg: Any, *, allow_unqualified: bool = False) -> dict[str, dict[str, Any]]:  # Match the canonical benchmark helper signature exactly.
        observed.update({"root": root_arg, "config": config_arg, "cases": tuple(cases_arg), "allow_unqualified": allow_unqualified})  # Record the complete schedule-aware call.
        return dict(reference_receipts)  # Return complete fake reference evidence for all sixteen cases.
    monkeypatch.setattr(ablations_module, "_protected_post_primary_freeze", lambda _request, _started: {"verified": True})  # Bypass exact Git and environment checks in this solve-free unit fixture.
    monkeypatch.setattr(ablations_module, "_validate_primary_completion", lambda _request, _cases, _started: {"complete": True, "reference_receipt_by_case": dict(reference_receipts)})  # Bypass raw enumeration while preserving the new denominator-pinning production contract.
    monkeypatch.setattr(manifest_module, "load_case_manifest", lambda _path, verify_checksum=True: {"cases": list(cases)})  # Return only the fake complete manifest unit.
    monkeypatch.setattr(benchmark_module, "select_manifest_cases", lambda _manifest, split: cases if split == "test" else ())  # Preserve the exact test-split selection request.
    monkeypatch.setattr(benchmark_module, "load_frozen_config", lambda _path: config)  # Return the exact config identity used in the forwarding assertion.
    monkeypatch.setattr(benchmark_module, "_preflight_references", fake_preflight_references)  # Enforce the current three-positional-argument helper contract.
    monkeypatch.setattr(benchmark_module, "_preflight_partitions", lambda _root, _config, selected: {str(case["case_id"]): {} for case in selected})  # Return complete solve-free fake partition evidence.
    monkeypatch.setattr(benchmark_module, "_preflight_models", lambda _root, _config: {"models": "loadable"})  # Return compact fake model construction evidence.
    request = AblationCampaignRequest(root=root, manifest_path=root / "protocol" / "case_manifest.json", frozen_config_path=root / "protocol" / "frozen_config.json", dry_run=True, allow_unqualified_references=True)  # Assemble the canonical amended readiness request.
    readiness = ablations_module.validate_ablation_readiness(request)  # Exercise the real call wiring without native work or formal writes.
    assert readiness.cases == cases and observed["root"] == root.resolve()  # Require the complete authenticated case unit and canonical root forwarding.
    assert observed["config"] is config and observed["cases"] == cases and observed["allow_unqualified"] is True  # Require the frozen schedule config and explicit waiver to reach reference preflight intact.


def test_ablation_dry_run_rejects_replaced_late_reference_cache_before_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # Prove a last-case denominator replacement fails during global solve-free readiness.
    import visionamr.bridge_case_manifest as manifest_module  # Patch only the authenticated manifest boundary for this synthetic complete case unit.
    import visionamr.vla.four_way_benchmark as benchmark_module  # Patch only solve-free benchmark preflight dependencies.
    root = tmp_path / "campaign"  # Isolate all synthetic cache and primary evidence below pytest storage.
    (root / "protocol").mkdir(parents=True)  # Create the canonical request parents without creating ablation output.
    (root / "test").mkdir()  # Create the canonical irreversible-marker parent.
    case_ids = tuple(f"test_{index:02d}" for index in range(16))  # Preserve the complete ascending blind case unit.
    cases = tuple({"case_id": case_id, "split": "test", "geometry_hash": f"{index:064x}"} for index, case_id in enumerate(case_ids))  # Build compact authenticated-manifest stand-ins.
    started = {"schema": "wmvla-four-way-test-started-v1", "protocol_id": "WMVLA-4WAY-P1", "one_shot": True, "resume_allowed": False, "allow_unqualified_references": False}  # Build the minimum valid one-shot primary boundary.
    (root / "test" / "TEST_STARTED.json").write_text(json.dumps(started, sort_keys=True), encoding="utf-8")  # Publish only the synthetic primary marker required by readiness.
    primary_receipts = {case_id: {"ledger_sha256": "a" * 64, "reference_b_sha256": "b" * 64, "qualification": True} for case_id in case_ids}  # Pin one complete denominator identity per primary case.
    current_receipts = {case_id: dict(receipt) for case_id, receipt in primary_receipts.items()}  # Start from an otherwise identical current solve-free cache verification.
    current_receipts[case_ids[-1]]["reference_b_sha256"] = "c" * 64  # Replace only the compact B bytes for the final manifest case.
    monkeypatch.setattr(ablations_module, "_protected_post_primary_freeze", lambda _request, _started: {"verified": True})  # Bypass unrelated Git and environment checks without weakening the production comparison.
    monkeypatch.setattr(ablations_module, "_validate_primary_completion", lambda _request, _cases, _started: {"reference_receipt_by_case": primary_receipts})  # Inject a complete already-deep-validated primary denominator ledger.
    monkeypatch.setattr(manifest_module, "load_case_manifest", lambda _path, verify_checksum=True: {"cases": list(cases)})  # Return the complete synthetic manifest unit.
    monkeypatch.setattr(benchmark_module, "select_manifest_cases", lambda _manifest, split: cases if split == "test" else ())  # Preserve canonical test selection.
    monkeypatch.setattr(benchmark_module, "load_frozen_config", lambda _path: {"allow_unqualified_references": False})  # Keep strict reference policy identical across request and marker.
    monkeypatch.setattr(benchmark_module, "_preflight_references", lambda _root, _config, _cases, allow_unqualified=False: current_receipts)  # Expose the simulated late-case cache replacement through the real readiness comparison.
    monkeypatch.setattr(benchmark_module, "_preflight_partitions", lambda *_args, **_kwargs: pytest.fail("partition preflight must not run after reference replacement"))  # Prove readiness stops at the replaced denominator before later preflights.
    monkeypatch.setattr(benchmark_module, "_preflight_models", lambda *_args, **_kwargs: pytest.fail("model preflight must not run after reference replacement"))  # Prove no learned runtime is constructed after the mismatch.
    request = AblationCampaignRequest(root=root, manifest_path=root / "protocol" / "case_manifest.json", frozen_config_path=root / "protocol" / "frozen_config.json", dry_run=True)  # Request the complete solve-free execution plan.
    with pytest.raises(ValueError, match="current Reference B preflight differs"):  # Require an explicit primary-versus-current denominator mismatch.
        run_ablation_campaign(request, executor=object())  # Supply an unusable executor to prove validation fails before dispatch can reach it.
    assert not (root / "ablations").exists()  # Prove the failed dry-run created neither a marker nor a per-case output tree.


def test_pristine_ablation_check_rejects_symlink_root_without_traversal(tmp_path: Path) -> None:  # Prevent a formal output alias from escaping or redirecting the one-shot evidence tree.
    root = tmp_path / "campaign"  # Select a fresh synthetic campaign boundary.
    outside = tmp_path / "outside"  # Select a separate directory whose contents must remain untouched.
    root.mkdir()  # Create only the canonical campaign parent.
    outside.mkdir()  # Create the prospective external symlink target.
    (root / "ablations").symlink_to(outside, target_is_directory=True)  # Substitute the formal evidence root with an external directory alias.
    with pytest.raises(ValueError, match="not pristine"):  # Require the explicit one-shot pristine-output violation.
        ablations_module._assert_pristine_ablation_output(root)  # Exercise the final pre-marker guard without following the alias.
    assert list(outside.iterdir()) == []  # Prove rejection neither traversed nor wrote into the external target.


def test_ablation_dry_run_deep_validates_corrupt_last_primary_job_before_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # Prove global readiness reaches a malformed late-case K6 artifact before any new diagnostic solve.
    import visionamr.vla.four_way_benchmark as benchmark_module  # Patch only grid cardinality so the deep-artifact fixture remains fast while retaining all sixteen cases.
    root = tmp_path / "campaign"  # Isolate the complete synthetic primary tree below pytest storage.
    cases = tuple({"case_id": f"test_{index:02d}", "split": "test", "geometry_hash": f"{index:064x}", "config_hash": f"{index + 100:064x}", "parameters": {"fake": float(index)}} for index in range(16))  # Build the complete ascending blind manifest unit with every records identity field.
    assert len(cases) * len(benchmark_module.BUDGETS) * len(benchmark_module.ALL_METHODS) == 336  # Lock the production helper's full primary-grid cardinality independently from the compact fixture below.
    monkeypatch.setattr(benchmark_module, "BUDGETS", (ABLATION_BUDGET,))  # Keep the exact WM-full reuse budget while avoiding redundant copies in this focused traversal test.
    monkeypatch.setattr(benchmark_module, "ALL_METHODS", ("world_model_vla",))  # Keep the strictest trace-bearing method while preserving first-to-last case traversal.
    case_ids = [str(case["case_id"]) for case in cases]  # Preserve the authenticated ascending case order used by TEST_STARTED and the plan.
    reference_receipts = {case_id: {"ledger_sha256": "a" * 64, "reference_b_sha256": "b" * 64, "qualification": True} for case_id in case_ids}  # Pin one full valid denominator receipt per synthetic primary case.
    outcomes: list[dict[str, Any]] = []  # Collect the invocation-level terminal ledger in exact case order.
    final_prefix_path: Path | None = None  # Retain the final case's K-grid path for one deliberate late corruption.
    for case in cases:  # Materialize every compact-grid primary job before readiness starts.
        case_id = str(case["case_id"])  # Normalize the active authenticated case coordinate.
        job = {"case_id": case_id, "split": "test", "geometry_hash": str(case["geometry_hash"]), "equation_budget": ABLATION_BUDGET, "method": "world_model_vla", "max_solves": ABLATION_MAX_SOLVES}  # Reconstruct the exact primary job identity.
        output = root / "test" / case_id / str(ABLATION_BUDGET) / "world_model_vla"  # Resolve the canonical raw trajectory directory.
        output.mkdir(parents=True)  # Create only this synthetic job's parent tree.
        final_state_path = output / "final_state.npz"  # Select the mandatory no-pickle terminal archive.
        np.savez_compressed(final_state_path, nodes=np.empty((0, 3), dtype=float), cells=np.empty((0, 4), dtype=int), eta2=np.empty((0,), dtype=float), region_labels=np.empty((0,), dtype=int), available=np.asarray([False]), source=np.asarray(["unavailable"]))  # Persist the exact six-field shape-safe empty-state contract.
        final_state_sha = ablations_module._sha256_file(final_state_path)  # Bind the terminal status to the exact archive bytes.
        common = {"protocol_id": "WMVLA-4WAY-P1", "job": job}  # Reuse the exact protocol and job identity across mandatory JSON artifacts.
        records = {"schema": "wmvla-four-way-method-result-v1", **common, "case": {"case_id": case_id, "parameters": case["parameters"], "config_hash": case["config_hash"], "geometry_hash": case["geometry_hash"]}, "reference_b": {**reference_receipts[case_id], "usage": "posthoc_only", "used_online": False}, "completed": True, "failure": None, "records": []}  # Build a complete zero-row natural terminal trajectory with posthoc-only denominator provenance.
        (output / "records.json").write_text(json.dumps(records, sort_keys=True), encoding="utf-8")  # Persist the strict method record before status.
        for filename, schema in (("mesh_receipts.json", "wmvla-four-way-mesh-receipts-v1"), ("action_log.json", "wmvla-four-way-action-log-v1"), ("timing.json", "wmvla-four-way-timing-v1")):  # Enumerate all remaining common JSON contracts.
            (output / filename).write_text(json.dumps({"schema": schema, **common}, sort_keys=True), encoding="utf-8")  # Persist one exact schema-and-job-bound artifact.
        prefix_rows = [{"case_id": case_id, "method": "world_model_vla", "solves": solve_limit, "equation_budget": ABLATION_BUDGET, "energy_error": None, "energy_ok": False, "qoi_error": None, "qoi_ok": False} for solve_limit in benchmark_module.SOLVE_LIMITS]  # Build the complete registered K={2,3,4,6} grid with honest unavailable metrics.
        final_prefix_path = output / "prefix_results.json"  # Retain this path so the loop's last assignment identifies test_15.
        final_prefix_path.write_text(json.dumps({"schema": "wmvla-four-way-prefix-results-v1", **common, "derivation": "best_feasible_actual_prefix", "rows": prefix_rows}, sort_keys=True), encoding="utf-8")  # Persist the complete true-prefix artifact.
        trace = {"schema": "wmvla-four-way-prediction-trace-v1", "protocol_id": "WMVLA-4WAY-P1", "case_id": case_id, "variant": "wm_full", "seed": None, "transitions": [], "incomplete_transitions": []}  # Build a valid behavior-neutral empty diagnostic trace.
        (output / "prediction_trace.json").write_text(json.dumps(trace, sort_keys=True), encoding="utf-8")  # Persist the mandatory WM-full trace for deep parsing.
        status = {"schema": "wmvla-four-way-method-result-v1", **common, "completed": True, "successful_solve_count": 0, "failure": None, "final_state": {"path": "final_state.npz", "sha256": final_state_sha}}  # Build the atomic terminal marker bound to the raw archive.
        status_path = output / "status.json"  # Select the sole terminal status identity.
        status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")  # Publish the terminal marker after every raw artifact.
        outcomes.append({"job": job, "status": "completed", "completed": True, "status_path": str(status_path), "successful_solve_count": 0})  # Add the exact summary row validated against durable status.
    assert final_prefix_path is not None and "test_15" in str(final_prefix_path)  # Prove the deliberate corruption targets only the final manifest case.
    corrupt_prefix = json.loads(final_prefix_path.read_text(encoding="utf-8"))  # Decode the previously valid final K grid.
    corrupt_prefix["rows"] = corrupt_prefix["rows"][:-1]  # Remove only K=6 so the defect would otherwise surface at late reuse time.
    final_prefix_path.write_text(json.dumps(corrupt_prefix, sort_keys=True), encoding="utf-8")  # Publish the late-case malformed primary artifact before readiness begins.
    plan = {"diagnostic_plan": benchmark_module.build_diagnostic_plan(case_ids), "reference_preflight": reference_receipts}  # Recreate the exact predeclared diagnostic and denominator preflight ledger.
    summary = {"schema": "wmvla-four-way-execution-summary-v1", "protocol_id": "WMVLA-4WAY-P1", "plan": plan, "job_outcomes": outcomes, "completed_job_count": len(outcomes), "successful_job_count": len(outcomes), "failed_job_count": 0, "terminal_job_count": len(outcomes), "all_jobs_completed": True}  # Build a superficially successful invocation summary whose last raw job is corrupt.
    summary_path = root / "test" / "EXECUTION_SUMMARY.json"  # Select the canonical complete primary summary path.
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")  # Persist the aggregate claim before exercising deep validation.
    started = {"case_order": case_ids, "case_count": 16, "budgets": [ABLATION_BUDGET], "methods": ["world_model_vla"]}  # Bind the compact test grid to the complete blind case order.
    request = AblationCampaignRequest(root=root, manifest_path=root / "protocol" / "case_manifest.json", frozen_config_path=root / "protocol" / "frozen_config.json", dry_run=True)  # Request a solve-free formal plan against the synthetic primary tree.
    executor_calls: list[str] = []  # Record any forbidden dispatch beyond readiness.
    class FailIfUsed:  # Provide an executor sentinel that makes every accidental phase call visible.
        def __getattr__(self, name: str) -> Any:  # Intercept any unexpected reuse or native method lookup.
            executor_calls.append(name)  # Preserve the exact leaked dispatch name for the final assertion.
            return lambda *_args, **_kwargs: pytest.fail(f"executor called before deep readiness completed: {name}")  # Fail immediately if orchestration reaches execution.
    def validate_with_real_primary(candidate: AblationCampaignRequest) -> AblationReadiness:  # Route the dry-run through the production deep primary validator only.
        ablations_module._validate_primary_completion(candidate, cases, started)  # Traverse all sixteen compact jobs and raise on the final malformed K grid.
        return _campaign_readiness()  # Supply a complete stand-in only if the expected corruption were incorrectly accepted.
    monkeypatch.setattr(ablations_module, "validate_ablation_readiness", validate_with_real_primary)  # Preserve real global primary validation while isolating unrelated freeze/model inputs.
    with pytest.raises(ValueError, match="primary prefix row count mismatch.*test_15"):  # Require the precise late-case K-grid failure during dry-run readiness.
        run_ablation_campaign(request, executor=FailIfUsed())  # Exercise the public formal entrypoint without native tools or formal-root writes.
    assert executor_calls == [] and not (root / "ablations").exists()  # Prove validation failed before executor access, markers, or new-solve directories.
