"""Focused tests for the frozen bridge Double-DQN schedule and scoring contract."""  # Describe the pure no-solver verification surface.
from __future__ import annotations  # Postpone annotation evaluation consistently with the implementation.
from collections import Counter  # Count deterministic case-budget exposure across three hundred episodes.
from dataclasses import asdict  # Serialize typed synthetic validation observations exactly like real shard reports.
import json  # Inspect the assembled unified freeze index without executing a blind test.
import math  # Verify fixed failure scores remain finite and use logarithmic aggregation.
from pathlib import Path  # Build isolated fake seed-shard artifact trees under pytest temporary directories.
import runpy  # Inspect the CLI's executable defaults without launching a campaign process.
from types import SimpleNamespace  # Build minimal successful solve-record fixtures without Gmsh or CalculiX.
import pytest  # Assert numerical equality and explicit malformed-grid failures.
from visionamr.baselines.bridge_rl import EQUATION_BUDGETS  # Import the three frozen active budgets.
from visionamr.baselines.bridge_rl import COMMON_NODAL_GRADATION  # Import the shared PR-40 V0 nodal size-field smoothing value.
from visionamr.baselines.bridge_rl import RL_SEEDS  # Import the three frozen independent training seeds.
from visionamr.baselines.bridge_rl import SOLVE_PREFIXES  # Import the four frozen real-solve delivery prefixes.
from visionamr.baselines.bridge_rl import TRAIN_EPISODES  # Import the exact attempted episode count per policy seed.
from visionamr.baselines.bridge_rl import VALIDATION_EPISODES  # Import the twelve eligible checkpoint episodes.
from visionamr.baselines.bridge_rl import VALIDATION_FAILURE_ERROR  # Import the fixed finite failed-point metric penalty.
from visionamr.baselines.bridge_rl import ValidationObservation  # Import typed checkpoint-selection evidence.
from visionamr.baselines.bridge_rl import RL_SCHEMA  # Import the exact shard and freeze artifact schema identity.
from visionamr.baselines.bridge_rl import PROTOCOL_ID  # Import the exact four-way protocol identity for fake audit artifacts.
from visionamr.baselines.bridge_rl import RETAINED_NUMERICAL_FAILURE_TYPES  # Verify the sample-level exception boundary excludes campaign-integrity failures.
from visionamr.baselines.bridge_rl import _base_dqn_config  # Build a real loadable frozen architecture checkpoint for shard verification.
from visionamr.baselines.bridge_rl import _load_reference_b  # Verify the strict or amended choice reaches the canonical reference loader exactly.
from visionamr.baselines.bridge_rl import _median_rows  # Verify pointwise three-seed aggregation from transparent prefix evidence.
from visionamr.baselines.bridge_rl import _sha256_file  # Hash fake report and model bytes through the production verifier.
from visionamr.baselines.bridge_rl import _write_json  # Persist strict fake shard artifacts through the production JSON boundary.
from visionamr.baselines.bridge_rl import _write_sha256_sidecar  # Seal each fake shard index exactly like the parallel trainer.
from visionamr.baselines.bridge_rl import assemble_bridge_rl  # Verify strict three-seed merge and no-best-seed retention.
from visionamr.baselines.bridge_rl import build_training_plan  # Verify the complete no-solver preregistration record.
from visionamr.baselines.bridge_rl import build_training_reference_policy  # Verify strict default and fixed-amendment authorization independently from native work.
from visionamr.baselines.bridge_rl import select_validation_checkpoint  # Verify all twelve checkpoints participate in selection.
from visionamr.baselines.bridge_rl import training_assignment  # Verify deterministic case and literal budget cycling.
from visionamr.baselines.bridge_rl import trajectory_prefix_result  # Verify best-feasible-prefix and failure propagation semantics.
from visionamr.baselines.bridge_rl import validation_selection_key  # Verify the exact finite lexicographic metric.
from visionamr.bridge_case_manifest import build_case_manifest, write_case_manifest  # Generate or persist the deterministic frozen manifest and sidecar.
from visionamr.baselines.rl_dqn import DQNPolicy  # Create real Torch state-dict bytes without any solver or training episode.
from visionamr.calculix import CalculiXExecutionError  # Construct one explicitly typed native solver failure for boundary verification.
from visionamr.mesher import GmshMeshingError  # Construct one explicitly typed native meshing failure for boundary verification.
from visionamr.vla.four_way_references import UNQUALIFIED_AUTHORIZATION  # Reuse the fixed user amendment token in isolated authorization tests.

def _validation_case_ids() -> list[str]:  # Build the exact eight synthetic validation identities.
    return [f"validation_{index:02d}" for index in range(8)]  # Return unique stable identifiers in deterministic order.

def _validation_observations(energy: float = 0.8, qoi: float = 0.9) -> list[ValidationObservation]:  # Build a complete passing eight-case-by-three-budget grid.
    return [ValidationObservation(case_id=case_id, equation_budget=budget, energy_error=energy, qoi_error=qoi, ok=True, budget_violation=False, solve_attempts=2) for case_id in _validation_case_ids() for budget in EQUATION_BUDGETS]  # Give every authorized point one finite successful observation.

def test_training_plan_has_three_models_and_no_blind_identifiers() -> None:  # Verify the dry-run plan exposes only training and validation data boundaries.
    manifest = build_case_manifest()  # Generate the checksummed design content without filesystem or native solves.
    plan = build_training_plan(manifest)  # Materialize the exact three-policy training and selection schedule.
    blind_ids = {str(case["case_id"]) for case in manifest["cases"] if case["split"] == "test"}  # Collect blind identities only inside this isolation test.
    serialized_plan = str(plan)  # Flatten the plan to prove it did not copy any blind identifier.
    assert plan["seeds"] == list(RL_SEEDS) and len(plan["policies"]) == 3  # Require exactly one budget-conditioned policy per independent seed.
    assert all(policy["episodes"] == TRAIN_EPISODES for policy in plan["policies"])  # Require exactly three hundred episode attempts per seed.
    assert all(policy["checkpoint_episodes"] == list(VALIDATION_EPISODES) for policy in plan["policies"])  # Require all twelve preregistered validation events.
    assert plan["nodal_gradation"] == COMMON_NODAL_GRADATION == 1.0  # Freeze the same explicit V0 nodal gradation used by every method.
    assert plan["test_split_access"] is False and all(case_id not in serialized_plan for case_id in blind_ids)  # Prevent blind identifiers or parameters from entering the training plan.

def test_cli_uses_protocol_partition_root_and_strict_reference_defaults() -> None:  # Pin the executable's canonical shared-partition path and fail-closed validation switches.
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "train_bridge_rl.py"))  # Load declarations without satisfying the script's __main__ execution guard.
    parser = namespace["_parser"]()  # Build the real phase parser rather than copying its defaults into the test.
    arguments = parser.parse_args(["train"])  # Parse the ordinary training invocation with no exceptional waiver flags.
    assert arguments.partition_root == namespace["CAMPAIGN_ROOT"] / "protocol" / "partitions"  # Require the same protocol-owned partition registry consumed by the shared adapter.
    assert arguments.allow_unqualified_references is False and arguments.expedited_reference_levels is None  # Keep complete_unqualified validation evidence rejected unless both flags appear.

def test_training_reference_policy_rejects_implicit_waiver_and_authenticates_fixed_amendment(tmp_path: Path) -> None:  # Verify only the exact two-flag user amendment can enable complete_unqualified references.
    protocol_directory = tmp_path / "campaign" / "protocol"  # Reproduce the canonical manifest and amendment parent without touching formal artifacts.
    protocol_directory.mkdir(parents=True)  # Create only the isolated protocol fixture directory.
    manifest_path = protocol_directory / "case_manifest.json"  # Resolve the path used to anchor amendment discovery.
    manifest_path.write_text("{}\n", encoding="utf-8")  # Provide harmless local bytes because this helper authenticates policy rather than manifest schema.
    strict_policy = build_training_reference_policy(manifest_path)  # Exercise the ordinary no-amendment path explicitly.
    assert strict_policy == {"allow_unqualified_references": False, "expedited_reference_levels": None, "authorization": None, "reference_execution_amendment": None}  # Pin the complete fail-closed serialized policy.
    with pytest.raises(ValueError, match="requires --allow-unqualified-references"):  # Reject a shortened ladder without the conspicuous opt-in switch.
        build_training_reference_policy(manifest_path, expedited_reference_levels=2)  # Supply only the depth half of the required pair.
    with pytest.raises(ValueError, match="requires --expedited-reference-levels 2"):  # Reject a vague unqualified-reference waiver with no frozen depth.
        build_training_reference_policy(manifest_path, allow_unqualified_references=True)  # Supply only the opt-in half of the required pair.
    with pytest.raises(FileNotFoundError, match="amendment"):  # Require durable pre-freeze human evidence before exceptional use.
        build_training_reference_policy(manifest_path, allow_unqualified_references=True, expedited_reference_levels=2)  # Supply both switches while the amendment is absent.
    amendment_path = protocol_directory / "EXPEDITED_EXECUTION_AMENDMENT.md"  # Resolve the sole canonical user amendment filename.
    amendment_path.write_text(f"# Local test amendment\n\n- 授权标识：`{UNQUALIFIED_AUTHORIZATION}`\n", encoding="utf-8")  # Write the exact named token into isolated transparent bytes.
    amended_policy = build_training_reference_policy(manifest_path, allow_unqualified_references=True, expedited_reference_levels=2)  # Authenticate the complete exceptional pair and canonical amendment.
    assert amended_policy["allow_unqualified_references"] is True and amended_policy["expedited_reference_levels"] == 2  # Preserve the exact operational choice.
    assert amended_policy["authorization"] == UNQUALIFIED_AUTHORIZATION  # Preserve the fixed user authorization identity rather than an arbitrary CLI token.
    assert amended_policy["reference_execution_amendment"]["sha256"] == _sha256_file(amendment_path)  # Bind the policy to all exact local amendment bytes.
    plan = build_training_plan(build_case_manifest(), amended_policy)  # Materialize a no-solver plan using the authenticated exceptional policy.
    assert plan["validation_reference_policy"] == amended_policy  # Write the choice at the plan summary level.
    assert all(policy["validation_reference_policy"] == amended_policy for policy in plan["policies"])  # Write the same choice into every fixed-seed model plan.

def test_rl_reference_loader_forwards_strict_default_and_only_explicit_amendment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # Verify the production adapter never turns an omitted policy into implicit complete_unqualified acceptance.
    calls: list[tuple[bool, int | None]] = []  # Record only the two qualification arguments relevant to this boundary.
    sentinel = object()  # Provide a unique fake reference result for both loader calls.
    def fake_load_reference_b(_root: Path, *, case_id: str, problem: object, runner: object, allow_unqualified: bool, expedited_levels: int | None) -> object:  # Mirror the canonical keyword-only reference API at the mocked boundary.
        assert case_id == "validation_case" and problem is not None and runner is not None  # Confirm the ordinary case binding remains intact.
        calls.append((allow_unqualified, expedited_levels))  # Capture exact policy propagation without opening a cache.
        return sentinel  # Return the harmless local object expected by the adapter.
    monkeypatch.setattr("visionamr.vla.four_way_references.load_reference_b", fake_load_reference_b)  # Replace only the lazy canonical loader for this isolated test.
    case = {"case_id": "validation_case"}  # Build the minimum already-authorized validation case identity.
    assert _load_reference_b(tmp_path, case, object(), object()) is sentinel  # Exercise omitted-policy behavior and require strict defaults.
    protocol_directory = tmp_path / "protocol"  # Create the canonical local location used by exceptional policy authentication.
    protocol_directory.mkdir()  # Create only the isolated amendment parent.
    manifest_path = protocol_directory / "case_manifest.json"  # Anchor the policy to its local protocol directory.
    manifest_path.write_text("{}\n", encoding="utf-8")  # Supply harmless local manifest bytes outside the loader's concern.
    amendment_path = protocol_directory / "EXPEDITED_EXECUTION_AMENDMENT.md"  # Resolve the fixed amendment filename.
    amendment_path.write_text(f"- 授权标识：`{UNQUALIFIED_AUTHORIZATION}`\n", encoding="utf-8")  # Supply the exact dedicated authorization field.
    amended_policy = build_training_reference_policy(manifest_path, allow_unqualified_references=True, expedited_reference_levels=2)  # Build the only accepted exceptional policy.
    assert _load_reference_b(tmp_path, case, object(), object(), amended_policy) is sentinel  # Forward the explicit authenticated exceptional choice.
    assert calls == [(False, None), (True, 2)]  # Prove strict omission and amended opt-in reach the cache verifier without implicit allowance.

def test_episode_schedule_cycles_budgets_exactly_and_rotates_cases() -> None:  # Verify all budgets receive one hundred episodes without locking a case to one budget.
    training_cases = [case for case in build_case_manifest()["cases"] if case["split"] == "train"]  # Isolate the exact twenty-four manifest training cases.
    assignments = [training_assignment(episode, training_cases) for episode in range(1, TRAIN_EPISODES + 1)]  # Expand the entire deterministic schedule.
    budget_counts = Counter(budget for _case, budget in assignments)  # Count literal cyclic budget exposure.
    first_288 = Counter((str(case["case_id"]), budget) for case, budget in assignments[:288])  # Inspect twelve complete twenty-four-case passes.
    assert budget_counts == Counter({30000: 100, 60000: 100, 120000: 100})  # Require exact one-third exposure at every active budget.
    assert set(first_288.values()) == {4} and len(first_288) == 24 * 3  # Require every case to see every budget four times before the final twelve episodes.
    assert [assignments[index][1] for index in range(6)] == [30000, 60000, 120000, 30000, 60000, 120000]  # Pin the literal public budget cycle.

def test_only_explicit_native_numerical_errors_are_retained_as_failed_samples(tmp_path: Path) -> None:  # Verify schema, hash, configuration, policy, and serialization failures abort the campaign.
    ccx_failure = CalculiXExecutionError("failed", returncode=1, wall_s=0.1, log_path=tmp_path / "ccx.log", workdir=tmp_path)  # Build one typed CalculiX execution failure with complete native provenance.
    gmsh_failure = GmshMeshingError("no simplices")  # Build one typed Gmsh materialization failure.
    integrity_failures = (ValueError("schema"), FileNotFoundError("hash"), TypeError("config"), FloatingPointError("policy"), json.JSONDecodeError("serialization", "{}", 0))  # Build representative campaign-invalidating non-native exceptions.
    assert isinstance(ccx_failure, RETAINED_NUMERICAL_FAILURE_TYPES) and isinstance(gmsh_failure, RETAINED_NUMERICAL_FAILURE_TYPES)  # Retain only explicitly classified native numerical outcomes.
    assert all(not isinstance(exception, RETAINED_NUMERICAL_FAILURE_TYPES) for exception in integrity_failures)  # Force every configuration or evidence-integrity error to propagate and invalidate the campaign.

def test_validation_key_is_finite_failure_aware_and_order_independent() -> None:  # Verify the five-component checkpoint score never uses NaN or infinity.
    observations = _validation_observations()  # Build the complete successful twenty-four-point validation grid.
    passing = validation_selection_key(observations, 25, _validation_case_ids())  # Score the first eligible checkpoint.
    failed = list(observations)  # Copy the immutable observations for one controlled failure.
    failed[0] = ValidationObservation(case_id=failed[0].case_id, equation_budget=failed[0].equation_budget, energy_error=math.nan, qoi_error=None, ok=False, budget_violation=True, solve_attempts=1, failure={"exception_type": "FloatingPointError"})  # Retain one malformed numerical point and one budget violation.
    scored = validation_selection_key(list(reversed(failed)), 25, list(reversed(_validation_case_ids())))  # Re-score in reverse order to test ordering invariance.
    expected_energy_log = (23 * math.log(0.8) + math.log(VALIDATION_FAILURE_ERROR)) / 24  # Compute the declared fixed-penalty energy log mean.
    expected_qoi_log = (23 * math.log(0.9) + math.log(VALIDATION_FAILURE_ERROR)) / 24  # Compute the declared fixed-penalty QoI log mean.
    assert passing[:4] == pytest.approx((0, math.log(0.8), math.log(0.9), 0))  # Preserve ordinary equal-weight log means and zero failures.
    assert scored[0] == 1 and scored[1] == pytest.approx(expected_energy_log) and scored[2] == pytest.approx(expected_qoi_log)  # Apply one finite penalty to both invalid metrics.
    assert scored[3:] == (1, 25) and all(math.isfinite(value) for value in scored[1:3])  # Retain the budget violation and finite selection values.

def test_checkpoint_selection_requires_all_twelve_and_uses_earlier_final_tie_break() -> None:  # Verify selection cannot omit candidates or favor a later equivalent checkpoint.
    reports = []  # Accumulate one transparent raw report at every frozen checkpoint episode.
    for episode in VALIDATION_EPISODES:  # Cover the complete twelve-candidate checkpoint set.
        metric = 0.5 if episode in (50, 75) else 1.0  # Give episodes fifty and seventy-five the same unique best scientific score.
        observations = _validation_observations(metric, metric)  # Build the exact twenty-four-point evidence for this checkpoint.
        reports.append({"checkpoint_episode": episode, "checkpoint_file": f"checkpoint_{episode}.pt", "observations": [observation.__dict__ for observation in observations]})  # Retain raw observations and a stable model path.
    selected = select_validation_checkpoint(list(reversed(reports)), _validation_case_ids())  # Make selection independent of report serialization order.
    assert selected["checkpoint_episode"] == 50 and selected["checkpoint_file"] == "checkpoint_50.pt"  # Use earlier episode only after the first four key components tie.
    with pytest.raises(ValueError, match="all 12"):  # Reject a favorable incomplete checkpoint candidate set.
        select_validation_checkpoint(reports[:-1], _validation_case_ids())  # Remove one scheduled candidate and require hard failure.

def test_prefix_delivery_excludes_over_budget_solves_and_retains_failure_boundary() -> None:  # Verify each K result comes from one actual trajectory with conservative failure semantics.
    records = [SimpleNamespace(solve_index=1, n_equations=20000, e_energy=0.8, e_qoi=0.7), SimpleNamespace(solve_index=2, n_equations=29000, e_energy=0.6, e_qoi=0.65), SimpleNamespace(solve_index=3, n_equations=31000, e_energy=0.4, e_qoi=0.3)]  # Build two feasible solves followed by one over-budget solve.
    successful = {"case_id": "case", "equation_budget": 30000, "status": "ok", "failure_at_solve": None, "records": records}  # Represent one normal trajectory that terminates after a budget overshoot.
    k2 = trajectory_prefix_result(successful, 2)  # Score the first two real solves.
    k4 = trajectory_prefix_result(successful, 4)  # Score the held trajectory including the actual over-budget attempt.
    assert k2["energy_error"] == pytest.approx(0.6) and k2["budget_violation"] is False  # Select the best two-solve feasible energy value.
    assert k4["energy_error"] == pytest.approx(0.6) and k4["qoi_error"] == pytest.approx(0.65)  # Exclude the better-looking over-budget solve from both delivered metrics.
    assert k4["budget_violation"] is True and k4["hold_last_after_stop"] is True  # Disclose the overshoot and held-last later prefixes.
    failed = {**successful, "status": "numerical_failure", "failure_at_solve": 3}  # Make the third real-solve attempt fail after two valid results.
    assert trajectory_prefix_result(failed, 2)["energy_ok"] is True  # Preserve a completed prefix strictly before the failure.
    assert trajectory_prefix_result(failed, 4)["energy_ok"] is False  # Invalidate every public prefix that reaches the failed attempt.

def test_pointwise_median_uses_all_three_seed_outcomes_for_every_operating_point() -> None:  # Verify no global or per-point best-seed selection enters the RL denominator.
    rows = []  # Accumulate one complete synthetic case across all public K and B points.
    for budget in EQUATION_BUDGETS:  # Cover every active equation budget.
        for solves in SOLVE_PREFIXES:  # Cover every public real-solve prefix.
            for seed, value, ok in zip(RL_SEEDS, (0.8, 1.0, None), (True, True, False), strict=True):  # Provide two finite policies and one failed policy in frozen seed order.
                rows.append({"case_id": "case", "solves": solves, "equation_budget": budget, "seed": seed, "energy_error": value, "qoi_error": value, "energy_ok": ok, "qoi_ok": ok, "budget_violation": False})  # Retain the failed seed instead of dropping it.
    medians = _median_rows(list(reversed(rows)), ["case"])  # Aggregate from deliberately reversed execution evidence.
    assert len(medians) == len(EQUATION_BUDGETS) * len(SOLVE_PREFIXES)  # Require the complete one-case operating grid.
    assert all(row["energy_error"] == pytest.approx(1.0) and row["qoi_error"] == pytest.approx(1.0) for row in medians)  # Rank the failed seed last so the worse finite policy is the median.
    assert all(row["seed_order"] == list(RL_SEEDS) and row["aggregation"].startswith("pointwise") for row in medians)  # Preserve source-seed identities and median semantics in every row.

def _fake_complete_seed_shard(root: Path, manifest_path: Path, manifest: dict, seed: int, supplied_reference_policy: dict | None = None) -> Path:  # Build one complete no-solver seed shard under a caller-selected authenticated policy.
    shard = root / f"shard_{seed}"  # Isolate this synthetic independent fixed-seed result.
    seed_root = shard / f"seed_{seed}"  # Reproduce the real trainer's detailed evidence directory.
    checkpoints = seed_root / "checkpoints"  # Reproduce all twelve eligible model-candidate locations.
    checkpoints.mkdir(parents=True)  # Create the isolated fake shard tree.
    model_path = shard / f"rl_seed{seed}.pt"  # Resolve the shard-local selected deployment model.
    DQNPolicy(_base_dqn_config(seed)).save(model_path)  # Write a real loadable state dict without performing a training update.
    model_bytes = model_path.read_bytes()  # Read the small state dict once for exact checkpoint duplication.
    reference_policy = build_training_reference_policy(manifest_path) if supplied_reference_policy is None else dict(supplied_reference_policy)  # Use strict production defaults unless a test explicitly supplies authenticated amendment evidence.
    training_cases = [case for case in manifest["cases"] if case["split"] == "train"]  # Isolate the authorized synthetic training split.
    episodes = []  # Accumulate exactly three hundred transparent scheduled episode outcomes.
    for episode in range(1, TRAIN_EPISODES + 1):  # Reproduce every case and literal budget assignment without native solves.
        case, budget = training_assignment(episode, training_cases)  # Reuse the production preregistered schedule.
        episodes.append({"episode": episode, "case_id": str(case["case_id"]), "equation_budget": budget, "status": "ok", "solve_attempts": 1, "gradient_updates": 1})  # Retain one finite synthetic solve and update cost per episode.
    budget_counts = {str(budget): 100 for budget in EQUATION_BUDGETS}  # Record the exact verified one-third exposure summary.
    training_history = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "training_history", "seed": seed, "validation_reference_policy": reference_policy, "episodes_requested": TRAIN_EPISODES, "episodes_attempted": TRAIN_EPISODES, "numeric_failure_episodes": 0, "budget_counts": budget_counts, "episodes": episodes}  # Assemble the complete fake raw training report with explicit strict denominator intent.
    training_history_path = seed_root / "training_history.json"  # Resolve the production history filename.
    _write_json(training_history_path, training_history)  # Persist strict finite raw training evidence.
    validation_cases = [case for case in manifest["cases"] if case["split"] == "validation"]  # Isolate the authorized synthetic checkpoint-selection split.
    validation_ids = [str(case["case_id"]) for case in validation_cases]  # Preserve exact validation identities for production key recomputation.
    observations = [ValidationObservation(case_id=case_id, equation_budget=budget, energy_error=0.8, qoi_error=0.9, ok=True, budget_violation=False, solve_attempts=1) for case_id in validation_ids for budget in EQUATION_BUDGETS]  # Build one complete equal-score twenty-four-point grid.
    reports = []  # Accumulate all twelve eligible validation reports and loadable checkpoints.
    for checkpoint_episode in VALIDATION_EPISODES:  # Cover every preregistered checkpoint candidate.
        checkpoint_path = checkpoints / f"episode_{checkpoint_episode:04d}.pt"  # Resolve the real trainer's checkpoint filename.
        checkpoint_path.write_bytes(model_bytes)  # Make each synthetic candidate a byte-identical loadable state dict.
        reports.append({"checkpoint_episode": checkpoint_episode, "checkpoint_file": str(checkpoint_path.relative_to(shard)), "checkpoint_sha256": _sha256_file(checkpoint_path), "validation_reference_policy": reference_policy, "observations": [asdict(item) for item in observations], "real_solve_attempts": len(observations), "wall_s": 0.25})  # Retain raw observations, cost, exact candidate identity, and strict qualification policy.
    selected = dict(select_validation_checkpoint(reports, validation_ids))  # Recompute the deterministic earliest equal-score checkpoint.
    selected["checkpoint_sha256"] = _sha256_file(shard / selected["checkpoint_file"])  # Bind the selected candidate bytes.
    selected["model_file"] = model_path.name  # Bind deployment to the shard-local model artifact.
    selected["model_sha256"] = _sha256_file(model_path)  # Bind deployment to exact loadable state-dict bytes.
    selected["seed"] = seed  # Bind the selection to this sole independent seed.
    selected["budget_conditioned"] = True  # Declare one policy serves all three active budgets.
    selected["validation_reference_policy"] = reference_policy  # Bind the selected synthetic checkpoint to the same explicit strict denominator contract.
    validation_history = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "validation_history", "seed": seed, "validation_reference_policy": reference_policy, "reports": reports, "selected": selected}  # Assemble every raw candidate, recomputable choice, and shared reference policy.
    validation_history_path = seed_root / "validation_history.json"  # Resolve the production validation-history filename.
    _write_json(validation_history_path, validation_history)  # Persist strict finite checkpoint-selection evidence.
    cost = {"seed": seed, "validation_reference_policy": reference_policy, "episodes_attempted": TRAIN_EPISODES, "numeric_failure_episodes": 0, "training_real_solve_attempts": TRAIN_EPISODES, "validation_real_solve_attempts": len(VALIDATION_EPISODES) * len(observations), "gradient_updates": TRAIN_EPISODES, "training_and_validation_wall_s": 1.0, "validation_wall_s": 0.5, "optimizer_training_wall_s": 0.5, "budget_episode_counts": budget_counts, "model_parameter_count": sum(parameter.numel() for parameter in DQNPolicy(_base_dqn_config(seed)).q.parameters()), "selected_checkpoint_episode": int(selected["checkpoint_episode"]), "selected_validation_key": dict(selected["selection_key"])}  # Build a complete internally consistent synthetic offline-cost record with explicit strict intent.
    cost_path = shard / "rl_training_cost.json"  # Resolve the shard-local aggregate cost filename.
    _write_json(cost_path, {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "seed_training_cost", "seed": seed, "validation_reference_policy": reference_policy, "shard_wall_s": 1.0, "cost": cost})  # Persist the independently hashed cost artifact and its strict policy.
    artifacts = {"model": {"path": model_path.name, "sha256": _sha256_file(model_path)}, "training_history": {"path": str(training_history_path.relative_to(shard)), "sha256": _sha256_file(training_history_path)}, "validation_history": {"path": str(validation_history_path.relative_to(shard)), "sha256": _sha256_file(validation_history_path)}, "training_cost": {"path": cost_path.name, "sha256": _sha256_file(cost_path)}}  # Bind every fake shard artifact to exact bytes.
    shard_index = {"schema": RL_SCHEMA, "protocol_id": PROTOCOL_ID, "phase": "seed_shard", "TEST_NOT_RUN": True, "test_split_access": False, "seed": seed, "manifest_file": manifest_path.name, "manifest_sha256": _sha256_file(manifest_path), "validation_reference_policy": reference_policy, "dqn_config": asdict(_base_dqn_config(seed)), "finite_element_contract": {"nodal_gradation": COMMON_NODAL_GRADATION, "source": "PR40_V0_default_tool_behavior"}, "episode_contract": {"episodes": TRAIN_EPISODES, "budgets": list(EQUATION_BUDGETS), "episodes_per_budget": 100}, "checkpoint_contract": {"episodes": list(VALIDATION_EPISODES), "validation_case_count": 8, "budgets": list(EQUATION_BUDGETS), "selection_order": ["failure_points", "finite_penalty_energy_log_mean", "finite_penalty_qoi_log_mean", "budget_violations", "checkpoint_episode"]}, "selected": selected, "cost": cost, "artifacts": artifacts}  # Assemble the complete fake pre-test shard identity with explicit strict policy.
    index_path = shard / "rl_seed_shard.json"  # Resolve the production shard-index filename.
    _write_json(index_path, shard_index)  # Publish the fake shard only after every required artifact exists.
    _write_sha256_sidecar(index_path)  # Seal exact index bytes for production verification.
    return shard  # Return the complete independent fixed-seed shard directory.

def test_parallel_seed_assembly_recomputes_artifacts_and_retains_all_three(tmp_path: Path) -> None:  # Verify fake complete shards merge without ranking or dropping a seed.
    manifest_path, _checksum_path, _digest = write_case_manifest(tmp_path / "protocol")  # Persist one authentic manifest and checksum for exact shard binding.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # Load the same transparent manifest content for fake schedule generation.
    shards = [_fake_complete_seed_shard(tmp_path, manifest_path, manifest, seed) for seed in RL_SEEDS]  # Build exactly three independently sealed complete fixed-seed artifacts.
    output = tmp_path / "assembled"  # Select a new unified model-set directory.
    result = assemble_bridge_rl(manifest_path, list(reversed(shards)), output)  # Assemble from reverse completion order to prove seed-order independence.
    freeze = json.loads((output / "rl_freeze_index.json").read_text(encoding="utf-8"))  # Inspect the authenticated unified pre-test contract.
    assert result["model_count"] == 3 and result["TEST_NOT_RUN"] is True  # Require a complete pre-blind three-model freeze.
    assert [model["seed"] for model in freeze["models"]] == list(RL_SEEDS)  # Retain every policy in frozen seed order without best-seed elimination.
    assert freeze["seed_merge_rule"] == "retain_all_three_without_cross_seed_selection"  # Make the no-best-seed merge rule machine-readable.
    assert freeze["finite_element_contract"]["nodal_gradation"] == 1.0  # Preserve common V0 size-field smoothing through parallel assembly.
    assert freeze["validation_reference_policy"]["allow_unqualified_references"] is False  # Preserve the fake shards' explicit strict policy through the unified freeze index.
    with pytest.raises(ValueError, match="exactly the three frozen seeds"):  # Reject three inputs that duplicate one seed instead of covering all three.
        assemble_bridge_rl(manifest_path, [shards[0], shards[0], shards[0]], tmp_path / "invalid_assembled")  # Prove assembly cannot turn repeated candidates into a best-seed model set.

def test_parallel_assembly_rehashes_amendment_before_freeze(tmp_path: Path) -> None:  # Verify an authorized shard cannot freeze after its human amendment bytes change.
    protocol_directory = tmp_path / "protocol"  # Create an isolated canonical protocol directory for manifest and amendment evidence.
    manifest_path, _checksum_path, _digest = write_case_manifest(protocol_directory)  # Persist an authentic checksummed manifest without formal data.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # Load the same manifest content used by synthetic schedule evidence.
    amendment_path = protocol_directory / "EXPEDITED_EXECUTION_AMENDMENT.md"  # Resolve the fixed pre-freeze amendment filename.
    amendment_path.write_text(f"- 授权标识：`{UNQUALIFIED_AUTHORIZATION}`\n", encoding="utf-8")  # Publish exact local authorization bytes before shard creation.
    reference_policy = build_training_reference_policy(manifest_path, allow_unqualified_references=True, expedited_reference_levels=2)  # Capture the authenticated two-level policy receipt.
    shard = _fake_complete_seed_shard(tmp_path, manifest_path, manifest, RL_SEEDS[0], reference_policy)  # Seal one complete synthetic seed against the original amendment hash.
    amendment_path.write_text(f"- 授权标识：`{UNQUALIFIED_AUTHORIZATION}`\nchanged after training\n", encoding="utf-8")  # Simulate a post-training authorization edit while retaining the token.
    output = tmp_path / "must_not_freeze"  # Select a destination whose absence proves fail-before-publication behavior.
    with pytest.raises(ValueError, match="no longer matches"):  # Require exact live amendment bytes rather than token-only acceptance during assembly.
        assemble_bridge_rl(manifest_path, [shard, shard, shard], output)  # Reach the first shard's amendment rehash before any duplicate-seed merge logic.
    assert not output.exists()  # Stop before creating a freeze directory when amendment evidence is stale.
