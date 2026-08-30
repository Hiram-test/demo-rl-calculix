"""Aggregate frozen four-way raw evidence into deterministic gates and reports."""  # Describe the module's sole post-execution responsibility.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
import csv  # Import standards-compliant tabular artifact generation.
from dataclasses import asdict  # Import typed observation serialization for raw aggregate JSON.
import hashlib  # Import exact artifact identity generation for the final audit index.
import json  # Import strict machine-readable evidence loading and persistence.
import math  # Import finite prediction-calibration calculations.
import os  # Import atomic same-filesystem publication.
from pathlib import Path  # Import portable campaign paths.
from statistics import median  # Import the deterministic odd-sample diagnostic median.
from typing import Any, Iterable, Mapping, Sequence  # Import explicit heterogeneous artifact contracts.
from ..bridge_case_manifest import load_case_manifest  # Reuse the authenticated manifest and frozen test identities.
from .four_way_benchmark import ALL_METHODS, BUDGETS, MAX_SOLVES, PROTOCOL_ID, RL_METHODS, SOLVE_LIMITS  # Reuse the exact execution grid and labels.
from .four_way_stats import DorflerObservation, FallbackEvidence, MechanismObservation, PairwiseObservation, PRIMARY_COMPETITORS, PRIMARY_OPERATING_POINTS, TargetDominanceCheck, TimeObservation, build_pairwise_ratio_rows, evaluate_dorfler_safety, evaluate_final_gate, evaluate_online_time, evaluate_primary_competitor, evaluate_world_model_mechanism, failure_aware_median_error  # Reuse the sole frozen statistical and final-gate implementation.

ANALYSIS_SCHEMA = "wmvla-four-way-analysis-v1"  # Identify the deterministic aggregate artifact family.
AGGREGATE_METHODS = ("world_model_vla", "local_prediction", "supervised", "rl_median", "dorfler")  # Freeze the reported four-way methods plus safety floor.

class IncompleteEvidenceError(RuntimeError):  # Distinguish missing raw jobs from numerical method failures retained inside jobs.
    """Report absent or malformed evidence that prevents complete-grid scoring."""  # Explain the exception category.

def _read_json(path: Path) -> Any:  # Load one strict UTF-8 JSON evidence artifact.
    return json.loads(path.read_text(encoding="utf-8"))  # Preserve transparent mapping and list structures.

def _json_safe(value: Any) -> Any:  # Normalize aggregate objects while rejecting nonstandard floating-point output.
    if isinstance(value, Mapping):  # Recurse through named evidence mappings.
        return {str(key): _json_safe(item) for key, item in value.items()}  # Preserve every key under JSON string semantics.
    if isinstance(value, (list, tuple)):  # Recurse through ordered evidence sequences.
        return [_json_safe(item) for item in value]  # Preserve execution and manifest order.
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):  # Preserve standard JSON primitives.
        return value  # Return the already safe value unchanged.
    if isinstance(value, float):  # Validate floating-point statistics explicitly.
        if not math.isfinite(value):  # Reject NaN and infinities that could hide failures.
            raise ValueError("aggregate evidence contains a nonfinite float")  # Stop before publishing invalid JSON or CSV.
        return float(value)  # Return a finite built-in scalar.
    if hasattr(value, "item"):  # Normalize NumPy scalar wrappers without importing their concrete types.
        return _json_safe(value.item())  # Reapply all finite-value rules to the Python scalar.
    return str(value)  # Preserve unknown textual diagnostics transparently.

def _write_json(path: Path, payload: Any) -> None:  # Persist one complete aggregate JSON artifact atomically.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the exact aggregate directory.
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")  # Isolate an interrupted write from readers.
    temporary.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Serialize stable strict JSON.
    os.replace(temporary, path)  # Publish the complete artifact atomically on the same filesystem.

def _csv_cell(value: Any) -> Any:  # Convert nested evidence into stable CSV cells.
    if value is None:  # Represent unavailable metrics as empty cells rather than textual zero.
        return ""  # Preserve failure semantics without inventing a value.
    if isinstance(value, bool):  # Normalize boolean text independently from integers.
        return "true" if value else "false"  # Use lowercase JSON-compatible spelling.
    if isinstance(value, (Mapping, list, tuple)):  # Preserve nested provenance in one auditable cell.
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)  # Encode compact deterministic JSON.
    return value  # Preserve finite scalar values directly.

def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:  # Persist deterministic union-column CSV atomically.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the exact aggregate directory.
    columns = sorted({str(key) for row in rows for key in row})  # Build a stable complete column vocabulary without dropping optional evidence.
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")  # Isolate incomplete tabular output.
    with temporary.open("w", encoding="utf-8", newline="") as handle:  # Apply the CSV module's platform-independent newline contract.
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")  # Reject unexpected row mutation after column collection.
        writer.writeheader()  # Emit an explicit schema row even for machine consumption.
        for row in rows:  # Preserve caller-supplied deterministic row order.
            writer.writerow({key: _csv_cell(row.get(key)) for key in columns})  # Encode every declared column and empty missing optional cell.
    os.replace(temporary, path)  # Atomically publish the complete CSV.

def _sha256_file(path: Path) -> str:  # Hash exact aggregate artifact bytes for the delivery index.
    digest = hashlib.sha256()  # Initialize the collision-resistant digest.
    with path.open("rb") as handle:  # Stream potentially large CSV or JSON evidence.
        for block in iter(lambda: handle.read(1024 * 1024), b""):  # Read deterministic bounded blocks to EOF.
            digest.update(block)  # Incorporate every exact byte.
    return digest.hexdigest()  # Return the complete lowercase hexadecimal identity.

def _method_dir(root: Path, case_id: str, budget: int, method: str) -> Path:  # Resolve one protocol-layout raw trajectory directory.
    return root / "test" / case_id / str(int(budget)) / method  # Bind exact case, public equation budget, and frozen method label.

def _validate_prefix_row(row: Mapping[str, Any], case_id: str, budget: int, method: str, solves: int) -> dict[str, Any]:  # Normalize one raw true-prefix row against its directory identity.
    expected = {"case_id": case_id, "equation_budget": int(budget), "method": method, "solves": int(solves)}  # Build the immutable row coordinates.
    if any(row.get(key) != value for key, value in expected.items()):  # Reject cross-job copies and mislabeled prefixes.
        raise IncompleteEvidenceError(f"prefix row identity mismatch for {case_id}/{budget}/{method}/K{solves}")  # Identify the exact malformed point.
    normalized = dict(row)  # Preserve all raw safety and failure fields for later tables.
    for metric in ("energy_error", "qoi_error"):  # Validate both independent delivered errors.
        value = normalized.get(metric)  # Read the optional best-feasible prefix value.
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):  # Reject nonfinite or negative relative errors.
            normalized[metric] = None  # Retain the point as a visible failure rather than accepting malformed data.
            normalized[metric.replace("error", "ok")] = False  # Keep its machine status aligned with the missing value.
    return normalized  # Return the identity-bound finite row.

def load_raw_prefix_rows(root: Path, case_ids: Sequence[str], *, allow_incomplete: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:  # Load the exact complete raw grid while retaining absent-job coverage evidence.
    rows: list[dict[str, Any]] = []  # Accumulate all case-budget-method-prefix rows.
    missing: list[dict[str, Any]] = []  # Accumulate absent or malformed job artifacts without silently dropping them.
    for case_id in sorted(case_ids):  # Preserve the protocol's blind execution order.
        for budget in BUDGETS:  # Inspect every independent public budget trajectory.
            for method in ALL_METHODS:  # Inspect WM, LP, supervised, all RL seeds, and independent Dörfler.
                path = _method_dir(root, case_id, budget, method) / "prefix_results.json"  # Resolve the mandatory derived raw prefix artifact.
                try:  # Convert filesystem and structural gaps into explicit coverage rows.
                    payload = _read_json(path)  # Parse the complete job prefix artifact.
                    values = payload.get("rows") if isinstance(payload, Mapping) else None  # Read the four-row list only from a named object.
                    if not isinstance(values, list) or len(values) != len(SOLVE_LIMITS):  # Require exactly K=2,3,4,6 once.
                        raise IncompleteEvidenceError("prefix artifact does not contain four rows")  # Preserve an exact structural failure.
                    by_k = {int(value.get("solves", -1)): value for value in values if isinstance(value, Mapping)}  # Index rows by solve limit while rejecting scalars.
                    if set(by_k) != set(SOLVE_LIMITS):  # Require the exact registered prefix set.
                        raise IncompleteEvidenceError("prefix artifact solve limits differ from 2,3,4,6")  # Reject duplicate or post-hoc K values.
                    rows.extend(_validate_prefix_row(by_k[solves], case_id, budget, method, solves) for solves in SOLVE_LIMITS)  # Retain all identity-bound true-prefix values.
                except (OSError, json.JSONDecodeError, IncompleteEvidenceError, TypeError, ValueError) as exception:  # Retain every unreadable or malformed job.
                    missing.append({"case_id": case_id, "equation_budget": int(budget), "method": method, "path": str(path), "error_type": type(exception).__name__, "error": str(exception)[:1000]})  # Preserve finite actionable coverage evidence.
    expected_count = len(case_ids) * len(BUDGETS) * len(ALL_METHODS) * len(SOLVE_LIMITS)  # Compute the exact required raw row cardinality.
    if (missing or len(rows) != expected_count) and not allow_incomplete:  # Fail before statistics on any absent or malformed raw job.
        raise IncompleteEvidenceError(f"raw grid incomplete: loaded {len(rows)}/{expected_count} rows with {len(missing)} missing jobs")  # Report complete coverage counts without selective scoring.
    return rows, missing  # Return valid raw rows and explicit missing-job evidence.

def build_rl_median_rows(raw_rows: Sequence[Mapping[str, Any]], case_ids: Sequence[str]) -> list[dict[str, Any]]:  # Compute the required pointwise three-policy median on the full twelve-point grid.
    indexed = {(str(row["case_id"]), int(row["solves"]), int(row["equation_budget"]), str(row["method"])): row for row in raw_rows if str(row.get("method")) in RL_METHODS}  # Index every seed point by exact public coordinates.
    medians: list[dict[str, Any]] = []  # Accumulate one transparent median row per case, K, and B.
    for case_id in sorted(case_ids):  # Preserve blind case order.
        for budget in BUDGETS:  # Preserve public budget order.
            for solves in SOLVE_LIMITS:  # Preserve true-prefix order.
                outcomes = [indexed.get((case_id, solves, budget, method)) for method in RL_METHODS]  # Read all three policy-index outcomes at this exact point.
                if any(outcome is None for outcome in outcomes):  # Refuse best-of-fewer seed aggregation.
                    raise IncompleteEvidenceError(f"RL median lacks three seed rows for {case_id}, K={solves}, B={budget}")  # Identify the incomplete denominator.
                seed_rows = [dict(outcome) for outcome in outcomes if outcome is not None]  # Narrow optional values after the complete-seed check.
                energy = failure_aware_median_error([row.get("energy_error") for row in seed_rows], [bool(row.get("energy_ok")) for row in seed_rows])  # Rank failed energy outcomes after all finite values and select the middle seed pointwise.
                qoi = failure_aware_median_error([row.get("qoi_error") for row in seed_rows], [bool(row.get("qoi_ok")) for row in seed_rows])  # Apply the independent pointwise failure-aware median to QoI.
                medians.append({"case_id": case_id, "method": "rl_median", "solves": int(solves), "equation_budget": int(budget), "energy_error": energy, "qoi_error": qoi, "energy_ok": energy is not None, "qoi_ok": qoi is not None, "budget_violation": any(bool(row.get("budget_violation")) for row in seed_rows), "seed_budget_violation_count": sum(bool(row.get("budget_violation")) for row in seed_rows), "seed_energy_outcomes": [{"policy_index": index, "value": row.get("energy_error"), "ok": bool(row.get("energy_ok"))} for index, row in enumerate(seed_rows)], "seed_qoi_outcomes": [{"policy_index": index, "value": row.get("qoi_error"), "ok": bool(row.get("qoi_ok"))} for index, row in enumerate(seed_rows)], "aggregation": "pointwise_three_seed_median_failures_after_finite"})  # Preserve the denominator and every source outcome for audit.
    return medians  # Return the complete 16-by-12 RL-median grid.

def build_primary_table(raw_rows: Sequence[Mapping[str, Any]], rl_medians: Sequence[Mapping[str, Any]], case_ids: Sequence[str]) -> list[dict[str, Any]]:  # Assemble the complete reported five-method twelve-point grid.
    retained = [dict(row) for row in raw_rows if str(row.get("method")) in ("world_model_vla", "local_prediction", "supervised", "dorfler")]  # Retain the three primary non-RL methods and safety comparator.
    combined = retained + [dict(row) for row in rl_medians]  # Add only the pointwise RL median, never a best seed.
    expected = len(case_ids) * len(BUDGETS) * len(SOLVE_LIMITS) * len(AGGREGATE_METHODS)  # Compute exact complete table cardinality.
    if len(combined) != expected:  # Refuse a misleading partial primary_results table.
        raise IncompleteEvidenceError(f"aggregate method grid has {len(combined)} rows; expected {expected}")  # Report the exact cardinality gap.
    for row in combined:  # Mark the six preregistered points without discarding the other six.
        row["primary_operating_point"] = (int(row["solves"]), int(row["equation_budget"])) in PRIMARY_OPERATING_POINTS  # Preserve complete twelve-point evidence and predeclared gate membership.
    order = {method: index for index, method in enumerate(AGGREGATE_METHODS)}  # Build stable method reporting order.
    return sorted(combined, key=lambda row: (str(row["case_id"]), int(row["equation_budget"]), int(row["solves"]), order[str(row["method"])]))  # Return deterministic case-budget-K-method rows.

def _table_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int, int], Mapping[str, Any]]:  # Index aggregate rows by exact method and public operating coordinates.
    indexed = {(str(row["case_id"]), str(row["method"]), int(row["solves"]), int(row["equation_budget"])): row for row in rows}  # Build one unique lookup key per row.
    if len(indexed) != len(rows):  # Reject duplicate evidence that could replace an unfavorable outcome.
        raise IncompleteEvidenceError("aggregate table contains duplicate case-method-K-B rows")  # Preserve one-to-one public-grid scoring.
    return indexed  # Return the complete unique lookup.

def evaluate_primary_gates(rows: Sequence[Mapping[str, Any]], case_ids: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # Evaluate all three primary competitors on the six preregistered points.
    indexed = _table_index(rows)  # Build exact paired lookups across the complete twelve-point table.
    reports: dict[str, Any] = {}  # Collect one seven-term machine gate per competitor.
    ratio_rows: list[dict[str, Any]] = []  # Collect transparent long-form energy and QoI ratios.
    for competitor in PRIMARY_COMPETITORS:  # Evaluate LP, supervised, and pointwise RL median independently.
        observations: list[PairwiseObservation] = []  # Build the complete sixteen-case by six-point paired grid.
        for case_id in sorted(case_ids):  # Cluster all six points under each blind case identity.
            for solves, budget in PRIMARY_OPERATING_POINTS:  # Use only the six fixed gate points.
                wm = indexed[(case_id, "world_model_vla", solves, budget)]  # Read the WM numerator.
                other = indexed[(case_id, competitor, solves, budget)]  # Read the named competitor denominator.
                observations.append(PairwiseObservation(case_id=case_id, competitor=competitor, solves=solves, equation_budget=budget, wm_energy_error=wm.get("energy_error"), competitor_energy_error=other.get("energy_error"), wm_qoi_error=wm.get("qoi_error"), competitor_qoi_error=other.get("qoi_error"), wm_energy_ok=bool(wm.get("energy_ok")), competitor_energy_ok=bool(other.get("energy_ok")), wm_qoi_ok=bool(wm.get("qoi_ok")), competitor_qoi_ok=bool(other.get("qoi_ok")), wm_budget_violation=bool(wm.get("budget_violation")), wm_proactive_action=bool(wm.get("certified_proactive_action"))))  # Preserve failures, budget events, and actual certified proactive execution.
        reports[competitor] = evaluate_primary_competitor(observations, competitor)  # Apply geometric, case-bootstrap, breadth, tail, QoI, budget, and mechanism-coverage gates.
        ratio_rows.extend(build_pairwise_ratio_rows(observations, competitor))  # Emit every finite failure-aware ratio used by the decision.
    return reports, sorted(ratio_rows, key=lambda row: (str(row["case_id"]), str(row["competitor"]), int(row["equation_budget"]), int(row["solves"])))  # Return deterministic gate and ratio evidence.

def _action_log(root: Path, case_id: str, budget: int) -> Mapping[str, Any]:  # Load one WM action log for safety, calibration, and mechanism evidence.
    path = _method_dir(root, case_id, budget, "world_model_vla") / "action_log.json"  # Resolve the exact primary WM trajectory log.
    payload = _read_json(path)  # Parse the complete per-job action artifact.
    if not isinstance(payload, Mapping):  # Require the named action-log schema.
        raise IncompleteEvidenceError(f"WM action log is malformed: {path}")  # Reject opaque or truncated safety evidence.
    return payload  # Return the transparent action mapping.

def evaluate_dorfler_gate(root: Path, rows: Sequence[Mapping[str, Any]], case_ids: Sequence[str]) -> dict[str, Any]:  # Evaluate structural and empirical Dörfler safety on all primary points.
    indexed = _table_index(rows)  # Build exact WM and independent-Dörfler lookups.
    observations: list[DorflerObservation] = []  # Collect the complete paired empirical grid.
    for case_id in sorted(case_ids):  # Preserve blind case clustering.
        for solves, budget in PRIMARY_OPERATING_POINTS:  # Restrict empirical safety to preregistered points.
            wm = indexed[(case_id, "world_model_vla", solves, budget)]  # Read the WM safety numerator.
            dorfler = indexed[(case_id, "dorfler", solves, budget)]  # Read the independent exact-Dörfler denominator.
            observations.append(DorflerObservation(case_id=case_id, solves=solves, equation_budget=budget, wm_energy_error=wm.get("energy_error"), dorfler_energy_error=dorfler.get("energy_error"), wm_ok=bool(wm.get("energy_ok")), dorfler_ok=bool(dorfler.get("energy_ok")), wm_budget_violation=bool(wm.get("budget_violation"))))  # Preserve every failure and budget event.
    dominance: list[TargetDominanceCheck] = []  # Collect one structural receipt per actually solved WM target.
    fallbacks: list[FallbackEvidence] = []  # Collect every planner or tool fallback event.
    for case_id in sorted(case_ids):  # Inspect all independent WM budget trajectories.
        for budget in BUDGETS:  # Include structural evidence from the complete registered resource grid.
            payload = _action_log(root, case_id, budget)  # Load this trajectory's exact action evidence.
            action_container = payload.get("actions", {})  # Read the method-specific action payload.
            certificates_value = action_container.get("certificates", []) if isinstance(action_container, Mapping) else []  # Read exact materialization certificates.
            decisions_value = action_container.get("decisions", []) if isinstance(action_container, Mapping) else []  # Read corresponding planner decisions.
            records_payload = _read_json(_method_dir(root, case_id, budget, "world_model_vla") / "records.json")  # Load successful real solves to distinguish materialized from actually executed targets.
            record_count = len(records_payload.get("records", [])) if isinstance(records_payload, Mapping) and isinstance(records_payload.get("records"), list) else 0  # Count completed real WM solves.
            executed_count = min(len(certificates_value) if isinstance(certificates_value, list) else 0, max(record_count - 1, 0))  # A target is actually executed only when its next real solve completed.
            certificates = certificates_value[:executed_count] if isinstance(certificates_value, list) else []  # Exclude unsolved preflight-only terminal candidates.
            decisions = decisions_value if isinstance(decisions_value, list) else []  # Normalize optional planner evidence.
            for index, certificate in enumerate(certificates):  # Verify every target that reached a subsequent real solve.
                if not isinstance(certificate, Mapping):  # Treat malformed certificates as failed structural evidence.
                    certificate = {}  # Preserve a deterministic failed receipt rather than dropping the action.
                action_id = f"{case_id}:B{budget}:A{index + 1}:{certificate.get('target_sha256', 'missing')}"  # Build a globally unique content-bound action identity.
                source = str(certificate.get("source", ""))  # Read actual tool-selected provenance before compiled-field classification.
                proactive = source == "world_model" and any(int(depth) > 0 for depth in certificate.get("executed_action", []))  # Determine whether a compiled proactive identity is required.
                base_hash = str(certificate.get("base_compiled_field_sha256", ""))  # Read the compiled exact-Dörfler field identity.
                world_hash = str(certificate.get("world_compiled_field_sha256", ""))  # Read the compiled proactive field identity when applicable.
                passed = str(certificate.get("schema_version", "")) == "wmvla.mcp-tool.v2" and bool(certificate.get("base_target_included")) and bool(certificate.get("no_coarsening")) and bool(certificate.get("compiled_dorfler_included")) and float(certificate.get("compiled_max_dorfler_violation", float("inf"))) <= 1.0e-12 and float(certificate.get("compiled_field_gradation", float("nan"))) == 1.0 and len(base_hash) == 64 and (not proactive or len(world_hash) == 64)  # Require complete v2 compiled-field dominance under the frozen tolerance and gradation.
                dominance.append(TargetDominanceCheck(case_id=case_id, action_id=action_id, passed=passed))  # Retain the structural gate observation.
                decision = decisions[index] if index < len(decisions) and isinstance(decisions[index], Mapping) else {}  # Align the planner decision by executed action index.
                executed_depth = certificate.get("executed_action", [])  # Read the exact materialized extra-depth vector.
                fallback_required = not bool(decision.get("accepted")) or source != "world_model"  # Classify planner rejection, distrust, or tool resource fallback.
                if fallback_required:  # Retain every required safety fallback independently.
                    trigger = str(certificate.get("reason") or decision.get("reason") or "unspecified_fallback")  # Preserve a nonempty exact trigger for the stats contract.
                    fallbacks.append(FallbackEvidence(case_id=case_id, trigger=f"B{budget}:A{index + 1}:{trigger}", dorfler_executed=not any(int(depth) > 0 for depth in executed_depth)))  # Require the actual action to become pure Dörfler.
    return evaluate_dorfler_safety(observations, dominance, fallbacks)  # Apply nodewise, aggregate, worst-point, budget, and fallback conjunctions.

def _timing_payload(root: Path, case_id: str, budget: int, method: str) -> Mapping[str, Any]:  # Load one raw method timing artifact.
    payload = _read_json(_method_dir(root, case_id, budget, method) / "timing.json")  # Parse the exact job timing file.
    values = payload.get("timing_s") if isinstance(payload, Mapping) else None  # Read only the named separated timing mapping.
    if not isinstance(values, Mapping):  # Require explicit measured timing components.
        raise IncompleteEvidenceError(f"timing evidence is malformed for {case_id}/{budget}/{method}")  # Prevent implicit zero timing.
    normalized = dict(values)  # Copy measured components before attaching trajectory-level status provenance.
    normalized["_trajectory_completed"] = payload.get("completed") is True  # Make a typed partial native-failure timing visibly invalid for the engineering gate.
    return normalized  # Return transparent measured components plus completion provenance.

def evaluate_time_gate(root: Path, case_ids: Sequence[str]) -> dict[str, Any]:  # Evaluate online efficiency at the fixed B=60000 full trajectory.
    observations: list[TimeObservation] = []  # Collect exactly one paired WM/LP timing observation per blind case.
    for case_id in sorted(case_ids):  # Preserve deterministic case pairing.
        wm = _timing_payload(root, case_id, 60000, "world_model_vla")  # Read complete WM separated timing.
        lp = _timing_payload(root, case_id, 60000, "local_prediction")  # Read paired LP complete online timing.
        visual = float(wm.get("visual_partition", 0.0)) + float(wm.get("visual_partition_s", 0.0))  # Include frozen-spec verification and new-stack visual-cache work once.
        world = float(wm.get("world_model_planning", wm.get("world_model_s", 0.0))) + float(wm.get("unattributed_python", 0.0)) + float(wm.get("harness_setup_s", 0.0))  # Conservatively assign pipeline residual and external model/config setup to WM online work.
        tools = float(wm.get("parameter_tools", wm.get("parameter_tools_s", 0.0)))  # Read deterministic parameter and certification work.
        gmsh = float(wm.get("gmsh_remeshing", wm.get("gmsh_s", 0.0)))  # Read all exact candidate remeshing work.
        calculix = float(wm.get("calculix", wm.get("calculix_s", 0.0)))  # Read counted native solver wall time.
        lp_total = float(lp.get("online_total_s", lp.get("trajectory_total", 0.0)))  # Read the complete paired local-prediction online wall clock.
        visual = visual if bool(wm.get("_trajectory_completed")) and bool(lp.get("_trajectory_completed")) else -1.0  # Force typed partial WM or LP trajectories into the time gate's explicit invalid-case path.
        observations.append(TimeObservation(case_id=case_id, vlm_calls=int(wm.get("vlm_calls", 0)), visual_partition_s=visual, world_model_s=world, parameter_tools_s=tools, gmsh_s=gmsh, calculix_s=calculix, lp_online_total_s=lp_total))  # Preserve every separated component without cross-case pooling.
    return evaluate_online_time(observations)  # Apply one-VLM, median overhead, and aggregate WM/LP wall-time gates.

def _ablation_case_path(root: Path, case_id: str) -> Path | None:  # Resolve one canonical per-case mandatory diagnostic summary.
    case_root = root / "ablations" / case_id  # Resolve the protocol-mandated per-case ablation directory.
    candidates = (case_root / "ablation_case.json", case_root / "summary.json", case_root / "diagnostic_summary.json")  # Support the frozen module filename and transparent compatibility aliases.
    return next((path for path in candidates if path.is_file()), None)  # Select the first explicit existing artifact deterministically.

def evaluate_mechanism_gate(root: Path, case_ids: Sequence[str]) -> dict[str, Any]:  # Evaluate attribution from mandatory full, h1, and random-safe diagnostics.
    observations: list[MechanismObservation] = []  # Collect exactly one fixed K=6, B=60000 row per blind case.
    missing: list[str] = []  # Retain absent or malformed diagnostic case identities.
    source_paths: list[str] = []  # Preserve exact diagnostic artifact provenance.
    for case_id in sorted(case_ids):  # Preserve blind case order.
        path = _ablation_case_path(root, case_id)  # Resolve this case's mandatory diagnostic summary.
        if path is None:  # Fail closed when any of the sixteen diagnostics is absent.
            missing.append(case_id)  # Retain the missing case explicitly.
            continue  # Inspect all remaining cases for a complete repair list.
        try:  # Convert schema and numerical errors into incomplete mechanism evidence.
            payload = _read_json(path)  # Parse the case-level frozen diagnostic artifact.
            if not isinstance(payload, Mapping) or payload.get("schema") != "wmvla-four-way-ablation-case-v1" or payload.get("case_id") != case_id:  # Require exact schema and case identity.
                raise ValueError("ablation schema or case_id mismatch")  # Reject stale or cross-case diagnostic results.
            point = payload.get("operating_point", {})  # Read the fixed diagnostic operating point.
            if not isinstance(point, Mapping) or int(point.get("equation_budget", -1)) != 60000 or int(point.get("max_solves", -1)) != 6:  # Require B=60000 and K=6.
                raise ValueError("ablation operating point is not B=60000, K=6")  # Reject post-hoc diagnostic resources.
            variants = payload.get("variants", {})  # Read all six mandatory variant records.
            required = {"wm_full", "wm_h1", "wm_prior_only", "wm_no_history", "random_safe_extra", "oracle_future_hit"}  # Freeze the mandatory diagnostic vocabulary.
            if not isinstance(variants, Mapping) or not required.issubset(variants):  # Require every diagnostic even though the mechanism gate uses a strict subset.
                raise ValueError("ablation case lacks one or more mandatory variants")  # Prevent selective mechanism reporting.
            full = variants["wm_full"]  # Read the full-controller observation.
            h1 = variants["wm_h1"]  # Read the otherwise-identical horizon-one control.
            random_safe = variants["random_safe_extra"]  # Read the five-seed random-safe control aggregate.
            if not all(isinstance(value, Mapping) for value in (full, h1, random_safe)):  # Require transparent named variant records.
                raise ValueError("ablation variant is malformed")  # Reject scalar or opaque variant evidence.
            seed_values = random_safe.get("seeds", {})  # Read all five random control outcomes.
            if not isinstance(seed_values, Mapping) or len(seed_values) != 5:  # Require the preregistered five-seed control basis.
                raise ValueError("random_safe_extra must contain five seeds")  # Reject best-of-fewer or omitted failures.
            common_hashes = [str(value.get("common_probe_sha256", "")) for value in (full, h1, random_safe)]  # Read common-probe identities from compared variants.
            common_probe = bool(common_hashes[0]) and len(set(common_hashes)) == 1  # Require one nonempty identical common uniform mesh hash.
            matched_budget = all(bool(value.get("matched_budget")) for value in (full, h1, random_safe))  # Require equal K and B without extra feedback solves.
            isolation = all(bool(value.get("competitor_isolation")) for value in (full, h1, random_safe))  # Require no LP, supervised, RL, or future-reference leakage.
            observations.append(MechanismObservation(case_id=case_id, wm_full_energy_error=full.get("energy_error"), wm_h1_energy_error=h1.get("energy_error"), random_safe_median_energy_error=random_safe.get("median_energy_error"), wm_full_ok=bool(full.get("ok")), wm_h1_ok=bool(h1.get("ok")), random_safe_ok=bool(random_safe.get("median_ok")), certified_proactive_actions=int(full.get("certified_proactive_actions", 0)), executed_proactive_actions=int(full.get("executed_proactive_actions", 0)), common_uniform_probe=common_probe, matched_solve_budget=matched_budget, competitor_isolation=isolation))  # Preserve failure-aware fixed-point mechanism evidence.
            source_paths.append(str(path))  # Preserve the exact successfully parsed artifact path.
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exception:  # Retain malformed diagnostics as missing mechanism evidence.
            missing.append(f"{case_id}:{type(exception).__name__}:{str(exception)[:300]}")  # Provide a bounded actionable reason.
    if missing or len(observations) != len(case_ids):  # Fail closed without calling a complete-grid statistic on partial diagnostics.
        return {"schema": "wmvla-four-way-mechanism-v1", "protocol_id": PROTOCOL_ID, "case_count": len(observations), "expected_case_count": len(case_ids), "source_paths": source_paths, "missing_or_invalid_cases": missing, "gates": {"complete_mandatory_ablations": False}, "passed": False}  # Return an explicit non-passing attribution result.
    result = evaluate_world_model_mechanism(observations)  # Apply proactive, certification, isolation, h1, random, and case-bootstrap conjunctions.
    result["source_paths"] = source_paths  # Bind the machine decision to all sixteen exact diagnostic artifacts.
    return result  # Return the complete mechanism attribution evidence.

def prediction_calibration_rows(root: Path, case_ids: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # Extract every primary WM realized transition and summarize available calibrated quantities.
    rows: list[dict[str, Any]] = []  # Collect one prediction-versus-realization row per completed transition.
    fallback_counts: dict[str, int] = {}  # Count exact planner fallback reasons without merging categories.
    accepted_count = 0  # Count planner-accepted proactive candidates.
    accepted_improved = 0  # Count accepted candidates whose next real estimator improved.
    for case_id in sorted(case_ids):  # Preserve blind case order.
        for budget in BUDGETS:  # Include every independent public budget trajectory.
            action = _action_log(root, case_id, budget)  # Load planner decisions and exact mesh certificates.
            action_values = action.get("actions", {})  # Read the method-specific payload wrapper.
            decisions = action_values.get("decisions", []) if isinstance(action_values, Mapping) else []  # Read planned action evidence.
            certificates = action_values.get("certificates", []) if isinstance(action_values, Mapping) else []  # Read exact resource and target evidence.
            records_payload = _read_json(_method_dir(root, case_id, budget, "world_model_vla") / "records.json")  # Load realized solve records.
            records = records_payload.get("records", []) if isinstance(records_payload, Mapping) else []  # Normalize the transparent record list.
            if not isinstance(decisions, list) or not isinstance(certificates, list) or not isinstance(records, list):  # Reject malformed calibration source arrays.
                raise IncompleteEvidenceError(f"WM calibration sources malformed for {case_id}/B{budget}")  # Prevent silent transition omission.
            realized_count = min(len(decisions), len(certificates), max(len(records) - 1, 0))  # Restrict to actions followed by a completed real solve.
            for index in range(realized_count):  # Pair each pre-action prediction with its actual successor.
                decision = decisions[index] if isinstance(decisions[index], Mapping) else {}  # Normalize planner evidence.
                certificate = certificates[index] if isinstance(certificates[index], Mapping) else {}  # Normalize exact resource evidence.
                successor = records[index + 1] if isinstance(records[index + 1], Mapping) else {}  # Read the realized next solve.
                extra = successor.get("extra", {}) if isinstance(successor.get("extra", {}), Mapping) else {}  # Read transition audit metadata attached by the new pipeline.
                actual_error_ratio = extra.get("wmvla_actual_error_ratio")  # Read the realized total estimator ratio.
                predicted_upper = decision.get("predicted_error_ratio_upper", extra.get("wmvla_predicted_error_ratio_upper"))  # Read the pre-execution conservative error bound.
                predicted_equations = certificate.get("estimated_equations")  # Read exact solve-free active-DOF preflight.
                actual_equations = successor.get("n_equations")  # Read realized active equations from CalculiX.
                error_log_abs = None if actual_error_ratio is None or predicted_upper is None or float(actual_error_ratio) <= 0.0 or float(predicted_upper) <= 0.0 else abs(math.log(float(actual_error_ratio) / float(predicted_upper)))  # Measure available upper-bound log discrepancy transparently.
                equation_mape = None if predicted_equations in (None, 0) or actual_equations is None else abs(float(actual_equations) - float(predicted_equations)) / max(abs(float(actual_equations)), 1.0)  # Measure exact preflight resource percentage error.
                covered = None if actual_error_ratio is None or predicted_upper is None else float(actual_error_ratio) <= float(predicted_upper)  # Test one-sided conservative error-bound coverage.
                accepted = bool(decision.get("accepted"))  # Read proactive planner acceptance before tool fallback.
                improved = None if actual_error_ratio is None else float(actual_error_ratio) < 1.0  # Classify realized global estimator improvement.
                accepted_count += int(accepted)  # Accumulate accepted action count.
                accepted_improved += int(accepted and improved is True)  # Accumulate successful accepted actions.
                reason = str(decision.get("reason", "missing_reason"))  # Read the exact planner acceptance or fallback reason.
                if not accepted:  # Count every fallback category separately.
                    fallback_counts[reason] = fallback_counts.get(reason, 0) + 1  # Increment the exact reason count.
                rows.append({"case_id": case_id, "equation_budget": int(budget), "transition": index + 1, "predicted_error_ratio_upper": predicted_upper, "actual_error_ratio": actual_error_ratio, "upper_bound_log_absolute_error": error_log_abs, "error_interval_covered": covered, "predicted_equations": predicted_equations, "actual_equations": actual_equations, "equation_mape": equation_mape, "planner_accepted": accepted, "actual_improvement": improved, "planner_reason": reason, "certificate_source": certificate.get("source"), "certificate_accepted": certificate.get("accepted"), "target_sha256": certificate.get("target_sha256")})  # Preserve all available prediction, realization, uncertainty-bound, resource, and fallback evidence.
    log_errors = [float(row["upper_bound_log_absolute_error"]) for row in rows if row.get("upper_bound_log_absolute_error") is not None]  # Collect finite available error-bound discrepancies.
    equation_errors = [float(row["equation_mape"]) for row in rows if row.get("equation_mape") is not None]  # Collect finite resource percentage errors.
    coverage = [bool(row["error_interval_covered"]) for row in rows if row.get("error_interval_covered") is not None]  # Collect defined interval-coverage observations.
    aggregate = {"schema": "wmvla-four-way-prediction-calibration-v1", "protocol_id": PROTOCOL_ID, "realized_transition_count": len(rows), "available_error_bound_log_mae": None if not log_errors else float(sum(log_errors) / len(log_errors)), "equation_mape": None if not equation_errors else float(sum(equation_errors) / len(equation_errors)), "one_sided_error_bound_coverage": None if not coverage else float(sum(coverage) / len(coverage)), "proactive_acceptance_rate": None if not rows else float(accepted_count / len(rows)), "accepted_action_real_improvement_rate": None if accepted_count == 0 else float(accepted_improved / accepted_count), "fallback_reason_counts": fallback_counts, "error_metric_note": "runtime stores conservative upper ratio; raw mean-prediction log-MAE must come from diagnostic traces", "candidate_rank_spearman": None}  # Summarize available metrics while explicitly marking unavailable required diagnostics.
    return rows, aggregate  # Return transparent transition rows and non-fabricated aggregate diagnostics.

def _write_text(path: Path, content: str) -> None:  # Publish one deterministic UTF-8 report atomically.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the exact report parent directory.
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")  # Isolate an interrupted report write from readers.
    temporary.write_text(content, encoding="utf-8")  # Persist the complete already-rendered Markdown document.
    os.replace(temporary, path)  # Publish the report atomically on the same filesystem.

def _analysis_boundary(root: Path, manifest_path: Path, allow_incomplete: bool) -> tuple[list[str], list[dict[str, Any]]]:  # Authenticate the blind case set and irreversible campaign markers before aggregation.
    manifest = load_case_manifest(manifest_path, verify_checksum=True)  # Reuse the sole schema, checksum, geometry, and split validator.
    case_ids = sorted(str(case["case_id"]) for case in manifest["cases"] if case.get("split") == "test")  # Recover only manifest-owned blind identities in canonical order.
    issues: list[dict[str, Any]] = []  # Collect explicit boundary problems for fail-closed diagnostic analysis.
    if len(case_ids) != 16:  # Require the protocol's exact blind cardinality independently from raw rows.
        issues.append({"kind": "manifest_case_count", "expected": 16, "observed": len(case_ids)})  # Preserve the exact case-count mismatch.
    started_path = root / "test" / "TEST_STARTED.json"  # Resolve the irreversible one-shot marker.
    try:  # Validate marker schema, case order, and full resource grid.
        started = _read_json(started_path)  # Parse the complete machine-written start marker.
        if not isinstance(started, Mapping) or started.get("schema") != "wmvla-four-way-test-started-v1" or started.get("case_order") != case_ids or tuple(started.get("budgets", ())) != BUDGETS or tuple(started.get("methods", ())) != ALL_METHODS:  # Require exact manifest and runner agreement.
            raise ValueError("TEST_STARTED identity or full-grid contract mismatch")  # Surface a stale, partial, or hand-edited campaign marker.
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exception:  # Retain missing or malformed start evidence explicitly.
        issues.append({"kind": "test_started", "path": str(started_path), "error_type": type(exception).__name__, "error": str(exception)[:1000]})  # Preserve an actionable bounded marker failure.
    invalid_path = root / "test" / "TEST_INVALID.json"  # Resolve an explicit fatal campaign-invalid marker.
    if invalid_path.exists():  # Refuse to score protocol, API, reference, or programming failures as numerical method outcomes.
        issues.append({"kind": "test_invalid", "path": str(invalid_path), "payload": _read_json(invalid_path)})  # Preserve the exact fatal stop boundary in coverage evidence.
    if issues and not allow_incomplete:  # Stop default scientific analysis on any invalid or incomplete campaign boundary.
        raise IncompleteEvidenceError(f"analysis boundary invalid with {len(issues)} issue(s)")  # Require explicit diagnostic opt-in for fail-closed partial artifacts.
    return case_ids, issues  # Return canonical cases and transparent boundary evidence.

def _reference_qualification_evidence(root: Path, case_ids: Sequence[str], *, allow_unqualified_references: bool, allow_incomplete: bool) -> dict[str, Any]:  # Audit every posthoc denominator receipt and preserve original qualification separately from authorization.
    started_path = root / "test" / "TEST_STARTED.json"  # Resolve the campaign-level pre-solve qualification disclosure.
    try:  # Retain missing start evidence only in explicitly incomplete diagnostic mode.
        started = _read_json(started_path)  # Parse the same irreversible marker authenticated by the analysis boundary.
    except (OSError, json.JSONDecodeError) as exception:  # Convert an absent marker into an explicit unavailable receipt for diagnostic output.
        if not allow_incomplete:  # Keep complete analysis strict even if a caller bypasses the normal boundary helper.
            raise IncompleteEvidenceError("reference qualification lacks TEST_STARTED evidence") from exception  # Refuse an undisclosed denominator policy.
        return {"available": False, "REFERENCE_QUALIFIED": False, "allow_unqualified_references": bool(allow_unqualified_references), "reason": f"{type(exception).__name__}:{str(exception)[:500]}", "rows": []}  # Publish no inferred convergence claim.
    if not isinstance(started, Mapping):  # Require named start-marker fields before reading waiver intent.
        raise IncompleteEvidenceError("TEST_STARTED reference qualification evidence is malformed")  # Treat scalar start evidence as an integrity failure.
    start_allows_unqualified = started.get("allow_unqualified_references") is True  # Read the exact effective runtime waiver disclosed before the first solve.
    start_expedited_levels = started.get("expedited_reference_levels")  # Read the authenticated frozen amendment depth disclosed beside the waiver.
    start_amendment_sha256 = started.get("reference_execution_amendment_sha256")  # Read the exact protected human amendment identity disclosed before the first solve.
    rows: list[dict[str, Any]] = []  # Collect one raw method denominator receipt per scheduled trajectory.
    missing: list[str] = []  # Retain absent or malformed records artifacts without dropping a case.
    for case_id in sorted(case_ids):  # Preserve blind case order across all three budgets and seven raw methods.
        for budget in BUDGETS:  # Audit each independently run public budget trajectory.
            for method in ALL_METHODS:  # Require the identical case-level denominator disclosure from every online method record.
                path = _method_dir(root, case_id, budget, method) / "records.json"  # Resolve the common raw records artifact.
                try:  # Convert only unavailable evidence into the explicit incomplete collection.
                    payload = _read_json(path)  # Parse exact posthoc provenance after raw grid validation.
                    reference = payload.get("reference_b") if isinstance(payload, Mapping) else None  # Read the sole posthoc denominator receipt.
                    if not isinstance(reference, Mapping) or reference.get("used_online") is not False or reference.get("usage") != "posthoc_only":  # Require explicit truth isolation on every method.
                        raise ValueError("records reference_b lacks posthoc-only provenance")  # Reject a leaked or ambiguous denominator receipt.
                    qualified = reference.get("qualification") is True  # Preserve original A/B threshold qualification independently from operational authorization.
                    status = str(reference.get("status", ""))  # Read the immutable qualified or non-qualified terminal status.
                    authorization = reference.get("authorization")  # Read the explicit user authorization only for operational non-qualified B.
                    if qualified and status != "complete":  # Require the explicit qualified terminal status beside every threshold-qualified denominator.
                        raise ValueError("qualified reference has an incompatible status")  # Reject contradictory threshold evidence.
                    amendment = reference.get("execution_amendment")  # Read exact strict-null or user-authorized effective schedule evidence.
                    amendment_sha256 = reference.get("reference_execution_amendment_sha256")  # Read the frozen authorization-artifact identity copied into the posthoc receipt.
                    if not qualified and (status != "complete_unqualified" or not authorization or not isinstance(amendment, Mapping) or type(start_expedited_levels) is not int or reference.get("expedited_levels") != start_expedited_levels or amendment.get("expedited_levels") != start_expedited_levels or not isinstance(start_amendment_sha256, str) or len(start_amendment_sha256) != 64 or amendment_sha256 != start_amendment_sha256):  # Require every non-qualified denominator to reproduce the start-disclosed frozen schedule and authorization bytes.
                        raise ValueError("non-qualified reference lacks operational authorization evidence")  # Never reinterpret failed 0.5% agreement as convergence.
                    rows.append({"case_id": case_id, "equation_budget": int(budget), "method": method, "qualification": qualified, "status": status, "authorization": authorization, "expedited_reference_levels": reference.get("expedited_levels"), "reference_execution_amendment_sha256": amendment_sha256, "reference_b_sha256": str(reference.get("reference_b_sha256", "")), "path": str(path)})  # Preserve complete method-level denominator, authorization identity, and amended-schedule provenance for audit.
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exception:  # Retain missing or malformed per-job evidence transparently.
                    missing.append(f"{case_id}/B{budget}/{method}:{type(exception).__name__}:{str(exception)[:300]}")  # Bound the repair list without hiding coordinates.
    if missing and not allow_incomplete:  # Refuse a complete aggregate without every denominator receipt.
        raise IncompleteEvidenceError(f"reference qualification evidence incomplete for {len(missing)} job(s)")  # Prevent selective qualification reporting.
    qualified = bool(rows) and not missing and all(bool(row["qualification"]) for row in rows)  # Compute the campaign-level original threshold result without authorization substitution.
    unqualified_case_ids = sorted({str(row["case_id"]) for row in rows if not bool(row["qualification"])})  # Collapse repeated job receipts to the affected blind identities.
    if unqualified_case_ids and (not allow_unqualified_references or not start_allows_unqualified):  # Require both pre-solve runner disclosure and explicit analyzer acknowledgement.
        raise IncompleteEvidenceError("campaign uses non-qualified Reference B; pass --allow-unqualified-references only for an already disclosed frozen waiver")  # Default to strict qualification while permitting the user-authorized nonblocking path.
    started_qualified = started.get("REFERENCE_QUALIFIED")  # Read the prominent campaign-level pre-solve threshold result.
    if not missing and rows and started_qualified is not qualified:  # Cross-check the start marker against all method-level posthoc receipts.
        raise IncompleteEvidenceError("TEST_STARTED REFERENCE_QUALIFIED disagrees with raw records")  # Treat a disclosure mismatch as campaign integrity failure.
    return {"available": not missing and bool(rows), "REFERENCE_QUALIFIED": qualified, "allow_unqualified_references": start_allows_unqualified, "analyzer_acknowledged_unqualified": bool(allow_unqualified_references), "unqualified_case_ids": unqualified_case_ids, "job_receipt_count": len(rows), "expected_job_receipt_count": len(case_ids) * len(BUDGETS) * len(ALL_METHODS), "missing_or_invalid": missing, "rows": rows}  # Return exact threshold, authorization, coverage, and source evidence.

def build_failure_matrix(root: Path, raw_rows: Sequence[Mapping[str, Any]], rl_rows: Sequence[Mapping[str, Any]], missing_jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:  # Build one transparent success/failure row per raw and pointwise-RL operating point.
    cache: dict[tuple[str, int, str], dict[str, Any]] = {}  # Cache one job status and native failure payload across its four true prefixes.
    output: list[dict[str, Any]] = []  # Collect raw policy and aggregate RL-median failure evidence.
    for row_value in raw_rows:  # Preserve every successfully loaded raw trajectory prefix.
        row = dict(row_value)  # Copy the source row before adding failure-matrix fields.
        key = (str(row["case_id"]), int(row["equation_budget"]), str(row["method"]))  # Identify the exact independent trajectory.
        if key not in cache:  # Load its completion marker at most once.
            status_path = _method_dir(root, key[0], key[1], key[2]) / "status.json"  # Resolve the atomic per-job completion evidence.
            try:  # Retain malformed or absent status as transparent artifact failure.
                status_value = _read_json(status_path)  # Parse the complete status document.
                cache[key] = dict(status_value) if isinstance(status_value, Mapping) else {"status_error": "status root is not an object"}  # Normalize only a named status mapping.
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exception:  # Preserve exact status-read failure.
                cache[key] = {"status_error": f"{type(exception).__name__}:{str(exception)[:500]}"}  # Store a bounded machine-readable reason.
        status = cache[key]  # Reuse trajectory-level completion and native failure evidence.
        failure = status.get("failure", {}) if isinstance(status.get("failure", {}), Mapping) else {}  # Normalize the typed numerical failure mapping.
        if bool(row.get("budget_violation")):  # Give resource violations their own hard failure category.
            category = "budget_violation"  # Name the active-equation cap breach explicitly.
        elif bool(row.get("failure_affects_prefix")):  # Classify prefixes at or after a retained native failure.
            category = str(failure.get("category", "native_failure"))  # Preserve CalculiX versus Gmsh when available.
        elif not bool(row.get("energy_ok")) or not bool(row.get("qoi_ok")):  # Retain missing delivered metrics independently.
            category = "metric_unavailable"  # Avoid treating a null error as a successful zero.
        elif status.get("status_error") is not None:  # Surface missing completion-marker integrity after metric checks.
            category = "status_artifact_error"  # Name the raw evidence problem explicitly.
        else:  # Classify a fully delivered prefix as successful.
            category = "ok"  # Preserve an unambiguous success label.
        output.append({"case_id": key[0], "equation_budget": key[1], "method": key[2], "solves": int(row["solves"]), "aggregate_policy": False, "category": category, "energy_ok": bool(row.get("energy_ok")), "qoi_ok": bool(row.get("qoi_ok")), "energy_error": row.get("energy_error"), "qoi_error": row.get("qoi_error"), "budget_violation": bool(row.get("budget_violation")), "failure_affects_prefix": bool(row.get("failure_affects_prefix")), "successful_solves_available": int(row.get("successful_solves_available", 0)), "trajectory_completed": bool(status.get("completed", False)), "exception_type": failure.get("exception_type"), "failure_message": failure.get("message"), "calculix_returncode": failure.get("calculix_returncode"), "status_error": status.get("status_error")})  # Preserve metric, native, resource, and artifact evidence together.
    for row_value in rl_rows:  # Add the required pointwise three-policy median outcome explicitly.
        row = dict(row_value)  # Copy the aggregate row before classification.
        output.append({"case_id": str(row["case_id"]), "equation_budget": int(row["equation_budget"]), "method": "rl_median", "solves": int(row["solves"]), "aggregate_policy": True, "category": "ok" if bool(row.get("energy_ok")) and bool(row.get("qoi_ok")) and not bool(row.get("budget_violation")) else "aggregate_failure", "energy_ok": bool(row.get("energy_ok")), "qoi_ok": bool(row.get("qoi_ok")), "energy_error": row.get("energy_error"), "qoi_error": row.get("qoi_error"), "budget_violation": bool(row.get("budget_violation")), "seed_budget_violation_count": int(row.get("seed_budget_violation_count", 0)), "seed_energy_outcomes": row.get("seed_energy_outcomes"), "seed_qoi_outcomes": row.get("seed_qoi_outcomes")})  # Preserve all source-seed outcomes rather than only the median scalar.
    for missing in missing_jobs:  # Expand each absent or malformed raw job over every required public prefix.
        for solves in SOLVE_LIMITS:  # Make every missing delivered operating point visible in the matrix.
            output.append({"case_id": str(missing.get("case_id", "")), "equation_budget": int(missing.get("equation_budget", -1)), "method": str(missing.get("method", "")), "solves": int(solves), "aggregate_policy": False, "category": "missing_raw_job", "energy_ok": False, "qoi_ok": False, "energy_error": None, "qoi_error": None, "budget_violation": False, "failure_affects_prefix": True, "path": missing.get("path"), "error_type": missing.get("error_type"), "failure_message": missing.get("error")})  # Preserve the exact path and read/validation failure.
    return sorted(output, key=lambda row: (str(row["case_id"]), int(row["equation_budget"]), int(row["solves"]), str(row["method"])))  # Return deterministic complete failure evidence.

def _diagnostic_prediction_rows(root: Path, case_ids: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:  # Prefer content-bound full-WM diagnostic traces when all sixteen summaries expose them.
    from .four_way_ablations import aggregate_diagnostic_trace_files, load_diagnostic_trace  # Reuse exact trace schema validation and mandatory aggregate definitions.
    paths: list[Path] = []  # Collect only reused primary WM-full traces, excluding control and oracle transitions.
    for case_id in sorted(case_ids):  # Require one case-bound WM-full trace per blind case.
        summary_path = _ablation_case_path(root, case_id)  # Resolve the canonical diagnostic case summary.
        if summary_path is None:  # Fall back to primary-runtime limited diagnostics when ablations are incomplete.
            return None  # Preserve honest absence without fabricating exact trace metrics.
        payload = _read_json(summary_path)  # Parse the already schema-checked style of diagnostic summary.
        variants = payload.get("variants", {}) if isinstance(payload, Mapping) else {}  # Read the six variant receipts.
        full = variants.get("wm_full", {}) if isinstance(variants, Mapping) else {}  # Select only the content-bound primary full controller.
        trace_value = full.get("trace_path") if isinstance(full, Mapping) else None  # Read its exact persisted prediction trace.
        if not trace_value:  # Reject a case lacking transition diagnostics.
            return None  # Allow the analyzer to emit explicitly limited primary-runtime metrics instead.
        trace_path = Path(str(trace_value))  # Normalize the persisted trace path.
        if not trace_path.is_absolute():  # Resolve portable campaign-relative trace receipts.
            trace_path = root / trace_path  # Bind the path beneath the selected campaign root.
        if not trace_path.is_file():  # Reject dangling diagnostic evidence.
            return None  # Preserve honest fallback without omitting other final artifacts.
        paths.append(trace_path)  # Retain this exact case trace for content-bound aggregation.
    rows: list[dict[str, Any]] = []  # Collect every validated completed real transition for CSV delivery.
    for path in sorted(paths, key=str):  # Preserve deterministic source and transition order.
        for transition in load_diagnostic_trace(path):  # Reconstruct every exact immutable transition record.
            row = asdict(transition)  # Expand regional vectors and candidate predictions transparently.
            row["trace_path"] = str(path)  # Bind the row back to its exact persisted source.
            row["trace_sha256"] = _sha256_file(path)  # Bind source bytes independently from summary metadata.
            rows.append(row)  # Retain the complete diagnostic transition evidence.
    aggregate = aggregate_diagnostic_trace_files(paths)  # Compute log-MAE, MAPE, upper coverage, fallbacks, and honest Spearman availability.
    return rows, aggregate  # Return complete trace rows and content-bound mandatory aggregate diagnostics.

def _offline_training_seconds(root: Path) -> tuple[dict[str, float], dict[str, Any]]:  # Extract actual frozen training-and-validation wall costs under method-specific reviewed schemas.
    path = root / "training" / "training_costs.json"  # Resolve the freeze-protected combined cost source.
    payload = _read_json(path)  # Parse the complete embedded source receipts.
    if not isinstance(payload, Mapping) or payload.get("protocol_id") != PROTOCOL_ID or payload.get("schema") != "wmvla-four-way-training-costs-v1":  # Require the exact freeze aggregate contract.
        raise IncompleteEvidenceError("training/training_costs.json has an incompatible schema")  # Prevent inferred or hand-entered offline costs.
    sources = payload.get("sources", [])  # Read all three exact source payloads.
    if not isinstance(sources, list):  # Require a transparent source collection.
        raise IncompleteEvidenceError("training cost aggregate lacks sources")  # Refuse an opaque total without method provenance.
    indexed = {str(source.get("method")): source.get("payload") for source in sources if isinstance(source, Mapping)}  # Index embedded protected payloads by learned method.
    if set(indexed) != {"world_model", "supervised", "rl"} or any(not isinstance(value, Mapping) for value in indexed.values()):  # Require all and only the three learned families.
        raise IncompleteEvidenceError("training cost sources do not cover world_model, supervised, and rl")  # Preserve full offline-cost coverage.
    world = indexed["world_model"]  # Read the actual world acquisition and fit cost source.
    supervised = indexed["supervised"]  # Read expert, three-network, and validation-selection costs.
    rl = indexed["rl"]  # Read the three-seed training-and-validation aggregate.
    values = {"world_model_vla": float(world["training_wall_s"]), "supervised": float(supervised["expert_generation_wall_s"]) + float(supervised["network_training_wall_s"]) + float(supervised["validation_wall_s"]), "rl_median": float(rl["sum_seed_training_and_validation_wall_s"])}  # Apply the preregistered non-parallel consumed-time definitions exactly.
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):  # Require finite nonnegative actual wall clocks.
        raise IncompleteEvidenceError("offline training wall-clock evidence is nonfinite or negative")  # Refuse undefined amortization arithmetic.
    receipt = {"path": str(path), "sha256": _sha256_file(path), "definitions": {"world_model_vla": "world_model.training_wall_s", "supervised": "expert_generation_wall_s + network_training_wall_s + validation_wall_s", "rl_median": "sum_seed_training_and_validation_wall_s (three seeds, non-parallel consumed elapsed)"}}  # Preserve exact source bytes and field formulas.
    return values, receipt  # Return method totals and complete source provenance.

def _online_representative_seconds(root: Path, case_ids: Sequence[str]) -> tuple[dict[str, float], dict[str, Any]]:  # Compute fixed B=60000, full-K=6 representative online costs over sixteen paired cases.
    wm_values: list[float] = []  # Collect one full-trajectory WM online total per case.
    supervised_values: list[float] = []  # Collect one strict two-solve held-last supervised online total per case.
    rl_values: list[float] = []  # Collect the sum of all three policy online totals required for one RL-median scientific output per case.
    sources: list[dict[str, Any]] = []  # Preserve exact timing artifact identities used by amortization.
    for case_id in sorted(case_ids):  # Pair methods within every blind case before taking case medians.
        wm_payload = _timing_payload(root, case_id, 60000, "world_model_vla")  # Read the complete independent WM B60000 trajectory timing.
        supervised_payload = _timing_payload(root, case_id, 60000, "supervised")  # Read the complete strict two-solve supervised deployment timing.
        wm_values.append(float(wm_payload["online_total_s"]))  # Use the unified method-level full trajectory wall clock.
        supervised_values.append(float(supervised_payload["online_total_s"]))  # Use the two-solve cost even though K=6 holds its second result.
        seed_totals: list[float] = []  # Collect all three required policy trajectory costs for this case.
        for method in RL_METHODS:  # Include every frozen policy rather than the selected median policy alone.
            seed_payload = _timing_payload(root, case_id, 60000, method)  # Read this independent seed's complete B60000 trajectory timing.
            seed_totals.append(float(seed_payload["online_total_s"]))  # Retain its full K=6 online total.
            seed_path = _method_dir(root, case_id, 60000, method) / "timing.json"  # Resolve exact source provenance.
            sources.append({"case_id": case_id, "method": method, "path": str(seed_path), "sha256": _sha256_file(seed_path)})  # Bind every RL component timing to exact bytes.
        rl_values.append(float(sum(seed_totals)))  # Charge all three seed deployments needed to compute the scientific pointwise median.
        for method in ("world_model_vla", "supervised"):  # Bind both non-RL representative source files.
            source_path = _method_dir(root, case_id, 60000, method) / "timing.json"  # Resolve this method's exact timing artifact.
            sources.append({"case_id": case_id, "method": method, "path": str(source_path), "sha256": _sha256_file(source_path)})  # Preserve exact timing provenance.
    values_by_method = {"world_model_vla": wm_values, "supervised": supervised_values, "rl_median": rl_values}  # Group the three preregistered deployment-cost definitions.
    if any(len(values) != 16 or any(not math.isfinite(value) or value < 0.0 for value in values) for values in values_by_method.values()):  # Require full finite case coverage.
        raise IncompleteEvidenceError("representative online timing lacks sixteen finite nonnegative case values")  # Refuse selective or invalid medians.
    medians = {method: float(median(values)) for method, values in values_by_method.items()}  # Compute the fixed case median for each deployed scientific output.
    receipt = {"operating_point": {"equation_budget": 60000, "max_solves": 6}, "case_count": 16, "aggregation": {"world_model_vla": "median of 16 B60000, K=6-capped terminal trajectory online_total_s", "supervised": "median of 16 strict two-solve deployment online_total_s; K>2 hold-last", "rl_median": "median of 16 per-case sums of all three B60000, K=6-capped seed trajectory online_total_s"}, "per_case_seconds": values_by_method, "sources": sources}  # Preserve exact definitions, retained terminal costs, and source identities.
    return medians, receipt  # Return representative slopes and complete online provenance.

def _cost_intersection(wm_train: float, wm_online: float, competitor_train: float, competitor_online: float) -> dict[str, Any]:  # Solve and classify WM-versus-competitor amortized cost over integer deployment counts n>=1.
    intercept = float(wm_train - competitor_train)  # Compute WM-minus-competitor offline cost at n=0.
    slope = float(wm_online - competitor_online)  # Compute WM-minus-competitor marginal online cost per deployment.
    tolerance = 1.0e-12 * max(abs(wm_online), abs(competitor_online), 1.0)  # Define a scale-aware exact-parallel numerical tolerance.
    if abs(slope) <= tolerance:  # Handle parallel amortization lines without dividing by an unstable near-zero value.
        wm_always = intercept <= 0.0  # Compare identical-slope lines by their frozen offline intercepts.
        return {"status": "always" if wm_always else "never", "parallel": True, "real_intersection_n": None, "integer_domain": "n>=1", "wm_no_more_expensive_integer_range": {"min_n": 1 if wm_always else None, "max_n": None, "kind": "all" if wm_always else "none"}, "difference_formula": {"intercept_seconds": intercept, "slope_seconds_per_case": slope}}  # Report all or no integer deployments explicitly.
    crossing = float(-intercept / slope)  # Solve Tw+nOw = Tc+nOc in real deployment-count space.
    difference_at_one = intercept + slope  # Classify the first scientifically meaningful integer deployment.
    if slope < 0.0:  # WM becomes relatively cheaper as deployment count increases.
        if difference_at_one <= 0.0:  # WM is already no more expensive at n=1 and remains so thereafter.
            status = "always"  # Name full integer-domain superiority.
            integer_range = {"min_n": 1, "max_n": None, "kind": "all"}  # Preserve the minimal integer range.
        else:  # A positive first-deployment difference crosses once at a later deployment count.
            first = max(1, int(math.ceil(crossing - 1.0e-12)))  # Find the first integer at or beyond the real equality within numerical tolerance.
            status = "crosses"  # Name the finite amortization crossover.
            integer_range = {"min_n": first, "max_n": None, "kind": "lower_bounded"}  # Report the complete WM-no-more-expensive integer range.
    else:  # WM becomes relatively more expensive as deployment count increases.
        if difference_at_one > 0.0:  # WM is already more expensive at n=1 and never recovers.
            status = "never"  # Name absence of a scientific-domain WM cost advantage.
            integer_range = {"min_n": None, "max_n": None, "kind": "none"}  # Report an empty integer range explicitly.
        else:  # WM begins no more expensive but crosses to more expensive at a later count.
            last = int(math.floor(crossing + 1.0e-12))  # Find the last integer at or before real equality within numerical tolerance.
            status = "always" if last < 1 else "crosses"  # Distinguish an impossible edge from the normal bounded advantage interval.
            integer_range = {"min_n": 1, "max_n": max(last, 0), "kind": "upper_bounded"} if last >= 1 else {"min_n": None, "max_n": None, "kind": "none"}  # Report the exact initial integer range.
    return {"status": status, "parallel": False, "real_intersection_n": crossing, "integer_domain": "n>=1", "wm_no_more_expensive_integer_range": integer_range, "difference_formula": {"intercept_seconds": intercept, "slope_seconds_per_case": slope}}  # Return real and integer crossover evidence together.

def amortized_cost_report(root: Path, case_ids: Sequence[str], *, allow_unavailable: bool) -> dict[str, Any]:  # Build T_m(n)=training+n*online and WM crossovers from frozen cost evidence.
    try:  # Permit explicit unavailable output only for authorized incomplete diagnostic analysis.
        offline, offline_receipt = _offline_training_seconds(root)  # Load protected actual training and validation costs.
        online, online_receipt = _online_representative_seconds(root, case_ids)  # Compute fixed full-trajectory representative deployment slopes.
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, IncompleteEvidenceError) as exception:  # Preserve exact absence without a fabricated zero cost.
        if not allow_unavailable:  # Make a complete scientific analysis require the protected cost inputs promised by the freeze.
            raise  # Surface malformed or absent cost evidence to the caller.
        return {"schema": "wmvla-four-way-amortized-cost-v1", "protocol_id": PROTOCOL_ID, "available": False, "reason": f"{type(exception).__name__}:{str(exception)[:1000]}"}  # Emit an honest fail-closed diagnostic receipt.
    methods = {method: {"training_seconds": float(offline[method]), "representative_online_seconds_per_case": float(online[method]), "formula": f"T_{method}(n)={offline[method]}+n*{online[method]}"} for method in ("world_model_vla", "supervised", "rl_median")}  # Publish exact line intercepts, slopes, and formulas.
    crossovers = {competitor: _cost_intersection(offline["world_model_vla"], online["world_model_vla"], offline[competitor], online[competitor]) for competitor in ("supervised", "rl_median")}  # Compare WM independently with both learned baselines.
    return {"schema": "wmvla-four-way-amortized-cost-v1", "protocol_id": PROTOCOL_ID, "available": True, "deployment_count_domain": "integer n>=1", "formula": "T_m(n)=T_m,train+n*T_m,online", "methods": methods, "wm_crossovers": crossovers, "offline_receipt": offline_receipt, "online_receipt": online_receipt}  # Return complete actual cost and crossover evidence.

def _false_gate_result(schema: str, reason: str) -> dict[str, Any]:  # Construct one explicit fail-closed gate for diagnostic incomplete analysis.
    return {"schema": schema, "protocol_id": PROTOCOL_ID, "passed": False, "gates": {"complete_evidence": False}, "reason": reason}  # Never allow missing raw evidence to become a vacuous pass.

def _report_lines(final_gate: Mapping[str, Any], primary: Mapping[str, Mapping[str, Any]], dorfler: Mapping[str, Any], mechanism: Mapping[str, Any], timing: Mapping[str, Any], coverage: Mapping[str, Any], amortized: Mapping[str, Any] | None = None, reference_evidence: Mapping[str, Any] | None = None) -> list[str]:  # Render the fixed seven-line first page followed by prominent reference qualification and concrete evidence.
    names = ("DORFLER_SAFE", "BEAT_LOCAL_PREDICTION", "BEAT_SUPERVISED", "BEAT_RL", "WORLD_MODEL_MECHANISM", "ONLINE_TIME_ACCEPTABLE", "OVERALL_WIN")  # Freeze the exact required conclusion order.
    widths = {"DORFLER_SAFE": 24, "BEAT_LOCAL_PREDICTION": 24, "BEAT_SUPERVISED": 24, "BEAT_RL": 24, "WORLD_MODEL_MECHANISM": 24, "ONLINE_TIME_ACCEPTABLE": 24, "OVERALL_WIN": 24}  # Align only presentation whitespace without changing labels.
    lines = [f"{name:<{widths[name]}}= {'true' if bool(final_gate.get(name, False)) else 'false'}" for name in names]  # Place machine conclusions before any title or narrative.
    reference_qualified = bool(reference_evidence.get("REFERENCE_QUALIFIED")) if isinstance(reference_evidence, Mapping) else False  # Preserve original 0.5% threshold qualification without treating authorization as convergence.
    lines.extend(("", f"REFERENCE_QUALIFIED     = {'true' if reference_qualified else 'false'}", "", "# WMVLA-4WAY-P1 执行报告", "", "## 未通过的具体门", ""))  # Place the nonblocking but prominent reference qualification immediately after the fixed seven machine results.
    failures: list[str] = []  # Collect every false atomic gate with its parent result.
    for label, result in (("LOCAL_PREDICTION", primary.get("local_prediction", {})), ("SUPERVISED", primary.get("supervised", {})), ("RL", primary.get("rl_median", {})), ("DORFLER", dorfler), ("WORLD_MODEL_MECHANISM", mechanism), ("ONLINE_TIME", timing)):  # Inspect every primary, safety, attribution, and time conjunction.
        gates = result.get("gates", {}) if isinstance(result, Mapping) else {}  # Read named atomic gates only from mappings.
        if isinstance(gates, Mapping):  # Preserve exact machine gate vocabulary.
            failures.extend(f"- {label}.{name} = false" for name, passed in sorted(gates.items()) if not bool(passed))  # List every failed atomic requirement without qualitative substitution.
    if not bool(coverage.get("complete")):  # Surface a mandatory delivery gap even when available partial scientific component gates happen to pass.
        failures.append("- ANALYSIS.COMPLETE_EVIDENCE = false")  # Name the exact publication guard that prevents an incomplete overall-win claim.
    lines.extend(failures or ["- 无；所有列出的原子门均为 true。"] )  # State only the concrete absence of failed atomic gates when applicable.
    lines.extend(("", "## 证据覆盖", "", f"- 测试工况：{coverage.get('case_count', 0)}/16", f"- 原始前缀行：{coverage.get('raw_prefix_row_count', 0)}/{coverage.get('expected_raw_prefix_row_count', 0)}", f"- 缺失或损坏 job：{coverage.get('missing_job_count', 0)}", f"- 边界问题：{coverage.get('boundary_issue_count', 0)}", f"- 参考解原始资格：REFERENCE_QUALIFIED={'true' if reference_qualified else 'false'}", f"- 非资格工况：{reference_evidence.get('unqualified_case_ids', []) if isinstance(reference_evidence, Mapping) else []}", "", "## 统计边界", "", "- 主要胜负仅使用预注册的六个 `(K,B)` 点；完整 12 点仍保存在 `primary_results.csv`。", "- RL 为每个工况与运行点三个冻结策略的失败感知中位数。", "- `REFERENCE_QUALIFIED=false` 是显式授权的运行证据限制，不称为收敛，也不改写原 0.5% 检验。", "- `OVERALL_WIN` 不包含独立报告的在线时间门。", "", "## 摊销成本与交叉点", ""))  # Explain reference limitations, fixed scoring, and the separately reported preregistered cost section.
    if not isinstance(amortized, Mapping) or not bool(amortized.get("available")):  # Preserve honest absence for explicitly incomplete diagnostic analysis.
        lines.append(f"- 不可用：{amortized.get('reason', 'missing amortized-cost evidence') if isinstance(amortized, Mapping) else 'missing amortized-cost evidence'}")  # Report the bounded evidence reason without inventing a zero cost.
    else:  # Render exact frozen intercepts, representative slopes, and WM crossover classifications.
        methods = amortized.get("methods", {})  # Read the three learned-method line definitions.
        for method in ("world_model_vla", "supervised", "rl_median"):  # Preserve the fixed scientific-output order.
            values = methods.get(method, {}) if isinstance(methods, Mapping) else {}  # Normalize a method receipt for concise report rendering.
            lines.append(f"- `{method}`: T(n)={values.get('training_seconds')} + n×{values.get('representative_online_seconds_per_case')} 秒")  # Print exact offline and representative online costs.
        crossovers = amortized.get("wm_crossovers", {})  # Read WM-versus-competitor crossover receipts.
        for competitor in ("supervised", "rl_median"):  # Report both learned comparator crossovers independently.
            crossing = crossovers.get(competitor, {}) if isinstance(crossovers, Mapping) else {}  # Normalize one crossover result.
            integer_range = crossing.get("wm_no_more_expensive_integer_range", {}) if isinstance(crossing, Mapping) else {}  # Read the complete integer-domain result.
            lines.append(f"- WM vs `{competitor}`: status={crossing.get('status')}, real_n={crossing.get('real_intersection_n')}, integer_range={integer_range}")  # Expose real intersection and minimal integer range without qualitative substitution.
    lines.append("")  # End the fixed report with one terminal section separator.
    return lines  # Return deterministic report content ready for atomic publication.

def analyze_four_way(root: Path | str, manifest_path: Path | str, *, allow_incomplete: bool = False, allow_unqualified_references: bool = False) -> dict[str, Any]:  # Orchestrate deterministic aggregation with explicit non-qualified-reference acknowledgement.
    campaign = Path(root).resolve()  # Normalize the selected campaign root for every raw and output path.
    manifest = Path(manifest_path).resolve()  # Normalize the exact authenticated manifest input.
    case_ids, boundary_issues = _analysis_boundary(campaign, manifest, allow_incomplete)  # Establish exact blind identities and one-shot campaign validity.
    raw_rows, missing_jobs = load_raw_prefix_rows(campaign, case_ids, allow_incomplete=allow_incomplete)  # Load the complete raw prefix grid or retain explicit diagnostic gaps.
    reference_evidence = _reference_qualification_evidence(campaign, case_ids, allow_unqualified_references=allow_unqualified_references, allow_incomplete=allow_incomplete)  # Audit posthoc-only receipts and enforce the explicit operational waiver boundary.
    expected_raw = len(case_ids) * len(BUDGETS) * len(ALL_METHODS) * len(SOLVE_LIMITS)  # Compute exact primary raw coverage independently.
    raw_complete = not boundary_issues and not missing_jobs and len(raw_rows) == expected_raw  # Separate primary raw-grid readiness from mandatory diagnostic and cost delivery.
    coverage = {"schema": "wmvla-four-way-analysis-coverage-v1", "protocol_id": PROTOCOL_ID, "case_count": len(case_ids), "expected_case_count": 16, "raw_prefix_row_count": len(raw_rows), "expected_raw_prefix_row_count": expected_raw, "missing_job_count": len(missing_jobs), "missing_jobs": list(missing_jobs), "boundary_issue_count": len(boundary_issues), "boundary_issues": boundary_issues, "allow_incomplete": bool(allow_incomplete), "raw_complete": raw_complete, "complete": False}  # Preserve primary coverage while reserving final completeness for all mandatory evidence families.
    rl_rows: list[dict[str, Any]] = []  # Reserve pointwise RL aggregates for both complete and fail-closed paths.
    primary_rows: list[dict[str, Any]] = [dict(row) for row in raw_rows]  # Preserve any valid raw evidence even when the aggregate grid is incomplete.
    pairwise_rows: list[dict[str, Any]] = []  # Reserve transparent paired ratio output.
    if raw_complete:  # Evaluate all fixed statistics only on the exact complete validated primary grid.
        rl_rows = build_rl_median_rows(raw_rows, case_ids)  # Compute three-seed pointwise failure-aware medians.
        primary_rows = build_primary_table(raw_rows, rl_rows, case_ids)  # Assemble all five reported methods over all twelve public points.
        primary_reports, pairwise_rows = evaluate_primary_gates(primary_rows, case_ids)  # Evaluate all three seven-term primary conjunctions.
        dorfler = evaluate_dorfler_gate(campaign, primary_rows, case_ids)  # Evaluate v2 structural and independent empirical Dörfler safety.
        timing = evaluate_time_gate(campaign, case_ids)  # Evaluate the separate fixed online engineering-efficiency gate.
        mechanism = evaluate_mechanism_gate(campaign, case_ids)  # Evaluate mandatory attribution controls fail-closed when diagnostics are absent.
    else:  # Produce explicit false gates for authorized incomplete diagnostic analysis.
        reason = "campaign boundary or raw prefix grid is incomplete"  # Name the sole fail-closed condition without implying scientific comparison.
        primary_reports = {name: _false_gate_result("wmvla-four-way-primary-v1", reason) for name in PRIMARY_COMPETITORS}  # Fail every primary competitor gate visibly.
        dorfler = _false_gate_result("wmvla-four-way-dorfler-safety-v1", reason)  # Fail safety without structural or empirical completeness.
        timing = _false_gate_result("wmvla-four-way-online-time-v1", reason)  # Fail timing without sixteen paired observations.
        mechanism = _false_gate_result("wmvla-four-way-mechanism-v1", reason)  # Fail attribution without complete mandatory diagnostics.
    for row in primary_rows:  # Mark every delivered metric row with the same prominent original Reference-B qualification result.
        row["REFERENCE_QUALIFIED"] = bool(reference_evidence.get("REFERENCE_QUALIFIED"))  # Never let an authorized non-qualified denominator appear converged in the primary table.
        row["reference_unqualified_authorized"] = bool(not reference_evidence.get("REFERENCE_QUALIFIED") and reference_evidence.get("allow_unqualified_references"))  # Distinguish operational authorization from threshold qualification.
    for row in pairwise_rows:  # Carry the identical qualification limitation into every paired metric ratio.
        row["REFERENCE_QUALIFIED"] = bool(reference_evidence.get("REFERENCE_QUALIFIED"))  # Make denominator status visible beside each statistical input.
    diagnostic_prediction = _diagnostic_prediction_rows(campaign, case_ids) if not boundary_issues else None  # Prefer exact content-bound primary traces only for a valid campaign boundary.
    if diagnostic_prediction is not None:  # Deliver full protocol prediction metrics when diagnostic traces are complete.
        prediction_rows, prediction_aggregate = diagnostic_prediction  # Read complete transition rows and exact aggregate definitions.
        prediction_source = "content_bound_wm_full_diagnostic_traces"  # Identify the authoritative diagnostic source.
    elif raw_complete:  # Fall back to limited fields already emitted by the frozen primary runtime only in explicit incomplete-diagnostic mode.
        prediction_rows, prediction_aggregate = prediction_calibration_rows(campaign, case_ids)  # Compute only measurable upper-bound and resource diagnostics honestly.
        prediction_source = "primary_runtime_limited_fields"  # Disclose unavailable mean and true candidate-ranking quantities.
    else:  # Emit an explicit empty calibration in incomplete diagnostic mode.
        prediction_rows = []  # Avoid reading absent action logs after an invalid campaign boundary.
        prediction_aggregate = {"schema": "wmvla-four-way-prediction-calibration-v1", "protocol_id": PROTOCOL_ID, "realized_transition_count": 0, "complete": False, "reason": "campaign evidence incomplete"}  # Preserve honest non-measurability.
        prediction_source = "unavailable_incomplete_campaign"  # Name the absence explicitly.
    amortized = amortized_cost_report(campaign, case_ids, allow_unavailable=bool(allow_incomplete))  # Require frozen actual costs by default and preserve explicit absence only in authorized diagnostic mode.
    mechanism_complete = int(mechanism.get("case_count", 0)) == 16 and len(mechanism.get("source_paths", [])) == 16 and not mechanism.get("missing_or_invalid_cases")  # Require all six ablation variants to be validated for all blind cases independent of whether their mechanism gates pass.
    prediction_complete = diagnostic_prediction is not None  # Require all sixteen content-bound WM-full traces instead of accepting the limited runtime fallback as a complete delivery.
    amortized_complete = bool(amortized.get("available"))  # Require actual protected training and representative online cost evidence.
    reference_evidence_complete = bool(reference_evidence.get("available"))  # Require every raw trajectory to disclose the identical posthoc denominator status.
    coverage.update({"mandatory_ablations_complete": mechanism_complete, "prediction_diagnostics_complete": prediction_complete, "amortized_cost_complete": amortized_complete, "reference_evidence_complete": reference_evidence_complete, "REFERENCE_QUALIFIED": bool(reference_evidence.get("REFERENCE_QUALIFIED")), "reference_unqualified_case_count": len(reference_evidence.get("unqualified_case_ids", [])), "allow_unqualified_references": bool(reference_evidence.get("allow_unqualified_references")), "complete": raw_complete and mechanism_complete and prediction_complete and amortized_complete and reference_evidence_complete})  # Bind completeness to every evidence family without making qualification itself a win threshold.
    if not bool(coverage["complete"]) and not allow_incomplete:  # Refuse a nominally scientific default aggregate when mandatory diagnostics or costs are absent.
        raise IncompleteEvidenceError("analysis lacks complete raw, mandatory-ablation, prediction-diagnostic, or amortized-cost evidence")  # Require explicit fail-closed diagnostic opt-in for partial delivery.
    final_gate = evaluate_final_gate(primary_reports, dorfler, mechanism, timing)  # Apply the sole frozen conjunction and report time separately.
    if not bool(coverage["complete"]):  # Prevent explicit incomplete mode from publishing a scientific overall-win claim from partial evidence.
        final_gate["OVERALL_WIN"] = False  # Keep partial component reports available while failing the sole headline scientific conclusion.
        final_gate["incomplete_evidence_override"] = True  # Disclose that publication completeness, not an undeclared scientific threshold, forced the headline false.
        final_gate["failed_overall_terms"] = list(final_gate.get("failed_overall_terms", [])) + (["ANALYSIS_COMPLETE"] if "ANALYSIS_COMPLETE" not in final_gate.get("failed_overall_terms", []) else [])  # Make the concrete missing publication term visible in the fixed report.
    final_gate["analysis_complete"] = bool(coverage["complete"])  # Prevent a fail-closed artifact from being mistaken for a complete scientific result.
    final_gate["REFERENCE_QUALIFIED"] = bool(reference_evidence.get("REFERENCE_QUALIFIED"))  # Prominently retain the original reference convergence outcome without changing preregistered win gates.
    final_gate["reference_qualification_is_overall_term"] = False  # Record the user-authorized nonblocking amendment explicitly.
    final_gate["coverage"] = coverage  # Bind the final decision to exact raw and boundary completeness.
    failure_rows = build_failure_matrix(campaign, raw_rows, rl_rows, missing_jobs)  # Deliver numerical, resource, metric, and artifact failures without dropping cases.
    aggregate_root = campaign / "aggregate"  # Resolve the protocol-mandated aggregate output directory.
    _write_csv(aggregate_root / "primary_results.csv", primary_rows)  # Persist the complete twelve-point five-method table or transparent partial raw table.
    _write_json(aggregate_root / "primary_results.json", {"schema": ANALYSIS_SCHEMA, "protocol_id": PROTOCOL_ID, "rows": primary_rows})  # Provide the machine-readable counterpart to the primary CSV.
    _write_csv(aggregate_root / "pairwise_ratios.csv", pairwise_rows)  # Persist every primary paired energy and QoI ratio.
    _write_json(aggregate_root / "pairwise_ratios.json", {"schema": ANALYSIS_SCHEMA, "protocol_id": PROTOCOL_ID, "rows": pairwise_rows})  # Provide the machine-readable counterpart to paired ratios.
    bootstrap = {"schema": ANALYSIS_SCHEMA, "protocol_id": PROTOCOL_ID, "primary": primary_reports, "dorfler_safety": dorfler, "world_model_mechanism": mechanism, "online_time": timing, "amortized_cost": amortized, "reference_qualification": {key: value for key, value in reference_evidence.items() if key != "rows"}, "coverage": coverage}  # Assemble every gate plus concise cost and denominator-qualification receipts.
    _write_json(aggregate_root / "bootstrap.json", bootstrap)  # Persist exact seeds, replicates, bounds, and gate thresholds from four_way_stats.
    _write_csv(aggregate_root / "prediction_calibration.csv", prediction_rows)  # Persist every measurable predicted-versus-realized transition.
    _write_json(aggregate_root / "prediction_calibration.json", {"schema": ANALYSIS_SCHEMA, "protocol_id": PROTOCOL_ID, "source": prediction_source, "aggregate": prediction_aggregate, "rows": prediction_rows})  # Preserve full calibration definitions and honest unavailable metrics.
    _write_csv(aggregate_root / "failure_matrix.csv", failure_rows)  # Persist every failure and successful public point for audit.
    _write_json(aggregate_root / "failure_matrix.json", {"schema": ANALYSIS_SCHEMA, "protocol_id": PROTOCOL_ID, "rows": failure_rows})  # Provide the complete machine-readable failure matrix.
    _write_json(aggregate_root / "amortized_cost.json", amortized)  # Persist actual frozen training costs, representative B60000/K6 online costs, and WM crossovers.
    _write_json(aggregate_root / "reference_qualification.json", {"schema": "wmvla-four-way-reference-qualification-v1", "protocol_id": PROTOCOL_ID, **reference_evidence})  # Persist every method-level denominator receipt and original qualification result.
    _write_json(aggregate_root / "coverage.json", coverage)  # Persist exact raw and campaign-boundary coverage separately.
    _write_json(aggregate_root / "final_gate.json", final_gate)  # Persist the sole fixed machine conclusion after every supporting artifact exists.
    report_path = campaign / "EXECUTION_REPORT.md"  # Resolve the fixed top-level execution report path.
    _write_text(report_path, "\n".join(_report_lines(final_gate, primary_reports, dorfler, mechanism, timing, coverage, amortized, reference_evidence)) + "\n")  # Publish seven fixed machine lines followed immediately by reference qualification and concrete evidence.
    artifact_paths = [aggregate_root / name for name in ("primary_results.csv", "primary_results.json", "pairwise_ratios.csv", "pairwise_ratios.json", "bootstrap.json", "prediction_calibration.csv", "prediction_calibration.json", "failure_matrix.csv", "failure_matrix.json", "amortized_cost.json", "reference_qualification.json", "coverage.json", "final_gate.json")] + [report_path]  # Enumerate every delivered aggregate and report artifact including cost and reference qualification.
    artifact_index = {"schema": "wmvla-four-way-analysis-artifacts-v1", "protocol_id": PROTOCOL_ID, "artifacts": [{"path": str(path.relative_to(campaign)), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size} for path in artifact_paths]}  # Bind every completed output to exact bytes.
    _write_json(aggregate_root / "artifact_index.json", artifact_index)  # Publish the non-self-referential delivery index last.
    return {"schema": ANALYSIS_SCHEMA, "protocol_id": PROTOCOL_ID, "root": str(campaign), "complete": bool(coverage["complete"]), "final_gate": final_gate, "coverage": coverage, "artifact_index": str(aggregate_root / "artifact_index.json"), "report": str(report_path)}  # Return concise CLI evidence without duplicating all rows.
