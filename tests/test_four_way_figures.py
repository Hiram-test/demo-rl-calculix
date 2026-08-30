"""Synthetic-only tests for sealed four-way figure generation and provenance."""  # Keep formal blind evidence outside every test path.
from __future__ import annotations  # Postpone annotation evaluation for supported runtimes.

import csv  # Create analyzer-shaped deterministic CSV counterparts under tmp_path.
import hashlib  # Build exact synthetic aggregate artifact indexes.
import json  # Create strict and intentionally malformed aggregate JSON evidence.
import math  # Compute internally consistent synthetic geometric and calibration summaries.
from pathlib import Path  # Address only pytest-provided temporary campaign roots.
import subprocess  # Exercise the delivered CLI in an isolated no-solver smoke test.
import sys  # Launch the CLI with the same verified Python interpreter as pytest.

import pytest  # Assert fail-closed evidence behavior and deterministic output.

from visionamr.vla.four_way_analysis import AGGREGATE_METHODS, ANALYSIS_SCHEMA  # Reuse the exact aggregate method and schema identities.
from visionamr.vla.four_way_benchmark import BUDGETS, PROTOCOL_ID, SOLVE_LIMITS  # Reuse the public grid identities.
from visionamr.vla.four_way_figures import EXPECTED_ANALYSIS_ARTIFACTS, FIGURE_SCHEMA, FigureEvidenceError, generate_four_way_figures  # Exercise the complete public figure boundary and expected analyzer delivery set.
from visionamr.vla.four_way_stats import BOOTSTRAP_CONFIDENCE, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, PRIMARY_COMPETITORS, PRIMARY_OPERATING_POINTS  # Reuse the preregistered comparison and uncertainty coordinates.


def _json_cell(value: object) -> object:  # Match the analyzer's deterministic CSV cell normalization.
    if value is None:  # Preserve explicitly unavailable optional values as blank cells.
        return ""  # Avoid inventing a textual measurement.
    if isinstance(value, bool):  # Normalize booleans independently from integers.
        return "true" if value else "false"  # Match analyzer lowercase JSON-compatible spelling.
    if isinstance(value, (dict, list, tuple)):  # Preserve nested diagnostic vectors in one stable cell.
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)  # Match compact strict analyzer serialization.
    return value  # Preserve finite scalar values directly.


def _write_json(path: Path, payload: object, *, allow_nan: bool = False) -> None:  # Write one deterministic synthetic aggregate document.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the selected tmp_path campaign subtree.
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=allow_nan) + "\n", encoding="utf-8")  # Match analyzer formatting and terminal newline.


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:  # Write one analyzer-shaped deterministic synthetic table.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the selected tmp_path aggregate directory.
    columns = sorted({str(key) for row in rows for key in row})  # Match the analyzer's stable union-column vocabulary.
    with path.open("w", encoding="utf-8", newline="") as handle:  # Apply platform-independent CSV newline handling.
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")  # Reject accidental helper-field divergence.
        writer.writeheader()  # Emit the mandatory explicit schema row.
        for row in rows:  # Preserve deterministic synthetic row order.
            writer.writerow({key: _json_cell(row.get(key)) for key in columns})  # Encode every cell exactly like the analyzer.


def _sha(path: Path) -> str:  # Hash one complete synthetic artifact exactly.
    return hashlib.sha256(path.read_bytes()).hexdigest()  # Return the full lowercase SHA-256 identity.


def _refresh_analysis_index(campaign: Path) -> None:  # Rebuild the non-self-referential analyzer artifact index after a deliberate test mutation.
    artifacts = []  # Collect all thirteen required aggregate and report receipts.
    for relative in EXPECTED_ANALYSIS_ARTIFACTS:  # Preserve the current analyzer delivery order.
        path = campaign / relative  # Resolve only the synthetic tmp_path artifact.
        artifacts.append({"path": relative, "sha256": _sha(path), "size_bytes": path.stat().st_size})  # Bind exact mutated or original bytes.
    _write_json(campaign / "aggregate" / "artifact_index.json", {"schema": "wmvla-four-way-analysis-artifacts-v1", "protocol_id": PROTOCOL_ID, "artifacts": artifacts})  # Publish the complete synthetic analyzer index.


def _geomean(values: list[float]) -> float:  # Compute an internally consistent positive synthetic ratio summary.
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))  # Apply equal observation weighting.


def _build_campaign(root: Path) -> Path:  # Build one complete analyzer-shaped campaign solely beneath tmp_path.
    campaign = root  # Retain the explicit temporary campaign identity.
    aggregate = campaign / "aggregate"  # Resolve the only data directory consumed by the renderer.
    aggregate.mkdir(parents=True, exist_ok=False)  # Prove each test starts with a brand-new synthetic aggregate.
    case_ids = tuple(f"test_case_{index:02d}" for index in range(16))  # Create sixteen opaque synthetic case identities.
    all_points = tuple((solves, budget) for budget in BUDGETS for solves in SOLVE_LIMITS)  # Build the complete twelve-point public grid.
    factors = {"world_model_vla": 0.62, "local_prediction": 1.00, "supervised": 0.86, "rl_median": 0.79, "dorfler": 0.72}  # Define deterministic non-degenerate method differences for rendering only.
    primary_rows: list[dict[str, object]] = []  # Collect the exact 960-row five-method table.
    for case_index, case_id in enumerate(case_ids):  # Populate every synthetic case without sampling.
        for solves, budget in all_points:  # Populate all twelve public operating points.
            base = 0.04 + 0.001 * case_index + 0.002 * solves + 0.00000008 * budget  # Derive a finite positive case-and-point-specific base error.
            for method in AGGREGATE_METHODS:  # Populate all five reported aggregate methods.
                energy = float(base * factors[method])  # Create one finite positive relative energy error.
                primary_rows.append({"case_id": case_id, "method": method, "solves": solves, "equation_budget": budget, "energy_error": energy, "qoi_error": float(energy * 1.25), "energy_ok": True, "qoi_ok": True, "budget_violation": False, "primary_operating_point": (solves, budget) in PRIMARY_OPERATING_POINTS})  # Preserve all fields required by the figure validator.
    primary_rows.sort(key=lambda row: (str(row["case_id"]), int(row["equation_budget"]), int(row["solves"]), AGGREGATE_METHODS.index(str(row["method"]))))  # Match the analyzer's deterministic row order.
    _write_csv(aggregate / "primary_results.csv", primary_rows)  # Publish the synthetic primary CSV counterpart.
    _write_json(aggregate / "primary_results.json", {"schema": ANALYSIS_SCHEMA, "protocol_id": PROTOCOL_ID, "rows": primary_rows})  # Publish the synthetic primary JSON source.
    primary_index = {(str(row["case_id"]), str(row["method"]), int(row["solves"]), int(row["equation_budget"])): row for row in primary_rows}  # Index exact primary errors for ratio construction.
    pairwise_rows: list[dict[str, object]] = []  # Collect the exact 288-row comparison grid.
    for case_id in case_ids:  # Populate every synthetic case.
        for competitor in PRIMARY_COMPETITORS:  # Populate all three preregistered denominators.
            for solves, budget in PRIMARY_OPERATING_POINTS:  # Populate all six preregistered operating points.
                wm = primary_index[(case_id, "world_model_vla", solves, budget)]  # Read the synthetic WM numerator.
                other = primary_index[(case_id, competitor, solves, budget)]  # Read the matching synthetic competitor denominator.
                pairwise_rows.append({"protocol_id": PROTOCOL_ID, "case_id": case_id, "competitor": competitor, "solves": solves, "equation_budget": budget, "energy_ratio": float(wm["energy_error"]) / float(other["energy_error"]), "energy_status": "success", "qoi_ratio": float(wm["qoi_error"]) / float(other["qoi_error"]), "qoi_status": "success", "wm_budget_violation": False, "wm_proactive_action": True})  # Preserve complete finite paired evidence.
    pairwise_rows.sort(key=lambda row: (str(row["case_id"]), str(row["competitor"]), int(row["equation_budget"]), int(row["solves"])))  # Match analyzer comparison ordering.
    _write_csv(aggregate / "pairwise_ratios.csv", pairwise_rows)  # Publish the synthetic ratio CSV counterpart.
    _write_json(aggregate / "pairwise_ratios.json", {"schema": ANALYSIS_SCHEMA, "protocol_id": PROTOCOL_ID, "rows": pairwise_rows})  # Publish the synthetic ratio JSON source.
    raw_prefix_count = 16 * len(BUDGETS) * 7 * len(SOLVE_LIMITS)  # Match the seven raw methods currently executed before RL-median aggregation.
    coverage = {"schema": "wmvla-four-way-analysis-coverage-v1", "protocol_id": PROTOCOL_ID, "case_count": 16, "expected_case_count": 16, "raw_prefix_row_count": raw_prefix_count, "expected_raw_prefix_row_count": raw_prefix_count, "missing_job_count": 0, "missing_jobs": [], "boundary_issue_count": 0, "boundary_issues": [], "allow_incomplete": False, "complete": True}  # Describe an exact complete synthetic analysis boundary.
    primary_reports: dict[str, object] = {}  # Collect all three internally consistent bootstrap reports.
    for competitor in PRIMARY_COMPETITORS:  # Build one report per preregistered denominator.
        ratios = [float(row["energy_ratio"]) for row in pairwise_rows if row["competitor"] == competitor]  # Read all 96 synthetic energy ratios.
        estimate = _geomean(ratios)  # Compute the exact all-point geometric mean.
        interval = {"point_estimate": estimate, "lower": estimate * 0.96, "upper": estimate * 1.04, "confidence": BOOTSTRAP_CONFIDENCE, "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "resampling_unit": "case"}  # Provide a finite ordered synthetic interval under the frozen metadata.
        primary_reports[competitor] = {"schema": "wmvla-four-way-primary-v1", "protocol_id": PROTOCOL_ID, "competitor": competitor, "case_count": 16, "operating_point_count": len(PRIMARY_OPERATING_POINTS), "observation_count": 16 * len(PRIMARY_OPERATING_POINTS), "energy": {"geometric_mean_ratio": estimate, "bootstrap_ci_95": interval}, "passed": True}  # Preserve every field consumed by the figure validator.
    bootstrap = {"schema": ANALYSIS_SCHEMA, "protocol_id": PROTOCOL_ID, "primary": primary_reports, "dorfler_safety": {}, "world_model_mechanism": {}, "online_time": {}, "amortized_cost": {}, "coverage": coverage}  # Assemble a complete analyzer-shaped uncertainty document.
    _write_json(aggregate / "bootstrap.json", bootstrap)  # Publish registered point and interval evidence.
    calibration_rows: list[dict[str, object]] = []  # Collect complete content-bound-style real transitions.
    for case_index, case_id in enumerate(case_ids):  # Ensure every synthetic blind identity contributes transitions.
        for step in (1, 2):  # Provide two deterministic real transitions per case.
            predicted_error = float(8.0 / (1.0 + case_index + step))  # Create a positive global total-error prediction.
            actual_error = float(predicted_error * (0.88 + 0.01 * ((case_index + step) % 5)))  # Create a nearby positive realization.
            predicted_equations = int(9000 + 240 * case_index + 700 * step)  # Create a positive active-equation prediction.
            actual_equations = int(predicted_equations + 40 * ((case_index % 3) - 1))  # Create a nearby positive CalculiX measurement.
            calibration_rows.append({"case_id": case_id, "variant": "wm_full", "seed": None, "step": step, "predicted_total_error": predicted_error, "actual_total_error": actual_error, "predicted_total_error_upper": predicted_error * 1.15, "predicted_equations": predicted_equations, "actual_equations": actual_equations, "predicted_equations_upper": float(predicted_equations * 1.08)})  # Preserve every field required for two-panel calibration.
    error_log_mae = float(sum(abs(math.log(float(row["predicted_total_error"])) - math.log(float(row["actual_total_error"]))) for row in calibration_rows) / len(calibration_rows))  # Compute the exact full-diagnostic global error log-MAE.
    equation_mape = float(sum(abs(float(row["predicted_equations"]) - float(row["actual_equations"])) / float(row["actual_equations"]) for row in calibration_rows) / len(calibration_rows))  # Compute the exact active-equation MAPE.
    error_coverage = float(sum(float(row["actual_total_error"]) <= float(row["predicted_total_error_upper"]) for row in calibration_rows) / len(calibration_rows))  # Compute exact one-sided error coverage.
    equation_coverage = float(sum(float(row["actual_equations"]) <= float(row["predicted_equations_upper"]) for row in calibration_rows) / len(calibration_rows))  # Compute exact one-sided resource coverage.
    calibration_aggregate = {"schema": "wmvla-four-way-prediction-aggregate-v1", "protocol_id": PROTOCOL_ID, "transition_count": len(calibration_rows), "case_count": 16, "total_error_log_mae": error_log_mae, "equation_mape": equation_mape, "prediction_interval_coverage": {"total_error_upper": error_coverage, "equations_upper": equation_coverage}}  # Assemble finite exact calibration summary metrics.
    _write_csv(aggregate / "prediction_calibration.csv", calibration_rows)  # Publish the synthetic transition CSV counterpart.
    _write_json(aggregate / "prediction_calibration.json", {"schema": ANALYSIS_SCHEMA, "protocol_id": PROTOCOL_ID, "source": "content_bound_wm_full_diagnostic_traces", "aggregate": calibration_aggregate, "rows": calibration_rows})  # Publish the content-bound-style JSON calibration source.
    failure_rows = [{"case_id": case_id, "method": "world_model_vla", "solves": 2, "equation_budget": 30000, "category": "ok"} for case_id in case_ids]  # Provide transparent case coverage without a fabricated failure.
    _write_csv(aggregate / "failure_matrix.csv", failure_rows)  # Publish the synthetic failure CSV counterpart.
    _write_json(aggregate / "failure_matrix.json", {"schema": ANALYSIS_SCHEMA, "protocol_id": PROTOCOL_ID, "rows": failure_rows})  # Publish the synthetic failure JSON source.
    methods = {method: {"training_seconds": float(100.0 + 10.0 * index), "representative_online_seconds_per_case": float(4.0 + index), "formula": "synthetic test receipt"} for index, method in enumerate(("world_model_vla", "supervised", "rl_median"))}  # Create finite actual-cost-shaped evidence for all learned methods.
    _write_json(aggregate / "amortized_cost.json", {"schema": "wmvla-four-way-amortized-cost-v1", "protocol_id": PROTOCOL_ID, "available": True, "methods": methods})  # Publish the complete synthetic cost receipt.
    _write_json(aggregate / "coverage.json", coverage)  # Publish the sole synthetic completeness boundary.
    final_gate = {"schema": "wmvla-four-way-final-gate-v1", "protocol_id": PROTOCOL_ID, "DORFLER_SAFE": True, "BEAT_LOCAL_PREDICTION": True, "BEAT_SUPERVISED": True, "BEAT_RL": True, "WORLD_MODEL_MECHANISM": True, "ONLINE_TIME_ACCEPTABLE": True, "OVERALL_WIN": True, "analysis_complete": True, "coverage": coverage}  # Provide all seven complete machine conclusions without influencing plot data.
    _write_json(aggregate / "final_gate.json", final_gate)  # Publish the complete synthetic final gate.
    (campaign / "EXECUTION_REPORT.md").write_text("DORFLER_SAFE = true\nSynthetic tmp_path report only.\n", encoding="utf-8")  # Provide the indexed report counterpart without formal evidence.
    _refresh_analysis_index(campaign)  # Seal every synthetic aggregate artifact by full hash and exact size.
    (campaign / "test").mkdir()  # Add a raw-tree sentinel proving plot generation does not require its contents.
    (campaign / "test" / "DO_NOT_READ.txt").write_text("not aggregate evidence\n", encoding="utf-8")  # Keep an irrelevant raw-like file outside all indexed inputs.
    return campaign  # Return the complete temporary campaign root.


def _figure_hashes(campaign: Path) -> dict[str, str]:  # Read generated image hashes from the completed figure index.
    index = json.loads((campaign / "figures" / "artifact_index.json").read_text(encoding="utf-8"))  # Load only the newly generated tmp_path index.
    return {str(item["path"]): str(item["sha256"]) for item in index["figures"]}  # Normalize the six exact image identities.


def test_complete_synthetic_aggregate_renders_reproducibly(tmp_path: Path) -> None:  # Verify byte-identical PNG/SVG output from identical sealed aggregate bytes.
    first = _build_campaign(tmp_path / "campaign_a")  # Build the first independent complete synthetic campaign.
    second = _build_campaign(tmp_path / "campaign_b")  # Build a byte-identical second synthetic campaign.
    result_first = generate_four_way_figures(first)  # Render the first complete aggregate without any solver or formal read.
    result_second = generate_four_way_figures(second)  # Render the second complete aggregate under the same pinned renderer.
    assert result_first["schema"] == FIGURE_SCHEMA  # Require the versioned completion schema.
    assert result_first["figure_count"] == 3 and result_first["file_count"] == 6  # Require all three PNG/SVG pairs.
    assert _figure_hashes(first) == _figure_hashes(second)  # Prove byte-level reproducibility for every image format.
    assert (first / "figures" / "artifact_index.json").read_bytes() == (second / "figures" / "artifact_index.json").read_bytes()  # Prove the full provenance index is path-independent and deterministic.
    sidecar = (first / "figures" / "artifact_index.sha256").read_text(encoding="ascii").split()[0]  # Read the complete self-hash sidecar.
    assert sidecar == _sha(first / "figures" / "artifact_index.json")  # Verify the non-circular index identity.
    index = json.loads((first / "figures" / "artifact_index.json").read_text(encoding="utf-8"))  # Inspect complete generated provenance.
    assert len(index["input_artifacts"]) == len(EXPECTED_ANALYSIS_ARTIFACTS) + 1  # Require all analyzer outputs plus its index receipt.
    assert all(len(item["sha256"]) == 64 for item in (*index["input_artifacts"], *index["generator_sources"], *index["figures"]))  # Require complete SHA-256 values throughout the index.
    assert (first / "test" / "DO_NOT_READ.txt").read_text(encoding="utf-8") == "not aggregate evidence\n"  # Confirm the irrelevant raw sentinel remains untouched.


def test_stale_aggregate_hash_fails_before_output(tmp_path: Path) -> None:  # Verify altered aggregate bytes cannot produce any public figures.
    campaign = _build_campaign(tmp_path / "stale_hash")  # Build one complete sealed synthetic campaign.
    path = campaign / "aggregate" / "primary_results.json"  # Select one indexed plotted source.
    path.write_bytes(path.read_bytes() + b" ")  # Alter exact bytes without updating the analyzer receipt.
    with pytest.raises(FigureEvidenceError, match="(?:size|SHA-256) mismatch"):  # Require an exact-byte identity failure rather than a partial render.
        generate_four_way_figures(campaign)  # Attempt figure generation from stale evidence.
    assert not (campaign / "figures").exists()  # Prove validation finishes before public output creation.


@pytest.mark.parametrize("mutation", ("missing", "nonfinite"))  # Exercise both required-field absence and nonstandard-number rejection.
def test_missing_or_nonfinite_calibration_fails_closed(tmp_path: Path, mutation: str) -> None:  # Verify no calibration gap can silently select another plot or invent a value.
    campaign = _build_campaign(tmp_path / mutation)  # Build a fresh complete sealed synthetic campaign.
    path = campaign / "aggregate" / "prediction_calibration.json"  # Select the required calibration JSON source.
    payload = json.loads(path.read_text(encoding="utf-8"))  # Load only synthetic tmp_path evidence for deliberate mutation.
    if mutation == "missing":  # Remove one required plotted realization.
        payload["rows"][0].pop("actual_total_error")  # Create an explicit missing-data condition.
        _write_json(path, payload)  # Persist valid JSON with an incomplete required row.
    else:  # Insert a nonstandard nonfinite scientific value.
        payload["rows"][0]["actual_total_error"] = math.nan  # Create an explicit NaN condition.
        _write_json(path, payload, allow_nan=True)  # Persist intentionally invalid extended JSON for parser rejection.
    _refresh_analysis_index(campaign)  # Keep hashes current so the semantic validator reaches the intended defect.
    with pytest.raises(FigureEvidenceError):  # Require either missing-field or strict-JSON evidence rejection.
        generate_four_way_figures(campaign)  # Attempt rendering without permitting a fallback or pseudovalue.
    assert not (campaign / "figures").exists()  # Prove no public partial output is created.


def test_incomplete_coverage_and_existing_output_are_rejected(tmp_path: Path) -> None:  # Verify both campaign completeness and immutable destination boundaries.
    incomplete = _build_campaign(tmp_path / "incomplete")  # Build one otherwise complete sealed synthetic campaign.
    coverage_path = incomplete / "aggregate" / "coverage.json"  # Select the sole completeness document.
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))  # Load synthetic coverage for deliberate mutation.
    coverage["complete"] = False  # Mark the analyzer result explicitly incomplete.
    _write_json(coverage_path, coverage)  # Persist the mutated coverage artifact.
    _refresh_analysis_index(incomplete)  # Update exact receipts so semantic validation detects incompleteness.
    with pytest.raises(FigureEvidenceError, match="complete non-diagnostic aggregate"):  # Require an explicit completeness-boundary failure.
        generate_four_way_figures(incomplete)  # Attempt rendering an incomplete aggregate.
    assert not (incomplete / "figures").exists()  # Prove no output is created for incomplete evidence.
    occupied = _build_campaign(tmp_path / "occupied")  # Build a separate complete synthetic campaign.
    (occupied / "figures").mkdir()  # Create a pre-existing destination that must remain immutable.
    (occupied / "figures" / "owner.txt").write_text("existing user artifact\n", encoding="utf-8")  # Place a recoverable ownership sentinel in the directory.
    with pytest.raises(FigureEvidenceError, match="already exists"):  # Require refusal before aggregate reads or writes.
        generate_four_way_figures(occupied)  # Attempt overwrite of a nonempty user-owned destination.
    assert (occupied / "figures" / "owner.txt").read_text(encoding="utf-8") == "existing user artifact\n"  # Prove the existing artifact is preserved byte-for-byte.


def test_plot_cli_smoke_uses_explicit_tmp_path_campaign(tmp_path: Path) -> None:  # Verify the delivered command publishes strict completion JSON without any default formal read.
    campaign = _build_campaign(tmp_path / "cli_campaign")  # Build one complete sealed synthetic campaign beneath tmp_path.
    repository = Path(__file__).resolve().parents[1]  # Locate the checked-out CLI independently from pytest's working directory.
    completed = subprocess.run([sys.executable, str(repository / "scripts" / "plot_four_way_results.py"), "--root", str(campaign)], cwd=repository, text=True, capture_output=True, check=False)  # Execute only the explicit synthetic campaign with no solver process.
    assert completed.returncode == 0, completed.stderr  # Require successful atomic rendering and expose structured failure output when it regresses.
    payload = json.loads(completed.stdout)  # Parse the CLI's strict machine-readable completion record.
    assert payload["schema"] == FIGURE_SCHEMA and payload["file_count"] == 6  # Require the versioned complete three-pair result.
    assert Path(payload["artifact_index"]) == campaign / "figures" / "artifact_index.json"  # Require output only beneath the selected tmp_path campaign.
