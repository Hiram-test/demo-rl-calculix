"""Render deterministic protocol figures from a sealed complete four-way aggregate."""  # Define the module's post-analysis and no-solver responsibility.
from __future__ import annotations  # Postpone annotation evaluation for supported Python runtimes.

import csv  # Parse analyzer-written tabular counterparts for semantic cross-checks.
import hashlib  # Recompute every consumed and generated artifact identity.
import json  # Load strict aggregate evidence and publish a deterministic figure index.
import math  # Validate finite measurements and compute descriptive log-space summaries.
import os  # Publish the completed figure directory atomically on the campaign filesystem.
from pathlib import Path  # Resolve campaign-contained artifacts without raw-test discovery.
import re  # Validate complete lowercase SHA-256 strings exactly.
import shutil  # Remove only a private failed temporary render directory.
from statistics import median  # Compute the explicitly labelled case median for the primary plot.
import tempfile  # Isolate all render writes until every requested format succeeds.
from typing import Any, Mapping, Sequence  # Describe strict heterogeneous aggregate contracts.

import matplotlib as mpl  # Configure a noninteractive and reproducible rendering backend.
mpl.use("Agg")  # Prevent display state or GUI availability from affecting batch rendering.
import matplotlib.pyplot as plt  # Build the three fixed scientific figure layouts.

from .four_way_analysis import AGGREGATE_METHODS, ANALYSIS_SCHEMA  # Reuse analyzer method and document identities.
from .four_way_benchmark import ALL_METHODS, BUDGETS, PROTOCOL_ID, SOLVE_LIMITS  # Reuse the exact raw methods, public twelve-point grid, and protocol identity.
from .four_way_stats import BOOTSTRAP_CONFIDENCE, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, PRIMARY_COMPETITORS, PRIMARY_OPERATING_POINTS  # Reuse the preregistered six points, competitors, and uncertainty contract.

FIGURE_SCHEMA = "wmvla-four-way-figures-v1"  # Version the deterministic render collection and its audit index.
EXPECTED_CASES = 16  # Require the full blind-case aggregate without opening the blind manifest or raw test tree.
EXPECTED_ANALYSIS_ARTIFACTS = ("aggregate/primary_results.csv", "aggregate/primary_results.json", "aggregate/pairwise_ratios.csv", "aggregate/pairwise_ratios.json", "aggregate/bootstrap.json", "aggregate/prediction_calibration.csv", "aggregate/prediction_calibration.json", "aggregate/failure_matrix.csv", "aggregate/failure_matrix.json", "aggregate/amortized_cost.json", "aggregate/coverage.json", "aggregate/final_gate.json", "EXECUTION_REPORT.md")  # Mirror the complete current analyzer delivery index exactly.
FIGURE_STEMS = ("primary_energy_errors", "pairwise_energy_ratios", "prediction_calibration")  # Freeze the small high-information figure collection.
METHOD_LABELS = {"world_model_vla": "WM-VLA", "local_prediction": "Local prediction", "supervised": "Supervised", "rl_median": "RL median", "dorfler": "Dörfler"}  # Use stable human-readable method labels without changing data identities.
METHOD_COLORS = {"world_model_vla": "#006D77", "local_prediction": "#E29578", "supervised": "#7B2CBF", "rl_median": "#2A9D8F", "dorfler": "#5F6C7B"}  # Freeze a color-blind-conscious five-method palette.
METHOD_MARKERS = {"world_model_vla": "o", "local_prediction": "s", "supervised": "^", "rl_median": "D", "dorfler": "P"}  # Distinguish methods when figures are printed without color.
COMPETITOR_LABELS = {"local_prediction": "WM / Local prediction", "supervised": "WM / Supervised", "rl_median": "WM / RL median"}  # Name every ratio denominator explicitly.
RC_PARAMS = {"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 10.0, "axes.labelsize": 9.0, "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6, "legend.fontsize": 8.0, "lines.linewidth": 1.5, "savefig.dpi": 180.0, "svg.hashsalt": "WMVLA-4WAY-P1-figures-v1"}  # Pin presentation choices and SVG identifier hashing.


class FigureEvidenceError(ValueError):  # Distinguish invalid sealed inputs from rendering-library failures.
    """Report aggregate evidence that cannot support the frozen figures."""  # Explain the fail-closed exception boundary.


def _reject_json_constant(value: str) -> None:  # Reject JSON extensions such as NaN and Infinity at parse time.
    raise FigureEvidenceError(f"nonstandard JSON constant {value!r}")  # Prevent nonfinite evidence from reaching any plot.


def _validate_json_numbers(value: Any, context: str) -> None:  # Recursively reject nonfinite numeric values while permitting explicit optional nulls.
    if isinstance(value, Mapping):  # Traverse every named field in deterministic source evidence.
        for key, item in value.items():  # Inspect all mapping values without selecting favorable fields.
            _validate_json_numbers(item, f"{context}.{key}")  # Preserve a precise diagnostic path for malformed evidence.
        return  # Stop after validating the mapping recursively.
    if isinstance(value, list):  # Traverse every row and nested vector in source order.
        for index, item in enumerate(value):  # Validate every sequence item rather than a sampled subset.
            _validate_json_numbers(item, f"{context}[{index}]")  # Preserve the exact failing sequence position.
        return  # Stop after validating the sequence recursively.
    if isinstance(value, float) and not math.isfinite(value):  # Detect parser-produced or programmatic nonfinite floats.
        raise FigureEvidenceError(f"{context} is nonfinite")  # Refuse invalid scientific numbers before output creation.


def _read_json(path: Path) -> Any:  # Load one strict UTF-8 JSON artifact with finite-number validation.
    try:  # Convert filesystem, encoding, and syntax failures into the public evidence error.
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)  # Parse standards-compliant JSON only.
    except FigureEvidenceError:  # Preserve already contextualized constant failures unchanged.
        raise  # Surface the direct evidence error to the caller.
    except (OSError, UnicodeError, json.JSONDecodeError) as exception:  # Bound all ordinary artifact-read failures.
        raise FigureEvidenceError(f"cannot read strict JSON {path}: {type(exception).__name__}: {exception}") from exception  # Identify the exact malformed artifact.
    _validate_json_numbers(payload, str(path))  # Reject any nonfinite values regardless of nesting depth.
    return payload  # Return only strict finite-or-explicit-null evidence.


def _mapping(value: Any, context: str) -> Mapping[str, Any]:  # Require a named object at one schema boundary.
    if not isinstance(value, Mapping):  # Reject scalar and list substitutions.
        raise FigureEvidenceError(f"{context} must be an object")  # Report the exact incompatible boundary.
    return value  # Return the narrowed mapping.


def _list(value: Any, context: str) -> list[Any]:  # Require a concrete JSON array at one schema boundary.
    if not isinstance(value, list):  # Reject mappings and scalar substitutions.
        raise FigureEvidenceError(f"{context} must be an array")  # Report the exact incompatible boundary.
    return value  # Return the narrowed list.


def _finite(value: Any, context: str, *, minimum: float | None = None, strict: bool = False) -> float:  # Normalize one required finite plotted scalar.
    if isinstance(value, bool) or not isinstance(value, (int, float)):  # Exclude booleans and textual coercions from scientific fields.
        raise FigureEvidenceError(f"{context} must be a finite number")  # Refuse missing, blank, or string-valued metrics.
    numeric = float(value)  # Normalize supported JSON integer and floating-point numbers.
    if not math.isfinite(numeric):  # Reject NaN and infinities independently from JSON parsing.
        raise FigureEvidenceError(f"{context} must be finite")  # Prevent invalid axes or summaries.
    if minimum is not None and (numeric <= minimum if strict else numeric < minimum):  # Apply the field's physical lower bound.
        relation = ">" if strict else ">="  # Describe the precise accepted interval.
        raise FigureEvidenceError(f"{context} must be {relation} {minimum}")  # Refuse physically invalid plotted evidence.
    return numeric  # Return the validated built-in float.


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:  # Require one exact JSON integer identity or cardinality.
    if isinstance(value, bool) or not isinstance(value, int):  # Reject booleans, floats, and textual integer coercions.
        raise FigureEvidenceError(f"{context} must be an integer")  # Preserve schema-level type identity.
    if minimum is not None and value < minimum:  # Apply an optional nonnegative or positive lower bound.
        raise FigureEvidenceError(f"{context} must be at least {minimum}")  # Reject impossible cardinalities.
    return int(value)  # Return a built-in integer explicitly.


def _boolean(value: Any, context: str) -> bool:  # Require an exact JSON boolean instead of truthiness.
    if not isinstance(value, bool):  # Reject integers, strings, and missing values.
        raise FigureEvidenceError(f"{context} must be a boolean")  # Preserve analyzer semantics exactly.
    return bool(value)  # Return the validated boolean.


def _sha256_file(path: Path) -> str:  # Hash every byte of one consumed or generated artifact.
    digest = hashlib.sha256()  # Initialize the protocol's collision-resistant digest.
    with path.open("rb") as handle:  # Stream large tables and raster figures without loading them whole.
        for block in iter(lambda: handle.read(1024 * 1024), b""):  # Read bounded blocks until exact EOF.
            digest.update(block)  # Incorporate every byte in order.
    return digest.hexdigest()  # Return the full lowercase 64-hex identity.


def _validate_protocol(payload: Mapping[str, Any], schema: str, context: str) -> None:  # Bind one aggregate document to the exact protocol and schema.
    if payload.get("schema") != schema or payload.get("protocol_id") != PROTOCOL_ID:  # Reject legacy, foreign, or unlabeled artifacts.
        raise FigureEvidenceError(f"{context} has incompatible schema or protocol_id")  # Stop before mixing campaigns.


def _campaign_path(campaign: Path, relative: str) -> Path:  # Resolve one declared path while preventing traversal outside the campaign.
    candidate = campaign / relative  # Join only the analyzer-declared portable relative path.
    try:  # Convert traversal into an explicit evidence error.
        candidate.resolve().relative_to(campaign)  # Require the final target to remain under the selected campaign root.
    except ValueError as exception:  # Catch an escaped path or symlink target.
        raise FigureEvidenceError(f"artifact path escapes campaign root: {relative}") from exception  # Refuse external or blind-root redirection.
    return candidate  # Return the campaign-contained path.


def _verify_analysis_index(campaign: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:  # Recompute the complete analyzer delivery index before reading plot data.
    index_path = campaign / "aggregate" / "artifact_index.json"  # Resolve only the sealed aggregate index, never the raw test tree.
    if index_path.is_symlink():  # Reject an index redirected to another campaign.
        raise FigureEvidenceError("aggregate/artifact_index.json must not be a symlink")  # Preserve an unambiguous campaign boundary.
    payload = _mapping(_read_json(index_path), "aggregate/artifact_index.json")  # Load the strict index document.
    _validate_protocol(payload, "wmvla-four-way-analysis-artifacts-v1", "aggregate/artifact_index.json")  # Require the exact analyzer index family.
    artifacts = _list(payload.get("artifacts"), "aggregate artifact index artifacts")  # Read every declared analyzer output.
    declared: dict[str, dict[str, Any]] = {}  # Index verified entries by portable campaign-relative path.
    for position, raw_entry in enumerate(artifacts):  # Verify every declaration without silently ignoring extras.
        entry = _mapping(raw_entry, f"artifact_index.artifacts[{position}]")  # Require a named path, hash, and size object.
        relative = entry.get("path")  # Read the declared portable path.
        digest = entry.get("sha256")  # Read the declared exact-byte identity.
        size = entry.get("size_bytes")  # Read the declared byte cardinality.
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():  # Reject missing, empty, or absolute paths.
            raise FigureEvidenceError(f"artifact index entry {position} has an invalid path")  # Prevent ambiguous source resolution.
        if relative in declared:  # Reject duplicate paths that could shadow unfavorable bytes.
            raise FigureEvidenceError(f"artifact index duplicates {relative}")  # Preserve one-to-one content binding.
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:  # Require the complete canonical SHA-256 representation.
            raise FigureEvidenceError(f"artifact index has an invalid SHA-256 for {relative}")  # Reject truncated or uppercase identities.
        expected_size = _integer(size, f"artifact index size for {relative}", minimum=0)  # Require a nonnegative exact byte count.
        path = _campaign_path(campaign, relative)  # Resolve the declared file beneath the campaign.
        if path.is_symlink() or not path.is_file():  # Reject missing, directory, and direct symlink substitutions.
            raise FigureEvidenceError(f"declared artifact is not a regular file: {relative}")  # Stop before reading incomplete output.
        observed = _sha256_file(path)  # Recompute the full digest from current bytes.
        if observed != digest:  # Compare exact lowercase identities without normalization.
            raise FigureEvidenceError(f"artifact SHA-256 mismatch for {relative}")  # Refuse altered or cross-run evidence.
        if path.stat().st_size != expected_size:  # Independently verify exact byte cardinality after content identity.
            raise FigureEvidenceError(f"artifact size mismatch for {relative}")  # Refuse internally inconsistent analyzer receipts.
        declared[relative] = {"path": relative, "sha256": observed, "size_bytes": expected_size}  # Retain the verified portable receipt.
    if set(declared) != set(EXPECTED_ANALYSIS_ARTIFACTS):  # Require every current analyzer deliverable exactly once.
        missing = sorted(set(EXPECTED_ANALYSIS_ARTIFACTS) - set(declared))  # Identify absent mandatory artifacts deterministically.
        extra = sorted(set(declared) - set(EXPECTED_ANALYSIS_ARTIFACTS))  # Identify undeclared schema extensions explicitly.
        raise FigureEvidenceError(f"aggregate artifact set mismatch; missing={missing}, extra={extra}")  # Refuse partial or incompatible aggregate families.
    for relative in EXPECTED_ANALYSIS_ARTIFACTS:  # Validate strict JSON syntax throughout the sealed aggregate family.
        if relative.endswith(".json"):  # Restrict structural parsing to JSON while hashes cover CSV and Markdown bytes.
            _read_json(_campaign_path(campaign, relative))  # Reject nonfinite JSON even in non-plotted supporting evidence.
    index_receipt = {"path": "aggregate/artifact_index.json", "sha256": _sha256_file(index_path), "size_bytes": index_path.stat().st_size}  # Bind the otherwise non-self-referential analyzer index itself.
    return declared, index_receipt  # Return complete verified inputs and the index's own receipt.


def _read_csv(path: Path, context: str) -> list[dict[str, str]]:  # Load one mandatory UTF-8 CSV with a unique explicit header.
    try:  # Convert encoding, filesystem, and CSV failures into the public evidence error.
        with path.open("r", encoding="utf-8", newline="") as handle:  # Preserve standards-compliant analyzer newline handling.
            reader = csv.DictReader(handle)  # Decode named rows without positional assumptions.
            fields = reader.fieldnames  # Read the mandatory header once.
            if not fields or any(not field for field in fields) or len(fields) != len(set(fields)):  # Require nonempty unique column names.
                raise FigureEvidenceError(f"{context} has an invalid CSV header")  # Refuse ambiguous field selection.
            rows = [dict(row) for row in reader]  # Retain every row exactly once.
    except FigureEvidenceError:  # Preserve already contextualized header failures.
        raise  # Surface the direct evidence error unchanged.
    except (OSError, UnicodeError, csv.Error) as exception:  # Catch ordinary tabular read failures.
        raise FigureEvidenceError(f"cannot read CSV {context}: {type(exception).__name__}: {exception}") from exception  # Identify the failing aggregate table.
    if not rows:  # Reject empty tables even if a header exists.
        raise FigureEvidenceError(f"{context} has no data rows")  # Prevent vacuous figures.
    if any(None in row for row in rows):  # Detect rows with more cells than the declared header.
        raise FigureEvidenceError(f"{context} contains an over-wide row")  # Reject malformed tabular evidence.
    return rows  # Return the complete decoded table.


def _csv_int(value: str | None, context: str) -> int:  # Parse one required integer CSV identity without accepting floating notation.
    if value is None or re.fullmatch(r"-?[0-9]+", value) is None:  # Require a canonical decimal integer cell.
        raise FigureEvidenceError(f"{context} must be an integer CSV cell")  # Reject blanks and silent coercions.
    return int(value)  # Return the exact parsed integer.


def _csv_float(value: str | None, context: str, *, minimum: float | None = None, strict: bool = False) -> float:  # Parse one required finite CSV metric.
    if value is None or not value.strip():  # Reject missing and blank plotted cells explicitly.
        raise FigureEvidenceError(f"{context} is missing")  # Preserve fail-closed missing-data behavior.
    try:  # Convert only syntactically numeric cells.
        numeric = float(value)  # Parse analyzer-written decimal or scientific notation.
    except ValueError as exception:  # Catch textual substitutions.
        raise FigureEvidenceError(f"{context} is not numeric") from exception  # Identify the invalid cell.
    return _finite(numeric, context, minimum=minimum, strict=strict)  # Apply finite and physical-bound validation.


def _csv_bool(value: str | None, context: str) -> bool:  # Parse the analyzer's lowercase JSON-compatible boolean cells.
    if value not in ("true", "false"):  # Reject blanks, case changes, and numeric truthiness.
        raise FigureEvidenceError(f"{context} must be true or false")  # Preserve exact boolean semantics.
    return value == "true"  # Return the validated boolean.


def _validate_coverage(payload: Mapping[str, Any]) -> tuple[int, int]:  # Require an exact complete sixteen-case analyzer boundary.
    _validate_protocol(payload, "wmvla-four-way-analysis-coverage-v1", "aggregate/coverage.json")  # Bind coverage to the current protocol.
    case_count = _integer(payload.get("case_count"), "coverage.case_count", minimum=0)  # Read observed blind-case cardinality.
    expected_cases = _integer(payload.get("expected_case_count"), "coverage.expected_case_count", minimum=0)  # Read the frozen required cardinality.
    raw_count = _integer(payload.get("raw_prefix_row_count"), "coverage.raw_prefix_row_count", minimum=0)  # Read observed raw grid size.
    expected_raw = _integer(payload.get("expected_raw_prefix_row_count"), "coverage.expected_raw_prefix_row_count", minimum=0)  # Read required raw grid size.
    frozen_raw = EXPECTED_CASES * len(BUDGETS) * len(SOLVE_LIMITS) * len(ALL_METHODS)  # Compute the protocol's exact raw prefix grid independently from the receipt.
    missing_count = _integer(payload.get("missing_job_count"), "coverage.missing_job_count", minimum=0)  # Read missing or malformed job count.
    boundary_count = _integer(payload.get("boundary_issue_count"), "coverage.boundary_issue_count", minimum=0)  # Read campaign-integrity issue count.
    if case_count != EXPECTED_CASES or expected_cases != EXPECTED_CASES or raw_count != expected_raw or expected_raw != frozen_raw or missing_count != 0 or boundary_count != 0:  # Require full observed and independently computed frozen coverage.
        raise FigureEvidenceError("coverage.json does not describe a complete sixteen-case raw grid")  # Refuse partial or boundary-invalid campaigns.
    if _boolean(payload.get("allow_incomplete"), "coverage.allow_incomplete") or not _boolean(payload.get("complete"), "coverage.complete"):  # Exclude diagnostic partial-analysis mode.
        raise FigureEvidenceError("figures require a complete non-diagnostic aggregate")  # Prevent fail-closed artifacts from looking scientific.
    if _list(payload.get("missing_jobs"), "coverage.missing_jobs") or _list(payload.get("boundary_issues"), "coverage.boundary_issues"):  # Require empty detailed issue collections too.
        raise FigureEvidenceError("coverage detail arrays are nonempty")  # Detect inconsistent summary counts.
    return case_count, raw_count  # Return validated cardinalities for the output index.


def _validate_primary(payload: Mapping[str, Any], csv_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, Any]], tuple[str, ...]]:  # Validate the complete five-method twelve-point table and CSV counterpart.
    _validate_protocol(payload, ANALYSIS_SCHEMA, "aggregate/primary_results.json")  # Bind the primary table to the analyzer family.
    raw_rows = _list(payload.get("rows"), "primary_results.rows")  # Read every JSON primary row.
    expected_count = EXPECTED_CASES * len(BUDGETS) * len(SOLVE_LIMITS) * len(AGGREGATE_METHODS)  # Compute the exact five-method public grid size.
    if len(raw_rows) != expected_count or len(csv_rows) != expected_count:  # Require both JSON and CSV to contain 960 rows.
        raise FigureEvidenceError(f"primary_results cardinality must be {expected_count} in JSON and CSV")  # Reject omissions, duplicates, or divergent counterparts.
    rows: list[dict[str, Any]] = []  # Normalize only fully finite complete primary observations.
    json_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}  # Enforce unique case-method-point identities.
    for position, raw in enumerate(raw_rows):  # Validate every primary row without filtering failures.
        row = _mapping(raw, f"primary_results.rows[{position}]")  # Require one named observation object.
        case_id = row.get("case_id")  # Read the opaque case identity only from aggregate output.
        method = row.get("method")  # Read the exact aggregate method label.
        if not isinstance(case_id, str) or not case_id or method not in AGGREGATE_METHODS:  # Reject missing cases and unauthorized methods.
            raise FigureEvidenceError(f"primary row {position} has invalid case_id or method")  # Identify the malformed observation.
        solves = _integer(row.get("solves"), f"primary row {position} solves")  # Read the public real-solve prefix.
        budget = _integer(row.get("equation_budget"), f"primary row {position} equation_budget")  # Read the public equation cap.
        energy = _finite(row.get("energy_error"), f"primary row {position} energy_error", minimum=0.0)  # Require a finite nonnegative delivered energy error.
        qoi = _finite(row.get("qoi_error"), f"primary row {position} qoi_error", minimum=0.0)  # Require the independent finite nonnegative QoI guard value.
        if not _boolean(row.get("energy_ok"), f"primary row {position} energy_ok") or not _boolean(row.get("qoi_ok"), f"primary row {position} qoi_ok"):  # Reject unavailable delivered metrics even if a stale number remains.
            raise FigureEvidenceError(f"primary row {position} marks a required metric unavailable")  # Preserve fail-closed complete-data plotting.
        budget_violation = _boolean(row.get("budget_violation"), f"primary row {position} budget_violation")  # Preserve resource status for audit metadata.
        primary = _boolean(row.get("primary_operating_point"), f"primary row {position} primary_operating_point")  # Read preregistered point membership.
        if (solves not in SOLVE_LIMITS or budget not in BUDGETS or primary != ((solves, budget) in PRIMARY_OPERATING_POINTS)):  # Require exact point coordinates and membership.
            raise FigureEvidenceError(f"primary row {position} has an invalid public operating point")  # Reject post-hoc point labels.
        key = (case_id, str(method), solves, budget)  # Construct the unique primary coordinate.
        if key in json_by_key:  # Detect duplicate observations before plotting.
            raise FigureEvidenceError(f"primary_results duplicates {key}")  # Prevent accidental weighting changes.
        normalized = {"case_id": case_id, "method": str(method), "solves": solves, "equation_budget": budget, "energy_error": energy, "qoi_error": qoi, "budget_violation": budget_violation, "primary_operating_point": primary}  # Retain only validated plotted and audit fields.
        json_by_key[key] = normalized  # Register one unique JSON observation.
        rows.append(normalized)  # Preserve all finite observations for descriptive summaries.
    case_ids = tuple(sorted({row["case_id"] for row in rows}))  # Recover opaque cases only from the aggregate table.
    if len(case_ids) != EXPECTED_CASES:  # Require sixteen unique aggregate case identities.
        raise FigureEvidenceError("primary_results must contain exactly sixteen unique cases")  # Refuse duplicated or incomplete case sets.
    expected_keys = {(case_id, method, solves, budget) for case_id in case_ids for method in AGGREGATE_METHODS for budget in BUDGETS for solves in SOLVE_LIMITS}  # Construct the exact complete Cartesian product.
    if set(json_by_key) != expected_keys:  # Detect any missing or extra coordinate independently from row count.
        raise FigureEvidenceError("primary_results does not cover the exact case-method-K-B grid")  # Refuse selectively absent points.
    csv_by_key: dict[tuple[str, str, int, int], tuple[float, float, bool]] = {}  # Cross-check the independently delivered CSV counterpart.
    for position, row in enumerate(csv_rows):  # Validate every CSV primary observation.
        case_id = row.get("case_id")  # Read the opaque CSV case identity.
        method = row.get("method")  # Read the exact CSV method identity.
        if not case_id or method not in AGGREGATE_METHODS:  # Reject missing or unauthorized coordinates.
            raise FigureEvidenceError(f"primary_results.csv row {position} has invalid identity")  # Identify the malformed CSV row.
        solves = _csv_int(row.get("solves"), f"primary CSV row {position} solves")  # Parse the exact solve prefix.
        budget = _csv_int(row.get("equation_budget"), f"primary CSV row {position} budget")  # Parse the exact equation cap.
        key = (case_id, str(method), solves, budget)  # Construct the JSON-compatible coordinate.
        if key in csv_by_key:  # Reject duplicate CSV rows independently.
            raise FigureEvidenceError(f"primary_results.csv duplicates {key}")  # Preserve one-to-one counterparts.
        csv_by_key[key] = (_csv_float(row.get("energy_error"), f"primary CSV row {position} energy_error", minimum=0.0), _csv_float(row.get("qoi_error"), f"primary CSV row {position} qoi_error", minimum=0.0), _csv_bool(row.get("primary_operating_point"), f"primary CSV row {position} primary flag"))  # Parse every plotted counterpart without blanks.
    if set(csv_by_key) != expected_keys:  # Require the same exact coordinate grid in CSV.
        raise FigureEvidenceError("primary_results CSV and JSON coordinates differ")  # Refuse stale counterpart tables.
    for key, values in csv_by_key.items():  # Compare all plotted values across the two delivered encodings.
        source = json_by_key[key]  # Read the matching validated JSON observation.
        if values != (source["energy_error"], source["qoi_error"], source["primary_operating_point"]):  # Require exact numeric and flag agreement.
            raise FigureEvidenceError(f"primary_results CSV and JSON values differ at {key}")  # Stop on cross-format divergence.
    return rows, case_ids  # Return the complete finite grid and canonical aggregate case order.


def _validate_pairwise(payload: Mapping[str, Any], csv_rows: Sequence[Mapping[str, str]], case_ids: Sequence[str]) -> list[dict[str, Any]]:  # Validate all three complete six-point paired-ratio distributions.
    _validate_protocol(payload, ANALYSIS_SCHEMA, "aggregate/pairwise_ratios.json")  # Bind ratios to the analyzer family.
    raw_rows = _list(payload.get("rows"), "pairwise_ratios.rows")  # Read every JSON ratio observation.
    expected_count = EXPECTED_CASES * len(PRIMARY_OPERATING_POINTS) * len(PRIMARY_COMPETITORS)  # Compute the exact 288-row primary comparison grid.
    if len(raw_rows) != expected_count or len(csv_rows) != expected_count:  # Require complete JSON and CSV counterparts.
        raise FigureEvidenceError(f"pairwise_ratios cardinality must be {expected_count} in JSON and CSV")  # Reject selective comparisons.
    rows: list[dict[str, Any]] = []  # Normalize complete finite ratio evidence.
    json_by_key: dict[tuple[str, str, int, int], tuple[float, float]] = {}  # Enforce unique comparator coordinates.
    for position, raw in enumerate(raw_rows):  # Validate every paired observation.
        row = _mapping(raw, f"pairwise_ratios.rows[{position}]")  # Require a named ratio object.
        if row.get("protocol_id") != PROTOCOL_ID:  # Bind each long-form row to the protocol too.
            raise FigureEvidenceError(f"pairwise row {position} has a foreign protocol_id")  # Refuse mixed reports.
        case_id = row.get("case_id")  # Read the opaque case identity.
        competitor = row.get("competitor")  # Read the exact denominator identity.
        if case_id not in case_ids or competitor not in PRIMARY_COMPETITORS:  # Restrict to the validated case set and three competitors.
            raise FigureEvidenceError(f"pairwise row {position} has invalid identity")  # Identify the malformed ratio.
        solves = _integer(row.get("solves"), f"pairwise row {position} solves")  # Read the preregistered solve prefix.
        budget = _integer(row.get("equation_budget"), f"pairwise row {position} budget")  # Read the preregistered equation cap.
        if (solves, budget) not in PRIMARY_OPERATING_POINTS:  # Exclude all six exploratory public points.
            raise FigureEvidenceError(f"pairwise row {position} is not a primary operating point")  # Preserve preregistration.
        energy = _finite(row.get("energy_ratio"), f"pairwise row {position} energy_ratio", minimum=0.0, strict=True)  # Require a positive finite failure-aware energy ratio.
        qoi = _finite(row.get("qoi_ratio"), f"pairwise row {position} qoi_ratio", minimum=0.0, strict=True)  # Require a positive finite failure-aware QoI ratio.
        if not isinstance(row.get("energy_status"), str) or not row.get("energy_status") or not isinstance(row.get("qoi_status"), str) or not row.get("qoi_status"):  # Require transparent ratio classifications.
            raise FigureEvidenceError(f"pairwise row {position} lacks ratio status")  # Refuse opaque penalized ratios.
        _boolean(row.get("wm_budget_violation"), f"pairwise row {position} budget flag")  # Validate the resource status even though it is not plotted.
        _boolean(row.get("wm_proactive_action"), f"pairwise row {position} proactive flag")  # Validate mechanism coverage metadata.
        key = (str(case_id), str(competitor), solves, budget)  # Construct the unique paired coordinate.
        if key in json_by_key:  # Reject duplicate case-point comparisons.
            raise FigureEvidenceError(f"pairwise_ratios duplicates {key}")  # Prevent implicit reweighting.
        json_by_key[key] = (energy, qoi)  # Register exact JSON ratios for CSV comparison.
        rows.append({"case_id": str(case_id), "competitor": str(competitor), "solves": solves, "equation_budget": budget, "energy_ratio": energy, "qoi_ratio": qoi})  # Retain the plotted finite fields.
    expected_keys = {(case_id, competitor, solves, budget) for case_id in case_ids for competitor in PRIMARY_COMPETITORS for solves, budget in PRIMARY_OPERATING_POINTS}  # Construct the complete paired Cartesian product.
    if set(json_by_key) != expected_keys:  # Detect any omitted or unauthorized coordinate.
        raise FigureEvidenceError("pairwise_ratios does not cover the exact case-competitor primary grid")  # Refuse incomplete distributions.
    csv_by_key: dict[tuple[str, str, int, int], tuple[float, float]] = {}  # Cross-check all CSV paired values independently.
    for position, row in enumerate(csv_rows):  # Validate every CSV ratio row.
        case_id = row.get("case_id")  # Read the CSV case identity.
        competitor = row.get("competitor")  # Read the CSV denominator identity.
        if case_id not in case_ids or competitor not in PRIMARY_COMPETITORS:  # Restrict to validated identities.
            raise FigureEvidenceError(f"pairwise_ratios.csv row {position} has invalid identity")  # Identify the malformed counterpart.
        key = (str(case_id), str(competitor), _csv_int(row.get("solves"), f"pairwise CSV row {position} solves"), _csv_int(row.get("equation_budget"), f"pairwise CSV row {position} budget"))  # Construct the exact public coordinate.
        if key in csv_by_key:  # Reject duplicated CSV evidence.
            raise FigureEvidenceError(f"pairwise_ratios.csv duplicates {key}")  # Preserve one counterpart per JSON row.
        csv_by_key[key] = (_csv_float(row.get("energy_ratio"), f"pairwise CSV row {position} energy_ratio", minimum=0.0, strict=True), _csv_float(row.get("qoi_ratio"), f"pairwise CSV row {position} qoi_ratio", minimum=0.0, strict=True))  # Parse both required finite ratios.
    if csv_by_key != json_by_key:  # Require exact coordinates and numeric values across encodings.
        raise FigureEvidenceError("pairwise_ratios CSV and JSON differ")  # Stop before plotting an ambiguous source.
    return rows  # Return all 288 validated energy and QoI ratios.


def _geometric_mean(values: Sequence[float]) -> float:  # Compute a finite positive descriptive mean without external state.
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):  # Require positive finite ratios before logarithms.
        raise FigureEvidenceError("geometric mean requires positive finite values")  # Refuse malformed paired distributions.
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))  # Apply equal row weighting exactly.


def _validate_bootstrap(payload: Mapping[str, Any], ratio_rows: Sequence[Mapping[str, Any]], coverage: Mapping[str, Any]) -> dict[str, dict[str, float]]:  # Validate preregistered point estimates and case-bootstrap intervals.
    _validate_protocol(payload, ANALYSIS_SCHEMA, "aggregate/bootstrap.json")  # Bind the uncertainty report to the analyzer family.
    if payload.get("coverage") != coverage:  # Require identical complete coverage embedded in the bootstrap report.
        raise FigureEvidenceError("bootstrap coverage differs from coverage.json")  # Refuse cross-run supporting evidence.
    primary = _mapping(payload.get("primary"), "bootstrap.primary")  # Read the three competitor reports.
    if set(primary) != set(PRIMARY_COMPETITORS):  # Require all and only preregistered denominators.
        raise FigureEvidenceError("bootstrap.primary has an incompatible competitor set")  # Prevent best-comparator selection.
    output: dict[str, dict[str, float]] = {}  # Normalize the exact energy point estimate and interval for plotting.
    for competitor in PRIMARY_COMPETITORS:  # Validate each comparison independently.
        report = _mapping(primary.get(competitor), f"bootstrap.primary.{competitor}")  # Read one primary gate report.
        _validate_protocol(report, "wmvla-four-way-primary-v1", f"bootstrap primary {competitor}")  # Require the exact primary schema.
        if report.get("competitor") != competitor or _integer(report.get("case_count"), f"{competitor} case_count") != EXPECTED_CASES or _integer(report.get("operating_point_count"), f"{competitor} operating_point_count") != len(PRIMARY_OPERATING_POINTS) or _integer(report.get("observation_count"), f"{competitor} observation_count") != EXPECTED_CASES * len(PRIMARY_OPERATING_POINTS):  # Require exact comparison identity and cardinality.
            raise FigureEvidenceError(f"bootstrap primary report is incomplete for {competitor}")  # Refuse partial uncertainty intervals.
        energy = _mapping(report.get("energy"), f"bootstrap primary {competitor} energy")  # Read the registered energy statistic.
        estimate = _finite(energy.get("geometric_mean_ratio"), f"{competitor} geometric_mean_ratio", minimum=0.0, strict=True)  # Require a positive finite point estimate.
        interval = _mapping(energy.get("bootstrap_ci_95"), f"{competitor} bootstrap_ci_95")  # Read case-cluster interval metadata.
        point = _finite(interval.get("point_estimate"), f"{competitor} bootstrap point", minimum=0.0, strict=True)  # Require the unresampled estimate.
        lower = _finite(interval.get("lower"), f"{competitor} bootstrap lower", minimum=0.0, strict=True)  # Require the finite lower limit.
        upper = _finite(interval.get("upper"), f"{competitor} bootstrap upper", minimum=0.0, strict=True)  # Require the finite upper limit.
        confidence = _finite(interval.get("confidence"), f"{competitor} bootstrap confidence", minimum=0.0, strict=True)  # Read the frozen confidence level.
        replicates = _integer(interval.get("replicates"), f"{competitor} bootstrap replicates", minimum=1)  # Read the frozen resample count.
        seed = _integer(interval.get("seed"), f"{competitor} bootstrap seed")  # Read the frozen bootstrap seed.
        if confidence != BOOTSTRAP_CONFIDENCE or replicates != BOOTSTRAP_REPLICATES or seed != BOOTSTRAP_SEED or interval.get("resampling_unit") != "case":  # Require the preregistered case-bootstrap contract.
            raise FigureEvidenceError(f"bootstrap metadata differs from the frozen contract for {competitor}")  # Reject post-hoc uncertainty settings.
        if lower > upper or not math.isclose(point, estimate, rel_tol=1.0e-12, abs_tol=1.0e-15):  # Require ordered percentile limits and the same unresampled point estimate without assuming a percentile CI must contain it.
            raise FigureEvidenceError(f"bootstrap interval is inconsistent for {competitor}")  # Refuse malformed interval graphics.
        observed = _geometric_mean([float(row["energy_ratio"]) for row in ratio_rows if row["competitor"] == competitor])  # Recompute the all-point geometric mean from sealed long-form rows.
        if not math.isclose(observed, estimate, rel_tol=1.0e-12, abs_tol=1.0e-15):  # Cross-check the reported point statistic against source ratios.
            raise FigureEvidenceError(f"bootstrap estimate does not match pairwise_ratios for {competitor}")  # Stop on cross-artifact divergence.
        output[competitor] = {"estimate": estimate, "lower": lower, "upper": upper}  # Retain only exact plotted interval values.
    return output  # Return all three validated energy-ratio intervals.


def _validate_calibration(payload: Mapping[str, Any], csv_rows: Sequence[Mapping[str, str]], case_ids: Sequence[str]) -> dict[str, Any]:  # Validate a complete full-trace or limited-runtime prediction calibration table.
    _validate_protocol(payload, ANALYSIS_SCHEMA, "aggregate/prediction_calibration.json")  # Bind calibration to the analyzer family.
    source = payload.get("source")  # Read the analyzer's explicit diagnostic provenance mode.
    raw_rows = _list(payload.get("rows"), "prediction_calibration.rows")  # Read every completed transition observation.
    aggregate = _mapping(payload.get("aggregate"), "prediction_calibration.aggregate")  # Read aggregate calibration metrics.
    if not raw_rows or len(raw_rows) != len(csv_rows):  # Require a nonempty one-to-one CSV and JSON transition table.
        raise FigureEvidenceError("prediction calibration requires nonempty equal-cardinality JSON and CSV rows")  # Refuse unavailable or truncated diagnostics.
    full_mode = source == "content_bound_wm_full_diagnostic_traces"  # Detect the preferred full world-model diagnostic source.
    limited_mode = source == "primary_runtime_limited_fields"  # Detect the explicitly limited primary-runtime source.
    if not full_mode and not limited_mode:  # Reject unavailable and unknown calibration modes.
        raise FigureEvidenceError("prediction calibration source is unavailable or unsupported")  # Prevent substituting invented diagnostics.
    rows: list[dict[str, Any]] = []  # Normalize the finite fields required by the fixed two-panel calibration plot.
    keys: set[tuple[Any, ...]] = set()  # Enforce one row per declared transition identity.
    json_values: dict[tuple[Any, ...], tuple[float, float, float, float, float, float]] = {}  # Bind every plotted JSON scalar to its transition identity for CSV cross-checks.
    for position, raw in enumerate(raw_rows):  # Validate every completed transition without sampling.
        row = _mapping(raw, f"prediction_calibration.rows[{position}]")  # Require a named transition record.
        case_id = row.get("case_id")  # Read the aggregate-owned opaque case identity.
        if case_id not in case_ids:  # Require calibration to use the exact primary case set.
            raise FigureEvidenceError(f"calibration row {position} has an invalid case_id")  # Refuse cross-campaign rows.
        if full_mode:  # Validate full content-bound diagnostic fields.
            step = _integer(row.get("step"), f"calibration row {position} step", minimum=0)  # Read the real pre-action solve index.
            key = (str(case_id), step)  # Identify one WM-full transition per case and step.
            if row.get("variant") != "wm_full":  # Exclude controls and oracle transitions from primary calibration.
                raise FigureEvidenceError(f"calibration row {position} is not wm_full")  # Preserve the analyzer's stated source.
            predicted_error = _finite(row.get("predicted_total_error"), f"calibration row {position} predicted_total_error", minimum=0.0)  # Require the finite mean total-error prediction.
            actual_error = _finite(row.get("actual_total_error"), f"calibration row {position} actual_total_error", minimum=0.0)  # Require the finite realized total error.
            upper_error = _finite(row.get("predicted_total_error_upper"), f"calibration row {position} predicted_total_error_upper", minimum=0.0)  # Require the emitted one-sided error bound.
            predicted_equations = _finite(row.get("predicted_equations"), f"calibration row {position} predicted_equations", minimum=0.0, strict=True)  # Require a positive mean resource prediction.
            actual_equations = _finite(row.get("actual_equations"), f"calibration row {position} actual_equations", minimum=0.0, strict=True)  # Require a positive real CalculiX equation count.
            upper_equations = _finite(row.get("predicted_equations_upper"), f"calibration row {position} predicted_equations_upper", minimum=0.0, strict=True)  # Require the one-sided resource bound.
        else:  # Validate limited primary-runtime ratio and equation fields honestly.
            budget = _integer(row.get("equation_budget"), f"calibration row {position} equation_budget")  # Read the independent trajectory budget.
            transition = _integer(row.get("transition"), f"calibration row {position} transition", minimum=1)  # Read the one-based completed action number.
            key = (str(case_id), budget, transition)  # Identify one primary-runtime transition.
            predicted_error = _finite(row.get("predicted_error_ratio_upper"), f"calibration row {position} predicted_error_ratio_upper", minimum=0.0)  # Require the available conservative error-ratio prediction.
            actual_error = _finite(row.get("actual_error_ratio"), f"calibration row {position} actual_error_ratio", minimum=0.0)  # Require the realized error ratio.
            upper_error = predicted_error  # Record that the only available predictor is already the emitted upper bound.
            predicted_equations = _finite(row.get("predicted_equations"), f"calibration row {position} predicted_equations", minimum=0.0, strict=True)  # Require a positive preflight resource prediction.
            actual_equations = _finite(row.get("actual_equations"), f"calibration row {position} actual_equations", minimum=0.0, strict=True)  # Require a positive real resource measurement.
            upper_equations = predicted_equations  # Preserve the limited source without inventing a second resource bound.
        if key in keys:  # Reject duplicated transitions that could reweight calibration.
            raise FigureEvidenceError(f"prediction calibration duplicates transition {key}")  # Preserve one observation per identity.
        keys.add(key)  # Register the validated unique transition.
        json_values[key] = (predicted_error, actual_error, upper_error, predicted_equations, actual_equations, upper_equations)  # Preserve exact plotted JSON values for counterpart validation.
        rows.append({"case_id": str(case_id), "predicted_error": predicted_error, "actual_error": actual_error, "upper_error": upper_error, "predicted_equations": predicted_equations, "actual_equations": actual_equations, "upper_equations": upper_equations})  # Retain the exact plotted means, realizations, and bounds.
    if {row["case_id"] for row in rows} != set(case_ids):  # Require at least one completed real transition from every blind case.
        raise FigureEvidenceError("prediction calibration does not cover all sixteen cases")  # Refuse a selectively available calibration plot.
    csv_values: dict[tuple[Any, ...], tuple[float, float, float, float, float, float]] = {}  # Cross-check transition identities and every plotted scalar in the analyzer-written CSV.
    for position, row in enumerate(csv_rows):  # Validate every CSV transition identity.
        case_id = row.get("case_id")  # Read the CSV case identity.
        if case_id not in case_ids:  # Restrict to the validated primary case set.
            raise FigureEvidenceError(f"prediction_calibration.csv row {position} has invalid case_id")  # Identify cross-campaign evidence.
        key = (str(case_id), _csv_int(row.get("step"), f"calibration CSV row {position} step")) if full_mode else (str(case_id), _csv_int(row.get("equation_budget"), f"calibration CSV row {position} budget"), _csv_int(row.get("transition"), f"calibration CSV row {position} transition"))  # Reconstruct the mode-specific unique coordinate.
        if key in csv_values:  # Reject duplicate CSV identities.
            raise FigureEvidenceError(f"prediction_calibration.csv duplicates {key}")  # Preserve one-to-one counterparts.
        if full_mode:  # Parse all six full diagnostic fields used by the calibration figure.
            plotted = (_csv_float(row.get("predicted_total_error"), f"calibration CSV row {position} predicted_total_error", minimum=0.0), _csv_float(row.get("actual_total_error"), f"calibration CSV row {position} actual_total_error", minimum=0.0), _csv_float(row.get("predicted_total_error_upper"), f"calibration CSV row {position} predicted_total_error_upper", minimum=0.0), _csv_float(row.get("predicted_equations"), f"calibration CSV row {position} predicted_equations", minimum=0.0, strict=True), _csv_float(row.get("actual_equations"), f"calibration CSV row {position} actual_equations", minimum=0.0, strict=True), _csv_float(row.get("predicted_equations_upper"), f"calibration CSV row {position} predicted_equations_upper", minimum=0.0, strict=True))  # Preserve exact full predictor, realization, and bound values.
        else:  # Parse all four limited fields and preserve their declared upper-bound semantics.
            csv_predicted_error = _csv_float(row.get("predicted_error_ratio_upper"), f"calibration CSV row {position} predicted_error_ratio_upper", minimum=0.0)  # Parse the available conservative error-ratio prediction.
            csv_predicted_equations = _csv_float(row.get("predicted_equations"), f"calibration CSV row {position} predicted_equations", minimum=0.0, strict=True)  # Parse the available resource prediction.
            plotted = (csv_predicted_error, _csv_float(row.get("actual_error_ratio"), f"calibration CSV row {position} actual_error_ratio", minimum=0.0), csv_predicted_error, csv_predicted_equations, _csv_float(row.get("actual_equations"), f"calibration CSV row {position} actual_equations", minimum=0.0, strict=True), csv_predicted_equations)  # Preserve the limited source without inventing separate mean or bound fields.
        csv_values[key] = plotted  # Register every exact plotted CSV scalar under its unique transition.
    if csv_values != json_values:  # Require JSON and CSV to describe identical transitions and plotted values.
        raise FigureEvidenceError("prediction calibration CSV and JSON differ")  # Refuse stale counterpart tables or scalar divergence.
    if full_mode:  # Validate mandatory full-diagnostic aggregate quantities used in figure annotation.
        _validate_protocol(aggregate, "wmvla-four-way-prediction-aggregate-v1", "prediction calibration aggregate")  # Require the full diagnostic aggregate schema.
        if _integer(aggregate.get("transition_count"), "calibration aggregate transition_count", minimum=1) != len(rows) or _integer(aggregate.get("case_count"), "calibration aggregate case_count", minimum=1) != EXPECTED_CASES:  # Require exact transition and case coverage.
            raise FigureEvidenceError("full calibration aggregate cardinality differs from rows")  # Refuse partial summary metrics.
        error_metric = _finite(aggregate.get("total_error_log_mae"), "calibration total_error_log_mae", minimum=0.0)  # Read the mandatory global error log-MAE.
        equation_metric = _finite(aggregate.get("equation_mape"), "calibration equation_mape", minimum=0.0)  # Read the mandatory resource MAPE.
        intervals = _mapping(aggregate.get("prediction_interval_coverage"), "calibration prediction_interval_coverage")  # Read one-sided coverage metrics.
        error_coverage = _finite(intervals.get("total_error_upper"), "calibration total_error_upper coverage", minimum=0.0)  # Read error-bound coverage.
        equation_coverage = _finite(intervals.get("equations_upper"), "calibration equations_upper coverage", minimum=0.0)  # Read resource-bound coverage.
        observed_error_metric = float(sum(abs(math.log(max(row["predicted_error"], 1.0e-300)) - math.log(max(row["actual_error"], 1.0e-300))) for row in rows) / len(rows))  # Recompute the protocol's total-error log-MAE from exact plotted transitions.
        observed_equation_metric = float(sum(abs(row["predicted_equations"] - row["actual_equations"]) / max(row["actual_equations"], 1.0) for row in rows) / len(rows))  # Recompute active-equation MAPE from exact plotted transitions.
        observed_error_coverage = float(sum(row["actual_error"] <= row["upper_error"] for row in rows) / len(rows))  # Recompute emitted global-error upper-bound coverage.
        observed_equation_coverage = float(sum(row["actual_equations"] <= row["upper_equations"] for row in rows) / len(rows))  # Recompute emitted resource upper-bound coverage.
    else:  # Validate all limited-runtime metrics needed for honest annotation.
        _validate_protocol(aggregate, "wmvla-four-way-prediction-calibration-v1", "limited prediction calibration aggregate")  # Require the limited diagnostic schema.
        if _integer(aggregate.get("realized_transition_count"), "limited calibration transition count", minimum=1) != len(rows):  # Require exact row cardinality.
            raise FigureEvidenceError("limited calibration aggregate cardinality differs from rows")  # Refuse partial summary metrics.
        error_metric = _finite(aggregate.get("available_error_bound_log_mae"), "limited calibration error-bound log-MAE", minimum=0.0)  # Require the honestly available bound discrepancy.
        equation_metric = _finite(aggregate.get("equation_mape"), "limited calibration equation_mape", minimum=0.0)  # Require resource prediction error.
        error_coverage = _finite(aggregate.get("one_sided_error_bound_coverage"), "limited calibration error coverage", minimum=0.0)  # Require available one-sided coverage.
        equation_coverage = float(sum(row["actual_equations"] <= row["upper_equations"] for row in rows) / len(rows))  # Derive exact limited resource coverage from the same sealed rows.
        observed_error_metric = float(sum(abs(math.log(max(row["actual_error"], 1.0e-300) / max(row["predicted_error"], 1.0e-300))) for row in rows) / len(rows))  # Recompute available conservative-bound log discrepancy exactly.
        observed_equation_metric = float(sum(abs(row["predicted_equations"] - row["actual_equations"]) / max(row["actual_equations"], 1.0) for row in rows) / len(rows))  # Recompute limited active-equation MAPE exactly.
        observed_error_coverage = float(sum(row["actual_error"] <= row["upper_error"] for row in rows) / len(rows))  # Recompute limited one-sided error coverage exactly.
        observed_equation_coverage = equation_coverage  # Preserve the row-derived limited resource coverage as its own consistency target.
    if error_coverage > 1.0 or equation_coverage > 1.0:  # Require probability-scale coverage values.
        raise FigureEvidenceError("prediction coverage metrics must lie in [0,1]")  # Refuse invalid calibration annotations.
    comparisons = ((error_metric, observed_error_metric, "error metric"), (equation_metric, observed_equation_metric, "equation metric"), (error_coverage, observed_error_coverage, "error coverage"), (equation_coverage, observed_equation_coverage, "equation coverage"))  # Pair every displayed aggregate with an independently recomputed row statistic.
    if any(not math.isclose(claimed, observed, rel_tol=1.0e-12, abs_tol=1.0e-15) for claimed, observed, _name in comparisons):  # Require cross-artifact calibration consistency without display rounding.
        mismatches = [name for claimed, observed, name in comparisons if not math.isclose(claimed, observed, rel_tol=1.0e-12, abs_tol=1.0e-15)]  # Identify every inconsistent displayed quantity deterministically.
        raise FigureEvidenceError(f"prediction calibration aggregate differs from rows for {mismatches}")  # Refuse annotations unsupported by transition evidence.
    return {"mode": "full_mean_predictions" if full_mode else "limited_upper_bound_predictions", "rows": rows, "error_metric": error_metric, "equation_metric": equation_metric, "error_coverage": error_coverage, "equation_coverage": equation_coverage}  # Return complete finite plot-ready calibration evidence.


def _validate_supporting_artifacts(payloads: Mapping[str, Any], coverage: Mapping[str, Any], case_ids: Sequence[str]) -> None:  # Validate complete final-gate, cost, and failure receipts even though they are not plotted.
    final_gate = _mapping(payloads["aggregate/final_gate.json"], "aggregate/final_gate.json")  # Read the final machine conclusion.
    _validate_protocol(final_gate, "wmvla-four-way-final-gate-v1", "aggregate/final_gate.json")  # Require the exact final-gate schema.
    if not _boolean(final_gate.get("analysis_complete"), "final_gate.analysis_complete") or final_gate.get("coverage") != coverage:  # Require a complete decision bound to identical coverage.
        raise FigureEvidenceError("final_gate is not bound to the complete coverage artifact")  # Refuse partial scientific conclusions.
    for name in ("DORFLER_SAFE", "BEAT_LOCAL_PREDICTION", "BEAT_SUPERVISED", "BEAT_RL", "WORLD_MODEL_MECHANISM", "ONLINE_TIME_ACCEPTABLE", "OVERALL_WIN"):  # Validate all seven fixed report booleans without requiring success.
        _boolean(final_gate.get(name), f"final_gate.{name}")  # Preserve complete pass-or-fail reporting.
    amortized = _mapping(payloads["aggregate/amortized_cost.json"], "aggregate/amortized_cost.json")  # Read protected actual time evidence.
    _validate_protocol(amortized, "wmvla-four-way-amortized-cost-v1", "aggregate/amortized_cost.json")  # Require the current cost schema.
    if not _boolean(amortized.get("available"), "amortized_cost.available"):  # Complete analysis promises actual training and online costs.
        raise FigureEvidenceError("complete figures require available amortized cost evidence")  # Refuse a nominally complete but cost-incomplete campaign.
    methods = _mapping(amortized.get("methods"), "amortized_cost.methods")  # Read the three learned-method line definitions.
    if set(methods) != {"world_model_vla", "supervised", "rl_median"}:  # Require the frozen learned-method cost set.
        raise FigureEvidenceError("amortized cost method set is incompatible")  # Reject omitted training cost evidence.
    for method, raw_values in methods.items():  # Validate every actual intercept and representative slope.
        values = _mapping(raw_values, f"amortized_cost.methods.{method}")  # Require a named cost record.
        _finite(values.get("training_seconds"), f"{method} training_seconds", minimum=0.0)  # Require finite nonnegative actual offline cost.
        _finite(values.get("representative_online_seconds_per_case"), f"{method} online_seconds", minimum=0.0)  # Require finite nonnegative representative deployment cost.
    failures = _mapping(payloads["aggregate/failure_matrix.json"], "aggregate/failure_matrix.json")  # Read the complete failure/success matrix counterpart.
    _validate_protocol(failures, ANALYSIS_SCHEMA, "aggregate/failure_matrix.json")  # Bind failure evidence to the analyzer family.
    failure_rows = _list(failures.get("rows"), "failure_matrix.rows")  # Require a nonempty transparent grid.
    if not failure_rows or {str(_mapping(row, "failure row").get("case_id")) for row in failure_rows} != set(case_ids):  # Require all sixteen aggregate cases to appear.
        raise FigureEvidenceError("failure_matrix does not cover the complete aggregate case set")  # Refuse incomplete campaign support evidence.


def _load_figure_evidence(campaign: Path) -> dict[str, Any]:  # Verify and normalize every sealed input before creating a figure directory.
    verified, index_receipt = _verify_analysis_index(campaign)  # Authenticate all analyzer artifacts and the non-self-referential index.
    json_names = [relative for relative in EXPECTED_ANALYSIS_ARTIFACTS if relative.endswith(".json")]  # Enumerate strict JSON support files only.
    payloads = {relative: _read_json(_campaign_path(campaign, relative)) for relative in json_names}  # Load verified aggregate JSON without touching raw test evidence.
    coverage = _mapping(payloads["aggregate/coverage.json"], "aggregate/coverage.json")  # Read the sole coverage boundary document.
    case_count, raw_count = _validate_coverage(coverage)  # Require a complete non-diagnostic campaign aggregate.
    primary_csv = _read_csv(campaign / "aggregate" / "primary_results.csv", "aggregate/primary_results.csv")  # Load the exact primary table counterpart.
    primary, case_ids = _validate_primary(_mapping(payloads["aggregate/primary_results.json"], "aggregate/primary_results.json"), primary_csv)  # Validate the full finite five-method grid.
    pairwise_csv = _read_csv(campaign / "aggregate" / "pairwise_ratios.csv", "aggregate/pairwise_ratios.csv")  # Load all paired-ratio counterparts.
    pairwise = _validate_pairwise(_mapping(payloads["aggregate/pairwise_ratios.json"], "aggregate/pairwise_ratios.json"), pairwise_csv, case_ids)  # Validate the complete six-point comparison grid.
    bootstrap = _validate_bootstrap(_mapping(payloads["aggregate/bootstrap.json"], "aggregate/bootstrap.json"), pairwise, coverage)  # Validate registered uncertainty intervals against ratio rows.
    calibration_csv = _read_csv(campaign / "aggregate" / "prediction_calibration.csv", "aggregate/prediction_calibration.csv")  # Load all prediction-versus-realization counterparts.
    calibration = _validate_calibration(_mapping(payloads["aggregate/prediction_calibration.json"], "aggregate/prediction_calibration.json"), calibration_csv, case_ids)  # Require complete finite calibration evidence.
    _validate_supporting_artifacts(payloads, coverage, case_ids)  # Require complete final, cost, and failure evidence before visualization.
    inputs = [verified[relative] for relative in EXPECTED_ANALYSIS_ARTIFACTS] + [index_receipt]  # Include every analyzer output plus the analyzer index's own full hash.
    return {"campaign": campaign, "case_ids": case_ids, "case_count": case_count, "raw_prefix_row_count": raw_count, "primary": primary, "pairwise": pairwise, "bootstrap": bootstrap, "calibration": calibration, "inputs": sorted(inputs, key=lambda item: item["path"])}  # Return only complete plot-ready data and exact provenance.


def _quantile_r7(values: Sequence[float], probability: float) -> float:  # Compute the standard linear-interpolated descriptive quartile deterministically.
    ordered = sorted(float(value) for value in values)  # Remove source-row ordering from the summary.
    if not ordered:  # Reject an undefined summary explicitly.
        raise FigureEvidenceError("quantile requires at least one value")  # Prevent empty method-point displays.
    position = (len(ordered) - 1) * float(probability)  # Compute the Hyndman-Fan type-7 fractional index.
    lower = int(math.floor(position))  # Locate the lower bracketing observation.
    upper = int(math.ceil(position))  # Locate the upper bracketing observation.
    fraction = position - lower  # Compute the interpolation weight.
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)  # Return the exact interpolated quantile.


def _point_label(point: tuple[int, int]) -> str:  # Render one preregistered operating point compactly.
    solves, budget = point  # Unpack the public coordinates.
    return f"K={solves}\nB={budget // 1000}k"  # Use exact integer-thousands notation for the three public budgets.


def _set_nonnegative_scale(axis: Any, values: Sequence[float], dimension: str = "y") -> dict[str, Any]:  # Choose a transparent scale that preserves valid exact zeros.
    finite = [float(value) for value in values]  # Normalize already validated plot values.
    positives = [value for value in finite if value > 0.0]  # Identify values compatible with a logarithmic scale.
    setter = axis.set_yscale if dimension == "y" else axis.set_xscale  # Select the requested Matplotlib dimension setter.
    if positives and len(positives) == len(finite):  # Use log scale only when it preserves every observation.
        setter("log")  # Improve dynamic-range readability without modifying any value.
        return {"scale": "log", "linthresh": None}  # Record the exact display transform.
    if positives:  # Preserve a mixture of exact zeros and positive measurements.
        threshold = min(positives) / 10.0  # Derive a small linear neighborhood solely from observed positive resolution.
        setter("symlog", linthresh=threshold)  # Display zero and positive orders of magnitude without a pseudocount.
        return {"scale": "symlog", "linthresh": threshold}  # Record the derived transform completely.
    setter("linear")  # Display an all-zero valid dataset without an undefined log axis.
    return {"scale": "linear", "linthresh": None}  # Record the exact fallback transform.


def _primary_figure(evidence: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:  # Plot median and interquartile energy errors for all methods at all six primary points.
    rows = evidence["primary"]  # Read the fully validated five-method grid.
    fig, axis = plt.subplots(figsize=(9.2, 5.2))  # Create one compact landscape panel suitable for review reports.
    x_values = list(range(len(PRIMARY_OPERATING_POINTS)))  # Assign fixed equally spaced primary-point positions.
    all_values: list[float] = []  # Collect exact observations for transparent axis selection.
    summaries: dict[str, list[dict[str, float]]] = {}  # Retain plotted descriptive values for the artifact index.
    for method in AGGREGATE_METHODS:  # Plot methods in the frozen aggregate reporting order.
        medians: list[float] = []  # Collect case medians across the six primary points.
        lowers: list[float] = []  # Collect case first quartiles.
        uppers: list[float] = []  # Collect case third quartiles.
        summaries[method] = []  # Initialize complete point metadata for this method.
        for solves, budget in PRIMARY_OPERATING_POINTS:  # Summarize each preregistered point independently.
            values = [float(row["energy_error"]) for row in rows if row["method"] == method and row["solves"] == solves and row["equation_budget"] == budget]  # Select all sixteen exact case observations.
            if len(values) != EXPECTED_CASES:  # Defend the renderer independently from prior validation.
                raise FigureEvidenceError(f"primary plot lacks sixteen values for {method}/K{solves}/B{budget}")  # Stop before partial display.
            center = float(median(values))  # Compute the labelled case median.
            lower = _quantile_r7(values, 0.25)  # Compute the labelled interquartile lower edge.
            upper = _quantile_r7(values, 0.75)  # Compute the labelled interquartile upper edge.
            medians.append(center)  # Append this point's center in public-point order.
            lowers.append(lower)  # Append this point's lower quartile.
            uppers.append(upper)  # Append this point's upper quartile.
            all_values.extend(values)  # Retain every raw observation for scale selection.
            summaries[method].append({"solves": int(solves), "equation_budget": int(budget), "median": center, "q1": lower, "q3": upper, "case_count": EXPECTED_CASES})  # Record exact descriptive statistics without claiming a new gate.
        axis.fill_between(x_values, lowers, uppers, color=METHOD_COLORS[method], alpha=0.10, linewidth=0.0)  # Show the case interquartile envelope.
        axis.plot(x_values, medians, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], label=METHOD_LABELS[method], markersize=5.0)  # Connect only like-method point medians.
    scale = _set_nonnegative_scale(axis, all_values, "y")  # Preserve exact zeros while exposing useful dynamic range.
    axis.set_xticks(x_values, [_point_label(point) for point in PRIMARY_OPERATING_POINTS])  # Label all and only six preregistered points.
    axis.set_xlabel("Preregistered operating point")  # Identify the public K and B coordinate axis.
    axis.set_ylabel("Relative energy error (median and IQR across cases)")  # State both metric and descriptive aggregation.
    axis.set_title("Five-method energy error at the six primary operating points")  # Name the complete comparison without a win claim.
    axis.legend(ncol=3, frameon=False, loc="best")  # Decode all method colors and markers compactly.
    axis.grid(True, which="both", axis="y")  # Aid magnitude comparison on linear or logarithmic scales.
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.91)  # Pin layout explicitly instead of relying on environment-dependent auto-layout.
    return fig, {"statistic": "case median with R7 interquartile range", "axis": scale, "summaries": summaries}  # Return the figure and complete display metadata.


def _ratio_figure(evidence: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:  # Plot every paired energy-ratio distribution with registered global confidence intervals.
    rows = evidence["pairwise"]  # Read the complete finite paired comparison grid.
    intervals = evidence["bootstrap"]  # Read exact registered energy-ratio intervals.
    fig, axes = plt.subplots(1, len(PRIMARY_COMPETITORS), figsize=(12.6, 4.6), sharey=True)  # Create one denominator-specific panel per primary competitor.
    output: dict[str, Any] = {}  # Record exact distribution and interval display metadata.
    all_values = [float(row["energy_ratio"]) for row in rows]  # Collect all 288 paired ratios for a common scale.
    point_positions = list(range(1, len(PRIMARY_OPERATING_POINTS) + 1))  # Assign fixed boxplot positions.
    offsets = [(-0.14 + 0.28 * index / (EXPECTED_CASES - 1)) for index in range(EXPECTED_CASES)]  # Spread case dots deterministically without random jitter.
    for axis, competitor in zip(axes, PRIMARY_COMPETITORS, strict=True):  # Build each denominator panel in the frozen competitor order.
        distributions: list[list[float]] = []  # Collect sixteen-case values at each primary point.
        for solves, budget in PRIMARY_OPERATING_POINTS:  # Preserve the fixed six-point x order.
            selected = sorted((str(row["case_id"]), float(row["energy_ratio"])) for row in rows if row["competitor"] == competitor and row["solves"] == solves and row["equation_budget"] == budget)  # Tie every plotted dot to sorted opaque case identity.
            if len(selected) != EXPECTED_CASES:  # Defend the renderer from incomplete distributions.
                raise FigureEvidenceError(f"ratio plot lacks sixteen values for {competitor}/K{solves}/B{budget}")  # Stop before partial display.
            distributions.append([value for _case, value in selected])  # Retain this complete case distribution.
        boxes = axis.boxplot(distributions, positions=point_positions, widths=0.50, patch_artist=True, showfliers=False, medianprops={"color": "#1B1B1B", "linewidth": 1.2}, whiskerprops={"color": "#444444"}, capprops={"color": "#444444"})  # Draw compact quartile and whisker summaries without duplicating outliers.
        for patch in boxes["boxes"]:  # Apply one consistent comparison fill.
            patch.set(facecolor="#9BD3D0", edgecolor="#277DA1", alpha=0.55)  # Keep raw dots visually dominant.
        for position, values in zip(point_positions, distributions, strict=True):  # Overlay every exact case ratio deterministically.
            axis.scatter([position + offset for offset in offsets], values, s=7.0, color="#264653", alpha=0.48, linewidths=0.0, zorder=3)  # Expose distribution density and extremes without stochastic jitter.
        interval = intervals[competitor]  # Read the registered all-six-point case-bootstrap summary.
        axis.axhspan(interval["lower"], interval["upper"], color="#F4A261", alpha=0.16, zorder=0, label="95% case-bootstrap CI")  # Show the global registered uncertainty interval.
        axis.axhline(interval["estimate"], color="#E76F51", linestyle="--", linewidth=1.4, label="Global geometric mean")  # Mark the matching global point estimate.
        axis.axhline(1.0, color="#111111", linestyle=":", linewidth=1.0, label="Parity")  # Identify equal energy error without declaring a threshold adjustment.
        axis.set_xticks(point_positions, [_point_label(point) for point in PRIMARY_OPERATING_POINTS], rotation=0)  # Label every preregistered operating point.
        axis.set_title(COMPETITOR_LABELS[competitor])  # State the exact numerator and denominator.
        axis.set_xlabel("Operating point")  # Identify x-coordinate semantics.
        output[competitor] = {"global_geometric_mean": interval["estimate"], "bootstrap_ci_95": [interval["lower"], interval["upper"]], "case_count_per_point": EXPECTED_CASES}  # Preserve exact registered overlay values.
    scale = _set_nonnegative_scale(axes[0], all_values, "y")  # Apply one common positive ratio scale through the shared y-axis.
    axes[0].set_ylabel("Failure-aware paired energy ratio (WM / competitor)")  # State ratio direction and retained-failure semantics.
    handles, labels = axes[-1].get_legend_handles_labels()  # Read the three shared overlay meanings once.
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))  # Keep a single compact legend outside data panels.
    fig.suptitle("Paired energy-ratio distributions and preregistered case-bootstrap intervals", y=0.98, fontsize=11.0)  # Name both raw distributions and registered uncertainty.
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.84, wspace=0.12)  # Pin all panel margins explicitly.
    return fig, {"statistic": "all sixteen case ratios per point; registered all-point case-bootstrap CI", "axis": scale, "comparisons": output}  # Return the complete figure and display provenance.


def _scatter_scale(axis: Any, x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, Any]:  # Apply matched transparent axes and draw the identity line.
    combined = [float(value) for value in (*x_values, *y_values)]  # Collect already validated predictor and realization values.
    positives = [value for value in combined if value > 0.0]  # Detect whether logarithmic axes preserve every point.
    if positives and len(positives) == len(combined):  # Prefer log-log calibration for strictly positive quantities.
        low = min(positives) / 1.25  # Add deterministic multiplicative breathing room below observed values.
        high = max(positives) * 1.25  # Add deterministic multiplicative breathing room above observed values.
        axis.set_xscale("log")  # Display predictor magnitude logarithmically.
        axis.set_yscale("log")  # Match the realization transform exactly.
        scale = "log"  # Record the common transform.
    else:  # Preserve exact zeros without adding a pseudocount.
        low = 0.0  # Anchor the physical nonnegative origin.
        high = max(combined) * 1.05 if max(combined) > 0.0 else 1.0  # Derive a finite upper extent from observed values.
        scale = "linear"  # Record the common zero-preserving transform.
    axis.set_xlim(low, high)  # Apply identical predictor bounds.
    axis.set_ylim(low, high)  # Apply identical realization bounds.
    axis.plot([low, high], [low, high], color="#333333", linestyle=":", linewidth=1.0, zorder=0)  # Draw exact prediction-equals-realization parity.
    axis.set_aspect("equal", adjustable="box")  # Preserve geometric interpretation of distance from parity.
    return {"scale": scale, "limits": [low, high]}  # Return exact display-transform metadata.


def _calibration_figure(evidence: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:  # Plot prediction versus realization for error and active equations.
    calibration = evidence["calibration"]  # Read complete finite transition diagnostics.
    rows = calibration["rows"]  # Read every validated completed transition.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))  # Create matched error and resource calibration panels.
    error_x = [float(row["predicted_error"]) for row in rows]  # Collect mean or explicitly limited upper error predictions.
    error_y = [float(row["actual_error"]) for row in rows]  # Collect exact realized error values.
    error_covered = [float(row["actual_error"]) <= float(row["upper_error"]) for row in rows]  # Classify one-sided error-bound coverage from source fields.
    equation_x = [float(row["predicted_equations"]) for row in rows]  # Collect active-equation predictions.
    equation_y = [float(row["actual_equations"]) for row in rows]  # Collect real CalculiX active-equation counts.
    equation_covered = [float(row["actual_equations"]) <= float(row["upper_equations"]) for row in rows]  # Classify available resource-bound coverage.
    for axis, x_values, y_values, covered, title, x_label, y_label in ((axes[0], error_x, error_y, error_covered, "Total-error calibration", "Predicted total error" if calibration["mode"] == "full_mean_predictions" else "Predicted error-ratio upper bound", "Realized total error" if calibration["mode"] == "full_mean_predictions" else "Realized error ratio"), (axes[1], equation_x, equation_y, equation_covered, "Active-equation calibration", "Predicted active equations", "Realized active equations")):  # Configure both calibration panels under one fixed rule.
        colors = ["#2A9D8F" if flag else "#E76F51" for flag in covered]  # Encode bound coverage without changing point coordinates.
        axis.scatter(x_values, y_values, c=colors, s=22.0, alpha=0.68, edgecolors="white", linewidths=0.25)  # Show every completed transition.
        axis.set_title(title)  # Name the calibrated quantity.
        axis.set_xlabel(x_label)  # State predictor semantics exactly.
        axis.set_ylabel(y_label)  # State realization semantics exactly.
        axis.grid(True, which="both")  # Aid distance-to-parity comparison on either scale.
    error_scale = _scatter_scale(axes[0], error_x, error_y)  # Apply matched data-driven error axes and parity line.
    equation_scale = _scatter_scale(axes[1], equation_x, equation_y)  # Apply matched data-driven resource axes and parity line.
    axes[0].text(0.03, 0.97, f"log-MAE = {calibration['error_metric']:.4g}\nupper coverage = {calibration['error_coverage']:.1%}", transform=axes[0].transAxes, va="top", ha="left", bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#BBBBBB"})  # Annotate exact analyzer metrics compactly.
    axes[1].text(0.03, 0.97, f"MAPE = {calibration['equation_metric']:.1%}\nupper coverage = {calibration['equation_coverage']:.1%}", transform=axes[1].transAxes, va="top", ha="left", bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#BBBBBB"})  # Annotate exact resource metrics compactly.
    covered_handle = plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="#2A9D8F", markeredgecolor="none", label="Within emitted upper bound")  # Build a stable legend proxy for covered transitions.
    missed_handle = plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="#E76F51", markeredgecolor="none", label="Above emitted upper bound")  # Build a stable legend proxy for misses.
    fig.legend(handles=[covered_handle, missed_handle], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))  # Decode bound-coverage colors once.
    fig.suptitle(f"Prediction calibration from {len(rows)} completed real transitions", y=0.98, fontsize=11.0)  # State the exact non-synthetic observation count.
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.84, wspace=0.28)  # Pin panel geometry explicitly.
    return fig, {"mode": calibration["mode"], "transition_count": len(rows), "error_axis": error_scale, "equation_axis": equation_scale, "error_log_mae": calibration["error_metric"], "equation_mape": calibration["equation_metric"], "error_upper_coverage": calibration["error_coverage"], "equation_upper_coverage": calibration["equation_coverage"]}  # Return the figure and exact calibration display metadata.


def _save_figure(fig: Any, directory: Path, stem: str) -> list[Path]:  # Persist one figure in deterministic raster and vector forms.
    png_path = directory / f"{stem}.png"  # Resolve the fixed raster filename.
    svg_path = directory / f"{stem}.svg"  # Resolve the fixed vector filename.
    fig.savefig(png_path, format="png", dpi=180, facecolor="white", edgecolor="white", metadata={"Software": "visionamr WMVLA-4WAY-P1 figure renderer v1"})  # Write a fixed-resolution PNG without a wall-clock field.
    fig.savefig(svg_path, format="svg", facecolor="white", edgecolor="white", metadata={"Date": None, "Creator": "visionamr WMVLA-4WAY-P1 figure renderer v1"})  # Write SVG with deterministic IDs and no generated date.
    return [png_path, svg_path]  # Return both exact completed paths.


def _write_json(path: Path, payload: Any) -> None:  # Persist one strict deterministic JSON document.
    _validate_json_numbers(payload, str(path))  # Refuse nonfinite display metadata before serialization.
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Write stable key order and one terminal newline.


def _generator_receipts() -> list[dict[str, Any]]:  # Hash the exact module and CLI sources that generate the figures.
    repository = Path(__file__).resolve().parents[2]  # Locate the checked-out repository from this package module.
    paths = (repository / "visionamr" / "vla" / "four_way_figures.py", repository / "scripts" / "plot_four_way_results.py")  # Enumerate both executable source boundaries.
    receipts: list[dict[str, Any]] = []  # Collect portable full-hash source identities.
    for path in paths:  # Hash both required generator sources.
        if not path.is_file():  # Refuse an incomplete source delivery.
            raise FigureEvidenceError(f"figure generator source is missing: {path}")  # Preserve reproducibility provenance.
        receipts.append({"path": str(path.relative_to(repository)), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size})  # Bind exact checked-out bytes.
    return receipts  # Return deterministic module-then-CLI source order.


def generate_four_way_figures(campaign_root: Path | str) -> dict[str, Any]:  # Validate a sealed aggregate and atomically publish the three fixed PNG/SVG figure pairs.
    campaign = Path(campaign_root).resolve()  # Normalize the caller-selected campaign root once.
    if not campaign.is_dir():  # Require an existing campaign container without searching defaults.
        raise FigureEvidenceError(f"campaign root is not a directory: {campaign}")  # Refuse accidental creation or blind-root discovery.
    destination = campaign / "figures"  # Resolve the protocol-mandated output directory.
    if destination.exists():  # Refuse overwrite, rerender selection, or mixing with prior figures.
        raise FigureEvidenceError(f"figure output already exists: {destination}")  # Preserve one immutable render per aggregate.
    evidence = _load_figure_evidence(campaign)  # Complete all input hash, schema, cardinality, and finite-data checks before writing.
    generators = _generator_receipts()  # Bind exact generator source bytes before any render.
    temporary = Path(tempfile.mkdtemp(prefix=".figures.tmp.", dir=campaign))  # Create one private same-filesystem staging directory.
    figure_paths: list[Path] = []  # Collect all six completed image paths for hashing.
    plot_metadata: dict[str, Any] = {}  # Collect exact plot definitions and display transformations.
    try:  # Ensure a failed Matplotlib write leaves no public or private partial figure set.
        with mpl.rc_context(RC_PARAMS):  # Apply all deterministic presentation settings only within this render.
            for stem, builder in (("primary_energy_errors", _primary_figure), ("pairwise_energy_ratios", _ratio_figure), ("prediction_calibration", _calibration_figure)):  # Render the frozen small figure collection in stable order.
                fig, metadata = builder(evidence)  # Build one figure solely from validated aggregate fields.
                try:  # Close native figure resources even when one format fails.
                    figure_paths.extend(_save_figure(fig, temporary, stem))  # Persist both PNG and SVG into private staging.
                finally:  # Always release Matplotlib state before the next plot.
                    plt.close(fig)  # Prevent cross-figure artist or memory leakage.
                plot_metadata[stem] = metadata  # Retain exact statistics and transforms after successful writes.
        figure_receipts = []  # Build portable full-hash output identities.
        for path in figure_paths:  # Hash every generated PNG and SVG byte.
            canonical = f"figures/{path.name}"  # Record the final protocol-relative path rather than temporary staging.
            media_type = "image/png" if path.suffix == ".png" else "image/svg+xml"  # Declare the exact output representation.
            figure_receipts.append({"path": canonical, "media_type": media_type, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size})  # Bind complete bytes and sizes.
        if len(figure_receipts) != len(FIGURE_STEMS) * 2:  # Require both representations for every fixed figure.
            raise FigureEvidenceError("renderer did not produce all three PNG/SVG pairs")  # Refuse a partial public directory.
        index = {"schema": FIGURE_SCHEMA, "protocol_id": PROTOCOL_ID, "source_boundary": "sealed aggregate artifacts only; no manifest, raw test, solver log, model, or reference reads", "aggregate_complete": True, "case_count": evidence["case_count"], "raw_prefix_row_count": evidence["raw_prefix_row_count"], "input_artifacts": evidence["inputs"], "generator_sources": generators, "figures": sorted(figure_receipts, key=lambda item: item["path"]), "plot_definitions": plot_metadata, "determinism": {"matplotlib_backend": "Agg", "matplotlib_version": mpl.__version__, "svg_hashsalt": RC_PARAMS["svg.hashsalt"], "png_dpi": 180, "wall_clock_metadata": False, "random_jitter": False}}  # Assemble complete input, code, output, and display provenance without a timestamp.
        index_path = temporary / "artifact_index.json"  # Resolve the non-self-referential figure delivery index.
        _write_json(index_path, index)  # Persist the strict deterministic index after all figure hashes exist.
        index_sha = _sha256_file(index_path)  # Compute the full identity of the completed index itself.
        sidecar = temporary / "artifact_index.sha256"  # Resolve the conventional self-hash sidecar.
        sidecar.write_text(f"{index_sha}  artifact_index.json\n", encoding="ascii")  # Bind the index bytes without a circular JSON field.
        os.replace(temporary, destination)  # Atomically expose the complete immutable figure collection.
    except BaseException:  # Clean only the private staging directory on any validation or rendering failure.
        if temporary.exists():  # Avoid touching an already atomically published destination.
            shutil.rmtree(temporary)  # Remove incomplete private render artifacts recoverably within task scope.
        raise  # Preserve the original failure type and traceback for review.
    return {"schema": FIGURE_SCHEMA, "protocol_id": PROTOCOL_ID, "campaign_root": str(campaign), "figure_count": len(FIGURE_STEMS), "file_count": len(FIGURE_STEMS) * 2, "artifact_index": str(destination / "artifact_index.json"), "artifact_index_sha256": index_sha, "artifact_index_sidecar": str(destination / "artifact_index.sha256")}  # Return concise strict completion evidence.
