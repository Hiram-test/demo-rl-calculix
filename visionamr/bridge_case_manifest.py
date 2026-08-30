"""Build and verify the frozen four-way bridge case manifest."""  # Describe the module's single responsibility.
from __future__ import annotations  # Postpone annotation evaluation for broad interpreter compatibility.
from collections.abc import Mapping  # Import the read-only mapping contract used by manifest records.
import hashlib  # Import SHA-256 for canonical case and artifact identities.
import json  # Import deterministic JSON serialization for hashes and manifest files.
import math  # Import finite-value checks for sampled physical parameters.
from pathlib import Path  # Import portable paths for manifest persistence and verification.
from typing import Any  # Import the heterogeneous JSON value annotation.
import numpy as np  # Import the deterministic PCG64 sampler and vectorized distance calculations.
from .bridge_cases import make_box_girder_diaphragm  # Reuse the canonical topology-preserving bridge factory.

MANIFEST_SCHEMA = "visionamr-bridge-case-manifest-v1"  # Freeze the machine-readable manifest schema identifier.
PROTOCOL_ID = "WMVLA-4WAY-P1"  # Tie every generated case to the frozen four-way experiment protocol.
FAMILY_NAME = "box_girder_diaphragm"  # Freeze the only bridge family admitted by this manifest.
FACTORY_PATH = "visionamr.bridge_cases.make_box_girder_diaphragm"  # Record the exact reconstruction entry point.
FROZEN_SEED = 20260830  # Freeze the Latin-hypercube random seed required by the protocol.
CASE_COUNT = 48  # Freeze the complete number of train, validation, and blind-test cases.
MAXIMIN_CANDIDATES = 4096  # Freeze the deterministic random-candidate pool used by the maximin search.
FLOAT_DECIMALS = 12  # Quantize stored physical inputs for portable canonical hashes.
SPLIT_COUNTS = (("train", 24), ("validation", 8), ("test", 16))  # Freeze the ordered 24/8/16 data partition.
PARAMETER_SPECS = (  # Freeze parameter order, lower bound, upper bound, and engineering unit.
    ("wheel_offset_x", -140.0, 140.0, "mm"),  # Define the longitudinal wheel-offset interval.
    ("wheel_offset_y", -75.0, 75.0, "mm"),  # Define the transverse wheel-offset interval.
    ("opening_radius", 48.0, 76.0, "mm"),  # Define the circular access-opening radius interval.
    ("diaphragm_thickness", 24.0, 40.0, "mm"),  # Define the transverse diaphragm thickness interval.
    ("pressure", 2.8, 6.0, "MPa"),  # Define the wheel-patch pressure interval.
    ("support_width", 55.0, 90.0, "mm"),  # Define the longitudinal bearing-strip width interval.
)  # Complete the immutable six-dimensional parameter specification.
FACTORY_FIXED_KWARGS = {  # Freeze every topology-preserving factory argument outside the sampled six dimensions.
    "length": 600.0,  # Freeze the longitudinal segment length in millimetres.
    "width": 360.0,  # Freeze the transverse steel-box width in millimetres.
    "height": 260.0,  # Freeze the steel-box depth in millimetres.
    "top_thickness": 24.0,  # Freeze the top-plate thickness in millimetres.
    "bottom_thickness": 20.0,  # Freeze the bottom-plate thickness in millimetres.
    "web_thickness": 18.0,  # Freeze both longitudinal web thicknesses in millimetres.
    "frame_width": 18.0,  # Freeze the access-opening frame width in millimetres.
    "wheel_size": (150.0, 110.0),  # Freeze the longitudinal and transverse wheel-patch dimensions.
}  # Complete the canonical fixed factory configuration.

def _canonical_json_bytes(payload: object) -> bytes:  # Serialize a JSON-compatible payload into one canonical byte sequence.
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")  # Exclude whitespace, key-order, locale, and NaN ambiguity.

def _sha256_payload(payload: object) -> str:  # Hash a canonical JSON-compatible payload with SHA-256.
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()  # Return the complete lowercase hexadecimal digest.

def _parameter_names() -> tuple[str, ...]:  # Return the frozen parameter order used by the LHS columns.
    return tuple(spec[0] for spec in PARAMETER_SPECS)  # Preserve the protocol table order exactly.

def _full_factory_kwargs(parameters: Mapping[str, float]) -> dict[str, Any]:  # Convert one six-dimensional record into explicit factory keyword arguments.
    kwargs: dict[str, Any] = dict(FACTORY_FIXED_KWARGS)  # Start from the frozen topology-preserving geometry constants.
    kwargs["wheel_offset"] = (float(parameters["wheel_offset_x"]), float(parameters["wheel_offset_y"]))  # Recombine the two sampled wheel coordinates for the existing factory.
    kwargs["opening_radius"] = float(parameters["opening_radius"])  # Pass the sampled opening radius without changing topology.
    kwargs["diaphragm_thickness"] = float(parameters["diaphragm_thickness"])  # Pass the sampled diaphragm thickness.
    kwargs["pressure"] = float(parameters["pressure"])  # Pass the sampled wheel pressure as the sole non-geometric variable.
    kwargs["support_width"] = float(parameters["support_width"])  # Pass the sampled bearing-strip width.
    return kwargs  # Return a complete configuration independent of future factory-default changes.

def _case_hashes(parameters: Mapping[str, float]) -> tuple[str, str]:  # Derive canonical geometry and full-configuration identities for one case.
    factory_kwargs = _full_factory_kwargs(parameters)  # Expand the sampled record into the complete factory configuration.
    geometry_kwargs = {key: value for key, value in factory_kwargs.items() if key != "pressure"}  # Exclude the load magnitude from the geometric identity.
    geometry_payload = {"factory": FACTORY_PATH, "topology": f"{FAMILY_NAME}-v1", "kwargs": geometry_kwargs}  # Bind geometry values to the exact factory and topology version.
    config_payload = {"factory": FACTORY_PATH, "topology": f"{FAMILY_NAME}-v1", "kwargs": factory_kwargs}  # Bind all geometry and load values to the exact factory.
    return _sha256_payload(geometry_payload), _sha256_payload(config_payload)  # Return full collision-resistant digests for independent audit.

def validate_geometry_parameters(parameters: Mapping[str, float]) -> None:  # Reject out-of-range or geometrically impossible bridge configurations.
    expected_names = set(_parameter_names())  # Collect the exact six names admitted by the frozen protocol.
    supplied_names = set(parameters)  # Collect the names supplied by the candidate case.
    if supplied_names != expected_names:  # Reject missing fields and unregistered extra design variables.
        raise ValueError(f"parameter names must be exactly {sorted(expected_names)}")  # Report the immutable six-dimensional contract.
    for name, lower, upper, _unit in PARAMETER_SPECS:  # Check every sampled scalar against its exact frozen range.
        value = float(parameters[name])  # Normalize numerical scalar types before validation.
        if not math.isfinite(value):  # Reject NaN and infinite values before geometric arithmetic.
            raise ValueError(f"{name} must be finite")  # Identify the non-finite parameter precisely.
        if value < lower or value > upper:  # Enforce inclusive bounds from the frozen parameter table.
            raise ValueError(f"{name}={value} lies outside [{lower}, {upper}]")  # Report the precise interval violation.
    kwargs = _full_factory_kwargs(parameters)  # Expand the candidate into explicit geometric constants and variables.
    length = float(kwargs["length"])  # Read the frozen segment length for longitudinal containment checks.
    width = float(kwargs["width"])  # Read the frozen box width for transverse containment checks.
    height = float(kwargs["height"])  # Read the frozen box depth for vertical containment checks.
    bottom_thickness = float(kwargs["bottom_thickness"])  # Read the diaphragm's lower physical boundary.
    web_thickness = float(kwargs["web_thickness"])  # Read the diaphragm's transverse inner boundaries.
    frame_width = float(kwargs["frame_width"])  # Read the fixed opening-frame envelope expansion.
    wheel_length, wheel_width = (float(value) for value in kwargs["wheel_size"])  # Unpack the frozen rectangular wheel footprint.
    wheel_center_x = 0.5 * length + float(parameters["wheel_offset_x"])  # Compute the sampled longitudinal wheel centre.
    wheel_center_y = 0.5 * width + float(parameters["wheel_offset_y"])  # Compute the sampled transverse wheel centre.
    wheel_inside_x = wheel_center_x - 0.5 * wheel_length >= 0.0 and wheel_center_x + 0.5 * wheel_length <= length  # Require the complete footprint inside the top plate longitudinally.
    wheel_inside_y = wheel_center_y - 0.5 * wheel_width >= 0.0 and wheel_center_y + 0.5 * wheel_width <= width  # Require the complete footprint inside the top plate transversely.
    if not wheel_inside_x or not wheel_inside_y:  # Reject any wheel face that crosses a top-plate boundary.
        raise ValueError("wheel patch must lie completely inside the top plate")  # Preserve the protocol's load-face feasibility rule.
    opening_radius = float(parameters["opening_radius"])  # Read the sampled access-opening radius.
    opening_center_y = 0.5 * width  # Reproduce the canonical factory's transverse opening centre.
    opening_center_z = 0.5 * height  # Reproduce the canonical factory's vertical opening centre.
    opening_frame_extent = opening_radius + frame_width  # Include the complete rectangular opening frame in the envelope.
    frame_inside_y = opening_center_y - opening_frame_extent >= web_thickness and opening_center_y + opening_frame_extent <= width - web_thickness  # Require the opening and frame to remain between the webs.
    frame_inside_z = opening_center_z - opening_frame_extent >= bottom_thickness and opening_center_z + opening_frame_extent <= height  # Require the opening and frame to remain inside the diaphragm height.
    if not frame_inside_y or not frame_inside_z:  # Reject a hole or opening frame that leaves the diaphragm plate.
        raise ValueError("opening and frame must lie completely inside the diaphragm")  # Preserve the protocol's diaphragm-containment rule.
    support_width = float(parameters["support_width"])  # Read the sampled longitudinal bearing-strip width.
    left_support_center = 0.12 * length  # Reproduce the canonical fixed-bearing location.
    right_support_center = 0.88 * length  # Reproduce the canonical roller-bearing location.
    supports_inside = left_support_center - 0.5 * support_width >= 0.0 and right_support_center + 0.5 * support_width <= length  # Require both imprinted bearing strips to stay on the bottom plate.
    if not supports_inside:  # Reject support faces that would leave the model domain.
        raise ValueError("support strips must lie completely inside the bottom plate")  # Report the bearing-footprint feasibility failure.

def _latin_hypercube_candidate(rng: np.random.Generator) -> np.ndarray:  # Draw one exact 48-by-six randomized Latin hypercube in the unit cube.
    points = np.empty((CASE_COUNT, len(PARAMETER_SPECS)), dtype=float)  # Allocate the normalized candidate design.
    for column in range(len(PARAMETER_SPECS)):  # Fill each parameter dimension with every stratum exactly once.
        strata = rng.permutation(CASE_COUNT)  # Assign the 48 strata to cases without replacement.
        jitter = rng.random(CASE_COUNT)  # Draw one uniform within-stratum offset for every case.
        points[:, column] = (strata + jitter) / CASE_COUNT  # Place each point strictly inside its assigned stratum.
    return points  # Return a valid six-dimensional Latin hypercube candidate.

def _minimum_pair_distance(points: np.ndarray) -> float:  # Measure the normalized Euclidean separation of the closest two cases.
    differences = points[:, None, :] - points[None, :, :]  # Form all ordered pairwise coordinate differences.
    squared_distances = np.einsum("ijk,ijk->ij", differences, differences)  # Sum squared differences across the six normalized dimensions.
    np.fill_diagonal(squared_distances, np.inf)  # Exclude every point's zero self-distance from the minimum.
    return float(np.sqrt(np.min(squared_distances)))  # Return the candidate's classical maximin objective.

def _maximin_latin_hypercube() -> tuple[np.ndarray, float, int]:  # Select the best deterministic LHS from the frozen candidate pool.
    rng = np.random.Generator(np.random.PCG64(FROZEN_SEED))  # Use an explicit stable bit generator rather than a process-global RNG.
    best_points: np.ndarray | None = None  # Hold the best normalized design observed so far.
    best_distance = -math.inf  # Initialize the strict maximin objective below every feasible candidate.
    best_index = -1  # Record the zero-based winning candidate for audit reproduction.
    for candidate_index in range(MAXIMIN_CANDIDATES):  # Exhaust the complete preregistered candidate pool.
        points = _latin_hypercube_candidate(rng)  # Draw the next exact Latin hypercube from the same deterministic stream.
        minimum_distance = _minimum_pair_distance(points)  # Evaluate its closest normalized pair.
        if minimum_distance > best_distance:  # Keep only strict improvements so first occurrence breaks exact ties.
            best_points = points.copy()  # Copy the winner so later RNG draws cannot mutate it.
            best_distance = minimum_distance  # Freeze the improved maximin score.
            best_index = candidate_index  # Freeze the improved candidate's audit index.
    if best_points is None:  # Guard against an impossible empty frozen candidate pool.
        raise RuntimeError("maximin candidate pool produced no design")  # Fail explicitly rather than emitting an empty manifest.
    return best_points, best_distance, best_index  # Return the winning normalized design and complete audit metadata.

def build_case_manifest() -> dict[str, Any]:  # Build the unique frozen 48-case manifest entirely in memory.
    normalized_points, minimum_distance, candidate_index = _maximin_latin_hypercube()  # Generate the deterministic maximin Latin hypercube.
    lower_bounds = np.asarray([spec[1] for spec in PARAMETER_SPECS], dtype=float)  # Collect the six physical lower bounds in column order.
    upper_bounds = np.asarray([spec[2] for spec in PARAMETER_SPECS], dtype=float)  # Collect the six physical upper bounds in column order.
    physical_points = lower_bounds + normalized_points * (upper_bounds - lower_bounds)  # Scale every unit-cube coordinate into its exact engineering interval.
    split_labels = [split for split, count in SPLIT_COUNTS for _index in range(count)]  # Expand the immutable 24/8/16 split order to 48 labels.
    parameter_names = _parameter_names()  # Reuse the exact six-dimensional column order.
    cases: list[dict[str, Any]] = []  # Allocate the ordered canonical case records.
    for row_index in range(CASE_COUNT):  # Convert every selected LHS point into one reconstructable bridge case.
        parameters = {name: round(float(physical_points[row_index, column]), FLOAT_DECIMALS) for column, name in enumerate(parameter_names)}  # Quantize physical inputs before validation and hashing.
        validate_geometry_parameters(parameters)  # Enforce geometry feasibility before the case can enter any manifest.
        geometry_hash, config_hash = _case_hashes(parameters)  # Hash geometry and full factory configuration independently.
        case_id = f"BGD-{row_index + 1:03d}-{config_hash[:12]}"  # Bind a sortable frozen ordinal to the canonical configuration identity.
        cases.append({"case_id": case_id, "split": split_labels[row_index], "parameters": parameters, "geometry_hash": geometry_hash, "config_hash": config_hash})  # Store only auditable case identity, partition, inputs, and hashes.
    parameter_ranges = {name: [lower, upper] for name, lower, upper, _unit in PARAMETER_SPECS}  # Preserve exact inclusive physical ranges in the manifest.
    parameter_units = {name: unit for name, _lower, _upper, unit in PARAMETER_SPECS}  # Preserve engineering units independently from numerical bounds.
    manifest: dict[str, Any] = {"schema": MANIFEST_SCHEMA, "protocol_id": PROTOCOL_ID, "family": FAMILY_NAME, "factory": FACTORY_PATH, "seed": FROZEN_SEED, "case_count": CASE_COUNT, "split_counts": dict(SPLIT_COUNTS), "parameter_order": list(parameter_names), "parameter_ranges": parameter_ranges, "parameter_units": parameter_units, "sampler": {"method": "random-candidate-maximin-latin-hypercube", "bit_generator": "PCG64", "candidate_count": MAXIMIN_CANDIDATES, "selected_candidate_index": candidate_index, "normalized_minimum_pair_distance": round(minimum_distance, 15)}, "cases": cases}  # Assemble the complete frozen manifest without an unstable timestamp.
    validate_case_manifest(manifest)  # Recheck all schema, split, LHS, geometry, identity, and hash invariants before return.
    return manifest  # Return the valid in-memory manifest for persistence or test use.

def validate_case_manifest(manifest: Mapping[str, Any]) -> None:  # Validate a loaded or generated manifest without performing a finite-element solve.
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("protocol_id") != PROTOCOL_ID:  # Require the exact schema and experiment protocol.
        raise ValueError("manifest schema or protocol_id is not frozen WMVLA-4WAY-P1")  # Reject unrelated or stale manifests.
    if manifest.get("family") != FAMILY_NAME or manifest.get("factory") != FACTORY_PATH:  # Require the canonical bridge family and reconstruction factory.
        raise ValueError("manifest family or factory is not canonical")  # Reject topology substitutions.
    if manifest.get("seed") != FROZEN_SEED or manifest.get("case_count") != CASE_COUNT:  # Require the preregistered seed and total case count.
        raise ValueError("manifest seed or case_count differs from the frozen protocol")  # Reject resampled or incomplete campaigns.
    if manifest.get("split_counts") != dict(SPLIT_COUNTS):  # Require the exact ordered split cardinalities as a JSON mapping.
        raise ValueError("manifest split_counts must be 24/8/16")  # Reject any post-hoc split adjustment.
    expected_ranges = {name: [lower, upper] for name, lower, upper, _unit in PARAMETER_SPECS}  # Reconstruct the exact parameter bounds for comparison.
    if manifest.get("parameter_order") != list(_parameter_names()) or manifest.get("parameter_ranges") != expected_ranges:  # Require exact columns and ranges from the protocol.
        raise ValueError("manifest parameter order or ranges differ from the frozen protocol")  # Reject renamed, reordered, or rescaled variables.
    cases_value = manifest.get("cases")  # Read the heterogeneous JSON case container once.
    if not isinstance(cases_value, list) or len(cases_value) != CASE_COUNT:  # Require exactly 48 ordered case mappings.
        raise ValueError("manifest must contain exactly 48 cases")  # Reject malformed or truncated case collections.
    observed_split_counts = {split: 0 for split, _count in SPLIT_COUNTS}  # Initialize independent split-count verification.
    observed_case_ids: set[str] = set()  # Track case identity uniqueness across the manifest.
    observed_config_hashes: set[str] = set()  # Track full-configuration uniqueness across the manifest.
    normalized_columns = np.empty((CASE_COUNT, len(PARAMETER_SPECS)), dtype=float)  # Allocate normalized coordinates for the Latin property check.
    for row_index, case_value in enumerate(cases_value):  # Validate every case in its frozen manifest order.
        if not isinstance(case_value, Mapping):  # Reject scalar or list entries masquerading as cases.
            raise ValueError("every manifest case must be a mapping")  # Report the structural schema failure.
        split = case_value.get("split")  # Read the recorded partition label.
        if split not in observed_split_counts:  # Reject unregistered or missing partition labels.
            raise ValueError(f"invalid split for case row {row_index}")  # Identify the malformed case row.
        observed_split_counts[str(split)] += 1  # Count the validated split assignment.
        parameters_value = case_value.get("parameters")  # Read the six-dimensional physical parameter mapping.
        if not isinstance(parameters_value, Mapping):  # Reject absent or non-mapping parameter records.
            raise ValueError(f"parameters missing for case row {row_index}")  # Identify the malformed case row.
        parameters = {name: float(parameters_value[name]) for name in _parameter_names() if name in parameters_value}  # Normalize available numerical values for geometry and hash checks.
        validate_geometry_parameters(parameters)  # Re-run the pre-write physical and geometric feasibility gate.
        geometry_hash, config_hash = _case_hashes(parameters)  # Recompute both canonical identities from physical inputs.
        if case_value.get("geometry_hash") != geometry_hash or case_value.get("config_hash") != config_hash:  # Reject any input-to-hash inconsistency.
            raise ValueError(f"canonical hash mismatch for case row {row_index}")  # Identify the tampered or stale record.
        expected_case_id = f"BGD-{row_index + 1:03d}-{config_hash[:12]}"  # Reconstruct the sortable content-bound case identifier.
        if case_value.get("case_id") != expected_case_id:  # Require the exact ordinal and configuration prefix.
            raise ValueError(f"case_id mismatch for case row {row_index}")  # Reject renamed, reordered, or duplicated cases.
        if expected_case_id in observed_case_ids or config_hash in observed_config_hashes:  # Require unique case and full-configuration identities.
            raise ValueError(f"duplicate case identity for case row {row_index}")  # Reject accidental duplicate LHS rows.
        observed_case_ids.add(expected_case_id)  # Record the validated case identifier.
        observed_config_hashes.add(config_hash)  # Record the validated full-configuration digest.
        for column, (_name, lower, upper, _unit) in enumerate(PARAMETER_SPECS):  # Normalize every physical coordinate for the Latin property check.
            normalized_columns[row_index, column] = (parameters[_name] - lower) / (upper - lower)  # Map the exact stored value back to its unit interval.
    if observed_split_counts != dict(SPLIT_COUNTS):  # Compare independently counted partitions against the frozen 24/8/16 contract.
        raise ValueError("case split assignments do not match 24/8/16")  # Reject post-hoc relabeling even when top-level metadata is unchanged.
    for column, (name, _lower, _upper, _unit) in enumerate(PARAMETER_SPECS):  # Check every dimension independently for exact stratification.
        strata = np.floor(normalized_columns[:, column] * CASE_COUNT).astype(int)  # Recover the 48 zero-based Latin strata from stored values.
        strata = np.clip(strata, 0, CASE_COUNT - 1)  # Treat an exact inclusive upper endpoint as the final legal stratum.
        if sorted(strata.tolist()) != list(range(CASE_COUNT)):  # Require every stratum exactly once in each parameter dimension.
            raise ValueError(f"parameter {name} is not a 48-level Latin hypercube")  # Reject non-Latin or duplicated sampling.

def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:  # Serialize the human-readable manifest into deterministic persisted bytes.
    validate_case_manifest(manifest)  # Prevent invalid geometry or identities from reaching the filesystem.
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"  # Produce stable, reviewable UTF-8 JSON with one terminal newline.
    return text.encode("utf-8")  # Return the exact bytes covered by case_manifest.sha256.

def write_case_manifest(output_directory: Path | str, manifest: Mapping[str, Any] | None = None) -> tuple[Path, Path, str]:  # Persist the frozen manifest and its exact-byte SHA-256 sidecar.
    payload = build_case_manifest() if manifest is None else dict(manifest)  # Build the unique frozen design unless a test supplies an explicit payload.
    encoded = manifest_bytes(payload)  # Complete every schema and geometry check before creating the manifest file.
    digest = hashlib.sha256(encoded).hexdigest()  # Hash the exact UTF-8 bytes that will be written.
    output_path = Path(output_directory)  # Normalize the caller's target directory.
    manifest_path = output_path / "case_manifest.json"  # Freeze the protocol-required JSON filename.
    checksum_path = output_path / "case_manifest.sha256"  # Freeze the protocol-required checksum filename.
    output_path.mkdir(parents=True, exist_ok=True)  # Create the destination only after all feasibility checks pass.
    manifest_path.write_bytes(encoded)  # Persist the already-validated deterministic manifest bytes.
    checksum_path.write_text(f"{digest}  {manifest_path.name}\n", encoding="ascii")  # Persist a standard sha256sum-compatible sidecar.
    return manifest_path, checksum_path, digest  # Return exact artifact paths and digest for command-line reporting.

def load_case_manifest(manifest_path: Path | str, verify_checksum: bool = True) -> dict[str, Any]:  # Load, validate, and optionally checksum-verify a persisted case manifest.
    path = Path(manifest_path)  # Normalize the supplied manifest path.
    encoded = path.read_bytes()  # Read the exact persisted bytes before JSON decoding.
    payload = json.loads(encoded.decode("utf-8"))  # Decode the transparent JSON document.
    if not isinstance(payload, dict):  # Require one top-level JSON object.
        raise ValueError("case manifest root must be a JSON object")  # Reject structurally invalid manifest documents.
    validate_case_manifest(payload)  # Re-run schema, split, LHS, geometry, ID, and hash validation after loading.
    if verify_checksum:  # Enforce the protocol sidecar unless a caller explicitly requests structure-only loading.
        checksum_path = path.with_suffix(".sha256")  # Resolve case_manifest.sha256 beside case_manifest.json.
        checksum_fields = checksum_path.read_text(encoding="ascii").strip().split()  # Parse the standard digest and filename fields.
        if len(checksum_fields) != 2 or checksum_fields[1] != path.name:  # Require a precise sidecar reference to this manifest filename.
            raise ValueError("case_manifest.sha256 has an invalid format or filename")  # Reject ambiguous or misdirected checksum files.
        observed_digest = hashlib.sha256(encoded).hexdigest()  # Hash the exact bytes that were loaded.
        if checksum_fields[0] != observed_digest:  # Compare the recorded and independently observed SHA-256 values.
            raise ValueError("case_manifest.sha256 does not match case_manifest.json")  # Reject altered or partially written manifest artifacts.
    return payload  # Return the fully validated and optionally checksum-verified manifest.

def problem_from_case(case: Mapping[str, Any]):  # Reconstruct one canonical Problem object from a validated manifest case.
    parameters_value = case.get("parameters")  # Read the case's six-dimensional parameter record.
    if not isinstance(parameters_value, Mapping):  # Require the same mapping structure used by the manifest schema.
        raise ValueError("case parameters must be a mapping")  # Reject records that cannot reconstruct the factory call.
    parameters = {name: float(parameters_value[name]) for name in _parameter_names() if name in parameters_value}  # Normalize all available scalar parameters.
    validate_geometry_parameters(parameters)  # Enforce physical feasibility before constructing any Gmsh closure.
    geometry_hash, config_hash = _case_hashes(parameters)  # Recompute the canonical identities before factory dispatch.
    if case.get("geometry_hash") != geometry_hash or case.get("config_hash") != config_hash:  # Reject stale or tampered records before reconstruction.
        raise ValueError("case canonical hashes do not match its parameters")  # Preserve manifest-to-factory provenance.
    return make_box_girder_diaphragm(**_full_factory_kwargs(parameters))  # Dispatch the exact explicit configuration to the existing canonical factory.
