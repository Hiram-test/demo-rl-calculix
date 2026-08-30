"""Focused tests for the frozen bridge case-manifest and factory round trip."""  # Describe the validation surface covered by this module.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from copy import deepcopy  # Import safe manifest mutation for rejection tests.
import hashlib  # Import independent exact-byte checksum verification.
from pathlib import Path  # Import temporary artifact path annotations.
import numpy as np  # Import independent Latin-stratum recovery.
import pytest  # Import precise rejection and failure assertions.
from visionamr.bridge_case_manifest import CASE_COUNT, FACTORY_FIXED_KWARGS, FROZEN_SEED, PARAMETER_SPECS, SPLIT_COUNTS, _case_hashes, build_case_manifest, load_case_manifest, manifest_bytes, problem_from_case, validate_geometry_parameters, write_case_manifest  # Import only the frozen manifest API, fixed geometry, and hash helper under test.

FROZEN_MANIFEST_SHA256 = "0536da44e267b78692b5ac1b5df11b5a7fe51a6ec8e838a8f3f42d243b391aa7"  # Freeze the exact canonical 48-case artifact identity.

def test_frozen_maximin_lhs_is_byte_deterministic() -> None:  # Prove repeated generation produces the identical preregistered artifact.
    first = build_case_manifest()  # Generate the complete design from a fresh deterministic PCG64 stream.
    second = build_case_manifest()  # Regenerate independently without reusing any process-global state.
    assert first == second  # Require exact equality of samples, ordering, IDs, splits, hashes, and metadata.
    assert first["seed"] == FROZEN_SEED  # Require the protocol seed rather than a caller-selected seed.
    assert hashlib.sha256(manifest_bytes(first)).hexdigest() == FROZEN_MANIFEST_SHA256  # Pin the exact persisted JSON bytes against resampling drift.

def test_manifest_has_exact_splits_ranges_and_latin_strata() -> None:  # Verify the complete 48-case six-dimensional sampling contract.
    manifest = build_case_manifest()  # Generate the frozen case collection.
    cases = manifest["cases"]  # Read the ordered case records once.
    assert len(cases) == CASE_COUNT == 48  # Require exactly 48 cases without deletion or replacement.
    observed_counts = {split: sum(case["split"] == split for case in cases) for split, _count in SPLIT_COUNTS}  # Count each recorded partition independently.
    assert observed_counts == {"train": 24, "validation": 8, "test": 16}  # Require the exact fixed 24/8/16 split.
    assert len({case["case_id"] for case in cases}) == CASE_COUNT  # Require one stable unique case identifier per configuration.
    assert len({case["config_hash"] for case in cases}) == CASE_COUNT  # Require 48 distinct canonical full configurations.
    for name, lower, upper, _unit in PARAMETER_SPECS:  # Check bounds and Latin stratification in every frozen dimension.
        values = np.asarray([case["parameters"][name] for case in cases], dtype=float)  # Collect the stored physical coordinates.
        assert np.all(values >= lower) and np.all(values <= upper)  # Enforce the exact inclusive protocol range.
        strata = np.floor((values - lower) / (upper - lower) * CASE_COUNT).astype(int)  # Recover zero-based unit-cube strata.
        strata = np.clip(strata, 0, CASE_COUNT - 1)  # Map an exact upper endpoint to the final legal stratum.
        assert sorted(strata.tolist()) == list(range(CASE_COUNT))  # Require every one of 48 strata exactly once.

def test_geometry_and_config_hashes_are_canonical_and_distinct() -> None:  # Verify pressure does not masquerade as geometry while remaining part of the full configuration.
    case = build_case_manifest()["cases"][0]  # Select one complete validated case record.
    parameters = dict(case["parameters"])  # Copy the physical values without changing the manifest.
    geometry_hash, config_hash = _case_hashes(parameters)  # Recompute both canonical identities independently.
    assert geometry_hash == case["geometry_hash"]  # Require the stored geometry digest to match canonical JSON inputs.
    assert config_hash == case["config_hash"]  # Require the stored complete-configuration digest to match canonical JSON inputs.
    parameters["pressure"] = 2.8 if parameters["pressure"] != 2.8 else 6.0  # Change only the non-geometric wheel pressure within its frozen range.
    changed_geometry_hash, changed_config_hash = _case_hashes(parameters)  # Recompute identities after the load-only change.
    assert changed_geometry_hash == geometry_hash  # Require a load-only change to preserve geometric identity.
    assert changed_config_hash != config_hash  # Require the full configuration identity to include the load magnitude.

def test_manifest_case_round_trips_to_canonical_bridge_factory() -> None:  # Prove a manifest case reconstructs the existing topology-preserving Problem factory.
    case = build_case_manifest()["cases"][17]  # Select a non-boundary training case from the frozen order.
    problem = problem_from_case(case)  # Reconstruct the canonical bridge problem without running Gmsh or CalculiX.
    parameters = case["parameters"]  # Read the expected six sampled physical values.
    assert problem.name == "box_girder_diaphragm" and problem.dim == 3  # Require the intended three-dimensional steel-box family.
    assert problem.params["wheel_offset"] == (parameters["wheel_offset_x"], parameters["wheel_offset_y"])  # Require exact wheel-coordinate recombination.
    assert problem.params["opening_radius"] == parameters["opening_radius"]  # Require exact opening-radius dispatch.
    assert problem.params["diaphragm_thickness"] == parameters["diaphragm_thickness"]  # Require exact diaphragm-thickness dispatch.
    assert problem.params["pressure"] == parameters["pressure"]  # Require exact pressure dispatch.
    assert problem.params["support_width"] == parameters["support_width"]  # Require exact bearing-strip dispatch.

def test_write_load_and_checksum_verification_form_a_closed_loop(tmp_path: Path) -> None:  # Verify both required artifacts and strict exact-byte loading.
    manifest_path, checksum_path, digest = write_case_manifest(tmp_path / "protocol")  # Persist the generated manifest through the validated writer.
    assert manifest_path.name == "case_manifest.json" and checksum_path.name == "case_manifest.sha256"  # Require the protocol-mandated filenames.
    assert digest == FROZEN_MANIFEST_SHA256  # Require the persisted artifact to retain the frozen identity.
    assert checksum_path.read_text(encoding="ascii") == f"{digest}  case_manifest.json\n"  # Require a standard sha256sum-compatible sidecar.
    loaded = load_case_manifest(manifest_path)  # Load with schema, geometry, canonical hash, and sidecar verification enabled.
    assert loaded == build_case_manifest()  # Require a lossless persisted round trip.
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")  # Alter only exact bytes while leaving the JSON semantics valid.
    with pytest.raises(ValueError, match="does not match"):  # Require the checksum gate to detect semantically invisible byte tampering.
        load_case_manifest(manifest_path)  # Attempt to load the altered manifest under strict verification.

def test_invalid_geometry_is_rejected_before_manifest_write(tmp_path: Path) -> None:  # Prove an infeasible candidate cannot leave partial protocol artifacts.
    manifest = deepcopy(build_case_manifest())  # Copy the valid frozen manifest for controlled corruption.
    manifest["cases"][0]["parameters"]["wheel_offset_x"] = 1000.0  # Move the complete wheel patch far beyond the top plate.
    with pytest.raises(ValueError, match="outside"):  # Require the pre-write physical and geometric gate to fail explicitly.
        write_case_manifest(tmp_path / "invalid_protocol", manifest)  # Attempt to persist the corrupted case collection.
    assert not (tmp_path / "invalid_protocol").exists()  # Require failure before directory or manifest creation.

def test_geometry_gate_checks_full_wheel_and_opening_frame_envelopes(monkeypatch: pytest.MonkeyPatch) -> None:  # Exercise containment logic separately from the frozen ranges.
    case_parameters = dict(build_case_manifest()["cases"][0]["parameters"])  # Start from a valid six-dimensional sampled configuration.
    case_parameters["wheel_offset_y"] = 75.0  # Place the wheel at the largest legal transverse offset.
    validate_geometry_parameters(case_parameters)  # Confirm the frozen geometry accepts its own exact boundary range.
    monkeypatch.setitem(FACTORY_FIXED_KWARGS, "width", 240.0)  # Narrow only the fixed top plate so the legal offset crosses its boundary.
    with pytest.raises(ValueError, match="wheel patch"):  # Require complete rectangular footprint containment rather than centre-point containment.
        validate_geometry_parameters(case_parameters)  # Recheck the same legal sampled variables on the deliberately infeasible fixed geometry.
    monkeypatch.setitem(FACTORY_FIXED_KWARGS, "width", 360.0)  # Restore the canonical transverse width before testing the opening envelope.
    monkeypatch.setitem(FACTORY_FIXED_KWARGS, "height", 180.0)  # Shorten only the diaphragm so the framed opening crosses its lower boundary.
    case_parameters["wheel_offset_y"] = 0.0  # Centre the wheel so only the opening-frame constraint remains active.
    with pytest.raises(ValueError, match="opening and frame"):  # Require the full frame envelope, not merely the circular hole, inside the diaphragm.
        validate_geometry_parameters(case_parameters)  # Reject the deliberately infeasible framed opening before any geometry construction.
