#!/usr/bin/env python3  # Evaluate spatial actions supplied by the live GPT visual assistant.
"""Compile explicit GPT image decisions and evaluate frozen choices with CalculiX."""  # Declare that this script does not call a vision-language API.
from __future__ import annotations  # Permit modern type hints without eager evaluation.
import argparse  # Expose reproducible input and output locations.
import hashlib  # Preserve source-image, decision, model, and initial-state provenance.
import json  # Read the assistant's explicit spatial decision.
import os  # Set consistent native threading before numerical imports.
import re  # Recover the physical case identity and safe artifact names.
import sys  # Import the repository and its existing experiment helpers.
import time  # Measure actual compilation, prediction, and solver costs.
from dataclasses import asdict  # Serialize authentic CalculiX solve records.
from datetime import datetime, timezone  # Timestamp the saved pre-solve decisions.
from pathlib import Path  # Handle the supplied experiment directories.
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Keep the same one-thread solver convention as the original experiment.
ROOT = Path(__file__).resolve().parents[1]  # Resolve the source checkout independently of the current directory.
sys.path.insert(0, str(ROOT))  # Make the repository's numerical modules importable.
sys.path.insert(0, str(ROOT / "scripts"))  # Import the neighboring benchmark without running its guarded main function.
import numpy as np  # Perform deterministic region compilation and feature calculations.
from run_visual_world_experiment import case_problem, equations, materialize, normalize_target, reference_error, save_post, write_json  # Reuse the original physical factories, resource budget, mesher, and error evaluation.
from visionamr.experiment import FemRunner  # Route every new real solve through existing solver accounting.
from visionamr.fem_post import compute_post  # Reconstruct present or reference fields from saved displacements.
from visionamr.indicators import zz_indicator  # Measure error without using it to replace GPT-selected region positions.
from visionamr.mesher import Mesh  # Restore the authentic saved tetrahedral mesh.
from visionamr.vla.visual_spatial import features, rasterize, spatial_error  # Build the same numerical observation and transition targets used during training.
from visionamr.vla.visual_world_model import SpatialWorldModel  # Score explicit GPT candidates with the already-trained one-step model.
# These helpers only compile explicit coordinates and never inspect an error field.
def file_sha256(path: str | Path) -> str:  # Hash a supplied artifact without interpreting its contents.
    digest = hashlib.sha256()  # Initialize the standard SHA-256 accumulator.
    with Path(path).open("rb") as stream:  # Stream large numerical artifacts without duplicating them in memory.
        for block in iter(lambda: stream.read(1024 * 1024), b""):  # Read bounded chunks until the input ends.
            digest.update(block)  # Include each byte exactly once.
    return digest.hexdigest()  # Return the stable lowercase provenance digest.
def safe_name(value: object) -> str:  # Preserve readable candidate names while preventing path traversal.
    name = str(value)  # Accept ordinary JSON string-like identifiers.
    if not re.fullmatch(r"[\w-]+", name):  # Require a filename-safe nonempty identifier including Unicode word characters.
        raise ValueError("candidate names must contain only letters, numbers, underscores, or hyphens")  # Explain the artifact naming constraint.
    return name  # Use exactly the same name in decisions and saved outcomes.
def compile_region_target(mesh: Mesh, candidate: dict, bbox: np.ndarray) -> np.ndarray:  # Convert GPT's normalized three-dimensional boxes into nodal size ratios.
    limits = np.asarray(bbox, dtype=float)  # Read the present geometry's physical coordinate frame.
    limits = limits.reshape(2, 3) if limits.shape == (6,) else limits  # Accept the visual module's xmin,ymin,zmin,xmax,ymax,zmax metadata directly.
    if limits.shape != (2, 3) or not np.all(np.isfinite(limits)) or np.any(limits[1] <= limits[0]):  # Require an actual three-dimensional bounding box.
        raise ValueError("bbox must contain finite three-dimensional lower and upper corners")  # Reject ambiguous coordinate systems.
    positions = (mesh.nodes[:, :3] - limits[0]) / (limits[1] - limits[0])  # Express each existing node in the same normalized coordinates as GPT's decision.
    background = float(candidate.get("background_ratio", 1.0))  # Read the explicitly requested background size ratio.
    if not np.isfinite(background) or background <= 0.0:  # Require a valid mesh-size multiplier.
        raise ValueError("background_ratio must be finite and positive")  # Explain malformed background sizing.
    ratios = np.full(mesh.n_nodes, background, dtype=float)  # Initialize the entire source mesh from GPT's background instruction.
    regions = candidate.get("regions", [])  # Permit a deliberately uniform candidate with no local boxes.
    if not isinstance(regions, list):  # Require a concrete ordered collection of region objects.
        raise ValueError("regions must be a list of normalized three-dimensional boxes")  # Explain the expected JSON structure.
    for region in regions:  # Apply the exact regions supplied by GPT without deriving new locations.
        lower = np.asarray(region["lo"], dtype=float)  # Read the normalized lower corner.
        upper = np.asarray(region["hi"], dtype=float)  # Read the normalized upper corner.
        ratio = float(region["ratio"])  # Read the requested interior mesh-size ratio.
        halo = float(region.get("halo", 0.0))  # Read the normalized exterior transition thickness.
        if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite(np.r_[lower, upper])):  # Require three finite coordinates per box corner.
            raise ValueError("every region needs finite three-coordinate lo and hi values")  # Report an invalid spatial instruction.
        if np.any(lower < 0.0) or np.any(upper > 1.0) or np.any(upper <= lower):  # Keep nondegenerate region boxes inside the stated normalized frame.
            raise ValueError("region boxes must satisfy 0 <= lo < hi <= 1 in each axis")  # Prevent silent coordinate clipping or reinterpretation.
        if not np.isfinite(ratio) or ratio <= 0.0 or not np.isfinite(halo) or halo < 0.0:  # Validate the supplied interior ratio and smoothing thickness.
            raise ValueError("region ratio must be positive and halo non-negative, both finite")  # Explain malformed action magnitudes.
        exterior = np.maximum(np.maximum(lower - positions, positions - upper), 0.0)  # Find each node's distance components outside the requested box.
        distance = np.linalg.norm(exterior, axis=1)  # Use Euclidean distance in normalized bounding-box coordinates.
        if halo == 0.0:  # Preserve a requested sharp box boundary without inventing a smoothing scale.
            weight = (distance <= 1.0e-12).astype(float)  # Apply the interior ratio only to nodes within the box.
        else:  # Smoothly restore the background outside the explicitly supplied box.
            progress = np.clip(distance / halo, 0.0, 1.0)  # Normalize the exterior distance by GPT's halo width.
            weight = 1.0 - 3.0 * progress ** 2 + 2.0 * progress ** 3  # Use a continuous smoothstep with zero slope at both transition ends.
        requested = background + (ratio - background) * weight  # Blend the requested interior and background ratios.
        ratios = np.minimum(ratios, requested)  # Combine overlapping regions using the requested minimum-size rule.
    return mesh.node_sizes * ratios  # Convert explicit image-derived ratios to a physical nodal target field.
def restore_post(path: str | Path, problem):  # Restore a field using only a specifically allowed saved displacement artifact.
    with np.load(path, allow_pickle=False) as saved:  # Read finite-element arrays without executable object deserialization.
        mesh = Mesh(saved["nodes"].copy(), saved["cells"].copy(), 3)  # Reconstruct the saved three-dimensional mesh.
        displacement = saved["u"].copy()  # Retain only the displacement field needed for independent post-processing.
    return compute_post(mesh, problem, displacement)  # Reconstruct stresses and energy with the original constitutive model.
def main() -> None:  # Execute explicit live-assistant decisions without invoking or impersonating a VLM.
    parser = argparse.ArgumentParser(description=__doc__)  # Define the replayable experiment interface.
    parser.add_argument("--case-dir", type=Path, required=True)  # Locate the completed initial state of a held-out physical instance.
    parser.add_argument("--decision-json", type=Path, required=True)  # Read the root assistant's independently supplied GPT image decision.
    parser.add_argument("--output", type=Path, required=True)  # Choose a separate directory for this GPT-action experiment.
    parser.add_argument("--source-image", type=Path)  # Optionally verify the exact image bytes whose digest GPT recorded.
    parser.add_argument("--ranker-model", type=Path)  # Optionally freeze a separately trained action-benefit ranker before any new solve.
    args = parser.parse_args()  # Parse concrete execution options.
    if (args.output / "predictions_before_solves.json").exists():  # Preserve existing sealed choices and their linked physical outcomes during accidental reruns.
        raise FileExistsError("output already contains a sealed decision; choose a new output directory for replay")  # Require a new evidence location rather than silently replacing a completed decision record.
    started = time.perf_counter()  # Begin local preprocessing accounting after the external GPT decision already exists.
    case_dir = args.case_dir.resolve()  # Resolve the physical case directory without reading future branch results.
    match = re.fullmatch(r"(?:train|test)_(bearing|deck)_(\d+)", case_dir.name)  # Derive family and seed from the existing experiment's directory contract.
    if match is None:  # Require an unambiguous factory identity without consulting case.json or partial.json.
        raise ValueError("case-dir name must be train/test_bearing/deck_SEED")  # Explain how the initial state is associated with a physical problem.
    family, seed_text = match.groups()  # Read the physical family and seeded instance identifier.
    seed = int(seed_text)  # Convert the instance identifier to the original generator seed.
    manifest_path = case_dir.parent / "manifest.json"  # Locate shared experiment settings rather than future case outcomes.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # Read only the previously fixed growth and rasterization settings.
    growth = float(manifest["growth"])  # Reuse the common original resource multiplier.
    resolution = int(manifest["resolution"])  # Reuse the trained model's exact image resolution.
    decision_bytes = args.decision_json.read_bytes()  # Preserve the literal externally supplied decision document.
    decision = json.loads(decision_bytes)  # Parse the supplied image-derived spatial instructions.
    decision_hash = hashlib.sha256(decision_bytes).hexdigest()  # Bind all later predictions to the unchanged original JSON bytes.
    source_hash = str(decision["source_image_sha256"]).lower()  # Read the root assistant's declared image identity.
    if re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:  # Require a real SHA-256 representation in provenance metadata.
        raise ValueError("source_image_sha256 must contain 64 hexadecimal characters")  # Avoid accepting an unlabeled image identity as verified evidence.
    image_verified = False  # Distinguish recorded provenance from actual local image-byte verification.
    if args.source_image is not None:  # Verify the exact viewed image only when a path was supplied.
        image_verified = file_sha256(args.source_image) == source_hash  # Compare all image bytes with the predeclared digest.
        if not image_verified:  # Keep a mismatched image from being mislabeled as the visual decision source.
            raise ValueError("source image bytes do not match source_image_sha256")  # Report the concrete provenance mismatch.
    proposals = decision["candidates"]  # Read GPT's explicit spatial alternatives without generating new candidates.
    if not isinstance(proposals, list) or not proposals:  # Require at least one concrete executable visual action.
        raise ValueError("candidates must contain at least one explicit GPT spatial action")  # Explain why there is no action to evaluate.
    names = [safe_name(candidate["name"]) for candidate in proposals]  # Check names once before allocating output files.
    primary = safe_name(decision["primary"])  # Read the direct GPT choice before any new numerical outcome is available.
    if len(set(names)) != len(names) or primary not in names:  # Require unique candidate identities and a primary drawn from them.
        raise ValueError("candidate names must be unique and include primary")  # Reject an ambiguous decision without choosing for GPT.
    args.output.mkdir(parents=True, exist_ok=True)  # Prepare a separate output directory for the frozen decision and real outcomes.
    original_path = args.output / "originaldecision.json"  # Retain a clearly named byte-exact source decision artifact.
    original_path.write_bytes(decision_bytes)  # Save the raw GPT decision before candidate meshing or prediction.
    provenance = {"decision_source": "GPT decision supplied by live assistant", "vision_language_api_called_by_script": False, "future_batch_api_adapter_built": False, "source_image_sha256": source_hash, "source_image_hash_verified": image_verified, "decision_sha256": decision_hash, "case_dir": str(case_dir), "family": family, "seed": seed, "manifest_sha256": file_sha256(manifest_path), "initial_sha256": file_sha256(case_dir / "initial.npz"), "region_positions_from": "explicit GPT JSON only; no eta-based position replacement", "halo_metric": "Euclidean distance in normalized bbox coordinates with smoothstep transition"}  # Record the actual decision architecture and coordinate semantics.
    write_json(args.output / "originaldecision_provenance.json", provenance)  # Preserve metadata before any candidate outcome can be consulted.
    problem = case_problem(family, seed)  # Regenerate the original physical problem from its fixed factory seed.
    current = restore_post(case_dir / "initial.npz", problem)  # Read only the allowed common initial finite-element state.
    eta = zz_indicator(problem, current)  # Compute the present estimator solely for model features and later outcome ratios.
    state = rasterize(problem, current, eta, resolution=resolution)  # Recover the present image representation used by the fitted world model.
    initial_equations = equations(problem, current.mesh)  # Count actual free unknowns directly without reading future case records.
    cap = int(growth * initial_equations)  # Apply the exact original common free-equation budget.
    model_path = case_dir.parent / "visual_world_model.json"  # Locate the existing trained one-step model.
    model = SpatialWorldModel.load(model_path)  # Restore fixed training-only parameters without fitting on this test case.
    provenance["model_sha256"] = file_sha256(model_path)  # Bind the hybrid selector to the loaded model snapshot.
    entries, candidate_meshes = [], []  # Hold only present-state features and unsolved future mesh realizations.
    for name, proposal in zip(names, proposals):  # Compile every explicitly supplied GPT alternative before any new solve.
        raw_target = compile_region_target(current.mesh, proposal, state.bbox)  # Convert GPT's own spatial coordinates and ratios directly to nodal sizes.
        target = normalize_target(current.mesh, raw_target, growth, problem)  # Apply the same global resource normalization used by all other methods.
        candidate_mesh, executed, cost = materialize(problem, current.mesh, target, cap)  # Measure actual mesh closure and enforce the shared finite resource budget.
        observation = features(state, executed, current.mesh)  # Encode the actual current image together with the realized GPT action.
        observation = np.r_[observation, np.log1p(cost["n_equations"]), cost["n_equations"] / initial_equations]  # Match the existing model's exact resource-feature schema.
        action_path = args.output / f"target_{name}.npz"  # Name the source-to-candidate mesh action artifact.
        np.savez_compressed(action_path, source_nodes=current.mesh.nodes, source_cells=current.mesh.cells, source_node_sizes=current.mesh.node_sizes, raw_target=raw_target, normalized_target=target, executed_target=executed, candidate_nodes=candidate_mesh.nodes, candidate_cells=candidate_mesh.cells, bbox=state.bbox)  # Preserve GPT positions, subsequent global scaling, and the actual unsolved mesh mapping.
        entries.append({"name": name, "proposal": proposal, "cost": cost, "x": observation.tolist(), "action_artifact": action_path.name, "action_artifact_sha256": file_sha256(action_path)})  # Record only information available before future physics is solved.
        candidate_meshes.append(candidate_mesh)  # Retain the unsolved conformal mesh for subsequent evaluation.
    prediction_started = time.perf_counter()  # Count the trained world model's numerical inference time.
    predicted_spatial = model.predict(np.asarray([entry["x"] for entry in entries]))  # Predict each GPT candidate's next spatial error before querying its real outcome.
    prediction_s = time.perf_counter() - prediction_started  # Record model inference independently of expensive meshing.
    predicted_total = predicted_spatial.sum(axis=1)  # Use the original estimator-mass objective for hybrid scoring.
    hybrid = names[int(np.argmin(predicted_total))]  # Freeze the best predicted GPT alternative using no future branch result.
    ranker_choice, ranker_scores = None, None  # Distinguish runs that do not include the additional trained action selector.
    if args.ranker_model is not None:  # Evaluate only an explicitly supplied frozen training artifact.
        from train_gpt_action_ranker import load_ranker, predict_ranker  # Import the public action-ranking API without executing its training entry point.
        ranker = load_ranker(args.ranker_model)  # Load the model trained solely on the four training physical instances.
        ranker_scores = predict_ranker(ranker, np.asarray([entry["x"] for entry in entries])).tolist()  # Score present-state action features before any candidate physics is available.
        ranker_choice = names[int(np.argmin(ranker_scores))]  # Seal the minimum predicted action-benefit score as the actual ranker decision.
        provenance["action_ranker_sha256"] = file_sha256(args.ranker_model)  # Bind this prospective choice to the exact frozen training parameters.
    for entry, prediction, total in zip(entries, predicted_spatial, predicted_total):  # Associate each unsolved action with its sealed numerical prediction.
        entry["predicted_spatial_mass_ratio"] = prediction.tolist()  # Keep all 64 predicted spatial bins for later ranking audit.
        entry["predicted_total_eta2_ratio"] = float(total)  # Preserve the predicted next-to-current total estimator-mass ratio.
    frozen = {"provenance": provenance, "sealed_utc": datetime.now(timezone.utc).isoformat(), "primary_gpt": primary, "hybrid_world_model": hybrid, "initial_equations": initial_equations, "initial_eta2": float(eta.sum()), "growth": growth, "cap": cap, "resolution": resolution, "prediction_s": prediction_s, "local_preparation_s": time.perf_counter() - started, "candidates": entries}  # Record a complete choice made without new solver outcomes.
    frozen_path = args.output / "predictions_before_solves.json"  # Establish the specific pre-solve decision record.
    frozen.update({"action_ranker": ranker_choice, "action_ranker_scores": ranker_scores})  # Add the new selector to the same immutable pre-solve evidence record.
    write_json(frozen_path, frozen)  # Save both direct GPT and hybrid choices before the first new CalculiX solve.
    frozen_hash = file_sha256(frozen_path)  # Bind all reported outcomes to the exact pre-solve record.
    runner = FemRunner(problem, args.output, keep_files=True)  # Begin new real-solve accounting only after choices are saved.
    outcomes = []  # Keep measured future results separate from the frozen prediction objects.
    for entry, candidate_mesh in zip(entries, candidate_meshes):  # Evaluate alternatives offline after both online choices are fixed.
        branch_started = time.perf_counter()  # Include the branch solve, error estimation, and output rasterization.
        following, record = runner.solve_mesh(candidate_mesh, method=entry["name"], stage="gpt_visual_branch")  # Execute authentic CalculiX on the already recorded mesh.
        next_eta = zz_indicator(problem, following)  # Measure the actually attained next-state estimator.
        next_state = rasterize(problem, following, next_eta, resolution=resolution)  # Recover the actual spatial transition for verification.
        post_path = args.output / f"post_{entry['name']}.npz"  # Name the candidate's authentic mesh and solution artifact.
        save_post(post_path, following, next_eta)  # Save actual solver-derived fields without synthesized imagery.
        outcome = {"name": entry["name"], "record": asdict(record), "eta2": float(next_eta.sum()), "eta": float(np.sqrt(next_eta.sum())), "eta2_ratio_to_initial": float(next_eta.sum() / max(eta.sum(), 1.0e-30)), "actual_spatial_mass_ratio": (spatial_error(next_state) / max(eta.sum(), 1.0e-30)).tolist(), "branch_s": time.perf_counter() - branch_started, "mesh_s": entry["cost"]["mesh_s"], "post_artifact": post_path.name}  # Record genuine outcomes and their resource costs.
        outcomes.append(outcome)  # Keep the same candidate order as the frozen predictions.
        write_json(args.output / "outcomes_partial.json", {"predictions_before_solves_sha256": frozen_hash, "outcomes": outcomes})  # Checkpoint expensive solves without modifying the sealed choices.
        print(json.dumps({"action": outcome["name"], "eta": outcome["eta"], "equations": record.n_equations, "solve_s": record.wall_s}), flush=True)  # Stream only measured solver results.
    reference_path = case_dir / "reference.npz"  # Locate an optional common reference after every choice has already been saved.
    reference_used = reference_path.is_file()  # Check reference availability without generating additional evidence silently.
    if reference_used:  # Evaluate independent reference error only after the prediction record is sealed.
        reference = restore_post(reference_path, problem)  # Restore the existing common reference from its actual displacement field.
        for outcome in outcomes:  # Compare every new candidate against the same reference integration mesh.
            following = restore_post(args.output / outcome["post_artifact"], problem)  # Reload each unchanged new solver outcome.
            outcome["reference_error"], outcome["reference_mapping_miss"] = reference_error(problem, following, reference)  # Record approximate non-nested stress-energy error and point-location misses.
    outcome_by_name = {outcome["name"]: outcome for outcome in outcomes}  # Retrieve the two previously frozen choices without reselection.
    report = {"scope": "live GPT image decision followed by explicit spatial compilation; one-step trained world-model reranking", "provenance": provenance, "predictions_before_solves_sha256": frozen_hash, "primary_gpt": primary, "hybrid_world_model": hybrid, "primary_outcome": outcome_by_name[primary], "hybrid_outcome": outcome_by_name[hybrid], "candidates": outcomes, "initial_equations": initial_equations, "initial_eta2": float(eta.sum()), "cap": cap, "reference_used": reference_used, "reference_sha256": file_sha256(reference_path) if reference_used else None, "new_solver_calls": len(outcomes), "offline_counterfactual_solver_calls_beyond_one_selected_action": len(outcomes) - 1, "all_candidate_mesh_s": sum(entry["cost"]["mesh_s"] for entry in entries), "world_model_prediction_s": prediction_s, "local_preparation_s": frozen["local_preparation_s"], "live_gpt_decision_wall_s": None, "full_end_to_end_latency_measured": False, "total_local_evaluation_s": time.perf_counter() - started}  # Distinguish real evaluation costs from unmeasured live-assistant decision latency.
    report.update({"action_ranker": ranker_choice, "ranker_outcome": outcome_by_name[ranker_choice] if ranker_choice is not None else None, "action_ranker_scores": ranker_scores})  # Report the previously frozen ranker action without selecting from measured results.
    write_json(args.output / "summary.json", report)  # Persist the additional prospective selection alongside the existing direct GPT and model outcomes.
    print(json.dumps({"primary_gpt": primary, "hybrid_world_model": hybrid, "new_solver_calls": len(outcomes), "reference_used": reference_used, "summary": str(args.output / "summary.json")}, indent=2), flush=True)  # Return a compact pointer to the complete experiment evidence.
if __name__ == "__main__":  # Run only when explicitly invoked with a supplied GPT decision.
    main()  # Execute the action replay and real finite-element evaluation.
