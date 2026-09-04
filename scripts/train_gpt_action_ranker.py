#!/usr/bin/env python3  # Train an action-family-aware ranker using real training-case counterfactual solves.
"""GPT-template augmentation and fixed-alpha within-case action ranking; diagnostic probe reanalysis."""  # State that reused probes are not a fresh blind test.
from __future__ import annotations  # Support modern annotations without eager evaluation.
import argparse  # Select existing training evidence and a new output directory.
import copy  # Preserve the original GPT template while moving its spatial boxes.
import json  # Save explicit numerical training and evaluation evidence.
import os  # Configure consistent solver threading before numerical imports.
import sys  # Import existing repository experiment helpers.
import time  # Record real meshing, solver, fitting, and analysis costs.
from dataclasses import asdict  # Serialize original solver accounting records.
from datetime import datetime, timezone  # Timestamp training and prediction freezes.
from pathlib import Path  # Resolve experiment artifacts reproducibly.
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Match the established solver-thread convention.
ROOT = Path(__file__).resolve().parents[1]  # Locate the shared source checkout.
sys.path.insert(0, str(ROOT))  # Make the finite-element package importable.
sys.path.insert(0, str(ROOT / "scripts"))  # Reuse neighboring scripts without executing their main functions.
import numpy as np  # Implement the entire fixed-alpha ranking model.
from run_visual_world_experiment import case_problem, equations, materialize, normalize_target, save_post, write_json  # Retain identical physical cases, mesh budgets, and solver output formats.
from visionamr.experiment import FemRunner  # Count every new training solve through the authentic CalculiX runner.
from visionamr.indicators import zz_indicator  # Generate the requested actual estimator-mass training target.
from visionamr.vla.visual_spatial import features, rasterize  # Match existing visual-action feature construction exactly.
# Template augmentation transfers GPT's spatial action family without claiming new GPT image inspection.
def normalized_load_bbox(problem, mesh) -> np.ndarray:  # Derive the actual loading footprint from loaded boundary-facet vertices.
    selected = np.zeros(len(mesh.boundary_facets), dtype=bool)  # Start with no loaded physical boundary facets.
    for traction in problem.tractions:  # Include all actual traction-supported facets in the initial solved geometry.
        selected |= np.asarray(traction.facet_predicate(mesh.facet_centroids), dtype=bool)  # Use the executable physical loading predicate.
    if not np.any(selected):  # Explain a missing actual load footprint instead of guessing locations.
        raise ValueError("the training mesh has no loaded boundary facets")  # Reject an undefined template coordinate frame.
    vertices = mesh.nodes[mesh.boundary_facets[selected]].reshape(-1, 3)  # Gather the real vertices delimiting the loaded surface.
    bounds = np.asarray(problem.bbox, dtype=float).reshape(2, 3)  # Read the full physical bounding box.
    normalized = (vertices - bounds[0]) / (bounds[1] - bounds[0])  # Express the actual loading footprint in normalized coordinates.
    return np.stack([normalized.min(axis=0), normalized.max(axis=0)])  # Preserve lower and upper load-facet corners.
def template_source_bbox(problem) -> np.ndarray:  # Recover the source bearing load geometry without opening any test outcome artifact.
    parameters = problem.params  # Use only the explicit original physical generator parameters.
    size = np.asarray([parameters["W"], parameters["D"]], dtype=float)  # Read full bearing dimensions.
    center = 0.5 + np.asarray(parameters["offset"], dtype=float) / size  # Normalize the specified traction-patch center.
    half_width = 0.5 * np.asarray(parameters["patch"], dtype=float) / size  # Normalize the specified traction-patch half-widths.
    return np.array([[center[0] - half_width[0], center[1] - half_width[1], 1.0], [center[0] + half_width[0], center[1] + half_width[1], 1.0]])  # Return only source geometry, never test errors or displacements.
def instantiate_templates(templates: list[dict], source_bbox: np.ndarray, destination_bbox: np.ndarray) -> list[dict]:  # Move each GPT template corner relative to its nearest source loading edge.
    from run_gpt_visual_action import safe_name  # Import the training-only compiler helper lazily so the public scorer has no circular dependency.
    instantiated = copy.deepcopy(templates)  # Keep ratio, halo, background, vertical extent, and source decision unchanged.
    for candidate in instantiated:  # Instantiate all supplied GPT action families with equal treatment.
        safe_name(candidate["name"])  # Verify that each candidate can be stored under its original name.
        for region in candidate["regions"]:  # Move only the explicitly defined region boxes.
            for endpoint in ("lo", "hi"):  # Treat both corners consistently under an edge-relative transformation.
                point = np.asarray(region[endpoint], dtype=float)  # Read the source GPT normalized corner.
                for axis in (0, 1):  # Keep the original normalized depth intervals unchanged.
                    edge = int(abs(point[axis] - source_bbox[1, axis]) < abs(point[axis] - source_bbox[0, axis]))  # Identify the nearest source loading edge on this axis.
                    point[axis] += destination_bbox[edge, axis] - source_bbox[edge, axis]  # Translate the corner by the corresponding real loading-edge displacement.
                region[endpoint] = np.clip(point, 0.0, 1.0).tolist()  # Intersect translated regions with the physical bounding box when necessary.
    return instantiated  # Return deterministic template augmentation, not newly inspected GPT decisions.
def train_case(experiment: Path, output: Path, family: str, seed: int, growth: float, resolution: int, templates: list[dict], source_bbox: np.ndarray) -> dict:  # Generate three genuine new counterfactual transitions on one training geometry.
    from run_gpt_visual_action import compile_region_target, file_sha256, restore_post, safe_name  # Reuse action-compilation helpers only when collecting training data.
    case_dir = experiment / f"train_{family}_{seed}"  # Select only an original training physical instance.
    target_dir = output / f"train_{family}_{seed}"  # Keep new template branches separate from existing training outcomes.
    target_dir.mkdir(parents=True, exist_ok=True)  # Create the training-case evidence directory.
    problem = case_problem(family, seed)  # Recover exactly the original seeded physical problem.
    current = restore_post(case_dir / "initial.npz", problem)  # Restore the already solved initial state without a new initial solve.
    eta = zz_indicator(problem, current)  # Read the current physical estimator before proposing new actions.
    state = rasterize(problem, current, eta, resolution=resolution)  # Reuse the same numerical image representation as the existing experiment.
    initial_equations = equations(problem, current.mesh)  # Count the initial mesh's actual free unknowns.
    cap = int(growth * initial_equations)  # Reuse the original shared finite resource budget.
    load_bbox = normalized_load_bbox(problem, current.mesh)  # Recover this training instance's actual traction footprint.
    proposals = instantiate_templates(templates, source_bbox, load_bbox)  # Move the supplied GPT templates to this physical loading footprint.
    write_json(target_dir / "template_augmentation.json", {"source": "deterministic augmentation of a live-GPT template; no new GPT image inspection", "source_load_bbox": source_bbox.tolist(), "actual_load_facet_bbox": load_bbox.tolist(), "transformation": "nearest load-edge translation in normalized XY; fixed Z, ratio and halo; clip at domain bbox", "candidates": proposals, "initial_sha256": file_sha256(case_dir / "initial.npz"), "initial_equations": initial_equations, "cap": cap})  # Record exactly how training regions were generated before solving.
    runner = FemRunner(problem, target_dir, keep_files=True)  # Count only the new template-action training solves.
    entries = []  # Collect the new measured training transitions.
    for proposal in proposals:  # Evaluate each fixed template with identical resource treatment.
        name = safe_name(proposal["name"])  # Preserve the action-family identity in all saved files.
        branch_started = time.perf_counter()  # Count all local branch construction and execution work.
        raw_target = compile_region_target(current.mesh, proposal, state.bbox)  # Compile only template box positions and ratios, never eta-based replacement locations.
        normalized = normalize_target(current.mesh, raw_target, growth, problem)  # Apply the shared global budget normalization.
        candidate_mesh, executed, cost = materialize(problem, current.mesh, normalized, cap)  # Measure authentic remeshing and the realized free-equation count.
        x_values = np.r_[features(state, executed, current.mesh), np.log1p(cost["n_equations"]), cost["n_equations"] / initial_equations]  # Match the complete trained-model feature schema.
        np.savez_compressed(target_dir / f"target_{name}.npz", source_nodes=current.mesh.nodes, source_cells=current.mesh.cells, raw_target=raw_target, normalized_target=normalized, executed_target=executed, candidate_nodes=candidate_mesh.nodes, candidate_cells=candidate_mesh.cells)  # Preserve raw templates, global adjustments, and actual unsolved mesh mapping.
        write_json(target_dir / f"input_{name}.json", {"proposal": proposal, "x": x_values.tolist(), "cost": cost, "source": "training-template augmentation"})  # Store pre-action training inputs before this branch's solver call.
        following, record = runner.solve_mesh(candidate_mesh, method="template_" + name, stage="ranker_training")  # Obtain a real CalculiX training outcome for this action family.
        next_eta = zz_indicator(problem, following)  # Measure the actual next estimator mass without a surrogate label.
        save_post(target_dir / f"post_{name}.npz", following, next_eta)  # Preserve the original finite-element mesh, displacement, stress, and estimator arrays.
        entry = {"name": name, "source": "training-template augmentation", "x": x_values.tolist(), "eta2": float(next_eta.sum()), "eta2_ratio": float(next_eta.sum() / max(eta.sum(), 1.0e-30)), "record": asdict(record), "cost": cost, "branch_s": time.perf_counter() - branch_started}  # Retain the exact scalar training target and complete genuine solver record.
        entries.append(entry)  # Append one real observed action-conditioned transition.
        write_json(target_dir / "partial.json", {"seed": seed, "entries": entries})  # Checkpoint each newly completed training solve.
        print(json.dumps({"stage": "training", "seed": seed, "action": name, "eta2_ratio": entry["eta2_ratio"], "equations": record.n_equations}), flush=True)  # Stream measured training progress without reading test outcomes.
    result = {"seed": seed, "family": family, "initial_eta2": float(eta.sum()), "initial_equations": initial_equations, "cap": cap, "entries": entries}  # Assemble the new training-case evidence.
    write_json(target_dir / "case.json", result)  # Save the completed set of real template-action transitions.
    return result  # Return only training evidence to the ranker-fitting stage.
# Group centering learns action gains instead of confusing case difficulty with action quality.
def fit_ranker(X: np.ndarray, log_eta2: np.ndarray, groups: np.ndarray, alpha: float = 10.0) -> dict:  # Fit a signed within-case linear ridge scorer with fixed regularization.
    x_values = np.asarray(X, dtype=float)  # Read the complete training action-feature matrix.
    targets = np.asarray(log_eta2, dtype=float).reshape(-1)  # Read observed log estimator masses, which may legitimately be signed after centering.
    labels = np.asarray(groups)  # Preserve physical-case grouping throughout the contrast construction.
    centered_x, centered_y = np.zeros_like(x_values), np.zeros_like(targets)  # Allocate case-relative observations and actual action gains.
    for group in np.unique(labels):  # Construct contrasts only among actions from the same real initial state.
        selected = labels == group  # Select one training geometry's full action dictionary.
        centered_x[selected] = x_values[selected] - x_values[selected].mean(axis=0)  # Remove case-specific feature offsets without using other cases or test data.
        centered_y[selected] = targets[selected] - targets[selected].mean()  # Remove case difficulty while retaining action-dependent log-error differences.
    raw_scale = np.sqrt(np.mean(centered_x ** 2, axis=0))  # Estimate feature variability from training action contrasts only.
    active = raw_scale > np.finfo(float).eps * np.maximum(1.0, np.max(np.abs(x_values), axis=0))  # Ignore columns with no resolvable within-case action information.
    scale = np.where(active, raw_scale, 1.0)  # Avoid zero-variance division while preserving standard ridge feature units.
    standardized = np.where(active, centered_x / scale, 0.0)  # Express all training action contrasts in the fitted numerical coordinates.
    dual = np.linalg.solve(standardized @ standardized.T + float(alpha) * np.eye(len(standardized)), centered_y)  # Solve the small-sample dual ridge system without square-root clipping of signed gains.
    coefficient = standardized.T @ dual  # Store a compact primal scorer for arbitrary new candidate groups.
    fitted = standardized @ coefficient  # Measure apparent fit only on the explicitly labeled training contrasts.
    return {"schema": "gpt-template-within-case-ranker-v1", "alpha": float(alpha), "target": "log actual next eta2 minus same-training-case mean; lower score is better", "training_rows": int(len(x_values)), "training_groups": int(len(np.unique(labels))), "feature_width": int(x_values.shape[1]), "active_features": int(active.sum()), "x_scale": scale.tolist(), "active_mask": active.tolist(), "coefficient": coefficient.tolist(), "training_centered_log_eta2_rmse": float(np.sqrt(np.mean((fitted - centered_y) ** 2))), "hyperparameter_selection": "alpha fixed at 10; no test-target tuning"}  # Persist the complete signed ranking model and its honest training diagnostics.
def rank_candidates(model: dict, X: np.ndarray) -> np.ndarray:  # Score one candidate set using only its pre-solve features.
    x_values = np.asarray(X, dtype=float)  # Read the previously sealed candidate observations.
    centered = x_values - x_values.mean(axis=0)  # Remove a common query-case offset, which leaves linear action ordering unchanged.
    standardized = np.where(np.asarray(model["active_mask"], dtype=bool), centered / np.asarray(model["x_scale"]), 0.0)  # Apply only training-derived scaling and active feature selection.
    return standardized @ np.asarray(model["coefficient"])  # Predict signed relative log-error scores without requiring any query target.
def load_ranker(path: str | Path) -> dict:  # Expose a lightweight parameter loader for new blinded action experiments.
    model = json.loads(Path(path).read_text(encoding="utf-8"))  # Read only the previously frozen training snapshot.
    if model.get("schema") != "gpt-template-within-case-ranker-v1":  # Require matching score and feature semantics.
        raise ValueError("unsupported GPT action-ranker snapshot")  # Reject an incompatible model explicitly.
    width = int(model["feature_width"])  # Read the expected action-feature dimension.
    for name in ("x_scale", "active_mask", "coefficient"):  # Verify each fitted numerical parameter before inference.
        values = np.asarray(model[name], dtype=float)  # Inspect finite numeric values without executable deserialization.
        if values.shape != (width,) or not np.all(np.isfinite(values)):  # Require correctly aligned finite parameter arrays.
            raise ValueError(f"invalid ranker snapshot parameter {name}")  # Identify malformed snapshot contents.
    if np.any(np.asarray(model["x_scale"]) <= 0.0):  # Keep standardized inference denominators positive.
        raise ValueError("ranker x_scale must be positive")  # Explain an invalid numerical snapshot.
    return model  # Return fixed fitted parameters without reading or requiring test labels.
def predict_ranker(model: dict, X: np.ndarray) -> np.ndarray:  # Expose the requested reusable pre-solve ranking API.
    return rank_candidates(model, X)  # Return one relative log-error score per candidate, where smaller is preferred.
def main() -> None:  # Train first, freeze candidate rankings second, and inspect reused probe outcomes only afterward.
    from run_gpt_visual_action import file_sha256  # Import provenance hashing lazily to keep load_ranker and predict_ranker independently importable.
    parser = argparse.ArgumentParser(description=__doc__)  # Define the reproducible augmentation and ranking experiment.
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "runs" / "visual_wm_probe")  # Locate existing initial states, training branches, and the fixed split manifest.
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "gpt_ranker_training")  # Store new solver evidence, model parameters, and reanalysis outputs.
    parser.add_argument("--template-json", type=Path)  # Permit an explicitly supplied original GPT template document.
    parser.add_argument("--template-seed", type=int, default=901)  # Identify only the physical geometry used by the original template coordinates.
    args = parser.parse_args()  # Read concrete experiment settings without introducing test-tuned model options.
    began = time.perf_counter()  # Count all newly performed local experiment work.
    manifest_path = args.experiment_dir / "manifest.json"  # Read the original split and numerical settings.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # Use manifest data rather than test-case result files.
    family = str(manifest["family"])  # Preserve the original structural case family.
    if family != "bearing":  # Keep this template transformation tied to its actual rectangular top-load geometry.
        raise ValueError("this augmentation implements the bearing-load GPT template only")  # Avoid silently applying an incompatible physical template.
    train_seeds = [int(value) for value in manifest["train_seeds"].split(",")]  # Use only the originally designated training geometries.
    test_seeds = [int(value) for value in manifest["test_seeds"].split(",")]  # Retain the diagnostic probe identity without reading its targets.
    template_path = args.template_json or args.experiment_dir / f"test_{family}_{args.template_seed}" / "gpt_decision.json"  # Read only the original source decision, never its measured test outcome.
    template = json.loads(template_path.read_text(encoding="utf-8"))  # Preserve the live GPT template's explicit ratio, halo, and box definitions.
    templates = template["candidates"]  # Augment all supplied action families equally rather than selecting the test winner.
    source_bbox = template_source_bbox(case_problem(family, args.template_seed))  # Recover source-template loading edges from factory geometry alone.
    args.output.mkdir(parents=True, exist_ok=True)  # Prepare a separate experiment directory for the new training work.
    training_manifest = {"stage": "diagnosis-motivated probe reanalysis, not a new blind test", "template_source": "deterministic augmentation of original live-GPT boxes; no new GPT inspection of training images", "template_sha256": file_sha256(template_path), "source_image_sha256": template.get("source_image_sha256"), "source_load_bbox": source_bbox.tolist(), "train_seeds": train_seeds, "probe_seeds": test_seeds, "alpha": 10.0, "training_target": "actual sum eta2; within-case log contrast", "test_target_use_for_fitting": False, "test_target_use_for_hyperparameters": False, "manifest_sha256": file_sha256(manifest_path)}  # State the adaptation history, target, and limits before collecting new evidence.
    write_json(args.output / "training_manifest.json", training_manifest)  # Save the intended training protocol before new solves.
    augmented = [train_case(args.experiment_dir, args.output, family, seed, float(manifest["growth"]), int(manifest["resolution"]), templates, source_bbox) for seed in train_seeds]  # Generate the requested twelve new authentic training solves.
    x_rows, y_rows, group_rows, identities = [], [], [], []  # Assemble old and new training transitions with explicit physical-state grouping.
    old_count = 0  # Count reused training solver observations separately from newly executed ones.
    for new_case in augmented:  # Pair each augmentation with the same original training geometry.
        seed = new_case["seed"]  # Read the training identity, never a test-selection result.
        old_case = json.loads((args.experiment_dir / f"train_{family}_{seed}" / "case.json").read_text(encoding="utf-8"))  # Read only the old training branches' features and physical labels.
        old_count += len(old_case["entries"])  # Account for already available real solver evidence.
        for origin, entries in (("existing", old_case["entries"]), ("gpt_template", new_case["entries"])):  # Give old and newly augmented actions a common target definition.
            for entry in entries:  # Append each real training transition exactly once.
                x_rows.append(entry["x"])  # Retain the same image-action-resource feature schema.
                y_rows.append(float(np.log(max(entry["eta2"], 1.0e-30))))  # Use observed estimator mass with a numerical logarithm floor only.
                group_rows.append(seed)  # Keep contrasts within one common initial physical state.
                identities.append(f"{seed}:{origin}:{entry['name']}")  # Preserve provenance for every training row.
    X, y, groups = np.asarray(x_rows), np.asarray(y_rows), np.asarray(group_rows)  # Convert the completed training library to numerical arrays.
    np.savez_compressed(args.output / "training_data.npz", X=X, log_eta2=y, groups=groups, identities=np.asarray(identities, dtype=str))  # Save actual training evidence without any test-target arrays.
    fitting_started = time.perf_counter()  # Count regression fitting separately from offline solver-data generation.
    model = fit_ranker(X, y, groups, alpha=10.0)  # Fit the requested fixed-alpha within-case action-ranking objective.
    model_path = args.output / "action_ranker.json"  # Select the immutable fitted-parameter artifact for this run.
    write_json(model_path, model)  # Freeze model parameters before reading any diagnostic test outcomes.
    fit_s = time.perf_counter() - fitting_started  # Record actual numerical fitting effort.
    model_hash = file_sha256(model_path)  # Bind every subsequent ranking to exactly this trained model.
    predictions = []  # Collect all reused-probe candidate rankings without reading their targets.
    for seed in test_seeds:  # Predict every original diagnostic GPT candidate set using only saved pre-solve inputs.
        frozen_path = args.experiment_dir.parent / f"gpt_direct_{seed}" / "predictions_before_solves.json"  # Locate previously sealed GPT candidate feature rows.
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))  # Read only original proposals, features, and earlier predictions, not outcomes.
        candidates = frozen["candidates"]  # Keep the original GPT spatial alternatives unchanged.
        scores = rank_candidates(model, np.asarray([candidate["x"] for candidate in candidates]))  # Rank all explicit GPT actions with the frozen signed-gain model.
        selected = candidates[int(np.argmin(scores))]["name"]  # Select the lowest predicted relative log error before opening targets.
        predictions.append({"seed": seed, "selected": selected, "original_primary_gpt": frozen["primary_gpt"], "original_hybrid": frozen["hybrid_world_model"], "input_sha256": file_sha256(frozen_path), "scores": [{"name": candidate["name"], "predicted_relative_log_eta2": float(score)} for candidate, score in zip(candidates, scores)]})  # Preserve every ranked alternative and both original decisions.
    prediction_path = args.output / "predictions_before_evaluation.json"  # Identify the new ranking record created before target access.
    write_json(prediction_path, {"stage": "previously diagnosed probes; not a fresh blind evaluation", "frozen_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": model_hash, "predictions": predictions})  # Save all selections before opening any reused test outcomes.
    prediction_hash = file_sha256(prediction_path)  # Bind the following diagnostic analysis to unchanged model selections.
    evaluations = []  # Allocate diagnostic outcome comparisons after prediction freeze.
    for prediction in predictions:  # Evaluate each already frozen reanalysis selection without fitting or changing it.
        seed = prediction["seed"]  # Select the corresponding reused diagnostic probe.
        outcome_path = args.experiment_dir.parent / f"gpt_direct_{seed}" / "summary.json"  # Locate actual GPT candidate outcomes only after ranking freeze.
        outcomes = json.loads(outcome_path.read_text(encoding="utf-8"))  # Read this probe's known real solver results for evaluation alone.
        choices = {entry["name"]: entry for entry in outcomes["candidates"]}  # Associate preserved candidate identities with actual physical outcomes.
        chosen = choices[prediction["selected"]]  # Retrieve the learned ranker's frozen action outcome.
        best_eta = min(choices.values(), key=lambda entry: entry["eta2"])  # Label the retrospective estimator-optimal candidate without changing the chosen action.
        baseline = json.loads((args.experiment_dir / f"test_{family}_{seed}" / "case.json").read_text(encoding="utf-8"))  # Read existing classical comparator outcomes only at the evaluation stage.
        evaluation = {**prediction, "selected_eta2": chosen["eta2"], "estimator_oracle": best_eta["name"], "estimator_regret": chosen["eta2"] / best_eta["eta2"] - 1.0, "original_primary_eta2": choices[prediction["original_primary_gpt"]]["eta2"], "original_hybrid_eta2": choices[prediction["original_hybrid"]]["eta2"], "selected_equations": chosen["record"]["n_equations"]}  # Report direct evidence of estimator-ranking improvement or failure.
        if "reference_error" in chosen:  # Add secondary physical-error comparisons only when the original experiment actually computed them.
            dorfler = min((entry for entry in baseline["entries"] if entry["name"].startswith("dorfler")), key=lambda entry: entry["reference_error"])  # Retain the explicitly retrospective best-of-three Dörfler reference-error envelope.
            evaluation.update({"selected_reference_error": chosen["reference_error"], "dorfler_reference_oracle": dorfler["name"], "dorfler_reference_error": dorfler["reference_error"], "reference_ratio_to_dorfler_oracle": chosen["reference_error"] / dorfler["reference_error"], "original_primary_reference_error": choices[prediction["original_primary_gpt"]]["reference_error"], "original_hybrid_reference_error": choices[prediction["original_hybrid"]]["reference_error"]})  # Keep reference accuracy distinct from the ranker's estimator-mass training objective.
        evaluations.append(evaluation)  # Preserve the case-level diagnostic comparison without selecting favorable cases.
    report = {"protocol": training_manifest, "model_sha256": model_hash, "predictions_before_evaluation_sha256": prediction_hash, "old_training_transitions": old_count, "new_training_transitions": sum(len(case["entries"]) for case in augmented), "new_training_solver_calls": sum(len(case["entries"]) for case in augmented), "total_training_transitions": int(len(X)), "training_fit_s": fit_s, "training_centered_log_eta2_rmse": model["training_centered_log_eta2_rmse"], "probe_reanalysis": evaluations, "new_blind_generalization_claim": False, "total_local_s": time.perf_counter() - began}  # Summarize actual new training work and explicitly limited diagnostic findings.
    write_json(args.output / "summary.json", report)  # Save the complete requested training and reanalysis report.
    print(json.dumps(report, indent=2), flush=True)  # Return measured outcomes without rewriting original GPT decisions.
if __name__ == "__main__":  # Execute only on an explicit training-command invocation.
    main()  # Run augmentation, frozen ranking, and diagnostic reanalysis in the declared order.
