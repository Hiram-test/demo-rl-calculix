#!/usr/bin/env python3  # Audit archived predictions without running a finite-element solver.
"""Intervene on WM input images while holding archived candidate actions fixed."""  # Describe the deliberately limited causal control.
from __future__ import annotations  # Support deferred type annotations.
import argparse  # Expose the experiment output directory and explicit old test seeds.
from dataclasses import replace  # Change only the recorded image in an otherwise fixed visual state.
import hashlib  # Bind decisions to exact archived numerical inputs.
import json  # Store predictions before loading outcome metrics.
from pathlib import Path  # Resolve source and experiment paths.
import sys  # Import the repository correctly from any working directory.
import numpy as np  # Reconstruct numerical states and compare controlled interventions.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository independently of the current directory.
sys.path.insert(0, str(ROOT))  # Make the actual project modules importable.
sys.path.insert(0, str(ROOT / "scripts"))  # Resolve the existing experiment helper by its real location.
from run_visual_world_experiment import case_problem, equations, scalar_features  # Reuse seeded geometry and the exact existing resource and scalar-feature definitions.
from visionamr.fem_post import compute_post  # Recover physical fields from archived displacement without a new solve.
from visionamr.indicators import zz_indicator  # Recompute the original observable estimator independently.
from visionamr.mesher import Mesh  # Restore real nodal coordinates and element connectivity.
from visionamr.vla.visual_spatial import _view, features, rasterize  # Apply image interventions before WM feature extraction rather than before target decoding.
from visionamr.vla.visual_world_model import SpatialWorldModel  # Load the original frozen visual and separately trained scalar predictors.

def write_json(path, value):  # Persist numerical audit evidence atomically within the same directory.
    path = Path(path)  # Normalize the output filename.
    temporary = path.with_suffix(path.suffix + ".tmp")  # Avoid leaving a partially written JSON result after interruption.
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # Reject nonfinite values instead of emitting invalid evidence.
    temporary.replace(path)  # Publish the complete local audit artifact.

def digest(array):  # Bind an intervention to exact numerical contents and dimensions.
    value = np.ascontiguousarray(array, dtype="<f8")  # Normalize numerical serialization without modifying the input.
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()  # Include shape so differently arranged arrays have distinct identities.

def selection(names, totals, indices):  # Choose from a declared candidate subset using predictions only.
    if not indices:  # Permit archives without a visual-only action subset.
        return None  # Report the missing subset rather than inventing an action.
    winner = min(indices, key=lambda index: (float(totals[index]), names[index]))  # Resolve exact ties deterministically by action name.
    return names[winner]  # Return the fixed action identity selected by the model.

def seal_case(path, family, seed, visual_model, scalar_model):  # Construct every intervention without reading case.json or any physical branch outcome.
    problem = case_problem(family, seed)  # Restore the exact seeded problem used by the original experiment.
    with np.load(path / "initial.npz", allow_pickle=False) as initial:  # Read only the initial state available to the original decision maker.
        mesh = Mesh(initial["nodes"].copy(), initial["cells"].copy(), problem.dim)  # Restore the solved common initial mesh.
        post = compute_post(mesh, problem, initial["u"].copy())  # Recompute physical postprocessing from the original displacement solution.
        archived_eta = initial["eta2"].copy()  # Retain the archived estimator for a reconstruction consistency check.
    eta = zz_indicator(problem, post)  # Recover the current-state estimator from the physical solution.
    with np.load(path / "observation.npz", allow_pickle=False) as observation:  # Read the exact original image supplied to candidate generation.
        original_image = observation["image"].copy()  # Preserve every recorded image value without regenerating the visual input.
        original_bbox = tuple(observation["bbox"].tolist())  # Retain the original physical coordinate frame.
    state = rasterize(problem, post, eta, resolution=original_image.shape[1])  # Rebuild the sparse mesh-to-image correspondence required by feature extraction.
    reconstruction = {"eta2_max_abs_difference": float(np.max(np.abs(eta - archived_eta))), "image_max_abs_difference": float(np.max(np.abs(state.image - original_image))), "bbox_matches": bool(np.allclose(state.bbox, original_bbox, rtol=0.0, atol=1e-12))}  # Report reconstruction differences explicitly without silently replacing archived data.
    state = replace(state, image=original_image, bbox=original_bbox)  # Use the exact archived original image with reconstructed physical transport.
    initial_equations = equations(problem, mesh)  # Recompute initial active equations solely from geometry and constraints.
    target_paths = sorted(path.glob("target_*.npz"))  # Freeze the candidate dictionary before any outcome access.
    names, targets, resources, candidate_hashes = [], [], [], []  # Record shared actions and resources for every intervention.
    for target_path in target_paths:  # Restore each executed source-mesh sizing field once.
        name = target_path.stem[len("target_"):]  # Recover the original action identity from its artifact name.
        with np.load(target_path, allow_pickle=False) as target_data:  # Read the exact target rather than decoding a new action from an altered image.
            target = target_data["target"].copy()  # Keep the target fixed under every WM image intervention.
        branch_path = path / f"post_{name}.npz"  # Locate the archived candidate mesh used for pre-solve resource counting.
        if not branch_path.exists():  # Report an incomplete archive before producing a misleading candidate set.
            return None, f"missing archived candidate mesh: {branch_path.name}"  # Let the caller retry after the ongoing experiment finishes.
        with np.load(branch_path, allow_pickle=False) as branch:  # Read mesh geometry only; do not access future u, stress, eta2, vm, or energy arrays.
            candidate_mesh = Mesh(branch["nodes"].copy(), branch["cells"].copy(), problem.dim)  # Recover quantities already known at original pre-solve materialization time.
        names.append(name)  # Freeze candidate ordering across all predictors.
        targets.append(target)  # Preserve the actual spatial sizing field.
        resources.append(equations(problem, candidate_mesh))  # Hold the same realized active-equation count fixed in every feature vector.
        candidate_hashes.append(digest(target))  # Bind all intervention decisions to unchanged target contents.
    eligible = [index for index, name in enumerate(names) if not name.startswith("image_")]  # Match the original deployment candidate set while retaining decoder controls for inspection.
    visual_only = [index for index, name in enumerate(names) if name.startswith("visual_")]  # Separately test decisions restricted to genuinely image-produced candidates.
    if not eligible:  # Handle test cases whose candidate generation has not completed.
        return None, "no complete deployment candidates"  # Report a normal pending result.
    predictions, original_features = {}, None  # Collect sealed model outputs before evaluating any branch.
    for mode in ("visual", "shuffled", "constant"):  # Intervene on WM observation content while preserving all action inputs.
        image = original_image if mode == "visual" else _view(state, mode)  # Apply the existing deterministic OOD image controls to WM input.
        changed_state = replace(state, image=image)  # Keep geometry, transport, scalar state, and original total eta2 unchanged.
        rows = np.asarray([np.r_[features(changed_state, target, mesh), np.log1p(count), count / initial_equations] for target, count in zip(targets, resources)])  # Recompute action-conditioned WM features using identical fixed targets and resources.
        predicted = visual_model.predict(rows)  # Query only the already frozen visual model.
        totals = predicted.sum(axis=1)  # Use the original predicted total squared-estimator objective.
        label = "original" if mode == "visual" else mode  # Distinguish the factual WM input from the two intervention labels.
        if original_features is None:  # Preserve the factual feature matrix for numerical sensitivity measurements.
            original_features = rows.copy()  # Avoid allowing later interventions to mutate the reference features.
        predictions[label] = {"image_sha256": digest(image), "feature_sha256": digest(rows), "feature_l2_difference_from_original": float(np.linalg.norm(rows - original_features)), "predicted_total_eta2_ratio": totals.tolist(), "selected": selection(names, totals, eligible), "selected_visual_only": selection(names, totals, visual_only)}  # Seal every intervention decision without using realized errors.
    if scalar_model is not None:  # Include the separately trained no-image model when its archived snapshot is available.
        scalar_rows = np.asarray([scalar_features(post, eta, target, count) for target, count in zip(targets, resources)])  # Use identical fixed candidate maps and measured resources with spatial coordinates removed.
        totals = scalar_model.predict(scalar_rows).sum(axis=1)  # Query the archived model retrained from scratch on scalar features.
        predictions["scalar_retrained"] = {"feature_sha256": digest(scalar_rows), "predicted_total_eta2_ratio": totals.tolist(), "selected": selection(names, totals, eligible), "selected_visual_only": selection(names, totals, visual_only)}  # Record the meaningful learned-feature ablation separately from OOD interventions.
    result = {"case": path.name, "family": family, "seed": seed, "reconstruction": reconstruction, "initial_equations": initial_equations, "candidate_names": names, "candidate_equations": resources, "target_sha256": candidate_hashes, "deployment_candidates": [names[index] for index in eligible], "visual_only_candidates": [names[index] for index in visual_only], "predictions": predictions, "candidate_resource_source": "Archived branch mesh nodes/cells only; these geometric resources were known before the original solve."}  # Preserve all fixed inputs and predictions in a self-contained sealed decision record.
    return result, None  # Return predictions without any physical branch metric.

def assess_case(path, sealed):  # Read physical outcomes only after the caller has persisted every case's predictions.
    outcome = json.loads((path / "case.json").read_text(encoding="utf-8"))  # Open realized branch results for the first time in this audit.
    entries = {entry["name"]: entry for entry in outcome["entries"]}  # Match physical results by action identity rather than incidental ordering.
    available = [name for name in sealed["deployment_candidates"] if name in entries]  # Evaluate the same declared deployment set.
    metrics = {"eta_norm": lambda entry: float(np.sqrt(entry["eta2"]))}  # Report estimator norm rather than confusing squared indicators with error norm.
    if available and all("reference_error" in entries[name] for name in available):  # Use the common reference metric only when present for every compared action.
        metrics["reference_error"] = lambda entry: float(entry["reference_error"])  # Keep independently integrated physical accuracy separate from the training target.
    result = {"case": sealed["case"], "metrics": {}, "input_intervention_changes_selection": {}, "input_intervention_changes_visual_only_selection": {}}  # Initialize factual comparisons without changing any sealed choice.
    for metric, evaluate in metrics.items():  # Report both optimization-target and independent-reference performance where available.
        values = {name: evaluate(entries[name]) for name in available}  # Evaluate physical outcomes only within the fixed candidate set.
        dorfler = [name for name in available if name.startswith("dorfler_")]  # Identify the several tested classical bulk parameters.
        best = min(available, key=lambda name: (values[name], name))  # Compute an explicitly hindsight-only candidate oracle for regret evaluation.
        best_dorfler = min(dorfler, key=lambda name: (values[name], name)) if dorfler else None  # Compute the strongest measured classical candidate for honest comparison.
        report = {"candidate_values": values, "oracle_action": best, "oracle_value": values[best], "best_dorfler_action": best_dorfler, "best_dorfler_value": None if best_dorfler is None else values[best_dorfler], "selections": {}}  # Identify all reference choices as evaluation-only quantities.
        for label, prediction in sealed["predictions"].items():  # Score each frozen model and image intervention without refitting.
            selections = {}  # Keep shared-candidate and visual-only decisions distinct.
            for scope, key in (("shared_candidates", "selected"), ("visual_only", "selected_visual_only")):  # Evaluate both predeclared action subsets.
                name = prediction[key]  # Read the action already sealed before this outcome access.
                if name is None or name not in values:  # Handle an absent visual-only subset without fabricating a comparison.
                    selections[scope] = None  # Preserve the missing-result status.
                    continue  # Move to the next available selection scope.
                value = values[name]  # Read the realized accuracy of the selected fixed action.
                selections[scope] = {"action": name, "value": value, "relative_regret_to_shared_oracle": value / max(values[best], 1e-30) - 1.0, "ratio_to_best_dorfler": None if best_dorfler is None else value / max(values[best_dorfler], 1e-30), "ratio_to_analytic_density": None if "analytic_density" not in values else value / max(values["analytic_density"], 1e-30)}  # Report regret and strong-baseline ratios without calling a hindsight winner a deployed policy.
            report["selections"][label] = selections  # Store each frozen selector's measured outcome.
        original = report["selections"]["original"]  # Identify the factual fixed-model decision as the intervention reference.
        for label in ("shuffled", "constant"):  # Express whether changing the WM input helps or harms the frozen policy on these cases.
            for scope in ("shared_candidates", "visual_only"):  # Preserve both candidate scopes in intervention regret.
                changed, factual = report["selections"][label][scope], original[scope]  # Read only already evaluated frozen decisions.
                if changed is not None and factual is not None:  # Report a difference only for comparable decisions.
                    changed["relative_regret_change_from_original"] = changed["relative_regret_to_shared_oracle"] - factual["relative_regret_to_shared_oracle"]  # Keep the sign so an intervention that improves a weak model remains visible.
                    changed["value_ratio_to_original_selection"] = changed["value"] / max(factual["value"], 1e-30)  # Quantify realized intervention impact without implying in-distribution generalization.
        result["metrics"][metric] = report  # Store the complete independent metric table.
    for label in ("shuffled", "constant"):  # Record direct model-choice sensitivity independently of whether it improved physics.
        result["input_intervention_changes_selection"][label] = sealed["predictions"][label]["selected"] != sealed["predictions"]["original"]["selected"]  # Measure fixed-candidate decision dependence on WM image input.
        result["input_intervention_changes_visual_only_selection"][label] = sealed["predictions"][label]["selected_visual_only"] != sealed["predictions"]["original"]["selected_visual_only"]  # Measure dependence within the visual-only action subset.
    return result  # Return retrospective evaluations of previously sealed model decisions.

def main(argv=None):  # Provide a repeatable archive audit with no native solver calls.
    parser = argparse.ArgumentParser(description=__doc__)  # Expose the audit's limited experimental scope.
    parser.add_argument("--output-dir", required=True, type=Path)  # Read the existing experiment directory explicitly.
    parser.add_argument("--seeds", default="901,902,903")  # Limit the default audit to the three original held-out cases.
    args = parser.parse_args(argv)  # Parse command-line settings without modifying the experiment.
    root = args.output_dir.resolve()  # Resolve all archived paths consistently.
    if not root.exists():  # Treat a run that has not started as a normal no-results state.
        print(json.dumps({"status": "no_results", "reason": "output directory does not exist"}))  # Explain the absence of evidence without an exception.
        return 0  # Exit successfully so an ongoing experiment can be audited later.
    manifest_path = root / "manifest.json"  # Locate the original experimental configuration.
    if not manifest_path.exists() or not (root / "visual_world_model.json").exists():  # Permit invocation while training is still running.
        print(json.dumps({"status": "no_results", "reason": "manifest or fitted visual model is not available"}))  # Report a pending audit plainly.
        return 0  # Avoid blocking or rerunning training.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # Read configuration without any held-out branch outcomes.
    family = manifest["family"]  # Reuse the original physical problem family.
    requested = [int(value) for value in args.seeds.split(",") if value.strip()]  # Keep the explicit seed list in its requested order.
    paths = [(root / f"test_{family}_{seed}", seed) for seed in requested]  # Select only the declared old test cases.
    completed = [(path, seed) for path, seed in paths if (path / "case.json").exists() and (path / "initial.npz").exists() and (path / "observation.npz").exists()]  # Check completion by file existence without reading physical metrics.
    if not completed:  # Handle a training-only or incomplete archive gracefully.
        print(json.dumps({"status": "no_results", "reason": "no completed requested test cases"}))  # State that there are no results to evaluate yet.
        return 0  # Leave the original running experiment unaffected.
    visual_model = SpatialWorldModel.load(root / "visual_world_model.json")  # Load the frozen original visual predictor without retraining.
    scalar_path = root / "scalar_world_model.json"  # Locate the separately trained scalar ablation.
    scalar_model = SpatialWorldModel.load(scalar_path) if scalar_path.exists() else None  # Keep a missing ablation explicit instead of substituting a zero-image model.
    sealed, skipped = [], []  # Accumulate decisions for all cases before reading any outcomes.
    for path, seed in completed:  # Reconstruct only requested completed old cases.
        prediction, reason = seal_case(path, family, seed, visual_model, scalar_model)  # Freeze fixed-candidate intervention choices using initial observations only.
        if prediction is None:  # Preserve incomplete candidate archives as pending cases.
            skipped.append({"case": path.name, "reason": reason})  # Record why that case could not yet be audited.
        else:  # Keep complete prediction sets ready for sealing.
            sealed.append(prediction)  # Delay all physical evaluation until every decision has been written.
    notes = {"scope": "Retrospective fixed-model input intervention on archived one-step candidates; predictions are written before this script reads case.json outcomes, not a new prospective trial.", "fixed_inputs": "All target arrays, mesh transport, scalar state, original total eta2 and actual candidate equations are held fixed; only WM image input changes.", "ood_controls": "Shuffled and constant images are out-of-distribution sensitivity controls; changed choices alone do not prove useful visual understanding or generalization.", "scalar_control": "The scalar snapshot was independently retrained from scratch on the original shared training branches and is evaluated on the same fixed candidate set.", "decoder_controls": "Archived image_shuffled/image_constant actions alter the decoder; they remain recorded but are excluded from deployment choices as in the original experiment.", "reference_baselines": "Best Dorfler and best-candidate oracle are hindsight evaluation references, never model-selection inputs."}  # State the evidential limitations directly in the machine-readable artifact.
    decisions = {"status": "predictions_sealed" if sealed else "no_results", "notes": notes, "requested_seeds": requested, "visual_model_sha256": hashlib.sha256((root / "visual_world_model.json").read_bytes()).hexdigest(), "scalar_model_sha256": hashlib.sha256(scalar_path.read_bytes()).hexdigest() if scalar_path.exists() else None, "cases": sealed, "skipped": skipped}  # Bind the audit to exact frozen models and explicit test scope.
    write_json(root / "visual_causality_decisions.json", decisions)  # Persist every prediction before the first case.json outcome is read.
    assessed = [assess_case(root / item["case"], item) for item in sealed]  # Only now evaluate the previously frozen predictions against archived physics.
    report = {"status": "complete" if assessed else "no_results", "notes": notes, "cases": assessed, "skipped": skipped, "decisions_file": "visual_causality_decisions.json"}  # Keep decisions and retrospective outcomes in separate inspectable artifacts.
    write_json(root / "visual_causality_audit.json", report)  # Persist the complete causal sensitivity and realized regret report.
    print(json.dumps({"status": report["status"], "cases": len(assessed), "decisions": str(root / "visual_causality_decisions.json"), "audit": str(root / "visual_causality_audit.json")}))  # Report artifact completion without overselling visual causality.
    return 0  # Finish without launching training, remeshing, or CalculiX.

if __name__ == "__main__":  # Permit ordinary standalone execution.
    raise SystemExit(main())  # Return a conventional command-line exit status.
