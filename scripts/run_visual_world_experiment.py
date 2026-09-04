#!/usr/bin/env python3  # Run real CalculiX counterfactual experiments.
"""Visual spatial world-model experiment with held-out geometric cases."""  # State the evidence scope.
from __future__ import annotations  # Support modern type hints.
import argparse  # Parse reproducible experiment options.
import json  # Store readable observations and decisions.
import os  # Configure reproducible native threading.
import sys  # Locate the repository package.
import time  # Count end-to-end costs.
from dataclasses import asdict  # Serialize exact solver records.
from pathlib import Path  # Manage experiment artifacts.
import numpy as np  # Perform numerical operations.
from scipy.spatial import cKDTree  # Locate reference integration points.
ROOT = Path(__file__).resolve().parents[1]  # Resolve the source root.
sys.path.insert(0, str(ROOT))  # Import the restored project.
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Keep solver threads consistent.
from visionamr.experiment import FemRunner, initial_mesh  # Reuse genuine solver accounting.
from visionamr.geometry import sample_bearing_block, sample_deck_panel  # Import bridge component families.
from visionamr.fem_post import elastic_C  # Use the same constitutive tensor for error integration.
from visionamr.indicators import zz_indicator  # Estimate currently observable discretization error.
from visionamr.marking import dorfler_mark  # Import a strong classical marking baseline.
from visionamr.baselines.dorfler import refine_size_map  # Preserve the existing marking action.
from visionamr.mesher import generate_mesh, generate_uniform, Mesh  # Use identical Gmsh realization for every method.
from visionamr.sizefield import NodalSizeField, element_to_node_sizes  # Compile common continuous fields.
from visionamr.vla.visual_spatial import rasterize, make_targets, features, spatial_error  # Import the image-driven action interface.
from visionamr.vla.visual_world_model import SpatialWorldModel  # Import the trained action-conditional transition model.

def write_json(path, value):  # Save complete numerical evidence after each step.
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # Create the containing directory.
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))  # Preserve readable measurements.

def case_problem(family, seed):  # Make reproducible physically distinct problem instances.
    rng = np.random.default_rng(seed)  # Isolate the geometry seed from all model randomness.
    return sample_bearing_block(rng) if family == "bearing" else sample_deck_panel(rng)  # Select the real bridge family.

def equations(problem, mesh):  # Count free displacement unknowns independently of the solver.
    fixed = np.zeros((mesh.n_nodes, mesh.dim), dtype=bool)  # Allocate the constraint mask.
    for item in problem.constraints:  # Apply all actual boundary conditions.
        selected = item.node_predicate(mesh.nodes)  # Evaluate the mesh-independent support footprint.
        for dof in item.dofs:  # Respect one-based component numbering.
            fixed[selected, dof - 1] = True  # Mark the prescribed displacement component.
    return int(fixed.size - fixed.sum())  # Return actual free degrees of freedom.

def normalize_target(mesh, target, growth, problem):  # Give every action the same nominal resource growth.
    target = np.asarray(target) * np.median(mesh.node_sizes) / max(np.median(target), 1e-30)  # Normalize arbitrary size-field units before the common resource search.
    desired = float(growth) * mesh.n_cells  # Set the common element-volume prediction.
    low, high = 0.1, 10.0  # Bracket a uniform multiplier on the proposed spatial pattern.
    for _ in range(35):  # Solve the monotone sizing equation without physics queries.
        scale = np.sqrt(low * high)  # Bisect in logarithmic size space.
        h = np.clip(np.asarray(target) * scale, problem.h_min, problem.h0)  # Enforce shared physical mesh bounds.
        count = np.sum((mesh.cell_sizes / h[mesh.cells].mean(axis=1)) ** mesh.dim)  # Predict new cell count under that field.
        low, high = (scale, high) if count > desired else (low, scale)  # Move toward the common growth budget.
    return np.clip(np.asarray(target) * np.sqrt(low * high), problem.h_min, problem.h0)  # Return the budget-normalized field.

def candidates(problem, post, eta, state, growth):  # Build shared spatial alternatives before any future solve.
    mesh = post.mesh  # Read the actually solved current mesh.
    result = {}  # Keep explicit action provenance.
    for theta in (0.2, 0.5, 0.8):  # Include several substantial classical bulk parameters.
        result[f"dorfler_{theta:.1f}"] = refine_size_map(mesh, dorfler_mark(eta, theta))  # Apply the repository's exact local marking.
    result["uniform"] = mesh.node_sizes * growth ** (-1.0 / mesh.dim)  # Add uniform refinement as a reference.
    coefficient = eta / np.maximum(mesh.measures * mesh.cell_sizes ** 2, 1e-30)  # Recover the leading linear-FE local error coefficient.
    coefficient = np.maximum(coefficient, np.max(coefficient) * 1e-8)  # Regularize regions with numerically zero indicators.
    result["analytic_density"] = element_to_node_sizes(mesh, coefficient ** (-1.0 / (mesh.dim + 2.0)))  # Add classical continuous error equidistribution.
    images = make_targets(state, mesh, growth=growth, mode="visual", goal="energy")  # Let image contents generate independent locations.
    for name in ("err_s0", "err_s1", "err_s2", "err_focused", "err_diffuse"):  # Keep a fixed modest action dictionary.
        if name in images:  # Accept the explicit visual-module naming contract.
            result["visual_" + name] = images[name]  # Retain each image-produced size field.
    for mode in ("shuffled", "constant"):  # Generate intervention controls through the identical decoder.
        control = make_targets(state, mesh, growth=growth, mode=mode, goal="energy")  # Alter spatial information while preserving the action architecture.
        result["image_" + mode] = control["err_s1"]  # Use the same central smoothing scale for both controls.
    return {name: normalize_target(mesh, target, growth, problem) for name, target in result.items()}  # Apply one common budget rule to all methods.

def materialize(problem, mesh, target, cap):  # Measure Gmsh's real resource cost before solving.
    started = time.perf_counter()  # Include all candidate remeshing attempts.
    scale, best, attempts = 1.0, None, []  # Track closest feasible realization and exact effort.
    for iteration in range(5):  # Permit the same bounded resource correction for every method.
        scaled = np.clip(target * scale, problem.h_min, problem.h0)  # Preserve the spatial action under global normalization.
        field = NodalSizeField(mesh, scaled, gradation=0.9, h_min=problem.h_min, h_max=problem.h0)  # Compile the common continuous Gmsh field.
        candidate = generate_mesh(problem, field)  # Generate a real conformal candidate mesh.
        count = equations(problem, candidate)  # Count actual free displacement equations.
        attempts.append({"iteration": iteration, "n_equations": count, "scale": scale})  # Retain the complete mesh-only search cost.
        if count <= cap and (best is None or count > best[0]):  # Keep the best budget use among feasible candidates.
            best = (count, candidate, field._h.copy())  # Record the actual post-gradation source field.
        if count <= cap and count >= 0.96 * cap:  # Finish when the common budget is used closely.
            break  # Avoid optional mesh work once the concrete uncertainty is resolved.
        scale *= max(0.3, count / float(cap)) ** (1.0 / mesh.dim) * 1.01  # Correct observed size-count mismatch conservatively.
    if best is None:  # Handle a true inability to generate within the requested budget.
        raise RuntimeError(f"No feasible mesh under {cap} free equations; attempts={attempts}")  # Retain failure rather than accepting overshoot silently.
    return best[1], best[2], {"mesh_s": time.perf_counter() - started, "attempts": attempts, "n_equations": best[0]}  # Return measured resources and executed field.

def scalar_features(post, eta, target, new_equations):  # Build a retrainable statistics-only ablation.
    ratio = target / np.maximum(post.mesh.node_sizes, 1e-30)  # Describe action magnitude without its spatial arrangement.
    q = np.quantile(ratio, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))  # Retain a rich distributional action summary.
    return np.r_[np.log1p([eta.sum(), post.mesh.n_cells, post.mesh.n_nodes, new_equations]), q, np.quantile(eta / max(eta.sum(), 1e-30), (0.25, 0.5, 0.9, 1.0))]  # Exclude positions and images consistently in training and test.

def save_post(path, post, eta):  # Keep original fields needed for later audit and plotting.
    np.savez_compressed(path, nodes=post.mesh.nodes, cells=post.mesh.cells, u=post.u, stress=post.stress, eta2=eta, vm=post.vm_elem, energy=post.energy_elem)  # Save authentic solver-derived numerical arrays.

def reference_error(problem, coarse_post, fine_post):  # Evaluate a reference-field energy error on common fine-mesh quadrature.
    mesh, reference = coarse_post.mesh, fine_post.mesh  # Unpack the two non-nested real meshes.
    centers = reference.centroids  # Use fixed reference-cell centroid quadrature for every method.
    tree = cKDTree(mesh.centroids)  # Restrict expensive containment searches to nearby tetrahedra.
    vertices = mesh.nodes[mesh.cells]  # Gather candidate-cell nodal coordinates.
    inverse = np.linalg.inv(np.stack([vertices[:, i] - vertices[:, 0] for i in range(1, 4)], axis=2))  # Invert real tetrahedral coordinate maps.
    located = np.empty(reference.n_cells, dtype=int)  # Allocate the coarse-cell mapping for every fine quadrature point.
    missed = 0  # Count interpolation approximations explicitly.
    for start in range(0, reference.n_cells, 2000):  # Limit the point-location working memory.
        xyz = centers[start:start + 2000]  # Select a reproducible block of fine quadrature points.
        _, index = tree.query(xyz, k=min(64, mesh.n_cells))  # Query nearby physical cells rather than unconstrained Delaunay simplices.
        index = np.atleast_2d(index)  # Preserve batch dimensions for tiny smoke cases.
        bary = np.einsum("bkij,bkj->bki", inverse[index], xyz[:, None, :] - vertices[index, 0, :])  # Recover barycentric coordinates in each possible cell.
        valid = (bary.min(axis=2) >= -1e-7) & (bary.sum(axis=2) <= 1.0 + 1e-7)  # Test actual containment including shared-face tolerance.
        found = valid.any(axis=1)  # Identify every exactly located quadrature point.
        chosen = np.argmax(valid, axis=1)  # Select the first containing physical cell.
        located[start:start + len(xyz)] = index[np.arange(len(xyz)), chosen]  # Use the nearest cell only for explicitly counted misses.
        missed += int((~found).sum())  # Preserve the approximation fraction for interpretation.
    difference = fine_post.stress - coarse_post.stress[located]  # Compare stresses at identical physical points.
    compliance = np.linalg.inv(elastic_C(problem.material, 3))  # Convert stress difference to elastic energy norm.
    numerator = np.sum(np.einsum("mi,ij,mj->m", difference, compliance, difference) * reference.measures)  # Integrate reference-field discrepancy.
    denominator = np.sum(np.einsum("mi,ij,mj->m", fine_post.stress, compliance, fine_post.stress) * reference.measures)  # Normalize with the reference solution energy norm.
    return float(np.sqrt(numerator / max(denominator, 1e-30))), float(missed / reference.n_cells)  # Return measured approximate norm and mapping uncertainty.

def collect_case(args, seed, split, model=None, scalar_model=None):  # Execute same-state action forks with sealed prior predictions.
    path = args.output / f"{split}_{args.family}_{seed}"  # Keep every geometric case independent.
    path.mkdir(parents=True, exist_ok=True)  # Create the case directory.
    if (path / "case.json").exists():  # Reuse fully completed solver evidence after interruption.
        return json.loads((path / "case.json").read_text())  # Never rerun identical expensive branches without need.
    problem = case_problem(args.family, seed)  # Construct the seeded physical geometry and load.
    runner = FemRunner(problem, path, keep_files=True)  # Record every authentic CalculiX invocation.
    began = time.perf_counter()  # Measure common initialization overhead.
    mesh = initial_mesh(problem)  # Give every compared method the identical initial mesh.
    post, initial_record = runner.solve_mesh(mesh, method="common", stage="initial")  # Solve the common initial state exactly once.
    eta = zz_indicator(problem, post)  # Derive only currently available error indicators.
    state = rasterize(problem, post, eta, resolution=args.resolution)  # Build the actual spatial observation.
    initial_s = time.perf_counter() - began  # Include solve, estimation, and visual rasterization.
    save_post(path / "initial.npz", post, eta)  # Retain the actual initial field and mesh.
    np.savez_compressed(path / "observation.npz", image=state.image, bbox=state.bbox)  # Keep the exact image consumed by decision generation.
    cap = int(equations(problem, mesh) * args.growth)  # Set a common actual free-equation cap.
    proposals = candidates(problem, post, eta, state, args.growth)  # Materialize the fixed candidate dictionary from present information.
    entries, meshes = [], []  # Allocate branch metadata and unsolved candidate meshes.
    for name, target in proposals.items():  # Build all actions before seeing any branch result.
        candidate, executed, cost = materialize(problem, mesh, target, cap)  # Count real Gmsh closure and budget overhead.
        x = features(state, executed, mesh)  # Encode actual image-action spatial interactions.
        x = np.r_[x, np.log1p(cost["n_equations"]), cost["n_equations"] / initial_record.n_equations]  # Include measured candidate resource use.
        xs = scalar_features(post, eta, executed, cost["n_equations"])  # Build the no-image alternative with the same resources.
        entries.append({"name": name, "x": x.tolist(), "xs": xs.tolist(), "cost": cost})  # Persist only present-state decision inputs.
        meshes.append(candidate)  # Hold the unsolved branch mesh.
        np.savez_compressed(path / f"target_{name}.npz", target=executed)  # Save the exact executable action map.
    eligible = [i for i, entry in enumerate(entries) if not entry["name"].startswith("image_")]  # Keep destructive image interventions as separate controls.
    decisions = {}  # Freeze deployment decisions before future field queries.
    if model is not None:  # Use only the already trained transition model on held-out cases.
        prediction = model.predict(np.asarray([entry["x"] for entry in entries]))  # Predict future spatial error under all unsolved actions.
        totals = np.sum(prediction, axis=1)  # Aggregate predicted spatial error for the current energy objective.
        winner = min(eligible, key=lambda i: float(totals[i]))  # Select the minimum predicted next error without branch feedback.
        decisions["visual_wm"] = entries[winner]["name"]  # Record the causal visual-world-model action.
        decisions["visual_predictions"] = totals.tolist()  # Preserve all predictions for subsequent ranking audit.
    if scalar_model is not None:  # Evaluate a separately fitted statistics-only model.
        prediction = scalar_model.predict(np.asarray([entry["xs"] for entry in entries]))  # Use identical training branches with images removed.
        totals = np.sum(prediction, axis=1)  # Aggregate its spatial predictions by the same rule.
        winner = min(eligible, key=lambda i: float(totals[i]))  # Choose its own action without future feedback.
        decisions["scalar_wm"] = entries[winner]["name"]  # Record the scalar model decision.
        decisions["scalar_predictions"] = totals.tolist()  # Keep evidence of its counterfactual ranking.
    write_json(path / "decisions_before_solves.json", decisions)  # Seal held-out actions before any counterfactual solve.
    for entry, candidate in zip(entries, meshes):  # Query real branches only after decisions are frozen.
        started = time.perf_counter()  # Count this branch's solver and post-processing time.
        following, record = runner.solve_mesh(candidate, method=entry["name"], stage="branch")  # Execute authentic linear elasticity on the candidate.
        next_eta = zz_indicator(problem, following)  # Measure the resulting error estimator.
        next_state = rasterize(problem, following, next_eta, resolution=args.resolution)  # Rasterize the actual future field for WM supervision.
        entry.update({"record": asdict(record), "eta2": float(next_eta.sum()), "y": (spatial_error(next_state) / max(eta.sum(), 1e-30)).tolist(), "branch_s": time.perf_counter() - started})  # Store action-conditioned spatial targets and measured resources.
        save_post(path / f"post_{entry['name']}.npz", following, next_eta)  # Keep physical branch fields for independent checks.
        write_json(path / "partial.json", {"entries": entries, "decisions": decisions})  # Checkpoint each costly solver result.
        print(json.dumps({"split": split, "seed": seed, "action": entry["name"], "equations": record.n_equations, "eta": float(np.sqrt(next_eta.sum()))}), flush=True)  # Stream real progress without invented conclusions.
    result = {"seed": seed, "family": args.family, "split": split, "params": problem.params, "cap": cap, "initial": asdict(initial_record), "initial_eta2": float(eta.sum()), "initial_s": initial_s, "entries": entries, "decisions": decisions}  # Assemble a complete same-state branching experiment.
    if split == "test" and args.reference:  # Add independent reference-field accuracy only after predictions and branch results are fixed.
        fine_mesh = generate_uniform(problem, problem.h0 / args.reference_factor)  # Use a shared fine mesh without method-specific hotspot information.
        fine_post, fine_record = runner.solve_mesh(fine_mesh, method="reference", stage="fine")  # Count the reference solve separately from online decisions.
        save_post(path / "reference.npz", fine_post, zz_indicator(problem, fine_post))  # Retain the common reference field.
        result["reference"] = asdict(fine_record)  # Record exact reference resolution and runtime.
        for entry in entries:  # Compare every branch on one common fine integration mesh.
            data = np.load(path / f"post_{entry['name']}.npz")  # Reload the unmodified actual branch field.
            branch_post = type("StressField", (), {"mesh": Mesh(data["nodes"], data["cells"], 3), "stress": data["stress"]})()  # Supply only the arrays required by independent reference integration.
            entry["reference_error"], entry["reference_mapping_miss"] = reference_error(problem, branch_post, fine_post)  # Compute the non-nested reference-field error honestly.
    write_json(path / "case.json", result)  # Commit the complete reproducible case evidence.
    return result  # Return measurements for fitting or assessment.

def main():  # Run fixed-split training and held-out evaluation.
    parser = argparse.ArgumentParser(description=__doc__)  # Expose a reproducible command interface.
    parser.add_argument("--output", type=Path, required=True)  # Choose the experiment artifact directory.
    parser.add_argument("--family", choices=("bearing", "deck"), default="bearing")  # Select the bridge component family.
    parser.add_argument("--train-seeds", default="101,102,103,104")  # Separate training geometries explicitly.
    parser.add_argument("--test-seeds", default="901,902,903")  # Hold out entire physical instances.
    parser.add_argument("--growth", type=float, default=2.0)  # Set the shared finite resource increase.
    parser.add_argument("--resolution", type=int, default=24)  # Control the fixed spatial observation resolution.
    parser.add_argument("--reference", action="store_true")  # Enable independently integrated reference errors.
    parser.add_argument("--reference-factor", type=float, default=3.0)  # Set the shared fine reference resolution.
    args = parser.parse_args()  # Parse concrete execution options.
    args.output.mkdir(parents=True, exist_ok=True)  # Prepare output once.
    train_seeds = [int(value) for value in args.train_seeds.split(",") if value]  # Parse the training manifest.
    test_seeds = [int(value) for value in args.test_seeds.split(",") if value]  # Parse the held-out manifest.
    if set(train_seeds) & set(test_seeds):  # Prevent accidental reuse of an identical physical instance.
        raise ValueError("Training and test geometry seeds must be distinct")  # Explain the scientific data contract.
    write_json(args.output / "manifest.json", {**vars(args), "source_commit": "bc0af12360dabbcf6b25320c7e48586a32a08264", "metric": "actual CCX ZZ indicator; optional common fine-cell stress-energy quadrature", "scope": "same-state counterfactual forks; one-step learned spatial transition", "language_model": False})  # Declare exact provenance and avoid overstating VLA scope.
    train = [collect_case(args, seed, "train") for seed in train_seeds]  # Generate authentic counterfactual training data.
    rows = [entry for case in train for entry in case["entries"]]  # Use every training action with equal access for both models.
    x = np.asarray([entry["x"] for entry in rows])  # Assemble spatial visual-action features.
    xs = np.asarray([entry["xs"] for entry in rows])  # Assemble the retrained no-image ablation.
    y = np.asarray([entry["y"] for entry in rows])  # Assemble future spatial error mass targets.
    started = time.perf_counter()  # Count training overhead separately.
    model = SpatialWorldModel(alpha=10.0).fit(x, y)  # Fit a fixed regularization model without test tuning.
    scalar = SpatialWorldModel(alpha=10.0).fit(xs, y)  # Fit the same regression family to non-spatial inputs.
    model.save(args.output / "visual_world_model.json")  # Save the genuine learned spatial transition parameters.
    scalar.save(args.output / "scalar_world_model.json")  # Save the equally trained no-image comparator.
    training_s = time.perf_counter() - started  # Distinguish regression fitting from offline solver data generation.
    tests = [collect_case(args, seed, "test", model, scalar) for seed in test_seeds]  # Apply frozen models to unseen physical instances.
    report = {"training_cases": len(train), "training_transitions": len(rows), "training_fit_s": training_s, "training_solver_calls": sum(1 + len(case["entries"]) for case in train), "test_cases": len(tests), "cases": []}  # Declare data costs and actual evidence scale.
    for case in tests:  # Report each case before pooling any averages.
        actions = {entry["name"]: entry for entry in case["entries"]}  # Map named physical alternatives.
        metric = "reference_error" if args.reference else "eta2"  # Use only an actually computed evaluation quantity.
        decision = case["decisions"]  # Read the previously sealed model decisions.
        winner = actions[decision["visual_wm"]]  # Retrieve the selected branch's real outcome.
        best_dorfler = min((entry for name, entry in actions.items() if name.startswith("dorfler")), key=lambda entry: entry[metric])  # Report a strong retrospective best-of-three classical envelope.
        best = min(actions.values(), key=lambda entry: entry[metric])  # Compute a separately labeled ex-post candidate oracle.
        report["cases"].append({"seed": case["seed"], "selected": decision["visual_wm"], "scalar_selected": decision["scalar_wm"], "metric": metric, "selected_value": winner[metric], "dorfler_oracle": best_dorfler["name"], "ratio_to_dorfler_oracle": winner[metric] / best_dorfler[metric], "ratio_to_analytic": winner[metric] / actions["analytic_density"][metric], "ratio_to_scalar": winner[metric] / actions[decision["scalar_wm"]][metric], "candidate_oracle": best["name"], "regret": winner[metric] / best[metric] - 1.0, "equations": winner["record"]["n_equations"], "cap": case["cap"], "planner_mesh_s": sum(entry["cost"]["mesh_s"] for entry in actions.values()), "chosen_solve_s": winner["record"]["wall_s"]})  # Preserve resources, ranking regret, and each competing result.
    write_json(args.output / "summary.json", report)  # Save measured results without declaring an automatic win.
    print(json.dumps(report, indent=2), flush=True)  # Return concise final experiment evidence.

if __name__ == "__main__":  # Run only on explicit script execution.
    main()  # Execute the full controlled experiment.
