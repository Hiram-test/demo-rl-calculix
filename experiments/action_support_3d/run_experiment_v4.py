from __future__ import annotations  # Enable modern type annotations without runtime evaluation.
import csv  # Persist the calibrated local mesh size and achieved DOF for every candidate support.
from pathlib import Path  # Create deterministic calibration directories without platform-specific path operations.
import sys  # Preserve explicit command-line failure reporting for GitHub Actions.
import run_experiment as core  # Reuse the frozen case matrix, Gmsh model, parsers, reference setup, and Pareto output logic.
import run_experiment_v3 as v3  # Reuse the final mesh-invariant physical QoIs and fixed-load CalculiX solve from v0.3.

TARGET_DOF = 3000  # Compare local action supports at one common three-translational-DOF resource budget.
CALIBRATION_STEPS = 8  # Use eight bisection refinements after bracketing to obtain a close mesh-resource match cheaply.
MIN_LOCAL_H = 1.2  # Allow very small spatial atoms to refine strongly enough to reach the common DOF budget.
MAX_LOCAL_H = 9.5  # Allow large semantic supports to remain coarse enough to reach the same common DOF budget.
CALIBRATION_DIR = core.ROOT / "calibration"  # Isolate temporary mesh-only budget searches from the final solver case directories.
ORIGINAL_WRITE_OUTPUTS = core.write_outputs  # Preserve the shared output writer before installing the v0.4 budget-calibration extension.


def mesh_dof_for_size(case: dict, local_h: float, gmsh_bin: str, label: str) -> int:  # Generate one mesh-only probe and return its three-DOF-per-node resource proxy.
    probe_dir = CALIBRATION_DIR / case["id"] / label  # Allocate a deterministic directory for the current candidate and calibration probe.
    probe_dir.mkdir(parents=True, exist_ok=True)  # Create the probe directory and any missing parents idempotently.
    geo_path = probe_dir / "model.geo"  # Store the exact Gmsh sizing program used by this resource probe.
    msh_path = probe_dir / "model.msh"  # Store the resulting MSH2 mesh long enough to count nodes reliably.
    original_h = core.REFINED_H  # Preserve the shared default local size before temporarily applying the probe size.
    core.REFINED_H = local_h  # Inject the current calibration size into the validated shared Gmsh local-support generator.
    try:  # Guarantee restoration of the shared local-size constant even when Gmsh reports an error.
        geo_path.write_text(core.gmsh_geo_text(case["regions"], global_size=None), encoding="utf-8")  # Generate the same support geometry with only the local mesh size changed.
    finally:  # Restore shared benchmark state before any external command is launched.
        core.REFINED_H = original_h  # Return the shared generator to its committed default local size.
    core.run_command([gmsh_bin, str(geo_path.name), "-3", "-format", "msh2", "-o", str(msh_path.name)], probe_dir, probe_dir / "gmsh.log")  # Generate the calibration mesh without launching CalculiX.
    nodes, _tets = core.parse_msh2(msh_path)  # Parse only enough mesh data to count physical finite-element nodes.
    return 3 * len(nodes)  # Return the same three-DOF-per-node resource proxy used in the final precision-resource tables.


def calibrate_local_size(case: dict, gmsh_bin: str) -> tuple[float, int]:  # Find the local target size whose generated mesh is closest to the common DOF budget.
    evaluated: list[tuple[float, int]] = []  # Retain every probe so the closest actually generated resource level can be selected at the end.
    low_h = MIN_LOCAL_H  # Start the fine end of the admissible local-size interval.
    high_h = MAX_LOCAL_H  # Start the coarse end of the admissible local-size interval.
    low_dof = mesh_dof_for_size(case, low_h, gmsh_bin, "bound_fine")  # Measure the maximum practical resource level for this support family.
    high_dof = mesh_dof_for_size(case, high_h, gmsh_bin, "bound_coarse")  # Measure the minimum practical resource level for this support family.
    evaluated.append((low_h, low_dof))  # Preserve the fine boundary probe as a valid fallback candidate.
    evaluated.append((high_h, high_dof))  # Preserve the coarse boundary probe as a valid fallback candidate.
    if low_dof < high_dof:  # Detect an unexpected local nonmonotonic reversal before applying bisection logic.
        low_h, high_h = high_h, low_h  # Swap the size bounds so the first bound again corresponds to the higher measured DOF.
        low_dof, high_dof = high_dof, low_dof  # Swap the measured resource values consistently with the size bounds.
    for step in range(CALIBRATION_STEPS):  # Refine the size interval with a bounded number of mesh-only probes.
        mid_h = 0.5 * (low_h + high_h)  # Bisect the current target-size interval in physical mesh-size space.
        mid_dof = mesh_dof_for_size(case, mid_h, gmsh_bin, f"bisect_{step:02d}")  # Measure the actual generated DOF at the midpoint size.
        evaluated.append((mid_h, mid_dof))  # Preserve the midpoint probe for final nearest-budget selection.
        if mid_dof > TARGET_DOF:  # Detect a mesh that is finer and more expensive than the common resource budget.
            low_h = mid_h  # Increase the lower size bound so subsequent probes become coarser and cheaper.
            low_dof = mid_dof  # Preserve the corresponding high-resource measurement for interval bookkeeping.
        else:  # Handle a mesh at or below the common target resource level.
            high_h = mid_h  # Decrease the upper size bound so subsequent probes become finer and more expensive.
            high_dof = mid_dof  # Preserve the corresponding low-resource measurement for interval bookkeeping.
    best_h, best_dof = min(evaluated, key=lambda item: (abs(item[1] - TARGET_DOF), item[0]))  # Select the actually generated mesh closest to the target DOF with a deterministic size tie-break.
    return best_h, best_dof  # Return the calibrated local size and its measured mesh-only resource proxy.


def solve_case_v4(case: dict, gmsh_bin: str, ccx_bin: str) -> dict:  # Evaluate one case while enforcing a common resource budget on every local support candidate.
    if case["kind"] == "global":  # Preserve the original coarse baseline and globally fine reference without artificial local-budget calibration.
        row = v3.solve_case_v3(case, gmsh_bin, ccx_bin)  # Evaluate the global mesh case with the final fixed physical QoIs.
        row["local_h"] = None  # Mark the global case as having no local refinement-size parameter.
        row["target_dof"] = None  # Mark the global case as outside the local candidate budget-normalization experiment.
        row["calibration_dof"] = row["dof_proxy"]  # Record its actual resource proxy for completeness in the calibration table.
        return row  # Return the completed baseline or reference record without further changes.
    best_h, calibration_dof = calibrate_local_size(case, gmsh_bin)  # Find the support-specific mesh size that best matches the common DOF budget.
    original_h = core.REFINED_H  # Preserve the committed default local size before the final calibrated structural solve.
    core.REFINED_H = best_h  # Apply the support-specific calibrated mesh size to the shared validated Gmsh generator.
    try:  # Guarantee restoration of the shared benchmark constant even if the final structural solve fails.
        row = v3.solve_case_v3(case, gmsh_bin, ccx_bin)  # Solve the calibrated candidate using the same fixed-load physics and fixed material-point QoIs as v0.3.
    finally:  # Restore committed shared state after the support-specific final solve.
        core.REFINED_H = original_h  # Return the shared generator to its default local mesh size for the next case.
    row["local_h"] = best_h  # Preserve the calibrated physical mesh size used for this candidate support.
    row["target_dof"] = TARGET_DOF  # Preserve the common intended resource budget alongside the actual generated mesh cost.
    row["calibration_dof"] = calibration_dof  # Preserve the mesh-only resource measurement selected by the calibration stage.
    return row  # Return the precision-resource result plus budget-calibration metadata.


def write_outputs_v4(rows: list[dict], reference: dict) -> None:  # Preserve the standard result tables and add an explicit resource-normalization audit table.
    ORIGINAL_WRITE_OUTPUTS(rows, reference)  # Write the established CSV, manifest, summary, errors, and Pareto flags without changing their semantics.
    fields = ["id", "source", "qoi_target", "confidence", "local_h", "target_dof", "calibration_dof", "dof_proxy"]  # Define a compact schema for auditing whether candidate resource budgets were truly matched.
    with (core.RESULTS_DIR / "budget_calibration.csv").open("w", encoding="utf-8", newline="") as handle:  # Open the calibration audit table with portable CSV newline handling.
        writer = csv.DictWriter(handle, fieldnames=fields)  # Create a deterministic column writer for the budget-calibration metadata.
        writer.writeheader()  # Emit the calibration CSV header before any case records.
        for row in rows:  # Emit global and local cases in the same deterministic execution order as the main result table.
            writer.writerow({field: row.get(field) for field in fields})  # Restrict each record to the documented resource-normalization schema.


def main() -> int:  # Install common-budget calibration around the final physical-QoI evaluator and run the frozen candidate matrix.
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)  # Create the mesh-only calibration root before executing any candidate probes.
    core.solve_case = solve_case_v4  # Replace the shared per-case solver with the common-budget calibrated evaluator.
    core.write_outputs = write_outputs_v4  # Extend the shared result writer with an auditable local-size and achieved-DOF table.
    return core.main()  # Execute the same coarse/reference cases, sixteen spatial atoms, six frozen LLM supports, and Pareto analysis.


if __name__ == "__main__":  # Execute the common-budget benchmark only when this file is invoked as the program entry point.
    try:  # Preserve explicit workflow failure reporting around calibration, meshing, solving, and result writing.
        raise SystemExit(main())  # Run the resource-normalized benchmark and terminate with its returned shell status.
    except Exception as exc:  # Catch unexpected errors only at the outermost command-line boundary.
        print(f"[fatal-v4] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Emit a concise failure reason to the GitHub Actions log and persisted console tail.
        raise  # Re-raise the original exception so the workflow accurately records a failed calibrated experiment.
