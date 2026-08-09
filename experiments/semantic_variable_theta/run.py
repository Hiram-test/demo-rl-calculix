from __future__ import annotations  # Enable postponed evaluation of modern type annotations.
import csv  # Write compact region-theta diagnostics and resource-threshold tables.
import importlib.util  # Load the validated semantic-partition benchmark without duplicating solver code.
import json  # Persist per-round regional error densities and variable theta values.
from pathlib import Path  # Resolve repository-relative experiment paths portably.
import sys  # Register the dynamically loaded module and propagate explicit failures to GitHub Actions.

ROOT = Path(__file__).resolve().parent  # Anchor all transient cases and compact outputs to this experiment directory.
BASE_PATH = ROOT.parent / "semantic_partition_dorfler_v2" / "run.py"  # Reuse the validated full-domain partition, solver, estimator, and remesher.
CASES_DIR = ROOT / "cases"  # Store transient Gmsh and CalculiX files outside compact result artifacts.
RESULTS_DIR = ROOT / "results"  # Store only small tables, summaries, logs, and figures here.
AMR_ROUNDS = 6  # Solve the shared coarse state plus six adaptive refinement states for each method.
THETA_MIN = 0.10  # Keep every semantic region weakly active so background information is never permanently discarded.
THETA_MAX = 0.90  # Prevent a single hotspot region from consuming essentially all of its local estimator mass in one round.
DENSITY_POWER = 1.00  # Use a transparent linear response from regional estimator density to relative refinement intensity.
BISECTION_STEPS = 80  # Solve the single global-budget multiplier accurately without external optimization dependencies.

spec = importlib.util.spec_from_file_location("semantic_variable_theta_base", BASE_PATH)  # Build an import specification for the validated benchmark implementation.
if spec is None or spec.loader is None:  # Reject an unexpected repository state before launching any native numerical tool.
    raise RuntimeError(f"Unable to load semantic-partition base from {BASE_PATH}")  # Fail explicitly when the validated base implementation is unavailable.
base = importlib.util.module_from_spec(spec)  # Create the isolated module object that will host the validated benchmark implementation.
sys.modules[spec.name] = base  # Register the module so postponed annotations and internal imports resolve correctly.
spec.loader.exec_module(base)  # Execute the validated module without invoking its command-line main block.
base.CASES_DIR = CASES_DIR  # Redirect the base experiment transient numerical cases into this dedicated directory.
base.RESULTS_DIR = RESULTS_DIR  # Redirect the base experiment compact outputs into this dedicated directory.
base.AMR_ROUNDS = AMR_ROUNDS  # Extend the base experiment to the requested six adaptive refinement rounds.
base.core.CASES_DIR = CASES_DIR  # Redirect the shared numerical core transient cases into this dedicated directory.
base.core.RESULTS_DIR = RESULTS_DIR  # Redirect any shared-core compact outputs into this dedicated directory.
base.core.AMR_ROUNDS = AMR_ROUNDS  # Keep the shared numerical core round count synchronized with the wrapper.


def clip_theta(value: float) -> float:  # Apply the fixed admissible interval to one region-level bulk parameter.
    return min(THETA_MAX, max(THETA_MIN, value))  # Return the bounded regional refinement intensity.


def region_statistics(nodes: dict[int, tuple[float, float, float]], tets: list[tuple[int, tuple[int, int, int, int]]], eta2: dict[int, float], groups: dict[str, set[int]]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:  # Compute estimator mass, physical volume, and estimator density for every semantic region.
    connectivity_map = {element_id: connectivity for element_id, connectivity in tets}  # Build direct tetrahedral connectivity lookup for physical-volume integration.
    region_mass = {name: sum(eta2[element_id] for element_id in element_ids) for name, element_ids in groups.items()}  # Integrate the current element indicator over each semantic region.
    region_volume = {name: sum(base.core.tet_volume(nodes, connectivity_map[element_id]) for element_id in element_ids) for name, element_ids in groups.items()}  # Integrate actual tetrahedral volume in each semantic region.
    region_density = {name: (region_mass[name] / region_volume[name] if region_volume[name] > 0.0 else 0.0) for name in groups}  # Normalize regional estimator mass by physical volume so hotspot detection is mesh-density independent.
    return region_mass, region_volume, region_density  # Return all three regional diagnostics for allocation and auditability.


def theta_from_density(region_mass: dict[str, float], region_volume: dict[str, float], region_density: dict[str, float]) -> tuple[dict[str, float], float]:  # Detect hotspots from regional error density and solve variable theta values under a conserved global bulk budget.
    total_mass = sum(region_mass.values())  # Compute the complete current global estimator mass used by the conventional baseline.
    total_volume = sum(region_volume.values())  # Compute the complete current material volume represented by the semantic partition.
    if total_mass <= 0.0 or total_volume <= 0.0:  # Require nontrivial estimator mass and physical volume before allocating refinement intensity.
        raise RuntimeError("Regional estimator mass or physical volume is zero")  # Fail explicitly rather than generating undefined regional priorities.
    global_density = total_mass / total_volume  # Compute the physical-volume-averaged estimator density used only as a dimensionless reference scale.
    scores = {name: max(region_density[name] / global_density, 1.0e-12) ** DENSITY_POWER for name in region_mass}  # Convert each regional hotspot density into a positive dimensionless allocation score.
    target_mass = base.core.THETA * total_mass  # Preserve exactly the same continuous global bulk budget as conventional Dörfler before discrete prefix overshoot.
    def allocated_mass(multiplier: float) -> float:  # Evaluate total continuous estimator budget captured by one candidate global multiplier.
        return sum(clip_theta(multiplier * scores[name]) * region_mass[name] for name in region_mass)  # Sum region-specific theta times regional estimator mass across the exhaustive partition.
    lower = 0.0  # Start the multiplier bracket below every positive allocation level.
    upper = 1.0  # Start with an order-one multiplier before expanding the upper bracket if necessary.
    while allocated_mass(upper) < target_mass:  # Expand until the clipped variable-theta allocation reaches the common global bulk target.
        upper *= 2.0  # Double the upper multiplier monotonically because allocated mass is nondecreasing in the multiplier.
        if upper > 1.0e12:  # Guard against an impossible budget caused by invalid parameter bounds or numerical data.
            raise RuntimeError("Unable to bracket the conserved variable-theta bulk budget")  # Fail explicitly when the regional allocation constraints cannot satisfy the target.
    for _iteration in range(BISECTION_STEPS):  # Solve the single monotone budget equation to high numerical precision.
        midpoint = 0.5 * (lower + upper)  # Evaluate the centre of the current valid multiplier bracket.
        if allocated_mass(midpoint) < target_mass:  # Check whether the current variable-theta allocation is still below the global target.
            lower = midpoint  # Move the lower bracket upward when more regional bulk intensity is required.
        else:  # Handle a midpoint allocation that reaches or exceeds the common global target.
            upper = midpoint  # Move the upper bracket downward to obtain the smallest multiplier satisfying the budget.
    multiplier = upper  # Use the conservative upper bracket as the final globally budgeted multiplier.
    theta_by_region = {name: clip_theta(multiplier * scores[name]) for name in region_mass}  # Convert hotspot-density scores into the final bounded dynamic regional theta values.
    conserved_fraction = sum(theta_by_region[name] * region_mass[name] for name in region_mass) / total_mass  # Report the continuous global estimator fraction implied by the regional theta allocation.
    return theta_by_region, conserved_fraction  # Return the dynamic per-region bulk parameters and their globally conserved continuous budget fraction.


def variable_theta_mark(nodes: dict[int, tuple[float, float, float]], tets: list[tuple[int, tuple[int, int, int, int]]], eta2: dict[int, float], groups: dict[str, set[int]]) -> tuple[list[int], dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, int], float, float]:  # Apply region-detected variable-theta Dörfler marking and union all regional prefixes.
    total_mass = sum(eta2.values())  # Compute global estimator mass for diagnostics and discrete captured-fraction reporting.
    region_mass, region_volume, region_density = region_statistics(nodes, tets, eta2, groups)  # Detect current semantic-region error concentration using physical estimator density.
    theta_by_region, continuous_budget_fraction = theta_from_density(region_mass, region_volume, region_density)  # Allocate higher theta to current hotspot regions while conserving the global continuous bulk budget.
    marked_union: set[int] = set()  # Accumulate the union of all independently ranked regional Dörfler prefixes.
    marked_by_region: dict[str, int] = {}  # Record how many finite elements each semantic region receives this round.
    for region_name, element_ids in groups.items():  # Apply the dynamically allocated theta separately inside every exhaustive semantic region.
        local_mass = region_mass[region_name]  # Read the current estimator mass available inside this semantic region.
        if not element_ids or local_mass <= 0.0:  # Skip only a geometrically empty or numerically zero-mass region.
            marked_by_region[region_name] = 0  # Record zero regional investment for transparent diagnostics.
            continue  # Advance because no meaningful local Dörfler problem exists in this region.
        local_target = theta_by_region[region_name] * local_mass  # Convert the detected regional theta into the local bulk estimator target.
        ranked = sorted(((element_id, eta2[element_id]) for element_id in element_ids), key=lambda item: item[1], reverse=True)  # Rank finite elements only within the current semantic region using the unchanged estimator.
        local_captured = 0.0  # Track estimator mass captured by the current regional prefix.
        local_marked: list[int] = []  # Accumulate the minimal current-region prefix satisfying its dynamic bulk condition.
        for element_id, value in ranked:  # Traverse current-region elements from largest to smallest indicator value.
            local_marked.append(element_id)  # Add the current highest-ranked regional element to the refinement action.
            local_captured += value  # Add its estimator contribution to the regional bulk accumulation.
            if local_captured >= local_target:  # Stop once this region's dynamically allocated theta fraction has been captured.
                break  # Preserve the minimal descending-prefix Dörfler construction inside this semantic region.
        marked_union.update(local_marked)  # Merge the current hotspot-weighted regional action into the full refinement action.
        marked_by_region[region_name] = len(local_marked)  # Record actual regional finite-element investment after discrete Dörfler prefix selection.
    if not marked_union:  # Require at least one finite element to be refined in every nontrivial adaptive round.
        raise RuntimeError("Variable-theta semantic marking produced no marked elements")  # Fail explicitly instead of generating an unchanged adaptive mesh.
    captured_mass = sum(eta2[element_id] for element_id in marked_union)  # Compute the actual discrete estimator mass captured after regional prefix overshoot.
    return sorted(marked_union), region_mass, region_volume, region_density, theta_by_region, marked_by_region, continuous_budget_fraction, captured_mass / total_mass  # Return the complete action and all allocation diagnostics.


def run_method_variable_theta(method: str, reference_qoi: float, gmsh_bin: str, ccx_bin: str) -> tuple[list[dict], list[dict]]:  # Execute either conventional global Dörfler or the semantic variable-theta method through the same numerical pipeline.
    refinement_history: list[dict[str, float]] = []  # Accumulate all prior marked-element sizing neighbourhoods so adaptive refinement persists across remeshing.
    rows: list[dict] = []  # Accumulate one precision-resource record for every solved adaptive state.
    decisions: list[dict] = []  # Preserve detailed regional theta and hotspot diagnostics for every marking round.
    for round_index in range(AMR_ROUNDS + 1):  # Solve the shared coarse state followed by six adaptive states.
        state = base.core.solve_mesh(f"{method}_r{round_index}", gmsh_bin, ccx_bin, refinement_boxes=refinement_history, global_h=None)  # Generate and solve the current adaptive mesh from the accumulated history.
        row = {"method": method, "round": round_index, "dof": state["dof"], "elements": state["element_count"], "qoi": state["qoi"], "qoi_rel_error": base.core.relative_error(state["qoi"], reference_qoi), "marked_elements": 0, "captured_global_fraction": 0.0, "selected_regions": ""}  # Create the current precision-resource record before optional marking.
        if round_index < AMR_ROUNDS:  # Mark finite elements only when another adaptive state will be generated.
            eta2 = base.core.stress_jump_indicator(state["nodes"], state["tets"], state["stresses"])  # Compute the identical elementwise estimator used by both methods.
            if method == "global_dorfler":  # Apply the validated conventional global Dörfler baseline unchanged.
                all_elements = {element_id for element_id, _connectivity in state["tets"]}  # Admit every current finite element to the global ranking.
                marked, captured_fraction = base.core.dorfler_mark(eta2, all_elements)  # Apply the existing global descending-prefix Dörfler rule with theta equal to 0.50.
                theta_by_region = {"all_domain": base.core.THETA}  # Record the single conventional global bulk parameter for common diagnostics.
                region_mass = {"all_domain": sum(eta2.values())}  # Record complete estimator mass as one undivided global region.
                region_volume = {"all_domain": sum(base.core.tet_volume(state["nodes"], connectivity) for _element_id, connectivity in state["tets"])}  # Record complete physical volume for the baseline diagnostic schema.
                region_density = {"all_domain": region_mass["all_domain"] / region_volume["all_domain"]}  # Record complete-domain estimator density for the baseline diagnostic schema.
                marked_by_region = {"all_domain": len(marked)}  # Record the conventional number of marked finite elements.
                continuous_budget_fraction = base.core.THETA  # Record the exact continuous global bulk target used by the conventional baseline.
                captured_global_fraction = captured_fraction  # Record the actual discrete fraction captured by the global prefix.
            else:  # Apply semantic-region hotspot detection followed by dynamically variable regional theta values.
                groups = base.partition_elements(state["nodes"], state["tets"])  # Partition the complete current mesh into root, hole, and residual background regions.
                marked, region_mass, region_volume, region_density, theta_by_region, marked_by_region, continuous_budget_fraction, captured_global_fraction = variable_theta_mark(state["nodes"], state["tets"], eta2, groups)  # Allocate hotspot-heavy regional bulk parameters and perform regional Dörfler marking.
            connectivity_map = {element_id: connectivity for element_id, connectivity in state["tets"]}  # Build direct connectivity lookup for every marked element.
            new_boxes = [base.core.refinement_box(state["nodes"], connectivity_map[element_id]) for element_id in marked]  # Convert the chosen finite elements into the unchanged persistent local remeshing neighbourhoods.
            refinement_history.extend(new_boxes)  # Preserve this round's action in every subsequent adaptive mesh state.
            row["marked_elements"] = len(marked)  # Record total finite-element investment in this adaptive action.
            row["captured_global_fraction"] = captured_global_fraction  # Record actual globally normalized estimator mass captured after discrete marking.
            row["selected_regions"] = ";".join(f"{name}:{theta_by_region[name]:.4f}" for name in sorted(theta_by_region))  # Store compact dynamic regional theta values directly in the main trajectory table.
            decisions.append({"method": method, "round": round_index, "region_mass": region_mass, "region_volume": region_volume, "region_density": region_density, "theta_by_region": theta_by_region, "marked_by_region": marked_by_region, "continuous_budget_fraction": continuous_budget_fraction, "captured_global_fraction": captured_global_fraction})  # Persist complete hotspot-detection and resource-allocation diagnostics.
        rows.append(row)  # Append the completed current adaptive state to the method trajectory.
        print(f"[{method}] round={round_index} dof={row['dof']} error={row['qoi_rel_error']:.6e} theta={row['selected_regions']} marked={row['marked_elements']}", flush=True)  # Stream one concise progress line into GitHub Actions logs.
    return rows, decisions  # Return the complete precision-resource trajectory and regional allocation history.


base.run_method = run_method_variable_theta  # Replace only the adaptive marking loop while preserving geometry, estimator, solver, remesher, reference, and baseline implementation.
original_write_outputs = base.write_outputs  # Preserve the validated machine-readable output writer before replacing its human-readable interpretation.


def write_outputs_variable_theta(rows: list[dict], decisions: list[dict], reference: dict) -> None:  # Persist standard benchmark tables plus explicit per-region theta history for direct inspection.
    original_write_outputs(rows, decisions, reference)  # Write history.csv, region_decisions.json, same_accuracy.csv, manifest.json, and the base report using the common schemas.
    semantic_decisions = [decision for decision in decisions if decision["method"] == "semantic_partition_dorfler"]  # Extract only variable-theta semantic allocation decisions from the common decision log.
    theta_fields = ["round", "root_theta", "hole_theta", "background_theta", "root_density", "hole_density", "background_density", "root_marked", "hole_marked", "background_marked", "continuous_budget_fraction", "captured_global_fraction"]  # Define a compact tabular schema exposing hotspot-heavy resource allocation explicitly.
    with (RESULTS_DIR / "theta_history.csv").open("w", encoding="utf-8", newline="") as handle:  # Open the region-theta history with portable CSV newline handling.
        writer = csv.DictWriter(handle, fieldnames=theta_fields)  # Create a deterministic writer for regional intensity diagnostics.
        writer.writeheader()  # Emit the documented regional diagnostic column names.
        for decision in semantic_decisions:  # Emit one row per semantic adaptive marking round.
            theta_map = decision["theta_by_region"]  # Read the dynamically detected regional bulk parameters for the current round.
            density_map = decision["region_density"]  # Read the current physical estimator densities that generated the variable theta values.
            marked_map = decision["marked_by_region"]  # Read actual finite-element investment assigned to each semantic region.
            writer.writerow({"round": decision["round"], "root_theta": theta_map.get("root_region", 0.0), "hole_theta": theta_map.get("hole_region", 0.0), "background_theta": theta_map.get("background_region", 0.0), "root_density": density_map.get("root_region", 0.0), "hole_density": density_map.get("hole_region", 0.0), "background_density": density_map.get("background_region", 0.0), "root_marked": marked_map.get("root_region", 0), "hole_marked": marked_map.get("hole_region", 0), "background_marked": marked_map.get("background_region", 0), "continuous_budget_fraction": decision["continuous_budget_fraction"], "captured_global_fraction": decision["captured_global_fraction"]})  # Persist dynamic hotspot intensity and actual regional element investment in one auditable row.
    global_rows = [row for row in rows if row["method"] == "global_dorfler"]  # Extract the conventional global Dörfler trajectory for the concise report.
    semantic_rows = [row for row in rows if row["method"] == "semantic_partition_dorfler"]  # Extract the semantic variable-theta trajectory for the concise report.
    lines: list[str] = []  # Accumulate a compact GitHub-readable interpretation of the experiment.
    lines.append("# Region-detected variable-theta semantic Dörfler")  # State the exact semantic intervention in the report title.
    lines.append("")  # Add one Markdown spacer line.
    lines.append(f"Reference QoI: `{reference['qoi']:.12e}` mm at `{reference['dof']}` DOF proxy.")  # State the fixed numerical reference shared by both methods.
    lines.append("")  # Add one Markdown spacer line.
    lines.append("Each round computes estimator density E_r/V_r in the root, hole, and residual background regions. Hotter regions receive larger theta_r and cooler regions receive smaller theta_r. A single multiplier is solved so sum(theta_r E_r) equals theta_global sum(E_r) with theta_global=0.50 before discrete Dörfler prefix overshoot.")  # Define the dynamic regional resource-allocation rule precisely.
    lines.append("")  # Add one Markdown spacer line.
    lines.append("| round | global DOF | global error | semantic DOF | semantic error | regional theta values |")  # Start the direct trajectory and allocation table.
    lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")  # Add Markdown table alignment markers.
    for global_row, semantic_row in zip(global_rows, semantic_rows):  # Compare solved states with the same adaptive-round index while exposing semantic intensity.
        lines.append(f"| {global_row['round']} | {global_row['dof']} | {global_row['qoi_rel_error']:.6e} | {semantic_row['dof']} | {semantic_row['qoi_rel_error']:.6e} | {semantic_row['selected_regions']} |")  # Emit one roundwise precision-resource and variable-theta record.
    lines.append("")  # Add one Markdown spacer line.
    lines.append("Use `theta_history.csv` to verify that hotspot regions actually receive higher theta and more marked elements than the residual background when their estimator density is higher.")  # Point directly to the diagnostic that tests the user's hotspot-reinvestment requirement.
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")  # Replace the generic base report with the variable-theta interpretation.


base.write_outputs = write_outputs_variable_theta  # Install the variable-theta output writer before the shared experiment main function executes.


def main() -> int:  # Execute the conventional global baseline and the region-detected variable-theta semantic method.
    CASES_DIR.mkdir(parents=True, exist_ok=True)  # Ensure transient numerical cases have a destination before Gmsh starts.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure compact generated outputs have a destination before the first solve.
    return base.main()  # Run the shared reference solution, both adaptive histories, common-accuracy comparison, and variable-theta diagnostics.


if __name__ == "__main__":  # Execute the experiment only when this wrapper file is invoked as the program entry point.
    try:  # Keep GitHub Actions failure propagation explicit at the outermost command boundary.
        raise SystemExit(main())  # Run the complete experiment and return its shell status unchanged.
    except Exception as exc:  # Catch setup, meshing, solving, parsing, or allocation failures only to emit a concise diagnostic.
        print(f"[fatal-variable-theta] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Write one clear failure reason into the persisted workflow console log.
        raise  # Re-raise the original exception so GitHub Actions records a genuine numerical failure.
