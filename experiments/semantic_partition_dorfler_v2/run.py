from __future__ import annotations  # Enable compact modern type annotations without eager runtime evaluation.
import csv  # Write compact adaptive histories and common-accuracy comparisons.
import importlib.util  # Load the already validated Dörfler/CalculiX core from the sibling experiment.
import json  # Persist region-level decisions and the experiment manifest.
from pathlib import Path  # Resolve all repository-relative paths portably.
import sys  # Register the dynamically loaded core module and propagate CI failures.

ROOT = Path(__file__).resolve().parent  # Anchor all generated cases and results to this experiment directory.
CASES_DIR = ROOT / "cases"  # Store transient Gmsh and CalculiX files outside compact results.
RESULTS_DIR = ROOT / "results"  # Store only small numerical tables and summaries here.
CORE_PATH = ROOT.parent / "llm_dorfler_3d" / "run.py"  # Reuse the numerically validated solver, estimator, and remesher.
REGION_THETA = 0.50  # Use the same bulk fraction at the semantic-region level as the global Dörfler baseline.
AMR_ROUNDS = 5  # Run enough adaptive rounds to test whether the background region becomes important later.
ROOT_XMAX = 18.0  # Define the hand-drawn full-cross-section root band boundary in millimetres.
HOLE_REGION_RADIUS = 18.0  # Define the hand-drawn circular envelope around the through-hole in millimetres.

spec = importlib.util.spec_from_file_location("dorfler_core_v2", CORE_PATH)  # Build an import specification for the validated sibling implementation.
if spec is None or spec.loader is None:  # Reject an unexpected repository state before launching any native tool.
    raise RuntimeError(f"Unable to load validated Dörfler core from {CORE_PATH}")  # Fail explicitly when the shared numerical core is unavailable.
core = importlib.util.module_from_spec(spec)  # Create an isolated module object for the validated numerical implementation.
sys.modules[spec.name] = core  # Register the module so postponed annotations and internal imports resolve normally.
spec.loader.exec_module(core)  # Execute the shared module without invoking its command-line main block.
core.CASES_DIR = CASES_DIR  # Redirect transient numerical cases into the new full-domain-partition experiment.
core.RESULTS_DIR = RESULTS_DIR  # Redirect any compact shared outputs into the new result directory.
core.AMR_ROUNDS = AMR_ROUNDS  # Extend the adaptive history while preserving every other numerical setting.


def semantic_region(point: tuple[float, float, float]) -> str:  # Assign every physical point to exactly one LLM-defined semantic partition cell.
    x, _y, z = point  # Unpack the coordinates used by the root-band and hole-circle boundaries.
    radial2 = (x - core.HOLE_X) ** 2 + z ** 2  # Compute squared distance from the through-hole centre in the side-view plane.
    if radial2 <= HOLE_REGION_RADIUS ** 2 + core.GEOM_TOL:  # Give the hole envelope first priority where the two hand-drawn regions overlap slightly.
        return "hole_region"  # Assign the point to the circular hole semantic region.
    if x <= ROOT_XMAX + core.GEOM_TOL:  # Test whether the point lies in the full-cross-section root band.
        return "root_region"  # Assign the point to the root semantic region.
    return "background_region"  # Assign every remaining point to the residual background instead of discarding it.


def partition_elements(nodes: dict[int, tuple[float, float, float]], tets: list[tuple[int, tuple[int, int, int, int]]]) -> dict[str, set[int]]:  # Partition the complete current mesh into three exhaustive semantic regions.
    groups = {"root_region": set(), "hole_region": set(), "background_region": set()}  # Initialize one disjoint element set per semantic region.
    for element_id, connectivity in tets:  # Visit every current tetrahedron exactly once.
        centroid = core.tet_centroid(nodes, connectivity)  # Compute the mesh-independent physical centroid of the current element.
        groups[semantic_region(centroid)].add(element_id)  # Assign the element to exactly one semantic cell using the drawn boundaries.
    if sum(len(values) for values in groups.values()) != len(tets):  # Verify that the semantic partition covers the complete mesh without element loss.
        raise RuntimeError("Semantic partition failed to cover the complete current mesh")  # Reject any implementation that silently omits non-hotspot elements.
    return groups  # Return the exhaustive disjoint full-domain semantic partition.


def hierarchical_mark(eta2: dict[int, float], groups: dict[str, set[int]]) -> tuple[list[int], list[str], dict[str, float], float]:  # Perform region-level bulk selection followed by element-level Dörfler marking.
    total_global = sum(eta2.values())  # Compute the same global estimator mass used by conventional Dörfler.
    if total_global <= 0.0:  # Require a nontrivial estimator before any semantic decision is made.
        raise RuntimeError("Global estimator mass is zero")  # Fail explicitly when no refinement signal exists.
    region_mass = {name: sum(eta2[element_id] for element_id in element_ids) for name, element_ids in groups.items()}  # Aggregate current error-indicator mass inside each semantic region.
    ranked_regions = sorted(region_mass.items(), key=lambda item: item[1], reverse=True)  # Rank semantic regions by current estimator mass rather than freezing hotspot priority.
    region_target = REGION_THETA * total_global  # Require selected regions to contain the same bulk fraction of current global estimator mass.
    selected_regions: list[str] = []  # Accumulate the minimal descending-prefix set of semantic regions reaching the bulk target.
    selected_mass = 0.0  # Track estimator mass captured by the selected semantic regions.
    for name, mass in ranked_regions:  # Traverse semantic regions from largest to smallest current aggregate estimator mass.
        selected_regions.append(name)  # Admit the current region into this round's action support.
        selected_mass += mass  # Add the current region's estimator mass to the support total.
        if selected_mass >= region_target:  # Stop as soon as the region-level bulk criterion is satisfied.
            break  # Preserve a minimal region-level descending-prefix selection.
    eligible = set().union(*(groups[name] for name in selected_regions))  # Form the current semantic action support from the dynamically selected regions.
    ranked_elements = sorted(((element_id, eta2[element_id]) for element_id in eligible), key=lambda item: item[1], reverse=True)  # Rank finite elements only after the region-level semantic decision.
    element_target = core.THETA * total_global  # Require the final marked elements to capture exactly the same global estimator fraction as global Dörfler.
    marked: list[int] = []  # Accumulate the minimal eligible element prefix reaching the common absolute estimator target.
    captured = 0.0  # Track globally normalized estimator mass captured by the final marked elements.
    for element_id, value in ranked_elements:  # Traverse eligible elements from largest to smallest indicator.
        marked.append(element_id)  # Add the current element to the marked set.
        captured += value  # Add its indicator contribution to the common global bulk target.
        if captured >= element_target:  # Stop once the same absolute global Dörfler mass has been captured.
            break  # Preserve the minimal descending-prefix set inside the semantic support.
    if captured < element_target:  # Guard against a numerical inconsistency between region and element aggregation.
        raise RuntimeError("Selected semantic regions could not satisfy the global Dörfler bulk target")  # Fail rather than weaken the comparison criterion.
    return marked, selected_regions, region_mass, captured / total_global  # Return marked elements, dynamic region decision, region masses, and globally normalized captured fraction.


def run_method(method: str, reference_qoi: float, gmsh_bin: str, ccx_bin: str) -> tuple[list[dict], list[dict]]:  # Execute either conventional global Dörfler or the exhaustive semantic-partition hierarchy.
    refinement_history: list[dict[str, float]] = []  # Accumulate all previous marked-element sizing neighbourhoods so refinement persists across remeshing.
    rows: list[dict] = []  # Accumulate one precision-resource record per solved adaptive state.
    decisions: list[dict] = []  # Preserve per-round region masses, selected regions, and exact marked element counts.
    for round_index in range(AMR_ROUNDS + 1):  # Solve the shared coarse state followed by five adaptive states.
        state = core.solve_mesh(f"{method}_r{round_index}", gmsh_bin, ccx_bin, refinement_boxes=refinement_history, global_h=None)  # Generate and solve the current adaptive mesh from the accumulated history.
        row = {"method": method, "round": round_index, "dof": state["dof"], "elements": state["element_count"], "qoi": state["qoi"], "qoi_rel_error": core.relative_error(state["qoi"], reference_qoi), "marked_elements": 0, "captured_global_fraction": 0.0, "selected_regions": ""}  # Create the current precision-resource record before optional marking.
        if round_index < AMR_ROUNDS:  # Mark only when another adaptive mesh will be generated.
            eta2 = core.stress_jump_indicator(state["nodes"], state["tets"], state["stresses"])  # Compute the identical stress-jump estimator used by both methods.
            if method == "global_dorfler":  # Apply the conventional element-level baseline without semantic grouping.
                all_elements = {element_id for element_id, _connectivity in state["tets"]}  # Admit every current tetrahedron to global Dörfler ranking.
                marked, captured_fraction = core.dorfler_mark(eta2, all_elements)  # Apply the validated global descending-prefix Dörfler rule.
                selected_regions = ["all_domain"]  # Record that no semantic partition constrained the global baseline.
                region_mass = {"all_domain": sum(eta2.values())}  # Record the full estimator mass for a common decision-log schema.
                captured_global_fraction = captured_fraction  # Global Dörfler already normalizes its captured fraction by global estimator mass.
            else:  # Apply the new exhaustive semantic-partition hierarchy.
                groups = partition_elements(state["nodes"], state["tets"])  # Partition the complete current mesh into root, hole, and background semantic cells.
                marked, selected_regions, region_mass, captured_global_fraction = hierarchical_mark(eta2, groups)  # Dynamically choose semantic regions and then mark elements at the same global bulk target.
            connectivity_map = {element_id: connectivity for element_id, connectivity in state["tets"]}  # Build direct connectivity lookup for the chosen marked elements.
            new_boxes = [core.refinement_box(state["nodes"], connectivity_map[element_id]) for element_id in marked]  # Convert each marked tetrahedron into the same persistent local remeshing neighbourhood.
            refinement_history.extend(new_boxes)  # Preserve the current refinement action in all future adaptive meshes.
            row["marked_elements"] = len(marked)  # Record the number of marked finite elements in the current action.
            row["captured_global_fraction"] = captured_global_fraction  # Record estimator mass captured relative to the complete current mesh.
            row["selected_regions"] = "+".join(selected_regions)  # Record which semantic regions were dynamically active this round.
            decisions.append({"method": method, "round": round_index, "selected_regions": selected_regions, "region_mass": region_mass, "marked_elements": len(marked), "captured_global_fraction": captured_global_fraction})  # Persist compact region-level decision metadata.
        rows.append(row)  # Append the completed current state to the method trajectory.
        print(f"[{method}] round={round_index} dof={row['dof']} error={row['qoi_rel_error']:.6e} regions={row['selected_regions']} marked={row['marked_elements']}", flush=True)  # Stream one concise progress line into GitHub Actions logs.
    return rows, decisions  # Return the complete precision-resource trajectory and semantic decision history.


def first_reaching(rows: list[dict], threshold: float) -> dict | None:  # Find the minimum-resource solved state that reaches a requested common accuracy target.
    satisfying = [row for row in rows if float(row["qoi_rel_error"]) <= threshold]  # Collect all solved states meeting the common error requirement.
    return None if not satisfying else min(satisfying, key=lambda row: int(row["dof"]))  # Return the satisfying state with the smallest actual DOF cost.


def write_outputs(rows: list[dict], decisions: list[dict], reference: dict) -> None:  # Persist only resource-fair comparisons and transparent region-level decisions.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the compact result directory exists before writing artifacts.
    fields = ["method", "round", "dof", "elements", "qoi", "qoi_rel_error", "marked_elements", "captured_global_fraction", "selected_regions"]  # Define a stable trajectory table schema.
    with (RESULTS_DIR / "history.csv").open("w", encoding="utf-8", newline="") as handle:  # Open the adaptive history with portable CSV newline handling.
        writer = csv.DictWriter(handle, fieldnames=fields)  # Create a deterministic column writer.
        writer.writeheader()  # Emit the documented table header.
        for row in rows:  # Emit all global and semantic adaptive states.
            writer.writerow({field: row[field] for field in fields})  # Restrict every row to the stable documented schema.
    (RESULTS_DIR / "region_decisions.json").write_text(json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # Persist region masses and dynamically selected action supports for every round.
    global_rows = [row for row in rows if row["method"] == "global_dorfler"]  # Extract the conventional baseline trajectory.
    semantic_rows = [row for row in rows if row["method"] == "semantic_partition_dorfler"]  # Extract the exhaustive semantic-partition trajectory.
    thresholds = [0.10, 0.05, 0.03, 0.02, 0.015]  # Define common accuracy requirements used for fair resource comparison.
    threshold_records = []  # Accumulate minimum DOF required by each method at each common accuracy requirement.
    for threshold in thresholds:  # Evaluate both methods at the same error target rather than comparing unequal final-round resources.
        global_hit = first_reaching(global_rows, threshold)  # Find the cheapest global state satisfying the current accuracy target.
        semantic_hit = first_reaching(semantic_rows, threshold)  # Find the cheapest semantic state satisfying the same target.
        threshold_records.append({"target_rel_error": threshold, "global_dof": "" if global_hit is None else global_hit["dof"], "semantic_dof": "" if semantic_hit is None else semantic_hit["dof"]})  # Store only like-for-like accuracy-resource costs.
    with (RESULTS_DIR / "same_accuracy.csv").open("w", encoding="utf-8", newline="") as handle:  # Open the main fair-comparison table.
        writer = csv.DictWriter(handle, fieldnames=["target_rel_error", "global_dof", "semantic_dof"])  # Define the minimal common-accuracy schema.
        writer.writeheader()  # Emit the comparison header.
        writer.writerows(threshold_records)  # Persist all common-accuracy resource comparisons.
    manifest = {"reference": reference, "theta": core.THETA, "region_theta": REGION_THETA, "amr_rounds": AMR_ROUNDS, "qoi_point_mm": core.QOI_POINT, "partition": {"root_region": f"x <= {ROOT_XMAX} mm outside hole priority", "hole_region": f"circle radius {HOLE_REGION_RADIUS} mm around hole centre", "background_region": "all remaining material"}, "fairness_rule": "compare minimum DOF needed to reach the same QoI relative-error target; never infer superiority from unequal final-round DOF"}  # Record every comparison-defining setting and the corrected fairness rule.
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # Persist the corrected experiment definition.
    lines = ["# Full-domain semantic partition + Dörfler versus global Dörfler", "", f"Reference QoI: `{reference['qoi']:.12e}` mm at `{reference['dof']}` DOF proxy.", "", "The LLM drawing defines an exhaustive three-region partition: root band, hole circle, and the residual background. No finite element is permanently excluded. Every round recomputes aggregate estimator mass per region, selects semantic regions by a region-level bulk criterion, then applies element-level Dörfler inside those regions until the same absolute global estimator fraction is captured as the conventional baseline.", "", "Resource fairness: only minimum DOF needed to reach the same QoI error target is interpreted. Unequal final-round states are not called a win or an overtake.", "", "| target relative error | global DOF | semantic-partition DOF |", "| ---: | ---: | ---: |"]  # Start a concise GitHub-readable summary focused on the corrected causal question.
    for record in threshold_records:  # Append one like-for-like precision-resource comparison row per target.
        lines.append(f"| < {100.0 * float(record['target_rel_error']):.1f}% | {record['global_dof'] or 'not reached'} | {record['semantic_dof'] or 'not reached'} |")  # Report resource requirements at identical accuracy.
    lines.extend(["", "See `region_decisions.json` for whether the background region becomes active in later rounds."])  # Point directly to the diagnostic that tests the user's non-hotspot concern.
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")  # Persist the corrected human-readable summary.


def main() -> int:  # Execute the corrected full-domain semantic-partition experiment end to end.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure compact output paths exist before any possible numerical failure.
    CASES_DIR.mkdir(parents=True, exist_ok=True)  # Ensure transient case paths exist before meshing begins.
    gmsh_bin = core.find_executable("gmsh", [])  # Resolve the same Gmsh executable used by the validated experiments.
    ccx_bin = core.find_executable("ccx", ["ccx_2.23", "ccx_2.22", "ccx_2.21", "ccx_2.20"])  # Resolve the installed CalculiX executable across common Ubuntu package names.
    reference_state = core.solve_mesh("reference_global", gmsh_bin, ccx_bin, refinement_boxes=None, global_h=core.REFERENCE_H)  # Solve one common globally fine numerical reference before either adaptive history.
    reference = {"qoi": reference_state["qoi"], "dof": reference_state["dof"], "elements": reference_state["element_count"], "mesh_size": core.REFERENCE_H}  # Compact the common reference data used for both methods.
    global_rows, global_decisions = run_method("global_dorfler", reference["qoi"], gmsh_bin, ccx_bin)  # Execute conventional global Dörfler from the shared coarse state.
    semantic_rows, semantic_decisions = run_method("semantic_partition_dorfler", reference["qoi"], gmsh_bin, ccx_bin)  # Execute the corrected exhaustive semantic-partition hierarchy from the same coarse state.
    write_outputs(global_rows + semantic_rows, global_decisions + semantic_decisions, reference)  # Persist trajectories, common-accuracy comparisons, and region decisions.
    return 0  # Report successful completion after every compact result has been written.


if __name__ == "__main__":  # Execute only when this file is invoked directly by GitHub Actions or a local shell.
    try:  # Wrap the outermost call to preserve a concise CI-visible failure reason.
        raise SystemExit(main())  # Run the complete corrected benchmark and return its explicit shell status.
    except Exception as exc:  # Catch unexpected numerical or parsing failures at the command boundary.
        print(f"[fatal-v2] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Emit the exact failure class and message into the workflow log.
        raise  # Re-raise the original exception so GitHub Actions records a failed numerical run.
