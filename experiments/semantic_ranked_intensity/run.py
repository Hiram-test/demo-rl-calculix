from __future__ import annotations  # Enable compact postponed type annotations for the standalone benchmark.
import csv  # Write adaptive histories and common-accuracy comparisons as machine-readable tables.
import importlib.util  # Load the already validated geometry, solver, estimator, and semantic partition implementation.
import json  # Read the frozen LLM rank prior and persist region-level action diagnostics.
from pathlib import Path  # Resolve all repository-relative experiment paths portably on GitHub Actions.
import sys  # Register dynamically imported modules and propagate explicit failures to the workflow shell.

ROOT = Path(__file__).resolve().parent  # Anchor all experiment inputs, transient cases, and compact outputs here.
BASE_PATH = ROOT.parent / "semantic_partition_dorfler_v2" / "run.py"  # Reuse the validated full-domain root/hole/background partition and numerical core.
RANKING_PATH = ROOT / "semantic_ranking.json"  # Load the LLM-generated ordinal hotspot intensity prior frozen before solving.
CASES_DIR = ROOT / "cases"  # Keep transient Gmsh and CalculiX files outside compact result artifacts.
RESULTS_DIR = ROOT / "results"  # Store only CSV, JSON, Markdown, diagnostics, and plot inputs here.
AMR_ROUNDS = 5  # Solve the common coarse state plus five mark-refine-resolve transitions for each method.
BASE_Q = 0.80  # Define one universal refinement size ratio used as the unit intensity for both methods.
MIN_LOCAL_H = 2.0  # Keep adaptive local sizes above the globally finer reference mesh scale.
REFERENCE_H = 1.5  # Use a substantially finer global mesh as the fixed QoI reference for both trajectories.

spec = importlib.util.spec_from_file_location("ranked_semantic_base", BASE_PATH)  # Build an import specification for the validated sibling experiment.
if spec is None or spec.loader is None:  # Reject an unexpected repository state before any native numerical command starts.
    raise RuntimeError(f"Unable to load validated semantic partition base from {BASE_PATH}")  # Fail explicitly if the reusable implementation is unavailable.
base = importlib.util.module_from_spec(spec)  # Create the isolated imported module object.
sys.modules[spec.name] = base  # Register the module so postponed annotations and internal imports resolve correctly.
spec.loader.exec_module(base)  # Execute the validated module without invoking its command-line main block.
core = base.core  # Bind the validated Gmsh/CalculiX/estimator numerical core under a short local name.
core.CASES_DIR = CASES_DIR  # Redirect all transient numerical cases into this dedicated experiment directory.
core.RESULTS_DIR = RESULTS_DIR  # Redirect any shared-core compact outputs into this experiment result directory.
core.AMR_ROUNDS = AMR_ROUNDS  # Keep shared numerical iteration metadata synchronized with this benchmark.
core.REFINE_FACTOR = BASE_Q  # Make one baseline refinement action mean h_new = 0.80 h_current everywhere.
core.MIN_LOCAL_H = MIN_LOCAL_H  # Replace the old 3.2 mm floor with a lower floor suitable for this comparison.

ranking = json.loads(RANKING_PATH.read_text(encoding="utf-8"))  # Read the frozen LLM ordinal semantic hotspot prior.
REGION_LEVEL = {name: int(record["rank"]) for name, record in ranking["regions"].items()}  # Extract one integer refinement intensity level per semantic region.
for region_name in ("root_region", "hole_region", "background_region"):  # Require the complete exhaustive partition to have an explicit LLM intensity.
    if region_name not in REGION_LEVEL:  # Detect an incomplete or malformed semantic prior before solving.
        raise RuntimeError(f"Missing LLM refinement rank for {region_name}")  # Fail instead of silently assigning an undocumented default intensity.
    if REGION_LEVEL[region_name] < 1:  # Require every active domain to receive at least the common unit refinement intensity.
        raise RuntimeError(f"Invalid nonpositive refinement rank for {region_name}")  # Reject a semantic prior that would disable a region implicitly.


def same_theta_regional_mark(eta2: dict[int, float], groups: dict[str, set[int]]) -> tuple[dict[str, list[int]], dict[str, float], float]:  # Apply the identical Dörfler theta independently inside every semantic region.
    total_global = sum(eta2.values())  # Compute the complete current estimator mass for global diagnostics.
    if total_global <= 0.0:  # Require a nontrivial estimator distribution before any marking decision.
        raise RuntimeError("Global estimator mass is zero")  # Fail explicitly when the current state provides no actionable refinement signal.
    region_mass = {name: sum(eta2[element_id] for element_id in element_ids) for name, element_ids in groups.items()}  # Aggregate current estimator mass inside each semantic region.
    marked_by_region: dict[str, list[int]] = {}  # Store the minimal Dörfler prefix selected independently in every semantic region.
    captured_global = 0.0  # Accumulate the absolute estimator mass captured by the union of all regional marked prefixes.
    for region_name, element_ids in groups.items():  # Process the exhaustive root, hole, and background partition without gating any domain out.
        local_mass = region_mass[region_name]  # Read this region's current estimator mass.
        if not element_ids or local_mass <= 0.0:  # Skip only genuinely empty or numerically zero-signal regions.
            marked_by_region[region_name] = []  # Preserve an explicit empty record for downstream diagnostics.
            continue  # Advance because no local Dörfler action is required in this degenerate region.
        target = core.THETA * local_mass  # Use exactly the same theta as the conventional global Dörfler baseline.
        ranked = sorted(((element_id, eta2[element_id]) for element_id in element_ids), key=lambda item: item[1], reverse=True)  # Rank only the elements belonging to this semantic region.
        local_marked: list[int] = []  # Accumulate the minimal descending regional prefix satisfying the common bulk fraction.
        local_captured = 0.0  # Track estimator mass captured inside this semantic region.
        for element_id, value in ranked:  # Traverse regional elements from largest to smallest error indicator.
            local_marked.append(element_id)  # Mark the current highest-ranked regional element.
            local_captured += value  # Add its estimator contribution to the local bulk accumulation.
            if local_captured >= target:  # Stop when this region reaches the same fixed theta fraction of its own estimator mass.
                break  # Preserve the minimal ordinary Dörfler prefix within this semantic region.
        marked_by_region[region_name] = local_marked  # Save the exact selected element identifiers for region-dependent refinement intensity.
        captured_global += local_captured  # Add the regional captured mass to the full-domain diagnostic total.
    if not any(marked_by_region.values()):  # Require at least one marked element across the exhaustive semantic partition.
        raise RuntimeError("Regional fixed-theta marking produced no marked elements")  # Fail instead of generating an unchanged adaptive mesh.
    return marked_by_region, region_mass, captured_global / total_global  # Return regional marked sets, regional estimator masses, and the union's global captured fraction.


def ranked_refinement_box(nodes: dict[int, tuple[float, float, float]], connectivity: tuple[int, int, int, int], level: int) -> dict[str, float]:  # Convert one marked element into a target-size box whose depth follows the LLM ordinal intensity.
    box = core.refinement_box(nodes, connectivity)  # Build the validated one-level box using h_new = BASE_Q times the current local element scale.
    one_level_vin = float(box["vin"])  # Read the one-level target size produced by the common refinement operator.
    deeper_vin = one_level_vin * (BASE_Q ** (level - 1))  # Apply only additional powers of the same universal size ratio for stronger semantic intensity levels.
    box["vin"] = max(MIN_LOCAL_H, deeper_vin)  # Enforce the common numerical floor after converting ordinal rank into a physical target size.
    return box  # Return the ordinary geometric refinement neighbourhood with only its target size changed by semantic rank.


def run_method(method: str, reference_qoi: float, gmsh_bin: str, ccx_bin: str) -> tuple[list[dict], list[dict]]:  # Execute either global fixed-intensity Dörfler or fixed-theta semantic ranked-intensity refinement.
    refinement_history: list[dict[str, float]] = []  # Preserve every prior physical target-size box so refined regions remain refined on remeshing.
    rows: list[dict] = []  # Accumulate one precision-resource state per solved adaptive round.
    actions: list[dict] = []  # Accumulate transparent per-round region-level marking and intensity decisions.
    for round_index in range(AMR_ROUNDS + 1):  # Solve the shared coarse mesh and each successive adaptive mesh state.
        state = core.solve_mesh(f"{method}_r{round_index}", gmsh_bin, ccx_bin, refinement_boxes=refinement_history, global_h=None)  # Generate, solve, and parse the current physical mesh state.
        row = {"method": method, "round": round_index, "dof": state["dof"], "elements": state["element_count"], "qoi": state["qoi"], "qoi_rel_error": core.relative_error(state["qoi"], reference_qoi), "marked_elements": 0, "captured_global_fraction": 0.0}  # Record the solved precision-resource state before optional marking.
        if round_index < AMR_ROUNDS:  # Mark and construct the next mesh only when another adaptive state remains to be solved.
            eta2 = core.stress_jump_indicator(state["nodes"], state["tets"], state["stresses"])  # Compute the identical stress-jump estimator for both methods.
            connectivity_map = {element_id: connectivity for element_id, connectivity in state["tets"]}  # Build direct connectivity lookup for all marked element identifiers.
            if method == "global_dorfler":  # Apply the conventional baseline with one global element ranking and one unit refinement intensity.
                all_elements = set(connectivity_map)  # Admit every current tetrahedron to the baseline Dörfler competition.
                marked, captured_fraction = core.dorfler_mark(eta2, all_elements)  # Select the ordinary global minimal prefix at the unchanged theta value.
                new_boxes = [core.refinement_box(state["nodes"], connectivity_map[element_id]) for element_id in marked]  # Refine every baseline marked element by exactly one universal size-ratio level.
                refinement_history.extend(new_boxes)  # Preserve the baseline target-size action in all future remeshed states.
                row["marked_elements"] = len(marked)  # Record the baseline marked-set cardinality for diagnostic comparison.
                row["captured_global_fraction"] = captured_fraction  # Record the actual baseline estimator mass captured after discrete prefix overshoot.
                actions.append({"method": method, "round": round_index, "region": "all_domain", "theta": core.THETA, "rank": 1, "size_ratio": BASE_Q, "region_mass": sum(eta2.values()), "marked_elements": len(marked)})  # Persist the baseline action using the same schema as semantic region actions.
            else:  # Apply the semantic method with the same theta but LLM-ranked physical refinement depth per region.
                groups = base.partition_elements(state["nodes"], state["tets"])  # Partition the complete current mesh into root, hole, and residual background regions.
                marked_by_region, region_mass, captured_fraction = same_theta_regional_mark(eta2, groups)  # Detect marked elements independently in every region using the same fixed theta.
                total_marked = 0  # Count all semantic marked elements across the exhaustive partition.
                for region_name, marked_ids in marked_by_region.items():  # Convert each region's Dörfler set into region-ranked physical target sizes.
                    level = REGION_LEVEL[region_name]  # Read the frozen LLM ordinal hotspot intensity for this semantic region.
                    ratio = BASE_Q ** level  # Convert the ordinal level into the documented physical target-size ratio h_new/h_current.
                    for element_id in marked_ids:  # Apply the region's common intensity to every element detected by the same-theta regional Dörfler rule.
                        refinement_history.append(ranked_refinement_box(state["nodes"], connectivity_map[element_id], level))  # Add the region-ranked physical target-size box to persistent refinement history.
                    total_marked += len(marked_ids)  # Accumulate the complete semantic marked-set cardinality.
                    actions.append({"method": method, "round": round_index, "region": region_name, "theta": core.THETA, "rank": level, "size_ratio": ratio, "region_mass": region_mass[region_name], "marked_elements": len(marked_ids)})  # Persist the exact regional detection and intensity decision.
                row["marked_elements"] = total_marked  # Record the union cardinality of all fixed-theta regional marked sets.
                row["captured_global_fraction"] = captured_fraction  # Record the estimator mass captured by the regional Dörfler union relative to the global estimator mass.
        rows.append(row)  # Append the completed precision-resource state to the current method trajectory.
        print(f"[{method}] round={round_index} dof={row['dof']} error={row['qoi_rel_error']:.6e} marked={row['marked_elements']} captured={row['captured_global_fraction']:.6f}", flush=True)  # Stream one concise progress line into GitHub Actions logs.
    return rows, actions  # Return the complete solved trajectory and transparent action diagnostics.


def first_reaching(rows: list[dict], threshold: float) -> dict | None:  # Find the least-resource solved state satisfying one common error requirement.
    satisfying = [row for row in rows if float(row["qoi_rel_error"]) <= threshold]  # Collect every solved state that reaches the requested error threshold.
    return None if not satisfying else min(satisfying, key=lambda row: int(row["dof"]))  # Return the satisfying state with the smallest actual DOF resource cost.


def write_outputs(rows: list[dict], actions: list[dict], reference: dict) -> None:  # Persist compact reproducible tables and the main precision-resource interpretation.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the compact output directory exists before writing any artifact.
    history_fields = ["method", "round", "dof", "elements", "qoi", "qoi_rel_error", "marked_elements", "captured_global_fraction"]  # Define the stable adaptive-history schema.
    with (RESULTS_DIR / "history.csv").open("w", encoding="utf-8", newline="") as handle:  # Open the complete solved-state trajectory table.
        writer = csv.DictWriter(handle, fieldnames=history_fields)  # Create a deterministic CSV writer for both methods.
        writer.writeheader()  # Emit the documented history header.
        writer.writerows({field: row[field] for field in history_fields} for row in rows)  # Persist every solved global and semantic precision-resource state.
    action_fields = ["method", "round", "region", "theta", "rank", "size_ratio", "region_mass", "marked_elements"]  # Define the transparent action-level diagnostic schema.
    with (RESULTS_DIR / "region_actions.csv").open("w", encoding="utf-8", newline="") as handle:  # Open the region detection and intensity table.
        writer = csv.DictWriter(handle, fieldnames=action_fields)  # Create a deterministic action writer.
        writer.writeheader()  # Emit the action table header.
        writer.writerows({field: action[field] for field in action_fields} for action in actions)  # Persist every baseline and semantic refinement action.
    global_rows = [row for row in rows if row["method"] == "global_dorfler"]  # Extract the conventional global Dörfler trajectory.
    semantic_rows = [row for row in rows if row["method"] == "semantic_ranked_intensity"]  # Extract the LLM-ranked semantic refinement trajectory.
    thresholds = [0.10, 0.05, 0.03, 0.02, 0.015, 0.01]  # Define common accuracy targets for like-for-like resource comparison.
    comparisons: list[dict] = []  # Accumulate minimum DOF required by each method at every common accuracy target.
    for threshold in thresholds:  # Evaluate both trajectories at exactly the same true-QoI error requirement.
        global_hit = first_reaching(global_rows, threshold)  # Find the cheapest global state reaching this common target.
        semantic_hit = first_reaching(semantic_rows, threshold)  # Find the cheapest semantic state reaching the same target.
        comparisons.append({"target_rel_error": threshold, "global_dof": "" if global_hit is None else global_hit["dof"], "semantic_dof": "" if semantic_hit is None else semantic_hit["dof"]})  # Store only comparable resource costs.
    with (RESULTS_DIR / "same_accuracy.csv").open("w", encoding="utf-8", newline="") as handle:  # Open the primary fair-comparison table.
        writer = csv.DictWriter(handle, fieldnames=["target_rel_error", "global_dof", "semantic_dof"])  # Define the minimal common-accuracy schema.
        writer.writeheader()  # Emit the table header.
        writer.writerows(comparisons)  # Persist all common-accuracy resource comparisons.
    manifest = {"reference": reference, "theta": core.THETA, "base_size_ratio_q": BASE_Q, "minimum_local_h_mm": MIN_LOCAL_H, "reference_h_mm": REFERENCE_H, "amr_rounds": AMR_ROUNDS, "semantic_ranks": REGION_LEVEL, "semantic_size_ratios": {name: BASE_Q ** level for name, level in REGION_LEVEL.items()}, "definition": "same theta in every semantic region; LLM ordinal rank controls only physical refinement size ratio; global baseline uses the same theta and one unit refinement level", "fairness_rule": "interpret only true QoI error versus actual DOF and minimum DOF at common error targets"}  # Record every causal and numerical setting defining the experiment.
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # Persist the complete experiment definition for reproducibility.
    lines = ["# Fixed-theta Dörfler + LLM ranked refinement intensity", "", f"Reference QoI: `{reference['qoi']:.12e}` mm at `{reference['dof']}` DOF proxy.", "", f"All Dörfler detection uses the same `theta = {core.THETA:.2f}`. The global baseline refines every marked element by one size-ratio level `q = {BASE_Q:.2f}`. The semantic method keeps the same theta but maps frozen LLM ranks to physical size ratios: root `{BASE_Q ** REGION_LEVEL['root_region']:.3f}`, hole `{BASE_Q ** REGION_LEVEL['hole_region']:.3f}`, background `{BASE_Q ** REGION_LEVEL['background_region']:.3f}`.", "", "No theta modulation and no estimator-derived semantic ranking are used. The only semantic intervention is region-dependent refinement depth through a fixed ordinal size-ratio mapping.", "", "| target relative error | global DOF | semantic ranked-intensity DOF |", "| ---: | ---: | ---: |"]  # Start a concise GitHub-readable summary of the purified causal comparison.
    for record in comparisons:  # Add one row for every common true-QoI error target.
        lines.append(f"| < {record['target_rel_error']:.3f} | {record['global_dof']} | {record['semantic_dof']} |")  # Emit the minimum-resource comparison without comparing unequal final-round states.
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")  # Persist the compact human-readable interpretation beside the machine-readable tables.


def main() -> int:  # Execute the complete reference plus both adaptive histories and write reproducible outputs.
    CASES_DIR.mkdir(parents=True, exist_ok=True)  # Ensure transient numerical cases have a destination before Gmsh starts.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure compact output artifacts have a destination before the first solve.
    gmsh_bin = core.find_executable("gmsh", [])  # Resolve the GitHub runner's Gmsh executable through the validated core helper.
    ccx_bin = core.find_executable("ccx", ["ccx_2.23", "ccx_2.22", "ccx_2.21"])  # Resolve the available CalculiX executable across common Ubuntu package names.
    reference_state = core.solve_mesh("reference_ranked_intensity", gmsh_bin, ccx_bin, refinement_boxes=[], global_h=REFERENCE_H)  # Generate and solve the shared globally fine reference mesh.
    reference = {"qoi": reference_state["qoi"], "dof": reference_state["dof"], "elements": reference_state["element_count"], "global_h": REFERENCE_H}  # Freeze the common QoI reference metadata before either adaptive method runs.
    global_rows, global_actions = run_method("global_dorfler", reference["qoi"], gmsh_bin, ccx_bin)  # Run the conventional global Dörfler trajectory with one unit refinement level.
    semantic_rows, semantic_actions = run_method("semantic_ranked_intensity", reference["qoi"], gmsh_bin, ccx_bin)  # Run the same-theta semantic trajectory with LLM-ranked refinement depth.
    write_outputs(global_rows + semantic_rows, global_actions + semantic_actions, reference)  # Persist all fair precision-resource and action-level diagnostics.
    return 0  # Return success only after all numerical states and compact outputs are written.


if __name__ == "__main__":  # Execute the benchmark only when this file is invoked as the program entry point.
    try:  # Preserve explicit diagnostic propagation around native meshing, solving, parsing, and marking failures.
        raise SystemExit(main())  # Run the complete experiment and return its status unchanged to GitHub Actions.
    except Exception as exc:  # Catch the outermost failure only to print one concise diagnostic before re-raising.
        print(f"[fatal-ranked-intensity] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Record the exact high-level failure reason in the persisted workflow console log.
        raise  # Re-raise the original exception so the Actions job cannot appear successful after a numerical failure.
