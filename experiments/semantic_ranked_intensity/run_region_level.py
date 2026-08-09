from __future__ import annotations  # Enable postponed annotations for the compact correction wrapper.
import csv  # Rewrite the corrected region-action diagnostics using the existing stable schema.
import importlib.util  # Load the first ranked-intensity implementation and override only its semantic support rule.
import json  # Correct the manifest and summary to describe region-level Dörfler accurately.
from pathlib import Path  # Resolve the sibling implementation and result paths portably.
import sys  # Register the imported implementation module and propagate explicit failures.

ROOT = Path(__file__).resolve().parent  # Anchor this correction wrapper to the ranked-intensity experiment directory.
IMPL_PATH = ROOT / "run.py"  # Reuse the validated solver, estimator, size-ratio refinement operator, and output machinery.

spec = importlib.util.spec_from_file_location("ranked_intensity_impl", IMPL_PATH)  # Build an import specification for the existing implementation.
if spec is None or spec.loader is None:  # Reject an unexpected repository state before numerical execution.
    raise RuntimeError(f"Unable to load ranked-intensity implementation from {IMPL_PATH}")  # Fail explicitly when the reusable implementation cannot be imported.
impl = importlib.util.module_from_spec(spec)  # Create the isolated module object for the existing implementation.
sys.modules[spec.name] = impl  # Register the module so postponed annotations and internal imports resolve correctly.
spec.loader.exec_module(impl)  # Execute the implementation without invoking its command-line main block.
core = impl.core  # Bind the validated numerical core under a short local name.
impl.AMR_ROUNDS = 4  # Use four adaptive transitions to limit explosive whole-region refinement while preserving the comparison mechanism.
core.AMR_ROUNDS = impl.AMR_ROUNDS  # Keep the numerical core round metadata synchronized with the corrected wrapper.
original_run_method = impl.run_method  # Preserve the conventional global baseline implementation unchanged.
original_write_outputs = impl.write_outputs  # Preserve the stable CSV and common-accuracy output writer before correcting its interpretation text.


def region_level_dorfler(eta2: dict[int, float], groups: dict[str, set[int]]) -> tuple[list[str], dict[str, float], float]:  # Apply ordinary Dörfler bulk marking to semantic regions as the action units.
    total_mass = sum(eta2.values())  # Compute the complete current estimator mass used by the common theta target.
    if total_mass <= 0.0:  # Require a nontrivial estimator distribution before selecting any semantic support.
        raise RuntimeError("Global estimator mass is zero")  # Fail explicitly rather than generating an undefined action.
    region_mass = {name: sum(eta2[element_id] for element_id in element_ids) for name, element_ids in groups.items()}  # Aggregate element indicators into one additive estimator mass per semantic region.
    ranked_regions = sorted(region_mass.items(), key=lambda item: item[1], reverse=True)  # Rank semantic action units by their current aggregate estimator mass.
    target = core.THETA * total_mass  # Use exactly the same fixed theta as conventional element-level global Dörfler.
    selected_regions: list[str] = []  # Accumulate the minimal descending prefix of semantic regions reaching the common bulk target.
    captured = 0.0  # Track estimator mass contained in the selected semantic support.
    for region_name, mass in ranked_regions:  # Traverse semantic action units from largest to smallest aggregate estimator mass.
        selected_regions.append(region_name)  # Admit the current semantic region as one indivisible action-support unit.
        captured += mass  # Add the complete region estimator mass to the Dörfler bulk accumulation.
        if captured >= target:  # Stop as soon as the selected semantic regions reach the same global theta fraction.
            break  # Preserve the minimal region-prefix Dörfler construction.
    if not selected_regions:  # Require a nonempty semantic support before physical refinement.
        raise RuntimeError("Region-level Dörfler selected no semantic region")  # Fail explicitly instead of producing an unchanged mesh.
    return selected_regions, region_mass, captured / total_mass  # Return the chosen region support, all region masses, and the actual captured global fraction.


def corrected_run_method(method: str, reference_qoi: float, gmsh_bin: str, ccx_bin: str) -> tuple[list[dict], list[dict]]:  # Preserve the baseline and replace only the semantic action-support representation.
    if method == "global_dorfler":  # Keep conventional element-level Dörfler exactly as implemented in the parent experiment.
        return original_run_method(method, reference_qoi, gmsh_bin, ccx_bin)  # Reuse the baseline without changing theta, estimator, refinement ratio, or numerical pipeline.
    refinement_history: list[dict[str, float]] = []  # Preserve all prior selected-region target-size actions across remeshing.
    rows: list[dict] = []  # Accumulate one solved precision-resource state per adaptive round.
    actions: list[dict] = []  # Record selected semantic regions and their LLM-ranked physical refinement intensity.
    for round_index in range(impl.AMR_ROUNDS + 1):  # Solve the common coarse state and each subsequent region-refined state.
        state = core.solve_mesh(f"{method}_r{round_index}", gmsh_bin, ccx_bin, refinement_boxes=refinement_history, global_h=None)  # Generate and solve the current physical mesh state.
        row = {"method": method, "round": round_index, "dof": state["dof"], "elements": state["element_count"], "qoi": state["qoi"], "qoi_rel_error": core.relative_error(state["qoi"], reference_qoi), "marked_elements": 0, "captured_global_fraction": 0.0}  # Record the solved state before optional support selection.
        if round_index < impl.AMR_ROUNDS:  # Select and refine semantic supports only when another adaptive state remains.
            eta2 = core.stress_jump_indicator(state["nodes"], state["tets"], state["stresses"])  # Compute the identical stress-jump estimator used by the baseline.
            groups = impl.base.partition_elements(state["nodes"], state["tets"])  # Partition every current tetrahedron into exactly one root, hole, or background semantic action unit.
            selected_regions, region_mass, captured_fraction = region_level_dorfler(eta2, groups)  # Apply the same fixed theta directly at the semantic-region level.
            connectivity_map = {element_id: connectivity for element_id, connectivity in state["tets"]}  # Build direct lookup for every current tetrahedron selected through a semantic region.
            selected_element_count = 0  # Count all physical elements belonging to the selected region support for diagnostics only.
            for region_name in selected_regions:  # Refine each Dörfler-selected semantic region as one action-support unit.
                level = impl.REGION_LEVEL[region_name]  # Read the frozen LLM ordinal hotspot intensity assigned before solving.
                ratio = impl.BASE_Q ** level  # Convert the ordinal level into the physical target-size ratio h_new/h_current.
                element_ids = groups[region_name]  # Take the complete selected semantic region rather than re-running element-level Dörfler inside it.
                for element_id in element_ids:  # Convert every element currently representing the selected semantic region into the ranked size field.
                    refinement_history.append(impl.ranked_refinement_box(state["nodes"], connectivity_map[element_id], level))  # Apply only the LLM-ranked size ratio to the selected support.
                selected_element_count += len(element_ids)  # Accumulate the current selected support size in element units for transparent diagnostics.
                actions.append({"method": method, "round": round_index, "region": region_name, "theta": core.THETA, "rank": level, "size_ratio": ratio, "region_mass": region_mass[region_name], "marked_elements": len(element_ids)})  # Persist the region-level Dörfler decision and its physical refinement intensity.
            row["marked_elements"] = selected_element_count  # Record how many current mesh elements lie inside the selected semantic support.
            row["captured_global_fraction"] = captured_fraction  # Record the estimator fraction represented by the selected region support.
        rows.append(row)  # Append the completed precision-resource state to the semantic trajectory.
        print(f"[{method}] round={round_index} dof={row['dof']} error={row['qoi_rel_error']:.6e} support_elements={row['marked_elements']} captured={row['captured_global_fraction']:.6f}", flush=True)  # Stream one concise corrected progress line into Actions logs.
    return rows, actions  # Return the corrected semantic trajectory and selected-region diagnostics.


def corrected_write_outputs(rows: list[dict], actions: list[dict], reference: dict) -> None:  # Reuse stable machine-readable outputs and replace only the experiment definition text.
    original_write_outputs(rows, actions, reference)  # Write history.csv, region_actions.csv, same_accuracy.csv, manifest.json, and the initial summary.
    manifest_path = impl.RESULTS_DIR / "manifest.json"  # Locate the manifest generated by the parent writer.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # Load the machine-readable experiment definition for correction.
    manifest["amr_rounds"] = impl.AMR_ROUNDS  # Record the corrected four-transition numerical history length.
    manifest["definition"] = "same theta for baseline and semantic marking; baseline Dörfler action units are elements, semantic Dörfler action units are LLM-defined regions; selected semantic regions are refined wholesale using frozen LLM ordinal size-ratio ranks"  # State the corrected action-support representation precisely.
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # Persist the corrected reproducibility definition.
    global_rows = [row for row in rows if row["method"] == "global_dorfler"]  # Extract the unchanged baseline trajectory.
    semantic_rows = [row for row in rows if row["method"] == "semantic_ranked_intensity"]  # Extract the corrected semantic region-level trajectory.
    thresholds = [0.10, 0.05, 0.03, 0.02, 0.015, 0.01]  # Use the same true-QoI accuracy targets as the parent experiment.
    lines = ["# Region-level Dörfler + LLM ranked refinement intensity", "", f"Reference QoI: `{reference['qoi']:.12e}` mm at `{reference['dof']}` DOF proxy.", "", f"Both methods use the same `theta = {core.THETA:.2f}` and the same unit size ratio `q = {impl.BASE_Q:.2f}`. The baseline applies Dörfler to elements and refines marked elements by one level. The semantic method applies Dörfler directly to the three LLM-defined region action units, then refines each selected region wholesale with frozen ranks root=3 (`0.512h`), hole=2 (`0.640h`), background=1 (`0.800h`).", "", "There is no second element-level Dörfler inside a selected semantic region, so the semantic action-support representation is not collapsed back to the global element ranking.", "", "| target relative error | global DOF | semantic region-level DOF |", "| ---: | ---: | ---: |"]  # Start the corrected concise interpretation.
    for threshold in thresholds:  # Compare minimum actual DOF needed to reach each common accuracy target.
        global_hit = impl.first_reaching(global_rows, threshold)  # Find the cheapest baseline state reaching the target.
        semantic_hit = impl.first_reaching(semantic_rows, threshold)  # Find the cheapest corrected semantic state reaching the same target.
        lines.append(f"| < {threshold:.3f} | {'' if global_hit is None else global_hit['dof']} | {'' if semantic_hit is None else semantic_hit['dof']} |")  # Emit one like-for-like resource comparison row.
    (impl.RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")  # Replace the parent summary with the corrected region-level experiment description.


impl.run_method = corrected_run_method  # Install region-level semantic Dörfler while preserving the conventional baseline implementation.
impl.write_outputs = corrected_write_outputs  # Install corrected experiment-definition outputs before calling the shared main routine.


if __name__ == "__main__":  # Execute the corrected experiment only when this wrapper is invoked directly.
    try:  # Preserve explicit status propagation around all native numerical and output failures.
        raise SystemExit(impl.main())  # Run the shared reference, unchanged baseline, corrected semantic trajectory, and corrected outputs.
    except Exception as exc:  # Catch the outermost failure only to emit one concise diagnostic before re-raising.
        print(f"[fatal-region-ranked-intensity] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Record the high-level failure reason in the persisted workflow console log.
        raise  # Re-raise so GitHub Actions records a genuine failure.
