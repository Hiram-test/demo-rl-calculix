from __future__ import annotations  # Enable postponed type annotations for the small experiment wrapper.
import importlib.util  # Load the validated full-domain semantic-partition implementation without copying its numerical core.
from pathlib import Path  # Resolve repository-relative experiment paths portably on GitHub Actions.
import sys  # Register the imported module and propagate explicit failures to the shell.

ROOT = Path(__file__).resolve().parent  # Anchor all new cases and results to this dedicated experiment directory.
BASE_PATH = ROOT.parent / "semantic_partition_dorfler_v2" / "run.py"  # Reuse the validated partition geometry, CalculiX solver, estimator, and remesher.
CASES_DIR = ROOT / "cases"  # Store transient Gmsh and CalculiX files outside the compact result directory.
RESULTS_DIR = ROOT / "results"  # Store only compact CSV, JSON, Markdown, and plot artifacts here.
AMR_ROUNDS = 6  # Solve the shared coarse state plus six adaptive refinement states for each method.

spec = importlib.util.spec_from_file_location("local_normalized_base", BASE_PATH)  # Build an import specification for the validated sibling implementation.
if spec is None or spec.loader is None:  # Reject an unexpected repository state before numerical execution starts.
    raise RuntimeError(f"Unable to load semantic-partition base from {BASE_PATH}")  # Fail explicitly when the validated base implementation cannot be imported.
base = importlib.util.module_from_spec(spec)  # Create the isolated Python module object for the validated base implementation.
sys.modules[spec.name] = base  # Register the module so postponed annotations and internal imports resolve correctly.
spec.loader.exec_module(base)  # Execute the base module without invoking its command-line main block.
base.CASES_DIR = CASES_DIR  # Redirect the base experiment transient cases into this experiment directory.
base.RESULTS_DIR = RESULTS_DIR  # Redirect the base experiment compact outputs into this result directory.
base.AMR_ROUNDS = AMR_ROUNDS  # Extend the adaptive history to six refinement rounds.
base.core.CASES_DIR = CASES_DIR  # Redirect the shared numerical core transient cases into this experiment directory.
base.core.RESULTS_DIR = RESULTS_DIR  # Redirect any shared-core compact outputs into this result directory.
base.core.AMR_ROUNDS = AMR_ROUNDS  # Keep the numerical core round count synchronized with the wrapper.


def independently_normalized_mark(eta2: dict[int, float], groups: dict[str, set[int]]) -> tuple[list[int], list[str], dict[str, float], float]:  # Apply one ordinary Dörfler solve inside every semantic normalization domain and union the resulting marked sets.
    total_global = sum(eta2.values())  # Compute the complete current estimator mass for diagnostics and global consistency checks.
    if total_global <= 0.0:  # Require a nontrivial global estimator distribution before marking.
        raise RuntimeError("Global estimator mass is zero")  # Fail explicitly when the current mesh provides no actionable estimator information.
    region_mass = {name: sum(eta2[element_id] for element_id in element_ids) for name, element_ids in groups.items()}  # Aggregate the estimator mass inside each semantic normalization domain.
    marked_union: set[int] = set()  # Accumulate the union of independently marked elements across all semantic domains.
    active_regions: list[str] = []  # Record every nonempty semantic normalization domain used in this round.
    captured_global_mass = 0.0  # Track the absolute estimator mass captured by the final union of marked elements.
    for region_name, element_ids in groups.items():  # Process root, hole, and background as separate normalization domains.
        if not element_ids:  # Skip a semantic region only when the current mesh contains no centroid assigned to it.
            continue  # Advance to the next semantic normalization domain without creating an empty Dörfler problem.
        local_mass = region_mass[region_name]  # Read the current estimator mass inside this semantic domain.
        if local_mass <= 0.0:  # Skip a domain whose current estimator mass vanishes numerically.
            continue  # Advance because no local refinement signal exists in this domain.
        local_target = base.core.THETA * local_mass  # Apply the unchanged Dörfler fraction to this region's own estimator mass.
        ranked = sorted(((element_id, eta2[element_id]) for element_id in element_ids), key=lambda item: item[1], reverse=True)  # Rank only elements sharing the same semantic normalization domain.
        local_captured = 0.0  # Track the estimator mass accumulated inside the current region.
        local_marked: list[int] = []  # Accumulate the minimal descending prefix satisfying the current region's local bulk condition.
        for element_id, value in ranked:  # Traverse current-region elements from largest to smallest local indicator value.
            local_marked.append(element_id)  # Add the current highest-ranked regional element to the local marked prefix.
            local_captured += value  # Add the current element's estimator contribution to the local bulk accumulation.
            if local_captured >= local_target:  # Stop once the same theta fraction of the current region's estimator mass is captured.
                break  # Preserve the minimal descending-prefix Dörfler construction inside this semantic domain.
        marked_union.update(local_marked)  # Merge the current region's marked prefix into the global refinement action.
        active_regions.append(region_name)  # Record that this semantic normalization domain participated in the current action.
    if not marked_union:  # Require at least one marked finite element before attempting a remesh.
        raise RuntimeError("Independent semantic normalization produced no marked elements")  # Fail explicitly instead of silently generating an unchanged mesh.
    captured_global_mass = sum(eta2[element_id] for element_id in marked_union)  # Compute the absolute global estimator mass captured by the union of all local Dörfler prefixes.
    if captured_global_mass + 1.0e-12 < base.core.THETA * total_global:  # Verify the partition-of-unity implication of local bulk marking up to floating-point tolerance.
        raise RuntimeError("Independent regional Dörfler union failed the implied global bulk condition")  # Reject an implementation or partition inconsistency that violates the expected aggregate inequality.
    marked = sorted(marked_union)  # Convert the marked-element union into a deterministic ordered list for reproducible remeshing and logging.
    return marked, active_regions, region_mass, captured_global_mass / total_global  # Return the union, active normalization domains, regional masses, and globally normalized captured fraction.

base.hierarchical_mark = independently_normalized_mark  # Replace only the semantic marking rule while preserving geometry, estimator, solver, remeshing, and global baseline.
original_write_outputs = base.write_outputs  # Keep a reference to the validated output writer before wrapping its explanatory summary.


def write_outputs_with_local_normalization(rows: list[dict], decisions: list[dict], reference: dict) -> None:  # Reuse machine-readable outputs and replace only the human-readable interpretation with the correct experiment definition.
    original_write_outputs(rows, decisions, reference)  # Write history.csv, region_decisions.json, same_accuracy.csv, manifest.json, and the base summary first.
    global_rows = [row for row in rows if row["method"] == "global_dorfler"]  # Extract the conventional global Dörfler trajectory.
    semantic_rows = [row for row in rows if row["method"] == "semantic_partition_dorfler"]  # Extract the independently normalized semantic trajectory.
    lines: list[str] = []  # Accumulate a concise corrected Markdown report for direct GitHub inspection.
    lines.append("# Independent semantic normalization domains + Dörfler")  # State the exact semantic intervention in the report title.
    lines.append("")  # Add one Markdown spacer line.
    lines.append(f"Reference QoI: `{reference['qoi']:.12e}` mm at `{reference['dof']}` DOF proxy.")  # State the fixed reference shared by both adaptive histories.
    lines.append("")  # Add one Markdown spacer line.
    lines.append("The full domain is partitioned into three exhaustive semantic normalization domains: root, hole, and residual background. Every nonempty domain independently applies the same `theta = 0.50` Dörfler bulk rule to its own estimator mass, and the three marked prefixes are united before remeshing.")  # Define the accepted local-normalization algorithm precisely.
    lines.append("")  # Add one Markdown spacer line.
    lines.append("Because the regions form a disjoint exhaustive partition, the union automatically captures at least the same global theta fraction of estimator mass, while preventing one high-magnitude physical mechanism from completely suppressing another normalization domain.")  # State the structural consequence of independent regional normalization.
    lines.append("")  # Add one Markdown spacer line.
    lines.append("| round | global DOF | global relative error | semantic DOF | semantic relative error | semantic marked elements |")  # Start the direct trajectory table.
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")  # Add Markdown table alignment markers.
    for global_row, semantic_row in zip(global_rows, semantic_rows):  # Compare solved states with the same adaptive-round index side by side.
        lines.append(f"| {global_row['round']} | {global_row['dof']} | {global_row['qoi_rel_error']:.6e} | {semantic_row['dof']} | {semantic_row['qoi_rel_error']:.6e} | {semantic_row['marked_elements']} |")  # Emit one roundwise precision-resource comparison row.
    lines.append("")  # Add one Markdown spacer line.
    lines.append("Interpretation must use the error–DOF trajectory or minimum DOF at common error targets; equal-round states are not resource matched.")  # State the resource-fair comparison rule explicitly.
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")  # Replace the base explanatory summary with the correct local-normalization report.

base.write_outputs = write_outputs_with_local_normalization  # Install the corrected output writer into the imported experiment before execution.


def main() -> int:  # Execute the conventional baseline and the independently normalized semantic partition through the same numerical pipeline.
    CASES_DIR.mkdir(parents=True, exist_ok=True)  # Ensure transient numerical cases have a destination before Gmsh starts.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure compact result artifacts have a destination before the first solve.
    return base.main()  # Run the shared reference solution, both adaptive histories, region decision logs, and resource comparisons.


if __name__ == "__main__":  # Execute the experiment only when this wrapper is called as the program entry point.
    try:  # Keep GitHub Actions failure propagation explicit at the outermost command boundary.
        raise SystemExit(main())  # Run the complete experiment and return its shell status unchanged.
    except Exception as exc:  # Catch setup, meshing, solving, parsing, or marking failures only to emit a concise diagnostic.
        print(f"[fatal-local-normalization] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Write one clear failure reason into the persisted workflow console log.
        raise  # Re-raise the original exception so GitHub Actions records a genuine failure.
