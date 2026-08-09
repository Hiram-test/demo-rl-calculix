from __future__ import annotations  # Enable postponed type annotations for the corrected semantic wrapper.
import importlib.util  # Load the existing full-domain semantic-partition implementation without copying its solver core.
from pathlib import Path  # Resolve the existing experiment file through repository-relative paths.
import sys  # Register the dynamically loaded module and propagate explicit failures to GitHub Actions.

ROOT = Path(__file__).resolve().parent  # Anchor the corrected wrapper to the existing semantic-partition experiment directory.
BASE_PATH = ROOT / "run.py"  # Reuse the already validated full-domain partition, Gmsh, CalculiX, and output implementation.

spec = importlib.util.spec_from_file_location("semantic_partition_local_target_base", BASE_PATH)  # Build an import specification for the existing experiment implementation.
if spec is None or spec.loader is None:  # Guard against an unexpected repository state where the base file cannot be loaded.
    raise RuntimeError(f"Unable to load semantic-partition base from {BASE_PATH}")  # Fail explicitly before any numerical work begins.
base = importlib.util.module_from_spec(spec)  # Create the isolated module object that will host the imported implementation.
sys.modules[spec.name] = base  # Register the imported module so postponed annotations resolve correctly.
spec.loader.exec_module(base)  # Execute the existing experiment module without invoking its command-line main block.

def hierarchical_mark_local_target(eta2: dict[int, float], groups: dict[str, set[int]]) -> tuple[list[int], list[str], dict[str, float], float]:  # Select semantic regions first, then apply Dörfler relative to the selected candidate support.
    total_global = sum(eta2.values())  # Compute the complete current estimator mass for region-level comparison and diagnostics.
    if total_global <= 0.0:  # Require a nontrivial estimator distribution before semantic or element-level selection.
        raise RuntimeError("Global estimator mass is zero")  # Fail explicitly when no refinement signal exists.
    region_mass = {name: sum(eta2[element_id] for element_id in element_ids) for name, element_ids in groups.items()}  # Aggregate estimator mass inside each exhaustive semantic partition cell.
    ranked_regions = sorted(region_mass.items(), key=lambda item: item[1], reverse=True)  # Rank semantic cells by current estimator mass while allowing background to move up or down each round.
    region_target = base.REGION_THETA * total_global  # Use the existing region-level bulk rule only to decide which semantic cells form this round's candidate support.
    selected_regions: list[str] = []  # Accumulate the smallest descending-prefix set of semantic regions reaching the region-level target.
    selected_mass = 0.0  # Track estimator mass contained in the dynamically selected semantic support.
    for name, mass in ranked_regions:  # Traverse semantic regions from largest to smallest current aggregate estimator mass.
        selected_regions.append(name)  # Add the current region to this round's semantic action support.
        selected_mass += mass  # Add the current region's estimator mass to the support total.
        if selected_mass >= region_target:  # Stop once the semantic support captures the requested region-level bulk fraction.
            break  # Preserve the minimal region-level descending-prefix construction.
    eligible = set().union(*(groups[name] for name in selected_regions))  # Form the finite-element candidate pool from the dynamically selected semantic regions.
    eligible_mass = sum(eta2[element_id] for element_id in eligible)  # Compute the estimator mass of the actual candidate pool passed to Dörfler.
    if eligible_mass <= 0.0:  # Require a nontrivial candidate-pool estimator before local bulk marking.
        raise RuntimeError("Selected semantic support has zero estimator mass")  # Fail explicitly instead of silently reverting to the global method.
    ranked_elements = sorted(((element_id, eta2[element_id]) for element_id in eligible), key=lambda item: item[1], reverse=True)  # Rank only the elements admitted by the semantic support using the unchanged local indicator.
    element_target = base.core.THETA * eligible_mass  # Apply the same theta relative to the selected candidate support rather than forcing the semantic method to capture theta times the whole-domain estimator.
    marked: list[int] = []  # Accumulate the minimal descending-prefix element set satisfying the corrected local Dörfler criterion.
    captured = 0.0  # Track estimator mass captured inside the selected semantic support.
    for element_id, value in ranked_elements:  # Traverse eligible elements from largest to smallest indicator value.
        marked.append(element_id)  # Add the current highest-ranked eligible element to the marked set.
        captured += value  # Add the current element's estimator contribution to the local bulk accumulation.
        if captured >= element_target:  # Stop once the same theta fraction of the semantic candidate-pool estimator has been captured.
            break  # Preserve the minimal descending-prefix Dörfler construction inside the semantic support.
    return marked, selected_regions, region_mass, captured / total_global  # Return the corrected marked set and globally normalized captured mass for transparent comparison.

base.hierarchical_mark = hierarchical_mark_local_target  # Replace only the incorrectly normalized semantic element-level bulk target while keeping the full-domain partition and all numerical operators unchanged.

def main() -> int:  # Execute the corrected semantic-partition experiment with local candidate-pool Dörfler normalization.
    return base.main()  # Run the existing reference, global baseline, semantic history, common-accuracy comparison, and result writers unchanged.

if __name__ == "__main__":  # Execute the corrected experiment only when this wrapper is invoked as the program entry point.
    try:  # Keep GitHub Actions failure propagation explicit at the command boundary.
        raise SystemExit(main())  # Run the corrected experiment and return its shell status unchanged.
    except Exception as exc:  # Catch setup, meshing, solving, parsing, or marking failures only to emit a concise diagnostic.
        print(f"[fatal-local-target] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Write one clear failure reason to the persisted Actions log.
        raise  # Re-raise the original exception so GitHub Actions records a genuine failure.
