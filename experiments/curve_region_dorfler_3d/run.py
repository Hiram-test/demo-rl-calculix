from __future__ import annotations  # Enable compact forward-compatible type annotations.
import csv  # Write a compact threshold-comparison table beside the adaptive histories.
import importlib.util  # Load the already validated Dörfler solver implementation from the sibling experiment.
import json  # Read the frozen freeform semantic regions and write the experiment manifest.
import math  # Evaluate circular semantic-region membership in physical coordinates.
from pathlib import Path  # Resolve repository-relative experiment paths portably.
import sys  # Register the dynamically loaded solver module and propagate failures to CI.

ROOT = Path(__file__).resolve().parent  # Anchor all new curve-region experiment files to this directory.
CASES_DIR = ROOT / "cases"  # Store transient meshes and CalculiX files outside the compact results folder.
RESULTS_DIR = ROOT / "results"  # Persist only small tables, summaries, and diagrams here.
REGIONS_PATH = ROOT / "semantic_regions.json"  # Read the LLM-drawn geometric support frozen before solving.
CORE_PATH = ROOT.parent / "llm_dorfler_3d" / "run.py"  # Reuse the already validated mesh, solver, indicator, and refinement implementation.

spec = importlib.util.spec_from_file_location("dorfler_core", CORE_PATH)  # Build an import specification for the validated sibling solver file.
if spec is None or spec.loader is None:  # Reject an unexpected repository state where the sibling solver cannot be imported.
    raise RuntimeError(f"Unable to load validated Dörfler core from {CORE_PATH}")  # Fail explicitly before any numerical work begins.
core = importlib.util.module_from_spec(spec)  # Create the isolated Python module object for the validated solver implementation.
sys.modules[spec.name] = core  # Register the dynamic module so its internal imports and annotations resolve normally.
spec.loader.exec_module(core)  # Execute the validated solver module without running its command-line main block.
core.CASES_DIR = CASES_DIR  # Redirect all transient numerical cases into the new curve-region experiment directory.
core.RESULTS_DIR = RESULTS_DIR  # Redirect any shared compact outputs into the new curve-region result directory.


def load_regions() -> dict:  # Read the frozen LLM freeform semantic-region document committed before numerical evaluation.
    return json.loads(REGIONS_PATH.read_text(encoding="utf-8"))  # Parse the UTF-8 JSON document into ordinary Python containers.


def point_in_region(point: tuple[float, float, float], region: dict) -> bool:  # Test one tetrahedron centroid against one continuous semantic drawing primitive.
    x, _y, z = point  # Unpack the coordinates needed by the current root-band and hole-circle primitives.
    region_type = region["type"]  # Read the explicit primitive type so unsupported drawings cannot be silently approximated.
    if region_type == "root_band":  # Interpret the root annotation as a full-cross-section axial band beside the clamp.
        xmin = float(region["xmin_mm"])  # Read the lower axial edge of the hand-drawn root band.
        xmax = float(region["xmax_mm"])  # Read the upper axial edge of the hand-drawn root band.
        return xmin - core.GEOM_TOL <= x <= xmax + core.GEOM_TOL  # Accept every material point whose x coordinate falls inside the root band.
    if region_type == "hole_annulus":  # Interpret the hole annotation as a circular envelope in the x-z plane extruded along y.
        cx = float(region["center_x_mm"])  # Read the drawn circle centre along the beam axis.
        cz = float(region["center_z_mm"])  # Read the drawn circle centre in the vertical direction.
        outer_radius = float(region["outer_radius_mm"])  # Read the outer radius of the semantic circle drawn around the physical hole.
        radial_distance = math.hypot(x - cx, z - cz)  # Compute physical distance from the element centroid to the hole centre in the transverse section.
        return radial_distance <= outer_radius + core.GEOM_TOL  # Accept material elements whose centroids fall inside the drawn circular envelope.
    raise ValueError(f"Unsupported semantic region primitive: {region_type}")  # Fail explicitly when a new drawing primitive has not been implemented.


def eligible_elements(method: str, nodes: dict[int, tuple[float, float, float]], tets: list[tuple[int, tuple[int, int, int, int]]], selected_regions: list[dict]) -> set[int]:  # Build the global or freeform-semantic Dörfler candidate pool on the current adaptive mesh.
    if method == "global_dorfler":  # Preserve the conventional baseline with every current tetrahedron eligible for marking.
        return {element_id for element_id, _connectivity in tets}  # Return the complete current element identifier set unchanged.
    eligible: set[int] = set()  # Accumulate current elements whose centroids lie inside at least one LLM-drawn continuous region.
    for element_id, connectivity in tets:  # Test every current tetrahedron against the frozen semantic drawing.
        centroid = core.tet_centroid(nodes, connectivity)  # Compute the current element centroid in physical coordinates independently of mesh numbering.
        if any(point_in_region(centroid, region) for region in selected_regions):  # Keep the element when the human-like drawn support contains its centroid.
            eligible.add(element_id)  # Add the current element to the semantic Dörfler candidate pool.
    if not eligible:  # Reject a semantic drawing that accidentally contains no current finite elements.
        raise RuntimeError("Freeform LLM semantic support contains no current elements")  # Fail rather than silently reverting to global Dörfler.
    return eligible  # Return the current mesh elements admitted by the continuous semantic support.


core.eligible_elements = eligible_elements  # Replace only the old atom-gating function while leaving the validated Dörfler algorithm unchanged.


def first_reaching(rows: list[dict], threshold: float) -> dict | None:  # Find the first adaptive state whose QoI error reaches a requested target.
    for row in rows:  # Traverse the adaptive history in chronological order.
        if float(row["qoi_rel_error"]) <= threshold:  # Test whether the current state satisfies the requested precision threshold.
            return row  # Return the earliest satisfying state so its DOF is the relevant resource cost.
    return None  # Return no state when the finite adaptive history never reaches the requested threshold.


def write_region_preview(document: dict) -> None:  # Draw a dependency-free side-view SVG of the actual LLM semantic support used by the solver.
    width = 1100  # Define the SVG canvas width in pixels.
    height = 420  # Define the SVG canvas height in pixels.
    beam_x = 150  # Place the beam left face away from the text margin.
    beam_y = 125  # Place the beam vertically near the centre of the canvas.
    beam_w = 760  # Draw a long beam body whose horizontal scale represents the 120 mm model length.
    beam_h = 170  # Draw the 40 mm section as a readable side-view height.
    xscale = beam_w / core.LENGTH  # Convert physical axial millimetres into SVG pixels.
    zscale = beam_h / (2.0 * core.HALF_HEIGHT)  # Convert physical vertical millimetres into SVG pixels.
    hole_cx = beam_x + core.HOLE_X * xscale  # Map the physical hole centre x coordinate into the SVG side view.
    hole_cy = beam_y + beam_h / 2.0  # Place the transverse-hole centre on the side-view mid-height.
    hole_r = core.HOLE_RADIUS * zscale  # Map the actual radius-eight hole into SVG pixels.
    root_region = next(region for region in document["selected_regions"] if region["type"] == "root_band")  # Retrieve the frozen root-band drawing primitive.
    hole_region = next(region for region in document["selected_regions"] if region["type"] == "hole_annulus")  # Retrieve the frozen hole-circle drawing primitive.
    root_xmax = beam_x + float(root_region["xmax_mm"]) * xscale  # Map the root-band right edge into the SVG side view.
    semantic_r = float(hole_region["outer_radius_mm"]) * zscale  # Map the drawn circular envelope radius into SVG pixels.
    svg = []  # Accumulate simple SVG markup without external plotting dependencies.
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')  # Start the standalone SVG canvas.
    svg.append('<rect width="100%" height="100%" fill="white"/>')  # Paint a clean white presentation background.
    svg.append('<text x="40" y="52" font-size="28" font-family="sans-serif" font-weight="700">LLM 直接画出的语义候选区</text>')  # Add a concise Chinese figure title.
    svg.append(f'<rect x="{beam_x}" y="{beam_y}" width="{beam_w}" height="{beam_h}" rx="8" fill="#d9e1e8" stroke="#5d6d7e" stroke-width="3"/>')  # Draw the perforated cantilever body in a neutral engineering tone.
    svg.append(f'<rect x="{beam_x - 22}" y="{beam_y - 15}" width="22" height="{beam_h + 30}" fill="#707b86"/>')  # Draw the clamped support as a dark wall beside the root face.
    svg.append(f'<circle cx="{hole_cx:.2f}" cy="{hole_cy:.2f}" r="{hole_r:.2f}" fill="white" stroke="#5d6d7e" stroke-width="3"/>')  # Cut the visible circular through-hole into the side-view drawing.
    svg.append(f'<rect x="{beam_x}" y="{beam_y}" width="{root_xmax - beam_x:.2f}" height="{beam_h}" rx="8" fill="#4f86d9" fill-opacity="0.22" stroke="#2f68b3" stroke-width="5"/>')  # Overlay the LLM-drawn root band as a translucent blue support.
    svg.append(f'<circle cx="{hole_cx:.2f}" cy="{hole_cy:.2f}" r="{semantic_r:.2f}" fill="#e74c3c" fill-opacity="0.13" stroke="#d83a2e" stroke-width="6" stroke-dasharray="14 8"/>')  # Overlay the hand-drawn circular semantic envelope around the hole.
    svg.append(f'<circle cx="{hole_cx:.2f}" cy="{hole_cy:.2f}" r="{hole_r:.2f}" fill="white" stroke="#5d6d7e" stroke-width="3"/>')  # Redraw the physical hole above the translucent semantic envelope for clarity.
    svg.append(f'<line x1="{beam_x + beam_w}" y1="{beam_y + 25}" x2="{beam_x + beam_w}" y2="{beam_y + 115}" stroke="#e67e22" stroke-width="7"/>')  # Draw the eccentric downward load shaft at the free end.
    svg.append(f'<polygon points="{beam_x + beam_w - 14},{beam_y + 108} {beam_x + beam_w + 14},{beam_y + 108} {beam_x + beam_w},{beam_y + 138}" fill="#e67e22"/>')  # Draw the load arrow head in the same orange annotation color.
    svg.append(f'<text x="{beam_x + 12}" y="{beam_y + beam_h + 42}" font-size="22" font-family="sans-serif" fill="#2f68b3">根部约束带</text>')  # Label the first continuous semantic region below the beam.
    svg.append(f'<text x="{hole_cx - 95:.2f}" y="{beam_y - 28}" font-size="22" font-family="sans-serif" fill="#c0392b">围绕孔直接画圆</text>')  # Label the circular semantic region in the same red used by its outline.
    svg.append(f'<text x="{beam_x + beam_w - 95}" y="{beam_y + beam_h + 42}" font-size="20" font-family="sans-serif" fill="#d35400">偏心载荷</text>')  # Label the free-end load so the region selection remains physically interpretable.
    svg.append('<text x="40" y="385" font-size="21" font-family="sans-serif" fill="#34495e">Dörfler 只在蓝色根部带与红色圆形包围区覆盖的元素中排序；其余算法完全不变。</text>')  # State the exact experimental intervention directly in the figure.
    svg.append('</svg>')  # Close the standalone SVG document.
    (RESULTS_DIR / "semantic_regions.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")  # Persist the actual geometric support as a small reviewable vector image.


def write_outputs(rows: list[dict], mark_records: list[dict], reference: dict, document: dict) -> None:  # Persist the adaptive histories and the precision-threshold comparison for the two methods.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the compact results directory exists before writing any artifact.
    history_fields = ["method", "round", "dof", "elements", "qoi", "qoi_rel_error", "candidate_elements", "marked_elements", "captured_fraction"]  # Define the stable adaptive-history CSV schema.
    with (RESULTS_DIR / "history.csv").open("w", encoding="utf-8", newline="") as handle:  # Open the main numerical trajectory table with portable newline handling.
        writer = csv.DictWriter(handle, fieldnames=history_fields)  # Create a deterministic writer using the documented field order.
        writer.writeheader()  # Emit the adaptive-history column names before the numerical rows.
        for row in rows:  # Emit all global and curve-region adaptive states in execution order.
            writer.writerow({field: row[field] for field in history_fields})  # Restrict every output row to the stable documented schema.
    (RESULTS_DIR / "marks.json").write_text(json.dumps(mark_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # Preserve exact marked element identifiers for transparent auditability.
    global_rows = [row for row in rows if row["method"] == "global_dorfler"]  # Extract the conventional full-domain Dörfler trajectory.
    semantic_rows = [row for row in rows if row["method"] == "curve_region_dorfler"]  # Extract the freeform semantic-region gated trajectory.
    thresholds = [0.10, 0.05, 0.03]  # Compare resource use at three intuitive QoI relative-error targets.
    threshold_rows = []  # Accumulate one resource-comparison record per requested target precision.
    for threshold in thresholds:  # Evaluate both adaptive histories against each target error level.
        global_hit = first_reaching(global_rows, threshold)  # Find the earliest conventional state satisfying the current target.
        semantic_hit = first_reaching(semantic_rows, threshold)  # Find the earliest semantic-gated state satisfying the current target.
        threshold_rows.append({"target_rel_error": threshold, "global_dof": "" if global_hit is None else global_hit["dof"], "semantic_dof": "" if semantic_hit is None else semantic_hit["dof"]})  # Preserve only the resource cost needed to reach the common target.
    with (RESULTS_DIR / "thresholds.csv").open("w", encoding="utf-8", newline="") as handle:  # Open the compact common-accuracy comparison table.
        writer = csv.DictWriter(handle, fieldnames=["target_rel_error", "global_dof", "semantic_dof"])  # Define the minimal threshold-comparison schema.
        writer.writeheader()  # Emit the three threshold-comparison column names.
        writer.writerows(threshold_rows)  # Write all requested common-accuracy resource comparisons.
    manifest = {"reference": reference, "theta": core.THETA, "amr_rounds": core.AMR_ROUNDS, "qoi_point_mm": core.QOI_POINT, "semantic_regions": document["selected_regions"], "semantic_description": document["semantic_description"]}  # Record every comparison-defining setting without atom vocabulary.
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # Persist the complete compact experiment manifest.
    markdown = []  # Accumulate one concise GitHub-readable experiment summary.
    markdown.append("# Freeform LLM semantic regions + Dörfler versus global Dörfler")  # State the experiment comparison in the report title.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append(f"Reference QoI: `{reference['qoi']:.12e}` mm from a global `{core.REFERENCE_H}` mm mesh with `{reference['dof']}` DOF proxy.")  # State the fixed fine-mesh reference used by both methods.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("Frozen LLM support: a full-cross-section root band from x=0 to 18 mm plus a circular envelope of radius 18 mm centred on the radius-8 mm through-hole at x=35 mm.")  # Describe the human-like continuous drawing instead of any fixed atom grid.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("Both methods use the same stress-jump indicator, theta=0.50, descending-prefix Dörfler marking, local remeshing operator, reference solution, QoI and number of AMR rounds. The only intervention is whether Dörfler sees the whole mesh or only elements inside the frozen drawn semantic regions.")  # State the isolated causal comparison precisely.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("| round | global DOF | global error | curve-region DOF | curve-region error |")  # Start the same-round trajectory table.
    markdown.append("| ---: | ---: | ---: | ---: | ---: |")  # Add the Markdown alignment row.
    for global_row, semantic_row in zip(global_rows, semantic_rows):  # Compare the two histories at the same adaptive-round index.
        markdown.append(f"| {global_row['round']} | {global_row['dof']} | {global_row['qoi_rel_error']:.6e} | {semantic_row['dof']} | {semantic_row['qoi_rel_error']:.6e} |")  # Add one compact same-round precision-resource record.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("## Same-accuracy resource comparison")  # Introduce the more meaningful common-error comparison section.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("| target relative error | global DOF | curve-region DOF |")  # Start the common-accuracy resource table.
    markdown.append("| ---: | ---: | ---: |")  # Add the common-accuracy Markdown alignment row.
    for item in threshold_rows:  # Traverse the three requested precision targets in ascending strictness.
        markdown.append(f"| < {100.0 * item['target_rel_error']:.0f}% | {item['global_dof'] or 'not reached'} | {item['semantic_dof'] or 'not reached'} |")  # Report the first resource level reaching each common target.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("Interpretation: success means the drawn semantic support shifts the QoI-error-versus-DOF trajectory left/down; failure or late saturation means the drawing omitted regions that later become important.")  # State the intended interpretation without assuming a positive result.
    (RESULTS_DIR / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")  # Persist the concise human-readable report.
    write_region_preview(document)  # Persist a vector picture of the exact root-band and circular-hole support used in the numerical run.


def main() -> int:  # Execute the complete freeform semantic-region experiment using the validated solver core.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Create the compact result directory before any possible early numerical failure.
    CASES_DIR.mkdir(parents=True, exist_ok=True)  # Create the transient numerical case directory before meshing begins.
    document = load_regions()  # Read the LLM-drawn continuous support frozen before any estimator or reference solution is observed.
    selected_regions = document["selected_regions"]  # Extract the geometric primitives passed to the semantic eligibility function.
    gmsh_bin = core.find_executable("gmsh", [])  # Resolve the Gmsh executable through the validated sibling utility.
    ccx_bin = core.find_executable("ccx", ["ccx_2.23", "ccx_2.22", "ccx_2.21", "ccx_2.20"])  # Resolve the installed CalculiX executable across common Ubuntu package names.
    print(f"[setup] gmsh={gmsh_bin} ccx={ccx_bin} theta={core.THETA} regions={[region['id'] for region in selected_regions]}", flush=True)  # Record the exact tools and frozen semantic drawing in the Actions log.
    reference_state = core.solve_mesh("reference_global", gmsh_bin, ccx_bin, refinement_boxes=None, global_h=core.REFERENCE_H)  # Solve one common globally fine reference before either adaptive history.
    reference = {"qoi": reference_state["qoi"], "dof": reference_state["dof"], "elements": reference_state["element_count"], "mesh_size": core.REFERENCE_H}  # Compact the reference to the information needed for error evaluation.
    global_rows, global_marks = core.run_method("global_dorfler", reference["qoi"], gmsh_bin, ccx_bin, selected_regions)  # Run conventional full-domain Dörfler from the shared coarse state.
    semantic_rows, semantic_marks = core.run_method("curve_region_dorfler", reference["qoi"], gmsh_bin, ccx_bin, selected_regions)  # Run the identical Dörfler algorithm inside only the hand-drawn semantic support.
    write_outputs(global_rows + semantic_rows, global_marks + semantic_marks, reference, document)  # Persist both trajectories, common-accuracy costs, exact marks, and the semantic-region picture.
    return 0  # Report successful benchmark completion after every compact output is written.


if __name__ == "__main__":  # Execute the experiment only when this file is invoked as the command-line program.
    try:  # Wrap only the outermost call so CI receives a concise failure message without hiding the traceback.
        raise SystemExit(main())  # Run the complete freeform semantic-region benchmark and return its explicit shell status.
    except Exception as exc:  # Catch numerical, parsing, geometry, and configuration failures at the command boundary.
        print(f"[fatal-curve-region] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Emit the exact failure class and message into the persisted workflow console log.
        raise  # Re-raise the original failure so GitHub Actions records the benchmark as failed.
