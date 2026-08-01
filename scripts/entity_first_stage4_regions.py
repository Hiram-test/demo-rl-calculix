"""Apply deterministic regional mesh fields to frozen BREP entities and verify density contrast."""  # Keep region-field validation separate from PSO.
from __future__ import annotations  # Enable modern annotations on Actions Python.
from dataclasses import asdict  # Serialize immutable topology evidence.
from hashlib import sha256  # Prove that regional meshing does not modify the BREP entity.
from math import ceil, hypot, pi, sqrt  # Compute boundary sampling and mesh-density metrics.
from pathlib import Path  # Resolve downloaded entity evidence and regional outputs safely.
from statistics import median  # Summarize local and far-field element sizes robustly.
from typing import Callable, Sequence  # Declare explicit zone-classification contracts.
import argparse  # Parse Stage 2 entity and output directories.
import html  # Escape paths and labels written to actual-mesh SVG evidence.
import json  # Write machine-readable regional-field receipts.
import shutil  # Copy frozen BREP files and locate Gmsh.
import subprocess  # Execute Gmsh in deterministic batch mode.
import sys  # Import verified Stage 1 and Stage 2 modules.
ROOT = Path(__file__).resolve().parents[1]  # Resolve the repository root.
SCRIPTS = ROOT / "scripts"  # Resolve the repository script directory.
if str(SCRIPTS) not in sys.path:  # Ensure direct execution imports repository modules.
    sys.path.insert(0, str(SCRIPTS))  # Put repository scripts before ambient packages.
from entity_first_stage1_audit import audit_mesh, triangle_area  # Reuse verified topology, geometry, and quality checks.
from entity_first_stage2_gmsh import expected_entity, format_number, parse_msh2  # Reuse exact model facts and MSH2 parsing.
def run_gmsh(source_path: Path) -> None:  # Execute one regional mesh source without a GUI.
    executable = shutil.which("gmsh")  # Resolve the installed mature Gmsh executable.
    if executable is None:  # Detect a missing dependency before producing partial evidence.
        raise RuntimeError("gmsh executable is unavailable")  # Stop the regional stage explicitly.
    command = (executable, source_path.as_posix(), "-2", "-nopopup")  # Run two-dimensional meshing in explicit batch mode.
    completed = subprocess.run(command, cwd=source_path.parent, text=True, capture_output=True, check=False)  # Execute the exact generated source.
    (source_path.parent / "gmsh.stdout.log").write_text(completed.stdout, encoding="utf-8")  # Preserve complete standard output.
    (source_path.parent / "gmsh.stderr.log").write_text(completed.stderr, encoding="utf-8")  # Preserve complete standard error.
    if completed.returncode != 0:  # Reject any geometry-import or meshing failure.
        source = source_path.read_text(encoding="utf-8")  # Read the exact generated program for diagnosis.
        raise RuntimeError(f"Gmsh failed: {command}\nsource:\n{source}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")  # Return reproducible failure evidence.
def physical_group_source(width: float, height: float, holes: Sequence[dict[str, float]]) -> list[str]:  # Generate fixed entity and boundary classifications.
    epsilon = max(width, height) * 1.0e-7  # Define a small OpenCASCADE selection tolerance.
    lines = ['SetFactory("OpenCASCADE");']  # Use OpenCASCADE for imported BREP topology.
    lines.append("Mesh.MshFileVersion = 2.2;")  # Emit deterministic text MSH2 output.
    lines.append("Mesh.SaveAll = 0;")  # Save only explicit physical entities.
    lines.append(f"domain[] = Surface In BoundingBox{{{-epsilon}, {-epsilon}, {-epsilon}, {format_number(width + epsilon)}, {format_number(height + epsilon)}, {epsilon}}};")  # Select the material surface.
    lines.append('Physical Surface("DOMAIN", 1) = {domain[]};')  # Preserve the material-domain group.
    lines.append(f"left[] = Curve In BoundingBox{{{-epsilon}, {-epsilon}, {-epsilon}, {epsilon}, {format_number(height + epsilon)}, {epsilon}}};")  # Select the exact left edge.
    lines.append(f"right[] = Curve In BoundingBox{{{format_number(width - epsilon)}, {-epsilon}, {-epsilon}, {format_number(width + epsilon)}, {format_number(height + epsilon)}, {epsilon}}};")  # Select the exact right edge.
    lines.append(f"bottom[] = Curve In BoundingBox{{{-epsilon}, {-epsilon}, {-epsilon}, {format_number(width + epsilon)}, {epsilon}, {epsilon}}};")  # Select the exact bottom edge.
    lines.append(f"top[] = Curve In BoundingBox{{{-epsilon}, {format_number(height - epsilon)}, {-epsilon}, {format_number(width + epsilon)}, {format_number(height + epsilon)}, {epsilon}}};")  # Select the exact top edge.
    lines.append('Physical Curve("FIXED_EDGE", 101) = {left[]};')  # Preserve the support boundary name.
    lines.append('Physical Curve("LOAD_EDGE", 102) = {right[]};')  # Preserve the load boundary name.
    lines.append('Physical Curve("BOTTOM_EDGE", 103) = {bottom[]};')  # Preserve the bottom boundary name.
    lines.append('Physical Curve("TOP_EDGE", 104) = {top[]};')  # Preserve the top boundary name.
    for index, hole in enumerate(holes):  # Classify every exact circular hole independently.
        center_x = float(hole["center_x_mm"])  # Read the exact hole center x-coordinate.
        center_y = float(hole["center_y_mm"])  # Read the exact hole center y-coordinate.
        radius = float(hole["radius_mm"])  # Read the exact hole radius.
        lines.append(f"hole_{index}[] = Curve In BoundingBox{{{format_number(center_x - radius - epsilon)}, {format_number(center_y - radius - epsilon)}, {-epsilon}, {format_number(center_x + radius + epsilon)}, {format_number(center_y + radius + epsilon)}, {epsilon}}};")  # Select this circular boundary only.
        lines.append(f'Physical Curve("HOLE_{index}", {200 + index}) = {{hole_{index}[]}};')  # Preserve one named hole boundary.
    return lines  # Return fixed topology and Physical Group source.
def threshold_field(lines: list[str], *, field_id: int, curves_expression: str, size_min: float, size_max: float, distance_min: float, distance_max: float) -> int:  # Append one distance-threshold size field.
    distance_id = field_id  # Allocate the Distance field identifier.
    threshold_id = field_id + 1  # Allocate the matching Threshold field identifier.
    lines.append(f"Field[{distance_id}] = Distance;")  # Measure distance from selected entity curves.
    lines.append(f"Field[{distance_id}].CurvesList = {{{curves_expression}}};")  # Bind the exact Physical Curve selection.
    lines.append(f"Field[{distance_id}].Sampling = 120;")  # Sample curved geometry densely for stable distance evaluation.
    lines.append(f"Field[{threshold_id}] = Threshold;")  # Convert distance to bounded local mesh size.
    lines.append(f"Field[{threshold_id}].InField = {distance_id};")  # Use the matching distance field.
    lines.append(f"Field[{threshold_id}].SizeMin = {format_number(size_min)};")  # Set the local refinement size.
    lines.append(f"Field[{threshold_id}].SizeMax = {format_number(size_max)};")  # Recover the far-field background size.
    lines.append(f"Field[{threshold_id}].DistMin = {format_number(distance_min)};")  # Hold local size through the core zone.
    lines.append(f"Field[{threshold_id}].DistMax = {format_number(distance_max)};")  # Grade to background size through the transition zone.
    return threshold_id  # Return the field used in the final minimum composition.
def regional_geo(case: dict[str, object], brep_path: Path, msh_path: Path) -> str:  # Generate one regional mesh program on the frozen entity.
    width = float(case["width_mm"])  # Read the fixed entity width.
    height = float(case["height_mm"])  # Read the fixed entity height.
    holes = tuple(case.get("holes", ()))  # Read exact circular-hole facts.
    background = float(case["background_size_mm"])  # Read the far-field size.
    lines = physical_group_source(width, height, holes)  # Generate stable topology and boundary groups.
    lines.insert(1, f'Merge "{brep_path.as_posix()}";')  # Import the one frozen BREP before topology queries.
    threshold_ids: list[int] = []  # Collect local fields for the final minimum field.
    next_field = 1  # Allocate deterministic Gmsh field IDs.
    if case["case_id"] == "bearing_plate":  # Build two physically interpretable load-path end zones.
        fixed_threshold = threshold_field(lines, field_id=next_field, curves_expression="left[]", size_min=6.0, size_max=background, distance_min=15.0, distance_max=220.0)  # Refine the complete fixed edge and grade into the plate.
        threshold_ids.append(fixed_threshold)  # Preserve the fixed-edge field.
        next_field += 2  # Advance field IDs.
        load_threshold = threshold_field(lines, field_id=next_field, curves_expression="right[]", size_min=6.0, size_max=background, distance_min=15.0, distance_max=160.0)  # Refine the finite load edge and its transfer zone.
        threshold_ids.append(load_threshold)  # Preserve the load-edge field.
        next_field += 2  # Advance field IDs.
    else:  # Build independent exact-hole fields for one-hole and three-hole entities.
        local_sizes = tuple(float(value) for value in case["hole_local_sizes_mm"])  # Read one independent target size per hole.
        for index, (hole, local_size) in enumerate(zip(holes, local_sizes)):  # Configure each hole independently.
            radius = float(hole["radius_mm"])  # Read the exact radius for grading and boundary sampling.
            threshold_id = threshold_field(lines, field_id=next_field, curves_expression=f"hole_{index}[]", size_min=local_size, size_max=background, distance_min=0.35 * radius, distance_max=2.5 * radius)  # Refine the circular process zone and grade outward.
            threshold_ids.append(threshold_id)  # Preserve this independent hole field.
            next_field += 2  # Advance field IDs.
            segment_count = max(32, int(ceil(2.0 * pi * radius / local_size)))  # Derive circular boundary resolution from exact radius and requested size.
            lines.append(f"Transfinite Curve {{hole_{index}[]}} = {segment_count + 1} Using Progression 1;")  # Guarantee geometry sampling without fixing all holes to one segment count.
    minimum_id = next_field  # Allocate the final composed field ID.
    lines.append(f"Field[{minimum_id}] = Min;")  # Select the finest active local requirement.
    lines.append(f"Field[{minimum_id}].FieldsList = {{{', '.join(str(value) for value in threshold_ids)}}};")  # Compose all independent regions without changing geometry.
    lines.append(f"Background Field = {minimum_id};")  # Apply the regional size field to the fixed BREP.
    minimum_size = min([6.0] if case["case_id"] == "bearing_plate" else [float(value) for value in case["hole_local_sizes_mm"]])  # Determine the explicit global size lower bound.
    lines.append(f"Mesh.MeshSizeMin = {format_number(minimum_size)};")  # Prevent hidden sizes below the regional contract.
    lines.append(f"Mesh.MeshSizeMax = {format_number(background)};")  # Prevent hidden sizes above the background contract.
    lines.append("Mesh.MeshSizeFromPoints = 0;")  # Prevent CAD points from overriding the regional field.
    lines.append("Mesh.MeshSizeFromCurvature = 0;")  # Keep curvature handling separate from optimization variables.
    lines.append("Mesh.MeshSizeExtendFromBoundary = 0;")  # Prevent uncontrolled boundary-size propagation.
    lines.append("Mesh.Algorithm = 6;")  # Use Frontal-Delaunay on the exact surface.
    lines.append("Mesh 2;")  # Generate the two-dimensional finite-element mesh.
    lines.append(f'Save "{msh_path.as_posix()}";')  # Save the validated MSH2 output.
    return "\n".join(lines) + "\n"  # Return deterministic Gmsh source.
def characteristic_size(points: Sequence[tuple[float, float]], triangle: tuple[int, int, int]) -> tuple[float, float, float]:  # Compute centroid and equivalent edge size for one triangle.
    p0, p1, p2 = (points[index] for index in triangle)  # Resolve triangle coordinates.
    area = triangle_area(p0, p1, p2)  # Compute physical element area.
    size = sqrt(4.0 * area / sqrt(3.0))  # Convert area to the edge length of an equal-area equilateral triangle.
    centroid_x = (p0[0] + p1[0] + p2[0]) / 3.0  # Compute centroid x-coordinate.
    centroid_y = (p0[1] + p1[1] + p2[1]) / 3.0  # Compute centroid y-coordinate.
    return centroid_x, centroid_y, size  # Return robust spatial size evidence.
def zone_median(records: Sequence[tuple[float, float, float]], predicate: Callable[[float, float], bool], name: str) -> float:  # Compute one zone's robust element-size statistic.
    values = [size for x_value, y_value, size in records if predicate(x_value, y_value)]  # Select elements whose centroids lie in the named zone.
    if len(values) < 5:  # Require enough elements to support a regional conclusion.
        raise RuntimeError(f"zone {name} contains only {len(values)} elements")  # Reject an under-resolved or misclassified region.
    return float(median(values))  # Return the robust median equivalent size.
def density_receipt(case: dict[str, object], points: Sequence[tuple[float, float]], triangles: Sequence[tuple[int, int, int]]) -> dict[str, object]:  # Quantify actual regional refinement contrast.
    records = tuple(characteristic_size(points, triangle) for triangle in triangles)  # Compute spatial size evidence once.
    width = float(case["width_mm"])  # Read the entity width for zone definitions.
    height = float(case["height_mm"])  # Read the entity height for zone definitions.
    if case["case_id"] == "bearing_plate":  # Evaluate complete fixed/load zones against the midspan far field.
        fixed = zone_median(records, lambda x_value, y_value: x_value < 140.0, "fixed_end")  # Measure the full-height fixed-end zone.
        load = zone_median(records, lambda x_value, y_value: x_value > width - 120.0, "load_end")  # Measure the finite load-transfer zone.
        far = zone_median(records, lambda x_value, y_value: 0.42 * width < x_value < 0.58 * width, "midspan_far")  # Measure the deliberately coarse midspan.
        if fixed >= 0.65 * far or load >= 0.65 * far:  # Require clear visible and numerical refinement contrast.
            raise RuntimeError(f"bearing regional contrast is insufficient: fixed={fixed}, load={load}, far={far}")  # Reject weak or misleading allocation.
        return {"fixed_end_median_size": fixed, "load_end_median_size": load, "far_field_median_size": far, "fixed_to_far_ratio": fixed / far, "load_to_far_ratio": load / far}  # Preserve complete bearing density evidence.
    holes = tuple(case["holes"])  # Read exact hole geometry for ring classification.
    ring_medians: list[float] = []  # Collect one actual median size per hole.
    for index, hole in enumerate(holes):  # Measure each independent circular refinement zone.
        center_x = float(hole["center_x_mm"])  # Read the exact center x-coordinate.
        center_y = float(hole["center_y_mm"])  # Read the exact center y-coordinate.
        radius = float(hole["radius_mm"])  # Read the exact radius.
        ring_width = max(16.0, 0.55 * radius)  # Define a physical annular measurement band.
        ring = zone_median(records, lambda x_value, y_value, cx=center_x, cy=center_y, r=radius, rw=ring_width: abs(hypot(x_value - cx, y_value - cy) - r) < rw, f"hole_{index}_ring")  # Measure this actual annular mesh density.
        ring_medians.append(ring)  # Preserve the independent hole result.
    far = zone_median(records, lambda x_value, y_value: all(hypot(x_value - float(hole["center_x_mm"]), y_value - float(hole["center_y_mm"])) > 3.0 * float(hole["radius_mm"]) for hole in holes) and 0.12 * width < x_value < 0.88 * width and 0.12 * height < y_value < 0.88 * height, "far_field")  # Measure interior far-field density away from all holes and outer boundaries.
    if any(value >= 0.65 * far for value in ring_medians):  # Require every hole region to be clearly finer than the far field.
        raise RuntimeError(f"hole regional contrast is insufficient: rings={ring_medians}, far={far}")  # Reject low-distinction meshes.
    if case["case_id"] == "three_openings" and not (ring_medians[1] < ring_medians[0] < ring_medians[2]):  # Require the deliberately distinct middle-left-right hierarchy.
        raise RuntimeError(f"three-hole independent sizing order failed: {ring_medians}")  # Reject a field implementation that collapses three controls into one.
    return {"hole_ring_median_sizes": ring_medians, "far_field_median_size": far, "hole_to_far_ratios": [value / far for value in ring_medians]}  # Preserve complete hole-density evidence.
def write_svg(path: Path, points: Sequence[tuple[float, float]], triangles: Sequence[tuple[int, int, int]], width_mm: float, height_mm: float, title: str) -> None:  # Render actual triangle connectivity as a vector mesh figure.
    canvas_width = 1400.0 if width_mm / height_mm > 3.0 else 1000.0  # Give long bearing geometry enough horizontal resolution.
    canvas_height = max(260.0, canvas_width * height_mm / width_mm)  # Preserve the true model aspect ratio.
    padding = 30.0  # Reserve a small border around the actual mesh.
    scale = min((canvas_width - 2.0 * padding) / width_mm, (canvas_height - 2.0 * padding) / height_mm)  # Fit the entity without distortion.
    def transform(point: tuple[float, float]) -> tuple[float, float]:  # Map model coordinates to SVG coordinates.
        x_value = padding + point[0] * scale  # Map horizontal model coordinate.
        y_value = canvas_height - padding - point[1] * scale  # Invert and map vertical model coordinate.
        return x_value, y_value  # Return one canvas point.
    path_fragments: list[str] = []  # Collect actual triangle paths efficiently.
    for triangle in triangles:  # Draw every finite element from its true connectivity.
        transformed = [transform(points[index]) for index in triangle]  # Transform the three actual nodes.
        commands = " ".join(f"{x_value:.3f},{y_value:.3f}" for x_value, y_value in transformed)  # Render the polygon point list.
        path_fragments.append(f'<polygon points="{commands}" fill="none" stroke="#5f6368" stroke-width="0.45"/>')  # Draw one unfilled actual triangle.
    document = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width:.0f}" height="{canvas_height:.0f}" viewBox="0 0 {canvas_width:.3f} {canvas_height:.3f}">']  # Begin the SVG document.
    document.append('<rect width="100%" height="100%" fill="white"/>')  # Add a clean white background.
    document.append(f'<title>{html.escape(title)}</title>')  # Preserve a machine-readable case title.
    document.extend(path_fragments)  # Add every actual mesh element.
    document.append('</svg>')  # Close the SVG document.
    path.write_text("\n".join(document) + "\n", encoding="utf-8")  # Persist vector evidence without raster interpolation.
CASES = (  # Freeze deterministic regional-field smoke specifications.
    {"case_id": "bearing_plate", "width_mm": 1000.0, "height_mm": 100.0, "holes": (), "background_size_mm": 45.0},  # Define full-edge bearing refinement on one rectangle.
    {"case_id": "circular_opening", "width_mm": 240.0, "height_mm": 240.0, "holes": ({"center_x_mm": 120.0, "center_y_mm": 120.0, "radius_mm": 20.0},), "background_size_mm": 25.0, "hole_local_sizes_mm": (3.0,)},  # Define strong single-hole refinement.
    {"case_id": "three_openings", "width_mm": 600.0, "height_mm": 260.0, "holes": ({"center_x_mm": 130.0, "center_y_mm": 130.0, "radius_mm": 24.0}, {"center_x_mm": 300.0, "center_y_mm": 130.0, "radius_mm": 42.0}, {"center_x_mm": 450.0, "center_y_mm": 130.0, "radius_mm": 30.0}), "background_size_mm": 30.0, "hole_local_sizes_mm": (5.0, 2.5, 8.0)},  # Define visibly independent three-hole sizes.
)  # Complete the fixed regional-field suite.
def run_case(case: dict[str, object], stage2_root: Path, output_root: Path) -> dict[str, object]:  # Mesh and validate one frozen entity.
    case_id = str(case["case_id"])  # Resolve the stable case identifier.
    source_brep = stage2_root / case_id / "model.brep"  # Locate the Stage 2 frozen entity.
    if not source_brep.exists():  # Require exact prior-stage evidence.
        raise FileNotFoundError(source_brep)  # Refuse to regenerate geometry silently.
    case_root = output_root / case_id  # Allocate an isolated regional output directory.
    case_root.mkdir(parents=True, exist_ok=True)  # Create the case evidence directory.
    target_brep = case_root / "model.brep"  # Allocate a self-contained copy of the frozen entity.
    shutil.copyfile(source_brep, target_brep)  # Copy bytes without rebuilding or modifying geometry.
    source_digest = sha256(source_brep.read_bytes()).hexdigest()  # Fingerprint the Stage 2 entity.
    target_digest = sha256(target_brep.read_bytes()).hexdigest()  # Fingerprint the regional-stage entity copy.
    if source_digest != target_digest:  # Enforce byte-identical entity isolation.
        raise RuntimeError(f"BREP changed before regional meshing for {case_id}")  # Reject geometry mutation.
    msh_path = case_root / "coarse.msh"  # Use the common name consumed by the existing CalculiX stage.
    geo_path = case_root / "regional.geo"  # Allocate the regional Gmsh source.
    geo_path.write_text(regional_geo(case, target_brep, msh_path), encoding="utf-8")  # Persist the exact regional field program.
    run_gmsh(geo_path)  # Mesh the frozen BREP with regional fields.
    parsed = parse_msh2(msh_path)  # Parse actual node and triangle evidence.
    expected_components, expected_area, expected_holes = expected_entity(case)  # Compute exact entity facts.
    audit = audit_mesh(parsed.points, parsed.triangles, expected_components=expected_components, expected_area=expected_area, expected_hole_areas=expected_holes, area_tolerance=0.01, hole_tolerance=0.02, angle_limit=8.0, radius_ratio_limit=12.0)  # Apply hard topology, geometry, and quality gates.
    if not audit.ok:  # Reject a regional field that damages entity or mesh quality.
        raise RuntimeError(f"regional mesh audit failed for {case_id}: {audit.issues}")  # Preserve complete failure evidence.
    density = density_receipt(case, parsed.points, parsed.triangles)  # Quantify actual regional distinction.
    svg_path = case_root / "regional_mesh.svg"  # Allocate actual vector mesh evidence.
    write_svg(svg_path, parsed.points, parsed.triangles, float(case["width_mm"]), float(case["height_mm"]), f"{case_id} regional mesh")  # Draw true connectivity, not a generated illustration.
    receipt = {"case_id": case_id, "brep_sha256": source_digest, "entity_invariant": source_digest == target_digest, "nodes": len(parsed.points), "triangles": len(parsed.triangles), "audit": asdict(audit), "density": density, "mesh_file": msh_path.name, "svg_file": svg_path.name}  # Build complete regional evidence.
    (case_root / "manifest.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")  # Persist case-level evidence.
    return receipt  # Return the validated regional result.
def main() -> int:  # Execute the deterministic regional-field suite.
    parser = argparse.ArgumentParser(description="Validate regional Gmsh fields on frozen OpenCASCADE entities")  # Describe the Stage 4 gate.
    parser.add_argument("--stage2-root", required=True)  # Require the prior exact-entity artifact.
    parser.add_argument("--output-dir", required=True)  # Require an isolated regional evidence directory.
    args = parser.parse_args()  # Parse command-line arguments once.
    stage2_root = Path(args.stage2_root).resolve()  # Normalize the exact-entity input root.
    output_root = Path(args.output_dir).resolve()  # Normalize the regional output root.
    receipts = {str(case["case_id"]): run_case(case, stage2_root, output_root) for case in CASES}  # Build all regional meshes independently.
    suite = {"schema_version": "entity-first-stage4-regions/1.0", "cases": receipts, "all_valid": True}  # Build the complete Stage 4 receipt.
    (output_root / "stage4_receipt.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")  # Persist machine-readable suite evidence.
    print(json.dumps(suite, ensure_ascii=False, indent=2))  # Echo evidence to the Actions log.
    return 0  # Return success only after all topology, geometry, quality, and density gates pass.
if __name__ == "__main__":  # Run only when invoked directly.
    raise SystemExit(main())  # Propagate the process status to CI.