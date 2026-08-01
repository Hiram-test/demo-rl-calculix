"""Build exact OpenCASCADE entities from IMP JSON and mesh one frozen BREP."""  # Keep geometry creation separate from mesh-size experiments.
from __future__ import annotations  # Enable modern annotations on Actions Python.
from dataclasses import dataclass  # Store parsed Gmsh mesh evidence.
from hashlib import sha256  # Fingerprint the frozen BREP entity.
from math import pi  # Compute exact circular-hole areas.
from pathlib import Path  # Resolve generated model artifacts safely.
from typing import Sequence  # Declare explicit geometry and mesh contracts.
import argparse  # Parse IMP and output arguments.
import json  # Read system-generated IMP and write model manifests.
import shutil  # Locate the Gmsh executable.
import subprocess  # Execute Gmsh as the deterministic geometry/mesh authority.
import sys  # Import the Stage 1 hard gate and return process status.
ROOT = Path(__file__).resolve().parents[1]  # Resolve the repository root from the script location.
SCRIPTS = ROOT / "scripts"  # Resolve the directory containing the Stage 1 audit module.
if str(SCRIPTS) not in sys.path:  # Ensure direct execution can import the Stage 1 hard gate.
    sys.path.insert(0, str(SCRIPTS))  # Put repository scripts before ambient packages.
from entity_first_stage1_audit import audit_mesh  # Reuse the already verified topology and geometry checks.
@dataclass(frozen=True)  # Preserve parsed mesh evidence without later mutation.
class ParsedMesh:  # Store nodes, triangles, line groups, and physical names.
    points: tuple[tuple[float, float], ...]  # Store planar node coordinates in zero-based order.
    triangles: tuple[tuple[int, int, int], ...]  # Store zero-based triangular connectivity.
    physical_names: dict[tuple[int, int], str]  # Map physical dimension and tag to a stable name.
    line_group_counts: dict[str, int]  # Count boundary line elements in each named Physical Curve.
def format_number(value: float) -> str:  # Render geometry values deterministically in generated Gmsh source.
    return f"{float(value):.12g}"  # Avoid locale and unnecessary floating-point noise.
def validate_imp(spec: dict[str, object]) -> None:  # Validate the system-generated model input before invoking Gmsh.
    if spec.get("schema_version") != "entity-first-imp/1.0":  # Require an explicit versioned contract.
        raise ValueError("unsupported IMP schema_version")  # Reject ambiguous input semantics.
    if spec.get("kind") != "plate_2d":  # Restrict Stage 2 to exact two-dimensional plate entities.
        raise ValueError("Stage 2 supports only kind=plate_2d")  # Reject unsupported model dimensions.
    width = float(spec["width_mm"])  # Read the plate width in model units.
    height = float(spec["height_mm"])  # Read the plate height in model units.
    if width <= 0.0 or height <= 0.0:  # Require a positive geometric extent.
        raise ValueError("plate dimensions must be positive")  # Reject a non-physical plate.
    holes = list(spec.get("holes", []))  # Normalize the optional hole list.
    for index, hole in enumerate(holes):  # Validate every circular subtraction independently.
        if not isinstance(hole, dict):  # Require structured hole facts.
            raise ValueError(f"hole {index} must be an object")  # Reject scalar or free-text hole definitions.
        radius = float(hole["radius_mm"])  # Read the exact circular radius.
        center_x = float(hole["center_x_mm"])  # Read the exact horizontal center.
        center_y = float(hole["center_y_mm"])  # Read the exact vertical center.
        if radius <= 0.0:  # Require a positive circular radius.
            raise ValueError(f"hole {index} radius must be positive")  # Reject a degenerate disk.
        if center_x - radius <= 0.0 or center_x + radius >= width or center_y - radius <= 0.0 or center_y + radius >= height:  # Require each hole to remain strictly inside the plate.
            raise ValueError(f"hole {index} must remain inside the plate")  # Reject a topology-changing boundary intersection.
def geometry_geo(spec: dict[str, object], brep_path: Path) -> str:  # Generate geometry-only OpenCASCADE source.
    width = float(spec["width_mm"])  # Read the fixed plate width.
    height = float(spec["height_mm"])  # Read the fixed plate height.
    holes = list(spec.get("holes", []))  # Read exact circular-hole facts.
    lines = ['SetFactory("OpenCASCADE");']  # Select the exact OpenCASCADE geometry kernel.
    lines.append(f"Rectangle(1) = {{0, 0, 0, {format_number(width)}, {format_number(height)}}};")  # Create the plate entity.
    tool_tags: list[int] = []  # Collect disk surface tags for one Boolean subtraction.
    for index, hole in enumerate(holes):  # Create every exact hole entity.
        tag = 100 + index  # Allocate deterministic disk tags away from the plate tag.
        tool_tags.append(tag)  # Preserve the disk tag for BooleanDifference.
        lines.append(f"Disk({tag}) = {{{format_number(float(hole['center_x_mm']))}, {format_number(float(hole['center_y_mm']))}, 0, {format_number(float(hole['radius_mm']))}, {format_number(float(hole['radius_mm']))}}};")  # Create one exact circular disk.
    if tool_tags:  # Subtract holes only when the IMP contains them.
        tools = ", ".join(str(tag) for tag in tool_tags)  # Render deterministic disk surface tags.
        lines.append(f"domain[] = BooleanDifference{{ Surface{{1}}; Delete; }}{{ Surface{{{tools}}}; Delete; }};")  # Produce one plate surface with exact holes.
    else:  # Preserve the rectangle as the domain when no holes exist.
        lines.append("domain[] = {1};")  # Bind the plate surface to the common domain array.
    lines.append("Coherence;")  # Remove duplicate OpenCASCADE entities before export.
    lines.append(f'Save "{brep_path.as_posix()}";')  # Export the reusable geometry entity before any mesh settings exist.
    return "\n".join(lines) + "\n"  # Return deterministic Gmsh source text.
def mesh_geo(spec: dict[str, object], brep_path: Path, msh_path: Path, background_size: float, local_size: float) -> str:  # Generate mesh-only source importing the frozen entity.
    width = float(spec["width_mm"])  # Read the fixed plate width.
    height = float(spec["height_mm"])  # Read the fixed plate height.
    holes = list(spec.get("holes", []))  # Read exact circular-hole facts.
    epsilon = max(width, height) * 1.0e-7  # Define a small bounding-box classification tolerance.
    lines = ['SetFactory("OpenCASCADE");']  # Use the same geometry kernel for BREP import.
    lines.append(f'Merge "{brep_path.as_posix()}";')  # Import the one frozen entity used by every mesh-size experiment.
    lines.append("Mesh.MshFileVersion = 2.2;")  # Emit the simple deterministic MSH2 format for direct validation.
    lines.append("Mesh.SaveAll = 0;")  # Save only elements belonging to explicit Physical Groups.
    lines.append(f"domain[] = Surface In BoundingBox{{{-epsilon}, {-epsilon}, {-epsilon}, {format_number(width + epsilon)}, {format_number(height + epsilon)}, {epsilon}}};")  # Select the imported plate surface.
    lines.append('Physical Surface("DOMAIN", 1) = {domain[]};')  # Create the material-domain group required by later CalculiX export.
    lines.append(f"left[] = Curve In BoundingBox{{{-epsilon}, {-epsilon}, {-epsilon}, {epsilon}, {format_number(height + epsilon)}, {epsilon}}};")  # Select the exact left outer edge.
    lines.append(f"right[] = Curve In BoundingBox{{{format_number(width - epsilon)}, {-epsilon}, {-epsilon}, {format_number(width + epsilon)}, {format_number(height + epsilon)}, {epsilon}}};")  # Select the exact right outer edge.
    lines.append(f"bottom[] = Curve In BoundingBox{{{-epsilon}, {-epsilon}, {-epsilon}, {format_number(width + epsilon)}, {epsilon}, {epsilon}}};")  # Select the exact bottom outer edge.
    lines.append(f"top[] = Curve In BoundingBox{{{-epsilon}, {format_number(height - epsilon)}, {-epsilon}, {format_number(width + epsilon)}, {format_number(height + epsilon)}, {epsilon}}};")  # Select the exact top outer edge.
    lines.append('Physical Curve("FIXED_EDGE", 101) = {left[]};')  # Name the left edge for boundary-condition generation.
    lines.append('Physical Curve("LOAD_EDGE", 102) = {right[]};')  # Name the right edge for load generation.
    lines.append('Physical Curve("BOTTOM_EDGE", 103) = {bottom[]};')  # Preserve the bottom model boundary.
    lines.append('Physical Curve("TOP_EDGE", 104) = {top[]};')  # Preserve the top model boundary.
    field_ids: list[int] = []  # Collect local threshold fields for a final minimum field.
    next_field = 1  # Allocate deterministic field identifiers.
    for index, hole in enumerate(holes):  # Classify and refine every exact circular boundary.
        center_x = float(hole["center_x_mm"])  # Read the exact hole center x-coordinate.
        center_y = float(hole["center_y_mm"])  # Read the exact hole center y-coordinate.
        radius = float(hole["radius_mm"])  # Read the exact hole radius.
        array_name = f"hole_{index}"  # Allocate a deterministic Gmsh array name.
        lines.append(f"{array_name}[] = Curve In BoundingBox{{{format_number(center_x - radius - epsilon)}, {format_number(center_y - radius - epsilon)}, {-epsilon}, {format_number(center_x + radius + epsilon)}, {format_number(center_y + radius + epsilon)}, {epsilon}}};")  # Select exact circular boundary curves.
        lines.append(f'Physical Curve("HOLE_{index}", {200 + index}) = {{{array_name}[]}};')  # Name each hole boundary independently.
        distance_id = next_field  # Allocate a Distance field for this hole.
        threshold_id = next_field + 1  # Allocate the matching Threshold field.
        next_field += 2  # Advance field identifiers for the next hole.
        field_ids.append(threshold_id)  # Preserve the threshold field for the final minimum.
        lines.append(f"Field[{distance_id}] = Distance;")  # Measure distance from the exact hole curves.
        lines.append(f"Field[{distance_id}].CurvesList = {{{array_name}[]}};")  # Bind distance evaluation to this hole boundary.
        lines.append(f"Field[{distance_id}].Sampling = 100;")  # Sample curved geometry densely for a stable distance field.
        lines.append(f"Field[{threshold_id}] = Threshold;")  # Convert distance to a bounded mesh size.
        lines.append(f"Field[{threshold_id}].InField = {distance_id};")  # Use the matching hole distance.
        lines.append(f"Field[{threshold_id}].SizeMin = {format_number(local_size)};")  # Set the local hole-boundary size.
        lines.append(f"Field[{threshold_id}].SizeMax = {format_number(background_size)};")  # Recover the background size away from the hole.
        lines.append(f"Field[{threshold_id}].DistMin = {format_number(radius)};")  # Keep local size through one radius from the hole.
        lines.append(f"Field[{threshold_id}].DistMax = {format_number(3.0 * radius)};")  # Grade smoothly to background size by three radii.
        lines.append(f"Transfinite Curve {{{array_name}[]}} = 64 Using Progression 1;")  # Guarantee enough boundary segments to preserve circular geometry.
    if field_ids:  # Combine all hole refinement fields when holes exist.
        minimum_id = next_field  # Allocate the final minimum field identifier.
        fields = ", ".join(str(field_id) for field_id in field_ids)  # Render deterministic threshold field identifiers.
        lines.append(f"Field[{minimum_id}] = Min;")  # Select the finest applicable local field.
        lines.append(f"Field[{minimum_id}].FieldsList = {{{fields}}};")  # Combine all independent hole fields.
        lines.append(f"Background Field = {minimum_id};")  # Apply local refinement without changing geometry.
    lines.append(f"Mesh.MeshSizeMin = {format_number(min(local_size, background_size))};")  # Bound the minimum mesh size.
    lines.append(f"Mesh.MeshSizeMax = {format_number(max(local_size, background_size))};")  # Bound the maximum mesh size.
    lines.append("Mesh.MeshSizeFromPoints = 0;")  # Prevent imported CAD points from overriding the explicit field.
    lines.append("Mesh.MeshSizeFromCurvature = 0;")  # Keep curvature sampling separate from mesh optimization.
    lines.append("Mesh.MeshSizeExtendFromBoundary = 0;")  # Avoid hidden boundary-size propagation beyond the explicit field.
    lines.append("Mesh.Algorithm = 6;")  # Use the Frontal-Delaunay surface algorithm on the fixed entity.
    lines.append("Mesh 2;")  # Generate the two-dimensional finite-element mesh.
    lines.append(f'Save "{msh_path.as_posix()}";')  # Export the validated MSH2 evidence.
    return "\n".join(lines) + "\n"  # Return deterministic mesh source text.
def run_gmsh(source_path: Path) -> None:  # Execute one generated Gmsh program with strict error handling.
    executable = shutil.which("gmsh")  # Resolve the runner-provided Gmsh executable.
    if executable is None:  # Detect a missing mature meshing dependency explicitly.
        raise RuntimeError("gmsh executable is unavailable")  # Stop before producing partial evidence.
    completed = subprocess.run((executable, source_path.as_posix(), "-nopopup"), cwd=source_path.parent, text=True, capture_output=True, check=False)  # Execute Gmsh without a GUI.
    if completed.returncode != 0:  # Detect any geometry or meshing failure.
        raise RuntimeError(f"gmsh failed for {source_path.name}:\n{completed.stdout}\n{completed.stderr}")  # Preserve complete tool output.
def parse_msh2(path: Path) -> ParsedMesh:  # Parse nodes, triangles, and Physical Curve evidence from MSH2.
    lines = path.read_text(encoding="utf-8").splitlines()  # Read the deterministic text mesh.
    physical_names: dict[tuple[int, int], str] = {}  # Collect dimension/tag to name mappings.
    node_coordinates: dict[int, tuple[float, float]] = {}  # Collect Gmsh node IDs and planar coordinates.
    triangle_rows: list[tuple[int, int, int]] = []  # Collect triangle node IDs before zero-based remapping.
    line_rows: list[tuple[int, int, int]] = []  # Collect physical tag and line node IDs.
    index = 0  # Initialize the section parser cursor.
    while index < len(lines):  # Visit every MSH2 section.
        token = lines[index].strip()  # Read the current section token.
        if token == "$PhysicalNames":  # Parse named physical groups.
            count = int(lines[index + 1])  # Read the number of physical names.
            for offset in range(count):  # Parse every physical name row.
                dimension_text, tag_text, quoted_name = lines[index + 2 + offset].split(maxsplit=2)  # Split dimension, tag, and quoted name.
                physical_names[(int(dimension_text), int(tag_text))] = quoted_name.strip('"')  # Preserve the stable group name.
            index += count + 3  # Skip the section and its end marker.
            continue  # Continue with the next section.
        if token == "$Nodes":  # Parse all mesh nodes.
            count = int(lines[index + 1])  # Read the number of node rows.
            for offset in range(count):  # Parse every node row.
                values = lines[index + 2 + offset].split()  # Split the MSH2 node record.
                node_coordinates[int(values[0])] = (float(values[1]), float(values[2]))  # Preserve planar coordinates by Gmsh ID.
            index += count + 3  # Skip the section and its end marker.
            continue  # Continue with the next section.
        if token == "$Elements":  # Parse line and triangle elements.
            count = int(lines[index + 1])  # Read the number of element rows.
            for offset in range(count):  # Parse every element row.
                values = [int(value) for value in lines[index + 2 + offset].split()]  # Convert the complete record to integers.
                element_type = values[1]  # Read the Gmsh element type.
                number_of_tags = values[2]  # Read the number of element tags.
                tags = values[3 : 3 + number_of_tags]  # Extract physical and elementary tags.
                nodes = values[3 + number_of_tags :]  # Extract Gmsh node IDs.
                physical_tag = tags[0] if tags else 0  # Read the physical group tag when present.
                if element_type == 1 and len(nodes) == 2:  # Select two-node boundary lines.
                    line_rows.append((physical_tag, nodes[0], nodes[1]))  # Preserve group and connectivity.
                if element_type == 2 and len(nodes) == 3:  # Select three-node domain triangles.
                    triangle_rows.append((nodes[0], nodes[1], nodes[2]))  # Preserve triangle connectivity.
            index += count + 3  # Skip the section and its end marker.
            continue  # Continue with the next section.
        index += 1  # Advance past an unrelated line.
    ordered_ids = sorted(node_coordinates)  # Establish deterministic zero-based node order.
    id_to_index = {node_id: position for position, node_id in enumerate(ordered_ids)}  # Map Gmsh IDs to zero-based indices.
    points = tuple(node_coordinates[node_id] for node_id in ordered_ids)  # Freeze planar coordinates.
    triangles = tuple(tuple(id_to_index[node_id] for node_id in row) for row in triangle_rows)  # Remap triangle connectivity.
    line_group_counts: dict[str, int] = {}  # Count boundary elements by physical name.
    for physical_tag, _, _ in line_rows:  # Inspect every boundary line element.
        name = physical_names.get((1, physical_tag), f"UNNAMED_{physical_tag}")  # Resolve the Physical Curve name.
        line_group_counts[name] = line_group_counts.get(name, 0) + 1  # Increment the named group count.
    return ParsedMesh(points, triangles, physical_names, line_group_counts)  # Freeze parsed mesh evidence.
def expected_entity(spec: dict[str, object]) -> tuple[int, float, tuple[float, ...]]:  # Compute exact topology and area facts from IMP.
    holes = list(spec.get("holes", []))  # Read all exact circular holes.
    hole_areas = tuple(pi * float(hole["radius_mm"]) ** 2 for hole in holes)  # Compute each exact disk area.
    area = float(spec["width_mm"]) * float(spec["height_mm"]) - sum(hole_areas)  # Compute the exact material-domain area.
    return 1 + len(holes), area, hole_areas  # Return outer-plus-hole loop count and exact areas.
def build_case(spec: dict[str, object], output_dir: Path) -> dict[str, object]:  # Build one frozen BREP and two independent meshes.
    validate_imp(spec)  # Reject invalid model facts before touching Gmsh.
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the case evidence directory.
    brep_path = output_dir / "model.brep"  # Allocate the one geometry entity shared by all meshes.
    geometry_path = output_dir / "geometry.geo"  # Allocate geometry-only Gmsh source.
    geometry_path.write_text(geometry_geo(spec, brep_path), encoding="utf-8")  # Persist deterministic entity construction source.
    run_gmsh(geometry_path)  # Build and export the OpenCASCADE BREP once.
    if not brep_path.exists() or brep_path.stat().st_size == 0:  # Require a real reusable entity artifact.
        raise RuntimeError("Gmsh did not create the BREP entity")  # Reject a source-only or point-cloud result.
    brep_digest = sha256(brep_path.read_bytes()).hexdigest()  # Fingerprint the frozen geometry entity.
    expected_components, expected_area, hole_areas = expected_entity(spec)  # Compute exact IMP-derived geometry facts.
    mesh_receipts: dict[str, dict[str, object]] = {}  # Collect independent coarse and fine mesh evidence.
    mesh_settings = (("coarse", float(spec["mesh"]["background_size_mm"]), float(spec["mesh"]["local_size_mm"])), ("fine", float(spec["mesh"]["background_size_mm"]) * 0.65, float(spec["mesh"]["local_size_mm"]) * 0.65))  # Define two mesh settings on one entity.
    for label, background_size, local_size in mesh_settings:  # Generate two meshes on the same BREP.
        msh_path = output_dir / f"{label}.msh"  # Allocate the mesh evidence path.
        source_path = output_dir / f"{label}.geo"  # Allocate mesh-only Gmsh source.
        source_path.write_text(mesh_geo(spec, brep_path, msh_path, background_size, local_size), encoding="utf-8")  # Persist deterministic mesh field source.
        run_gmsh(source_path)  # Mesh the already frozen BREP entity.
        parsed = parse_msh2(msh_path)  # Parse exact node, element, and Physical Group evidence.
        audit = audit_mesh(parsed.points, parsed.triangles, expected_components=expected_components, expected_area=expected_area, expected_hole_areas=hole_areas, area_tolerance=0.01, hole_tolerance=0.02, angle_limit=8.0, radius_ratio_limit=12.0)  # Enforce topology, geometry, and quality before solving.
        required_groups = {"DOMAIN", "FIXED_EDGE", "LOAD_EDGE", "BOTTOM_EDGE", "TOP_EDGE"} | {f"HOLE_{index}" for index in range(len(hole_areas))}  # Define all required physical names.
        available_names = set(parsed.physical_names.values())  # Collect names actually exported by Gmsh.
        missing_groups = sorted(required_groups - available_names)  # Detect missing boundary or material groups.
        if missing_groups:  # Reject an entity that cannot generate deterministic solver sets.
            raise RuntimeError(f"missing Physical Groups for {label}: {missing_groups}")  # Preserve exact missing names.
        if not audit.ok:  # Reject geometry drift or unusable element quality before CalculiX.
            raise RuntimeError(f"mesh audit failed for {label}: {audit.issues}")  # Preserve the complete hard-gate evidence.
        mesh_receipts[label] = {"brep_sha256": brep_digest, "nodes": len(parsed.points), "triangles": len(parsed.triangles), "line_group_counts": parsed.line_group_counts, "audit": audit.__dict__, "msh_file": msh_path.name}  # Store deterministic mesh evidence.
    manifest = {"schema_version": "entity-first-gmsh-receipt/1.0", "model_id": spec["model_id"], "brep_file": brep_path.name, "brep_sha256": brep_digest, "geometry_source": geometry_path.name, "mesh_receipts": mesh_receipts, "entity_invariant": mesh_receipts["coarse"]["brep_sha256"] == mesh_receipts["fine"]["brep_sha256"]}  # Build the complete entity-first receipt.
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")  # Persist the machine-readable model evidence.
    if not manifest["entity_invariant"]:  # Refuse mesh-size experiments that do not share one entity.
        raise RuntimeError("coarse and fine meshes do not share the same BREP")  # Enforce strict geometry isolation.
    return manifest  # Return complete case evidence.
def smoke_specs() -> tuple[dict[str, object], ...]:  # Define deterministic IMP fixtures for entity generation tests.
    return ({"schema_version": "entity-first-imp/1.0", "model_id": "bearing_plate", "kind": "plate_2d", "width_mm": 1000.0, "height_mm": 100.0, "holes": [], "mesh": {"background_size_mm": 35.0, "local_size_mm": 18.0}}, {"schema_version": "entity-first-imp/1.0", "model_id": "circular_opening", "kind": "plate_2d", "width_mm": 240.0, "height_mm": 240.0, "holes": [{"center_x_mm": 120.0, "center_y_mm": 120.0, "radius_mm": 20.0}], "mesh": {"background_size_mm": 20.0, "local_size_mm": 5.0}}, {"schema_version": "entity-first-imp/1.0", "model_id": "three_openings", "kind": "plate_2d", "width_mm": 600.0, "height_mm": 260.0, "holes": [{"center_x_mm": 130.0, "center_y_mm": 130.0, "radius_mm": 24.0}, {"center_x_mm": 300.0, "center_y_mm": 130.0, "radius_mm": 42.0}, {"center_x_mm": 450.0, "center_y_mm": 130.0, "radius_mm": 30.0}], "mesh": {"background_size_mm": 35.0, "local_size_mm": 8.0}})  # Return the three current triangular model families.
def self_test(output_root: Path) -> None:  # Build exact entities and two meshes for every Stage 2 fixture.
    receipts: dict[str, object] = {}  # Collect case receipts for one suite-level artifact.
    for spec in smoke_specs():  # Test each independent geometry family.
        receipts[str(spec["model_id"])] = build_case(spec, output_root / str(spec["model_id"]))  # Build and validate one exact entity.
    (output_root / "stage2_receipt.json").write_text(json.dumps({"schema_version": "entity-first-stage2-suite/1.0", "cases": receipts}, ensure_ascii=False, indent=2), encoding="utf-8")  # Persist suite evidence.
def main() -> int:  # Execute smoke fixtures or one supplied IMP.
    parser = argparse.ArgumentParser(description="Build exact Gmsh entities from system-generated IMP JSON")  # Describe the Stage 2 tool.
    parser.add_argument("--self-test", action="store_true")  # Request deterministic three-case smoke tests.
    parser.add_argument("--imp")  # Accept one system-generated IMP JSON file.
    parser.add_argument("--output-dir", required=True)  # Require an isolated evidence directory.
    args = parser.parse_args()  # Parse command-line arguments once.
    output_dir = Path(args.output_dir).resolve()  # Normalize the evidence directory.
    if args.self_test:  # Run the deterministic entity suite.
        self_test(output_dir)  # Build and validate all Stage 2 fixtures.
        return 0  # Return success only after all exact entities pass.
    if not args.imp:  # Require a structured input outside self-test mode.
        parser.error("--imp is required without --self-test")  # Reject an ambiguous model request.
    spec = json.loads(Path(args.imp).read_text(encoding="utf-8"))  # Read the system-generated model input.
    build_case(spec, output_dir)  # Build one exact entity and two validated meshes.
    return 0  # Return success after complete evidence generation.
if __name__ == "__main__":  # Run only when invoked directly.
    raise SystemExit(main())  # Propagate the process status to CI.