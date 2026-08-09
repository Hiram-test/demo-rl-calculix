from __future__ import annotations  # Enable modern type annotations without runtime evaluation.
import math  # Compare node coordinates to fixed geometric monitor points with a Euclidean tolerance.
import sys  # Preserve the same explicit command-line failure behavior as the core benchmark.
import run_experiment as core  # Reuse the validated mesh parser, CalculiX deck writer, ranking logic, and case matrix from v0.1.

FIXED_POINT_TOL = 1.0e-6  # Require monitor nodes to coincide with geometry-defined points to numerical precision.


def gmsh_geo_text_v2(regions: list[dict] | None, global_size: float | None = None) -> str:  # Generate the same beam while embedding fixed physical monitor points on the hole surface.
    lines: list[str] = []  # Accumulate Gmsh statements so every generated statement carries an inline explanation.
    lines.append('SetFactory("OpenCASCADE"); // Use OpenCASCADE for the three-dimensional Boolean geometry and embedded monitor points.')  # Select the CAD kernel.
    lines.append(f"Box(1) = {{0, -{core.HALF_WIDTH}, -{core.HALF_HEIGHT}, {core.LENGTH}, {2 * core.HALF_WIDTH}, {2 * core.HALF_HEIGHT}}}; // Create the rectangular cantilever solid.")  # Create the beam volume.
    lines.append(f"Cylinder(2) = {{{core.HOLE_X}, -25, 0, 0, 50, 0, {core.HOLE_RADIUS}}}; // Create the transverse cylindrical cutter for the through-hole.")  # Create the hole cutter.
    lines.append("BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; }; // Create one explicitly tagged beam-minus-hole volume for deterministic downstream geometry operations.")  # Form the final solid with an explicit volume tag.
    lines.append(f"Point(1001) = {{{core.HOLE_X}, 0, {core.HOLE_RADIUS}}}; // Create the fixed upper-hole physical monitor point at the mid-width section.")  # Create the upper monitor point.
    lines.append(f"Point(1002) = {{{core.HOLE_X}, 0, -{core.HOLE_RADIUS}}}; // Create the fixed lower-hole physical monitor point at the mid-width section.")  # Create the lower monitor point.
    lines.append(f"topSurface() = Closest {{{core.HOLE_X}, 0, {core.HOLE_RADIUS}}} {{ Surface{{:}}; }}; // Identify the hole surface closest to the upper monitor point.")  # Identify the supporting upper surface.
    lines.append(f"bottomSurface() = Closest {{{core.HOLE_X}, 0, -{core.HOLE_RADIUS}}} {{ Surface{{:}}; }}; // Identify the hole surface closest to the lower monitor point.")  # Identify the supporting lower surface.
    lines.append("Point{1001} In Surface{topSurface(0)}; // Embed the upper physical monitor point so every mesh contains the same QoI node.")  # Embed the upper monitor point.
    lines.append("Point{1002} In Surface{bottomSurface(0)}; // Embed the lower physical monitor point so every mesh contains the same QoI node.")  # Embed the lower monitor point.
    lines.append("Mesh.ElementOrder = 1; // Use first-order tetrahedra so all candidate actions share one interpolation order.")  # Keep the element order fixed.
    lines.append("Mesh.Algorithm3D = 1; // Use the standard three-dimensional Delaunay tetrahedral mesher.")  # Choose the 3D meshing algorithm.
    lines.append("Mesh.CharacteristicLengthFromCurvature = 0; // Disable curvature-driven refinement so the explicit action field remains the only local sizing signal.")  # Disable curvature sizing.
    lines.append("Mesh.CharacteristicLengthExtendFromBoundary = 0; // Disable automatic boundary-size propagation beyond the requested support.")  # Disable boundary size extension.
    if global_size is not None:  # Handle the coarse and reference cases with one constant field.
        lines.append('Field[1] = MathEval; // Define a constant target size for the requested global mesh case.')  # Create the constant field.
        lines.append(f'Field[1].F = "{global_size}"; // Set the global target edge length in millimetres.')  # Set the constant size.
        lines.append("Background Field = 1; // Activate the global constant size field.")  # Activate the global field.
    else:  # Handle one or more overlapping local action-support boxes on the common coarse background.
        field_ids: list[int] = []  # Track all local box-field identifiers for the final overlap-preserving aggregator.
        for index, region in enumerate(regions or [], start=1):  # Convert every support primitive into one local box size field.
            if region.get("type") != "box":  # Reject unsupported primitives instead of silently approximating the LLM proposal.
                raise ValueError(f"Unsupported region primitive: {region.get('type')}")  # Fail explicitly when the proposal cannot be represented exactly.
            field_ids.append(index)  # Record the current field identifier for the final Min union.
            lines.append(f"Field[{index}] = Box; // Define local action-support box {index}.")  # Create the current box field.
            lines.append(f"Field[{index}].VIn = {core.REFINED_H}; // Use the common refined size inside the current action support.")  # Set the inside mesh size.
            lines.append(f"Field[{index}].VOut = {core.COARSE_H}; // Preserve the common coarse size outside the current action support.")  # Set the outside mesh size.
            lines.append(f"Field[{index}].XMin = {region['xmin']}; // Set the lower axial support bound.")  # Set the lower x bound.
            lines.append(f"Field[{index}].XMax = {region['xmax']}; // Set the upper axial support bound.")  # Set the upper x bound.
            lines.append(f"Field[{index}].YMin = {region['ymin']}; // Set the lower transverse support bound.")  # Set the lower y bound.
            lines.append(f"Field[{index}].YMax = {region['ymax']}; // Set the upper transverse support bound.")  # Set the upper y bound.
            lines.append(f"Field[{index}].ZMin = {region['zmin']}; // Set the lower vertical support bound.")  # Set the lower z bound.
            lines.append(f"Field[{index}].ZMax = {region['zmax']}; // Set the upper vertical support bound.")  # Set the upper z bound.
        min_id = len(field_ids) + 1  # Reserve the next field identifier for the support union.
        fields_csv = ", ".join(str(field_id) for field_id in field_ids)  # Format local field identifiers for Gmsh syntax.
        lines.append(f"Field[{min_id}] = Min; // Preserve overlaps by taking the minimum requested size across all support boxes.")  # Create the overlap union field.
        lines.append(f"Field[{min_id}].FieldsList = {{{fields_csv}}}; // Attach every support box to the overlap union field.")  # Configure the union members.
        lines.append(f"Background Field = {min_id}; // Activate the complete local action support as the mesh-size controller.")  # Activate the local field.
    lines.append("Mesh 3; // Generate the three-dimensional tetrahedral mesh after geometry and support fields are complete.")  # Trigger volume meshing.
    return "\n".join(lines) + "\n"  # Return a newline-terminated deterministic Gmsh program.


def squared_distance(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:  # Compute squared Euclidean distance without an unnecessary square root.
    return sum((value_a - value_b) ** 2 for value_a, value_b in zip(point_a, point_b))  # Return the coordinate-wise squared-distance sum.


def nearest_exact_node(nodes: dict[int, tuple[float, float, float]], target: tuple[float, float, float], label: str) -> int:  # Resolve one geometry-defined monitor point to its coincident mesh node.
    node_id, point = min(nodes.items(), key=lambda item: squared_distance(item[1], target))  # Find the mesh node closest to the fixed physical target.
    distance = math.sqrt(squared_distance(point, target))  # Convert the minimum squared distance into millimetres for validation.
    if distance > FIXED_POINT_TOL:  # Reject any mesh that failed to preserve the embedded physical monitor point.
        raise RuntimeError(f"Fixed point {label} missing: nearest node is {distance:.6e} mm away")  # Report the physical mismatch explicitly.
    return node_id  # Return the coincident node identifier after validating the geometry contract.


def select_node_sets_v2(nodes: dict[int, tuple[float, float, float]]) -> dict[str, list[int]]:  # Build mesh-invariant load and QoI sets from fixed physical points.
    fixed = [node_id for node_id, (x, _y, _z) in nodes.items() if abs(x) < FIXED_POINT_TOL]  # Clamp every node on the x=0 root face as before.
    load_upper = nearest_exact_node(nodes, (core.LENGTH, core.HALF_WIDTH, core.HALF_HEIGHT), "load_upper")  # Use the positive-y positive-z free-end corner as the first fixed load node.
    load_lower = nearest_exact_node(nodes, (core.LENGTH, core.HALF_WIDTH, -core.HALF_HEIGHT), "load_lower")  # Use the positive-y negative-z free-end corner as the second fixed load node.
    hole_top = nearest_exact_node(nodes, (core.HOLE_X, 0.0, core.HOLE_RADIUS), "hole_top")  # Resolve the embedded upper-hole mid-width monitor node.
    hole_bottom = nearest_exact_node(nodes, (core.HOLE_X, 0.0, -core.HOLE_RADIUS), "hole_bottom")  # Resolve the embedded lower-hole mid-width monitor node.
    if not fixed:  # Require a valid clamped face before launching the structural solve.
        raise RuntimeError("No nodes found on the fixed x=0 face")  # Fail explicitly when the geometry-to-boundary mapping breaks.
    return {"FIXED": fixed, "LOAD": [load_upper, load_lower], "HOLE_TOP": [hole_top], "HOLE_BOTTOM": [hole_bottom]}  # Return fixed sets whose physical locations are identical for every mesh.


def main() -> int:  # Replace only the geometry and node-set definitions while reusing the complete v0.1 execution and ranking pipeline.
    core.gmsh_geo_text = gmsh_geo_text_v2  # Replace the mesh generator with the fixed-monitor-point geometry implementation.
    core.select_node_sets = select_node_sets_v2  # Replace variable node averaging with mesh-invariant physical-point sets.
    return core.main()  # Execute the same baselines, 16 spatial atoms, six frozen LLM supports, and Pareto analysis.


if __name__ == "__main__":  # Execute the corrected benchmark only when this file is invoked as the program entry point.
    try:  # Preserve the same explicit CI failure behavior as the core benchmark.
        raise SystemExit(main())  # Run the corrected benchmark and terminate with its returned status code.
    except Exception as exc:  # Catch errors only at the outermost command-line boundary for concise CI diagnostics.
        print(f"[fatal-v2] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Report the corrected benchmark failure reason without hiding the exception.
        raise  # Re-raise the original exception so GitHub Actions records the failing run.
