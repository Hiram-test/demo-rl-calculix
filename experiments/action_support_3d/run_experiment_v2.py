from __future__ import annotations  # Enable modern type annotations without runtime evaluation.
import math  # Validate fixed geometric load nodes and evaluate tetrahedral interpolation tolerances.
from pathlib import Path  # Reuse repository-relative case paths without string-specific assumptions.
import sys  # Preserve explicit command-line failure reporting for GitHub Actions.
import run_experiment as core  # Reuse the validated Gmsh model, case matrix, parsers, rankings, and output writers from v0.1.

FIXED_POINT_TOL = 1.0e-6  # Require geometry-defined corner load nodes to coincide with mesh nodes to numerical precision.
INTERP_TOL = 1.0e-9  # Allow a small barycentric tolerance when a fixed physical point lies numerically on a tetrahedral face.
HOLE_POINT_TOP = (core.HOLE_X, 0.0, 10.0)  # Define a fixed material point two millimetres above the radius-eight hole at the mid-width plane.
HOLE_POINT_BOTTOM = (core.HOLE_X, 0.0, -10.0)  # Define the symmetric fixed material point two millimetres below the hole at the mid-width plane.


def squared_distance(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:  # Compute squared Euclidean distance without an unnecessary square root.
    return sum((value_a - value_b) ** 2 for value_a, value_b in zip(point_a, point_b))  # Return the coordinate-wise squared-distance sum.


def nearest_exact_node(nodes: dict[int, tuple[float, float, float]], target: tuple[float, float, float], label: str) -> int:  # Resolve one CAD corner to its coincident mesh node.
    node_id, point = min(nodes.items(), key=lambda item: squared_distance(item[1], target))  # Find the mesh node closest to the fixed physical target.
    distance = math.sqrt(squared_distance(point, target))  # Convert the minimum squared distance into millimetres for validation.
    if distance > FIXED_POINT_TOL:  # Reject a mesh that somehow lost a required CAD corner node.
        raise RuntimeError(f"Fixed point {label} missing: nearest node is {distance:.6e} mm away")  # Report the geometric mismatch explicitly.
    return node_id  # Return the coincident node identifier after validating the geometry contract.


def select_node_sets_v2(nodes: dict[int, tuple[float, float, float]]) -> dict[str, list[int]]:  # Build mesh-invariant boundary and load sets from CAD coordinates.
    fixed = [node_id for node_id, (x, _y, _z) in nodes.items() if abs(x) < FIXED_POINT_TOL]  # Clamp every mesh node on the physical x=0 end face.
    load_upper = nearest_exact_node(nodes, (core.LENGTH, core.HALF_WIDTH, core.HALF_HEIGHT), "load_upper")  # Use the positive-y positive-z free-end CAD corner as one fixed load node.
    load_lower = nearest_exact_node(nodes, (core.LENGTH, core.HALF_WIDTH, -core.HALF_HEIGHT), "load_lower")  # Use the positive-y negative-z free-end CAD corner as the second fixed load node.
    if not fixed:  # Require the clamped face to contain at least one mesh node before solving.
        raise RuntimeError("No nodes found on the fixed x=0 face")  # Fail explicitly when the boundary-condition geometry cannot be reconstructed.
    return {"FIXED": fixed, "LOAD": [load_upper, load_lower]}  # Return only mesh-invariant physical boundary and load sets.


def write_calculix_input_v2(path: Path, nodes: dict[int, tuple[float, float, float]], tets: list[tuple[int, tuple[int, int, int, int]]], sets: dict[str, list[int]]) -> None:  # Write a linear-static CalculiX deck that prints every nodal displacement for fixed-point interpolation.
    force_per_node = core.TOTAL_LOAD_Z / len(sets["LOAD"])  # Split the fixed total force equally between the two fixed positive-y free-end corners.
    all_node_ids = sorted(nodes)  # Build one deterministic all-node set so interior fixed-point values can be interpolated after the solve.
    with path.open("w", encoding="utf-8") as handle:  # Open the complete solver deck in deterministic write order.
        handle.write("*HEADING\n")  # Start the CalculiX model heading.
        handle.write("3D action-support semantics benchmark v0.2 fixed physical QoI\n")  # Identify the corrected mesh-invariant evaluation protocol in solver outputs.
        handle.write("*NODE\n")  # Start the nodal-coordinate block.
        for node_id, (x, y, z) in sorted(nodes.items()):  # Emit every mesh node in ascending identifier order.
            handle.write(f"{node_id}, {x:.12g}, {y:.12g}, {z:.12g}\n")  # Write one three-dimensional node record.
        handle.write("*ELEMENT, TYPE=C3D4, ELSET=SOLID\n")  # Start the first-order tetrahedral volume-element block.
        for element_id, connectivity in tets:  # Emit every tetrahedron in the validated Gmsh parse order.
            handle.write(f"{element_id}, {connectivity[0]}, {connectivity[1]}, {connectivity[2]}, {connectivity[3]}\n")  # Write one C3D4 connectivity record.
        core.write_id_block(handle, "*NSET, NSET=FIXED", sets["FIXED"])  # Write the geometry-defined clamped-end node set.
        core.write_id_block(handle, "*NSET, NSET=LOAD", sets["LOAD"])  # Write the two geometry-defined eccentric free-end load nodes.
        core.write_id_block(handle, "*NSET, NSET=ALLNODES", all_node_ids)  # Write every mesh node so all displacement values are available for interpolation.
        handle.write("*MATERIAL, NAME=STEEL\n")  # Define the single isotropic elastic material.
        handle.write("*ELASTIC\n")  # Start the elastic material-property card.
        handle.write(f"{core.YOUNG}, {core.POISSON}\n")  # Write Young modulus and Poisson ratio in the benchmark unit system.
        handle.write("*SOLID SECTION, ELSET=SOLID, MATERIAL=STEEL\n")  # Assign the material to every tetrahedral volume element.
        handle.write("*BOUNDARY\n")  # Start the displacement boundary-condition card.
        handle.write("FIXED, 1, 3, 0.0\n")  # Clamp all three translational degrees of freedom on the root face.
        handle.write("*STEP\n")  # Start the only linear-static load step.
        handle.write("*STATIC\n")  # Request static equilibrium.
        handle.write("*CLOAD\n")  # Start the concentrated nodal-load card.
        handle.write(f"LOAD, 3, {force_per_node:.12g}\n")  # Apply the same total z-force on every mesh through two fixed CAD corners.
        handle.write("*NODE PRINT, NSET=ALLNODES\n")  # Request all nodal displacements needed for mesh-independent physical-point interpolation.
        handle.write("U\n")  # Print the three translational displacement components for every node.
        handle.write("*END STEP\n")  # End the only analysis step.


def determinant3(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> float:  # Evaluate a three-by-three determinant from column vectors.
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1]) + c[0] * (a[1] * b[2] - a[2] * b[1])  # Expand the determinant explicitly for dependency-free barycentric interpolation.


def subtract(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:  # Subtract one three-dimensional coordinate vector from another.
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])  # Return the component-wise vector difference.


def barycentric_weights(point: tuple[float, float, float], vertices: list[tuple[float, float, float]]) -> tuple[float, float, float, float] | None:  # Compute linear C3D4 shape-function weights at one physical point.
    a, b, c, d = vertices  # Unpack the tetrahedral vertices in the same order as the element connectivity.
    ad = subtract(a, d)  # Form the first column of the affine tetrahedral coordinate matrix.
    bd = subtract(b, d)  # Form the second column of the affine tetrahedral coordinate matrix.
    cd = subtract(c, d)  # Form the third column of the affine tetrahedral coordinate matrix.
    pd = subtract(point, d)  # Form the physical-point vector measured from the fourth tetrahedral vertex.
    denominator = determinant3(ad, bd, cd)  # Compute six times the signed tetrahedral volume.
    if abs(denominator) < 1.0e-18:  # Reject numerically degenerate tetrahedra before dividing by the signed volume.
        return None  # Signal that this tetrahedron cannot support a stable interpolation calculation.
    weight_a = determinant3(pd, bd, cd) / denominator  # Solve the first affine coordinate by Cramer's rule.
    weight_b = determinant3(ad, pd, cd) / denominator  # Solve the second affine coordinate by Cramer's rule.
    weight_c = determinant3(ad, bd, pd) / denominator  # Solve the third affine coordinate by Cramer's rule.
    weight_d = 1.0 - weight_a - weight_b - weight_c  # Recover the fourth affine coordinate from partition of unity.
    return (weight_a, weight_b, weight_c, weight_d)  # Return the four C3D4 linear shape-function values.


def locate_point(nodes: dict[int, tuple[float, float, float]], tets: list[tuple[int, tuple[int, int, int, int]]], point: tuple[float, float, float], label: str) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float]]:  # Find a tetrahedron containing one fixed material-space QoI point.
    px, py, pz = point  # Unpack the physical query point once for inexpensive bounding-box rejection.
    for _element_id, connectivity in tets:  # Scan tetrahedra until one contains the fixed physical point.
        vertices = [nodes[node_id] for node_id in connectivity]  # Resolve the current tetrahedral connectivity to physical coordinates.
        if px < min(vertex[0] for vertex in vertices) - INTERP_TOL or px > max(vertex[0] for vertex in vertices) + INTERP_TOL:  # Reject tetrahedra whose axial bounds exclude the point.
            continue  # Advance immediately when the point cannot lie in the current tetrahedron.
        if py < min(vertex[1] for vertex in vertices) - INTERP_TOL or py > max(vertex[1] for vertex in vertices) + INTERP_TOL:  # Reject tetrahedra whose transverse bounds exclude the point.
            continue  # Advance immediately when the point cannot lie in the current tetrahedron.
        if pz < min(vertex[2] for vertex in vertices) - INTERP_TOL or pz > max(vertex[2] for vertex in vertices) + INTERP_TOL:  # Reject tetrahedra whose vertical bounds exclude the point.
            continue  # Advance immediately when the point cannot lie in the current tetrahedron.
        weights = barycentric_weights(point, vertices)  # Compute the exact C3D4 interpolation weights for the surviving tetrahedron.
        if weights is not None and min(weights) >= -INTERP_TOL and max(weights) <= 1.0 + INTERP_TOL:  # Accept the tetrahedron when all affine coordinates lie inside the closed simplex.
            return connectivity, weights  # Return the containing element connectivity and fixed-point shape-function weights.
    raise RuntimeError(f"Physical QoI point {label} was not located inside any C3D4 element")  # Fail explicitly if meshing or point placement invalidates the QoI definition.


def interpolate_component(connectivity: tuple[int, int, int, int], weights: tuple[float, float, float, float], values: dict[int, tuple[float, float, float]], component: int) -> float:  # Interpolate one displacement component at a fixed material point.
    return sum(weight * values[node_id][component] for node_id, weight in zip(connectivity, weights))  # Apply the C3D4 linear shape functions to the solved nodal displacement values.


def solve_case_v2(case: dict, gmsh_bin: str, ccx_bin: str) -> dict:  # Mesh, solve, and evaluate two mesh-invariant physical QoIs for one candidate support.
    case_dir = core.CASES_DIR / case["id"]  # Allocate the same deterministic case directory used by the v0.1 experiment driver.
    case_dir.mkdir(parents=True, exist_ok=True)  # Create the case directory and any missing parents idempotently.
    geo_path = case_dir / "model.geo"  # Store the generated Gmsh geometry and sizing script for auditability.
    msh_path = case_dir / "model.msh"  # Store the generated MSH2 tetrahedral mesh for CalculiX conversion.
    if case["kind"] == "global":  # Generate either the coarse baseline or globally fine numerical reference mesh.
        geo_path.write_text(core.gmsh_geo_text(None, global_size=case["mesh_size"]), encoding="utf-8")  # Reuse the already validated global-size Gmsh geometry program.
    else:  # Generate one locally refined mesh from an oracle atom or frozen LLM action support.
        geo_path.write_text(core.gmsh_geo_text(case["regions"], global_size=None), encoding="utf-8")  # Reuse the already validated overlap-preserving local support fields.
    core.run_command([gmsh_bin, str(geo_path.name), "-3", "-format", "msh2", "-o", str(msh_path.name)], case_dir, case_dir / "gmsh.log")  # Generate the tetrahedral volume mesh with the validated Gmsh path.
    nodes, tets = core.parse_msh2(msh_path)  # Parse the generated nodes and C3D4 connectivity into solver-neutral Python structures.
    sets = select_node_sets_v2(nodes)  # Reconstruct only fixed physical boundary and load sets from CAD coordinates.
    top_connectivity, top_weights = locate_point(nodes, tets, HOLE_POINT_TOP, "hole_top_material")  # Locate the fixed upper material point before the structural solve.
    bottom_connectivity, bottom_weights = locate_point(nodes, tets, HOLE_POINT_BOTTOM, "hole_bottom_material")  # Locate the fixed lower material point before the structural solve.
    inp_path = case_dir / "job.inp"  # Store the corrected CalculiX deck under the standard job name.
    write_calculix_input_v2(inp_path, nodes, tets, sets)  # Write the complete fixed-load all-displacement CalculiX input deck.
    core.run_command([ccx_bin, "-i", "job"], case_dir, case_dir / "ccx.log")  # Solve the current mesh with CalculiX and persist the complete console log.
    dat_path = case_dir / "job.dat"  # Locate the text displacement output produced by NODE PRINT.
    displacements = core.parse_dat_displacements(dat_path, set(nodes))  # Recover converged displacements for every mesh node exactly once.
    tip_uz = sum(displacements[node_id][2] for node_id in sets["LOAD"]) / len(sets["LOAD"])  # Define QoI A as mean z-displacement of the same two free-end CAD corners on every mesh.
    hole_top_uz = interpolate_component(top_connectivity, top_weights, displacements, 2)  # Interpolate vertical displacement at the fixed material point above the hole.
    hole_bottom_uz = interpolate_component(bottom_connectivity, bottom_weights, displacements, 2)  # Interpolate vertical displacement at the fixed material point below the hole.
    hole_opening = hole_top_uz - hole_bottom_uz  # Define QoI B as a mesh-invariant local differential displacement across the hole ligaments.
    return {"id": case["id"], "source": case["source"], "qoi_target": case.get("qoi_target", "both"), "confidence": case.get("confidence"), "nodes": len(nodes), "elements": len(tets), "dof_proxy": 3 * len(nodes), "tip_uz": tip_uz, "hole_opening": hole_opening}  # Return the compact solver-derived precision-resource record expected by the shared ranking pipeline.


def main() -> int:  # Replace only the case solver while preserving the frozen candidate matrix and shared Pareto analysis.
    core.solve_case = solve_case_v2  # Use fixed CAD loads and fixed material-point interpolation for every baseline and candidate mesh.
    return core.main()  # Execute the same two baselines, sixteen spatial atoms, six frozen LLM supports, and output ranking logic.


if __name__ == "__main__":  # Execute the corrected benchmark only when this file is invoked as the program entry point.
    try:  # Preserve explicit CI failure reporting around the shared benchmark driver.
        raise SystemExit(main())  # Run the mesh-invariant benchmark and terminate with its returned status code.
    except Exception as exc:  # Catch unexpected errors only at the outermost command-line boundary.
        print(f"[fatal-v2] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Emit a concise failure reason to the GitHub Actions log and committed console tail.
        raise  # Re-raise the original exception so the workflow check records a genuine failure.
