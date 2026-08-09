from __future__ import annotations  # Enable modern type annotations without runtime evaluation.
import sys  # Preserve explicit command-line failure reporting for GitHub Actions.
import run_experiment as core  # Reuse the validated case matrix, Gmsh generator, parsers, ranking logic, and result writer.
import run_experiment_v2 as v2  # Reuse the fixed-load CalculiX deck and dependency-free C3D4 physical-point interpolation utilities.

TIP_POINT = (110.0, 0.0, 0.0)  # Evaluate global vertical response at a fixed interior point ten millimetres upstream of the singular point loads.
LOCAL_POINT_TOP = (core.HOLE_X, 0.0, 10.0)  # Evaluate the upper local axial displacement at a fixed material point two millimetres outside the hole.
LOCAL_POINT_BOTTOM = (core.HOLE_X, 0.0, -10.0)  # Evaluate the lower local axial displacement at the symmetric fixed material point below the hole.


def solve_case_v3(case: dict, gmsh_bin: str, ccx_bin: str) -> dict:  # Mesh, solve, and evaluate the two final mesh-invariant QoIs for one action-support candidate.
    case_dir = core.CASES_DIR / case["id"]  # Allocate the same stable directory used by all prior benchmark revisions.
    case_dir.mkdir(parents=True, exist_ok=True)  # Create the case directory and any missing parents idempotently.
    geo_path = case_dir / "model.geo"  # Store the generated Gmsh geometry and action-support sizing program for auditability.
    msh_path = case_dir / "model.msh"  # Store the generated MSH2 tetrahedral mesh for CalculiX conversion.
    if case["kind"] == "global":  # Generate either the deliberately coarse baseline or globally fine numerical reference mesh.
        geo_path.write_text(core.gmsh_geo_text(None, global_size=case["mesh_size"]), encoding="utf-8")  # Reuse the already validated constant-size Gmsh model.
    else:  # Generate one locally refined mesh from an oracle atom or a frozen LLM semantic action support.
        geo_path.write_text(core.gmsh_geo_text(case["regions"], global_size=None), encoding="utf-8")  # Reuse the already validated overlap-preserving local support fields.
    core.run_command([gmsh_bin, str(geo_path.name), "-3", "-format", "msh2", "-o", str(msh_path.name)], case_dir, case_dir / "gmsh.log")  # Generate the C3D4-compatible tetrahedral mesh with Gmsh.
    nodes, tets = core.parse_msh2(msh_path)  # Parse the mesh into physical node coordinates and four-node tetrahedral connectivity.
    sets = v2.select_node_sets_v2(nodes)  # Apply the same fixed root face and the same two positive-y CAD-corner point loads on every mesh.
    tip_connectivity, tip_weights = v2.locate_point(nodes, tets, TIP_POINT, "global_tip_material")  # Locate the fixed interior global-response point independently of mesh numbering.
    top_connectivity, top_weights = v2.locate_point(nodes, tets, LOCAL_POINT_TOP, "local_top_material")  # Locate the fixed material point above the through-hole.
    bottom_connectivity, bottom_weights = v2.locate_point(nodes, tets, LOCAL_POINT_BOTTOM, "local_bottom_material")  # Locate the fixed material point below the through-hole.
    inp_path = case_dir / "job.inp"  # Store the CalculiX model deck under the stable job name expected by the shared driver.
    v2.write_calculix_input_v2(inp_path, nodes, tets, sets)  # Write the fixed-load deck and request all nodal displacements for physical-point interpolation.
    core.run_command([ccx_bin, "-i", "job"], case_dir, case_dir / "ccx.log")  # Solve the linear-static elasticity problem and preserve the full CalculiX console log.
    dat_path = case_dir / "job.dat"  # Locate the all-node displacement output requested by the v0.2 CalculiX deck.
    displacements = core.parse_dat_displacements(dat_path, set(nodes))  # Recover the converged three-component displacement vector at every mesh node.
    tip_uz = v2.interpolate_component(tip_connectivity, tip_weights, displacements, 2)  # Define QoI A as vertical displacement at the same interior material point on every mesh.
    local_top_ux = v2.interpolate_component(top_connectivity, top_weights, displacements, 0)  # Interpolate axial displacement above the hole where bending produces a strong signed signal.
    local_bottom_ux = v2.interpolate_component(bottom_connectivity, bottom_weights, displacements, 0)  # Interpolate axial displacement below the hole at the symmetric physical location.
    local_axial_difference = local_top_ux - local_bottom_ux  # Define QoI B as the local axial displacement difference across the two hole ligaments.
    return {"id": case["id"], "source": case["source"], "qoi_target": case.get("qoi_target", "both"), "confidence": case.get("confidence"), "nodes": len(nodes), "elements": len(tets), "dof_proxy": 3 * len(nodes), "tip_uz": tip_uz, "hole_opening": local_axial_difference}  # Return the shared output schema while using the legacy hole_opening column for the new local axial-difference QoI.


def main() -> int:  # Replace only the per-case QoI evaluator while preserving frozen candidate supports and identical mesh/refinement operators.
    core.solve_case = solve_case_v3  # Install the final fixed-material-point evaluator into the shared benchmark driver.
    return core.main()  # Execute the same coarse/reference meshes, sixteen spatial atoms, six frozen LLM candidates, and Pareto analysis.


if __name__ == "__main__":  # Execute the final benchmark revision only when this file is invoked as the program entry point.
    try:  # Preserve explicit workflow failure behavior around the shared benchmark driver.
        raise SystemExit(main())  # Run the final benchmark revision and terminate with its returned shell status.
    except Exception as exc:  # Catch unexpected errors only at the outermost command-line boundary.
        print(f"[fatal-v3] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Emit a concise failure reason to the GitHub Actions log and persisted console tail.
        raise  # Re-raise the original exception so the workflow accurately records a failed numerical run.
