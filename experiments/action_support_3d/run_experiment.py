from __future__ import annotations  # Enable modern type annotations without runtime evaluation.
import csv  # Write compact tabular experiment results.
import itertools  # Build the structured evaluation-atom grid.
import json  # Read the committed LLM region proposals and write manifests.
import math  # Compute geometric distances for monitor-node detection.
import os  # Read optional executable overrides from the environment.
from pathlib import Path  # Handle repository-relative paths portably.
import shutil  # Resolve Gmsh and CalculiX executables on the runner.
import subprocess  # Execute Gmsh and CalculiX as deterministic external solvers.
import sys  # Return a nonzero status when the experiment cannot be completed.

ROOT = Path(__file__).resolve().parent  # Anchor all generated files to this experiment directory.
CASES_DIR = ROOT / "cases"  # Store per-case mesh and solver files under one deterministic directory.
RESULTS_DIR = ROOT / "results"  # Store compact CSV, JSON, and Markdown outputs under one deterministic directory.
PROPOSALS_PATH = ROOT / "llm_regions.json"  # Load the frozen LLM action-support proposals from version control.
LENGTH = 120.0  # Define the cantilever length in millimetres.
HALF_WIDTH = 20.0  # Define the half-width of the square section in millimetres.
HALF_HEIGHT = 20.0  # Define the half-height of the square section in millimetres.
HOLE_X = 35.0  # Place the through-hole centre along the beam axis in millimetres.
HOLE_RADIUS = 8.0  # Define the through-hole radius in millimetres.
TOTAL_LOAD_Z = -1000.0  # Apply a total downward nodal load of one kilonewton.
COARSE_H = 10.0  # Use a deliberately coarse background mesh size in millimetres.
REFINED_H = 4.0  # Use a single local-refinement level for every action-support candidate.
REFERENCE_H = 2.5  # Use a globally fine mesh as the numerical reference for both QoIs.
YOUNG = 210000.0  # Use a steel-like Young modulus in megapascal.
POISSON = 0.30  # Use a steel-like Poisson ratio.


def find_executable(primary: str, fallbacks: list[str]) -> str:  # Resolve one required executable with explicit fallbacks.
    override = os.environ.get(primary.upper() + "_BIN")  # Allow the workflow or a local user to override the executable path.
    if override and Path(override).exists():  # Accept an explicit executable path when it exists.
        return override  # Return the explicit executable path unchanged.
    direct = shutil.which(primary)  # Search the runner PATH for the preferred executable name.
    if direct:  # Accept the preferred executable when it is available.
        return direct  # Return the preferred executable path.
    for name in fallbacks:  # Try each known package-specific fallback name in deterministic order.
        resolved = shutil.which(name)  # Search the runner PATH for the current fallback executable name.
        if resolved:  # Accept the first fallback executable that exists.
            return resolved  # Return the discovered fallback executable path.
    raise FileNotFoundError(f"Required executable not found: {primary}")  # Fail explicitly rather than silently skipping the solver.


def load_llm_proposals() -> dict:  # Read the frozen region proposals that represent the LLM selection stage.
    with PROPOSALS_PATH.open("r", encoding="utf-8") as handle:  # Open the committed proposal file with deterministic UTF-8 decoding.
        return json.load(handle)  # Parse the proposal document into ordinary Python containers.


def evaluation_atoms() -> list[dict]:  # Build sixteen mesh-independent rectangular atoms used only as an oracle baseline.
    atoms: list[dict] = []  # Accumulate atom definitions without exposing element identifiers to the LLM.
    x_edges = [0.0, 30.0, 60.0, 90.0, 120.0]  # Divide the beam length into four equal axial slabs.
    y_edges = [-20.0, 0.0, 20.0]  # Divide the width into two halves to retain torsional asymmetry.
    z_edges = [-20.0, 0.0, 20.0]  # Divide the height into two halves to retain bending asymmetry.
    for ix, iy, iz in itertools.product(range(4), range(2), range(2)):  # Enumerate every Cartesian product of the coarse oracle bins.
        atoms.append({  # Append one deterministic atom description to the oracle list.
            "id": f"A{ix}{iy}{iz}",  # Give each atom a compact stable identifier.
            "source": "oracle_atom",  # Mark this support as a non-semantic baseline region.
            "qoi_target": "both",  # Evaluate every atom against both QoIs after one solve.
            "confidence": None,  # Leave probability undefined because atoms are not LLM predictions.
            "regions": [{  # Represent the atom as one box that is independent of the finite-element numbering.
                "type": "box",  # Use the box primitive implemented by the Gmsh background field generator.
                "xmin": x_edges[ix],  # Set the atom lower axial coordinate.
                "xmax": x_edges[ix + 1],  # Set the atom upper axial coordinate.
                "ymin": y_edges[iy],  # Set the atom lower transverse coordinate.
                "ymax": y_edges[iy + 1],  # Set the atom upper transverse coordinate.
                "zmin": z_edges[iz],  # Set the atom lower vertical coordinate.
                "zmax": z_edges[iz + 1],  # Set the atom upper vertical coordinate.
            }],  # Close the single-region atom definition.
        })  # Finish the current atom record.
    return atoms  # Return all sixteen atom supports to the experiment driver.


def flatten_llm_proposals(document: dict) -> list[dict]:  # Convert the two QoI-specific proposal groups into one case list.
    supports: list[dict] = []  # Accumulate one executable mesh-refinement case per LLM candidate support.
    for qoi_name, payload in document["qois"].items():  # Traverse each frozen QoI-conditioned LLM proposal group.
        for proposal in payload["proposals"]:  # Traverse each ranked proposal emitted for the current QoI.
            supports.append({  # Create one executable support record while preserving semantic metadata.
                "id": proposal["id"],  # Preserve the stable proposal identifier from the committed LLM output.
                "source": "llm_semantic",  # Mark the support as an LLM-generated semantic action object.
                "qoi_target": qoi_name,  # Preserve which QoI conditioned the LLM selection.
                "confidence": proposal["confidence"],  # Preserve the model-reported candidate probability for later calibration studies.
                "regions": proposal["regions"],  # Preserve the mesh-independent geometric support primitives.
                "rationale": proposal["rationale"],  # Preserve the physical relation used by the model to justify the joint action.
            })  # Finish the current semantic support record.
    return supports  # Return all frozen LLM action supports.


def gmsh_geo_text(regions: list[dict] | None, global_size: float | None = None) -> str:  # Generate one self-contained OpenCASCADE Gmsh model and size field.
    lines: list[str] = []  # Accumulate Gmsh statements so every generated statement can carry an inline comment.
    lines.append('SetFactory("OpenCASCADE"); // Use the OpenCASCADE kernel for robust three-dimensional Boolean geometry.')  # Select the geometry kernel.
    lines.append(f"Box(1) = {{0, -{HALF_WIDTH}, -{HALF_HEIGHT}, {LENGTH}, {2 * HALF_WIDTH}, {2 * HALF_HEIGHT}}}; // Create the rectangular cantilever solid.")  # Create the beam volume.
    lines.append(f"Cylinder(2) = {{{HOLE_X}, -25, 0, 0, 50, 0, {HOLE_RADIUS}}}; // Create the transverse cylindrical cutter for the through-hole.")  # Create the hole cutter.
    lines.append("BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; } // Subtract the cylinder from the beam to form the through-hole.")  # Form the final domain.
    lines.append("Mesh.ElementOrder = 1; // Use first-order tetrahedra so refinement changes are easy to interpret.")  # Keep the element interpolation fixed.
    lines.append("Mesh.Algorithm3D = 1; // Use the standard three-dimensional Delaunay tetrahedral mesher.")  # Choose a robust tetrahedral mesher.
    lines.append("Mesh.CharacteristicLengthFromCurvature = 0; // Prevent curvature heuristics from adding an uncontrolled competing refinement rule.")  # Disable curvature sizing.
    lines.append("Mesh.CharacteristicLengthExtendFromBoundary = 0; // Prevent boundary sizes from propagating beyond the explicit action support.")  # Disable automatic size propagation.
    if global_size is not None:  # Handle the coarse and globally fine reference meshes with one constant field.
        lines.append('Field[1] = MathEval; // Define a constant global mesh-size field for the requested baseline case.')  # Create a constant size field.
        lines.append(f'Field[1].F = "{global_size}"; // Set the global target tetrahedral edge length in millimetres.')  # Assign the constant size.
        lines.append("Background Field = 1; // Activate the constant global field as the only mesh-size controller.")  # Activate the global size field.
    else:  # Handle one or more local action-support regions on top of the common coarse background.
        field_ids: list[int] = []  # Track all local box-field identifiers for the final Min aggregator.
        for index, region in enumerate(regions or [], start=1):  # Convert every semantic or oracle support primitive into one Gmsh box field.
            if region.get("type") != "box":  # Reject unsupported primitives explicitly so proposal semantics cannot be silently altered.
                raise ValueError(f"Unsupported region primitive: {region.get('type')}")  # Fail when a proposal cannot be represented exactly by this benchmark version.
            field_ids.append(index)  # Record the current box field identifier for the final union operation.
            lines.append(f"Field[{index}] = Box; // Define local action-support box {index}.")  # Create one local box mesh-size field.
            lines.append(f"Field[{index}].VIn = {REFINED_H}; // Refine inside this action-support box to the common local size.")  # Set the local refined size.
            lines.append(f"Field[{index}].VOut = {COARSE_H}; // Keep the background outside this action-support box deliberately coarse.")  # Set the outside coarse size.
            lines.append(f"Field[{index}].XMin = {region['xmin']}; // Set the box lower axial bound.")  # Set the lower x bound.
            lines.append(f"Field[{index}].XMax = {region['xmax']}; // Set the box upper axial bound.")  # Set the upper x bound.
            lines.append(f"Field[{index}].YMin = {region['ymin']}; // Set the box lower transverse bound.")  # Set the lower y bound.
            lines.append(f"Field[{index}].YMax = {region['ymax']}; // Set the box upper transverse bound.")  # Set the upper y bound.
            lines.append(f"Field[{index}].ZMin = {region['zmin']}; // Set the box lower vertical bound.")  # Set the lower z bound.
            lines.append(f"Field[{index}].ZMax = {region['zmax']}; // Set the box upper vertical bound.")  # Set the upper z bound.
        min_id = len(field_ids) + 1  # Reserve the next field identifier for the union-by-minimum aggregator.
        fields_csv = ", ".join(str(field_id) for field_id in field_ids)  # Format the local field identifiers for Gmsh syntax.
        lines.append(f"Field[{min_id}] = Min; // Combine overlapping local action boxes without forcing a disjoint partition.")  # Create the overlap-preserving aggregator.
        lines.append(f"Field[{min_id}].FieldsList = {{{fields_csv}}}; // Union the candidate action support through the minimum requested mesh size.")  # Assign the local fields to the aggregator.
        lines.append(f"Background Field = {min_id}; // Activate the combined semantic or oracle action support.")  # Activate the local support union.
    lines.append("Mesh 3; // Generate the three-dimensional tetrahedral mesh after all sizing rules are defined.")  # Trigger volume meshing.
    return "\n".join(lines) + "\n"  # Return a newline-terminated Gmsh script for reproducible file output.


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:  # Execute one external program and persist its combined output for CI diagnosis.
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)  # Run the command without hiding a nonzero exit status.
    log_path.write_text(completed.stdout or "", encoding="utf-8")  # Persist the complete tool output before interpreting the return code.
    if completed.returncode != 0:  # Treat any solver or mesher failure as a hard experiment failure.
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}")  # Raise a concise error that points to the saved log.


def parse_msh2(path: Path) -> tuple[dict[int, tuple[float, float, float]], list[tuple[int, tuple[int, int, int, int]]]]:  # Parse only the MSH2 nodes and four-node tetrahedra needed by CalculiX.
    lines = path.read_text(encoding="utf-8").splitlines()  # Read the small ASCII MSH2 file into memory for deterministic parsing.
    nodes: dict[int, tuple[float, float, float]] = {}  # Store every Gmsh node by its integer identifier.
    tets: list[tuple[int, tuple[int, int, int, int]]] = []  # Store only first-order tetrahedral volume elements.
    index = 0  # Traverse the file with an explicit line cursor.
    while index < len(lines):  # Continue until every MSH2 section has been inspected.
        token = lines[index].strip()  # Normalize the current section marker or data line.
        if token == "$Nodes":  # Parse the standard MSH2 node section when encountered.
            count = int(lines[index + 1].strip())  # Read the declared node count.
            for offset in range(count):  # Parse exactly the declared number of node records.
                parts = lines[index + 2 + offset].split()  # Split the current node record into identifier and coordinates.
                nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))  # Store the current node coordinates.
            index += count + 3  # Skip the node data and its closing marker.
            continue  # Resume scanning from the next section marker.
        if token == "$Elements":  # Parse the standard MSH2 element section when encountered.
            count = int(lines[index + 1].strip())  # Read the declared element count across all dimensions.
            for offset in range(count):  # Parse exactly the declared number of element records.
                parts = lines[index + 2 + offset].split()  # Split the current element record into metadata and connectivity.
                element_id = int(parts[0])  # Read the stable Gmsh element identifier.
                element_type = int(parts[1])  # Read the MSH2 numeric element type.
                tag_count = int(parts[2])  # Read how many metadata tags precede the connectivity.
                if element_type == 4:  # Keep only four-node tetrahedra because the CalculiX deck uses C3D4 elements.
                    start = 3 + tag_count  # Locate the first node identifier after the variable tag block.
                    connectivity = tuple(int(value) for value in parts[start:start + 4])  # Read the tetrahedral connectivity in Gmsh order.
                    tets.append((element_id, connectivity))  # Preserve the element identifier and its four nodes.
            index += count + 3  # Skip the element data and its closing marker.
            continue  # Resume scanning from the next section marker.
        index += 1  # Advance over any unrelated MSH2 section or marker.
    if not nodes or not tets:  # Reject malformed or unexpectedly empty meshes before creating a solver deck.
        raise RuntimeError("MSH2 parsing produced no nodes or no tetrahedra")  # Fail explicitly when Gmsh output is unusable.
    return nodes, tets  # Return the parsed volume mesh to the CalculiX deck writer.


def select_node_sets(nodes: dict[int, tuple[float, float, float]]) -> dict[str, list[int]]:  # Build boundary, load, and local-QoI node sets from geometry rather than mesh numbering.
    fixed: list[int] = []  # Accumulate nodes lying on the clamped end face.
    load: list[int] = []  # Accumulate nodes lying in the eccentric load patch on the free end face.
    hole_top: list[int] = []  # Accumulate nodes on the upper arc of the cylindrical hole boundary.
    hole_bottom: list[int] = []  # Accumulate nodes on the lower arc of the cylindrical hole boundary.
    for node_id, (x, y, z) in nodes.items():  # Classify each node using only its physical coordinates.
        if abs(x) < 1.0e-6:  # Detect the exact clamped end face produced by OpenCASCADE.
            fixed.append(node_id)  # Add the current node to the fixed boundary set.
        if abs(x - LENGTH) < 1.0e-6 and y >= 5.0 and abs(z) <= 12.0:  # Detect an eccentric free-end patch that creates bending and torsion.
            load.append(node_id)  # Add the current node to the load distribution set.
        radial_error = abs(math.hypot(x - HOLE_X, z) - HOLE_RADIUS)  # Measure distance from the analytical cylindrical hole surface in the x-z plane.
        if radial_error < 1.0e-5 and z >= 4.0:  # Detect the upper half of the through-hole boundary independently of y.
            hole_top.append(node_id)  # Add the current node to the upper hole monitor set.
        if radial_error < 1.0e-5 and z <= -4.0:  # Detect the lower half of the through-hole boundary independently of y.
            hole_bottom.append(node_id)  # Add the current node to the lower hole monitor set.
    if not fixed or not load or not hole_top or not hole_bottom:  # Require every physical set before launching CalculiX.
        raise RuntimeError(f"Missing node set: fixed={len(fixed)}, load={len(load)}, hole_top={len(hole_top)}, hole_bottom={len(hole_bottom)}")  # Report exact set sizes for mesh-debugging.
    return {"FIXED": fixed, "LOAD": load, "HOLE_TOP": hole_top, "HOLE_BOTTOM": hole_bottom}  # Return all physically defined node sets.


def write_id_block(handle, keyword: str, ids: list[int]) -> None:  # Write one CalculiX set with compact comma-separated identifiers.
    handle.write(keyword + "\n")  # Start the requested NSET or ELSET card on its own line.
    for start in range(0, len(ids), 16):  # Limit each generated set-data line to a manageable number of identifiers.
        handle.write(", ".join(str(value) for value in ids[start:start + 16]) + "\n")  # Write the current identifier chunk.


def write_calculix_input(path: Path, nodes: dict[int, tuple[float, float, float]], tets: list[tuple[int, tuple[int, int, int, int]]], sets: dict[str, list[int]]) -> None:  # Create a minimal linear-static CalculiX deck for one mesh.
    force_per_node = TOTAL_LOAD_Z / len(sets["LOAD"])  # Divide the requested total force equally over the geometry-defined load-patch nodes.
    with path.open("w", encoding="utf-8") as handle:  # Open the deck file once to preserve deterministic card order.
        handle.write("*HEADING\n")  # Start the CalculiX model heading.
        handle.write("3D action-support semantics benchmark\n")  # Identify the benchmark in solver outputs.
        handle.write("*NODE\n")  # Start the nodal-coordinate block.
        for node_id, (x, y, z) in sorted(nodes.items()):  # Emit nodes in ascending identifier order for reproducibility.
            handle.write(f"{node_id}, {x:.12g}, {y:.12g}, {z:.12g}\n")  # Write one three-dimensional node record.
        handle.write("*ELEMENT, TYPE=C3D4, ELSET=SOLID\n")  # Start the first-order tetrahedral element block.
        for element_id, connectivity in tets:  # Emit all tetrahedral volume elements in the parsed Gmsh order.
            handle.write(f"{element_id}, {connectivity[0]}, {connectivity[1]}, {connectivity[2]}, {connectivity[3]}\n")  # Write one C3D4 connectivity record.
        write_id_block(handle, "*NSET, NSET=FIXED", sets["FIXED"])  # Write the clamped-end node set.
        write_id_block(handle, "*NSET, NSET=LOAD", sets["LOAD"])  # Write the eccentric free-end load node set.
        write_id_block(handle, "*NSET, NSET=HOLE_TOP", sets["HOLE_TOP"])  # Write the upper hole-monitor node set.
        write_id_block(handle, "*NSET, NSET=HOLE_BOTTOM", sets["HOLE_BOTTOM"])  # Write the lower hole-monitor node set.
        handle.write("*MATERIAL, NAME=STEEL\n")  # Define the single isotropic elastic material.
        handle.write("*ELASTIC\n")  # Start the elastic property card.
        handle.write(f"{YOUNG}, {POISSON}\n")  # Write Young modulus and Poisson ratio.
        handle.write("*SOLID SECTION, ELSET=SOLID, MATERIAL=STEEL\n")  # Assign the material to every tetrahedral element.
        handle.write("*BOUNDARY\n")  # Start the displacement boundary-condition card.
        handle.write("FIXED, 1, 3, 0.0\n")  # Clamp all three translational degrees of freedom on the root face.
        handle.write("*STEP\n")  # Start the single linear-static load step.
        handle.write("*STATIC\n")  # Request a static equilibrium solution.
        handle.write("*CLOAD\n")  # Start the concentrated nodal-load card.
        handle.write(f"LOAD, 3, {force_per_node:.12g}\n")  # Apply equal z-direction force to every node in the eccentric load patch.
        handle.write("*NODE PRINT, NSET=LOAD\n")  # Request nodal displacements at the load patch for the tip-displacement QoI.
        handle.write("U\n")  # Print all three displacement components for the load-patch nodes.
        handle.write("*NODE PRINT, NSET=HOLE_TOP\n")  # Request nodal displacements on the upper hole boundary for the local QoI.
        handle.write("U\n")  # Print all three displacement components for the upper hole-monitor nodes.
        handle.write("*NODE PRINT, NSET=HOLE_BOTTOM\n")  # Request nodal displacements on the lower hole boundary for the local QoI.
        handle.write("U\n")  # Print all three displacement components for the lower hole-monitor nodes.
        handle.write("*END STEP\n")  # End the only analysis step.


def parse_dat_displacements(path: Path, monitored_ids: set[int]) -> dict[int, tuple[float, float, float]]:  # Recover the last printed displacement triplet for each requested monitor node.
    values: dict[int, tuple[float, float, float]] = {}  # Store the latest valid displacement row for every monitored node identifier.
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():  # Scan the complete CalculiX text output defensively.
        parts = raw_line.split()  # Tokenize the current output line on arbitrary whitespace.
        if len(parts) < 4:  # Ignore headers, blank lines, and unrelated short records.
            continue  # Advance immediately when the line cannot contain node plus three displacement values.
        try:  # Attempt to interpret the first four tokens as a node displacement record.
            node_id = int(parts[0])  # Parse the first token as an integer node identifier.
            vector = (float(parts[1]), float(parts[2]), float(parts[3]))  # Parse the next three tokens as displacement components.
        except ValueError:  # Ignore any line whose tokens are not a pure node displacement record.
            continue  # Advance without treating unrelated solver text as data.
        if node_id in monitored_ids:  # Keep only nodes that belong to one of the three requested monitor sets.
            values[node_id] = vector  # Overwrite with the latest row so the final converged increment is retained.
    missing = monitored_ids.difference(values)  # Identify monitor nodes that never appeared as valid displacement rows.
    if missing:  # Refuse to compute a QoI from an incomplete printed result set.
        raise RuntimeError(f"Missing displacement output for {len(missing)} monitored nodes")  # Report the size of the parsing failure.
    return values  # Return the final displacement vector for every requested monitor node.


def mean_component(ids: list[int], values: dict[int, tuple[float, float, float]], component: int) -> float:  # Compute one arithmetic mean displacement component over a geometry-defined node set.
    return sum(values[node_id][component] for node_id in ids) / len(ids)  # Average the requested component over the complete set.


def solve_case(case: dict, gmsh_bin: str, ccx_bin: str) -> dict:  # Mesh, solve, and extract both QoIs for one action-support case.
    case_dir = CASES_DIR / case["id"]  # Allocate a deterministic directory named after the current support identifier.
    case_dir.mkdir(parents=True, exist_ok=True)  # Create the case directory and any missing parents idempotently.
    geo_path = case_dir / "model.geo"  # Store the generated Gmsh geometry and sizing program for auditability.
    msh_path = case_dir / "model.msh"  # Store the generated MSH2 tetrahedral mesh for solver conversion.
    if case["kind"] == "global":  # Generate either a coarse or fine mesh using a constant global target size.
        geo_path.write_text(gmsh_geo_text(None, global_size=case["mesh_size"]), encoding="utf-8")  # Write the requested constant-size Gmsh program.
    else:  # Generate one local-refinement mesh from an oracle atom or frozen LLM support.
        geo_path.write_text(gmsh_geo_text(case["regions"], global_size=None), encoding="utf-8")  # Write the overlap-preserving local support field.
    run_command([gmsh_bin, str(geo_path.name), "-3", "-format", "msh2", "-o", str(msh_path.name)], case_dir, case_dir / "gmsh.log")  # Generate the volume mesh with Gmsh.
    nodes, tets = parse_msh2(msh_path)  # Parse the generated nodes and tetrahedra into a solver-neutral representation.
    sets = select_node_sets(nodes)  # Reconstruct all boundary and QoI sets directly from physical coordinates.
    inp_path = case_dir / "job.inp"  # Store the generated CalculiX model deck under the standard job name.
    write_calculix_input(inp_path, nodes, tets, sets)  # Write the complete linear-static CalculiX input deck.
    run_command([ccx_bin, "-i", "job"], case_dir, case_dir / "ccx.log")  # Solve the current mesh with CalculiX and preserve its console output.
    dat_path = case_dir / "job.dat"  # Locate the text output requested by the NODE PRINT cards.
    monitored_ids = set(sets["LOAD"] + sets["HOLE_TOP"] + sets["HOLE_BOTTOM"])  # Combine all monitor nodes for one robust parser pass.
    displacements = parse_dat_displacements(dat_path, monitored_ids)  # Recover the converged nodal displacement vectors.
    tip_uz = mean_component(sets["LOAD"], displacements, 2)  # Define QoI A as the average vertical displacement over the eccentric load patch.
    hole_top_uz = mean_component(sets["HOLE_TOP"], displacements, 2)  # Compute the mean upper-hole vertical displacement.
    hole_bottom_uz = mean_component(sets["HOLE_BOTTOM"], displacements, 2)  # Compute the mean lower-hole vertical displacement.
    hole_opening = hole_top_uz - hole_bottom_uz  # Define QoI B as the relative vertical opening of the through-hole boundary.
    return {  # Return one compact, solver-derived record for downstream precision-resource comparison.
        "id": case["id"],  # Preserve the support identifier.
        "source": case["source"],  # Preserve whether the case is baseline, oracle atom, or LLM semantic support.
        "qoi_target": case.get("qoi_target", "both"),  # Preserve which QoI conditioned the semantic proposal when applicable.
        "confidence": case.get("confidence"),  # Preserve the LLM confidence for future calibration analysis.
        "nodes": len(nodes),  # Record node count as a transparent mesh-size indicator.
        "elements": len(tets),  # Record tetrahedral element count as a transparent resource indicator.
        "dof_proxy": 3 * len(nodes),  # Use three translational nodal degrees of freedom as the common resource proxy.
        "tip_uz": tip_uz,  # Store QoI A in millimetres.
        "hole_opening": hole_opening,  # Store QoI B in millimetres.
    }  # Finish the case result record.


def relative_error(value: float, reference: float) -> float:  # Compute an absolute relative QoI error with a safe reference magnitude check.
    scale = max(abs(reference), 1.0e-16)  # Prevent accidental division by zero for a nearly vanishing reference QoI.
    return abs(value - reference) / scale  # Return the dimensionless relative error.


def pareto_ids(rows: list[dict], error_key: str) -> set[str]:  # Identify candidates that are nondominated in error versus DOF proxy.
    frontier: set[str] = set()  # Accumulate candidate identifiers that no other candidate strictly dominates.
    for row in rows:  # Test each non-reference candidate against every alternative candidate.
        dominated = False  # Assume the current candidate is nondominated until a counterexample is found.
        for other in rows:  # Search the complete comparison set for one candidate that is no worse in both objectives.
            if other["id"] == row["id"]:  # Skip the trivial self-comparison.
                continue  # Advance to a genuinely different candidate.
            no_more_dof = other["dof_proxy"] <= row["dof_proxy"]  # Check whether the alternative consumes no more resource.
            no_more_error = other[error_key] <= row[error_key]  # Check whether the alternative has no larger QoI error.
            strictly_better = other["dof_proxy"] < row["dof_proxy"] or other[error_key] < row[error_key]  # Require strict improvement in at least one objective.
            if no_more_dof and no_more_error and strictly_better:  # Detect standard Pareto domination in the two-objective plane.
                dominated = True  # Mark the current candidate as dominated.
                break  # Stop searching once one valid dominator has been found.
        if not dominated:  # Keep only candidates that survived every domination test.
            frontier.add(row["id"])  # Add the current candidate to the nondominated frontier set.
    return frontier  # Return identifiers rather than mutable row objects.


def write_outputs(rows: list[dict], reference: dict) -> None:  # Add reference-relative errors, Pareto flags, and compact human-readable rankings.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Create the output directory idempotently before writing any artifact.
    candidates = [row for row in rows if row["id"] != reference["id"]]  # Exclude the globally fine reference from candidate rankings.
    for row in candidates:  # Compute the two reference-relative precision metrics for every candidate mesh.
        row["tip_rel_error"] = relative_error(row["tip_uz"], reference["tip_uz"])  # Measure QoI A error against the globally fine reference.
        row["hole_rel_error"] = relative_error(row["hole_opening"], reference["hole_opening"])  # Measure QoI B error against the globally fine reference.
    tip_frontier = pareto_ids(candidates, "tip_rel_error")  # Compute the precision-resource Pareto frontier for tip displacement.
    hole_frontier = pareto_ids(candidates, "hole_rel_error")  # Compute the precision-resource Pareto frontier for local hole opening.
    for row in candidates:  # Attach explicit Boolean frontier flags for machine-readable downstream analysis.
        row["tip_pareto"] = row["id"] in tip_frontier  # Flag whether the candidate is nondominated for QoI A.
        row["hole_pareto"] = row["id"] in hole_frontier  # Flag whether the candidate is nondominated for QoI B.
    fields = ["id", "source", "qoi_target", "confidence", "nodes", "elements", "dof_proxy", "tip_uz", "hole_opening", "tip_rel_error", "hole_rel_error", "tip_pareto", "hole_pareto"]  # Define a stable CSV schema for reproducible comparisons.
    with (RESULTS_DIR / "results.csv").open("w", encoding="utf-8", newline="") as handle:  # Open the compact result table with portable CSV newline handling.
        writer = csv.DictWriter(handle, fieldnames=fields)  # Create a deterministic column writer using the stable schema.
        writer.writeheader()  # Emit the CSV header before any candidate data.
        for row in candidates:  # Emit every non-reference candidate in execution order.
            writer.writerow({field: row.get(field) for field in fields})  # Restrict each row to the documented output schema.
    manifest = {"reference": reference, "candidate_count": len(candidates), "tip_pareto_ids": sorted(tip_frontier), "hole_pareto_ids": sorted(hole_frontier)}  # Build a compact JSON summary for automated consumers.
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # Persist the summary in a stable readable JSON form.
    tip_sorted = sorted(candidates, key=lambda row: (row["tip_rel_error"], row["dof_proxy"]))  # Rank all candidates first by QoI A precision and then by resource use.
    hole_sorted = sorted(candidates, key=lambda row: (row["hole_rel_error"], row["dof_proxy"]))  # Rank all candidates first by QoI B precision and then by resource use.
    markdown: list[str] = []  # Accumulate a concise report that can be read directly in a GitHub artifact.
    markdown.append("# 3D action-support benchmark result")  # Add the report title.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append(f"Reference mesh: {reference['nodes']} nodes, {reference['elements']} C3D4 elements.")  # Report the numerical reference mesh size.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("## Tip-displacement QoI: ten lowest errors")  # Start the QoI A ranking section.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("| rank | id | source | dof proxy | relative error | Pareto |")  # Add the QoI A ranking table header.
    markdown.append("| ---: | --- | --- | ---: | ---: | :---: |")  # Add the QoI A ranking table separator.
    for rank, row in enumerate(tip_sorted[:10], start=1):  # Show only the ten most precise candidates for compactness.
        markdown.append(f"| {rank} | {row['id']} | {row['source']} | {row['dof_proxy']} | {row['tip_rel_error']:.6e} | {row['tip_pareto']} |")  # Add one QoI A ranking row.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("## Hole-opening QoI: ten lowest errors")  # Start the QoI B ranking section.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("| rank | id | source | dof proxy | relative error | Pareto |")  # Add the QoI B ranking table header.
    markdown.append("| ---: | --- | --- | ---: | ---: | :---: |")  # Add the QoI B ranking table separator.
    for rank, row in enumerate(hole_sorted[:10], start=1):  # Show only the ten most precise candidates for compactness.
        markdown.append(f"| {rank} | {row['id']} | {row['source']} | {row['dof_proxy']} | {row['hole_rel_error']:.6e} | {row['hole_pareto']} |")  # Add one QoI B ranking row.
    (RESULTS_DIR / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")  # Persist the human-readable result summary.


def build_cases() -> list[dict]:  # Assemble the two baselines, sixteen oracle atoms, and frozen LLM semantic supports.
    document = load_llm_proposals()  # Read the committed LLM proposals exactly once per experiment run.
    cases: list[dict] = [  # Start with global baseline meshes that do not encode any local action support.
        {"id": "coarse_global", "kind": "global", "mesh_size": COARSE_H, "source": "baseline"},  # Add the deliberately coarse initial mesh.
        {"id": "reference_global", "kind": "global", "mesh_size": REFERENCE_H, "source": "reference"},  # Add the globally fine numerical reference mesh.
    ]  # Finish the baseline case list.
    for support in evaluation_atoms() + flatten_llm_proposals(document):  # Add oracle atoms and semantic supports under the same one-action refinement operator.
        support["kind"] = "local"  # Mark every support candidate as a local background-field refinement case.
        cases.append(support)  # Append the current action-support candidate to the execution list.
    return cases  # Return the complete deterministic experiment matrix.


def main() -> int:  # Execute the complete benchmark and return a shell-compatible status code.
    CASES_DIR.mkdir(parents=True, exist_ok=True)  # Create the case-output directory before launching external tools.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Create the compact result directory before any solver run.
    gmsh_bin = find_executable("gmsh", [])  # Resolve the Gmsh executable installed by the GitHub Actions runner package step.
    ccx_bin = find_executable("ccx", ["ccx_2.21", "ccx_2.20", "ccx_2.19"])  # Resolve CalculiX across common Ubuntu binary naming variants.
    rows: list[dict] = []  # Accumulate one solver-derived row for each experiment case.
    for case in build_cases():  # Execute every baseline, atom, and semantic support under an identical physics model.
        print(f"[run] {case['id']}", flush=True)  # Expose current case progress in the GitHub Actions log without changing computation.
        rows.append(solve_case(case, gmsh_bin, ccx_bin))  # Mesh, solve, and extract both QoIs for the current action support.
    reference = next(row for row in rows if row["id"] == "reference_global")  # Locate the globally fine reference result by its stable identifier.
    write_outputs(rows, reference)  # Compute precision-resource metrics and write compact comparison artifacts.
    print(f"[done] wrote {RESULTS_DIR / 'results.csv'}", flush=True)  # Report the primary result artifact path in the CI log.
    return 0  # Return success after every case and output step completed without exception.


if __name__ == "__main__":  # Execute the benchmark only when this file is invoked as the program entry point.
    try:  # Convert any explicit benchmark exception into a readable CI failure line.
        raise SystemExit(main())  # Run the benchmark and terminate with its returned status code.
    except Exception as exc:  # Catch unexpected errors only at the outermost command-line boundary.
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Emit a concise failure reason to the GitHub Actions log.
        raise  # Re-raise the original exception so GitHub Actions records a failing job.
