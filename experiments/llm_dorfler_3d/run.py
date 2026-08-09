from __future__ import annotations  # Enable postponed evaluation of type annotations for a compact standalone script.
import csv  # Write the adaptive histories to a compact machine-readable table.
import itertools  # Build the fixed Cartesian atom vocabulary used only by the LLM gate.
import json  # Read the frozen LLM atom selection and write compact diagnostics.
import math  # Evaluate tetrahedral volumes, interpolation weights, and stress-jump norms.
import os  # Read optional executable overrides from the GitHub Actions environment.
from pathlib import Path  # Handle all repository-relative experiment paths portably.
import shutil  # Resolve Gmsh and CalculiX executables from the runner PATH.
import subprocess  # Execute deterministic Gmsh and CalculiX subprocesses.
import sys  # Propagate numerical failures to GitHub Actions with a nonzero exit code.

ROOT = Path(__file__).resolve().parent  # Anchor generated cases and results to this experiment directory.
CASES_DIR = ROOT / "cases"  # Keep transient meshes and solver files outside the compact result directory.
RESULTS_DIR = ROOT / "results"  # Persist only compact adaptive histories and summaries here.
SELECTION_PATH = ROOT / "llm_selection.json"  # Load the frozen semantic atom selection committed before solving.
LENGTH = 120.0  # Set the beam length in millimetres.
HALF_WIDTH = 20.0  # Set half the square-section width in millimetres.
HALF_HEIGHT = 20.0  # Set half the square-section height in millimetres.
HOLE_X = 35.0  # Place the transverse through-hole centre at x equals 35 millimetres.
HOLE_RADIUS = 8.0  # Set the transverse through-hole radius in millimetres.
TOTAL_LOAD_Z = -1000.0  # Apply a total downward load of one kilonewton on the positive-y half of the free face.
YOUNG = 210000.0  # Use a steel-like Young modulus in MPa.
POISSON = 0.30  # Use a steel-like Poisson ratio.
COARSE_H = 10.0  # Start every adaptive history from the same deliberately coarse global mesh size.
REFERENCE_H = 2.5  # Use one substantially finer global mesh as the fixed QoI reference.
MIN_LOCAL_H = 3.2  # Prevent repeated local refinement from becoming as fine as the numerical reference mesh.
REFINE_FACTOR = 0.65  # Reduce the characteristic size of every marked element neighbourhood by this factor.
BOX_PAD_FACTOR = 0.30  # Expand each marked tetrahedron bounding box slightly to obtain a stable remeshing neighbourhood.
THETA = 0.50  # Use exactly the same Dorfler bulk parameter in both adaptive methods.
AMR_ROUNDS = 4  # Perform four mark-refine cycles after the shared initial coarse solve.
QOI_POINT = (110.0, 0.0, 0.0)  # Measure vertical displacement at a fixed interior material point away from the load face.
GEOM_TOL = 1.0e-7  # Use a tight coordinate tolerance for physical boundary-node selection.
INTERP_TOL = 1.0e-9  # Allow a small barycentric tolerance when the QoI lies numerically on a tetrahedral face.


def find_executable(primary: str, fallbacks: list[str]) -> str:  # Resolve one native executable while allowing explicit runner overrides.
    override = os.environ.get(primary.upper() + "_BIN")  # Read an optional explicit executable path from the environment.
    if override and Path(override).exists():  # Accept the explicit override only when the file actually exists.
        return override  # Return the validated explicit path unchanged.
    direct = shutil.which(primary)  # Search the runner PATH for the preferred executable name.
    if direct:  # Accept the preferred executable when available.
        return direct  # Return the discovered preferred executable path.
    for fallback in fallbacks:  # Try package-specific fallback names in deterministic order.
        resolved = shutil.which(fallback)  # Search the runner PATH for the current fallback name.
        if resolved:  # Accept the first fallback executable that exists.
            return resolved  # Return the discovered fallback path.
    raise FileNotFoundError(f"Required executable not found: {primary}")  # Fail explicitly when the required native tool is unavailable.


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:  # Execute one native command and preserve complete combined output for diagnosis.
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)  # Run synchronously without suppressing a nonzero status.
    log_path.write_text(completed.stdout or "", encoding="utf-8")  # Persist the complete command output before interpreting its status.
    if completed.returncode != 0:  # Treat a mesher or solver failure as a hard experiment failure.
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}")  # Report the exact failed command while retaining its saved log.


def load_selection() -> dict:  # Read the semantic gate frozen before any numerical ranking is observed.
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))  # Parse the committed UTF-8 selection document into ordinary Python containers.


def atom_boxes(selection: dict) -> dict[str, dict[str, float]]:  # Reconstruct every fixed atom from the committed Cartesian grid description.
    x_edges = [float(value) for value in selection["atom_grid"]["x_edges_mm"]]  # Read the axial atom boundaries in millimetres.
    y_edges = [float(value) for value in selection["atom_grid"]["y_edges_mm"]]  # Read the transverse atom boundaries in millimetres.
    z_edges = [float(value) for value in selection["atom_grid"]["z_edges_mm"]]  # Read the vertical atom boundaries in millimetres.
    atoms: dict[str, dict[str, float]] = {}  # Accumulate one mesh-independent box for every atom identifier.
    for ix, iy, iz in itertools.product(range(len(x_edges) - 1), range(len(y_edges) - 1), range(len(z_edges) - 1)):  # Enumerate the complete Cartesian atom vocabulary.
        atom_id = f"A{ix}{iy}{iz}"  # Construct the stable identifier using the documented index rule.
        atoms[atom_id] = {"xmin": x_edges[ix], "xmax": x_edges[ix + 1], "ymin": y_edges[iy], "ymax": y_edges[iy + 1], "zmin": z_edges[iz], "zmax": z_edges[iz + 1]}  # Store the current closed atom box in physical coordinates.
    return atoms  # Return the complete atom dictionary for semantic eligibility tests.


def gmsh_geo_text(refinement_boxes: list[dict[str, float]] | None = None, global_h: float | None = None) -> str:  # Generate the common beam geometry and either global or accumulated local sizing fields.
    lines: list[str] = []  # Accumulate one deterministic Gmsh statement per list entry.
    lines.append('SetFactory("OpenCASCADE"); // Use OpenCASCADE for robust three-dimensional Boolean geometry.')  # Select the CAD kernel.
    lines.append(f"Box(1) = {{0, -{HALF_WIDTH}, -{HALF_HEIGHT}, {LENGTH}, {2.0 * HALF_WIDTH}, {2.0 * HALF_HEIGHT}}}; // Create the rectangular cantilever solid.")  # Create the beam body.
    lines.append(f"Cylinder(2) = {{{HOLE_X}, -25, 0, 0, 50, 0, {HOLE_RADIUS}}}; // Create the transverse cylindrical cutter along the y direction.")  # Create the through-hole cutter.
    lines.append("BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; } // Subtract the cylinder from the beam body.")  # Form the final perforated cantilever domain.
    lines.append("Mesh.ElementOrder = 1; // Use first-order tetrahedra for transparent local marking and remeshing.")  # Fix the interpolation order.
    lines.append("Mesh.Algorithm3D = 1; // Use the standard three-dimensional Delaunay tetrahedral mesher.")  # Choose the tetrahedral meshing algorithm.
    lines.append("Mesh.CharacteristicLengthFromCurvature = 0; // Disable curvature-driven sizing so refinement is controlled only by the experiment fields.")  # Remove curvature as a competing refinement rule.
    lines.append("Mesh.CharacteristicLengthExtendFromBoundary = 0; // Disable uncontrolled propagation of boundary sizes into the volume.")  # Remove automatic boundary-size extension.
    if global_h is not None:  # Handle the common coarse mesh and the globally fine reference with one constant field.
        lines.append('Field[1] = MathEval; // Define a constant global target-size field.')  # Create the global size field.
        lines.append(f'Field[1].F = "{global_h}"; // Set the requested global tetrahedral target size in millimetres.')  # Assign the global target size.
        lines.append("Background Field = 1; // Activate the constant field as the sole background-size controller.")  # Activate the global field.
    else:  # Handle one adaptive mesh reconstructed from the accumulated marked-element neighbourhoods.
        lines.append('Field[1] = MathEval; // Define the unchanged coarse background target-size field.')  # Create the common coarse background field.
        lines.append(f'Field[1].F = "{COARSE_H}"; // Keep all locations without prior marks at the original coarse size.')  # Assign the coarse background size.
        field_ids: list[int] = [1]  # Include the coarse background in the final minimum-size aggregation.
        for offset, box in enumerate(refinement_boxes or [], start=2):  # Convert each accumulated marked-element neighbourhood into one local box field.
            field_ids.append(offset)  # Record the current field identifier for the final Min aggregator.
            lines.append(f"Field[{offset}] = Box; // Define accumulated local refinement neighbourhood {offset - 1}.")  # Create the current local box field.
            lines.append(f"Field[{offset}].VIn = {box['vin']}; // Request the stored refined size inside this marked-element neighbourhood.")  # Set the local refined size.
            lines.append(f"Field[{offset}].VOut = {COARSE_H}; // Leave the surrounding background at the common coarse size.")  # Set the outside size.
            lines.append(f"Field[{offset}].XMin = {box['xmin']}; // Set the local neighbourhood lower x bound.")  # Set the lower x coordinate.
            lines.append(f"Field[{offset}].XMax = {box['xmax']}; // Set the local neighbourhood upper x bound.")  # Set the upper x coordinate.
            lines.append(f"Field[{offset}].YMin = {box['ymin']}; // Set the local neighbourhood lower y bound.")  # Set the lower y coordinate.
            lines.append(f"Field[{offset}].YMax = {box['ymax']}; // Set the local neighbourhood upper y bound.")  # Set the upper y coordinate.
            lines.append(f"Field[{offset}].ZMin = {box['zmin']}; // Set the local neighbourhood lower z bound.")  # Set the lower z coordinate.
            lines.append(f"Field[{offset}].ZMax = {box['zmax']}; // Set the local neighbourhood upper z bound.")  # Set the upper z coordinate.
        min_id = len(field_ids) + 1  # Reserve the next field identifier for the minimum-size union.
        fields_csv = ", ".join(str(value) for value in field_ids)  # Format all field identifiers for valid Gmsh list syntax.
        lines.append(f"Field[{min_id}] = Min; // Combine the coarse background with every accumulated marked-element neighbourhood.")  # Create the overlap-preserving union field.
        lines.append(f"Field[{min_id}].FieldsList = {{{fields_csv}}}; // Apply the smallest requested size wherever refinement neighbourhoods overlap.")  # Attach all component fields to the union.
        lines.append(f"Background Field = {min_id}; // Activate the complete adaptive sizing history.")  # Activate the accumulated adaptive field.
    lines.append("Mesh 3; // Generate the three-dimensional tetrahedral mesh after the sizing rule is complete.")  # Trigger volume meshing.
    return "\n".join(lines) + "\n"  # Return one newline-terminated deterministic Gmsh program.


def parse_msh2(path: Path) -> tuple[dict[int, tuple[float, float, float]], list[tuple[int, tuple[int, int, int, int]]]]:  # Parse only MSH2 nodes and first-order tetrahedra required by CalculiX.
    lines = path.read_text(encoding="utf-8").splitlines()  # Read the small ASCII MSH2 file into memory for deterministic scanning.
    nodes: dict[int, tuple[float, float, float]] = {}  # Store all volume-mesh node coordinates by identifier.
    tets: list[tuple[int, tuple[int, int, int, int]]] = []  # Store all four-node tetrahedral connectivities by element identifier.
    index = 0  # Traverse the MSH2 text with one explicit cursor.
    while index < len(lines):  # Continue until every MSH2 section has been inspected.
        token = lines[index].strip()  # Normalize the current section marker.
        if token == "$Nodes":  # Enter the standard MSH2 node section.
            count = int(lines[index + 1].strip())  # Read the declared node count.
            for offset in range(count):  # Parse exactly the declared number of node records.
                parts = lines[index + 2 + offset].split()  # Split the current node row into identifier and coordinates.
                nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))  # Store the current three-dimensional node coordinate.
            index += count + 3  # Skip the parsed node records and closing section marker.
            continue  # Resume scanning from the next MSH2 section.
        if token == "$Elements":  # Enter the standard MSH2 element section.
            count = int(lines[index + 1].strip())  # Read the declared total element count.
            for offset in range(count):  # Parse exactly the declared number of element records.
                parts = lines[index + 2 + offset].split()  # Split the current element row into metadata and connectivity.
                element_id = int(parts[0])  # Read the stable Gmsh element identifier.
                element_type = int(parts[1])  # Read the numeric MSH2 element type.
                tag_count = int(parts[2])  # Read the number of metadata tags preceding connectivity.
                if element_type == 4:  # Keep only first-order four-node tetrahedra.
                    start = 3 + tag_count  # Locate the first connectivity token after the metadata tags.
                    connectivity = tuple(int(value) for value in parts[start:start + 4])  # Parse the four tetrahedral node identifiers.
                    tets.append((element_id, connectivity))  # Preserve the volume element and its connectivity.
            index += count + 3  # Skip the parsed element records and closing section marker.
            continue  # Resume scanning from the next MSH2 section.
        index += 1  # Advance over unrelated MSH2 sections and markers.
    if not nodes or not tets:  # Reject a malformed or unexpectedly empty generated volume mesh.
        raise RuntimeError("MSH2 parsing produced no nodes or no C3D4 tetrahedra")  # Fail explicitly before launching CalculiX.
    return nodes, tets  # Return the solver-neutral volume mesh.


def squared_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:  # Compute squared Euclidean distance between two physical points.
    return sum((va - vb) ** 2 for va, vb in zip(a, b))  # Return the component-wise squared-distance sum.


def nearest_exact_node(nodes: dict[int, tuple[float, float, float]], target: tuple[float, float, float], label: str) -> int:  # Resolve one CAD corner or point to a coincident mesh node.
    node_id, coordinate = min(nodes.items(), key=lambda item: squared_distance(item[1], target))  # Find the nearest generated mesh node to the requested physical point.
    distance = math.sqrt(squared_distance(coordinate, target))  # Convert the minimum squared distance into millimetres.
    if distance > GEOM_TOL:  # Reject a mesh that failed to preserve the required CAD point.
        raise RuntimeError(f"Required physical node {label} missing by {distance:.6e} mm")  # Report the physical mismatch explicitly.
    return node_id  # Return the validated coincident mesh node identifier.


def select_node_sets(nodes: dict[int, tuple[float, float, float]]) -> dict[str, list[int]]:  # Build the common clamped face, distributed eccentric load patch, and all-node output set.
    fixed = [node_id for node_id, (x, _y, _z) in nodes.items() if abs(x) <= GEOM_TOL]  # Clamp every generated node on the physical x equals zero end face.
    load = [node_id for node_id, (x, y, _z) in nodes.items() if abs(x - LENGTH) <= GEOM_TOL and y >= -GEOM_TOL]  # Load the positive-y half of the free end face to create bending plus torsion without a single-node singularity.
    if not fixed or not load:  # Require both physical sets before writing the structural model.
        raise RuntimeError(f"Missing physical node set: fixed={len(fixed)}, load={len(load)}")  # Report exact set sizes when mesh-to-geometry mapping fails.
    return {"FIXED": sorted(fixed), "LOAD": sorted(load), "ALLNODES": sorted(nodes)}  # Return deterministic node lists for CalculiX cards.


def write_id_block(handle, keyword: str, ids: list[int]) -> None:  # Write one CalculiX node set using compact comma-separated identifier rows.
    handle.write(keyword + "\n")  # Start the requested set card on its own line.
    for start in range(0, len(ids), 16):  # Limit every generated identifier row to a manageable width.
        handle.write(", ".join(str(value) for value in ids[start:start + 16]) + "\n")  # Write the current deterministic identifier chunk.


def write_calculix_input(path: Path, nodes: dict[int, tuple[float, float, float]], tets: list[tuple[int, tuple[int, int, int, int]]], sets: dict[str, list[int]]) -> None:  # Write the common linear-static deck with all-node displacements and integration-point stresses.
    force_per_node = TOTAL_LOAD_Z / len(sets["LOAD"])  # Divide the fixed total load equally over the physical positive-y free-face nodes.
    with path.open("w", encoding="utf-8") as handle:  # Open the complete solver deck in deterministic write order.
        handle.write("*HEADING\n")  # Start the CalculiX model heading.
        handle.write("LLM semantic gating versus global Dorfler AMR\n")  # Identify this isolated experiment in solver output files.
        handle.write("*NODE\n")  # Start the nodal-coordinate block.
        for node_id, (x, y, z) in sorted(nodes.items()):  # Emit all nodes in ascending identifier order.
            handle.write(f"{node_id}, {x:.12g}, {y:.12g}, {z:.12g}\n")  # Write one three-dimensional node record.
        handle.write("*ELEMENT, TYPE=C3D4, ELSET=SOLID\n")  # Start the first-order tetrahedral element block.
        for element_id, connectivity in tets:  # Emit every tetrahedral element in parsed Gmsh order.
            handle.write(f"{element_id}, {connectivity[0]}, {connectivity[1]}, {connectivity[2]}, {connectivity[3]}\n")  # Write one C3D4 connectivity record.
        write_id_block(handle, "*NSET, NSET=FIXED", sets["FIXED"])  # Write the physical clamped-end node set.
        write_id_block(handle, "*NSET, NSET=LOAD", sets["LOAD"])  # Write the physical eccentric half-face load node set.
        write_id_block(handle, "*NSET, NSET=ALLNODES", sets["ALLNODES"])  # Write all mesh nodes for mesh-independent QoI interpolation.
        handle.write("*MATERIAL, NAME=STEEL\n")  # Define the single isotropic elastic material.
        handle.write("*ELASTIC\n")  # Start the elastic property card.
        handle.write(f"{YOUNG}, {POISSON}\n")  # Write Young modulus and Poisson ratio in the benchmark unit system.
        handle.write("*SOLID SECTION, ELSET=SOLID, MATERIAL=STEEL\n")  # Assign the elastic material to every tetrahedral volume element.
        handle.write("*BOUNDARY\n")  # Start the displacement boundary-condition block.
        handle.write("FIXED, 1, 3, 0.0\n")  # Clamp all three translational degrees of freedom on the root face.
        handle.write("*STEP\n")  # Start the only structural load step.
        handle.write("*STATIC\n")  # Request linear static equilibrium.
        handle.write("*CLOAD\n")  # Start the nodal-load block.
        handle.write(f"LOAD, 3, {force_per_node:.12g}\n")  # Apply the same total negative-z force on every adaptive mesh.
        handle.write("*NODE PRINT, NSET=ALLNODES\n")  # Request all converged nodal displacements for fixed-point interpolation.
        handle.write("U\n")  # Print the three translational displacement components.
        handle.write("*EL PRINT, ELSET=SOLID\n")  # Request integration-point stresses for the common local error indicator.
        handle.write("S\n")  # Print all six Cauchy stress components at integration points.
        handle.write("*END STEP\n")  # End the only static load step.


def determinant3(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> float:  # Evaluate a three-by-three determinant from column vectors.
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1]) + c[0] * (a[1] * b[2] - a[2] * b[1])  # Expand the determinant explicitly without external dependencies.


def subtract(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:  # Subtract one three-dimensional vector from another.
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])  # Return the component-wise vector difference.


def barycentric_weights(point: tuple[float, float, float], vertices: list[tuple[float, float, float]]) -> tuple[float, float, float, float] | None:  # Compute linear C3D4 interpolation weights at one physical point.
    a, b, c, d = vertices  # Unpack the four tetrahedral vertices in connectivity order.
    ad = subtract(a, d)  # Form the first affine coordinate column.
    bd = subtract(b, d)  # Form the second affine coordinate column.
    cd = subtract(c, d)  # Form the third affine coordinate column.
    pd = subtract(point, d)  # Form the query-point vector relative to the fourth vertex.
    denominator = determinant3(ad, bd, cd)  # Compute six times the signed tetrahedral volume.
    if abs(denominator) < 1.0e-18:  # Reject numerically degenerate tetrahedra before division.
        return None  # Signal that this element cannot support stable interpolation.
    wa = determinant3(pd, bd, cd) / denominator  # Solve the first barycentric coordinate by Cramer's rule.
    wb = determinant3(ad, pd, cd) / denominator  # Solve the second barycentric coordinate by Cramer's rule.
    wc = determinant3(ad, bd, pd) / denominator  # Solve the third barycentric coordinate by Cramer's rule.
    wd = 1.0 - wa - wb - wc  # Recover the fourth barycentric coordinate from partition of unity.
    return (wa, wb, wc, wd)  # Return the four linear shape-function values.


def locate_point(nodes: dict[int, tuple[float, float, float]], tets: list[tuple[int, tuple[int, int, int, int]]], point: tuple[float, float, float]) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float]]:  # Locate the fixed QoI material point inside one generated tetrahedron.
    px, py, pz = point  # Unpack the query point once for inexpensive bounding-box rejection.
    for _element_id, connectivity in tets:  # Scan tetrahedra until one contains the physical point.
        vertices = [nodes[node_id] for node_id in connectivity]  # Resolve the current connectivity to physical coordinates.
        if px < min(value[0] for value in vertices) - INTERP_TOL or px > max(value[0] for value in vertices) + INTERP_TOL:  # Reject tetrahedra outside the query x coordinate.
            continue  # Advance immediately when the point cannot lie in the current tetrahedron.
        if py < min(value[1] for value in vertices) - INTERP_TOL or py > max(value[1] for value in vertices) + INTERP_TOL:  # Reject tetrahedra outside the query y coordinate.
            continue  # Advance immediately when the point cannot lie in the current tetrahedron.
        if pz < min(value[2] for value in vertices) - INTERP_TOL or pz > max(value[2] for value in vertices) + INTERP_TOL:  # Reject tetrahedra outside the query z coordinate.
            continue  # Advance immediately when the point cannot lie in the current tetrahedron.
        weights = barycentric_weights(point, vertices)  # Compute the exact C3D4 shape-function weights for the surviving tetrahedron.
        if weights is not None and min(weights) >= -INTERP_TOL and max(weights) <= 1.0 + INTERP_TOL:  # Accept the element when all barycentric coordinates lie inside the closed simplex.
            return connectivity, weights  # Return the containing element and interpolation weights.
    raise RuntimeError("The fixed QoI point was not located inside any generated C3D4 element")  # Fail explicitly if remeshing invalidates the physical QoI definition.


def parse_dat(path: Path, node_ids: set[int], element_ids: set[int]) -> tuple[dict[int, tuple[float, float, float]], dict[int, tuple[float, float, float, float, float, float]]]:  # Parse the final all-node displacement block and integration-point stress table from CalculiX ASCII output.
    displacements: dict[int, tuple[float, float, float]] = {}  # Store the latest converged displacement vector for each mesh node.
    stress_samples: dict[int, list[tuple[float, float, float, float, float, float]]] = {}  # Accumulate every printed integration-point stress sample by element identifier.
    section = ""  # Track whether the current numeric rows belong to displacement or stress output.
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():  # Scan the complete solver text output defensively.
        lower = raw_line.lower()  # Normalize the current row for robust header detection.
        if "displacements" in lower:  # Detect a CalculiX nodal displacement table header.
            section = "u"  # Route following numeric rows to the displacement parser.
            continue  # Advance to the next line after consuming the header.
        if "stresses (elem" in lower:  # Detect the standard CalculiX integration-point stress table header.
            section = "s"  # Route following numeric rows to the stress parser.
            continue  # Advance to the next line after consuming the header.
        parts = raw_line.split()  # Tokenize the current row on arbitrary whitespace.
        if section == "u" and len(parts) >= 4:  # Attempt to parse one nodal displacement row only inside the displacement section.
            try:  # Guard against unrelated headers and total-force rows inside the text output.
                node_id = int(parts[0])  # Parse the first token as a node identifier.
                vector = (float(parts[1]), float(parts[2]), float(parts[3]))  # Parse the next three tokens as displacement components.
            except ValueError:  # Ignore rows that are not plain node-displacement records.
                continue  # Advance without modifying parsed displacement state.
            if node_id in node_ids:  # Keep only identifiers belonging to the current mesh node set.
                displacements[node_id] = vector  # Preserve the latest converged displacement vector for this node.
        if section == "s" and len(parts) >= 8:  # Attempt to parse one integration-point stress row only inside the stress section.
            try:  # Guard against unrelated stress-section text and headers.
                element_id = int(parts[0])  # Parse the first token as an element identifier.
                _integration_point = int(parts[1])  # Parse and deliberately retain no dependence on the integration-point number.
                stress = tuple(float(value) for value in parts[2:8])  # Parse the six Cauchy stress components in CalculiX order.
            except ValueError:  # Ignore rows that are not plain element stress records.
                continue  # Advance without modifying parsed stress state.
            if element_id in element_ids:  # Keep only identifiers belonging to the current volume mesh.
                stress_samples.setdefault(element_id, []).append(stress)  # Accumulate all integration-point samples for robust averaging.
    missing_nodes = node_ids.difference(displacements)  # Identify any nodes that never appeared in the displacement output.
    if missing_nodes:  # Refuse to interpolate the QoI from an incomplete solution field.
        raise RuntimeError(f"Missing displacement output for {len(missing_nodes)} mesh nodes")  # Report the exact size of the parsing failure.
    stresses: dict[int, tuple[float, float, float, float, float, float]] = {}  # Store one integration-point-averaged stress tensor per tetrahedron.
    for element_id in element_ids:  # Convert every element's printed integration-point samples into one representative tensor.
        samples = stress_samples.get(element_id, [])  # Retrieve all stress samples printed for the current element.
        if not samples:  # Require a valid stress tensor for every tetrahedron used by the error indicator.
            raise RuntimeError(f"Missing stress output for element {element_id}")  # Fail explicitly when the stress parser loses an element.
        stresses[element_id] = tuple(sum(sample[index] for sample in samples) / len(samples) for index in range(6))  # Average each tensor component across any available integration points.
    return displacements, stresses  # Return the complete converged nodal displacement field and element stress field.


def interpolate_component(connectivity: tuple[int, int, int, int], weights: tuple[float, float, float, float], values: dict[int, tuple[float, float, float]], component: int) -> float:  # Interpolate one displacement component at the fixed physical QoI point.
    return sum(weight * values[node_id][component] for node_id, weight in zip(connectivity, weights))  # Apply the C3D4 linear shape functions to the solved nodal values.


def tet_volume(nodes: dict[int, tuple[float, float, float]], connectivity: tuple[int, int, int, int]) -> float:  # Compute the physical volume of one generated tetrahedron.
    a, b, c, d = [nodes[node_id] for node_id in connectivity]  # Resolve the four tetrahedral vertices from the mesh node dictionary.
    volume6 = abs(determinant3(subtract(a, d), subtract(b, d), subtract(c, d)))  # Compute six times the absolute tetrahedral volume.
    return volume6 / 6.0  # Return the physical tetrahedral volume in cubic millimetres.


def tet_centroid(nodes: dict[int, tuple[float, float, float]], connectivity: tuple[int, int, int, int]) -> tuple[float, float, float]:  # Compute the physical centroid of one tetrahedral element.
    vertices = [nodes[node_id] for node_id in connectivity]  # Resolve the four tetrahedral vertex coordinates.
    return tuple(sum(vertex[index] for vertex in vertices) / 4.0 for index in range(3))  # Return the arithmetic mean coordinate in each direction.


def stress_distance_squared(a: tuple[float, float, float, float, float, float], b: tuple[float, float, float, float, float, float]) -> float:  # Measure the squared tensor jump using a symmetric-tensor Frobenius norm.
    differences = [va - vb for va, vb in zip(a, b)]  # Compute the six independent stress-component differences.
    return differences[0] ** 2 + differences[1] ** 2 + differences[2] ** 2 + 2.0 * (differences[3] ** 2 + differences[4] ** 2 + differences[5] ** 2)  # Count shear components twice in the full symmetric-tensor norm.


def stress_jump_indicator(nodes: dict[int, tuple[float, float, float]], tets: list[tuple[int, tuple[int, int, int, int]]], stresses: dict[int, tuple[float, float, float, float, float, float]]) -> dict[int, float]:  # Build one common element indicator from stress jumps across shared tetrahedral faces.
    face_map: dict[tuple[int, int, int], list[int]] = {}  # Map every unordered triangular face to the adjacent volume element identifiers.
    connectivity_map = {element_id: connectivity for element_id, connectivity in tets}  # Build direct connectivity lookup for volume weighting.
    for element_id, connectivity in tets:  # Enumerate all tetrahedral faces once.
        for face in itertools.combinations(connectivity, 3):  # Generate the four triangular faces of the current tetrahedron.
            face_map.setdefault(tuple(sorted(face)), []).append(element_id)  # Attach the current element to the unordered face key.
    neighbours: dict[int, list[int]] = {element_id: [] for element_id, _connectivity in tets}  # Initialize an adjacency list for every tetrahedral element.
    for adjacent in face_map.values():  # Inspect every unique triangular face in the mesh.
        if len(adjacent) == 2:  # Treat only true internal faces shared by exactly two tetrahedra as stress-jump interfaces.
            first, second = adjacent  # Unpack the two elements sharing the current internal face.
            neighbours[first].append(second)  # Register the second element as a neighbour of the first.
            neighbours[second].append(first)  # Register the first element as a neighbour of the second.
    eta2: dict[int, float] = {}  # Store squared local indicators so Dorfler accumulation is directly additive.
    for element_id, connectivity in tets:  # Evaluate one local jump indicator for every tetrahedron.
        local_neighbours = neighbours[element_id]  # Retrieve all face-sharing neighbours of the current element.
        if not local_neighbours:  # Handle an isolated pathological element defensively.
            eta2[element_id] = 0.0  # Assign zero jump information when no internal comparison exists.
            continue  # Advance to the next element.
        mean_jump = sum(stress_distance_squared(stresses[element_id], stresses[other]) for other in local_neighbours) / len(local_neighbours)  # Average squared tensor jumps over the local face neighbourhood.
        eta2[element_id] = tet_volume(nodes, connectivity) * mean_jump  # Weight the local stress variation by physical element volume.
    return eta2  # Return the common nonnegative elementwise indicator used by both marking methods.


def point_in_box(point: tuple[float, float, float], box: dict[str, float]) -> bool:  # Test whether one element centroid lies inside one closed semantic atom box.
    x, y, z = point  # Unpack the physical point once for readable bound checks.
    return box["xmin"] - GEOM_TOL <= x <= box["xmax"] + GEOM_TOL and box["ymin"] - GEOM_TOL <= y <= box["ymax"] + GEOM_TOL and box["zmin"] - GEOM_TOL <= z <= box["zmax"] + GEOM_TOL  # Return the conjunction of all three coordinate-range tests.


def eligible_elements(method: str, nodes: dict[int, tuple[float, float, float]], tets: list[tuple[int, tuple[int, int, int, int]]], selected_boxes: list[dict[str, float]]) -> set[int]:  # Build either the global Dorfler pool or the LLM-gated semantic pool on the current adaptive mesh.
    if method == "global_dorfler":  # Keep every current tetrahedron eligible for the conventional global baseline.
        return {element_id for element_id, _connectivity in tets}  # Return the complete current mesh element identifier set.
    eligible: set[int] = set()  # Accumulate current elements whose centroids lie inside at least one frozen LLM atom.
    for element_id, connectivity in tets:  # Test every current tetrahedron against the frozen semantic support.
        centroid = tet_centroid(nodes, connectivity)  # Compute the mesh-independent physical centroid of the current element.
        if any(point_in_box(centroid, box) for box in selected_boxes):  # Keep the element when its centroid belongs to the LLM-selected atom union.
            eligible.add(element_id)  # Add the current element to the semantic Dorfler candidate pool.
    if not eligible:  # Reject a semantic gate that accidentally contains no current finite elements.
        raise RuntimeError("LLM semantic gate contains no current elements")  # Fail explicitly instead of silently reverting to the global method.
    return eligible  # Return the current semantic candidate pool.


def dorfler_mark(eta2: dict[int, float], eligible: set[int]) -> tuple[list[int], float]:  # Apply the same theta-bulk marking rule inside the supplied candidate pool.
    ranked = sorted(((element_id, eta2[element_id]) for element_id in eligible), key=lambda item: item[1], reverse=True)  # Rank eligible elements by the common nonnegative local indicator.
    total = sum(value for _element_id, value in ranked)  # Compute the total indicator mass inside the current candidate pool.
    if total <= 0.0:  # Require a nontrivial error-indicator distribution before bulk marking.
        raise RuntimeError("Dorfler candidate pool has zero total indicator mass")  # Fail explicitly when the estimator contains no actionable information.
    target = THETA * total  # Set the common Dorfler bulk target using the same theta for both methods.
    marked: list[int] = []  # Accumulate the smallest descending-prefix set that reaches the target mass.
    accumulated = 0.0  # Track the indicator mass captured by the marked prefix.
    for element_id, value in ranked:  # Traverse eligible elements from largest to smallest indicator.
        marked.append(element_id)  # Add the current highest remaining indicator element to the marked set.
        accumulated += value  # Add its indicator contribution to the captured mass.
        if accumulated >= target:  # Stop as soon as the standard Dorfler bulk inequality is satisfied.
            break  # Preserve the minimal descending-prefix construction for the chosen ranking.
    return marked, accumulated / total  # Return the marked element identifiers and the actually captured candidate-pool fraction.


def refinement_box(nodes: dict[int, tuple[float, float, float]], connectivity: tuple[int, int, int, int]) -> dict[str, float]:  # Convert one marked tetrahedron into a mesh-independent local sizing neighbourhood for the next remesh.
    vertices = [nodes[node_id] for node_id in connectivity]  # Resolve all four tetrahedral vertex coordinates.
    volume = tet_volume(nodes, connectivity)  # Compute the current physical element volume.
    characteristic_h = max((6.0 * volume) ** (1.0 / 3.0), MIN_LOCAL_H)  # Estimate one robust local element length scale while respecting the refinement floor.
    pad = BOX_PAD_FACTOR * characteristic_h  # Expand the exact tetrahedral bounds slightly to avoid fragile centroid-only refinement.
    vin = max(MIN_LOCAL_H, REFINE_FACTOR * characteristic_h)  # Request a smaller local target size without ever reaching the reference-mesh size.
    return {"xmin": min(vertex[0] for vertex in vertices) - pad, "xmax": max(vertex[0] for vertex in vertices) + pad, "ymin": min(vertex[1] for vertex in vertices) - pad, "ymax": max(vertex[1] for vertex in vertices) + pad, "zmin": min(vertex[2] for vertex in vertices) - pad, "zmax": max(vertex[2] for vertex in vertices) + pad, "vin": vin}  # Return the complete local Gmsh Box-field specification.


def solve_mesh(label: str, gmsh_bin: str, ccx_bin: str, refinement_boxes: list[dict[str, float]] | None = None, global_h: float | None = None) -> dict:  # Generate, solve, and postprocess one mesh state shared by either adaptive method.
    case_dir = CASES_DIR / label  # Allocate a deterministic directory for the current method and adaptive round.
    case_dir.mkdir(parents=True, exist_ok=True)  # Create the current case directory and missing parents idempotently.
    geo_path = case_dir / "model.geo"  # Store the generated Gmsh geometry and accumulated sizing history.
    msh_path = case_dir / "model.msh"  # Store the generated ASCII MSH2 tetrahedral mesh.
    geo_path.write_text(gmsh_geo_text(refinement_boxes=refinement_boxes, global_h=global_h), encoding="utf-8")  # Write either the global-size or adaptive-size Gmsh program.
    run_command([gmsh_bin, geo_path.name, "-3", "-format", "msh2", "-o", msh_path.name], case_dir, case_dir / "gmsh.log")  # Generate the current three-dimensional volume mesh.
    nodes, tets = parse_msh2(msh_path)  # Parse generated nodes and C3D4 connectivities into Python containers.
    sets = select_node_sets(nodes)  # Reconstruct the physical clamp and distributed eccentric load patch from coordinates.
    qoi_connectivity, qoi_weights = locate_point(nodes, tets, QOI_POINT)  # Locate the fixed material-space QoI point before solving.
    inp_path = case_dir / "job.inp"  # Store the complete CalculiX input deck under a stable job name.
    write_calculix_input(inp_path, nodes, tets, sets)  # Write the common structural model and common output requests.
    run_command([ccx_bin, "-i", "job"], case_dir, case_dir / "ccx.log")  # Solve the current linear-static model with CalculiX.
    dat_path = case_dir / "job.dat"  # Locate the ASCII displacement and stress output produced by print cards.
    displacements, stresses = parse_dat(dat_path, set(nodes), {element_id for element_id, _connectivity in tets})  # Recover the complete solved displacement and element-stress fields.
    qoi = interpolate_component(qoi_connectivity, qoi_weights, displacements, 2)  # Evaluate vertical displacement at the same interior material point on every mesh.
    return {"nodes": nodes, "tets": tets, "stresses": stresses, "qoi": qoi, "dof": 3 * len(nodes), "element_count": len(tets)}  # Return only the state required for marking and precision-resource analysis.


def relative_error(value: float, reference: float) -> float:  # Compute absolute relative QoI error against the fixed globally fine reference solution.
    return abs(value - reference) / max(abs(reference), 1.0e-16)  # Protect the division while preserving the dimensionless precision metric.


def run_method(method: str, reference_qoi: float, gmsh_bin: str, ccx_bin: str, selected_boxes: list[dict[str, float]]) -> tuple[list[dict], list[dict]]:  # Execute one complete adaptive history using either global or LLM-gated Dorfler marking.
    refinement_history: list[dict[str, float]] = []  # Accumulate all marked-element neighbourhoods so refinement persists across remeshing rounds.
    rows: list[dict] = []  # Accumulate one precision-resource record for every solved adaptive mesh state.
    mark_records: list[dict] = []  # Preserve compact per-round marking metadata for auditability.
    for round_index in range(AMR_ROUNDS + 1):  # Solve the shared coarse state and then four successive adaptive states.
        state = solve_mesh(f"{method}_r{round_index}", gmsh_bin, ccx_bin, refinement_boxes=refinement_history, global_h=None)  # Generate and solve the current adaptive mesh from the accumulated mark history.
        row = {"method": method, "round": round_index, "dof": state["dof"], "elements": state["element_count"], "qoi": state["qoi"], "qoi_rel_error": relative_error(state["qoi"], reference_qoi), "candidate_elements": 0, "marked_elements": 0, "captured_fraction": 0.0}  # Create the current precision-resource record before optional marking.
        if round_index < AMR_ROUNDS:  # Mark and create refinement neighbourhoods only when another adaptive state will be solved.
            eta2 = stress_jump_indicator(state["nodes"], state["tets"], state["stresses"])  # Compute the common elementwise stress-jump indicator on the current mesh.
            eligible = eligible_elements(method, state["nodes"], state["tets"], selected_boxes)  # Apply either no gate or the frozen LLM atom gate before Dorfler ranking.
            marked, captured_fraction = dorfler_mark(eta2, eligible)  # Apply exactly the same theta-bulk marking rule inside the current candidate pool.
            connectivity_map = {element_id: connectivity for element_id, connectivity in state["tets"]}  # Build direct connectivity lookup for marked elements.
            new_boxes = [refinement_box(state["nodes"], connectivity_map[element_id]) for element_id in marked]  # Convert each marked finite element into a persistent local sizing neighbourhood.
            refinement_history.extend(new_boxes)  # Append the current marks so all earlier refinement decisions remain active on the next remesh.
            row["candidate_elements"] = len(eligible)  # Record the current size of the Dorfler candidate pool.
            row["marked_elements"] = len(marked)  # Record how many finite elements satisfy the bulk criterion through descending-prefix marking.
            row["captured_fraction"] = captured_fraction  # Record the actual indicator fraction captured by the marked set.
            mark_records.append({"method": method, "round": round_index, "candidate_elements": len(eligible), "marked_elements": marked, "captured_fraction": captured_fraction})  # Preserve the actual marked element identifiers for transparent diagnosis.
        rows.append(row)  # Append the completed current adaptive-state record to the method history.
        print(f"[{method}] round={round_index} dof={row['dof']} error={row['qoi_rel_error']:.6e} candidates={row['candidate_elements']} marked={row['marked_elements']}", flush=True)  # Stream one concise progress line into the GitHub Actions console log.
    return rows, mark_records  # Return the complete precision-resource trajectory and detailed marking metadata.


def write_outputs(rows: list[dict], mark_records: list[dict], reference: dict, selection: dict) -> None:  # Persist compact comparison tables and a directly readable Markdown summary.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Create the compact result directory before writing any artifact.
    fields = ["method", "round", "dof", "elements", "qoi", "qoi_rel_error", "candidate_elements", "marked_elements", "captured_fraction"]  # Define a stable CSV schema for the two adaptive histories.
    with (RESULTS_DIR / "history.csv").open("w", encoding="utf-8", newline="") as handle:  # Open the adaptive-history table with portable CSV newline handling.
        writer = csv.DictWriter(handle, fieldnames=fields)  # Create a deterministic CSV writer using the documented field order.
        writer.writeheader()  # Emit the column header before data rows.
        for row in rows:  # Emit all global and semantic adaptive states in execution order.
            writer.writerow({field: row[field] for field in fields})  # Restrict each row to the stable documented schema.
    (RESULTS_DIR / "marks.json").write_text(json.dumps(mark_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # Persist exact per-round marked element identifiers for auditability.
    manifest = {"reference": reference, "theta": THETA, "amr_rounds": AMR_ROUNDS, "selected_atoms": selection["selected_atoms"], "semantic_description": selection["semantic_description"], "qoi_point_mm": QOI_POINT}  # Build one compact experiment manifest containing all comparison-defining settings.
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # Persist the manifest in human-readable UTF-8 JSON.
    global_rows = [row for row in rows if row["method"] == "global_dorfler"]  # Extract the conventional global Dorfler trajectory.
    semantic_rows = [row for row in rows if row["method"] == "llm_gated_dorfler"]  # Extract the frozen-semantic-gate Dorfler trajectory.
    markdown: list[str] = []  # Accumulate one concise GitHub-readable experiment report.
    markdown.append("# LLM semantic partition + Dörfler versus global Dörfler")  # Add the report title.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append(f"Reference QoI: `{reference['qoi']:.12e}` mm from a global `{REFERENCE_H}` mm mesh with `{reference['dof']}` DOF proxy.")  # State the fixed numerical reference used for both methods.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append(f"Frozen LLM support: `{', '.join(selection['selected_atoms'])}` ({selection['selected_atom_count']} of 16 atoms).")  # State exactly which semantic atoms were selected before solving.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("Both methods use the same stress-jump element indicator, the same `theta = 0.50`, the same descending-prefix Dörfler rule, the same local remeshing operator, and the same stopping round. The only difference is the candidate pool presented to Dörfler.")  # State the isolation principle of the experiment.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("| round | global DOF | global rel. error | LLM-gated DOF | LLM-gated rel. error |")  # Start the direct trajectory-comparison table.
    markdown.append("| ---: | ---: | ---: | ---: | ---: |")  # Add the Markdown alignment row.
    for global_row, semantic_row in zip(global_rows, semantic_rows):  # Compare states with the same adaptive-round index side by side.
        markdown.append(f"| {global_row['round']} | {global_row['dof']} | {global_row['qoi_rel_error']:.6e} | {semantic_row['dof']} | {semantic_row['qoi_rel_error']:.6e} |")  # Add one same-round precision-resource comparison row.
    markdown.append("")  # Add one Markdown spacer line.
    final_global = global_rows[-1]  # Read the final conventional global Dorfler state.
    final_semantic = semantic_rows[-1]  # Read the final LLM-gated Dorfler state.
    markdown.append(f"Final-round global Dörfler: error `{final_global['qoi_rel_error']:.6e}` at `{final_global['dof']}` DOF proxy.")  # State the final conventional baseline result.
    markdown.append(f"Final-round LLM-gated Dörfler: error `{final_semantic['qoi_rel_error']:.6e}` at `{final_semantic['dof']}` DOF proxy.")  # State the final semantic-gate result without overinterpreting it.
    markdown.append("")  # Add one Markdown spacer line.
    markdown.append("Interpretation rule: a useful semantic partition should move the QoI-error-versus-DOF trajectory down and/or left; the LLM itself never ranks finite elements and never sees the estimator values.")  # State the single intended causal interpretation of the comparison.
    (RESULTS_DIR / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")  # Persist the concise summary beside the machine-readable outputs.


def main() -> int:  # Execute the complete isolated semantic-gating benchmark from frozen selection through both adaptive histories.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the compact result directory exists before any possible early failure.
    CASES_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the transient case directory exists before meshing begins.
    selection = load_selection()  # Read the LLM-selected atom combination frozen before numerical evaluation.
    atoms = atom_boxes(selection)  # Reconstruct the complete fixed atom vocabulary from physical grid boundaries.
    selected_boxes = [atoms[atom_id] for atom_id in selection["selected_atoms"]]  # Convert the selected atom identifiers into their physical box supports.
    gmsh_bin = find_executable("gmsh", [])  # Resolve the Gmsh executable required for every adaptive remesh.
    ccx_bin = find_executable("ccx", ["ccx_2.23", "ccx_2.22", "ccx_2.21", "ccx_2.20"])  # Resolve the installed CalculiX executable across common Ubuntu package names.
    print(f"[setup] gmsh={gmsh_bin} ccx={ccx_bin} theta={THETA} selected_atoms={selection['selected_atoms']}", flush=True)  # Record the exact executable paths and frozen semantic support in the CI log.
    reference_state = solve_mesh("reference_global", gmsh_bin, ccx_bin, refinement_boxes=None, global_h=REFERENCE_H)  # Solve the single globally fine reference mesh before either adaptive method.
    reference = {"qoi": reference_state["qoi"], "dof": reference_state["dof"], "elements": reference_state["element_count"], "mesh_size": REFERENCE_H}  # Compact the reference to the information needed for precision evaluation.
    global_rows, global_marks = run_method("global_dorfler", reference["qoi"], gmsh_bin, ccx_bin, selected_boxes)  # Execute conventional full-domain Dorfler marking from the shared coarse state.
    semantic_rows, semantic_marks = run_method("llm_gated_dorfler", reference["qoi"], gmsh_bin, ccx_bin, selected_boxes)  # Execute the identical Dorfler algorithm after the frozen six-atom semantic gate.
    write_outputs(global_rows + semantic_rows, global_marks + semantic_marks, reference, selection)  # Persist the two trajectories and exact marking decisions in one compact result set.
    return 0  # Report successful completion to the workflow after all result files are written.


if __name__ == "__main__":  # Execute the benchmark only when this file is invoked as the program entry point.
    try:  # Wrap the outermost call only to emit one concise CI-visible failure reason.
        raise SystemExit(main())  # Run the complete benchmark and return its explicit shell status.
    except Exception as exc:  # Catch unexpected numerical or parsing failures at the command boundary.
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Emit the exact failure class and message into the persisted workflow console log.
        raise  # Re-raise the original exception so GitHub Actions records the run as failed.
