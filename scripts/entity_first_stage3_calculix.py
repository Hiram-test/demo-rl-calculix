"""Convert validated Gmsh entities to CalculiX decks and run isolated smoke solves."""  # Keep solver validation behind the Stage 2 geometry gate.
from __future__ import annotations  # Enable modern annotations on Actions Python.
from dataclasses import dataclass  # Store parsed Gmsh evidence and solver receipts.
from pathlib import Path  # Resolve downloaded Stage 2 artifacts and solver work directories.
from typing import Iterable  # Declare explicit node-set and element contracts.
import argparse  # Parse Stage 2 artifact and output locations.
import json  # Write deterministic solver receipts.
import math  # Reject non-finite result values.
import re  # Extract printed displacement and reaction values from CalculiX output.
import shutil  # Locate the installed CalculiX executable.
import subprocess  # Execute the mature CalculiX solver in batch mode.
@dataclass(frozen=True)  # Prevent parsed mesh data from changing during deck generation.
class MeshModel:  # Store one validated two-dimensional finite-element model.
    nodes: dict[int, tuple[float, float, float]]  # Map Gmsh node IDs to coordinates.
    triangles: tuple[tuple[int, int, int], ...]  # Store domain triangle connectivity.
    node_sets: dict[str, tuple[int, ...]]  # Store named boundary node sets from Physical Curves.
@dataclass(frozen=True)  # Preserve solver evidence without later mutation.
class SolverReceipt:  # Store one CalculiX smoke-test result.
    case_id: str  # Identify the exact entity fixture.
    input_file: str  # Record the generated CalculiX deck.
    return_code: int  # Record the solver process result.
    dat_file: str  # Record the printed result file.
    frd_file: str  # Record the field result file.
    maximum_absolute_displacement: float  # Record a finite non-zero displacement check.
    reaction_sum_y: float  # Record the vertical support reaction sum.
    applied_load_y: float  # Record the exact applied vertical load.
    equilibrium_relative_error: float  # Record reaction-to-load equilibrium error.
def parse_msh2(path: Path) -> MeshModel:  # Parse one Stage 2 MSH2 file into solver-ready entities.
    lines = path.read_text(encoding="utf-8").splitlines()  # Read the deterministic text mesh.
    physical_names: dict[tuple[int, int], str] = {}  # Map physical dimension and tag to stable names.
    nodes: dict[int, tuple[float, float, float]] = {}  # Collect Gmsh nodes by their original IDs.
    triangles: list[tuple[int, int, int]] = []  # Collect domain triangle connectivity.
    set_nodes: dict[str, set[int]] = {}  # Collect unique nodes from each named Physical Curve.
    index = 0  # Initialize the MSH2 section cursor.
    while index < len(lines):  # Visit every section in the mesh file.
        token = lines[index].strip()  # Read the current section marker.
        if token == "$PhysicalNames":  # Parse Physical Group names.
            count = int(lines[index + 1])  # Read the number of physical-name rows.
            for offset in range(count):  # Visit every named group.
                dimension_text, tag_text, quoted_name = lines[index + 2 + offset].split(maxsplit=2)  # Split dimension, tag, and quoted name.
                physical_names[(int(dimension_text), int(tag_text))] = quoted_name.strip('"')  # Preserve the exact name.
            index += count + 3  # Skip the section and end marker.
            continue  # Continue with the next MSH2 section.
        if token == "$Nodes":  # Parse all mesh nodes.
            count = int(lines[index + 1])  # Read the node count.
            for offset in range(count):  # Visit every node row.
                values = lines[index + 2 + offset].split()  # Split node ID and coordinates.
                nodes[int(values[0])] = (float(values[1]), float(values[2]), float(values[3]))  # Preserve the original Gmsh node ID.
            index += count + 3  # Skip the section and end marker.
            continue  # Continue with the next MSH2 section.
        if token == "$Elements":  # Parse boundary lines and domain triangles.
            count = int(lines[index + 1])  # Read the element count.
            for offset in range(count):  # Visit every element row.
                values = [int(value) for value in lines[index + 2 + offset].split()]  # Convert the complete row to integers.
                element_type = values[1]  # Read the Gmsh element type.
                number_of_tags = values[2]  # Read the number of element tags.
                tags = values[3 : 3 + number_of_tags]  # Extract physical and elementary tags.
                connectivity = values[3 + number_of_tags :]  # Extract node IDs.
                physical_tag = tags[0] if tags else 0  # Resolve the physical tag when present.
                if element_type == 1 and len(connectivity) == 2:  # Select two-node Physical Curve elements.
                    name = physical_names.get((1, physical_tag), f"UNNAMED_{physical_tag}")  # Resolve the boundary group name.
                    set_nodes.setdefault(name, set()).update(connectivity)  # Add both endpoints to the named node set.
                if element_type == 2 and len(connectivity) == 3:  # Select three-node Physical Surface elements.
                    name = physical_names.get((2, physical_tag), f"UNNAMED_{physical_tag}")  # Resolve the material-domain group name.
                    if name == "DOMAIN":  # Accept only the validated material domain.
                        triangles.append(tuple(connectivity))  # Preserve triangle connectivity.
            index += count + 3  # Skip the section and end marker.
            continue  # Continue with the next MSH2 section.
        index += 1  # Advance past an unrelated line.
    if not nodes:  # Require an actual finite-element node set.
        raise ValueError(f"{path} contains no nodes")  # Reject an empty or malformed mesh.
    if not triangles:  # Require an actual domain discretization.
        raise ValueError(f"{path} contains no DOMAIN triangles")  # Reject a boundary-only export.
    required_sets = {"FIXED_EDGE", "LOAD_EDGE"}  # Define solver-critical boundary sets.
    missing_sets = sorted(required_sets - set(set_nodes))  # Detect missing Physical Curve mappings.
    if missing_sets:  # Refuse to infer boundary conditions geometrically after meshing.
        raise ValueError(f"{path} is missing node sets {missing_sets}")  # Preserve the exact missing contract names.
    frozen_sets = {name: tuple(sorted(values)) for name, values in set_nodes.items()}  # Freeze all boundary node sets deterministically.
    return MeshModel(nodes, tuple(triangles), frozen_sets)  # Return solver-ready model evidence.
def format_id_rows(values: Iterable[int], width: int = 16) -> list[str]:  # Format CalculiX set rows without truncating long sets.
    items = [str(int(value)) for value in values]  # Normalize every identifier to decimal text.
    return [", ".join(items[start : start + width]) for start in range(0, len(items), width)]  # Wrap identifiers at a stable width.
def build_deck(model: MeshModel, *, total_load_y: float, thickness_mm: float = 1.0, youngs_modulus_mpa: float = 210000.0, poisson_ratio: float = 0.3) -> str:  # Generate one auditable plane-stress CalculiX deck.
    if thickness_mm <= 0.0:  # Require a physical positive thickness.
        raise ValueError("thickness_mm must be positive")  # Reject a degenerate section.
    fixed_nodes = model.node_sets["FIXED_EDGE"]  # Read the exact support boundary generated by Gmsh.
    load_nodes = model.node_sets["LOAD_EDGE"]  # Read the exact load boundary generated by Gmsh.
    if not fixed_nodes or not load_nodes:  # Require both boundary groups to contain nodes.
        raise ValueError("FIXED_EDGE and LOAD_EDGE must be non-empty")  # Reject an under-defined solver model.
    nodal_load = total_load_y / len(load_nodes)  # Distribute the exact total force uniformly over the named load edge.
    lines = ["*HEADING"]  # Start the CalculiX input deck.
    lines.append("Entity-first Gmsh to CalculiX smoke model")  # Record the deterministic workflow purpose.
    lines.append("*NODE")  # Begin the node table.
    for node_id in sorted(model.nodes):  # Emit nodes in stable original Gmsh order.
        x, y, z = model.nodes[node_id]  # Resolve the node coordinate.
        lines.append(f"{node_id}, {x:.12g}, {y:.12g}, {z:.12g}")  # Write one CalculiX node row.
    lines.append("*ELEMENT, TYPE=CPS3, ELSET=DOMAIN")  # Begin three-node plane-stress elements.
    for element_id, connectivity in enumerate(model.triangles, start=1):  # Allocate stable CalculiX element IDs.
        lines.append(f"{element_id}, {connectivity[0]}, {connectivity[1]}, {connectivity[2]}")  # Write one triangle row.
    for set_name in sorted(model.node_sets):  # Emit every named Gmsh boundary as a CalculiX node set.
        lines.append(f"*NSET, NSET={set_name}")  # Begin one named node set.
        lines.extend(format_id_rows(model.node_sets[set_name]))  # Write the complete set without geometric reclassification.
    lines.append("*MATERIAL, NAME=STEEL")  # Define the smoke-test elastic material.
    lines.append("*ELASTIC")  # Begin isotropic linear elasticity data.
    lines.append(f"{youngs_modulus_mpa:.12g}, {poisson_ratio:.12g}")  # Write Young's modulus and Poisson ratio.
    lines.append("*SOLID SECTION, ELSET=DOMAIN, MATERIAL=STEEL")  # Assign the material and plane-stress thickness.
    lines.append(f"{thickness_mm:.12g}")  # Write the physical section thickness.
    lines.append("*BOUNDARY")  # Apply deterministic support constraints before the step.
    lines.append("FIXED_EDGE, 1, 2, 0.0")  # Fix both in-plane displacement components on the named left edge.
    lines.append("*STEP")  # Begin the static smoke-test load step.
    lines.append("*STATIC")  # Select a linear static solution procedure.
    lines.append("*CLOAD")  # Begin concentrated nodal loads.
    for node_id in load_nodes:  # Apply the edge load through exact named boundary nodes.
        lines.append(f"{node_id}, 2, {nodal_load:.12g}")  # Apply the uniformly divided vertical force.
    lines.append("*NODE PRINT, NSET=LOAD_EDGE")  # Request load-edge displacements in the text result file.
    lines.append("U")  # Print displacement components.
    lines.append("*NODE PRINT, NSET=FIXED_EDGE, TOTALS=YES")  # Request summed support reactions for equilibrium validation.
    lines.append("RF")  # Print reaction-force components and totals.
    lines.append("*NODE FILE")  # Request nodal field output in the FRD file.
    lines.append("U")  # Save displacement fields.
    lines.append("*EL FILE, ELSET=DOMAIN")  # Request element field output in the FRD file.
    lines.append("S")  # Save stress fields.
    lines.append("*END STEP")  # Close the static smoke-test step.
    return "\n".join(lines) + "\n"  # Return deterministic CalculiX source text.
def extract_floats(text: str) -> tuple[float, ...]:  # Extract finite decimal and scientific-notation values from solver text.
    pattern = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"  # Match standard CalculiX numeric output.
    values = tuple(float(token) for token in re.findall(pattern, text))  # Convert every numeric token.
    return tuple(value for value in values if math.isfinite(value))  # Discard non-finite values defensively.
def parse_displacement_max(dat_text: str) -> float:  # Extract a conservative non-zero displacement magnitude from the .dat file.
    lower = dat_text.lower()  # Normalize section matching without changing numbers.
    marker = lower.find("displacements")  # Locate the first displacement print section.
    if marker < 0:  # Require explicit requested displacement output.
        raise ValueError("CalculiX .dat file contains no displacement section")  # Reject an output-free solve.
    section = dat_text[marker : marker + 20000]  # Limit parsing to the displacement neighborhood.
    values = extract_floats(section)  # Parse all finite values in that section.
    magnitudes = [abs(value) for value in values if abs(value) < 1.0e12]  # Exclude impossible overflow values.
    if not magnitudes:  # Require at least one numeric displacement value.
        raise ValueError("CalculiX displacement section contains no finite values")  # Reject a malformed result.
    return max(magnitudes)  # Return the largest printed absolute value.
def parse_reaction_sum_y(dat_text: str) -> float:  # Extract the y reaction total from the printed TOTALS row.
    lower = dat_text.lower()  # Normalize section matching without changing numbers.
    marker = lower.find("forces")  # Locate the first reaction-force print section.
    if marker < 0:  # Require explicit requested reaction output.
        raise ValueError("CalculiX .dat file contains no reaction-force section")  # Reject a solve without equilibrium evidence.
    section = dat_text[marker : marker + 20000]  # Limit parsing to the reaction neighborhood.
    total_lines = [line for line in section.splitlines() if "total" in line.lower()]  # Find CalculiX TOTALS rows.
    if not total_lines:  # Require a solver-computed support total.
        raise ValueError("CalculiX reaction section contains no totals row")  # Reject node-by-node output without equilibrium evidence.
    values = extract_floats(total_lines[-1])  # Parse the final totals row.
    if len(values) < 2:  # Require at least x and y reaction components.
        raise ValueError("CalculiX reaction totals row is incomplete")  # Reject ambiguous equilibrium evidence.
    return values[-2] if len(values) >= 3 else values[1]  # Return the y component while tolerating an optional leading label number.
def find_ccx() -> str:  # Resolve the mature CalculiX executable across standard package names.
    for candidate in ("ccx", "ccx_2.21", "ccx_2.20", "ccx_2.19"):  # Check common Ubuntu and upstream command names.
        executable = shutil.which(candidate)  # Resolve the current candidate on PATH.
        if executable is not None:  # Accept the first installed solver binary.
            return executable  # Return the exact executable path.
    raise RuntimeError("CalculiX ccx executable is unavailable")  # Stop before claiming a solver test.
def run_case(case_id: str, msh_path: Path, output_root: Path, total_load_y: float = -1000.0) -> SolverReceipt:  # Generate and solve one validated model.
    model = parse_msh2(msh_path)  # Read exact Gmsh entities and Physical Group node sets.
    work_dir = output_root / case_id  # Allocate an isolated solver work directory.
    work_dir.mkdir(parents=True, exist_ok=True)  # Create the work directory without touching Stage 2 evidence.
    job_name = "smoke"  # Use a stable CalculiX job stem.
    input_path = work_dir / f"{job_name}.inp"  # Allocate the solver input path.
    input_path.write_text(build_deck(model, total_load_y=total_load_y), encoding="utf-8")  # Persist the deterministic solver deck.
    executable = find_ccx()  # Resolve the installed CalculiX binary.
    completed = subprocess.run((executable, "-i", job_name), cwd=work_dir, text=True, capture_output=True, check=False)  # Execute CalculiX in batch mode.
    (work_dir / "ccx.stdout.log").write_text(completed.stdout, encoding="utf-8")  # Preserve the complete standard output.
    (work_dir / "ccx.stderr.log").write_text(completed.stderr, encoding="utf-8")  # Preserve the complete standard error.
    if completed.returncode != 0:  # Reject any solver process failure.
        raise RuntimeError(f"CalculiX failed for {case_id} with rc={completed.returncode}:\n{completed.stdout}\n{completed.stderr}")  # Return exact solver evidence.
    dat_path = work_dir / f"{job_name}.dat"  # Resolve the printed result file.
    frd_path = work_dir / f"{job_name}.frd"  # Resolve the field result file.
    if not dat_path.exists() or dat_path.stat().st_size == 0:  # Require printed QoI evidence.
        raise RuntimeError(f"CalculiX created no non-empty .dat file for {case_id}")  # Reject an output-free process success.
    if not frd_path.exists() or frd_path.stat().st_size == 0:  # Require field evidence.
        raise RuntimeError(f"CalculiX created no non-empty .frd file for {case_id}")  # Reject a result without fields.
    dat_text = dat_path.read_text(encoding="utf-8", errors="replace")  # Read the complete text result.
    maximum_displacement = parse_displacement_max(dat_text)  # Verify finite non-zero structural response.
    reaction_sum_y = parse_reaction_sum_y(dat_text)  # Verify support equilibrium.
    equilibrium_error = abs(reaction_sum_y + total_load_y) / max(1.0, abs(total_load_y))  # Compare support reaction with the opposite applied load.
    if maximum_displacement <= 0.0 or not math.isfinite(maximum_displacement):  # Require a real finite response.
        raise RuntimeError(f"invalid displacement result for {case_id}: {maximum_displacement}")  # Reject a zero or non-finite solve.
    if equilibrium_error > 1.0e-4:  # Enforce static vertical-force equilibrium.
        raise RuntimeError(f"equilibrium error for {case_id} is {equilibrium_error:.6e}")  # Reject an incorrect deck or parser.
    return SolverReceipt(case_id, input_path.name, completed.returncode, dat_path.name, frd_path.name, maximum_displacement, reaction_sum_y, total_load_y, equilibrium_error)  # Freeze complete solver evidence.
def generator_self_test() -> None:  # Verify deck generation before executing CalculiX.
    model = MeshModel({1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0)}, ((1, 2, 3),), {"FIXED_EDGE": (1, 3), "LOAD_EDGE": (2,), "TOP_EDGE": (3,), "BOTTOM_EDGE": (1, 2)})  # Define the smallest valid triangular model.
    deck = build_deck(model, total_load_y=-10.0)  # Generate a deterministic smoke deck.
    assert "*ELEMENT, TYPE=CPS3, ELSET=DOMAIN" in deck  # Require the intended plane-stress element family.
    assert "FIXED_EDGE, 1, 2, 0.0" in deck  # Require exact support-set usage.
    assert "2, 2, -10" in deck  # Require total load distribution through the named load set.
    assert "*NODE PRINT, NSET=FIXED_EDGE, TOTALS=YES" in deck  # Require equilibrium evidence.
def run_suite(stage2_root: Path, output_root: Path) -> dict[str, object]:  # Solve the three exact Stage 2 entity families.
    generator_self_test()  # Prove deck generation before invoking CalculiX.
    cases = ("bearing_plate", "circular_opening", "three_openings")  # Freeze the Stage 3 smoke-test case order.
    receipts: dict[str, object] = {}  # Collect machine-readable solver evidence.
    for case_id in cases:  # Solve each entity independently.
        msh_path = stage2_root / case_id / "coarse.msh"  # Use the validated coarse mesh from the frozen BREP.
        if not msh_path.exists():  # Require the exact Stage 2 evidence.
            raise FileNotFoundError(msh_path)  # Refuse to regenerate or infer a missing mesh silently.
        receipt = run_case(case_id, msh_path, output_root)  # Generate the CalculiX deck and execute the solver.
        receipts[case_id] = receipt.__dict__  # Serialize immutable case evidence.
    suite = {"schema_version": "entity-first-stage3-calculix/1.0", "cases": receipts, "all_valid": True}  # Build the solver-suite receipt.
    (output_root / "stage3_receipt.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")  # Persist complete Stage 3 evidence.
    return suite  # Return the validated solver suite.
def main() -> int:  # Execute the generator self-test or the complete CalculiX smoke suite.
    parser = argparse.ArgumentParser(description="Run CalculiX on exact Stage 2 Gmsh entities")  # Describe the isolated solver gate.
    parser.add_argument("--generator-self-test", action="store_true")  # Allow code-only verification without CalculiX.
    parser.add_argument("--stage2-root")  # Accept the downloaded Stage 2 artifact root.
    parser.add_argument("--output-dir")  # Accept an isolated solver evidence directory.
    args = parser.parse_args()  # Parse arguments once.
    if args.generator_self_test:  # Run only the deck generator regression.
        generator_self_test()  # Execute deterministic deck assertions.
        print("entity-first Stage 3 generator self-test passed")  # Emit an explicit success marker.
        return 0  # Return success without starting CalculiX.
    if not args.stage2_root or not args.output_dir:  # Require both roots in solver mode.
        parser.error("--stage2-root and --output-dir are required")  # Reject an ambiguous solver invocation.
    suite = run_suite(Path(args.stage2_root).resolve(), Path(args.output_dir).resolve())  # Run all exact model families.
    print(json.dumps(suite, ensure_ascii=False, indent=2))  # Echo machine-readable evidence to the CI log.
    return 0  # Return success only after all three CalculiX solves pass.
if __name__ == "__main__":  # Run only when invoked directly.
    raise SystemExit(main())  # Propagate the process status to CI.