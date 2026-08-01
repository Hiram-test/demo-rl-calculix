"""Audit legacy V4 meshes before rebuilding the Gmsh/CalculiX workflow."""  # Keep Stage 1 independent from meshing, solving, and PSO.
from __future__ import annotations  # Enable modern type annotations on Actions Python.
from collections import Counter, defaultdict  # Count element edges and build boundary graphs.
from dataclasses import asdict, dataclass  # Store and serialize immutable audit evidence.
from math import acos, degrees, pi, sqrt  # Compute exact entity areas and triangle quality.
from pathlib import Path  # Resolve frozen artifact paths safely.
from typing import Iterable, Sequence  # Declare explicit mesh input contracts.
import argparse  # Parse self-test and artifact-audit modes.
import json  # Read PSO records and write the audit receipt.
import sys  # Return a failing status for an invalid mesh suite.
import numpy as np  # Read frozen raw NPZ meshes from the Actions artifact.
Point = tuple[float, float]  # Represent one planar node.
Triangle = tuple[int, int, int]  # Represent one three-node triangle.
Edge = tuple[int, int]  # Represent one undirected element edge.
@dataclass(frozen=True)  # Prevent audit evidence from changing after validation.
class AuditResult:  # Preserve topology, geometry, and quality evidence.
    ok: bool  # State whether every hard check passed.
    issues: tuple[str, ...]  # Record every failed hard check.
    boundary_components: int  # Record the number of free-edge loops.
    boundary_loop_areas: tuple[float, ...]  # Record outer and inner loop areas.
    open_boundary_vertices: tuple[int, ...]  # Record non-closed or non-manifold boundary vertices.
    mesh_area: float  # Record integrated triangle area.
    expected_area: float  # Record fixed entity area.
    relative_area_error: float  # Record mesh-to-entity area drift.
    minimum_angle_deg: float  # Record the worst element angle.
    maximum_radius_ratio: float  # Record the worst triangle distortion.
def canonical_edge(a: int, b: int) -> Edge:  # Normalize opposite edge orientations.
    return (a, b) if a < b else (b, a)  # Store the lower node index first.
def triangle_area(p0: Point, p1: Point, p2: Point) -> float:  # Compute one element area.
    cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])  # Evaluate the planar cross product.
    return abs(cross) * 0.5  # Convert the parallelogram area to a triangle area.
def edge_length(p0: Point, p1: Point) -> float:  # Compute one Euclidean edge length.
    return sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)  # Return the Euclidean norm.
def triangle_quality(p0: Point, p1: Point, p2: Point) -> tuple[float, float]:  # Compute minimum angle and radius ratio.
    a = edge_length(p1, p2)  # Measure the side opposite the first vertex.
    b = edge_length(p2, p0)  # Measure the side opposite the second vertex.
    c = edge_length(p0, p1)  # Measure the side opposite the third vertex.
    area = triangle_area(p0, p1, p2)  # Compute element area once.
    if area <= 0.0 or min(a, b, c) <= 0.0:  # Detect duplicate or collinear nodes.
        return 0.0, float("inf")  # Mark a degenerate element as invalid.
    angles: list[float] = []  # Collect all three interior angles.
    for opposite, side_1, side_2 in ((a, b, c), (b, c, a), (c, a, b)):  # Visit every vertex.
        cosine = (side_1 * side_1 + side_2 * side_2 - opposite * opposite) / (2.0 * side_1 * side_2)  # Apply the cosine rule.
        angles.append(degrees(acos(max(-1.0, min(1.0, cosine)))))  # Bound roundoff and convert to degrees.
    semiperimeter = 0.5 * (a + b + c)  # Compute the semiperimeter.
    inradius = area / semiperimeter  # Compute the inscribed-circle radius.
    circumradius = a * b * c / (4.0 * area)  # Compute the circumscribed-circle radius.
    return min(angles), circumradius / (2.0 * inradius)  # Normalize equilateral quality to one.
def boundary_graph(triangles: Sequence[Triangle]) -> dict[int, list[int]]:  # Build the graph of free element edges.
    counts: Counter[Edge] = Counter()  # Count ownership of every undirected edge.
    for n0, n1, n2 in triangles:  # Visit every triangular element.
        counts[canonical_edge(n0, n1)] += 1  # Count the first edge.
        counts[canonical_edge(n1, n2)] += 1  # Count the second edge.
        counts[canonical_edge(n2, n0)] += 1  # Count the third edge.
    adjacency: dict[int, list[int]] = defaultdict(list)  # Store free-edge neighbors.
    for (node_a, node_b), count in counts.items():  # Inspect every unique element edge.
        if count == 1:  # Select a free boundary edge.
            adjacency[node_a].append(node_b)  # Connect the first endpoint.
            adjacency[node_b].append(node_a)  # Connect the second endpoint.
    return dict(adjacency)  # Return a stable plain dictionary.
def connected_components(adjacency: dict[int, list[int]]) -> tuple[tuple[int, ...], ...]:  # Enumerate boundary graph components.
    seen: set[int] = set()  # Track assigned boundary vertices.
    components: list[tuple[int, ...]] = []  # Collect deterministic component vertex sets.
    for start in sorted(adjacency):  # Traverse vertices in stable order.
        if start in seen:  # Skip an already assigned vertex.
            continue  # Continue to the next seed.
        stack = [start]  # Initialize depth-first traversal.
        seen.add(start)  # Mark the seed before traversal.
        nodes: list[int] = []  # Collect vertices in this component.
        while stack:  # Continue until the component is exhausted.
            current = stack.pop()  # Remove one pending vertex.
            nodes.append(current)  # Preserve the current vertex.
            for neighbor in adjacency[current]:  # Visit every boundary neighbor.
                if neighbor not in seen:  # Detect a new component vertex.
                    seen.add(neighbor)  # Mark it before scheduling.
                    stack.append(neighbor)  # Schedule it for traversal.
        components.append(tuple(sorted(nodes)))  # Freeze this component.
    return tuple(components)  # Return all boundary components.
def ordered_cycle(component: Sequence[int], adjacency: dict[int, list[int]]) -> tuple[int, ...]:  # Order one degree-two component as a polygon.
    start = min(component)  # Select a deterministic start vertex.
    ordered = [start]  # Seed the polygon traversal.
    previous: int | None = None  # Record the vertex behind the traversal head.
    current = start  # Initialize the traversal head.
    for _ in range(len(component) - 1):  # Visit every remaining component vertex.
        neighbors = sorted(adjacency[current])  # Read the two cycle neighbors.
        next_node = neighbors[0] if neighbors[0] != previous else neighbors[1]  # Continue forward without reversing.
        if next_node == start:  # Detect premature closure.
            break  # Stop and expose the incomplete cycle through its area.
        ordered.append(next_node)  # Append the next polygon vertex.
        previous, current = current, next_node  # Advance traversal state.
    return tuple(ordered)  # Freeze the ordered cycle.
def polygon_area(points: Sequence[Point], cycle: Sequence[int]) -> float:  # Compute one boundary-loop area.
    doubled = 0.0  # Initialize the shoelace sum.
    for index, node_index in enumerate(cycle):  # Visit every ordered polygon vertex.
        next_index = cycle[(index + 1) % len(cycle)]  # Wrap the last vertex to the first.
        x0, y0 = points[node_index]  # Resolve the current coordinate.
        x1, y1 = points[next_index]  # Resolve the next coordinate.
        doubled += x0 * y1 - y0 * x1  # Add one shoelace cross term.
    return abs(doubled) * 0.5  # Convert signed doubled area to physical area.
def audit_mesh(points: Sequence[Point], triangles: Iterable[Triangle], *, expected_components: int, expected_area: float, expected_hole_areas: Sequence[float], area_tolerance: float = 0.01, hole_tolerance: float = 0.02, angle_limit: float = 10.0, radius_ratio_limit: float = 10.0) -> AuditResult:  # Define the pre-solve hard gate.
    elements = [tuple(map(int, triangle)) for triangle in triangles]  # Normalize connectivity once.
    issues: list[str] = []  # Collect all failures without hiding later evidence.
    mesh_area = 0.0  # Initialize integrated finite-element area.
    minimum_angle = float("inf")  # Initialize the worst-angle metric.
    maximum_ratio = 0.0  # Initialize the worst-distortion metric.
    for element_index, triangle in enumerate(elements):  # Inspect every element.
        if len(set(triangle)) != 3 or min(triangle) < 0 or max(triangle) >= len(points):  # Detect invalid connectivity.
            issues.append(f"element {element_index} has invalid connectivity")  # Preserve the failing element index.
            continue  # Skip unavailable geometry.
        p0, p1, p2 = (points[index] for index in triangle)  # Resolve element coordinates.
        mesh_area += triangle_area(p0, p1, p2)  # Accumulate physical area.
        angle, ratio = triangle_quality(p0, p1, p2)  # Compute element quality.
        minimum_angle = min(minimum_angle, angle)  # Update the global minimum angle.
        maximum_ratio = max(maximum_ratio, ratio)  # Update the global maximum distortion.
    adjacency = boundary_graph(elements)  # Build the free-edge topology.
    open_vertices = tuple(sorted(node for node, neighbors in adjacency.items() if len(neighbors) != 2))  # Detect open and non-manifold boundary nodes.
    components = connected_components(adjacency)  # Enumerate outer and inner boundary graphs.
    loop_areas = tuple(sorted((polygon_area(points, ordered_cycle(component, adjacency)) for component in components), reverse=True))  # Measure every boundary loop.
    if open_vertices:  # Reject non-cycle boundary graphs.
        issues.append(f"boundary is open or non-manifold at {open_vertices}")  # Preserve offending vertices.
    if len(components) != expected_components:  # Enforce entity topology.
        issues.append(f"expected {expected_components} boundary components but found {len(components)}")  # Report fake or missing holes.
    observed_holes = tuple(sorted(loop_areas[1:])) if loop_areas else tuple()  # Treat the largest loop as the outer entity boundary.
    expected_holes = tuple(sorted(float(value) for value in expected_hole_areas))  # Normalize hole contracts.
    if len(observed_holes) != len(expected_holes):  # Enforce the number of hole entities.
        issues.append(f"expected {len(expected_holes)} hole areas but found {len(observed_holes)}")  # Report hole-count mismatch.
    else:  # Compare corresponding holes only when counts match.
        for hole_index, (observed, expected) in enumerate(zip(observed_holes, expected_holes)):  # Inspect every hole independently.
            error = abs(observed - expected) / expected  # Compute normalized hole-area drift.
            if error > hole_tolerance:  # Enforce fixed hole geometry.
                issues.append(f"hole {hole_index} relative area error {error:.6f} exceeds {hole_tolerance:.6f}")  # Report exact drift.
    area_error = abs(mesh_area - expected_area) / expected_area  # Compute normalized domain-area drift.
    if area_error > area_tolerance:  # Enforce fixed model area.
        issues.append(f"relative area error {area_error:.6f} exceeds {area_tolerance:.6f}")  # Report exact material drift.
    if minimum_angle < angle_limit:  # Enforce the minimum-angle hard limit.
        issues.append(f"minimum angle {minimum_angle:.6f} deg is below {angle_limit:.6f} deg")  # Report exact quality failure.
    if maximum_ratio > radius_ratio_limit:  # Enforce the radius-ratio hard limit.
        issues.append(f"maximum radius ratio {maximum_ratio:.6f} exceeds {radius_ratio_limit:.6f}")  # Report exact distortion failure.
    return AuditResult(not issues, tuple(issues), len(components), loop_areas, open_vertices, mesh_area, expected_area, area_error, minimum_angle, maximum_ratio)  # Freeze all evidence.
CASE_CONTRACTS = {  # Freeze intended entity facts for triangular legacy cases.
    "bearing_load_introduction": {"components": 1, "area": 1000.0 * 100.0, "holes": ()},  # Define one rectangular surface with no hole.
    "web_circular_opening": {"components": 2, "area": 240.0 * 240.0 - pi * 20.0**2, "holes": (pi * 20.0**2,)},  # Define one exact circular opening.
    "diaphragm_multi_opening_budget": {"components": 4, "area": 600.0 * 260.0 - pi * (24.0**2 + 42.0**2 + 30.0**2), "holes": (pi * 24.0**2, pi * 42.0**2, pi * 30.0**2)},  # Define three exact circular openings.
}  # Complete the frozen entity contracts.
def self_test() -> None:  # Verify the audit code before it judges production artifacts.
    valid_points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))  # Define a complete unit square.
    valid = audit_mesh(valid_points, ((0, 1, 2), (0, 2, 3)), expected_components=1, expected_area=1.0, expected_hole_areas=())  # Audit the valid square.
    assert valid.ok, valid.issues  # Require the complete entity to pass.
    grid_points = tuple((float(x), float(y)) for y in range(4) for x in range(4))  # Define a four-by-four node lattice.
    grid_triangles: list[Triangle] = []  # Collect cells around an intentional interior deletion.
    for y in range(3):  # Visit each cell row.
        for x in range(3):  # Visit each cell column.
            if x == 1 and y == 1:  # Select the central cell.
                continue  # Delete it to create a fake internal hole.
            n00 = y * 4 + x  # Resolve the lower-left node.
            n10 = n00 + 1  # Resolve the lower-right node.
            n01 = n00 + 4  # Resolve the upper-left node.
            n11 = n01 + 1  # Resolve the upper-right node.
            grid_triangles.extend(((n00, n10, n11), (n00, n11, n01)))  # Mesh the current square.
    fake_hole = audit_mesh(grid_points, grid_triangles, expected_components=1, expected_area=9.0, expected_hole_areas=())  # Audit against an intact rectangle.
    assert not fake_hole.ok and fake_hole.boundary_components == 2, fake_hole  # Require detection of the artificial cavity.
    ring_points = ((0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0))  # Define an outer square and inner square.
    ring_triangles = ((0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))  # Mesh the square ring.
    wrong_hole = audit_mesh(ring_points, ring_triangles, expected_components=2, expected_area=8.0, expected_hole_areas=(0.5,))  # Declare an incorrect hole entity.
    assert not wrong_hole.ok and any("hole 0 relative area error" in issue for issue in wrong_hole.issues), wrong_hole  # Require hole-size drift detection.
    sliver = audit_mesh(((0.0, 0.0), (1.0, 0.0), (0.999999, 0.000001)), ((0, 1, 2),), expected_components=1, expected_area=0.0000005, expected_hole_areas=())  # Audit a nearly collinear element.
    assert not sliver.ok and any("minimum angle" in issue for issue in sliver.issues), sliver  # Require quality-failure detection.
def load_best_mesh(artifact_root: Path, case_id: str) -> tuple[np.ndarray, np.ndarray, Path]:  # Resolve one best-particle raw mesh.
    run_root = artifact_root / "runs" / case_id  # Resolve the case run directory.
    record = json.loads((run_root / "artifacts" / "region_pso_1.json").read_text(encoding="utf-8"))  # Parse the optimizer record.
    raw_path = run_root / record["best"]["raw_file"]  # Resolve the exact best-particle NPZ.
    with np.load(raw_path, allow_pickle=False) as raw:  # Open immutable raw evidence.
        nodes = np.asarray(raw["nodes"], dtype=float)  # Copy node coordinates.
        elements = np.asarray(raw["elements"], dtype=int)  # Copy element connectivity.
    return nodes, elements, raw_path  # Return data and provenance.
def audit_artifact(artifact_root: Path, report_path: Path) -> bool:  # Audit the frozen V4 best meshes.
    case_reports: dict[str, dict[str, object]] = {}  # Collect case-level evidence.
    all_valid = True  # Initialize suite status optimistically.
    for case_id, contract in CASE_CONTRACTS.items():  # Audit every configured case independently.
        nodes, elements, raw_path = load_best_mesh(artifact_root, case_id)  # Load the exact frozen mesh.
        if elements.ndim != 2 or elements.shape[1] != 3:  # Restrict Stage 1 to triangular legacy cases.
            case_reports[case_id] = {"ok": False, "issues": ["mesh is not three-node triangular"], "raw_file": str(raw_path)}  # Preserve unsupported connectivity.
            all_valid = False  # Mark the suite invalid.
            continue  # Continue to the next case.
        points = tuple((float(x), float(y)) for x, y in nodes[:, :2])  # Convert node coordinates to the pure audit API.
        triangles = tuple(tuple(int(value) for value in row) for row in elements)  # Convert connectivity to immutable triangles.
        result = audit_mesh(points, triangles, expected_components=int(contract["components"]), expected_area=float(contract["area"]), expected_hole_areas=tuple(float(value) for value in contract["holes"]))  # Apply hard topology, geometry, and quality checks.
        payload = asdict(result)  # Serialize the immutable result.
        payload["raw_file"] = str(raw_path.relative_to(artifact_root))  # Preserve artifact-relative provenance.
        case_reports[case_id] = payload  # Store complete case evidence.
        all_valid = all_valid and result.ok  # Update the suite result.
    receipt = {"schema_version": "entity-first-mesh-audit/1.0", "cases": case_reports, "all_valid": all_valid}  # Build the Stage 1 receipt.
    report_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the report directory exists.
    report_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")  # Persist deterministic evidence.
    print(json.dumps(receipt, ensure_ascii=False, indent=2))  # Echo evidence to the Actions log.
    return all_valid  # Return the suite validation state.
def main() -> int:  # Execute self-tests or the frozen-artifact audit.
    parser = argparse.ArgumentParser(description="Entity-first Stage 1 mesh audit")  # Describe the isolated gate.
    parser.add_argument("--self-test", action="store_true")  # Allow dependency-light verification of the audit code.
    parser.add_argument("--artifact-root")  # Accept a downloaded V4 artifact directory.
    parser.add_argument("--report")  # Accept a machine-readable report path.
    args = parser.parse_args()  # Parse arguments once.
    if args.self_test:  # Run code verification before artifact diagnosis.
        self_test()  # Execute deterministic regression checks.
        print("entity-first Stage 1 self-tests passed")  # Emit an explicit success marker.
        return 0  # Return success for verified audit code.
    if not args.artifact_root or not args.report:  # Require both artifact arguments in diagnostic mode.
        parser.error("--artifact-root and --report are required without --self-test")  # Reject an incomplete diagnostic invocation.
    return 0 if audit_artifact(Path(args.artifact_root).resolve(), Path(args.report).resolve()) else 1  # Propagate the hard-gate result.
if __name__ == "__main__":  # Run only when invoked directly.
    raise SystemExit(main())  # Propagate the process status to CI.