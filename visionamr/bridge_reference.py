# Moderate-cost geometry-aware reference solution for the three-dimensional bridge pier-cap benchmark.  # Module purpose.
from __future__ import annotations  # Enable postponed annotations without importing heavy solver types early.
from dataclasses import asdict  # Serialize the repository Reference record consistently.
from pathlib import Path  # Manage reusable reference artifacts in method-specific work directories.
import json  # Persist the independent reference and its declared refinement profile.
import math  # Evaluate scalar geometric distances efficiently inside the Gmsh size callback.
from .experiment import FemRunner, Reference  # Reuse honest CalculiX execution and reference metadata contracts.
from .geometry import Problem  # Read the immutable bridge-component geometry and material definition.
from .mesher import generate_mesh  # Generate the independently prescribed graded reference mesh.


_PROFILE_VERSION = "bridge_pier_cap_reference_v2"  # Freeze the benchmark reference field for reproducibility.


def _point_segment_distance(x: float, y: float, z: float, start: tuple[float, float, float], end: tuple[float, float, float]) -> float:  # Compute a scalar three-dimensional point-to-segment distance.
    ax, ay, az = start  # Read the segment start coordinates.
    bx, by, bz = end  # Read the segment end coordinates.
    vx = bx - ax  # Compute the segment x direction.
    vy = by - ay  # Compute the segment y direction.
    vz = bz - az  # Compute the segment z direction.
    wx = x - ax  # Compute the point offset in x.
    wy = y - ay  # Compute the point offset in y.
    wz = z - az  # Compute the point offset in z.
    denominator = max(vx * vx + vy * vy + vz * vz, 1.0e-30)  # Protect the projection against a degenerate segment.
    fraction = min(max((wx * vx + wy * vy + wz * vz) / denominator, 0.0), 1.0)  # Clamp the nearest-point parameter to the segment.
    dx = x - (ax + fraction * vx)  # Compute the nearest-point x residual.
    dy = y - (ay + fraction * vy)  # Compute the nearest-point y residual.
    dz = z - (az + fraction * vz)  # Compute the nearest-point z residual.
    return math.sqrt(dx * dx + dy * dy + dz * dz)  # Return the Euclidean distance.


def _graded_size(background: float, floor: float, distance: float, radius: float, exponent: float) -> float:  # Convert feature distance into a bounded smooth target size.
    normalized = min(max(float(distance) / max(float(radius), 1.0e-12), 0.0), 1.0)  # Normalize distance inside the declared influence width.
    return float(floor + (background - floor) * normalized ** float(exponent))  # Interpolate monotonically from feature floor to background.


def bridge_reference_profile(problem: Problem) -> dict[str, float | str]:  # Publish the complete independent reference-field contract.
    return {  # Return a JSON-compatible immutable profile description.
        "version": _PROFILE_VERSION,  # Preserve the exact field definition version.
        "background_h": float(problem.h_ref),  # Preserve the global background size.
        "bearing_floor_h": 22.0,  # Resolve pressure-patch edge concentrations more finely than every method mesh.
        "bearing_radius": 135.0,  # Restrict bearing-edge refinement to a local top-surface band.
        "duct_floor_h": 28.0,  # Resolve each prestressing-duct wall without refining its empty core.
        "duct_radius_band": 105.0,  # Restrict duct refinement to a narrow annular transfer zone.
        "junction_floor_h": 24.0,  # Resolve the cap-column re-entrant load-transfer perimeter.
        "junction_radius": 165.0,  # Restrict junction refinement to the local discontinuity zone.
        "base_floor_h": 36.0,  # Resolve fixed-base reaction gradients at moderate cost.
        "base_radius": 110.0,  # Restrict base refinement to the support perimeter.
        "gradation_exponent": 0.72,  # Use one smooth deterministic grading exponent for all feature classes.
    }  # Finish the reproducible profile.


def bridge_reference_size_fn(problem: Problem):  # Build a scalar geometry-aware Gmsh size callback for the bridge component.
    if problem.name != "bridge_pier_cap" or int(problem.dim) != 3:  # Restrict the specialized profile to its declared benchmark family.
        raise ValueError("bridge_reference_size_fn requires the 3-D bridge_pier_cap problem")  # Fail instead of silently applying wrong geometry.
    params = dict(problem.params)  # Copy the exact parameterized geometry definition.
    profile = bridge_reference_profile(problem)  # Read the frozen numerical refinement contract.
    length = float(params["length"])  # Read the cap-beam longitudinal length.
    width = float(params["width"])  # Read the cap-beam transverse width.
    cap_height = float(params["cap_height"])  # Read the cap-beam depth.
    column_height = float(params["column_height"])  # Read the modeled pier-column height.
    column_width = float(params["column_width"])  # Read the column longitudinal width.
    column_depth = float(params["column_depth"])  # Read the column transverse depth.
    bearing_size = tuple(float(value) for value in params["bearing_size"])  # Read the exact pressure-patch dimensions.
    bearing_centres = tuple(float(value) for value in params["bearing_centres"])  # Read both bearing centre coordinates.
    duct_radius = float(params["duct_radius"])  # Read the prestressing-duct radius.
    duct_y_fractions = tuple(float(value) for value in params["duct_y_fractions"])  # Read both transverse duct fractions.
    duct_z_fraction = float(params["duct_z_fraction"])  # Read the duct elevation fraction inside the cap.
    top_z = column_height + cap_height  # Compute the cap top elevation.
    cap_bottom_z = column_height  # Compute the cap soffit and column-cap interface elevation.
    centre_y = 0.5 * width  # Compute the transverse cap centreline.
    column_x0 = 0.5 * (length - column_width)  # Compute the left column face.
    column_x1 = 0.5 * (length + column_width)  # Compute the right column face.
    column_y0 = 0.5 * (width - column_depth)  # Compute the near column face.
    column_y1 = 0.5 * (width + column_depth)  # Compute the far column face.
    patch_half_x = 0.5 * bearing_size[0]  # Compute the pressure-patch half length.
    patch_half_y = 0.5 * bearing_size[1]  # Compute the pressure-patch half width.
    bearing_edges: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []  # Accumulate the eight exact top pressure edges.
    for centre_x in bearing_centres:  # Construct each pressure-patch perimeter independently.
        x0 = centre_x - patch_half_x  # Compute the left patch edge.
        x1 = centre_x + patch_half_x  # Compute the right patch edge.
        y0 = centre_y - patch_half_y  # Compute the near patch edge.
        y1 = centre_y + patch_half_y  # Compute the far patch edge.
        bearing_edges.extend([((x0, y0, top_z), (x1, y0, top_z)), ((x1, y0, top_z), (x1, y1, top_z)), ((x1, y1, top_z), (x0, y1, top_z)), ((x0, y1, top_z), (x0, y0, top_z))])  # Register the complete pressure perimeter.
    junction_edges = [((column_x0, column_y0, cap_bottom_z), (column_x1, column_y0, cap_bottom_z)), ((column_x1, column_y0, cap_bottom_z), (column_x1, column_y1, cap_bottom_z)), ((column_x1, column_y1, cap_bottom_z), (column_x0, column_y1, cap_bottom_z)), ((column_x0, column_y1, cap_bottom_z), (column_x0, column_y0, cap_bottom_z))]  # Define the cap-column interface perimeter.
    base_edges = [((column_x0, column_y0, 0.0), (column_x1, column_y0, 0.0)), ((column_x1, column_y0, 0.0), (column_x1, column_y1, 0.0)), ((column_x1, column_y1, 0.0), (column_x0, column_y1, 0.0)), ((column_x0, column_y1, 0.0), (column_x0, column_y0, 0.0))]  # Define the fixed-base reaction perimeter.
    duct_ys = tuple(width * value for value in duct_y_fractions)  # Compute both actual duct centreline y coordinates.
    duct_z = column_height + duct_z_fraction * cap_height  # Compute the common duct centreline elevation.
    background = float(profile["background_h"])  # Read the global background size once for the hot callback.
    exponent = float(profile["gradation_exponent"])  # Read the common smooth grading exponent once.

    def size(x: float, y: float, z: float = 0.0) -> float:  # Evaluate the independent target size at one Gmsh query point.
        target = background  # Start from the globally finer-than-field background mesh.
        for start, end in bearing_edges:  # Evaluate every exact pressure-patch edge.
            distance = _point_segment_distance(x, y, z, start, end)  # Measure three-dimensional distance to the loaded edge.
            target = min(target, _graded_size(background, float(profile["bearing_floor_h"]), distance, float(profile["bearing_radius"]), exponent))  # Apply narrow top-edge grading.
        for duct_y in duct_ys:  # Evaluate each actual longitudinal duct wall.
            radial_distance = abs(math.sqrt((y - duct_y) ** 2 + (z - duct_z) ** 2) - duct_radius)  # Measure distance to the cylindrical wall rather than its void centreline.
            target = min(target, _graded_size(background, float(profile["duct_floor_h"]), radial_distance, float(profile["duct_radius_band"]), exponent))  # Apply narrow annular grading along the full duct.
        for start, end in junction_edges:  # Evaluate every cap-column interface edge.
            distance = _point_segment_distance(x, y, z, start, end)  # Measure distance to the re-entrant load-transfer perimeter.
            target = min(target, _graded_size(background, float(profile["junction_floor_h"]), distance, float(profile["junction_radius"]), exponent))  # Apply local junction grading.
        for start, end in base_edges:  # Evaluate every fixed-base perimeter edge.
            distance = _point_segment_distance(x, y, z, start, end)  # Measure distance to the support reaction perimeter.
            target = min(target, _graded_size(background, float(profile["base_floor_h"]), distance, float(profile["base_radius"]), exponent))  # Apply moderate local base grading.
        return float(target)  # Return the smallest active geometry-controlled target size.

    return size  # Return the reusable scalar callback.


def bridge_reference_floor(problem: Problem) -> float:  # Return the actual minimum prescribed by the specialized field.
    profile = bridge_reference_profile(problem)  # Read every declared feature floor.
    floors = [float(profile["bearing_floor_h"]), float(profile["duct_floor_h"]), float(profile["junction_floor_h"]), float(profile["base_floor_h"])]  # Collect all physical feature floors.
    return float(min(float(problem.h_min), min(floors)))  # Respect both the problem and specialized reference minima.


class BridgeReferenceRunner(FemRunner):  # Reuse FemRunner while replacing only the independent reference-mesh prescription.
    def ensure_reference(self) -> Reference:  # Solve or load the moderate-cost bridge-specific reference once.
        ref_path = Path(self.workdir) / "reference.json"  # Locate the repository-compatible reference metadata file.
        profile_path = Path(self.workdir) / "reference_profile.json"  # Locate the specialized field declaration.
        if self.reference is not None:  # Reuse an already loaded reference in this runner.
            return self.reference  # Avoid any duplicate mesh generation or CalculiX call.
        if ref_path.exists():  # Reuse an identical copied or cached reference artifact.
            self.reference = Reference(**json.loads(ref_path.read_text(encoding="utf-8")))  # Restore the typed reference record.
            return self.reference  # Return without mixing method trajectories.
        mesh = generate_mesh(self.problem, bridge_reference_size_fn(self.problem), h_floor=bridge_reference_floor(self.problem))  # Generate the independently prescribed geometry-aware mesh.
        post, record = self._solve(mesh, method="bridge_reference", stage=_PROFILE_VERSION, count=False)  # Run one uncounted reference CalculiX solve through the standard backend.
        self.reference = Reference(U_total=post.U_total, qoi=post.qoi, n_equations=record.n_equations, n_elems=mesh.n_cells, h_ref=float(self.problem.h_ref))  # Construct repository-compatible reference metadata.
        ref_path.write_text(json.dumps(asdict(self.reference), indent=2), encoding="utf-8")  # Persist the reusable independent reference.
        profile_path.write_text(json.dumps(bridge_reference_profile(self.problem), indent=2), encoding="utf-8")  # Persist the exact geometry-aware field contract.
        return self.reference  # Return the completed independent reference.
