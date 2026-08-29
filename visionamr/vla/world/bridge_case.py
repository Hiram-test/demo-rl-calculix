"""Medium-complexity three-dimensional steel box-girder diaphragm benchmark."""  # Describe the bridge component represented by this module.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from collections.abc import Callable  # Import the boundary-predicate protocol.
import numpy as np  # Import array operations used by geometric predicates.
from ...geometry import Constraint, FeatureAnchor, Material, Problem, TractionSpec  # Reuse the repository finite-element problem contract.

def _box_predicate(lower: tuple[float, float, float], upper: tuple[float, float, float]) -> Callable[[np.ndarray], np.ndarray]:  # Build a remesh-stable axis-aligned boundary predicate.
    lo = np.asarray(lower, dtype=float)  # Convert the lower corner to a numerical vector.
    hi = np.asarray(upper, dtype=float)  # Convert the upper corner to a numerical vector.
    def predicate(points: np.ndarray) -> np.ndarray:  # Evaluate whether points fall inside the closed box.
        xyz = np.atleast_2d(points)[:, :3]  # Normalize the point array to three spatial coordinates.
        return np.all((xyz >= lo - 1.0e-8) & (xyz <= hi + 1.0e-8), axis=1)  # Return the tolerant inclusion mask.
    return predicate  # Return the reusable predicate closure.

def make_box_girder_diaphragm(*, length: float = 720.0, width: float = 900.0, height: float = 560.0, flange_t: float = 55.0, web_t: float = 45.0, diaphragm_t: float = 60.0, opening_r: float = 130.0, pressure: float = 2.4, load_offset_y: float = 105.0, h0: float = 80.0, h_ref: float = 42.0, h_min: float = 12.0) -> Problem:  # Construct a connected box-girder segment with an opened transverse diaphragm.
    values = np.asarray([length, width, height, flange_t, web_t, diaphragm_t, opening_r, pressure, h0, h_ref, h_min], dtype=float)  # Collect parameters for validation.
    if np.any(values <= 0.0):  # Reject non-physical dimensions, load, or mesh sizes.
        raise ValueError("all box-girder diaphragm parameters must be positive")  # Explain the invalid parameter contract.
    if 2.0 * flange_t >= height or 2.0 * web_t >= width:  # Ensure a non-empty internal box region.
        raise ValueError("flange and web thicknesses leave no internal box region")  # Report the impossible plate layout.
    if opening_r >= 0.42 * min(width - 2.0 * web_t, height - 2.0 * flange_t):  # Keep the inspection opening inside the diaphragm.
        raise ValueError("inspection opening is too large for the diaphragm")  # Report the topology violation.
    if not h_min <= h_ref <= h0:  # Require an ordered mesh-size hierarchy.
        raise ValueError("mesh sizes must satisfy h_min <= h_ref <= h0")  # Report the invalid meshing contract.
    x_mid = 0.5 * length  # Locate the transverse diaphragm station.
    y_mid = 0.5 * width  # Locate the box-girder transverse centreline.
    z_mid = 0.5 * height  # Locate the box-girder vertical centreline.
    load_x = x_mid + 0.16 * length  # Place the wheel patch eccentrically along the girder.
    load_y = y_mid + load_offset_y  # Place the wheel patch eccentrically across the girder.
    load_dx = 0.25 * length  # Set a finite longitudinal wheel-contact dimension.
    load_dy = 0.24 * width  # Set a finite transverse wheel-contact dimension.
    load_x0 = load_x - 0.5 * load_dx  # Compute the first wheel-patch x edge.
    load_x1 = load_x + 0.5 * load_dx  # Compute the second wheel-patch x edge.
    load_y0 = load_y - 0.5 * load_dy  # Compute the first wheel-patch y edge.
    load_y1 = load_y + 0.5 * load_dy  # Compute the second wheel-patch y edge.
    pad_dx = 0.28 * length  # Set each bearing-pad longitudinal footprint.
    pad_dy = max(2.4 * web_t, 0.14 * width)  # Cover the web-foot region without fixing the whole flange.
    pad_x0 = x_mid - 0.5 * pad_dx  # Compute the common bearing-pad first x edge.
    pad_x1 = x_mid + 0.5 * pad_dx  # Compute the common bearing-pad second x edge.
    left_y0 = 0.0  # Place the first bearing pad below the first web.
    left_y1 = pad_dy  # Compute the first bearing-pad inner edge.
    right_y0 = width - pad_dy  # Compute the second bearing-pad inner edge.
    right_y1 = width  # Place the second bearing pad below the second web.
    def build_geometry() -> None:  # Build one conformal OpenCASCADE solid assembly.
        import gmsh  # Import Gmsh lazily so unit tests do not require its native library.
        occ = gmsh.model.occ  # Select the OpenCASCADE geometry kernel.
        bottom = occ.addBox(0.0, 0.0, 0.0, length, width, flange_t)  # Create the bottom flange plate.
        top = occ.addBox(0.0, 0.0, height - flange_t, length, width, flange_t)  # Create the top flange plate.
        web_left = occ.addBox(0.0, 0.0, flange_t, length, web_t, height - 2.0 * flange_t)  # Create the first longitudinal web.
        web_right = occ.addBox(0.0, width - web_t, flange_t, length, web_t, height - 2.0 * flange_t)  # Create the second longitudinal web.
        diaphragm = occ.addBox(x_mid - 0.5 * diaphragm_t, web_t, flange_t, diaphragm_t, width - 2.0 * web_t, height - 2.0 * flange_t)  # Create the transverse diaphragm plate.
        fused, _ = occ.fuse([(3, bottom)], [(3, top), (3, web_left), (3, web_right), (3, diaphragm)], removeObject=True, removeTool=True)  # Fuse all plates into one connected structural body.
        opening = occ.addCylinder(x_mid - diaphragm_t, y_mid, z_mid, 2.0 * diaphragm_t, 0.0, 0.0, opening_r)  # Create a through-thickness circular inspection opening.
        opened, _ = occ.cut(fused, [(3, opening)], removeObject=True, removeTool=True)  # Cut the opening only where it intersects the diaphragm.
        wheel_face = occ.addRectangle(load_x0, load_y0, height, load_dx, load_dy)  # Imprint the finite wheel patch on the top flange.
        left_support_face = occ.addRectangle(pad_x0, left_y0, 0.0, pad_dx, pad_dy)  # Imprint the first support pad on the bottom flange.
        right_support_face = occ.addRectangle(pad_x0, right_y0, 0.0, pad_dx, pad_dy)  # Imprint the second support pad on the bottom flange.
        occ.fragment(opened, [(2, wheel_face), (2, left_support_face), (2, right_support_face)], removeObject=True, removeTool=True)  # Make load and support footprints exact under every remesh.
    wheel_predicate = _box_predicate((load_x0, load_y0, height - 1.0e-6), (load_x1, load_y1, height + 1.0e-6))  # Define the mesh-independent wheel-load surface.
    left_support_predicate = _box_predicate((pad_x0, left_y0, -1.0e-6), (pad_x1, left_y1, 1.0e-6))  # Define the first mesh-independent bearing surface.
    right_support_predicate = _box_predicate((pad_x0, right_y0, -1.0e-6), (pad_x1, right_y1, 1.0e-6))  # Define the second mesh-independent bearing surface.
    constraints = [Constraint(left_support_predicate, (1, 2, 3), "left_bearing_fixed"), Constraint(right_support_predicate, (2, 3), "right_bearing_guided")]  # Remove rigid-body motion while retaining longitudinal expansion.
    traction = TractionSpec(wheel_predicate, (0.0, 0.0, -pressure), "eccentric_wheel_patch")  # Apply a finite downward pressure footprint.
    features = [FeatureAnchor("wheel_center", load_x, load_y, height, "load"), FeatureAnchor("wheel_edge_longitudinal", load_x0, load_y, height, "load"), FeatureAnchor("wheel_edge_transverse", load_x, load_y0, height, "load"), FeatureAnchor("inspection_opening", x_mid, y_mid, z_mid, "hole", r=opening_r), FeatureAnchor("opening_upper_rim", x_mid, y_mid, z_mid + opening_r, "hole", r=0.35 * opening_r), FeatureAnchor("opening_lower_rim", x_mid, y_mid, z_mid - opening_r, "hole", r=0.35 * opening_r), FeatureAnchor("left_bearing", x_mid, 0.5 * pad_dy, 0.0, "support"), FeatureAnchor("right_bearing", x_mid, width - 0.5 * pad_dy, 0.0, "support"), FeatureAnchor("left_web_diaphragm", x_mid, web_t, z_mid, "corner"), FeatureAnchor("right_web_diaphragm", x_mid, width - web_t, z_mid, "corner"), FeatureAnchor("top_diaphragm_joint", x_mid, y_mid, height - flange_t, "corner"), FeatureAnchor("bottom_diaphragm_joint", x_mid, y_mid, flange_t, "corner")]  # Name the competing structural mechanisms for the vision partition.
    segments = [((load_x0, load_y0, height), (load_x1, load_y0, height)), ((load_x1, load_y0, height), (load_x1, load_y1, height)), ((load_x1, load_y1, height), (load_x0, load_y1, height)), ((load_x0, load_y1, height), (load_x0, load_y0, height)), ((pad_x0, left_y1, 0.0), (pad_x1, left_y1, 0.0)), ((pad_x0, right_y0, 0.0), (pad_x1, right_y0, 0.0)), ((x_mid, web_t, flange_t), (x_mid, web_t, height - flange_t)), ((x_mid, width - web_t, flange_t), (x_mid, width - web_t, height - flange_t)), ((x_mid, web_t, flange_t), (x_mid, width - web_t, flange_t)), ((x_mid, web_t, height - flange_t), (x_mid, width - web_t, height - flange_t))]  # Register load, support, web, and flange concentration lines for the reference mesh.
    parameters = {"length": float(length), "width": float(width), "height": float(height), "flange_t": float(flange_t), "web_t": float(web_t), "diaphragm_t": float(diaphragm_t), "opening_r": float(opening_r), "pressure": float(pressure), "load_offset_y": float(load_offset_y)}  # Record all physical parameters in the instance identity.
    return Problem(name="box_girder_diaphragm", dim=3, build_geometry=build_geometry, constraints=constraints, tractions=[traction], qoi_facet_predicate=wheel_predicate, material=Material(E=210.0e3, nu=0.3), h0=float(h0), h_ref=float(h_ref), h_min=float(h_min), bbox=(0.0, 0.0, 0.0, float(length), float(width), float(height)), features=features, singular_points=[], singular_segments=segments, params=parameters)  # Return the complete finite-element problem contract.

def make_box_girder_diaphragm_smoke() -> Problem:  # Construct a smaller CI instance with the same competing mechanisms.
    return make_box_girder_diaphragm(length=480.0, width=620.0, height=410.0, flange_t=48.0, web_t=42.0, diaphragm_t=50.0, opening_r=86.0, pressure=2.0, load_offset_y=72.0, h0=78.0, h_ref=52.0, h_min=18.0)  # Preserve topology while reducing the real-solver cost.
