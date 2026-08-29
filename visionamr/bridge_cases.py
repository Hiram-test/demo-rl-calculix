"""Medium-complexity three-dimensional bridge-component benchmarks."""  # Describe the module purpose.
from __future__ import annotations  # Enable postponed evaluation of annotations.
from collections.abc import Callable  # Import the callable protocol used by predicates.
import math  # Import trigonometric functions for circular opening segments.
import numpy as np  # Import numerical arrays for geometric predicates.
from .geometry import Constraint, FeatureAnchor, Material, Problem, TractionSpec  # Reuse the repository problem contract.

def _box_predicate(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> Callable[[np.ndarray], np.ndarray]:  # Build a mesh-independent box predicate.
    lower = np.asarray(lo, dtype=float)  # Convert the lower corner to a numerical array.
    upper = np.asarray(hi, dtype=float)  # Convert the upper corner to a numerical array.
    def predicate(points: np.ndarray) -> np.ndarray:  # Evaluate whether points lie inside the closed box.
        values = np.atleast_2d(points)[:, :3]  # Normalize the incoming coordinates to an n-by-three array.
        return np.all((values >= lower - 1.0e-9) & (values <= upper + 1.0e-9), axis=1)  # Apply a small geometric tolerance.
    return predicate  # Return the reusable predicate.

def _circle_segments(x_value: float, y_center: float, z_center: float, radius: float, count: int = 12) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:  # Approximate a circular stress line by straight segments.
    segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []  # Allocate the segment list.
    for index in range(count):  # Walk around the opening circumference.
        angle_a = 2.0 * math.pi * index / count  # Compute the first polar angle.
        angle_b = 2.0 * math.pi * (index + 1) / count  # Compute the second polar angle.
        point_a = (x_value, y_center + radius * math.cos(angle_a), z_center + radius * math.sin(angle_a))  # Form the first three-dimensional endpoint.
        point_b = (x_value, y_center + radius * math.cos(angle_b), z_center + radius * math.sin(angle_b))  # Form the second three-dimensional endpoint.
        segments.append((point_a, point_b))  # Store the local chord.
    return segments  # Return the full polygonal rim.

def make_box_girder_diaphragm(  # Define the canonical bridge-component factory.
    length: float = 600.0,  # Set the longitudinal segment length in millimetres.
    width: float = 360.0,  # Set the transverse box width in millimetres.
    height: float = 260.0,  # Set the box depth in millimetres.
    top_thickness: float = 24.0,  # Set the top-plate thickness in millimetres.
    bottom_thickness: float = 20.0,  # Set the bottom-plate thickness in millimetres.
    web_thickness: float = 18.0,  # Set each web thickness in millimetres.
    diaphragm_thickness: float = 30.0,  # Set the diaphragm longitudinal thickness in millimetres.
    opening_radius: float = 62.0,  # Set the diaphragm access-opening radius in millimetres.
    frame_width: float = 18.0,  # Set the local opening-frame width in millimetres.
    wheel_size: tuple[float, float] = (150.0, 110.0),  # Set the top wheel-patch dimensions.
    wheel_offset: tuple[float, float] = (45.0, 35.0),  # Set the wheel offset from the segment centre.
    pressure: float = 4.0,  # Set the downward patch pressure in megapascals.
    support_width: float = 70.0,  # Set the longitudinal width of each bearing strip.
) -> Problem:  # Return a complete repository Problem object.
    diaphragm_x = 0.5 * (length - diaphragm_thickness)  # Locate the diaphragm around the segment mid-plane.
    opening_y = 0.5 * width  # Centre the access opening transversely.
    opening_z = 0.50 * height  # Centre the access opening through the diaphragm depth.
    wheel_length, wheel_width = wheel_size  # Unpack the wheel-patch dimensions.
    wheel_x = 0.5 * length + wheel_offset[0]  # Locate the wheel patch longitudinally.
    wheel_y = 0.5 * width + wheel_offset[1]  # Locate the wheel patch transversely.
    wheel_x0 = wheel_x - 0.5 * wheel_length  # Compute the first wheel edge.
    wheel_x1 = wheel_x + 0.5 * wheel_length  # Compute the opposite wheel edge.
    wheel_y0 = wheel_y - 0.5 * wheel_width  # Compute the first transverse wheel edge.
    wheel_y1 = wheel_y + 0.5 * wheel_width  # Compute the opposite transverse wheel edge.
    left_support_x = 0.12 * length  # Locate the pinned bearing strip.
    right_support_x = 0.88 * length  # Locate the roller bearing strip.
    support_y0 = web_thickness  # Keep the support footprint inside the left web.
    support_y1 = width - web_thickness  # Keep the support footprint inside the right web.
    def build() -> None:  # Construct the exact OpenCASCADE solid assembly.
        import gmsh  # Import Gmsh only when geometry generation is requested.
        occ = gmsh.model.occ  # Select the OpenCASCADE geometry kernel.
        top_plate = occ.addBox(0.0, 0.0, height, length, width, top_thickness)  # Create the top plate.
        bottom_plate = occ.addBox(0.0, 0.0, 0.0, length, width, bottom_thickness)  # Create the bottom plate.
        left_web = occ.addBox(0.0, 0.0, bottom_thickness, length, web_thickness, height - bottom_thickness)  # Create the left web.
        right_web = occ.addBox(0.0, width - web_thickness, bottom_thickness, length, web_thickness, height - bottom_thickness)  # Create the right web.
        diaphragm = occ.addBox(diaphragm_x, web_thickness, bottom_thickness, diaphragm_thickness, width - 2.0 * web_thickness, height - bottom_thickness)  # Create the internal diaphragm.
        opening = occ.addCylinder(diaphragm_x - 1.0, opening_y, opening_z, diaphragm_thickness + 2.0, 0.0, 0.0, opening_radius)  # Create the through-opening cutter.
        diaphragm_parts, _ = occ.cut([(3, diaphragm)], [(3, opening)], removeObject=True, removeTool=True)  # Cut the circular access opening.
        frame_x = diaphragm_x - 0.5 * frame_width  # Extend the opening frame beyond both diaphragm faces.
        frame_depth = diaphragm_thickness + frame_width  # Set the longitudinal frame depth.
        top_frame = occ.addBox(frame_x, opening_y - opening_radius - frame_width, opening_z + opening_radius, frame_depth, 2.0 * (opening_radius + frame_width), frame_width)  # Create the upper opening-frame bar.
        bottom_frame = occ.addBox(frame_x, opening_y - opening_radius - frame_width, opening_z - opening_radius - frame_width, frame_depth, 2.0 * (opening_radius + frame_width), frame_width)  # Create the lower opening-frame bar.
        left_frame = occ.addBox(frame_x, opening_y - opening_radius - frame_width, opening_z - opening_radius, frame_depth, frame_width, 2.0 * opening_radius)  # Create the left opening-frame bar.
        right_frame = occ.addBox(frame_x, opening_y + opening_radius, opening_z - opening_radius, frame_depth, frame_width, 2.0 * opening_radius)  # Create the right opening-frame bar.
        volume_parts = [(3, top_plate), (3, bottom_plate), (3, left_web), (3, right_web)] + diaphragm_parts + [(3, top_frame), (3, bottom_frame), (3, left_frame), (3, right_frame)]  # Collect all connected structural volumes.
        fused, _ = occ.fuse([volume_parts[0]], volume_parts[1:], removeObject=True, removeTool=True)  # Fuse the plates, webs, diaphragm, and frame.
        wheel_face = occ.addRectangle(wheel_x0, wheel_y0, height + top_thickness, wheel_length, wheel_width)  # Imprint the wheel footprint on the top surface.
        left_support_face = occ.addRectangle(left_support_x - 0.5 * support_width, support_y0, 0.0, support_width, support_y1 - support_y0)  # Imprint the pinned support strip.
        right_support_face = occ.addRectangle(right_support_x - 0.5 * support_width, support_y0, 0.0, support_width, support_y1 - support_y0)  # Imprint the roller support strip.
        occ.fragment(fused, [(2, wheel_face), (2, left_support_face), (2, right_support_face)], removeObject=True, removeTool=True)  # Make load and support resultants independent of remeshing.
    def left_support_nodes(nodes: np.ndarray) -> np.ndarray:  # Select nodes on the pinned support footprint.
        on_bottom = nodes[:, 2] < 1.0e-9  # Restrict the set to the bottom surface.
        in_x = np.abs(nodes[:, 0] - left_support_x) <= 0.5 * support_width + 1.0e-9  # Apply the longitudinal strip width.
        in_y = (nodes[:, 1] >= support_y0 - 1.0e-9) & (nodes[:, 1] <= support_y1 + 1.0e-9)  # Apply the transverse strip width.
        return on_bottom & in_x & in_y  # Return the complete pinned-node mask.
    def right_support_nodes(nodes: np.ndarray) -> np.ndarray:  # Select nodes on the roller support footprint.
        on_bottom = nodes[:, 2] < 1.0e-9  # Restrict the set to the bottom surface.
        in_x = np.abs(nodes[:, 0] - right_support_x) <= 0.5 * support_width + 1.0e-9  # Apply the longitudinal strip width.
        in_y = (nodes[:, 1] >= support_y0 - 1.0e-9) & (nodes[:, 1] <= support_y1 + 1.0e-9)  # Apply the transverse strip width.
        return on_bottom & in_x & in_y  # Return the complete roller-node mask.
    constraints = [Constraint(left_support_nodes, (1, 2, 3), "left_bearing_pin"), Constraint(right_support_nodes, (3,), "right_bearing_roller")]  # Define a stable pin-roller support idealization.
    wheel_predicate = _box_predicate((wheel_x0, wheel_y0, height + top_thickness - 1.0e-6), (wheel_x1, wheel_y1, height + top_thickness + 1.0e-6))  # Define the exact loaded top patch.
    traction = TractionSpec(wheel_predicate, (0.0, 0.0, -pressure), "wheel_patch")  # Apply the vertical wheel pressure.
    rim_segments = _circle_segments(diaphragm_x, opening_y, opening_z, opening_radius) + _circle_segments(diaphragm_x + diaphragm_thickness, opening_y, opening_z, opening_radius)  # Resolve both opening rims in the reference.
    wheel_segments = [((wheel_x0, wheel_y0, height + top_thickness), (wheel_x1, wheel_y0, height + top_thickness)), ((wheel_x1, wheel_y0, height + top_thickness), (wheel_x1, wheel_y1, height + top_thickness)), ((wheel_x1, wheel_y1, height + top_thickness), (wheel_x0, wheel_y1, height + top_thickness)), ((wheel_x0, wheel_y1, height + top_thickness), (wheel_x0, wheel_y0, height + top_thickness))]  # Resolve the four wheel-patch singular lines.
    intersection_segments = [((diaphragm_x, web_thickness, bottom_thickness), (diaphragm_x, web_thickness, height)), ((diaphragm_x + diaphragm_thickness, web_thickness, bottom_thickness), (diaphragm_x + diaphragm_thickness, web_thickness, height)), ((diaphragm_x, width - web_thickness, bottom_thickness), (diaphragm_x, width - web_thickness, height)), ((diaphragm_x + diaphragm_thickness, width - web_thickness, bottom_thickness), (diaphragm_x + diaphragm_thickness, width - web_thickness, height))]  # Resolve diaphragm-to-web intersection lines.
    features = [FeatureAnchor("wheel_center", wheel_x, wheel_y, height + top_thickness, "load"), FeatureAnchor("wheel_edge_x0", wheel_x0, wheel_y, height + top_thickness, "load"), FeatureAnchor("wheel_edge_x1", wheel_x1, wheel_y, height + top_thickness, "load"), FeatureAnchor("wheel_edge_y0", wheel_x, wheel_y0, height + top_thickness, "load"), FeatureAnchor("wheel_edge_y1", wheel_x, wheel_y1, height + top_thickness, "load"), FeatureAnchor("opening_crown", diaphragm_x + 0.5 * diaphragm_thickness, opening_y, opening_z + opening_radius, "corner"), FeatureAnchor("opening_invert", diaphragm_x + 0.5 * diaphragm_thickness, opening_y, opening_z - opening_radius, "corner"), FeatureAnchor("opening_left_rim", diaphragm_x + 0.5 * diaphragm_thickness, opening_y - opening_radius, opening_z, "corner"), FeatureAnchor("opening_right_rim", diaphragm_x + 0.5 * diaphragm_thickness, opening_y + opening_radius, opening_z, "corner"), FeatureAnchor("left_web_diaphragm", diaphragm_x + 0.5 * diaphragm_thickness, web_thickness, 0.55 * height, "corner"), FeatureAnchor("right_web_diaphragm", diaphragm_x + 0.5 * diaphragm_thickness, width - web_thickness, 0.55 * height, "corner"), FeatureAnchor("left_bearing", left_support_x, 0.5 * width, 0.0, "support"), FeatureAnchor("right_bearing", right_support_x, 0.5 * width, 0.0, "support")]  # Name the competing bridge hot-spot mechanisms for the vision partition.
    return Problem(name="box_girder_diaphragm", dim=3, build_geometry=build, constraints=constraints, tractions=[traction], qoi_facet_predicate=wheel_predicate, material=Material(E=210.0e3, nu=0.30), h0=48.0, h_ref=18.0, h_min=6.0, bbox=(0.0, 0.0, 0.0, length, width, height + top_thickness), features=features, singular_points=[], singular_segments=wheel_segments + rim_segments + intersection_segments, params={"length": length, "width": width, "height": height, "top_thickness": top_thickness, "bottom_thickness": bottom_thickness, "web_thickness": web_thickness, "diaphragm_thickness": diaphragm_thickness, "opening_radius": opening_radius, "frame_width": frame_width, "wheel_size": wheel_size, "wheel_offset": wheel_offset, "pressure": pressure, "support_width": support_width})  # Assemble the complete three-dimensional bridge problem.

def sample_box_girder_diaphragm(rng: np.random.Generator) -> Problem:  # Draw a controlled family member for transition-library training.
    wheel_x = float(rng.uniform(-70.0, 70.0))  # Sample a longitudinal wheel offset.
    wheel_y = float(rng.uniform(-55.0, 55.0))  # Sample a transverse wheel offset.
    radius = float(rng.uniform(50.0, 72.0))  # Sample a practical access-opening radius.
    pressure = float(rng.uniform(3.0, 5.5))  # Sample the wheel pressure.
    diaphragm = float(rng.uniform(26.0, 36.0))  # Sample the diaphragm thickness.
    return make_box_girder_diaphragm(diaphragm_thickness=diaphragm, opening_radius=radius, wheel_offset=(wheel_x, wheel_y), pressure=pressure)  # Return the sampled bridge segment.
