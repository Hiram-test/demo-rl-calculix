"""Medium-complexity three-dimensional bridge steel-box diaphragm benchmark."""  # Describe the benchmark implemented by this module.

from __future__ import annotations  # Enable postponed evaluation of annotations.

from collections.abc import Callable  # Import the callable protocol used by geometric predicates.

import numpy as np  # Import NumPy for geometry coordinates and predicate evaluation.

from .geometry import Constraint  # Import the repository boundary-condition contract.
from .geometry import FeatureAnchor  # Import named structural anchors used by the vision partitioner.
from .geometry import Material  # Import the homogeneous material definition.
from .geometry import Problem  # Import the common finite-element problem contract.
from .geometry import TractionSpec  # Import the mesh-independent surface-traction contract.


def _box_predicate(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> Callable[[np.ndarray], np.ndarray]:  # Build a reusable axis-aligned geometric predicate.
    lower = np.asarray(lo, dtype=float)  # Convert the lower coordinate bound to a NumPy vector.
    upper = np.asarray(hi, dtype=float)  # Convert the upper coordinate bound to a NumPy vector.

    def predicate(points: np.ndarray) -> np.ndarray:  # Evaluate whether points lie inside the closed coordinate box.
        values = np.atleast_2d(np.asarray(points, dtype=float))[:, :3]  # Normalize the input to an N-by-3 coordinate array.
        return np.all((values >= lower - 1.0e-8) & (values <= upper + 1.0e-8), axis=1)  # Return the Boolean inclusion mask with a geometric tolerance.

    return predicate  # Return the completed point predicate.


def _ring_segments(x_value: float, y_center: float, z_center: float, radius: float, count: int = 16) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:  # Approximate a circular diaphragm-hole rim by short three-dimensional segments.
    angles = np.linspace(0.0, 2.0 * np.pi, count + 1)  # Create equally spaced angles including the closing angle.
    points = [(float(x_value), float(y_center + radius * np.cos(angle)), float(z_center + radius * np.sin(angle))) for angle in angles]  # Convert each polar sample to a Cartesian point on the y-z ring.
    return [(points[index], points[index + 1]) for index in range(count)]  # Return consecutive chords around the complete ring.


def make_steel_box_diaphragm(  # Construct the canonical bridge-component benchmark.
    length: float = 2400.0,  # Set the longitudinal segment length in millimetres.
    width: float = 1200.0,  # Set the external box-girder width in millimetres.
    height: float = 900.0,  # Set the external box-girder depth in millimetres.
    plate_t: float = 60.0,  # Set the top, bottom, and web solid-plate thickness.
    diaphragm_t: float = 80.0,  # Set the transverse diaphragm thickness.
    hole_radius: float = 220.0,  # Set the circular access-hole radius.
    rib_t: float = 45.0,  # Set each longitudinal rib thickness.
    rib_h: float = 150.0,  # Set each longitudinal rib depth below the top plate.
    wheel: tuple[float, float] = (420.0, 280.0),  # Set the rectangular wheel-contact footprint.
    wheel_offset: tuple[float, float] = (120.0, 110.0),  # Offset the wheel from the diaphragm and box centreline.
    pressure: float = 1.25,  # Set the downward wheel pressure in megapascals.
    support_length: float = 300.0,  # Set each bottom bearing-patch length.
    support_width: float = 420.0,  # Set each bottom bearing-patch width.
    h0: float = 180.0,  # Set the common coarse-mesh target size.
    h_ref: float = 65.0,  # Set the graded reference background target size.
    h_min: float = 24.0,  # Set the method-mesh minimum target size.
) -> Problem:  # Return a repository-standard finite-element problem.
    if min(length, width, height, plate_t, diaphragm_t, hole_radius, rib_t, rib_h, support_length, support_width, h0, h_ref, h_min) <= 0.0:  # Reject non-positive geometric or mesh parameters.
        raise ValueError("all steel-box dimensions and mesh sizes must be positive")  # Report the invalid benchmark definition.
    if 2.0 * plate_t >= min(width, height):  # Ensure that the hollow box has a non-empty interior.
        raise ValueError("plate_t is too large for the selected box dimensions")  # Report an impossible shell thickness.
    if 2.0 * hole_radius >= min(width - 2.0 * plate_t, height - 2.0 * plate_t):  # Ensure that the access hole fits inside the diaphragm.
        raise ValueError("access hole does not fit inside the diaphragm")  # Report an oversized access opening.
    diaphragm_x0 = 0.5 * length - 0.5 * diaphragm_t  # Locate the first face of the transverse diaphragm.
    diaphragm_x1 = diaphragm_x0 + diaphragm_t  # Locate the second face of the transverse diaphragm.
    wheel_length = float(wheel[0])  # Read the wheel footprint length.
    wheel_width = float(wheel[1])  # Read the wheel footprint width.
    wheel_center_x = 0.5 * length + float(wheel_offset[0])  # Locate the wheel centre longitudinally.
    wheel_center_y = 0.5 * width + float(wheel_offset[1])  # Locate the wheel centre transversely.
    wheel_x0 = wheel_center_x - 0.5 * wheel_length  # Locate the first wheel-patch x edge.
    wheel_x1 = wheel_center_x + 0.5 * wheel_length  # Locate the second wheel-patch x edge.
    wheel_y0 = wheel_center_y - 0.5 * wheel_width  # Locate the first wheel-patch y edge.
    wheel_y1 = wheel_center_y + 0.5 * wheel_width  # Locate the second wheel-patch y edge.
    if wheel_x0 <= 0.0 or wheel_x1 >= length or wheel_y0 <= 0.0 or wheel_y1 >= width:  # Ensure that the wheel footprint lies on the top plate.
        raise ValueError("wheel footprint must remain inside the top plate")  # Report an invalid load footprint.
    support_x_centres = (0.22 * length, 0.78 * length)  # Place two bearing patches beneath the bottom plate.
    support_y_center = 0.5 * width  # Centre both bearing patches transversely.
    support_boxes = []  # Allocate the list of support coordinate boxes.
    for centre_x in support_x_centres:  # Build one geometric box per support patch.
        support_boxes.append(((centre_x - 0.5 * support_length, support_y_center - 0.5 * support_width, -1.0e-6), (centre_x + 0.5 * support_length, support_y_center + 0.5 * support_width, 1.0e-6)))  # Store the closed bottom-face patch bounds.
    hole_y = 0.5 * width  # Place the diaphragm access hole on the transverse centreline.
    hole_z = 0.50 * height  # Place the diaphragm access hole near mid-depth.
    rib_centres = (0.32 * width, 0.68 * width)  # Place two longitudinal ribs under the top plate.

    def build_geometry() -> None:  # Build the conforming OpenCASCADE solid assembly.
        import gmsh  # Import Gmsh lazily so unit tests can import this module without native libraries.

        occ = gmsh.model.occ  # Obtain the OpenCASCADE geometry factory.
        top_plate = occ.addBox(0.0, 0.0, height - plate_t, length, width, plate_t)  # Create the solid top plate.
        bottom_plate = occ.addBox(0.0, 0.0, 0.0, length, width, plate_t)  # Create the solid bottom plate.
        left_web = occ.addBox(0.0, 0.0, plate_t, length, plate_t, height - 2.0 * plate_t)  # Create the left solid web.
        right_web = occ.addBox(0.0, width - plate_t, plate_t, length, plate_t, height - 2.0 * plate_t)  # Create the right solid web.
        diaphragm = occ.addBox(diaphragm_x0, plate_t, plate_t, diaphragm_t, width - 2.0 * plate_t, height - 2.0 * plate_t)  # Create the full transverse diaphragm blank.
        opening = occ.addCylinder(diaphragm_x0 - 1.0, hole_y, hole_z, diaphragm_t + 2.0, 0.0, 0.0, hole_radius)  # Create the x-directed cylindrical access-hole cutter.
        diaphragm_parts, _ = occ.cut([(3, diaphragm)], [(3, opening)], removeObject=True, removeTool=True)  # Cut the circular access opening through the diaphragm only.
        solids = [(3, top_plate), (3, bottom_plate), (3, left_web), (3, right_web)] + list(diaphragm_parts)  # Start the connected solid list with plates, webs, and the perforated diaphragm.
        for centre_y in rib_centres:  # Add the two longitudinal top-plate ribs.
            rib = occ.addBox(0.0, centre_y - 0.5 * rib_t, height - plate_t - rib_h, length, rib_t, rib_h)  # Create one solid rectangular rib touching the top plate and diaphragm.
            solids.append((3, rib))  # Add the rib to the Boolean fusion list.
        fused, _ = occ.fuse([solids[0]], solids[1:], removeObject=True, removeTool=True)  # Fuse all touching parts into a conforming connected steel solid.
        volumes = [dim_tag for dim_tag in fused if int(dim_tag[0]) == 3]  # Retain only resulting three-dimensional volume entities.
        if not volumes:  # Guard against an unexpected OpenCASCADE Boolean failure.
            raise RuntimeError("steel-box Boolean fusion produced no volume")  # Stop instead of meshing an invalid geometry.
        wheel_face = occ.addRectangle(wheel_x0, wheel_y0, height, wheel_length, wheel_width)  # Create the top-surface load imprint.
        support_faces = [occ.addRectangle(bounds[0][0], bounds[0][1], 0.0, support_length, support_width) for bounds in support_boxes]  # Create both bottom-surface support imprints.
        tools = [(2, wheel_face)] + [(2, face) for face in support_faces]  # Assemble all surface tools used to fragment the solid boundary.
        occ.fragment(volumes, tools, removeObject=True, removeTool=True)  # Imprint load and support footprints so resultants remain mesh independent.

    wheel_predicate = _box_predicate((wheel_x0, wheel_y0, height - 1.0e-6), (wheel_x1, wheel_y1, height + 1.0e-6))  # Build the top wheel-patch facet selector.
    support_one_predicate = _box_predicate(support_boxes[0][0], support_boxes[0][1])  # Build the pin-bearing node selector.
    support_two_predicate = _box_predicate(support_boxes[1][0], support_boxes[1][1])  # Build the roller-bearing node selector.
    constraints = [Constraint(support_one_predicate, (1, 2, 3), "bearing_pin"), Constraint(support_two_predicate, (3,), "bearing_roller_z")]  # Apply stable pin-and-roller support conditions.
    traction = TractionSpec(wheel_predicate, (0.0, 0.0, -float(pressure)), "wheel_patch")  # Apply the downward wheel pressure on the imprinted patch.
    singular_segments = []  # Allocate the list used by the graded reference mesh.
    singular_segments.extend([((wheel_x0, wheel_y0, height), (wheel_x1, wheel_y0, height)), ((wheel_x1, wheel_y0, height), (wheel_x1, wheel_y1, height)), ((wheel_x1, wheel_y1, height), (wheel_x0, wheel_y1, height)), ((wheel_x0, wheel_y1, height), (wheel_x0, wheel_y0, height))])  # Add all four wheel-patch edge concentration lines.
    for bounds in support_boxes:  # Add each bearing-patch perimeter to the reference grading targets.
        x0_value, y0_value, _ = bounds[0]  # Read the support lower coordinates.
        x1_value, y1_value, _ = bounds[1]  # Read the support upper coordinates.
        singular_segments.extend([((x0_value, y0_value, 0.0), (x1_value, y0_value, 0.0)), ((x1_value, y0_value, 0.0), (x1_value, y1_value, 0.0)), ((x1_value, y1_value, 0.0), (x0_value, y1_value, 0.0)), ((x0_value, y1_value, 0.0), (x0_value, y0_value, 0.0))])  # Add the four support-patch edges.
    singular_segments.extend(_ring_segments(diaphragm_x0, hole_y, hole_z, hole_radius))  # Add the first access-hole rim to the reference grading targets.
    singular_segments.extend(_ring_segments(diaphragm_x1, hole_y, hole_z, hole_radius))  # Add the second access-hole rim to the reference grading targets.
    for centre_y in rib_centres:  # Add both rib-to-top-plate junction lines.
        singular_segments.append(((0.0, centre_y, height - plate_t), (length, centre_y, height - plate_t)))  # Record the longitudinal rib junction line.
    for x_value in (diaphragm_x0, diaphragm_x1):  # Add representative diaphragm-to-box junction lines on both diaphragm faces.
        singular_segments.extend([((x_value, plate_t, plate_t), (x_value, plate_t, height - plate_t)), ((x_value, width - plate_t, plate_t), (x_value, width - plate_t, height - plate_t)), ((x_value, plate_t, plate_t), (x_value, width - plate_t, plate_t)), ((x_value, plate_t, height - plate_t), (x_value, width - plate_t, height - plate_t))])  # Record web, bottom, and top junction lines.
    features = [FeatureAnchor("wheel_center", wheel_center_x, wheel_center_y, height, "load"), FeatureAnchor("wheel_edge_x0", wheel_x0, wheel_center_y, height, "load"), FeatureAnchor("wheel_edge_x1", wheel_x1, wheel_center_y, height, "load"), FeatureAnchor("wheel_edge_y0", wheel_center_x, wheel_y0, height, "load"), FeatureAnchor("wheel_edge_y1", wheel_center_x, wheel_y1, height, "load"), FeatureAnchor("bearing_pin_center", support_x_centres[0], support_y_center, 0.0, "support"), FeatureAnchor("bearing_roller_center", support_x_centres[1], support_y_center, 0.0, "support"), FeatureAnchor("access_rim_top", 0.5 * length, hole_y, hole_z + hole_radius, "corner"), FeatureAnchor("access_rim_bottom", 0.5 * length, hole_y, hole_z - hole_radius, "corner"), FeatureAnchor("access_rim_left", 0.5 * length, hole_y - hole_radius, hole_z, "corner"), FeatureAnchor("access_rim_right", 0.5 * length, hole_y + hole_radius, hole_z, "corner"), FeatureAnchor("diaphragm_top_junction", 0.5 * length, 0.5 * width, height - plate_t, "corner"), FeatureAnchor("diaphragm_bottom_junction", 0.5 * length, 0.5 * width, plate_t, "corner"), FeatureAnchor("rib_one_junction", wheel_center_x, rib_centres[0], height - plate_t, "corner"), FeatureAnchor("rib_two_junction", wheel_center_x, rib_centres[1], height - plate_t, "corner")]  # Define semantic anchors without prescribing continuous mesh sizes.
    return Problem(name="steel_box_diaphragm", dim=3, build_geometry=build_geometry, constraints=constraints, tractions=[traction], qoi_facet_predicate=wheel_predicate, material=Material(E=210000.0, nu=0.30), h0=float(h0), h_ref=float(h_ref), h_min=float(h_min), bbox=(0.0, 0.0, 0.0, float(length), float(width), float(height)), features=features, singular_points=[], singular_segments=singular_segments, params={"length": float(length), "width": float(width), "height": float(height), "plate_t": float(plate_t), "diaphragm_t": float(diaphragm_t), "hole_radius": float(hole_radius), "rib_t": float(rib_t), "rib_h": float(rib_h), "wheel": (wheel_length, wheel_width), "wheel_offset": (float(wheel_offset[0]), float(wheel_offset[1])), "pressure": float(pressure), "support_length": float(support_length), "support_width": float(support_width)})  # Return the fully specified bridge-component problem.


def sample_steel_box_diaphragm(rng: np.random.Generator) -> Problem:  # Draw one parameterized member of the same bridge-component family.
    generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng  # Normalize the random-number source.
    hole_radius = float(generator.uniform(185.0, 245.0))  # Vary the access-hole radius within a realistic family range.
    wheel_length = float(generator.uniform(340.0, 480.0))  # Vary the wheel footprint length.
    wheel_width = float(generator.uniform(230.0, 330.0))  # Vary the wheel footprint width.
    wheel_offset_x = float(generator.uniform(-180.0, 220.0))  # Move the wheel across the diaphragm neighbourhood longitudinally.
    wheel_offset_y = float(generator.uniform(-180.0, 180.0))  # Move the wheel between the two rib lines transversely.
    pressure = float(generator.uniform(0.85, 1.55))  # Vary the applied patch pressure.
    return make_steel_box_diaphragm(hole_radius=hole_radius, wheel=(wheel_length, wheel_width), wheel_offset=(wheel_offset_x, wheel_offset_y), pressure=pressure)  # Return the sampled problem without changing its physical topology.
