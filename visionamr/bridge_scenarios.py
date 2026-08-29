# Medium-complexity three-dimensional bridge-component scenarios for WM-VLA evaluation.  # Module purpose.
from __future__ import annotations  # Enable postponed annotations for lightweight factory typing.
from collections.abc import Callable  # Type the exported scenario factory registry.
import numpy as np  # Build robust geometric predicates for loads, constraints, and QoI facets.
from .geometry import Constraint, FeatureAnchor, Material, Problem, TractionSpec  # Reuse the repository problem contract.


def _box_predicate(lower: tuple[float, float, float], upper: tuple[float, float, float]) -> Callable[[np.ndarray], np.ndarray]:  # Build a remesh-invariant box predicate.
    lo = np.asarray(lower, dtype=float)  # Normalize lower coordinates once.
    hi = np.asarray(upper, dtype=float)  # Normalize upper coordinates once.

    def predicate(points: np.ndarray) -> np.ndarray:  # Evaluate the geometric region on arbitrary mesh points.
        values = np.atleast_2d(np.asarray(points, dtype=float))[:, :3]  # Normalize input shape and physical coordinates.
        return np.all((values >= lo - 1.0e-8) & (values <= hi + 1.0e-8), axis=1)  # Return a robust Boolean mask.

    return predicate  # Return the reusable geometric predicate.


def make_bridge_pier_cap(  # Build a concrete pier-cap subassembly with competing three-dimensional hotspots.
    length: float = 3200.0,  # Set cap-beam longitudinal length in millimetres.
    width: float = 1000.0,  # Set cap-beam transverse width in millimetres.
    cap_height: float = 600.0,  # Set cap-beam vertical depth in millimetres.
    column_height: float = 1200.0,  # Set the modeled pier-column height below the cap.
    column_width: float = 820.0,  # Set the column dimension along the cap-beam axis.
    column_depth: float = 680.0,  # Set the column transverse dimension.
    bearing_size: tuple[float, float] = (420.0, 420.0),  # Set both top bearing-footprint dimensions.
    bearing_centres: tuple[float, float] = (800.0, 2400.0),  # Set left and right bearing centres along the beam.
    pressures: tuple[float, float] = (8.0, 11.0),  # Set unequal vertical bearing pressures in MPa.
    duct_radius: float = 65.0,  # Set the radius of two longitudinal prestressing ducts.
    duct_y_fractions: tuple[float, float] = (0.32, 0.68),  # Set the transverse duct locations as width fractions.
    duct_z_fraction: float = 0.38,  # Set the duct height within the cap as a depth fraction.
) -> Problem:  # Return a complete remesh-invariant finite-element problem.
    top_z = float(column_height + cap_height)  # Compute the top bearing elevation.
    cap_bottom_z = float(column_height)  # Compute the cap soffit elevation.
    centre_y = 0.5 * float(width)  # Compute the transverse centreline.
    column_x0 = 0.5 * (float(length) - float(column_width))  # Compute the left column face.
    column_x1 = 0.5 * (float(length) + float(column_width))  # Compute the right column face.
    column_y0 = 0.5 * (float(width) - float(column_depth))  # Compute the near column face.
    column_y1 = 0.5 * (float(width) + float(column_depth))  # Compute the far column face.
    patch_x = float(bearing_size[0])  # Read the bearing footprint length.
    patch_y = float(bearing_size[1])  # Read the bearing footprint width.
    patch_boxes: list[tuple[float, float, float, float]] = []  # Store both exact top pressure footprints.
    for centre_x in bearing_centres:  # Construct each bearing footprint deterministically.
        patch_boxes.append((float(centre_x) - 0.5 * patch_x, float(centre_x) + 0.5 * patch_x, centre_y - 0.5 * patch_y, centre_y + 0.5 * patch_y))  # Store x0, x1, y0, y1.
    duct_ys = tuple(float(width) * float(value) for value in duct_y_fractions)  # Compute both duct centreline y coordinates.
    duct_z = float(column_height + duct_z_fraction * cap_height)  # Compute the common duct centreline elevation.

    def build_geometry() -> None:  # Build and imprint the solid geometry through the Gmsh OCC API.
        import gmsh  # Import Gmsh only when meshing is actually requested.
        occ = gmsh.model.occ  # Use the OpenCASCADE geometry kernel.
        cap = occ.addBox(0.0, 0.0, cap_bottom_z, float(length), float(width), float(cap_height))  # Create the cap beam.
        column = occ.addBox(column_x0, column_y0, 0.0, float(column_width), float(column_depth), float(column_height))  # Create the pier column.
        body, _ = occ.fuse([(3, cap)], [(3, column)])  # Fuse cap and column into one load-transferring solid.
        cutters = []  # Accumulate longitudinal prestressing-duct volumes.
        for duct_y in duct_ys:  # Create each through duct independently.
            cylinder = occ.addCylinder(-2.0, duct_y, duct_z, float(length) + 4.0, 0.0, 0.0, float(duct_radius))  # Extend beyond both end faces for a clean cut.
            cutters.append((3, cylinder))  # Register the cylinder as a Boolean cutter.
        body, _ = occ.cut(body, cutters)  # Cut both ducts from the fused concrete body.
        patch_faces = []  # Accumulate load-imprinting top surfaces.
        for x0, x1, y0, y1 in patch_boxes:  # Create one coplanar rectangle per bearing footprint.
            patch_face = occ.addRectangle(x0, y0, top_z, x1 - x0, y1 - y0)  # Create the exact pressure surface.
            patch_faces.append((2, patch_face))  # Register the surface for fragmentation.
        occ.fragment(body, patch_faces)  # Imprint load boundaries so total forces stay mesh independent.

    def base_nodes(nodes: np.ndarray) -> np.ndarray:  # Select the complete pier-column base.
        points = np.atleast_2d(np.asarray(nodes, dtype=float))  # Normalize input node coordinates.
        in_column = (points[:, 0] >= column_x0 - 1.0e-8) & (points[:, 0] <= column_x1 + 1.0e-8) & (points[:, 1] >= column_y0 - 1.0e-8) & (points[:, 1] <= column_y1 + 1.0e-8)  # Restrict to the column footprint.
        return in_column & (points[:, 2] <= 1.0e-8)  # Return base nodes only.

    base_constraint = Constraint(base_nodes, (1, 2, 3), "column_base_fixed")  # Clamp all translational components at the base.
    patch_predicates = [_box_predicate((x0, y0, top_z - 1.0e-6), (x1, y1, top_z + 1.0e-6)) for x0, x1, y0, y1 in patch_boxes]  # Build remesh-invariant pressure predicates.
    tractions = [TractionSpec(predicate, (0.0, 0.0, -float(pressure)), f"bearing_patch_{index + 1}") for index, (predicate, pressure) in enumerate(zip(patch_predicates, pressures))]  # Create unequal bearing loads.

    def qoi_predicate(points: np.ndarray) -> np.ndarray:  # Select both bearing footprints for the displacement QoI.
        masks = [predicate(points) for predicate in patch_predicates]  # Evaluate each exact footprint.
        return np.logical_or.reduce(masks)  # Return their union.

    features: list[FeatureAnchor] = []  # Accumulate structural semantics for the one-shot visual partition.
    singular_segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []  # Accumulate known concentration lines for the reference field.
    for index, ((x0, x1, y0, y1), centre_x) in enumerate(zip(patch_boxes, bearing_centres), start=1):  # Describe each bearing patch.
        features.append(FeatureAnchor(f"bearing_{index}_centre", float(centre_x), centre_y, top_z, "load"))  # Mark the loaded centre.
        features.append(FeatureAnchor(f"bearing_{index}_edge_x0", x0, centre_y, top_z, "load"))  # Mark the left pressure edge.
        features.append(FeatureAnchor(f"bearing_{index}_edge_x1", x1, centre_y, top_z, "load"))  # Mark the right pressure edge.
        features.append(FeatureAnchor(f"bearing_{index}_edge_y0", float(centre_x), y0, top_z, "load"))  # Mark the near pressure edge.
        features.append(FeatureAnchor(f"bearing_{index}_edge_y1", float(centre_x), y1, top_z, "load"))  # Mark the far pressure edge.
        singular_segments.extend([((x0, y0, top_z), (x1, y0, top_z)), ((x1, y0, top_z), (x1, y1, top_z)), ((x1, y1, top_z), (x0, y1, top_z)), ((x0, y1, top_z), (x0, y0, top_z))])  # Register all patch-perimeter lines.
    junction_corners = ((column_x0, column_y0), (column_x1, column_y0), (column_x1, column_y1), (column_x0, column_y1))  # Define the cap-soffit re-entrant perimeter.
    for index, (x_value, y_value) in enumerate(junction_corners, start=1):  # Describe each column-cap junction corner.
        features.append(FeatureAnchor(f"column_cap_corner_{index}", x_value, y_value, cap_bottom_z, "corner"))  # Mark the three-dimensional re-entrant corner.
    singular_segments.extend([((column_x0, column_y0, cap_bottom_z), (column_x1, column_y0, cap_bottom_z)), ((column_x1, column_y0, cap_bottom_z), (column_x1, column_y1, cap_bottom_z)), ((column_x1, column_y1, cap_bottom_z), (column_x0, column_y1, cap_bottom_z)), ((column_x0, column_y1, cap_bottom_z), (column_x0, column_y0, cap_bottom_z))])  # Register the junction perimeter.
    for index, duct_y in enumerate(duct_ys, start=1):  # Describe each prestressing duct.
        features.append(FeatureAnchor(f"prestress_duct_{index}", 0.5 * float(length), duct_y, duct_z, "hole", 0.0))  # Mark the duct semantically without a misleading planar reference radius.
        singular_segments.extend([((0.0, duct_y - duct_radius, duct_z), (float(length), duct_y - duct_radius, duct_z)), ((0.0, duct_y + duct_radius, duct_z), (float(length), duct_y + duct_radius, duct_z)), ((0.0, duct_y, duct_z - duct_radius), (float(length), duct_y, duct_z - duct_radius)), ((0.0, duct_y, duct_z + duct_radius), (float(length), duct_y, duct_z + duct_radius))])  # Approximate duct-wall hotspot traces for the graded reference.
    features.append(FeatureAnchor("column_base_centre", 0.5 * float(length), centre_y, 0.0, "clamp"))  # Mark the fixed support region.
    features.append(FeatureAnchor("cap_midfield", 0.5 * float(length), centre_y, top_z - 0.5 * float(cap_height), "support"))  # Mark the global load-transfer body.
    base_segments = [((column_x0, column_y0, 0.0), (column_x1, column_y0, 0.0)), ((column_x1, column_y0, 0.0), (column_x1, column_y1, 0.0)), ((column_x1, column_y1, 0.0), (column_x0, column_y1, 0.0)), ((column_x0, column_y1, 0.0), (column_x0, column_y0, 0.0))]  # Define fixed-base reaction lines.
    singular_segments.extend(base_segments)  # Include fixed-base concentrations in the reference field.
    parameters = {"length": float(length), "width": float(width), "cap_height": float(cap_height), "column_height": float(column_height), "column_width": float(column_width), "column_depth": float(column_depth), "bearing_size": tuple(float(value) for value in bearing_size), "bearing_centres": tuple(float(value) for value in bearing_centres), "pressures": tuple(float(value) for value in pressures), "duct_radius": float(duct_radius), "duct_y_fractions": tuple(float(value) for value in duct_y_fractions), "duct_z_fraction": float(duct_z_fraction)}  # Preserve the complete reproducibility state.
    return Problem(  # Construct the repository-standard finite-element problem.
        name="bridge_pier_cap",  # Give the scenario a stable campaign family name.
        dim=3,  # Use full three-dimensional tetrahedral elasticity.
        build_geometry=build_geometry,  # Attach the OCC geometry builder.
        constraints=[base_constraint],  # Attach the fixed pier-column base.
        tractions=tractions,  # Attach both unequal bearing pressures.
        qoi_facet_predicate=qoi_predicate,  # Measure displacement over both bearing footprints.
        material=Material(E=34.0e3, nu=0.20),  # Use a representative linear-elastic concrete material in MPa.
        h0=float(cap_height) / 2.0,  # Use a coarse 300 mm-class development mesh by default.
        h_ref=float(cap_height) / 7.5,  # Use an 80 mm-class graded reference background.
        h_min=float(cap_height) / 24.0,  # Permit 25 mm-class local refinement around critical details.
        bbox=(0.0, 0.0, 0.0, float(length), float(width), top_z),  # Preserve the complete model bounding box.
        features=features,  # Expose bridge semantics to the one-shot visual head.
        singular_points=[],  # Use line and smooth-hole concentrations rather than isolated point singularities.
        singular_segments=singular_segments,  # Drive an independently graded reference mesh.
        params=parameters,  # Preserve exact scenario parameters for instance hashing.
    )  # Finish problem construction.


def sample_bridge_pier_cap(rng: np.random.Generator) -> Problem:  # Draw a nearby instance for transfer and OOD evaluation.
    left = float(rng.uniform(680.0, 920.0))  # Vary the left bearing location.
    right = float(rng.uniform(2280.0, 2520.0))  # Vary the right bearing location.
    pressure_left = float(rng.uniform(6.5, 10.0))  # Vary the left bearing pressure.
    pressure_right = float(rng.uniform(8.0, 13.0))  # Vary the right bearing pressure independently.
    radius = float(rng.uniform(55.0, 78.0))  # Vary duct diameter within a realistic moderate range.
    transverse_shift = float(rng.uniform(-0.025, 0.025))  # Vary both ducts without approaching the cap faces.
    return make_bridge_pier_cap(bearing_centres=(left, right), pressures=(pressure_left, pressure_right), duct_radius=radius, duct_y_fractions=(0.32 + transverse_shift, 0.68 + transverse_shift))  # Return the sampled bridge component.


BRIDGE_SCENARIO_FACTORIES: dict[str, Callable[[], Problem]] = {"bridge_pier_cap": make_bridge_pier_cap}  # Export canonical scenario factories.
