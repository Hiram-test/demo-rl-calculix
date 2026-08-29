# Geometry-only three-dimensional visual partition for the bridge pier-cap world-model benchmark.  # Module purpose.
from __future__ import annotations  # Enable postponed annotations for the problem contract.
from dataclasses import dataclass  # Configure the deterministic VLM stand-in compactly.
import numpy as np  # Construct section polygons from exact bridge geometry.
from ..geometry import Problem  # Read the immutable bridge-component geometry contract.
from .drawing import DrawnRegion, drawing_centroid_xyz, irregular_from_points, irregular_halo_2d, poly_tuple  # Build true section-aware regions.
from .grades import prior_h  # Convert visual ordinal grades through the shared deterministic mapping.
from .regions import Seed  # Return the fixed named region graph expected by WM-VLA.


@dataclass  # Configure section depths without exposing numerical mesh sizes to a language model.
class BridgePierCapVisionPartitioner:  # Emulate one accurate geometry-only VLM markup for this bridge family.
    bearing_slab_fraction: float = 0.16  # Localize each bearing region near the cap top rather than through the whole depth.
    junction_radius_fraction: float = 0.22  # Cover the two column-cap load-transfer corners in front view.
    base_slab_fraction: float = 0.10  # Localize the fixed-base region near the column foot.
    duct_radius_factor: float = 2.20  # Cover each longitudinal duct wall and its surrounding stress-transfer tube.

    def propose(self, problem: Problem, post=None, eta2=None) -> list[Seed]:  # Produce one solve-free fixed region graph.
        del post, eta2  # Prohibit solved-field information from entering the visual proposal.
        if problem.name != "bridge_pier_cap" or int(problem.dim) != 3:  # Restrict this exact geometry markup to its declared family.
            raise ValueError("BridgePierCapVisionPartitioner requires the 3-D bridge_pier_cap problem")  # Fail instead of silently misdrawing another case.
        params = dict(problem.params)  # Copy the reproducible geometric parameters.
        length = float(params["length"])  # Read the cap-beam length.
        width = float(params["width"])  # Read the cap-beam width.
        cap_height = float(params["cap_height"])  # Read the cap-beam depth.
        column_height = float(params["column_height"])  # Read the modeled pier-column height.
        column_width = float(params["column_width"])  # Read the column longitudinal dimension.
        column_depth = float(params["column_depth"])  # Read the column transverse dimension.
        bearing_size = tuple(float(value) for value in params["bearing_size"])  # Read the exact pressure-footprint dimensions.
        bearing_centres = tuple(float(value) for value in params["bearing_centres"])  # Read both bearing centres.
        duct_radius = float(params["duct_radius"])  # Read the prestressing-duct radius.
        duct_y_fractions = tuple(float(value) for value in params["duct_y_fractions"])  # Read both transverse duct positions.
        duct_z_fraction = float(params["duct_z_fraction"])  # Read the duct elevation fraction inside the cap.
        top_z = column_height + cap_height  # Compute the cap top elevation.
        cap_bottom_z = column_height  # Compute the cap soffit and column-cap interface elevation.
        centre_y = 0.5 * width  # Compute the cap transverse centreline.
        column_x0 = 0.5 * (length - column_width)  # Compute the left column face.
        column_x1 = 0.5 * (length + column_width)  # Compute the right column face.
        column_y0 = 0.5 * (width - column_depth)  # Compute the near column face.
        column_y1 = 0.5 * (width + column_depth)  # Compute the far column face.
        duct_ys = tuple(width * value for value in duct_y_fractions)  # Compute both duct y coordinates.
        duct_z = column_height + duct_z_fraction * cap_height  # Compute the common duct elevation.
        drawings: list[DrawnRegion] = []  # Accumulate the fixed region geometry in authoritative order.
        patch_half_x = 0.5 * bearing_size[0]  # Compute the pressure-patch half length.
        patch_half_y = 0.5 * bearing_size[1]  # Compute the pressure-patch half width.
        for index, centre_x in enumerate(bearing_centres, start=1):  # Draw each bearing independently because the pressures differ.
            corners = np.array([[centre_x - patch_half_x, centre_y - patch_half_y], [centre_x + patch_half_x, centre_y - patch_half_y], [centre_x + patch_half_x, centre_y + patch_half_y], [centre_x - patch_half_x, centre_y + patch_half_y]], dtype=float)  # Represent the exact loaded footprint in the x-y section.
            polygon = irregular_from_points(corners, 0.08 * bearing_size[0])  # Add a modest non-box transition halo around the pressure edges.
            drawings.append(DrawnRegion(f"bearing_{index}_top_zone", prior_h(1, problem.h0), "section", poly_tuple(polygon), "vision", plane="xy", cut=top_z - 0.06 * cap_height, slab=self.bearing_slab_fraction * cap_height, grade=1))  # Localize grade one to the top cap layer.
        for index, duct_y in enumerate(duct_ys, start=1):  # Draw each longitudinal prestressing duct in its natural y-z cross-section.
            polygon = irregular_halo_2d(np.array([duct_y, duct_z], dtype=float), self.duct_radius_factor * duct_radius, phase=0.55 * index)  # Cover the curved wall and a finite transfer halo.
            drawings.append(DrawnRegion(f"prestress_duct_{index}_tube", prior_h(2, problem.h0), "side", poly_tuple(polygon), "vision", grade=2))  # Extend the y-z duct region along its real longitudinal x direction.
        junction_radius = self.junction_radius_fraction * cap_height  # Compute a moderate local load-transfer radius.
        for label, face_x, phase in (("left", column_x0, 0.3), ("right", column_x1, 1.1)):  # Draw the two main shear-transfer corners separately.
            polygon = irregular_halo_2d(np.array([face_x, cap_bottom_z], dtype=float), junction_radius, phase=phase)  # Create a front-view x-z hotspot region.
            drawings.append(DrawnRegion(f"column_cap_{label}_junction", prior_h(2, problem.h0), "front", poly_tuple(polygon), "vision", grade=2))  # Extend across the cap width while staying local in x and z.
        base_corners = np.array([[column_x0, column_y0], [column_x1, column_y0], [column_x1, column_y1], [column_x0, column_y1]], dtype=float)  # Represent the exact fixed column footprint.
        base_polygon = irregular_from_points(base_corners, 0.06 * min(column_width, column_depth))  # Add a modest reaction-transfer halo around the base perimeter.
        drawings.append(DrawnRegion("column_base_reaction_zone", prior_h(3, problem.h0), "section", poly_tuple(base_polygon), "vision", plane="xy", cut=0.05 * column_height, slab=self.base_slab_fraction * column_height, grade=3))  # Localize the support region near the fixed base.
        field_point = (0.5 * length, centre_y, 0.5 * top_z)  # Place the unpainted-volume seed at the full model centre.
        seeds = [Seed(drawing.name, drawing_centroid_xyz(drawing, problem), drawing.h, drawing.origin) for drawing in drawings]  # Convert every true three-dimensional drawing into a stable graph seed.
        seeds.append(Seed("field", field_point, prior_h(5, problem.h0), origin="coarse"))  # Give all unpainted volume an explicit coarse visual grade.
        self.last_drawings = drawings  # Cache the exact drawings for deterministic Gmsh materialization.
        self.last_grades = {drawing.name: int(drawing.grade) for drawing in drawings}  # Cache only ordinal visual decisions.
        self.last_grades["field"] = 5  # Preserve the explicit remainder grade.
        return seeds  # Return the complete fixed region graph after one geometry-only observation.
