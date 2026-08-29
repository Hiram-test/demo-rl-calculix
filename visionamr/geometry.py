"""Benchmark problem definitions.

Problems are built with the Gmsh OCC API; boundary conditions, loads and
the quantity of interest are geometric predicates so they survive any
remesh.  Two 3-D bridge-component families are the main experimental
objects; two 2-D families remain as the fast development/CI substrate.

3-D bridge components:

* ``bearing_block`` -- bridge bearing plate under a local patch pressure
  from the girder flange (singular lines along the patch edges and the
  clamped bottom perimeter).
* ``deck_panel``    -- beam-bridge deck slab strip on two girder support
  strips under a wheel patch load (bending hotspot under the wheel plus
  reaction concentrations along the strip edges).

2-D substrate: ``lbracket`` (re-entrant corner singularity) and
``plate_holes`` (smooth hole hotspots).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Material:
    E: float = 210e3  # MPa
    nu: float = 0.3
    thickness: float = 1.0  # mm, 2-D plane stress only


@dataclass(frozen=True)
class FeatureAnchor:
    """Named structural feature usable for region naming and grading."""

    name: str
    x: float
    y: float
    z: float = 0.0
    kind: str = "corner"  # "corner" | "hole" | "clamp" | "load" | "support"
    r: float = 0.0

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass(frozen=True)
class Constraint:
    """Fix the given dofs (1-based: 1=ux, 2=uy, 3=uz) on matching nodes."""

    node_predicate: Callable[[np.ndarray], np.ndarray]
    dofs: tuple[int, ...]
    name: str = "fix"


@dataclass(frozen=True)
class TractionSpec:
    """Traction on boundary facets whose centroids satisfy the predicate.

    ``value`` is in MPa; nodal forces are assembled as
    t * facet_measure * (thickness in 2-D), split equally over facet nodes.
    """

    facet_predicate: Callable[[np.ndarray], np.ndarray]
    value: tuple[float, float, float]
    name: str = "load"


@dataclass
class Problem:
    name: str
    dim: int
    build_geometry: Callable[[], None]
    constraints: Sequence[Constraint]
    tractions: Sequence[TractionSpec]
    qoi_facet_predicate: Callable[[np.ndarray], np.ndarray]
    material: Material
    h0: float
    h_ref: float
    h_min: float
    bbox: tuple[float, float, float, float, float, float]
    features: list[FeatureAnchor] = field(default_factory=list)
    singular_points: list[tuple[float, float, float]] = field(default_factory=list)
    singular_segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = field(
        default_factory=list
    )
    params: dict = field(default_factory=dict)

    @property
    def instance_id(self) -> str:
        blob = repr(sorted(self.params.items())).encode()
        return f"{self.name}-{hashlib.sha256(blob).hexdigest()[:8]}"

    @property
    def diameter(self) -> float:
        b = self.bbox
        return float(np.linalg.norm([b[3] - b[0], b[4] - b[1], b[5] - b[2]]))


def drawing_bc_marks(problem: Problem) -> dict:
    """What the eye drawing may paint, taken from BCs — not a family hardcode.

    A full-bottom clamp overlay is only honest when a constraint is actually
    named ``bottom_fixed``.  The load-patch label is the traction name
    (``girder_patch`` vs ``wheel_patch``), so a deck slab is not captioned
    as a bearing.
    """

    full_bottom = any(c.name == "bottom_fixed" for c in problem.constraints)
    n_loads = sum(1 for f in problem.features if f.kind == "load")
    load_label = None
    if n_loads >= 3 and problem.tractions:
        load_label = str(problem.tractions[0].name).replace("_", " ")
    return {
        "full_bottom_clamp": bool(full_bottom),
        "load_patch_label": load_label,
    }


def _box_pred(lo, hi) -> Callable[[np.ndarray], np.ndarray]:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    def pred(pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)[:, : len(lo)]
        return np.all((pts >= lo - 1e-9) & (pts <= hi + 1e-9), axis=1)

    return pred


# ===========================================================================
# 3-D bridge components
# ===========================================================================


def make_bearing_block(
    W: float = 400.0,
    D: float = 400.0,
    H: float = 120.0,
    patch: tuple[float, float] = (140.0, 140.0),
    offset: tuple[float, float] = (40.0, 0.0),
    pressure: float = 12.0,
) -> Problem:
    """Bridge bearing plate: solid block, bottom fixed, top patch pressure.

    The patch models the girder-flange contact footprint; its edges carry
    3-D line singularities, the strongest AMR targets.
    """

    a, b = patch
    ox, oy = offset
    cx, cy = W / 2.0 + ox, D / 2.0 + oy
    px0, px1 = cx - a / 2.0, cx + a / 2.0
    py0, py1 = cy - b / 2.0, cy + b / 2.0

    def build() -> None:
        import gmsh

        occ = gmsh.model.occ
        box = occ.addBox(0.0, 0.0, 0.0, W, D, H)
        # imprint the load footprint so every mesh tiles the patch exactly
        # (mesh-independent load resultant)
        patch_face = occ.addRectangle(px0, py0, H, a, b)
        occ.fragment([(3, box)], [(2, patch_face)])

    bottom = Constraint(lambda n: n[:, 2] < 1e-9, (1, 2, 3), "bottom_fixed")
    patch_pred = _box_pred((px0, py0, H - 1e-6), (px1, py1, H + 1e-6))
    tr = TractionSpec(patch_pred, (0.0, 0.0, -pressure), "girder_patch")

    zt = H
    segs = [
        ((px0, py0, zt), (px1, py0, zt)),
        ((px1, py0, zt), (px1, py1, zt)),
        ((px1, py1, zt), (px0, py1, zt)),
        ((px0, py1, zt), (px0, py0, zt)),
        # clamped bottom perimeter
        ((0, 0, 0), (W, 0, 0)),
        ((W, 0, 0), (W, D, 0)),
        ((W, D, 0), (0, D, 0)),
        ((0, D, 0), (0, 0, 0)),
    ]
    features = [
        FeatureAnchor("patch_center", cx, cy, H, "load"),
        FeatureAnchor("patch_edge_x0", px0, cy, H, "load"),
        FeatureAnchor("patch_edge_x1", px1, cy, H, "load"),
        FeatureAnchor("patch_edge_y0", cx, py0, H, "load"),
        FeatureAnchor("patch_edge_y1", cx, py1, H, "load"),
        FeatureAnchor("bottom_center", W / 2, D / 2, 0.0, "clamp"),
        # clamped-edge reaction lines (textbook bearing-edge concentration)
        FeatureAnchor("bottom_edge_y0", W / 2, 0.0, 0.0, "support"),
        FeatureAnchor("bottom_edge_y1", W / 2, D, 0.0, "support"),
        FeatureAnchor("bottom_edge_x0", 0.0, D / 2, 0.0, "support"),
        FeatureAnchor("bottom_edge_x1", W, D / 2, 0.0, "support"),
    ]
    return Problem(
        name="bearing_block",
        dim=3,
        build_geometry=build,
        constraints=[bottom],
        tractions=[tr],
        qoi_facet_predicate=patch_pred,
        material=Material(),
        h0=H / 2.4,          # 50 mm
        h_ref=H / 10.0,      # 12 mm background, graded to edges
        h_min=H / 40.0,      # 3 mm
        bbox=(0, 0, 0, W, D, H),
        features=features,
        singular_points=[],
        singular_segments=segs,
        params={"W": W, "D": D, "H": H, "patch": patch, "offset": offset,
                "pressure": pressure},
    )


def sample_bearing_block(rng: np.random.Generator) -> Problem:
    a = float(rng.uniform(100.0, 180.0))
    b = float(rng.uniform(100.0, 180.0))
    ox = float(rng.uniform(-70.0, 70.0))
    oy = float(rng.uniform(-70.0, 70.0))
    return make_bearing_block(
        patch=(a, b), offset=(ox, oy), pressure=float(rng.uniform(8.0, 16.0))
    )


def make_deck_panel(
    L: float = 2400.0,
    B: float = 1600.0,
    T: float = 200.0,
    strip_w: float = 220.0,
    strip_off: float = 300.0,
    wheel: tuple[float, float] = (400.0, 250.0),
    wheel_pos: tuple[float, float] = (1200.0, 800.0),
    pressure: float = 1.0,
) -> Problem:
    """Beam-bridge deck slab strip on two girder support strips.

    Girders run along x at y = strip_off and y = B - strip_off (strip
    width ``strip_w``, uz fixed on the bottom face there).  A wheel patch
    presses on the top face.  Hotspots: under the wheel and along the
    inner strip edges (reaction line concentrations).
    """

    y1, y2 = strip_off, B - strip_off
    wa, wb = wheel
    wx, wy = wheel_pos
    wx0, wx1 = wx - wa / 2.0, wx + wa / 2.0
    wy0, wy1 = wy - wb / 2.0, wy + wb / 2.0

    def build() -> None:
        import gmsh

        occ = gmsh.model.occ
        box = occ.addBox(0.0, 0.0, 0.0, L, B, T)
        # imprint the wheel footprint and both support strips so loads and
        # constraints are mesh-independent
        wheel_face = occ.addRectangle(wx0, wy0, T, wa, wb)
        s1_face = occ.addRectangle(0.0, y1 - strip_w / 2.0, 0.0, L, strip_w)
        s2_face = occ.addRectangle(0.0, y2 - strip_w / 2.0, 0.0, L, strip_w)
        occ.fragment([(3, box)], [(2, wheel_face), (2, s1_face), (2, s2_face)])

    def strip_nodes(n: np.ndarray) -> np.ndarray:
        on_bot = n[:, 2] < 1e-9
        s1 = np.abs(n[:, 1] - y1) <= strip_w / 2.0 + 1e-9
        s2 = np.abs(n[:, 1] - y2) <= strip_w / 2.0 + 1e-9
        return on_bot & (s1 | s2)

    def strip1_lateral(n: np.ndarray) -> np.ndarray:
        return (n[:, 2] < 1e-9) & (np.abs(n[:, 1] - y1) <= strip_w / 2.0 + 1e-9)

    constraints = [
        Constraint(strip_nodes, (3,), "girder_strips_uz"),
        Constraint(strip1_lateral, (1, 2), "strip1_lateral"),
    ]
    wheel_pred = _box_pred((wx0, wy0, T - 1e-6), (wx1, wy1, T + 1e-6))
    tr = TractionSpec(wheel_pred, (0.0, 0.0, -pressure), "wheel_patch")

    zt = T
    segs = [
        ((wx0, wy0, zt), (wx1, wy0, zt)),
        ((wx1, wy0, zt), (wx1, wy1, zt)),
        ((wx1, wy1, zt), (wx0, wy1, zt)),
        ((wx0, wy1, zt), (wx0, wy0, zt)),
        # inner edges of the two support strips (reaction lines)
        ((0, y1 + strip_w / 2, 0), (L, y1 + strip_w / 2, 0)),
        ((0, y1 - strip_w / 2, 0), (L, y1 - strip_w / 2, 0)),
        ((0, y2 - strip_w / 2, 0), (L, y2 - strip_w / 2, 0)),
        ((0, y2 + strip_w / 2, 0), (L, y2 + strip_w / 2, 0)),
    ]
    features = [
        FeatureAnchor("wheel_center", wx, wy, T, "load"),
        FeatureAnchor("wheel_edge_x0", wx0, wy, T, "load"),
        FeatureAnchor("wheel_edge_x1", wx1, wy, T, "load"),
        FeatureAnchor("wheel_edge_y0", wx, wy0, T, "load"),
        FeatureAnchor("wheel_edge_y1", wx, wy1, T, "load"),
        FeatureAnchor("midspan", L / 2, B / 2, T / 2, "corner"),
    ]
    for si, ys in ((1, y1), (2, y2)):
        for tag, xs in (("a", L / 6), ("mid", L / 2), ("b", 5 * L / 6)):
            features.append(
                FeatureAnchor(f"girder_strip_{si}_{tag}", xs, ys, 0.0, "support")
            )
    return Problem(
        name="deck_panel",
        dim=3,
        build_geometry=build,
        constraints=constraints,
        tractions=[tr],
        qoi_facet_predicate=wheel_pred,
        material=Material(E=34e3, nu=0.2),  # concrete deck
        h0=T / 1.6,          # 125 mm
        h_ref=T / 4.5,       # ~44 mm background, graded to lines
        h_min=T / 20.0,      # 10 mm
        bbox=(0, 0, 0, L, B, T),
        features=features,
        singular_points=[],
        singular_segments=segs,
        params={"L": L, "B": B, "T": T, "strip_w": strip_w, "strip_off": strip_off,
                "wheel": wheel, "wheel_pos": wheel_pos, "pressure": pressure},
    )


def sample_deck_panel(rng: np.random.Generator) -> Problem:
    wx = float(rng.uniform(700.0, 1700.0))
    wy = float(rng.uniform(500.0, 1100.0))
    return make_deck_panel(
        wheel_pos=(wx, wy),
        wheel=(float(rng.uniform(300.0, 500.0)), float(rng.uniform(200.0, 320.0))),
        pressure=float(rng.uniform(0.7, 1.4)),
    )


# ===========================================================================
# 2-D development substrate
# ===========================================================================


def make_lbracket(
    size: float = 100.0,
    cut_frac_x: float = 0.5,
    cut_frac_y: float = 0.5,
    load: float = -10.0,
) -> Problem:
    """L-shaped bracket: left edge clamped, right edge loaded downward."""

    cx = size * cut_frac_x
    cy = size * cut_frac_y

    def build() -> None:
        import gmsh

        occ = gmsh.model.occ
        outer = occ.addRectangle(0.0, 0.0, 0.0, size, size)
        cut = occ.addRectangle(cx, cy, 0.0, size - cx + 1.0, size - cy + 1.0)
        occ.cut([(2, outer)], [(2, cut)])

    clamp = Constraint(lambda n: n[:, 0] < 1e-9, (1, 2), "clamp")
    load_pred = _box_pred((size - 1e-6, 0.0), (size + 1e-6, cy))
    tr = TractionSpec(load_pred, (0.0, load, 0.0), "tip_load")

    return Problem(
        name="lbracket",
        dim=2,
        build_geometry=build,
        constraints=[clamp],
        tractions=[tr],
        qoi_facet_predicate=load_pred,
        material=Material(),
        h0=size / 8.0,
        h_ref=size / 80.0,
        h_min=size / 512.0,
        bbox=(0.0, 0.0, 0.0, size, size, 0.0),
        features=[
            FeatureAnchor("reentrant_corner", cx, cy, 0.0, "corner"),
            FeatureAnchor("clamp_top", 0.0, size, 0.0, "clamp"),
            FeatureAnchor("clamp_bottom", 0.0, 0.0, 0.0, "clamp"),
            FeatureAnchor("load_edge", size, cy / 2.0, 0.0, "load"),
        ],
        singular_points=[(cx, cy, 0.0)],
        params={"size": size, "cut_frac_x": cut_frac_x, "cut_frac_y": cut_frac_y,
                "load": load},
    )


def sample_lbracket(rng: np.random.Generator) -> Problem:
    return make_lbracket(
        cut_frac_x=float(rng.uniform(0.4, 0.65)),
        cut_frac_y=float(rng.uniform(0.4, 0.65)),
        load=float(rng.uniform(-14.0, -6.0)),
    )


def make_plate_holes(
    width: float = 200.0,
    height: float = 100.0,
    holes: Sequence[tuple[float, float, float]] = ((100.0, 50.0, 12.0),),
    tension: float = 100.0,
) -> Problem:
    """Rectangular plate with circular holes, clamped left, tension right."""

    holes = tuple((float(x), float(y), float(r)) for x, y, r in holes)

    def build() -> None:
        import gmsh

        occ = gmsh.model.occ
        plate = occ.addRectangle(0.0, 0.0, 0.0, width, height)
        tools = [(2, occ.addDisk(x, y, 0.0, r, r)) for x, y, r in holes]
        if tools:
            occ.cut([(2, plate)], tools)

    clamp = Constraint(lambda n: n[:, 0] < 1e-9, (1, 2), "clamp")
    load_pred = _box_pred((width - 1e-6, 0.0), (width + 1e-6, height))
    tr = TractionSpec(load_pred, (tension, 0.0, 0.0), "tension")

    features = [
        FeatureAnchor("clamp_edge", 0.0, height / 2.0, 0.0, "clamp"),
        FeatureAnchor("load_edge", width, height / 2.0, 0.0, "load"),
    ]
    for i, (x, y, r) in enumerate(holes):
        features.append(FeatureAnchor(f"hole_{i}", x, y, 0.0, "hole", r=r))

    return Problem(
        name="plate_holes",
        dim=2,
        build_geometry=build,
        constraints=[clamp],
        tractions=[tr],
        qoi_facet_predicate=load_pred,
        material=Material(),
        h0=height / 5.0,
        h_ref=height / 64.0,
        h_min=height / 400.0,
        bbox=(0.0, 0.0, 0.0, width, height, 0.0),
        features=features,
        params={"width": width, "height": height, "holes": holes, "tension": tension},
    )


def sample_plate_holes(rng: np.random.Generator) -> Problem:
    width, height = 200.0, 100.0
    n_holes = int(rng.integers(1, 4))
    holes: list[tuple[float, float, float]] = []
    attempts = 0
    while len(holes) < n_holes and attempts < 200:
        attempts += 1
        r = float(rng.uniform(7.0, 14.0))
        x = float(rng.uniform(35.0 + r, width - 35.0 - r))
        y = float(rng.uniform(20.0 + r, height - 20.0 - r))
        if all((x - hx) ** 2 + (y - hy) ** 2 > (r + hr + 12.0) ** 2 for hx, hy, hr in holes):
            holes.append((x, y, r))
    tension = float(rng.uniform(60.0, 140.0))
    return make_plate_holes(holes=holes, tension=tension)


def make_bearing_hole(
    W: float = 400.0,
    D: float = 400.0,
    H: float = 120.0,
    patch: tuple[float, float] = (140.0, 140.0),
    offset: tuple[float, float] = (40.0, 0.0),
    pressure: float = 12.0,
    hole_r: float = 24.0,
    hole_gap: float = 25.0,
) -> Problem:
    """Failure-probe family: bearing block with a transverse duct hole.

    A through-going horizontal duct (grout/anchor sleeve) runs along y at
    mid-height, ``hole_gap`` clear of the loaded patch edge.  The rim
    concentration lines (Kirsch extremes at x = hx ± r) exist on the
    drawing but are barely visible to a ZZ indicator on the h0 probe —
    the probe-blindness target.  Load and constraints match the parent
    ``bearing_block`` family, so the parent's supervised model applies
    syntactically while the topology is outside its training support.
    """

    a, b = patch
    ox, oy = offset
    cx, cy = W / 2.0 + ox, D / 2.0 + oy
    px0, px1 = cx - a / 2.0, cx + a / 2.0
    py0, py1 = cy - b / 2.0, cy + b / 2.0
    hx = px0 - hole_gap - hole_r
    hz = H / 2.0

    def build() -> None:
        import gmsh

        occ = gmsh.model.occ
        box = occ.addBox(0.0, 0.0, 0.0, W, D, H)
        duct = occ.addCylinder(hx, 0.0, hz, 0.0, D, 0.0, hole_r)
        occ.cut([(3, box)], [(3, duct)])
        patch_face = occ.addRectangle(px0, py0, H, a, b)
        occ.fragment([(3, box)], [(2, patch_face)])

    bottom = Constraint(lambda n: n[:, 2] < 1e-9, (1, 2, 3), "bottom_fixed")
    patch_pred = _box_pred((px0, py0, H - 1e-6), (px1, py1, H + 1e-6))
    tr = TractionSpec(patch_pred, (0.0, 0.0, -pressure), "girder_patch")

    zt = H
    segs = [
        ((px0, py0, zt), (px1, py0, zt)),
        ((px1, py0, zt), (px1, py1, zt)),
        ((px1, py1, zt), (px0, py1, zt)),
        ((px0, py1, zt), (px0, py0, zt)),
        ((0, 0, 0), (W, 0, 0)),
        ((W, 0, 0), (W, D, 0)),
        ((W, D, 0), (0, D, 0)),
        ((0, D, 0), (0, 0, 0)),
        # duct rim concentration lines (Kirsch extremes, along the axis)
        ((hx - hole_r, 0.0, hz), (hx - hole_r, D, hz)),
        ((hx + hole_r, 0.0, hz), (hx + hole_r, D, hz)),
    ]
    features = [
        FeatureAnchor("patch_center", cx, cy, H, "load"),
        FeatureAnchor("patch_edge_x0", px0, cy, H, "load"),
        FeatureAnchor("patch_edge_x1", px1, cy, H, "load"),
        FeatureAnchor("patch_edge_y0", cx, py0, H, "load"),
        FeatureAnchor("patch_edge_y1", cx, py1, H, "load"),
        FeatureAnchor("bottom_center", W / 2, D / 2, 0.0, "clamp"),
        FeatureAnchor("bottom_edge_y0", W / 2, 0.0, 0.0, "support"),
        FeatureAnchor("bottom_edge_y1", W / 2, D, 0.0, "support"),
        FeatureAnchor("bottom_edge_x0", 0.0, D / 2, 0.0, "support"),
        FeatureAnchor("bottom_edge_x1", W, D / 2, 0.0, "support"),
        # r=0: the x-y radial hole grading does not fit a horizontal duct;
        # the rim lines above carry the reference grading instead.
        FeatureAnchor("duct_hole", hx, D / 2, hz, "hole", r=0.0),
    ]
    return Problem(
        name="bearing_hole",
        dim=3,
        build_geometry=build,
        constraints=[bottom],
        tractions=[tr],
        qoi_facet_predicate=patch_pred,
        material=Material(),
        h0=H / 2.4,
        h_ref=H / 10.0,
        h_min=H / 40.0,
        bbox=(0, 0, 0, W, D, H),
        features=features,
        singular_points=[],
        singular_segments=segs,
        params={"W": W, "D": D, "H": H, "patch": patch, "offset": offset,
                "pressure": pressure, "hole_r": hole_r, "hole_gap": hole_gap},
    )


def sample_bearing_hole(rng: np.random.Generator) -> Problem:
    """Load parameters inside the parent training support; only the duct
    (absent from the parent family) varies — isolates the topology shift."""

    a = float(rng.uniform(130.0, 170.0))
    b = float(rng.uniform(130.0, 170.0))
    ox = float(rng.uniform(10.0, 50.0))
    oy = float(rng.uniform(-40.0, 40.0))
    return make_bearing_hole(
        patch=(a, b),
        offset=(ox, oy),
        pressure=float(rng.uniform(9.0, 15.0)),
        hole_r=float(rng.uniform(22.0, 28.0)),
    )


def make_deck_opening(
    L: float = 2400.0,
    B: float = 1600.0,
    T: float = 200.0,
    strip_w: float = 220.0,
    strip_off: float = 300.0,
    wheel: tuple[float, float] = (400.0, 250.0),
    wheel_pos: tuple[float, float] = (1200.0, 800.0),
    pressure: float = 1.0,
    open_r: float = 80.0,
    open_gap: float = 40.0,
) -> Problem:
    """Failure-probe family: deck panel with a circular service opening.

    A through-thickness opening sits between the wheel patch and the near
    support strip, cutting the load path.  In-plane flow concentrates on
    the rim's x-extremes (vertical lines at x = ox ± r).  Same load and
    strip layout as the parent ``deck_panel`` family.
    """

    y1, y2 = strip_off, B - strip_off
    wa, wb = wheel
    wx, wy = wheel_pos
    wx0, wx1 = wx - wa / 2.0, wx + wa / 2.0
    wy0, wy1 = wy - wb / 2.0, wy + wb / 2.0
    ocx = wx
    ocy = wy0 - open_gap - open_r

    def build() -> None:
        import gmsh

        occ = gmsh.model.occ
        box = occ.addBox(0.0, 0.0, 0.0, L, B, T)
        opening = occ.addCylinder(ocx, ocy, 0.0, 0.0, 0.0, T, open_r)
        occ.cut([(3, box)], [(3, opening)])
        wheel_face = occ.addRectangle(wx0, wy0, T, wa, wb)
        s1_face = occ.addRectangle(0.0, y1 - strip_w / 2.0, 0.0, L, strip_w)
        s2_face = occ.addRectangle(0.0, y2 - strip_w / 2.0, 0.0, L, strip_w)
        occ.fragment([(3, box)], [(2, wheel_face), (2, s1_face), (2, s2_face)])

    def strip_nodes(n: np.ndarray) -> np.ndarray:
        on_bot = n[:, 2] < 1e-9
        s1 = np.abs(n[:, 1] - y1) <= strip_w / 2.0 + 1e-9
        s2 = np.abs(n[:, 1] - y2) <= strip_w / 2.0 + 1e-9
        return on_bot & (s1 | s2)

    def strip1_lateral(n: np.ndarray) -> np.ndarray:
        return (n[:, 2] < 1e-9) & (np.abs(n[:, 1] - y1) <= strip_w / 2.0 + 1e-9)

    constraints = [
        Constraint(strip_nodes, (3,), "girder_strips_uz"),
        Constraint(strip1_lateral, (1, 2), "strip1_lateral"),
    ]
    wheel_pred = _box_pred((wx0, wy0, T - 1e-6), (wx1, wy1, T + 1e-6))
    tr = TractionSpec(wheel_pred, (0.0, 0.0, -pressure), "wheel_patch")

    zt = T
    segs = [
        ((wx0, wy0, zt), (wx1, wy0, zt)),
        ((wx1, wy0, zt), (wx1, wy1, zt)),
        ((wx1, wy1, zt), (wx0, wy1, zt)),
        ((wx0, wy1, zt), (wx0, wy0, zt)),
        ((0, y1 + strip_w / 2, 0), (L, y1 + strip_w / 2, 0)),
        ((0, y1 - strip_w / 2, 0), (L, y1 - strip_w / 2, 0)),
        ((0, y2 - strip_w / 2, 0), (L, y2 - strip_w / 2, 0)),
        ((0, y2 + strip_w / 2, 0), (L, y2 + strip_w / 2, 0)),
        # opening rim concentration lines (through thickness)
        ((ocx - open_r, ocy, 0.0), (ocx - open_r, ocy, T)),
        ((ocx + open_r, ocy, 0.0), (ocx + open_r, ocy, T)),
    ]
    features = [
        FeatureAnchor("wheel_center", wx, wy, T, "load"),
        FeatureAnchor("wheel_edge_x0", wx0, wy, T, "load"),
        FeatureAnchor("wheel_edge_x1", wx1, wy, T, "load"),
        FeatureAnchor("wheel_edge_y0", wx, wy0, T, "load"),
        FeatureAnchor("wheel_edge_y1", wx, wy1, T, "load"),
        FeatureAnchor("midspan", L / 2, B / 2, T / 2, "corner"),
        FeatureAnchor("service_opening", ocx, ocy, T, "hole", r=open_r),
    ]
    for si, ys in ((1, y1), (2, y2)):
        for tag, xs in (("a", L / 6), ("mid", L / 2), ("b", 5 * L / 6)):
            features.append(
                FeatureAnchor(f"girder_strip_{si}_{tag}", xs, ys, 0.0, "support")
            )
    return Problem(
        name="deck_opening",
        dim=3,
        build_geometry=build,
        constraints=constraints,
        tractions=[tr],
        qoi_facet_predicate=wheel_pred,
        material=Material(E=34e3, nu=0.2),
        h0=T / 1.6,
        h_ref=T / 4.5,
        h_min=T / 20.0,
        bbox=(0, 0, 0, L, B, T),
        features=features,
        singular_points=[],
        singular_segments=segs,
        params={"L": L, "B": B, "T": T, "strip_w": strip_w, "strip_off": strip_off,
                "wheel": wheel, "wheel_pos": wheel_pos, "pressure": pressure,
                "open_r": open_r, "open_gap": open_gap},
    )


def sample_deck_opening(rng: np.random.Generator) -> Problem:
    """Load parameters inside the parent training support; only the opening
    (absent from the parent family) varies."""

    wx = float(rng.uniform(1000.0, 1400.0))
    wy = float(rng.uniform(820.0, 900.0))
    return make_deck_opening(
        wheel_pos=(wx, wy),
        wheel=(float(rng.uniform(350.0, 450.0)), float(rng.uniform(220.0, 300.0))),
        pressure=float(rng.uniform(0.9, 1.3)),
        open_r=float(rng.uniform(70.0, 90.0)),
    )


def analytic_load_resultant(problem: Problem) -> np.ndarray:
    """Nominal traction resultant p×A (2-D: × thickness).  Gate G7 target."""

    p = problem.params
    t = problem.material.thickness
    if problem.name in ("bearing_block", "bearing_hole"):
        a, b = p["patch"]
        return np.array([0.0, 0.0, -float(p["pressure"]) * a * b])
    if problem.name in ("deck_panel", "deck_opening"):
        wa, wb = p["wheel"]
        return np.array([0.0, 0.0, -float(p["pressure"]) * wa * wb])
    if problem.name == "lbracket":
        size = float(p["size"])
        cy = size * float(p["cut_frac_y"])
        load = float(p["load"])
        return np.array([0.0, load * cy * t, 0.0])
    if problem.name == "plate_holes":
        height = float(p["height"])
        tension = float(p["tension"])
        return np.array([tension * height * t, 0.0, 0.0])
    raise ValueError(f"no analytic resultant for {problem.name}")


PROBLEM_FACTORIES = {
    "bearing_block": make_bearing_block,
    "deck_panel": make_deck_panel,
    "lbracket": make_lbracket,
    "plate_holes": make_plate_holes,
    "bearing_hole": make_bearing_hole,
    "deck_opening": make_deck_opening,
}

def sample_bearing_block_ood(rng: np.random.Generator) -> Problem:
    """Out-of-distribution bearing instance: every parameter lies outside
    the training sampler's support (patch 100–180, offset ±70, p 8–16)."""

    a = float(rng.uniform(190.0, 205.0))
    b = float(rng.uniform(190.0, 205.0))
    ox = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(75.0, 82.0))
    oy = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(75.0, 82.0))
    return make_bearing_block(
        patch=(a, b), offset=(ox, oy), pressure=float(rng.uniform(18.0, 22.0))
    )


def sample_deck_panel_ood(rng: np.random.Generator) -> Problem:
    """Out-of-distribution deck instance: wheel position, footprint, and
    pressure all outside the training support (x 700–1700, y 500–1100,
    wheel 300–500 × 200–320, p 0.7–1.4)."""

    wx = float(rng.uniform(1850.0, 2000.0))
    wy = float(rng.uniform(1150.0, 1250.0))
    return make_deck_panel(
        wheel_pos=(wx, wy),
        wheel=(float(rng.uniform(510.0, 560.0)), float(rng.uniform(330.0, 360.0))),
        pressure=float(rng.uniform(1.6, 2.0)),
    )


SAMPLERS = {
    "bearing_block": sample_bearing_block,
    "deck_panel": sample_deck_panel,
    "lbracket": sample_lbracket,
    "plate_holes": sample_plate_holes,
    "bearing_hole": sample_bearing_hole,
    "deck_opening": sample_deck_opening,
}

OOD_SAMPLERS = {
    "bearing_block": sample_bearing_block_ood,
    "deck_panel": sample_deck_panel_ood,
}
