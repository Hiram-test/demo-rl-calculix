"""Benchmark problem definitions.

Every problem is a 2-D plane-stress domain built with the Gmsh OCC API.
A ``Problem`` carries geometry construction, boundary conditions, loads,
the quantity of interest, and named feature anchors that vision/region
methods may use to name regions.

Two parametric families are provided:

* ``lbracket`` -- L-shaped bracket with a re-entrant corner (classic AFEM
  singularity benchmark; energy-norm rate of uniform meshing is limited
  by the corner singularity, so correct adaptive methods must win).
* ``plate_holes`` -- rectangular plate with 1-3 circular holes under
  tension (smooth but with strong stress-concentration hotspots).

Instance samplers generate randomized train/test instances for the
learning-based methods.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Material:
    E: float = 210e3          # MPa
    nu: float = 0.3
    thickness: float = 1.0    # mm


@dataclass(frozen=True)
class FeatureAnchor:
    """Named geometric feature usable for region naming."""

    name: str
    x: float
    y: float
    kind: str  # "corner" | "hole" | "clamp" | "load"
    r: float = 0.0  # characteristic radius (holes)


@dataclass(frozen=True)
class TractionSpec:
    """Traction applied on boundary edges whose midpoints satisfy the predicate.

    ``value`` is a surface traction in MPa (force per area); nodal forces
    are assembled as t * edge_length * thickness, half to each edge node.
    """

    edge_predicate: Callable[[np.ndarray], np.ndarray]
    value: tuple[float, float]
    name: str = "load"


@dataclass
class Problem:
    name: str
    build_geometry: Callable[[], None]
    clamp_predicate: Callable[[np.ndarray], np.ndarray]
    tractions: Sequence[TractionSpec]
    qoi_edge_predicate: Callable[[np.ndarray], np.ndarray]
    material: Material
    h0: float                 # initial/background mesh size
    h_ref: float              # reference-mesh size (error origin)
    h_min: float              # hard floor for any method
    bbox: tuple[float, float, float, float]
    features: list[FeatureAnchor] = field(default_factory=list)
    singular_points: list[tuple[float, float]] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def instance_id(self) -> str:
        blob = repr(sorted(self.params.items())).encode()
        return f"{self.name}-{hashlib.sha256(blob).hexdigest()[:8]}"


# ---------------------------------------------------------------------------
# helpers


def _edge_mid_between(lo: np.ndarray, hi: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    def pred(mid: np.ndarray) -> np.ndarray:
        mid = np.atleast_2d(mid)
        return np.all((mid >= lo - 1e-9) & (mid <= hi + 1e-9), axis=1)

    return pred


# ---------------------------------------------------------------------------
# L-bracket family


def make_lbracket(
    size: float = 100.0,
    cut_frac_x: float = 0.5,
    cut_frac_y: float = 0.5,
    load: float = -10.0,
) -> Problem:
    """L-shaped bracket.

    Domain: [0,size]^2 minus the upper-right rectangle
    [cx,size] x [cy,size].  Re-entrant 270-degree corner at (cx, cy).
    Left edge clamped, right edge (below the cut) loaded downward.
    """

    cx = size * cut_frac_x
    cy = size * cut_frac_y

    def build() -> None:
        import gmsh

        occ = gmsh.model.occ
        outer = occ.addRectangle(0.0, 0.0, 0.0, size, size)
        cut = occ.addRectangle(cx, cy, 0.0, size - cx + 1.0, size - cy + 1.0)
        occ.cut([(2, outer)], [(2, cut)])

    clamp = lambda nodes: nodes[:, 0] < 1e-9
    load_pred = _edge_mid_between((size - 1e-6, 0.0), (size + 1e-6, cy))
    tr = TractionSpec(load_pred, (0.0, load), name="tip_load")

    return Problem(
        name="lbracket",
        build_geometry=build,
        clamp_predicate=clamp,
        tractions=[tr],
        qoi_edge_predicate=load_pred,
        material=Material(),
        h0=size / 8.0,
        h_ref=size / 80.0,
        h_min=size / 512.0,
        bbox=(0.0, 0.0, size, size),
        features=[
            FeatureAnchor("reentrant_corner", cx, cy, "corner"),
            FeatureAnchor("clamp_top", 0.0, size, "clamp"),
            FeatureAnchor("clamp_bottom", 0.0, 0.0, "clamp"),
            FeatureAnchor("load_edge", size, cy / 2.0, "load"),
        ],
        singular_points=[(cx, cy)],
        params={
            "size": size,
            "cut_frac_x": cut_frac_x,
            "cut_frac_y": cut_frac_y,
            "load": load,
        },
    )


def sample_lbracket(rng: np.random.Generator) -> Problem:
    return make_lbracket(
        cut_frac_x=float(rng.uniform(0.4, 0.65)),
        cut_frac_y=float(rng.uniform(0.4, 0.65)),
        load=float(rng.uniform(-14.0, -6.0)),
    )


# ---------------------------------------------------------------------------
# plate-with-holes family


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

    clamp = lambda nodes: nodes[:, 0] < 1e-9
    load_pred = _edge_mid_between((width - 1e-6, 0.0), (width + 1e-6, height))
    tr = TractionSpec(load_pred, (tension, 0.0), name="tension")

    features = [
        FeatureAnchor("clamp_edge", 0.0, height / 2.0, "clamp"),
        FeatureAnchor("load_edge", width, height / 2.0, "load"),
    ]
    for i, (x, y, r) in enumerate(holes):
        features.append(FeatureAnchor(f"hole_{i}", x, y, "hole", r=r))

    return Problem(
        name="plate_holes",
        build_geometry=build,
        clamp_predicate=clamp,
        tractions=[tr],
        qoi_edge_predicate=load_pred,
        material=Material(),
        h0=height / 5.0,
        h_ref=height / 64.0,
        h_min=height / 400.0,
        bbox=(0.0, 0.0, width, height),
        features=features,
        singular_points=[],
        params={
            "width": width,
            "height": height,
            "holes": holes,
            "tension": tension,
        },
    )


def sample_plate_holes(rng: np.random.Generator) -> Problem:
    """Randomized plate instance with 1-3 non-overlapping holes."""

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


PROBLEM_FACTORIES = {
    "lbracket": make_lbracket,
    "plate_holes": make_plate_holes,
}

SAMPLERS = {
    "lbracket": sample_lbracket,
    "plate_holes": sample_plate_holes,
}
