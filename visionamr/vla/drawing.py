"""Irregular vision-drawn regions: polygons on ortho views, then a size.

The eye draws a non-box outline (top / front / side, or a section cut)
and then assigns one fineness to that region.  Unpainted volume still
gets an eye-assigned remainder size; it is not silently dumped at h0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from ..geometry import Problem


@dataclass(frozen=True)
class DrawnRegion:
    """One AI-drawn irregular region: a view-plane polygon and an eye size."""

    name: str
    h: float
    view: str
    polygon: tuple[tuple[float, float], ...]
    origin: str = "vision"
    plane: str | None = None
    cut: float | None = None
    slab: float | None = None

    def poly_array(self) -> np.ndarray:
        return np.asarray(self.polygon, dtype=float)

VIEW_AXES = {
    "top": (0, 1),
    "front": (0, 2),
    "side": (1, 2),
}

PLANE_AXES = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}

ORTHO_PLANE = {"top": "xy", "front": "xz", "side": "yz"}
REMAINDER_NAMES = frozenset({"field", "bulk", "remainder", "unpainted"})


def drawing_plane(region: DrawnRegion) -> str:
    if region.view == "section":
        plane = region.plane or "xy"
    else:
        plane = ORTHO_PLANE.get(region.view, "xy")
    if plane not in PLANE_AXES:
        raise ValueError(f"unknown plane {plane!r}")
    return plane


def drawing_axes(region: DrawnRegion) -> tuple[int, int]:
    return PLANE_AXES[drawing_plane(region)]


def unused_axis(plane: str) -> int:
    ax0, ax1 = PLANE_AXES[plane]
    return ({0, 1, 2} - {ax0, ax1}).pop()


def section_slab(region: DrawnRegion, problem: Problem) -> float:
    if region.slab is not None:
        return float(region.slab)
    plane = drawing_plane(region)
    k = unused_axis(plane)
    span = float(problem.bbox[k + 3] - problem.bbox[k])
    return 0.28 * max(span, 1e-9)


def origin_for_name(name: str) -> str:
    return "coarse" if name.lower() in REMAINDER_NAMES else "vision"


def point_hits_drawing(xyz: np.ndarray, region: DrawnRegion, problem: Problem) -> bool:
    """True if a model point falls in this drawing (section = slab around the cut)."""

    pt = np.asarray(xyz, float).reshape(3)
    plane = drawing_plane(region)
    ax0, ax1 = PLANE_AXES[plane]
    inside = bool(points_in_poly(pt[None, [ax0, ax1]], region.poly_array())[0])
    if not inside:
        return False
    if region.view == "section":
        k = unused_axis(plane)
        return abs(pt[k] - float(region.cut)) <= section_slab(region, problem)
    return True


def size_at_xyz(
    xyz: np.ndarray,
    drawings: list[DrawnRegion],
    remainder_h: float,
    problem: Problem,
) -> float:
    """Eye size at a point: finest hitting drawing, else the remainder assignment."""

    hits = [d.h for d in drawings if point_hits_drawing(xyz, d, problem)]
    return float(min(hits) if hits else remainder_h)


def drawings_size_fn(drawings: list[DrawnRegion], remainder_h: float, problem: Problem):
    """Gmsh callback: drawings + remainder, no background mesh, no solve."""

    def fn(x, y, z):
        return size_at_xyz((x, y, z), drawings, remainder_h, problem)

    return fn


def require_fineness(item: dict, label: str) -> float:
    if "fineness_fraction" not in item:
        raise ValueError(f"{label} missing fineness_fraction")
    return float(np.clip(item["fineness_fraction"], 0.1, 1.0))


def poly_tuple(pts: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in np.asarray(pts, float).reshape(-1, 2))


def irregular_halo_2d(
    center: np.ndarray, radius: float, *, n: int = 10, phase: float = 0.0
) -> np.ndarray:
    """Non-rectangular, non-elliptic blob around a point."""

    c = np.asarray(center, float).reshape(2)
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radius * (
        1.0
        + 0.35 * np.sin(2.0 * ang + phase)
        + 0.18 * np.cos(3.0 * ang + 0.7 * phase)
    )
    return c + np.column_stack((r * np.cos(ang), r * np.sin(ang)))


def irregular_from_points(pts2: np.ndarray, pad: float) -> np.ndarray:
    """Convex hull of the points plus a sheared halo so rectangles do not stay boxes."""

    pts2 = np.asarray(pts2, float).reshape(-1, 2)
    if len(pts2) < 3:
        c = pts2.mean(axis=0) if len(pts2) else np.zeros(2)
        return irregular_halo_2d(c, max(pad, 1e-6))
    c = pts2.mean(axis=0)
    extra = []
    for p in pts2:
        v = p - c
        extra.append(c + 1.15 * v + 0.12 * np.array([-v[1], v[0]]))
    allp = np.vstack([pts2, extra])
    try:
        hull = ConvexHull(allp)
        return allp[hull.vertices]
    except Exception:  # noqa: BLE001  -- degenerate clouds fall back to a halo
        return irregular_halo_2d(c, max(pad, float(np.std(allp)) + 1e-6))


def points_in_poly(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized even-odd ray cast. ``xy`` is (N, 2), ``poly`` is (M, 2)."""

    xy = np.asarray(xy, float).reshape(-1, 2)
    poly = np.asarray(poly, float).reshape(-1, 2)
    if len(xy) == 0 or len(poly) < 3:
        return np.zeros(len(xy), dtype=bool)
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = (yi > y) != (yj > y)
        den = yj - yi
        den = np.where(np.abs(den) < 1e-30, 1e-30, den)
        xinters = (xj - xi) * (y - yi) / den + xi
        inside ^= cond & (x < xinters)
        j = i
    return inside


def drawing_centroid_xyz(region: DrawnRegion, problem: Problem) -> tuple[float, float, float]:
    c = np.asarray(region.polygon, float).mean(axis=0)
    b = problem.bbox
    xyz = [0.5 * (b[k] + b[k + 3]) for k in range(3)]
    plane = drawing_plane(region)
    ax0, ax1 = PLANE_AXES[plane]
    xyz[ax0] = float(c[0])
    xyz[ax1] = float(c[1])
    if region.view == "section" and region.cut is not None:
        xyz[unused_axis(plane)] = float(region.cut)
    if problem.dim == 2:
        xyz[2] = 0.0
    lo = problem.bbox[:3]
    hi = problem.bbox[3:]
    xyz = [min(max(xyz[k], lo[k]), hi[k]) for k in range(3)]
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def view_for_feature(feature, problem: Problem) -> str:
    if problem.dim == 2:
        return "top"
    if feature.kind == "hole" and problem.name == "bearing_hole":
        return "front"
    return "top"


def halo_drawing(
    name: str,
    xyz: np.ndarray,
    h: float,
    problem: Problem,
    *,
    view: str = "top",
    radius: float | None = None,
    phase: float = 0.0,
    origin: str = "vision",
) -> DrawnRegion:
    ax0, ax1 = VIEW_AXES[view]
    r = radius if radius is not None else 0.06 * problem.diameter
    poly = irregular_halo_2d(np.asarray(xyz, float)[[ax0, ax1]], r, phase=phase)
    return DrawnRegion(name, float(h), view, poly_tuple(poly), origin)


def markup_from_spec(spec: dict, problem: Problem, max_regions: int = 12) -> list[DrawnRegion]:
    """Parse VLM JSON: drawn regions preferred; old seed lists become irregular blobs."""

    if not isinstance(spec, dict):
        raise ValueError("JSON must be an object")
    drawings: list[DrawnRegion] = []
    raw_regions = spec.get("regions")
    raw_seeds = spec.get("seeds")
    if isinstance(raw_regions, list) and raw_regions:
        for i, item in enumerate(raw_regions[:max_regions]):
            if not isinstance(item, dict):
                raise ValueError(f"region {i} is not an object")
            view = str(item.get("view", "top"))
            if view not in VIEW_AXES and view != "section":
                raise ValueError(f"region {i} unknown view {view!r}")
            plane = item.get("plane")
            cut = item.get("cut")
            slab = item.get("slab")
            if view == "section":
                plane = str(plane or "xy")
                if plane not in PLANE_AXES:
                    raise ValueError(f"region {i} unknown section plane {plane!r}")
                if cut is None:
                    raise ValueError(f"region {i} section needs cut")
                cut = float(cut)
            elif plane is not None:
                plane = str(plane)
            poly = item.get("polygon")
            if not isinstance(poly, list) or len(poly) < 3:
                raise ValueError(f"region {i} needs a polygon of at least 3 points")
            pts = []
            for p in poly:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    raise ValueError(f"region {i} polygon vertex is not a pair")
                pts.append((float(p[0]), float(p[1])))
            frac = require_fineness(item, f"region {i}")
            name = str(item.get("name", f"llm_region_{i}"))[:48]
            drawings.append(
                DrawnRegion(
                    name,
                    frac * problem.h0,
                    view,
                    tuple(pts),
                    origin_for_name(name),
                    plane=plane,
                    cut=cut if cut is None else float(cut),
                    slab=None if slab is None else float(slab),
                )
            )
    elif isinstance(raw_seeds, list) and raw_seeds:
        lo = np.array(problem.bbox[:3], dtype=float)
        hi = np.array(problem.bbox[3:], dtype=float)
        for i, item in enumerate(raw_seeds[:max_regions]):
            if not isinstance(item, dict):
                raise ValueError(f"seed {i} is not an object")
            try:
                xyz = np.array(
                    [float(item["x"]), float(item.get("y", 0.0)), float(item.get("z", 0.0))]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"seed {i} missing numeric x/y/z") from exc
            xyz = np.clip(xyz, lo, hi)
            if problem.dim == 2:
                xyz[2] = 0.0
            frac = require_fineness(item, f"seed {i}")
            name = str(item.get("name", f"llm_seed_{i}"))[:48]
            drawings.append(
                halo_drawing(
                    name, xyz, frac * problem.h0, problem,
                    view="top", radius=0.07 * problem.diameter, phase=0.4 * i,
                    origin=origin_for_name(name),
                )
            )
    else:
        raise ValueError("JSON missing 'regions' or 'seeds'")
    if not drawings:
        raise ValueError("VLM returned no usable regions")
    return drawings
