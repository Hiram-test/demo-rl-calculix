from __future__ import annotations

from collections import Counter
from math import cos, pi, sin
from typing import Callable

import numpy as np

from .fem import Mesh


def _boundary_edges(elements: np.ndarray) -> list[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    directed: dict[tuple[int, int], tuple[int, int]] = {}
    for conn in elements:
        for i in range(4):
            a = int(conn[i])
            b = int(conn[(i + 1) % 4])
            key = (min(a, b), max(a, b))
            counts[key] += 1
            directed[key] = (a, b)
    return [directed[k] for k, c in counts.items() if c == 1]


def _classify_rect_outer_edges(nodes: np.ndarray, edges: list[tuple[int, int]], tol: float = 1e-8) -> dict[str, list[tuple[int, int]]]:
    xmin, ymin = nodes.min(axis=0)
    xmax, ymax = nodes.max(axis=0)
    out = {"left": [], "right": [], "bottom": [], "top": [], "boundary": list(edges)}
    scale = max(xmax - xmin, ymax - ymin, 1.0)
    eps = tol * scale
    for e in edges:
        p = nodes[list(e)]
        if np.all(np.abs(p[:, 0] - xmin) <= eps):
            out["left"].append(e)
        elif np.all(np.abs(p[:, 0] - xmax) <= eps):
            out["right"].append(e)
        elif np.all(np.abs(p[:, 1] - ymin) <= eps):
            out["bottom"].append(e)
        elif np.all(np.abs(p[:, 1] - ymax) <= eps):
            out["top"].append(e)
    return out


def structured_rect_mesh(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    *,
    cell_keep: Callable[[float, float], bool] | None = None,
) -> Mesh:
    x = np.asarray(x_coords, dtype=float)
    y = np.asarray(y_coords, dtype=float)
    nx = len(x) - 1
    ny = len(y) - 1
    nodes = np.array([(xx, yy) for yy in y for xx in x], dtype=float)

    elements: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            cx = 0.5 * (x[i] + x[i + 1])
            cy = 0.5 * (y[j] + y[j + 1])
            if cell_keep is not None and not cell_keep(cx, cy):
                continue
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n3 = (j + 1) * (nx + 1) + i
            n2 = n3 + 1
            elements.append([n0, n1, n2, n3])

    elems = np.asarray(elements, dtype=int)
    used = np.unique(elems.ravel())
    remap = -np.ones(len(nodes), dtype=int)
    remap[used] = np.arange(len(used))
    nodes = nodes[used]
    elems = remap[elems]
    bedges = _boundary_edges(elems)
    edge_sets = _classify_rect_outer_edges(nodes, bedges)
    return Mesh(nodes=nodes, elements=elems, edge_sets=edge_sets)


def rectangle_mesh(length: float, height: float, nx: int, ny: int, *, centered_y: bool = True) -> Mesh:
    x = np.linspace(0.0, length, nx + 1)
    if centered_y:
        y = np.linspace(-height / 2.0, height / 2.0, ny + 1)
    else:
        y = np.linspace(0.0, height, ny + 1)
    mesh = structured_rect_mesh(x, y)
    mesh.metadata.update({"kind": "rectangle", "length": length, "height": height, "nx": nx, "ny": ny})
    return mesh


def _outer_rect_radius(theta: float, half_w: float, half_h: float) -> float:
    c = abs(cos(theta))
    s = abs(sin(theta))
    rx = np.inf if c < 1e-14 else half_w / c
    ry = np.inf if s < 1e-14 else half_h / s
    return float(min(rx, ry))


def _rounded_rect_sdf(x: float, y: float, half_w: float, half_h: float, radius: float) -> float:
    if radius <= 0.0:
        return max(abs(x) - half_w, abs(y) - half_h)
    bx = half_w - radius
    by = half_h - radius
    qx = abs(x) - bx
    qy = abs(y) - by
    ox = max(qx, 0.0)
    oy = max(qy, 0.0)
    return float(np.hypot(ox, oy) + min(max(qx, qy), 0.0) - radius)


def _inner_rounded_rect_radius(theta: float, half_w: float, half_h: float, radius: float) -> float:
    if radius <= 1e-12:
        c = abs(cos(theta))
        s = abs(sin(theta))
        rx = np.inf if c < 1e-14 else half_w / c
        ry = np.inf if s < 1e-14 else half_h / s
        return float(min(rx, ry))
    lo = 0.0
    hi = np.hypot(half_w, half_h) * 1.5 + radius
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        value = _rounded_rect_sdf(mid * cos(theta), mid * sin(theta), half_w, half_h, radius)
        if value < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def radial_hole_mesh(
    width: float,
    height: float,
    inner_radius_fn: Callable[[float], float],
    ntheta: int,
    nr: int,
    *,
    cluster: float = 3.0,
    kind: str = "radial_hole",
) -> Mesh:
    if ntheta % 8 != 0:
        raise ValueError("ntheta must be divisible by 8 so rectangle corners are represented exactly")
    half_w = width / 2.0
    half_h = height / 2.0
    theta = np.linspace(0.0, 2.0 * pi, ntheta, endpoint=False)
    if cluster <= 1e-12:
        svals = np.linspace(0.0, 1.0, nr + 1)
    else:
        raw = np.linspace(0.0, 1.0, nr + 1)
        svals = (np.exp(cluster * raw) - 1.0) / (np.exp(cluster) - 1.0)

    nodes: list[tuple[float, float]] = []
    for s in svals:
        for th in theta:
            rin = inner_radius_fn(float(th))
            rout = _outer_rect_radius(float(th), half_w, half_h)
            r = rin + s * (rout - rin)
            nodes.append((r * cos(th), r * sin(th)))

    elements: list[list[int]] = []
    for j in range(nr):
        for i in range(ntheta):
            ip = (i + 1) % ntheta
            inner_i = j * ntheta + i
            inner_ip = j * ntheta + ip
            outer_i = (j + 1) * ntheta + i
            outer_ip = (j + 1) * ntheta + ip
            elements.append([inner_i, outer_i, outer_ip, inner_ip])

    arr_nodes = np.asarray(nodes, dtype=float)
    arr_elems = np.asarray(elements, dtype=int)
    inner_edges = [(i, (i + 1) % ntheta) for i in range(ntheta)]
    outer_offset = nr * ntheta
    outer_edges = [
        (outer_offset + i, outer_offset + ((i + 1) % ntheta)) for i in range(ntheta)
    ]
    edge_sets = _classify_rect_outer_edges(arr_nodes, outer_edges, tol=1e-7)
    edge_sets["inner_hole"] = inner_edges
    edge_sets["boundary"] = inner_edges + outer_edges
    mesh = Mesh(nodes=arr_nodes, elements=arr_elems, edge_sets=edge_sets)
    mesh.metadata.update({"kind": kind, "width": width, "height": height, "ntheta": ntheta, "nr": nr})
    return mesh


def circular_hole_mesh(width: float, height: float, radius: float, ntheta: int, nr: int, *, cluster: float = 3.0) -> Mesh:
    mesh = radial_hole_mesh(
        width,
        height,
        lambda _theta: radius,
        ntheta,
        nr,
        cluster=cluster,
        kind="circular_hole",
    )
    mesh.metadata["hole_radius"] = radius
    return mesh


def rounded_rect_hole_mesh(
    width: float,
    height: float,
    hole_half_width: float,
    hole_half_height: float,
    corner_radius: float,
    ntheta: int,
    nr: int,
    *,
    cluster: float = 3.0,
) -> Mesh:
    if corner_radius < 0.0 or corner_radius > min(hole_half_width, hole_half_height):
        raise ValueError("invalid corner radius")
    mesh = radial_hole_mesh(
        width,
        height,
        lambda th: _inner_rounded_rect_radius(th, hole_half_width, hole_half_height, corner_radius),
        ntheta,
        nr,
        cluster=cluster,
        kind="rounded_rect_hole",
    )
    mesh.metadata.update(
        {
            "hole_half_width": hole_half_width,
            "hole_half_height": hole_half_height,
            "corner_radius": corner_radius,
        }
    )
    return mesh


def central_crack_mesh(width: float, height: float, half_crack: float, nx: int, ny: int) -> Mesh:
    if nx % 10 != 0 or ny % 2 != 0:
        raise ValueError("central crack mesh expects nx divisible by 10 and even ny")
    x = np.linspace(-width / 2.0, width / 2.0, nx + 1)
    y = np.linspace(-height / 2.0, height / 2.0, ny + 1)
    if np.min(np.abs(x - half_crack)) > 1e-9 or np.min(np.abs(x + half_crack)) > 1e-9:
        raise ValueError("crack tips must align with x grid")
    zero_j = int(np.argmin(np.abs(y)))
    if abs(y[zero_j]) > 1e-12:
        raise ValueError("y=0 must be present")

    registry: dict[tuple[int, int, str], int] = {}
    coords: list[tuple[float, float]] = []

    def node(ix: int, iy: int, region: str) -> int:
        xx = float(x[ix])
        yy = float(y[iy])
        side = "shared"
        if iy == zero_j and abs(xx) < half_crack - 1e-10:
            side = region
        key = (ix, iy, side)
        if key not in registry:
            registry[key] = len(coords)
            coords.append((xx, yy))
        return registry[key]

    elements: list[list[int]] = []
    for j in range(ny):
        region = "lower" if y[j + 1] <= 0.0 + 1e-12 else "upper"
        for i in range(nx):
            n0 = node(i, j, region)
            n1 = node(i + 1, j, region)
            n2 = node(i + 1, j + 1, region)
            n3 = node(i, j + 1, region)
            elements.append([n0, n1, n2, n3])

    arr_nodes = np.asarray(coords, dtype=float)
    arr_elems = np.asarray(elements, dtype=int)
    bedges = _boundary_edges(arr_elems)
    edge_sets = _classify_rect_outer_edges(arr_nodes, bedges)
    crack_upper: list[tuple[int, int]] = []
    crack_lower: list[tuple[int, int]] = []
    for e in bedges:
        p = arr_nodes[list(e)]
        if np.all(np.abs(p[:, 1]) < 1e-9) and np.max(np.abs(p[:, 0])) <= half_crack + 1e-9:
            if p[1, 0] > p[0, 0]:
                crack_lower.append(e)
            else:
                crack_upper.append(e)
    edge_sets["crack_upper"] = crack_upper
    edge_sets["crack_lower"] = crack_lower
    mesh = Mesh(nodes=arr_nodes, elements=arr_elems, edge_sets=edge_sets)
    mesh.metadata.update(
        {"kind": "central_crack", "width": width, "height": height, "half_crack": half_crack, "nx": nx, "ny": ny}
    )
    return mesh


def select_edges_by_midpoint(mesh: Mesh, edge_set: str, predicate: Callable[[float, float], bool]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for edge in mesh.edge_sets.get(edge_set, []):
        mid = mesh.nodes[list(edge)].mean(axis=0)
        if predicate(float(mid[0]), float(mid[1])):
            out.append(edge)
    return out


def nodes_on_coordinate(mesh: Mesh, *, x: float | None = None, y: float | None = None, tol: float = 1e-8) -> np.ndarray:
    mask = np.ones(len(mesh.nodes), dtype=bool)
    scale = max(np.ptp(mesh.nodes[:, 0]), np.ptp(mesh.nodes[:, 1]), 1.0)
    eps = tol * scale
    if x is not None:
        mask &= np.abs(mesh.nodes[:, 0] - x) <= eps
    if y is not None:
        mask &= np.abs(mesh.nodes[:, 1] - y) <= eps
    return np.flatnonzero(mask)
