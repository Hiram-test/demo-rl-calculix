"""Rendering: field views for the vision head and evidence figures.

2-D problems render as a single field plot; 3-D problems render as three
orthographic surface views (top / front / side), which is also what the
multimodal vision head receives (GReFEM-style ortho views).
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import PolyCollection

from .fem_post import PostState
from .geometry import Problem

_VIEWS = [
    ("top (x-y)", (0, 1), 2, +1),
    ("front (x-z)", (0, 2), 1, -1),
    ("side (y-z)", (1, 2), 0, +1),
]


def _surface_view(ax, mesh, values, axes_pair, depth_axis, depth_sign, cmap="turbo",
                  categorical=False, vmax=None):
    bf = mesh.boundary_facets
    owners = mesh.boundary_facet_owners
    pts = mesh.nodes[bf][:, :, axes_pair]
    depth = mesh.nodes[bf][:, :, depth_axis].mean(axis=1) * depth_sign
    order = np.argsort(depth)
    if values.shape[0] == mesh.n_nodes:
        fvals = values[bf].mean(axis=1)
    else:
        fvals = values[owners]
    pc = PolyCollection(
        pts[order],
        array=fvals[order],
        cmap=cmap,
        edgecolors="k",
        linewidths=0.1,
    )
    if vmax is not None:
        pc.set_clim(0, vmax)
    ax.add_collection(pc)
    lo = mesh.nodes[:, axes_pair].min(axis=0)
    hi = mesh.nodes[:, axes_pair].max(axis=0)
    pad = 0.03 * np.linalg.norm(hi - lo)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    return pc


def render_field_png(
    problem: Problem, post: PostState, path: Path | None = None, dpi: int = 130
) -> bytes:
    """Von Mises field with axes in model units (the vision input)."""

    mesh = post.mesh
    if mesh.dim == 2:
        tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.cells)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        tc = ax.tripcolor(tri, post.vm_node, shading="gouraud", cmap="turbo")
        ax.triplot(tri, lw=0.15, color="k", alpha=0.25)
        fig.colorbar(tc, ax=ax, label="von Mises [MPa]")
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"{problem.name}: response field")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        vmax = float(post.vm_node.max())
        for ax, (title, pair, dax, sgn) in zip(axes, _VIEWS):
            pc = _surface_view(ax, mesh, post.vm_node, pair, dax, sgn, vmax=vmax)
            ax.set_title(f"{problem.name}: {title}")
            ax.set_xlabel("xyz"[pair[0]] + " [mm]")
            ax.set_ylabel("xyz"[pair[1]] + " [mm]")
        fig.colorbar(pc, ax=axes, label="von Mises [MPa]", shrink=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    data = buf.getvalue()
    if path is not None:
        Path(path).write_bytes(data)
    return data


def plot_partition(
    problem: Problem,
    post: PostState,
    labels: np.ndarray,
    seeds,
    path: Path,
    title: str = "",
) -> None:
    """Partition regions (element labels) with seed markers."""

    mesh = post.mesh
    R = max(int(labels.max()) + 1, 1)
    if mesh.dim == 2:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        pts = mesh.nodes[mesh.cells][:, :, :2]
        pc = PolyCollection(
            pts, array=labels.astype(float), cmap="tab20",
            edgecolors="k", linewidths=0.1,
        )
        pc.set_clim(-0.5, 19.5)
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect("equal")
        for s in seeds:
            ax.plot(*s.point()[:2], "k*", ms=10, mec="w")
            ax.annotate(f"{s.name}\nh={s.h:.2f}", s.point()[:2], fontsize=6,
                        color="k", va="bottom")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(title)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        lab = labels.astype(float)
        for ax, (vtitle, pair, dax, sgn) in zip(axes, _VIEWS):
            pc = _surface_view(ax, mesh, lab, pair, dax, sgn, cmap="tab20")
            pc.set_clim(-0.5, 19.5)
            for s in seeds:
                p = s.point()[list(pair)]
                ax.plot(*p, "k*", ms=9, mec="w")
                ax.annotate(s.name, p, fontsize=5, va="bottom")
            ax.set_title(vtitle)
            ax.set_xlabel("xyz"[pair[0]] + " [mm]")
            ax.set_ylabel("xyz"[pair[1]] + " [mm]")
        fig.suptitle(title)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_mesh(mesh, path: Path, title: str = "", values=None, label: str = "") -> None:
    if mesh.dim == 2:
        tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.cells)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        if values is not None:
            tc = ax.tripcolor(tri, values, cmap="viridis")
            fig.colorbar(tc, ax=ax, label=label)
        ax.triplot(tri, lw=0.2, color="k", alpha=0.6)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        vals = values if values is not None else mesh.cell_sizes
        for ax, (vtitle, pair, dax, sgn) in zip(axes, _VIEWS):
            pc = _surface_view(ax, mesh, np.asarray(vals), pair, dax, sgn,
                               cmap="viridis")
            ax.set_title(vtitle)
            ax.set_xlabel("xyz"[pair[0]] + " [mm]")
            ax.set_ylabel("xyz"[pair[1]] + " [mm]")
        fig.colorbar(pc, ax=axes, label=label or "element size [mm]", shrink=0.8)
        fig.suptitle(title)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


METHOD_STYLE = {
    "uniform": dict(color="0.45", marker="s", ls="--"),
    "dorfler_zz": dict(color="tab:blue", marker="o", ls="-"),
    "local_prediction": dict(color="tab:green", marker="^", ls="-"),
    "vla": dict(color="tab:red", marker="*", ls="-", ms=11),
    "rl_dqn": dict(color="tab:purple", marker="D", ls="-"),
    "supervised": dict(color="tab:orange", marker="v", ls="-"),
}


def plot_error_curves(records: list, path: Path, *, x: str = "n_equations",
                      y: str = "e_energy", title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    by_method: dict[str, list] = {}
    for r in records:
        by_method.setdefault(r.method, []).append(r)
    for method, recs in by_method.items():
        style = METHOD_STYLE.get(method, dict(marker="."))
        xs = [getattr(r, x) for r in recs]
        ys = [getattr(r, y) for r in recs]
        ax.loglog(xs, ys, label=method, **style)
    ax.set_xlabel("equations $N$" if x == "n_equations" else x)
    ax.set_ylabel("relative energy error" if y == "e_energy" else y)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_error_vs_solves(records: list, path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    by_method: dict[str, list] = {}
    for r in records:
        by_method.setdefault(r.method, []).append(r)
    for method, recs in by_method.items():
        style = METHOD_STYLE.get(method, dict(marker="."))
        if method == "local_prediction":
            best: dict[int, list] = {}
            for r in recs:
                best.setdefault(r.extra.get("budget", 0), []).append(r)
            recs = max(best.values(), key=lambda rr: rr[-1].n_equations)
        xs = list(range(1, len(recs) + 1))
        ys = [r.e_energy for r in recs]
        ax.semilogy(xs, ys, label=method, **style)
    ax.set_xlabel("global solves (cumulative)")
    ax.set_ylabel("relative energy error")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
