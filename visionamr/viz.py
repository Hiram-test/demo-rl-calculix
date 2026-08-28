"""Rendering: field views for the vision head and evidence figures."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from .fem_post import PostState
from .geometry import Problem


def render_field_png(
    problem: Problem, post: PostState, path: Path | None = None, dpi: int = 130
) -> bytes:
    """Von Mises field on the mesh with axes in model units (vision input)."""

    mesh = post.mesh
    tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    tc = ax.tripcolor(tri, post.vm_node, shading="gouraud", cmap="turbo")
    ax.triplot(tri, lw=0.15, color="k", alpha=0.25)
    fig.colorbar(tc, ax=ax, label="von Mises [MPa]")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{problem.name}: response field")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    data = buf.getvalue()
    if path is not None:
        Path(path).write_bytes(data)
    return data


def plot_mesh(
    mesh, path: Path, title: str = "", regions=None, values=None, label: str = ""
) -> None:
    tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if values is not None:
        tc = ax.tripcolor(tri, values, cmap="viridis")
        fig.colorbar(tc, ax=ax, label=label)
    ax.triplot(tri, lw=0.2, color="k", alpha=0.6)
    if regions:
        for r in regions:
            ax.add_patch(
                plt.Rectangle(
                    (r.xmin, r.ymin),
                    r.xmax - r.xmin,
                    r.ymax - r.ymin,
                    fill=False,
                    edgecolor="crimson",
                    lw=1.4,
                )
            )
            ax.annotate(
                f"{r.name}\nh={r.h:.2f}",
                (r.xmin, r.ymax),
                fontsize=6,
                color="crimson",
                va="bottom",
            )
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
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
