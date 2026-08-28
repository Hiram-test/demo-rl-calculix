"""Regenerate the VLA evidence figures (partition view, meshes, curves).

Reuses the cached reference in an existing benchmark directory; the VLA
itself costs three CalculiX solves per problem.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from visionamr.experiment import FemRunner, initial_mesh
from visionamr.fem_post import compute_post
from visionamr.geometry import PROBLEM_FACTORIES
from visionamr.indicators import zz_indicator
from visionamr.mesher import generate_mesh
from visionamr.sizefield import Region, RegionSizeField
from visionamr.viz import plot_mesh, render_field_png
from visionamr.vla.partition import ScriptedVisionPartitioner
from visionamr.vla.pipeline import VLAConfig, run_vla


def figures_for(problem_name: str, bench_dir: Path, out_dir: Path) -> None:
    problem = PROBLEM_FACTORIES[problem_name]()
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = FemRunner(problem, bench_dir / "figs_tmp")
    # reuse the benchmark reference if present
    ref_src = bench_dir / "reference.json"
    if ref_src.exists():
        shutil.copy(ref_src, runner.workdir / "reference.json")
    runner.ensure_reference()

    partitioner = ScriptedVisionPartitioner()
    mesh0 = initial_mesh(problem)
    post0, _ = runner.solve_mesh(mesh0, method="fig", stage="probe")
    eta2_0 = zz_indicator(problem, post0)
    render_field_png(problem, post0, out_dir / f"{problem_name}_probe_field.png")
    regions = partitioner.partition(problem, post0, eta2_0)
    plot_mesh(
        mesh0,
        out_dir / f"{problem_name}_partition.png",
        title=f"{problem_name}: vision partition on probe solve",
        regions=regions,
    )

    res = run_vla(runner, partitioner, VLAConfig(n_eq_budget=8000))
    final_mesh = runner.last_mesh
    plot_mesh(
        final_mesh,
        out_dir / f"{problem_name}_vla_certified_mesh.png",
        title=(
            f"{problem_name}: VLA certified mesh "
            f"(3 solves, s={res.s:.3f}, kappa={res.kappa:.3f})"
        ),
    )

    for fig in ("fig_error_vs_N.png", "fig_error_vs_solves.png"):
        src = bench_dir / fig
        if src.exists():
            shutil.copy(src, out_dir / f"{problem_name}_{fig}")
    shutil.rmtree(runner.workdir, ignore_errors=True)


if __name__ == "__main__":
    figures_for("lbracket", Path("results/bench_lbracket"), Path("docs/evidence"))
    figures_for("plate_holes", Path("results/bench_plate"), Path("docs/evidence"))
    print("figures written to docs/evidence/")
