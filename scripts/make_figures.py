"""Regenerate the VLA evidence figures (partition views, meshes, curves).

Reuses the cached reference in an existing benchmark directory; the VLA
itself costs its usual few CalculiX solves per problem.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionamr.experiment import FemRunner, initial_mesh
from visionamr.geometry import PROBLEM_FACTORIES
from visionamr.indicators import zz_indicator
from visionamr.viz import plot_mesh, plot_partition, render_field_png
from visionamr.vla.partition import ScriptedVisionPartitioner
from visionamr.vla.pipeline import VLAConfig, run_vla
from visionamr.vla.regions import Partition

BUDGETS = {"bearing_block": 8000, "deck_panel": 20000, "lbracket": 8000,
           "plate_holes": 8000}


def figures_for(problem_name: str, bench_dir: Path, out_dir: Path) -> None:
    problem = PROBLEM_FACTORIES[problem_name]()
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = FemRunner(problem, bench_dir / "figs_tmp")
    ref_src = bench_dir / "reference.json"
    if ref_src.exists():
        shutil.copy(ref_src, runner.workdir / "reference.json")
    runner.ensure_reference()

    partitioner = ScriptedVisionPartitioner()
    mesh0 = initial_mesh(problem)
    post0, _ = runner.solve_mesh(mesh0, method="fig", stage="probe")
    eta2_0 = zz_indicator(problem, post0)
    render_field_png(problem, post0, out_dir / f"{problem_name}_probe_field.png")

    seeds = partitioner.propose(problem, post0, eta2_0)
    part = Partition(seeds, problem)
    labels = part.assign(mesh0)
    plot_partition(
        problem, post0, labels, seeds,
        out_dir / f"{problem_name}_partition.png",
        title=f"{problem_name}: seed-grown geodesic partition on the probe",
    )

    res = run_vla(
        runner, partitioner,
        VLAConfig(n_eq_budget=BUDGETS[problem_name], max_solves=2),
    )
    plot_mesh(
        runner.last_mesh,
        out_dir / f"{problem_name}_vla_certified_mesh.png",
        title=(
            f"{problem_name}: VLA certified mesh "
            f"({res.solves} solves, {len(res.seeds_final)} regions)"
        ),
    )

    for fig in ("fig_error_vs_N.png", "fig_error_vs_solves.png"):
        src = bench_dir / fig
        if src.exists():
            shutil.copy(src, out_dir / f"{problem_name}_{fig}")
    shutil.rmtree(runner.workdir, ignore_errors=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or ("bearing_block", "deck_panel"):
        figures_for(name, Path(f"results/bench_{name}"), Path("docs/evidence"))
        print(f"{name} figures written")
