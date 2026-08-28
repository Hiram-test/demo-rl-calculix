"""The VLA short loop: probe -> vision partition + one communication round
-> regional solve -> (optional region edit) -> PSO calibration -> certify.

Global solve accounting (all through FemRunner):

  solve 1  "probe"      uniform h0 mesh; vision input and residual source
  solve 2  "regional"   mesh from the negotiated region sizes
  solve 3  "certified"  mesh from the PSO-calibrated sizes

The three meshes must have pairwise distinct equation counts (gate),
otherwise the run records a failure scenario instead of faking a third
point.  An optional continuous variant inserts extra
communication+solve rounds for the ablation of "keep deciding" vs the
three-solve short loop.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..experiment import FemRunner, initial_mesh
from ..indicators import zz_indicator
from ..mesher import generate_mesh
from .agents import AgentConfig, communication_round, revise_regions
from .pso import PSOConfig, calibrate, fit_surrogate
from .regions import RegionGraph


@dataclass
class VLAConfig:
    n_eq_budget: int = 8000
    error_share_target: float = 0.7   # accuracy limit as share of probe error
    allow_region_edit: bool = True
    extra_rounds: int = 0             # continuous-variant ablation
    gradation: float = 0.9
    agent: AgentConfig = field(default_factory=AgentConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)


@dataclass
class VLAResult:
    regions_initial: list
    regions_final: list
    sizes_delegated: list
    sizes_negotiated: list
    sizes_certified: list
    s: float
    kappa: float
    n_distinct_gate: bool
    solves: int
    info: dict


def run_vla(
    runner: FemRunner,
    partitioner,
    config: VLAConfig | None = None,
    *,
    method: str = "vla",
) -> VLAResult:
    cfg = config or VLAConfig()
    problem = runner.problem
    runner.ensure_reference()

    # ---- solve 1: probe (vision input + residual source)
    mesh0 = initial_mesh(problem)
    post0, rec0 = runner.solve_mesh(mesh0, method=method, stage="probe")
    eta2_0 = zz_indicator(problem, post0)
    rec0.extra["sum_eta2"] = float(eta2_0.sum())

    # ---- vision partition (region count follows the view)
    regions = partitioner.partition(problem, post0, eta2_0)
    graph = RegionGraph.build(
        regions, problem.h0, problem, gradation=cfg.gradation
    )
    sizes_delegated = graph.sizes().copy()

    # ---- one communication round on probe residuals
    feats0 = graph.features(post0, eta2_0)
    h_probe = np.full(len(graph.regions), problem.h0)  # nominal probe sizes
    cfg.agent.error_share_target = cfg.error_share_target
    h1, hb1, round_info = communication_round(
        graph, feats0, n_eq_budget=cfg.n_eq_budget, cfg=cfg.agent
    )
    graph = graph.with_sizes(h1, hb1)

    # ---- solve 2: regional mesh
    mesh1 = generate_mesh(problem, graph.size_field())
    post1, rec1 = runner.solve_mesh(mesh1, method=method, stage="regional")
    eta2_1 = zz_indicator(problem, post1)
    rec1.extra["sum_eta2"] = float(eta2_1.sum())
    rec1.extra["regions"] = [r.name for r in graph.regions]

    # ---- optional continuous variant: keep negotiating
    for k in range(cfg.extra_rounds):
        feats_k = graph.features(post1, eta2_1)
        hk, hbk, _ = communication_round(
            graph, feats_k, n_eq_budget=cfg.n_eq_budget, cfg=cfg.agent
        )
        graph = graph.with_sizes(hk, hbk)
        mesh1 = generate_mesh(problem, graph.size_field())
        post1, rec1 = runner.solve_mesh(mesh1, method=method, stage=f"extra{k}")
        eta2_1 = zz_indicator(problem, post1)

    # ---- VLA may edit its own regions if the background stayed hot
    feats1 = graph.features(post1, eta2_1)
    E_probe = feats0.err_sum.copy()
    if cfg.allow_region_edit:
        graph2 = revise_regions(graph, feats1, post1, eta2_1)
        if len(graph2.regions) != len(graph.regions):
            # re-aggregate over the edited region set; new regions have no
            # probe pair, so their exponent stays at the default
            n_old = len(graph.regions)
            graph = graph2
            feats1 = graph.features(post1, eta2_1)
            h_probe = np.concatenate([h_probe, graph.sizes()[n_old:]])
            E_probe = np.concatenate([E_probe, feats1.err_sum[n_old:]])

    # ---- PSO calibration under (accuracy limit, resource budget)
    sur = fit_surrogate(graph, E_probe, h_probe, feats1, cfg.pso)
    err_limit = cfg.error_share_target * float(eta2_0.sum())
    h2, hb2, pso_info = calibrate(
        graph, sur, err_limit=err_limit, n_eq_budget=cfg.n_eq_budget, cfg=cfg.pso
    )
    graph = graph.with_sizes(h2, hb2)

    # ---- solve 3: certified mesh
    mesh2 = generate_mesh(problem, graph.size_field())
    post2, rec2 = runner.solve_mesh(mesh2, method=method, stage="certified")
    eta2_2 = zz_indicator(problem, post2)
    rec2.extra["sum_eta2"] = float(eta2_2.sum())
    rec2.extra["pso"] = {k: v for k, v in pso_info.items() if k != "tau"}

    ns = {rec0.n_equations, rec1.n_equations, rec2.n_equations}
    gate = len(ns) == 3
    if not gate:
        rec2.extra["failure_scenario"] = "equation_counts_collide"

    return VLAResult(
        regions_initial=[r.name for r in regions],
        regions_final=[r.name for r in graph.regions],
        sizes_delegated=[float(v) for v in sizes_delegated],
        sizes_negotiated=[float(v) for v in h1],
        sizes_certified=[float(v) for v in h2],
        s=pso_info["s"],
        kappa=pso_info["kappa"],
        n_distinct_gate=gate,
        solves=3 + cfg.extra_rounds,
        info={"round": round_info, "pso": pso_info},
    )
