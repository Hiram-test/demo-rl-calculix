"""The VLA short loop: an adaptive few-solve remeshing protocol.

Global solve accounting (all through FemRunner):

  solve 1        "probe"       uniform h0 mesh; vision input + residual source
  solve 2        "regional"    partition sizes from region-level error
                               equidistribution + one communication round + PSO
  solve 3..k-1   "cal{r}"      optional further rounds: exponent refit,
                               communication round, region splitting, PSO
  solve k        "certified"   final calibrated mesh

The loop is adaptive: it stops early when the measured indicator meets
the accuracy limit within the resource budget, and never exceeds
``max_solves``.  The design target is to beat the Doerfler loop at
matched few solve counts (asymptotic optimality is ceded to Doerfler).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..baselines.local_prediction import predicted_sizes
from ..experiment import FemRunner, initial_mesh
from ..indicators import zz_indicator
from ..mesher import generate_mesh
from .agents import AgentConfig, communication_round
from .pso import PSOConfig, calibrate, fit_surrogate
from .regions import Partition


@dataclass
class VLAConfig:
    n_eq_budget: int = 8000
    max_solves: int = 4               # probe included; adaptive, not fixed
    error_share_target: float = 0.25  # accuracy limit vs probe indicator
    min_budget_use: float = 0.7       # early stop needs this utilization
    allow_split: bool = True
    early_stop: bool = True
    gradation: float = 0.9
    init_ratio_bounds: tuple[float, float] = (1.0 / 8.0, 1.8)
    agent: AgentConfig = field(default_factory=AgentConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)


_EQ_RATIO_CEIL = {2: 1.6, 3: 0.62}


def _eq_ratio(problem, rec) -> float:
    """Equations-per-element estimate.

    Coarse meshes overestimate the fine-mesh ratio in 3-D (boundary
    dominated), so cap by the asymptotic prior until the mesh is large
    enough to trust the measurement.
    """

    measured = rec.n_equations / max(rec.n_elems, 1)
    if rec.n_elems >= 4000:
        return measured
    return min(measured, _EQ_RATIO_CEIL[problem.dim])


@dataclass
class VLAResult:
    seeds_initial: list
    seeds_final: list
    sizes_initial: list
    sizes_final: list
    s_last: float
    kappa_last: float
    n_distinct_gate: bool
    solves: int
    stopped_early: bool
    info: dict


def _region_geomean_sizes(h_elem: np.ndarray, labels: np.ndarray, R: int,
                          fallback: float) -> np.ndarray:
    out = np.full(R, fallback)
    for i in range(R):
        m = labels == i
        if m.any():
            out[i] = float(np.exp(np.mean(np.log(np.maximum(h_elem[m], 1e-12)))))
    return out


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

    # ---- solve 1: probe --------------------------------------------------
    mesh = initial_mesh(problem)
    post, rec = runner.solve_mesh(mesh, method=method, stage="probe")
    eta2 = zz_indicator(problem, post)
    rec.extra["sum_eta2"] = float(eta2.sum())
    err_limit = cfg.error_share_target * float(eta2.sum())
    eq_per_elem = _eq_ratio(problem, rec)

    # ---- vision partition + region-level equidistribution init ----------
    seeds = partitioner.propose(problem, post, eta2)
    part = Partition(seeds, problem, gradation=cfg.gradation)
    labels = part.assign(mesh)
    elems_budget = cfg.n_eq_budget / eq_per_elem
    h_elem = predicted_sizes(
        mesh, eta2, n_target=elems_budget, ratio_bounds=cfg.init_ratio_bounds,
        d=problem.dim,
    )
    h_init = np.clip(
        _region_geomean_sizes(h_elem, labels, len(seeds), problem.h0),
        problem.h_min, problem.h0,
    )
    part = part.with_sizes(h_init)
    seeds_initial = [s.name for s in seeds]
    sizes_initial = [float(v) for v in h_init]

    # first communication round + PSO on probe measurements
    feats = part.features(post, eta2, labels)
    adjacency = part.adjacency(mesh, labels)
    cfg.agent.error_share_target = cfg.error_share_target
    history: list[tuple[np.ndarray, np.ndarray]] = [
        (np.full(len(seeds), problem.h0), feats.err_sum.copy())
    ]
    h_agent, round_info = communication_round(
        part, feats, adjacency,
        n_eq_budget=cfg.n_eq_budget, eq_per_elem=eq_per_elem, cfg=cfg.agent,
    )
    # blend: equidistribution magnitudes with the negotiated correction
    h_plus = np.clip(
        np.sqrt(h_init * h_agent), problem.h_min, problem.h0
    )
    sur = fit_surrogate(part, feats, np.full(len(seeds), problem.h0), [], cfg.pso)
    A = part.adjacency_matrix(mesh, labels)
    h_cal, pso_info = calibrate(
        part, h_plus, sur, A,
        err_limit=err_limit, n_eq_budget=cfg.n_eq_budget,
        eq_per_elem=eq_per_elem, cfg=cfg.pso,
    )
    part = part.with_sizes(h_cal)

    stopped_early = False
    s_last, k_last = pso_info["s"], pso_info["kappa"]
    solve_idx = 1

    # ---- solves 2..k ------------------------------------------------------
    while solve_idx < cfg.max_solves:
        solve_idx += 1
        is_final = solve_idx == cfg.max_solves
        stage = (
            "regional" if solve_idx == 2 else
            ("certified" if is_final else f"cal{solve_idx}")
        )
        anchor_h = part.sizes().copy()
        new_mesh = generate_mesh(problem, part.size_field(mesh, labels))
        mesh = new_mesh
        post, rec = runner.solve_mesh(mesh, method=method, stage=stage)
        eta2 = zz_indicator(problem, post)
        labels = part.assign(mesh)
        feats = part.features(post, eta2, labels)
        eq_per_elem = _eq_ratio(problem, rec)
        rec.extra.update(
            sum_eta2=float(eta2.sum()),
            regions={s.name: float(h) for s, h in zip(part.seeds, part.sizes())},
        )

        if is_final:
            rec.extra["pso"] = {k: v for k, v in pso_info.items() if k != "tau"}
            break

        # early certification: accuracy met AND the budget actually used
        if (
            cfg.early_stop
            and float(eta2.sum()) <= err_limit
            and cfg.min_budget_use * cfg.n_eq_budget
            <= rec.n_equations
            <= cfg.n_eq_budget
        ):
            rec.stage = f"{stage}_certified_early"
            rec.extra["stopped_early"] = True
            stopped_early = True
            break

        # ---- decide the next mesh ---------------------------------------
        adjacency = part.adjacency(mesh, labels)
        h_agent, round_info = communication_round(
            part, feats, adjacency,
            n_eq_budget=cfg.n_eq_budget, eq_per_elem=eq_per_elem, cfg=cfg.agent,
        )
        part_next = part.with_sizes(h_agent)

        anchor_vec = anchor_h
        if cfg.allow_split and cfg.max_solves - solve_idx >= 1:
            grown = part_next.split_concentrated(post, eta2, labels)
            if len(grown.seeds) > len(part_next.seeds):
                n_new = len(grown.seeds) - len(part_next.seeds)
                part_next = grown
                labels = part_next.assign(mesh)
                feats = part_next.features(post, eta2, labels)
                # children inherit the parent's anchor size
                parents = [
                    s.name.rsplit("_hot", 1)[0] for s in part_next.seeds[-n_new:]
                ]
                name_to_idx = {s.name: i for i, s in enumerate(part.seeds)}
                extra_anchor = [
                    anchor_h[name_to_idx.get(p, 0)] for p in parents
                ]
                anchor_vec = np.concatenate([anchor_h, np.array(extra_anchor)])

        sur = fit_surrogate(part_next, feats, anchor_vec, history, cfg.pso)
        A = part_next.adjacency_matrix(mesh, labels)
        h_cal, pso_info = calibrate(
            part_next, part_next.sizes(), sur, A,
            err_limit=err_limit, n_eq_budget=cfg.n_eq_budget,
            eq_per_elem=eq_per_elem, cfg=cfg.pso,
        )
        s_last, k_last = pso_info["s"], pso_info["kappa"]
        history.append((anchor_vec.copy(), feats.err_sum.copy()))
        part = part_next.with_sizes(h_cal)

    vla_records = [r for r in runner.records if r.method == method]
    ns = {r.n_equations for r in vla_records}
    gate = len(ns) == len(vla_records)
    if not gate:
        vla_records[-1].extra["failure_scenario"] = "equation_counts_collide"

    return VLAResult(
        seeds_initial=seeds_initial,
        seeds_final=[s.name for s in part.seeds],
        sizes_initial=sizes_initial,
        sizes_final=[float(v) for v in part.sizes()],
        s_last=float(s_last),
        kappa_last=float(k_last),
        n_distinct_gate=gate,
        solves=len(vla_records),
        stopped_early=stopped_early,
        info={"round": round_info, "pso": pso_info},
    )
