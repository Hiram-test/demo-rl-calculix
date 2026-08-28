"""The VLA short loop: real-workflow remeshing (skill: vla-real-workflow).

Global solve accounting (all through FemRunner):

  draw             geometry / loads / supports; no CalculiX
  solve 1          "first"      mesh from eye sizes (not uniform h0)
  revise (mid)     communication + optional split, then PSO
  solve 2..k-1     "revise{r}"
  last revision    PSO only (no communication, no split, no eye, no LP)
  solve k          "certified"

The loop is adaptive: it stops early when the measured indicator meets
the accuracy limit within the resource budget, and never exceeds
``max_solves``.  PSO fitness is the last solve's η² plus N ~ h^{-d};
there is no η ~ h^q error surrogate.  Asymptotic optimality is ceded
to Doerfler.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..experiment import FemRunner
from ..indicators import zz_indicator
from ..mesher import generate_mesh
from .agents import AgentConfig, communication_round
from .drawing import drawings_size_fn
from .pso import PSOConfig, calibrate_measured
from .regions import Partition


@dataclass
class VLAConfig:
    n_eq_budget: int = 8000
    max_solves: int = 4               # probe included; adaptive, not fixed
    error_share_target: float = 0.25  # accuracy limit vs first-solve indicator
    min_budget_use: float = 0.7       # early stop needs this utilization
    allow_split: bool = True
    early_stop: bool = True
    gradation: float = 0.9
    init_ratio_bounds: tuple[float, float] = (1.0 / 8.0, 1.8)
    pso_pos_bound_later: float = 0.45  # trust region after the first jump
    final_budget_safety: float = 0.85  # certification round: never overshoot
    min_predicted_gain: float = 0.15   # retired: was E_pred inplace stop
    inplace_min_use: float = 0.88      # retired with the error surrogate
    partition_mode: str = "drawn"      # drawn | geodesic | linf_box (AB2)
    allow_communication: bool = True   # AB5
    allow_pso: bool = True             # AB6
    pso_mode: str = "sk"               # sk | s_only | nelder (AB8)
    use_measured_exponents: bool = True  # AB9
    use_resource_drift: bool = True    # AB10
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


def vision_assigned_sizes(part, problem) -> np.ndarray:
    """Eye-assigned region sizes: the drawing's fineness, not LP paint."""

    return np.clip(part.sizes(), problem.h_min, problem.h0)


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

    # ---- draw on the drawing; no solve -----------------------------------
    seeds = partitioner.propose(problem)
    drawings = list(getattr(partitioner, "last_drawings", []) or [])
    part = Partition(
        seeds, problem, gradation=cfg.gradation, assign_mode=cfg.partition_mode,
        drawings=drawings,
    )
    h_init = vision_assigned_sizes(part, problem)
    part = part.with_sizes(h_init)
    seeds_initial = [s.name for s in seeds]
    sizes_initial = [float(v) for v in h_init]
    remainder_h = next(
        (s.h for s in part.seeds if s.origin == "coarse"), float(problem.h0)
    )
    mesh = generate_mesh(problem, drawings_size_fn(drawings, remainder_h, problem))

    # ---- solve 1: first analysis on the drawn mesh -----------------------
    post, rec = runner.solve_mesh(mesh, method=method, stage="first")
    eta2 = zz_indicator(problem, post)
    labels = part.assign(mesh)
    feats = part.features(post, eta2, labels)
    rec.extra["sum_eta2"] = float(eta2.sum())
    rec.extra["regions"] = {s.name: float(h) for s, h in zip(part.seeds, part.sizes())}
    err_limit = cfg.error_share_target * float(eta2.sum())
    eq_per_elem = _eq_ratio(problem, rec)
    cfg.agent.error_share_target = cfg.error_share_target

    stopped_early = False
    round_info: dict = {"skipped": True, "reason": "no_revision_yet"}
    pso_info: dict = {"s": 0.0, "kappa": 0.0, "R_pred_elems": None}
    s_last, k_last = 0.0, 0.0
    solve_idx = 1

    # ---- later solves: residual revision; last revision is PSO only ------
    while solve_idx < cfg.max_solves:
        next_is_final = solve_idx + 1 == cfg.max_solves
        drift = 1.0
        if cfg.use_resource_drift and pso_info.get("R_pred_elems"):
            drift = rec.n_elems / max(pso_info["R_pred_elems"], 1.0)

        if next_is_final:
            # last revision uses PSO only — no communication, no split, no eye
            round_info = {"skipped": True, "reason": "final_revision_pso"}
            part_next = part
            pso_cfg = replace(
                cfg.pso,
                pos_bound=cfg.pso_pos_bound_later,
                budget_safety=cfg.final_budget_safety,
            )
        else:
            adjacency = part.adjacency(mesh, labels)
            if cfg.allow_communication:
                h_agent, round_info = communication_round(
                    part, feats, adjacency,
                    n_eq_budget=cfg.n_eq_budget, eq_per_elem=eq_per_elem,
                    cfg=cfg.agent, p_vec=None,
                )
                part_next = part.with_sizes(h_agent)
            else:
                round_info = {"skipped": True}
                part_next = part
            if cfg.allow_split:
                grown = part_next.split_concentrated(post, eta2, labels)
                if len(grown.seeds) > len(part_next.seeds):
                    part_next = grown
                    labels = part_next.assign(mesh)
                    feats = part_next.features(post, eta2, labels)
            pso_cfg = replace(cfg.pso, pos_bound=cfg.pso_pos_bound_later)

        A = part_next.adjacency_matrix(mesh, labels)
        if cfg.allow_pso:
            h_cal, pso_info = calibrate_measured(
                part_next, part_next.sizes(), feats, A,
                n_eq_budget=cfg.n_eq_budget,
                eq_per_elem=eq_per_elem, resource_drift=drift, cfg=pso_cfg,
                mode=cfg.pso_mode,
            )
        else:
            h_cal = part_next.sizes()
            pso_info = {"s": 0.0, "kappa": 0.0, "R_pred_elems": None}
        if next_is_final:
            pso_info["final_revision"] = "pso"
        s_last, k_last = pso_info["s"], pso_info["kappa"]

        part = part_next.with_sizes(h_cal)
        labels = part.assign(mesh)

        solve_idx += 1
        is_final = solve_idx == cfg.max_solves
        stage = "certified" if is_final else f"revise{solve_idx}"
        mesh = generate_mesh(problem, part.size_field(mesh, labels))
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
            rec.extra["final_revision"] = "pso"
            break
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

    vla_records = [r for r in runner.records if r.method == method]
    ns = {r.n_equations for r in vla_records}
    gate = len(ns) == len(vla_records)
    if not gate:
        vla_records[-1].extra["failure_scenario"] = "equation_counts_collide"

    # first solve is already the drawn mesh and may be the deliverable
    candidates = [
        r
        for r in vla_records
        if r.n_equations <= cfg.n_eq_budget and "sum_eta2" in r.extra
    ]
    if not candidates:
        candidates = vla_records
    pick = min(candidates, key=lambda r: r.extra.get("sum_eta2", float("inf")))
    pick.extra["certified_pick"] = True

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
