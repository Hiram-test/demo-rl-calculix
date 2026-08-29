"""The VLA short loop: judge, call tools, then again.

  judge            draw regions; who is finer (grade 1..5, not h)
  tools            GRADE_PRIOR map (feats=None, evals=0) → mesh → solve
  judge            look at occupancy and leftover; maybe re-grade, still no h
  tools            one closed-form tweak (evals ≤ 1) → mesh → solve

The first mesh may overshoot.  That is the next glance, not a search bug.
The eye never tunes or delegates sizes.  Tools own the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..experiment import FemRunner
from ..indicators import zz_indicator
from ..mesher import generate_mesh
from .agents import AgentConfig, communication_round
from .drawing import drawings_size_fn, drawings_with_sizes
from .grades import grade_from_frac
from .pso import PSOConfig, calibrate_grades
from .regions import Partition


@dataclass
class VLAConfig:
    n_eq_budget: int = 8000
    max_solves: int = 2               # first + one certified remesh
    error_share_target: float = 0.25  # accuracy limit vs first-solve indicator
    min_budget_use: float = 0.7       # early stop needs this utilization
    allow_split: bool = False         # AB4; main method does not re-draw
    early_stop: bool = True
    gradation: float = 0.9
    init_ratio_bounds: tuple[float, float] = (1.0 / 8.0, 1.8)
    pso_pos_bound_later: float = 0.45  # trust region after the first jump
    final_budget_safety: float = 0.85  # certification round: never overshoot
    min_predicted_gain: float = 0.15   # retired: was E_pred inplace stop
    inplace_min_use: float = 0.88      # retired with the error surrogate
    partition_mode: str = "drawn"      # drawn | geodesic | linf_box (AB2)
    allow_communication: bool = False  # AB5; η-share talk is not the main method
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
    """Placeholder sizes from current partition; the tool overwrites them."""

    return np.clip(part.sizes(), problem.h_min, problem.h0)


def _grades_array(partitioner, part) -> np.ndarray:
    gmap = getattr(partitioner, "last_grades", None) or {}
    h0 = part.problem.h0
    out = []
    for s in part.seeds:
        if s.name in gmap:
            out.append(int(gmap[s.name]))
        elif s.origin == "coarse":
            out.append(int(gmap.get("field", 5)))
        else:
            out.append(grade_from_frac(s.h / h0))
    return np.asarray(out, dtype=int)


def _observation(rec, n_eq_budget: int, sizes: dict) -> dict:
    """Tool result handed back to vision: this solve + leftover resource."""

    used = int(rec.n_equations)
    remaining = max(n_eq_budget - used, 0)
    return {
        "n_equations": used,
        "n_elems": int(rec.n_elems),
        "budget": int(n_eq_budget),
        "remaining": remaining,
        "sizes": dict(sizes),
    }


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
    grades = _grades_array(partitioner, part)
    eq_guess = _EQ_RATIO_CEIL[problem.dim]
    if cfg.allow_pso:
        h_init, pso_info = calibrate_grades(
            part, grades, None,
            n_eq_budget=cfg.n_eq_budget, eq_per_elem=eq_guess, cfg=cfg.pso,
        )
    else:
        h_init = vision_assigned_sizes(part, problem)
        pso_info = {"s": 0.0, "kappa": 0.0, "applied": False, "role": "pso_off"}
    part = part.with_sizes(h_init)
    drawings = drawings_with_sizes(
        drawings, [s.name for s in part.seeds], part.sizes()
    )
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
    round_info: dict = {"tool": "first_solve"}
    s_last, k_last = float(pso_info.get("s", 0.0)), float(pso_info.get("kappa", 0.0))
    solve_idx = 1
    thoughts: list[str] = []
    rec.extra["grades"] = [int(v) for v in grades]
    rec.extra["pso"] = {k: v for k, v in pso_info.items() if k != "tau"}

    # ---- vision (re-grade) + tools (tweak / mesh / solve) -------------------
    while solve_idx < cfg.max_solves:
        obs = _observation(
            rec, cfg.n_eq_budget,
            {s.name: float(h) for s, h in zip(part.seeds, part.sizes())},
        )
        rec.extra["observation"] = {
            k: obs[k] for k in ("n_equations", "budget", "remaining")
        }
        decide = getattr(partitioner, "revise", None)
        decision = decide(problem, obs) if callable(decide) else None
        if decision is None and cfg.allow_communication:
            adjacency = part.adjacency(mesh, labels)
            h_agent, round_info = communication_round(
                part, feats, adjacency,
                n_eq_budget=cfg.n_eq_budget, eq_per_elem=eq_per_elem,
                cfg=cfg.agent, p_vec=None,
            )
            decision = {
                "thought": "AB5 communication",
                "sizes": {s.name: float(h) for s, h in zip(part.seeds, h_agent)},
                "stop": False,
                "source": "ab5_comm",
            }
        if decision is not None:
            thoughts.append(str(decision.get("thought") or ""))
            rec.extra["thought"] = thoughts[-1]
            if decision.get("stop"):
                rec.extra["stopped_by"] = "decision"
                stopped_early = True
                break
        grades = _grades_array(partitioner, part)
        part_next = part
        if cfg.allow_split:
            grown = part_next.split_concentrated(post, eta2, labels)
            if len(grown.seeds) > len(part_next.seeds):
                part_next = grown
                labels = part_next.assign(mesh)
                feats = part_next.features(post, eta2, labels)
                grades = _grades_array(partitioner, part_next)

        if cfg.allow_pso:
            h_cal, pso_info = calibrate_grades(
                part_next, grades, feats,
                n_eq_budget=cfg.n_eq_budget,
                eq_per_elem=eq_per_elem,
                h_anchor=part.sizes(),
                cfg=cfg.pso,
            )
        elif decision is not None and decision.get("sizes"):
            h_cal = np.array(
                [float(decision["sizes"].get(s.name, s.h)) for s in part_next.seeds]
            )
            pso_info = {"s": 0.0, "kappa": 0.0, "applied": False, "role": "pso_off"}
        else:
            h_cal = part_next.sizes()
            pso_info = {"s": 0.0, "kappa": 0.0, "applied": False, "role": "pso_off"}
        s_last, k_last = float(pso_info.get("s", 0.0)), float(pso_info.get("kappa", 0.0))
        round_info = {
            "source": None if decision is None else decision.get("source"),
            "remaining": obs["remaining"],
            "grades": [int(v) for v in grades],
        }

        part = part_next.with_sizes(h_cal)
        drawings = drawings_with_sizes(
            drawings, [s.name for s in part.seeds], part.sizes()
        )
        remainder_h = next(
            (s.h for s in part.seeds if s.origin == "coarse"), remainder_h
        )

        solve_idx += 1
        is_final = solve_idx == cfg.max_solves
        stage = "certified" if is_final else f"revise{solve_idx}"
        mesh = generate_mesh(problem, drawings_size_fn(drawings, remainder_h, problem))
        post, rec = runner.solve_mesh(mesh, method=method, stage=stage)
        eta2 = zz_indicator(problem, post)
        labels = part.assign(mesh)
        feats = part.features(post, eta2, labels)
        eq_per_elem = _eq_ratio(problem, rec)
        rec.extra.update(
            sum_eta2=float(eta2.sum()),
            regions={s.name: float(h) for s, h in zip(part.seeds, part.sizes())},
            grades=[int(v) for v in grades],
            thought=thoughts[-1] if thoughts else "",
            pso={k: v for k, v in pso_info.items() if k != "tau"},
            final_revision="grades_pso",
        )
        if is_final:
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
        info={"round": round_info, "pso": pso_info, "thoughts": thoughts},
    )
