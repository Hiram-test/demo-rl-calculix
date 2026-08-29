"""Regional sub-agents: one communication round of size negotiation.

Each partition region is an agent holding one size.  In a round it sees
its own residual/resource shares against volume-proportional targets,
its neighbours' sizes (smoothness coupling), and the parent agent's
global budget pressure, then performs one bounded log-step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regions import Partition, RegionFeatures


@dataclass
class AgentConfig:
    w_err: float = 4.0
    w_res: float = 1.0
    regularization: float = 0.5
    max_log_step: float = 0.35
    neighbor_coupling: float = 0.08
    global_share: float = 0.6
    p_error: float = 2.0     # smooth-rate prior d log(sum eta^2)/d log h
    error_share_target: float = 0.7


def communication_round(
    partition: Partition,
    feats: RegionFeatures,
    adjacency: list[set[int]],
    *,
    n_eq_budget: int,
    eq_per_elem: float,
    cfg: AgentConfig | None = None,
    p_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """One round; returns (new region sizes, info).

    ``p_vec`` supplies per-region *measured* error exponents (fitted from
    two consecutive real solves).  This is the counter to the classic
    one-shot remeshing failure mode: assuming one smooth convergence rate
    everywhere is what makes ZZ-style prediction oscillate at
    singularities (Onate-Bugeda 1993); the agents negotiate with the rate
    each region actually exhibited.
    """

    cfg = cfg or AgentConfig()
    problem = partition.problem
    d = float(problem.dim)
    R = len(partition.seeds)
    elems_budget = max(n_eq_budget / eq_per_elem, 1.0)

    shares = feats.volume / max(feats.volume.sum(), 1e-30)
    err_limit = cfg.error_share_target * feats.total_err
    E_tgt = np.maximum(err_limit * shares, 1e-30)
    R_tgt = np.maximum(elems_budget * shares, 1e-12)

    E_i = np.maximum(feats.err_sum, 1e-30)
    R_i = np.maximum(feats.elems.astype(float), 1.0)
    e_log = np.log(E_i / E_tgt)
    r_log = np.log(R_i / R_tgt)

    if p_vec is not None:
        p = np.clip(np.asarray(p_vec, dtype=float), 0.5, 5.0)
    else:
        p = np.full(R, cfg.p_error)
    denom = cfg.w_err * p**2 + cfg.w_res * d**2 + cfg.regularization
    delta = np.clip(
        (cfg.w_res * d * r_log - cfg.w_err * p * e_log) / denom,
        -cfg.max_log_step,
        cfg.max_log_step,
    )

    log_h = np.log(partition.sizes())
    nb_term = np.zeros(R)
    for i, nbs in enumerate(adjacency):
        if nbs:
            nb_term[i] = np.mean([log_h[j] - log_h[i] for j in nbs])
    delta = delta + cfg.neighbor_coupling * nb_term

    g = np.clip(
        (cfg.global_share / d) * np.log(max(feats.total_elems, 1) / elems_budget),
        -cfg.max_log_step,
        cfg.max_log_step,
    )

    h_new = np.clip(
        partition.sizes() * np.exp(delta + g), problem.h_min, problem.h0
    )
    info = {
        "delta": delta.tolist(),
        "global_step": float(g),
        "err_log_ratio": e_log.tolist(),
        "res_log_ratio": r_log.tolist(),
    }
    return h_new, info
