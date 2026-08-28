"""Two-coordinate PSO calibration under accuracy and resource limits.

After the communication round the proposed region sizes h+ are
calibrated with a particle swarm over (s, kappa):

    h_i(s, kappa) = h+_i * exp(s + kappa * tau_i),

where s is a global log-scale and tau is a resource-neutral transfer
direction built from measured marginal efficiencies.  Fitness is a
weighted penalty around the pre-declared accuracy limit and resource
budget (meet the accuracy, prefer fewer resources), evaluated on a
power-law surrogate anchored at the latest real solve (its region
exponents are fitted from the two most recent solves).  The winning
particle is certified with one real solve by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regions import Partition, RegionFeatures


@dataclass
class PSOConfig:
    n_particles: int = 9
    generations: int = 6
    pos_bound: float = 0.8
    vel_bound: float = 0.15
    inertia: float = 0.35
    cognitive: float = 0.8
    social: float = 1.2
    seed: int = 17
    w_err: float = 200.0        # quadratic pull toward the accuracy limit
    w_err_lin: float = 10.0     # monotone accuracy drive (always on)
    w_res_over: float = 400.0   # the resource budget is the hard cap
    w_res_under: float = 2.0
    w_quality: float = 50.0
    w_dev: float = 0.05
    max_neighbor_ratio: float = 1.8
    budget_safety: float = 0.97  # final projection headroom on the surrogate
    q_default: float = 2.0       # d log(sum eta^2) / d log h until measured
    q_bounds: tuple[float, float] = (0.6, 4.5)


@dataclass
class Surrogate:
    """Power-law response anchored at the latest real solve (nominal sizes)."""

    E_ref: np.ndarray   # region eta^2 sums at anchor
    R_ref: np.ndarray   # region element counts at anchor
    h_ref: np.ndarray   # nominal region sizes at anchor
    q: np.ndarray       # fitted d log(eta^2) / d log h
    d: float            # resource dimension (2 or 3)

    def predict(self, h: np.ndarray) -> tuple[float, float]:
        ratio = np.maximum(h, 1e-12) / np.maximum(self.h_ref, 1e-12)
        E = float(np.sum(self.E_ref * ratio**self.q))
        R = float(np.sum(self.R_ref * ratio**-self.d))
        return E, R


def fit_surrogate(
    partition: Partition,
    feats_now: RegionFeatures,
    h_now_nominal: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray]],
    cfg: PSOConfig,
) -> Surrogate:
    """Anchor at the latest solve; fit exponents from the last two solves.

    ``history`` holds (nominal sizes, region eta^2 sums) of earlier
    solves whose leading len() entries correspond to current region ids
    (new split regions simply have no history and keep the default q).
    """

    R = len(partition.seeds)
    q = np.full(R, cfg.q_default)
    if history:
        h_prev, E_prev = history[-1]
        k = min(len(h_prev), R)
        with np.errstate(divide="ignore", invalid="ignore"):
            dlogh = np.log(h_now_nominal[:k] / np.maximum(h_prev[:k], 1e-12))
            dlogE = np.log(
                np.maximum(feats_now.err_sum[:k], 1e-30)
                / np.maximum(E_prev[:k], 1e-30)
            )
            mask = np.abs(dlogh) > 0.05
            q[:k][mask] = np.clip(dlogE[mask] / dlogh[mask], *cfg.q_bounds)
    return Surrogate(
        E_ref=np.maximum(feats_now.err_sum, 1e-30),
        R_ref=np.maximum(feats_now.elems.astype(float), 1.0),
        h_ref=np.maximum(h_now_nominal, 1e-12),
        q=q,
        d=float(partition.problem.dim),
    )


def transfer_direction(sur: Surrogate) -> np.ndarray:
    """Resource-neutral transfer: refine where marginal efficiency is high."""

    marg = (sur.q * sur.E_ref) / np.maximum(sur.d * sur.R_ref, 1e-12)
    tau = -np.log(np.maximum(marg, 1e-30))
    w = sur.R_ref / sur.R_ref.sum()
    tau = tau - float(np.sum(w * tau))
    m = np.abs(tau).max()
    return tau / m if m > 1e-12 else np.zeros_like(tau)


def calibrate(
    partition: Partition,
    h_plus: np.ndarray,
    sur: Surrogate,
    adjacency_matrix: np.ndarray,
    *,
    err_limit: float,
    n_eq_budget: int,
    eq_per_elem: float,
    cfg: PSOConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """PSO over (s, kappa) around the proposal h_plus; returns (sizes, info)."""

    cfg = cfg or PSOConfig()
    problem = partition.problem
    rng = np.random.default_rng(cfg.seed)
    tau = transfer_direction(sur)
    elems_budget = max(n_eq_budget / eq_per_elem, 1.0)
    edges = np.argwhere(np.triu(adjacency_matrix) > 0)

    def decode(s: float, k: float) -> np.ndarray:
        return np.clip(h_plus * np.exp(s + k * tau), problem.h_min, problem.h0)

    def fitness(s: float, k: float) -> float:
        h = decode(s, k)
        E, R = sur.predict(h)
        # accuracy: quadratic pull toward the limit when feasible, plus an
        # always-on monotone drive so an infeasible limit cannot leverage
        # the swarm across the (hard) resource cap
        e_plus = min(max(np.log(E / max(err_limit, 1e-30)), 0.0), 1.0)
        e_lin = float(np.clip(np.log(E / max(err_limit, 1e-30)), -4.0, 4.0))
        r_log = np.log(R / elems_budget)
        r_plus, r_minus = max(r_log, 0.0), max(-r_log, 0.0)
        Q = 0.0
        for i, j in edges:
            ratio = max(h[i], h[j]) / max(min(h[i], h[j]), 1e-12)
            Q += max(np.log(ratio / cfg.max_neighbor_ratio), 0.0) ** 2
        return (
            cfg.w_err * e_plus**2
            + cfg.w_err_lin * e_lin
            + cfg.w_res_over * r_plus**2
            + cfg.w_res_under * r_minus**2
            + cfg.w_quality * Q
            + cfg.w_dev * (s**2 + k**2)
        )

    # seeded swarm: mother particle, closed-form budget and accuracy
    # scales (the classic scalar knob), and axis probes
    E0, R0 = sur.predict(h_plus)
    s_budget = float(
        np.clip(np.log(R0 / elems_budget) / sur.d, -cfg.pos_bound, cfg.pos_bound)
    )
    q_mean = float(np.average(sur.q, weights=np.maximum(sur.E_ref, 1e-30)))
    s_accuracy = float(
        np.clip(
            np.log(max(err_limit, 1e-30) / max(E0, 1e-30)) / max(q_mean, 0.5),
            -cfg.pos_bound,
            cfg.pos_bound,
        )
    )
    init = [
        (0.0, 0.0),
        (s_budget, 0.0),
        (s_accuracy, 0.0),
        (0.12, 0.0),
        (-0.12, 0.0),
        (0.0, 0.12),
        (0.0, -0.12),
    ]
    while len(init) < cfg.n_particles:
        init.append(tuple(rng.uniform(-0.2, 0.2, size=2)))
    pos = np.array(init[: cfg.n_particles], dtype=float)
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_f = np.array([fitness(*p) for p in pos])
    g_idx = int(np.argmin(pbest_f))
    gbest, gbest_f = pbest[g_idx].copy(), float(pbest_f[g_idx])
    evals = len(pos)

    for _ in range(cfg.generations):
        r1 = rng.random(pos.shape)
        r2 = rng.random(pos.shape)
        vel = (
            cfg.inertia * vel
            + cfg.cognitive * r1 * (pbest - pos)
            + cfg.social * r2 * (gbest[None, :] - pos)
        )
        vel = np.clip(vel, -cfg.vel_bound, cfg.vel_bound)
        pos = np.clip(pos + vel, -cfg.pos_bound, cfg.pos_bound)
        for i, p in enumerate(pos):
            f = fitness(*p)
            evals += 1
            if f < pbest_f[i]:
                pbest_f[i], pbest[i] = f, p.copy()
                if f < gbest_f:
                    gbest_f, gbest = f, p.copy()

    s, k = float(gbest[0]), float(gbest[1])
    h_final = decode(s, k)

    # certification semantics: the budget is a hard cap.  Project the
    # winner onto the surrogate budget surface (closed-form global scale).
    cap = cfg.budget_safety * elems_budget
    for _ in range(3):
        _, R_chk = sur.predict(h_final)
        if R_chk <= cap:
            break
        s += float(np.log(R_chk / cap)) / sur.d
        h_final = decode(s, k)

    E_pred, R_pred = sur.predict(h_final)
    info = {
        "s": s,
        "kappa": k,
        "tau": tau.tolist(),
        "fitness": float(gbest_f),
        "surrogate_evals": evals,
        "E_pred": E_pred,
        "R_pred_elems": R_pred,
        "err_limit": err_limit,
        "elems_budget": elems_budget,
    }
    return h_final, info
