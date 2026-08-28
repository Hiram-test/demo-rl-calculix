"""Two-coordinate PSO calibration under accuracy and resource limits.

After the communication round the region sizes are calibrated with a
particle swarm over (s, kappa):

    h_i(s, kappa) = h_i * exp(s + kappa * tau_i),      tau_bg = 0,

where s is a global log-scale and tau is a resource-neutral transfer
direction built from measured marginal efficiencies.  Fitness is a
weighted penalty around the pre-declared accuracy limit and resource
budget (meet the accuracy, prefer fewer resources), evaluated on a
power-law surrogate whose region exponents are fitted from the two real
solves already made.  The winning particle is certified with one real
solve by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents import EQ_PER_ELEM
from .regions import RegionFeatures, RegionGraph


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
    w_err: float = 200.0
    w_res_over: float = 200.0
    w_res_under: float = 2.0
    w_quality: float = 50.0
    w_dev: float = 0.05
    max_neighbor_ratio: float = 1.8
    q_default: float = 2.0     # eta^2 ~ h^q
    q_bounds: tuple[float, float] = (0.6, 4.0)


@dataclass
class Surrogate:
    """Power-law response anchored at the latest real solve."""

    E_ref: np.ndarray   # region eta^2 sums at anchor
    R_ref: np.ndarray   # region element counts at anchor
    h_ref: np.ndarray   # region sizes at anchor (measured)
    q: np.ndarray       # fitted d log(eta^2) / d log h
    E_bg: float
    R_bg: float
    h_bg: float

    def predict(self, h: np.ndarray, h_bg: float) -> tuple[float, float]:
        ratio = np.maximum(h, 1e-12) / np.maximum(self.h_ref, 1e-12)
        E = float(np.sum(self.E_ref * ratio**self.q))
        R = float(np.sum(self.R_ref * ratio**-2.0))
        rb = max(h_bg, 1e-12) / max(self.h_bg, 1e-12)
        E += self.E_bg * rb**2.0
        R += self.R_bg * rb**-2.0
        return E, R


def fit_surrogate(
    graph: RegionGraph,
    E_probe: np.ndarray,
    h_probe_nominal: np.ndarray,
    feats_now: RegionFeatures,
    cfg: PSOConfig,
) -> Surrogate:
    """Fit per-region exponents from the probe and current solves.

    The surrogate is anchored at the *nominal* region sizes of the last
    real solve, so the mother particle (s, kappa) = (0, 0) reproduces the
    measured error and element counts exactly; gradation overhead is
    absorbed into the anchor.
    """

    h_now = np.maximum(graph.sizes(), 1e-12)
    q = np.full(len(graph.regions), cfg.q_default)
    E0 = np.maximum(E_probe, 1e-30)
    E1 = np.maximum(feats_now.err_sum, 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlogh = np.log(h_now / np.maximum(h_probe_nominal, 1e-12))
        dlogE = np.log(E1 / E0)
        mask = np.abs(dlogh) > 0.05
        q[mask] = np.clip(dlogE[mask] / dlogh[mask], *cfg.q_bounds)
    return Surrogate(
        E_ref=E1,
        R_ref=np.maximum(feats_now.elems.astype(float), 1.0),
        h_ref=h_now,
        q=q,
        E_bg=float(feats_now.bg_err),
        R_bg=float(max(feats_now.bg_elems, 1)),
        h_bg=float(graph.h_background),
    )


def transfer_direction(sur: Surrogate) -> np.ndarray:
    """Resource-neutral transfer: refine where marginal efficiency is high."""

    marg = (sur.q * sur.E_ref) / np.maximum(2.0 * sur.R_ref, 1e-12)
    tau = -np.log(np.maximum(marg, 1e-30))
    w = sur.R_ref / sur.R_ref.sum()
    tau = tau - float(np.sum(w * tau))
    m = np.abs(tau).max()
    return tau / m if m > 1e-12 else np.zeros_like(tau)


def calibrate(
    graph: RegionGraph,
    sur: Surrogate,
    *,
    err_limit: float,
    n_eq_budget: int,
    cfg: PSOConfig | None = None,
) -> tuple[np.ndarray, float, dict]:
    """PSO over (s, kappa); returns (region sizes, background size, info)."""

    cfg = cfg or PSOConfig()
    problem = graph.problem
    rng = np.random.default_rng(cfg.seed)
    tau = transfer_direction(sur)
    h0_vec = graph.sizes()
    hb0 = graph.h_background
    elems_budget = max(n_eq_budget / EQ_PER_ELEM, 1.0)

    A = graph.adjacency_matrix()
    edges = np.argwhere(np.triu(A) > 0)

    def decode(s: float, k: float) -> tuple[np.ndarray, float]:
        h = np.clip(h0_vec * np.exp(s + k * tau), problem.h_min, problem.h0)
        hb = float(np.clip(hb0 * np.exp(s), problem.h_min, problem.h0))
        return h, hb

    def fitness(s: float, k: float) -> float:
        h, hb = decode(s, k)
        E, R = sur.predict(h, hb)
        e_plus = max(np.log(E / max(err_limit, 1e-30)), 0.0)
        r_log = np.log(R / elems_budget)
        r_plus, r_minus = max(r_log, 0.0), max(-r_log, 0.0)
        Q = 0.0
        for i, j in edges:
            ratio = max(h[i], h[j]) / max(min(h[i], h[j]), 1e-12)
            Q += max(np.log(ratio / cfg.max_neighbor_ratio), 0.0) ** 2
        return (
            cfg.w_err * e_plus**2
            + cfg.w_res_over * r_plus**2
            + cfg.w_res_under * r_minus**2
            + cfg.w_quality * Q
            + cfg.w_dev * (s**2 + k**2)
        )

    # deterministic seeded swarm: mother particle, closed-form budget and
    # accuracy scales (the classic scalar knob), and axis probes
    E0, R0 = sur.predict(h0_vec, hb0)
    s_budget = float(np.clip(0.5 * np.log(R0 / elems_budget), -cfg.pos_bound, cfg.pos_bound))
    q_mean = float(np.average(sur.q, weights=np.maximum(sur.E_ref, 1e-30)))
    s_accuracy = float(
        np.clip(np.log(max(err_limit, 1e-30) / max(E0, 1e-30)) / max(q_mean, 0.5),
                -cfg.pos_bound, cfg.pos_bound)
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
    h_final, hb_final = decode(s, k)
    E_pred, R_pred = sur.predict(h_final, hb_final)
    info = {
        "s": s,
        "kappa": k,
        "tau": tau.tolist(),
        "fitness": gbest_f,
        "surrogate_evals": evals,
        "E_pred": E_pred,
        "R_pred_elems": R_pred,
        "err_limit": err_limit,
        "elems_budget": elems_budget,
    }
    return h_final, hb_final, info
