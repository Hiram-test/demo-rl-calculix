"""(s, κ) PSO on the last measured residual and the mesh-count budget.

    h_i(s, kappa) = h+_i * exp(s + kappa * tau_i)

The live path is ``calibrate_grades``: the model chose discrete
levels; the tool maps them to ``GRADE_PRIOR`` and, if a last mesh
exists, applies one closed-form scale.  ``calibrate_measured``
stays for older unit tests.  ``run_vla`` must not call ``fit_surrogate``.

``Surrogate`` / ``fit_surrogate`` / ``calibrate`` stay for unit tests of
the retired power-law fitness.  ``run_vla`` must not call them.
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


def transfer_from_measure(err_sum: np.ndarray, elems: np.ndarray) -> np.ndarray:
    """Refine where the last solve's residual per element is high.  No q-fit."""

    dens = np.asarray(err_sum, float) / np.maximum(np.asarray(elems, float), 1.0)
    tau = -np.log(np.maximum(dens, 1e-30))
    w = np.maximum(elems.astype(float), 1.0)
    w = w / w.sum()
    tau = tau - float(np.sum(w * tau))
    m = float(np.abs(tau).max())
    return tau / m if m > 1e-12 else np.zeros_like(tau)


def resource_elems(h: np.ndarray, h_ref: np.ndarray, n_ref: np.ndarray, d: float) -> float:
    """Element-count scaling from the last mesh.  Not a PDE error model."""

    ratio = np.maximum(h, 1e-12) / np.maximum(h_ref, 1e-12)
    return float(np.sum(np.maximum(n_ref, 1.0) * ratio ** (-d)))


def calibrate_measured(
    partition: Partition,
    h_plus: np.ndarray,
    feats: RegionFeatures,
    adjacency_matrix: np.ndarray,
    *,
    n_eq_budget: int,
    eq_per_elem: float,
    resource_drift: float = 1.0,
    cfg: PSOConfig | None = None,
    mode: str = "sk",
    h_anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Last-revision PSO: measured residual + budget.  No error surrogate.

    ``h_plus`` is the drawing prior.  ``h_anchor`` is the size field that
    generated the current mesh (usually the same eye sizes).  Resource
    scaling N ~ (h / h_anchor)^{-d} must not use smeared h_meas — Gmsh
    gradation makes measured sizes coarser than the request, which makes
    the eye field look over budget and the swarm coarsens it away.
    """

    cfg = cfg or PSOConfig()
    if mode not in ("sk", "s_only", "nelder"):
        raise ValueError(f"unknown measured-PSO mode {mode!r}")
    problem = partition.problem
    rng = np.random.default_rng(cfg.seed)
    d = float(problem.dim)
    h_ref = np.maximum(h_anchor if h_anchor is not None else h_plus, 1e-12)
    n_ref = np.maximum(feats.elems.astype(float), 1.0)
    err = np.maximum(feats.err_sum, 1e-30)
    share = err / err.sum()
    tau = transfer_from_measure(feats.err_sum, feats.elems)
    drift = float(np.clip(resource_drift, 1.0, 1.4))
    elems_budget = max(n_eq_budget / eq_per_elem, 1.0) / drift
    edges = np.argwhere(np.triu(adjacency_matrix) > 0)

    def decode(s: float, k: float) -> np.ndarray:
        return np.clip(h_plus * np.exp(s + k * tau), problem.h_min, problem.h0)

    def score_h(h: np.ndarray, *, s_pen: float = 0.0, k_pen: float = 0.0) -> float:
        R = resource_elems(h, h_ref, n_ref, d)
        r_log = np.log(R / elems_budget)
        r_plus, r_minus = max(r_log, 0.0), max(-r_log, 0.0)
        align = float(np.sum(share * (h / h_ref)))
        Q = 0.0
        for i, j in edges:
            ratio = max(h[i], h[j]) / max(min(h[i], h[j]), 1e-12)
            Q += max(np.log(ratio / cfg.max_neighbor_ratio), 0.0) ** 2
        return (
            cfg.w_res_over * r_plus**2
            + max(cfg.w_res_under, 80.0) * r_minus**2
            + cfg.w_quality * Q
            + 20.0 * align
            + cfg.w_dev * (s_pen**2 + k_pen**2)
        )

    def fitness(s: float, k: float) -> float:
        return score_h(decode(s, k), s_pen=s, k_pen=k)

    if mode == "s_only":
        ss = np.linspace(-cfg.pos_bound, cfg.pos_bound, 41)
        scores = np.array([fitness(float(s), 0.0) for s in ss])
        evals = len(ss)
        s = float(ss[int(np.argmin(scores))])
        lo, hi = max(s - 0.08, -cfg.pos_bound), min(s + 0.08, cfg.pos_bound)
        for _ in range(20):
            m1 = lo + 0.38 * (hi - lo)
            m2 = lo + 0.62 * (hi - lo)
            f1, f2 = fitness(m1, 0.0), fitness(m2, 0.0)
            evals += 2
            if f1 < f2:
                hi = m2
            else:
                lo = m1
        s, k = 0.5 * (lo + hi), 0.0
        gbest_f = fitness(s, k)
        evals += 1
        h_final = decode(s, k)
    elif mode == "nelder":
        from scipy.optimize import minimize

        x0 = np.zeros(len(h_plus))

        def f_nm(x):
            x = np.clip(x, -cfg.pos_bound, cfg.pos_bound)
            h = np.clip(h_plus * np.exp(x), problem.h_min, problem.h0)
            return score_h(h, s_pen=float(np.mean(x)), k_pen=0.0)

        res = minimize(
            f_nm, x0, method="Nelder-Mead",
            options={"maxiter": 80, "xatol": 1e-3, "fatol": 1e-4, "disp": False},
        )
        evals = int(res.nfev)
        x = np.clip(res.x, -cfg.pos_bound, cfg.pos_bound)
        h_final = np.clip(h_plus * np.exp(x), problem.h_min, problem.h0)
        s = float(np.mean(x))
        k = 0.0
        gbest_f = float(res.fun)
    else:
        R0 = resource_elems(h_plus, h_ref, n_ref, d)
        s_budget = float(
            np.clip(np.log(max(R0, 1.0) / elems_budget) / d, -cfg.pos_bound, cfg.pos_bound)
        )
        init = [
            (0.0, 0.0),
            (s_budget, 0.0),
            (0.0, 0.15),
            (0.0, -0.15),
            (0.10, 0.0),
            (-0.10, 0.0),
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
    cap = cfg.budget_safety * elems_budget
    for _ in range(3):
        R_chk = resource_elems(h_final, h_ref, n_ref, d)
        if R_chk <= cap:
            break
        scale = float(np.log(R_chk / cap)) / d
        s += scale
        h_final = np.clip(h_final * np.exp(scale), problem.h_min, problem.h0)
    R_pred = resource_elems(h_final, h_ref, n_ref, d)
    return h_final, {
        "s": s,
        "kappa": k,
        "tau": tau.tolist(),
        "fitness": float(gbest_f),
        "evals": evals,
        "R_pred_elems": R_pred,
        "elems_budget": elems_budget,
        "resource_drift": drift,
        "mode": "measured",
        "final_revision": "pso",
    }


def project_feasible(
    partition: Partition,
    h_model: np.ndarray,
    feats: RegionFeatures,
    *,
    n_eq_budget: int,
    eq_per_elem: float,
    h_anchor: np.ndarray | None = None,
    cfg: PSOConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Reliability only: if the decision overshoots the budget, scale it back.

    Does not re-rank by residual.  The next sizes come from the agent.
    """

    cfg = cfg or PSOConfig()
    problem = partition.problem
    d = float(problem.dim)
    h_ref = np.maximum(h_anchor if h_anchor is not None else h_model, 1e-12)
    n_ref = np.maximum(feats.elems.astype(float), 1.0)
    cap = max(n_eq_budget / max(eq_per_elem, 1e-12), 1.0)
    h = np.clip(np.asarray(h_model, float), problem.h_min, problem.h0)
    R = resource_elems(h, h_ref, n_ref, d)
    if R <= cap:
        return h, {
            "s": 0.0,
            "kappa": 0.0,
            "applied": False,
            "role": "reliable",
            "R_pred_elems": R,
            "elems_budget": cap,
            "mode": "project_feasible",
        }
    s = float(np.log(R / cap) / d)
    h = np.clip(h * np.exp(s), problem.h_min, problem.h0)
    R2 = resource_elems(h, h_ref, n_ref, d)
    return h, {
        "s": s,
        "kappa": 0.0,
        "applied": True,
        "role": "unreliable_overshoot",
        "R_pred_elems": R2,
        "elems_budget": cap,
        "mode": "project_feasible",
    }


def calibrate_grades(
    partition: Partition,
    grades: np.ndarray,
    feats: RegionFeatures | None,
    *,
    n_eq_budget: int,
    eq_per_elem: float,
    h_anchor: np.ndarray | None = None,
    cfg: PSOConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Map grades to h.  First call (feats=None) is priors only, evals=0.

    With a last mesh, one closed-form N-scale, evals ≤ 1.  No search.
    Inaccuracy is left for the next vision pass, like a person glancing
    again.  The first mesh may overshoot; that is not a search bug.
    """

    from .grades import GRADE_PRIOR, MIN_STEP, parse_grade

    del cfg
    problem = partition.problem
    d = float(problem.dim)
    g = np.array([parse_grade(int(v), "grade") for v in np.asarray(grades).ravel()], int)
    levels = sorted({int(x) for x in g})
    h = np.array([GRADE_PRIOR[int(v)] * problem.h0 for v in g], float)
    ordered = levels
    for a, b in zip(ordered, ordered[1:]):
        hi = float(h[g == a][0] * MIN_STEP)
        h[g == b] = np.maximum(h[g == b], hi)
    h = np.clip(h, problem.h_min, problem.h0)
    elems_budget = max(n_eq_budget / max(eq_per_elem, 1e-12), 1.0)
    evals = 0
    s = 0.0
    R_pred = None
    if feats is not None:
        h_ref = np.maximum(h_anchor if h_anchor is not None else feats.h_meas, 1e-12)
        n_ref = np.maximum(feats.elems.astype(float), 1.0)
        R = resource_elems(h, h_ref, n_ref, d)
        evals = 1
        target = elems_budget if R > elems_budget else 0.97 * elems_budget
        s = float(np.log(max(R, 1.0) / max(target, 1.0)) / d)
        h = np.clip(h * np.exp(s), problem.h_min, problem.h0)
        R_pred = float(target)
    h_lev = {int(lev): float(h[g == lev][0]) for lev in levels}
    return h, {
        "s": s,
        "kappa": 0.0,
        "applied": feats is not None,
        "role": "one_shot_tweak",
        "grades": [int(v) for v in g],
        "h_by_grade": h_lev,
        "R_pred_elems": R_pred,
        "elems_budget": elems_budget,
        "evals": evals,
        "mode": "calibrate_grades",
    }


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
    resource_drift: float = 1.0,
    cfg: PSOConfig | None = None,
    mode: str = "sk",
) -> tuple[np.ndarray, dict]:
    """Calibrate region sizes around the proposal h_plus; returns (sizes, info).

    ``mode``:
      * ``sk``     -- (s, κ) PSO (default, the paper method)
      * ``s_only`` -- AB8: global log-scale only (κ ≡ 0)
      * ``nelder`` -- AB8: per-region Nelder–Mead on log sizes

    ``resource_drift`` is the measured realized/predicted element ratio of
    the previous round (Gmsh gradation bands and mesh-generator behaviour
    are not in the power-law surrogate).  Because the budget is a hard
    cap, the correction may only *tighten* the surrogate budget (drift
    clipped below at 1): under-realization is a soft cost, overshoot is a
    contract violation.
    """

    cfg = cfg or PSOConfig()
    problem = partition.problem
    rng = np.random.default_rng(cfg.seed)
    tau = transfer_direction(sur)
    drift = float(np.clip(resource_drift, 1.0, 1.4))
    elems_budget = max(n_eq_budget / eq_per_elem, 1.0) / drift
    edges = np.argwhere(np.triu(adjacency_matrix) > 0)

    def decode(s: float, k: float) -> np.ndarray:
        return np.clip(h_plus * np.exp(s + k * tau), problem.h_min, problem.h0)

    def score_h(h: np.ndarray, *, s_pen: float = 0.0, k_pen: float = 0.0) -> float:
        E, R = sur.predict(h)
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
            + cfg.w_dev * (s_pen**2 + k_pen**2)
        )

    def fitness(s: float, k: float) -> float:
        return score_h(decode(s, k), s_pen=s, k_pen=k)

    evals = 0
    if mode == "s_only":
        ss = np.linspace(-cfg.pos_bound, cfg.pos_bound, 41)
        scores = np.array([fitness(float(s), 0.0) for s in ss])
        evals = len(ss)
        s = float(ss[int(np.argmin(scores))])
        # local refine
        lo, hi = max(s - 0.08, -cfg.pos_bound), min(s + 0.08, cfg.pos_bound)
        for _ in range(20):
            m1 = lo + 0.38 * (hi - lo)
            m2 = lo + 0.62 * (hi - lo)
            f1, f2 = fitness(m1, 0.0), fitness(m2, 0.0)
            evals += 2
            if f1 < f2:
                hi = m2
            else:
                lo = m1
        s, k = 0.5 * (lo + hi), 0.0
        gbest_f = fitness(s, k)
        evals += 1
        h_final = decode(s, k)
    elif mode == "nelder":
        from scipy.optimize import minimize

        x0 = np.zeros(len(h_plus))

        def f_nm(x):
            x = np.clip(x, -cfg.pos_bound, cfg.pos_bound)
            h = np.clip(h_plus * np.exp(x), problem.h_min, problem.h0)
            return score_h(h, s_pen=float(np.mean(x)), k_pen=0.0)

        res = minimize(
            f_nm, x0, method="Nelder-Mead",
            options={"maxiter": 80, "xatol": 1e-3, "fatol": 1e-4, "disp": False},
        )
        evals = int(res.nfev)
        x = np.clip(res.x, -cfg.pos_bound, cfg.pos_bound)
        h_final = np.clip(h_plus * np.exp(x), problem.h_min, problem.h0)
        s = float(np.mean(x))
        k = 0.0
        gbest_f = float(res.fun)
    else:
        if mode != "sk":
            raise ValueError(f"unknown PSO mode {mode!r}")
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
        scale = float(np.log(R_chk / cap)) / sur.d
        s += scale
        h_final = np.clip(h_final * np.exp(scale), problem.h_min, problem.h0)

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
        "resource_drift": drift,
        "mode": mode,
    }
    return h_final, info
