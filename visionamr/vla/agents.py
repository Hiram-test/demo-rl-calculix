"""Regional sub-agents: one communication round of size negotiation.

Each region is an agent holding one size.  In a single round it sees its
own residual/resource shares against area-proportional targets, its
neighbours' sizes (smoothness coupling), and the parent agent's global
budget pressure, then performs one bounded log-step.  The background
participates through the global term only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regions import RegionFeatures, RegionGraph

EQ_PER_ELEM = 1.5  # empirical CPS3 ratio n_equations / n_elements


@dataclass
class AgentConfig:
    w_err: float = 4.0
    w_res: float = 1.0
    regularization: float = 0.5
    max_log_step: float = 0.35
    neighbor_coupling: float = 0.08
    global_share: float = 0.6
    p_error: float = 1.3     # assumed local error order until measured
    d_resource: float = 2.0  # 2-D: elements ~ h^-2
    error_share_target: float = 0.7  # aim to keep 70% of current total error


def communication_round(
    graph: RegionGraph,
    feats: RegionFeatures,
    *,
    n_eq_budget: int,
    cfg: AgentConfig | None = None,
) -> tuple[np.ndarray, float, dict]:
    """One round; returns (new region sizes, new background size, info)."""

    cfg = cfg or AgentConfig()
    problem = graph.problem
    R = len(graph.regions)
    elems_budget = max(n_eq_budget / EQ_PER_ELEM, 1.0)

    xmin, ymin, xmax, ymax = problem.bbox
    bbox_area = (xmax - xmin) * (ymax - ymin)
    region_area = feats.area
    bg_area = max(bbox_area - float(region_area.sum()), 1e-12)
    shares = np.concatenate([region_area, [bg_area]])
    shares = shares / shares.sum()

    err_limit = cfg.error_share_target * feats.total_err
    E_tgt = err_limit * shares
    R_tgt = elems_budget * shares

    E_i = np.maximum(feats.err_sum, 1e-30)
    R_i = np.maximum(feats.elems.astype(float), 1.0)
    e_log = np.log(E_i / np.maximum(E_tgt[:R], 1e-30))
    r_log = np.log(R_i / np.maximum(R_tgt[:R], 1e-12))

    p, d = cfg.p_error, cfg.d_resource
    denom = cfg.w_err * p**2 + cfg.w_res * d**2 + cfg.regularization
    delta = np.clip(
        (cfg.w_res * d * r_log - cfg.w_err * p * e_log) / denom,
        -cfg.max_log_step,
        cfg.max_log_step,
    )

    # neighbour smoothness coupling
    log_h = np.log(graph.sizes())
    nb_term = np.zeros(R)
    for i, nbs in enumerate(graph.adjacency):
        if nbs:
            nb_term[i] = np.mean([log_h[j] - log_h[i] for j in nbs])
    delta = delta + cfg.neighbor_coupling * nb_term

    # parent agent: shared budget pressure on everyone incl. background
    g = np.clip(
        (cfg.global_share / d) * np.log(max(feats.total_elems, 1) / elems_budget),
        -cfg.max_log_step,
        cfg.max_log_step,
    )

    h_new = np.clip(
        graph.sizes() * np.exp(delta + g), problem.h_min, problem.h0
    )
    h_bg_new = float(
        np.clip(graph.h_background * np.exp(g), problem.h_min, problem.h0)
    )
    info = {
        "delta": delta.tolist(),
        "global_step": float(g),
        "err_log_ratio": e_log.tolist(),
        "res_log_ratio": r_log.tolist(),
    }
    return h_new, h_bg_new, info


def revise_regions(
    graph: RegionGraph,
    feats: RegionFeatures,
    post,
    eta2: np.ndarray,
    *,
    bg_err_threshold: float = 0.30,
    max_new: int = 2,
) -> RegionGraph:
    """VLA may edit its own region set: if the background holds too much
    residual, promote its worst hotspot cluster(s) to new regions."""

    if feats.bg_err <= bg_err_threshold * feats.total_err:
        return graph
    mesh = post.mesh
    owner = graph.assign_elements(mesh)
    bg_mask = owner == -1
    if not bg_mask.any():
        return graph
    from .partition import ScriptedVisionPartitioner

    # cluster only background elements by zeroing others' indicator
    eta_bg = np.where(bg_mask, eta2, 0.0)
    sub = ScriptedVisionPartitioner(quantile=0.985, max_regions=max_new)
    # build a pseudo post whose nodal field highlights background residual
    node_eta = np.zeros(mesh.n_nodes)
    for k in range(3):
        np.add.at(node_eta, mesh.tris[:, k], eta_bg / 3.0)
    import copy

    pseudo = copy.copy(post)
    pseudo.vm_node = node_eta
    new_regions = sub.partition(graph.problem, pseudo, eta_bg)
    if not new_regions:
        return graph
    regions = list(graph.regions)
    for r in new_regions:
        regions.append(
            r.with_h(min(0.6 * graph.h_background, graph.problem.h0))
        )
    return RegionGraph.build(
        regions, graph.h_background, graph.problem, gradation=graph.gradation
    )
