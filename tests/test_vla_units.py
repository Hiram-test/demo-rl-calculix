import numpy as np

from visionamr.geometry import make_plate_holes
from visionamr.sizefield import Region
from visionamr.vla.pso import PSOConfig, Surrogate, calibrate, transfer_direction
from visionamr.vla.regions import RegionGraph


def graph_two_regions():
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)
    regions = [
        Region("left", 0.0, 0.0, 0.6, 1.0, h=0.1),
        Region("right", 0.7, 0.0, 1.3, 1.0, h=0.1),
    ]
    return RegionGraph.build(regions, 0.2, problem), problem


def test_adjacency_via_padding():
    graph, _ = graph_two_regions()
    assert 1 in graph.adjacency[0]
    assert 0 in graph.adjacency[1]


def test_transfer_direction_zero_mean_and_signs():
    sur = Surrogate(
        E_ref=np.array([10.0, 0.1]),
        R_ref=np.array([100.0, 100.0]),
        h_ref=np.array([0.1, 0.1]),
        q=np.array([2.0, 2.0]),
        E_bg=0.0,
        R_bg=1.0,
        h_bg=0.2,
    )
    tau = transfer_direction(sur)
    assert abs(np.max(np.abs(tau)) - 1.0) < 1e-9
    # region 0 has higher marginal efficiency -> refine it when kappa > 0
    assert tau[0] < 0 < tau[1]


def test_pso_respects_resource_budget_on_surrogate():
    graph, problem = graph_two_regions()
    sur = Surrogate(
        E_ref=np.array([1.0, 1.0]),
        R_ref=np.array([2000.0, 2000.0]),
        h_ref=np.array([0.1, 0.1]),
        q=np.array([2.0, 2.0]),
        E_bg=0.1,
        R_bg=1000.0,
        h_bg=0.2,
    )
    # anchor uses 5000 elements but the budget only allows ~2000
    h, hb, info = calibrate(
        graph, sur, err_limit=1e9, n_eq_budget=3000, cfg=PSOConfig(seed=3)
    )
    E_pred, R_pred = sur.predict(h, hb)
    assert R_pred <= 1.10 * info["elems_budget"]  # within 10% of the cap
    assert info["s"] > 0.0  # budget pressure coarsens


def test_pso_refines_when_error_limit_binds():
    graph, problem = graph_two_regions()
    sur = Surrogate(
        E_ref=np.array([5.0, 5.0]),
        R_ref=np.array([100.0, 100.0]),
        h_ref=np.array([0.1, 0.1]),
        q=np.array([2.0, 2.0]),
        E_bg=0.0,
        R_bg=100.0,
        h_bg=0.2,
    )
    h, hb, info = calibrate(
        graph, sur, err_limit=1.0, n_eq_budget=10**7, cfg=PSOConfig(seed=3)
    )
    assert info["s"] < 0.0  # must refine to chase the accuracy limit
