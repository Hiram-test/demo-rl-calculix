import numpy as np

from visionamr.geometry import make_plate_holes
from visionamr.vla.pso import PSOConfig, Surrogate, calibrate, transfer_direction
from visionamr.vla.regions import Partition, Seed


def two_region_partition():
    problem = make_plate_holes(width=2.0, height=1.0, holes=(), tension=100.0)
    part = Partition(
        [Seed("left", (0.5, 0.5, 0.0), h=0.1), Seed("right", (1.5, 0.5, 0.0), h=0.1)],
        problem,
    )
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    return part, A


def test_transfer_direction_zero_mean_and_signs():
    sur = Surrogate(
        E_ref=np.array([10.0, 0.1]),
        R_ref=np.array([100.0, 100.0]),
        h_ref=np.array([0.1, 0.1]),
        q=np.array([2.0, 2.0]),
        d=2.0,
    )
    tau = transfer_direction(sur)
    assert abs(np.max(np.abs(tau)) - 1.0) < 1e-9
    # region 0 has higher marginal efficiency -> refine it when kappa > 0
    assert tau[0] < 0 < tau[1]


def test_pso_respects_resource_budget_on_surrogate():
    part, A = two_region_partition()
    sur = Surrogate(
        E_ref=np.array([1.0, 1.0]),
        R_ref=np.array([2500.0, 2500.0]),
        h_ref=np.array([0.1, 0.1]),
        q=np.array([2.0, 2.0]),
        d=2.0,
    )
    # anchor holds 5000 elements; budget allows only ~2000
    h, info = calibrate(
        part, np.array([0.1, 0.1]), sur, A,
        err_limit=1e9, n_eq_budget=3000, eq_per_elem=1.5, cfg=PSOConfig(seed=3),
    )
    E_pred, R_pred = sur.predict(h)
    assert R_pred <= 1.10 * info["elems_budget"]
    assert info["s"] > 0.0  # budget pressure coarsens


def test_pso_refines_when_error_limit_binds():
    part, A = two_region_partition()
    sur = Surrogate(
        E_ref=np.array([5.0, 5.0]),
        R_ref=np.array([100.0, 100.0]),
        h_ref=np.array([0.1, 0.1]),
        q=np.array([2.0, 2.0]),
        d=2.0,
    )
    h, info = calibrate(
        part, np.array([0.1, 0.1]), sur, A,
        err_limit=1.0, n_eq_budget=10**7, eq_per_elem=1.5, cfg=PSOConfig(seed=3),
    )
    assert info["s"] < 0.0  # must refine to chase the accuracy limit


def test_resource_drift_tightens_budget():
    """A measured overshoot (drift > 1) must shrink the effective budget."""

    part, A = two_region_partition()
    sur = Surrogate(
        E_ref=np.array([1.0, 1.0]),
        R_ref=np.array([1500.0, 1500.0]),
        h_ref=np.array([0.1, 0.1]),
        q=np.array([2.0, 2.0]),
        d=2.0,
    )
    h_plain, info_plain = calibrate(
        part, np.array([0.1, 0.1]), sur, A,
        err_limit=1e9, n_eq_budget=4500, eq_per_elem=1.5, cfg=PSOConfig(seed=3),
    )
    h_drift, info_drift = calibrate(
        part, np.array([0.1, 0.1]), sur, A,
        err_limit=1e9, n_eq_budget=4500, eq_per_elem=1.5,
        resource_drift=1.3, cfg=PSOConfig(seed=3),
    )
    assert info_drift["elems_budget"] < info_plain["elems_budget"]
    _, R_plain = sur.predict(h_plain)
    _, R_drift = sur.predict(h_drift)
    assert R_drift < R_plain  # coarser sizes to compensate the overshoot


def test_communication_round_uses_measured_exponents():
    """Regions with a slow measured rate get treated more cautiously."""

    import copy

    from visionamr.vla.agents import AgentConfig, communication_round
    from visionamr.vla.regions import RegionFeatures

    part, _ = two_region_partition()
    feats = RegionFeatures(
        err_sum=np.array([5.0, 5.0]),      # same error excess
        elems=np.array([1000.0, 1000.0]),  # resource exactly on target
        vm_max=np.array([1.0, 1.0]),
        vm_mean=np.array([1.0, 1.0]),
        h_meas=np.array([0.1, 0.1]),
        volume=np.array([0.6, 0.6]),
        total_err=10.0,
        total_elems=2000,
    )
    adjacency = [{1}, {0}]
    # region 0 slow rate (singular, q=0.8), region 1 smooth (q=2.0)
    h, info = communication_round(
        part, feats, adjacency, n_eq_budget=3000, eq_per_elem=1.5,
        cfg=AgentConfig(neighbor_coupling=0.0), p_vec=np.array([0.8, 2.0]),
    )
    d0, d1 = info["delta"]
    assert d0 != d1  # measured rates differentiate otherwise identical regions


def test_surrogate_dimension_changes_resource_scaling():
    sur2 = Surrogate(
        E_ref=np.array([1.0]), R_ref=np.array([1000.0]),
        h_ref=np.array([0.1]), q=np.array([2.0]), d=2.0,
    )
    sur3 = Surrogate(
        E_ref=np.array([1.0]), R_ref=np.array([1000.0]),
        h_ref=np.array([0.1]), q=np.array([2.0]), d=3.0,
    )
    _, R2 = sur2.predict(np.array([0.05]))
    _, R3 = sur3.predict(np.array([0.05]))
    assert np.isclose(R2, 4000.0)
    assert np.isclose(R3, 8000.0)
