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


def test_vla_deliverable_holds_certified_after_early_stop():
    """A2′ must keep reporting the certified iterate for k > n_solves."""

    from visionamr.campaign import vla_deliverable

    recs = [
        {"method": "vla", "n_equations": 1200, "e_energy": 0.48,
         "extra": {"sum_eta2": 12.0}},
        {"method": "vla", "n_equations": 7200, "e_energy": 0.21,
         "extra": {"sum_eta2": 3.0}},
        {"method": "vla", "n_equations": 7500, "e_energy": 0.198,
         "extra": {"sum_eta2": 2.4}},
    ]
    pick3 = vla_deliverable(recs, 3, 8000)
    pick6 = vla_deliverable(recs, 6, 8000)
    assert pick3 is not None and pick6 is not None
    assert pick3["e_energy"] == pick6["e_energy"] == 0.198


def test_gate_g2_fails_when_handwritten_cell_inserts_found():
    """Review lock: G2 pass is false iff add_cells/insert_cells hits exist."""

    from pathlib import Path

    from visionamr.campaign import gate_g2

    g = gate_g2()
    assert g["pass"] == (len(g["hits"]) == 0)
    src = (Path(__file__).resolve().parents[1] / "visionamr" / "campaign.py").read_text()
    assert "len(hits) == 0" in src


def test_s8_whitelist_figure_helpers_are_wired():
    """Plan §8 bars/boxplots must stay callable without remeshing."""

    from pathlib import Path

    import pytest

    src = (Path(__file__).resolve().parents[1] / "visionamr" / "viz.py").read_text()
    for token in ("def plot_training_cost_bars", "def plot_test_boxplots"):
        assert token in src

    pytest.importorskip("matplotlib")

    from visionamr.viz import plot_test_boxplots, plot_training_cost_bars

    out = Path("/tmp/visionamr_s8_figs")
    out.mkdir(exist_ok=True)
    plot_training_cost_bars(
        [
            {"family": "bearing_block", "kind": "supervised_experts", "n_experts": 24},
            {"family": "bearing_block", "kind": "rl_s0", "train_solves": 240, "episodes": 120},
        ],
        out / "training_cost.png",
        title="A4",
    )
    plot_test_boxplots(
        {
            "bearing_block": [
                {"key": "test_9000", "k": {2: {"dorfler": 0.4, "vla": 0.3},
                                            3: {"dorfler": 0.3, "vla": 0.25},
                                            4: {"dorfler": 0.2, "vla": 0.22}}},
            ]
        },
        out / "test_boxplots.png",
        title="IQR",
    )
    assert (out / "training_cost.png").stat().st_size > 0
    assert (out / "test_boxplots.png").stat().st_size > 0


def test_s7_keeps_plan_ablations_ab7_and_ab9_to_ab11():
    """Review lock: AB7 rows and plan §5 AB9–AB11 must stay wired."""

    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "visionamr" / "campaign.py").read_text()
    for token in (
        "vla_ab7_k",
        "vla_ab9_fixed_q",
        "vla_ab10_nodrift",
        "vla_ab10_safety092",
        "vla_ab11_no_inplace",
    ):
        assert token in text


def test_reference_floor_honours_lbracket_corner_grading():
    """G3: reference mesh floor must be at least as fine as h_ref/48."""

    from visionamr.experiment import reference_floor
    from visionamr.geometry import make_lbracket

    problem = make_lbracket()
    floor = reference_floor(problem)
    assert floor <= problem.h_ref / 48.0 + 1e-15
    assert floor < problem.h_min


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
