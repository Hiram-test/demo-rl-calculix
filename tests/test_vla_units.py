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


def test_budget_rows_lp_stays_same_tier():
    """G6: local-prediction budget scatter must not stitch larger tiers."""

    from visionamr.report import _lp_same_tier, budget_rows, PILOT_EQ

    b = PILOT_EQ["bearing_block"]
    tier = _lp_same_tier("bearing_block", "canonical", b)
    assert tier
    last = tier[-1]["n_equations"]
    rows = [
        r for r in budget_rows()
        if r["method"] == "local_prediction"
        and r["family"] == "bearing_block"
        and r["key"] == "canonical"
    ]
    assert len(rows) == 1
    assert rows[0]["n_eq"] == last
    # the dumped file's last record is the largest (stitched) tier
    from visionamr.report import _series
    dumped_tail = _series("bearing_block", "canonical", "records_local_prediction.json")[-1]["n_equations"]
    assert last != dumped_tail


def test_s8_figures_never_stitch_llm_into_scripted_vla_series():
    """G6/§8: two VLA runs must not merge into one solves curve."""

    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "visionamr" / "campaign.py").read_text()
    assert 'if "llm" in glob' in src
    assert '"method": "vla_llm"' in src


def test_a2prime_discloses_dorfler_budget_and_learned_columns():
    """Audit lock: A2′ shows Dörfler N/B and the learned deliverables."""

    from visionamr.report import error_at_k_table

    row = error_at_k_table()["canonical"]["bearing_block"]
    # learned methods were run (S5/S6): probe + deploy, held after k=2
    assert row[2]["supervised"] is not None
    assert row[2]["rl_dqn"] is not None
    assert row[6]["supervised"] == row[2]["supervised"]
    # S2 caps Dörfler at the largest tier; over-pilot rounds must be visible
    fracs = [row[k]["dorfler_frac"] for k in range(1, 7) if row[k].get("dorfler_frac")]
    assert fracs and any(f > 1.05 for f in fracs)
    for k in range(1, 7):
        if row[k]["dorfler"] is not None:
            assert row[k]["dorfler_n"] is not None


def test_vla_init_does_not_paint_lp_sizes():
    """Init sizes come from the eye after drawing regions, not LP geomean."""

    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "visionamr" / "vla" / "pipeline.py").read_text()
    assert "predicted_sizes" not in src
    assert "vision_assigned_sizes" in src
    assert src.index("propose") < src.index("solve_mesh")


def test_pipeline_has_no_error_surrogate():
    """PSO is already written; the live loop must not fit η ~ h^q."""

    from pathlib import Path

    from visionamr.vla.pipeline import VLAConfig

    root = Path(__file__).resolve().parents[1]
    src = (root / "visionamr" / "vla" / "pipeline.py").read_text()
    assert "fit_surrogate" not in src
    assert "calibrate_measured" not in src
    assert "project_feasible" in src
    assert "revise" in src
    assert "from .pso import" in src and "calibrate," not in src.split("from .pso import", 1)[1].split("\n", 1)[0]
    assert "drawings_size_fn" in src
    scripted = (root / "visionamr" / "vla" / "partition.py").read_text()
    propose_body = scripted.split("def propose")[1].split("def _drawing_frac")[0]
    assert "vm_node" not in propose_body
    assert "del post, eta2" in propose_body
    cfg = VLAConfig()
    assert cfg.max_solves == 2
    assert cfg.allow_communication is False
    assert cfg.allow_split is False


def test_calibrate_measured_respects_budget_without_predicting_error():
    """Last revision: measured residual + N~h^{-d}.  No E_pred."""

    from visionamr.vla.pso import PSOConfig, calibrate_measured, resource_elems
    from visionamr.vla.regions import RegionFeatures

    part, A = two_region_partition()
    feats = RegionFeatures(
        err_sum=np.array([8.0, 0.5]),
        elems=np.array([2500.0, 2500.0]),
        vm_max=np.array([1.0, 1.0]),
        vm_mean=np.array([1.0, 1.0]),
        h_meas=np.array([0.1, 0.1]),
        volume=np.array([1.0, 1.0]),
        total_err=8.5,
        total_elems=5000,
    )
    h, info = calibrate_measured(
        part, np.array([0.1, 0.1]), feats, A,
        n_eq_budget=3000, eq_per_elem=1.5, cfg=PSOConfig(seed=3),
    )
    assert "E_pred" not in info
    assert info["mode"] == "measured"
    R = resource_elems(h, np.array([0.1, 0.1]), np.array([2500.0, 2500.0]), 2.0)
    assert R <= 1.10 * info["elems_budget"]
    assert info["s"] > 0.0
    assert h[0] <= h[1]  # more residual → keep or refine that region relative to the other


def test_calibrate_measured_spends_unused_budget():
    """Under the cap, measured PSO must refine — the budget is a target band."""

    from visionamr.vla.pso import PSOConfig, calibrate_measured, resource_elems
    from visionamr.vla.regions import RegionFeatures

    part, A = two_region_partition()
    feats = RegionFeatures(
        err_sum=np.array([6.0, 1.0]),
        elems=np.array([400.0, 400.0]),
        vm_max=np.array([1.0, 1.0]),
        vm_mean=np.array([1.0, 1.0]),
        h_meas=np.array([0.2, 0.2]),
        volume=np.array([1.0, 1.0]),
        total_err=7.0,
        total_elems=800,
    )
    h, info = calibrate_measured(
        part, np.array([0.2, 0.2]), feats, A,
        n_eq_budget=4500, eq_per_elem=1.5, cfg=PSOConfig(seed=3),
    )
    R = resource_elems(h, feats.h_meas, feats.elems.astype(float), 2.0)
    assert R > 800.0
    assert info["s"] < 0.0


def test_calibrate_measured_anchors_on_last_mesh_not_proposal():
    """N ~ h^{-d} must start from the mesh that produced n_ref."""

    from visionamr.vla.pso import PSOConfig, calibrate_measured, resource_elems
    from visionamr.vla.regions import RegionFeatures

    part, A = two_region_partition()
    # last mesh was coarse (h_meas=0.2); communication already proposed 0.05
    feats = RegionFeatures(
        err_sum=np.array([4.0, 4.0]),
        elems=np.array([800.0, 800.0]),
        vm_max=np.array([1.0, 1.0]),
        vm_mean=np.array([1.0, 1.0]),
        h_meas=np.array([0.2, 0.2]),
        volume=np.array([1.0, 1.0]),
        total_err=8.0,
        total_elems=1600,
    )
    proposed = np.array([0.05, 0.05])
    h, info = calibrate_measured(
        part, proposed, feats, A,
        n_eq_budget=2400, eq_per_elem=1.5, cfg=PSOConfig(seed=3),
        h_anchor=feats.h_meas,
    )
    R = resource_elems(h, feats.h_meas, feats.elems.astype(float), 2.0)
    assert R <= 1.10 * info["elems_budget"]
    # naive h_ref=h_plus would think N is still 1600 and accept 0.05
    assert float(h.mean()) > 0.08


def test_agent_revise_is_next_decision_from_result_and_leftover():
    """VLA think: this solve + remaining resource → next sizes.  No API."""

    from pathlib import Path

    from visionamr.geometry import make_bearing_block
    from visionamr.vla.partition import CachedDrawingPartitioner

    root = Path(__file__).resolve().parents[1]
    problem = make_bearing_block()
    eye = root / "tests" / "fixtures" / "bearing_block_eye.json"
    rev = root / "tests" / "fixtures" / "bearing_block_eye_revise1.json"
    head = CachedDrawingPartitioner(str(eye), revisions=[str(rev)])
    head.propose(problem)
    sizes0 = {d.name: d.h for d in head.last_drawings}
    sizes0["field"] = 0.72 * problem.h0
    out = head.revise(
        problem,
        {"n_equations": 6591, "budget": 8000, "remaining": 1409, "sizes": sizes0},
    )
    assert out is not None
    assert "还剩" in out["thought"] or "剩余" in out["thought"]
    assert out["sizes"]["patch_rim_x0"] < sizes0["patch_rim_x0"]
    assert out["sizes"]["patch_rim_x0"] < out["sizes"]["field"]
    assert head.revise(problem, {"sizes": sizes0}) is None


def test_project_feasible_only_pulls_back_overshoot():
    from visionamr.vla.pso import PSOConfig, project_feasible, resource_elems
    from visionamr.vla.regions import RegionFeatures

    part, _A = two_region_partition()
    feats = RegionFeatures(
        err_sum=np.array([1.0, 1.0]),
        elems=np.array([800.0, 800.0]),
        vm_max=np.array([1.0, 1.0]),
        vm_mean=np.array([1.0, 1.0]),
        h_meas=np.array([0.2, 0.2]),
        volume=np.array([1.0, 1.0]),
        total_err=2.0,
        total_elems=1600,
    )
    ok, info_ok = project_feasible(
        part, np.array([0.2, 0.2]), feats,
        n_eq_budget=4000, eq_per_elem=1.5, h_anchor=feats.h_meas,
        cfg=PSOConfig(seed=1),
    )
    assert info_ok["applied"] is False
    assert np.allclose(ok, [0.2, 0.2])
    blown, info = project_feasible(
        part, np.array([0.04, 0.04]), feats,
        n_eq_budget=2400, eq_per_elem=1.5, h_anchor=feats.h_meas,
        cfg=PSOConfig(seed=1),
    )
    assert info["applied"] is True
    R = resource_elems(blown, feats.h_meas, feats.elems.astype(float), 2.0)
    assert R <= 1.05 * info["elems_budget"]
    assert "E_pred" not in info
    assert "tau" not in info


def test_drawings_with_sizes_keeps_polygons():
    from visionamr.geometry import make_bearing_block
    from visionamr.vla.drawing import DrawnRegion, drawings_with_sizes

    problem = make_bearing_block()
    d = DrawnRegion("patch_core", 0.3 * problem.h0, "top", ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)))
    out = drawings_with_sizes([d], ["patch_core"], np.array([0.5 * problem.h0]))
    assert out[0].polygon == d.polygon
    assert out[0].h == 0.5 * problem.h0


def test_skill_locks_measured_pso_and_forbids_error_surrogate():
    from pathlib import Path

    skill = (
        Path(__file__).resolve().parents[1]
        / ".cursor" / "skills" / "vla-real-workflow" / "SKILL.md"
    ).read_text()
    assert "project_feasible" in skill
    assert "calibrate_measured" in skill
    assert "fit_surrogate" in skill
    assert "revise" in skill
    assert "计算结果" in skill and "资源存量" in skill
    assert "不许误差代理" in skill or "禁止" in skill and "η ~ h^q" in skill


def test_a2prime_tables_flagged_until_drawn_init_rerun():
    """Honesty: do not apply the new-init narrative to pre-change dumps."""

    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "visionamr" / "report.py").read_text()
    assert "尚未用新初始化重跑" in src
    assert "通信/PSO/分裂/硬帽/就地认证仍是实例循环" not in src
    assert "两轮定稿" in src


def test_e1_does_not_claim_v2_lp_is_monotonic():
    """Honesty lock: v2 7-solve LP also rebounds; do not call it 单调."""

    from pathlib import Path

    from visionamr.report import build_all_tables, render_results_md

    src = (Path(__file__).resolve().parents[1] / "visionamr" / "report.py").read_text()
    assert "单调进平台" not in src
    assert "单调进平台" not in render_results_md(build_all_tables())


def test_supervised_weakness_is_not_unseen_coverage():
    """Honesty lock: infinite supervised would swallow unseen-topology coverage.

    The scientific contradiction is label circularity (experts are Dörfler)
    plus deploy-time probe blindness — not 'the model never saw this hole'.
    """

    from pathlib import Path

    from visionamr.report import build_all_tables, render_results_md

    src = (Path(__file__).resolve().parents[1] / "visionamr" / "report.py").read_text()
    md = render_results_md(build_all_tables())
    for text in (src, md):
        assert "这正是要测的" not in text
        assert "标签预言机" in text
        assert "一步回归发明不了" in text or "一步回归从这根探针发明不了" in text
        assert "不是「没见过的孔所以监督差」" in text or "禁止把「未见拓扑 / 冻结母族」写成科学结论" in text


def test_ood_samplers_draw_outside_training_support():
    """E3 validity: every OOD parameter must sit outside the sampler range."""

    import numpy as np

    from visionamr.geometry import OOD_SAMPLERS

    for seed in (9500, 9501, 9502, 9503):
        p = OOD_SAMPLERS["bearing_block"](np.random.default_rng(seed)).params
        assert min(p["patch"]) > 180.0
        assert min(abs(p["offset"][0]), abs(p["offset"][1])) > 70.0
        assert p["pressure"] > 16.0
        q = OOD_SAMPLERS["deck_panel"](np.random.default_rng(seed)).params
        assert q["wheel_pos"][0] > 1700.0 and q["wheel_pos"][1] > 1100.0
        assert q["wheel"][0] > 500.0 and q["wheel"][1] > 320.0
        assert q["pressure"] > 1.4


def test_weakness_evidence_tables_are_populated():
    """E1/E2/E3 audit tables must be backed by dumped records."""

    from visionamr.report import (
        budget_deviation_stats,
        lp_naive_diagnostic,
        ood_generalization_table,
    )

    naive = lp_naive_diagnostic()
    for fam in ("bearing_block", "deck_panel"):
        assert len(naive[fam]["rows"]) == 7
        assert naive[fam]["uptick_after_best"] is True

    dev = budget_deviation_stats()
    for fam in ("bearing_block", "deck_panel"):
        for m in ("local_prediction", "supervised", "vla"):
            assert dev[fam][m]["n"] > 0 and dev[fam][m]["median"] is not None

    ood = ood_generalization_table()
    for fam in ("bearing_block", "deck_panel"):
        assert len(ood[fam]["rows"]) == 4
        assert ood[fam]["median_gap_sup_minus_vla_ood"] is not None
        assert ood[fam]["median_gap_sup_minus_vla_test"] is not None


def test_failure_probe_families_are_geometrically_valid():
    """E4 validity: hole clear of patch/walls, opening clear of strip/wheel."""

    import numpy as np

    from visionamr.geometry import (
        PROBLEM_FACTORIES,
        SAMPLERS,
        analytic_load_resultant,
    )

    for maker in (
        lambda: PROBLEM_FACTORIES["bearing_hole"](),
        lambda: SAMPLERS["bearing_hole"](np.random.default_rng(9600)),
        lambda: SAMPLERS["bearing_hole"](np.random.default_rng(9601)),
    ):
        p = maker().params
        a, _ = p["patch"]
        px0 = p["W"] / 2 + p["offset"][0] - a / 2
        hx = px0 - p["hole_gap"] - p["hole_r"]
        assert hx - p["hole_r"] > 10.0            # clear of the x=0 wall
        assert px0 - (hx + p["hole_r"]) == p["hole_gap"]
        assert p["H"] / 2 + p["hole_r"] < p["H"]  # inside the block height

    for maker in (
        lambda: PROBLEM_FACTORIES["deck_opening"](),
        lambda: SAMPLERS["deck_opening"](np.random.default_rng(9600)),
        lambda: SAMPLERS["deck_opening"](np.random.default_rng(9601)),
    ):
        prob = maker()
        p = prob.params
        _, wb = p["wheel"]
        wy0 = p["wheel_pos"][1] - wb / 2
        ocy = wy0 - p["open_gap"] - p["open_r"]
        strip_top = p["strip_off"] + p["strip_w"] / 2
        assert ocy - p["open_r"] > strip_top + 20.0   # clear of the strip
        assert wy0 - (ocy + p["open_r"]) == p["open_gap"]
        F = analytic_load_resultant(prob)
        assert F[2] < 0

    # the scripted head must receive the hole anchor (drawing knowledge)
    for fam in ("bearing_hole", "deck_opening"):
        prob = PROBLEM_FACTORIES[fam]()
        assert any(f.kind == "hole" for f in prob.features)


def test_failure_probe_table_is_backed_by_records():
    """E4 audit lock: the report table reads dumped CalculiX records."""

    from visionamr.report import failure_probe_table

    fp = failure_probe_table()
    assert "bearing_hole" in fp
    info = fp["bearing_hole"]
    assert len(info["rows"]) == 3
    for r in info["rows"]:
        assert r["supervised_e"] is not None
        assert r["vla_e"] is not None
        assert r["rim_h_supervised"] is not None
    assert info["median_gap_sup_minus_vla_fp"] is not None
    assert info["median_rim_h_ratio_sup_over_vla"] is not None


def test_lp_rounds_diag_stays_out_of_the_main_lp_series():
    """G6: the 7-solve diagnostic must not join the plan-locked short run."""

    from visionamr.report import _lp_same_tier, lp_rounds_diagnostic, PILOT_EQ

    diag = lp_rounds_diagnostic()
    assert "bearing_block" in diag and len(diag["bearing_block"]["rows"]) == 7
    # main same-tier pick is still the 3-solve short run
    tier = _lp_same_tier("bearing_block", "canonical", PILOT_EQ["bearing_block"])
    assert len(tier) == 3
    assert all(r.get("method") == "local_prediction" for r in tier)


def test_training_cost_counts_supervised_offline_solves():
    """A4: the expert bank's real CalculiX solves are on the books."""

    from visionamr.report import training_cost_rows

    rows = {(r["family"], r["kind"]): r for r in training_cost_rows()}
    for fam in ("bearing_block", "deck_panel"):
        sup = rows[(fam, "supervised_experts")]
        assert sup["n_experts"] == 24
        assert sup["train_solves"] is not None and sup["train_solves"] > 100


def test_learned_test_summary_covers_3d_test_set():
    """Plan §1.3: learned deploy on the 8 test instances is tabulated."""

    from visionamr.report import learned_test_summary

    summary = learned_test_summary()
    for fam in ("bearing_block", "deck_panel"):
        assert fam in summary
        assert summary[fam]["supervised"]["n"] == 8
        assert summary[fam]["rl_dqn_s0"]["n"] == 8


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


def test_s4_two_solves_and_ab4_ab5_turn_split_comm_on():
    """Main method is two solves, no split/comm. AB4/AB5 are the on variants."""

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "visionamr" / "campaign.py").read_text()
    assert 'method="vla_ab4_split"' in text
    assert '{"allow_split": True}' in text
    assert 'method="vla_ab5_comm"' in text
    assert '{"allow_communication": True}' in text
    assert "vla_ab4_nosplit" not in text
    assert "vla_ab5_nocomm" not in text
    assert "max_solves=2" in text
    assert "max_solves=4" not in text
    report = (root / "visionamr" / "report.py").read_text()
    assert '("vla_ab4_split", "AB4 split")' in report
    assert '("vla_ab5_comm", "AB5 comm")' in report
    assert '("vla_ab4_nosplit"' not in report
    assert '("vla_ab5_nocomm"' not in report
    plan = (root / "docs" / "EXPERIMENT_PLAN.md").read_text()
    assert "max_solves 2" in plan
    assert "vla_ab4_split" in plan
    assert "vla_ab5_comm" in plan


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
