"""Tables, statistics, and Results markdown for the campaign.

Numbers come only from dumped CalculiX-backed records.  Missing cells
stay blank; hypotheses are judged 成立 / 不成立 / 证据不足.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .campaign import (
    CAMPAIGN,
    FAMILIES_2D,
    FAMILIES_3D,
    PILOT_EQ,
    RESULTS,
    TEST_SEEDS,
    elem_budget,
    instance_dir,
    vla_deliverable,
)


def _e(rec: dict | None) -> float | None:
    if rec is None:
        return None
    v = rec.get("e_energy")
    return None if v is None else float(v)


def _series(fam, key, filename) -> list[dict]:
    path = instance_dir(fam, key) / filename
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("records", [])


def _vla_file(fam, key, head, n_eq) -> list[dict]:
    path = instance_dir(fam, key) / f"records_vla_{head}_b{n_eq}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("records", [])


def _held(series: list[dict], k: int) -> dict | None:
    """Deliverable after k solves: the last record with solve index <= k."""

    if not series:
        return None
    return series[min(k, len(series)) - 1]


def _lp_same_tier(fam: str, key: str, n_eq_budget: int) -> list[dict]:
    """One local-prediction budget group, matched to the VLA equation tier.

    Local-prediction dumps store several short runs in one file.  Stitching
    them is a G6 blacklist item; the extra.budget field is an element count.
    """

    groups: dict[int, list] = {}
    for rec in _series(fam, key, "records_local_prediction.json"):
        groups.setdefault(int(rec.get("extra", {}).get("budget", 0)), []).append(rec)
    if not groups:
        return []
    dim = 3 if fam in FAMILIES_3D else 2
    target = elem_budget(n_eq_budget, dim)
    return groups[min(groups, key=lambda bb: abs(bb - target))]


def error_at_k_table(families=FAMILIES_3D, head: str = "scripted") -> dict:
    """Headline A2′: e_E after k global solves, canonical + test median."""

    out: dict = {"canonical": {}, "test_median": {}, "test_raw": {}}
    for fam in families:
        b = PILOT_EQ[fam]
        # canonical
        dor = _series(fam, "canonical", "records_dorfler.json")
        vla = _vla_file(fam, "canonical", head, b)
        # same resource tier as the VLA column (do not stitch larger LP budgets)
        lp = _lp_same_tier(fam, "canonical", b)
        sup = _series(fam, "canonical", "records_supervised.json")
        rls = [
            _series(fam, "canonical", f"records_rl_dqn_s{seed}.json")
            for seed in range(3)
        ]
        row = {}
        for k in range(1, 7):
            vla_pick = vla_deliverable(vla, k, b)
            dor_rec = dor[k - 1] if k <= len(dor) else None
            dor_n = dor_rec.get("n_equations") if dor_rec else None
            rl_es = [_e(_held(r, k)) for r in rls]
            rl_es = [e for e in rl_es if e is not None]
            row[k] = {
                "dorfler": _e(dor_rec),
                # A2′ compares at equal k; the resource side must stay
                # visible because S2 runs Dörfler to the largest tier.
                "dorfler_n": dor_n,
                "dorfler_frac": (dor_n / b) if dor_n else None,
                # hold the certified iterate after early stop: A2' is
                # error after k solves, not "error only while still iterating"
                "vla": _e(vla_pick),
                "vla_n": vla_pick.get("n_equations") if vla_pick else None,
                "local_prediction": _e(lp[k - 1]) if k <= len(lp) else None,
                # learned deploys stop at k=2; hold their deliverable
                "supervised": _e(_held(sup, k)),
                "rl_dqn": float(np.median(rl_es)) if rl_es else None,
            }
        out["canonical"][fam] = row

        # test set
        raw = []
        for s in TEST_SEEDS:
            key = f"test_{s}"
            dser = _series(fam, key, "records_dorfler.json")
            vser = _vla_file(fam, key, head, b)
            raw.append(
                {
                    "key": key,
                    "k": {
                        k: {
                            "dorfler": _e(dser[k - 1]) if k <= len(dser) else None,
                            "vla": _e(vla_deliverable(vser, k, b)),
                        }
                        for k in (2, 3, 4)
                    },
                }
            )
        out["test_raw"][fam] = raw
        med = {}
        for k in (2, 3, 4):
            dv = [p["k"][k]["dorfler"] for p in raw if p["k"][k]["dorfler"] is not None]
            vv = [p["k"][k]["vla"] for p in raw if p["k"][k]["vla"] is not None]
            med[k] = {
                "dorfler_median": float(np.median(dv)) if dv else None,
                "dorfler_iqr": (
                    [float(np.percentile(dv, 25)), float(np.percentile(dv, 75))]
                    if len(dv) >= 2 else None
                ),
                "vla_median": float(np.median(vv)) if vv else None,
                "vla_iqr": (
                    [float(np.percentile(vv, 25)), float(np.percentile(vv, 75))]
                    if len(vv) >= 2 else None
                ),
                "n_dorfler": len(dv),
                "n_vla": len(vv),
            }
        out["test_median"][fam] = med
    return out


def speedup_table(families=FAMILIES_3D, head: str = "scripted") -> dict:
    """A2″: solves for VLA to reach Dörfler cycle-4 / cycle-6 error."""

    out = {}
    for fam in families:
        b = PILOT_EQ[fam]
        dor = _series(fam, "canonical", "records_dorfler.json")
        vla = _vla_file(fam, "canonical", head, b)
        fam_row = {}
        for target_k in (4, 6):
            if target_k > len(dor) or dor[target_k - 1].get("e_energy") is None:
                fam_row[f"dorfler_k{target_k}"] = None
                continue
            target = float(dor[target_k - 1]["e_energy"])
            vla_k = None
            for k in range(1, 8):
                rec = vla_deliverable(vla, k, b)
                if rec is not None and rec.get("e_energy") is not None and rec["e_energy"] <= target:
                    vla_k = k
                    break
            fam_row[f"dorfler_k{target_k}"] = {
                "target_e": target,
                "vla_solves": vla_k,
                "speedup": (target_k / vla_k) if vla_k else None,
            }
        out[fam] = fam_row
    return out


def budget_rows(families=FAMILIES_3D, head: str = "scripted") -> list[dict]:
    rows = []
    for fam in families:
        b = PILOT_EQ[fam]
        for key in ["canonical"] + [f"test_{s}" for s in TEST_SEEDS]:
            vla = _vla_file(fam, key, head, b)
            pick = vla_deliverable(vla, 99, b)
            if pick:
                rows.append(
                    {
                        "method": "vla",
                        "family": fam,
                        "key": key,
                        "n_eq": pick["n_equations"],
                        "budget": b,
                        "e_energy": pick.get("e_energy"),
                    }
                )
            lp = _lp_same_tier(fam, key, b)
            if lp:
                last = lp[-1]
                rows.append(
                    {
                        "method": "local_prediction",
                        "family": fam,
                        "key": key,
                        "n_eq": last["n_equations"],
                        "budget": b,
                        "e_energy": last.get("e_energy"),
                    }
                )
            dor = _series(fam, key, "records_dorfler.json")
            if dor:
                last = dor[-1]
                rows.append(
                    {
                        "method": "dorfler_zz",
                        "family": fam,
                        "key": key,
                        "n_eq": last["n_equations"],
                        "budget": b,
                        "e_energy": last.get("e_energy"),
                    }
                )
    return rows


def wilcoxon_h1(error_table: dict) -> dict:
    """Paired Wilcoxon on test-set e_E at k=2,3,4.  Never invented."""

    from scipy.stats import wilcoxon

    out = {}
    for fam, raw in error_table.get("test_raw", {}).items():
        out[fam] = {}
        for k in (2, 3, 4):
            pairs = [
                (p["k"][k]["vla"], p["k"][k]["dorfler"])
                for p in raw
                if p["k"][k]["vla"] is not None and p["k"][k]["dorfler"] is not None
            ]
            if len(pairs) < 6:
                out[fam][k] = {
                    "n": len(pairs),
                    "p": None,
                    "median_diff": None,
                    "judgment": "证据不足",
                }
                continue
            v = np.array([a for a, _ in pairs])
            d = np.array([b for _, b in pairs])
            diff = v - d
            try:
                stat = wilcoxon(diff, alternative="less", zero_method="wilcox")
                p = float(stat.pvalue)
            except ValueError:
                p = None
            med = float(np.median(diff))
            if p is None:
                j = "证据不足"
            elif p < 0.05 and med < 0:
                j = "成立"
            else:
                j = "不成立"
            out[fam][k] = {
                "n": len(pairs),
                "p": p,
                "median_diff": med,
                "judgment": j,
            }
    return out


def ablation_rows(families=FAMILIES_3D) -> dict:
    names = [
        ("vla", "full"),
        ("vla_ab1_random", "AB1 random"),
        ("vla_ab2_box", "AB2 box"),
        ("vla_ab3_no_anchor", "AB3 no-anchor"),
        ("vla_ab4_nosplit", "AB4 no-split"),
        ("vla_ab5_nocomm", "AB5 no-comm"),
        ("vla_ab6_nopso", "AB6 no-PSO"),
        ("vla_ab7_k3", "AB7 k=3"),
        ("vla_ab7_k4", "AB7 k=4"),
        ("vla_ab7_k5", "AB7 k=5"),
        ("vla_ab7_k6", "AB7 k=6"),
        ("vla_ab8_s_only", "AB8 s-only"),
        ("vla_ab8_nelder", "AB8 nelder"),
        ("vla_ab9_fixed_q", "AB9 fixed-q"),
        ("vla_ab10_nodrift", "AB10 no-drift"),
        ("vla_ab10_safety092", "AB10 safety 0.92"),
        ("vla_ab10_safety097", "AB10 safety 0.97"),
        ("vla_ab11_no_inplace", "AB11 no-inplace"),
    ]
    out = {}
    for fam in families:
        b = PILOT_EQ[fam]
        rows = []
        for method, label in names:
            if method == "vla":
                recs = _vla_file(fam, "canonical", "scripted", b)
            else:
                recs = _series(fam, "canonical", f"records_{method}.json")
            pick = vla_deliverable(recs, 99, b)
            if pick:
                rows.append(
                    {
                        "name": label,
                        "e_energy": pick.get("e_energy"),
                        "n_eq": pick.get("n_equations"),
                        "solves": pick.get("solve_index"),
                    }
                )
        out[fam] = rows
    return out


def llm_fallback_rate(families=FAMILIES_3D) -> dict:
    n, n_fb = 0, 0
    details = []
    for fam in families:
        b = PILOT_EQ[fam]
        for key in ["canonical"] + [f"test_{s}" for s in TEST_SEEDS]:
            path = instance_dir(fam, key) / f"records_vla_llm_b{b}.json"
            if not path.exists():
                continue
            n += 1
            info = json.loads(path.read_text()).get("vision", {})
            src = info.get("source")
            if src != "llm" and src != "llm_cache":
                n_fb += 1
            details.append({"family": fam, "key": key, "source": src})
    return {
        "n": n,
        "n_fallback": n_fb,
        "rate": (n_fb / n) if n else None,
        "details": details,
    }


def judge_hypotheses(error_table, speedup, budgets, wilcox) -> dict:
    """H1–H4 from measured tables only.  Never upgrades a gap to a claim."""

    h: dict = {}
    canon_ok = []
    missing = False
    for fam, row in error_table.get("canonical", {}).items():
        for k in (2, 3, 4):
            v = row.get(k, {}).get("vla")
            d = row.get(k, {}).get("dorfler")
            if v is None or d is None:
                missing = True
            else:
                canon_ok.append(v < d)
        sp = ((speedup.get(fam) or {}).get("dorfler_k4") or {}).get("speedup")
        if sp is None:
            missing = True
        else:
            canon_ok.append(sp >= 1.5)
    wj = [w["judgment"] for fam in wilcox.values() for w in fam.values()]
    if not canon_ok:
        h["H1"] = "证据不足"
    elif not all(canon_ok):
        h["H1"] = "不成立"
    elif wj and any(j == "不成立" for j in wj):
        h["H1"] = "不成立"
    elif wj and all(j == "成立" for j in wj):
        h["H1"] = "成立"
    else:
        h["H1"] = "canonical 误差@k 与加速比已测；测试集 Wilcoxon 证据不足" if missing or not wj or all(j == "证据不足" for j in wj) else "证据不足"

    # H2 budget band [90%, 105%] for VLA deliverable
    vla_b = [r for r in budgets if r["method"] == "vla" and r.get("n_eq") and r.get("budget")]
    if not vla_b:
        h["H2"] = "证据不足"
    else:
        fracs = [r["n_eq"] / r["budget"] for r in vla_b]
        ok = all(0.90 <= f <= 1.05 for f in fracs)
        h["H2"] = "成立" if ok else "不成立"
        h["H2_fracs"] = fracs

    # H3 stays 证据不足: 3D plan-scale ledgers exist, but there is no
    # pre-registered same-order test that would license 成立. Over-cap
    # RL deploys and leftover 2D runs are not that test.
    h["H3"] = "证据不足"
    # H4: after VLA stops, its deliverable is held; Dörfler keeps iterating.
    # S2 runs the classical loop to the largest budget tier, so also record
    # the crossover restricted to budget-compliant Dörfler iterates
    # (N <= 1.05 x pilot budget, the H2 band upper edge).
    h4 = {}
    h4_capped = {}
    for fam, row in error_table.get("canonical", {}).items():
        v_hold = None
        k_star = None
        k_star_capped = None
        for k in range(1, 7):
            cell = row.get(k, {})
            v = cell.get("vla")
            d = cell.get("dorfler")
            frac = cell.get("dorfler_frac")
            if v is not None:
                v_hold = v
            if v_hold is None or d is None:
                continue
            if d < v_hold:
                if k_star is None:
                    k_star = k
                if k_star_capped is None and frac is not None and frac <= 1.05:
                    k_star_capped = k
        h4[fam] = k_star
        h4_capped[fam] = k_star_capped
    if not error_table.get("canonical"):
        h["H4"] = "证据不足"
    else:
        h["H4"] = h4
        h["H4_capped"] = h4_capped
    return h


def learned_scale() -> dict:
    """What was actually trained, inferred from artifacts. Not the plan."""

    out: dict = {}
    for fam in FAMILIES_3D + FAMILIES_2D:
        entry: dict = {}
        meta = CAMPAIGN / fam / "supervised" / "experts" / "expert_meta.json"
        if meta.exists():
            payload = json.loads(meta.read_text())
            entry["supervised_experts"] = len(payload) if isinstance(payload, list) else None
        n_rl = 0
        for seed in range(3):
            hist = CAMPAIGN / fam / f"rl_seed{seed}" / "training" / "training_history.json"
            rec = CAMPAIGN / fam / "canonical" / f"records_rl_dqn_s{seed}.json"
            if hist.exists():
                payload = json.loads(hist.read_text())
                entry[f"rl_episodes_s{seed}"] = len(payload) if isinstance(payload, list) else None
                n_rl += 1
            elif rec.exists():
                entry[f"rl_episodes_s{seed}"] = None
                n_rl += 1
        if n_rl:
            entry["rl_seeds"] = n_rl
        if entry:
            out[fam] = entry
    return out


def _lp_diag_table(fname: str, families=FAMILIES_3D) -> dict:
    """Per-solve table + convergence stats for one LP diagnostic dump."""

    out = {}
    for fam in families:
        recs = _series(fam, "canonical", fname)
        if not recs:
            continue
        b = PILOT_EQ[fam]
        rows = [
            {
                "solve": r.get("solve_index"),
                "n_eq": r.get("n_equations"),
                "frac": (r.get("n_equations") or 0) / b,
                "e_energy": r.get("e_energy"),
            }
            for r in recs
        ]
        es = [r["e_energy"] for r in rows if r["e_energy"] is not None]
        e3 = rows[2]["e_energy"] if len(rows) >= 3 else None
        e_best = min(es) if es else None
        k_best = (es.index(e_best) + 1) if es else None
        out[fam] = {
            "rows": rows,
            "e_at_3_solves": e3,
            "e_best": e_best,
            "k_best": k_best,
            "e_final": es[-1] if es else None,
            "rel_gain_after_3": (1.0 - e_best / e3) if (e3 and e_best) else None,
            "rel_degradation_after_best": (
                (es[-1] / e_best - 1.0) if (es and e_best) else None
            ),
            "uptick_after_best": bool(
                k_best is not None
                and any(es[i] > es[i - 1] for i in range(k_best, len(es)))
            ),
        }
    return out


def lp_rounds_diagnostic(families=FAMILIES_3D) -> dict:
    """Audit table: local prediction rerun past its 3-solve short run."""

    return _lp_diag_table("records_lp_rounds_diag.json", families)


def lp_naive_diagnostic(families=FAMILIES_3D) -> dict:
    """Weakness evidence: literature deployment (p=1, wide coarsening)."""

    return _lp_diag_table("records_lp_naive_diag.json", families)


def budget_deviation_stats(families=FAMILIES_3D) -> dict:
    """A3/H2 evidence: uncertified methods drift from the budget.

    Fractions are in each method's own contract space: LP and supervised
    promise an element count, VLA promises an equation cap.
    """

    out = {}
    for fam in families:
        b_eq = PILOT_EQ[fam]
        dim = 3 if fam in FAMILIES_3D else 2
        fracs: dict[str, list[float]] = {
            "local_prediction": [],
            "supervised": [],
            "vla": [],
        }
        for key in ["canonical"] + [f"test_{s}" for s in TEST_SEEDS]:
            groups: dict[int, list] = {}
            for rec in _series(fam, key, "records_local_prediction.json"):
                groups.setdefault(int(rec.get("extra", {}).get("budget", 0)), []).append(rec)
            for bb, rr in groups.items():
                if bb > 0 and rr[-1].get("n_elems"):
                    fracs["local_prediction"].append(rr[-1]["n_elems"] / bb)
            sup = _series(fam, key, "records_supervised.json")
            if sup and sup[-1].get("n_elems"):
                fracs["supervised"].append(sup[-1]["n_elems"] / elem_budget(b_eq, dim))
            pick = vla_deliverable(_vla_file(fam, key, "scripted", b_eq), 99, b_eq)
            if pick and pick.get("n_equations"):
                fracs["vla"].append(pick["n_equations"] / b_eq)
        out[fam] = {
            m: {
                "n": len(v),
                "median": float(np.median(v)) if v else None,
                "min": float(np.min(v)) if v else None,
                "max": float(np.max(v)) if v else None,
                "n_in_band_90_105": int(sum(0.90 <= f <= 1.05 for f in v)),
            }
            for m, v in fracs.items()
        }
    return out


OOD_SEEDS = list(range(9500, 9504))


def ood_generalization_table(families=FAMILIES_3D) -> dict:
    """Weakness evidence: learned size field under distribution shift.

    Same three deployments on instances drawn outside the training
    sampler's support; the sup-vs-vla gap shift isolates the shift effect
    (both see a new instance; only supervised carries trained weights).
    """

    out = {}
    for fam in families:
        b = PILOT_EQ[fam]
        dim = 3 if fam in FAMILIES_3D else 2
        rows = []
        gaps_ood = []
        for seed in OOD_SEEDS:
            key = f"ood_{seed}"
            sup = _series(fam, key, "records_supervised.json")
            lp = _series(fam, key, "records_local_prediction.json")
            pick = vla_deliverable(_vla_file(fam, key, "scripted", b), 99, b)
            if not (sup and lp and pick):
                continue
            e_s, e_v = sup[-1].get("e_energy"), pick.get("e_energy")
            rows.append(
                {
                    "key": key,
                    "supervised_e": e_s,
                    "supervised_frac": (sup[-1].get("n_elems") or 0) / elem_budget(b, dim),
                    "lp3_e": lp[-1].get("e_energy"),
                    "lp3_frac": (lp[-1].get("n_elems") or 0) / elem_budget(b, dim),
                    "vla_e": e_v,
                    "vla_frac": (pick.get("n_equations") or 0) / b,
                }
            )
            if e_s is not None and e_v is not None:
                gaps_ood.append(e_s - e_v)
        if not rows:
            continue
        # in-distribution contrast on the 8 test instances
        gaps_test = []
        for s in TEST_SEEDS:
            key = f"test_{s}"
            sup = _series(fam, key, "records_supervised.json")
            pick = vla_deliverable(_vla_file(fam, key, "scripted", b), 99, b)
            if sup and pick and sup[-1].get("e_energy") is not None and pick.get("e_energy") is not None:
                gaps_test.append(sup[-1]["e_energy"] - pick["e_energy"])
        out[fam] = {
            "rows": rows,
            "median_gap_sup_minus_vla_ood": float(np.median(gaps_ood)) if gaps_ood else None,
            "median_gap_sup_minus_vla_test": float(np.median(gaps_test)) if gaps_test else None,
        }
    return out


def training_cost_rows() -> list[dict]:
    """A4: offline training solves, never mixed into the deploy k-axis."""

    rows: list[dict] = []
    for fam in FAMILIES_3D + FAMILIES_2D:
        meta = CAMPAIGN / fam / "supervised" / "experts" / "expert_meta.json"
        if meta.exists():
            payload = json.loads(meta.read_text())
            cost = meta.parent / "training_cost.json"
            train_solves = None
            if cost.exists():
                train_solves = json.loads(cost.read_text()).get("total_solves")
            rows.append(
                {
                    "family": fam,
                    "kind": "supervised_experts",
                    "n_experts": len(payload) if isinstance(payload, list) else None,
                    "episodes": None,
                    "train_solves": train_solves,
                }
            )
        for seed in range(3):
            hist = CAMPAIGN / fam / f"rl_seed{seed}" / "training" / "training_history.json"
            if not hist.exists():
                continue
            payload = json.loads(hist.read_text())
            if not isinstance(payload, list):
                continue
            rows.append(
                {
                    "family": fam,
                    "kind": f"rl_s{seed}",
                    "n_experts": None,
                    "episodes": len(payload),
                    "train_solves": int(sum(int(x.get("solves") or 0) for x in payload)),
                }
            )
    return rows


def learned_deploy_rows(families=FAMILIES_3D) -> list[dict]:
    """Last deploy iterate of supervised / RL. Not mixed into the VLA k-axis."""

    rows: list[dict] = []
    for fam in families:
        b = PILOT_EQ[fam]
        for method, fname in (
            ("supervised", "records_supervised.json"),
            ("rl_dqn_s0", "records_rl_dqn_s0.json"),
            ("rl_dqn_s1", "records_rl_dqn_s1.json"),
            ("rl_dqn_s2", "records_rl_dqn_s2.json"),
        ):
            recs = _series(fam, "canonical", fname)
            if not recs:
                continue
            last = recs[-1]
            n_eq = last.get("n_equations")
            rows.append(
                {
                    "family": fam,
                    "key": "canonical",
                    "method": method,
                    "solves": last.get("solve_index"),
                    "n_eq": n_eq,
                    "e_energy": last.get("e_energy"),
                    "budget": b,
                    "frac": (n_eq / b) if n_eq else None,
                    "over_cap": bool(n_eq is not None and n_eq > b),
                }
            )
    return rows


def learned_test_summary(families=FAMILIES_3D) -> dict:
    """Plan §1.3: learned methods are judged on the test set, not upgraded to H3."""

    out: dict = {}
    for fam in families:
        b = PILOT_EQ[fam]
        fam_row = {}
        for method, fname in (
            ("supervised", "records_supervised.json"),
            ("rl_dqn_s0", "records_rl_dqn_s0.json"),
            ("rl_dqn_s1", "records_rl_dqn_s1.json"),
            ("rl_dqn_s2", "records_rl_dqn_s2.json"),
        ):
            es, ns, fracs = [], [], []
            for seed in TEST_SEEDS:
                recs = _series(fam, f"test_{seed}", fname)
                if not recs:
                    continue
                last = recs[-1]
                e = last.get("e_energy")
                n = last.get("n_equations")
                if e is not None:
                    es.append(float(e))
                if n is not None:
                    ns.append(int(n))
                    fracs.append(n / b)
            if not es:
                continue
            fam_row[method] = {
                "n": len(es),
                "e_median": float(np.median(es)),
                "e_iqr": [float(np.percentile(es, 25)), float(np.percentile(es, 75))],
                "n_eq_median": float(np.median(ns)) if ns else None,
                "frac_median": float(np.median(fracs)) if fracs else None,
                "n_over_cap": int(sum(f > 1.0 for f in fracs)),
            }
        if fam_row:
            out[fam] = fam_row
    return out


def build_all_tables(campaign_dir: Path | None = None) -> dict:
    err = error_at_k_table()
    sp = speedup_table()
    bgt = budget_rows()
    wil = wilcoxon_h1(err)
    hyp = judge_hypotheses(err, sp, bgt, wil)
    return {
        "error_at_k": err,
        "speedup": sp,
        "budget_rows": bgt,
        "wilcoxon": wil,
        "ablation": ablation_rows(),
        "llm_fallback": llm_fallback_rate(),
        "learned_scale": learned_scale(),
        "learned_deploy": learned_deploy_rows(),
        "learned_test": learned_test_summary(),
        "training_cost": training_cost_rows(),
        "lp_rounds_diag": lp_rounds_diagnostic(),
        "lp_naive_diag": lp_naive_diagnostic(),
        "budget_deviation": budget_deviation_stats(),
        "ood_generalization": ood_generalization_table(),
        "hypotheses": hyp,
        "campaign_dir": str(campaign_dir or CAMPAIGN),
    }


def write_tables(tables: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "all.json").write_text(json.dumps(tables, indent=1, default=str))
    # markdown fragments
    lines = ["# Campaign tables", ""]
    lines.append("## Error @ k (canonical, pilot budget)")
    lines.append("")
    lines.append("| family | k | Dörfler | VLA | local_prediction |")
    lines.append("|---|---:|---:|---:|---:|")
    for fam, row in tables["error_at_k"].get("canonical", {}).items():
        for k, cell in row.items():
            def fmt(v):
                return "—" if v is None else f"{v:.4f}"
            lines.append(
                f"| {fam} | {k} | {fmt(cell['dorfler'])} | {fmt(cell['vla'])} | "
                f"{fmt(cell['local_prediction'])} |"
            )
    (outdir / "error_at_k.md").write_text("\n".join(lines) + "\n")


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_results_md(tables: dict) -> str:
    hyp = tables.get("hypotheses", {})
    lines = [
        "# Results (campaign execution of EXPERIMENT_PLAN.md)",
        "",
        "本文只报告本仓库本机战役写入 `results/campaign/` 的 CalculiX 记录。",
        "未跑完的格子留空。假设判定只允许：成立 / 不成立 / 证据不足。",
        "不把试点登记数字写成新结论。",
        "",
        "## 假设判定",
        "",
        f"- **H1**（少求解次数处 VLA 优于 Dörfler，加速比 ≥ 1.5×）：**{hyp.get('H1', '证据不足')}**",
        f"- **H2**（预算占用 ∈ [90%, 105%]）：**{hyp.get('H2', '证据不足')}**",
        f"- **H3**（免训练达到学习方法同量级）：**{hyp.get('H3', '证据不足')}**",
        f"- **H4**（交叉点 k*，Dörfler 未封顶）：`{hyp.get('H4', '证据不足')}`",
        f"- H4 预算内交叉（Dörfler N ≤ 1.05×试点档时）：`{hyp.get('H4_capped', '证据不足')}`",
        "",
        "## A2′ 误差 @ k 次全局求解（canonical，试点预算档）",
        "",
        "带 `*` 的 Dörfler 轮次其 N 超过试点预算 105%（S2 经典循环封顶在最大档，",
        "资源侧见 N/B 列与 A3）。监督/RL 第 2 次求解后交付并保持。",
        "",
        "| family | k | Dörfler | Dörfler N/B | VLA (scripted) | local_prediction | supervised | RL (3种子中位) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fam, row in tables.get("error_at_k", {}).get("canonical", {}).items():
        for k, cell in row.items():
            frac = cell.get("dorfler_frac")
            over = frac is not None and frac > 1.05
            dtxt = _fmt(cell.get("dorfler"))
            if dtxt != "—" and over:
                dtxt += "*"
            lines.append(
                f"| {fam} | {k} | {dtxt} | {_fmt(frac, 2)} | {_fmt(cell.get('vla'))} | "
                f"{_fmt(cell.get('local_prediction'))} | {_fmt(cell.get('supervised'))} | "
                f"{_fmt(cell.get('rl_dqn'))} |"
            )
    lines += [
        "",
        "## A2″ 加速比（到达 Dörfler 第 4/6 轮误差）",
        "",
        "```json",
        json.dumps(tables.get("speedup", {}), indent=1, default=str),
        "```",
        "",
        "## 测试集 Wilcoxon（H1）",
        "",
        "```json",
        json.dumps(tables.get("wilcoxon", {}), indent=1, default=str),
        "```",
        "",
        "## 测试集中位数 [IQR]（计划 §4）",
        "",
        "```json",
        json.dumps(tables.get("error_at_k", {}).get("test_median", {}), indent=1, default=str),
        "```",
        "",
        "## A4 训练成本（离线求解，不进部署 k 轴）",
        "",
        "监督行的 train_solves 为专家库制造（探针+Dörfler 到帽）实际发生的 CalculiX 求解数，",
        "计自各 expert 运行目录；经典方法与 VLA 无离线成本。",
        "",
        "```json",
        json.dumps(tables.get("training_cost", {}), indent=1, default=str),
        "```",
        "",
    ]
    diag = tables.get("lp_rounds_diag") or {}
    if diag:
        lines += [
            "## 局部预测轮数诊断（canonical，试点档；审核补充）",
            "",
            "计划 §3.3 锁定每档 probe+2 轮=3 次求解；此诊断放开到 7 次以核查该限制是否压误差。",
            "",
            "| family | solve | N | N/B | e_E |",
            "|---|---:|---:|---:|---:|",
        ]
        for fam, info in diag.items():
            for r in info.get("rows", []):
                lines.append(
                    f"| {fam} | {r['solve']} | {r['n_eq']} | "
                    f"{_fmt(r['frac'], 2)} | {_fmt(r['e_energy'])} |"
                )
        for fam, info in diag.items():
            gain = info.get("rel_gain_after_3")
            lines.append("")
            lines.append(
                f"- {fam}：3 次求解 e={_fmt(info.get('e_at_3_solves'))}，"
                f"7 次内最优 e={_fmt(info.get('e_best'))}（第 {info.get('k_best')} 次），"
                f"额外 4 次求解的相对改善 {_fmt(100 * gain if gain is not None else None, 1)}%；"
                f"最优后出现回弹：{'是' if info.get('uptick_after_best') else '否'}。"
            )
    lines += [
        "",
        "## LLM 视觉头回退率",
        "",
        "```json",
        json.dumps(tables.get("llm_fallback", {}), indent=1, default=str),
        "```",
        "",
        "## 消融（canonical × 试点档）",
        "",
        "```json",
        json.dumps(tables.get("ablation", {}), indent=1, default=str),
        "```",
        "",
        "## 学习方法测试集中位数（计划 §1.3；不升格为 H3）",
        "",
        "```json",
        json.dumps(tables.get("learned_test", {}), indent=1, default=str),
        "```",
        "",
        "## 学习方法部署账本（3D canonical；不升格为 H3）",
        "",
        "实际训练规模（从产物推断；3D 对齐计划 24 专家 / 120×3）：",
        "",
        "```json",
        json.dumps(tables.get("learned_scale", {}), indent=1, default=str),
        "```",
        "",
        "| family | method | solves | n_eq | e_energy | budget frac | over cap |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for r in tables.get("learned_deploy", []) or []:
        frac = r.get("frac")
        lines.append(
            f"| {r.get('family')} | {r.get('method')} | {r.get('solves')} | "
            f"{r.get('n_eq')} | {_fmt(r.get('e_energy'))} | "
            f"{_fmt(frac)} | {'yes' if r.get('over_cap') else 'no'} |"
        )
    lines += ["", "## 审核补充：对照方法缺点实证（全部为真实 CalculiX 求解）", ""]
    naive = tables.get("lp_naive_diag") or {}
    if naive:
        lines += [
            "### E1 局部预测·朴素部署振荡（§3.3.1 失稳模式复现）",
            "",
            "同一逐单元预测循环，改用文献朴素配方：指数 p=1、尺寸比界 [1/6, 3.0]",
            "（v2 锁定值为 p=(d+2)/2、尺寸比界 [1/6, 1.8]）。",
            "",
            "| family | solve | N | N/B | e_E |",
            "|---|---:|---:|---:|---:|",
        ]
        for fam, info in naive.items():
            for r in info.get("rows", []):
                lines.append(
                    f"| {fam} | {r['solve']} | {r['n_eq']} | "
                    f"{_fmt(r['frac'], 2)} | {_fmt(r['e_energy'])} |"
                )
        v2diag = tables.get("lp_rounds_diag") or {}
        for fam, info in naive.items():
            deg = info.get("rel_degradation_after_best")
            v2_deg = (v2diag.get(fam) or {}).get("rel_degradation_after_best")
            lines.append("")
            lines.append(
                f"- {fam}：最优 e={_fmt(info.get('e_best'))}（第 {info.get('k_best')} 次），"
                f"末轮 e={_fmt(info.get('e_final'))}，最优后恶化 "
                f"{_fmt(100 * deg if deg is not None else None, 1)}%；"
                f"回弹：{'是' if info.get('uptick_after_best') else '否'}。"
                f"对照：v2 修正版（上节）最优后恶化 "
                f"{_fmt(100 * v2_deg if v2_deg is not None else None, 1)}%"
                f"{'（亦有回弹）' if (v2diag.get(fam) or {}).get('uptick_after_best') else ''}。"
            )
        lines.append("")
    dev = tables.get("budget_deviation") or {}
    if dev:
        lines += [
            "### E2 无认证语义的预算偏差（canonical+测试 8，按各自契约空间）",
            "",
            "LP/监督承诺单元数（frac=单元数/单元预算）；VLA 承诺方程帽（frac=N/预算）。",
            "",
            "| family | method | n | median | min | max | in [0.90,1.05] |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for fam, methods in dev.items():
            for m, s in methods.items():
                lines.append(
                    f"| {fam} | {m} | {s['n']} | {_fmt(s['median'], 2)} | "
                    f"{_fmt(s['min'], 2)} | {_fmt(s['max'], 2)} | "
                    f"{s['n_in_band_90_105']}/{s['n']} |"
                )
        lines.append("")
    ood = tables.get("ood_generalization") or {}
    if ood:
        lines += [
            "### E3 监督·分布偏移（OOD 实例：全部参数在训练采样器支撑集外）",
            "",
            "同场部署：监督（2 次求解）、LP 短跑（3 次）、VLA scripted。",
            "免训练方法对新实例本来就是零样本；差距变化隔离偏移效应。",
            "",
            "| family | key | supervised e (frac) | lp3 e (frac) | VLA e (frac) |",
            "|---|---|---:|---:|---:|",
        ]
        for fam, info in ood.items():
            for r in info.get("rows", []):
                lines.append(
                    f"| {fam} | {r['key']} | {_fmt(r['supervised_e'])} ({_fmt(r['supervised_frac'], 2)}) | "
                    f"{_fmt(r['lp3_e'])} ({_fmt(r['lp3_frac'], 2)}) | "
                    f"{_fmt(r['vla_e'])} ({_fmt(r['vla_frac'], 2)}) |"
                )
        for fam, info in ood.items():
            lines.append("")
            lines.append(
                f"- {fam}：监督−VLA 中位差距，分布内 "
                f"{_fmt(info.get('median_gap_sup_minus_vla_test'))} → OOD "
                f"{_fmt(info.get('median_gap_sup_minus_vla_ood'))}。"
            )
        lines.append("")
    lines += [
        "",
        "## 诚实边界",
        "",
        "- Dörfler 的渐近最优性不在本文争夺范围；k* 交叉若出现必须画出。",
        "- A2′ 的 Dörfler 列不按试点档封顶（S2 给经典循环的帽是最大档）；超 105% 的轮次带 `*`，预算内交叉在假设判定单列。A2″ 的第 4/6 轮目标误差同样取自该未封顶序列。",
        "- 局部预测是逐单元一步预测，不是分区方法；其预算偏差如实列入。",
        "- VLA 第 2 次求解的初始尺寸复用同一逐单元等分布预测（按区几何平均，§3.6），故 k=2 处两者相近是结构性的；其后 VLA 走实测指数/漂移反馈/硬帽投影/就地认证（AB5/AB6/AB9–AB11 量化各自贡献），局部预测走再等分布（轮数诊断见上）。",
        "- 监督专家库由训练实例上的 Dörfler-到帽循环蒸馏（离线求解已计入 A4）；其部署无预算帽语义，canonical 交付停在 44–52% 档位。",
        "- LLM 头失败回退 Scripted 时计入回退率，不把 Scripted 数字标成 LLM。",
        "- 训练期求解（监督专家库、RL episode）单列，不混进部署 k 轴。",
        "- 论文主文只报 3D。S5 3D 监督 24 专家；S6 3D RL 120 回合 × 3 种子。H3 仍为证据不足。",
        "- 图政策（§8）：局部预测按预算档分列，不跨档拼线；e_E–N 图标注加密原子，不是单一 Pareto；k* 画在误差@k 主图上。",
        "- 学习方法按 §1.3 在测试集 8 实例上汇总；该表不升格为 H3。",
        "",
    ]
    gates_path = RESULTS / "gates.json"
    if gates_path.exists():
        try:
            gates = json.loads(gates_path.read_text())
        except json.JSONDecodeError:
            gates = {}
        lines += [
            "## S3 验收门",
            "",
            "| gate | pass |",
            "|---|---|",
        ]
        for name in ("G1", "G2", "G3", "G4", "G5", "G7"):
            g = gates.get(name) or {}
            lines.append(f"| {name} | {'PASS' if g.get('pass') else 'FAIL'} |")
        lines.append("")
    return "\n".join(lines) + "\n"
