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
    EQ_PER_ELEM,
    FAMILIES_2D,
    FAMILIES_3D,
    PILOT_EQ,
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


def error_at_k_table(families=FAMILIES_3D, head: str = "scripted") -> dict:
    """Headline A2′: e_E after k global solves, canonical + test median."""

    out: dict = {"canonical": {}, "test_median": {}, "test_raw": {}}
    for fam in families:
        b = PILOT_EQ[fam]
        # canonical
        dor = _series(fam, "canonical", "records_dorfler.json")
        vla = _vla_file(fam, "canonical", head, b)
        lp_all = _series(fam, "canonical", "records_local_prediction.json")
        lp_groups: dict[int, list] = {}
        for r in lp_all:
            lp_groups.setdefault(int(r.get("extra", {}).get("budget", 0)), []).append(r)
        # same resource tier as the VLA column (do not stitch larger LP budgets)
        dim = 3 if fam in FAMILIES_3D else 2
        target_elems = elem_budget(b, dim)
        lp = []
        if lp_groups:
            key_b = min(lp_groups, key=lambda bb: abs(bb - target_elems))
            lp = lp_groups[key_b]
        row = {}
        for k in range(1, 7):
            row[k] = {
                "dorfler": _e(dor[k - 1]) if k <= len(dor) else None,
                # hold the certified iterate after early stop: A2' is
                # error after k solves, not "error only while still iterating"
                "vla": _e(vla_deliverable(vla, k, b)),
                "local_prediction": _e(lp[k - 1]) if k <= len(lp) else None,
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
            lp = _series(fam, key, "records_local_prediction.json")
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

    # H3 stays 证据不足 until learned methods exist at the *plan* scale
    # (24 experts / 120–300 episodes × 3 seeds) and a locked comparison
    # is run. Reduced-scale ledgers below are not that comparison.
    h["H3"] = "证据不足"
    # H4: after VLA stops, its deliverable is held; Dörfler keeps iterating.
    h4 = {}
    for fam, row in error_table.get("canonical", {}).items():
        v_hold = None
        k_star = None
        for k in range(1, 7):
            v = row.get(k, {}).get("vla")
            d = row.get(k, {}).get("dorfler")
            if v is not None:
                v_hold = v
            if v_hold is None or d is None:
                continue
            if d < v_hold:
                k_star = k
                break
        h4[fam] = k_star
    if not error_table.get("canonical"):
        h["H4"] = "证据不足"
    else:
        h["H4"] = h4
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
        hist = CAMPAIGN / fam / "rl_seed0" / "training" / "training_history.json"
        if hist.exists():
            payload = json.loads(hist.read_text())
            entry["rl_episodes_s0"] = len(payload) if isinstance(payload, list) else None
            entry["rl_seeds"] = 1
        if entry:
            out[fam] = entry
    return out


def learned_deploy_rows(families=FAMILIES_3D + FAMILIES_2D) -> list[dict]:
    """Last deploy iterate of supervised / RL. Not mixed into the VLA k-axis."""

    rows: list[dict] = []
    for fam in families:
        b = PILOT_EQ[fam]
        for method, fname in (
            ("supervised", "records_supervised.json"),
            ("rl_dqn_s0", "records_rl_dqn_s0.json"),
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
        f"- **H4**（交叉点 k*）：`{hyp.get('H4', '证据不足')}`",
        "",
        "## A2′ 误差 @ k 次全局求解（canonical，试点预算档）",
        "",
        "| family | k | Dörfler | VLA (scripted) | local_prediction |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam, row in tables.get("error_at_k", {}).get("canonical", {}).items():
        for k, cell in row.items():
            lines.append(
                f"| {fam} | {k} | {_fmt(cell.get('dorfler'))} | {_fmt(cell.get('vla'))} | "
                f"{_fmt(cell.get('local_prediction'))} |"
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
        "## 学习方法部署账本（canonical；不升格为 H3）",
        "",
        "实际训练规模（从产物推断，小于 EXPERIMENT_PLAN 的 24 专家 / 120–300×3 种子）：",
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
    lines += [
        "",
        "## 诚实边界",
        "",
        "- Dörfler 的渐近最优性不在本文争夺范围；k* 交叉若出现必须画出。",
        "- 局部预测是逐单元一步预测，不是分区方法；其预算偏差如实列入。",
        "- LLM 头失败回退 Scripted 时计入回退率，不把 Scripted 数字标成 LLM。",
        "- 训练期求解（监督专家库、RL episode）单列，不混进部署 k 轴。",
        "- 学习方法按缩小规模登记：deck 监督 8 专家，plate_holes RL 80×1，deck RL 40×1。H3 仍为证据不足。",
        "",
    ]
    return "\n".join(lines) + "\n"
