"""Walk the vla-real-workflow skill on one 3-D instance.

  1. load the agent judgment (regions + grades; no sizes)
  2. run_vla: judge → one-shot tool tweak → mesh → solve → judge → tweak
  3. one-step LP: probe + one predicted remesh
  4. compare e_E, e_qoi, N/B, and whether the grade ranking survived

No Gmsh size-search loop.  The eye does not assign parameters.
Does not write into results/campaign/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionamr.baselines.local_prediction import run_local_prediction
from visionamr.experiment import FemRunner
from visionamr.geometry import PROBLEM_FACTORIES
from visionamr.vla.partition import CachedDrawingPartitioner
from visionamr.vla.pipeline import VLAConfig, run_vla

EQ_PER_ELEM = {2: 1.5, 3: 0.62}


def _row(r, n_eq_budget: int) -> dict:
    return {
        "k": r.solve_index,
        "method": r.method,
        "stage": r.stage,
        "n_equations": r.n_equations,
        "n_elems": r.n_elems,
        "n_over_budget": r.n_equations / n_eq_budget,
        "e_energy": r.e_energy,
        "e_qoi": r.e_qoi,
        "sum_eta2": r.extra.get("sum_eta2"),
        "budget_ok": r.n_equations <= n_eq_budget,
        "regions": r.extra.get("regions"),
        "grades": r.extra.get("grades"),
        "pso_evals": (r.extra.get("pso") or {}).get("evals"),
    }


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or len(a) != len(b):
        return None
    from scipy.stats import spearmanr

    rho, _ = spearmanr(a, b)
    return float(rho) if np.isfinite(rho) else None


def _table(recs, n_eq_budget: int) -> str:
    lines = [
        f"{'k':>2} {'stage':18s} {'N_eq':>8} {'N/B':>6} {'e_E':>8} {'e_qoi':>8} {'Ση²':>10}"
    ]
    for r in recs:
        e = r.e_energy if r.e_energy is None else float(r.e_energy)
        q = r.e_qoi if r.e_qoi is None else float(r.e_qoi)
        eta = r.extra.get("sum_eta2", float("nan"))
        e = float("nan") if e is None else e
        q = float("nan") if q is None else q
        lines.append(
            f"{r.solve_index:2d} {r.stage:18s} {r.n_equations:8d} "
            f"{r.n_equations / n_eq_budget:6.2f} {e:8.4f} {q:8.4f} {eta:10.3f}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="bearing_block")
    ap.add_argument("--eye", default="tests/fixtures/bearing_block_eye.json")
    ap.add_argument(
        "--revise",
        default="tests/fixtures/bearing_block_eye_revise1.json",
        help="next judgment after the first solve (grades only, no API)",
    )
    ap.add_argument("--n-eq-budget", type=int, default=8000)
    ap.add_argument("--max-solves", type=int, default=2)
    ap.add_argument("--out", default="/opt/cursor/artifacts/vla_skill_walkthrough")
    ap.add_argument(
        "--reference",
        default="results/campaign/bearing_block/canonical/reference.json",
    )
    args = ap.parse_args()

    out = Path(args.out)
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    ref = Path(args.reference)
    if ref.exists():
        shutil.copy(ref, work / "reference.json")

    problem = PROBLEM_FACTORIES[args.problem]()
    n_elem = int(round(args.n_eq_budget / EQ_PER_ELEM[problem.dim]))
    revisions = [args.revise] if args.revise else []
    head = CachedDrawingPartitioner(args.eye, revisions=revisions)
    seeds = head.propose(problem)
    grades0 = dict(head.last_grades)
    print(
        f"[judge] {len(head.last_drawings)} regions grades={grades0}",
        flush=True,
    )

    runner = FemRunner(problem, work, ccx_timeout=1800.0)
    ref_rec = runner.ensure_reference()
    print(f"[ref] N={ref_rec.n_equations} U={ref_rec.U_total:.6g}", flush=True)

    cfg = VLAConfig(n_eq_budget=args.n_eq_budget, max_solves=args.max_solves)
    vla = run_vla(runner, head, cfg, method="vla_eye")
    vla_recs = [r for r in runner.records if r.method == "vla_eye"]
    for r in vla_recs:
        r.extra.setdefault("n_eq_budget", args.n_eq_budget)

    runner.reset_counter()
    run_local_prediction(runner, budgets=[n_elem], rounds=1, method="lp_eye")
    lp_recs = [r for r in runner.records if r.method == "lp_eye"]
    for r in lp_recs:
        r.extra.setdefault("n_eq_budget", args.n_eq_budget)

    names = [s.name for s in seeds]
    g_final = dict(getattr(head, "last_grades", {}) or {})
    rho = _spearman(
        [float(grades0.get(n, 5)) for n in names if n in g_final],
        [float(g_final[n]) for n in names if n in g_final],
    )
    rem_g = g_final.get("field", 5)
    rim_g = [g_final[n] for n in names if n in g_final and ("rim" in n or "patch" in n)]
    rank_ok = bool(rim_g) and max(rim_g) < rem_g

    print("\n[VLA eye]\n" + _table(vla_recs, args.n_eq_budget), flush=True)
    print("\n[LP one-step]\n" + _table(lp_recs, args.n_eq_budget), flush=True)
    print(
        f"[rank] spearman(grades0, grades_final)={rho}  "
        f"rims_finer_than_remainder={rank_ok}  "
        f"thoughts={ (vla.info or {}).get('thoughts') }",
        flush=True,
    )

    payload = {
        "protocol": "vla-real-workflow",
        "problem": problem.instance_id,
        "n_eq_budget": args.n_eq_budget,
        "eye": args.eye,
        "vla_defaults": {
            "max_solves": cfg.max_solves,
            "allow_communication": cfg.allow_communication,
            "allow_split": cfg.allow_split,
        },
        "grades0": grades0,
        "thoughts": (vla.info or {}).get("thoughts"),
        "vla": asdict(vla),
        "vla_records": [_row(r, args.n_eq_budget) for r in vla_recs],
        "lp_records": [_row(r, args.n_eq_budget) for r in lp_recs],
        "drawing_rank": {
            "spearman": rho,
            "rims_finer_than_remainder": rank_ok,
            "grades0": grades0,
            "grades_final": g_final,
        },
        "note": "new-protocol trial; do not paste into old A2′ tables",
    }
    (out / "comparison.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {out / 'comparison.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
