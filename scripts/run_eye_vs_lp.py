"""Walk the vla-real-workflow skill on one 3-D instance.

  1. load the agent eye drawing (no solve)
  2. scale the drawn sizes to the element budget (Gmsh only; ranking fixed)
  3. run_vla: first solve + agent decision from result/leftover (PSO only if overshoot)
  4. one-step LP: probe + one predicted remesh
  5. compare e_E, e_qoi, N/B, and whether the drawing ranking survived

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
from visionamr.vla.drawing import scale_drawings_to_elem_budget
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
        e = r.e_energy if r.e_energy is not None else float("nan")
        q = r.e_qoi if r.e_qoi is not None else float("nan")
        eta = r.extra.get("sum_eta2", float("nan"))
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
        help="agent's next decision after the first solve (no API)",
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
    raw = CachedDrawingPartitioner(args.eye)
    seeds = raw.propose(problem)
    drawings = list(raw.last_drawings)
    rem = next(s.h for s in seeds if s.origin == "coarse")
    print(f"[eye] loaded {len(drawings)} drawings remainder={rem:.2f}", flush=True)
    scaled, rem, n_mesh = scale_drawings_to_elem_budget(
        drawings, rem, problem, n_elem
    )
    spec_path = out / "eye_budget_scaled.json"
    spec_path.write_text(
        json.dumps(
            {
                "remainder_fineness_fraction": rem / problem.h0,
                "regions": [
                    {
                        "name": d.name,
                        "view": d.view,
                        "fineness_fraction": d.h / problem.h0,
                        "polygon": [list(p) for p in d.polygon],
                        **({} if d.plane is None else {"plane": d.plane}),
                        **({} if d.cut is None else {"cut": d.cut}),
                        **({} if d.slab is None else {"slab": d.slab}),
                    }
                    for d in scaled
                ],
            },
            indent=1,
        )
    )
    print(f"[eye] budget-scaled mesh n_elem={n_mesh} target={n_elem}", flush=True)
    h_eye = {d.name: d.h for d in scaled}
    h_eye["field"] = rem

    revisions = [args.revise] if args.revise else []
    head = CachedDrawingPartitioner(str(spec_path), revisions=revisions)
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

    h_final = {
        n: h for n, h in zip(vla.seeds_final, vla.sizes_final) if n in h_eye
    }
    names = [n for n in h_eye if n in h_final]
    rho = _spearman([h_eye[n] for n in names], [h_final[n] for n in names])
    rem_h = h_final.get("field") or rem
    rims = [h_final[n] for n in names if "rim" in n or "patch" in n]
    rank_ok = bool(rims) and rem_h > 0 and min(rims) < rem_h

    print("\n[VLA eye]\n" + _table(vla_recs, args.n_eq_budget), flush=True)
    print("\n[LP one-step]\n" + _table(lp_recs, args.n_eq_budget), flush=True)
    print(
        f"[rank] spearman(h_eye, h_final)={rho}  "
        f"rims_finer_than_remainder={rank_ok}",
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
        "first_mesh_elems": n_mesh,
        "thoughts": (vla.info or {}).get("thoughts"),
        "vla": asdict(vla),
        "vla_records": [_row(r, args.n_eq_budget) for r in vla_recs],
        "lp_records": [_row(r, args.n_eq_budget) for r in lp_recs],
        "drawing_rank": {
            "spearman": rho,
            "rims_finer_than_remainder": rank_ok,
            "h_eye": h_eye,
            "h_final": h_final,
        },
        "note": "new-protocol trial; do not paste into old A2′ tables",
    }
    (out / "comparison.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {out / 'comparison.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
