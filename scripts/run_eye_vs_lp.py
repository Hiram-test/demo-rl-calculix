"""Walk the vla-real-workflow skill on one 3-D instance.

  1. load the agent eye drawing (no solve)
  2. scale the drawn sizes to the element budget (Gmsh only)
  3. run_vla: first solve on that mesh, residual mid-rounds, last PSO
  4. run one-step local prediction at the same equation budget
  5. compare e@k honestly

Does not write into results/campaign/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionamr.baselines.local_prediction import run_local_prediction
from visionamr.experiment import FemRunner
from visionamr.geometry import PROBLEM_FACTORIES
from visionamr.vla.drawing import scale_drawings_to_elem_budget
from visionamr.vla.partition import CachedDrawingPartitioner
from visionamr.vla.pipeline import VLAConfig, run_vla

EQ_PER_ELEM = {2: 1.5, 3: 0.62}


def _row(r) -> dict:
    return {
        "k": r.solve_index,
        "method": r.method,
        "stage": r.stage,
        "n_equations": r.n_equations,
        "n_elems": r.n_elems,
        "e_energy": r.e_energy,
        "sum_eta2": r.extra.get("sum_eta2"),
        "budget_ok": r.n_equations <= r.extra.get("n_eq_budget", 10**18)
        if "n_eq_budget" in r.extra
        else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="bearing_block")
    ap.add_argument("--eye", default="tests/fixtures/bearing_block_eye.json")
    ap.add_argument("--n-eq-budget", type=int, default=8000)
    ap.add_argument("--max-solves", type=int, default=4)
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

    head = CachedDrawingPartitioner(str(spec_path))
    runner = FemRunner(problem, work, ccx_timeout=1800.0)
    ref_rec = runner.ensure_reference()
    print(f"[ref] N={ref_rec.n_equations} U={ref_rec.U_total:.6g}", flush=True)

    cfg = VLAConfig(n_eq_budget=args.n_eq_budget, max_solves=args.max_solves)
    vla = run_vla(runner, head, cfg, method="vla_eye")
    vla_recs = [r for r in runner.records if r.method == "vla_eye"]
    for r in vla_recs:
        r.extra.setdefault("n_eq_budget", args.n_eq_budget)
    print(
        f"[vla] solves={vla.solves} early={vla.stopped_early} "
        f"s={vla.s_last:.3f} kappa={vla.kappa_last:.3f} "
        f"final={vla.info.get('pso', {}).get('final_revision')} "
        f"mode={vla.info.get('pso', {}).get('mode')}",
        flush=True,
    )

    runner.reset_counter()
    run_local_prediction(runner, budgets=[n_elem], rounds=2, method="lp_eye")
    lp_recs = [r for r in runner.records if r.method == "lp_eye"]
    for r in lp_recs:
        r.extra.setdefault("n_eq_budget", args.n_eq_budget)

    def table(recs):
        lines = [f"{'k':>2} {'stage':18s} {'N_eq':>8} {'e_E':>8} {'Ση²':>10} {'N/B':>6}"]
        for r in recs:
            e = r.e_energy if r.e_energy is not None else float("nan")
            eta = r.extra.get("sum_eta2", float("nan"))
            lines.append(
                f"{r.solve_index:2d} {r.stage:18s} {r.n_equations:8d} "
                f"{e:8.4f} {eta:10.3f} {r.n_equations / args.n_eq_budget:6.2f}"
            )
        return "\n".join(lines)

    print("\n[VLA eye]\n" + table(vla_recs), flush=True)
    print("\n[LP one-step]\n" + table(lp_recs), flush=True)

    payload = {
        "protocol": "vla-real-workflow",
        "problem": problem.instance_id,
        "n_eq_budget": args.n_eq_budget,
        "eye": args.eye,
        "scaled_eye": str(spec_path),
        "first_mesh_elems": n_mesh,
        "vla": asdict(vla),
        "vla_records": [_row(r) for r in vla_recs],
        "lp_records": [_row(r) for r in lp_recs],
        "note": "new-protocol trial; do not paste into old A2′ tables",
    }
    (out / "comparison.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {out / 'comparison.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
