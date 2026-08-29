"""Multi-method benchmark on one problem instance.

Methods and their global-solve accounting:

  uniform            geometric ladder until the DOF cap ("just add DOF")
  dorfler_zz         SOLVE-ESTIMATE-MARK-REMESH loop until cap (many solves)
  local_prediction   probe + 2 predicted remeshes per budget (3 solves each)
  vla                adaptive short loop, at most --vla-max-solves solves
  supervised         probe + 1 learned remesh (2 solves; offline training)
  rl_dqn             probe + k policy steps (k+1 solves; offline training)

Learned methods are optional (--with-learned); the 3-D pilot runs the
classical methods plus VLA.

Usage:
  python3 scripts/run_benchmark.py --problem bearing_block --n-eq-budget 8000
  python3 scripts/run_benchmark.py --problem deck_panel --n-eq-budget 20000
  python3 scripts/run_benchmark.py --problem lbracket --with-learned
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionamr.baselines.dorfler import run_dorfler
from visionamr.baselines.local_prediction import run_local_prediction
from visionamr.baselines.uniform import run_uniform_ladder
from visionamr.experiment import FemRunner
from visionamr.geometry import PROBLEM_FACTORIES, SAMPLERS
from visionamr.viz import plot_error_curves, plot_error_vs_solves
from visionamr.vla.partition import ScriptedVisionPartitioner
from visionamr.vla.pipeline import VLAConfig, run_vla

EQ_PER_ELEM = {2: 1.5, 3: 0.62}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="bearing_block", choices=list(PROBLEM_FACTORIES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-eq-budget", type=int, default=8000)
    ap.add_argument("--vla-max-solves", type=int, default=2)
    ap.add_argument("--with-learned", action="store_true")
    ap.add_argument("--rl-episodes", type=int, default=40)
    ap.add_argument("--sup-experts", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out or f"results/bench_{args.problem}")
    out.mkdir(parents=True, exist_ok=True)
    problem = PROBLEM_FACTORIES[args.problem]()
    partitioner = ScriptedVisionPartitioner()
    budget_elems = int(args.n_eq_budget / EQ_PER_ELEM[problem.dim])

    runner = FemRunner(problem, out)
    ref = runner.ensure_reference()
    print(f"[ref] N_eq={ref.n_equations} U={ref.U_total:.6g} qoi={ref.qoi:.6g}")

    # ---- classical methods -------------------------------------------------
    run_uniform_ladder(runner, n_steps=8, n_eq_cap=int(1.2 * args.n_eq_budget))
    run_dorfler(runner, theta=0.5, max_rounds=12, n_eq_cap=args.n_eq_budget)
    run_local_prediction(
        runner, budgets=[budget_elems // 4, budget_elems // 2, budget_elems], rounds=2
    )

    # ---- VLA ---------------------------------------------------------------
    vla_res = run_vla(
        runner,
        partitioner,
        VLAConfig(n_eq_budget=args.n_eq_budget, max_solves=args.vla_max_solves),
    )
    (out / "vla_result.json").write_text(
        json.dumps(asdict(vla_res), indent=1, default=str)
    )
    print(
        f"[vla] gate={vla_res.n_distinct_gate} solves={vla_res.solves} "
        f"early={vla_res.stopped_early} s={vla_res.s_last:.3f} "
        f"kappa={vla_res.kappa_last:.3f}\n[vla] seeds={vla_res.seeds_final}"
    )

    # ---- learned methods (optional) ---------------------------------------
    if args.with_learned:
        sampler = SAMPLERS[args.problem]

        from visionamr.baselines.supervised import (
            deploy_supervised,
            generate_expert_dataset,
            train_supervised,
        )

        train_problems = [
            sampler(np.random.default_rng(1000 + i)) for i in range(args.sup_experts)
        ]
        ds = generate_expert_dataset(
            train_problems, out / "supervised_training",
            n_eq_cap=args.n_eq_budget, max_rounds=10,
        )
        model = train_supervised(ds)
        deploy_supervised(runner, model, n_elem_budget=budget_elems)

        from visionamr.baselines.rl_dqn import DQNConfig, evaluate_dqn, train_dqn

        cfg = DQNConfig(n_eq_budget=args.n_eq_budget, max_steps=6)
        policy, _ = train_dqn(
            lambda ep: sampler(np.random.default_rng(2000 + ep)),
            partitioner,
            out / "rl_training",
            episodes=args.rl_episodes,
            cfg=cfg,
        )
        policy.save(out / "rl_training" / "policy.pt")
        evaluate_dqn(runner, policy, partitioner, cfg=cfg)

    # ---- reporting ---------------------------------------------------------
    runner.dump(out / "records.json")

    print(f"\n{'method':>18} {'stage':>16} {'N_eq':>7} {'elems':>7} {'e_E':>8} {'e_qoi':>8}")
    for r in runner.records:
        print(
            f"{r.method:>18} {r.stage:>16} {r.n_equations:>7} {r.n_elems:>7} "
            f"{r.e_energy:8.4f} {r.e_qoi:8.4f}"
        )

    # headline: error at matched few solve counts (VLA target: beat Doerfler).
    # The VLA column reports its deliverable after k solves: the solved,
    # budget-compliant mesh its own indicator ranks best (return-best-iterate
    # semantics; no oracle involved in the pick).
    by_method: dict[str, list] = {}
    for r in runner.records:
        by_method.setdefault(r.method, []).append(r)
    print("\nerror after k global solves (headline axis):")
    print(f"{'k':>3} {'dorfler':>10} {'vla':>10} {'local_pred(best)':>18}")
    dor = by_method.get("dorfler_zz", [])
    vla = by_method.get("vla", [])
    lp: dict[int, list] = {}
    for r in by_method.get("local_prediction", []):
        lp.setdefault(r.extra.get("budget", 0), []).append(r)
    lp_best = max(lp.values(), key=lambda rr: rr[-1].n_equations) if lp else []

    def vla_deliverable(k: int):
        seen = vla[:k]
        cands = [
            r for r in seen[1:]
            if r.n_equations <= args.n_eq_budget and "sum_eta2" in r.extra
        ]
        if not cands:
            return seen[-1] if seen else None
        return min(cands, key=lambda r: r.extra["sum_eta2"])

    for k in range(1, max(len(dor), len(vla)) + 1):
        d = f"{dor[k-1].e_energy:.4f}" if k <= len(dor) else "-"
        vr = vla_deliverable(k) if k <= len(vla) else None
        v = f"{vr.e_energy:.4f}" if vr is not None else "-"
        l = f"{lp_best[k-1].e_energy:.4f}" if k <= len(lp_best) else "-"
        print(f"{k:>3} {d:>10} {v:>10} {l:>18}")

    plot_error_curves(
        runner.records, out / "fig_error_vs_N.png",
        title=f"{problem.name}: energy error vs DOF",
    )
    plot_error_vs_solves(
        runner.records, out / "fig_error_vs_solves.png",
        title=f"{problem.name}: error vs number of global solves",
    )
    print(f"\nresults in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
