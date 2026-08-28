"""Six-method benchmark on one problem instance (demo-scale campaign).

Methods and their global-solve accounting:

  uniform            ladder of ~7 solves (control: "just add DOF")
  dorfler_zz         SOLVE-ESTIMATE-MARK-REMESH loop until cap (many solves)
  local_prediction   probe + 2 predicted remeshes per budget (3 solves each)
  supervised         probe + 1 learned remesh (2 solves; offline training)
  rl_dqn             probe + k policy steps (k+1 solves; offline training)
  vla                probe + regional + certified (3 solves)

Usage:
  python3 scripts/run_benchmark.py --problem lbracket --out results/bench_lbracket \
      [--rl-episodes 40] [--sup-experts 6] [--skip-learned]
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
from visionamr.viz import plot_error_curves, plot_mesh, render_field_png
from visionamr.vla.partition import ScriptedVisionPartitioner
from visionamr.vla.pipeline import VLAConfig, run_vla


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="lbracket", choices=list(PROBLEM_FACTORIES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-eq-budget", type=int, default=8000)
    ap.add_argument("--rl-episodes", type=int, default=40)
    ap.add_argument("--sup-experts", type=int, default=6)
    ap.add_argument("--skip-learned", action="store_true")
    args = ap.parse_args()

    out = Path(args.out or f"results/bench_{args.problem}")
    out.mkdir(parents=True, exist_ok=True)
    problem = PROBLEM_FACTORIES[args.problem]()
    partitioner = ScriptedVisionPartitioner()
    budget_elems = int(args.n_eq_budget / 1.5)

    runner = FemRunner(problem, out)
    ref = runner.ensure_reference()
    print(f"[ref] N_eq={ref.n_equations} U={ref.U_total:.6g} qoi={ref.qoi:.6g}")

    # ---- classical methods -------------------------------------------------
    run_uniform_ladder(runner, n_steps=7)
    run_dorfler(runner, theta=0.5, max_rounds=12, n_eq_cap=args.n_eq_budget)
    run_local_prediction(
        runner, budgets=[budget_elems // 4, budget_elems // 2, budget_elems], rounds=2
    )

    # ---- VLA ---------------------------------------------------------------
    vla_res = run_vla(
        runner, partitioner, VLAConfig(n_eq_budget=args.n_eq_budget)
    )
    (out / "vla_result.json").write_text(
        json.dumps(asdict(vla_res), indent=1, default=str)
    )
    print(
        f"[vla] gate={vla_res.n_distinct_gate} s={vla_res.s:.3f} "
        f"kappa={vla_res.kappa:.3f} regions={vla_res.regions_final}"
    )

    # ---- learned methods ---------------------------------------------------
    if not args.skip_learned:
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

    print(f"\n{'method':>18} {'stage':>14} {'N_eq':>7} {'elems':>6} {'e_E':>8} {'e_qoi':>8} {'solve#':>6}")
    for r in runner.records:
        print(
            f"{r.method:>18} {r.stage:>14} {r.n_equations:>7} {r.n_elems:>6} "
            f"{r.e_energy:8.4f} {r.e_qoi:8.4f} {r.solve_index:>6}"
        )

    # figures: error vs N and error vs solve count within each method
    plot_error_curves(
        runner.records, out / "fig_error_vs_N.png",
        title=f"{problem.name}: energy error vs DOF",
    )

    # per-method cumulative-solve curves
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visionamr.viz import METHOD_STYLE

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    by_method: dict[str, list] = {}
    for r in runner.records:
        by_method.setdefault(r.method, []).append(r)
    for method, recs in by_method.items():
        style = METHOD_STYLE.get(method, dict(marker="."))
        # local prediction: one short run per budget; plot its best budget run
        if method == "local_prediction":
            best: dict[int, list] = {}
            for r in recs:
                best.setdefault(r.extra.get("budget", 0), []).append(r)
            recs = max(best.values(), key=lambda rr: rr[-1].n_equations)
        xs = list(range(1, len(recs) + 1))
        ys = [r.e_energy for r in recs]
        ax.semilogy(xs, ys, label=method, **style)
    ax.set_xlabel("global solves (cumulative)")
    ax.set_ylabel("relative energy error")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{problem.name}: error vs number of global solves")
    fig.tight_layout()
    fig.savefig(out / "fig_error_vs_solves.png", dpi=170)
    plt.close(fig)

    print(f"\nresults in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
