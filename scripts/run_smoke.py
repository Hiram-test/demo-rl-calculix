"""Smoke check for the core pipeline and acceptance gate G1.

Gate G1: on the singular L-bracket, the element-wise ZZ+Doerfler loop
must dominate the uniform ladder on the energy-error-vs-DOF axis.  The
old hand-rolled implementation failed this; a correct deployment must
not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionamr.baselines.dorfler import run_dorfler
from visionamr.baselines.local_prediction import run_local_prediction
from visionamr.baselines.uniform import run_uniform_ladder
from visionamr.experiment import FemRunner
from visionamr.geometry import make_lbracket


def main() -> int:
    workdir = Path("results/smoke_lbracket")
    problem = make_lbracket()
    runner = FemRunner(problem, workdir)
    ref = runner.ensure_reference()
    print(f"reference: N_eq={ref.n_equations} U={ref.U_total:.6g} qoi={ref.qoi:.6g}")

    run_uniform_ladder(runner, n_steps=7)
    run_dorfler(runner, theta=0.5, max_rounds=10, n_eq_cap=ref.n_equations // 3)
    run_local_prediction(runner, budgets=[1000, 4000], rounds=2)

    runner.dump()
    rows = {}
    for r in runner.records:
        rows.setdefault(r.method, []).append(r)

    print(f"\n{'method':>18} {'stage':>8} {'N_eq':>7} {'elems':>6} {'e_E':>8} {'e_qoi':>8}")
    for method, recs in rows.items():
        for r in recs:
            print(
                f"{method:>18} {r.stage:>8} {r.n_equations:>7} {r.n_elems:>6} "
                f"{r.e_energy:8.4f} {r.e_qoi:8.4f}"
            )

    # ---- gate G1: compare energy error at matched DOF (interpolate uniform)
    import numpy as np

    uni = sorted(rows["uniform"], key=lambda r: r.n_equations)
    dor = rows["dorfler_zz"]
    uni_N = np.array([r.n_equations for r in uni], dtype=float)
    uni_e = np.array([r.e_energy for r in uni], dtype=float)
    ok, checked = True, 0
    for r in dor:
        if r.n_equations < uni_N[0] or r.n_equations > uni_N[-1]:
            continue
        e_uni = np.exp(np.interp(np.log(r.n_equations), np.log(uni_N), np.log(uni_e)))
        checked += 1
        if r.e_energy > e_uni * 1.02:
            ok = False
            print(f"G1 FAIL at N={r.n_equations}: dorfler {r.e_energy:.4f} vs uniform {e_uni:.4f}")
    print(f"\nGate G1 (dorfler <= uniform at matched N, {checked} points): {'PASS' if ok and checked else 'FAIL'}")
    return 0 if ok and checked else 1


if __name__ == "__main__":
    raise SystemExit(main())
