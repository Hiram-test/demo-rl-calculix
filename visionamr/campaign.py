"""Campaign orchestration for docs/EXPERIMENT_PLAN.md (steps S0–S9).

Does not invent solve products: every tabulated point is a FemRunner
record backed by a CalculiX log.  Hypotheses H1–H4 are judged from
those records or left as 'insufficient evidence'.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .baselines.dorfler import run_dorfler
from .baselines.local_prediction import run_local_prediction
from .baselines.uniform import run_uniform_ladder
from .calculix import assemble_nodal_forces, default_ccx_cmd
from .experiment import FemRunner, SolveRecord
from .geometry import (
    OOD_SAMPLERS,
    PROBLEM_FACTORIES,
    SAMPLERS,
    analytic_load_resultant,
)
from .mesher import generate_uniform
from .vla.partition import (
    LLMVisionPartitioner,
    RandomSeedPartitioner,
    ScriptedVisionPartitioner,
)
from .vla.pipeline import VLAConfig, run_vla

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CAMPAIGN = RESULTS / "campaign"

EQ_PER_ELEM = {2: 1.5, 3: 0.62}

BUDGETS_EQ = {
    "bearing_block": (4000, 8000, 16000),
    "deck_panel": (10000, 20000, 40000),
    "lbracket": (4000, 8000, 16000),
    "plate_holes": (4000, 8000, 16000),
    "bearing_hole": (4000, 8000, 16000),
    "deck_opening": (10000, 20000, 40000),
}
PILOT_EQ = {
    "bearing_block": 8000,
    "deck_panel": 20000,
    "lbracket": 8000,
    "plate_holes": 8000,
    "bearing_hole": 8000,
    "deck_opening": 20000,
}
FAMILIES_3D = ("bearing_block", "deck_panel")
FAMILIES_2D = ("lbracket", "plate_holes")
TRAIN_SEEDS = list(range(1000, 1024))
TEST_SEEDS = list(range(9000, 9008))
PSO_SEED = 17


def elem_budget(n_eq: int, dim: int) -> int:
    return max(int(round(n_eq / EQ_PER_ELEM[dim])), 1)


def instance_key_problem(family: str, key: str):
    if key == "canonical":
        return PROBLEM_FACTORIES[family]()
    kind, _, raw = key.partition("_")
    seed = int(raw)
    rng = np.random.default_rng(seed)
    if kind == "ood":
        return OOD_SAMPLERS[family](rng)
    return SAMPLERS[family](rng)


def instance_dir(family: str, key: str) -> Path:
    return CAMPAIGN / family / key


def make_runner(problem, workdir: Path, *, timeout: float | None = None) -> FemRunner:
    if timeout is None:
        timeout = 1800.0 if problem.dim == 3 else 300.0
    workdir.mkdir(parents=True, exist_ok=True)
    return FemRunner(problem, workdir, ccx_timeout=timeout)


def _dump_slice(runner: FemRunner, method: str, path: Path) -> None:
    recs = [r for r in runner.records if r.method == method]
    path.write_text(
        json.dumps(
            {
                "problem": runner.problem.instance_id,
                "params": runner.problem.params,
                "method": method,
                "records": [asdict(r) for r in recs],
            },
            indent=1,
            default=str,
        )
    )


def _records_exist(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(data.get("records"))


# ---------------------------------------------------------------------------
# S0
# ---------------------------------------------------------------------------


def step_s0() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    import gmsh
    import numpy
    import scipy

    import visionamr

    ccx = default_ccx_cmd()
    try:
        proc = subprocess.run([ccx, "-v"], capture_output=True, text=True, timeout=20)
        ccx_ver = (proc.stdout + proc.stderr).strip().splitlines()[-1]
    except Exception as exc:  # noqa: BLE001
        ccx_ver = f"unreadable: {exc}"
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "git": git,
        "python": sys.version.split()[0],
        "gmsh": gmsh.__version__,
        "ccx_cmd": ccx,
        "ccx_version": ccx_ver,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "visionamr": visionamr.__version__,
        "pso_seed": PSO_SEED,
        "gmsh_threads": 1,
        "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "2")),
        "train_seeds": TRAIN_SEEDS,
        "test_seeds": TEST_SEEDS,
        "budgets_eq": {k: list(v) for k, v in BUDGETS_EQ.items()},
        "pilot_eq": PILOT_EQ,
        "eq_per_elem_prior": EQ_PER_ELEM,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = RESULTS / "locked.json"
    path.write_text(json.dumps(payload, indent=1))
    (CAMPAIGN).mkdir(parents=True, exist_ok=True)
    print(f"[S0] wrote {path}")
    return payload


# ---------------------------------------------------------------------------
# S1 references
# ---------------------------------------------------------------------------


def iter_instances(families, *, include_train: bool = False):
    for fam in families:
        yield fam, "canonical"
        for s in TEST_SEEDS:
            yield fam, f"test_{s}"
        if include_train:
            for s in TRAIN_SEEDS:
                yield fam, f"train_{s}"


def recompute_energy_errors(root: Path | None = None) -> int:
    """Rewrite e_energy from stored U_total after a reference is updated."""

    n = 0
    root = root or CAMPAIGN
    for ref_path in root.rglob("reference.json"):
        ref = json.loads(ref_path.read_text())
        U_ref = float(ref["U_total"])
        q_ref = float(ref["qoi"])
        for rec_path in ref_path.parent.glob("records*.json"):
            data = json.loads(rec_path.read_text())
            changed = False
            for r in data.get("records", []):
                U = r.get("U_total")
                if U is None:
                    continue
                gap = max(U_ref - float(U), 0.0)
                e = float((gap / U_ref) ** 0.5)
                extra = r.setdefault("extra", {})
                if float(U) > U_ref:
                    extra["above_reference"] = True
                elif extra.get("above_reference"):
                    extra.pop("above_reference", None)
                if r.get("e_energy") != e:
                    r["e_energy"] = e
                    changed = True
                if q_ref:
                    r["e_qoi"] = abs(float(r.get("qoi", 0.0)) - q_ref) / abs(q_ref)
                    changed = True
                n += 1
            if changed:
                rec_path.write_text(json.dumps(data, indent=1, default=str))
    return n


def step_s1(
    families=FAMILIES_3D, *, include_train: bool = False, instance_keys=None
) -> list[dict]:
    out = []
    pairs = (
        [(fam, k) for fam in families for k in instance_keys]
        if instance_keys is not None
        else list(iter_instances(families, include_train=include_train))
    )
    for fam, key in pairs:
        problem = instance_key_problem(fam, key)
        d = instance_dir(fam, key)
        runner = make_runner(problem, d)
        t0 = time.perf_counter()
        ref = runner.ensure_reference()
        row = {
            "family": fam,
            "key": key,
            "instance_id": problem.instance_id,
            "n_equations": ref.n_equations,
            "n_elems": ref.n_elems,
            "U_total": ref.U_total,
            "qoi": ref.qoi,
            "wall_s": time.perf_counter() - t0,
        }
        out.append(row)
        print(
            f"[S1] {fam}/{key} N={ref.n_equations} elems={ref.n_elems} "
            f"U={ref.U_total:.6g} ({row['wall_s']:.1f}s)"
        )
    (CAMPAIGN / "s1_references.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# S2 classical methods
# ---------------------------------------------------------------------------


def step_s2(families=FAMILIES_3D, instance_keys=None) -> None:
    if instance_keys is None:
        instance_keys = ["canonical"] + [f"test_{s}" for s in TEST_SEEDS]
    for fam in families:
        budgets = BUDGETS_EQ[fam]
        for key in instance_keys:
            _run_classical(fam, key, budgets)


def _run_classical(fam: str, key: str, budgets: tuple[int, ...]) -> None:
    problem = instance_key_problem(fam, key)
    d = instance_dir(fam, key)
    runner = make_runner(problem, d)
    runner.ensure_reference()
    dim = problem.dim
    max_eq = max(budgets)
    e_budgets = [elem_budget(b, dim) for b in budgets]

    uni_path = d / "records_uniform.json"
    if not _records_exist(uni_path):
        runner.reset_counter()
        n0 = len(runner.records)
        run_uniform_ladder(runner, n_steps=8, n_eq_cap=int(1.2 * max_eq))
        _dump_slice(runner, "uniform", uni_path)
        print(f"[S2] {fam}/{key} uniform {len(runner.records) - n0} solves")
    else:
        print(f"[S2] {fam}/{key} uniform cached")

    dor_path = d / "records_dorfler.json"
    if not _records_exist(dor_path):
        runner.reset_counter()
        n0 = len(runner.records)
        run_dorfler(runner, theta=0.5, max_rounds=12, n_eq_cap=max_eq)
        _dump_slice(runner, "dorfler_zz", dor_path)
        print(f"[S2] {fam}/{key} dorfler {len(runner.records) - n0} solves")
    else:
        print(f"[S2] {fam}/{key} dorfler cached")

    lp_path = d / "records_local_prediction.json"
    if not _records_exist(lp_path):
        runner.reset_counter()
        n0 = len(runner.records)
        run_local_prediction(runner, budgets=e_budgets, rounds=2)
        _dump_slice(runner, "local_prediction", lp_path)
        print(f"[S2] {fam}/{key} local_prediction {len(runner.records) - n0} solves")
    else:
        print(f"[S2] {fam}/{key} local_prediction cached")

    runner.dump(d / "records_classical.json")


def run_lp_rounds_diagnostic(families=FAMILIES_3D, rounds: int = 6) -> None:
    """Audit: rerun local prediction past its plan-locked 3-solve short run.

    The plan (§3.3) fixes probe+2 rounds per tier; §3.3.1 cites the known
    oscillation failure mode of longer element-wise prediction loops.  This
    diagnostic measures it on the canonical instances instead of asserting
    it.  Dumped to a separate file so the main LP series stays a short run.
    """

    for fam in families:
        b = PILOT_EQ[fam]
        d = instance_dir(fam, "canonical")
        out = d / "records_lp_rounds_diag.json"
        if _records_exist(out):
            print(f"[LPdiag] {fam} cached")
            continue
        problem = instance_key_problem(fam, "canonical")
        runner = make_runner(problem, d)
        runner.ensure_reference()
        runner.reset_counter()
        run_local_prediction(
            runner,
            budgets=[elem_budget(b, problem.dim)],
            rounds=rounds,
            method="local_prediction_r6",
        )
        _dump_slice(runner, "local_prediction_r6", out)
        print(f"[LPdiag] {fam} done")


def run_lp_naive_diagnostic(families=FAMILIES_3D, rounds: int = 6) -> None:
    """Weakness evidence (plan §3.3.1): the literature deployment oscillates.

    Runs the element-wise prediction loop with the *uncorrected* exponent
    (p=1) and the wide coarsening bound instead of the v2-locked
    (p=(d+2)/2, clip 1.8).  Dumped separately; never joins the S2 series.
    """

    for fam in families:
        b = PILOT_EQ[fam]
        d = instance_dir(fam, "canonical")
        out = d / "records_lp_naive_diag.json"
        if _records_exist(out):
            print(f"[LPnaive] {fam} cached")
            continue
        problem = instance_key_problem(fam, "canonical")
        runner = make_runner(problem, d)
        runner.ensure_reference()
        runner.reset_counter()
        run_local_prediction(
            runner,
            budgets=[elem_budget(b, problem.dim)],
            rounds=rounds,
            method="local_prediction_naive",
            p=1.0,
            ratio_bounds=(1.0 / 6.0, 3.0),
        )
        _dump_slice(runner, "local_prediction_naive", out)
        print(f"[LPnaive] {fam} done")


OOD_SEEDS = list(range(9500, 9504))


def run_ood_generalization(families=FAMILIES_3D, seeds=None) -> None:
    """Weakness evidence: supervised is family-sampler-bound.

    Deploys the trained supervised model, the LP short run, and the VLA
    scripted loop on instances drawn *outside* the training sampler's
    support.  Training-free methods see a new instance either way; the
    learned size field faces distribution shift.
    """

    from .baselines.supervised import SizeMLP, SupervisedConfig, deploy_supervised

    seeds = list(seeds) if seeds is not None else OOD_SEEDS
    for fam in families:
        b = PILOT_EQ[fam]
        model_path = CAMPAIGN / fam / "supervised" / "model.pt"
        model = None
        if model_path.exists():
            model = SizeMLP(SupervisedConfig())
            model.load(model_path)
        for seed in seeds:
            key = f"ood_{seed}"
            d = instance_dir(fam, key)
            problem = instance_key_problem(fam, key)

            sup_out = d / "records_supervised.json"
            if model is not None and not _records_exist(sup_out):
                runner = make_runner(problem, d)
                runner.ensure_reference()
                runner.reset_counter()
                deploy_supervised(
                    runner, model, n_elem_budget=elem_budget(b, problem.dim)
                )
                _dump_slice(runner, "supervised", sup_out)
                print(f"[OOD] {fam}/{key} supervised done")

            lp_out = d / "records_local_prediction.json"
            if not _records_exist(lp_out):
                runner = make_runner(problem, d)
                runner.ensure_reference()
                runner.reset_counter()
                run_local_prediction(
                    runner, budgets=[elem_budget(b, problem.dim)], rounds=2
                )
                _dump_slice(runner, "local_prediction", lp_out)
                print(f"[OOD] {fam}/{key} local_prediction done")

            _run_vla_one(fam, key, b, head="scripted")


# ---------------------------------------------------------------------------
# Failure-probe families (audit): drawing feature invisible to the probe
# ---------------------------------------------------------------------------

FP_FAMILIES = ("bearing_hole", "deck_opening")
FP_PARENT = {"bearing_hole": "bearing_block", "deck_opening": "deck_panel"}
FP_KEYS = ("canonical", "fp_9600", "fp_9601")


def _rim_distance(problem, pts: np.ndarray) -> np.ndarray:
    """Distance of points to the family's hole rim surface."""

    p = problem.params
    if problem.name == "bearing_hole":
        a, _ = p["patch"]
        px0 = p["W"] / 2.0 + p["offset"][0] - a / 2.0
        hx = px0 - p["hole_gap"] - p["hole_r"]
        hz = p["H"] / 2.0
        rad = np.sqrt((pts[:, 0] - hx) ** 2 + (pts[:, 2] - hz) ** 2)
        return np.abs(rad - p["hole_r"])
    if problem.name == "deck_opening":
        _, wb = p["wheel"]
        wx, wy = p["wheel_pos"]
        ocx, ocy = wx, wy - wb / 2.0 - p["open_gap"] - p["open_r"]
        rad = np.sqrt((pts[:, 0] - ocx) ** 2 + (pts[:, 1] - ocy) ** 2)
        return np.abs(rad - p["open_r"])
    raise ValueError(problem.name)


def _rim_mean_h(problem, mesh, band: float) -> float | None:
    d = _rim_distance(problem, mesh.centroids)
    m = d <= band
    if not m.any():
        return None
    return float(np.exp(np.mean(np.log(mesh.cell_sizes[m]))))


def run_failure_probe_families(families=FP_FAMILIES, keys=FP_KEYS) -> None:
    """Weakness evidence on purpose-built families.

    The hole/opening is on the drawing but barely present in the h0
    probe indicator (rim η² share of a few percent).  A one-shot map
    from that probe cannot invent rim grading; capacity on the parent
    measure does not create the missing information.  Covering the new
    topology in the expert bank would mean running Dörfler-to-cap on it
    first — the oracle is the baseline being replaced.  Parent models
    are frozen to isolate the deploy-time bottleneck, not to claim a
    network cannot learn holes.  VLA sees the rim through the drawing
    anchor, not a larger bank.
    """

    from .baselines.supervised import SizeMLP, SupervisedConfig, deploy_supervised
    from .experiment import initial_mesh
    from .indicators import zz_indicator

    for fam in families:
        b = PILOT_EQ[fam]
        parent = FP_PARENT[fam]
        model_path = CAMPAIGN / parent / "supervised" / "model.pt"
        model = None
        if model_path.exists():
            model = SizeMLP(SupervisedConfig())
            model.load(model_path)
        for key in keys:
            d = instance_dir(fam, key)
            diag_path = d / "fp_diag.json"
            if diag_path.exists():
                print(f"[FP] {fam}/{key} cached")
                continue
            problem = instance_key_problem(fam, key)
            band = 0.75 * float(problem.params.get("hole_r") or problem.params.get("open_r"))
            diag: dict = {"family": fam, "key": key, "parent_model": parent,
                          "rim_band": band, "n_eq_budget": b}

            runner = make_runner(problem, d, timeout=1800.0)
            ref = runner.ensure_reference()
            diag["reference_n_eq"] = ref.n_equations
            print(f"[FP] {fam}/{key} reference N={ref.n_equations}")

            # probe: what the coarse indicator actually sees at the rim
            probe_path = d / "records_fp_probe.json"
            runner.reset_counter()
            mesh0 = initial_mesh(problem)
            post0, rec0 = runner.solve_mesh(mesh0, method="fp_probe", stage="probe")
            eta2 = zz_indicator(problem, post0)
            rim_mask = _rim_distance(problem, mesh0.centroids) <= band
            rec0.extra["sum_eta2"] = float(eta2.sum())
            rec0.extra["rim_eta2_share"] = float(eta2[rim_mask].sum() / max(eta2.sum(), 1e-30))
            diag["probe_rim_eta2_share"] = rec0.extra["rim_eta2_share"]
            diag["rim_h_probe"] = _rim_mean_h(problem, mesh0, band)
            _dump_slice(runner, "fp_probe", probe_path)

            if model is not None:
                runner = make_runner(problem, d, timeout=1800.0)
                runner.ensure_reference()
                runner.reset_counter()
                deploy_supervised(runner, model, n_elem_budget=elem_budget(b, problem.dim))
                diag["rim_h_supervised"] = _rim_mean_h(problem, runner.last_mesh, band)
                _dump_slice(runner, "supervised", d / "records_supervised.json")

            runner = make_runner(problem, d, timeout=1800.0)
            runner.ensure_reference()
            runner.reset_counter()
            run_local_prediction(runner, budgets=[elem_budget(b, problem.dim)], rounds=2)
            diag["rim_h_lp"] = _rim_mean_h(problem, runner.last_mesh, band)
            _dump_slice(runner, "local_prediction", d / "records_local_prediction.json")

            runner = make_runner(problem, d, timeout=1800.0)
            runner.ensure_reference()
            runner.reset_counter()
            partitioner = ScriptedVisionPartitioner()
            cfg = VLAConfig(n_eq_budget=b, max_solves=4, pso=VLAConfig().pso)
            res = run_vla(runner, partitioner, cfg, method="vla")
            recs = [r for r in runner.records if r.method == "vla"]
            for r in recs:
                r.extra.setdefault("n_eq_budget", b)
                r.extra.setdefault("head", "scripted")
            (d / f"records_vla_scripted_b{b}.json").write_text(json.dumps({
                "vla_result": asdict(res),
                "problem": problem.instance_id,
                "params": problem.params,
                "method": "vla",
                "head": "scripted",
                "n_eq_budget": b,
                "vision": getattr(partitioner, "last_info", {}),
                "records": [asdict(r) for r in recs],
            }, indent=1, default=str))
            diag["rim_h_vla"] = _rim_mean_h(problem, runner.last_mesh, band)

            diag_path.write_text(json.dumps(diag, indent=1))
            print(f"[FP] {fam}/{key} done: {diag}")


# ---------------------------------------------------------------------------
# S3 gates
# ---------------------------------------------------------------------------


def gate_g1(workdir: Path | None = None) -> dict:
    """2-D L-bracket: Dörfler e_E <= uniform×1.02 at matched N."""

    from .geometry import make_lbracket

    workdir = workdir or (RESULTS / "gates" / "g1_lbracket")
    problem = make_lbracket()
    runner = make_runner(problem, workdir, timeout=300.0)
    runner.ensure_reference()
    run_uniform_ladder(runner, n_steps=7)
    run_dorfler(runner, theta=0.5, max_rounds=10, n_eq_cap=runner.reference.n_equations // 3)
    runner.dump(workdir / "records.json")
    rows: dict[str, list[SolveRecord]] = {}
    for r in runner.records:
        rows.setdefault(r.method, []).append(r)
    uni = sorted(rows["uniform"], key=lambda r: r.n_equations)
    uni_N = np.array([r.n_equations for r in uni], dtype=float)
    uni_e = np.array([r.e_energy for r in uni], dtype=float)
    ok, checked, fails = True, 0, []
    for r in rows["dorfler_zz"]:
        if r.n_equations < uni_N[0] or r.n_equations > uni_N[-1]:
            continue
        e_uni = float(np.exp(np.interp(np.log(r.n_equations), np.log(uni_N), np.log(uni_e))))
        checked += 1
        if r.e_energy > e_uni * 1.02:
            ok = False
            fails.append({"N": r.n_equations, "dorfler": r.e_energy, "uniform": e_uni})
    return {
        "gate": "G1",
        "pass": bool(ok and checked),
        "checked": checked,
        "fails": fails,
    }


def gate_g3(scan_root: Path | None = None) -> dict:
    """above_reference must be zero across dumped records."""

    scan_root = scan_root or CAMPAIGN
    n_rec, n_bad = 0, 0
    bad = []
    for path in scan_root.rglob("records*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for r in data.get("records", []):
            n_rec += 1
            if r.get("extra", {}).get("above_reference"):
                n_bad += 1
                bad.append(str(path))
    return {
        "gate": "G3",
        "pass": n_bad == 0 and n_rec > 0,
        "n_records": n_rec,
        "n_above_reference": n_bad,
        "paths": bad[:20],
    }


def gate_g7() -> dict:
    """Imprinted load resultant equals p×A on two distinct meshes per 3-D family."""

    rows = []
    ok = True
    for fam in FAMILIES_3D:
        problem = PROBLEM_FACTORIES[fam]()
        target = analytic_load_resultant(problem)
        for tag, h in (("coarse", problem.h0), ("fine", problem.h0 / 1.6)):
            mesh = generate_uniform(problem, h)
            F = assemble_nodal_forces(mesh, problem).sum(axis=0)
            rel = float(np.linalg.norm(F - target) / max(np.linalg.norm(target), 1e-30))
            passed = rel < 1e-6
            ok = ok and passed
            rows.append(
                {
                    "family": fam,
                    "mesh": tag,
                    "n_elems": mesh.n_cells,
                    "F": F.tolist(),
                    "target": target.tolist(),
                    "rel_err": rel,
                    "pass": passed,
                }
            )
    return {"gate": "G7", "pass": ok, "rows": rows}


def gate_g2() -> dict:
    """Gmsh is the only mesh source: no handwritten connectivity outside mesher."""

    import ast

    root = Path(__file__).resolve().parent
    hits = []
    for path in root.rglob("*.py"):
        if path.name == "mesher.py":
            continue
        text = path.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"add_cells", "insert_cells"}:
                    hits.append(f"{path.name}:{node.lineno}")
    return {
        "gate": "G2",
        "pass": len(hits) == 0,
        "hits": hits,
        "note": "structural: only mesher.py calls gmsh",
    }


def gate_g4() -> dict:
    """VLA solves in one dump must have distinct N; collisions are failures."""

    n_ok, n_fail, fails = 0, 0, []
    for path in CAMPAIGN.rglob("records_vla_*.json"):
        recs = json.loads(path.read_text()).get("records", [])
        vla = [r for r in recs if str(r.get("method", "")).startswith("vla")]
        ns = [r.get("n_equations") for r in vla]
        if len(ns) >= 2 and len(set(ns)) < len(ns):
            n_fail += 1
            fails.append(str(path.relative_to(CAMPAIGN)))
        else:
            n_ok += 1
    return {
        "gate": "G4",
        "pass": n_fail == 0 and n_ok > 0,
        "checked": n_ok + n_fail,
        "fails": fails[:20],
    }


def gate_g5() -> dict:
    """Every tabulated point has CalculiX-backed U and equation count."""

    n, bad = 0, 0
    sample = []
    for path in CAMPAIGN.rglob("records*.json"):
        try:
            recs = json.loads(path.read_text()).get("records", [])
        except json.JSONDecodeError:
            continue
        for r in recs:
            n += 1
            if r.get("U_total") is None or r.get("n_equations") is None:
                bad += 1
                if len(sample) < 10:
                    sample.append(str(path.relative_to(CAMPAIGN)))
    return {"gate": "G5", "pass": bad == 0 and n > 0, "n_records": n, "n_missing": bad, "paths": sample}


def step_s3() -> dict:
    g1 = gate_g1()
    g7 = gate_g7()
    g3 = gate_g3()
    g2 = gate_g2()
    g4 = gate_g4()
    g5 = gate_g5()
    payload = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G7": g7}
    path = RESULTS / "gates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    for name, g in payload.items():
        print(f"[S3] {name}: {'PASS' if g['pass'] else 'FAIL'} {g}")
    return payload


# ---------------------------------------------------------------------------
# S4 VLA
# ---------------------------------------------------------------------------


def _vla_path(d: Path, head: str, n_eq: int) -> Path:
    return d / f"records_vla_{head}_b{n_eq}.json"


def _run_vla_one(
    fam: str,
    key: str,
    n_eq: int,
    *,
    head: str,
    cfg_kwargs: dict | None = None,
    method: str = "vla",
) -> dict:
    problem = instance_key_problem(fam, key)
    d = instance_dir(fam, key)
    out = _vla_path(d, head, n_eq) if method == "vla" else d / f"records_{method}.json"
    if _records_exist(out):
        print(f"[S4] {fam}/{key} {head}/{method} b{n_eq} cached")
        return json.loads(out.read_text()).get("vla_result", {})

    runner = make_runner(problem, d)
    runner.ensure_reference()
    runner.reset_counter()
    n0 = len(runner.records)

    if head == "llm":
        cache = d / "llm_seeds.json"
        partitioner = LLMVisionPartitioner(
            cache_path=str(cache) if cache.exists() else None,
            dump_dir=str(d / "llm_dump"),
        )
    elif head == "random":
        # cardinality matched to a scripted proposal on the same probe later;
        # the caller should pass n_seeds via cfg_kwargs['n_random']
        kw = dict(cfg_kwargs or {})
        n_random = kw.pop("n_random", 10)
        rng_seed = kw.pop("rng_seed", 0)
        cfg_kwargs = kw
        partitioner = RandomSeedPartitioner(n_seeds=n_random, rng_seed=rng_seed)
    elif head == "scripted_no_anchor":
        partitioner = ScriptedVisionPartitioner(anchor_kinds=())
    else:
        partitioner = ScriptedVisionPartitioner()

    cfg = VLAConfig(n_eq_budget=n_eq, max_solves=4, pso=VLAConfig().pso)
    # PSO seed is already 17 in PSOConfig
    if cfg_kwargs:
        for k, v in cfg_kwargs.items():
            if k == "pso_budget_safety":
                from dataclasses import replace

                cfg = replace(cfg, pso=replace(cfg.pso, budget_safety=v))
            elif hasattr(cfg, k):
                setattr(cfg, k, v)

    res = run_vla(runner, partitioner, cfg, method=method)
    recs = [r for r in runner.records[n0:] if r.method == method]
    for r in recs:
        r.extra.setdefault("n_eq_budget", n_eq)
        r.extra.setdefault("head", head)
    info = {
        "vla_result": asdict(res),
        "problem": problem.instance_id,
        "params": problem.params,
        "method": method,
        "head": head,
        "n_eq_budget": n_eq,
        "vision": getattr(partitioner, "last_info", {}),
        "records": [asdict(r) for r in recs],
    }
    out.write_text(json.dumps(info, indent=1, default=str))
    print(
        f"[S4] {fam}/{key} {head}/{method} b{n_eq} solves={res.solves} "
        f"gate={res.n_distinct_gate} early={res.stopped_early} "
        f"vision={info['vision']}"
    )
    return asdict(res)


def step_s4(
    families=FAMILIES_3D,
    instance_keys=None,
    *,
    scripted_all_budgets: bool = True,
    llm: bool = True,
) -> None:
    if instance_keys is None:
        instance_keys = ["canonical"] + [f"test_{s}" for s in TEST_SEEDS]
    for fam in families:
        budgets = BUDGETS_EQ[fam] if scripted_all_budgets else (PILOT_EQ[fam],)
        for key in instance_keys:
            for b in budgets:
                _run_vla_one(fam, key, b, head="scripted")
            if llm:
                _run_vla_one(fam, key, PILOT_EQ[fam], head="llm")


# ---------------------------------------------------------------------------
# S5 / S6 learned
# ---------------------------------------------------------------------------


def step_s5(families=FAMILIES_2D, *, n_experts: int | None = None, n_eq: int | None = None) -> None:
    from .baselines.supervised import deploy_supervised, generate_expert_dataset, train_supervised

    n_experts = n_experts if n_experts is not None else len(TRAIN_SEEDS)
    for fam in families:
        budget = n_eq or PILOT_EQ[fam]
        d = CAMPAIGN / fam / "supervised"
        d.mkdir(parents=True, exist_ok=True)
        model_path = d / "model.pt"
        ds_path = d / "expert_dataset.npz"
        train_problems = [
            SAMPLERS[fam](np.random.default_rng(s)) for s in TRAIN_SEEDS[:n_experts]
        ]
        if not ds_path.exists():
            generate_expert_dataset(
                train_problems, d / "experts", n_eq_cap=budget, max_rounds=10
            )
            # generate_expert_dataset writes expert_dataset.npz inside experts/
            src = d / "experts" / "expert_dataset.npz"
            if src.exists():
                ds_path.write_bytes(src.read_bytes())
        if not ds_path.exists():
            print(f"[S5] {fam} no expert dataset")
            continue
        model = train_supervised(ds_path)
        model.save(model_path)
        for key in ["canonical"] + [f"test_{s}" for s in TEST_SEEDS]:
            inst = instance_dir(fam, key)
            out = inst / "records_supervised.json"
            if _records_exist(out):
                print(f"[S5] {fam}/{key} supervised cached")
                continue
            problem = instance_key_problem(fam, key)
            runner = make_runner(problem, inst)
            runner.ensure_reference()
            runner.reset_counter()
            deploy_supervised(
                runner, model, n_elem_budget=elem_budget(budget, problem.dim)
            )
            _dump_slice(runner, "supervised", out)
            print(f"[S5] {fam}/{key} supervised done")


def step_s6(
    families=FAMILIES_2D,
    *,
    episodes: int | None = None,
    n_seeds: int = 3,
    seeds: tuple[int, ...] | None = None,
) -> None:
    from .baselines.rl_dqn import DQNConfig, evaluate_dqn, train_dqn

    partitioner = ScriptedVisionPartitioner()
    seed_list = list(seeds) if seeds is not None else list(range(n_seeds))
    for fam in families:
        budget = PILOT_EQ[fam]
        ep = episodes if episodes is not None else (300 if fam in FAMILIES_2D else 120)
        for seed in seed_list:
            d = CAMPAIGN / fam / f"rl_seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            policy_path = d / "policy.pt"
            cfg = DQNConfig(n_eq_budget=budget, max_steps=6, seed=seed)
            sampler = SAMPLERS[fam]
            if not policy_path.exists():
                policy, _ = train_dqn(
                    lambda ep_i, s=sampler: s(np.random.default_rng(2000 + 1000 * seed + ep_i)),
                    partitioner,
                    d / "training",
                    episodes=ep,
                    cfg=cfg,
                )
                policy.save(policy_path)
            else:
                from .baselines.rl_dqn import DQNPolicy

                policy = DQNPolicy(cfg)
                policy.load(policy_path)
            for key in ["canonical"] + [f"test_{s}" for s in TEST_SEEDS]:
                inst = instance_dir(fam, key)
                out = inst / f"records_rl_dqn_s{seed}.json"
                if _records_exist(out):
                    print(f"[S6] {fam}/{key} rl seed={seed} cached")
                    continue
                problem = instance_key_problem(fam, key)
                runner = make_runner(problem, inst)
                runner.ensure_reference()
                runner.reset_counter()
                evaluate_dqn(runner, policy, partitioner, cfg=cfg, method=f"rl_dqn_s{seed}")
                _dump_slice(runner, f"rl_dqn_s{seed}", out)
                print(f"[S6] {fam}/{key} rl seed={seed} done")


# ---------------------------------------------------------------------------
# S7 ablations (canonical × 2 families × pilot budget)
# ---------------------------------------------------------------------------


def step_s7(families=FAMILIES_3D) -> None:
    for fam in families:
        b = PILOT_EQ[fam]
        key = "canonical"
        # AB1 random (cardinality from a cheap scripted propose is handled
        # inside with a default n_seeds=10; a later pass can match exactly)
        _run_vla_one(
            fam, key, b, head="random", method="vla_ab1_random",
            cfg_kwargs={"n_random": 10, "rng_seed": 1},
        )
        _run_vla_one(
            fam, key, b, head="scripted", method="vla_ab2_box",
            cfg_kwargs={"partition_mode": "linf_box"},
        )
        _run_vla_one(
            fam, key, b, head="scripted_no_anchor", method="vla_ab3_no_anchor",
        )
        _run_vla_one(
            fam, key, b, head="scripted", method="vla_ab4_nosplit",
            cfg_kwargs={"allow_split": False},
        )
        _run_vla_one(
            fam, key, b, head="scripted", method="vla_ab5_nocomm",
            cfg_kwargs={"allow_communication": False},
        )
        _run_vla_one(
            fam, key, b, head="scripted", method="vla_ab6_nopso",
            cfg_kwargs={"allow_pso": False},
        )
        for ksol in (3, 4, 5, 6):
            _run_vla_one(
                fam, key, b, head="scripted", method=f"vla_ab7_k{ksol}",
                cfg_kwargs={"max_solves": ksol},
            )
        _run_vla_one(
            fam, key, b, head="scripted", method="vla_ab8_s_only",
            cfg_kwargs={"pso_mode": "s_only"},
        )
        _run_vla_one(
            fam, key, b, head="scripted", method="vla_ab8_nelder",
            cfg_kwargs={"pso_mode": "nelder"},
        )
        # §5 AB9–AB11 (S7 card lists AB1–AB8; these three are in the plan table)
        _run_vla_one(
            fam, key, b, head="scripted", method="vla_ab9_fixed_q",
            cfg_kwargs={"use_measured_exponents": False},
        )
        _run_vla_one(
            fam, key, b, head="scripted", method="vla_ab10_nodrift",
            cfg_kwargs={"use_resource_drift": False},
        )
        _run_vla_one(
            fam, key, b, head="scripted", method="vla_ab10_safety092",
            cfg_kwargs={"final_budget_safety": 0.92},
        )
        _run_vla_one(
            fam, key, b, head="scripted", method="vla_ab10_safety097",
            cfg_kwargs={"final_budget_safety": 0.97},
        )
        _run_vla_one(
            fam, key, b, head="scripted", method="vla_ab11_no_inplace",
            cfg_kwargs={"inplace_min_use": 9.0},
        )


# ---------------------------------------------------------------------------
# S8 / S9  (see report.py)
# ---------------------------------------------------------------------------


def load_method_records(fam: str, key: str, glob: str) -> list[dict]:
    d = instance_dir(fam, key)
    recs = []
    for path in sorted(d.glob(glob)):
        data = json.loads(path.read_text())
        recs.extend(data.get("records", []))
    return recs


def vla_deliverable(records: list[dict], k: int, n_eq_budget: int) -> dict | None:
    seen = [r for r in records if r.get("method", "").startswith("vla")][:k]
    cands = [
        r
        for r in seen[1:]
        if r.get("n_equations", 10**18) <= n_eq_budget and "sum_eta2" in r.get("extra", {})
    ]
    if not cands:
        return seen[-1] if seen else None
    return min(cands, key=lambda r: r["extra"]["sum_eta2"])


def _method_series(records: list[dict], method: str) -> list[dict]:
    return [r for r in records if r.get("method") == method]


def write_whitelist_mesh_figures(figdir: Path) -> list[str]:
    """Plan §8: partition tri-views and certified-mesh tri-views.

    Seed locations come from a fresh scripted probe (same head as S4).
    Certified sizes are taken from the stored campaign dump by seed name
    so the figure matches the tabulated deliverable, not a new VLA loop.
    """

    from .indicators import zz_indicator
    from .mesher import generate_mesh, generate_uniform
    from .vla.regions import Partition, Seed
    from .viz import plot_mesh, plot_partition

    written: list[str] = []
    for fam in FAMILIES_3D:
        problem = PROBLEM_FACTORIES[fam]()
        budget = PILOT_EQ[fam]
        dump = instance_dir(fam, "canonical") / f"records_vla_scripted_b{budget}.json"
        if not dump.exists():
            continue
        payload = json.loads(dump.read_text())
        pick = None
        for rec in payload.get("records", []):
            if rec.get("extra", {}).get("certified_pick"):
                pick = rec
        regions = ((pick or (payload.get("records") or [{}])[-1]).get("extra") or {}).get("regions") or {}
        workdir = figdir / "_meshes" / fam
        runner = make_runner(problem, workdir, timeout=180.0)
        mesh0 = generate_uniform(problem, problem.h0)
        post, _ = runner.solve_mesh(mesh0, method="vla_evidence", stage="probe")
        eta2 = zz_indicator(problem, post)
        head = ScriptedVisionPartitioner()
        seeds = head.propose(problem, post, eta2)
        sized = [
            Seed(s.name, s.xyz, float(regions.get(s.name, s.h)), s.origin) for s in seeds
        ]
        part = Partition(sized, problem, drawings=list(getattr(head, "last_drawings", []) or []))
        labels = part.assign(post.mesh)
        part_path = figdir / f"{fam}_partition.png"
        plot_partition(
            problem, post, labels, part.seeds, part_path,
            title=f"{fam}: drawn irregular partition (scripted)",
        )
        written.append(part_path.name)
        cert = generate_mesh(problem, part.size_field(post.mesh, labels))
        mesh_path = figdir / f"{fam}_certified_mesh.png"
        plot_mesh(cert, mesh_path, title=f"{fam}: certified size-field mesh")
        written.append(mesh_path.name)
    return written


def step_s8() -> dict:
    from . import report
    from .viz import (
        plot_ablation_bars,
        plot_budget_scatter,
        plot_error_curves,
        plot_error_vs_solves,
        plot_test_boxplots,
        plot_training_cost_bars,
    )

    tables = report.build_all_tables(CAMPAIGN)
    figdir = RESULTS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    report.write_tables(tables, RESULTS / "tables")
    # figures from canonical classical + vla records
    for fam in FAMILIES_3D:
        d = instance_dir(fam, "canonical")
        recs = []
        for glob in (
            "records_uniform.json",
            "records_dorfler.json",
            "records_local_prediction.json",
            "records_vla_scripted_b*.json",
            "records_vla_llm_b*.json",
            "records_supervised.json",
            "records_rl_dqn_s0.json",
            "records_rl_dqn_s1.json",
            "records_rl_dqn_s2.json",
        ):
            loaded = load_method_records(fam, "canonical", glob.replace("b*", f"b{PILOT_EQ[fam]}"))
            if "llm" in glob:
                # both heads dump method="vla"; keep them separate series so
                # the solves axis never stitches two runs into one curve
                loaded = [{**r, "method": "vla_llm"} for r in loaded]
            recs.extend(loaded)
        # wrap as simple namespace for viz
        class R:
            def __init__(self, d):
                self.__dict__.update(d)
                if not hasattr(self, "extra"):
                    self.extra = {}

        wrapped = [R(r) for r in recs]
        if wrapped:
            plot_error_curves(
                wrapped, figdir / f"{fam}_error_vs_N.png",
                title=f"{fam}: energy error vs DOF",
            )
            plot_error_vs_solves(
                wrapped, figdir / f"{fam}_error_vs_solves.png",
                title=f"{fam}: error vs number of global solves",
                n_eq_budget=PILOT_EQ[fam],
            )
    # budget scatter + ablations if present
    scatter_rows = tables.get("budget_rows", [])
    if scatter_rows:
        plot_budget_scatter(scatter_rows, figdir / "budget_compliance.png",
                            title="budget compliance")
    for fam in FAMILIES_3D:
        ab = tables.get("ablation", {}).get(fam, [])
        if ab:
            plot_ablation_bars(ab, figdir / f"{fam}_ablation.png",
                               title=f"{fam} ablations (canonical, pilot budget)")
    cost_rows = [r for r in tables.get("training_cost") or [] if r.get("family") in FAMILIES_3D]
    if cost_rows:
        plot_training_cost_bars(
            cost_rows, figdir / "training_cost.png",
            title="A4 training cost (3D; offline only)",
        )
    test_raw = (tables.get("error_at_k") or {}).get("test_raw") or {}
    test_raw = {fam: test_raw[fam] for fam in FAMILIES_3D if fam in test_raw}
    if test_raw:
        plot_test_boxplots(
            test_raw, figdir / "test_boxplots.png",
            title="test-set energy error at k=2/3/4",
        )
    try:
        mesh_figs = write_whitelist_mesh_figures(figdir)
        print(f"[S8] whitelist mesh figures: {mesh_figs}")
    except Exception as exc:
        print(f"[S8] whitelist mesh figures skipped: {exc}")
    (RESULTS / "s8_tables.json").write_text(json.dumps(tables, indent=1, default=str))
    print(f"[S8] tables and figures in {RESULTS}")
    return tables


def step_s9(tables: dict | None = None) -> Path:
    from . import report

    if tables is None:
        tables = json.loads((RESULTS / "s8_tables.json").read_text()) if (RESULTS / "s8_tables.json").exists() else report.build_all_tables(CAMPAIGN)
    path = ROOT / "docs" / "RESULTS.md"
    path.write_text(report.render_results_md(tables))
    print(f"[S9] wrote {path}")
    return path


def run_steps(steps: list[str], **kwargs) -> None:
    dispatch = {
        "S0": step_s0,
        "S1": lambda: step_s1(
            kwargs.get("families") or FAMILIES_3D,
            include_train=kwargs.get("include_train", False),
            instance_keys=kwargs.get("instance_keys"),
        ),
        "S2": lambda: step_s2(
            kwargs.get("families") or FAMILIES_3D,
            kwargs.get("instance_keys"),
        ),
        "S3": step_s3,
        "S4": lambda: step_s4(
            kwargs.get("families") or FAMILIES_3D,
            kwargs.get("instance_keys"),
            scripted_all_budgets=not kwargs.get("pilot_only", False),
            llm=not kwargs.get("no_llm", False),
        ),
        "S5": lambda: step_s5(
            kwargs.get("learn_families") or FAMILIES_2D,
            n_experts=kwargs.get("n_experts"),
        ),
        "S6": lambda: step_s6(
            kwargs.get("learn_families") or FAMILIES_2D,
            episodes=kwargs.get("rl_episodes"),
            n_seeds=kwargs.get("n_seeds", 3),
            seeds=kwargs.get("rl_seeds"),
        ),
        "S7": lambda: step_s7(kwargs.get("families") or FAMILIES_3D),
        "S8": step_s8,
        "S9": step_s9,
    }
    for s in steps:
        s = s.upper()
        if s not in dispatch:
            raise SystemExit(f"unknown step {s!r}; valid: {sorted(dispatch)}")
        print(f"==== {s} ====")
        dispatch[s]()
