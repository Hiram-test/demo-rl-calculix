from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .fem import nearest_node
from .meshes import nodes_on_coordinate
from .scenarios import ScenarioRun


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _find_ccx() -> str:
    direct = shutil.which("ccx")
    if direct:
        return direct
    for name in ("ccx_2.21", "ccx_2.20", "ccx_2.19"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("CalculiX executable not found")


def _baseline_key(run: ScenarioRun) -> str:
    preferred = {
        "bearing_load_introduction": "point_L3",
        "web_circular_opening": "L3",
        "cracked_tension_panel": "L3",
        "diaphragm_rectangular_opening": "sharp_L3",
    }[run.scenario_id]
    if preferred not in run.meshes:
        raise KeyError(f"missing baseline mesh {preferred}")
    return preferred


def _constraints(run: ScenarioRun, key: str) -> dict[int, float]:
    mesh = run.meshes[key]
    if run.scenario_id == "bearing_load_introduction":
        left = nodes_on_coordinate(mesh, x=float(mesh.nodes[:, 0].min()))
        out = {2 * int(n): 0.0 for n in left}
        out.update({2 * int(n) + 1: 0.0 for n in left})
        return out
    xmin, ymin = mesh.nodes.min(axis=0)
    xmax = mesh.nodes[:, 0].max()
    n1 = nearest_node(mesh, (float(xmin), float(ymin)))
    n2 = nearest_node(mesh, (float(xmax), float(ymin)))
    return {2 * n1: 0.0, 2 * n1 + 1: 0.0, 2 * n2 + 1: 0.0}


def _check_node_component(run: ScenarioRun, key: str) -> tuple[int, int]:
    mesh = run.meshes[key]
    xmax = mesh.nodes[:, 0].max()
    ymax = mesh.nodes[:, 1].max()
    if run.scenario_id == "bearing_load_introduction":
        return nearest_node(mesh, (float(xmax), 0.0)), 2
    if run.scenario_id in {"web_circular_opening", "diaphragm_rectangular_opening"}:
        return nearest_node(mesh, (float(xmax), 0.0)), 1
    return nearest_node(mesh, (0.0, float(ymax))), 2


def _write_inp(run: ScenarioRun, key: str, path: Path, check_node: int) -> None:
    mesh = run.meshes[key]
    sol = run.solutions[key]
    young = float(run.metadata["young"])
    poisson = float(run.metadata["poisson"])
    thickness = float(run.metadata["thickness"])
    constraints = _constraints(run, key)

    lines = ["*HEADING", f"CalculiX cross-check: {run.scenario_id}", "*NODE,NSET=NALL"]
    for i, (x, y) in enumerate(mesh.nodes, start=1):
        lines.append(f"{i},{x:.12g},{y:.12g},0.0")
    lines.append("*ELEMENT,TYPE=CPS4,ELSET=EALL")
    for i, conn in enumerate(mesh.elements, start=1):
        lines.append(f"{i}," + ",".join(str(int(n) + 1) for n in conn))
    lines.extend([
        "*MATERIAL,NAME=STEEL",
        "*ELASTIC",
        f"{young:.12g},{poisson:.12g}",
        "*SOLID SECTION,ELSET=EALL,MATERIAL=STEEL",
        f"{thickness:.12g}",
        "*NSET,NSET=CHECK",
        str(check_node + 1),
        "*BOUNDARY",
    ])
    for dof, value in sorted(constraints.items()):
        node = dof // 2 + 1
        component = dof % 2 + 1
        lines.append(f"{node},{component},{component},{value:.12g}")
    lines.extend(["*STEP", "*STATIC", "*CLOAD"])
    f = sol.load_vector.reshape((-1, 2))
    for node, pair in enumerate(f, start=1):
        for comp, value in enumerate(pair, start=1):
            if abs(float(value)) > 1e-12:
                lines.append(f"{node},{comp},{float(value):.12g}")
    lines.extend(["*NODE PRINT,NSET=CHECK,GLOBAL=YES", "U", "*END STEP"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_displacement(dat_path: Path, node_id: int) -> tuple[float, float, float]:
    lines = dat_path.read_text(encoding="utf-8", errors="replace").splitlines()
    active = False
    pattern = re.compile(rf"^\s*{node_id}\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})")
    candidates: list[tuple[float, float, float]] = []
    for line in lines:
        if "displacements" in line.lower():
            active = True
            continue
        if active:
            match = pattern.match(line)
            if match:
                candidates.append(tuple(float(x.replace("D", "E").replace("d", "e")) for x in match.groups()))
    if not candidates:
        for line in lines:
            match = pattern.match(line)
            if match:
                candidates.append(tuple(float(x.replace("D", "E").replace("d", "e")) for x in match.groups()))
    if not candidates:
        raise RuntimeError(f"could not parse displacement for node {node_id} from {dat_path}")
    return candidates[-1]


def run_calculix_crosschecks(runs: list[ScenarioRun], output_dir: Path, *, tolerance: float = 0.08) -> dict[str, Any]:
    ccx = _find_ccx()
    root = output_dir / "calculix_crosscheck"
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for run in runs:
        key = _baseline_key(run)
        node, component = _check_node_component(run, key)
        case_dir = root / run.scenario_id
        case_dir.mkdir(parents=True, exist_ok=True)
        job = "check"
        inp = case_dir / f"{job}.inp"
        _write_inp(run, key, inp, node)
        proc = subprocess.run([ccx, job], cwd=case_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, check=False)
        (case_dir / "ccx_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        dat = case_dir / f"{job}.dat"
        if proc.returncode != 0 or not dat.exists():
            results.append({"scenario_id": run.scenario_id, "passed": False, "error": f"ccx return code {proc.returncode}; dat_exists={dat.exists()}", "stdout_tail": proc.stdout[-2000:]})
            continue
        try:
            ccx_u = _parse_displacement(dat, node + 1)[component - 1]
            ref_u = float(run.solutions[key].displacements[node, component - 1])
            rel = abs(ccx_u - ref_u) / max(abs(ccx_u), abs(ref_u), 1e-30)
            results.append({
                "scenario_id": run.scenario_id,
                "mesh_key": key,
                "node": node + 1,
                "component": component,
                "reference_displacement": ref_u,
                "calculix_displacement": ccx_u,
                "relative_difference": rel,
                "passed": bool(rel <= tolerance),
            })
        except Exception as exc:
            results.append({"scenario_id": run.scenario_id, "passed": False, "error": f"{type(exc).__name__}: {exc}", "dat_tail": dat.read_text(encoding="utf-8", errors="replace")[-4000:]})
    receipt = {
        "solver": ccx,
        "tolerance": tolerance,
        "scenario_count": len(runs),
        "passed_count": sum(bool(r.get("passed")) for r in results),
        "valid": all(bool(r.get("passed")) for r in results) and len(results) == len(runs),
        "results": results,
    }
    (output_dir / "calculix_crosscheck.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt
