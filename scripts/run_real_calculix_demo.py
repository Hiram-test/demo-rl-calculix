#!/usr/bin/env python3
"""Run a real CalculiX mesh study and build an AI analysis packet.

The script performs no physical classification. It creates three meshes, runs
CalculiX, extracts solver quantities and packages them with the user's question
for an AI model to interpret.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mesh_need import build_analysis_packet  # noqa: E402


FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?")


def _node_id(i: int, j: int, nx: int) -> int:
    return j * (nx + 1) + i + 1


def _format_ids(ids: list[int], per_line: int = 16) -> list[str]:
    return [
        ", ".join(str(value) for value in ids[start : start + per_line])
        for start in range(0, len(ids), per_line)
    ]


def write_input(path: Path, nx: int, ny: int) -> dict[str, int | float]:
    length = 100.0
    height = 20.0
    thickness = 1.0
    young = 210000.0
    poisson = 0.3
    load = -1000.0

    dx = length / nx
    dy = height / ny
    lines = ["*HEADING", f"CalculiX mesh study: {nx} x {ny}", "*NODE"]

    for j in range(ny + 1):
        for i in range(nx + 1):
            nid = _node_id(i, j, nx)
            lines.append(f"{nid}, {i * dx:.12g}, {j * dy:.12g}, 0.0")

    lines.append("*ELEMENT, TYPE=CPS4, ELSET=EALL")
    eid = 1
    for j in range(ny):
        for i in range(nx):
            n1 = _node_id(i, j, nx)
            n2 = _node_id(i + 1, j, nx)
            n3 = _node_id(i + 1, j + 1, nx)
            n4 = _node_id(i, j + 1, nx)
            lines.append(f"{eid}, {n1}, {n2}, {n3}, {n4}")
            eid += 1

    left_nodes = [_node_id(0, j, nx) for j in range(ny + 1)]
    probe_node = _node_id(nx // 2, ny // 2, nx)
    load_node = _node_id(nx, ny, nx)

    lines.extend(["*NSET, NSET=LEFT", *_format_ids(left_nodes)])
    lines.extend(["*NSET, NSET=PROBE", str(probe_node)])
    lines.extend(
        [
            "*MATERIAL, NAME=STEEL",
            "*ELASTIC",
            f"{young}, {poisson}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL",
            str(thickness),
            "*BOUNDARY",
            "LEFT, 1, 2, 0.0",
            "*STEP",
            "*STATIC",
            "*CLOAD",
            f"{load_node}, 2, {load}",
            "*NODE PRINT, NSET=PROBE",
            "U",
            "*EL PRINT, ELSET=EALL",
            "S",
            "*END STEP",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "nx": nx,
        "ny": ny,
        "elements": nx * ny,
        "nodes": (nx + 1) * (ny + 1),
        "mesh_size": max(dx, dy),
        "probe_node": probe_node,
        "load_node": load_node,
    }


def _numbers(line: str) -> list[float]:
    return [float(token.replace("D", "E").replace("d", "e")) for token in FLOAT_RE.findall(line)]


def parse_dat(path: Path, probe_node: int) -> tuple[float, float]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_displacements = False
    in_stresses = False
    displacement_started = False
    stress_started = False
    probe_u2: float | None = None
    von_mises_values: list[float] = []

    for line in lines:
        lower = line.lower()
        if "displacements" in lower and "probe" in lower:
            in_displacements = True
            in_stresses = False
            displacement_started = False
            continue
        if "stresses" in lower and "eall" in lower:
            in_stresses = True
            in_displacements = False
            stress_started = False
            continue

        if in_displacements:
            values = _numbers(line)
            if len(values) >= 3 and int(round(values[0])) == probe_node:
                probe_u2 = values[2]
                displacement_started = True
                continue
            if displacement_started and not line.strip():
                in_displacements = False

        if in_stresses:
            values = _numbers(line)
            if len(values) >= 8:
                sxx, syy, szz, sxy, syz, szx = values[2:8]
                vm2 = 0.5 * (
                    (sxx - syy) ** 2
                    + (syy - szz) ** 2
                    + (szz - sxx) ** 2
                ) + 3.0 * (sxy**2 + syz**2 + szx**2)
                von_mises_values.append(math.sqrt(max(vm2, 0.0)))
                stress_started = True
                continue
            if stress_started and not line.strip():
                in_stresses = False

    if probe_u2 is None:
        raise RuntimeError(f"Could not parse PROBE displacement from {path}")
    if not von_mises_values:
        raise RuntimeError(f"Could not parse integration-point stresses from {path}")
    return max(von_mises_values), probe_u2


def resolve_ccx(explicit: str | None) -> str:
    if explicit:
        candidate = shutil.which(explicit) or explicit
        if Path(candidate).exists() or shutil.which(candidate):
            return candidate
        raise FileNotFoundError(f"CalculiX executable not found: {explicit}")
    for name in ("ccx", "ccx_2.21", "ccx_2.22", "ccx_2.23"):
        candidate = shutil.which(name)
        if candidate:
            return candidate
    raise FileNotFoundError("CalculiX executable not found on PATH")


def solve_mesh(ccx: str, output_dir: Path, nx: int, ny: int) -> dict[str, float | int | str]:
    job = f"point_load_{nx}x{ny}"
    inp = output_dir / f"{job}.inp"
    metadata = write_input(inp, nx, ny)
    completed = subprocess.run(
        [ccx, job],
        cwd=output_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    (output_dir / f"{job}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"CalculiX failed for {job}:\n{completed.stdout}")

    dat = output_dir / f"{job}.dat"
    if not dat.exists():
        raise RuntimeError(f"CalculiX did not create {dat.name}")
    peak_stress, probe_u2 = parse_dat(dat, int(metadata["probe_node"]))
    return {**metadata, "job": job, "global_peak_stress": peak_stress, "probe_u2": probe_u2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccx", default=None)
    parser.add_argument("--output-dir", default="artifacts/real-calculix")
    args = parser.parse_args()

    ccx = resolve_ccx(args.ccx)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history = [solve_mesh(ccx, output_dir, nx, ny) for nx, ny in [(20, 4), (40, 8), (80, 16)]]
    evidence = {
        "solver": {
            "executable": ccx,
            "version_output": subprocess.run(
                [ccx, "-v"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
            ).stdout.strip(),
        },
        "benchmark": "plane_stress_cantilever_with_corner_nodal_load",
        "model": {
            "geometry_mm": [100.0, 20.0],
            "element_type": "CPS4",
            "left_boundary": "all left-edge nodes constrained in x and y",
            "load": "-1000 N vertical CLOAD on the upper-right node",
            "probe": "vertical displacement at fixed point (50 mm, 10 mm)",
        },
        "mesh_history": history,
    }
    case = {
        "question": "右上角附近结果随网格变化。这个现象说明什么，下一步最有区分力的计算是什么？",
        "intended_use": "理解载荷引入区与远场响应对网格的敏感性",
        "qoi": "尚未最终确定；候选包括右上角局部应力和远场固定点位移",
        "model_context": evidence["model"],
    }
    packet = build_analysis_packet(case, evidence)

    (output_dir / "solver_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "ai_analysis_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(packet, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
