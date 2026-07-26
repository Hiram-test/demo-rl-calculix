#!/usr/bin/env python3
"""Run a real CalculiX mesh study and feed solver results to the diagnosis layer.

The controlled benchmark is a plane-stress cantilever plate with a concentrated
load at the upper-right corner. The local stress at the load application point
is singular, while a displacement measured at a fixed point away from the load
should approach a stable value. Three independently generated meshes are solved
by CalculiX; no response values are pre-filled.
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

from mesh_need import diagnose_question  # noqa: E402


FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?")


def _node_id(i: int, j: int, nx: int) -> int:
    return j * (nx + 1) + i + 1


def _format_ids(ids: list[int], per_line: int = 16) -> list[str]:
    return [", ".join(str(value) for value in ids[start : start + per_line]) for start in range(0, len(ids), per_line)]


def write_input(path: Path, nx: int, ny: int) -> dict[str, int | float]:
    length = 100.0
    height = 20.0
    thickness = 1.0
    young = 210000.0
    poisson = 0.3
    load = -1000.0

    dx = length / nx
    dy = height / ny
    lines = [
        "*HEADING",
        f"Real CalculiX point-load singularity study: {nx} x {ny}",
        "*NODE",
    ]

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
            # CalculiX *EL PRINT stress rows contain element, integration point,
            # followed by six Cartesian stress components.
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
        raise RuntimeError(
            f"CalculiX failed for {job} with code {completed.returncode}:\n{completed.stdout}"
        )

    dat = output_dir / f"{job}.dat"
    if not dat.exists():
        raise RuntimeError(f"CalculiX did not create {dat.name}")
    peak_stress, probe_u2 = parse_dat(dat, int(metadata["probe_node"]))
    return {
        **metadata,
        "job": job,
        "peak_stress": peak_stress,
        "reference_qoi": probe_u2,
    }


def relative_change(previous: float, current: float) -> float:
    return abs(current - previous) / max(abs(previous), 1.0e-15)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccx", default=None)
    parser.add_argument("--output-dir", default="artifacts/real-calculix")
    args = parser.parse_args()

    ccx = resolve_ccx(args.ccx)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meshes = [(20, 4), (40, 8), (80, 16)]
    history = [solve_mesh(ccx, output_dir, nx, ny) for nx, ny in meshes]
    peaks = [float(row["peak_stress"]) for row in history]
    references = [float(row["reference_qoi"]) for row in history]
    peak_monotonic = all(b > a for a, b in zip(peaks, peaks[1:]))
    reference_last_change = relative_change(references[-2], references[-1])

    case = {
        "question": (
            "悬臂板右上角施加集中力。加载点附近最大应力随网格细化升高，"
            "但我关心的是远离加载点的固定位置位移。是否应该把最大应力热点"
            "交给PSO继续加密？"
        ),
        "intended_use": "判断远离集中力加载点的结构位移响应",
        "qoi": "板中部固定物理位置的竖向位移",
        "mesh_history": history,
        "acceptance": {"reference_relative_change_max": 0.03},
    }
    diagnosis = diagnose_question(case)
    result = {
        "solver": {
            "executable": ccx,
            "version_output": subprocess.run(
                [ccx, "-v"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
            ).stdout.strip(),
        },
        "benchmark": "plane_stress_cantilever_with_corner_point_load",
        "mesh_history": history,
        "observed": {
            "peak_stress_strictly_increases": peak_monotonic,
            "peak_last_to_first_ratio": peaks[-1] / peaks[0],
            "reference_qoi_last_relative_change": reference_last_change,
        },
        "diagnosis": diagnosis,
    }

    result_path = output_dir / "real_calculix_diagnosis.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not peak_monotonic:
        raise SystemExit("Expected the point-load peak stress to increase monotonically")
    if reference_last_change > 0.03:
        raise SystemExit(
            f"Fixed reference displacement did not stabilize: last change={reference_last_change:.3%}"
        )
    if diagnosis.get("recommended_skill") != "qoi_and_singularity_guard":
        raise SystemExit(f"Unexpected diagnosis: {diagnosis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
