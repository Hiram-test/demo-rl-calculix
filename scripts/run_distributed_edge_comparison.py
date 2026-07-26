#!/usr/bin/env python3
"""Compare a corner nodal load with a finite right-edge load segment.

The deterministic code only creates models, runs CalculiX, and records numerical
observations. It does not decide which physical explanation is correct.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mesh_need import build_analysis_packet  # noqa: E402
from run_real_calculix_demo import _format_ids, _node_id, resolve_ccx  # noqa: E402
from run_upper_right_refinement import (  # noqa: E402
    HEIGHT,
    LENGTH,
    PATCH_X_MIN,
    PATCH_Y_MIN,
    RING_INNER,
    RING_OUTER,
    _element_centroid,
    _nearest_gauss_distance,
    _relative_change,
    parse_dat,
)


TOTAL_VERTICAL_LOAD = -1000.0
LOAD_SEGMENT_HEIGHT = 2.5
THICKNESS = 1.0
YOUNG = 210000.0
POISSON = 0.3


def build_distributed_nodal_loads(
    nx: int,
    ny: int,
    *,
    segment_height: float = LOAD_SEGMENT_HEIGHT,
    total_vertical_load: float = TOTAL_VERTICAL_LOAD,
) -> dict[int, float]:
    """Return consistent nodal forces for a uniform vertical edge-line load."""

    dy = HEIGHT / ny
    segment_count_float = segment_height / dy
    segment_count = int(round(segment_count_float))
    if segment_count < 1 or not math.isclose(
        segment_count_float, segment_count, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError(
            f"segment height {segment_height} mm is not aligned with ny={ny} (dy={dy} mm)"
        )

    start_j = ny - segment_count
    line_load = total_vertical_load / segment_height
    nodal_loads: dict[int, float] = {}
    for j in range(start_j, ny):
        bottom = _node_id(nx, j, nx)
        top = _node_id(nx, j + 1, nx)
        equivalent = line_load * dy / 2.0
        nodal_loads[bottom] = nodal_loads.get(bottom, 0.0) + equivalent
        nodal_loads[top] = nodal_loads.get(top, 0.0) + equivalent

    if not math.isclose(
        sum(nodal_loads.values()), total_vertical_load, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise RuntimeError("distributed nodal loads do not preserve the requested resultant")
    return nodal_loads


def write_distributed_input(path: Path, nx: int, ny: int) -> dict[str, Any]:
    dx = LENGTH / nx
    dy = HEIGHT / ny
    nodal_loads = build_distributed_nodal_loads(nx, ny)
    lines = ["*HEADING", f"Distributed edge comparison: {nx} x {ny}", "*NODE"]

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
    lines.extend(["*NSET, NSET=LEFT", *_format_ids(left_nodes)])
    lines.extend(["*NSET, NSET=PROBE", str(probe_node)])
    lines.extend(
        [
            "*MATERIAL, NAME=STEEL",
            "*ELASTIC",
            f"{YOUNG}, {POISSON}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL",
            str(THICKNESS),
            "*BOUNDARY",
            "LEFT, 1, 2, 0.0",
            "*STEP",
            "*STATIC",
            "*CLOAD",
        ]
    )
    for node, force in sorted(nodal_loads.items()):
        lines.append(f"{node}, 2, {force:.12g}")
    lines.extend(
        [
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
        "load_node_count": len(nodal_loads),
        "load_segment_height_mm": LOAD_SEGMENT_HEIGHT,
        "line_load_n_per_mm": TOTAL_VERTICAL_LOAD / LOAD_SEGMENT_HEIGHT,
        "applied_total_vertical_force_n": sum(nodal_loads.values()),
    }


def solve_one(ccx: str, output_dir: Path, nx: int, ny: int) -> dict[str, Any]:
    job = f"distributed_edge_{nx}x{ny}"
    inp = output_dir / f"{job}.inp"
    metadata = write_distributed_input(inp, nx, ny)
    completed = subprocess.run(
        [ccx, job],
        cwd=output_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=360,
        check=False,
    )
    (output_dir / f"{job}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"CalculiX failed for {job}:\n{completed.stdout}")

    dat = output_dir / f"{job}.dat"
    stress_rows, probe_u2 = parse_dat(dat, int(metadata["probe_node"]))
    corner_element = nx * ny
    corner_values = [vm for eid, _ip, vm in stress_rows if eid == corner_element]
    if not corner_values:
        raise RuntimeError(f"No stresses found for upper-right element {corner_element}")

    ring_values: list[float] = []
    patch_values: list[float] = []
    for element, _ip, vm in stress_rows:
        cx, cy = _element_centroid(element, nx, ny)
        distance = math.hypot(LENGTH - cx, HEIGHT - cy)
        if RING_INNER <= distance <= RING_OUTER:
            ring_values.append(vm)
        if cx >= PATCH_X_MIN and cy >= PATCH_Y_MIN:
            patch_values.append(vm)

    nearest_distance = _nearest_gauss_distance(nx, ny)
    corner_peak = max(corner_values)
    return {
        **metadata,
        "job": job,
        "global_peak_stress": max(vm for _eid, _ip, vm in stress_rows),
        "upper_right_corner_element": corner_element,
        "upper_right_corner_peak_stress": corner_peak,
        "nearest_integration_point_distance": nearest_distance,
        "corner_peak_times_distance": corner_peak * nearest_distance,
        "upper_right_ring_sample_count": len(ring_values),
        "upper_right_ring_mean_stress": statistics.fmean(ring_values),
        "upper_right_ring_peak_stress": max(ring_values),
        "upper_right_patch_sample_count": len(patch_values),
        "upper_right_patch_mean_stress": statistics.fmean(patch_values),
        "upper_right_patch_peak_stress": max(patch_values),
        "remote_probe_vertical_displacement": probe_u2,
    }


def summarize(history: list[dict[str, Any]]) -> dict[str, Any]:
    corner = [float(row["upper_right_corner_peak_stress"]) for row in history]
    scaled = [float(row["corner_peak_times_distance"]) for row in history]
    ring = [float(row["upper_right_ring_mean_stress"]) for row in history]
    patch = [float(row["upper_right_patch_mean_stress"]) for row in history]
    probe = [float(row["remote_probe_vertical_displacement"]) for row in history]
    scaled_mean = statistics.fmean(scaled)
    return {
        "corner_peak_successive_ratios": [b / a for a, b in zip(corner, corner[1:])],
        "corner_peak_times_distance_mean": scaled_mean,
        "corner_peak_times_distance_coefficient_of_variation": (
            statistics.pstdev(scaled) / abs(scaled_mean)
        ),
        "ring_mean_last_relative_change": _relative_change(ring[-2], ring[-1]),
        "patch_mean_last_relative_change": _relative_change(patch[-2], patch[-1]),
        "remote_probe_last_relative_change": _relative_change(probe[-2], probe[-1]),
    }


def build_matching_comparison(
    point_history: list[dict[str, Any]], distributed_history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    point_by_mesh = {(int(row["nx"]), int(row["ny"])): row for row in point_history}
    comparison: list[dict[str, Any]] = []
    for distributed in distributed_history:
        key = (int(distributed["nx"]), int(distributed["ny"]))
        point = point_by_mesh.get(key)
        if point is None:
            continue
        point_peak = float(point["upper_right_corner_peak_stress"])
        distributed_peak = float(distributed["upper_right_corner_peak_stress"])
        point_scaled = float(point["corner_peak_times_distance"])
        distributed_scaled = float(distributed["corner_peak_times_distance"])
        comparison.append(
            {
                "nx": key[0],
                "ny": key[1],
                "point_corner_peak_stress": point_peak,
                "distributed_corner_peak_stress": distributed_peak,
                "distributed_to_point_corner_peak_ratio": distributed_peak / point_peak,
                "point_corner_peak_times_distance": point_scaled,
                "distributed_corner_peak_times_distance": distributed_scaled,
                "distributed_to_point_scaled_peak_ratio": distributed_scaled / point_scaled,
                "point_remote_probe_vertical_displacement": float(
                    point["remote_probe_vertical_displacement"]
                ),
                "distributed_remote_probe_vertical_displacement": float(
                    distributed["remote_probe_vertical_displacement"]
                ),
            }
        )
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccx", default=None)
    parser.add_argument(
        "--point-evidence",
        default="artifacts/upper-right-refinement/solver_evidence.json",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/distributed-edge-comparison"
    )
    args = parser.parse_args()

    ccx = resolve_ccx(args.ccx)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    point_evidence = json.loads(Path(args.point_evidence).read_text(encoding="utf-8"))

    distributed_history = [
        solve_one(ccx, output_dir, nx, ny)
        for nx, ny in [(40, 8), (80, 16), (160, 32), (320, 64)]
    ]
    distributed_evidence = {
        "solver": {
            "executable": ccx,
            "version_output": subprocess.run(
                [ccx, "-v"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            ).stdout.strip(),
        },
        "benchmark": "upper_right_finite_edge_load_refinement",
        "model": {
            "geometry_mm": [LENGTH, HEIGHT],
            "element_type": "CPS4",
            "left_boundary": "fully fixed in x and y",
            "load": (
                "-1000 N resultant represented as a uniform vertical line load over "
                f"the upper {LOAD_SEGMENT_HEIGHT} mm of the right edge"
            ),
            "load_discretization": "consistent equivalent nodal forces on aligned edge segments",
        },
        "regions": point_evidence.get("regions", {}),
        "mesh_history": distributed_history,
        "derived_numerical_summaries": summarize(distributed_history),
    }
    combined_evidence = {
        "controlled_comparison": {
            "held_constant": [
                "100 mm by 20 mm plane-stress geometry",
                "1 mm thickness",
                "steel elastic constants",
                "CPS4 elements",
                "left-edge x/y constraints",
                "-1000 N total vertical resultant",
                "region definitions and remote probe location",
            ],
            "changed": (
                "load introduction: one corner node versus a uniform vertical line load "
                f"over {LOAD_SEGMENT_HEIGHT} mm of the right edge"
            ),
            "point_case_meshes": [
                [int(row["nx"]), int(row["ny"])]
                for row in point_evidence["mesh_history"]
            ],
            "distributed_case_meshes": [
                [int(row["nx"]), int(row["ny"])] for row in distributed_history
            ],
        },
        "point_load_case": point_evidence,
        "distributed_edge_case": distributed_evidence,
        "matching_mesh_observations": build_matching_comparison(
            point_evidence["mesh_history"], distributed_history
        ),
    }
    case = {
        "question": (
            "比较单节点集中力和2.5 mm右边缘均布竖向线载荷的网格序列。不要预设"
            "集中力一定是唯一原因；根据两组原始证据判断哪些解释被支持、被削弱或"
            "仍无法区分，并提出下一项最小计算。"
        ),
        "intended_use": "判断局部峰值是否由载荷理想化主导，以及哪一种结果可支持工程决策",
        "qoi": "角点局部应力、固定区域统计量和远场固定点位移",
        "model_context": combined_evidence["controlled_comparison"],
    }
    packet = build_analysis_packet(case, combined_evidence)

    (output_dir / "distributed_solver_evidence.json").write_text(
        json.dumps(distributed_evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "comparison_solver_evidence.json").write_text(
        json.dumps(combined_evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "comparison_ai_analysis_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "distributed_edge_refinement.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distributed_history[0].keys()))
        writer.writeheader()
        writer.writerows(distributed_history)

    print(json.dumps(packet, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
