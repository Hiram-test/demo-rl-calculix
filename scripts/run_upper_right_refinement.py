#!/usr/bin/env python3
"""Run a focused right-upper-corner refinement study for AI interpretation.

The script extracts region-specific quantities but deliberately makes no claim
about the governing physical mechanism and applies no physics decision rules.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mesh_need import build_analysis_packet  # noqa: E402
from run_real_calculix_demo import _numbers, resolve_ccx, write_input  # noqa: E402


LENGTH = 100.0
HEIGHT = 20.0
RING_INNER = 5.0
RING_OUTER = 10.0
PATCH_X_MIN = 90.0
PATCH_Y_MIN = 10.0


def _von_mises(values: list[float]) -> float:
    sxx, syy, szz, sxy, sxz, syz = values
    vm2 = 0.5 * (
        (sxx - syy) ** 2
        + (syy - szz) ** 2
        + (szz - sxx) ** 2
    ) + 3.0 * (sxy**2 + sxz**2 + syz**2)
    return math.sqrt(max(vm2, 0.0))


def parse_dat(path: Path, probe_node: int) -> tuple[list[tuple[int, int, float]], float]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_displacements = False
    in_stresses = False
    displacement_started = False
    stress_started = False
    probe_u2: float | None = None
    stress_rows: list[tuple[int, int, float]] = []

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
                stress_rows.append(
                    (int(round(values[0])), int(round(values[1])), _von_mises(values[2:8]))
                )
                stress_started = True
                continue
            if stress_started and not line.strip():
                in_stresses = False

    if probe_u2 is None:
        raise RuntimeError(f"Could not parse PROBE displacement from {path}")
    if not stress_rows:
        raise RuntimeError(f"Could not parse integration-point stresses from {path}")
    return stress_rows, probe_u2


def _element_centroid(element: int, nx: int, ny: int) -> tuple[float, float]:
    index = element - 1
    i = index % nx
    j = index // nx
    dx = LENGTH / nx
    dy = HEIGHT / ny
    return (i + 0.5) * dx, (j + 0.5) * dy


def _nearest_gauss_distance(nx: int, ny: int) -> float:
    dx = LENGTH / nx
    dy = HEIGHT / ny
    offset = (1.0 - 1.0 / math.sqrt(3.0)) / 2.0
    return math.hypot(offset * dx, offset * dy)


def _relative_change(previous: float, current: float) -> float:
    return abs(current - previous) / max(abs(previous), 1.0e-15)


def solve_one(ccx: str, output_dir: Path, nx: int, ny: int) -> dict[str, float | int | str]:
    job = f"upper_right_{nx}x{ny}"
    inp = output_dir / f"{job}.inp"
    metadata = write_input(inp, nx, ny)
    completed = subprocess.run(
        [ccx, job],
        cwd=output_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
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
        raise RuntimeError(f"No stress rows found for upper-right element {corner_element}")

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccx", default=None)
    parser.add_argument("--output-dir", default="artifacts/upper-right-refinement")
    args = parser.parse_args()

    ccx = resolve_ccx(args.ccx)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history = [
        solve_one(ccx, output_dir, nx, ny)
        for nx, ny in [(20, 4), (40, 8), (80, 16), (160, 32)]
    ]

    corner_peaks = [float(row["upper_right_corner_peak_stress"]) for row in history]
    scaled_peaks = [float(row["corner_peak_times_distance"]) for row in history]
    ring_means = [float(row["upper_right_ring_mean_stress"]) for row in history]
    patch_means = [float(row["upper_right_patch_mean_stress"]) for row in history]
    probe_displacements = [float(row["remote_probe_vertical_displacement"]) for row in history]
    scaled_mean = statistics.fmean(scaled_peaks)

    evidence = {
        "solver": {
            "executable": ccx,
            "version_output": subprocess.run(
                [ccx, "-v"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
            ).stdout.strip(),
        },
        "benchmark": "upper_right_nodal_load_refinement",
        "model": {
            "geometry_mm": [LENGTH, HEIGHT],
            "element_type": "CPS4",
            "load": "-1000 N vertical CLOAD at the upper-right node",
            "left_boundary": "fully fixed in x and y",
        },
        "regions": {
            "corner": "single element touching the upper-right loaded node",
            "ring_mm": [RING_INNER, RING_OUTER],
            "patch_mm": {"x_min": PATCH_X_MIN, "y_min": PATCH_Y_MIN},
            "remote_probe": "fixed point at (50 mm, 10 mm)",
        },
        "mesh_history": history,
        "derived_numerical_summaries": {
            "corner_peak_successive_ratios": [b / a for a, b in zip(corner_peaks, corner_peaks[1:])],
            "corner_peak_times_distance_mean": scaled_mean,
            "corner_peak_times_distance_coefficient_of_variation": (
                statistics.pstdev(scaled_peaks) / abs(scaled_mean)
            ),
            "ring_mean_last_relative_change": _relative_change(ring_means[-2], ring_means[-1]),
            "patch_mean_last_relative_change": _relative_change(patch_means[-2], patch_means[-1]),
            "remote_probe_last_relative_change": _relative_change(
                probe_displacements[-2], probe_displacements[-1]
            ),
        },
    }
    case = {
        "question": (
            "请分析右上角附近的网格加密结果。不要预设它是奇异性；比较多个可能的"
            "载荷引入、边界、离散和结果提取解释，并提出最小的区分性试验。"
        ),
        "intended_use": "判断右上角局部结果是否能支持工程决策，以及下一步模型应如何改进",
        "qoi": "尚待AI根据工程用途和证据澄清",
        "model_context": {**evidence["model"], **evidence["regions"]},
    }
    packet = build_analysis_packet(case, evidence)

    (output_dir / "solver_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "ai_analysis_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = output_dir / "upper_right_refinement.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(json.dumps(packet, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
