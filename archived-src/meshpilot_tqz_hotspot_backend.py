"""Real Gmsh/CalculiX hotspot extraction for the TQZ support family.

A uniform coarse solve is post-processed into element von Mises stress.  The upper
part of the local concrete block is divided into virtual cells; each cell receives
a stress-plus-neighbour-contrast score.  Adjacent high-score cells are merged and
the top regions become fixed mesh-control candidates for one PSO run.

No surrogate, response surface, or synthetic objective is used.  Every objective
value consumed by the optimizers comes from a real CalculiX solve.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from meshpilot_tqz_backend import (
    SupportMesh,
    SupportRunResult,
    TQZCase,
    TQZMaterial,
    TQZMeshSpec,
    block_dimensions,
    command_available,
    compute_nodal_loads,
    load_resultants,
    parse_frd_displacements,
    parse_msh2_tetra,
    select_boundary_nodes,
    split_command,
)
from meshpilot_tqz_dense_tags import renumber_support_mesh


Box = Tuple[float, float, float, float, float, float]
CellIndex = Tuple[int, int, int]
Position = Tuple[int, ...]


@dataclass(frozen=True)
class HotspotSpec:
    grid_x: int = 5
    grid_y: int = 4
    grid_z: int = 3
    candidate_count: int = 6
    z_min_ratio: float = 0.20
    stress_weight: float = 0.70
    contrast_weight: float = 0.30
    merge_ratio: float = 0.72
    max_cells_per_region: int = 3
    expansion_ratio: float = 0.08
    match_max_cost: float = 0.90

    def validated(self) -> "HotspotSpec":
        if min(self.grid_x, self.grid_y, self.grid_z) < 1:
            raise ValueError("hotspot grid dimensions must be positive")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if not 0.0 <= self.z_min_ratio < 1.0:
            raise ValueError("z_min_ratio must lie in [0, 1)")
        if self.stress_weight < 0.0 or self.contrast_weight < 0.0:
            raise ValueError("hotspot weights cannot be negative")
        if self.stress_weight + self.contrast_weight <= 0.0:
            raise ValueError("at least one hotspot weight must be positive")
        if not 0.0 <= self.merge_ratio <= 1.0:
            raise ValueError("merge_ratio must lie in [0, 1]")
        if self.max_cells_per_region < 1:
            raise ValueError("max_cells_per_region must be positive")
        if self.expansion_ratio < 0.0:
            raise ValueError("expansion_ratio cannot be negative")
        if self.match_max_cost <= 0.0:
            raise ValueError("match_max_cost must be positive")
        return self


@dataclass(frozen=True)
class HotspotCell:
    index: CellIndex
    bounds: Box
    element_count: int
    mean_mises: float
    p90_mises: float
    max_mises: float
    stress_signal: float
    contrast: float
    score: float


@dataclass(frozen=True)
class HotspotCandidate:
    candidate_id: int
    bounds: Box
    center_normalized: Tuple[float, float, float]
    size_normalized: Tuple[float, float, float]
    score: float
    stress_signal: float
    contrast: float
    element_count: int
    cells: Tuple[CellIndex, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["bounds"] = list(self.bounds)
        value["center_normalized"] = list(self.center_normalized)
        value["size_normalized"] = list(self.size_normalized)
        value["cells"] = [list(item) for item in self.cells]
        return value


@dataclass(frozen=True)
class HotspotAnalysis:
    result: SupportRunResult
    mesh: SupportMesh
    displacements: Mapping[int, Tuple[float, float, float]]
    element_von_mises: Mapping[int, float]


@dataclass(frozen=True)
class HotspotMatch:
    source_candidate_id: int
    target_candidate_id: int
    source_level: int
    cost: float

    def to_dict(self) -> dict:
        return asdict(self)


def _run_command(
    command: Sequence[str],
    cwd: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _compact_number(value: float, zero_tolerance: float = 1.0e-9) -> str:
    number = float(value)
    if abs(number) < zero_tolerance:
        return "0"
    return f"{number:.10g}"


def _id_lines(values: Iterable[int], per_line: int = 8) -> List[str]:
    ids = [int(value) for value in values]
    return [
        ", ".join(str(value) for value in ids[start : start + per_line])
        for start in range(0, len(ids), per_line)
    ]


def _write_geo(
    filepath: Path,
    case: TQZCase,
    spec: TQZMeshSpec,
    global_size: float,
    candidate_boxes: Sequence[Box],
    candidate_sizes: Sequence[float],
) -> None:
    if len(candidate_boxes) != len(candidate_sizes):
        raise ValueError("candidate box and size counts differ")
    length, width, depth = block_dimensions(case, spec)
    lines = [
        'SetFactory("OpenCASCADE");',
        (
            "Box(1) = {"
            f"{_compact_number(-0.5 * length)}, "
            f"{_compact_number(-0.5 * width)}, 0, "
            f"{_compact_number(length)}, {_compact_number(width)}, "
            f"{_compact_number(depth)}"
            "};"
        ),
        'Physical Volume("DOMAIN") = {1};',
        "Mesh.MshFileVersion = 2.2;",
        "Mesh.ElementOrder = 1;",
        "Mesh.Algorithm3D = 1;",
        "Mesh.Optimize = 1;",
        "Mesh.MeshSizeExtendFromBoundary = 0;",
        "Mesh.MeshSizeFromPoints = 0;",
        "Mesh.MeshSizeFromCurvature = 0;",
    ]

    field_ids: List[int] = []
    for field_id, (box, size) in enumerate(
        zip(candidate_boxes, candidate_sizes), start=1
    ):
        x_min, x_max, y_min, y_max, z_min, z_max = box
        field_ids.append(field_id)
        lines.extend(
            [
                f"Field[{field_id}] = Box;",
                f"Field[{field_id}].VIn = {_compact_number(size)};",
                f"Field[{field_id}].VOut = {_compact_number(global_size)};",
                f"Field[{field_id}].XMin = {_compact_number(x_min)};",
                f"Field[{field_id}].XMax = {_compact_number(x_max)};",
                f"Field[{field_id}].YMin = {_compact_number(y_min)};",
                f"Field[{field_id}].YMax = {_compact_number(y_max)};",
                f"Field[{field_id}].ZMin = {_compact_number(z_min)};",
                f"Field[{field_id}].ZMax = {_compact_number(z_max)};",
            ]
        )

    if field_ids:
        minimum_field = len(field_ids) + 1
        lines.extend(
            [
                f"Field[{minimum_field}] = Min;",
                (
                    f"Field[{minimum_field}].FieldsList = "
                    "{" + ", ".join(str(value) for value in field_ids) + "};"
                ),
                f"Background Field = {minimum_field};",
            ]
        )
    minimum_size = min([float(global_size), *[float(v) for v in candidate_sizes]])
    maximum_size = max([float(global_size), *[float(v) for v in candidate_sizes]])
    lines.extend(
        [
            f"Mesh.MeshSizeMin = {_compact_number(minimum_size)};",
            f"Mesh.MeshSizeMax = {_compact_number(maximum_size)};",
        ]
    )
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_deck(
    filepath: Path,
    case: TQZCase,
    material: TQZMaterial,
    mesh: SupportMesh,
    fixed_nodes: Sequence[int],
    loaded_nodes: Sequence[int],
    loads: Mapping[int, Tuple[float, float, float]],
) -> None:
    lines = [
        "*HEADING",
        f"MeshPilot hotspot TQZ benchmark: {case.case_id}",
        "*NODE",
    ]
    for node_id, (x, y, z) in sorted(mesh.nodes.items()):
        lines.append(
            f"{node_id}, {_compact_number(x)}, {_compact_number(y)}, "
            f"{_compact_number(z)}"
        )
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=EALL")
    for element_id, connectivity in sorted(mesh.tetrahedra.items()):
        lines.append(
            f"{element_id}, {connectivity[0]}, {connectivity[1]}, "
            f"{connectivity[2]}, {connectivity[3]}"
        )
    lines.extend(["*NSET, NSET=FIXED", *_id_lines(fixed_nodes)])
    lines.extend(["*NSET, NSET=LOADED", *_id_lines(loaded_nodes)])
    lines.extend(
        [
            "*MATERIAL, NAME=CONCRETE",
            "*ELASTIC",
            (
                f"{_compact_number(material.young_modulus_mpa)}, "
                f"{_compact_number(material.poisson_ratio)}"
            ),
            "*SOLID SECTION, ELSET=EALL, MATERIAL=CONCRETE",
            "*BOUNDARY",
            "FIXED, 1, 3, 0",
            "*STEP",
            "*STATIC",
            "1, 1, 1e-5, 1",
            "*CLOAD",
        ]
    )
    for node_id in loaded_nodes:
        fx, fy, fz = loads[int(node_id)]
        if abs(fx) > 0.0:
            lines.append(f"{node_id}, 1, {_compact_number(fx)}")
        if abs(fy) > 0.0:
            lines.append(f"{node_id}, 2, {_compact_number(fy)}")
        if abs(fz) > 0.0:
            lines.append(f"{node_id}, 3, {_compact_number(fz)}")
    lines.extend(["*NODE FILE", "U", "*END STEP"])
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconstruct_element_von_mises(
    mesh: SupportMesh,
    displacements: Mapping[int, Tuple[float, float, float]],
    material: TQZMaterial,
) -> Dict[int, float]:
    """Reconstruct constant C3D4 stress from the real CalculiX displacement field."""

    young = float(material.young_modulus_mpa)
    poisson = float(material.poisson_ratio)
    lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    shear = young / (2.0 * (1.0 + poisson))
    constitutive = np.asarray(
        [
            [lame + 2.0 * shear, lame, lame, 0.0, 0.0, 0.0],
            [lame, lame + 2.0 * shear, lame, 0.0, 0.0, 0.0],
            [lame, lame, lame + 2.0 * shear, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, shear, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, shear, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, shear],
        ],
        dtype=np.float64,
    )

    result: Dict[int, float] = {}
    for element_id, connectivity in mesh.tetrahedra.items():
        coordinates = np.asarray(
            [mesh.nodes[node_id] for node_id in connectivity],
            dtype=np.float64,
        )
        nodal_displacement = np.asarray(
            [displacements[node_id] for node_id in connectivity],
            dtype=np.float64,
        ).reshape(-1)
        coefficient = np.column_stack((np.ones(4), coordinates))
        try:
            inverse = np.linalg.inv(coefficient)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"degenerate tetrahedron {element_id}") from exc
        gradients = inverse[1:, :].T
        strain_matrix = np.zeros((6, 12), dtype=np.float64)
        for local_index, (dx, dy, dz) in enumerate(gradients):
            base = 3 * local_index
            strain_matrix[0, base] = dx
            strain_matrix[1, base + 1] = dy
            strain_matrix[2, base + 2] = dz
            strain_matrix[3, base] = dy
            strain_matrix[3, base + 1] = dx
            strain_matrix[4, base + 1] = dz
            strain_matrix[4, base + 2] = dy
            strain_matrix[5, base] = dz
            strain_matrix[5, base + 2] = dx
        stress = constitutive @ (strain_matrix @ nodal_displacement)
        sx, sy, sz, txy, tyz, tzx = stress
        result[element_id] = float(
            math.sqrt(
                max(
                    0.0,
                    0.5
                    * (
                        (sx - sy) ** 2
                        + (sy - sz) ** 2
                        + (sz - sx) ** 2
                    )
                    + 3.0 * (txy**2 + tyz**2 + tzx**2),
                )
            )
        )
    return result


def run_hotspot_analysis(
    case: TQZCase,
    material: TQZMaterial,
    spec: TQZMeshSpec,
    workdir: str | Path,
    gmsh_cmd: str | Sequence[str] = "gmsh",
    ccx_cmd: str | Sequence[str] = "ccx",
    *,
    global_size: float,
    candidate_boxes: Sequence[Box] = (),
    candidate_sizes: Sequence[float] = (),
) -> HotspotAnalysis:
    """Generate, solve, and fully post-process one real CalculiX analysis."""

    if not command_available(gmsh_cmd):
        raise FileNotFoundError(f"Gmsh command is unavailable: {gmsh_cmd}")
    if not command_available(ccx_cmd):
        raise FileNotFoundError(f"CalculiX command is unavailable: {ccx_cmd}")
    case = case.validated()
    material = material.validated()
    spec = spec.validated()
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    geo_path = root / "model.geo"
    msh_path = root / "model.msh"
    inp_path = root / "model.inp"
    _write_geo(
        geo_path,
        case,
        spec,
        float(global_size),
        tuple(candidate_boxes),
        tuple(float(value) for value in candidate_sizes),
    )

    gmsh_command = split_command(gmsh_cmd) + [
        geo_path.name,
        "-3",
        "-format",
        "msh2",
        "-o",
        msh_path.name,
        "-v",
        "2",
    ]
    gmsh_result = _run_command(
        gmsh_command,
        root,
        spec.solver_timeout_seconds,
    )
    (root / "gmsh.stdout.log").write_text(
        gmsh_result.stdout or "", encoding="utf-8"
    )
    (root / "gmsh.stderr.log").write_text(
        gmsh_result.stderr or "", encoding="utf-8"
    )
    if gmsh_result.returncode != 0 or not msh_path.exists():
        raise RuntimeError(f"Gmsh failed with return code {gmsh_result.returncode}")

    raw_mesh = parse_msh2_tetra(msh_path)
    mesh = renumber_support_mesh(raw_mesh)
    fixed_nodes, loaded_nodes = select_boundary_nodes(case, spec, mesh)
    loads = compute_nodal_loads(case, loaded_nodes, mesh.nodes)
    total_fx, total_fz, moment_y = load_resultants(loads, mesh.nodes)
    _write_deck(
        inp_path,
        case,
        material,
        mesh,
        fixed_nodes,
        loaded_nodes,
        loads,
    )

    ccx_command = split_command(ccx_cmd) + ["-i", "model"]
    ccx_result = _run_command(
        ccx_command,
        root,
        spec.solver_timeout_seconds,
    )
    (root / "ccx.stdout.log").write_text(
        ccx_result.stdout or "", encoding="utf-8"
    )
    (root / "ccx.stderr.log").write_text(
        ccx_result.stderr or "", encoding="utf-8"
    )
    frd_path = root / "model.frd"
    if ccx_result.returncode != 0 or not frd_path.exists():
        raise RuntimeError(f"CalculiX failed with return code {ccx_result.returncode}")

    displacements = parse_frd_displacements(frd_path)
    missing_displacements = set(mesh.nodes) - set(displacements)
    if missing_displacements:
        raise RuntimeError(
            f"FRD is missing {len(missing_displacements)} displacement records"
        )
    loaded_displacements = [displacements[node_id] for node_id in loaded_nodes]
    compliance = 0.0
    max_displacement = 0.0
    for node_id, vector in displacements.items():
        max_displacement = max(
            max_displacement,
            math.sqrt(sum(component * component for component in vector)),
        )
        if node_id in loads:
            force = loads[node_id]
            compliance += sum(force[index] * vector[index] for index in range(3))
    mean_vertical = abs(
        float(np.mean([vector[2] for vector in loaded_displacements]))
    )
    run_result = SupportRunResult(
        qoi=mean_vertical,
        mean_vertical_displacement=mean_vertical,
        max_displacement=float(max_displacement),
        compliance=float(compliance),
        element_count=len(mesh.tetrahedra),
        node_count=len(mesh.nodes),
        loaded_node_count=len(loaded_nodes),
        total_vertical_force=float(total_fz),
        total_horizontal_force=float(total_fx),
        applied_moment_y=float(moment_y),
        workdir=str(root),
        mesh_signature=(len(mesh.tetrahedra), len(mesh.nodes), len(loaded_nodes)),
    )
    von_mises = reconstruct_element_von_mises(mesh, displacements, material)
    (root / "result.json").write_text(
        json.dumps(run_result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "element_von_mises.json").write_text(
        json.dumps(
            {str(key): value for key, value in von_mises.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return HotspotAnalysis(
        result=run_result,
        mesh=mesh,
        displacements=displacements,
        element_von_mises=von_mises,
    )


def _cell_neighbours(index: CellIndex) -> Tuple[CellIndex, ...]:
    ix, iy, iz = index
    return (
        (ix - 1, iy, iz),
        (ix + 1, iy, iz),
        (ix, iy - 1, iz),
        (ix, iy + 1, iz),
        (ix, iy, iz - 1),
        (ix, iy, iz + 1),
    )


def score_hotspot_cells(
    case: TQZCase,
    mesh_spec: TQZMeshSpec,
    hotspot_spec: HotspotSpec,
    analysis: HotspotAnalysis,
) -> Tuple[HotspotCell, ...]:
    """Score real coarse-mesh cells by stress magnitude and neighbour contrast."""

    hotspot_spec = hotspot_spec.validated()
    length, width, depth = block_dimensions(case, mesh_spec)
    x_edges = np.linspace(-0.5 * length, 0.5 * length, hotspot_spec.grid_x + 1)
    y_edges = np.linspace(-0.5 * width, 0.5 * width, hotspot_spec.grid_y + 1)
    z_edges = np.linspace(
        depth * hotspot_spec.z_min_ratio,
        depth,
        hotspot_spec.grid_z + 1,
    )
    values: Dict[CellIndex, List[float]] = {}
    for element_id, connectivity in analysis.mesh.tetrahedra.items():
        centroid = np.mean(
            np.asarray(
                [analysis.mesh.nodes[node_id] for node_id in connectivity],
                dtype=np.float64,
            ),
            axis=0,
        )
        if centroid[2] < z_edges[0] or centroid[2] > z_edges[-1]:
            continue
        ix = int(np.clip(np.searchsorted(x_edges, centroid[0], side="right") - 1, 0, hotspot_spec.grid_x - 1))
        iy = int(np.clip(np.searchsorted(y_edges, centroid[1], side="right") - 1, 0, hotspot_spec.grid_y - 1))
        iz = int(np.clip(np.searchsorted(z_edges, centroid[2], side="right") - 1, 0, hotspot_spec.grid_z - 1))
        values.setdefault((ix, iy, iz), []).append(
            float(analysis.element_von_mises[element_id])
        )
    if not values:
        raise RuntimeError("coarse analysis produced no hotspot cells")

    raw: Dict[CellIndex, dict] = {}
    for index, stresses in values.items():
        array = np.asarray(stresses, dtype=np.float64)
        mean = float(np.mean(array))
        p90 = float(np.percentile(array, 90.0))
        maximum = float(np.max(array))
        signal = 0.45 * mean + 0.55 * p90
        ix, iy, iz = index
        raw[index] = {
            "bounds": (
                float(x_edges[ix]),
                float(x_edges[ix + 1]),
                float(y_edges[iy]),
                float(y_edges[iy + 1]),
                float(z_edges[iz]),
                float(z_edges[iz + 1]),
            ),
            "count": int(len(stresses)),
            "mean": mean,
            "p90": p90,
            "max": maximum,
            "signal": signal,
        }

    contrasts: Dict[CellIndex, float] = {}
    for index, item in raw.items():
        neighbour_signals = [
            raw[neighbour]["signal"]
            for neighbour in _cell_neighbours(index)
            if neighbour in raw
        ]
        contrasts[index] = float(
            np.mean(
                [abs(item["signal"] - value) for value in neighbour_signals]
            )
            if neighbour_signals
            else 0.0
        )
    stress_scale = max(max(item["signal"] for item in raw.values()), 1.0e-12)
    contrast_scale = max(max(contrasts.values()), 1.0e-12)
    weight_total = hotspot_spec.stress_weight + hotspot_spec.contrast_weight

    cells = []
    for index, item in raw.items():
        score = (
            hotspot_spec.stress_weight * item["signal"] / stress_scale
            + hotspot_spec.contrast_weight * contrasts[index] / contrast_scale
        ) / weight_total
        cells.append(
            HotspotCell(
                index=index,
                bounds=item["bounds"],
                element_count=item["count"],
                mean_mises=item["mean"],
                p90_mises=item["p90"],
                max_mises=item["max"],
                stress_signal=item["signal"],
                contrast=contrasts[index],
                score=float(score),
            )
        )
    cells.sort(key=lambda item: (-item.score, item.index))
    return tuple(cells)


def _region_candidate(
    candidate_id: int,
    region: Sequence[HotspotCell],
    case: TQZCase,
    mesh_spec: TQZMeshSpec,
    hotspot_spec: HotspotSpec,
) -> HotspotCandidate:
    length, width, depth = block_dimensions(case, mesh_spec)
    x_min = min(cell.bounds[0] for cell in region)
    x_max = max(cell.bounds[1] for cell in region)
    y_min = min(cell.bounds[2] for cell in region)
    y_max = max(cell.bounds[3] for cell in region)
    z_min = min(cell.bounds[4] for cell in region)
    z_max = max(cell.bounds[5] for cell in region)
    expand_x = hotspot_spec.expansion_ratio * (x_max - x_min)
    expand_y = hotspot_spec.expansion_ratio * (y_max - y_min)
    expand_z = hotspot_spec.expansion_ratio * (z_max - z_min)
    bounds = (
        max(-0.5 * length, x_min - expand_x),
        min(0.5 * length, x_max + expand_x),
        max(-0.5 * width, y_min - expand_y),
        min(0.5 * width, y_max + expand_y),
        max(0.0, z_min - expand_z),
        min(depth, z_max + expand_z),
    )
    center = (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )
    size = (
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )
    return HotspotCandidate(
        candidate_id=candidate_id,
        bounds=tuple(float(value) for value in bounds),
        center_normalized=(
            float((center[0] + 0.5 * length) / length),
            float((center[1] + 0.5 * width) / width),
            float(center[2] / depth),
        ),
        size_normalized=(
            float(size[0] / length),
            float(size[1] / width),
            float(size[2] / depth),
        ),
        score=float(max(cell.score for cell in region)),
        stress_signal=float(max(cell.stress_signal for cell in region)),
        contrast=float(max(cell.contrast for cell in region)),
        element_count=int(sum(cell.element_count for cell in region)),
        cells=tuple(sorted(cell.index for cell in region)),
    )


def select_hotspot_candidates(
    case: TQZCase,
    mesh_spec: TQZMeshSpec,
    hotspot_spec: HotspotSpec,
    cells: Sequence[HotspotCell],
) -> Tuple[HotspotCandidate, ...]:
    """Merge adjacent high-score cells and retain the strongest fixed candidates."""

    hotspot_spec = hotspot_spec.validated()
    by_index = {cell.index: cell for cell in cells}
    used: set[CellIndex] = set()
    regions: List[List[HotspotCell]] = []

    for seed in cells:
        if seed.index in used:
            continue
        region = [seed]
        used.add(seed.index)
        frontier = sorted(
            (
                by_index[index]
                for index in _cell_neighbours(seed.index)
                if index in by_index and index not in used
            ),
            key=lambda item: (-item.score, item.index),
        )
        while frontier and len(region) < hotspot_spec.max_cells_per_region:
            candidate = frontier.pop(0)
            if candidate.index in used:
                continue
            if candidate.score < hotspot_spec.merge_ratio * seed.score:
                continue
            region.append(candidate)
            used.add(candidate.index)
            for neighbour_index in _cell_neighbours(candidate.index):
                if neighbour_index in by_index and neighbour_index not in used:
                    frontier.append(by_index[neighbour_index])
            frontier.sort(key=lambda item: (-item.score, item.index))
        regions.append(region)
        if len(regions) >= hotspot_spec.candidate_count:
            break

    if len(regions) < hotspot_spec.candidate_count:
        for cell in cells:
            if cell.index not in used:
                regions.append([cell])
                used.add(cell.index)
                if len(regions) >= hotspot_spec.candidate_count:
                    break
    if len(regions) < hotspot_spec.candidate_count:
        raise RuntimeError(
            f"only {len(regions)} hotspot regions found; "
            f"{hotspot_spec.candidate_count} required"
        )

    candidates = [
        _region_candidate(
            index + 1,
            region,
            case,
            mesh_spec,
            hotspot_spec,
        )
        for index, region in enumerate(regions[: hotspot_spec.candidate_count])
    ]
    candidates.sort(key=lambda item: (-item.score, item.candidate_id))
    return tuple(
        HotspotCandidate(
            candidate_id=index + 1,
            bounds=candidate.bounds,
            center_normalized=candidate.center_normalized,
            size_normalized=candidate.size_normalized,
            score=candidate.score,
            stress_signal=candidate.stress_signal,
            contrast=candidate.contrast,
            element_count=candidate.element_count,
            cells=candidate.cells,
        )
        for index, candidate in enumerate(candidates)
    )


def hotspot_match_cost(
    source: HotspotCandidate,
    target: HotspotCandidate,
) -> float:
    center = np.asarray(source.center_normalized) - np.asarray(
        target.center_normalized
    )
    size = np.asarray(source.size_normalized) - np.asarray(target.size_normalized)
    return float(
        np.linalg.norm(center)
        + 0.30 * np.linalg.norm(size)
        + 0.15 * abs(float(source.score) - float(target.score))
    )


def match_hotspot_levels(
    source_candidates: Sequence[HotspotCandidate],
    source_levels: Sequence[int],
    target_candidates: Sequence[HotspotCandidate],
    *,
    max_cost: float,
    default_level: int = 0,
) -> Tuple[Position, Tuple[HotspotMatch, ...]]:
    """Greedily transfer levels between physically corresponding normalized hotspots."""

    if len(source_candidates) != len(source_levels):
        raise ValueError("source candidate and level counts differ")
    target_levels = [int(default_level)] * len(target_candidates)
    pairs = []
    for source_index, source in enumerate(source_candidates):
        for target_index, target in enumerate(target_candidates):
            pairs.append(
                (
                    hotspot_match_cost(source, target),
                    source_index,
                    target_index,
                )
            )
    pairs.sort()
    used_source: set[int] = set()
    used_target: set[int] = set()
    matches: List[HotspotMatch] = []
    for cost, source_index, target_index in pairs:
        if cost > max_cost:
            break
        if source_index in used_source or target_index in used_target:
            continue
        used_source.add(source_index)
        used_target.add(target_index)
        level = int(source_levels[source_index])
        target_levels[target_index] = level
        matches.append(
            HotspotMatch(
                source_candidate_id=source_candidates[source_index].candidate_id,
                target_candidate_id=target_candidates[target_index].candidate_id,
                source_level=level,
                cost=float(cost),
            )
        )
    matches.sort(key=lambda item: item.target_candidate_id)
    return tuple(target_levels), tuple(matches)
