"""Three-dimensional Gmsh/CalculiX backend for the TQZ(XII) support family.

The model is intentionally a local linear-elastic benchmark, not an engineering
acceptance model. A rectangular C50 concrete block represents the local girder
bottom/pedestal region. The support's effective load footprint is taken from the
C x D dimensions in the drawing-derived contract. The horizontal design force
is 1.5*Ag times the nominal vertical capacity, matching the 0.15P/0.225P/0.30P
bearing labels. Its height H is retained as a moment arm by adding a statically
equivalent linear vertical-load gradient over the loaded top nodes.

Six mesh-control patches form a 3 x 2 partition of the loaded footprint. A PSO
particle chooses one L0-L3 mesh size for each patch. Gmsh creates first-order
tetrahedra and CalculiX solves a C3D4 static elasticity problem.
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
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


Position = Tuple[int, ...]
Command = Sequence[str]


@dataclass(frozen=True)
class TQZCase:
    case_id: str
    bearing_model: str
    nominal_vertical_capacity_kN: float
    ag: float
    A: float
    B: float
    C: float
    D: float
    H: float
    scope_status: str = "interpolation"

    def validated(self) -> "TQZCase":
        values = {
            "nominal_vertical_capacity_kN": self.nominal_vertical_capacity_kN,
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "D": self.D,
            "H": self.H,
        }
        for name, value in values.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= float(self.ag) <= 1.0:
            raise ValueError("ag must lie in [0, 1]")
        if self.C > self.A + 1.0e-9 or self.D > self.B + 1.0e-9:
            raise ValueError("effective C x D load footprint must fit inside A x B")
        return self

    @property
    def horizontal_ratio(self) -> float:
        return 1.5 * float(self.ag)

    @property
    def vertical_force_n(self) -> float:
        return float(self.nominal_vertical_capacity_kN) * 1_000.0

    @property
    def horizontal_force_n(self) -> float:
        return self.horizontal_ratio * self.vertical_force_n


@dataclass(frozen=True)
class TQZMaterial:
    name: str = "C50 concrete"
    young_modulus_mpa: float = 35_500.0
    poisson_ratio: float = 0.20

    def validated(self) -> "TQZMaterial":
        if self.young_modulus_mpa <= 0.0:
            raise ValueError("young_modulus_mpa must be positive")
        if not -0.49 < self.poisson_ratio < 0.49:
            raise ValueError("poisson_ratio must lie in (-0.49, 0.49)")
        return self


@dataclass(frozen=True)
class TQZMeshSpec:
    mesh_levels_mm: Tuple[float, ...] = (180.0, 130.0, 95.0, 70.0)
    reference_mesh_size_mm: float = 55.0
    block_margin_x_mm: float = 260.0
    block_margin_y_mm: float = 260.0
    block_depth_mm: float = 600.0
    patch_depth_mm: float = 280.0
    element_budget: int = 18_000
    resource_weight: float = 0.025
    budget_penalty: float = 5.0
    solver_timeout_seconds: int = 300

    def validated(self) -> "TQZMeshSpec":
        if len(self.mesh_levels_mm) < 2:
            raise ValueError("mesh_levels_mm must contain at least two levels")
        if any(float(value) <= 0.0 for value in self.mesh_levels_mm):
            raise ValueError("mesh levels must be positive")
        if any(
            self.mesh_levels_mm[index + 1] >= self.mesh_levels_mm[index]
            for index in range(len(self.mesh_levels_mm) - 1)
        ):
            raise ValueError("mesh levels must decrease from L0 to the finest level")
        if self.reference_mesh_size_mm <= 0.0:
            raise ValueError("reference_mesh_size_mm must be positive")
        if self.reference_mesh_size_mm >= min(self.mesh_levels_mm):
            raise ValueError("reference mesh must be finer than all optimization levels")
        if min(
            self.block_margin_x_mm,
            self.block_margin_y_mm,
            self.block_depth_mm,
            self.patch_depth_mm,
        ) <= 0.0:
            raise ValueError("local block dimensions must be positive")
        if self.patch_depth_mm > self.block_depth_mm:
            raise ValueError("patch_depth_mm cannot exceed block_depth_mm")
        if self.element_budget < 1:
            raise ValueError("element_budget must be positive")
        return self


@dataclass(frozen=True)
class SupportMesh:
    nodes: Mapping[int, Tuple[float, float, float]]
    tetrahedra: Mapping[int, Tuple[int, int, int, int]]


@dataclass(frozen=True)
class SupportRunResult:
    qoi: float
    mean_vertical_displacement: float
    max_displacement: float
    compliance: float
    element_count: int
    node_count: int
    loaded_node_count: int
    total_vertical_force: float
    total_horizontal_force: float
    applied_moment_y: float
    workdir: str
    mesh_signature: Tuple[int, int, int]

    def to_dict(self) -> dict:
        return asdict(self)


def split_command(command: str | Sequence[str]) -> list[str]:
    if not isinstance(command, str):
        values = [str(item) for item in command]
        if not values:
            raise ValueError("command cannot be empty")
        return values
    stripped = command.strip()
    if not stripped:
        raise ValueError("command cannot be empty")
    expanded = os.path.expandvars(os.path.expanduser(stripped.strip('"')))
    if Path(expanded).exists():
        return [expanded]
    return shlex.split(stripped, posix=os.name != "nt")


def command_available(command: str | Sequence[str]) -> bool:
    executable = split_command(command)[0].strip('"')
    return Path(executable).exists() or shutil.which(executable) is not None


def _run_command(command: Command, cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def parse_msh2_tetra(filepath: str | Path) -> SupportMesh:
    """Read nodes and first-order tetrahedra from an ASCII Gmsh 2.2 file."""

    path = Path(filepath)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nodes: Dict[int, Tuple[float, float, float]] = {}
    tetrahedra: Dict[int, Tuple[int, int, int, int]] = {}
    index = 0
    while index < len(lines):
        marker = lines[index].strip()
        if marker == "$Nodes":
            count = int(lines[index + 1].strip())
            for offset in range(count):
                parts = lines[index + 2 + offset].split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed node record: {lines[index + 2 + offset]!r}")
                nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
            index += count + 2
        elif marker == "$Elements":
            count = int(lines[index + 1].strip())
            for offset in range(count):
                parts = lines[index + 2 + offset].split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed element record: {lines[index + 2 + offset]!r}")
                element_id = int(parts[0])
                element_type = int(parts[1])
                number_of_tags = int(parts[2])
                connectivity = [int(value) for value in parts[3 + number_of_tags :]]
                if element_type == 4:
                    if len(connectivity) != 4:
                        raise ValueError("Gmsh tetrahedron does not have four nodes")
                    tetrahedra[element_id] = tuple(connectivity)  # type: ignore[assignment]
            index += count + 2
        else:
            index += 1
    if not nodes:
        raise ValueError(f"No nodes found in {path}")
    if not tetrahedra:
        raise ValueError(f"No first-order tetrahedra found in {path}")
    return SupportMesh(nodes=nodes, tetrahedra=tetrahedra)


def parse_frd_displacements(filepath: str | Path) -> Dict[int, Tuple[float, float, float]]:
    """Read the last ASCII DISP block in a CalculiX FRD file."""

    path = Path(filepath)
    result: Dict[int, Tuple[float, float, float]] = {}
    current: Dict[int, Tuple[float, float, float]] = {}
    in_disp = False
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if not tokens:
            continue
        if tokens[0] == "-4":
            in_disp = len(tokens) > 1 and tokens[1].upper().startswith("DISP")
            if in_disp:
                current = {}
            continue
        if in_disp and tokens[0] == "-1":
            try:
                node_id = int(raw_line[3:13].strip())
                fields = [raw_line[start : start + 12] for start in (13, 25, 37)]
                if any(not field.strip() for field in fields):
                    raise ValueError("incomplete fixed-width record")
                values = [float(field.replace("D", "E").replace("d", "e")) for field in fields]
            except (ValueError, IndexError):
                if len(tokens) < 4:
                    continue
                node_id = int(tokens[1])
                values = [
                    float(value.replace("D", "E").replace("d", "e"))
                    for value in tokens[2:5]
                ]
                values.extend([0.0] * (3 - len(values)))
            current[node_id] = (values[0], values[1], values[2])
            continue
        if in_disp and tokens[0] == "-3":
            if current:
                result = current
            in_disp = False
    if not result:
        raise ValueError(f"No ASCII DISP dataset found in {path}")
    return result


def block_dimensions(case: TQZCase, spec: TQZMeshSpec) -> Tuple[float, float, float]:
    case = case.validated()
    spec = spec.validated()
    return (
        float(case.A + 2.0 * spec.block_margin_x_mm),
        float(case.B + 2.0 * spec.block_margin_y_mm),
        float(spec.block_depth_mm),
    )


def patch_boxes(case: TQZCase, spec: TQZMeshSpec) -> Tuple[Tuple[float, float, float, float, float, float], ...]:
    """Return six 3x2 boxes covering the effective C x D top footprint."""

    case = case.validated()
    spec = spec.validated()
    x_edges = np.linspace(-case.C * 0.5, case.C * 0.5, 4)
    y_edges = np.linspace(-case.D * 0.5, case.D * 0.5, 3)
    epsilon = max(case.A, case.B) * 1.0e-8
    z_min = spec.block_depth_mm - spec.patch_depth_mm
    z_max = spec.block_depth_mm + epsilon
    boxes = []
    for iy in range(2):
        for ix in range(3):
            boxes.append(
                (
                    float(x_edges[ix] - epsilon),
                    float(x_edges[ix + 1] + epsilon),
                    float(y_edges[iy] - epsilon),
                    float(y_edges[iy + 1] + epsilon),
                    float(z_min),
                    float(z_max),
                )
            )
    return tuple(boxes)


def _write_geo(
    filepath: Path,
    case: TQZCase,
    spec: TQZMeshSpec,
    global_size: float,
    patch_sizes: Sequence[float],
) -> None:
    if len(patch_sizes) != 6:
        raise ValueError("exactly six patch mesh sizes are required")
    length, width, depth = block_dimensions(case, spec)
    lines = [
        'SetFactory("OpenCASCADE");',
        f"Box(1) = {{{-0.5 * length:.16g}, {-0.5 * width:.16g}, 0, {length:.16g}, {width:.16g}, {depth:.16g}}};",
        'Physical Volume("DOMAIN") = {1};',
        "Mesh.MshFileVersion = 2.2;",
        "Mesh.ElementOrder = 1;",
        "Mesh.Algorithm3D = 1;",
        "Mesh.Optimize = 1;",
        "Mesh.MeshSizeExtendFromBoundary = 0;",
        "Mesh.MeshSizeFromPoints = 0;",
        "Mesh.MeshSizeFromCurvature = 0;",
    ]
    fields = []
    for field_id, (box, size) in enumerate(zip(patch_boxes(case, spec), patch_sizes), start=1):
        x_min, x_max, y_min, y_max, z_min, z_max = box
        fields.append(field_id)
        lines.extend(
            [
                f"Field[{field_id}] = Box;",
                f"Field[{field_id}].VIn = {float(size):.16g};",
                f"Field[{field_id}].VOut = {float(global_size):.16g};",
                f"Field[{field_id}].XMin = {x_min:.16g};",
                f"Field[{field_id}].XMax = {x_max:.16g};",
                f"Field[{field_id}].YMin = {y_min:.16g};",
                f"Field[{field_id}].YMax = {y_max:.16g};",
                f"Field[{field_id}].ZMin = {z_min:.16g};",
                f"Field[{field_id}].ZMax = {z_max:.16g};",
            ]
        )
    min_field = len(fields) + 1
    lines.extend(
        [
            f"Field[{min_field}] = Min;",
            f"Field[{min_field}].FieldsList = {{{', '.join(str(value) for value in fields)}}};",
            f"Background Field = {min_field};",
            f"Mesh.MeshSizeMin = {min(min(patch_sizes), global_size):.16g};",
            f"Mesh.MeshSizeMax = {max(max(patch_sizes), global_size):.16g};",
        ]
    )
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_boundary_nodes(
    case: TQZCase,
    spec: TQZMeshSpec,
    mesh: SupportMesh,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    length, width, depth = block_dimensions(case, spec)
    tolerance = max(length, width, depth) * 1.0e-6
    fixed = sorted(node_id for node_id, (_, _, z) in mesh.nodes.items() if abs(z) <= tolerance)
    loaded = sorted(
        node_id
        for node_id, (x, y, z) in mesh.nodes.items()
        if abs(z - depth) <= tolerance
        and abs(x) <= 0.5 * case.C + tolerance
        and abs(y) <= 0.5 * case.D + tolerance
    )
    if not fixed:
        raise RuntimeError("mesh has no nodes on the fixed bottom face")
    if len(loaded) < 4:
        raise RuntimeError("mesh has too few nodes on the loaded top footprint")
    return tuple(fixed), tuple(loaded)


def compute_nodal_loads(
    case: TQZCase,
    loaded_node_ids: Iterable[int],
    nodes: Mapping[int, Tuple[float, float, float]],
) -> Dict[int, Tuple[float, float, float]]:
    """Distribute vertical force, horizontal force, and H-arm moment exactly."""

    case = case.validated()
    ids = tuple(int(value) for value in loaded_node_ids)
    if len(ids) < 2:
        raise ValueError("at least two loaded nodes are required")
    x = np.asarray([nodes[node_id][0] for node_id in ids], dtype=np.float64)
    vertical = case.vertical_force_n
    horizontal = case.horizontal_force_n
    moment_y = horizontal * case.H
    system = np.asarray(
        [[float(len(ids)), float(x.sum())], [float(x.sum()), float(x @ x)]],
        dtype=np.float64,
    )
    right_hand_side = np.asarray([-vertical, -moment_y], dtype=np.float64)
    if abs(float(np.linalg.det(system))) <= 1.0e-12:
        raise RuntimeError("loaded nodes cannot represent the horizontal-force moment arm")
    intercept, slope = np.linalg.solve(system, right_hand_side)
    vertical_forces = intercept + slope * x
    fx = horizontal / len(ids)
    return {
        node_id: (float(fx), 0.0, float(vertical_forces[index]))
        for index, node_id in enumerate(ids)
    }


def load_resultants(
    loads: Mapping[int, Tuple[float, float, float]],
    nodes: Mapping[int, Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    total_fx = sum(value[0] for value in loads.values())
    total_fz = sum(value[2] for value in loads.values())
    moment_y = sum(nodes[node_id][0] * value[2] for node_id, value in loads.items())
    return float(total_fx), float(total_fz), float(moment_y)


def _format_id_lines(values: Iterable[int], per_line: int = 16) -> list[str]:
    ids = [int(value) for value in values]
    return [", ".join(str(value) for value in ids[start : start + per_line]) for start in range(0, len(ids), per_line)]


def _write_deck(
    filepath: Path,
    case: TQZCase,
    material: TQZMaterial,
    mesh: SupportMesh,
    fixed_nodes: Sequence[int],
    loaded_nodes: Sequence[int],
    loads: Mapping[int, Tuple[float, float, float]],
) -> None:
    lines = ["*HEADING", f"MeshPilot TQZ local support benchmark: {case.case_id}", "*NODE"]
    for node_id, (x, y, z) in sorted(mesh.nodes.items()):
        lines.append(f"{node_id}, {x:.16g}, {y:.16g}, {z:.16g}")
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=EALL")
    for element_id, connectivity in sorted(mesh.tetrahedra.items()):
        lines.append(f"{element_id}, {connectivity[0]}, {connectivity[1]}, {connectivity[2]}, {connectivity[3]}")
    lines.extend(["*NSET, NSET=FIXED", *_format_id_lines(fixed_nodes)])
    lines.extend(["*NSET, NSET=LOADED", *_format_id_lines(loaded_nodes)])
    lines.extend(
        [
            "*MATERIAL, NAME=CONCRETE",
            "*ELASTIC",
            f"{material.young_modulus_mpa:.16g}, {material.poisson_ratio:.16g}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=CONCRETE",
            "*BOUNDARY",
            "FIXED, 1, 3, 0.0",
            "*STEP",
            "*STATIC",
            "1.0, 1.0, 1.0e-05, 1.0",
            "*CLOAD",
        ]
    )
    for node_id in loaded_nodes:
        fx, fy, fz = loads[int(node_id)]
        if abs(fx) > 0.0:
            lines.append(f"{node_id}, 1, {fx:.16g}")
        if abs(fy) > 0.0:
            lines.append(f"{node_id}, 2, {fy:.16g}")
        if abs(fz) > 0.0:
            lines.append(f"{node_id}, 3, {fz:.16g}")
    lines.extend(["*NODE FILE", "U", "*END STEP"])
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_support_analysis(
    case: TQZCase,
    material: TQZMaterial,
    spec: TQZMeshSpec,
    workdir: str | Path,
    gmsh_cmd: str | Sequence[str] = "gmsh",
    ccx_cmd: str | Sequence[str] = "ccx",
    *,
    global_size: float,
    patch_sizes: Sequence[float],
) -> SupportRunResult:
    """Generate, solve, and post-process one local TQZ mesh configuration."""

    case = case.validated()
    material = material.validated()
    spec = spec.validated()
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    geo_path = root / "model.geo"
    msh_path = root / "model.msh"
    inp_path = root / "model.inp"
    _write_geo(geo_path, case, spec, float(global_size), tuple(float(v) for v in patch_sizes))

    gmsh_command = split_command(gmsh_cmd) + [
        str(geo_path.name),
        "-3",
        "-format",
        "msh2",
        "-o",
        str(msh_path.name),
        "-v",
        "2",
    ]
    gmsh_result = _run_command(gmsh_command, root, spec.solver_timeout_seconds)
    (root / "gmsh.stdout.log").write_text(gmsh_result.stdout or "", encoding="utf-8")
    (root / "gmsh.stderr.log").write_text(gmsh_result.stderr or "", encoding="utf-8")
    if gmsh_result.returncode != 0 or not msh_path.exists():
        raise RuntimeError(f"Gmsh failed with return code {gmsh_result.returncode}")

    mesh = parse_msh2_tetra(msh_path)
    fixed_nodes, loaded_nodes = select_boundary_nodes(case, spec, mesh)
    loads = compute_nodal_loads(case, loaded_nodes, mesh.nodes)
    total_fx, total_fz, moment_y = load_resultants(loads, mesh.nodes)
    _write_deck(inp_path, case, material, mesh, fixed_nodes, loaded_nodes, loads)

    ccx_command = split_command(ccx_cmd) + ["-i", "model"]
    ccx_result = _run_command(ccx_command, root, spec.solver_timeout_seconds)
    (root / "ccx.stdout.log").write_text(ccx_result.stdout or "", encoding="utf-8")
    (root / "ccx.stderr.log").write_text(ccx_result.stderr or "", encoding="utf-8")
    frd_path = root / "model.frd"
    if ccx_result.returncode != 0 or not frd_path.exists():
        raise RuntimeError(f"CalculiX failed with return code {ccx_result.returncode}")

    displacements = parse_frd_displacements(frd_path)
    loaded_disp = []
    compliance = 0.0
    max_displacement = 0.0
    for node_id, vector in displacements.items():
        max_displacement = max(max_displacement, math.sqrt(sum(component * component for component in vector)))
        if node_id in loads:
            loaded_disp.append(vector)
            force = loads[node_id]
            compliance += force[0] * vector[0] + force[1] * vector[1] + force[2] * vector[2]
    if len(loaded_disp) != len(loaded_nodes):
        missing = len(loaded_nodes) - len(loaded_disp)
        raise RuntimeError(f"FRD is missing {missing} loaded-node displacement records")
    mean_uz = float(np.mean([value[2] for value in loaded_disp]))
    mean_vertical = abs(mean_uz)
    result = SupportRunResult(
        qoi=float(mean_vertical),
        mean_vertical_displacement=float(mean_vertical),
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
    (root / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result
