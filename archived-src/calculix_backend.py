"""Local Gmsh + CalculiX backend for the state-aware graph DQN.

The backend uses a rectangular plate (optionally with a circular hole), divides
it into *virtual cells*, and controls one mesh-size value per virtual cell.  Gmsh
creates a first-order triangular mesh from a background size field.  CalculiX
solves a 2-D plane-strain cantilever problem.  Displacements are read from the
ASCII FRD file, and element stress/strain-energy features are reconstructed from
linear-triangle kinematics.

No solver executable is imported into Python.  Paths are supplied with
``GMSH_CMD``/``CCX_CMD`` or constructor arguments, making the same code usable on
Windows and Linux.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from mesh_goal import GoalCondition
from state_aware_dqn_agent import COARSEN, REFINE, GraphState, build_graph_state


CommandRunner = Callable[[Sequence[str], Path, int], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class PlateConfig:
    length: float = 10.0
    height: float = 4.0
    thickness: float = 1.0
    young_modulus: float = 210_000.0
    poisson_ratio: float = 0.30
    load_x: float = 0.0
    load_y: float = -1_000.0
    hole_center_x: float = 4.0
    hole_center_y: float = 2.0
    hole_radius: float = 0.75
    cells_x: int = 8
    cells_y: int = 4
    gmsh_algorithm: int = 6

    def validated(self) -> "PlateConfig":
        if self.length <= 0 or self.height <= 0 or self.thickness <= 0:
            raise ValueError("Plate dimensions and thickness must be positive")
        if self.young_modulus <= 0:
            raise ValueError("young_modulus must be positive")
        if not -0.49 < self.poisson_ratio < 0.49:
            raise ValueError("poisson_ratio must be between -0.49 and 0.49")
        if self.cells_x < 1 or self.cells_y < 1:
            raise ValueError("cells_x and cells_y must be positive")
        if self.hole_radius < 0:
            raise ValueError("hole_radius cannot be negative")
        if self.hole_radius > 0:
            margin_x = min(self.hole_center_x, self.length - self.hole_center_x)
            margin_y = min(self.hole_center_y, self.height - self.hole_center_y)
            if self.hole_radius >= min(margin_x, margin_y):
                raise ValueError("The circular hole must lie strictly inside the plate")
        return self

    @classmethod
    def from_json(cls, filepath: str | Path) -> "PlateConfig":
        with open(filepath, "r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise ValueError("Plate config JSON must contain one object")
        supported = set(cls.__dataclass_fields__)
        return cls(**{key: value[key] for key in supported if key in value}).validated()


@dataclass(frozen=True)
class VirtualCell:
    cell_id: int
    ix: int
    iy: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) * 0.5, (self.y_min + self.y_max) * 0.5)


@dataclass
class Msh2Mesh:
    nodes: Dict[int, Tuple[float, float, float]]
    triangles: Dict[int, Tuple[int, int, int]]


@dataclass
class CalculixRunResult:
    qoi: float
    element_count: int
    node_count: int
    displacements: Dict[int, Tuple[float, float, float]]
    cell_to_elements: Dict[int, list[int]]
    cell_energy: Dict[int, float]
    cell_features: Dict[int, list[float]]
    mesh_signature: Tuple[int, Tuple[Tuple[int, int], ...]]
    workdir: str
    mesh: Msh2Mesh = field(repr=False)


def _default_command_runner(
    command: Sequence[str], cwd: Path, timeout_seconds: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def split_command(command: str | Sequence[str]) -> list[str]:
    if not isinstance(command, str):
        result = [str(item) for item in command]
        if not result:
            raise ValueError("Command sequence cannot be empty")
        return result
    command = command.strip()
    if not command:
        raise ValueError("Command cannot be empty")
    expanded = os.path.expandvars(os.path.expanduser(command.strip('"')))
    if Path(expanded).exists():
        return [expanded]
    return shlex.split(command, posix=os.name != "nt")


def command_available(command: str | Sequence[str]) -> bool:
    executable = split_command(command)[0].strip('"')
    return Path(executable).exists() or shutil.which(executable) is not None


def parse_msh2(filepath: str | Path) -> Msh2Mesh:
    """Parse the ASCII node and first-order triangle blocks of a Gmsh 2.2 file."""

    path = Path(filepath)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nodes: Dict[int, Tuple[float, float, float]] = {}
    triangles: Dict[int, Tuple[int, int, int]] = {}
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
                if element_type == 2:  # three-node triangle
                    if len(connectivity) != 3:
                        raise ValueError("Gmsh triangle does not have three nodes")
                    triangles[element_id] = tuple(connectivity)  # type: ignore[assignment]
            index += count + 2
        else:
            index += 1
    if not nodes:
        raise ValueError(f"No nodes found in {path}")
    if not triangles:
        raise ValueError(f"No first-order triangles found in {path}")
    return Msh2Mesh(nodes=nodes, triangles=triangles)


def parse_frd_displacements(filepath: str | Path) -> Dict[int, Tuple[float, float, float]]:
    """Read the last ASCII ``DISP`` dataset from a CalculiX FRD file."""

    result: Dict[int, Tuple[float, float, float]] = {}
    current: Dict[int, Tuple[float, float, float]] = {}
    in_disp = False
    path = Path(filepath)
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
            # Native CalculiX FRD result records are fixed-width: I3, I10,
            # followed by E12.5 fields.  Adjacent signed values therefore do
            # not necessarily contain whitespace (for example
            # ``9.06566E-05-9.63721E-02``), so a plain ``split`` is unsafe.
            try:
                node_id = int(raw_line[3:13].strip())
                fields = [raw_line[start : start + 12] for start in (13, 25, 37)]
                if any(not field.strip() for field in fields):
                    raise ValueError("incomplete fixed-width FRD record")
                values = [
                    float(field.replace("D", "E").replace("d", "e"))
                    for field in fields
                ]
            except (ValueError, IndexError):
                # Keep compatibility with compact hand-written fixtures and
                # third-party converters that emit whitespace-delimited FRD.
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


def triangle_response(
    coordinates: Sequence[Tuple[float, float]],
    displacements: Sequence[Tuple[float, float]],
    young_modulus: float,
    poisson_ratio: float,
    thickness: float,
) -> Tuple[float, float, np.ndarray]:
    """Return ``(von_mises, strain_energy, strain)`` for one CPE3 triangle."""

    xy = np.asarray(coordinates, dtype=np.float64)
    u = np.asarray(displacements, dtype=np.float64).reshape(6)
    twice_area = float(
        np.linalg.det(
            np.asarray(
                [
                    [1.0, xy[0, 0], xy[0, 1]],
                    [1.0, xy[1, 0], xy[1, 1]],
                    [1.0, xy[2, 0], xy[2, 1]],
                ]
            )
        )
    )
    area = abs(twice_area) * 0.5
    if area <= 1.0e-15:
        raise ValueError("Degenerate triangular element")
    b = np.asarray([xy[1, 1] - xy[2, 1], xy[2, 1] - xy[0, 1], xy[0, 1] - xy[1, 1]])
    c = np.asarray([xy[2, 0] - xy[1, 0], xy[0, 0] - xy[2, 0], xy[1, 0] - xy[0, 0]])
    sign = 1.0 if twice_area > 0 else -1.0
    b *= sign
    c *= sign
    b_matrix = np.asarray(
        [
            [b[0], 0.0, b[1], 0.0, b[2], 0.0],
            [0.0, c[0], 0.0, c[1], 0.0, c[2]],
            [c[0], b[0], c[1], b[1], c[2], b[2]],
        ],
        dtype=np.float64,
    ) / (2.0 * area)
    strain = b_matrix @ u
    factor = young_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    constitutive = factor * np.asarray(
        [
            [1.0 - poisson_ratio, poisson_ratio, 0.0],
            [poisson_ratio, 1.0 - poisson_ratio, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson_ratio)],
        ]
    )
    stress = constitutive @ strain
    sigma_x, sigma_y, tau_xy = (float(stress[0]), float(stress[1]), float(stress[2]))
    lame_lambda = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    sigma_z = lame_lambda * float(strain[0] + strain[1])
    von_mises = math.sqrt(
        max(
            0.0,
            0.5
            * (
                (sigma_x - sigma_y) ** 2
                + (sigma_y - sigma_z) ** 2
                + (sigma_z - sigma_x) ** 2
            )
            + 3.0 * tau_xy**2,
        )
    )
    strain_energy = 0.5 * float(strain @ stress) * area * thickness
    return von_mises, strain_energy, strain


class StateAwareCalculixEnv:
    """State-aware adaptive mesh environment driven by local Gmsh and CalculiX."""

    BASE_CELL_FEATURE_NAMES = (
        "mean_mises",
        "std_mises",
        "max_mises",
        "mean_energy_density",
        "max_energy_density",
        "cell_energy_fraction",
        "relative_cell_energy_error",
        "mean_displacement",
        "max_displacement",
        "displacement_spread",
        "element_count_over_limit",
        "cell_element_fraction",
        "centroid_x",
        "centroid_y",
        "cell_width",
        "cell_height",
        "distance_to_hole",
        "on_fixed_boundary",
        "on_loaded_boundary",
        "has_elements",
    )
    DYNAMIC_CELL_FEATURE_NAMES = (
        "mesh_size_over_global",
        "log_mesh_size_over_global",
        "mesh_size_position_in_bounds",
        "refine_is_valid",
        "coarsen_is_valid",
        "last_action_was_refine",
        "last_action_was_coarsen",
        "steps_since_last_action",
        "last_action_was_ineffective",
    )
    CELL_FEATURE_DIM = len(BASE_CELL_FEATURE_NAMES) + len(DYNAMIC_CELL_FEATURE_NAMES)
    GLOBAL_FEATURE_NAMES = (
        "resource_usage",
        "remaining_budget",
        "load_or_step_fraction",
        "consecutive_failure_fraction",
        "last_reward",
        "relative_qoi_error",
        "accuracy_progress",
        "mean_mesh_size_over_global",
        "std_mesh_size_over_global",
        "min_mesh_size_over_global",
        "max_mesh_size_over_global",
        "last_mesh_was_unchanged",
        "last_transition_rolled_back",
        "goal_accuracy_priority",
        "goal_resource_priority",
        "goal_localization_priority",
        "goal_reserve_budget_fraction",
        "goal_target_relative_error",
    )

    def __init__(
        self,
        plate: Optional[PlateConfig] = None,
        simulations_root: str = "simulations_calculix_v2",
        gmsh_cmd: str | Sequence[str] = "gmsh",
        ccx_cmd: str | Sequence[str] = "ccx",
        global_mesh_size: float = 0.80,
        cell_min_mesh_size: Optional[float] = 0.15,
        cell_max_mesh_size: Optional[float] = 1.60,
        max_elements: int = 20_000,
        min_elements: int = 50,
        refine_step_size: float = 0.20,
        coarsen_step_size: float = 0.20,
        max_consecutive_failures: int = 5,
        solver_timeout_seconds: int = 300,
        command_runner: Optional[CommandRunner] = None,
    ) -> None:
        self.plate = (plate or PlateConfig()).validated()
        self.simulations_root = Path(simulations_root)
        self.gmsh_cmd = split_command(gmsh_cmd)
        self.ccx_cmd = split_command(ccx_cmd)
        self.global_mesh_size = float(global_mesh_size)
        self.cell_min_mesh_size = (
            float(cell_min_mesh_size) if cell_min_mesh_size is not None else None
        )
        self.cell_max_mesh_size = (
            float(cell_max_mesh_size) if cell_max_mesh_size is not None else None
        )
        if self.global_mesh_size <= 0:
            raise ValueError("global_mesh_size must be positive")
        if self.cell_min_mesh_size is not None and self.cell_min_mesh_size <= 0:
            raise ValueError("cell_min_mesh_size must be positive")
        if self.cell_max_mesh_size is not None and self.cell_max_mesh_size <= 0:
            raise ValueError("cell_max_mesh_size must be positive")
        if (
            self.cell_min_mesh_size is not None
            and self.cell_max_mesh_size is not None
            and self.cell_min_mesh_size >= self.cell_max_mesh_size
        ):
            raise ValueError("cell_min_mesh_size must be smaller than cell_max_mesh_size")
        self.max_elements = int(max_elements)
        self.min_elements = int(min_elements)
        self.refine_step_size = float(refine_step_size)
        self.coarsen_step_size = float(coarsen_step_size)
        if not 0.0 < self.refine_step_size < 1.0:
            raise ValueError("refine_step_size must lie in (0, 1)")
        if self.coarsen_step_size <= 0:
            raise ValueError("coarsen_step_size must be positive")
        self._max_consecutive_failures = int(max_consecutive_failures)
        self.solver_timeout_seconds = int(solver_timeout_seconds)
        self._command_runner = command_runner or _default_command_runner

        self.virtual_cells = self._build_virtual_cells()
        self.cell_adjacency = self._build_cell_adjacency()
        self.cell_mesh_density = {
            cell_id: self.global_mesh_size for cell_id in self.virtual_cells
        }
        self.current_goal = GoalCondition().normalized()
        self.baseline_qoi: Optional[float] = None
        self.baseline_cell_energy: Dict[int, float] = {}
        self.initial_qoi: Optional[float] = None
        self.step_index = 0
        self.run_id = "run"
        self._consecutive_failures = 0
        self._last_result: Optional[CalculixRunResult] = None
        self._last_reward = 0.0
        self._last_info: Dict[str, Any] = {}
        self._previous_element_count: Optional[int] = None
        self._last_action_by_cell: Dict[int, int] = {}
        self._last_action_step_by_cell: Dict[int, int] = {}
        self._temporarily_blocked_until: Dict[Tuple[int, int], int] = {}

    @property
    def global_feature_dim(self) -> int:
        return len(self.GLOBAL_FEATURE_NAMES)

    def set_goal(self, goal: GoalCondition) -> None:
        self.current_goal = goal.normalized()

    def preflight(self) -> dict[str, Any]:
        return {
            "gmsh_command": self.gmsh_cmd,
            "gmsh_available": command_available(self.gmsh_cmd),
            "ccx_command": self.ccx_cmd,
            "ccx_available": command_available(self.ccx_cmd),
            "simulation_root": str(self.simulations_root.resolve()),
            "plate": asdict(self.plate),
        }

    def _build_virtual_cells(self) -> Dict[int, VirtualCell]:
        cells: Dict[int, VirtualCell] = {}
        dx = self.plate.length / self.plate.cells_x
        dy = self.plate.height / self.plate.cells_y
        for iy in range(self.plate.cells_y):
            for ix in range(self.plate.cells_x):
                x_min = ix * dx
                x_max = (ix + 1) * dx
                y_min = iy * dy
                y_max = (iy + 1) * dy
                corners = ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max))
                fully_inside_hole = False
                if self.plate.hole_radius > 0:
                    fully_inside_hole = all(
                        math.hypot(x - self.plate.hole_center_x, y - self.plate.hole_center_y)
                        < self.plate.hole_radius
                        for x, y in corners
                    )
                if fully_inside_hole:
                    continue
                cell_id = iy * self.plate.cells_x + ix + 1
                cells[cell_id] = VirtualCell(
                    cell_id=cell_id,
                    ix=ix,
                    iy=iy,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
        return cells

    def _build_cell_adjacency(self) -> Dict[int, list[int]]:
        by_grid = {(cell.ix, cell.iy): cell_id for cell_id, cell in self.virtual_cells.items()}
        adjacency: Dict[int, list[int]] = {}
        for cell_id, cell in self.virtual_cells.items():
            neighbors = []
            for delta_x, delta_y in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = by_grid.get((cell.ix + delta_x, cell.iy + delta_y))
                if neighbor is not None:
                    neighbors.append(neighbor)
            adjacency[cell_id] = sorted(neighbors)
        return adjacency

    def _cell_for_point(self, x: float, y: float) -> Optional[int]:
        ix = min(max(int(math.floor(x / self.plate.length * self.plate.cells_x)), 0), self.plate.cells_x - 1)
        iy = min(max(int(math.floor(y / self.plate.height * self.plate.cells_y)), 0), self.plate.cells_y - 1)
        cell_id = iy * self.plate.cells_x + ix + 1
        return cell_id if cell_id in self.virtual_cells else None

    def _write_geo(self, filepath: Path, mesh_sizes: Mapping[int, float]) -> None:
        p = self.plate
        tolerance = max(p.length, p.height) * 1.0e-7
        lines = [
            'SetFactory("OpenCASCADE");',
            f"Rectangle(1) = {{0, 0, 0, {p.length:.16g}, {p.height:.16g}}};",
        ]
        if p.hole_radius > 0:
            lines.extend(
                [
                    f"Disk(2) = {{{p.hole_center_x:.16g}, {p.hole_center_y:.16g}, 0, {p.hole_radius:.16g}, {p.hole_radius:.16g}}};",
                    "domain[] = BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; };",
                    'Physical Surface("DOMAIN") = {domain[]};',
                ]
            )
        else:
            lines.append('Physical Surface("DOMAIN") = {1};')
        lines.extend(
            [
                f"fixed[] = Curve In BoundingBox{{{-tolerance:.16g}, {-tolerance:.16g}, {-tolerance:.16g}, {tolerance:.16g}, {p.height + tolerance:.16g}, {tolerance:.16g}}};",
                f"loaded[] = Curve In BoundingBox{{{p.length - tolerance:.16g}, {-tolerance:.16g}, {-tolerance:.16g}, {p.length + tolerance:.16g}, {p.height + tolerance:.16g}, {tolerance:.16g}}};",
                'Physical Curve("FIXED") = {fixed[]};',
                'Physical Curve("LOADED") = {loaded[]};',
                "Mesh.MshFileVersion = 2.2;",
                "Mesh.ElementOrder = 1;",
                f"Mesh.Algorithm = {int(p.gmsh_algorithm)};",
                "Mesh.RecombineAll = 0;",
                "Mesh.SaveAll = 1;",
                "Mesh.MeshSizeExtendFromBoundary = 0;",
                "Mesh.MeshSizeFromPoints = 0;",
                "Mesh.MeshSizeFromCurvature = 0;",
            ]
        )
        field_ids = []
        v_out = max(mesh_sizes.values()) * 1.0e4
        for field_id, (cell_id, cell) in enumerate(sorted(self.virtual_cells.items()), start=1):
            field_ids.append(field_id)
            size = float(mesh_sizes[cell_id])
            epsilon_x = (cell.x_max - cell.x_min) * 1.0e-8
            epsilon_y = (cell.y_max - cell.y_min) * 1.0e-8
            lines.extend(
                [
                    f"Field[{field_id}] = Box;",
                    f"Field[{field_id}].VIn = {size:.16g};",
                    f"Field[{field_id}].VOut = {v_out:.16g};",
                    f"Field[{field_id}].XMin = {cell.x_min - epsilon_x:.16g};",
                    f"Field[{field_id}].XMax = {cell.x_max + epsilon_x:.16g};",
                    f"Field[{field_id}].YMin = {cell.y_min - epsilon_y:.16g};",
                    f"Field[{field_id}].YMax = {cell.y_max + epsilon_y:.16g};",
                    "Field[{0}].ZMin = -1;".format(field_id),
                    "Field[{0}].ZMax = 1;".format(field_id),
                ]
            )
        min_field = len(field_ids) + 1
        lines.extend(
            [
                f"Field[{min_field}] = Min;",
                f"Field[{min_field}].FieldsList = {{{', '.join(str(value) for value in field_ids)}}};",
                f"Background Field = {min_field};",
            ]
        )
        if self.cell_min_mesh_size is not None:
            lines.append(f"Mesh.MeshSizeMin = {self.cell_min_mesh_size:.16g};")
        if self.cell_max_mesh_size is not None:
            lines.append(f"Mesh.MeshSizeMax = {self.cell_max_mesh_size:.16g};")
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_calculix_deck(self, filepath: Path, mesh: Msh2Mesh) -> None:
        p = self.plate
        tolerance = max(p.length, p.height) * 1.0e-6
        fixed_nodes = sorted(
            node_id for node_id, (x, _, _) in mesh.nodes.items() if abs(x) <= tolerance
        )
        loaded_nodes = sorted(
            node_id
            for node_id, (x, _, _) in mesh.nodes.items()
            if abs(x - p.length) <= tolerance
        )
        if not fixed_nodes:
            raise RuntimeError("Gmsh mesh has no nodes on the fixed x=0 boundary")
        if not loaded_nodes:
            raise RuntimeError("Gmsh mesh has no nodes on the loaded x=L boundary")

        lines = ["*HEADING", "State-aware Gmsh + CalculiX adaptive mesh benchmark", "*NODE"]
        for node_id, (x, y, z) in sorted(mesh.nodes.items()):
            lines.append(f"{node_id}, {x:.16g}, {y:.16g}, {z:.16g}")
        lines.append("*ELEMENT, TYPE=CPE3, ELSET=EALL")
        for element_id, nodes in sorted(mesh.triangles.items()):
            lines.append(f"{element_id}, {nodes[0]}, {nodes[1]}, {nodes[2]}")
        lines.extend(["*NSET, NSET=NALL"])
        self._append_ids(lines, sorted(mesh.nodes))
        lines.extend(["*NSET, NSET=FIXED"])
        self._append_ids(lines, fixed_nodes)
        lines.extend(["*NSET, NSET=LOADED"])
        self._append_ids(lines, loaded_nodes)
        lines.extend(
            [
                "*MATERIAL, NAME=MAT",
                "*ELASTIC",
                f"{p.young_modulus:.16g}, {p.poisson_ratio:.16g}",
                "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT",
                f"{p.thickness:.16g}",
                "*STEP",
                "*STATIC",
                "0.1, 1.0",
                "*BOUNDARY",
                "FIXED, 1, 2, 0.0",
                "*CLOAD",
            ]
        )
        load_x = p.load_x / len(loaded_nodes)
        load_y = p.load_y / len(loaded_nodes)
        for node_id in loaded_nodes:
            if abs(load_x) > 0:
                lines.append(f"{node_id}, 1, {load_x:.16g}")
            if abs(load_y) > 0:
                lines.append(f"{node_id}, 2, {load_y:.16g}")
        lines.extend(["*NODE FILE", "U", "*END STEP"])
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _append_ids(lines: list[str], values: Sequence[int], width: int = 16) -> None:
        for start in range(0, len(values), width):
            lines.append(", ".join(str(value) for value in values[start : start + width]))

    def _run_checked(self, command: Sequence[str], cwd: Path, label: str) -> None:
        completed = self._command_runner(command, cwd, self.solver_timeout_seconds)
        (cwd / f"{label}.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
        (cwd / f"{label}.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"{label} failed with exit code {completed.returncode}; "
                f"see {label}.stdout.log and {label}.stderr.log in {cwd}"
            )

    def _run_analysis(self, workdir: Path, mesh_sizes: Mapping[int, float]) -> CalculixRunResult:
        workdir.mkdir(parents=True, exist_ok=True)
        geo_path = workdir / "model.geo"
        msh_path = workdir / "mesh.msh"
        deck_path = workdir / "model.inp"
        self._write_geo(geo_path, mesh_sizes)
        self._run_checked(
            [*self.gmsh_cmd, "-2", geo_path.name, "-format", "msh2", "-o", msh_path.name],
            workdir,
            "gmsh",
        )
        mesh = parse_msh2(msh_path)
        self._write_calculix_deck(deck_path, mesh)
        self._run_checked([*self.ccx_cmd, "-i", "model"], workdir, "ccx")
        frd_path = workdir / "model.frd"
        if not frd_path.exists():
            raise RuntimeError(f"CalculiX did not create {frd_path}")
        displacements = parse_frd_displacements(frd_path)
        return self._postprocess(workdir, mesh, displacements)

    def _postprocess(
        self,
        workdir: Path,
        mesh: Msh2Mesh,
        displacements: Mapping[int, Tuple[float, float, float]],
    ) -> CalculixRunResult:
        cell_to_elements = {cell_id: [] for cell_id in self.virtual_cells}
        element_mises: Dict[int, float] = {}
        element_energy: Dict[int, float] = {}
        element_disp: Dict[int, list[float]] = {}
        element_cell: Dict[int, int] = {}
        total_energy = 0.0
        p = self.plate

        for element_id, connectivity in mesh.triangles.items():
            coordinates_3d = [mesh.nodes[node_id] for node_id in connectivity]
            centroid_x = sum(value[0] for value in coordinates_3d) / 3.0
            centroid_y = sum(value[1] for value in coordinates_3d) / 3.0
            cell_id = self._cell_for_point(centroid_x, centroid_y)
            if cell_id is None:
                continue
            if any(node_id not in displacements for node_id in connectivity):
                raise RuntimeError(f"FRD result is missing displacement for element {element_id}")
            xy = [(value[0], value[1]) for value in coordinates_3d]
            uv = [
                (displacements[node_id][0], displacements[node_id][1])
                for node_id in connectivity
            ]
            mises, energy, _ = triangle_response(
                xy, uv, p.young_modulus, p.poisson_ratio, p.thickness
            )
            magnitudes = [math.hypot(value[0], value[1]) for value in uv]
            cell_to_elements[cell_id].append(element_id)
            element_cell[element_id] = cell_id
            element_mises[element_id] = mises
            element_energy[element_id] = energy
            element_disp[element_id] = magnitudes
            total_energy += energy

        # The linear static strain energy is also 1/2 F^T U.  The element sum is
        # used because it naturally yields cell-wise contributions.
        qoi = float(total_energy)
        cell_energy = {
            cell_id: float(sum(element_energy.get(element_id, 0.0) for element_id in ids))
            for cell_id, ids in cell_to_elements.items()
        }
        cell_features = self._build_base_cell_features(
            mesh=mesh,
            cell_to_elements=cell_to_elements,
            cell_energy=cell_energy,
            element_mises=element_mises,
            element_energy=element_energy,
            element_disp=element_disp,
            total_energy=total_energy,
        )
        signature = (
            len(mesh.triangles),
            tuple(sorted((cell_id, len(ids)) for cell_id, ids in cell_to_elements.items())),
        )
        return CalculixRunResult(
            qoi=qoi,
            element_count=len(mesh.triangles),
            node_count=len(mesh.nodes),
            displacements=dict(displacements),
            cell_to_elements=cell_to_elements,
            cell_energy=cell_energy,
            cell_features=cell_features,
            mesh_signature=signature,
            workdir=str(workdir),
            mesh=mesh,
        )

    def _build_base_cell_features(
        self,
        mesh: Msh2Mesh,
        cell_to_elements: Mapping[int, Sequence[int]],
        cell_energy: Mapping[int, float],
        element_mises: Mapping[int, float],
        element_energy: Mapping[int, float],
        element_disp: Mapping[int, Sequence[float]],
        total_energy: float,
    ) -> Dict[int, list[float]]:
        p = self.plate
        stress_scale = max(
            math.hypot(p.load_x, p.load_y) / max(p.height * p.thickness, 1.0e-12),
            1.0e-12,
        )
        energy_density_scale = max(stress_scale**2 / p.young_modulus, 1.0e-12)
        displacement_scale = max(
            math.hypot(p.load_x, p.load_y)
            * p.length
            / max(p.young_modulus * p.height * p.thickness, 1.0e-12),
            1.0e-12,
        )
        all_elements = max(sum(len(ids) for ids in cell_to_elements.values()), 1)
        features: Dict[int, list[float]] = {}

        for cell_id, cell in self.virtual_cells.items():
            ids = list(cell_to_elements.get(cell_id, []))
            mises_values = np.asarray([element_mises[element_id] for element_id in ids], dtype=np.float64)
            energy_values = np.asarray([element_energy[element_id] for element_id in ids], dtype=np.float64)
            disp_values = np.asarray(
                [value for element_id in ids for value in element_disp[element_id]], dtype=np.float64
            )
            if mises_values.size == 0:
                mises_values = np.asarray([0.0])
                energy_values = np.asarray([0.0])
                disp_values = np.asarray([0.0])
            element_area = (cell.x_max - cell.x_min) * (cell.y_max - cell.y_min) / max(len(ids), 1)
            energy_density = energy_values / max(element_area * p.thickness, 1.0e-12)
            center_x, center_y = cell.center
            hole_distance = 1.0
            if p.hole_radius > 0:
                hole_distance = max(
                    0.0,
                    math.hypot(center_x - p.hole_center_x, center_y - p.hole_center_y)
                    - p.hole_radius,
                ) / math.hypot(p.length, p.height)
            baseline_energy = self.baseline_cell_energy.get(cell_id)
            relative_cell_error = 0.0
            if baseline_energy is not None:
                raw = abs(float(cell_energy.get(cell_id, 0.0)) - baseline_energy) / (
                    abs(baseline_energy) + 1.0e-12
                )
                relative_cell_error = float(np.tanh(np.log1p(raw)))
            scale_log = lambda value, scale: float(np.tanh(np.log1p(abs(float(value)) / scale)))
            row = [
                scale_log(mises_values.mean(), stress_scale),
                scale_log(mises_values.std(), stress_scale),
                scale_log(mises_values.max(), stress_scale),
                scale_log(energy_density.mean(), energy_density_scale),
                scale_log(energy_density.max(), energy_density_scale),
                float(cell_energy.get(cell_id, 0.0)) / (abs(total_energy) + 1.0e-12),
                relative_cell_error,
                scale_log(disp_values.mean(), displacement_scale),
                scale_log(disp_values.max(), displacement_scale),
                scale_log(disp_values.max() - disp_values.min(), displacement_scale),
                len(ids) / max(float(self.max_elements), 1.0),
                len(ids) / float(all_elements),
                center_x / p.length,
                center_y / p.height,
                (cell.x_max - cell.x_min) / p.length,
                (cell.y_max - cell.y_min) / p.height,
                hole_distance,
                float(cell.ix == 0),
                float(cell.ix == p.cells_x - 1),
                float(bool(ids)),
            ]
            if len(row) != len(self.BASE_CELL_FEATURE_NAMES):
                raise RuntimeError("CalculiX base feature schema length changed unexpectedly")
            features[cell_id] = row
        return features

    def _baseline_cache_key(self, baseline_mesh_size: float) -> str:
        payload = {
            "plate": asdict(self.plate),
            "baseline_mesh_size": float(baseline_mesh_size),
            "schema": 1,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    def compute_baseline(
        self,
        cache_dir: str = "checkpoints_calculix_v2/baseline_cache",
        use_cache: bool = True,
        baseline_mesh_size: float = 0.25,
    ) -> Optional[float]:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = cache_root / f"calculix_{self._baseline_cache_key(baseline_mesh_size)}.json"
        if use_cache and cache_file.exists():
            value = json.loads(cache_file.read_text(encoding="utf-8"))
            self.baseline_qoi = float(value["baseline_qoi"])
            self.baseline_cell_energy = {
                int(key): float(item) for key, item in value.get("baseline_cell_energy", {}).items()
            }
            return self.baseline_qoi

        saved_sizes = dict(self.cell_mesh_density)
        try:
            baseline_sizes = {cell_id: float(baseline_mesh_size) for cell_id in self.virtual_cells}
            result = self._run_analysis(
                self.simulations_root / "_baseline" / self._baseline_cache_key(baseline_mesh_size),
                baseline_sizes,
            )
            self.baseline_qoi = result.qoi
            self.baseline_cell_energy = dict(result.cell_energy)
            cache_file.write_text(
                json.dumps(
                    {
                        "baseline_qoi": self.baseline_qoi,
                        "baseline_cell_energy": self.baseline_cell_energy,
                        "plate": asdict(self.plate),
                        "baseline_mesh_size": float(baseline_mesh_size),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return self.baseline_qoi
        finally:
            self.cell_mesh_density = saved_sizes

    def reset(self, run_id: Optional[str] = None) -> dict[str, Any]:
        self.run_id = run_id or "run"
        self.step_index = 0
        self._consecutive_failures = 0
        self._last_action_by_cell.clear()
        self._last_action_step_by_cell.clear()
        self._temporarily_blocked_until.clear()
        self.cell_mesh_density = {
            cell_id: self.global_mesh_size for cell_id in self.virtual_cells
        }
        result = self._run_analysis(
            self.simulations_root / self.run_id / "step_0000",
            self.cell_mesh_density,
        )
        self._last_result = result
        self.initial_qoi = result.qoi
        self._previous_element_count = result.element_count
        self._last_reward = 0.0
        self._last_info = self._result_info(result)
        return {
            "last_reward": 0.0,
            "cell_features": result.cell_features,
            "resource_usage": self._extract_resource_usage(),
            "global_features": {"qoi": result.qoi},
        }

    def _result_info(self, result: CalculixRunResult) -> Dict[str, Any]:
        return {
            "backend": "calculix",
            "qoi": result.qoi,
            "allse": result.qoi,
            "cell_strain_energy": result.cell_energy,
            "cell_features": result.cell_features,
            "total_elements": result.element_count,
            "num_nodes": result.node_count,
            "resource_usage": result.element_count / max(float(self.max_elements), 1.0),
            "workdir": result.workdir,
        }

    def _physical_action_mask(
        self, cell_ids: Iterable[int], goal: GoalCondition
    ) -> Dict[int, list[bool]]:
        resource_usage = self._extract_resource_usage()
        reserve_limit = 1.0 - goal.reserve_budget_fraction
        has_elements = (
            self._last_result.cell_to_elements if self._last_result is not None else {}
        )
        masks: Dict[int, list[bool]] = {}
        for raw_cell_id in cell_ids:
            cell_id = int(raw_cell_id)
            current = float(self.cell_mesh_density[cell_id])
            refined = current * (1.0 - self.refine_step_size)
            coarsened = current * (1.0 + self.coarsen_step_size)
            active = bool(has_elements.get(cell_id, []))
            refine_valid = active and resource_usage < reserve_limit - 1.0e-9
            if self.cell_min_mesh_size is not None:
                refine_valid = refine_valid and refined >= self.cell_min_mesh_size - 1.0e-9
            coarsen_valid = active
            if self.cell_max_mesh_size is not None:
                coarsen_valid = coarsen_valid and coarsened <= self.cell_max_mesh_size + 1.0e-9
            masks[cell_id] = [bool(refine_valid), bool(coarsen_valid)]
        return masks

    def get_action_mask(
        self,
        cell_ids: Optional[Iterable[int]] = None,
        goal: Optional[GoalCondition] = None,
    ) -> Dict[int, list[bool]]:
        goal = (goal or self.current_goal).normalized()
        ids = tuple(int(value) for value in (cell_ids or sorted(self.virtual_cells)))
        physical = self._physical_action_mask(ids, goal)
        masked = {cell_id: list(row) for cell_id, row in physical.items()}
        for cell_id in ids:
            for action in (REFINE, COARSEN):
                blocked_until = self._temporarily_blocked_until.get((cell_id, action))
                if blocked_until is not None and self.step_index <= blocked_until:
                    masked[cell_id][action] = False
        return masked if any(any(row) for row in masked.values()) else physical

    def _relative_cell_energy_error(self, cell_id: int) -> float:
        if self._last_result is None or cell_id not in self.baseline_cell_energy:
            return 0.0
        current = self._last_result.cell_energy.get(cell_id, 0.0)
        baseline = self.baseline_cell_energy[cell_id]
        raw = abs(current - baseline) / (abs(baseline) + 1.0e-12)
        return float(np.tanh(np.log1p(raw)))

    def get_augmented_cell_observations(
        self, goal: Optional[GoalCondition] = None
    ) -> Dict[int, Dict[str, Any]]:
        if self._last_result is None:
            return {}
        goal = (goal or self.current_goal).normalized()
        masks = self.get_action_mask(self.virtual_cells, goal)
        lower = self.cell_min_mesh_size or 0.0
        upper = self.cell_max_mesh_size or max(self.global_mesh_size * 2.0, lower + 1.0)
        span = max(upper - lower, 1.0e-12)
        observations: Dict[int, Dict[str, Any]] = {}
        for cell_id in sorted(self.virtual_cells):
            base = list(self._last_result.cell_features.get(cell_id, []))
            if len(base) != len(self.BASE_CELL_FEATURE_NAMES):
                raise RuntimeError(f"Unexpected CalculiX cell feature length for cell {cell_id}")
            mesh_size = self.cell_mesh_density[cell_id]
            ratio = mesh_size / self.global_mesh_size
            last_action = self._last_action_by_cell.get(cell_id)
            last_step = self._last_action_step_by_cell.get(cell_id)
            recency = 1.0 if last_step is None else min(max((self.step_index - last_step) / 20.0, 0.0), 1.0)
            ineffective = any(
                self._temporarily_blocked_until.get((cell_id, action), -1) >= self.step_index
                for action in (REFINE, COARSEN)
            )
            dynamic = [
                ratio,
                float(np.tanh(math.log(max(ratio, 1.0e-12)))),
                float(np.clip((mesh_size - lower) / span, 0.0, 1.0)),
                float(masks[cell_id][REFINE]),
                float(masks[cell_id][COARSEN]),
                float(last_action == REFINE),
                float(last_action == COARSEN),
                float(recency),
                float(ineffective),
            ]
            observations[cell_id] = {"self": base + dynamic, "neighbors": []}
        return observations

    def relative_qoi_error(self) -> Optional[float]:
        if self._last_result is None or self.baseline_qoi is None:
            return None
        return abs(self._last_result.qoi - self.baseline_qoi) / (
            abs(self.baseline_qoi) + 1.0e-12
        )

    def _extract_resource_usage(self, info: Optional[Mapping[str, Any]] = None) -> float:
        if info is not None and info.get("resource_usage") is not None:
            return float(info["resource_usage"])
        if self._last_result is None:
            return 0.0
        return self._last_result.element_count / max(float(self.max_elements), 1.0)

    def get_global_feature_vector(
        self, goal: Optional[GoalCondition] = None, max_steps: int = 100
    ) -> list[float]:
        goal = (goal or self.current_goal).normalized()
        resource_usage = self._extract_resource_usage()
        relative_error = self.relative_qoi_error() or 0.0
        transformed_error = float(np.tanh(np.log1p(relative_error)))
        accuracy_progress = 0.0
        if (
            self._last_result is not None
            and self.initial_qoi is not None
            and self.baseline_qoi is not None
        ):
            initial_error = abs(self.initial_qoi - self.baseline_qoi)
            current_error = abs(self._last_result.qoi - self.baseline_qoi)
            if initial_error > 1.0e-12:
                accuracy_progress = float(np.clip(1.0 - current_error / initial_error, -1.0, 1.0))
        ratios = np.asarray(
            [value / self.global_mesh_size for value in self.cell_mesh_density.values()],
            dtype=np.float64,
        )
        features = [
            float(np.clip(resource_usage, 0.0, 2.0)),
            max(0.0, 1.0 - resource_usage),
            min(self.step_index / max(float(max_steps), 1.0), 1.0),
            min(self._consecutive_failures / max(float(self._max_consecutive_failures), 1.0), 1.0),
            float(np.tanh(self._last_reward / 10.0)),
            transformed_error,
            accuracy_progress,
            float(ratios.mean()),
            float(ratios.std()),
            float(ratios.min()),
            float(ratios.max()),
            float(bool(self._last_info.get("mesh_unchanged", False))),
            float(bool(self._last_info.get("state_rollback", False))),
            goal.accuracy_priority,
            goal.resource_priority,
            goal.localization_priority,
            goal.reserve_budget_fraction,
            float(np.tanh(np.log1p(goal.target_relative_error))),
        ]
        if len(features) != self.global_feature_dim:
            raise RuntimeError("CalculiX global feature schema length changed unexpectedly")
        return features

    def build_state(
        self, goal: Optional[GoalCondition] = None, max_steps: int = 100
    ) -> GraphState:
        goal = (goal or self.current_goal).normalized()
        observations = self.get_augmented_cell_observations(goal)
        action_mask = self.get_action_mask(observations, goal)
        return build_graph_state(
            cell_observations=observations,
            cell_adjacency=self.cell_adjacency,
            global_features=self.get_global_feature_vector(goal, max_steps),
            action_mask=action_mask,
            num_actions=2,
        )

    def step(self, action_params: Mapping[int, int]):
        self.step_index += 1
        backup_sizes = dict(self.cell_mesh_density)
        previous_result = self._last_result
        previous_relative_error = self.relative_qoi_error()
        previous_local_errors = {
            int(cell_id): self._relative_cell_energy_error(int(cell_id))
            for cell_id in action_params
        }
        invalid = []
        physical_mask = self._physical_action_mask(action_params.keys(), self.current_goal)
        for raw_cell_id, raw_action in action_params.items():
            cell_id = int(raw_cell_id)
            action = int(raw_action)
            if cell_id not in self.cell_mesh_density or action not in (REFINE, COARSEN):
                invalid.append((cell_id, action))
                continue
            if not physical_mask.get(cell_id, [False, False])[action]:
                invalid.append((cell_id, action))
                continue
            current = self.cell_mesh_density[cell_id]
            self.cell_mesh_density[cell_id] = (
                current * (1.0 - self.refine_step_size)
                if action == REFINE
                else current * (1.0 + self.coarsen_step_size)
            )
        if invalid:
            self.cell_mesh_density = backup_sizes
            return self._failure_transition(
                action_params, "invalid_action", -2.0, {"invalid_actions": invalid}
            )

        try:
            result = self._run_analysis(
                self.simulations_root / self.run_id / f"step_{self.step_index:04d}",
                self.cell_mesh_density,
            )
            if result.element_count >= self.max_elements:
                raise RuntimeError(
                    f"Element count {result.element_count} reached max_elements={self.max_elements}"
                )
            if result.element_count < self.min_elements:
                raise RuntimeError(
                    f"Element count {result.element_count} is below min_elements={self.min_elements}"
                )
        except Exception as exc:
            self.cell_mesh_density = backup_sizes
            return self._failure_transition(
                action_params, "solver_or_mesh_failure", -5.0, {"exception": str(exc)}
            )

        mesh_unchanged = bool(
            previous_result is not None and result.mesh_signature == previous_result.mesh_signature
        )
        self._last_result = result
        new_relative_error = self.relative_qoi_error()
        accuracy_improvement = 0.0
        if previous_relative_error is not None and new_relative_error is not None:
            accuracy_improvement = previous_relative_error - new_relative_error
        accuracy_component = self.current_goal.accuracy_priority * float(
            np.tanh(10.0 * accuracy_improvement)
        )
        previous_elements = (
            previous_result.element_count if previous_result is not None else result.element_count
        )
        resource_delta = max(0, result.element_count - previous_elements) / max(float(self.max_elements), 1.0)
        resource_component = -self.current_goal.resource_priority * 5.0 * resource_delta
        localization_improvement = 0.0
        compared = 0
        for raw_cell_id in action_params:
            cell_id = int(raw_cell_id)
            localization_improvement += previous_local_errors.get(cell_id, 0.0) - self._relative_cell_energy_error(cell_id)
            compared += 1
        if compared:
            localization_improvement /= compared
        localization_component = self.current_goal.localization_priority * localization_improvement
        ineffective_penalty = -0.25 if mesh_unchanged else 0.0
        reward = float(
            accuracy_component
            + resource_component
            + localization_component
            + ineffective_penalty
        )
        self._last_reward = reward
        self._previous_element_count = result.element_count
        self._consecutive_failures = 0
        info = self._result_info(result)
        info.update(
            {
                "reward_components": {
                    "accuracy_improvement": accuracy_improvement,
                    "accuracy_component": accuracy_component,
                    "resource_delta_elements": result.element_count - previous_elements,
                    "resource_component": resource_component,
                    "localization_improvement": localization_improvement,
                    "localization_component": localization_component,
                    "ineffective_penalty": ineffective_penalty,
                    "goal_condition": self.current_goal.to_dict(),
                },
                "mesh_unchanged": mesh_unchanged,
                "state_rollback": False,
                "cell_rewards": {int(cell_id): reward for cell_id in action_params},
            }
        )
        for raw_cell_id, raw_action in action_params.items():
            cell_id = int(raw_cell_id)
            action = int(raw_action)
            self._last_action_by_cell[cell_id] = action
            self._last_action_step_by_cell[cell_id] = self.step_index
            if mesh_unchanged:
                self._temporarily_blocked_until[(cell_id, action)] = self.step_index
        self._last_info = info
        obs = {
            "last_reward": reward,
            "cell_features": result.cell_features,
            "resource_usage": self._extract_resource_usage(info),
            "global_features": {"qoi": result.qoi},
        }
        return obs, reward, False, info

    def _failure_transition(
        self,
        action_params: Mapping[int, int],
        penalty_type: str,
        reward: float,
        details: Mapping[str, Any],
    ):
        self._consecutive_failures += 1
        for raw_cell_id, raw_action in action_params.items():
            cell_id = int(raw_cell_id)
            action = int(raw_action)
            self._last_action_by_cell[cell_id] = action
            self._last_action_step_by_cell[cell_id] = self.step_index
            self._temporarily_blocked_until[(cell_id, action)] = self.step_index
        self._last_reward = float(reward)
        info = {
            "backend": "calculix",
            "penalty_type": penalty_type,
            "penalty_value": float(reward),
            "state_rollback": True,
            "mesh_unchanged": False,
            "consecutive_failures": self._consecutive_failures,
            **dict(details),
        }
        if self._last_result is not None:
            info.update(self._result_info(self._last_result))
        self._last_info = info
        done = self._consecutive_failures >= self._max_consecutive_failures
        obs = {
            "last_reward": float(reward),
            "cell_features": (
                self._last_result.cell_features if self._last_result is not None else {}
            ),
            "resource_usage": self._extract_resource_usage(info),
            "global_features": {},
        }
        return obs, float(reward), done, info
