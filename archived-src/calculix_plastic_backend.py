"""Incremental elastoplastic Gmsh + CalculiX backend for mesh-control DQN.

The environment is deliberately different from the original static elastic demo:

* a thin plate with a circular hole is modeled with ``CPS3`` plane-stress elements;
* J2 incremental plasticity with isotropic hardening is solved by CalculiX;
* the right edge is displacement-controlled and the left edge is minimally
  constrained, so the reaction-displacement path is a meaningful global target;
* an episode follows the loading path.  At every decision the agent changes one
  virtual-patch mesh size and the load advances to the next increment;
* after remeshing the complete monotonic history is recomputed on the new mesh.
  This is more expensive than state transfer, but avoids corrupting plastic
  history variables while the method is being validated;
* fine-mesh references are used only by the training reward and evaluation.
  They are never included in the policy state, which prevents oracle leakage.

The graph nodes are fixed *control patches*, not finite elements.  This provides a
stable action space while the underlying Gmsh mesh can change at every step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from calculix_backend import (
    Msh2Mesh,
    VirtualCell,
    command_available,
    parse_frd_displacements,
    parse_msh2,
    split_command,
)
from mesh_goal import GoalCondition
from state_aware_dqn_agent import COARSEN, KEEP, REFINE, GraphState, build_graph_state


CommandRunner = Callable[[Sequence[str], Path, int], subprocess.CompletedProcess[str]]


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


@dataclass(frozen=True)
class PlasticPlateConfig:
    """Geometry, material and loading contract for the plastic benchmark.

    Units are intentionally user-consistent.  The supplied example uses mm, N
    and MPa.
    """

    length: float = 10.0
    height: float = 4.0
    thickness: float = 1.0
    young_modulus: float = 210_000.0
    poisson_ratio: float = 0.30
    yield_stress: float = 250.0
    hardening_modulus: float = 1_500.0
    plastic_curve_end_strain: float = 0.12
    target_displacement_x: float = 0.05
    load_steps: int = 20
    plastic_threshold: float = 1.0e-5
    hole_center_x: float = 5.0
    hole_center_y: float = 2.0
    hole_radius: float = 0.75
    cells_x: int = 8
    cells_y: int = 4
    gmsh_algorithm: int = 6
    mesh_transition_fraction: float = 0.20
    max_neighbor_size_ratio: float = 2.25
    solver_initial_increment: float = 0.05
    solver_min_increment: float = 1.0e-6
    solver_max_increment: float = 0.10

    def validated(self) -> "PlasticPlateConfig":
        if self.length <= 0 or self.height <= 0 or self.thickness <= 0:
            raise ValueError("Plate dimensions and thickness must be positive")
        if self.young_modulus <= 0:
            raise ValueError("young_modulus must be positive")
        if not -0.49 < self.poisson_ratio < 0.49:
            raise ValueError("poisson_ratio must lie between -0.49 and 0.49")
        if self.yield_stress <= 0:
            raise ValueError("yield_stress must be positive")
        if self.hardening_modulus < 0:
            raise ValueError("hardening_modulus cannot be negative")
        if self.plastic_curve_end_strain <= 0:
            raise ValueError("plastic_curve_end_strain must be positive")
        if self.target_displacement_x <= 0:
            raise ValueError("target_displacement_x must be positive")
        if self.load_steps < 3:
            raise ValueError("load_steps must be at least three")
        if self.cells_x < 1 or self.cells_y < 1:
            raise ValueError("cells_x and cells_y must be positive")
        if self.hole_radius < 0:
            raise ValueError("hole_radius cannot be negative")
        if self.hole_radius > 0:
            margin_x = min(self.hole_center_x, self.length - self.hole_center_x)
            margin_y = min(self.hole_center_y, self.height - self.hole_center_y)
            if self.hole_radius >= min(margin_x, margin_y):
                raise ValueError("The circular hole must lie strictly inside the plate")
        if not 0.0 <= self.mesh_transition_fraction <= 1.0:
            raise ValueError("mesh_transition_fraction must lie in [0, 1]")
        if self.max_neighbor_size_ratio <= 1.0:
            raise ValueError("max_neighbor_size_ratio must be greater than one")
        if not 0 < self.solver_initial_increment <= 1.0:
            raise ValueError("solver_initial_increment must lie in (0, 1]")
        if not 0 < self.solver_min_increment <= self.solver_initial_increment:
            raise ValueError("solver_min_increment must be positive and no larger than initial")
        if not self.solver_initial_increment <= self.solver_max_increment <= 1.0:
            raise ValueError("solver_max_increment must be between initial increment and one")
        return self

    @classmethod
    def from_json(cls, filepath: str | Path) -> "PlasticPlateConfig":
        with open(filepath, "r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise ValueError("Plastic plate config JSON must contain one object")
        supported = set(cls.__dataclass_fields__)
        return cls(**{key: value[key] for key in supported if key in value}).validated()


@dataclass(frozen=True)
class PlasticDatFrame:
    time: float
    reaction_forces: Dict[int, Tuple[float, float, float]]
    stresses: Dict[int, Tuple[float, float, float, float, float, float]]
    peeq: Dict[int, float]

    @property
    def total_reaction(self) -> Tuple[float, float, float]:
        if not self.reaction_forces:
            return (0.0, 0.0, 0.0)
        values = np.asarray(list(self.reaction_forces.values()), dtype=np.float64)
        total = values.sum(axis=0)
        return (float(total[0]), float(total[1]), float(total[2]))


@dataclass
class PlasticRunResult:
    qoi: float
    reaction_force_x: float
    reaction_force_y: float
    load_step: int
    load_fraction: float
    prescribed_displacement: float
    element_count: int
    node_count: int
    displacements: Dict[int, Tuple[float, float, float]]
    element_mises: Dict[int, float]
    element_peeq: Dict[int, float]
    cell_to_elements: Dict[int, list[int]]
    cell_mean_peeq: Dict[int, float]
    cell_max_peeq: Dict[int, float]
    cell_plastic_work: Dict[int, float]
    cell_features: Dict[int, list[float]]
    total_plastic_work: float
    plastic_zone_fraction: float
    max_peeq: float
    mesh_signature: Tuple[int, int, str]
    workdir: str
    mesh: Msh2Mesh = field(repr=False)


@dataclass(frozen=True)
class PlasticReferenceFrame:
    load_step: int
    load_fraction: float
    reaction_force_x: float
    total_plastic_work: float
    plastic_zone_fraction: float
    max_peeq: float
    cell_mean_peeq: Dict[int, float]
    cell_max_peeq: Dict[int, float]

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["cell_mean_peeq"] = {str(k): float(v) for k, v in self.cell_mean_peeq.items()}
        value["cell_max_peeq"] = {str(k): float(v) for k, v in self.cell_max_peeq.items()}
        return value

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "PlasticReferenceFrame":
        return cls(
            load_step=int(value["load_step"]),
            load_fraction=float(value["load_fraction"]),
            reaction_force_x=float(value["reaction_force_x"]),
            total_plastic_work=float(value["total_plastic_work"]),
            plastic_zone_fraction=float(value["plastic_zone_fraction"]),
            max_peeq=float(value["max_peeq"]),
            cell_mean_peeq={int(k): float(v) for k, v in value["cell_mean_peeq"].items()},
            cell_max_peeq={int(k): float(v) for k, v in value["cell_max_peeq"].items()},
        )


_HEADER_TIME = re.compile(r"\btime\s+([-+0-9.EeDd]+)\s*$")


def _parse_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _read_numeric_block(lines: Sequence[str], start: int, min_columns: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and not lines[index].strip():
        index += 1
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            break
        parts = stripped.split()
        if len(parts) < min_columns:
            break
        try:
            int(parts[0])
        except ValueError:
            break
        rows.append(parts)
        index += 1
    return rows, index


def parse_calculix_plastic_dat(filepath: str | Path) -> PlasticDatFrame:
    """Parse the last reaction, integration-point stress and PEEQ blocks.

    ``*NODE PRINT`` and ``*EL PRINT`` produce plain ASCII blocks in ``.dat``.
    Integration-point values are averaged per original two-dimensional element.
    """

    path = Path(filepath)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    reactions: Dict[int, Tuple[float, float, float]] = {}
    stress_samples: Dict[int, list[np.ndarray]] = {}
    peeq_samples: Dict[int, list[float]] = {}
    last_time = 0.0
    index = 0
    while index < len(lines):
        stripped = lines[index].strip().lower()
        time_match = _HEADER_TIME.search(lines[index])
        block_time = _parse_float(time_match.group(1)) if time_match else last_time
        if stripped.startswith("forces (fx,fy,fz) for set"):
            rows, index = _read_numeric_block(lines, index + 1, 4)
            reactions = {
                int(row[0]): (_parse_float(row[1]), _parse_float(row[2]), _parse_float(row[3]))
                for row in rows
            }
            last_time = block_time
            continue
        if stripped.startswith("stresses (elem, integ.pnt."):
            rows, index = _read_numeric_block(lines, index + 1, 8)
            current: Dict[int, list[np.ndarray]] = {}
            for row in rows:
                element_id = int(row[0])
                current.setdefault(element_id, []).append(
                    np.asarray([_parse_float(value) for value in row[2:8]], dtype=np.float64)
                )
            stress_samples = current
            last_time = block_time
            continue
        if stripped.startswith("equivalent plastic strain"):
            rows, index = _read_numeric_block(lines, index + 1, 3)
            current_peeq: Dict[int, list[float]] = {}
            for row in rows:
                current_peeq.setdefault(int(row[0]), []).append(_parse_float(row[2]))
            peeq_samples = current_peeq
            last_time = block_time
            continue
        index += 1

    if not reactions:
        raise ValueError(f"No reaction-force block found in {path}")
    if not stress_samples:
        raise ValueError(f"No stress block found in {path}")
    if not peeq_samples:
        raise ValueError(f"No equivalent-plastic-strain block found in {path}")

    stresses = {
        element_id: tuple(float(value) for value in np.mean(samples, axis=0))
        for element_id, samples in stress_samples.items()
    }
    peeq = {
        element_id: float(np.mean(samples)) for element_id, samples in peeq_samples.items()
    }
    return PlasticDatFrame(
        time=float(last_time),
        reaction_forces=reactions,
        stresses=stresses,
        peeq=peeq,
    )


def _triangle_area(coordinates: Sequence[Tuple[float, float, float]]) -> float:
    (x1, y1, _), (x2, y2, _), (x3, y3, _) = coordinates
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def _von_mises_3d(stress: Sequence[float]) -> float:
    sxx, syy, szz, sxy, sxz, syz = (float(value) for value in stress)
    value = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
    value += 3.0 * (sxy**2 + sxz**2 + syz**2)
    return math.sqrt(max(0.0, value))


class StateAwareCalculixPlasticEnv:
    """Patch-level adaptive mesh environment along an elastoplastic load path."""

    BASE_CELL_FEATURE_NAMES = (
        "mean_mises_over_yield",
        "std_mises_over_yield",
        "max_mises_over_yield",
        "mean_peeq_scaled",
        "std_peeq_scaled",
        "max_peeq_scaled",
        "plastic_element_fraction",
        "plastic_work_fraction",
        "mean_displacement_over_target",
        "max_displacement_over_target",
        "displacement_spread_over_target",
        "element_count_over_limit",
        "cell_element_fraction",
        "centroid_x",
        "centroid_y",
        "cell_width",
        "cell_height",
        "distance_to_hole",
        "on_left_boundary",
        "on_right_boundary",
        "has_elements",
        "mean_yield_margin",
        "max_yield_margin",
        "mean_element_area_over_cell_area",
    )
    DYNAMIC_CELL_FEATURE_NAMES = (
        "mesh_size_over_global",
        "log_mesh_size_over_global",
        "mesh_size_position_in_bounds",
        "refine_is_valid",
        "coarsen_is_valid",
        "keep_is_valid",
        "last_action_was_refine",
        "last_action_was_coarsen",
        "last_action_was_keep",
        "steps_since_last_action",
        "last_action_was_ineffective",
        "mean_neighbor_log_size_jump",
    )
    CELL_FEATURE_DIM = len(BASE_CELL_FEATURE_NAMES) + len(DYNAMIC_CELL_FEATURE_NAMES)
    GLOBAL_FEATURE_NAMES = (
        "resource_usage",
        "remaining_budget",
        "load_fraction",
        "consecutive_failure_fraction",
        "last_reward",
        "reaction_over_nominal_yield",
        "tangent_stiffness_over_elastic",
        "plastic_zone_fraction",
        "max_peeq_scaled",
        "plastic_work_scaled",
        "mean_mesh_size_over_global",
        "std_mesh_size_over_global",
        "min_mesh_size_over_global",
        "max_mesh_size_over_global",
        "mean_neighbor_log_size_jump",
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
        plate: Optional[PlasticPlateConfig] = None,
        simulations_root: str = "simulations_calculix_plastic_v1",
        gmsh_cmd: str | Sequence[str] = "gmsh",
        ccx_cmd: str | Sequence[str] = "ccx",
        global_mesh_size: float = 0.80,
        cell_min_mesh_size: Optional[float] = 0.15,
        cell_max_mesh_size: Optional[float] = 1.60,
        max_elements: int = 8_000,
        min_elements: int = 50,
        refine_step_size: float = 0.20,
        coarsen_step_size: float = 0.20,
        max_consecutive_failures: int = 5,
        solver_timeout_seconds: int = 300,
        command_runner: Optional[CommandRunner] = None,
    ) -> None:
        self.plate = (plate or PlasticPlateConfig()).validated()
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
        if not 0 < self.refine_step_size < 1:
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
        self.baseline_frames: Dict[int, PlasticReferenceFrame] = {}
        self.step_index = 0
        self.load_step_index = 1
        self.run_id = "run"
        self._consecutive_failures = 0
        self._last_result: Optional[PlasticRunResult] = None
        self._previous_result: Optional[PlasticRunResult] = None
        self._last_reward = 0.0
        self._last_info: Dict[str, Any] = {}
        self._last_action_by_cell: Dict[int, int] = {}
        self._last_action_step_by_cell: Dict[int, int] = {}
        self._temporarily_blocked_until: Dict[Tuple[int, int], int] = {}
        self._last_tangent_ratio = 1.0
        self.allow_early_stop = False
        # Unlike the legacy static environments, a plastic load-path decision
        # must be allowed to advance the load without forcing a mesh mutation.
        # KEEP is represented once (on one canonical active patch) to avoid N
        # duplicate no-op candidates with identical physical meaning.
        self.num_actions = 3

    @property
    def global_feature_dim(self) -> int:
        return len(self.GLOBAL_FEATURE_NAMES)

    @property
    def episode_horizon(self) -> int:
        return self.plate.load_steps - 1

    @property
    def load_fraction(self) -> float:
        return float(self.load_step_index) / float(self.plate.load_steps)

    def set_goal(self, goal: GoalCondition) -> None:
        self.current_goal = goal.normalized()

    def preflight(self) -> dict[str, Any]:
        return {
            "backend": "calculix-plastic",
            "gmsh_command": self.gmsh_cmd,
            "gmsh_available": command_available(self.gmsh_cmd),
            "ccx_command": self.ccx_cmd,
            "ccx_available": command_available(self.ccx_cmd),
            "simulation_root": str(self.simulations_root.resolve()),
            "episode_horizon": self.episode_horizon,
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
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = by_grid.get((cell.ix + dx, cell.iy + dy))
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
                f"left[] = Curve In BoundingBox{{{-tolerance:.16g}, {-tolerance:.16g}, {-tolerance:.16g}, {tolerance:.16g}, {p.height + tolerance:.16g}, {tolerance:.16g}}};",
                f"right[] = Curve In BoundingBox{{{p.length - tolerance:.16g}, {-tolerance:.16g}, {-tolerance:.16g}, {p.length + tolerance:.16g}, {p.height + tolerance:.16g}, {tolerance:.16g}}};",
                'Physical Curve("LEFT") = {left[]};',
                'Physical Curve("RIGHT") = {right[]};',
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
        field_ids: list[int] = []
        v_out = max(mesh_sizes.values()) * 1.0e4
        for field_id, (cell_id, cell) in enumerate(sorted(self.virtual_cells.items()), start=1):
            field_ids.append(field_id)
            size = float(mesh_sizes[cell_id])
            transition = p.mesh_transition_fraction * min(
                cell.x_max - cell.x_min, cell.y_max - cell.y_min
            )
            lines.extend(
                [
                    f"Field[{field_id}] = Box;",
                    f"Field[{field_id}].VIn = {size:.16g};",
                    f"Field[{field_id}].VOut = {v_out:.16g};",
                    f"Field[{field_id}].XMin = {cell.x_min:.16g};",
                    f"Field[{field_id}].XMax = {cell.x_max:.16g};",
                    f"Field[{field_id}].YMin = {cell.y_min:.16g};",
                    f"Field[{field_id}].YMax = {cell.y_max:.16g};",
                    f"Field[{field_id}].ZMin = -1;",
                    f"Field[{field_id}].ZMax = 1;",
                    f"Field[{field_id}].Thickness = {transition:.16g};",
                ]
            )
        minimum_field = len(field_ids) + 1
        lines.extend(
            [
                f"Field[{minimum_field}] = Min;",
                f"Field[{minimum_field}].FieldsList = {{{', '.join(str(value) for value in field_ids)}}};",
                f"Background Field = {minimum_field};",
            ]
        )
        if self.cell_min_mesh_size is not None:
            lines.append(f"Mesh.MeshSizeMin = {self.cell_min_mesh_size:.16g};")
        if self.cell_max_mesh_size is not None:
            lines.append(f"Mesh.MeshSizeMax = {self.cell_max_mesh_size:.16g};")
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _append_ids(lines: list[str], values: Sequence[int], width: int = 16) -> None:
        for start in range(0, len(values), width):
            lines.append(", ".join(str(value) for value in values[start : start + width]))

    def _write_calculix_deck(
        self,
        filepath: Path,
        mesh: Msh2Mesh,
        load_step: int,
    ) -> None:
        p = self.plate
        tolerance = max(p.length, p.height) * 1.0e-6
        left_nodes = sorted(
            node_id for node_id, (x, _, _) in mesh.nodes.items() if abs(x) <= tolerance
        )
        right_nodes = sorted(
            node_id for node_id, (x, _, _) in mesh.nodes.items() if abs(x - p.length) <= tolerance
        )
        if not left_nodes:
            raise RuntimeError("Gmsh mesh has no nodes on the x=0 boundary")
        if not right_nodes:
            raise RuntimeError("Gmsh mesh has no nodes on the x=L boundary")
        anchor = min(left_nodes, key=lambda node_id: (mesh.nodes[node_id][1], node_id))
        load_fraction = float(load_step) / float(p.load_steps)
        prescribed = p.target_displacement_x * load_fraction
        second_plastic_stress = p.yield_stress + p.hardening_modulus * p.plastic_curve_end_strain

        lines = [
            "*HEADING",
            "State-aware displacement-controlled elastoplastic plate with a hole",
            "*NODE",
        ]
        for node_id, (x, y, z) in sorted(mesh.nodes.items()):
            lines.append(f"{node_id}, {x:.16g}, {y:.16g}, {z:.16g}")
        lines.append("*ELEMENT, TYPE=CPS3, ELSET=EALL")
        for element_id, nodes in sorted(mesh.triangles.items()):
            lines.append(f"{element_id}, {nodes[0]}, {nodes[1]}, {nodes[2]}")
        lines.append("*NSET, NSET=NALL")
        self._append_ids(lines, sorted(mesh.nodes))
        lines.append("*NSET, NSET=LEFT")
        self._append_ids(lines, left_nodes)
        lines.append("*NSET, NSET=RIGHT")
        self._append_ids(lines, right_nodes)
        lines.extend(["*NSET, NSET=ANCHOR", str(anchor)])
        lines.extend(
            [
                "*MATERIAL, NAME=STEEL",
                "*ELASTIC",
                f"{p.young_modulus:.16g}, {p.poisson_ratio:.16g}",
                "*PLASTIC, HARDENING=ISOTROPIC",
                f"{p.yield_stress:.16g}, 0.0",
                f"{second_plastic_stress:.16g}, {p.plastic_curve_end_strain:.16g}",
                "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL",
                f"{p.thickness:.16g}",
                "*STEP, NLGEOM=NO, INC=1000",
                "*STATIC",
                f"{p.solver_initial_increment:.16g}, 1.0, {p.solver_min_increment:.16g}, {p.solver_max_increment:.16g}",
                "*BOUNDARY",
                "LEFT, 1, 1, 0.0",
                "ANCHOR, 2, 2, 0.0",
                f"RIGHT, 1, 1, {prescribed:.16g}",
                "*NODE FILE, FREQUENCY=999999",
                "U",
                "*EL FILE, FREQUENCY=999999",
                "S, PEEQ",
                "*NODE PRINT, NSET=RIGHT, TOTALS=YES, FREQUENCY=999999",
                "RF",
                "*EL PRINT, ELSET=EALL, FREQUENCY=999999",
                "S, PEEQ",
                "*END STEP",
            ]
        )
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _run_checked(self, command: Sequence[str], cwd: Path, label: str) -> None:
        completed = self._command_runner(command, cwd, self.solver_timeout_seconds)
        (cwd / f"{label}.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
        (cwd / f"{label}.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"{label} failed with exit code {completed.returncode}; "
                f"see {label}.stdout.log and {label}.stderr.log in {cwd}"
            )

    def _run_analysis(
        self,
        workdir: Path,
        mesh_sizes: Mapping[int, float],
        load_step: int,
    ) -> PlasticRunResult:
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
        self._write_calculix_deck(deck_path, mesh, load_step)
        self._run_checked([*self.ccx_cmd, "-i", "model"], workdir, "ccx")
        frd_path = workdir / "model.frd"
        dat_path = workdir / "model.dat"
        if not frd_path.exists():
            raise RuntimeError(f"CalculiX did not create {frd_path}")
        if not dat_path.exists():
            raise RuntimeError(f"CalculiX did not create {dat_path}")
        displacements = parse_frd_displacements(frd_path)
        frame = parse_calculix_plastic_dat(dat_path)
        return self._postprocess(workdir, mesh, displacements, frame, load_step)

    def _postprocess(
        self,
        workdir: Path,
        mesh: Msh2Mesh,
        displacements: Mapping[int, Tuple[float, float, float]],
        frame: PlasticDatFrame,
        load_step: int,
    ) -> PlasticRunResult:
        p = self.plate
        cell_to_elements = {cell_id: [] for cell_id in self.virtual_cells}
        element_mises: Dict[int, float] = {}
        element_peeq: Dict[int, float] = {}
        element_plastic_work: Dict[int, float] = {}
        element_disp: Dict[int, list[float]] = {}
        element_area: Dict[int, float] = {}
        total_plastic_work = 0.0
        total_area = 0.0
        plastic_area = 0.0

        for element_id, connectivity in mesh.triangles.items():
            if element_id not in frame.stresses or element_id not in frame.peeq:
                raise RuntimeError(f"DAT output is missing element {element_id}")
            coordinates = [mesh.nodes[node_id] for node_id in connectivity]
            centroid_x = sum(value[0] for value in coordinates) / 3.0
            centroid_y = sum(value[1] for value in coordinates) / 3.0
            cell_id = self._cell_for_point(centroid_x, centroid_y)
            if cell_id is None:
                continue
            if any(node_id not in displacements for node_id in connectivity):
                raise RuntimeError(f"FRD output is missing displacement for element {element_id}")
            area = _triangle_area(coordinates)
            peeq = max(0.0, float(frame.peeq[element_id]))
            mises = _von_mises_3d(frame.stresses[element_id])
            plastic_work_density = p.yield_stress * peeq + 0.5 * p.hardening_modulus * peeq**2
            plastic_work = plastic_work_density * area * p.thickness
            magnitudes = [
                math.sqrt(sum(component * component for component in displacements[node_id]))
                for node_id in connectivity
            ]
            cell_to_elements[cell_id].append(element_id)
            element_mises[element_id] = mises
            element_peeq[element_id] = peeq
            element_plastic_work[element_id] = plastic_work
            element_disp[element_id] = magnitudes
            element_area[element_id] = area
            total_plastic_work += plastic_work
            total_area += area
            if peeq > p.plastic_threshold:
                plastic_area += area

        total_elements = len(mesh.triangles)
        if total_elements <= 0:
            raise RuntimeError("Generated mesh contains no triangles")
        plastic_zone_fraction = plastic_area / max(total_area, 1.0e-12)
        max_peeq = max(element_peeq.values(), default=0.0)
        cell_mean_peeq: Dict[int, float] = {}
        cell_max_peeq: Dict[int, float] = {}
        cell_plastic_work: Dict[int, float] = {}
        for cell_id, element_ids in cell_to_elements.items():
            values = np.asarray(
                [element_peeq[element_id] for element_id in element_ids],
                dtype=np.float64,
            )
            weights = np.asarray(
                [element_area[element_id] for element_id in element_ids],
                dtype=np.float64,
            )
            if values.size and float(weights.sum()) > 1.0e-15:
                cell_mean_peeq[cell_id] = float(np.average(values, weights=weights))
                cell_max_peeq[cell_id] = float(np.max(values))
            else:
                cell_mean_peeq[cell_id] = 0.0
                cell_max_peeq[cell_id] = 0.0
            cell_plastic_work[cell_id] = float(
                sum(element_plastic_work[element_id] for element_id in element_ids)
            )

        total_reaction = frame.total_reaction
        reaction_x = abs(float(total_reaction[0]))
        reaction_y = float(total_reaction[1])
        cell_features = self._build_base_cell_features(
            mesh=mesh,
            cell_to_elements=cell_to_elements,
            element_mises=element_mises,
            element_peeq=element_peeq,
            element_plastic_work=element_plastic_work,
            element_disp=element_disp,
            element_area=element_area,
            cell_plastic_work=cell_plastic_work,
            total_plastic_work=total_plastic_work,
        )
        digest_payload = []
        for node_id, coordinates in sorted(mesh.nodes.items()):
            digest_payload.append(
                f"N,{node_id},{coordinates[0]:.10e},{coordinates[1]:.10e},{coordinates[2]:.10e}"
            )
        for element_id, connectivity in sorted(mesh.triangles.items()):
            digest_payload.append(
                f"E,{element_id},{connectivity[0]},{connectivity[1]},{connectivity[2]}"
            )
        mesh_digest = hashlib.sha256("\n".join(digest_payload).encode("ascii")).hexdigest()[:20]
        signature = (total_elements, len(mesh.nodes), mesh_digest)
        load_fraction = float(load_step) / float(p.load_steps)
        return PlasticRunResult(
            qoi=reaction_x,
            reaction_force_x=reaction_x,
            reaction_force_y=reaction_y,
            load_step=int(load_step),
            load_fraction=load_fraction,
            prescribed_displacement=p.target_displacement_x * load_fraction,
            element_count=total_elements,
            node_count=len(mesh.nodes),
            displacements=dict(displacements),
            element_mises=element_mises,
            element_peeq=element_peeq,
            cell_to_elements=cell_to_elements,
            cell_mean_peeq=cell_mean_peeq,
            cell_max_peeq=cell_max_peeq,
            cell_plastic_work=cell_plastic_work,
            cell_features=cell_features,
            total_plastic_work=float(total_plastic_work),
            plastic_zone_fraction=float(plastic_zone_fraction),
            max_peeq=float(max_peeq),
            mesh_signature=signature,
            workdir=str(workdir),
            mesh=mesh,
        )

    def _build_base_cell_features(
        self,
        mesh: Msh2Mesh,
        cell_to_elements: Mapping[int, Sequence[int]],
        element_mises: Mapping[int, float],
        element_peeq: Mapping[int, float],
        element_plastic_work: Mapping[int, float],
        element_disp: Mapping[int, Sequence[float]],
        element_area: Mapping[int, float],
        cell_plastic_work: Mapping[int, float],
        total_plastic_work: float,
    ) -> Dict[int, list[float]]:
        p = self.plate
        total_elements = max(sum(len(ids) for ids in cell_to_elements.values()), 1)
        target_disp = max(p.target_displacement_x, 1.0e-12)
        peeq_scale = max(p.plastic_curve_end_strain * 0.25, 1.0e-4)
        features: Dict[int, list[float]] = {}
        for cell_id, cell in self.virtual_cells.items():
            ids = list(cell_to_elements.get(cell_id, []))
            mises = np.asarray([element_mises[element_id] for element_id in ids], dtype=np.float64)
            peeq = np.asarray([element_peeq[element_id] for element_id in ids], dtype=np.float64)
            disp = np.asarray(
                [value for element_id in ids for value in element_disp[element_id]],
                dtype=np.float64,
            )
            areas = np.asarray([element_area[element_id] for element_id in ids], dtype=np.float64)
            if not ids:
                mises = np.asarray([0.0])
                peeq = np.asarray([0.0])
                disp = np.asarray([0.0])
                areas = np.asarray([0.0])
            stress_ratio = mises / p.yield_stress
            peeq_scaled = np.tanh(np.log1p(peeq / peeq_scale))

            def weighted_mean_std(values: np.ndarray) -> tuple[float, float]:
                weight_sum = float(areas.sum())
                if values.size == 0 or weight_sum <= 1.0e-15:
                    return 0.0, 0.0
                mean = float(np.average(values, weights=areas))
                variance = float(np.average((values - mean) ** 2, weights=areas))
                return mean, math.sqrt(max(variance, 0.0))

            stress_mean, stress_std = weighted_mean_std(stress_ratio)
            peeq_mean, peeq_std = weighted_mean_std(peeq_scaled)
            yield_excess = np.maximum(stress_ratio - 1.0, 0.0)
            yield_mean, _ = weighted_mean_std(yield_excess)
            active_area_fraction = (
                float(areas[peeq > p.plastic_threshold].sum())
                / max(float(areas.sum()), 1.0e-12)
            )
            element_disp_mean = np.asarray(
                [float(np.mean(element_disp[element_id])) for element_id in ids],
                dtype=np.float64,
            )
            if ids and float(areas.sum()) > 1.0e-15:
                displacement_mean = float(np.average(element_disp_mean, weights=areas))
            else:
                displacement_mean = 0.0

            center_x, center_y = cell.center
            hole_distance = 1.0
            if p.hole_radius > 0:
                hole_distance = max(
                    0.0,
                    math.hypot(center_x - p.hole_center_x, center_y - p.hole_center_y)
                    - p.hole_radius,
                ) / math.hypot(p.length, p.height)
            cell_area = (cell.x_max - cell.x_min) * (cell.y_max - cell.y_min)
            row = [
                stress_mean,
                stress_std,
                float(stress_ratio.max()),
                peeq_mean,
                peeq_std,
                float(peeq_scaled.max()),
                active_area_fraction,
                float(cell_plastic_work.get(cell_id, 0.0)) / (abs(total_plastic_work) + 1.0e-12),
                displacement_mean / target_disp,
                float(disp.max()) / target_disp,
                float(disp.max() - disp.min()) / target_disp,
                len(ids) / max(float(self.max_elements), 1.0),
                len(ids) / float(total_elements),
                center_x / p.length,
                center_y / p.height,
                (cell.x_max - cell.x_min) / p.length,
                (cell.y_max - cell.y_min) / p.height,
                hole_distance,
                float(cell.ix == 0),
                float(cell.ix == p.cells_x - 1),
                float(bool(ids)),
                yield_mean,
                float(yield_excess.max()),
                float(areas.mean()) / max(cell_area, 1.0e-12),
            ]
            if len(row) != len(self.BASE_CELL_FEATURE_NAMES):
                raise RuntimeError("Plastic base cell feature schema changed unexpectedly")
            features[cell_id] = row
        return features

    def _baseline_cache_key(self, baseline_mesh_size: float) -> str:
        payload = {
            "plate": asdict(self.plate),
            "baseline_mesh_size": float(baseline_mesh_size),
            "schema": 4,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    def compute_baseline(
        self,
        cache_dir: str = "checkpoints_calculix_plastic/baseline_cache",
        use_cache: bool = True,
        baseline_mesh_size: float = 0.25,
    ) -> Optional[float]:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = self._baseline_cache_key(baseline_mesh_size)
        cache_file = cache_root / f"calculix_plastic_{cache_key}.json"
        if use_cache and cache_file.exists():
            value = json.loads(cache_file.read_text(encoding="utf-8"))
            self.baseline_frames = {
                int(key): PlasticReferenceFrame.from_json(frame)
                for key, frame in value["frames"].items()
            }
            final = self.baseline_frames.get(self.plate.load_steps)
            return final.reaction_force_x if final is not None else None

        saved_sizes = dict(self.cell_mesh_density)
        saved_min_size = self.cell_min_mesh_size
        saved_max_size = self.cell_max_mesh_size
        frames: Dict[int, PlasticReferenceFrame] = {}
        try:
            # The training action bounds must not silently clamp a finer reference
            # request.  Use a dedicated baseline envelope so, for example, asking
            # for h=0.08 does not actually produce the training minimum h=0.12.
            reference_size = float(baseline_mesh_size)
            if reference_size <= 0.0:
                raise ValueError("baseline_mesh_size must be positive")
            self.cell_min_mesh_size = max(reference_size * 0.5, 1.0e-12)
            self.cell_max_mesh_size = reference_size
            baseline_sizes = {
                cell_id: reference_size for cell_id in self.virtual_cells
            }
            baseline_root = self.simulations_root / "_plastic_baseline" / cache_key
            for load_step in range(1, self.plate.load_steps + 1):
                result = self._run_analysis(
                    baseline_root / f"load_{load_step:03d}",
                    baseline_sizes,
                    load_step,
                )
                frames[load_step] = PlasticReferenceFrame(
                    load_step=load_step,
                    load_fraction=result.load_fraction,
                    reaction_force_x=result.reaction_force_x,
                    total_plastic_work=result.total_plastic_work,
                    plastic_zone_fraction=result.plastic_zone_fraction,
                    max_peeq=result.max_peeq,
                    cell_mean_peeq=dict(result.cell_mean_peeq),
                    cell_max_peeq=dict(result.cell_max_peeq),
                )
            self.baseline_frames = frames
            cache_file.write_text(
                json.dumps(
                    {
                        "schema": 4,
                        "plate": asdict(self.plate),
                        "baseline_mesh_size": float(baseline_mesh_size),
                        "frames": {str(key): frame.to_json() for key, frame in frames.items()},
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return frames[self.plate.load_steps].reaction_force_x
        finally:
            self.cell_mesh_density = saved_sizes
            self.cell_min_mesh_size = saved_min_size
            self.cell_max_mesh_size = saved_max_size

    def reset(self, run_id: Optional[str] = None) -> dict[str, Any]:
        self.run_id = run_id or "plastic_run"
        self.step_index = 0
        self.load_step_index = 1
        self._consecutive_failures = 0
        self._last_action_by_cell.clear()
        self._last_action_step_by_cell.clear()
        self._temporarily_blocked_until.clear()
        self.cell_mesh_density = {
            cell_id: self.global_mesh_size for cell_id in self.virtual_cells
        }
        result = self._run_analysis(
            self.simulations_root / self.run_id / "step_0000_load_001",
            self.cell_mesh_density,
            self.load_step_index,
        )
        self._previous_result = None
        self._last_result = result
        self._last_tangent_ratio = 1.0
        self._last_reward = 0.0
        self._last_info = self._result_info(result)
        return {
            "last_reward": 0.0,
            "cell_features": result.cell_features,
            "resource_usage": self._extract_resource_usage(),
            "global_features": {
                "reaction_force_x": result.reaction_force_x,
                "load_fraction": result.load_fraction,
            },
        }

    def _result_info(self, result: PlasticRunResult) -> Dict[str, Any]:
        metrics = self.current_error_metrics(result)
        return {
            "backend": "calculix-plastic",
            "qoi": result.qoi,
            "reaction_force_x": result.reaction_force_x,
            "reaction_force_y": result.reaction_force_y,
            "load_step": result.load_step,
            "load_fraction": result.load_fraction,
            "prescribed_displacement": result.prescribed_displacement,
            "total_plastic_work": result.total_plastic_work,
            "plastic_zone_fraction": result.plastic_zone_fraction,
            "max_peeq": result.max_peeq,
            "cell_mean_peeq": result.cell_mean_peeq,
            "cell_max_peeq": result.cell_max_peeq,
            "cell_plastic_work": result.cell_plastic_work,
            "cell_features": result.cell_features,
            "total_elements": result.element_count,
            "num_nodes": result.node_count,
            "resource_usage": result.element_count / max(float(self.max_elements), 1.0),
            "workdir": result.workdir,
            "error_metrics": metrics,
        }

    def _reference_for(self, load_step: Optional[int] = None) -> Optional[PlasticReferenceFrame]:
        return self.baseline_frames.get(int(load_step or self.load_step_index))

    def current_error_metrics(
        self, result: Optional[PlasticRunResult] = None
    ) -> Dict[str, float]:
        result = result or self._last_result
        if result is None:
            return {
                "reaction_error": 0.0,
                "plastic_work_error": 0.0,
                "plastic_profile_error": 0.0,
                "plastic_front_error": 0.0,
                "composite_error": 0.0,
            }
        reference = self._reference_for(result.load_step)
        if reference is None:
            return {
                "reaction_error": 0.0,
                "plastic_work_error": 0.0,
                "plastic_profile_error": 0.0,
                "plastic_front_error": 0.0,
                "composite_error": 0.0,
            }
        reaction_error = abs(result.reaction_force_x - reference.reaction_force_x) / (
            abs(reference.reaction_force_x) + 1.0e-12
        )
        work_scale = max(
            abs(reference.total_plastic_work),
            self.plate.yield_stress
            * self.plate.length
            * self.plate.height
            * self.plate.thickness
            * 1.0e-7,
        )
        plastic_work_error = abs(
            result.total_plastic_work - reference.total_plastic_work
        ) / work_scale
        cell_ids = sorted(self.virtual_cells)
        current_profile = np.asarray(
            [result.cell_mean_peeq.get(cell_id, 0.0) for cell_id in cell_ids],
            dtype=np.float64,
        )
        reference_profile = np.asarray(
            [reference.cell_mean_peeq.get(cell_id, 0.0) for cell_id in cell_ids],
            dtype=np.float64,
        )
        profile_scale = max(
            float(np.abs(reference_profile).sum()),
            self.plate.plastic_threshold * max(len(cell_ids), 1),
        )
        plastic_profile_error = float(np.abs(current_profile - reference_profile).sum() / profile_scale)
        current_active = current_profile > self.plate.plastic_threshold
        reference_active = reference_profile > self.plate.plastic_threshold
        union = int(np.logical_or(current_active, reference_active).sum())
        intersection = int(np.logical_and(current_active, reference_active).sum())
        plastic_front_error = 0.0 if union == 0 else 1.0 - intersection / float(union)
        accuracy_error = 0.75 * reaction_error + 0.25 * min(plastic_work_error, 5.0)
        localization_error = (
            0.70 * min(plastic_profile_error, 5.0)
            + 0.30 * plastic_front_error
        )
        goal = self.current_goal.normalized()
        physics_weight = max(
            goal.accuracy_priority + goal.localization_priority,
            1.0e-12,
        )
        composite = (
            goal.accuracy_priority * accuracy_error
            + goal.localization_priority * localization_error
        ) / physics_weight
        return {
            "reaction_error": float(reaction_error),
            "plastic_work_error": float(plastic_work_error),
            "plastic_profile_error": float(plastic_profile_error),
            "plastic_front_error": float(plastic_front_error),
            "accuracy_error": float(accuracy_error),
            "localization_error": float(localization_error),
            "composite_error": float(composite),
        }

    def relative_qoi_error(self) -> Optional[float]:
        if self._last_result is None or not self.baseline_frames:
            return None
        return self.current_error_metrics()["composite_error"]

    def evaluation_metrics(self) -> Dict[str, float]:
        metrics = self.current_error_metrics()
        metrics.update(
            {
                "resource_usage": self._extract_resource_usage(),
                "load_fraction": self.load_fraction,
                "reaction_force_x": self._last_result.reaction_force_x if self._last_result else 0.0,
                "plastic_zone_fraction": self._last_result.plastic_zone_fraction if self._last_result else 0.0,
                "max_peeq": self._last_result.max_peeq if self._last_result else 0.0,
            }
        )
        return metrics

    def _extract_resource_usage(self, info: Optional[Mapping[str, Any]] = None) -> float:
        if info is not None and info.get("resource_usage") is not None:
            return float(info["resource_usage"])
        if self._last_result is None:
            return 0.0
        return self._last_result.element_count / max(float(self.max_elements), 1.0)

    def _mesh_gradation(self, proposed: Optional[Mapping[int, float]] = None) -> float:
        sizes = proposed or self.cell_mesh_density
        jumps = []
        visited: set[Tuple[int, int]] = set()
        for cell_id, neighbors in self.cell_adjacency.items():
            for neighbor in neighbors:
                pair = tuple(sorted((cell_id, neighbor)))
                if pair in visited:
                    continue
                visited.add(pair)
                first = max(float(sizes[cell_id]), 1.0e-12)
                second = max(float(sizes[neighbor]), 1.0e-12)
                jumps.append(abs(math.log(first / second)))
        return float(np.mean(jumps)) if jumps else 0.0

    def _proposal_respects_gradation(self, cell_id: int, proposed_size: float) -> bool:
        limit = self.plate.max_neighbor_size_ratio
        for neighbor in self.cell_adjacency.get(cell_id, []):
            neighbor_size = float(self.cell_mesh_density[neighbor])
            ratio = max(proposed_size, neighbor_size) / max(min(proposed_size, neighbor_size), 1.0e-12)
            if ratio > limit + 1.0e-9:
                return False
        return True

    def _physical_action_mask(
        self, cell_ids: Iterable[int], goal: GoalCondition
    ) -> Dict[int, list[bool]]:
        ids = tuple(int(cell_id) for cell_id in cell_ids)
        if self.load_step_index >= self.plate.load_steps:
            return {cell_id: [False, False, False] for cell_id in ids}
        resource_usage = self._extract_resource_usage()
        reserve_limit = 1.0 - goal.reserve_budget_fraction
        active_elements = (
            self._last_result.cell_to_elements if self._last_result is not None else {}
        )
        active_cell_ids = sorted(
            cell_id
            for cell_id in self.virtual_cells
            if bool(active_elements.get(cell_id, []))
        )
        canonical_keep_cell = active_cell_ids[0] if active_cell_ids else None
        masks: Dict[int, list[bool]] = {}
        for cell_id in ids:
            current = float(self.cell_mesh_density[cell_id])
            refined = current * (1.0 - self.refine_step_size)
            coarsened = current * (1.0 + self.coarsen_step_size)
            active = bool(active_elements.get(cell_id, []))
            refine_valid = active and resource_usage < reserve_limit - 1.0e-9
            if self.cell_min_mesh_size is not None:
                refine_valid = refine_valid and refined >= self.cell_min_mesh_size - 1.0e-9
            refine_valid = refine_valid and self._proposal_respects_gradation(cell_id, refined)
            coarsen_valid = active
            if self.cell_max_mesh_size is not None:
                coarsen_valid = coarsen_valid and coarsened <= self.cell_max_mesh_size + 1.0e-9
            coarsen_valid = coarsen_valid and self._proposal_respects_gradation(cell_id, coarsened)
            keep_valid = active and cell_id == canonical_keep_cell
            masks[cell_id] = [
                bool(refine_valid),
                bool(coarsen_valid),
                bool(keep_valid),
            ]
        return masks

    def get_action_mask(
        self,
        cell_ids: Optional[Iterable[int]] = None,
        goal: Optional[GoalCondition] = None,
    ) -> Dict[int, list[bool]]:
        goal = (goal or self.current_goal).normalized()
        if cell_ids is None:
            cell_ids = sorted(self.virtual_cells)
        ids = tuple(int(cell_id) for cell_id in cell_ids)
        physical = self._physical_action_mask(ids, goal)
        masked = {cell_id: list(row) for cell_id, row in physical.items()}
        for cell_id in ids:
            for action in (REFINE, COARSEN, KEEP):
                blocked_until = self._temporarily_blocked_until.get((cell_id, action))
                if blocked_until is not None and self.step_index <= blocked_until:
                    masked[cell_id][action] = False
        if any(any(row) for row in masked.values()):
            return masked
        return physical

    def get_augmented_cell_observations(
        self, goal: Optional[GoalCondition] = None
    ) -> Dict[int, Dict[str, Any]]:
        goal = (goal or self.current_goal).normalized()
        if self._last_result is None:
            return {}
        masks = self.get_action_mask(self.virtual_cells, goal)
        lower = self.cell_min_mesh_size or 0.0
        upper = self.cell_max_mesh_size or max(self.global_mesh_size * 2.0, lower + 1.0)
        span = max(upper - lower, 1.0e-12)
        normalizer = max(self.global_mesh_size, 1.0e-12)
        observations: Dict[int, Dict[str, Any]] = {}
        for cell_id in sorted(self.virtual_cells):
            base = list(self._last_result.cell_features.get(cell_id, []))
            if len(base) != len(self.BASE_CELL_FEATURE_NAMES):
                raise RuntimeError("Plastic cell feature vector has an unexpected length")
            mesh_size = float(self.cell_mesh_density[cell_id])
            ratio = mesh_size / normalizer
            last_action = self._last_action_by_cell.get(cell_id)
            last_step = self._last_action_step_by_cell.get(cell_id)
            recency = 1.0 if last_step is None else min(max((self.step_index - last_step) / 20.0, 0.0), 1.0)
            ineffective = any(
                self._temporarily_blocked_until.get((cell_id, action), -1) >= self.step_index
                for action in (REFINE, COARSEN, KEEP)
            )
            neighbor_jumps = [
                abs(
                    math.log(
                        max(mesh_size, 1.0e-12)
                        / max(float(self.cell_mesh_density[neighbor]), 1.0e-12)
                    )
                )
                for neighbor in self.cell_adjacency.get(cell_id, [])
            ]
            dynamic = [
                ratio,
                float(np.tanh(math.log(max(ratio, 1.0e-12)))),
                float(np.clip((mesh_size - lower) / span, 0.0, 1.0)),
                float(masks[cell_id][REFINE]),
                float(masks[cell_id][COARSEN]),
                float(masks[cell_id][KEEP]),
                float(last_action == REFINE),
                float(last_action == COARSEN),
                float(last_action == KEEP),
                recency,
                float(ineffective),
                float(np.mean(neighbor_jumps)) if neighbor_jumps else 0.0,
            ]
            observations[cell_id] = {
                "self": base + dynamic,
                "neighbors": [
                    {"cell_id": neighbor, "features": self._last_result.cell_features.get(neighbor, [])}
                    for neighbor in self.cell_adjacency.get(cell_id, [])
                ],
            }
        return observations

    def get_global_feature_vector(
        self, goal: Optional[GoalCondition] = None, max_steps: int = 100
    ) -> list[float]:
        del max_steps  # Load fraction is the physically meaningful progress coordinate.
        goal = (goal or self.current_goal).normalized()
        resource_usage = self._extract_resource_usage()
        result = self._last_result
        nominal_yield_force = max(
            self.plate.yield_stress * self.plate.height * self.plate.thickness,
            1.0e-12,
        )
        reaction_ratio = result.reaction_force_x / nominal_yield_force if result else 0.0
        zone_fraction = result.plastic_zone_fraction if result else 0.0
        max_peeq = result.max_peeq if result else 0.0
        plastic_work_scale = max(
            self.plate.yield_stress
            * self.plate.length
            * self.plate.height
            * self.plate.thickness
            * self.plate.plastic_curve_end_strain,
            1.0e-12,
        )
        plastic_work_scaled = result.total_plastic_work / plastic_work_scale if result else 0.0
        ratios = np.asarray(
            [value / self.global_mesh_size for value in self.cell_mesh_density.values()],
            dtype=np.float64,
        )
        features = [
            float(np.clip(resource_usage, 0.0, 2.0)),
            max(0.0, 1.0 - resource_usage),
            self.load_fraction,
            min(self._consecutive_failures / max(float(self._max_consecutive_failures), 1.0), 1.0),
            float(np.tanh(self._last_reward / 5.0)),
            float(np.tanh(reaction_ratio)),
            float(np.clip(self._last_tangent_ratio, -2.0, 2.0)),
            float(zone_fraction),
            float(np.tanh(np.log1p(max_peeq / max(self.plate.plastic_curve_end_strain * 0.25, 1.0e-4)))),
            float(np.tanh(plastic_work_scaled)),
            float(ratios.mean()),
            float(ratios.std()),
            float(ratios.min()),
            float(ratios.max()),
            self._mesh_gradation(),
            float(bool(self._last_info.get("mesh_unchanged", False))),
            float(bool(self._last_info.get("state_rollback", False))),
            goal.accuracy_priority,
            goal.resource_priority,
            goal.localization_priority,
            goal.reserve_budget_fraction,
            float(np.tanh(np.log1p(goal.target_relative_error))),
        ]
        if len(features) != self.global_feature_dim:
            raise RuntimeError("Plastic global feature schema changed unexpectedly")
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
            num_actions=self.num_actions,
        )

    def step(self, action_params: Mapping[int, int]):
        if self.load_step_index >= self.plate.load_steps:
            return {}, 0.0, True, {"backend": "calculix-plastic", "load_path_complete": True}
        self.step_index += 1
        backup_sizes = dict(self.cell_mesh_density)
        previous_result = self._last_result
        invalid = []
        if len(action_params) != 1:
            invalid.append(("action_count", len(action_params)))
        physical_mask = self._physical_action_mask(self.virtual_cells, self.current_goal)
        intentional_keep = bool(action_params) and all(
            int(raw_action) == KEEP for raw_action in action_params.values()
        )
        for raw_cell_id, raw_action in action_params.items():
            cell_id = int(raw_cell_id)
            action = int(raw_action)
            if cell_id not in self.cell_mesh_density or action not in (REFINE, COARSEN, KEEP):
                invalid.append((cell_id, action))
                continue
            if not physical_mask.get(cell_id, [False] * self.num_actions)[action]:
                invalid.append((cell_id, action))
                continue
            current = self.cell_mesh_density[cell_id]
            if action == REFINE:
                self.cell_mesh_density[cell_id] = current * (1.0 - self.refine_step_size)
            elif action == COARSEN:
                self.cell_mesh_density[cell_id] = current * (1.0 + self.coarsen_step_size)
            else:
                # KEEP advances the physical load path on the unchanged mesh.
                self.cell_mesh_density[cell_id] = current
        if invalid:
            self.cell_mesh_density = backup_sizes
            return self._failure_transition(
                action_params, "invalid_action", -2.0, {"invalid_actions": invalid}
            )

        next_load_step = self.load_step_index + 1
        try:
            result = self._run_analysis(
                self.simulations_root
                / self.run_id
                / f"step_{self.step_index:04d}_load_{next_load_step:03d}",
                self.cell_mesh_density,
                next_load_step,
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
        self._previous_result = previous_result
        self._last_result = result
        self.load_step_index = next_load_step
        if previous_result is not None:
            delta_reaction = result.reaction_force_x - previous_result.reaction_force_x
            delta_disp = result.prescribed_displacement - previous_result.prescribed_displacement
            elastic_stiffness = (
                self.plate.young_modulus
                * self.plate.height
                * self.plate.thickness
                / self.plate.length
            )
            self._last_tangent_ratio = delta_reaction / max(delta_disp * elastic_stiffness, 1.0e-12)
        metrics = self.current_error_metrics(result)
        resource_usage = self._extract_resource_usage({"resource_usage": result.element_count / self.max_elements})
        accuracy_cost = self.current_goal.accuracy_priority * (
            0.75 * metrics["reaction_error"] + 0.25 * min(metrics["plastic_work_error"], 5.0)
        )
        localization_cost = self.current_goal.localization_priority * (
            0.70 * min(metrics["plastic_profile_error"], 5.0)
            + 0.30 * metrics["plastic_front_error"]
        )
        resource_cost = self.current_goal.resource_priority * resource_usage
        gradation_cost = 0.05 * self._mesh_gradation()
        ineffective_penalty = 0.20 if mesh_unchanged and not intentional_keep else 0.0
        reward = -float(
            accuracy_cost
            + localization_cost
            + resource_cost
            + gradation_cost
            + ineffective_penalty
        )
        done = self.load_step_index >= self.plate.load_steps
        if done and metrics["composite_error"] <= self.current_goal.target_relative_error:
            reward += 1.0
        self._last_reward = reward
        self._consecutive_failures = 0
        info = self._result_info(result)
        info.update(
            {
                "reward_components": {
                    "accuracy_cost": accuracy_cost,
                    "localization_cost": localization_cost,
                    "resource_cost": resource_cost,
                    "gradation_cost": gradation_cost,
                    "ineffective_penalty": ineffective_penalty,
                    "terminal_target_bonus": float(
                        done and metrics["composite_error"] <= self.current_goal.target_relative_error
                    ),
                    "goal_condition": self.current_goal.to_dict(),
                },
                "mesh_unchanged": mesh_unchanged,
                "intentional_keep": intentional_keep,
                "state_rollback": False,
                "load_path_complete": done,
                "cell_rewards": {int(cell_id): reward for cell_id in action_params},
            }
        )
        for raw_cell_id, raw_action in action_params.items():
            cell_id = int(raw_cell_id)
            action = int(raw_action)
            self._last_action_by_cell[cell_id] = action
            self._last_action_step_by_cell[cell_id] = self.step_index
            if mesh_unchanged and not intentional_keep:
                self._temporarily_blocked_until[(cell_id, action)] = self.step_index
        self._last_info = info
        obs = {
            "last_reward": reward,
            "cell_features": result.cell_features,
            "resource_usage": resource_usage,
            "global_features": {
                "reaction_force_x": result.reaction_force_x,
                "load_fraction": result.load_fraction,
                "plastic_zone_fraction": result.plastic_zone_fraction,
            },
        }
        return obs, reward, done, info

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
            "backend": "calculix-plastic",
            "penalty_type": penalty_type,
            "penalty_value": float(reward),
            "state_rollback": True,
            "mesh_unchanged": False,
            "consecutive_failures": self._consecutive_failures,
            "load_step": self.load_step_index,
            "load_fraction": self.load_fraction,
            **dict(details),
        }
        if self._last_result is not None:
            info.update(self._result_info(self._last_result))
        self._last_info = info
        done = self._consecutive_failures >= self._max_consecutive_failures
        obs = {
            "last_reward": float(reward),
            "cell_features": self._last_result.cell_features if self._last_result else {},
            "resource_usage": self._extract_resource_usage(info),
            "global_features": {},
        }
        return obs, float(reward), done, info
