from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .fem import Mesh, Solution
    from .regions import Region

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")
_MESH_COUNTER = itertools.count(1)
_SOLVE_COUNTER = itertools.count(1)


def _fmt(value: float) -> str:
    return f"{float(value):.14g}"


def _resolve_executable(candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError(f"{label} executable not found: {', '.join(candidates)}")


def resolve_gmsh() -> str:
    return _resolve_executable(("gmsh",), "Gmsh")


def resolve_ccx() -> str:
    return _resolve_executable(("ccx", "ccx_2.23", "ccx_2.22", "ccx_2.21", "ccx_2.20", "ccx_2.19"), "CalculiX")


def _root_from_env(name: str, fallback: str) -> Path:
    configured = os.environ.get(name, "").strip()
    root = Path(configured).resolve() if configured else Path(tempfile.gettempdir()) / fallback
    root.mkdir(parents=True, exist_ok=True)
    return root


def _new_work_dir(kind: str) -> Path:
    root = _root_from_env("BRIDGE_AGENT_NUMERICAL_EVIDENCE_ROOT", "bridge-agent-entity-calculix")
    token = f"{kind}_{os.getpid()}_{next(_MESH_COUNTER):06d}"
    path = root / token
    path.mkdir(parents=True, exist_ok=False)
    return path


def _run(command: list[str], cwd: Path, log_name: str, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    (cwd / log_name).write_text(completed.stdout, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed rc={completed.returncode}: {' '.join(command)}\n{completed.stdout[-5000:]}")
    return completed


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _entity_descriptor(x0: float, y0: float, width: float, height: float, holes: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "x0": round(float(x0), 12),
        "y0": round(float(y0), 12),
        "width": round(float(width), 12),
        "height": round(float(height), 12),
        "holes": [
            {
                "x": round(float(hole["x"]), 12),
                "y": round(float(hole["y"]), 12),
                "r": round(float(hole["r"]), 12),
            }
            for hole in holes
        ],
    }


def _acquire_lock(lock_path: Path, timeout: float = 120.0) -> int:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for entity-cache lock: {lock_path}")
            time.sleep(0.15)


def _ensure_brep(x0: float, y0: float, width: float, height: float, holes: list[dict[str, float]]) -> tuple[Path, str, dict[str, Any]]:
    descriptor = _entity_descriptor(x0, y0, width, height, holes)
    key = hashlib.sha256(_canonical_json(descriptor)).hexdigest()
    cache_root = _root_from_env("BRIDGE_AGENT_ENTITY_CACHE_ROOT", "bridge-agent-entity-cache") / key
    cache_root.mkdir(parents=True, exist_ok=True)
    brep_path = cache_root / "model.brep"
    manifest_path = cache_root / "entity_manifest.json"
    if brep_path.is_file() and manifest_path.is_file():
        digest = hashlib.sha256(brep_path.read_bytes()).hexdigest()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("brep_sha256") == digest and manifest.get("descriptor") == descriptor:
            return brep_path, digest, manifest
    lock_path = cache_root / ".build.lock"
    lock_fd = _acquire_lock(lock_path)
    try:
        if brep_path.is_file() and manifest_path.is_file():
            digest = hashlib.sha256(brep_path.read_bytes()).hexdigest()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("brep_sha256") == digest and manifest.get("descriptor") == descriptor:
                return brep_path, digest, manifest
        geometry_path = cache_root / "geometry.geo"
        lines = [
            'SetFactory("OpenCASCADE");',
            f"Rectangle(1) = {{{_fmt(x0)}, {_fmt(y0)}, 0, {_fmt(width)}, {_fmt(height)}}};",
        ]
        hole_tags: list[int] = []
        for index, hole in enumerate(holes, start=2):
            hole_tags.append(index)
            lines.append(f"Disk({index}) = {{{_fmt(hole['x'])}, {_fmt(hole['y'])}, 0, {_fmt(hole['r'])}, {_fmt(hole['r'])}}};")
        if hole_tags:
            lines.append(f"domain[] = BooleanDifference{{ Surface{{1}}; Delete; }}{{ Surface{{{', '.join(str(tag) for tag in hole_tags)}}}; Delete; }};")
        lines.extend(["Coherence;", f'Save "{brep_path.as_posix()}";'])
        geometry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _run([resolve_gmsh(), geometry_path.name, "-0", "-v", "2"], cache_root, "gmsh_geometry.log")
        if not brep_path.is_file() or brep_path.stat().st_size == 0:
            raise RuntimeError("Gmsh did not create the expected BREP")
        digest = hashlib.sha256(brep_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": "bridge-agent-exact-entity/1.0",
            "descriptor": descriptor,
            "brep_sha256": digest,
            "gmsh": resolve_gmsh(),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return brep_path, digest, manifest
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def _regular_background_pos(path: Path, x0: float, y0: float, width: float, height: float, size_at: Callable[[np.ndarray], np.ndarray], nx: int = 40, ny: int = 30) -> None:
    xs = np.linspace(x0, x0 + width, max(8, int(nx)) + 1)
    ys = np.linspace(y0, y0 + height, max(8, int(ny)) + 1)
    grid = np.asarray([(x_value, y_value) for y_value in ys for x_value in xs], dtype=float)
    values = np.asarray(size_at(grid), dtype=float)
    if values.shape != (len(grid),) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("background size function returned invalid values")
    rows = ['View "entity-first-size-field" {']
    stride = len(xs)
    for iy in range(len(ys) - 1):
        for ix in range(len(xs) - 1):
            n00 = iy * stride + ix
            n10 = n00 + 1
            n01 = n00 + stride
            n11 = n01 + 1
            for a, b, c in ((n00, n10, n11), (n00, n11, n01)):
                coordinates = [grid[a], grid[b], grid[c]]
                size_values = [values[a], values[b], values[c]]
                rows.append(
                    "ST(" + ",".join(_fmt(component) for point in coordinates for component in (point[0], point[1], 0.0)) + "){" + ",".join(_fmt(value) for value in size_values) + "};"
                )
    rows.append("};")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _boundary_loop_count(edges: list[tuple[int, int]]) -> int:
    if not edges:
        return 0
    adjacency: dict[int, set[int]] = {}
    for first, second in edges:
        adjacency.setdefault(int(first), set()).add(int(second))
        adjacency.setdefault(int(second), set()).add(int(first))
    unvisited = set(adjacency)
    components = 0
    while unvisited:
        components += 1
        stack = [unvisited.pop()]
        while stack:
            node = stack.pop()
            for neighbour in adjacency.get(node, set()):
                if neighbour in unvisited:
                    unvisited.remove(neighbour)
                    stack.append(neighbour)
    return components


def _parse_msh2(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[int, int]]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    physical_names: dict[tuple[int, int], str] = {}
    node_coordinates: dict[int, tuple[float, float]] = {}
    triangle_rows: list[tuple[int, int, int]] = []
    line_rows: list[tuple[str, int, int]] = []
    index = 0
    while index < len(lines):
        token = lines[index].strip()
        if token == "$PhysicalNames":
            count = int(lines[index + 1])
            for offset in range(count):
                dimension_text, tag_text, quoted_name = lines[index + 2 + offset].split(maxsplit=2)
                physical_names[(int(dimension_text), int(tag_text))] = quoted_name.strip('"')
            index += count + 3
            continue
        if token == "$Nodes":
            count = int(lines[index + 1])
            for offset in range(count):
                values = lines[index + 2 + offset].split()
                node_coordinates[int(values[0])] = (float(values[1]), float(values[2]))
            index += count + 3
            continue
        if token == "$Elements":
            count = int(lines[index + 1])
            for offset in range(count):
                values = [int(value) for value in lines[index + 2 + offset].split()]
                element_type = values[1]
                tag_count = values[2]
                tags = values[3 : 3 + tag_count]
                connectivity = values[3 + tag_count :]
                physical_tag = tags[0] if tags else 0
                if element_type == 1 and len(connectivity) == 2:
                    line_rows.append((physical_names.get((1, physical_tag), f"UNNAMED_{physical_tag}"), connectivity[0], connectivity[1]))
                elif element_type == 2 and len(connectivity) == 3 and physical_names.get((2, physical_tag)) == "DOMAIN":
                    triangle_rows.append((connectivity[0], connectivity[1], connectivity[2]))
            index += count + 3
            continue
        index += 1
    ordered_ids = sorted(node_coordinates)
    if not ordered_ids or not triangle_rows:
        raise ValueError(f"MSH2 file contains no DOMAIN mesh: {path}")
    id_to_index = {node_id: position for position, node_id in enumerate(ordered_ids)}
    nodes = np.asarray([node_coordinates[node_id] for node_id in ordered_ids], dtype=float)
    triangles = np.asarray([[id_to_index[node_id] for node_id in row] for row in triangle_rows], dtype=int)
    for row_index, row in enumerate(triangles):
        coordinates = nodes[row]
        twice_area = float(np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0]))
        if twice_area < 0:
            triangles[row_index, 1], triangles[row_index, 2] = triangles[row_index, 2], triangles[row_index, 1]
        elif twice_area <= 1.0e-14:
            raise ValueError("Gmsh produced a degenerate triangle")
    groups: dict[str, list[tuple[int, int]]] = {}
    for name, first, second in line_rows:
        groups.setdefault(name, []).append((id_to_index[first], id_to_index[second]))
    return nodes, triangles, groups


def _triangle_areas(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    coordinates = nodes[elements]
    return 0.5 * np.abs(np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]))


def _quality_metrics(nodes: np.ndarray, elements: np.ndarray) -> dict[str, float | int]:
    all_angles: list[float] = []
    all_aspects: list[float] = []
    invalid = 0
    for connectivity in elements:
        coordinates = nodes[connectivity]
        lengths = np.asarray([np.linalg.norm(coordinates[(index + 1) % 3] - coordinates[index]) for index in range(3)], dtype=float)
        if np.min(lengths) <= 1.0e-12:
            invalid += 1
            continue
        a_value, b_value, c_value = lengths
        cosines = [
            (a_value * a_value + c_value * c_value - b_value * b_value) / (2.0 * a_value * c_value),
            (a_value * a_value + b_value * b_value - c_value * c_value) / (2.0 * a_value * b_value),
            (b_value * b_value + c_value * c_value - a_value * a_value) / (2.0 * b_value * c_value),
        ]
        all_angles.extend(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))).tolist())
        area = 0.5 * abs(float(np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])))
        altitude = 2.0 * area / max(lengths)
        all_aspects.append(float(max(lengths) / max(altitude, 1.0e-12)))
    if not all_angles:
        return {"min_angle_deg": 0.0, "p05_angle_deg": 0.0, "max_aspect_ratio": float("inf"), "p95_aspect_ratio": float("inf"), "invalid_count": invalid}
    return {
        "min_angle_deg": float(np.min(all_angles)),
        "p05_angle_deg": float(np.percentile(all_angles, 5)),
        "max_aspect_ratio": float(np.max(all_aspects)),
        "p95_aspect_ratio": float(np.percentile(all_aspects, 95)),
        "invalid_count": int(invalid),
    }


def _apply_crack_seam(nodes: np.ndarray, elements: np.ndarray, crack_edges: list[tuple[int, int]], crack_y: float, half_crack: float) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[int, int]]], dict[str, np.ndarray], dict[str, Any]]:
    if not crack_edges:
        raise ValueError("embedded crack guide has no line elements")
    crack_nodes = sorted({node for edge in crack_edges for node in edge}, key=lambda node: nodes[node, 0])
    if len(crack_nodes) < 5:
        raise ValueError("embedded crack guide is too coarse")
    internal_nodes = crack_nodes[1:-1]
    duplicate: dict[int, int] = {}
    new_nodes = nodes.tolist()
    for node in internal_nodes:
        duplicate[int(node)] = len(new_nodes)
        new_nodes.append(nodes[int(node)].tolist())
    new_elements = elements.copy()
    for element_index, connectivity in enumerate(elements):
        centroid_y = float(np.mean(nodes[connectivity, 1]))
        if centroid_y < crack_y - 1.0e-10:
            new_elements[element_index] = [duplicate.get(int(node), int(node)) for node in connectivity]
    upper_edges = [(int(first), int(second)) for first, second in crack_edges]
    lower_edges = [(duplicate.get(int(first), int(first)), duplicate.get(int(second), int(second))) for first, second in crack_edges]
    center_node = min(internal_nodes, key=lambda node: abs(float(nodes[int(node), 0])))
    mouth_nodes = np.asarray([int(center_node), int(duplicate[int(center_node)])], dtype=int)
    pair_error = max(float(np.linalg.norm(np.asarray(new_nodes[new]) - np.asarray(new_nodes[old]))) for old, new in duplicate.items())
    evidence = {
        "representation": "intact BREP plus embedded guide plus duplicated coincident seam nodes",
        "half_crack": float(half_crack),
        "paired_internal_nodes": len(duplicate),
        "maximum_pair_coordinate_error": pair_error,
        "upper_lower_node_sets_disjoint": not bool(set(internal_nodes) & set(duplicate.values())),
        "shared_tip_nodes": [int(crack_nodes[0]), int(crack_nodes[-1])],
        "crack_length": float(nodes[crack_nodes[-1], 0] - nodes[crack_nodes[0], 0]),
    }
    return np.asarray(new_nodes, dtype=float), new_elements, {"crack_upper": upper_edges, "crack_lower": lower_edges}, {"crack_mouth": mouth_nodes}, evidence


def _mesh_from_exact_entity(*, x0: float, y0: float, width: float, height: float, holes: list[dict[str, float]], size_at: Callable[[np.ndarray], np.ndarray], metadata: dict[str, Any], crack: dict[str, float] | None = None, field_resolution: tuple[int, int] = (40, 30)) -> "Mesh":
    from .fem import Mesh
    from .regions import assign_region_owner

    brep_path, entity_digest, entity_manifest = _ensure_brep(x0, y0, width, height, holes)
    work_dir = _new_work_dir("mesh")
    field_path = work_dir / "background.pos"
    msh_path = work_dir / "mesh.msh"
    geo_path = work_dir / "mesh.geo"
    _regular_background_pos(field_path, x0, y0, width, height, size_at, nx=field_resolution[0], ny=field_resolution[1])
    tolerance = max(width, height, 1.0) * 1.0e-7
    lines = [
        'SetFactory("OpenCASCADE");',
        f'Merge "{brep_path.as_posix()}";',
        "Mesh.MshFileVersion = 2.2;",
        "Mesh.SaveAll = 0;",
        f"domain[] = Surface In BoundingBox{{{_fmt(x0 - tolerance)}, {_fmt(y0 - tolerance)}, {-tolerance:.14g}, {_fmt(x0 + width + tolerance)}, {_fmt(y0 + height + tolerance)}, {tolerance:.14g}}};",
        'Physical Surface("DOMAIN", 1) = {domain[]};',
        f"left[] = Curve In BoundingBox{{{_fmt(x0 - tolerance)}, {_fmt(y0 - tolerance)}, {-tolerance:.14g}, {_fmt(x0 + tolerance)}, {_fmt(y0 + height + tolerance)}, {tolerance:.14g}}};",
        f"right[] = Curve In BoundingBox{{{_fmt(x0 + width - tolerance)}, {_fmt(y0 - tolerance)}, {-tolerance:.14g}, {_fmt(x0 + width + tolerance)}, {_fmt(y0 + height + tolerance)}, {tolerance:.14g}}};",
        f"bottom[] = Curve In BoundingBox{{{_fmt(x0 - tolerance)}, {_fmt(y0 - tolerance)}, {-tolerance:.14g}, {_fmt(x0 + width + tolerance)}, {_fmt(y0 + tolerance)}, {tolerance:.14g}}};",
        f"top[] = Curve In BoundingBox{{{_fmt(x0 - tolerance)}, {_fmt(y0 + height - tolerance)}, {-tolerance:.14g}, {_fmt(x0 + width + tolerance)}, {_fmt(y0 + height + tolerance)}, {tolerance:.14g}}};",
        'Physical Curve("FIXED_EDGE", 101) = {left[]};',
        'Physical Curve("LOAD_EDGE", 102) = {right[]};',
        'Physical Curve("BOTTOM_EDGE", 103) = {bottom[]};',
        'Physical Curve("TOP_EDGE", 104) = {top[]};',
    ]
    for index, hole in enumerate(holes):
        radius = float(hole["r"])
        lines.extend([
            f"hole_{index}[] = Curve In BoundingBox{{{_fmt(float(hole['x']) - radius - tolerance)}, {_fmt(float(hole['y']) - radius - tolerance)}, {-tolerance:.14g}, {_fmt(float(hole['x']) + radius + tolerance)}, {_fmt(float(hole['y']) + radius + tolerance)}, {tolerance:.14g}}};",
            f'Physical Curve("HOLE_{index}", {200 + index}) = {{hole_{index}[]}};',
        ])
    if crack is not None:
        x_start = float(crack["x_start"])
        x_end = float(crack["x_end"])
        crack_y = float(crack["y"])
        target_size = max(float(crack.get("line_size", 1.0)), 1.0e-6)
        segments = max(20, int(math.ceil(abs(x_end - x_start) / target_size)))
        if segments % 2:
            segments += 1
        lines.extend([
            "crack_p0 = newp;",
            f"Point(crack_p0) = {{{_fmt(x_start)}, {_fmt(crack_y)}, 0, {_fmt(target_size)}}};",
            "crack_p1 = newp;",
            f"Point(crack_p1) = {{{_fmt(x_end)}, {_fmt(crack_y)}, 0, {_fmt(target_size)}}};",
            "crack_line = newl;",
            "Line(crack_line) = {crack_p0, crack_p1};",
            "Curve{crack_line} In Surface{domain[]};",
            'Physical Curve("CRACK_GUIDE", 300) = {crack_line};',
            f"Transfinite Curve {{crack_line}} = {segments + 1} Using Progression 1;",
        ])
    lines.extend([
        f'Merge "{field_path.as_posix()}";',
        "Background Mesh View[0];",
        "Mesh.MeshSizeExtendFromBoundary = 0;",
        "Mesh.MeshSizeFromPoints = 0;",
        "Mesh.MeshSizeFromCurvature = 0;",
        "Mesh.Algorithm = 6;",
        "Mesh.Optimize = 1;",
        "Mesh 2;",
        f'Save "{msh_path.as_posix()}";',
    ])
    geo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    digest_before = hashlib.sha256(brep_path.read_bytes()).hexdigest()
    _run([resolve_gmsh(), geo_path.name, "-2", "-format", "msh2", "-v", "2"], work_dir, "gmsh_mesh.log")
    digest_after = hashlib.sha256(brep_path.read_bytes()).hexdigest()
    if digest_before != entity_digest or digest_after != entity_digest:
        raise RuntimeError("exact BREP changed during meshing")
    nodes, elements, physical_groups = _parse_msh2(msh_path)
    edge_sets = {
        "left": list(physical_groups.get("FIXED_EDGE", [])),
        "right": list(physical_groups.get("LOAD_EDGE", [])),
        "bottom": list(physical_groups.get("BOTTOM_EDGE", [])),
        "top": list(physical_groups.get("TOP_EDGE", [])),
    }
    hole_edges: list[tuple[int, int]] = []
    for index in range(len(holes)):
        group = list(physical_groups.get(f"HOLE_{index}", []))
        if not group:
            raise RuntimeError(f"missing exact hole boundary group HOLE_{index}")
        edge_sets[f"hole_{index}"] = group
        hole_edges.extend(group)
    if len(holes) == 1:
        edge_sets["inner_hole"] = list(edge_sets["hole_0"])
    node_sets: dict[str, np.ndarray] = {}
    seam_evidence: dict[str, Any] = {}
    if crack is not None:
        nodes, elements, seam_sets, seam_node_sets, seam_evidence = _apply_crack_seam(nodes, elements, list(physical_groups.get("CRACK_GUIDE", [])), float(crack["y"]), 0.5 * abs(float(crack["x_end"]) - float(crack["x_start"])))
        edge_sets.update(seam_sets)
        node_sets.update(seam_node_sets)
    outer_edges = edge_sets["left"] + edge_sets["right"] + edge_sets["bottom"] + edge_sets["top"]
    edge_sets["boundary"] = outer_edges + hole_edges + edge_sets.get("crack_upper", []) + edge_sets.get("crack_lower", [])
    areas = _triangle_areas(nodes, elements)
    exact_area = width * height - sum(math.pi * float(hole["r"]) ** 2 for hole in holes)
    mesh_area = float(np.sum(areas))
    area_error = abs(mesh_area - exact_area) / max(abs(exact_area), 1.0e-12)
    loop_count = _boundary_loop_count(outer_edges + hole_edges)
    if loop_count != 1 + len(holes):
        raise RuntimeError(f"boundary-loop mismatch: expected {1 + len(holes)}, found {loop_count}")
    if area_error > 0.02:
        raise RuntimeError(f"mesh-domain area drift {area_error:.3%}")
    quality = _quality_metrics(nodes, elements)
    if int(quality["invalid_count"]) != 0 or float(quality["min_angle_deg"]) < 4.0:
        raise RuntimeError(f"invalid Gmsh quality: {quality}")
    complete_metadata = dict(metadata)
    complete_metadata.update({
        "backend": "exact-gmsh-opencascade",
        "entity_descriptor": entity_manifest["descriptor"],
        "entity_sha256": entity_digest,
        "entity_path": str(brep_path),
        "mesh_work_dir": str(work_dir),
        "mesh_geo": str(geo_path),
        "mesh_msh": str(msh_path),
        "exact_area": exact_area,
        "mesh_area": mesh_area,
        "relative_area_error": area_error,
        "boundary_loop_count": loop_count,
        "quality": quality,
        "seam": seam_evidence,
    })
    mesh = Mesh(nodes, elements, "T3", edge_sets, node_sets=node_sets, metadata=complete_metadata)
    regions = complete_metadata.get("regions")
    importance = complete_metadata.get("importance")
    if isinstance(regions, list) and regions:
        from .regions import regions_from_payload
        region_objects = regions_from_payload(regions)
        centers = nodes[elements].mean(axis=1)
        owner = assign_region_owner(centers, region_objects, importance if isinstance(importance, dict) else {})
        allocation = {region.region_id: int(np.sum(owner == index)) for index, region in enumerate(region_objects)}
        allocation["__background__"] = int(np.sum(owner < 0))
        mesh.metadata["region_element_allocation"] = allocation
    receipt = {
        "schema_version": "bridge-agent-entity-gmsh-mesh/1.0",
        "entity_sha256": entity_digest,
        "nodes": len(nodes),
        "elements": len(elements),
        "exact_area": exact_area,
        "mesh_area": mesh_area,
        "relative_area_error": area_error,
        "boundary_loop_count": loop_count,
        "quality": quality,
        "seam": seam_evidence,
        "metadata": complete_metadata,
    }
    (work_dir / "mesh_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return mesh


def _constant_size(value: float) -> Callable[[np.ndarray], np.ndarray]:
    safe = max(float(value), 1.0e-5)
    return lambda points: np.full(len(points), safe, dtype=float)


def _hole_size_function(holes: list[dict[str, float]], boundary_sizes: list[float], background: float) -> Callable[[np.ndarray], np.ndarray]:
    background = max(float(background), 1.0e-5)
    boundary_sizes = [max(float(value), 1.0e-5) for value in boundary_sizes]
    def size_at(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        values = np.full(len(p), background, dtype=float)
        for hole, boundary_size in zip(holes, boundary_sizes):
            radius = float(hole["r"])
            radial_distance = np.abs(np.hypot(p[:, 0] - float(hole["x"]), p[:, 1] - float(hole["y"])) - radius)
            transition = max(2.5 * radius, 4.0 * boundary_size)
            blend = np.clip(radial_distance / transition, 0.0, 1.0)
            candidate = boundary_size + (background - boundary_size) * blend
            values = np.minimum(values, candidate)
        return np.maximum(values, min([background, *boundary_sizes]) * 0.75)
    return size_at


def _region_size_function(regions: list["Region"], region_sizes: dict[str, float], background: float, importance: dict[str, float] | None = None) -> Callable[[np.ndarray], np.ndarray]:
    from .regions import assign_region_owner, region_boundary_distance, role_rank
    importance = importance or {}
    background = max(float(background), 1.0e-5)
    coarse_roles = {"coarsenable", "low_importance", "background_release"}
    all_sizes = [background, *[max(float(value), 1.0e-5) for value in region_sizes.values()]]
    global_min = min(all_sizes)
    global_max = max(all_sizes)
    def size_at(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        result = np.full(len(p), background, dtype=float)
        owner = assign_region_owner(p, regions, importance)
        for index, region in enumerate(regions):
            result[owner == index] = max(float(region_sizes.get(region.region_id, background)), 1.0e-5)
        for index, region in enumerate(regions):
            fine_size = max(float(region_sizes.get(region.region_id, background)), 1.0e-5)
            if fine_size >= background or role_rank(region) < 3:
                continue
            signed_distance = region_boundary_distance(region, p)
            outside_distance = np.maximum(signed_distance, 0.0)
            transition_width = max(2.4 * background, 4.0 * fine_size)
            allowed = owner < 0
            for other_index, other in enumerate(regions):
                if other.role in coarse_roles or role_rank(other) <= 3:
                    allowed |= owner == other_index
            influence = (outside_distance < transition_width) & allowed
            grown = fine_size * np.power(1.24, outside_distance / max(fine_size, 1.0e-12))
            result[influence] = np.minimum(result[influence], np.minimum(grown[influence], background))
        return np.clip(result, 0.80 * global_min, 1.08 * global_max)
    return size_at


def rectangle_q4_exact(length: float, height: float, nx: int, ny: int, *, centered: bool = True) -> "Mesh":
    x0 = 0.0
    y0 = -float(height) / 2.0 if centered else 0.0
    target = max(float(length) / max(int(nx), 1), float(height) / max(int(ny), 1))
    return _mesh_from_exact_entity(x0=x0, y0=y0, width=float(length), height=float(height), holes=[], size_at=_constant_size(target), metadata={"kind": "rectangle", "length": float(length), "height": float(height), "nx": int(nx), "ny": int(ny)}, field_resolution=(max(12, min(70, int(nx))), max(8, min(40, int(ny)))))


def circular_hole_q4_exact(width: float, height: float, radius: float, theta: np.ndarray, nr: int, *, cluster: float = 2.8) -> "Mesh":
    del cluster
    theta = np.asarray(theta, dtype=float)
    boundary_size = 2.0 * math.pi * float(radius) / max(len(theta), 16)
    background = max((min(float(width), float(height)) / max(int(nr), 1)) * 0.45, 1.6 * boundary_size)
    hole = {"x": 0.0, "y": 0.0, "r": float(radius)}
    return _mesh_from_exact_entity(x0=-float(width) / 2.0, y0=-float(height) / 2.0, width=float(width), height=float(height), holes=[hole], size_at=_hole_size_function([hole], [boundary_size], background), metadata={"kind": "circular_hole", "width": float(width), "height": float(height), "radius": float(radius), "ntheta": len(theta), "nr": int(nr)})


def multi_hole_tri_mesh_exact(width: float, height: float, holes: list[dict[str, float]], boundary_counts: list[int], *, interior_spacing: float = 24.0, radial_rings: int = 3) -> "Mesh":
    del radial_rings
    if len(holes) != len(boundary_counts):
        raise ValueError("hole/count mismatch")
    normalized_holes = [{"x": float(hole["x"]), "y": float(hole["y"]), "r": float(hole["r"])} for hole in holes]
    boundary_sizes = [2.0 * math.pi * float(hole["r"]) / max(int(count), 16) for hole, count in zip(normalized_holes, boundary_counts)]
    return _mesh_from_exact_entity(x0=-float(width) / 2.0, y0=-float(height) / 2.0, width=float(width), height=float(height), holes=normalized_holes, size_at=_hole_size_function(normalized_holes, boundary_sizes, float(interior_spacing)), metadata={"kind": "multi_hole_plate", "width": float(width), "height": float(height), "holes": normalized_holes, "boundary_counts": [int(value) for value in boundary_counts], "interior_spacing": float(interior_spacing)})


def _crack_size_function(width: float, height: float, half_crack: float, h_tip: float, h_far: float, h_y_near: float, h_y_far: float, tip_zone: float) -> Callable[[np.ndarray], np.ndarray]:
    h_tip = max(float(h_tip), 1.0e-5)
    h_far = max(float(h_far), h_tip)
    h_y_near = max(float(h_y_near), h_tip)
    h_y_far = max(float(h_y_far), h_y_near)
    tip_zone = max(float(tip_zone), 2.0 * h_tip)
    def size_at(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        values = np.full(len(p), h_far, dtype=float)
        tip_distance = np.minimum(np.hypot(p[:, 0] - half_crack, p[:, 1]), np.hypot(p[:, 0] + half_crack, p[:, 1]))
        tip_blend = np.clip(tip_distance / tip_zone, 0.0, 1.0)
        values = np.minimum(values, h_tip + (h_far - h_tip) * tip_blend)
        wake_mask = (np.abs(p[:, 0]) <= half_crack + tip_zone) & (np.abs(p[:, 1]) <= max(3.0 * h_y_near, 0.05 * height))
        wake_distance = np.abs(p[:, 1])
        wake_blend = np.clip(wake_distance / max(6.0 * h_y_near, 0.10 * height), 0.0, 1.0)
        wake_size = h_y_near + (h_y_far - h_y_near) * wake_blend
        values[wake_mask] = np.minimum(values[wake_mask], wake_size[wake_mask])
        return np.maximum(values, 0.78 * h_tip)
    return size_at


def central_crack_graded_q4_exact(width: float, height: float, half_crack: float, *, h_tip: float, h_far: float, h_y_near: float | None = None, h_y_far: float | None = None, tip_zone: float | None = None) -> "Mesh":
    h_y_near = float(h_y_near if h_y_near is not None else h_tip)
    h_y_far = float(h_y_far if h_y_far is not None else h_far)
    tip_zone = float(tip_zone if tip_zone is not None else max(2.5 * h_tip, 0.12 * half_crack))
    crack = {"x_start": -float(half_crack), "x_end": float(half_crack), "y": 0.0, "line_size": min(float(h_tip), h_y_near)}
    return _mesh_from_exact_entity(x0=-float(width) / 2.0, y0=-float(height) / 2.0, width=float(width), height=float(height), holes=[], size_at=_crack_size_function(float(width), float(height), float(half_crack), float(h_tip), float(h_far), h_y_near, h_y_far, tip_zone), metadata={"kind": "central_crack", "width": float(width), "height": float(height), "half_crack": float(half_crack), "h_tip": float(h_tip), "h_far": float(h_far), "h_y_near": h_y_near, "h_y_far": h_y_far, "tip_zone": tip_zone}, crack=crack)


def central_crack_q4_exact(width: float, height: float, half_crack: float, nx: int, ny: int) -> "Mesh":
    background = max(float(width) / max(int(nx), 1), float(height) / max(int(ny), 1))
    return central_crack_graded_q4_exact(width, height, half_crack, h_tip=0.45 * background, h_far=background, h_y_near=0.55 * background, h_y_far=background)


def adaptive_region_tri_mesh_exact(width: float, height: float, holes: list[dict[str, float]], regions: list["Region"], region_sizes: dict[str, float], background_size: float, importance: dict[str, float] | None = None, max_depth: int = 11) -> "Mesh":
    del max_depth
    normalized_holes = [{"x": float(hole["x"]), "y": float(hole["y"]), "r": float(hole["r"])} for hole in holes]
    return _mesh_from_exact_entity(x0=-float(width) / 2.0, y0=-float(height) / 2.0, width=float(width), height=float(height), holes=normalized_holes, size_at=_region_size_function(regions, region_sizes, float(background_size), importance), metadata={"kind": "adaptive_region_tri", "width": float(width), "height": float(height), "holes": normalized_holes, "background_size": float(background_size), "region_sizes": {str(key): float(value) for key, value in region_sizes.items()}, "regions": [region.to_dict() for region in regions], "importance": importance or {}})


def uniform_tri_mesh_exact(width: float, height: float, holes: list[dict[str, float]], size: float) -> "Mesh":
    normalized_holes = [{"x": float(hole["x"]), "y": float(hole["y"]), "r": float(hole["r"])} for hole in holes]
    return _mesh_from_exact_entity(x0=-float(width) / 2.0, y0=-float(height) / 2.0, width=float(width), height=float(height), holes=normalized_holes, size_at=_constant_size(float(size)), metadata={"kind": "uniform_tri", "width": float(width), "height": float(height), "holes": normalized_holes, "background_size": float(size)})


def uniform_mesh_matching_budget_exact(width: float, height: float, holes: list[dict[str, float]], target_elements: int, *, h_low: float = 1.0, h_high: float = 80.0, iterations: int = 14) -> "Mesh":
    target = max(int(target_elements), 1)
    domain_area = float(width) * float(height) - sum(math.pi * float(hole["r"]) ** 2 for hole in holes)
    lower = max(float(h_low), min(float(width), float(height)) / 2500.0)
    upper = max(float(h_high), lower * 1.01)
    estimate = float(np.clip(math.sqrt(max(4.0 * domain_area / (math.sqrt(3.0) * target), 1.0e-12)), lower, upper))
    cache: dict[float, "Mesh"] = {}
    best: "Mesh" | None = None
    best_difference = float("inf")
    def evaluate(size: float) -> int:
        nonlocal best, best_difference
        clipped = float(np.clip(size, lower, upper))
        key = round(clipped, 8)
        if key not in cache:
            cache[key] = uniform_tri_mesh_exact(width, height, holes, clipped)
        mesh = cache[key]
        difference = abs(len(mesh.elements) - target)
        if difference < best_difference:
            best = mesh
            best_difference = difference
        return len(mesh.elements)
    history: list[tuple[float, int]] = []
    current = estimate
    for _ in range(max(7, min(int(iterations), 14))):
        count = evaluate(current)
        history.append((current, count))
        if abs(count - target) / target < 0.012:
            break
        next_size = float(np.clip(current * (max(count, 1) / target) ** 0.45, lower, upper))
        if abs(next_size - current) / max(current, 1.0e-12) < 0.002:
            break
        current = next_size
    if history:
        best_size = min(history, key=lambda row: abs(row[1] - target))[0]
        for factor in (0.94, 0.97, 0.985, 1.015, 1.03, 1.06):
            evaluate(best_size * factor)
    if best is None:
        raise RuntimeError("uniform budget search produced no mesh")
    best.metadata["budget_target_elements"] = target
    best.metadata["budget_difference_elements"] = int(len(best.elements) - target)
    best.metadata["budget_search_evaluations"] = len(cache)
    return best


def crack_uniform_mesh_matching_budget_exact(width: float, height: float, half_crack: float, target_elements: int, *, h_low: float | None = None, h_high: float | None = None, iterations: int = 18) -> "Mesh":
    target = max(int(target_elements), 1)
    lower = float(h_low if h_low is not None else min(width, height) / 300.0)
    upper = float(h_high if h_high is not None else max(width, height) / 6.0)
    estimate = float(np.clip(math.sqrt(float(width) * float(height) / target), lower, upper))
    cache: dict[float, "Mesh"] = {}
    best: "Mesh" | None = None
    best_difference = float("inf")
    def evaluate(size: float) -> int:
        nonlocal best, best_difference
        clipped = float(np.clip(size, lower, upper))
        key = round(clipped, 8)
        if key not in cache:
            cache[key] = central_crack_graded_q4_exact(width, height, half_crack, h_tip=clipped, h_far=clipped, h_y_near=clipped, h_y_far=clipped, tip_zone=max(2.5 * clipped, 0.12 * half_crack))
        mesh = cache[key]
        difference = abs(len(mesh.elements) - target)
        if difference < best_difference:
            best = mesh
            best_difference = difference
        return len(mesh.elements)
    current = estimate
    history: list[tuple[float, int]] = []
    for _ in range(max(8, int(iterations))):
        count = evaluate(current)
        history.append((current, count))
        if abs(count - target) / target < 0.012:
            break
        next_size = float(np.clip(current * (max(count, 1) / target) ** 0.45, lower, upper))
        if abs(next_size - current) / max(current, 1.0e-12) < 0.002:
            break
        current = next_size
    if history:
        best_size = min(history, key=lambda row: abs(row[1] - target))[0]
        for factor in (0.94, 0.97, 0.985, 1.015, 1.03, 1.06):
            evaluate(best_size * factor)
    if best is None:
        raise RuntimeError("crack budget search produced no mesh")
    best.metadata["budget_target_elements"] = target
    best.metadata["budget_difference_elements"] = int(len(best.elements) - target)
    best.metadata["budget_search_evaluations"] = len(cache)
    return best


def _parse_dat_displacements(path: Path, expected_nodes: int) -> np.ndarray | None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    values: dict[int, tuple[float, float]] = {}
    mode = False
    started = False
    for line in lines:
        lower = line.lower()
        if "displacements" in lower and "nall" in lower:
            mode = True
            started = False
            values = {}
            continue
        if mode:
            numbers = [float(token.replace("D", "E").replace("d", "e")) for token in _FLOAT_RE.findall(line)]
            if len(numbers) >= 4:
                node_id = int(round(numbers[0]))
                values[node_id] = (float(numbers[1]), float(numbers[2]))
                started = True
                continue
            if started and not line.strip():
                mode = False
    if len(values) != expected_nodes:
        return None
    return np.asarray([values[index] for index in range(1, expected_nodes + 1)], dtype=float)


def _parse_frd_displacements(path: Path, expected_nodes: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start_indices = [index for index, line in enumerate(lines) if "-4  DISP" in line]
    if not start_indices:
        raise RuntimeError("CalculiX FRD contains no DISP dataset")
    start = start_indices[-1]
    values: dict[int, tuple[float, float]] = {}
    for line in lines[start + 1 :]:
        if line.lstrip().startswith("-3"):
            break
        if not line.lstrip().startswith("-1"):
            continue
        numbers = [float(token.replace("D", "E").replace("d", "e")) for token in _FLOAT_RE.findall(line)]
        if len(numbers) >= 5:
            node_id = int(round(numbers[1]))
            values[node_id] = (float(numbers[2]), float(numbers[3]))
    if len(values) != expected_nodes:
        missing = sorted(set(range(1, expected_nodes + 1)) - set(values))[:20]
        raise RuntimeError(f"FRD displacement parse incomplete: {len(values)}/{expected_nodes}, missing={missing}")
    return np.asarray([values[index] for index in range(1, expected_nodes + 1)], dtype=float)


def _postprocess_linear(mesh: "Mesh", young: float, poisson: float, thickness: float, load_vector: np.ndarray, displacements: np.ndarray) -> "Solution":
    from .fem import Solution, assemble, plane_stress_matrix, _q4_B, _tri_B
    stiffness = assemble(mesh, young, poisson, thickness)
    displacement_vector = np.asarray(displacements, dtype=float).reshape(-1)
    force_vector = np.asarray(load_vector, dtype=float).reshape(-1)
    reactions = stiffness @ displacement_vector - force_vector
    constitutive = plane_stress_matrix(young, poisson)
    element_count = len(mesh.elements)
    centers = np.zeros((element_count, 2), dtype=float)
    stress = np.zeros((element_count, 3), dtype=float)
    strain = np.zeros((element_count, 3), dtype=float)
    von_mises = np.zeros(element_count, dtype=float)
    areas = np.zeros(element_count, dtype=float)
    nodal_accumulator = np.zeros((len(mesh.nodes), 3), dtype=float)
    nodal_weight = np.zeros(len(mesh.nodes), dtype=float)
    for element_index, connectivity in enumerate(mesh.elements):
        coordinates = mesh.nodes[connectivity]
        degrees = np.asarray([[2 * int(node), 2 * int(node) + 1] for node in connectivity], dtype=int).ravel()
        if mesh.element_type == "Q4":
            strain_operator, determinant = _q4_B(coordinates, 0.0, 0.0)
            area = 4.0 * determinant
        else:
            strain_operator, area = _tri_B(coordinates)
        element_strain = strain_operator @ displacement_vector[degrees]
        element_stress = constitutive @ element_strain
        centers[element_index] = np.mean(coordinates, axis=0)
        strain[element_index] = element_strain
        stress[element_index] = element_stress
        sx_value, sy_value, txy_value = element_stress
        von_mises[element_index] = math.sqrt(max(sx_value * sx_value - sx_value * sy_value + sy_value * sy_value + 3.0 * txy_value * txy_value, 0.0))
        areas[element_index] = area
        for node in connectivity:
            nodal_accumulator[int(node)] += element_stress * area
            nodal_weight[int(node)] += area
    nodal_weight[nodal_weight == 0.0] = 1.0
    nodal_stress = nodal_accumulator / nodal_weight[:, None]
    sx_values, sy_values, txy_values = nodal_stress.T
    nodal_von_mises = np.sqrt(np.maximum(sx_values * sx_values - sx_values * sy_values + sy_values * sy_values + 3.0 * txy_values * txy_values, 0.0))
    strain_energy = float(0.5 * displacement_vector @ (stiffness @ displacement_vector))
    external_half_work = float(0.5 * force_vector @ displacement_vector)
    denominator = max(abs(strain_energy), abs(external_half_work), 1.0e-30)
    return Solution(displacements=np.asarray(displacements, dtype=float), reactions=np.asarray(reactions, dtype=float).reshape((-1, 2)), element_centers=centers, element_stress=stress, element_strain=strain, element_von_mises=von_mises, nodal_stress=nodal_stress, nodal_von_mises=nodal_von_mises, load_vector=force_vector, stiffness=stiffness, strain_energy=strain_energy, external_half_work=external_half_work, energy_balance_rel=abs(strain_energy - external_half_work) / denominator, element_areas=areas)


def calculix_solve_linear(mesh: "Mesh", young: float, poisson: float, thickness: float, load_vector: np.ndarray, constraints: dict[int, float]) -> "Solution":
    solver_root = _root_from_env("BRIDGE_AGENT_NUMERICAL_EVIDENCE_ROOT", "bridge-agent-entity-calculix")
    job_index = next(_SOLVE_COUNTER)
    job_name = f"linear_{os.getpid()}_{job_index:06d}"
    work_dir = solver_root / job_name
    work_dir.mkdir(parents=True, exist_ok=False)
    input_path = work_dir / f"{job_name}.inp"
    node_count = len(mesh.nodes)
    element_count = len(mesh.elements)
    element_type = "CPS4" if mesh.element_type == "Q4" else "CPS3"
    fixed_nodes = sorted({int(dof) // 2 + 1 for dof in constraints})
    lines = ["*HEADING", "Original V4 Agent with exact Gmsh entities and CalculiX environment", "*NODE,NSET=NALL"]
    for node_id, (x_value, y_value) in enumerate(mesh.nodes, start=1):
        lines.append(f"{node_id},{_fmt(x_value)},{_fmt(y_value)},0")
    lines.append(f"*ELEMENT,TYPE={element_type},ELSET=EALL")
    for element_id, connectivity in enumerate(mesh.elements, start=1):
        lines.append(f"{element_id}," + ",".join(str(int(node) + 1) for node in connectivity))
    lines.extend(["*MATERIAL,NAME=STEEL", "*ELASTIC", f"{_fmt(young)},{_fmt(poisson)}", "*SOLID SECTION,ELSET=EALL,MATERIAL=STEEL", _fmt(thickness)])
    if fixed_nodes:
        lines.extend(["*NSET,NSET=FIXED", ",".join(str(node) for node in fixed_nodes)])
    lines.append("*BOUNDARY")
    for dof, value in sorted(constraints.items()):
        node_id = int(dof) // 2 + 1
        component = int(dof) % 2 + 1
        lines.append(f"{node_id},{component},{component},{_fmt(value)}")
    lines.extend(["*STEP,NLGEOM=NO", "*STATIC", "1.0,1.0", "*CLOAD"])
    reshaped_load = np.asarray(load_vector, dtype=float).reshape((-1, 2))
    for node_id, pair in enumerate(reshaped_load, start=1):
        for component, value in enumerate(pair, start=1):
            if abs(float(value)) > 1.0e-12:
                lines.append(f"{node_id},{component},{_fmt(value)}")
    lines.extend(["*NODE PRINT,NSET=NALL", "U"])
    if fixed_nodes:
        lines.extend(["*NODE PRINT,NSET=FIXED,TOTALS=YES", "RF"])
    lines.extend(["*NODE FILE", "U,RF", "*END STEP", ""])
    input_path.write_text("\n".join(lines), encoding="utf-8")
    ccx = resolve_ccx()
    completed = _run([ccx, "-i", job_name], work_dir, "ccx.stdout.log", timeout=900.0)
    dat_path = work_dir / f"{job_name}.dat"
    frd_path = work_dir / f"{job_name}.frd"
    displacements = _parse_dat_displacements(dat_path, node_count) if dat_path.is_file() else None
    if displacements is None:
        if not frd_path.is_file():
            raise RuntimeError("CalculiX produced neither a complete DAT displacement block nor FRD output")
        displacements = _parse_frd_displacements(frd_path, node_count)
    solution = _postprocess_linear(mesh, float(young), float(poisson), float(thickness), np.asarray(load_vector, dtype=float), displacements)
    equilibrium = np.sum(solution.reactions, axis=0) + np.sum(np.asarray(load_vector, dtype=float).reshape((-1, 2)), axis=0)
    denominator = max(float(np.linalg.norm(np.sum(np.asarray(load_vector, dtype=float).reshape((-1, 2)), axis=0))), 1.0)
    equilibrium_error = float(np.linalg.norm(equilibrium) / denominator)
    if not np.all(np.isfinite(solution.displacements)) or not np.all(np.isfinite(solution.element_stress)):
        raise RuntimeError("CalculiX returned non-finite fields")
    if equilibrium_error > 1.0e-5:
        raise RuntimeError(f"CalculiX equilibrium gate failed: {equilibrium_error}")
    mesh.metadata["last_calculix_solve"] = {"solver": ccx, "work_dir": str(work_dir), "input": str(input_path), "dat": str(dat_path), "frd": str(frd_path), "return_code": completed.returncode, "equilibrium_error": equilibrium_error, "energy_balance_rel": solution.energy_balance_rel}
    receipt = {"schema_version": "bridge-agent-calculix-linear/1.0", "solver": ccx, "nodes": node_count, "elements": element_count, "element_type": element_type, "entity_sha256": mesh.metadata.get("entity_sha256"), "equilibrium_error": equilibrium_error, "strain_energy": solution.strain_energy, "external_half_work": solution.external_half_work, "energy_balance_rel": solution.energy_balance_rel, "files": {"inp": str(input_path), "dat": str(dat_path), "frd": str(frd_path), "stdout": str(work_dir / "ccx.stdout.log")}}
    (work_dir / "solve_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return solution


def _numbers(line: str) -> list[float]:
    return [float(token.replace("D", "E").replace("d", "e")) for token in _FLOAT_RE.findall(line)]


def _parse_plastic_dat(path: Path, probe_node: int, cmod_nodes: list[int]) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    mode: str | None = None
    started = False
    stresses: list[tuple[int, int, list[float]]] = []
    peeq: list[tuple[int, int, float]] = []
    displacements: dict[int, list[float]] = {}
    for line in lines:
        lower = line.lower()
        if "stresses" in lower and "eall" in lower:
            mode = "stress"
            started = False
            continue
        if ("equivalent plastic strain" in lower or "peeq" in lower) and "eall" in lower:
            mode = "peeq"
            started = False
            continue
        if "displacements" in lower and ("probe" in lower or "cmod" in lower):
            mode = "disp"
            started = False
            continue
        numbers = _numbers(line)
        if mode == "stress" and len(numbers) >= 8:
            stresses.append((int(round(numbers[0])), int(round(numbers[1])), [float(value) for value in numbers[2:8]]))
            started = True
            continue
        if mode == "peeq" and len(numbers) >= 3:
            peeq.append((int(round(numbers[0])), int(round(numbers[1])), float(numbers[2])))
            started = True
            continue
        if mode == "disp" and len(numbers) >= 4:
            displacements[int(round(numbers[0]))] = [float(numbers[1]), float(numbers[2]), float(numbers[3])]
            started = True
            continue
        if started and not line.strip():
            mode = None
            started = False
    if probe_node not in displacements:
        requested = {probe_node, *cmod_nodes}
        for line in lines:
            numbers = _numbers(line)
            if len(numbers) >= 4 and int(round(numbers[0])) in requested:
                displacements[int(round(numbers[0]))] = [float(numbers[1]), float(numbers[2]), float(numbers[3])]
    return {"stresses": stresses, "peeq": peeq, "displacements": displacements}


def _connected_plastic_zone(mesh: "Mesh", peeq_array: np.ndarray, tip: np.ndarray, relative_threshold: float = 0.01, absolute_threshold: float = 1.0e-6) -> dict[str, Any]:
    values = np.asarray(peeq_array, dtype=float)
    maximum = float(np.max(values)) if len(values) else 0.0
    threshold = max(float(absolute_threshold), float(relative_threshold) * maximum)
    active = np.flatnonzero(values >= threshold)
    centers = np.asarray([mesh.nodes[connectivity].mean(axis=0) for connectivity in mesh.elements], dtype=float)
    if len(active) == 0:
        return {"threshold": threshold, "max_peeq": maximum, "active_count": 0, "connected_count": 0, "connected_indices": [], "radius": 0.0, "x_extent": 0.0, "y_extent": 0.0, "seed_index": None}
    active_set = set(int(index) for index in active)
    node_to_active: dict[int, list[int]] = {}
    for element_index in active:
        for node in mesh.elements[int(element_index)]:
            node_to_active.setdefault(int(node), []).append(int(element_index))
    seed = int(active[np.argmin(np.linalg.norm(centers[active] - np.asarray(tip, dtype=float), axis=1))])
    queue = [seed]
    connected = {seed}
    while queue:
        element_index = queue.pop()
        neighbours: set[int] = set()
        for node in mesh.elements[element_index]:
            neighbours.update(node_to_active.get(int(node), []))
        for neighbour in neighbours:
            if neighbour in active_set and neighbour not in connected:
                connected.add(neighbour)
                queue.append(neighbour)
    indices = np.asarray(sorted(connected), dtype=int)
    delta = centers[indices] - np.asarray(tip, dtype=float)
    return {"threshold": threshold, "max_peeq": maximum, "active_count": int(len(active)), "connected_count": int(len(indices)), "connected_indices": indices.tolist(), "radius": float(np.max(np.linalg.norm(delta, axis=1))) if len(indices) else 0.0, "x_extent": float(np.max(np.abs(delta[:, 0]))) if len(indices) else 0.0, "y_extent": float(np.max(np.abs(delta[:, 1]))) if len(indices) else 0.0, "seed_index": seed}


def run_elastoplastic_crack_exact(mesh: "Mesh", out_dir: Path, *, young: float, poisson: float, thickness: float, sigma: float, yield_curve: list[tuple[float, float]], job: str) -> dict[str, Any]:
    from .fem import edge_traction, nearest_node
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / f"{job}.inp"
    nodes = mesh.nodes
    elements = mesh.elements
    xmin, ymin = nodes.min(axis=0)
    xmax = float(nodes[:, 0].max())
    ymax = float(nodes[:, 1].max())
    first_constraint = nearest_node(mesh, (float(xmin), float(ymin)))
    second_constraint = nearest_node(mesh, (xmax, float(ymin)))
    load_vector = edge_traction(mesh, mesh.edge_sets["top"], (0.0, float(sigma)), float(thickness)) + edge_traction(mesh, mesh.edge_sets["bottom"], (0.0, -float(sigma)), float(thickness))
    cmod_nodes = np.asarray(mesh.node_sets.get("crack_mouth", []), dtype=int)
    if len(cmod_nodes) < 2:
        raise RuntimeError("exact crack seam contains no paired crack-mouth nodes")
    probe = nearest_node(mesh, (0.0, ymax))
    element_type = "CPS4" if mesh.element_type == "Q4" else "CPS3"
    lines = ["*HEADING", "Original V4 exact-seam elastoplastic crack", "*NODE,NSET=NALL"]
    for node_id, (x_value, y_value) in enumerate(nodes, start=1):
        lines.append(f"{node_id},{_fmt(x_value)},{_fmt(y_value)},0")
    lines.append(f"*ELEMENT,TYPE={element_type},ELSET=EALL")
    for element_id, connectivity in enumerate(elements, start=1):
        lines.append(f"{element_id}," + ",".join(str(int(node) + 1) for node in connectivity))
    lines.extend(["*MATERIAL,NAME=STEEL", "*ELASTIC", f"{_fmt(young)},{_fmt(poisson)}", "*PLASTIC"])
    for stress, plastic_strain in yield_curve:
        lines.append(f"{_fmt(stress)},{_fmt(plastic_strain)}")
    lines.extend(["*SOLID SECTION,ELSET=EALL,MATERIAL=STEEL", _fmt(thickness), "*NSET,NSET=PROBE", str(probe + 1), "*NSET,NSET=CMOD", ",".join(str(int(node) + 1) for node in cmod_nodes[:2])])
    lines.extend(["*BOUNDARY", f"{first_constraint + 1},1,2,0", f"{second_constraint + 1},2,2,0", "*STEP,NLGEOM=NO", "*STATIC", "0.02,1.0,1e-07,0.08", "*CLOAD"])
    for node_id, pair in enumerate(load_vector.reshape((-1, 2)), start=1):
        for component, value in enumerate(pair, start=1):
            if abs(float(value)) > 1.0e-12:
                lines.append(f"{node_id},{component},{_fmt(value)}")
    lines.extend(["*NODE PRINT,NSET=PROBE", "U", "*NODE PRINT,NSET=CMOD", "U", "*EL PRINT,ELSET=EALL", "S,PEEQ", "*NODE FILE", "U", "*EL FILE", "S,PEEQ", "*END STEP", ""])
    input_path.write_text("\n".join(lines), encoding="utf-8")
    ccx = resolve_ccx()
    completed = _run([ccx, "-i", job], out_dir, f"{job}.stdout.txt", timeout=1200.0)
    dat_path = out_dir / f"{job}.dat"
    if not dat_path.is_file():
        raise RuntimeError("CalculiX plastic solve produced no DAT file")
    parsed = _parse_plastic_dat(dat_path, probe + 1, [int(node) + 1 for node in cmod_nodes[:2]])
    element_peeq: dict[int, float] = {}
    for element_id, integration_point, value in parsed["peeq"]:
        del integration_point
        element_peeq[element_id] = max(element_peeq.get(element_id, 0.0), float(value))
    centers = np.asarray([mesh.nodes[connectivity].mean(axis=0) for connectivity in mesh.elements], dtype=float)
    peeq_array = np.zeros(len(centers), dtype=float)
    for element_id, value in element_peeq.items():
        if 1 <= element_id <= len(peeq_array):
            peeq_array[element_id - 1] = value
    right_tip = np.asarray([float(mesh.metadata["half_crack"]), 0.0], dtype=float)
    zone = _connected_plastic_zone(mesh, peeq_array, right_tip)
    displacement_rows = parsed["displacements"]
    probe_displacement = displacement_rows.get(probe + 1, [None, None, None])
    upper_node = int(cmod_nodes[0]) + 1
    lower_node = int(cmod_nodes[1]) + 1
    upper_displacement = displacement_rows.get(upper_node, [0.0, 0.0, 0.0])
    lower_displacement = displacement_rows.get(lower_node, [0.0, 0.0, 0.0])
    cmod = abs(float(upper_displacement[1]) - float(lower_displacement[1]))
    stress_von_mises: list[float] = []
    for element_id, integration_point, values in parsed["stresses"]:
        del element_id, integration_point
        sx_value, sy_value, sz_value, sxy_value, sxz_value, syz_value = values
        stress_von_mises.append(math.sqrt(max(0.5 * ((sx_value - sy_value) ** 2 + (sy_value - sz_value) ** 2 + (sz_value - sx_value) ** 2) + 3.0 * (sxy_value ** 2 + sxz_value ** 2 + syz_value ** 2), 0.0)))
    connected_mask = np.zeros(len(centers), dtype=np.uint8)
    if zone["connected_indices"]:
        connected_mask[np.asarray(zone["connected_indices"], dtype=int)] = 1
    field_path = out_dir / f"{job}_plastic_field.npz"
    np.savez_compressed(field_path, centers=centers, peeq=peeq_array, connected_plastic_zone=connected_mask, threshold=np.asarray([zone["threshold"]]))
    result = {"solver": ccx, "job": job, "return_code": completed.returncode, "mesh": {"nodes": len(nodes), "elements": len(elements), "element_type": element_type, "entity_sha256": mesh.metadata.get("entity_sha256"), "seam": mesh.metadata.get("seam")}, "material": {"young": float(young), "poisson": float(poisson), "yield_curve": yield_curve}, "nominal_stress": float(sigma), "max_integration_point_von_mises": max(stress_von_mises, default=None), "max_peeq": float(zone["max_peeq"]), "yielded_element_count": int(zone["connected_count"]), "total_thresholded_element_count": int(zone["active_count"]), "plastic_zone_radius_from_right_tip": float(zone["radius"]), "plastic_zone_x_extent_from_right_tip": float(zone["x_extent"]), "plastic_zone_y_extent_from_right_tip": float(zone["y_extent"]), "plastic_zone_threshold": float(zone["threshold"]), "plastic_zone_method": "tip-connected thresholded PEEQ component", "probe_displacement": probe_displacement, "crack_mouth_opening_displacement": cmod, "files": {"inp": str(input_path), "dat": str(dat_path), "frd": str(out_dir / f"{job}.frd"), "sta": str(out_dir / f"{job}.sta"), "stdout": str(out_dir / f"{job}.stdout.txt"), "plastic_field": str(field_path)}}
    (out_dir / f"{job}_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
