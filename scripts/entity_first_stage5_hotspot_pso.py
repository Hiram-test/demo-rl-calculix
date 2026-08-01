"""Run actual Gmsh hotspot PSO on four fixed models without DeepSeek or FE-error objectives."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import acos, ceil, degrees, exp, hypot, log, pi, sqrt
from pathlib import Path
from random import Random
from statistics import median
from typing import Callable, Sequence
import argparse
import html
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from entity_first_stage1_audit import audit_mesh, triangle_area
from entity_first_stage2_gmsh import format_number

Point = tuple[float, float]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class Case:
    case_id: str
    width: float
    height: float
    holes: tuple[tuple[float, float, float], ...]
    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    target_elements: int
    target_ratios: tuple[float, ...]


@dataclass(frozen=True)
class Evaluation:
    valid: bool
    score: float
    position: tuple[float, ...]
    nodes: int
    triangles: int
    minimum_angle_deg: float
    maximum_radius_ratio: float
    relative_area_error: float
    entity_sha256: str
    density: dict[str, object]
    seam: dict[str, object]
    issues: tuple[str, ...]


CASES: dict[str, Case] = {
    "bearing_plate": Case(
        "bearing_plate",
        1000.0,
        100.0,
        (),
        ("fixed_size", "load_size", "fixed_extent", "load_extent", "background_size"),
        (3.0, 3.0, 70.0, 70.0, 28.0),
        (14.0, 14.0, 300.0, 260.0, 70.0),
        2200,
        (0.20, 0.20),
    ),
    "circular_opening": Case(
        "circular_opening",
        240.0,
        240.0,
        ((120.0, 120.0, 20.0),),
        ("hotspot_size", "hotspot_halfwidth", "hotspot_halfheight", "background_size"),
        (1.5, 8.0, 6.0, 14.0),
        (8.0, 38.0, 32.0, 42.0),
        1800,
        (0.18,),
    ),
    "three_openings": Case(
        "three_openings",
        600.0,
        260.0,
        ((130.0, 130.0, 24.0), (300.0, 130.0, 42.0), (450.0, 130.0, 30.0)),
        ("left_size", "middle_size", "right_size", "left_extent", "middle_extent", "right_extent", "background_size"),
        (2.0, 1.5, 3.0, 10.0, 14.0, 12.0, 18.0),
        (10.0, 10.0, 12.0, 50.0, 72.0, 58.0, 45.0),
        3600,
        (0.24, 0.13, 0.34),
    ),
    "cracked_web": Case(
        "cracked_web",
        200.0,
        200.0,
        (),
        ("tip_size", "wake_size", "tip_extent", "wake_halfheight", "background_size"),
        (0.8, 1.5, 6.0, 4.0, 10.0),
        (5.0, 10.0, 30.0, 35.0, 35.0),
        3200,
        (0.12, 0.30),
    ),
}


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required executable is unavailable: {name}")
    return path


def run_gmsh(source_path: Path, dimension: int = 2) -> None:
    command = (executable("gmsh"), source_path.as_posix(), f"-{dimension}", "-nopopup")
    completed = subprocess.run(command, cwd=source_path.parent, text=True, capture_output=True, check=False)
    (source_path.parent / f"{source_path.stem}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (source_path.parent / f"{source_path.stem}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Gmsh failed for {source_path.name}:\n{completed.stdout}\n{completed.stderr}")


def ensure_crack_brep(entity_root: Path) -> Path:
    case_root = entity_root / "cracked_web"
    brep_path = case_root / "model.brep"
    if brep_path.exists() and brep_path.stat().st_size > 0:
        return brep_path
    case_root.mkdir(parents=True, exist_ok=True)
    geo_path = case_root / "geometry.geo"
    geo_path.write_text(
        'SetFactory("OpenCASCADE");\nRectangle(1) = {0, 0, 0, 200, 200};\nSave "'
        + brep_path.as_posix()
        + '";\n',
        encoding="utf-8",
    )
    run_gmsh(geo_path, dimension=0)
    if not brep_path.exists() or brep_path.stat().st_size == 0:
        raise RuntimeError("failed to create intact crack-plate BREP")
    return brep_path


def entity_path(case: Case, entity_root: Path) -> Path:
    if case.case_id == "cracked_web":
        return ensure_crack_brep(entity_root)
    path = entity_root / case.case_id / "model.brep"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def fixed_source(case: Case, brep_path: Path) -> list[str]:
    epsilon = max(case.width, case.height) * 1.0e-7
    lines = ['SetFactory("OpenCASCADE");']
    lines.append(f'Merge "{brep_path.as_posix()}";')
    lines.append("Mesh.MshFileVersion = 2.2;")
    lines.append("Mesh.SaveAll = 0;")
    lines.append(
        f"domain[] = Surface In BoundingBox{{{-epsilon}, {-epsilon}, {-epsilon}, {format_number(case.width + epsilon)}, {format_number(case.height + epsilon)}, {epsilon}}};"
    )
    lines.append('Physical Surface("DOMAIN", 1) = {domain[]};')
    lines.append(
        f"left[] = Curve In BoundingBox{{{-epsilon}, {-epsilon}, {-epsilon}, {epsilon}, {format_number(case.height + epsilon)}, {epsilon}}};"
    )
    lines.append(
        f"right[] = Curve In BoundingBox{{{format_number(case.width - epsilon)}, {-epsilon}, {-epsilon}, {format_number(case.width + epsilon)}, {format_number(case.height + epsilon)}, {epsilon}}};"
    )
    lines.append(
        f"bottom[] = Curve In BoundingBox{{{-epsilon}, {-epsilon}, {-epsilon}, {format_number(case.width + epsilon)}, {epsilon}, {epsilon}}};"
    )
    lines.append(
        f"top[] = Curve In BoundingBox{{{-epsilon}, {format_number(case.height - epsilon)}, {-epsilon}, {format_number(case.width + epsilon)}, {format_number(case.height + epsilon)}, {epsilon}}};"
    )
    lines.append('Physical Curve("FIXED_EDGE", 101) = {left[]};')
    lines.append('Physical Curve("LOAD_EDGE", 102) = {right[]};')
    lines.append('Physical Curve("BOTTOM_EDGE", 103) = {bottom[]};')
    lines.append('Physical Curve("TOP_EDGE", 104) = {top[]};')
    for index, (center_x, center_y, radius) in enumerate(case.holes):
        lines.append(
            f"hole_{index}[] = Curve In BoundingBox{{{format_number(center_x - radius - epsilon)}, {format_number(center_y - radius - epsilon)}, {-epsilon}, {format_number(center_x + radius + epsilon)}, {format_number(center_y + radius + epsilon)}, {epsilon}}};"
        )
        lines.append(f'Physical Curve("HOLE_{index}", {200 + index}) = {{hole_{index}[]}};')
    return lines


def add_box(
    lines: list[str],
    field_id: int,
    local_size: float,
    background_size: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    thickness: float,
) -> int:
    lines.append(f"Field[{field_id}] = Box;")
    lines.append(f"Field[{field_id}].VIn = {format_number(local_size)};")
    lines.append(f"Field[{field_id}].VOut = {format_number(background_size)};")
    lines.append(f"Field[{field_id}].XMin = {format_number(xmin)};")
    lines.append(f"Field[{field_id}].XMax = {format_number(xmax)};")
    lines.append(f"Field[{field_id}].YMin = {format_number(ymin)};")
    lines.append(f"Field[{field_id}].YMax = {format_number(ymax)};")
    lines.append("Field[{0}].ZMin = -1;".format(field_id))
    lines.append("Field[{0}].ZMax = 1;".format(field_id))
    lines.append(f"Field[{field_id}].Thickness = {format_number(max(thickness, 0.5))};")
    return field_id


def candidate_source(case: Case, position: Sequence[float], brep_path: Path, msh_path: Path) -> str:
    values = dict(zip(case.names, position))
    lines = fixed_source(case, brep_path)
    fields: list[int] = []
    next_field = 1
    background = values["background_size"]
    if case.case_id == "bearing_plate":
        fixed_extent = values["fixed_extent"]
        load_extent = values["load_extent"]
        fields.append(
            add_box(
                lines,
                next_field,
                values["fixed_size"],
                background,
                0.0,
                fixed_extent,
                0.0,
                case.height,
                0.35 * fixed_extent,
            )
        )
        next_field += 1
        fields.append(
            add_box(
                lines,
                next_field,
                values["load_size"],
                background,
                case.width - load_extent,
                case.width,
                0.0,
                case.height,
                0.35 * load_extent,
            )
        )
        next_field += 1
    elif case.case_id == "circular_opening":
        center_x, center_y, radius = case.holes[0]
        halfwidth = values["hotspot_halfwidth"]
        halfheight = values["hotspot_halfheight"]
        for hotspot_y in (center_y - radius, center_y + radius):
            fields.append(
                add_box(
                    lines,
                    next_field,
                    values["hotspot_size"],
                    background,
                    center_x - halfwidth,
                    center_x + halfwidth,
                    hotspot_y - halfheight,
                    hotspot_y + halfheight,
                    max(halfwidth, halfheight),
                )
            )
            next_field += 1
        segments = max(48, int(ceil(2.0 * pi * radius / max(values["hotspot_size"], 0.5))))
        lines.append(f"Transfinite Curve {{hole_0[]}} = {segments + 1} Using Progression 1;")
    elif case.case_id == "three_openings":
        size_names = ("left_size", "middle_size", "right_size")
        extent_names = ("left_extent", "middle_extent", "right_extent")
        for index, (center_x, center_y, radius) in enumerate(case.holes):
            extent = values[extent_names[index]]
            halfwidth = extent
            halfheight = 0.72 * extent
            for hotspot_y in (center_y - radius, center_y + radius):
                fields.append(
                    add_box(
                        lines,
                        next_field,
                        values[size_names[index]],
                        background,
                        center_x - halfwidth,
                        center_x + halfwidth,
                        hotspot_y - halfheight,
                        hotspot_y + halfheight,
                        extent,
                    )
                )
                next_field += 1
            segments = max(48, int(ceil(2.0 * pi * radius / max(values[size_names[index]], 0.5))))
            lines.append(f"Transfinite Curve {{hole_{index}[]}} = {segments + 1} Using Progression 1;")
    else:
        lines.append("crack_left = newp;")
        lines.append("Point(crack_left) = {70, 100, 0, 1};")
        lines.append("crack_right = newp;")
        lines.append("Point(crack_right) = {130, 100, 0, 1};")
        lines.append("crack_line = newl;")
        lines.append("Line(crack_line) = {crack_left, crack_right};")
        lines.append("Curve{crack_line} In Surface{domain[0]};")
        lines.append('Physical Curve("CRACK_GUIDE", 300) = {crack_line};')
        tip_extent = values["tip_extent"]
        for tip_x in (70.0, 130.0):
            fields.append(
                add_box(
                    lines,
                    next_field,
                    values["tip_size"],
                    background,
                    tip_x - tip_extent,
                    tip_x + tip_extent,
                    100.0 - tip_extent,
                    100.0 + tip_extent,
                    tip_extent,
                )
            )
            next_field += 1
        wake_height = values["wake_halfheight"]
        fields.append(
            add_box(
                lines,
                next_field,
                values["wake_size"],
                background,
                70.0,
                130.0,
                100.0 - wake_height,
                100.0 + wake_height,
                wake_height,
            )
        )
        next_field += 1
        crack_segments = max(60, int(ceil(60.0 / max(values["wake_size"], 0.5))))
        lines.append(f"Transfinite Curve {{crack_line}} = {crack_segments + 1} Using Progression 1;")
    minimum_field = next_field
    lines.append(f"Field[{minimum_field}] = Min;")
    lines.append(f"Field[{minimum_field}].FieldsList = {{{', '.join(str(value) for value in fields)}}};")
    lines.append(f"Background Field = {minimum_field};")
    local_sizes = [value for name, value in values.items() if name.endswith("size") and name != "background_size"]
    lines.append(f"Mesh.MeshSizeMin = {format_number(min(local_sizes))};")
    lines.append(f"Mesh.MeshSizeMax = {format_number(background)};")
    lines.append("Mesh.MeshSizeFromPoints = 0;")
    lines.append("Mesh.MeshSizeFromCurvature = 0;")
    lines.append("Mesh.MeshSizeExtendFromBoundary = 0;")
    lines.append("Mesh.Algorithm = 6;")
    lines.append("Mesh 2;")
    lines.append(f'Save "{msh_path.as_posix()}";')
    return "\n".join(lines) + "\n"


def parse_msh2(path: Path) -> tuple[list[Point], list[Triangle], dict[str, list[tuple[int, int]]]]:
    text = path.read_text(encoding="utf-8").splitlines()
    physical_names: dict[tuple[int, int], str] = {}
    node_coordinates: dict[int, Point] = {}
    triangle_rows: list[tuple[int, int, int]] = []
    line_rows: list[tuple[str, int, int]] = []
    index = 0
    while index < len(text):
        token = text[index].strip()
        if token == "$PhysicalNames":
            count = int(text[index + 1])
            for offset in range(count):
                dimension_text, tag_text, quoted_name = text[index + 2 + offset].split(maxsplit=2)
                physical_names[(int(dimension_text), int(tag_text))] = quoted_name.strip('"')
            index += count + 3
            continue
        if token == "$Nodes":
            count = int(text[index + 1])
            for offset in range(count):
                values = text[index + 2 + offset].split()
                node_coordinates[int(values[0])] = (float(values[1]), float(values[2]))
            index += count + 3
            continue
        if token == "$Elements":
            count = int(text[index + 1])
            for offset in range(count):
                values = [int(value) for value in text[index + 2 + offset].split()]
                element_type = values[1]
                tag_count = values[2]
                tags = values[3 : 3 + tag_count]
                connectivity = values[3 + tag_count :]
                physical_tag = tags[0] if tags else 0
                if element_type == 1 and len(connectivity) == 2:
                    line_rows.append(
                        (physical_names.get((1, physical_tag), f"UNNAMED_{physical_tag}"), connectivity[0], connectivity[1])
                    )
                if element_type == 2 and len(connectivity) == 3 and physical_names.get((2, physical_tag)) == "DOMAIN":
                    triangle_rows.append(tuple(connectivity))
            index += count + 3
            continue
        index += 1
    ordered_ids = sorted(node_coordinates)
    id_to_index = {node_id: position for position, node_id in enumerate(ordered_ids)}
    points = [node_coordinates[node_id] for node_id in ordered_ids]
    triangles = [tuple(id_to_index[node_id] for node_id in row) for row in triangle_rows]
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for name, first, second in line_rows:
        groups[name].append((id_to_index[first], id_to_index[second]))
    if not points or not triangles:
        raise ValueError("MSH2 file contains no DOMAIN mesh")
    return points, triangles, dict(groups)


def create_crack_seam(
    points: list[Point], triangles: list[Triangle], crack_edges: Sequence[tuple[int, int]]
) -> tuple[list[Point], list[Triangle], dict[str, object]]:
    if not crack_edges:
        raise ValueError("CRACK_GUIDE contains no line elements")
    crack_nodes = sorted({node for edge in crack_edges for node in edge}, key=lambda node: points[node][0])
    internal_nodes = crack_nodes[1:-1]
    if len(internal_nodes) < 8:
        raise ValueError(f"crack guide has only {len(internal_nodes)} internal nodes")
    duplicate: dict[int, int] = {}
    new_points = list(points)
    for node in internal_nodes:
        duplicate[node] = len(new_points)
        new_points.append(points[node])
    new_triangles: list[Triangle] = []
    for triangle in triangles:
        centroid_y = sum(points[node][1] for node in triangle) / 3.0
        if centroid_y < 100.0 - 1.0e-9:
            new_triangles.append(tuple(duplicate.get(node, node) for node in triangle))
        else:
            new_triangles.append(triangle)
    pair_error = max(
        hypot(new_points[new_node][0] - new_points[old_node][0], new_points[new_node][1] - new_points[old_node][1])
        for old_node, new_node in duplicate.items()
    )
    upper_internal = set(internal_nodes)
    lower_internal = set(duplicate.values())
    if upper_internal & lower_internal:
        raise ValueError("crack seam node sets overlap")
    evidence = {
        "left_tip_index": crack_nodes[0],
        "right_tip_index": crack_nodes[-1],
        "paired_internal_nodes": len(duplicate),
        "maximum_pair_coordinate_error": pair_error,
        "upper_lower_node_sets_disjoint": True,
    }
    return new_points, new_triangles, evidence


def equivalent_records(points: Sequence[Point], triangles: Sequence[Triangle]) -> tuple[tuple[float, float, float], ...]:
    records: list[tuple[float, float, float]] = []
    for triangle in triangles:
        p0, p1, p2 = (points[node] for node in triangle)
        area = triangle_area(p0, p1, p2)
        size = sqrt(4.0 * area / sqrt(3.0))
        records.append(((p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0, size))
    return tuple(records)


def zone_median(
    records: Sequence[tuple[float, float, float]], predicate: Callable[[float, float], bool], name: str
) -> float:
    values = [size for x_value, y_value, size in records if predicate(x_value, y_value)]
    if len(values) < 5:
        raise ValueError(f"zone {name} contains only {len(values)} elements")
    return float(median(values))


def density_metrics(
    case: Case, position: Sequence[float], records: Sequence[tuple[float, float, float]]
) -> tuple[dict[str, object], tuple[float, ...]]:
    values = dict(zip(case.names, position))
    if case.case_id == "bearing_plate":
        fixed = zone_median(records, lambda x_value, y_value: x_value < 0.75 * values["fixed_extent"], "fixed")
        load = zone_median(
            records,
            lambda x_value, y_value: x_value > case.width - 0.75 * values["load_extent"],
            "load",
        )
        far = zone_median(records, lambda x_value, y_value: 0.42 * case.width < x_value < 0.58 * case.width, "far")
        ratios = (fixed / far, load / far)
        return {
            "fixed_end_median_size": fixed,
            "load_end_median_size": load,
            "far_field_median_size": far,
            "ratios": ratios,
        }, ratios
    if case.case_id == "circular_opening":
        center_x, center_y, radius = case.holes[0]
        halfwidth = values["hotspot_halfwidth"]
        halfheight = values["hotspot_halfheight"]
        hotspot = zone_median(
            records,
            lambda x_value, y_value: abs(x_value - center_x) < 0.75 * halfwidth
            and min(abs(y_value - (center_y - radius)), abs(y_value - (center_y + radius))) < 0.75 * halfheight,
            "top_bottom_hotspots",
        )
        far = zone_median(
            records,
            lambda x_value, y_value: hypot(x_value - center_x, y_value - center_y) > 3.3 * radius
            and 0.10 * case.width < x_value < 0.90 * case.width
            and 0.10 * case.height < y_value < 0.90 * case.height,
            "far",
        )
        ratios = (hotspot / far,)
        return {"hotspot_median_size": hotspot, "far_field_median_size": far, "ratios": ratios}, ratios
    if case.case_id == "three_openings":
        size_names = ("left_size", "middle_size", "right_size")
        extent_names = ("left_extent", "middle_extent", "right_extent")
        hotspot_sizes: list[float] = []
        for index, (center_x, center_y, radius) in enumerate(case.holes):
            extent = values[extent_names[index]]
            hotspot_sizes.append(
                zone_median(
                    records,
                    lambda x_value, y_value, cx=center_x, cy=center_y, r=radius, e=extent: abs(x_value - cx) < 0.72 * e
                    and min(abs(y_value - (cy - r)), abs(y_value - (cy + r))) < 0.52 * e,
                    f"hole_{index}_hotspots",
                )
            )
        far = zone_median(
            records,
            lambda x_value, y_value: all(
                hypot(x_value - center_x, y_value - center_y) > 2.8 * radius
                for center_x, center_y, radius in case.holes
            )
            and 0.08 * case.width < x_value < 0.92 * case.width
            and 0.08 * case.height < y_value < 0.92 * case.height,
            "far",
        )
        ratios = tuple(size / far for size in hotspot_sizes)
        return {"hotspot_median_sizes": hotspot_sizes, "far_field_median_size": far, "ratios": ratios}, ratios
    tip_extent = values["tip_extent"]
    wake_height = values["wake_halfheight"]
    tip = zone_median(
        records,
        lambda x_value, y_value: min(hypot(x_value - 70.0, y_value - 100.0), hypot(x_value - 130.0, y_value - 100.0))
        < 0.72 * tip_extent,
        "tips",
    )
    wake = zone_median(
        records,
        lambda x_value, y_value: 76.0 < x_value < 124.0
        and abs(y_value - 100.0) < 0.65 * wake_height
        and min(hypot(x_value - 70.0, y_value - 100.0), hypot(x_value - 130.0, y_value - 100.0))
        > 0.82 * tip_extent,
        "wake",
    )
    far = zone_median(
        records,
        lambda x_value, y_value: abs(y_value - 100.0) > 2.4 * wake_height
        and 25.0 < x_value < 175.0
        and 20.0 < y_value < 180.0,
        "far",
    )
    ratios = (tip / far, wake / far)
    return {"tip_median_size": tip, "wake_median_size": wake, "far_field_median_size": far, "ratios": ratios}, ratios


def exact_area(case: Case) -> float:
    return case.width * case.height - sum(pi * radius * radius for _, _, radius in case.holes)


def expected_components(case: Case) -> int:
    return 1 + len(case.holes)


def expected_hole_areas(case: Case) -> tuple[float, ...]:
    return tuple(pi * radius * radius for _, _, radius in case.holes)


def score_candidate(case: Case, ratios: Sequence[float], triangle_count: int, minimum_angle: float, radius_ratio: float) -> float:
    contrast = sum(log(max(actual, 1.0e-6) / target) ** 2 for actual, target in zip(ratios, case.target_ratios))
    budget = 0.22 * log(max(triangle_count, 1) / case.target_elements) ** 2
    angle_penalty = 0.08 * max(0.0, (14.0 - minimum_angle) / 6.0) ** 2
    ratio_penalty = 0.04 * max(0.0, (radius_ratio - 5.0) / 5.0) ** 2
    distinction = 0.0
    if case.case_id == "three_openings":
        target_order = (1, 0, 2)
        actual_order = tuple(sorted(range(3), key=lambda index: ratios[index]))
        distinction = 0.5 if actual_order != target_order else 0.0
    return contrast + budget + angle_penalty + ratio_penalty + distinction


def evaluate(case: Case, position: Sequence[float], brep_path: Path, workspace: Path, keep: bool = False) -> Evaluation:
    candidate_key = "_".join(f"{value:.5f}" for value in position)
    candidate_root = workspace / candidate_key
    candidate_root.mkdir(parents=True, exist_ok=True)
    msh_path = candidate_root / "candidate.msh"
    geo_path = candidate_root / "candidate.geo"
    entity_digest_before = sha256(brep_path.read_bytes()).hexdigest()
    try:
        geo_path.write_text(candidate_source(case, position, brep_path, msh_path), encoding="utf-8")
        run_gmsh(geo_path)
        entity_digest_after = sha256(brep_path.read_bytes()).hexdigest()
        if entity_digest_after != entity_digest_before:
            raise RuntimeError("BREP digest changed during candidate meshing")
        points, triangles, groups = parse_msh2(msh_path)
        audit = audit_mesh(
            points,
            triangles,
            expected_components=expected_components(case),
            expected_area=exact_area(case),
            expected_hole_areas=expected_hole_areas(case),
            area_tolerance=0.01,
            hole_tolerance=0.02,
            angle_limit=7.0,
            radius_ratio_limit=14.0,
        )
        if not audit.ok:
            raise RuntimeError("; ".join(audit.issues))
        seam: dict[str, object] = {}
        output_points = points
        output_triangles = triangles
        if case.case_id == "cracked_web":
            output_points, output_triangles, seam = create_crack_seam(points, triangles, groups.get("CRACK_GUIDE", ()))
        records = equivalent_records(points, triangles)
        density, ratios = density_metrics(case, position, records)
        score = score_candidate(
            case,
            ratios,
            len(triangles),
            audit.minimum_angle_deg,
            audit.maximum_radius_ratio,
        )
        result = Evaluation(
            True,
            score,
            tuple(float(value) for value in position),
            len(output_points),
            len(output_triangles),
            audit.minimum_angle_deg,
            audit.maximum_radius_ratio,
            audit.relative_area_error,
            entity_digest_before,
            density,
            seam,
            (),
        )
        if keep:
            (candidate_root / "evaluation.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except Exception as error:
        return Evaluation(
            False,
            1.0e6,
            tuple(float(value) for value in position),
            0,
            0,
            0.0,
            float("inf"),
            float("inf"),
            entity_digest_before,
            {},
            {},
            (f"{type(error).__name__}: {error}",),
        )
    finally:
        if not keep:
            shutil.rmtree(candidate_root, ignore_errors=True)


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def seeded_positions(case: Case, particles: int, random: Random) -> list[list[float]]:
    midpoint = [(lower + upper) * 0.5 for lower, upper in zip(case.lower, case.upper)]
    aggressive = [lower + 0.18 * (upper - lower) for lower, upper in zip(case.lower, case.upper)]
    conservative = [lower + 0.72 * (upper - lower) for lower, upper in zip(case.lower, case.upper)]
    positions = [midpoint, aggressive, conservative]
    while len(positions) < particles:
        positions.append([random.uniform(lower, upper) for lower, upper in zip(case.lower, case.upper)])
    return positions[:particles]


def run_pso(
    case: Case,
    brep_path: Path,
    output_root: Path,
    particles: int,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    random = Random(seed)
    positions = seeded_positions(case, particles, random)
    velocities = [
        [random.uniform(-0.05, 0.05) * (upper - lower) for lower, upper in zip(case.lower, case.upper)]
        for _ in range(particles)
    ]
    personal_positions = [list(position) for position in positions]
    personal_scores = [float("inf")] * particles
    global_position = list(positions[0])
    global_score = float("inf")
    history: list[dict[str, object]] = []
    cache: dict[tuple[float, ...], Evaluation] = {}
    workspace = output_root / "candidates"
    workspace.mkdir(parents=True, exist_ok=True)
    for iteration in range(iterations):
        rows: list[dict[str, object]] = []
        for particle_index, position in enumerate(positions):
            key = tuple(round(value, 5) for value in position)
            evaluation = cache.get(key)
            if evaluation is None:
                evaluation = evaluate(case, position, brep_path, workspace)
                cache[key] = evaluation
            rows.append({"particle": particle_index, "evaluation": asdict(evaluation)})
            if evaluation.score < personal_scores[particle_index]:
                personal_scores[particle_index] = evaluation.score
                personal_positions[particle_index] = list(position)
            if evaluation.score < global_score:
                global_score = evaluation.score
                global_position = list(position)
        history.append(
            {
                "iteration": iteration + 1,
                "global_best_score": global_score,
                "global_best_position": global_position,
                "particles": rows,
            }
        )
        inertia = 0.78 - 0.34 * iteration / max(1, iterations - 1)
        for particle_index in range(particles):
            for dimension, (lower, upper) in enumerate(zip(case.lower, case.upper)):
                r1 = random.random()
                r2 = random.random()
                velocity = (
                    inertia * velocities[particle_index][dimension]
                    + 1.55 * r1 * (personal_positions[particle_index][dimension] - positions[particle_index][dimension])
                    + 1.55 * r2 * (global_position[dimension] - positions[particle_index][dimension])
                )
                limit = 0.24 * (upper - lower)
                velocity = clip(velocity, -limit, limit)
                velocities[particle_index][dimension] = velocity
                positions[particle_index][dimension] = clip(positions[particle_index][dimension] + velocity, lower, upper)
    best_root = output_root / "best"
    shutil.rmtree(best_root, ignore_errors=True)
    best_root.mkdir(parents=True, exist_ok=True)
    best = evaluate(case, global_position, brep_path, best_root, keep=True)
    if not best.valid:
        raise RuntimeError(f"PSO ended with no valid best candidate: {best.issues}")
    source_root = best_root / "_".join(f"{value:.5f}" for value in global_position)
    for source in source_root.iterdir():
        if source.is_file():
            shutil.copy2(source, best_root / source.name)
    shutil.rmtree(source_root, ignore_errors=True)
    shutil.rmtree(workspace, ignore_errors=True)
    receipt = {
        "schema_version": "entity-first-hotspot-pso/1.0",
        "case": asdict(case),
        "particles": particles,
        "iterations": iterations,
        "actual_unique_gmsh_evaluations": len(cache),
        "seed": seed,
        "best": asdict(best),
        "history": history,
    }
    (output_root / "pso_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def write_svg(case: Case, msh_path: Path, output_path: Path) -> None:
    points, triangles, groups = parse_msh2(msh_path)
    canvas_width = 1400.0 if case.width / case.height > 3.0 else 1000.0
    canvas_height = max(300.0, canvas_width * case.height / case.width)
    padding = 30.0
    scale = min((canvas_width - 2.0 * padding) / case.width, (canvas_height - 2.0 * padding) / case.height)
    def transform(point: Point) -> tuple[float, float]:
        return padding + point[0] * scale, canvas_height - padding - point[1] * scale
    document = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width:.0f}" height="{canvas_height:.0f}" viewBox="0 0 {canvas_width:.3f} {canvas_height:.3f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<title>{html.escape(case.case_id)} hotspot PSO mesh</title>',
    ]
    for triangle in triangles:
        transformed = [transform(points[index]) for index in triangle]
        coordinates = " ".join(f"{x_value:.3f},{y_value:.3f}" for x_value, y_value in transformed)
        document.append(f'<polygon points="{coordinates}" fill="none" stroke="#5f6368" stroke-width="0.45"/>')
    if case.case_id == "cracked_web":
        x0, y0 = transform((70.0, 100.0))
        x1, y1 = transform((130.0, 100.0))
        document.append(f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" stroke="black" stroke-width="2.0"/>')
    document.append("</svg>")
    output_path.write_text("\n".join(document) + "\n", encoding="utf-8")


def optimizer_self_test() -> None:
    random = Random(7)
    position = [4.0, -3.0]
    velocity = [0.0, 0.0]
    personal = list(position)
    global_best = [1.0, 2.0]
    for _ in range(20):
        for dimension in range(2):
            velocity[dimension] = 0.6 * velocity[dimension] + 1.5 * random.random() * (global_best[dimension] - position[dimension])
            position[dimension] += velocity[dimension]
    assert hypot(position[0] - global_best[0], position[1] - global_best[1]) < 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic hotspot PSO on fixed Gmsh entities")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--case", choices=sorted(CASES))
    parser.add_argument("--entity-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--particles", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.self_test:
        optimizer_self_test()
        print("hotspot PSO algorithm self-test passed")
        return 0
    if not args.case or not args.entity_root or not args.output_dir:
        parser.error("--case, --entity-root, and --output-dir are required")
    if args.particles < 4 or args.iterations < 2:
        parser.error("particles must be >=4 and iterations must be >=2")
    case = CASES[args.case]
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    brep_path = entity_path(case, Path(args.entity_root).resolve())
    receipt = run_pso(case, brep_path, output_root, args.particles, args.iterations, args.seed)
    write_svg(case, output_root / "best" / "candidate.msh", output_root / "best" / "hotspot_mesh.svg")
    print(json.dumps({"case": case.case_id, "best": receipt["best"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
