"""CalculiX compatibility helpers for drawing-derived TQZ Gmsh meshes.

Two input issues are normalized without changing the physical model:

1. OpenCASCADE/Gmsh tags may be sparse, so node/element labels are remapped to
   dense 1-based ranges and connectivity follows the same mapping.
2. Geometry operations may leave coordinate noise around zero with long
   scientific strings.  CalculiX 2.21 can reject those node cards, so values
   negligible at the millimetre scale are written as zero with compact fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from meshpilot_tqz_backend import SupportMesh, TQZCase, TQZMaterial


def renumber_support_mesh(mesh: SupportMesh) -> SupportMesh:
    if not mesh.nodes:
        raise ValueError("mesh has no nodes")
    if not mesh.tetrahedra:
        raise ValueError("mesh has no tetrahedra")

    node_map = {
        old_id: new_id
        for new_id, old_id in enumerate(sorted(mesh.nodes), start=1)
    }
    nodes = {
        node_map[old_id]: tuple(float(value) for value in mesh.nodes[old_id])
        for old_id in sorted(mesh.nodes)
    }
    tetrahedra: Dict[int, Tuple[int, int, int, int]] = {}
    for new_element_id, old_element_id in enumerate(
        sorted(mesh.tetrahedra), start=1
    ):
        connectivity = mesh.tetrahedra[old_element_id]
        try:
            tetrahedra[new_element_id] = tuple(
                node_map[node_id] for node_id in connectivity
            )  # type: ignore[assignment]
        except KeyError as exc:
            raise ValueError(
                f"tetrahedron {old_element_id} references missing node {exc.args[0]}"
            ) from exc

    if max(nodes) != len(nodes):
        raise RuntimeError("node labels are not dense after normalization")
    if max(tetrahedra) != len(tetrahedra):
        raise RuntimeError("element labels are not dense after normalization")
    return SupportMesh(nodes=nodes, tetrahedra=tetrahedra)


def compact_number(value: float, *, zero_tolerance: float = 1.0e-9) -> str:
    number = float(value)
    if abs(number) < zero_tolerance:
        return "0"
    text = f"{number:.12g}"
    if len(text) > 20:
        text = f"{number:.10e}"
    if len(text) > 20:
        raise ValueError(f"numeric field is too long for CalculiX: {text}")
    return text


def _format_id_lines(values: Iterable[int], per_line: int = 12) -> list[str]:
    ids = [int(value) for value in values]
    return [
        ", ".join(str(value) for value in ids[start : start + per_line])
        for start in range(0, len(ids), per_line)
    ]


def write_calculix_deck(
    filepath: Path,
    case: TQZCase,
    material: TQZMaterial,
    mesh: SupportMesh,
    fixed_nodes: Sequence[int],
    loaded_nodes: Sequence[int],
    loads: Mapping[int, Tuple[float, float, float]],
) -> None:
    node_count = len(mesh.nodes)
    element_count = len(mesh.tetrahedra)
    if max(mesh.nodes, default=0) != node_count:
        raise RuntimeError("CalculiX deck requires dense node labels 1..N")
    if max(mesh.tetrahedra, default=0) != element_count:
        raise RuntimeError("CalculiX deck requires dense element labels 1..E")
    if any(node_id < 1 or node_id > node_count for node_id in fixed_nodes):
        raise RuntimeError("fixed set contains an undefined node")
    if any(node_id < 1 or node_id > node_count for node_id in loaded_nodes):
        raise RuntimeError("loaded set contains an undefined node")

    lines = [
        "*HEADING",
        f"MeshPilot TQZ local support benchmark: {case.case_id}",
        "*NODE",
    ]
    for node_id, (x, y, z) in sorted(mesh.nodes.items()):
        lines.append(
            f"{node_id}, {compact_number(x)}, {compact_number(y)}, {compact_number(z)}"
        )
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=EALL")
    for element_id, connectivity in sorted(mesh.tetrahedra.items()):
        lines.append(
            f"{element_id}, {connectivity[0]}, {connectivity[1]}, "
            f"{connectivity[2]}, {connectivity[3]}"
        )
    lines.extend(["*NSET, NSET=FIXED", *_format_id_lines(fixed_nodes)])
    lines.extend(["*NSET, NSET=LOADED", *_format_id_lines(loaded_nodes)])
    lines.extend(
        [
            "*MATERIAL, NAME=CONCRETE",
            "*ELASTIC",
            f"{compact_number(material.young_modulus_mpa)}, {compact_number(material.poisson_ratio)}",
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
            lines.append(f"{node_id}, 1, {compact_number(fx)}")
        if abs(fy) > 0.0:
            lines.append(f"{node_id}, 2, {compact_number(fy)}")
        if abs(fz) > 0.0:
            lines.append(f"{node_id}, 3, {compact_number(fz)}")
    lines.extend(["*NODE FILE", "U", "*END STEP"])
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
