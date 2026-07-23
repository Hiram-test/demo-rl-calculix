"""Dense-label compatibility for Gmsh meshes consumed by CalculiX.

OpenCASCADE/Gmsh node and element tags are not guaranteed to be contiguous.
CalculiX 2.21 sizes some arrays from the number of records and rejects a deck
when a referenced label is greater than that count.  Normalize all labels and
connectivity before the deck is written.
"""
from __future__ import annotations

from typing import Dict, Tuple

from meshpilot_tqz_backend import SupportMesh


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
