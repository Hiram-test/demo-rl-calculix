"""Post-processing of plane-stress CPS3 solutions.

Element stresses, strain energy, and the quantity of interest are
reconstructed from the FRD displacement field; this keeps the error
metrics solver-independent and exact for linear triangles.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .geometry import Material, Problem
from .mesher import TriMesh


def plane_stress_C(mat: Material) -> np.ndarray:
    f = mat.E / (1.0 - mat.nu**2)
    return f * np.array(
        [[1.0, mat.nu, 0.0], [mat.nu, 1.0, 0.0], [0.0, 0.0, (1.0 - mat.nu) / 2.0]]
    )


def element_B(mesh: TriMesh) -> np.ndarray:
    """Strain-displacement matrices (m, 3, 6) for CST triangles."""

    p = mesh.nodes[mesh.tris]
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    m = len(mesh.tris)
    B = np.zeros((m, 3, 6))
    for i, (b, c) in enumerate(((b1, c1), (b2, c2), (b3, c3))):
        B[:, 0, 2 * i] = b
        B[:, 1, 2 * i + 1] = c
        B[:, 2, 2 * i] = c
        B[:, 2, 2 * i + 1] = b
    B /= A2[:, None, None]
    return B


@dataclass
class PostState:
    """Everything downstream methods need from one solve."""

    mesh: TriMesh
    u: np.ndarray             # (n, 2)
    stress: np.ndarray        # (m, 3) sxx, syy, sxy per element
    strain: np.ndarray        # (m, 3)
    vm_elem: np.ndarray       # (m,)
    vm_node: np.ndarray       # (n,)
    energy_elem: np.ndarray   # (m,) strain energy per element
    U_total: float
    qoi: float                # length-weighted mean |u| on the QoI edge


def von_mises(stress: np.ndarray) -> np.ndarray:
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)


def compute_post(mesh: TriMesh, problem: Problem, u: np.ndarray) -> PostState:
    mat = problem.material
    C = plane_stress_C(mat)
    B = element_B(mesh)
    ue = u[mesh.tris].reshape(len(mesh.tris), 6)
    strain = np.einsum("mij,mj->mi", B, ue)
    stress = strain @ C.T
    energy = 0.5 * np.einsum("mi,mi->m", stress, strain) * mesh.areas * mat.thickness
    vm_e = von_mises(stress)

    # area-weighted nodal von Mises
    vm_n = np.zeros(mesh.n_nodes)
    wsum = np.zeros(mesh.n_nodes)
    for k in range(3):
        np.add.at(vm_n, mesh.tris[:, k], vm_e * mesh.areas)
        np.add.at(wsum, mesh.tris[:, k], mesh.areas)
    wsum[wsum == 0] = 1.0
    vm_n /= wsum

    qoi = _edge_qoi(mesh, problem, u)
    return PostState(
        mesh=mesh,
        u=u,
        stress=stress,
        strain=strain,
        vm_elem=vm_e,
        vm_node=vm_n,
        energy_elem=energy,
        U_total=float(energy.sum()),
        qoi=qoi,
    )


def _edge_qoi(mesh: TriMesh, problem: Problem, u: np.ndarray) -> float:
    be = mesh.boundary_edges
    mids = 0.5 * (mesh.nodes[be[:, 0]] + mesh.nodes[be[:, 1]])
    mask = problem.qoi_edge_predicate(mids)
    if not mask.any():
        return float(np.linalg.norm(u, axis=1).max())
    L = np.linalg.norm(mesh.nodes[be[:, 0]] - mesh.nodes[be[:, 1]], axis=1)[mask]
    umag = np.linalg.norm(u, axis=1)
    edge_mean = 0.5 * (umag[be[mask, 0]] + umag[be[mask, 1]])
    return float(np.sum(edge_mean * L) / np.sum(L))
