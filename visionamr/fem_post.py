"""Post-processing of CPS3 / C3D4 solutions.

Element stresses, strain energy, and the quantity of interest are
reconstructed from the FRD displacement field (exact for linear
simplices), keeping error metrics solver-independent.

Voigt order: 2-D [xx, yy, xy]; 3-D [xx, yy, zz, xy, yz, zx]
(engineering shear strains).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Material, Problem
from .mesher import Mesh


def elastic_C(mat: Material, dim: int) -> np.ndarray:
    if dim == 2:  # plane stress
        f = mat.E / (1.0 - mat.nu**2)
        return f * np.array(
            [[1.0, mat.nu, 0.0], [mat.nu, 1.0, 0.0], [0.0, 0.0, (1.0 - mat.nu) / 2.0]]
        )
    lam = mat.E * mat.nu / ((1.0 + mat.nu) * (1.0 - 2.0 * mat.nu))
    mu = mat.E / (2.0 * (1.0 + mat.nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2.0 * mu
    C[3:, 3:] = np.diag([mu, mu, mu])
    return C


def shape_gradients(mesh: Mesh) -> np.ndarray:
    """Gradients of the linear shape functions: (m, dim+1, dim)."""

    p = mesh.nodes[mesh.cells]
    m = mesh.n_cells
    if mesh.dim == 2:
        J = np.stack(
            [p[:, 1, :2] - p[:, 0, :2], p[:, 2, :2] - p[:, 0, :2]], axis=1
        )  # (m, 2, 2)
    else:
        J = np.stack(
            [p[:, 1, :3] - p[:, 0, :3], p[:, 2, :3] - p[:, 0, :3], p[:, 3, :3] - p[:, 0, :3]],
            axis=1,
        )
    Jinv = np.linalg.inv(J)
    d = mesh.dim
    ref_grads = np.vstack([-np.ones((1, d)), np.eye(d)])  # (d+1, d)
    # grad_x N = grad_xi N @ (dx/dxi)^-1 with dx/dxi = J^T (J rows are edges)
    return np.einsum("nd,mkd->mnk", ref_grads, Jinv)


def element_B(mesh: Mesh) -> np.ndarray:
    """Strain-displacement matrices: (m, n_voigt, (dim+1)*dim)."""

    g = shape_gradients(mesh)  # (m, d+1, d)
    m = mesh.n_cells
    if mesh.dim == 2:
        B = np.zeros((m, 3, 6))
        for i in range(3):
            gx, gy = g[:, i, 0], g[:, i, 1]
            B[:, 0, 2 * i] = gx
            B[:, 1, 2 * i + 1] = gy
            B[:, 2, 2 * i] = gy
            B[:, 2, 2 * i + 1] = gx
        return B
    B = np.zeros((m, 6, 12))
    for i in range(4):
        gx, gy, gz = g[:, i, 0], g[:, i, 1], g[:, i, 2]
        c = 3 * i
        B[:, 0, c] = gx
        B[:, 1, c + 1] = gy
        B[:, 2, c + 2] = gz
        B[:, 3, c] = gy
        B[:, 3, c + 1] = gx
        B[:, 4, c + 1] = gz
        B[:, 4, c + 2] = gy
        B[:, 5, c] = gz
        B[:, 5, c + 2] = gx
    return B


@dataclass
class PostState:
    """Everything downstream methods need from one solve."""

    mesh: Mesh
    u: np.ndarray             # (n, 3)
    stress: np.ndarray        # (m, n_voigt)
    strain: np.ndarray        # (m, n_voigt)
    vm_elem: np.ndarray       # (m,)
    vm_node: np.ndarray       # (n,)
    energy_elem: np.ndarray   # (m,)
    U_total: float
    qoi: float                # measure-weighted mean |u| on the QoI facets


def von_mises(stress: np.ndarray, dim: int) -> np.ndarray:
    if dim == 2:
        sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
        return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
    sxx, syy, szz, sxy, syz, szx = (stress[:, k] for k in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
    )


def compute_post(mesh: Mesh, problem: Problem, u: np.ndarray) -> PostState:
    mat = problem.material
    d = mesh.dim
    C = elastic_C(mat, d)
    B = element_B(mesh)
    ue = u[:, :d][mesh.cells].reshape(mesh.n_cells, (d + 1) * d)
    strain = np.einsum("mij,mj->mi", B, ue)
    stress = strain @ C.T
    scale = mat.thickness if d == 2 else 1.0
    energy = 0.5 * np.einsum("mi,mi->m", stress, strain) * mesh.measures * scale
    vm_e = von_mises(stress, d)

    vm_n = np.zeros(mesh.n_nodes)
    wsum = np.zeros(mesh.n_nodes)
    for k in range(d + 1):
        np.add.at(vm_n, mesh.cells[:, k], vm_e * mesh.measures)
        np.add.at(wsum, mesh.cells[:, k], mesh.measures)
    wsum[wsum == 0] = 1.0
    vm_n /= wsum

    qoi = _facet_qoi(mesh, problem, u)
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


def _facet_qoi(mesh: Mesh, problem: Problem, u: np.ndarray) -> float:
    bf = mesh.boundary_facets
    mask = problem.qoi_facet_predicate(mesh.facet_centroids)
    if not mask.any():
        return float(np.linalg.norm(u, axis=1).max())
    w = mesh.facet_measures[mask]
    umag = np.linalg.norm(u, axis=1)
    facet_mean = umag[bf[mask]].mean(axis=1)
    return float(np.sum(facet_mean * w) / np.sum(w))
