from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class Mesh:
    nodes: np.ndarray
    elements: np.ndarray
    edge_sets: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, float | str | int] = field(default_factory=dict)

    def copy(self) -> "Mesh":
        return Mesh(
            nodes=self.nodes.copy(),
            elements=self.elements.copy(),
            edge_sets={k: list(v) for k, v in self.edge_sets.items()},
            node_sets={k: v.copy() for k, v in self.node_sets.items()},
            metadata=dict(self.metadata),
        )


@dataclass
class Solution:
    displacements: np.ndarray
    reactions: np.ndarray
    element_centers: np.ndarray
    element_strain: np.ndarray
    element_stress: np.ndarray
    element_von_mises: np.ndarray
    nodal_stress: np.ndarray
    nodal_von_mises: np.ndarray
    stiffness: csr_matrix
    load_vector: np.ndarray
    strain_energy: float
    external_half_work: float
    energy_balance_rel: float


def plane_stress_matrix(young: float, poisson: float) -> np.ndarray:
    fac = young / (1.0 - poisson**2)
    return fac * np.array(
        [
            [1.0, poisson, 0.0],
            [poisson, 1.0, 0.0],
            [0.0, 0.0, (1.0 - poisson) / 2.0],
        ],
        dtype=float,
    )


def shape_derivatives(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.array(
        [
            [-(1.0 - eta), +(1.0 - eta), +(1.0 + eta), -(1.0 + eta)],
            [-(1.0 - xi), -(1.0 + xi), +(1.0 + xi), +(1.0 - xi)],
        ],
        dtype=float,
    )


def element_B(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    dnat = shape_derivatives(xi, eta)
    jac = dnat @ coords
    det = float(np.linalg.det(jac))
    if det <= 0.0:
        raise ValueError(f"non-positive element Jacobian: {det}")
    grad = np.linalg.solve(jac, dnat)
    B = np.zeros((3, 8), dtype=float)
    for i in range(4):
        dnx, dny = grad[:, i]
        B[0, 2 * i] = dnx
        B[1, 2 * i + 1] = dny
        B[2, 2 * i] = dny
        B[2, 2 * i + 1] = dnx
    return B, det


def element_stiffness(coords: np.ndarray, D: np.ndarray, thickness: float) -> np.ndarray:
    ke = np.zeros((8, 8), dtype=float)
    gp = 1.0 / np.sqrt(3.0)
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            B, det = element_B(coords, xi, eta)
            ke += B.T @ D @ B * det * thickness
    return ke


def assemble_stiffness(mesh: Mesh, young: float, poisson: float, thickness: float) -> csr_matrix:
    D = plane_stress_matrix(young, poisson)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for conn in mesh.elements:
        coords = mesh.nodes[conn]
        ke = element_stiffness(coords, D, thickness)
        dofs = np.array([[2 * n, 2 * n + 1] for n in conn], dtype=int).ravel()
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        vals.extend(ke.ravel().tolist())
    ndof = 2 * len(mesh.nodes)
    return coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()


def edge_traction_load(
    mesh: Mesh,
    edges: Iterable[tuple[int, int]],
    traction: tuple[float, float],
    thickness: float,
) -> np.ndarray:
    f = np.zeros(2 * len(mesh.nodes), dtype=float)
    tvec = np.asarray(traction, dtype=float)
    for n1, n2 in edges:
        p1 = mesh.nodes[n1]
        p2 = mesh.nodes[n2]
        length = float(np.linalg.norm(p2 - p1))
        nodal = tvec * thickness * length / 2.0
        f[2 * n1 : 2 * n1 + 2] += nodal
        f[2 * n2 : 2 * n2 + 2] += nodal
    return f


def point_load(mesh: Mesh, node: int, load: tuple[float, float]) -> np.ndarray:
    f = np.zeros(2 * len(mesh.nodes), dtype=float)
    f[2 * node : 2 * node + 2] = np.asarray(load, dtype=float)
    return f


def solve_linear_plane_stress(
    mesh: Mesh,
    young: float,
    poisson: float,
    thickness: float,
    load_vector: np.ndarray,
    constraints: dict[int, float],
) -> Solution:
    K = assemble_stiffness(mesh, young, poisson, thickness)
    ndof = K.shape[0]
    f = np.asarray(load_vector, dtype=float).copy()
    if f.shape != (ndof,):
        raise ValueError(f"load vector has shape {f.shape}, expected {(ndof,)}")

    fixed = np.array(sorted(constraints), dtype=int)
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    u = np.zeros(ndof, dtype=float)
    for dof, value in constraints.items():
        u[dof] = value

    rhs = f[free]
    if len(fixed):
        rhs = rhs - K[free][:, fixed] @ u[fixed]
    u[free] = spsolve(K[free][:, free], rhs)
    reactions = K @ u - f

    D = plane_stress_matrix(young, poisson)
    ne = len(mesh.elements)
    centers = np.zeros((ne, 2), dtype=float)
    strains = np.zeros((ne, 3), dtype=float)
    stresses = np.zeros((ne, 3), dtype=float)
    vm = np.zeros(ne, dtype=float)
    nodal_acc = np.zeros((len(mesh.nodes), 3), dtype=float)
    nodal_count = np.zeros(len(mesh.nodes), dtype=float)

    for eid, conn in enumerate(mesh.elements):
        coords = mesh.nodes[conn]
        centers[eid] = coords.mean(axis=0)
        B, _ = element_B(coords, 0.0, 0.0)
        dofs = np.array([[2 * n, 2 * n + 1] for n in conn], dtype=int).ravel()
        strain = B @ u[dofs]
        stress = D @ strain
        strains[eid] = strain
        stresses[eid] = stress
        sx, sy, txy = stress
        vm[eid] = np.sqrt(max(sx * sx - sx * sy + sy * sy + 3.0 * txy * txy, 0.0))
        for n in conn:
            nodal_acc[n] += stress
            nodal_count[n] += 1.0

    nodal_count[nodal_count == 0.0] = 1.0
    nodal_stress = nodal_acc / nodal_count[:, None]
    sx, sy, txy = nodal_stress.T
    nodal_vm = np.sqrt(np.maximum(sx * sx - sx * sy + sy * sy + 3.0 * txy * txy, 0.0))

    strain_energy = float(0.5 * u @ (K @ u))
    external_half_work = float(0.5 * f @ u)
    denom = max(abs(strain_energy), abs(external_half_work), 1.0e-30)
    energy_balance_rel = abs(strain_energy - external_half_work) / denom

    return Solution(
        displacements=u.reshape((-1, 2)),
        reactions=reactions.reshape((-1, 2)),
        element_centers=centers,
        element_strain=strains,
        element_stress=stresses,
        element_von_mises=vm,
        nodal_stress=nodal_stress,
        nodal_von_mises=nodal_vm,
        stiffness=K,
        load_vector=f,
        strain_energy=strain_energy,
        external_half_work=external_half_work,
        energy_balance_rel=energy_balance_rel,
    )


def nearest_node(mesh: Mesh, point: tuple[float, float]) -> int:
    p = np.asarray(point, dtype=float)
    return int(np.argmin(np.linalg.norm(mesh.nodes - p, axis=1)))


def nearest_element(solution: Solution, point: tuple[float, float]) -> int:
    p = np.asarray(point, dtype=float)
    return int(np.argmin(np.linalg.norm(solution.element_centers - p, axis=1)))


def element_characteristic_size(mesh: Mesh) -> np.ndarray:
    sizes = np.zeros(len(mesh.elements), dtype=float)
    for i, conn in enumerate(mesh.elements):
        pts = mesh.nodes[conn]
        lengths = [np.linalg.norm(pts[(j + 1) % 4] - pts[j]) for j in range(4)]
        sizes[i] = float(np.sqrt(np.prod(sorted(lengths)[:2])))
    return sizes
