"""Supervised baseline: learned one-shot size field from expert meshes.

There is no human expert-mesh corpus, so the expert is manufactured (the
separate data line the study requires): on every training instance the
element-wise ZZ+Doerfler loop is run to a DOF cap and its final mesh
defines the expert nodal size field.  A small MLP learns
log(h_expert/h0) from probe-solve features; at deployment the predicted
field is scaled by one scalar to meet the element budget (the standard
budget knob of MeshingNet/AMBER-style methods) and Gmsh remeshes once.

Deployment cost: 2 global solves (probe + final).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..experiment import FemRunner, initial_mesh
from ..indicators import zz_indicator
from ..mesher import TriMesh, generate_mesh
from ..sizefield import NodalSizeField

N_FEATURES = 8


def node_features(problem, mesh: TriMesh, post, eta2: np.ndarray) -> np.ndarray:
    """Per-node features from a probe solve (instance-agnostic scaling)."""

    xmin, ymin, xmax, ymax = problem.bbox
    diam = math.hypot(xmax - xmin, ymax - ymin)
    x = (mesh.nodes[:, 0] - xmin) / (xmax - xmin)
    y = (mesh.nodes[:, 1] - ymin) / (ymax - ymin)

    vm = post.vm_node
    vm_n = np.log(np.maximum(vm, 1e-12) / max(vm.mean(), 1e-12))

    eta_node = np.zeros(mesh.n_nodes)
    for k in range(3):
        np.add.at(eta_node, mesh.tris[:, k], eta2 / 3.0)
    eta_n = np.log(np.maximum(eta_node, 1e-30) / max(eta_node.mean(), 1e-30))

    clamp_nodes = mesh.nodes[problem.clamp_predicate(mesh.nodes)]
    d_clamp = (
        cKDTree(clamp_nodes).query(mesh.nodes)[0] / diam
        if len(clamp_nodes)
        else np.ones(mesh.n_nodes)
    )
    loads = [f for f in problem.features if f.kind == "load"]
    others = [f for f in problem.features if f.kind in ("hole", "corner")]
    d_load = (
        np.min(
            [np.hypot(mesh.nodes[:, 0] - f.x, mesh.nodes[:, 1] - f.y) for f in loads],
            axis=0,
        )
        / diam
        if loads
        else np.ones(mesh.n_nodes)
    )
    d_feat = (
        np.min(
            [np.hypot(mesh.nodes[:, 0] - f.x, mesh.nodes[:, 1] - f.y) for f in others],
            axis=0,
        )
        / diam
        if others
        else np.ones(mesh.n_nodes)
    )
    h0_n = np.full(mesh.n_nodes, problem.h0 / diam)
    return np.column_stack([x, y, vm_n, eta_n, d_clamp, d_load, d_feat, h0_n]).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# expert dataset


def generate_expert_dataset(
    problems,
    workdir: Path,
    *,
    n_eq_cap: int = 9000,
    theta: float = 0.5,
    max_rounds: int = 10,
) -> Path:
    """Run the classic loop on each instance; label = final nodal sizes."""

    from .dorfler import run_dorfler

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    X_all, y_all = [], []
    meta = []
    for i, problem in enumerate(problems):
        inst_dir = workdir / f"expert_{i:03d}"
        runner = FemRunner(problem, inst_dir)
        runner.ensure_reference()

        mesh0 = initial_mesh(problem)
        post0, _ = runner.solve_mesh(mesh0, method="expert_probe", stage="probe")
        eta2_0 = zz_indicator(problem, post0)
        feats = node_features(problem, mesh0, post0, eta2_0)

        run_dorfler(runner, theta=theta, max_rounds=max_rounds, n_eq_cap=n_eq_cap,
                    method="expert_dorfler")
        expert_mesh: TriMesh = runner.last_mesh  # final Doerfler mesh
        h_expert = expert_mesh.node_sizes
        interp = NodalSizeField(
            expert_mesh, h_expert, gradation=1.2, h_min=problem.h_min, h_max=problem.h0
        )
        labels = np.array(
            [interp(px, py) for px, py in mesh0.nodes], dtype=np.float32
        )
        y = np.log(np.maximum(labels, problem.h_min) / problem.h0).astype(np.float32)
        X_all.append(feats)
        y_all.append(y)
        meta.append({"instance": problem.instance_id, "n_nodes": len(y)})

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    out = workdir / "expert_dataset.npz"
    np.savez_compressed(out, X=X, y=y)
    (workdir / "expert_meta.json").write_text(json.dumps(meta, indent=1))
    return out


# ---------------------------------------------------------------------------
# model


@dataclass
class SupervisedConfig:
    hidden: int = 64
    lr: float = 2e-3
    epochs: int = 300
    batch: int = 4096
    seed: int = 0


class SizeMLP:
    def __init__(self, cfg: SupervisedConfig):
        import torch
        import torch.nn as nn

        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        import torch

        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y).unsqueeze(1)
        n = len(Xt)
        losses = []
        for ep in range(self.cfg.epochs):
            perm = torch.randperm(n)
            ep_loss = 0.0
            for i in range(0, n, self.cfg.batch):
                idx = perm[i : i + self.cfg.batch]
                opt.zero_grad()
                pred = self.net(Xt[idx])
                loss = torch.nn.functional.mse_loss(pred, yt[idx])
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach()) * len(idx)
            losses.append(ep_loss / n)
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.net(torch.from_numpy(X)).squeeze(1).numpy()

    def save(self, path: Path) -> None:
        import torch

        torch.save(self.net.state_dict(), path)

    def load(self, path: Path) -> None:
        import torch

        self.net.load_state_dict(torch.load(path, weights_only=True))


def train_supervised(dataset_path: Path, cfg: SupervisedConfig | None = None) -> SizeMLP:
    cfg = cfg or SupervisedConfig()
    data = np.load(dataset_path)
    model = SizeMLP(cfg)
    losses = model.fit(data["X"], data["y"])
    print(f"[supervised] final train MSE={losses[-1]:.4f} on {len(data['y'])} nodes")
    return model


# ---------------------------------------------------------------------------
# deployment


def deploy_supervised(
    runner: FemRunner,
    model: SizeMLP,
    *,
    n_elem_budget: int,
    method: str = "supervised",
) -> None:
    """Probe solve -> predicted size field -> budget scalar -> one remesh."""

    problem = runner.problem
    runner.ensure_reference()
    mesh0 = initial_mesh(problem)
    post0, rec0 = runner.solve_mesh(mesh0, method=method, stage="probe")
    eta2_0 = zz_indicator(problem, post0)
    feats = node_features(problem, mesh0, post0, eta2_0)
    log_ratio = np.clip(
        model.predict(feats), math.log(problem.h_min / problem.h0), 0.5
    )
    h_pred = problem.h0 * np.exp(log_ratio)

    # one budget scalar c (MeshingNet/AMBER-style): elements ~ sum 2A/h^2,
    # calibrated against what Gmsh actually produced on the probe mesh
    tri_h = h_pred[mesh0.tris].mean(axis=1)
    theory_probe = float(np.sum(2.0 * mesh0.areas / mesh0.tri_sizes**2))
    cal = mesh0.n_tris / max(theory_probe, 1e-12)

    def elems(c: float) -> float:
        return cal * float(
            np.sum(2.0 * mesh0.areas / np.maximum(c * tri_h, problem.h_min) ** 2)
        )

    lo, hi = 0.2, 5.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if elems(mid) > n_elem_budget:
            lo = mid
        else:
            hi = mid
    c = math.sqrt(lo * hi)

    field = NodalSizeField(
        mesh0, c * h_pred, gradation=0.9, h_min=problem.h_min, h_max=problem.h0
    )
    mesh1 = generate_mesh(problem, field)
    _, rec1 = runner.solve_mesh(
        mesh1, method=method, stage="deployed", extra={"budget_scalar": c}
    )
