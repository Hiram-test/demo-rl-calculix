"""RL baseline: region-graph Double DQN with a GCN encoder.

Positioning (fixed by the study design): unlike element-agent RL-AMR
(ASMR and successors), the agent's graph nodes are the *vision-drawn
regions* -- the same objects the VLA method controls -- plus one
background node.  The region set stays fixed during an episode (only the
VLA method may edit regions).  Each step the policy picks one node whose
size is multiplied by ``gamma`` (refine) or stops; every step costs one
real Gmsh remesh + CalculiX solve.

Reward uses the ZZ indicator total (no oracle reference), minus a
per-solve cost and a budget-violation penalty, so the trained policy has
to trade accuracy against remeshing cost like every other method.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..experiment import FemRunner, initial_mesh
from ..indicators import zz_indicator
from ..mesher import generate_mesh
from ..vla.regions import RegionGraph

FEATURE_DIM = 8


@dataclass
class DQNConfig:
    gamma_refine: float = 0.7      # size multiplier per refine action
    max_steps: int = 8
    n_eq_budget: int = 8000
    discount: float = 0.95
    lr: float = 1e-3
    batch_size: int = 32
    replay_size: int = 4000
    target_sync: int = 200
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.6
    hidden: int = 32
    reward_error_scale: float = 10.0
    reward_solve_cost: float = 0.15
    reward_budget_penalty: float = 2.0
    reward_stop_bonus: float = 1.0
    seed: int = 0


def _build_torch_model(hidden: int):
    import torch
    import torch.nn as nn

    class GCNQ(nn.Module):
        """Two-layer GCN over the region graph + per-node and stop heads."""

        def __init__(self) -> None:
            super().__init__()
            self.lin1 = nn.Linear(FEATURE_DIM, hidden)
            self.lin2 = nn.Linear(hidden, hidden)
            self.node_head = nn.Linear(hidden, 1)
            self.stop_head = nn.Linear(hidden, 1)

        def forward(self, X: "torch.Tensor", A_hat: "torch.Tensor") -> "torch.Tensor":
            h = torch.relu(self.lin1(A_hat @ X))
            h = torch.relu(self.lin2(A_hat @ h))
            q_nodes = self.node_head(h).squeeze(-1)          # (n_nodes,)
            q_stop = self.stop_head(h.mean(dim=0))           # (1,)
            return torch.cat([q_nodes, q_stop])              # refine each node | stop

    return GCNQ()


def _norm_adjacency(A: np.ndarray) -> np.ndarray:
    A = A + np.eye(len(A))
    d = A.sum(axis=1)
    D = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    return D @ A @ D


class RegionRefineEnv:
    """One episode = one problem instance with a fixed region template."""

    def __init__(
        self,
        runner: FemRunner,
        partitioner,
        cfg: DQNConfig,
        *,
        method: str = "rl_dqn",
    ) -> None:
        self.runner = runner
        self.partitioner = partitioner
        self.cfg = cfg
        self.method = method

    # ------------------------------------------------------------------
    def reset(self):
        problem = self.runner.problem
        mesh = initial_mesh(problem)
        post, rec = self.runner.solve_mesh(mesh, method=self.method, stage="probe")
        eta2 = zz_indicator(problem, post)
        regions = self.partitioner.partition(problem, post, eta2)
        self.graph = RegionGraph.build(regions, problem.h0, problem)
        self.h_bg = problem.h0
        self.steps = 0
        self.last_rec = rec
        self._update_state(post, eta2)
        return self.state()

    def _update_state(self, post, eta2) -> None:
        self.post = post
        self.eta2 = eta2
        self.feats = self.graph.features(post, eta2)
        self.log_eta = 0.5 * math.log(max(float(eta2.sum()), 1e-30))

    # ------------------------------------------------------------------
    def state(self) -> tuple[np.ndarray, np.ndarray]:
        """Node features (regions + background) and normalized adjacency."""

        f = self.feats
        problem = self.runner.problem
        R = len(self.graph.regions)
        n_eq = self.last_rec.n_equations
        rows = []
        vm_gmax = max(float(self.post.vm_elem.max()), 1e-12)
        bbox_area = (problem.bbox[2] - problem.bbox[0]) * (
            problem.bbox[3] - problem.bbox[1]
        )
        for i in range(R):
            rows.append(
                [
                    math.log(self.graph.regions[i].h / problem.h0),
                    f.err_share[i],
                    f.elem_share[i],
                    f.vm_max[i] / vm_gmax,
                    f.area[i] / bbox_area,
                    0.0,
                    (self.cfg.max_steps - self.steps) / self.cfg.max_steps,
                    n_eq / self.cfg.n_eq_budget,
                ]
            )
        rows.append(
            [
                math.log(self.h_bg / problem.h0),
                f.bg_err / max(f.total_err, 1e-30),
                f.bg_elems / max(f.total_elems, 1),
                1.0,
                1.0 - float(f.area.sum()) / bbox_area,
                1.0,
                (self.cfg.max_steps - self.steps) / self.cfg.max_steps,
                n_eq / self.cfg.n_eq_budget,
            ]
        )
        X = np.asarray(rows, dtype=np.float32)
        A = self.graph.adjacency_matrix()
        n = len(A)
        Af = np.zeros((n + 1, n + 1))
        Af[:n, :n] = A
        Af[n, :n] = 1.0  # background touches every region
        Af[:n, n] = 1.0
        return X, _norm_adjacency(Af).astype(np.float32)

    @property
    def n_actions(self) -> int:
        return len(self.graph.regions) + 2  # refine each region | refine bg | stop

    # ------------------------------------------------------------------
    def step(self, action: int):
        cfg = self.cfg
        problem = self.runner.problem
        R = len(self.graph.regions)
        if action == R + 1:  # stop
            within = self.last_rec.n_equations <= cfg.n_eq_budget
            reward = cfg.reward_stop_bonus if within else -cfg.reward_budget_penalty
            return self.state(), reward, True, {"stop": True}

        if action == R:
            self.h_bg = max(self.h_bg * cfg.gamma_refine, problem.h_min)
        else:
            sizes = self.graph.sizes()
            sizes[action] = max(sizes[action] * cfg.gamma_refine, problem.h_min)
            self.graph = self.graph.with_sizes(sizes, self.h_bg)
        self.graph = self.graph.with_sizes(self.graph.sizes(), self.h_bg)

        mesh = generate_mesh(problem, self.graph.size_field())
        post, rec = self.runner.solve_mesh(
            mesh, method=self.method, stage=f"step{self.steps + 1}"
        )
        eta2 = zz_indicator(problem, post)
        prev_log_eta = self.log_eta
        self.last_rec = rec
        self.steps += 1
        self._update_state(post, eta2)

        reward = (
            cfg.reward_error_scale * (prev_log_eta - self.log_eta)
            - cfg.reward_solve_cost
        )
        done = self.steps >= cfg.max_steps
        over = rec.n_equations > cfg.n_eq_budget
        if over:
            reward -= cfg.reward_budget_penalty
            done = True
        return self.state(), float(reward), done, {"over_budget": over}


@dataclass
class Transition:
    X: np.ndarray
    A: np.ndarray
    action: int
    reward: float
    X2: np.ndarray
    A2: np.ndarray
    done: bool


class DQNPolicy:
    def __init__(self, cfg: DQNConfig):
        import torch

        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.q = _build_torch_model(cfg.hidden)
        self.q_target = _build_torch_model(cfg.hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay: list[Transition] = []
        self.grad_steps = 0

    # ------------------------------------------------------------------
    def q_values(self, X: np.ndarray, A: np.ndarray, target: bool = False) -> np.ndarray:
        import torch

        model = self.q_target if target else self.q
        with torch.no_grad():
            return model(torch.from_numpy(X), torch.from_numpy(A)).numpy()

    def act(self, X: np.ndarray, A: np.ndarray, eps: float, rng: random.Random) -> int:
        n_actions = X.shape[0] + 1
        if rng.random() < eps:
            return rng.randrange(n_actions)
        return int(np.argmax(self.q_values(X, A)))

    # ------------------------------------------------------------------
    def push(self, t: Transition) -> None:
        self.replay.append(t)
        if len(self.replay) > self.cfg.replay_size:
            self.replay.pop(0)

    def learn(self, rng: random.Random) -> float | None:
        import torch

        cfg = self.cfg
        if len(self.replay) < cfg.batch_size:
            return None
        batch = rng.sample(self.replay, cfg.batch_size)
        losses = []
        self.opt.zero_grad()
        for t in batch:
            q = self.q(torch.from_numpy(t.X), torch.from_numpy(t.A))
            q_sa = q[t.action]
            with torch.no_grad():
                if t.done:
                    target = torch.tensor(t.reward)
                else:
                    q2_online = self.q(torch.from_numpy(t.X2), torch.from_numpy(t.A2))
                    a_star = int(torch.argmax(q2_online))
                    q2_t = self.q_target(torch.from_numpy(t.X2), torch.from_numpy(t.A2))
                    target = t.reward + cfg.discount * q2_t[a_star]
            losses.append(torch.nn.functional.smooth_l1_loss(q_sa, target))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        self.grad_steps += 1
        if self.grad_steps % cfg.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss)

    def save(self, path: Path) -> None:
        import torch

        torch.save(self.q.state_dict(), path)

    def load(self, path: Path) -> None:
        import torch

        sd = torch.load(path, weights_only=True)
        self.q.load_state_dict(sd)
        self.q_target.load_state_dict(sd)


def train_dqn(
    make_instance,
    partitioner,
    workdir: Path,
    *,
    episodes: int = 150,
    cfg: DQNConfig | None = None,
    log_every: int = 10,
) -> tuple[DQNPolicy, list[dict]]:
    """``make_instance(episode) -> Problem`` supplies training instances."""

    cfg = cfg or DQNConfig()
    rng = random.Random(cfg.seed)
    policy = DQNPolicy(cfg)
    history: list[dict] = []
    workdir = Path(workdir)

    for ep in range(episodes):
        problem = make_instance(ep)
        ep_dir = workdir / f"ep{ep:04d}"
        runner = FemRunner(problem, ep_dir)
        env = RegionRefineEnv(runner, partitioner, cfg)
        X, A = env.reset()
        frac = min(ep / max(cfg.eps_decay_frac * episodes, 1), 1.0)
        eps = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac
        total_r, done = 0.0, False
        while not done:
            a = policy.act(X, A, eps, rng)
            (X2, A2), r, done, _ = env.step(a)
            policy.push(Transition(X, A, a, r, X2, A2, done))
            policy.learn(rng)
            X, A = X2, A2
            total_r += r
        history.append(
            {
                "episode": ep,
                "eps": eps,
                "reward": total_r,
                "final_eta": math.exp(env.log_eta),
                "final_neq": env.last_rec.n_equations,
                "solves": env.steps + 1,
            }
        )
        if (ep + 1) % log_every == 0:
            recent = history[-log_every:]
            print(
                f"[dqn] ep {ep + 1}/{episodes} eps={eps:.2f} "
                f"reward={np.mean([h['reward'] for h in recent]):.2f} "
                f"eta={np.mean([h['final_eta'] for h in recent]):.4f}"
            )
        # training solves are not part of any evaluation record
        import shutil

        shutil.rmtree(ep_dir, ignore_errors=True)

    (workdir / "training_history.json").write_text(json.dumps(history, indent=1))
    return policy, history


def evaluate_dqn(
    runner: FemRunner,
    policy: DQNPolicy,
    partitioner,
    *,
    cfg: DQNConfig | None = None,
    method: str = "rl_dqn",
) -> dict:
    """Frozen greedy policy on one instance; every solve is counted."""

    cfg = cfg or policy.cfg
    env = RegionRefineEnv(runner, partitioner, cfg, method=method)
    X, A = env.reset()
    done, steps = False, 0
    while not done:
        a = int(np.argmax(policy.q_values(X, A)))
        (X, A), _, done, info = env.step(a)
        steps += 1
    return {
        "solves": env.steps + 1,
        "final_neq": env.last_rec.n_equations,
        "final_eta": math.exp(env.log_eta),
    }
