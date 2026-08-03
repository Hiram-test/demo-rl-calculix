"""State-aware graph DQN for adaptive finite-element mesh control.

The original project chooses one global candidate ``(cell, action)`` per step.
This module makes the Bellman update consistent with that policy: the next-state
bootstrap is the maximum over *all valid cell-action pairs*, not only over the
same cell that was selected in the current state.

The network also mixes local GCN embeddings with graph-level pooled context and
explicit global/task features.  Therefore a change elsewhere in the mesh,
the remaining resource budget, or the engineering objective can change every
candidate Q-value.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


REFINE = 0
COARSEN = 1
KEEP = 2
ACTION_NAMES = {REFINE: "refine", COARSEN: "coarsen", KEEP: "keep"}


try:  # Prefer PyG when it is installed, but keep the code runnable without it.
    from torch_geometric.nn import GCNConv as _PyGGCNConv  # type: ignore
except Exception:  # pragma: no cover - exercised on installations without PyG.
    _PyGGCNConv = None


class NativeGCNConv(nn.Module):
    """Small, dependency-free implementation of normalized GCN convolution.

    It implements ``D^{-1/2} (A + I) D^{-1/2} X W`` with ``index_add_``.
    The fallback is intentionally simple; existing installations can continue
    to use :class:`torch_geometric.nn.GCNConv` automatically.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must have shape [N, F], got {tuple(x.shape)}")
        num_nodes = x.shape[0]
        if num_nodes == 0:
            return x.new_zeros((0, self.linear.out_features))

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
            )

        device = x.device
        loops = torch.arange(num_nodes, device=device, dtype=torch.long)
        loop_edges = torch.stack([loops, loops], dim=0)
        edges = torch.cat([edge_index.to(device=device, dtype=torch.long), loop_edges], dim=1)

        # edge_index stores source -> destination.
        source, destination = edges[0], edges[1]
        degree = x.new_zeros(num_nodes)
        degree.index_add_(0, destination, torch.ones_like(destination, dtype=x.dtype))
        degree_inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
        norm = degree_inv_sqrt[source] * degree_inv_sqrt[destination]

        messages = x[source] * norm.unsqueeze(-1)
        aggregated = x.new_zeros(x.shape)
        aggregated.index_add_(0, destination, messages)
        return self.linear(aggregated) + self.bias


def make_gcn_layer(in_channels: int, out_channels: int) -> nn.Module:
    """Create a non-cached GCN layer suitable for a changing mesh graph."""

    if _PyGGCNConv is not None:
        return _PyGGCNConv(
            in_channels,
            out_channels,
            cached=False,
            add_self_loops=True,
            normalize=True,
        )
    return NativeGCNConv(in_channels, out_channels)


@dataclass(frozen=True)
class GraphState:
    """Immutable tensor snapshot of one mesh-decision state."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    global_features: torch.Tensor
    action_mask: torch.Tensor
    cell_ids: Tuple[int, ...]

    def validate(self, num_actions: int) -> None:
        if self.node_features.ndim != 2:
            raise ValueError("node_features must have shape [N, F]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.global_features.ndim != 2 or self.global_features.shape[0] != 1:
            raise ValueError("global_features must have shape [1, G]")
        expected_mask_shape = (self.node_features.shape[0], num_actions)
        if tuple(self.action_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"action_mask must have shape {expected_mask_shape}, "
                f"got {tuple(self.action_mask.shape)}"
            )
        if len(self.cell_ids) != self.node_features.shape[0]:
            raise ValueError("cell_ids length must equal the number of graph nodes")
        if self.edge_index.numel() > 0:
            if int(self.edge_index.min()) < 0:
                raise ValueError("edge_index contains a negative node index")
            if int(self.edge_index.max()) >= self.node_features.shape[0]:
                raise ValueError("edge_index refers to a node outside the graph")

    def snapshot(self) -> "GraphState":
        """Return a detached CPU copy safe for replay-buffer storage."""

        return GraphState(
            node_features=self.node_features.detach().clone().cpu(),
            edge_index=self.edge_index.detach().clone().cpu(),
            global_features=self.global_features.detach().clone().cpu(),
            action_mask=self.action_mask.detach().clone().bool().cpu(),
            cell_ids=tuple(int(cell_id) for cell_id in self.cell_ids),
        )

    def to(self, device: torch.device | str) -> "GraphState":
        return GraphState(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            global_features=self.global_features.to(device),
            action_mask=self.action_mask.to(device),
            cell_ids=self.cell_ids,
        )


@dataclass(frozen=True)
class Transition:
    state: GraphState
    action_node: int
    action_type: int
    reward: float
    next_state: GraphState
    done: bool
    cell_id: int
    n_steps: int = 1


class ReplayBufferV2:
    """Ring buffer that stores immutable graph snapshots."""

    VERSION = 2

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.storage: List[Transition] = []
        self.ptr = 0

    def add(
        self,
        state: GraphState,
        action_node: int,
        action_type: int,
        reward: float,
        next_state: GraphState,
        done: bool,
        cell_id: Optional[int] = None,
        n_steps: int = 1,
    ) -> None:
        state_snapshot = state.snapshot()
        next_snapshot = next_state.snapshot()
        if not 0 <= action_node < len(state_snapshot.cell_ids):
            raise IndexError(f"action_node {action_node} is outside the current graph")
        if cell_id is None:
            cell_id = state_snapshot.cell_ids[action_node]
        item = Transition(
            state=state_snapshot,
            action_node=int(action_node),
            action_type=int(action_type),
            reward=float(reward),
            next_state=next_snapshot,
            done=bool(done),
            cell_id=int(cell_id),
            n_steps=max(1, int(n_steps)),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        if not self.storage:
            return []
        return random.sample(self.storage, k=min(int(batch_size), len(self.storage)))

    def __len__(self) -> int:
        return len(self.storage)

    def save(self, filepath: str) -> None:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "version": self.VERSION,
                "capacity": self.capacity,
                "storage": self.storage,
                "ptr": self.ptr,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        version = int(checkpoint.get("version", 1))
        if version != self.VERSION:
            raise RuntimeError(
                "The replay-buffer format changed in V2. Start with a clean buffer "
                f"or migrate it explicitly; found version {version}."
            )
        self.capacity = int(checkpoint["capacity"])
        self.storage = list(checkpoint["storage"])
        self.ptr = int(checkpoint["ptr"])


def _coerce_feature_vector(value: Any) -> List[float]:
    if isinstance(value, torch.Tensor):
        return [float(x) for x in value.detach().cpu().view(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if value is None:
        return [0.0]
    return [float(value)]


def build_graph_state(
    cell_observations: Mapping[int, Mapping[str, Any]],
    cell_adjacency: Mapping[int, Iterable[int]],
    global_features: Sequence[float] | torch.Tensor,
    action_mask: Optional[Mapping[int, Sequence[bool]] | torch.Tensor] = None,
    num_actions: int = 2,
) -> GraphState:
    """Convert environment dictionaries into a validated graph snapshot."""

    cell_ids = tuple(sorted(int(cell_id) for cell_id in cell_observations.keys()))
    if not cell_ids:
        if isinstance(global_features, torch.Tensor):
            global_tensor = global_features.detach().clone().float().reshape(1, -1)
        else:
            global_tensor = torch.tensor(list(global_features), dtype=torch.float32).reshape(1, -1)
        return GraphState(
            node_features=torch.zeros((0, 1), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            global_features=global_tensor,
            action_mask=torch.zeros((0, num_actions), dtype=torch.bool),
            cell_ids=(),
        )

    feature_rows = [
        _coerce_feature_vector(cell_observations[cell_id].get("self", []))
        for cell_id in cell_ids
    ]
    feature_dim = max(len(row) for row in feature_rows)
    feature_rows = [row + [0.0] * (feature_dim - len(row)) for row in feature_rows]
    node_features = torch.tensor(feature_rows, dtype=torch.float32)

    index_by_cell = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    edges: set[Tuple[int, int]] = set()
    for cell_id in cell_ids:
        source = index_by_cell[cell_id]
        for neighbor in cell_adjacency.get(cell_id, []):
            try:
                neighbor_id = int(neighbor)
            except (TypeError, ValueError):
                continue
            if neighbor_id not in index_by_cell:
                continue
            destination = index_by_cell[neighbor_id]
            edges.add((source, destination))
            edges.add((destination, source))
    if edges:
        edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    if isinstance(global_features, torch.Tensor):
        global_tensor = global_features.detach().clone().float().reshape(1, -1)
    else:
        global_tensor = torch.tensor(list(global_features), dtype=torch.float32).reshape(1, -1)

    if action_mask is None:
        mask_tensor = torch.ones((len(cell_ids), num_actions), dtype=torch.bool)
    elif isinstance(action_mask, torch.Tensor):
        mask_tensor = action_mask.detach().clone().bool()
    else:
        rows: List[List[bool]] = []
        for cell_id in cell_ids:
            row = list(bool(value) for value in action_mask.get(cell_id, [True] * num_actions))
            if len(row) != num_actions:
                raise ValueError(
                    f"Action mask for cell {cell_id} has {len(row)} entries; "
                    f"expected {num_actions}."
                )
            rows.append(row)
        mask_tensor = torch.tensor(rows, dtype=torch.bool)

    state = GraphState(
        node_features=node_features,
        edge_index=edge_index,
        global_features=global_tensor,
        action_mask=mask_tensor,
        cell_ids=cell_ids,
    )
    state.validate(num_actions)
    return state


class GraphDuelingQNetwork(nn.Module):
    """Graph-conditioned dueling Q network for global cell-action candidates."""

    def __init__(
        self,
        feat_dim: int,
        global_dim: int,
        hidden_dim: int = 96,
        num_actions: int = 2,
        num_gcn_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_gcn_layers < 1:
            raise ValueError("num_gcn_layers must be at least one")
        self.feat_dim = int(feat_dim)
        self.global_dim = int(global_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_actions = int(num_actions)
        self.dropout = float(dropout)

        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        in_dim = self.feat_dim
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(make_gcn_layer(in_dim, self.hidden_dim))
            self.norm_layers.append(nn.LayerNorm(self.hidden_dim))
            in_dim = self.hidden_dim

        # graph_context = mean pooling + max pooling + explicit global features
        context_dim = 2 * self.hidden_dim + self.global_dim
        self.value_head = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(self.hidden_dim + context_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions),
        )

    def encode(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = node_features
        for index, (gcn, norm) in enumerate(zip(self.gcn_layers, self.norm_layers)):
            updated = norm(gcn(x, edge_index))
            updated = F.relu(updated)
            if index > 0 and updated.shape == x.shape:
                updated = updated + x
            x = F.dropout(updated, p=self.dropout, training=self.training)

        if x.shape[0] == 0:
            context = torch.cat(
                [
                    x.new_zeros((1, self.hidden_dim)),
                    x.new_zeros((1, self.hidden_dim)),
                    global_features,
                ],
                dim=-1,
            )
            return x, context

        pooled_mean = x.mean(dim=0, keepdim=True)
        pooled_max = x.max(dim=0, keepdim=True).values
        context = torch.cat([pooled_mean, pooled_max, global_features], dim=-1)
        return x, context

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        x, context = self.encode(node_features, edge_index, global_features)
        if x.shape[0] == 0:
            return x.new_zeros((0, self.num_actions))

        state_value = self.value_head(context)  # [1, 1]
        repeated_context = context.expand(x.shape[0], -1)
        advantages = self.advantage_head(torch.cat([x, repeated_context], dim=-1))

        # One global action is selected from all cell-action pairs, so centre the
        # advantages over that complete candidate set rather than per cell.
        centred_advantages = advantages - advantages.mean()
        return state_value.expand_as(centred_advantages) + centred_advantages


def masked_flat_argmax(
    q_values: torch.Tensor,
    action_mask: torch.Tensor,
) -> Optional[Tuple[int, int, float]]:
    """Return the best valid ``(node, action, value)`` over the whole graph."""

    if q_values.shape != action_mask.shape:
        raise ValueError(
            f"q_values and action_mask must have the same shape, got "
            f"{tuple(q_values.shape)} and {tuple(action_mask.shape)}"
        )
    if q_values.numel() == 0 or not bool(action_mask.any()):
        return None
    masked = q_values.masked_fill(~action_mask, -torch.inf)
    flat_index = int(masked.reshape(-1).argmax().item())
    num_actions = q_values.shape[1]
    node_index = flat_index // num_actions
    action_type = flat_index % num_actions
    return node_index, action_type, float(masked[node_index, action_type].item())


class StateAwareDQNAgent:
    """Double/dueling graph DQN with a global cell-action Bellman target."""

    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        global_dim: int,
        feat_dim: Optional[int] = None,
        hidden_dim: int = 96,
        num_actions: int = 2,
        num_gcn_layers: int = 3,
        device: Optional[str] = None,
        gamma: float = 0.99,
        lr: float = 3e-4,
        dropout: float = 0.0,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feat_dim = feat_dim
        self.global_dim = int(global_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_actions = int(num_actions)
        self.num_gcn_layers = int(num_gcn_layers)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.dropout = float(dropout)

        self._q_net: Optional[GraphDuelingQNetwork] = None
        self._target_net: Optional[GraphDuelingQNetwork] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        if feat_dim is not None:
            self._ensure_networks(int(feat_dim))

    @property
    def q_net(self) -> GraphDuelingQNetwork:
        if self._q_net is None:
            raise RuntimeError("Network has not been initialized from a graph state")
        return self._q_net

    @property
    def target_net(self) -> GraphDuelingQNetwork:
        if self._target_net is None:
            raise RuntimeError("Target network has not been initialized")
        return self._target_net

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            raise RuntimeError("Optimizer has not been initialized")
        return self._optimizer

    def _ensure_networks(self, feat_dim: int) -> None:
        if self._q_net is not None:
            if self.feat_dim != feat_dim:
                raise RuntimeError(
                    f"Node feature dimension changed from {self.feat_dim} to {feat_dim}. "
                    "Use a fixed feature schema rather than reinitializing the Q network."
                )
            return

        self.feat_dim = int(feat_dim)
        kwargs = dict(
            feat_dim=self.feat_dim,
            global_dim=self.global_dim,
            hidden_dim=self.hidden_dim,
            num_actions=self.num_actions,
            num_gcn_layers=self.num_gcn_layers,
            dropout=self.dropout,
        )
        self._q_net = GraphDuelingQNetwork(**kwargs).to(self.device)
        self._target_net = GraphDuelingQNetwork(**kwargs).to(self.device)
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.AdamW(self._q_net.parameters(), lr=self.lr)

    def q_values(self, state: GraphState, target: bool = False) -> torch.Tensor:
        state.validate(self.num_actions)
        self._ensure_networks(state.node_features.shape[1])
        device_state = state.to(self.device)
        network = self.target_net if target else self.q_net
        return network(
            device_state.node_features,
            device_state.edge_index,
            device_state.global_features,
        )

    def select_action(self, state: GraphState, epsilon: float = 0.0) -> Optional[Dict[str, Any]]:
        state.validate(self.num_actions)
        if state.node_features.shape[0] == 0 or not bool(state.action_mask.any()):
            return None
        self._ensure_networks(state.node_features.shape[1])

        valid_pairs = state.action_mask.nonzero(as_tuple=False)
        if random.random() < float(epsilon):
            selected = valid_pairs[random.randrange(valid_pairs.shape[0])]
            node_index = int(selected[0].item())
            action_type = int(selected[1].item())
            q_value: Optional[float] = None
            strategy = "explore"
        else:
            self.q_net.eval()
            with torch.no_grad():
                q_values = self.q_values(state)
                selected = masked_flat_argmax(q_values, state.action_mask.to(self.device))
            if selected is None:
                return None
            node_index, action_type, q_value = selected
            strategy = "exploit"

        return {
            "cell_id": int(state.cell_ids[node_index]),
            "node_index": node_index,
            "action": action_type,
            "action_name": ACTION_NAMES.get(action_type, str(action_type)),
            "q_value": q_value,
            "strategy": strategy,
        }

    @staticmethod
    def select_next_candidate(
        online_q_values: torch.Tensor,
        next_action_mask: torch.Tensor,
    ) -> Optional[Tuple[int, int, float]]:
        """Select the global Double-DQN candidate used by the Bellman target."""

        return masked_flat_argmax(online_q_values, next_action_mask)

    def train_step(
        self,
        replay: ReplayBufferV2,
        batch_size: int = 32,
        target_update_tau: float = 0.01,
    ) -> Optional[float]:
        if len(replay) < max(8, min(int(batch_size), 8)):
            return None
        transitions = replay.sample(batch_size)
        if not transitions:
            return None

        self._ensure_networks(transitions[0].state.node_features.shape[1])
        self.q_net.train()
        losses: List[torch.Tensor] = []

        for transition in transitions:
            state = transition.state.to(self.device)
            next_state = transition.next_state.to(self.device)
            state.validate(self.num_actions)
            next_state.validate(self.num_actions)

            if not bool(state.action_mask[transition.action_node, transition.action_type]):
                # A corrupted or obsolete transition should not affect training.
                continue

            current_q_values = self.q_net(
                state.node_features,
                state.edge_index,
                state.global_features,
            )
            q_selected = current_q_values[
                transition.action_node,
                transition.action_type,
            ]

            with torch.no_grad():
                bootstrap = q_selected.new_zeros(())
                if not transition.done and bool(next_state.action_mask.any()):
                    next_online = self.q_net(
                        next_state.node_features,
                        next_state.edge_index,
                        next_state.global_features,
                    )
                    candidate = self.select_next_candidate(
                        next_online,
                        next_state.action_mask,
                    )
                    if candidate is not None:
                        next_node, next_action, _ = candidate
                        next_target = self.target_net(
                            next_state.node_features,
                            next_state.edge_index,
                            next_state.global_features,
                        )
                        bootstrap = next_target[next_node, next_action]

                discount = self.gamma ** transition.n_steps
                target = q_selected.new_tensor(transition.reward) + discount * bootstrap

            losses.append(F.smooth_l1_loss(q_selected, target))

        if not losses:
            return None

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                self.q_net.parameters(), self.target_net.parameters()
            ):
                target_parameter.mul_(1.0 - target_update_tau)
                target_parameter.add_(target_update_tau * online_parameter)

        return float(loss.item())

    def state_sensitivity(
        self,
        state_a: GraphState,
        state_b: GraphState,
    ) -> Dict[str, float]:
        """Diagnostic used to verify that changed states change Q-values."""

        if state_a.cell_ids != state_b.cell_ids:
            raise ValueError("Sensitivity comparison requires aligned cell IDs")
        self.q_net.eval()
        with torch.no_grad():
            q_a = self.q_values(state_a).detach().cpu()
            q_b = self.q_values(state_b).detach().cpu()
        delta = (q_b - q_a).abs()
        return {
            "mean_abs_q_delta": float(delta.mean().item()),
            "max_abs_q_delta": float(delta.max().item()),
            "fraction_changed": float((delta > 1e-6).float().mean().item()),
        }

    def save_checkpoint(self, filepath: str) -> None:
        if self._q_net is None:
            raise RuntimeError("Cannot save an uninitialized agent")
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "version": self.CHECKPOINT_VERSION,
                "q_net_state_dict": self.q_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "feat_dim": self.feat_dim,
                "global_dim": self.global_dim,
                "hidden_dim": self.hidden_dim,
                "num_actions": self.num_actions,
                "num_gcn_layers": self.num_gcn_layers,
                "gamma": self.gamma,
                "lr": self.lr,
                "dropout": self.dropout,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        version = int(checkpoint.get("version", 1))
        if version != self.CHECKPOINT_VERSION:
            raise RuntimeError(
                "V1 checkpoints use an inconsistent per-cell Bellman target and are "
                "not loaded into the V2 architecture. Start a new V2 run; the old "
                "checkpoint can remain as an experimental baseline."
            )
        if int(checkpoint["global_dim"]) != self.global_dim:
            raise RuntimeError(
                f"Checkpoint global_dim={checkpoint['global_dim']} does not match "
                f"the current schema ({self.global_dim})."
            )

        self.hidden_dim = int(checkpoint["hidden_dim"])
        self.num_actions = int(checkpoint["num_actions"])
        self.num_gcn_layers = int(checkpoint["num_gcn_layers"])
        self.gamma = float(checkpoint["gamma"])
        self.lr = float(checkpoint["lr"])
        self.dropout = float(checkpoint.get("dropout", 0.0))
        self._ensure_networks(int(checkpoint["feat_dim"]))
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
