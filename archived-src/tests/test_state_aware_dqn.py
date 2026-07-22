from __future__ import annotations

import unittest

import torch

from state_aware_dqn_agent import (
    GraphState,
    ReplayBufferV2,
    StateAwareDQNAgent,
    build_graph_state,
    masked_flat_argmax,
)


def make_state(
    node_rows,
    edges,
    global_features=(0.2, 0.8, 0.0),
    mask=None,
    cell_ids=None,
):
    num_nodes = len(node_rows)
    if mask is None:
        mask = torch.ones((num_nodes, 2), dtype=torch.bool)
    if cell_ids is None:
        cell_ids = tuple(range(10, 10 + num_nodes))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return GraphState(
        node_features=torch.tensor(node_rows, dtype=torch.float32),
        edge_index=edge_index,
        global_features=torch.tensor(global_features, dtype=torch.float32).reshape(1, -1),
        action_mask=mask,
        cell_ids=tuple(cell_ids),
    )


class StateAwareDQNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_masked_argmax_is_global_over_cells_and_actions(self):
        q_values = torch.tensor([[1.0, 2.0], [9.0, 3.0], [4.0, 8.0]])
        mask = torch.tensor([[True, True], [False, True], [True, True]])
        selected = masked_flat_argmax(q_values, mask)
        self.assertIsNotNone(selected)
        node, action, value = selected
        self.assertEqual((node, action), (2, 1))
        self.assertEqual(value, 8.0)

    def test_remote_state_change_changes_same_cell_q_values(self):
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        state_a = make_state(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            edges,
        )
        # Cell 0 itself is unchanged. Only a remote cell and one global budget
        # feature change. A state-conditioned policy must still update cell 0 Q.
        state_b = make_state(
            [[0.1, 0.2], [0.3, 0.4], [2.5, -1.2]],
            edges,
            global_features=(0.9, 0.1, 1.0),
        )
        agent = StateAwareDQNAgent(
            feat_dim=2,
            global_dim=3,
            hidden_dim=24,
            num_gcn_layers=2,
            device="cpu",
        )
        with torch.no_grad():
            q_a = agent.q_values(state_a).cpu()
            q_b = agent.q_values(state_b).cpu()
        self.assertGreater(float((q_b[0] - q_a[0]).abs().max()), 1.0e-7)

    def test_replay_buffer_stores_immutable_snapshots(self):
        state = make_state([[1.0, 2.0], [3.0, 4.0]], [(0, 1), (1, 0)])
        next_state = make_state([[2.0, 3.0], [4.0, 5.0]], [(0, 1), (1, 0)])
        replay = ReplayBufferV2(capacity=4)
        replay.add(state, 0, 1, 0.5, next_state, False)

        state.node_features[0, 0] = 999.0
        next_state.global_features[0, 0] = 999.0
        stored = replay.storage[0]
        self.assertEqual(float(stored.state.node_features[0, 0]), 1.0)
        self.assertAlmostEqual(float(stored.next_state.global_features[0, 0]), 0.2, places=6)

    def test_training_accepts_changed_graph_size(self):
        state = make_state([[0.1, 0.2], [0.3, 0.4]], [(0, 1), (1, 0)])
        next_state = make_state(
            [[0.1, 0.2], [0.3, 0.4], [0.7, 0.8]],
            [(0, 1), (1, 0), (1, 2), (2, 1)],
            cell_ids=(10, 11, 12),
        )
        replay = ReplayBufferV2(capacity=16)
        for index in range(8):
            replay.add(
                state,
                action_node=index % 2,
                action_type=index % 2,
                reward=0.1 * index,
                next_state=next_state,
                done=False,
            )

        agent = StateAwareDQNAgent(
            feat_dim=2,
            global_dim=3,
            hidden_dim=16,
            num_gcn_layers=2,
            device="cpu",
        )
        loss = agent.train_step(replay, batch_size=8)
        self.assertIsInstance(loss, float)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_graph_builder_preserves_stable_cell_mapping(self):
        observations = {
            42: {"self": [4.2, 0.0]},
            7: {"self": [0.7, 1.0]},
        }
        state = build_graph_state(
            observations,
            {7: [42], 42: [7]},
            global_features=[0.0, 1.0],
            action_mask={7: [True, False], 42: [False, True]},
        )
        self.assertEqual(state.cell_ids, (7, 42))
        self.assertTrue(torch.allclose(state.node_features, torch.tensor([[0.7, 1.0], [4.2, 0.0]])))
        self.assertEqual(state.action_mask.tolist(), [[True, False], [False, True]])


if __name__ == "__main__":
    unittest.main()
