from typing import List, Dict, Optional, Tuple, Any
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphQNetwork(nn.Module):
    """
    Graph Convolutional Network for Q-value estimation.
    Processes entire graph structure and outputs Q-values for all nodes simultaneously.

    Inputs:
      - node_features: tensor [N, F] where N is number of nodes (cells), F is feature dim
      - edge_index: tensor [2, E] where E is number of edges
      - global_feat: tensor [1, G] or [N, G] for global features (optional)

    Output:
      - Q-values: tensor [N, A] where A is number of actions (3: refine, coarsen, no-op)
    """

    def __init__(self,
                 feat_dim: int = 3,
                 hidden_dim: int = 64,
                 num_actions: int = 3,
                 global_dim: int = 0,
                 num_gcn_layers: int = 3):
        super().__init__()
        self.num_actions = num_actions
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.global_dim = global_dim
        self.num_gcn_layers = num_gcn_layers

        # GCN layers for graph convolution
        self.gcn_layers = nn.ModuleList()
        # First layer: input is node features
        self.gcn_layers.append(GCNConv(feat_dim, hidden_dim))
        # Intermediate layers
        for _ in range(num_gcn_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        # Last layer: output hidden representation
        if num_gcn_layers > 1:
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Combine node features with global features and output Q-values
        # Input: [hidden_dim + global_dim] -> [num_actions]
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                global_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through graph network.
        
        Args:
            node_features: [N, F] node feature matrix
            edge_index: [2, E] edge connectivity in COO format
            global_feat: [1, G] or [N, G] global features (optional)
        
        Returns:
            q_values: [N, A] Q-values for each node
        """
        # Graph convolution layers
        x = node_features
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
            x = F.relu(x, inplace=True)

        # Add global features if provided
        if global_feat is not None:
            # If global_feat is [1, G], broadcast to [N, G]
            if global_feat.shape[0] == 1:
                global_feat = global_feat.expand(x.shape[0], -1)
            # Concatenate node features with global features
            x = torch.cat([x, global_feat], dim=-1)
        else:
            # If no global features, create zero tensor
            global_feat = torch.zeros((x.shape[0], self.global_dim), 
                                     dtype=x.dtype, device=x.device)
            x = torch.cat([x, global_feat], dim=-1)

        # Output Q-values for each node
        q_values = self.q_head(x)
        return q_values


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        # Storage format: (node_features, edge_index, global_feat, actions, rewards, next_node_features,
        #                  next_edge_index, next_global_feat, done, action_mask)
        # rewards/action_mask are [N] tensors.
        self.storage: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                 torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.ptr = 0

    def add(self,
            node_features: torch.Tensor,  # [N, F]
            edge_index: torch.Tensor,      # [2, E]
            global_feat: torch.Tensor,     # [1, G] or [N, G]
            actions: torch.Tensor,         # [N] actions for all nodes
            reward,                        # float or torch.Tensor [N] - per-cell rewards
            next_node_features: torch.Tensor,  # [N, F]
            next_edge_index: torch.Tensor,     # [2, E]
            next_global_feat: torch.Tensor,    # [1, G] or [N, G]
            done: bool,
            action_mask: Optional[torch.Tensor] = None):
        # Convert reward to tensor if it's a scalar (for backward compatibility)
        if isinstance(reward, (int, float)):
            # Scalar reward: broadcast to all nodes
            reward_tensor = torch.full((node_features.shape[0],), float(reward), dtype=torch.float32)
        elif isinstance(reward, torch.Tensor):
            # Already a tensor, ensure correct shape
            reward_tensor = reward.clone().detach().float()
            if reward_tensor.dim() == 0:
                # Scalar tensor: broadcast to all nodes
                reward_tensor = reward_tensor.expand(node_features.shape[0])
        else:
            raise ValueError(f"reward must be float or torch.Tensor, got {type(reward)}")
        
        if action_mask is None:
            action_mask_tensor = torch.ones(node_features.shape[0], dtype=torch.bool)
        else:
            action_mask_tensor = action_mask.clone().detach().bool()
            if action_mask_tensor.dim() == 0:
                action_mask_tensor = action_mask_tensor.view(1)
            if action_mask_tensor.shape[0] != node_features.shape[0]:
                action_mask_tensor = torch.ones(node_features.shape[0], dtype=torch.bool)

        item = (
            node_features, edge_index, global_feat,
            actions, reward_tensor,  # [N] tensors
            next_node_features, next_edge_index, next_global_feat,
            torch.tensor([done], dtype=torch.bool),
            action_mask_tensor
        )
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int):
        import random
        batch = random.sample(self.storage, k=min(batch_size, len(self.storage)))
        if not batch:
            return tuple()
        transposed = list(zip(*batch))
        if len(transposed) == 9:
            # Backward compatibility: old entries without action_mask
            node_features_list = transposed[0]
            default_masks = [
                torch.ones(nf.shape[0], dtype=torch.bool) if isinstance(nf, torch.Tensor) else torch.ones(0, dtype=torch.bool)
                for nf in node_features_list
            ]
            transposed.append(default_masks)
        return tuple(transposed)

    def __len__(self):
        return len(self.storage)
    
    def save(self, filepath: str):
        """保存replay buffer到文件"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        checkpoint = {
            'capacity': self.capacity,
            'storage': self.storage,
            'ptr': self.ptr,
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str):
        """从文件加载replay buffer"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        checkpoint = torch.load(filepath, map_location='cpu')
        self.capacity = checkpoint['capacity']
        self.storage = checkpoint['storage']
        self.ptr = checkpoint['ptr']
        # Backward compatibility: ensure action_mask exists
        if self.storage and len(self.storage[0]) == 9:
            converted = []
            for item in self.storage:
                (node_features, edge_index, global_feat, actions, reward_tensor,
                 next_node_features, next_edge_index, next_global_feat, done_flag) = item
                mask = torch.ones(node_features.shape[0], dtype=torch.bool)
                converted.append((
                    node_features, edge_index, global_feat,
                    actions, reward_tensor,
                    next_node_features, next_edge_index, next_global_feat,
                    done_flag, mask
                ))
            self.storage = converted


class DQNAgent:
    """
    Graph-based DQN for simultaneous decisions on all cells.
    Uses Graph Convolutional Network (GCN) to process entire graph structure.

    Provides:
      - act_on_graph(graph_data) -> actions dict {cell_id: action} for all cells
      - policy_callback_all_cells(cell_observations, cell_adjacency) -> actions dict
      - optional train_step(replay)
    """
    def __init__(self,
                 feat_dim: int = None,  # None表示动态确定
                 hidden_dim: int = 64,
                 num_actions: int = 3,  # 3 actions: refine, coarsen, no-op
                 device: Optional[str] = None,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 global_dim: int = 0,
                 num_gcn_layers: int = 3):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # feat_dim可以为None，首次使用时动态确定
        self.feat_dim = feat_dim
        self._q_net = None
        self._tgt_net = None
        self._hidden_dim = hidden_dim
        self._num_actions = num_actions
        self._global_dim = global_dim
        self._num_gcn_layers = num_gcn_layers
        self._optimizer = None
        self._lr = lr
        self.gamma = gamma
        self.num_actions = num_actions
        self.global_dim = global_dim
    
    def _ensure_networks(self, feat_dim: int):
        """确保网络已初始化，如果feat_dim改变则重新创建"""
        if self._q_net is None or self.feat_dim != feat_dim:
            self.feat_dim = feat_dim
            self._q_net = GraphQNetwork(
                feat_dim, self._hidden_dim, self._num_actions, 
                global_dim=self._global_dim, num_gcn_layers=self._num_gcn_layers
            ).to(self.device)
            self._tgt_net = GraphQNetwork(
                feat_dim, self._hidden_dim, self._num_actions,
                global_dim=self._global_dim, num_gcn_layers=self._num_gcn_layers
            ).to(self.device)
            self._tgt_net.load_state_dict(self._q_net.state_dict())
            self._optimizer = torch.optim.Adam(self._q_net.parameters(), lr=self._lr)
    
    @property
    def q_net(self):
        if self._q_net is None:
            raise RuntimeError("Network not initialized. Call _ensure_networks first.")
        return self._q_net
    
    @property
    def tgt_net(self):
        if self._tgt_net is None:
            raise RuntimeError("Network not initialized. Call _ensure_networks first.")
        return self._tgt_net
    
    @property
    def optimizer(self):
        if self._optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call _ensure_networks first.")
        return self._optimizer

    @staticmethod
    def build_graph_data(cell_observations: Dict, cell_adjacency: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
        """
        从cell观测和邻接关系构建图数据。
        
        Args:
            cell_observations: {cell_id: {'self': [features], 'neighbors': [...]}}
            cell_adjacency: {cell_id: [neighbor_ids]}
        
        Returns:
            node_features: [N, F] 节点特征矩阵
            edge_index: [2, E] 边索引
            cell_id_to_index: {cell_id: index} 映射，用于后续恢复cell_id
        """
        # 获取所有cell_id并排序
        all_cell_ids = sorted(cell_observations.keys())
        num_nodes = len(all_cell_ids)
        
        if num_nodes == 0:
            # 空图
            return torch.zeros((0, 1), dtype=torch.float32), torch.zeros((2, 0), dtype=torch.long), {}
        
        # 创建cell_id到索引的映射
        cell_id_to_index = {cell_id: idx for idx, cell_id in enumerate(all_cell_ids)}
        
        # 构建节点特征矩阵 [N, F]
        node_features_list = []
        for cell_id in all_cell_ids:
            cell_obs = cell_observations[cell_id]
            self_feat = cell_obs.get('self', [])
            if isinstance(self_feat, (list, tuple)):
                feat_vec = [float(x) for x in self_feat]
            else:
                feat_vec = [0.0]
            node_features_list.append(feat_vec)
        
        # 确保所有特征维度一致
        max_feat_dim = max(len(f) for f in node_features_list) if node_features_list else 1
        node_features_list = [
            f if len(f) == max_feat_dim else (f + [0.0] * (max_feat_dim - len(f)) if len(f) < max_feat_dim else f[:max_feat_dim])
            for f in node_features_list
        ]
        
        node_features = torch.tensor(node_features_list, dtype=torch.float32)
        
        # 构建边索引 [2, E]
        edge_list = []
        for cell_id in all_cell_ids:
            cell_idx = cell_id_to_index[cell_id]
            neighbors = cell_adjacency.get(cell_id, [])
            for neighbor_id in neighbors:
                if neighbor_id in cell_id_to_index:
                    neighbor_idx = cell_id_to_index[neighbor_id]
                    # 添加双向边（无向图）
                    edge_list.append([cell_idx, neighbor_idx])
                    edge_list.append([neighbor_idx, cell_idx])
        
        # 去重边（如果存在重复）
        if edge_list:
            edge_list = list(set(tuple(e) for e in edge_list))
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return node_features, edge_index, cell_id_to_index

    def act_on_graph(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                     global_feat: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """
        对图进行前向传播，返回所有节点的动作。
        
        Args:
            node_features: [N, F] 节点特征
            edge_index: [2, E] 边索引
            global_feat: [1, G] 全局特征
            epsilon: epsilon-greedy探索率
        
        Returns:
            actions: [N] 每个节点的动作
        """
        import random
        
        feat_dim = node_features.shape[1]
        self._ensure_networks(feat_dim)
        
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        global_feat = global_feat.to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net(node_features, edge_index, global_feat)  # [N, A]
        
        # Epsilon-greedy: 对每个节点独立决策
        num_nodes = q_values.shape[0]
        actions = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
        
        for i in range(num_nodes):
            if random.random() < epsilon:
                actions[i] = random.randrange(self.num_actions)
            else:
                actions[i] = torch.argmax(q_values[i], dim=-1)
        
        return actions

    def policy_callback_all_cells(self, cell_observations: Dict, cell_adjacency: Dict, 
                                   epsilon: float = 0.0) -> Dict[int, int]:
        """
        对所有cell同时进行决策，返回每个cell的动作。
        
        Args:
            cell_observations: {cell_id: {'self': [features], 'neighbors': [...]}}
            cell_adjacency: {cell_id: [neighbor_ids]}
            epsilon: epsilon-greedy探索率
        
        Returns:
            actions_dict: {cell_id: action} 每个cell的动作
        """
        # 构建图数据
        node_features, edge_index, cell_id_to_index = self.build_graph_data(
            cell_observations, cell_adjacency
        )
        # 创建空的全局特征张量
        global_feat = torch.zeros((1, 0), dtype=torch.float32)
        
        if node_features.shape[0] == 0:
            return {}
        
        # 获取所有节点的动作
        actions = self.act_on_graph(node_features, edge_index, global_feat, epsilon)
        
        # 将动作映射回cell_id
        index_to_cell_id = {idx: cell_id for cell_id, idx in cell_id_to_index.items()}
        actions_dict = {index_to_cell_id[i]: int(actions[i].item()) for i in range(len(actions))}
        
        return actions_dict

    def select_single_cell_action(self, cell_observations: Dict, cell_adjacency: Dict,
                                  epsilon: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        选择单个cell执行动作，用于串行动作模式。
        
        Returns dict包含：
            {
              'cell_id': int,
              'action': int,
              'q_value': float or None,
              'strategy': 'explore' or 'exploit'
            }
        """
        node_features, edge_index, cell_id_to_index = self.build_graph_data(
            cell_observations, cell_adjacency
        )
        if node_features.shape[0] == 0:
            return None
        
        global_feat = torch.zeros((1, 0), dtype=torch.float32)
        cell_ids = sorted(cell_observations.keys())
        num_nodes = len(cell_ids)
        import random
        
        # 探索：随机选择cell和动作
        if random.random() < epsilon:
            selected_idx = random.randrange(num_nodes)
            selected_action = random.randrange(self.num_actions)
            selected_q = None
            strategy = 'explore'
        else:
            feat_dim = node_features.shape[1]
            self._ensure_networks(feat_dim)
            node_features_device = node_features.to(self.device)
            edge_index_device = edge_index.to(self.device)
            global_feat_device = global_feat.to(self.device)
            with torch.no_grad():
                q_values = self.q_net(node_features_device, edge_index_device, global_feat_device)
                best_values, best_actions = torch.max(q_values, dim=1)
                selected_idx = torch.argmax(best_values).item()
                selected_action = int(best_actions[selected_idx].item())
                selected_q = float(best_values[selected_idx].item())
            strategy = 'exploit'
        
        selected_cell_id = cell_ids[selected_idx]
        return {
            'cell_id': selected_cell_id,
            'action': selected_action,
            'q_value': selected_q,
            'strategy': strategy
        }

    def train_step(self, replay: ReplayBuffer, batch_size: int = 64, tgt_update_tau: float = 0.005) -> Optional[float]:
        """
        训练一步，从replay buffer中采样batch并更新网络。
        
        使用梯度累积处理所有样本，而不是只用第一个样本。
        """
        if len(replay) < 8:
            return None
        
        # 采样batch: (node_features, edge_index, global_feat, actions, reward, next_node_features, next_edge_index, next_global_feat, done, action_mask)
        sample = replay.sample(batch_size)
        if not sample:
            return None
        (s_node_feat, s_edge_idx, s_global, a, r,
         ns_node_feat, ns_edge_idx, ns_global, done, action_masks) = sample
        
        # 获取第一个样本的特征维度
        if len(s_node_feat) == 0:
            return None
        
        feat_dim = s_node_feat[0].shape[1]
        self._ensure_networks(feat_dim)
        
        # 使用梯度累积：对所有样本累积梯度
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        valid_samples = 0
        
        for i in range(min(batch_size, len(s_node_feat))):
            # 获取单个样本
            node_feat = s_node_feat[i].to(self.device)  # [N, F]
            edge_idx = s_edge_idx[i].to(self.device)    # [2, E]
            # 在训练时仍然需要处理全局特征，因为replay buffer中存储了它
            global_feat = s_global[i].to(self.device)   # [1, G]
            actions = a[i].to(self.device)              # [N]
            rewards = r[i].to(self.device)              # [N] - per-cell rewards
            action_mask = action_masks[i].to(self.device).bool() if action_masks else torch.ones(
                node_feat.shape[0], dtype=torch.bool, device=self.device)
            
            next_node_feat = ns_node_feat[i].to(self.device)  # [N, F]
            next_edge_idx = ns_edge_idx[i].to(self.device)    # [2, E]
            next_global_feat = ns_global[i].to(self.device)  # [1, G]
            done_flag = done[i].item()
            
            if node_feat.shape[0] == 0:
                continue  # 跳过空图
            
            if action_mask.shape[0] != node_feat.shape[0]:
                action_mask = torch.ones(node_feat.shape[0], dtype=torch.bool, device=self.device)
            
            if not action_mask.any():
                continue
            
            # 【安全检查】确保当前状态和下一状态的节点数量一致
            if node_feat.shape[0] != next_node_feat.shape[0]:
                print(f"  [WARNING] Skipping corrupted sample from replay buffer: "
                      f"node count mismatch ({node_feat.shape[0]} vs {next_node_feat.shape[0]})")
                print(f"  [HINT] This is likely old data. Consider clearing replay buffer for clean training.")
                continue  # 跳过这个样本
            
            # 当前状态Q值
            q_values = self.q_net(node_feat, edge_idx, global_feat)  # [N, A]
            
            # 选择执行的动作对应的Q值
            # actions: [N], q_values: [N, A]
            q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [N]
            
            # 目标Q值（使用target network）
            with torch.no_grad():
                next_q_values = self.tgt_net(next_node_feat, next_edge_idx, next_global_feat)  # [N, A]
                next_q_max = torch.max(next_q_values, dim=1).values  # [N]
                # 使用per-cell的reward：每个cell有自己的reward
                # rewards: [N], next_q_max: [N] -> target: [N]
                target = rewards + (1.0 - float(done_flag)) * self.gamma * next_q_max  # [N]

            # 只对有效节点（执行了动作的cell）计算loss
            valid_indices = action_mask.nonzero(as_tuple=False).view(-1)
            q_selected = q_selected[valid_indices]
            target = target[valid_indices]
            if q_selected.numel() == 0:
                continue
            
            # 计算损失（对所有节点平均）
            loss = F.smooth_l1_loss(q_selected, target)
            
            # 【修复】累积梯度：对每个样本的loss进行反向传播，梯度会自动累积
            # 除以batch_size以保持梯度的平均值（避免梯度爆炸）
            (loss / batch_size).backward()
            
            total_loss += loss.item()
            valid_samples += 1
        
        if valid_samples == 0:
            return None
        
        # 执行优化步骤（使用累积的梯度）
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        # Soft update target network
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.tgt_net.parameters()):
                tp.data.mul_(1.0 - tgt_update_tau).add_(tgt_update_tau * p.data)
        
        return total_loss / valid_samples if valid_samples > 0 else None
    
    def save_checkpoint(self, filepath: str):
        """保存DQN agent的checkpoint"""
        if self._q_net is None:
            raise RuntimeError("Network not initialized. Cannot save checkpoint.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        checkpoint = {
            'q_net_state_dict': self._q_net.state_dict(),
            'tgt_net_state_dict': self._tgt_net.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'feat_dim': self.feat_dim,
            'hidden_dim': self._hidden_dim,
            'num_actions': self._num_actions,
            'global_dim': self._global_dim,
            'lr': self._lr,
            'gamma': self.gamma,
            'device': str(self.device),
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """从文件加载checkpoint并恢复agent状态"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 恢复超参数
        self.feat_dim = checkpoint.get('feat_dim')
        self._hidden_dim = checkpoint.get('hidden_dim', self._hidden_dim)
        self._num_actions = checkpoint.get('num_actions', self._num_actions)
        self._global_dim = checkpoint.get('global_dim', self._global_dim)
        self._lr = checkpoint.get('lr', self._lr)
        self.gamma = checkpoint.get('gamma', self.gamma)
        
        # 如果feat_dim已知，初始化网络
        if self.feat_dim is not None:
            self._ensure_networks(self.feat_dim)
            
            # 加载网络状态
            self._q_net.load_state_dict(checkpoint['q_net_state_dict'])
            self._tgt_net.load_state_dict(checkpoint['tgt_net_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


